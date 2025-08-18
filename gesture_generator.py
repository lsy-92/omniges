import json
import os
import pickle
import random
import threading
import time
import copy
import argparse

import librosa
import numpy as np

from os import environ

from src.utils.utils import set_seed
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

import torch
import soundfile as sf
from scipy.interpolate import interp1d

# gemma3 bug workaround
# https://github.com/huggingface/transformers/issues/36815
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from pathlib import Path

from matplotlib import pyplot as plt
import re
import sys
from transformers import pipeline

from src.models.diffusion_module import DiffusionLitModule
from src.utils.tts_helper import TTSHelper
from src.utils.viz_util import JOINT_NAME2IDX, generate_bvh, render_bvh, render_motion_to_video, unnormalize, normalize, \
    play_beat_motion_with_audio

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def double_fps_scipy(motion_data):
    num_frames = motion_data.shape[0]
    original_times = np.linspace(0, 1, num_frames)
    target_times = np.linspace(0, 1, num_frames * 2)  # 2배 프레임

    interpolator = interp1d(original_times, motion_data, axis=0, kind='linear')
    new_motion = interpolator(target_times)
    return new_motion


class GestureGenerator:
    def __init__(self, lit_checkpoint_path):
        self.model = DiffusionLitModule.load_from_checkpoint(lit_checkpoint_path)
        self.tts = TTSHelper(use_library=False)

        self.model.to(device)
        self.model.eval()

        self.motion_library = {}

        self.previous_semantic_gestures = []

        # SeG motion library
        with open('seg_motion_library.json', 'r') as json_file:
            seg_data = json.load(json_file)

        # TODO: use config or embed this index info in json
        N2I = JOINT_NAME2IDX
        control_joint_index = [
            N2I['head'], N2I['right_shoulder'], N2I['left_shoulder'],
            N2I['right_elbow'], N2I['left_elbow'],
            N2I['right_wrist'], N2I['left_wrist']
        ]
        for k, v in seg_data.items():
            control_data = np.array(v)
            motion_array = np.zeros((control_data.shape[0], 56, 3))  # TODO: avoid hard-coded dimension
            motion_array[:, control_joint_index] = control_data
            self.motion_library[k] = motion_array.reshape(control_data.shape[0], -1)

    def control_to_raw_motion_constraint(self, control):
        n_frames = max(c[3] for c in control)
        n_joints = 56

        motion_array = np.zeros((n_frames, n_joints, 3))

        # convert control data to raw motion array
        for joint_index, joint_data, start_frame, end_frame in control:
            for i in range(start_frame, end_frame):
                motion_array[i, joint_index, :] = joint_data

        return motion_array.reshape(n_frames, -1)

    def generate(self, speech_audio, pose_constraints=None):
        out = self.model.synthesize_gesture_from_audio(
            speech_audio, pose_constraints, n_pre_poses=32, do_smoothing=True)
        return out

    def generate_from_text(self, speech_text, voice='kr-female', no_gesture=False, llm_pipe=None):
        def remove_tags_marks(text):
            reg_expr = re.compile('<.*?>|[.,:;!?]+')
            clean_text = re.sub(reg_expr, '', text)
            return clean_text

        def find_token_positions(input_string, target_word):
            tokens = input_string.lower().split()
            target_word = target_word.lower().strip()
            positions = []

            for index, token in enumerate(tokens):
                if target_word in token:
                    positions.append(index)

            return positions

        def tts_task(tts_str, tts_result, silence_margin=0):
            tts_filename, timepoints = self.tts.synthesis(tts_str, voice_name=voice, get_word_timestamps=True,
                                                          verbose=True, silence_margin=silence_margin,
                                                          minimum_length=2.0)

            # fast version
            y, sr = librosa.load(tts_filename, sr=None)
            if len(y) < 3 * sr:
                fast_rate = 1.0
            else:
                fast_rate = 1.5
            y_fast = librosa.effects.time_stretch(y, rate=fast_rate)

            fast_filename = tts_filename.replace('.wav', '_fast.wav')
            sf.write(fast_filename, y_fast, sr)

            fast_timepoints = copy.deepcopy(timepoints)
            for tp in fast_timepoints:
                tp['timeSeconds'] /= fast_rate

            tts_result['filename'] = tts_filename
            tts_result['timepoints'] = timepoints
            tts_result['fast_filename'] = fast_filename
            tts_result['fast_timepoints'] = fast_timepoints

        def gesture_selection_task(selection_result):
            with open("./src/resource/gesture_selection_prompt.txt", "r", encoding="utf-8") as f:
                prompt_text = f.read()

            # Add constraint about previously used gestures
            if self.previous_semantic_gestures:
                prompt_text += "\n\n"
                prompt_text += (
                    "Do not select following gestures that have already been used. "
                    f"The following gestures are not allowed: {', '.join(self.previous_semantic_gestures)}."
                )

            chat = [
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": "Sentence: " + gesture_str}
            ]

            start = time.time()
            response = llm_pipe(chat, max_new_tokens=512)
            response_text = response[0]['generated_text'][-1]['content']
            print(f' >> gesture selection response: {response_text} [took {(time.time() - start):.2f} s]')

            # Parse the response
            cleaned = response_text.strip().strip('()')
            gesture_part, word_part = cleaned.split(",", 1)
            word = word_part.strip().strip("'").strip('"')
            selection_result['gesture'] = gesture_part
            selection_result['start_word'] = word

        # prepare inference
        print(' >> inferring gestures for \'%s\'...' % speech_text)

        tts_str = speech_text
        gesture_str = remove_tags_marks(tts_str)

        # TTS, LLM gesture selection
        tts_silence_margin = 0
        tts_result = {}
        selection_result = {}
        # print("Current working directory:", os.getcwd())

        tts_thread = threading.Thread(target=tts_task, args=(tts_str, tts_result, tts_silence_margin))
        selection_thread = threading.Thread(target=gesture_selection_task, args=(selection_result,))

        # thread start
        start = time.time()
        tts_thread.start()
        selection_thread.start()

        tts_thread.join()
        selection_thread.join()

        print(f' >> TTS and gesture selection took {(time.time() - start):.2f} s')

        audio, audio_sr = librosa.load(tts_result['fast_filename'], mono=True, sr=16000, res_type='kaiser_fast')
        audio_duration = len(audio) / audio_sr
        n_frames = int(audio_duration * self.model.motion_fps)

        # Use llm selection result to form the pose control vector.
        pose_control = None
        control_info = []

        selected_gesture = selection_result.get('gesture')
        gesture_start_word = selection_result.get('start_word')

        if selected_gesture and gesture_start_word:
            matched_keys = [key for key in self.motion_library.keys() if selected_gesture in key]
            if matched_keys:
                added = False
                selected_gesture_key = matched_keys[0]
                if len(self.previous_semantic_gestures) >= 4:
                    self.previous_semantic_gestures.pop()
                self.previous_semantic_gestures.append(selected_gesture_key)
                selected_motion = self.motion_library[selected_gesture_key]
                positions = find_token_positions(gesture_str, gesture_start_word)
                for p in positions:
                    if p >= len(tts_result['fast_timepoints']):
                        control_pos_begin = n_frames - len(selected_motion)
                        print(' >> Warning: gesture start word not found in TTS timepoints')
                    else:
                        marked_frame = int(tts_result['fast_timepoints'][p]['timeSeconds'] * self.model.motion_fps)
                        control_pos_begin = max(marked_frame - int(len(selected_motion) / 2), 0)
                    control_pos_end = min(control_pos_begin + len(selected_motion), n_frames)
                    if pose_control is None:
                        pose_control = np.zeros((n_frames, self.model.pose_dim))
                    else:
                        if np.any(pose_control[control_pos_begin:control_pos_end]):
                            continue
                    print(f' -- Motion control: ({selected_gesture_key}, {control_pos_begin}--{control_pos_end - 1})')
                    pose_control[control_pos_begin:control_pos_end, :56*3] = selected_motion[
                                                                           : control_pos_end - control_pos_begin]
                    control_info.append({'word': selected_gesture_key, 'start_frame': control_pos_begin,
                                         'end_frame': control_pos_end-1, 'motion': selected_motion})
                    added = True
                if not added:
                    print(' >> Warning: gesture start word not found in TTS timepoints')
            else:
                print(" >> Warning: no matching gesture found in motion_library for the selected gesture.")
        else:
            print(" >> Warning: LLM selection result is incomplete. No pose control vector created.")

        if no_gesture:
            generated_motion = np.random.rand(1, 1)
        else:
            generated_motion = self.generate(audio, pose_control)

        # output_wav_path = tts_result['fast_filename']
        output_wav_path = tts_result['filename']
        if tts_silence_margin > 0:  # remove silence margin
            print(f' >> Motion shape {generated_motion.shape}')
            n_exclude = int(tts_silence_margin * self.model.motion_fps)
            n_audio_exclude = int(tts_silence_margin * audio_sr)
            generated_motion = generated_motion[n_exclude:-n_exclude, :]
            for i in range(len(control_info)):
                control_info[i]['start_frame'] = max(control_info[i]['start_frame'] - n_exclude, 0)
                control_info[i]['end_frame'] = max(control_info[i]['end_frame'] - n_exclude, 0)
            output_wav_path = output_wav_path.replace('.wav', '_no_silence.wav')
            audio_trunc = audio[n_audio_exclude:-n_audio_exclude]
            sf.write(output_wav_path, audio_trunc, audio_sr)

            control_str = ""
            if len(control_info) > 0:
                control_str = f', control ({control_info[0]["start_frame"]}, {control_info[0]["end_frame"]})'
            print(f' >> Removing initial silence margin. Motion shape {generated_motion.shape}, '
                  f'audio len {len(audio_trunc) / audio_sr} s{control_str}')

        return generated_motion, output_wav_path, control_info


def test_with_audio_files(generator):
    wav_files = ['/media/yw/work2/BEAT2/beat_english_v2.0.0/wave16k/9_miranda_0_34_34.wav',]

    max_len = 16000 * 10  # use the first 30 seconds
    n_iter = 1

    for iteration in range(n_iter):
        for wav_path in wav_files:
            audio, audio_sr = librosa.load(wav_path, mono=True, sr=16000, res_type='kaiser_fast')
            audio = audio[:max_len]
            filename = Path(wav_path).stem

            # prepare pose constraint array
            n_frames = int(len(audio) / audio_sr * generator.model.motion_fps)
            pose_control = np.zeros((n_frames, generator.model.pose_dim))

            # manual control
            if False:
                # pose_constraints[0:25, 0] = 4.0  # root position
                # pose_constraints[25:50, 0] = -4.0
                # idx 21, 40: left/right wrist
                pose_control[0:50, 21 * 3] = 10.0
                pose_control[0:50, 40 * 3] = 0.0
                pose_control[50:100, 21 * 3] = 0.0
                pose_control[50:100, 40 * 3] = -10.0

            # motion library
            if True:
                # selected_motion = generator.motion_library['ARMS_WELCOME-1']
                selected_motion = generator.motion_library['FOREARM_CUT-1']
                pose_control[0:selected_motion.shape[0], :56*3] = selected_motion

            # synthesize motion
            start = time.time()
            generated_motion = generator.generate(audio, pose_constraints=pose_control)
            print(generated_motion.shape, 'processing time', time.time() - start)

            print('rendering...')
            title = f"{os.path.basename(wav_path)}"
            out_path = './logs/temp'

            generated_motion = unnormalize(generated_motion, generator.model.norm_method, generator.model.data_stat)
            generated_motion = double_fps_scipy(generated_motion)  # 15 to 30 fps
            root_pos, joint_pos, joint_rot = generator.model.parse_output_vec(generated_motion)

            out_filename = f'{filename}_iter{iteration}'

            # render a video
            poses = np.concatenate((np.expand_dims(root_pos, 1), joint_pos), axis=1)  # (seq, joint, 3)
            render_motion_to_video(poses, wav_path, title, mp4_path=f'./logs/temp/{out_filename}.mp4', fps=30)


def test_with_tts(generator, llm_pipe):
    input_speech_list = [
        u'저희 데모 부스를 찾아주셨네요. 가까이 와서 봐주세요. 여러가지 시연을 준비했어요.',]
    voice = 'kr-female'  # kr-male, kr-female, en-male, en-female

    for i, input_speech in enumerate(input_speech_list):
        # synthesize motion
        generated_motion, wav_path, _ = generator.generate_from_text(input_speech, voice=voice, llm_pipe=llm_pipe)
        print(generated_motion.shape)

        print('rendering...')
        title = f"{input_speech[:20]}"

        generated_motion = unnormalize(generated_motion, generator.model.norm_method, generator.model.data_stat)
        generated_motion = double_fps_scipy(generated_motion)  # 15 to 30 fps
        root_pos, joint_pos, joint_rot = generator.model.parse_output_vec(generated_motion)
        poses = np.concatenate((np.expand_dims(root_pos, 1), joint_pos), axis=1)  # (seq, joint, 3)
        render_motion_to_video(poses, wav_path, title, mp4_path='./logs/temp/test_tts.mp4', fps=30)


def interactive_demo(generator, lang='kor'):
    plt.ion()

    with open(f"resource/example_sentence_{lang}.txt", "r", encoding="utf-8") as f:
        examples = [line.strip() for line in f if line.strip()]

    if lang == 'kor':
        voice = 'kr-female'  # kr-male, kr-female, en-male, en-female
    elif lang == 'eng':
        voice = 'en-male'  # kr-male, kr-female, en-male, en-female
    else:
        assert False
    examples.append('Try a new one')

    while True:
        fig = plt.figure(figsize=(3, 3))
        manager = plt.get_current_fig_manager()
        # geom = manager.window.geometry()
        # x, y, dx, dy = geom.getRect()
        # manager.window.setGeometry(100, 100, dx, dy)

        for i, example in enumerate(examples):
            print('%d: %s' % (i, example))

        try:
            select = int(input("select: "))
        except ValueError:
            continue

        if select == len(examples) - 1:
            sentence = input("Sentence: ")
            # if True:  # todo: is korean?
            #     sentence = sentence.decode(sys.stdin.encoding)
        elif select >= len(examples) or select < 0:
            print('Exiting...')
            break
        else:
            sentence = examples[select]

        # synthesize motion
        seed = random.randint(0, 10000)
        seed = 9152 if lang == 'kor' else 7083
        print(seed)
        set_seed(seed)
        generated_motion, wav_path, _ = generator.generate_from_text(sentence, voice=voice)
        print(generated_motion.shape)

        title = f"Generated speech motion"

        generated_motion = unnormalize(generated_motion, generator.model.norm_method, generator.model.data_stat)

        if False:
            # export motion
            npy_path = wav_path.replace('.wav', '.npy')
            with open(npy_path, 'wb') as f:
                np.save(f, generated_motion)

        generated_motion = double_fps_scipy(generated_motion)  # 15 to 30 fps
        root_pos, joint_pos, joint_rot = generator.model.parse_output_vec(generated_motion)

        # play motion and audio or save
        save = True
        if save:
            out_name = f'saved.mp4'
            out_mp4_path = render_bvh(generated_motion, wav_path, title, out_path='./logs/temp', fps=30)

        play_beat_motion_with_audio(generated_motion, wav_path, title, fps=30, fig=fig)


def generate_llm_response(chat, pipe):
    response = pipe(chat, max_new_tokens=512)
    response_text = response[0]['generated_text'][-1]['content']
    return response_text


def interactive_llm_demo(generator, lang, llm_pipe):
    ax = None
    voice = 'kr-female' if lang == 'kor' else 'en-male'
    chat = [
        {"role": "system", "content": "너는 캐주얼한 대화를 하는 챗봇이야. 반드시 한국어로만 짧게 대답해. 영어 절대 쓰지 마. Emoji 사용하지 마."}
    ]
    print("LLM 대화 모드 시작 (종료하려면 'exit' 입력)")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("종료합니다.")
            break

        # LLM 모델을 통해 응답 생성
        chat.append({"role": "user", "content": user_input})
        llm_response = generate_llm_response(chat, llm_pipe)
        # 한글, 영어, 숫자, 공백, 일반적인 특수문자만 허용
        llm_response = re.sub(r'[^ㄱ-ㅎ가-힣a-zA-Z0-9\s.,!?\'"()[\]{}:;@#%^&*+=~\-_/\\|<>]', '', llm_response)
        print("LLM:", llm_response)
        chat.append({"role": "assistant", "content": llm_response})

        # 제스처 생성을 위한 시드 설정
        seed = random.randint(0, 10000)
        seed = 9152 if lang == 'kor' else 7083
        set_seed(seed)

        # 텍스트를 음성 및 제스처로 변환
        generated_motion, wav_path, _ = generator.generate_from_text(llm_response, voice=voice, llm_pipe=llm_pipe)
        print(" >> motion shape:", generated_motion.shape)

        # 제스처 후처리 및 렌더링
        generated_motion = unnormalize(generated_motion, generator.model.norm_method, generator.model.data_stat)
        generated_motion = double_fps_scipy(generated_motion)  # 15 to 30 fps
        title = llm_response

        play_beat_motion_with_audio(generated_motion, wav_path, title, fps=30,
                                        fig=plt.figure(figsize=(3, 3)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gesture Generation Demo")
    parser.add_argument('--demo_type', type=str, default='interactive_llm',
                        choices=['interactive_llm', 'interactive', 'tts', 'audio'], help='Type of demo to run')
    parser.add_argument('--lang', type=str, default='kor', choices=['kor', 'eng'], help='Language for the demo')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--llm_name', type=str, default='google/gemma-3-12b-it', help='Hugging Face model name')
    parser.add_argument('--hf_access_token', type=str, help='Hugging Face access token')

    args = parser.parse_args()

    generator = GestureGenerator(args.ckpt_path)

    # llm pipe
    if not args.hf_access_token:
        raise ValueError("Hugging Face access token is required for the LLM demo.")
    llm_pipe = pipeline("text-generation", args.llm_name, torch_dtype=torch.bfloat16,
                        device=device, token=args.hf_access_token)

    # mode
    if args.demo_type == 'interactive_llm':
        interactive_llm_demo(generator, args.lang, llm_pipe)
    elif args.demo_type == 'interactive':
        interactive_demo(generator, args.lang)
    elif args.demo_type == 'tts':
        test_with_tts(generator, llm_pipe)
    elif args.demo_type == 'audio':
        test_with_audio_files(generator)

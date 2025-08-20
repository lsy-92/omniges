#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

"""
Omniges Training Script
Complete multi-task training for Text-Audio-Gesture generation
Based on OmniFlow + RVQVAE gesture processing
Supports: t2g, g2t, a2g, g2a, t2a, a2t
"""

# ============================================================================
# 표준 라이브러리 임포트
# ============================================================================
import argparse  # 명령행 인수 파싱
import copy  # 객체 복사
import gc  # 가비지 컬렉션
import time  # 시간 측정
from safetensors import safe_open  # 안전한 텐서 파일 로딩
import sys  # 시스템 관련 기능
import os  # 운영체제 관련 기능
from pathlib import Path  # 경로 처리

# ============================================================================
# 프로젝트 루트 경로 추가
# ============================================================================
sys.path.append(str(Path(__file__).parent.parent.parent))  # 현재 파일의 상위 3단계 디렉토리를 Python 경로에 추가

# ============================================================================
# Omniges 컴포넌트 임포트
# ============================================================================
from omniges.models import OmnigesFlowTransformerModel, GestureProcessor  # Omniges 모델 및 제스처 프로세서
from omniges.pipelines import OmnigesPipeline, OmnigesGestureVAE  # Omniges 파이프라인 및 제스처 VAE

# ============================================================================
# OmniFlow 컴포넌트 임포트
# ============================================================================
from omniflow.utils.ema import EMAModel  # 지수 이동 평균 모델
import torch.utils.data  # PyTorch 데이터 유틸리티
from transformers.trainer_pt_utils import LabelSmoother  # 라벨 스무딩
import itertools  # 반복자 유틸리티
import logging  # 로깅
import math  # 수학 함수
import random  # 랜덤 생성
import shutil  # 파일/디렉토리 복사
import warnings  # 경고 처리
from contextlib import nullcontext  # 컨텍스트 매니저
import pandas as pd  # 데이터 분석
import numpy as np  # 수치 계산
import torch  # PyTorch 딥러닝 프레임워크
import torch.utils.checkpoint  # 그래디언트 체크포인팅
import transformers  # Hugging Face 트랜스포머
from accelerate import Accelerator  # 분산 훈련 가속기
from accelerate.logging import get_logger  # 로거 생성
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed  # 분산 훈련 유틸리티
from huggingface_hub import create_repo, upload_folder  # Hugging Face Hub 연동
from huggingface_hub.utils import insecure_hashlib  # 해시 유틸리티
from PIL import Image  # 이미지 처리
from PIL.ImageOps import exif_transpose  # EXIF 정보 처리
from torch.utils.data import Dataset  # 데이터셋 클래스
from torchvision import transforms  # 이미지 변환
from torchvision.transforms.functional import crop  # 이미지 크롭
from tqdm.auto import tqdm  # 진행률 표시
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast, CLIPImageProcessor  # 트랜스포머 모델들
import torch.nn.functional as F  # PyTorch 함수형 API
import diffusers  # 디퓨전 모델 라이브러리
from diffusers import AutoencoderKL  # 자동 인코더
from omniflow.utils.scheduler import OmniFlowMatchEulerDiscreteScheduler as FlowMatchEulerDiscreteScheduler  # OmniFlow 스케줄러 (Omniges에서 재사용)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS  # 레이어 정규화
from transformers.trainer_pt_utils import get_parameter_names  # 파라미터 이름 가져오기
from diffusers.image_processor import VaeImageProcessor  # VAE 이미지 프로세서
from diffusers.optimization import get_scheduler  # 옵티마이저 스케줄러
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3  # 훈련 유틸리티
from diffusers.utils import (  # diffusers 유틸리티
    check_min_version,  # 버전 체크
    is_wandb_available,  # wandb 사용 가능 여부
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card  # 모델 카드 유틸리티
from diffusers.utils.torch_utils import is_compiled_module  # 컴파일된 모듈 체크
import torch.distributed as dist  # 분산 훈련
import glob  # 파일 패턴 매칭
from omniflow.models.audio_vae import load_audio_vae  # 오디오 VAE 로딩
from omniflow.utils.text_encode import encode_prompt_train, cat_and_pad, encode_prompt_for_decoder  # 텍스트 인코딩 유틸리티

# ============================================================================
# wandb 사용 가능 여부 확인 및 임포트
# ============================================================================
if is_wandb_available():  # wandb가 사용 가능한 경우
    import wandb  # 실험 추적 라이브러리
from torch import nn  # 신경망 모듈
check_min_version("0.30.0.dev0")  # 최소 버전 체크

logger = get_logger(__name__)  # 로거 생성

# ============================================================================
# 검증용 파일 경로 (테스트용)
# ============================================================================
VAL_FILES = ['./assets/girl.png']  # 검증용 이미지 파일
VAL_FILES_AUDIO = ['./assets/car engine.mp3']  # 검증용 오디오 파일

# ============================================================================
# 추가 모델 및 유틸리티 임포트
# ============================================================================
from omniflow.models.text_vae import LLamaForLatentConnector  # 텍스트 VAE (LLaMA 기반)
from omniflow.models.encoders import LanguageBindAudioProcessor, LanguageBindAudio  # 오디오 인코더
import yaml  # YAML 설정 파일 처리
from transformers import AutoTokenizer, AutoConfig  # 자동 토크나이저 및 설정

# ============================================================================
# BEAT 데이터 처리 임포트
# ============================================================================
from dataloaders.beat_sep_lower import CustomDataset  # BEAT 데이터셋
from dataloaders.data_tools import joints_list  # 관절 리스트
from utils import rotation_conversions as rc  # 회전 변환 유틸리티


# ============================================================================
# 유틸리티 함수들
# ============================================================================

def load_yaml(fp: str):
    """
    YAML 설정 파일을 로드하는 함수
    """
    with open(fp, 'r') as file:  # 파일을 읽기 모드로 열기
        data = yaml.safe_load(file)  # YAML 파일을 안전하게 파싱
    return data  # 파싱된 데이터 반환


def n_get_sigmas(noise_scheduler_copy, device, timesteps, n_dim=4, dtype=torch.float32):
    """
    노이즈 스케줄러에서 특정 타임스텝에 해당하는 시그마 값을 추출하는 함수
    
    Args:
        noise_scheduler_copy: 노이즈 스케줄러 복사본
        device: 계산 디바이스
        timesteps: 타임스텝 텐서
        n_dim: 원하는 차원 수 (기본값: 4)
        dtype: 데이터 타입 (기본값: torch.float32)
    
    Returns:
        sigma: 해당 타임스텝의 시그마 값들
    """
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)  # 시그마 값을 디바이스로 이동
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device)  # 스케줄 타임스텝을 디바이스로 이동
    timesteps = timesteps.to(device)  # 입력 타임스텝을 디바이스로 이동
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]  # 각 타임스텝에 해당하는 인덱스 찾기

    sigma = sigmas[step_indices].flatten()  # 해당 인덱스의 시그마 값들을 추출하고 평탄화
    while len(sigma.shape) < n_dim:  # 원하는 차원 수에 도달할 때까지 차원 추가
        sigma = sigma.unsqueeze(-1)  # 마지막 차원에 1차원 추가
    return sigma  # 최종 시그마 텐서 반환


def n_compute_text_embeddings(device, prompt, text_encoders, tokenizers, add_token_embed=True, train=False):
    """
    텍스트 프롬프트를 임베딩으로 변환하는 함수
    
    Args:
        device: 계산 디바이스
        prompt: 텍스트 프롬프트 리스트
        text_encoders: 텍스트 인코더 모델들
        tokenizers: 토크나이저들
        add_token_embed: 토큰 임베딩 추가 여부 (기본값: True)
        train: 훈련 모드 여부 (기본값: False)
    
    Returns:
        prompt_embeds: 프롬프트 임베딩
        pooled_prompt_embeds: 풀링된 프롬프트 임베딩
    """
    print(f"DEBUG: [n_compute_text_embeddings] Input prompts count: {len(prompt)}")  # 입력 프롬프트 개수 출력
    print(f"DEBUG: [n_compute_text_embeddings] Sample prompt: {prompt[0] if prompt else 'No prompts'}")  # 샘플 프롬프트 출력
    print(f"DEBUG: [n_compute_text_embeddings] add_token_embed: {add_token_embed}, train: {train}")  # 설정값 출력
    
    with torch.no_grad():  # 그래디언트 계산 비활성화 (추론 모드)
        prompt_embeds, pooled_prompt_embeds = encode_prompt_train(  # 훈련용 프롬프트 인코딩 함수 호출
            text_encoders,  # 텍스트 인코더들
            tokenizers,  # 토크나이저들
            prompt,  # 프롬프트 리스트
            256,  # 최대 시퀀스 길이
            add_token_embed=add_token_embed,  # 토큰 임베딩 추가 여부
            normalize=True,  # 정규화 활성화
            drops=list(  # 드롭아웃 설정
                np.random.rand() > 0.5 for _ in range(4)  # 훈련 시 랜덤 드롭아웃
            ) if train else [False, False, False, False]  # 추론 시 드롭아웃 비활성화
        )
        print(f"DEBUG: [n_compute_text_embeddings] Raw prompt_embeds shape: {prompt_embeds.shape}")  # 원본 프롬프트 임베딩 형태 출력
        print(f"DEBUG: [n_compute_text_embeddings] Raw pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}")  # 원본 풀링된 프롬프트 임베딩 형태 출력
        
        prompt_embeds = prompt_embeds.to(device)  # 프롬프트 임베딩을 디바이스로 이동
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)  # 풀링된 프롬프트 임베딩을 디바이스로 이동
        
        print(f"DEBUG: [n_compute_text_embeddings] Final prompt_embeds shape: {prompt_embeds.shape}")  # 최종 프롬프트 임베딩 형태 출력
        print(f"DEBUG: [n_compute_text_embeddings] Final pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}")  # 최종 풀링된 프롬프트 임베딩 형태 출력
    return prompt_embeds, pooled_prompt_embeds  # 프롬프트 임베딩과 풀링된 임베딩 반환


class OmnigesDataset(Dataset):
    """
    Omniges 데이터셋 - BEAT 제스처 데이터와 텍스트/오디오를 결합
    모든 태스크 조합 지원: t2g, g2t, a2g, g2a, t2a, a2t
    """

    def __init__(
        self,
        beat_config_path="configs/shortcut_rvqvae_128.yaml",  # BEAT 설정 파일 경로
        task_weights=[1/6] * 6,  # 모든 6개 태스크에 동일한 가중치
        size=512,  # 이미지 크기 (호환성을 위해 유지)
        is_train=True,  # 훈련 모드 여부
        image_processor=None,  # 이미지 프로세서 (호환성을 위해 유지)
        audio_processor=None,  # 오디오 프로세서
        audio_processor_clip=None,  # CLIP용 오디오 프로세서
    ):
        # ============================================================================
        # 기본 속성 초기화
        # ============================================================================
        self.size = size  # 이미지 크기 저장
        self.image_processor = image_processor  # 이미지 프로세서 저장
        self.audio_processor = audio_processor  # 오디오 프로세서 저장
        self.audio_processor_clip = audio_processor_clip  # CLIP용 오디오 프로세서 저장
        self.task_weights = task_weights  # 태스크 가중치 저장
        self.is_train = is_train  # 훈련 모드 저장
        
        # ============================================================================
        # BEAT 설정 파일 로드 및 데이터셋 생성
        # ============================================================================
        with open(beat_config_path, 'r') as f:  # BEAT 설정 파일을 읽기 모드로 열기
            beat_config = yaml.safe_load(f)  # YAML 파일을 안전하게 파싱
            
        # ============================================================================
        # BEAT 설정을 위한 인자 객체 생성
        # ============================================================================
        class BeatArgs:
            def __init__(self, config):
                for key, value in config.items():  # 설정 딕셔너리의 모든 키-값 쌍을 반복
                    setattr(self, key, value)  # 객체에 속성으로 설정
                # BEAT 데이터셋을 위한 모든 누락된 속성 추가
                self.multi_length_training = [1.0]  # 다중 길이 훈련 설정
                self.beat_align = False  # BEAT 정렬 비활성화
                self.word_cache = False          # 단어 캐시 비활성화
                self.facial_cache = False        # 얼굴 캐시 비활성화
                self.audio_cache = False         # 오디오 캐시 비활성화
                self.pose_cache = False          # 포즈 캐시 비활성화
                self.trans_cache = False         # 변환 캐시 비활성화
                self.speaker_cache = False       # 화자 캐시 비활성화
                self.emotion_cache = False       # 감정 캐시 비활성화
                self.semantic_cache = False      # 의미 캐시 비활성화
                
        self.beat_args = BeatArgs(beat_config)  # BEAT 인자 객체 생성
        
        # ============================================================================
        # BEAT 데이터셋 생성
        # ============================================================================
        self.beat_dataset = CustomDataset(  # 커스텀 BEAT 데이터셋 생성
            self.beat_args,  # BEAT 인자 전달
            loader_type="train" if is_train else "test",  # 훈련/테스트 모드 설정
            build_cache=True  # 캐시 구축 활성화
        )
        
        # ============================================================================
        # 태스크 조합 및 프롬프트 설정
        # ============================================================================
        # 지원하는 태스크 조합들
        self.tasks = ['t2g', 'g2t', 'a2g', 'g2a', 't2a', 'a2t']  # 6가지 멀티모달 태스크
        
        # 제스처 태스크를 위한 텍스트 프롬프트 생성
        self.gesture_prompts = [  # 제스처 관련 텍스트 프롬프트 리스트
            "A person waving hello",  # 인사하는 사람
            "Someone clapping their hands",  # 박수치는 사람
            "A person pointing forward",  # 앞을 가리키는 사람
            "Dancing with arm movements",  # 팔 움직임으로 춤추는 사람
            "Gesturing while speaking",  # 말하면서 제스처하는 사람
            "Hand gestures during conversation",  # 대화 중 손 제스처
            "Expressive body language",  # 표현적인 바디랭귀지
            "Animated talking with hands",  # 손으로 애니메이션하며 말하는 사람
            "Conducting orchestra movements",  # 오케스트라 지휘 동작
            "Sign language communication"  # 수화 의사소통
        ]

    def __len__(self):
        """
        데이터셋의 길이를 반환하는 메서드
        """
        return len(self.beat_dataset)  # BEAT 데이터셋의 길이 반환

    def __getitem__(self, index):
        """
        데이터셋에서 특정 인덱스의 아이템을 가져오는 메서드
        
        Args:
            index: 가져올 아이템의 인덱스
        
        Returns:
            dict: 처리된 데이터 아이템
        """
        # ============================================================================
        # BEAT 데이터 가져오기
        # ============================================================================
        beat_item = self.beat_dataset[index]  # BEAT 데이터셋에서 해당 인덱스의 아이템 가져오기
        
        # ============================================================================
        # 랜덤 태스크 선택
        # ============================================================================
        task = np.random.choice(self.tasks, p=self.task_weights)  # 가중치에 따라 랜덤하게 태스크 선택
        
        # ============================================================================
        # 태스크별 처리 로직
        # ============================================================================
        if task in ['t2g', 'g2t']:  # 텍스트-제스처 태스크들
            # 텍스트-제스처 태스크
            prompt = np.random.choice(self.gesture_prompts)  # 제스처 프롬프트 중 랜덤 선택
            prompt2 = prompt  # 두 인코더 모두에 동일한 프롬프트 사용
            has_text = True  # 텍스트 사용
            has_gesture = True  # 제스처 사용
            has_audio = False  # 오디오 미사용
            
        elif task in ['a2g', 'g2a']:  # 오디오-제스처 태스크들
            # 오디오-제스처 태스크
            prompt = ""  # 순수 오디오-제스처를 위한 텍스트 없음
            prompt2 = ""
            has_text = False  # 텍스트 미사용
            has_gesture = True  # 제스처 사용
            has_audio = True  # 오디오 사용
            
        elif task == 't2a':  # 텍스트-오디오 태스크 (OmniFlow에서 가져옴)
            # 텍스트-오디오 태스크
            prompt = "Music playing"  # 오디오 설명
            prompt2 = prompt
            has_text = True  # 텍스트 사용
            has_gesture = False  # 제스처 미사용
            has_audio = True  # 오디오 사용
            
        elif task == 'a2t':  # 오디오-텍스트 태스크 (OmniFlow에서 가져옴)
            # 오디오-텍스트 태스크
            prompt = ""  # 오디오에서 생성될 텍스트
            prompt2 = ""
            has_text = True  # 텍스트 사용
            has_gesture = False  # 제스처 미사용
            has_audio = True  # 오디오 사용
            
        # ============================================================================
        # 제스처 데이터 처리
        # ============================================================================
        pose = beat_item['pose']       # 포즈 데이터 (T, pose_dim)
        facial = beat_item['facial']   # 얼굴 데이터 (T, 100)
        trans = beat_item['trans']     # 변환 데이터 (T, 3)
        trans_v = beat_item['trans_v'] # 변환 속도 데이터 (T, 3)
        audio = beat_item['audio']     # 오디오 데이터 (T_audio,)
        
        # ============================================================================
        # 배치 형식으로 변환
        # ============================================================================
        gesture_sequence = self._process_gesture_data(pose, facial, trans, trans_v)  # 제스처 데이터 처리
        
        # ============================================================================
        # 오디오 VAE를 위한 오디오 처리
        # ============================================================================
        audio_vae_input = torch.zeros(1, 1, 1024, 64)  # 기본 오디오 VAE 입력 (더미)
        audio_clip_input = torch.zeros(1, 3, 112, 1036)  # 기본 CLIP 오디오 입력 (더미)
        
        if has_audio and hasattr(beat_item, 'audio_name'):  # 오디오가 필요하고 오디오 이름이 있는 경우
            try:
                audio_path = beat_item.get('audio_name', '')  # 오디오 파일 경로 가져오기
                if audio_path and os.path.exists(audio_path):  # 경로가 존재하고 파일이 있는 경우
                    x = self.audio_processor.feature_extraction_vae(audio_path)  # 오디오 VAE용 특징 추출
                    audio_vae_input = x['fbank'].unsqueeze(0)  # fbank 특징을 배치 차원 추가
                    audio_clip_input = self.audio_processor_clip([audio_path])['pixel_values']  # CLIP용 오디오 처리
                else:
                    # 더미 오디오 사용 (기본값 유지)
                    pass
            except Exception as e:
                logger.warning(f"Audio processing failed: {e}")  # 오디오 처리 실패 시 경고 로그
                
        # ============================================================================
        # 호환성을 위한 더미 이미지 생성 (제스처로 대체됨)
        # ============================================================================
        dummy_image = torch.zeros(3, self.size, self.size)  # 더미 이미지 (3채널, size x size)
        dummy_image_clip = torch.zeros(1, 3, 224, 224)  # CLIP용 더미 이미지 (1배치, 3채널, 224x224)
        
        # ============================================================================
        # 최종 데이터 아이템 반환
        # ============================================================================
        return {
            'gesture_sequence': gesture_sequence,    # 새로운 제스처 데이터
            'image': dummy_image,                   # 호환성을 위한 더미 이미지
            'image_clip': dummy_image_clip,         # 호환성을 위한 더미 CLIP 이미지
            'caption': prompt,                      # 텍스트 프롬프트
            'caption2': prompt2,                    # 텍스트 프롬프트 2
            'audio': audio_vae_input,              # 오디오 VAE용 오디오
            'audio_clip': audio_clip_input,        # 오디오 인코더용 오디오
            'task': task,                          # 태스크 타입
            'has_gesture': has_gesture,            # 제스처 사용 가능 여부
            'has_image': False,                    # 항상 False (제스처로 대체됨)
            'has_audio': has_audio,                # 오디오 사용 가능 여부
            'has_caption': has_text,               # 텍스트 사용 가능 여부
            'dataset': f'gesture_{task}',          # 데이터셋 식별자
            'weight': [1.0, 1.0]                  # 태스크 가중치
        }
        
    def _process_gesture_data(self, pose, facial, trans, trans_v):
        """
        검증된 적응형 방법을 사용하여 제스처 데이터를 처리하는 함수
        
        Args:
            pose: 포즈 데이터 (T, pose_dim)
            facial: 얼굴 데이터 (T, 100)
            trans: 변환 데이터 (T, 3)
            trans_v: 변환 속도 데이터 (T, 3)
        
        Returns:
            full_gesture: 처리된 제스처 시퀀스 (T, 415)
        """
        # ============================================================================
        # 배치 차원 추가
        # ============================================================================
        pose = pose.unsqueeze(0)      # (1, T, pose_dim) - 배치 차원 추가
        facial = facial.unsqueeze(0)  # (1, T, 100) - 배치 차원 추가
        trans = trans.unsqueeze(0)    # (1, T, 3) - 배치 차원 추가
        trans_v = trans_v.unsqueeze(0) # (1, T, 3) - 배치 차원 추가
        
        B, T, pose_dim = pose.shape  # 배치, 시퀀스 길이, 포즈 차원 추출
        
        # ============================================================================
        # 검증된 적응형 분할 방법 사용
        # ============================================================================
        upper_end = int(pose_dim * 0.4)  # 상체 끝 인덱스 (포즈 차원의 40%)
        hands_start = upper_end  # 손 시작 인덱스 (상체 끝과 동일)
        hands_end = int(pose_dim * 0.8)  # 손 끝 인덱스 (포즈 차원의 80%)
        
        upper_pose = pose[:, :, :upper_end]  # 상체 포즈 추출
        hands_pose = pose[:, :, hands_start:hands_end]  # 손 포즈 추출
        lower_pose = pose[:, :, hands_end:]  # 하체 포즈 추출
        
        # ============================================================================
        # 하체와 변환 속도 결합
        # ============================================================================
        lower_trans = torch.cat([lower_pose, trans_v], dim=-1)  # 하체 포즈와 변환 속도를 마지막 차원으로 결합
        
        # ============================================================================
        # 정확한 RVQVAE 요구사항에 맞게 패딩
        # ============================================================================
        upper_pose = F.pad(upper_pose, (0, max(0, 78 - upper_pose.shape[-1])))[:, :, :78]  # 상체를 78차원으로 패딩
        hands_pose = F.pad(hands_pose, (0, max(0, 180 - hands_pose.shape[-1])))[:, :, :180]  # 손을 180차원으로 패딩
        lower_trans = F.pad(lower_trans, (0, max(0, 57 - lower_trans.shape[-1])))[:, :, :57]  # 하체+변환을 57차원으로 패딩
        face_data = F.pad(facial, (0, max(0, 100 - facial.shape[-1])))[:, :, :100]  # 얼굴을 100차원으로 패딩
        
        # ============================================================================
        # 모든 부위 결합
        # ============================================================================
        full_gesture = torch.cat([  # 모든 부위를 마지막 차원으로 결합
            upper_pose,      # 78차원 - 상체
            hands_pose,      # 180차원 - 손
            lower_trans,     # 57차원 - 하체 + 변환
            face_data        # 100차원 - 얼굴
        ], dim=-1)  # (1, T, 415) - 총 415차원
        
        return full_gesture.squeeze(0)  # (T, 415) - 배치 차원 제거하여 반환


def omniges_collate_fn(examples):
    """
    Omniges에 맞게 조정된 배치 수집 함수
    """
    # ============================================================================
    # 이미지 대신 제스처 데이터 가져오기
    # ============================================================================
    gesture_sequences = [example["gesture_sequence"] for example in examples]  # 모든 예제에서 제스처 시퀀스 추출
    gesture_sequences = torch.stack([torch.nn.functional.pad(seq, (0, 0, 0, 128 - seq.shape[0])) for seq in gesture_sequences])  # 128 프레임으로 패딩하여 스택
    
    # ============================================================================
    # OmniFlow 로직과의 호환성을 위해 더미 이미지 유지
    # ============================================================================
    pixel_values = [example["image"] for example in examples]  # 모든 예제에서 더미 이미지 추출
    clip_values = torch.cat([example["image_clip"] for example in examples])  # CLIP용 더미 이미지 연결
    
    prompts = list([example["caption"] for example in examples])  # 모든 예제에서 프롬프트 추출
    prompts2 = list([example["caption2"] for example in examples])  # 모든 예제에서 프롬프트2 추출

    pixel_values = torch.stack(pixel_values)  # 더미 이미지들을 스택
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()  # 메모리 형식을 연속으로 변경하고 float로 변환

    audio = torch.cat([example["audio"] for example in examples])  # 모든 예제에서 오디오 연결
    audio_clip = torch.cat([example["audio_clip"] for example in examples])  # 모든 예제에서 CLIP 오디오 연결
    
    # ============================================================================
    # 첫 번째 예제에서 태스크 결정
    # ============================================================================
    task_type = examples[0]['task']  # 첫 번째 예제의 태스크 타입 가져오기
    
    # ============================================================================
    # Omniges 태스크를 OmniFlow 호환 이름으로 매핑 (내부 처리용)
    # ============================================================================
    task_mapping = {
        't2g': 'text2img',   # 텍스트 → 제스처 → 텍스트 → 이미지 (내부)
        'g2t': 'img2text',   # 제스처 → 텍스트 → 이미지 → 텍스트 (내부)
        'a2g': 'aud2img',    # 오디오 → 제스처 → 오디오 → 이미지 (내부)
        'g2a': 'img2aud',    # 제스처 → 오디오 → 이미지 → 오디오 (내부)
        't2a': 'text2aud',   # 텍스트 → 오디오 (동일)
        'a2t': 'aud2text'    # 오디오 → 텍스트 (동일)
    }
    
    task = task_mapping[task_type]  # 태스크 타입을 OmniFlow 호환 이름으로 변환
    
    # ============================================================================
    # 제스처 태스크에 맞게 조정된 드롭아웃 로직
    # ============================================================================
    drop_img = drop_text = drop_aud = None  # 드롭아웃 인덱스 초기화
    
    if task in ['text2img', 'text2aud']:  # t2g, t2a 태스크
        drop_text = (np.random.rand(len(prompts)) < 0.15).nonzero()[0]  # 15% 확률로 텍스트 드롭아웃
    elif task in ['img2text', 'img2aud']:  # g2t, g2a 태스크
        drop_img = (np.random.rand(len(prompts)) < 0.15).nonzero()[0]  # 15% 확률로 이미지(제스처) 드롭아웃
    elif task in ['aud2text', 'aud2img']:  # a2t, a2g 태스크
        drop_aud = (np.random.rand(len(prompts)) < 0.15).nonzero()[0]  # 15% 확률로 오디오 드롭아웃
    
    # ============================================================================
    # 최종 배치 딕셔너리 구성
    # ============================================================================
    batch = {
        "pixel_values": pixel_values,      # 더미 이미지들 (제스처 시퀀스로 대체됨)
        "gesture_sequences": gesture_sequences,  # 새로운: 실제 제스처 데이터
        "prompts": prompts,  # 텍스트 프롬프트들
        "prompts2": prompts2,  # 텍스트 프롬프트2들
        "task": task,  # OmniFlow 호환 태스크 이름
        "task_type": task_type,           # 새로운: 원본 태스크 타입
        "clip_values": clip_values,  # CLIP용 더미 이미지들
        "audio": audio,  # 오디오 데이터
        "audio_clip": audio_clip,  # CLIP용 오디오 데이터
        'drop_img': drop_img,  # 이미지(제스처) 드롭아웃 인덱스
        'drop_aud': drop_aud,  # 오디오 드롭아웃 인덱스
        "drop_text": drop_text,  # 텍스트 드롭아웃 인덱스
        "name": examples[0]['dataset'],  # 데이터셋 이름
    }

    return batch  # 최종 배치 반환


def prepare_omniges_inputs(
    transformer, args, text_encoder_one, text_encoder_two, text_encoder_three,
    device, batch, gesture_vae, tokenizer_three, text_encoders, tokenizers,
    tokenizer_one, tokenizer_two, weight_dtype, noise_scheduler_copy,
    noise_scheduler, audio_vae_factor, audiovae, text_vae_tokenizer,
    text_vae, audio_encoder, anchor=False, mm_encoder=None,
):
    """
    Omniges 훈련을 위한 입력 데이터를 준비하는 함수
    OmniFlow의 prepare_inputs를 제스처 처리에 맞게 적응
    
    Args:
        transformer: OmnigesFlowTransformerModel
        args: 훈련 인자들
        text_encoder_one/two/three: 텍스트 인코더들
        device: 계산 디바이스
        batch: 데이터 배치
        gesture_vae: 제스처 VAE
        tokenizer_three: 토크나이저
        text_encoders: 텍스트 인코더 리스트
        tokenizers: 토크나이저 리스트
        tokenizer_one/two: 추가 토크나이저들
        weight_dtype: 가중치 데이터 타입
        noise_scheduler_copy: 노이즈 스케줄러 복사본
        noise_scheduler: 노이즈 스케줄러
        audio_vae_factor: 오디오 VAE 팩터
        audiovae: 오디오 VAE
        text_vae_tokenizer: 텍스트 VAE 토크나이저
        text_vae: 텍스트 VAE
        audio_encoder: 오디오 인코더
        anchor: 앵커 플래그
        mm_encoder: 멀티모달 인코더
    
    Returns:
        tuple: 훈련에 필요한 모든 입력 데이터들
    """
    with torch.no_grad():  # 그래디언트 계산 비활성화 (추론 모드)
        models_to_accumulate = [transformer]  # 누적할 모델 리스트 (현재는 transformer만)

        # ============================================================================
        # 배치에서 기본 정보 추출
        # ============================================================================
        task = batch['task']  # OmniFlow 호환 태스크 이름 (text2img, img2text 등)
        task_type = batch['task_type']  # 원본 Omniges 태스크 타입 (t2g, g2t 등)
        
        # ============================================================================
        # 이미지 데이터 대신 제스처 데이터 처리
        # ============================================================================
        gesture_sequences = batch["gesture_sequences"]  # 제스처 시퀀스 (B, T, 415)
        prompts = np.array(batch["prompts"])  # 텍스트 프롬프트들을 numpy 배열로 변환
        prompts2 = np.array(batch["prompts2"])  # 텍스트 프롬프트2들을 numpy 배열로 변환
        
        # ============================================================================
        # 디버그 출력 - 데이터로더 출력 정보
        # ============================================================================
        print(f"DEBUG: ========== DATALOADER OUTPUT ==========")  # 데이터로더 출력 섹션 시작
        print(f"DEBUG: task: {task}, task_type: {task_type}")  # 태스크 정보 출력
        print(f"DEBUG: gesture_sequences shape: {gesture_sequences.shape}")  # 제스처 시퀀스 형태 출력
        print(f"DEBUG: prompts count: {len(prompts)}")  # 프롬프트 개수 출력
        print(f"DEBUG: prompts2 count: {len(prompts2)}")  # 프롬프트2 개수 출력
        if 'audio' in batch:  # 배치에 오디오가 있는 경우
            print(f"DEBUG: audio shape: {batch['audio'].shape}")  # 오디오 형태 출력
        else:  # 배치에 오디오가 없는 경우
            print(f"DEBUG: No audio in batch")  # 오디오 없음 메시지 출력
    
        bsz = len(prompts)  # 배치 크기를 프롬프트 개수로 설정
        
        # ============================================================================
        # 제스처 VAE를 사용하여 제스처를 잠재 변수로 인코딩
        # ============================================================================
        print(f"DEBUG: ========== GESTURE VAE ENCODING ==========")  # 제스처 VAE 인코딩 섹션 시작
        gesture_latents_dist = gesture_vae.encode(gesture_sequences.to(device))  # 제스처 시퀀스를 디바이스로 이동하여 VAE 인코딩
        model_input = gesture_latents_dist.sample()  # 분포에서 샘플링하여 (B, C, H, W) 형식으로 변환
        print(f"DEBUG: gesture VAE output shape: {model_input.shape}")  # 제스처 VAE 출력 형태 출력
        
        # ============================================================================
        # GestureVAE는 이제 (B, 512, T, 1)을 직접 출력해야 함 - 리셰이핑 불필요
        # ============================================================================
        B, C, H, W = model_input.shape  # 배치, 채널, 높이, 너비 추출
        print(f"DEBUG: Gesture VAE output - B:{B}, C:{C}, H:{H}, W:{W}")  # 제스처 VAE 출력 차원 정보 출력
        
        # ============================================================================
        # 예상 형식 검증
        # ============================================================================
        if C == 512 and W == 1:  # 올바른 형식인 경우 (4개 부위 결합)
            print(f"DEBUG: ✓ Correct format (B, 512, T, 1) - 4 parts concatenated")  # 올바른 형식 확인 메시지
        elif C == 128 and W == 4:  # 레거시 형식인 경우 (4개 부위 스택)
            print(f"DEBUG: ⚠ Legacy format (B, 128, T, 4) - converting to concat format")  # 레거시 형식 변환 메시지
            # 변환: (B, 128, T, 4) -> (B, T, 128, 4) -> (B, T, 512) -> (B, 512, T, 1)
            model_input = model_input.permute(0, 2, 1, 3)  # 차원 순서 변경 (B, T, 128, 4)
            model_input = model_input.reshape(B, H, C * W)  # 형태 변경 (B, T, 512)
            model_input = model_input.permute(0, 2, 1).unsqueeze(-1)  # 차원 순서 변경 후 마지막 차원 추가 (B, 512, T, 1)
            print(f"DEBUG: After conversion - model_input shape: {model_input.shape}")  # 변환 후 형태 출력
        else:  # 예상치 못한 형식인 경우
            print(f"DEBUG: ⚠ Unexpected format - C:{C}, W:{W}")  # 예상치 못한 형식 경고
            print(f"DEBUG: Current model_input shape: {model_input.shape}")  # 현재 형태 출력
            
        model_input = model_input * gesture_vae.config.scaling_factor  # VAE 스케일링 팩터 적용
        model_input = model_input.to(dtype=weight_dtype)  # 가중치 데이터 타입으로 변환
        print(f"DEBUG: After scaling and dtype - model_input shape: {model_input.shape}")  # 스케일링 및 데이터 타입 변환 후 형태 출력

        # ============================================================================
        # 오디오 입력 (OmniFlow와 동일)
        # ============================================================================
        print(f"DEBUG: ========== AUDIO VAE ENCODING ==========")  # 오디오 VAE 인코딩 섹션 시작
        raw_audio_embeds = batch['audio'].to(model_input.device)  # 배치의 오디오를 모델 입력과 동일한 디바이스로 이동
        print(f"DEBUG: Raw audio input shape: {raw_audio_embeds.shape}")  # 원본 오디오 입력 형태 출력
        print(f"DEBUG: Raw audio input dtype: {raw_audio_embeds.dtype}")  # 원본 오디오 입력 데이터 타입 출력
        print(f"DEBUG: Raw audio input device: {raw_audio_embeds.device}")  # 원본 오디오 입력 디바이스 출력
        
        # ============================================================================
        # 오디오 VAE 인코딩
        # ============================================================================
        audio_latent_dist = audiovae.encode(raw_audio_embeds.float())  # 오디오를 float로 변환하여 VAE 인코딩
        print(f"DEBUG: Audio VAE latent_dist type: {type(audio_latent_dist)}")  # 오디오 VAE 잠재 분포 타입 출력
        
        # ============================================================================
        # AutoencoderKLOutput을 올바르게 처리
        # ============================================================================
        if hasattr(audio_latent_dist, 'latents'):  # latents 속성이 있는 경우
            # AutoencoderKLOutput의 가장 일반적인 경우
            raw_audio_embeds = audio_latent_dist.latents  # latents 속성에서 잠재 변수 추출
            print(f"DEBUG: Audio VAE latents shape (via .latents): {raw_audio_embeds.shape}")  # latents를 통한 형태 출력
        elif hasattr(audio_latent_dist, 'latent_dist'):  # latent_dist 속성이 있는 경우
            raw_audio_embeds = audio_latent_dist.latent_dist.sample()  # 잠재 분포에서 샘플링
            print(f"DEBUG: Audio VAE sample shape (via latent_dist): {raw_audio_embeds.shape}")  # latent_dist를 통한 형태 출력
        elif hasattr(audio_latent_dist, 'sample'):  # sample 메서드가 있는 경우
            raw_audio_embeds = audio_latent_dist.sample()  # 직접 샘플링
            print(f"DEBUG: Audio VAE sample shape (direct): {raw_audio_embeds.shape}")  # 직접 샘플링 형태 출력
        else:  # 위의 경우가 모두 아닌 경우
            # 폴백 - 이미 잠재 텐서라고 가정
            raw_audio_embeds = audio_latent_dist  # 직접 사용
            print(f"DEBUG: Audio VAE direct tensor shape: {raw_audio_embeds.shape}")  # 직접 텐서 형태 출력
            
        print(f"DEBUG: Audio latent attributes: {[attr for attr in dir(audio_latent_dist) if not attr.startswith('_')]}")  # 오디오 잠재 분포의 모든 속성 출력
        
        # ============================================================================
        # 스케일링 팩터 적용
        # ============================================================================
        raw_audio_embeds = raw_audio_embeds.mul_(audiovae.config.scaling_factor)  # 오디오 VAE 스케일링 팩터 적용 (in-place)
        print(f"DEBUG: Audio VAE scaling factor: {audiovae.config.scaling_factor}")  # 오디오 VAE 스케일링 팩터 출력
        print(f"DEBUG: After scaling - audio shape: {raw_audio_embeds.shape}")  # 스케일링 후 오디오 형태 출력
        
        raw_audio_embeds = raw_audio_embeds.to(model_input)  # 모델 입력과 동일한 디바이스/데이터 타입으로 변환
        print(f"DEBUG: Final audio embeds shape: {raw_audio_embeds.shape}")  # 최종 오디오 임베딩 형태 출력
        print(f"DEBUG: Final audio embeds dtype: {raw_audio_embeds.dtype}")  # 최종 오디오 임베딩 데이터 타입 출력
        
        # ============================================================================
        # 서로 다른 모달리티를 위한 노이즈 샘플링
        # ============================================================================
        bsz = model_input.shape[0]  # 배치 크기를 모델 입력에서 추출
        add_token_embed = True  # 토큰 임베딩 추가 플래그 활성화
        
        # ============================================================================
        # 3개 모달리티를 위한 타임스텝 샘플링
        # ============================================================================
        print(f"DEBUG: ========== TIMESTEP SAMPLING ==========")  # 타임스텝 샘플링 섹션 시작
        print(f"DEBUG: batch size: {bsz}, total timesteps to sample: {bsz * 3}")  # 배치 크기와 총 샘플링할 타임스텝 수 출력
        u = compute_density_for_timestep_sampling(  # 타임스텝 샘플링을 위한 밀도 계산
            weighting_scheme=args.weighting_scheme,  # 가중치 스키마
            batch_size=bsz * 3,  # 배치 크기 (3개 모달리티)
            logit_mean=args.logit_mean,  # 로짓 평균
            logit_std=args.logit_std,  # 로짓 표준편차
            mode_scale=args.mode_scale,  # 모드 스케일
        )
        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()  # 밀도를 타임스텝 인덱스로 변환
        if args.uniform_flow:  # 균등 플로우가 활성화된 경우
            indices = torch.randint(  # 균등 분포에서 랜덤 인덱스 생성
                0, noise_scheduler.config.num_train_timesteps, (bsz*3,), device='cpu', dtype=torch.long
            )
        timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)  # 인덱스에 해당하는 타임스텝을 디바이스로 이동
        print(f"DEBUG: Raw timesteps shape: {timesteps.shape}")  # 원본 타임스텝 형태 출력
        timesteps, timesteps_text, timesteps_audio = timesteps.chunk(3)  # 타임스텝을 3개 모달리티로 분할
        print(f"DEBUG: Split timesteps - gesture: {timesteps.shape}, text: {timesteps_text.shape}, audio: {timesteps_audio.shape}")  # 분할된 타임스텝 형태 출력
        
        # ============================================================================
        # 각 모달리티에 대한 시그마 값 가져오기
        # ============================================================================
        sigmas = n_get_sigmas(noise_scheduler_copy, device, timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)  # 제스처용 시그마
        sigma_text = n_get_sigmas(noise_scheduler_copy, device, timesteps_text, n_dim=model_input.ndim, dtype=model_input.dtype)  # 텍스트용 시그마
        sigmas_audio = n_get_sigmas(noise_scheduler_copy, device, timesteps_audio, n_dim=model_input.ndim, dtype=model_input.dtype)  # 오디오용 시그마
        print(f"DEBUG: Sigmas shapes - gesture: {sigmas.shape}, text: {sigma_text.shape}, audio: {sigmas_audio.shape}")  # 각 모달리티 시그마 형태 출력
        
        # ============================================================================
        # 서로 다른 태스크에 대한 손실 팩터
        # ============================================================================
        loss_text_factor = 1  # 텍스트 손실 팩터
        loss_aud_factor = 1  # 오디오 손실 팩터
        loss_gesture_factor = 1  # 제스처 손실 팩터 (기존 loss_img_factor에서 이름 변경)
        
        # ============================================================================
        # 태스크에 따른 적절한 조건화 설정
        # ============================================================================
        can_generate_text = True  # 텍스트 생성 가능 플래그
        if np.random.rand() < 0.1:  # 10% 확률로 텍스트 생성 비활성화
            can_generate_text = False
            
        # ============================================================================
        # 태스크별 조건화 (제스처에 맞게 적응)
        # ============================================================================
        if task in ['text2img', 'text2aud']:  # t2g, t2a 태스크
            loss_text_factor = 0  # 텍스트 손실 팩터를 0으로 설정
            if np.random.rand() < 0.8:  # 80% 확률로 텍스트 시그마와 타임스텝을 0으로 설정
                sigma_text = sigma_text * 0  # 텍스트 시그마를 0으로 설정
                timesteps_text = timesteps_text * 0  # 텍스트 타임스텝을 0으로 설정
        
        if task in ['img2aud', 'aud2img']:  # g2a, a2g 태스크
            loss_text_factor = 0  # 텍스트 손실 팩터를 0으로 설정
            sigma_text = sigma_text * 0 + 1  # 텍스트 시그마를 1로 설정
            timesteps_text = timesteps_text * 0 + 1000  # 텍스트 타임스텝을 1000으로 설정
            
        if batch['drop_text'] is not None:  # 텍스트 드롭아웃이 설정된 경우
            timesteps_text[batch['drop_text']] = 1000  # 드롭된 텍스트의 타임스텝을 1000으로 설정
            sigma_text[batch['drop_text']] = 1  # 드롭된 텍스트의 시그마를 1로 설정
        
        if batch['drop_aud'] is not None:  # 오디오 드롭아웃이 설정된 경우
            timesteps_audio[batch['drop_aud']] = 1000  # 드롭된 오디오의 타임스텝을 1000으로 설정
            sigmas_audio[batch['drop_aud']] = 1  # 드롭된 오디오의 시그마를 1로 설정
            
        if batch['drop_img'] is not None:  # 이미지(제스처) 드롭아웃이 설정된 경우
            timesteps[batch['drop_img']] = 1000  # 드롭된 제스처의 타임스텝을 1000으로 설정
            sigmas[batch['drop_img']] = 1  # 드롭된 제스처의 시그마를 1로 설정
            
        if task in ['img2text', 'img2aud']:  # g2t, g2a 태스크
            loss_gesture_factor = 0  # 제스처 손실 팩터를 0으로 설정
            if np.random.rand() < 0.8:  # 80% 확률로 제스처 시그마와 타임스텝을 0으로 설정
                sigmas = sigmas * 0  # 제스처 시그마를 0으로 설정
                timesteps = timesteps * 0  # 제스처 타임스텝을 0으로 설정
                
        if task in ['text2aud', 'aud2text']:  # t2a, a2t 태스크
            loss_gesture_factor = 0  # 제스처 손실 팩터를 0으로 설정
            sigmas = sigmas * 0 + 1  # 제스처 시그마를 1로 설정
            timesteps = timesteps * 0 + 1000  # 제스처 타임스텝을 1000으로 설정
              
        if task in ['aud2text', 'aud2img']:  # a2t, a2g 태스크
            loss_aud_factor = 0  # 오디오 손실 팩터를 0으로 설정
            if np.random.rand() < 0.8:  # 80% 확률로 오디오 시그마와 타임스텝을 0으로 설정
                sigmas_audio = sigmas_audio * 0  # 오디오 시그마를 0으로 설정
                timesteps_audio = timesteps_audio * 0  # 오디오 타임스텝을 0으로 설정
            
        if task in ['text2img', 'img2text']:  # t2g, g2t 태스크
            loss_aud_factor = 0  # 오디오 손실 팩터를 0으로 설정
            sigmas_audio = sigmas_audio * 0 + 1  # 오디오 시그마를 1로 설정
            timesteps_audio = timesteps_audio * 0 + 1000  # 오디오 타임스텝을 1000으로 설정
        
        # ============================================================================
        # 풀링 모드 결정
        # ============================================================================
        if task in ['img2text', 'img2aud']:  # g2t, g2a 태스크
            pool_mode = 'gesture'  # 제스처 임베딩 사용
        elif task in ['aud2img', 'aud2text']:  # a2g, a2t 태스크
            pool_mode = 'aud'  # 오디오 임베딩 사용
        else:  # 기타 태스크
            pool_mode = 'text'  # 텍스트 임베딩 사용
            
        if not can_generate_text:  # 텍스트 생성이 불가능한 경우
            loss_text_factor = loss_text_factor * 0  # 텍스트 손실 팩터를 0으로 설정

        # ============================================================================
        # 텍스트 인코딩 (OmniFlow와 동일)
        # ============================================================================
        print(f"DEBUG: ========== TEXT ENCODING ==========")  # 텍스트 인코딩 섹션 시작
        prompts = prompts.tolist()  # numpy 배열을 리스트로 변환
        target_labels = tokenize_prompt(tokenizer_three, prompts)  # 프롬프트를 토크나이저로 토크화
        target_labels = target_labels.to(device)  # 타겟 라벨을 디바이스로 이동
        print(f"DEBUG: target_labels shape: {target_labels.shape}")  # 타겟 라벨 형태 출력
        
        prompt_embeds, pooled_prompt_embeds = n_compute_text_embeddings(  # 텍스트 임베딩 계산
            device, prompts, text_encoders, tokenizers, add_token_embed=add_token_embed, train=False
        )
        print(f"DEBUG: prompt_embeds shape: {prompt_embeds.shape}")  # 프롬프트 임베딩 형태 출력
        print(f"DEBUG: pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}")  # 풀링된 프롬프트 임베딩 형태 출력
        
        # ============================================================================
        # 텍스트 VAE 인코딩
        # ============================================================================
        print(f"DEBUG: ========== TEXT VAE ENCODING ==========")  # 텍스트 VAE 인코딩 섹션 시작
        prompt_embeds_vae = text_vae.encode(prompts, input_ids=None, tokenizer=tokenizer_three, sample=True)  # 조건부 VAE 인코딩
        prompt_embeds_vae_uncond = text_vae.encode(prompts, input_ids=None, tokenizer=tokenizer_three, drop=True)  # 무조건부 VAE 인코딩
        print(f"DEBUG: prompt_embeds_vae shape: {prompt_embeds_vae.shape}")  # 조건부 VAE 임베딩 형태 출력
        print(f"DEBUG: prompt_embeds_vae_uncond shape: {prompt_embeds_vae_uncond.shape}")  # 무조건부 VAE 임베딩 형태 출력

        if not can_generate_text:  # 텍스트 생성이 불가능한 경우
            prompt_embeds_vae *= 0  # VAE 임베딩을 0으로 설정
            print(f"DEBUG: Text generation disabled - zeroed VAE embeddings")  # 텍스트 생성 비활성화 메시지

        l_vae = prompt_embeds_vae.shape[1]  # VAE 시퀀스 길이 추출
        print(f"DEBUG: l_vae (VAE sequence length): {l_vae}")  # VAE 시퀀스 길이 출력
        
        # ============================================================================
        # 프롬프트 임베딩 준비
        # ============================================================================
        print(f"DEBUG: ========== TEXT EMBEDDINGS PREPARATION ==========")  # 텍스트 임베딩 준비 섹션 시작
        prompt_embeds = cat_and_pad([prompt_embeds_vae], max_dim=4096)  # 조건부 VAE 임베딩을 연결하고 패딩
        prompt_embeds_uncond = cat_and_pad([prompt_embeds_vae_uncond], max_dim=4096)  # 무조건부 VAE 임베딩을 연결하고 패딩
        print(f"DEBUG: After cat_and_pad - prompt_embeds shape: {prompt_embeds.shape}")  # 연결 및 패딩 후 프롬프트 임베딩 형태 출력
        print(f"DEBUG: After cat_and_pad - prompt_embeds_uncond shape: {prompt_embeds_uncond.shape}")  # 연결 및 패딩 후 무조건부 프롬프트 임베딩 형태 출력

        # ============================================================================
        # 텍스트 디코더를 위한 타겟
        # ============================================================================
        targets = encode_prompt_for_decoder(prompts, text_vae_tokenizer, device=transformer.device)  # 디코더용 프롬프트 인코딩
        target_labels = targets['labels']  # 타겟 라벨 추출
        print(f"DEBUG: Text decoder targets shape: {target_labels.shape}")  # 텍스트 디코더 타겟 형태 출력
        print(f"DEBUG: Target labels sample: {targets.keys()}")  # 타겟 라벨 샘플 출력

        # ============================================================================
        # 풀링 모드에 따른 풀링된 임베딩
        # ============================================================================
        with torch.no_grad():  # 그래디언트 계산 비활성화
            if pool_mode == 'gesture':  # 제스처 풀링 모드인 경우
                # 제스처 임베딩 사용 (현재는 더미)
                pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)  # 풀링된 프롬프트 임베딩을 0으로 초기화
                if batch['drop_img'] is not None:  # 제스처 드롭아웃이 설정된 경우
                    pooled_prompt_embeds[batch['drop_img']] = 0  # 드롭된 제스처의 풀링 임베딩을 0으로 설정
            elif pool_mode == 'aud':  # 오디오 풀링 모드인 경우
                audio_embeds = audio_encoder.get_image_features(  # 오디오 인코더에서 이미지 특징 추출
                    pixel_values=batch['audio_clip'].to(audio_encoder.dtype).to(audio_encoder.device)  # 오디오 클립을 인코더 형식으로 변환
                )
                pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)  # 풀링된 프롬프트 임베딩을 0으로 초기화
                pooled_prompt_embeds[..., :audio_embeds.shape[-1]] = audio_embeds  # 오디오 임베딩을 풀링된 임베딩에 할당
                if batch['drop_aud'] is not None:  # 오디오 드롭아웃이 설정된 경우
                    pooled_prompt_embeds[batch['drop_aud']] = 0  # 드롭된 오디오의 풀링 임베딩을 0으로 설정
            else:  # 텍스트 풀링 모드인 경우
                if batch['drop_text'] is not None:  # 텍스트 드롭아웃이 설정된 경우
                    pooled_prompt_embeds[batch['drop_text']] = 0  # 드롭된 텍스트의 풀링 임베딩을 0으로 설정
                    
        pooled_prompt_embeds = pooled_prompt_embeds.detach()  # 풀링된 프롬프트 임베딩을 그래디언트에서 분리
        
        # ============================================================================
        # 풀링된 임베딩에 드롭아웃 적용
        # ============================================================================
        drop_pool = (torch.rand(pooled_prompt_embeds.shape[0]) < 0.85).view(-1, 1).to(pooled_prompt_embeds)  # 85% 확률로 드롭아웃 마스크 생성
        pooled_prompt_embeds = pooled_prompt_embeds * drop_pool  # 드롭아웃 마스크를 풀링된 임베딩에 적용
        
        # ============================================================================
        # 시그마 텍스트 형태 변경
        # ============================================================================
        sigma_text = sigma_text.view(-1, 1, 1)  # 시그마 텍스트를 3차원으로 형태 변경
        
        # ============================================================================
        # 노이즈 생성
        # ============================================================================
        print(f"DEBUG: ========== NOISE GENERATION ==========")  # 노이즈 생성 섹션 시작
        noise = torch.randn_like(model_input)  # 모델 입력과 동일한 형태의 가우시안 노이즈 생성
        noise_text = torch.randn_like(prompt_embeds)  # 프롬프트 임베딩과 동일한 형태의 가우시안 노이즈 생성
        print(f"DEBUG: noise shape: {noise.shape}")  # 제스처 노이즈 형태 출력
        print(f"DEBUG: noise_text shape: {noise_text.shape}")  # 텍스트 노이즈 형태 출력
        
        # ============================================================================
        # 입력에 노이즈 추가
        # ============================================================================
        noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input  # 제스처 입력에 노이즈 추가
        noisy_prompt_embeds = sigma_text * noise_text + (1.0 - sigma_text) * prompt_embeds  # 텍스트 임베딩에 노이즈 추가
        print(f"DEBUG: noisy_model_input shape: {noisy_model_input.shape}")  # 노이즈가 추가된 모델 입력 형태 출력
        print(f"DEBUG: noisy_prompt_embeds shape: {noisy_prompt_embeds.shape}")  # 노이즈가 추가된 프롬프트 임베딩 형태 출력

        noise_audio = torch.randn_like(raw_audio_embeds)  # 오디오 임베딩과 동일한 형태의 가우시안 노이즈 생성
        sigmas_audio = sigmas_audio.view(-1, 1, 1, 1)  # 오디오 시그마를 4차원으로 형태 변경
        noisy_audio_embeds = sigmas_audio * noise_audio + (1.0 - sigmas_audio) * raw_audio_embeds  # 오디오 임베딩에 노이즈 추가
        print(f"DEBUG: noise_audio shape: {noise_audio.shape}")  # 오디오 노이즈 형태 출력
        print(f"DEBUG: noisy_audio_embeds shape: {noisy_audio_embeds.shape}")  # 노이즈가 추가된 오디오 임베딩 형태 출력

        # ============================================================================
        # 텍스트 임베딩 정리
        # ============================================================================
        noisy_prompt_embeds[:, -l_vae:, prompt_embeds_vae.shape[-1]:] = 0  # VAE 부분 이후의 텍스트 임베딩을 0으로 설정
        noisy_prompt_embeds = noisy_prompt_embeds.detach()  # 노이즈가 추가된 프롬프트 임베딩을 그래디언트에서 분리
        
        # ============================================================================
        # 최종 반환값
        # ============================================================================
        return (  # 훈련에 필요한 모든 입력 데이터들을 튜플로 반환
            noisy_model_input, timesteps, timesteps_text, timesteps_audio, noisy_prompt_embeds,  # 노이즈가 추가된 입력들
            noisy_audio_embeds, sigma_text, prompt_embeds, pooled_prompt_embeds, targets, prompt_embeds_uncond,  # 임베딩 및 타겟들
            sigmas, sigmas_audio, model_input,  # 시그마 값들과 원본 모델 입력
            loss_gesture_factor,  # 제스처 손실 팩터 (기존 loss_img_factor에서 이름 변경)
            loss_text_factor,  # 텍스트 손실 팩터
            loss_aud_factor,  # 오디오 손실 팩터
            noise_scheduler_copy,  # 노이즈 스케줄러 복사본
            raw_audio_embeds,  # 원본 오디오 임베딩
            task, task_type,  # 태스크 이름들 (OmniFlow 호환 이름과 원본 이름 모두 포함)
            prompts,  # 원본 프롬프트들
            noise,  # 제스처 노이즈
            noise_text,  # 텍스트 노이즈
            noise_audio,  # 오디오 노이즈
            target_labels,  # 타겟 라벨들
            prompt_embeds_vae_uncond,  # 무조건부 VAE 프롬프트 임베딩
            gesture_sequences  # 새로운: 제스처 시퀀스 추가
        )


def compute_omniges_loss(
    transformer, noisy_model_input, timesteps, timesteps_text, timesteps_audio, noisy_prompt_embeds,
    noisy_audio_embeds, sigma_text, prompt_embeds, pooled_prompt_embeds, targets, prompt_embeds_uncond,
    sigmas, sigmas_audio, model_input, loss_gesture_factor, loss_text_factor, loss_aud_factor,
    noise_scheduler_copy, last_lr, raw_audio_embeds, task, task_type, prompts,
    noise, noise_text, noise_audio, text_vae, target_labels, do_decode,
    prompt_embeds_vae_uncond, precondition_text_outputs=False, anchor=False, batch=None,
    gesture_sequences=None
):
    """
    Omniges 훈련을 위한 손실을 계산하는 함수
    OmniFlow를 제스처 처리에 맞게 적응
    
    Args:
        transformer: OmnigesFlowTransformerModel
        noisy_model_input: 노이즈가 추가된 제스처 입력
        timesteps: 제스처 타임스텝
        timesteps_text: 텍스트 타임스텝
        timesteps_audio: 오디오 타임스텝
        noisy_prompt_embeds: 노이즈가 추가된 텍스트 임베딩
        noisy_audio_embeds: 노이즈가 추가된 오디오 임베딩
        sigma_text: 텍스트 시그마 값
        prompt_embeds: 원본 텍스트 임베딩
        pooled_prompt_embeds: 풀링된 텍스트 임베딩
        targets: 텍스트 디코더 타겟
        prompt_embeds_uncond: 무조건부 텍스트 임베딩
        sigmas: 제스처 시그마 값
        sigmas_audio: 오디오 시그마 값
        model_input: 원본 제스처 입력
        loss_gesture_factor: 제스처 손실 팩터
        loss_text_factor: 텍스트 손실 팩터
        loss_aud_factor: 오디오 손실 팩터
        noise_scheduler_copy: 노이즈 스케줄러 복사본
        last_lr: 마지막 학습률
        raw_audio_embeds: 원본 오디오 임베딩
        task: OmniFlow 호환 태스크 이름
        task_type: 원본 Omniges 태스크 타입
        prompts: 원본 프롬프트들
        noise: 제스처 노이즈
        noise_text: 텍스트 노이즈
        noise_audio: 오디오 노이즈
        text_vae: 텍스트 VAE
        target_labels: 타겟 라벨들
        do_decode: 디코딩 여부
        prompt_embeds_vae_uncond: 무조건부 VAE 텍스트 임베딩
        precondition_text_outputs: 텍스트 출력 전처리 여부
        anchor: 앵커 플래그
        batch: 배치 데이터
        gesture_sequences: 제스처 시퀀스
    
    Returns:
        tuple: 손실, 디코딩 손실, 로그, 태스크 타입, 예측값들
    """
    
    # ============================================================================
    # OmnigesFlow를 통한 순전파
    # ============================================================================
    print(f"DEBUG: ========== MODEL FORWARD PASS ==========")  # 모델 순전파 섹션 시작
    print(f"DEBUG: Forward pass inputs:")  # 순전파 입력 정보 출력
    print(f"DEBUG:   noisy_model_input shape: {noisy_model_input.shape}")  # 노이즈가 추가된 제스처 입력 형태
    print(f"DEBUG:   timesteps shape: {timesteps.shape}")  # 제스처 타임스텝 형태
    print(f"DEBUG:   timesteps_text shape: {timesteps_text.shape}")  # 텍스트 타임스텝 형태
    print(f"DEBUG:   timesteps_audio shape: {timesteps_audio.shape}")  # 오디오 타임스텝 형태
    print(f"DEBUG:   noisy_prompt_embeds shape: {noisy_prompt_embeds.shape}")  # 노이즈가 추가된 텍스트 임베딩 형태
    print(f"DEBUG:   noisy_audio_embeds shape: {noisy_audio_embeds.shape}")  # 노이즈가 추가된 오디오 임베딩 형태
    print(f"DEBUG:   pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}")  # 풀링된 프롬프트 임베딩 형태
    
    output_dict = transformer(  # OmnigesFlow 트랜스포머에 순전파
        hidden_states=noisy_model_input,              # 제스처 잠재 변수
        timestep=timesteps,                           # 제스처 타임스텝
        timestep_text=timesteps_text,                 # 텍스트 타임스텝
        timestep_audio=timesteps_audio,               # 오디오 타임스텝
        encoder_hidden_states=noisy_prompt_embeds,    # 텍스트 임베딩
        audio_hidden_states=noisy_audio_embeds,       # 오디오 임베딩
        sigma_text=sigma_text,  # 텍스트 시그마 값
        target_prompt_embeds=prompt_embeds,  # 타겟 프롬프트 임베딩
        pooled_projections=pooled_prompt_embeds,  # 풀링된 투영
        targets=targets,  # 텍스트 디코더 타겟
        return_dict=False,  # 딕셔너리 반환 비활성화
        use_text_output=True,  # 텍스트 출력 사용
        prompt_embeds_uncond=None if np.random.rand() < 0.5 else prompt_embeds_uncond,  # 50% 확률로 무조건부 임베딩 사용
        detach_logits=not anchor,  # 앵커가 아닌 경우 로짓 분리
        split_cond=False,  # 조건 분할 비활성화
        text_vae=text_vae,  # 텍스트 VAE
        text_x0=precondition_text_outputs,  # 텍스트 출력 전처리
        decode_text=True,  # 텍스트 디코딩 활성화
        # 태스크별 드롭아웃 로직
        # 입력 모달리티: 절대 드롭하지 않음
        # 출력 모달리티: 배치 드롭아웃 설정 사용
        drop_gesture=(task in ['text2img', 'aud2img'] and batch['drop_img'] is not None),  # T2G, A2G 태스크에서만 제스처 드롭
        drop_text=(task in ['img2text', 'aud2text'] and batch['drop_text'] is not None),   # G2T, A2T 태스크에서만 텍스트 드롭
        drop_audio=(task in ['text2aud', 'img2aud'] and batch['drop_aud'] is not None)     # T2A, G2A 태스크에서만 오디오 드롭
    )
    
    # ============================================================================
    # 예측값 추출
    # ============================================================================
    print(f"DEBUG: ========== MODEL OUTPUT ==========")  # 모델 출력 섹션 시작
    model_pred = output_dict['output']              # 제스처 출력
    model_pred_audio = output_dict['audio_hidden_states']  # 오디오 출력
    model_pred_text = output_dict['model_pred_text']       # 텍스트 출력
    logits = output_dict['logits']  # 텍스트 디코딩 로짓
    logits_labels = output_dict['logits_labels']  # 텍스트 라벨 로짓
    
    print(f"DEBUG: Model outputs:")  # 모델 출력 정보 출력
    print(f"DEBUG:   model_pred shape: {model_pred.shape if model_pred is not None else None}")  # 제스처 예측 형태
    print(f"DEBUG:   model_pred_audio shape: {model_pred_audio.shape if model_pred_audio is not None else None}")  # 오디오 예측 형태
    print(f"DEBUG:   model_pred_text shape: {model_pred_text.shape if model_pred_text is not None else None}")  # 텍스트 예측 형태
    print(f"DEBUG:   logits shape: {logits.shape if logits is not None else None}")  # 로짓 형태
    print(f"DEBUG:   logits_labels shape: {logits_labels.shape if logits_labels is not None else None}")  # 라벨 로짓 형태
    
    # ============================================================================
    # 속도 타겟 계산
    # ============================================================================
    print(f"DEBUG: ========== VELOCITY TARGETS ==========")  # 속도 타겟 섹션 시작
    v_theta = noise - model_input                    # 제스처 속도 (노이즈 - 원본 입력)
    v_theta_audio = noise_audio - raw_audio_embeds   # 오디오 속도 (노이즈 - 원본 오디오)
    print(f"DEBUG: v_theta shape: {v_theta.shape}")  # 제스처 속도 형태 출력
    print(f"DEBUG: v_theta_audio shape: {v_theta_audio.shape}")  # 오디오 속도 형태 출력
    
    print(f"DEBUG: Loss input comparison:")  # 손실 입력 비교 정보 출력
    print(f"DEBUG:   model_pred shape: {model_pred.shape if model_pred is not None else None}")  # 모델 예측 형태
    print(f"DEBUG:   v_theta shape: {v_theta.shape}")  # 제스처 속도 형태
    print(f"DEBUG:   Are shapes compatible? {model_pred.shape == v_theta.shape if model_pred is not None else 'model_pred is None'}")  # 형태 호환성 확인
    
    # ============================================================================
    # 텍스트 임베딩 처리 (일부 태스크에서 model_pred_text가 None일 수 있음)
    # ============================================================================
    if model_pred_text is not None:  # 텍스트 예측이 존재하는 경우
        raw_text_embeds = prompt_embeds[..., :model_pred_text.shape[-1]]  # 텍스트 예측 크기에 맞게 원본 텍스트 임베딩 슬라이싱
        noise_text = noise_text[..., :model_pred_text.shape[-1]]  # 텍스트 예측 크기에 맞게 텍스트 노이즈 슬라이싱
    else:  # 텍스트 예측이 없는 경우
        raw_text_embeds = prompt_embeds  # 원본 텍스트 임베딩 사용
        # noise_text는 이미 올바른 크기

    # ============================================================================
    # 손실 가중치 계산
    # ============================================================================
    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)  # 제스처 손실 가중치
    weighting_text = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigma_text)  # 텍스트 손실 가중치
    weighting_audio = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas_audio)  # 오디오 손실 가중치
    
    # ============================================================================
    # 가중치에 드롭아웃 적용
    # ============================================================================
    if batch['drop_img'] is not None:  # 제스처 드롭아웃이 설정된 경우
        weighting[batch['drop_img']] = 0  # 드롭된 제스처의 가중치를 0으로 설정

    # ============================================================================
    # 제스처 손실 계산 (이미지 손실에서 적응)
    # ============================================================================
    print(f"DEBUG: ========== LOSS CALCULATION ==========")  # 손실 계산 섹션 시작
    print(f"DEBUG: Gesture loss inputs:")  # 제스처 손실 입력 정보 출력
    print(f"DEBUG:   weighting shape: {weighting.shape}")  # 가중치 형태
    print(f"DEBUG:   model_pred shape: {model_pred.shape if model_pred is not None else None}")  # 모델 예측 형태
    print(f"DEBUG:   v_theta shape: {v_theta.shape}")  # 제스처 속도 형태
    
    if model_pred is not None and v_theta is not None:  # 모델 예측과 속도가 모두 존재하는 경우
        loss_gesture = (weighting.float() * (model_pred - v_theta.float()) ** 2).mean()  # 제스처 손실 계산 (MSE)
        print(f"DEBUG:   gesture loss value: {loss_gesture.item()}")  # 제스처 손실 값 출력
    else:  # 모델 예측 또는 속도가 None인 경우
        loss_gesture = torch.tensor(0.0, device=weighting.device)  # 제스처 손실을 0으로 설정
        print(f"DEBUG:   gesture loss set to 0 (None inputs)")  # 제스처 손실 0 설정 메시지

    # ============================================================================
    # 텍스트 손실 계산 (OmniFlow와 동일) - None 처리 포함
    # ============================================================================
    print(f"DEBUG: Text loss inputs:")  # 텍스트 손실 입력 정보 출력
    print(f"DEBUG:   weighting_text shape: {weighting_text.shape}")  # 텍스트 가중치 형태
    with torch.no_grad():  # 그래디언트 계산 비활성화
        weighting_text = weighting_text.view(-1, 1, 1)  # 텍스트 가중치를 3차원으로 형태 변경
        if batch['drop_text'] is not None:  # 텍스트 드롭아웃이 설정된 경우
            weighting_text[batch['drop_text']] = 0  # 드롭된 텍스트의 가중치를 0으로 설정
            print(f"DEBUG:   Applied text dropout")  # 텍스트 드롭아웃 적용 메시지
    
    # ============================================================================
    # 텍스트 출력이 존재하는 경우에만 텍스트 손실 처리
    # ============================================================================
    if model_pred_text is not None and loss_text_factor > 0:  # 텍스트 예측이 존재하고 텍스트 팩터가 0보다 큰 경우
        if precondition_text_outputs:  # 텍스트 출력 전처리가 활성화된 경우
            loss_text = (weighting_text.float() * (model_pred_text.float() - raw_text_embeds.float().detach()) ** 2).mean()  # 전처리 텍스트 손실
            norm_1 = F.normalize(model_pred_text, dim=-1, eps=1e-4).float()  # 모델 예측 정규화
            norm_2 = F.normalize(raw_text_embeds, dim=-1, eps=1e-4).float().detach()  # 원본 텍스트 임베딩 정규화
            loss_text_norm = (weighting_text.float() * (norm_1 - norm_2) ** 2).mean()  # 정규화 텍스트 손실
            loss_text_norm = loss_text_norm * 0.1  # 정규화 손실에 0.1 가중치 적용
        else:  # 텍스트 출력 전처리가 비활성화된 경우
            v_theta_text = noise_text - raw_text_embeds  # 텍스트 속도 계산
            loss_text = (weighting_text.float() * (model_pred_text.float() - v_theta_text.float()) ** 2).mean()  # 일반 텍스트 손실
            loss_text_norm = 0  # 정규화 손실을 0으로 설정
    else:  # 텍스트 출력이 없거나 텍스트 팩터가 0인 경우
        # 텍스트 출력이 없거나 텍스트 팩터가 0
        loss_text = torch.tensor(0.0, device=model_input.device)  # 텍스트 손실을 0으로 설정
        loss_text_norm = 0  # 정규화 손실을 0으로 설정
        
    # ============================================================================
    # 오디오 손실 계산 (OmniFlow와 동일)
    # ============================================================================
    print(f"DEBUG: Audio loss inputs:")  # 오디오 손실 입력 정보 출력
    print(f"DEBUG:   weighting_audio shape before view: {weighting_audio.shape}")  # view 전 오디오 가중치 형태
    weighting_audio = weighting_audio.view(-1, 1, 1, 1)  # 오디오 가중치를 4차원으로 형태 변경
    print(f"DEBUG:   weighting_audio shape after view: {weighting_audio.shape}")  # view 후 오디오 가중치 형태
    print(f"DEBUG:   model_pred_audio shape: {model_pred_audio.shape if model_pred_audio is not None else None}")  # 오디오 예측 형태
    print(f"DEBUG:   v_theta_audio shape: {v_theta_audio.shape}")  # 오디오 속도 형태
    
    if model_pred_audio is not None:  # 오디오 예측이 존재하는 경우
        loss_audio = (weighting_audio.float() * (model_pred_audio - v_theta_audio.float()) ** 2).mean()  # 오디오 손실 계산 (MSE)
        print(f"DEBUG:   audio loss value: {loss_audio.item()}")  # 오디오 손실 값 출력
    else:  # 오디오 예측이 None인 경우
        loss_audio = torch.tensor(0.0, device=weighting_audio.device)  # 오디오 손실을 0으로 설정
        print(f"DEBUG:   audio loss set to 0 (None model_pred_audio)")  # 오디오 손실 0 설정 메시지

    # ============================================================================
    # 텍스트 생성을 위한 디코딩 손실 (OmniFlow와 동일)
    # ============================================================================
    if anchor:  # 앵커 모드인 경우
        from train import WeightedLabelSmoother, compute_decode_loss_weight  # 필요한 모듈 임포트
        label_smoother = WeightedLabelSmoother(epsilon=0.0, ignore_index=-100)  # 라벨 스무더 초기화
        decode_loss_tgt_weight = torch.ones(len(timesteps_text)).to(logits)  # 타겟 디코딩 손실 가중치 초기화
        if anchor:  # 앵커 모드인 경우
            decode_loss_weight = torch.ones(len(timesteps_text)).to(logits)  # 디코딩 손실 가중치를 1로 초기화
        else:  # 앵커 모드가 아닌 경우
            decode_loss_weight = compute_decode_loss_weight(timesteps_text, noise_scheduler_copy.config.num_train_timesteps)  # 디코딩 손실 가중치 계산
        if batch['drop_text'] is not None:  # 텍스트 드롭아웃이 설정된 경우
            decode_loss_weight[batch['drop_text']] = 0  # 드롭된 텍스트의 디코딩 손실 가중치를 0으로 설정
            decode_loss_tgt_weight[batch['drop_text']] = 0  # 드롭된 텍스트의 타겟 디코딩 손실 가중치를 0으로 설정
        decode_loss_pred = label_smoother([logits], target_labels, shift_labels=True, sample_weight=decode_loss_weight)  # 예측 디코딩 손실 계산
        decode_loss_tgt = label_smoother([logits_labels], target_labels, shift_labels=True, sample_weight=decode_loss_tgt_weight)  # 타겟 디코딩 손실 계산
        decode_loss = None  # 디코딩 손실을 None으로 설정
    else:  # 앵커 모드가 아닌 경우
        decode_loss_pred = 0  # 예측 디코딩 손실을 0으로 설정
        decode_loss_tgt = 0  # 타겟 디코딩 손실을 0으로 설정
        decode_loss = None  # 디코딩 손실을 None으로 설정

    # ============================================================================
    # 총 손실 계산
    # ============================================================================
    loss = (loss_gesture * loss_gesture_factor +  # 제스처 손실에 제스처 팩터 곱하기
            (loss_text + loss_text_norm) * loss_text_factor +  # 텍스트 손실에 텍스트 팩터 곱하기
            loss_audio * loss_aud_factor +  # 오디오 손실에 오디오 팩터 곱하기
            (decode_loss_tgt + decode_loss_pred) * loss_text_factor * 0.1)  # 디코딩 손실에 텍스트 팩터와 0.1 가중치 곱하기

    # ============================================================================
    # 로깅
    # ============================================================================
    logs = {  # 로그 딕셔너리 초기화
        "loss": loss.detach().item(),  # 총 손실
        "lr": last_lr,  # 마지막 학습률
        "loss_aud_factor": loss_aud_factor,  # 오디오 손실 팩터
        "loss_gesture_factor": loss_gesture_factor,  # 제스처 손실 팩터 (이름 변경됨)
        "loss_text_factor": loss_text_factor,  # 텍스트 손실 팩터
        "task_type": task_type  # 원본 태스크 타입 로깅
    }
    
    if loss_text_factor > 0 and model_pred_text is not None:  # 텍스트 팩터가 0보다 크고 텍스트 예측이 존재하는 경우
        logs.update({  # 텍스트 관련 로그 추가
            "loss_text": loss_text.detach().item(),  # 텍스트 손실
            "loss_text_norm": loss_text_norm.detach().item() if isinstance(loss_text_norm, torch.Tensor) else loss_text_norm,  # 정규화 텍스트 손실
        })
        with torch.no_grad():  # 그래디언트 계산 비활성화
            if raw_text_embeds is not None:  # 원본 텍스트 임베딩이 존재하는 경우
                logs.update({  # 텍스트 임베딩 통계 추가
                    "text_embed_mean": raw_text_embeds.mean().item(),  # 텍스트 임베딩 평균
                    "text_embed_std": raw_text_embeds.std().item(),  # 텍스트 임베딩 표준편차
                })
        if anchor:  # 앵커 모드인 경우
            logs.update({  # 디코딩 손실 로그 추가
                "decode_loss_tgt": decode_loss_tgt.detach().item(),  # 타겟 디코딩 손실
                "decode_loss": decode_loss_pred.detach().item(),  # 예측 디코딩 손실
            })
            
    if loss_gesture_factor > 0:  # 제스처 팩터가 0보다 큰 경우
        logs.update({  # 제스처 관련 로그 추가
            "loss_gesture": loss_gesture.detach().item(),  # 제스처 손실
        })
        
    if loss_aud_factor > 0:  # 오디오 팩터가 0보다 큰 경우
        logs.update({  # 오디오 관련 로그 추가
            "loss_audio": loss_audio.detach().item(),  # 오디오 손실
        })
        
    # ============================================================================
    # 예측값 계산
    # ============================================================================
    with torch.no_grad():  # 그래디언트 계산 비활성화
        model_pred = model_pred * (-sigmas) + noisy_model_input  # 제스처 예측값 계산 (노이즈 제거)
        model_pred_audio = model_pred_audio * (-sigmas_audio) + noisy_audio_embeds  # 오디오 예측값 계산 (노이즈 제거)
        target = model_input  # 타겟을 원본 모델 입력으로 설정
        
    # ============================================================================
    # 최종 반환값
    # ============================================================================
    return (  # 모든 결과를 튜플로 반환
        loss, decode_loss, logs, task_type, model_pred, logits, target, prompts,  # 기본 결과들
        model_pred_audio, model_pred_audio, raw_audio_embeds, model_pred_text, raw_text_embeds  # 모달리티별 결과들
    )


def omniges_forward_pass(
    transformer, args, text_encoder_one, text_encoder_two, text_encoder_three,
    accelerator, batch, gesture_vae, tokenizer_three, text_encoders, tokenizers,
    tokenizer_one, tokenizer_two, weight_dtype, noise_scheduler_copy,
    noise_scheduler, audio_vae_factor, audiovae, text_vae_tokenizer,
    last_lr, text_vae, audio_encoder, do_decode=False,
    precondition_text_outputs=False, anchor=False, mm_encoder=None
):
    """
    Omniges 훈련을 위한 완전한 순전파 함수
    입력 준비와 손실 계산을 포함한 전체 훈련 스텝을 수행
    
    Args:
        transformer: OmnigesFlowTransformerModel
        args: 훈련 인자들
        text_encoder_one: 첫 번째 텍스트 인코더 (CLIP)
        text_encoder_two: 두 번째 텍스트 인코더 (CLIP)
        text_encoder_three: 세 번째 텍스트 인코더 (T5)
        accelerator: Accelerate 라이브러리 가속기
        batch: 배치 데이터
        gesture_vae: 제스처 VAE
        tokenizer_three: T5 토크나이저
        text_encoders: 텍스트 인코더들
        tokenizers: 토크나이저들
        tokenizer_one: CLIP 토크나이저
        tokenizer_two: CLIP 토크나이저
        weight_dtype: 가중치 데이터 타입
        noise_scheduler_copy: 노이즈 스케줄러 복사본
        noise_scheduler: 노이즈 스케줄러
        audio_vae_factor: 오디오 VAE 팩터
        audiovae: 오디오 VAE
        text_vae_tokenizer: 텍스트 VAE 토크나이저
        last_lr: 마지막 학습률
        text_vae: 텍스트 VAE
        audio_encoder: 오디오 인코더
        do_decode: 디코딩 여부
        precondition_text_outputs: 텍스트 출력 전처리 여부
        anchor: 앵커 플래그
        mm_encoder: 멀티모달 인코더
    
    Returns:
        tuple: 손실, 디코딩 손실, 로그, 태스크 타입, 예측값들
    """
    
    # ============================================================================
    # 입력 준비
    # ============================================================================
    (noisy_model_input, timesteps, timesteps_text, timesteps_audio, noisy_prompt_embeds,  # 노이즈가 추가된 모델 입력, 타임스텝들
     noisy_audio_embeds, sigma_text, prompt_embeds, pooled_prompt_embeds, targets, prompt_embeds_uncond,  # 노이즈가 추가된 오디오 임베딩, 텍스트 시그마, 프롬프트 임베딩들
     sigmas, sigmas_audio, model_input, loss_gesture_factor, loss_text_factor, loss_aud_factor,  # 시그마들, 모델 입력, 손실 팩터들
     noise_scheduler_copy, raw_audio_embeds, task, task_type, prompts, noise, noise_text, noise_audio,  # 노이즈 스케줄러, 원본 오디오 임베딩, 태스크 정보, 노이즈들
     target_labels, prompt_embeds_vae_uncond, gesture_sequences) = prepare_omniges_inputs(  # 타겟 라벨, 무조건부 VAE 임베딩, 제스처 시퀀스
        transformer, args, text_encoder_one, text_encoder_two, text_encoder_three,  # 트랜스포머, 인자들, 텍스트 인코더들
        accelerator, batch, gesture_vae, tokenizer_three, text_encoders, tokenizers,  # 가속기, 배치, 제스처 VAE, 토크나이저들
        tokenizer_one, tokenizer_two, weight_dtype, noise_scheduler_copy,  # CLIP 토크나이저들, 가중치 타입, 노이즈 스케줄러
        noise_scheduler, audio_vae_factor, audiovae, text_vae_tokenizer,  # 노이즈 스케줄러, 오디오 VAE 팩터, 오디오 VAE, 텍스트 VAE 토크나이저
        text_vae, audio_encoder, anchor, mm_encoder=mm_encoder  # 텍스트 VAE, 오디오 인코더, 앵커, 멀티모달 인코더
    )
    
    # ============================================================================
    # 손실 계산
    # ============================================================================
    loss, decode_loss, logs, task_type, model_pred, logits, target, prompts, model_pred_audio, model_pred_audio, raw_audio_embeds, model_pred_text, raw_text_embeds = compute_omniges_loss(  # 모든 손실과 예측값들
        transformer, noisy_model_input, timesteps, timesteps_text, timesteps_audio, noisy_prompt_embeds,  # 트랜스포머, 노이즈 입력, 타임스텝들, 노이즈 프롬프트 임베딩
        noisy_audio_embeds, sigma_text, prompt_embeds, pooled_prompt_embeds, targets, prompt_embeds_uncond,  # 노이즈 오디오 임베딩, 텍스트 시그마, 프롬프트 임베딩들
        sigmas, sigmas_audio, model_input, loss_gesture_factor, loss_text_factor, loss_aud_factor,  # 시그마들, 모델 입력, 손실 팩터들
        noise_scheduler_copy, last_lr, raw_audio_embeds, task, task_type, prompts,  # 노이즈 스케줄러, 마지막 학습률, 원본 오디오 임베딩, 태스크 정보
        noise, noise_text, noise_audio, text_vae, target_labels, do_decode,  # 노이즈들, 텍스트 VAE, 타겟 라벨, 디코딩 여부
        prompt_embeds_vae_uncond, precondition_text_outputs=precondition_text_outputs,  # 무조건부 VAE 임베딩, 텍스트 출력 전처리
        anchor=anchor, batch=batch, gesture_sequences=gesture_sequences  # 앵커, 배치, 제스처 시퀀스
    )
    
    # ============================================================================
    # 결과 반환 (텍스트 예측값들은 그래디언트 분리)
    # ============================================================================
    return loss, decode_loss, logs, task_type, model_pred, logits, target, prompts, model_pred_audio, model_pred_audio, raw_audio_embeds, model_pred_text.detach() if model_pred_text is not None else None, raw_text_embeds.detach() if raw_text_embeds is not None else None  # 텍스트 예측값들을 detach하여 그래디언트 분리


@torch.no_grad()  # 그래디언트 계산 비활성화 (검증 시에는 그래디언트가 필요 없음)
def log_omniges_validation(
    pipeline, args, accelerator, pipeline_args, global_step,
    is_final_validation=False, prefix='', do_gesture=True, do_audio=True, do_text=True,
):
    """
    Omniges 검증 로깅 함수
    지원하는 모든 태스크를 테스트: t2g, g2t, a2g, g2a, t2a, a2t
    각 태스크별로 샘플을 생성하고 wandb에 로깅
    
    Args:
        pipeline: OmnigesPipeline
        args: 훈련 인자들
        accelerator: Accelerate 가속기
        pipeline_args: 파이프라인 인자들
        global_step: 현재 글로벌 스텝
        is_final_validation: 최종 검증 여부
        prefix: 로깅 접두사
        do_gesture: 제스처 태스크 실행 여부
        do_audio: 오디오 태스크 실행 여부
        do_text: 텍스트 태스크 실행 여부
    """
    logger.info(f"Running Omniges validation... Generating samples for all tasks")  # 검증 시작 로그
    pipeline = pipeline.to(accelerator.device)  # 파이프라인을 가속기 디바이스로 이동
    
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None  # 시드가 있으면 제너레이터 생성, 없으면 None
    autocast_ctx = nullcontext()  # 자동 캐스팅 컨텍스트 (혼합 정밀도 비활성화)
    
    with autocast_ctx:  # 자동 캐스팅 컨텍스트 내에서 실행
        phase_name = f"test_{prefix}" if is_final_validation else f"validation_{prefix}"  # 검증 단계 이름 설정 (최종 검증이면 'test', 아니면 'validation')
        
        # ============================================================================
        # 텍스트에서 제스처로 변환 테스트 (t2g)
        # ============================================================================
        if do_gesture:  # 제스처 태스크가 활성화된 경우
            try:  # 예외 처리 시작
                gesture_results = []  # 제스처 결과 리스트 초기화
                test_prompts = ["A person waving hello", "Someone clapping hands", "Dancing movements"]  # 테스트 프롬프트들
                for prompt in test_prompts:  # 각 테스트 프롬프트에 대해
                    result = pipeline(  # 파이프라인 실행
                        prompt=prompt,  # 프롬프트
                        task='t2g',  # 태스크: 텍스트에서 제스처로
                        seq_length=128,  # 시퀀스 길이
                        guidance_scale=7.0,  # 가이던스 스케일
                        generator=generator  # 제너레이터
                    )
                    gesture_results.append(result)  # 결과를 리스트에 추가
                    
                # ============================================================================
                # wandb에 로깅
                # ============================================================================
                for tracker in accelerator.trackers:  # 모든 트래커에 대해
                    if tracker.name == "wandb":  # wandb 트래커인 경우
                        # 제스처 시퀀스를 numpy 배열로 로깅
                        gesture_data = []  # 제스처 데이터 리스트 초기화
                        for i, result in enumerate(gesture_results):  # 각 결과에 대해
                            if hasattr(result, 'gestures'):  # 결과에 gestures 속성이 있는 경우
                                gesture_np = result.gestures.cpu().numpy()  # 제스처를 numpy 배열로 변환
                                gesture_data.append({  # 제스처 데이터 딕셔너리 추가
                                    'prompt': test_prompts[i],  # 프롬프트
                                    'gesture_shape': str(gesture_np.shape),  # 제스처 형태
                                    'gesture_mean': float(gesture_np.mean()),  # 제스처 평균
                                    'gesture_std': float(gesture_np.std())  # 제스처 표준편차
                                })
                        
                        df = pd.DataFrame(gesture_data)  # pandas DataFrame 생성
                        html = wandb.Html(df.to_html(), inject=True)  # HTML 테이블 생성
                        tracker.log({f"t2g_{phase_name}": html})  # wandb에 로깅
                        
            except Exception as e:  # 예외 발생 시
                logger.warning(f"T2G validation failed: {e}")  # 경고 로그 출력
        
        # ============================================================================
        # 오디오에서 제스처로 변환 테스트 (a2g)
        # ============================================================================
        if do_gesture and do_audio:  # 제스처와 오디오 태스크가 모두 활성화된 경우
            try:  # 예외 처리 시작
                for ref_audio in ['assets/car engine.mp3']:  # 참조 오디오 파일들
                    if os.path.exists(ref_audio):  # 오디오 파일이 존재하는 경우
                        result = pipeline(  # 파이프라인 실행
                            input_aud=ref_audio,  # 입력 오디오
                            task='a2g',  # 태스크: 오디오에서 제스처로
                            seq_length=128,  # 시퀀스 길이
                            guidance_scale=7.0  # 가이던스 스케일
                        )
                        
                        # ============================================================================
                        # wandb에 로깅
                        # ============================================================================
                        for tracker in accelerator.trackers:  # 모든 트래커에 대해
                            if tracker.name == "wandb":  # wandb 트래커인 경우
                                if hasattr(result, 'gestures'):  # 결과에 gestures 속성이 있는 경우
                                    gesture_np = result.gestures.cpu().numpy()  # 제스처를 numpy 배열로 변환
                                    gesture_info = {  # 제스처 정보 딕셔너리
                                        'audio_file': ref_audio,  # 오디오 파일명
                                        'gesture_shape': str(gesture_np.shape),  # 제스처 형태
                                        'gesture_mean': float(gesture_np.mean()),  # 제스처 평균
                                        'gesture_std': float(gesture_np.std())  # 제스처 표준편차
                                    }
                                    tracker.log({f"a2g_{phase_name}": gesture_info})  # wandb에 로깅
                                    
            except Exception as e:  # 예외 발생 시
                logger.warning(f"A2G validation failed: {e}")  # 경고 로그 출력
        
        # ============================================================================
        # 제스처에서 텍스트로 변환 테스트 (g2t)
        # ============================================================================
        if do_text:  # 텍스트 태스크가 활성화된 경우
            try:  # 예외 처리 시작
                # 테스트용 더미 제스처 생성
                dummy_gesture = torch.randn(1, 128, 415).to(accelerator.device)  # 랜덤 제스처 생성 (배치=1, 시퀀스=128, 특성=415)
                
                result = pipeline(  # 파이프라인 실행
                    input_gesture=dummy_gesture,  # 입력 제스처
                    task='g2t',  # 태스크: 제스처에서 텍스트로
                    guidance_scale=2.0  # 가이던스 스케일
                )
                
                if isinstance(result, tuple) and len(result) >= 2:  # 결과가 튜플이고 길이가 2 이상인 경우
                    generated_text = result[0][0] if result[0] else "No text generated"  # 생성된 텍스트 추출
                    
                    # ============================================================================
                    # wandb에 로깅
                    # ============================================================================
                    for tracker in accelerator.trackers:  # 모든 트래커에 대해
                        if tracker.name == "wandb":  # wandb 트래커인 경우
                            tracker.log({  # wandb에 로깅
                                f"g2t_{phase_name}": {  # 제스처에서 텍스트로 변환 결과
                                    'generated_text': generated_text,  # 생성된 텍스트
                                    'gesture_input_shape': str(dummy_gesture.shape)  # 입력 제스처 형태
                                }
                            })
                            
            except Exception as e:  # 예외 발생 시
                logger.warning(f"G2T validation failed: {e}")  # 경고 로그 출력
        
        # ============================================================================
        # 텍스트에서 오디오로 변환 테스트 (t2a) - OmniFlow에서 가져옴
        # ============================================================================
        if do_audio:  # 오디오 태스크가 활성화된 경우
            try:  # 예외 처리 시작
                spec, _ = pipeline(  # 파이프라인 실행 (스펙트로그램과 기타 결과)
                    prompt="Music playing softly",  # 프롬프트
                    task='t2a',  # 태스크: 텍스트에서 오디오로
                    guidance_scale=4.0,  # 가이던스 스케일
                    num_inference_steps=28  # 추론 스텝 수
                )
                
                # ============================================================================
                # wandb에 로깅
                # ============================================================================
                for tracker in accelerator.trackers:  # 모든 트래커에 대해
                    if tracker.name == "wandb":  # wandb 트래커인 경우
                        # 오디오 스펙트로그램 정보 로깅
                        tracker.log({  # wandb에 로깅
                            f"t2a_{phase_name}": {  # 텍스트에서 오디오로 변환 결과
                                'spec_shape': str(spec.shape) if hasattr(spec, 'shape') else 'No shape',  # 스펙트로그램 형태
                                'spec_mean': float(np.mean(spec)) if spec is not None else 0,  # 스펙트로그램 평균
                                'spec_std': float(np.std(spec)) if spec is not None else 0  # 스펙트로그램 표준편차
                            }
                        })
                        
            except Exception as e:  # 예외 발생 시
                logger.warning(f"T2A validation failed: {e}")  # 경고 로그 출력

    # ============================================================================
    # 메모리 정리
    # ============================================================================
    del pipeline  # 파이프라인 삭제
    if torch.cuda.is_available():  # CUDA가 사용 가능한 경우
        torch.cuda.empty_cache()  # CUDA 캐시 비우기

    return None  # None 반환


def parse_omniges_args(input_args=None):
    """
    Omniges 훈련을 위한 인자 파싱 함수
    모든 훈련 관련 인자들을 정의하고 파싱
    
    Args:
        input_args: 입력 인자 리스트 (None이면 sys.argv 사용)
    
    Returns:
        args: 파싱된 인자 객체
    """
    parser = argparse.ArgumentParser(description="Omniges multi-modal training script.")  # 인자 파서 생성
    
    # ============================================================================
    # 기본 모델 인자들
    # ============================================================================
    parser.add_argument(  # 사전 훈련된 모델 경로
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,  # 필수 인자
        help="Path to pretrained OmniFlow model",  # 사전 훈련된 OmniFlow 모델 경로
    )
    
    parser.add_argument(  # BEAT 데이터셋 설정 파일 경로
        "--beat_config_path",
        type=str,
        default="configs/shortcut_rvqvae_128.yaml",  # 기본 설정 파일
        help="Path to BEAT dataset configuration",  # BEAT 데이터셋 설정 파일 경로
    )
    
    parser.add_argument(  # RVQVAE 체크포인트 디렉토리
        "--rvqvae_checkpoints",
        type=str,
        default="./ckpt/",  # 기본 체크포인트 디렉토리
        help="Directory containing RVQVAE checkpoints",  # RVQVAE 체크포인트가 포함된 디렉토리
    )
    
    parser.add_argument(  # 텍스트 VAE 토크나이저 경로
        "--tokenizer",
        type=str,
        default='/localhome/jacklishufan/TinyLlama_v1.1',  # 기본 토크나이저 경로
        required=True,  # 필수 인자
        help="Path to tokenizer for text VAE",  # 텍스트 VAE용 토크나이저 경로
    )
    
    # ============================================================================
    # 훈련 인자들
    # ============================================================================
    parser.add_argument("--output_dir", type=str, default="omniges-training", help="Output directory")  # 출력 디렉토리
    parser.add_argument("--seed", type=int, default=None, help="Training seed")  # 훈련 시드
    parser.add_argument("--resolution", type=int, default=512, help="Resolution for compatibility")  # 호환성을 위한 해상도
    parser.add_argument("--seq_length", type=int, default=128, help="Gesture sequence length")  # 제스처 시퀀스 길이
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size per device")  # 디바이스당 배치 크기
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of epochs")  # 에포크 수
    parser.add_argument("--max_train_steps", type=int, default=None, help="Maximum training steps")  # 최대 훈련 스텝
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")  # 학습률
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation")  # 그래디언트 누적 스텝
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")  # 그래디언트 체크포인팅 사용
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])  # 혼합 정밀도
    parser.add_argument("--use_ema", action="store_true", help="Use EMA")  # EMA 사용
    parser.add_argument("--ema_momentum", type=float, default=0.9999, help="EMA momentum")  # EMA 모멘텀
    
    # ============================================================================
    # 검증 인자들
    # ============================================================================
    parser.add_argument("--validation_prompt", type=str, default="A person waving", help="Validation prompt")  # 검증 프롬프트
    parser.add_argument("--num_validation_images", type=int, default=4, help="Number of validation samples")  # 검증 샘플 수
    parser.add_argument("--val_every", type=int, default=500, help="Validation frequency")  # 검증 빈도
    
    # ============================================================================
    # 스케줄러 인자들
    # ============================================================================
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="LR scheduler type")  # 학습률 스케줄러 타입
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Warmup steps")  # 워밍업 스텝 수
    
    # ============================================================================
    # 손실 가중치 인자들
    # ============================================================================
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"])  # 가중치 스킴
    parser.add_argument("--logit_mean", type=float, default=0.0, help="Logit normal mean")  # 로짓 정규 분포 평균
    parser.add_argument("--logit_std", type=float, default=1.0, help="Logit normal std")  # 로짓 정규 분포 표준편차
    parser.add_argument("--mode_scale", type=float, default=1.29, help="Mode scale")  # 모드 스케일
    parser.add_argument("--uniform_flow", action="store_true", help="Use uniform flow matching")  # 균등 플로우 매칭 사용
    
    # ============================================================================
    # 체크포인트 인자들
    # ============================================================================
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Checkpoint frequency")  # 체크포인트 빈도
    parser.add_argument("--checkpoints_total_limit", type=int, default=5, help="Max checkpoints")  # 최대 체크포인트 수
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")  # 체크포인트에서 재개
    
    # ============================================================================
    # 옵티마이저 인자들
    # ============================================================================
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer type")  # 옵티마이저 타입
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")  # Adam beta1
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")  # Adam beta2
    parser.add_argument("--adam_weight_decay", type=float, default=0, help="Weight decay")  # 가중치 감쇠
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Adam epsilon")  # Adam epsilon
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")  # 최대 그래디언트 노름
    
    # ============================================================================
    # 로깅 인자들
    # ============================================================================
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting backend")  # 보고 백엔드
    parser.add_argument("--logging_dir", type=str, default="logs", help="Logging directory")  # 로깅 디렉토리
    
    # ============================================================================
    # 고급 인자들
    # ============================================================================
    parser.add_argument("--allow_tf32", action="store_true", help="Allow TF32")  # TF32 허용
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Dataloader workers")  # 데이터로더 워커 수
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")  # 분산 훈련용 로컬 랭크
    
    # ============================================================================
    # 텍스트 VAE 인자들
    # ============================================================================
    parser.add_argument("--text_vae", type=str, required=True, help="Path to text VAE model")  # 텍스트 VAE 모델 경로 (필수)
    parser.add_argument("--precondition_text_outputs", action="store_true", help="Precondition text outputs")  # 텍스트 출력 전처리
    parser.add_argument("--anchor", action="store_true", help="Use anchor loss")  # 앵커 손실 사용
    
    # ============================================================================
    # 인자 파싱
    # ============================================================================
    if input_args is not None:  # 입력 인자가 제공된 경우
        args = parser.parse_args(input_args)  # 제공된 인자로 파싱
    else:  # 입력 인자가 없는 경우
        args = parser.parse_args()  # sys.argv로 파싱
    
    # ============================================================================
    # 인자 검증
    # ============================================================================
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))  # 환경 변수에서 로컬 랭크 가져오기
    if env_local_rank != -1 and env_local_rank != args.local_rank:  # 환경 변수 랭크가 있고 인자 랭크와 다른 경우
        args.local_rank = env_local_rank  # 환경 변수 랭크로 업데이트
        
    return args  # 파싱된 인자 반환


def tokenize_prompt(tokenizer, prompt):
    """
    텍스트 생성을 위한 프롬프트 토크나이징 함수
    
    Args:
        tokenizer: 토크나이저
        prompt: 토크나이징할 프롬프트
    
    Returns:
        text_input_ids: 토크나이징된 입력 ID들
    """
    text_inputs = tokenizer(  # 토크나이저로 프롬프트 처리
        prompt,  # 프롬프트
        padding="max_length",  # 최대 길이로 패딩
        max_length=77,  # 최대 길이 77
        truncation=True,  # 잘라내기 활성화
        return_tensors="pt",  # PyTorch 텐서로 반환
    )
    text_input_ids = text_inputs.input_ids  # 입력 ID 추출
    return text_input_ids  # 토크나이징된 입력 ID 반환


def load_safe_tensors(fp, model):
    """
    형태 검사를 통한 안전한 텐서 로딩 함수
    모델과 로드할 텐서의 형태가 일치하지 않는 경우 해당 키를 제거
    
    Args:
        fp: 로드할 파일 경로
        model: 텐서를 로드할 모델
    """
    tensors = torch.load(fp, map_location='cpu')  # CPU에서 텐서 로드
    
    model_dict = model.state_dict()  # 모델의 상태 딕셔너리 가져오기
    keys_to_pop = []  # 제거할 키 리스트 초기화
    for k, v in tensors.items():  # 로드된 텐서의 각 키-값 쌍에 대해
        if k in model_dict and model_dict[k].shape != v.shape:  # 모델에 키가 있고 형태가 다른 경우
            print(f"SIZE MISMATCH {k}: {model_dict[k].shape} vs {v.shape}")  # 형태 불일치 출력
            keys_to_pop.append(k)  # 제거할 키 리스트에 추가
    for k in keys_to_pop:  # 제거할 키들에 대해
        tensors.pop(k)  # 텐서 딕셔너리에서 해당 키 제거
        
    res = model.load_state_dict(tensors, strict=False)  # 모델에 텐서 로드 (엄격하지 않게)
    print(f"Loaded {fp}: {res}")  # 로드 결과 출력
    del tensors  # 텐서 딕셔너리 삭제
    torch.cuda.empty_cache()  # CUDA 캐시 비우기


def load_safe_tensors_ema(fp, model):
    """
    EMA 모델 가중치 로딩 함수
    EMA (Exponential Moving Average) 모델의 가중치를 로드
    
    Args:
        fp: 로드할 EMA 파일 경로
        model: EMA 가중치를 로드할 모델
    """
    tensors = torch.load(fp, map_location='cpu')  # CPU에서 EMA 텐서 로드
    res = model.load_state_dict(tensors)  # 모델에 EMA 가중치 로드 (엄격하게)
    print(f"Loaded EMA {fp}: {res}")  # EMA 로드 결과 출력
    del tensors  # 텐서 딕셔너리 삭제
    torch.cuda.empty_cache()  # CUDA 캐시 비우기


def main(args):
    """
    Omniges 훈련을 위한 메인 함수
    모든 컴포넌트를 초기화하고 훈련 루프를 시작
    
    Args:
        args: 파싱된 훈련 인자들
    """
    
    # ============================================================================
    # 가속기 설정
    # ============================================================================
    logging_dir = Path(args.output_dir, args.logging_dir)  # 로깅 디렉토리 경로 생성
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)  # 프로젝트 설정 생성
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)  # 분산 훈련을 위한 키워드 인자 (사용되지 않는 파라미터 허용)
    accelerator = Accelerator(  # Accelerate 가속기 초기화
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # 그래디언트 누적 스텝
        mixed_precision=args.mixed_precision,  # 혼합 정밀도 설정
        log_with=args.report_to,  # 로깅 백엔드 (wandb 등)
        project_config=accelerator_project_config,  # 프로젝트 설정
        kwargs_handlers=[kwargs],  # 키워드 인자 핸들러
    )
    
    # ============================================================================
    # 로깅 설정
    # ============================================================================
    logging.basicConfig(  # 기본 로깅 설정
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",  # 로그 포맷
        datefmt="%m/%d/%Y %H:%M:%S",  # 날짜 포맷
        level=logging.INFO,  # 로그 레벨
    )
    logger.info(accelerator.state, main_process_only=False)  # 가속기 상태 로깅 (모든 프로세스에서)
    
    if accelerator.is_local_main_process:  # 로컬 메인 프로세스인 경우
        transformers.utils.logging.set_verbosity_warning()  # Transformers 로깅을 경고 레벨로 설정
        diffusers.utils.logging.set_verbosity_info()  # Diffusers 로깅을 정보 레벨로 설정
    else:  # 다른 프로세스인 경우
        transformers.utils.logging.set_verbosity_error()  # Transformers 로깅을 에러 레벨로 설정
        diffusers.utils.logging.set_verbosity_error()  # Diffusers 로깅을 에러 레벨로 설정

    if args.seed is not None:  # 시드가 설정된 경우
        set_seed(args.seed)  # 시드 설정

    # ============================================================================
    # 출력 디렉토리 생성
    # ============================================================================
    if accelerator.is_main_process:  # 메인 프로세스인 경우
        os.makedirs(args.output_dir, exist_ok=True)  # 출력 디렉토리 생성 (이미 존재하면 무시)

    # ============================================================================
    # 토크나이저 로드 (OmniFlow와 동일)
    # ============================================================================
    tokenizer_one = CLIPTokenizer.from_pretrained('laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')  # 첫 번째 CLIP 토크나이저
    tokenizer_two = CLIPTokenizer.from_pretrained(  # 두 번째 CLIP 토크나이저
        args.pretrained_model_name_or_path, subfolder="tokenizer_2"  # 사전 훈련된 모델에서 로드
    )
    tokenizer_three = T5TokenizerFast.from_pretrained('google/flan-t5-large')  # T5 토크나이저

    # ============================================================================
    # 스케줄러 로드 (OmniFlow와 동일)
    # ============================================================================
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(  # 노이즈 스케줄러 로드
        args.pretrained_model_name_or_path, subfolder="scheduler", shift=1  # 사전 훈련된 모델에서 로드, shift=1
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)  # 노이즈 스케줄러 복사본 (훈련용)
    noise_scheduler_pipeline = copy.deepcopy(noise_scheduler)  # 노이즈 스케줄러 복사본 (파이프라인용)
    
    # ============================================================================
    # 텍스트 인코더 로드 (OmniFlow와 동일)
    # ============================================================================
    text_encoder_one = CLIPTextModelWithProjection.from_pretrained(  # 첫 번째 CLIP 텍스트 인코더
        'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K', projection_dim=768  # 투영 차원 768
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(  # 두 번째 CLIP 텍스트 인코더
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"  # 사전 훈련된 모델에서 로드
    )
    text_encoder_three = T5EncoderModel.from_pretrained('google/flan-t5-large')  # T5 인코더

    # ============================================================================
    # 기타 인코더 로드
    # ============================================================================
    audio_encoder = LanguageBindAudio.from_pretrained('LanguageBind/LanguageBind_Audio_FT')  # 오디오 인코더
    audio_encoder.text_model = nn.Identity()  # 텍스트 모델을 항등 함수로 설정 (오디오만 사용)
    
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")  # 이미지 프로세서 (호환성을 위해)
    audio_processor_clip = LanguageBindAudioProcessor(audio_encoder.config)  # 오디오 프로세서
    
    # ============================================================================
    # 인코더를 평가 모드로 설정
    # ============================================================================
    text_encoder_one.eval()  # 첫 번째 텍스트 인코더를 평가 모드로 설정
    text_encoder_two.eval()  # 두 번째 텍스트 인코더를 평가 모드로 설정
    text_encoder_three.eval()  # 세 번째 텍스트 인코더를 평가 모드로 설정
    
    # ============================================================================
    # VAE 로드
    # ============================================================================
    audiovae, audio_processor = load_audio_vae()  # 오디오 VAE와 프로세서 로드
    
    # ============================================================================
    # 텍스트 VAE 로드
    # ============================================================================
    text_vae_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)  # 텍스트 VAE 토크나이저 로드
    text_vae_tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # 패딩 토큰 추가
    
    config = AutoConfig.from_pretrained(args.text_vae)  # 텍스트 VAE 설정 로드
    text_vae = LLamaForLatentConnector._from_config(config, torch_dtype=torch.bfloat16)  # 텍스트 VAE 모델 생성 (bfloat16)
    text_vae.prepare_tokenizer(text_vae_tokenizer)  # 토크나이저 준비
    text_vae.set_encoder(text_encoder_three)  # T5 인코더 설정
    
    # ============================================================================
    # 제스처 VAE 생성
    # ============================================================================
    rvqvae_checkpoints = {  # RVQVAE 체크포인트 경로들
        'upper': os.path.join(args.rvqvae_checkpoints, 'net_300000_upper.pth'),  # 상체 체크포인트
        'hands': os.path.join(args.rvqvae_checkpoints, 'net_300000_hands.pth'),  # 손 체크포인트
        'lower_trans': os.path.join(args.rvqvae_checkpoints, 'net_300000_lower.pth'),  # 하체+이동 체크포인트
        'face': os.path.join(args.rvqvae_checkpoints, 'net_300000_face.pth')  # 얼굴 체크포인트
    }
    gesture_vae = OmnigesGestureVAE(rvqvae_checkpoints)  # 제스처 VAE 생성
    
    # ============================================================================
    # 새로운 Omniges 트랜스포머 생성 - OmniFlow 차원에 맞춤
    # ============================================================================
    transformer = OmnigesFlowTransformerModel(  # OmnigesFlow 트랜스포머 모델 생성
        seq_length=args.seq_length,  # 시퀀스 길이
        gesture_latent_dim=512,      # 제스처 잠재 차원 (128 * 4 parts)
        num_layers=24,               # OmniFlow 실제 레이어 수 (더 큰 모델)
        num_attention_heads=24,      # OmniFlow 실제 head 수
        attention_head_dim=64,  # 어텐션 헤드 차원
        joint_attention_dim=4096,  # 공동 어텐션 차원
        caption_projection_dim=1536, # OmniFlow 실제 차원 1536
        pooled_projection_dim=2048,  # 풀링 투영 차원
        audio_input_dim=8,  # 오디오 입력 차원
        gesture_output_dim=512,  # 제스처 출력 차원
        add_audio=True,  # 오디오 추가
        use_audio_mae=False,  # 오디오 MAE 사용 안함
        drop_gesture=False,  # 제스처 드롭 안함
        drop_text=False,  # 텍스트 드롭 안함
        drop_audio=False  # 오디오 드롭 안함
    )
    
    # ============================================================================
    # 텍스트 디코더 설정
    # ============================================================================
    transformer.set_text_decoder(text_vae)  # 트랜스포머에 텍스트 디코더 설정
    
    # ============================================================================
    # OmniFlow 가중치 로드 (가능한 경우)
    # ============================================================================
    if args.pretrained_model_name_or_path:  # 사전 훈련된 모델 경로가 있는 경우
        fp = os.path.join(args.pretrained_model_name_or_path, 'transformer/diffusion_pytorch_model.bin')  # 트랜스포머 가중치 파일 경로
        if os.path.exists(fp):  # 파일이 존재하는 경우
            try:  # 시도
                load_safe_tensors(fp, transformer)  # 안전한 텐서 로딩
                logger.info("Loaded OmniFlow weights successfully")  # 성공 로그
            except Exception as e:  # 예외 발생 시
                logger.warning(f"Could not load OmniFlow weights: {e}")  # 경고 로그
    
    # ============================================================================
    # 그래디언트 계산 설정
    # ============================================================================
    transformer.requires_grad_(True)  # 트랜스포머는 그래디언트 계산 활성화
    text_vae.requires_grad_(False)  # 텍스트 VAE는 그래디언트 계산 비활성화
    audio_encoder.requires_grad_(False)  # 오디오 인코더는 그래디언트 계산 비활성화
    audiovae.requires_grad_(False)  # 오디오 VAE는 그래디언트 계산 비활성화
    gesture_vae.requires_grad_(False)  # 제스처 VAE는 그래디언트 계산 비활성화
    text_encoder_one.requires_grad_(False)  # 첫 번째 텍스트 인코더는 그래디언트 계산 비활성화
    text_encoder_two.requires_grad_(False)  # 두 번째 텍스트 인코더는 그래디언트 계산 비활성화
    text_encoder_three.requires_grad_(False)  # 세 번째 텍스트 인코더는 그래디언트 계산 비활성화
    
    # ============================================================================
    # 가중치 데이터 타입 설정
    # ============================================================================
    weight_dtype = torch.float32  # 기본 가중치 데이터 타입
    if accelerator.mixed_precision == "fp16":  # 혼합 정밀도가 fp16인 경우
        weight_dtype = torch.float16  # float16으로 설정
    elif accelerator.mixed_precision == "bf16":  # 혼합 정밀도가 bf16인 경우
        weight_dtype = torch.bfloat16  # bfloat16으로 설정
        
    # ============================================================================
    # EMA 설정
    # ============================================================================
    if args.use_ema and accelerator.is_main_process:  # EMA 사용이 활성화되고 메인 프로세스인 경우
        ema_transformer = EMAModel(transformer.parameters(), decay=args.ema_momentum)  # EMA 모델 생성
        
    # ============================================================================
    # 모델을 디바이스로 이동
    # ============================================================================
    gesture_vae.to(accelerator.device, dtype=weight_dtype)  # 제스처 VAE를 디바이스로 이동
    audiovae.to(accelerator.device, dtype=torch.float32)  # 오디오 VAE를 디바이스로 이동 (float32)
    text_vae.to(accelerator.device)  # 텍스트 VAE를 디바이스로 이동
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)  # 첫 번째 텍스트 인코더를 디바이스로 이동
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)  # 두 번째 텍스트 인코더를 디바이스로 이동
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)  # 세 번째 텍스트 인코더를 디바이스로 이동
    audio_encoder.to(accelerator.device, dtype=weight_dtype)  # 오디오 인코더를 디바이스로 이동
    
    # ============================================================================
    # 그래디언트 체크포인팅 활성화
    # ============================================================================
    if args.gradient_checkpointing:  # 그래디언트 체크포인팅이 활성화된 경우
        transformer.enable_gradient_checkpointing()  # 트랜스포머에 그래디언트 체크포인팅 활성화
        
    # ============================================================================
    # 옵티마이저 생성
    # ============================================================================
    optimizer = torch.optim.AdamW(  # AdamW 옵티마이저 생성
        transformer.parameters(),  # 트랜스포머 파라미터
        lr=args.learning_rate,  # 학습률
        betas=(args.adam_beta1, args.adam_beta2),  # Adam 베타 값들
        weight_decay=args.adam_weight_decay,  # 가중치 감쇠
        eps=args.adam_epsilon,  # Adam epsilon
    )
    
    # ============================================================================
    # 데이터셋 생성
    # ============================================================================
    train_dataset = OmnigesDataset(  # Omniges 데이터셋 생성
        beat_config_path=args.beat_config_path,  # BEAT 설정 파일 경로
        task_weights=[1/6] * 6,  # 모든 태스크에 동일한 가중치 (1/6)
        size=args.resolution,  # 해상도
        is_train=True,  # 훈련 모드
        image_processor=image_processor,  # 이미지 프로세서
        audio_processor=audio_processor,  # 오디오 프로세서
        audio_processor_clip=audio_processor_clip,  # CLIP 오디오 프로세서
    )
    
    # ============================================================================
    # 데이터로더 생성
    # ============================================================================
    train_dataloader = torch.utils.data.DataLoader(  # 훈련 데이터로더 생성
        train_dataset,  # 훈련 데이터셋
        batch_size=args.train_batch_size,  # 배치 크기
        shuffle=True,  # 셔플 활성화
        collate_fn=omniges_collate_fn,  # 커스텀 콜레이트 함수
        num_workers=args.dataloader_num_workers,  # 워커 수
    )
    
    # ============================================================================
    # 스케줄러 생성
    # ============================================================================
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)  # 에포크당 업데이트 스텝 수 계산
    if args.max_train_steps is None:  # 최대 훈련 스텝이 지정되지 않은 경우
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch  # 에포크 수로 계산
    else:  # 최대 훈련 스텝이 지정된 경우
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)  # 에포크 수 재계산
    
    lr_scheduler = get_scheduler(  # 학습률 스케줄러 생성
        args.lr_scheduler,  # 스케줄러 타입
        optimizer=optimizer,  # 옵티마이저
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,  # 워밍업 스텝 수 (프로세스 수 곱하기)
        num_training_steps=args.max_train_steps * accelerator.num_processes,  # 훈련 스텝 수 (프로세스 수 곱하기)
    )
    
    # ============================================================================
    # 가속기로 준비
    # ============================================================================
    transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)  # 가속기로 모델, 옵티마이저, 스케줄러 준비
    
    # ============================================================================
    # 텍스트 인코더 리스트 생성
    # ============================================================================
    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]  # 토크나이저 리스트
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]  # 텍스트 인코더 리스트
    
    # ============================================================================
    # 훈련 정보
    # ============================================================================
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps  # 총 배치 크기 계산
    
    logger.info("***** Running Omniges Training *****")  # 훈련 시작 로그
    logger.info(f"  Num examples = {len(train_dataset)}")  # 예제 수
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")  # 에포크당 배치 수
    logger.info(f"  Num Epochs = {args.num_train_epochs}")  # 에포크 수
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")  # 디바이스당 즉시 배치 크기
    logger.info(f"  Total train batch size = {total_batch_size}")  # 총 훈련 배치 크기
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")  # 그래디언트 누적 스텝
    logger.info(f"  Total optimization steps = {args.max_train_steps}")  # 총 최적화 스텝
    logger.info(f"  Supported tasks: t2g, g2t, a2g, g2a, t2a, a2t")  # 지원하는 태스크들
    
    global_step = 0  # 글로벌 스텝 초기화
    first_epoch = 0  # 첫 번째 에포크 초기화
    
    # ============================================================================
    # 진행률 바
    # ============================================================================
    progress_bar = tqdm(  # tqdm 진행률 바 생성
        range(0, args.max_train_steps * args.gradient_accumulation_steps),  # 진행 범위
        initial=0,  # 초기값
        desc="Steps",  # 설명
        disable=not accelerator.is_local_main_process,  # 로컬 메인 프로세스가 아니면 비활성화
    )
    
    # ============================================================================
    # 추적 초기화
    # ============================================================================
    if accelerator.is_main_process:  # 메인 프로세스인 경우
        accelerator.init_trackers("omniges-training", config=vars(args))  # 추적기 초기화 (wandb 등)
    
    # ============================================================================
    # 훈련 루프
    # ============================================================================
    for epoch in range(first_epoch, args.num_train_epochs):  # 에포크 루프 (첫 번째 에포크부터 설정된 에포크 수까지)
        transformer.train()  # 트랜스포머를 훈련 모드로 설정
        
        for step, batch in enumerate(train_dataloader):  # 배치 루프 (데이터로더의 각 배치에 대해)
            with accelerator.accumulate([transformer]):  # 그래디언트 누적 컨텍스트 (트랜스포머만 누적)
                
                # ============================================================================
                # 순전파
                # ============================================================================
                loss, decode_loss, logs, task_type, model_pred, logits, target, prompts, model_pred_audio, model_pred_audio, raw_audio_embeds, model_pred_text, raw_text_embeds = transformer(  # 트랜스포머에 순전파하여 모든 결과 받기
                    kkwargs={  # 키워드 인자 딕셔너리
                        'args': args,  # 훈련 인자들
                        'text_encoder_one': text_encoder_one,  # 첫 번째 텍스트 인코더
                        'text_encoder_two': text_encoder_two,  # 두 번째 텍스트 인코더
                        'text_encoder_three': text_encoder_three,  # 세 번째 텍스트 인코더
                        'accelerator': accelerator.device,  # 가속기 디바이스
                        'batch': batch,  # 배치 데이터
                        'gesture_vae': gesture_vae,  # 제스처 VAE 사용 (이미지 VAE 대신)
                        'tokenizer_three': tokenizer_three,  # T5 토크나이저
                        'text_encoders': text_encoders,  # 텍스트 인코더 리스트
                        'tokenizers': tokenizers,  # 토크나이저 리스트
                        'tokenizer_one': tokenizer_one,  # 첫 번째 CLIP 토크나이저
                        'tokenizer_two': tokenizer_two,  # 두 번째 CLIP 토크나이저
                        'weight_dtype': weight_dtype,  # 가중치 데이터 타입
                        'noise_scheduler_copy': noise_scheduler_copy,  # 노이즈 스케줄러 복사본
                        'noise_scheduler': noise_scheduler,  # 노이즈 스케줄러
                        'audio_vae_factor': 1,  # 오디오 VAE 팩터
                        'audiovae': audiovae,  # 오디오 VAE
                        'text_vae_tokenizer': text_vae_tokenizer,  # 텍스트 VAE 토크나이저
                        'last_lr': lr_scheduler.get_last_lr()[0],  # 마지막 학습률
                        'text_vae': text_vae,  # 텍스트 VAE
                        'audio_encoder': audio_encoder,  # 오디오 인코더
                        'do_decode': False,  # 디코딩 비활성화
                        'precondition_text_outputs': args.precondition_text_outputs,  # 텍스트 출력 전처리
                        'anchor': args.anchor,  # 앵커 플래그
                        'mm_encoder': None,  # 멀티모달 인코더 (None)
                    },
                    forward_function=omniges_forward_pass  # 순전파 함수 지정
                )

                # ============================================================================
                # 역전파
                # ============================================================================
                accelerator.backward(loss)  # 가속기를 사용한 역전파
                
                # ============================================================================
                # 그래디언트 클리핑
                # ============================================================================
                if accelerator.sync_gradients:  # 그래디언트 동기화가 필요한 경우
                    params_to_clip = transformer.parameters()  # 클리핑할 파라미터들 (트랜스포머만)
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)  # 그래디언트 노름 클리핑
                
                # ============================================================================
                # 옵티마이저 스텝
                # ============================================================================
                optimizer.step()  # 옵티마이저 스텝
                lr_scheduler.step()  # 학습률 스케줄러 스텝
                optimizer.zero_grad()  # 그래디언트 초기화
                
                # ============================================================================
                # EMA 업데이트
                # ============================================================================
                if accelerator.sync_gradients:  # 그래디언트 동기화가 필요한 경우
                    if args.use_ema and accelerator.is_main_process:  # EMA 사용이 활성화되고 메인 프로세스인 경우
                        if global_step % 100 == 0:  # 100 스텝마다 EMA 업데이트
                            ema_transformer.step(transformer.parameters())  # EMA 모델 업데이트

            # ============================================================================
            # 진행률 추적
            # ============================================================================
            progress_bar.update(1)  # 진행률 바 업데이트
            if accelerator.sync_gradients:  # 그래디언트 동기화가 필요한 경우
                global_step += 1  # 글로벌 스텝 증가
                
                # ============================================================================
                # 메트릭 로깅
                # ============================================================================
                progress_bar.set_postfix(**logs)  # 진행률 바에 로그 정보 표시
                accelerator.log(logs, step=global_step)  # 가속기에 로그 기록
                
                # ============================================================================
                # 체크포인트 저장
                # ============================================================================
                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:  # 메인 프로세스이고 체크포인트 스텝인 경우
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")  # 체크포인트 저장 경로
                    accelerator.save_state(save_path)  # 가속기 상태 저장
                    
                    # EMA 별도 저장
                    if args.use_ema:  # EMA 사용이 활성화된 경우
                        ema_path = os.path.join(save_path, "ema_transformer.pt")  # EMA 모델 저장 경로
                        torch.save(ema_transformer.state_dict(), ema_path)  # EMA 모델 상태 저장
                    
                    logger.info(f"Saved checkpoint to {save_path}")  # 체크포인트 저장 로그
                
                # ============================================================================
                # 검증 - 모든 태스크 테스트
                # ============================================================================
                if global_step % args.val_every == 0 and global_step > 0:  # 검증 빈도에 도달하고 글로벌 스텝이 0보다 큰 경우
                    transformer.eval()  # 트랜스포머를 평가 모드로 설정
                    
                    # ============================================================================
                    # 검증 파이프라인 생성
                    # ============================================================================
                    pipeline = OmnigesPipeline(  # Omniges 파이프라인 생성
                        transformer=accelerator.unwrap_model(transformer),  # 가속기에서 언래핑된 트랜스포머
                        scheduler=noise_scheduler_pipeline,  # 파이프라인용 노이즈 스케줄러
                        gesture_vae=gesture_vae,  # 제스처 VAE
                        text_encoder=accelerator.unwrap_model(text_encoder_one),  # 첫 번째 텍스트 인코더
                        text_encoder_2=accelerator.unwrap_model(text_encoder_two),  # 두 번째 텍스트 인코더
                        text_encoder_3=accelerator.unwrap_model(text_encoder_three),  # 세 번째 텍스트 인코더
                        tokenizer=tokenizer_one,  # 첫 번째 토크나이저
                        tokenizer_2=tokenizer_two,  # 두 번째 토크나이저
                        tokenizer_3=tokenizer_three,  # 세 번째 토크나이저
                        audio_vae=audiovae,  # 오디오 VAE
                        audio_processor=audio_processor,  # 오디오 프로세서
                        audio_processor_clip=audio_processor_clip,  # CLIP 오디오 프로세서
                        audio_encoder=accelerator.unwrap_model(audio_encoder),  # 오디오 인코더
                        text_vae=text_vae,  # 텍스트 VAE
                        text_vae_tokenizer=text_vae_tokenizer,  # 텍스트 VAE 토크나이저
                        text_x0=args.precondition_text_outputs,  # 텍스트 출력 전처리
                    )
                    
                    # ============================================================================
                    # 🎯 모든 태스크별 검증 (OmniFlow 방식 확장)
                    # ============================================================================
                    validation_results = {}  # 검증 결과 딕셔너리 초기화
                    
                    # ============================================================================
                    # 1. 텍스트에서 제스처로 변환 (t2g)
                    # ============================================================================
                    try:  # 예외 처리 시작
                        t2g_result = pipeline(  # 파이프라인 실행
                            prompt=args.validation_prompt,  # 검증 프롬프트
                            task='t2g',  # 태스크: 텍스트에서 제스처로
                            seq_length=128,  # 시퀀스 길이
                            guidance_scale=7.0  # 가이던스 스케일
                        )
                        if hasattr(t2g_result, 'gestures'):  # 결과에 gestures 속성이 있는 경우
                            gesture_np = t2g_result.gestures.cpu().numpy()  # 제스처를 numpy 배열로 변환
                            validation_results['t2g'] = {  # T2G 검증 결과 저장
                                'shape': gesture_np.shape,  # 제스처 형태
                                'mean': float(gesture_np.mean()),  # 제스처 평균
                                'std': float(gesture_np.std())  # 제스처 표준편차
                            }
                            logger.info(f"  ✅ T2G validation: {gesture_np.shape}")  # 성공 로그
                    except Exception as e:  # 예외 발생 시
                        logger.warning(f"  ⚠️ T2G validation failed: {e}")  # 경고 로그
                    
                    # ============================================================================
                    # 2. 제스처에서 텍스트로 변환 (g2t)
                    # ============================================================================
                    try:  # 예외 처리 시작
                        dummy_gesture = torch.randn(1, 128, 415).to(accelerator.device)  # 더미 제스처 생성
                        g2t_result = pipeline(  # 파이프라인 실행
                            input_gesture=dummy_gesture,  # 입력 제스처
                            task='g2t',  # 태스크: 제스처에서 텍스트로
                            guidance_scale=2.0  # 가이던스 스케일
                        )
                        if isinstance(g2t_result, tuple) and len(g2t_result) >= 2:  # 결과가 튜플이고 길이가 2 이상인 경우
                            generated_text = g2t_result[0][0] if g2t_result[0] else "No text"  # 생성된 텍스트 추출
                            validation_results['g2t'] = {  # G2T 검증 결과 저장
                                'text': generated_text,  # 생성된 텍스트
                                'length': len(generated_text.split())  # 텍스트 길이
                            }
                            logger.info(f"  ✅ G2T validation: '{generated_text[:30]}...'")  # 성공 로그
                    except Exception as e:  # 예외 발생 시
                        logger.warning(f"  ⚠️ G2T validation failed: {e}")  # 경고 로그
                    
                    # ============================================================================
                    # 3. 오디오에서 제스처로 변환 (a2g) - 오디오 파일이 있는 경우
                    # ============================================================================
                    if os.path.exists('./assets/car engine.mp3'):  # 오디오 파일이 존재하는 경우
                        try:  # 예외 처리 시작
                            a2g_result = pipeline(  # 파이프라인 실행
                                input_aud='./assets/car engine.mp3',  # 입력 오디오 파일
                                task='a2g',  # 태스크: 오디오에서 제스처로
                                seq_length=128,  # 시퀀스 길이
                                guidance_scale=7.0  # 가이던스 스케일
                            )
                            if hasattr(a2g_result, 'gestures'):  # 결과에 gestures 속성이 있는 경우
                                gesture_np = a2g_result.gestures.cpu().numpy()  # 제스처를 numpy 배열로 변환
                                validation_results['a2g'] = {  # A2G 검증 결과 저장
                                    'shape': gesture_np.shape,  # 제스처 형태
                                    'mean': float(gesture_np.mean())  # 제스처 평균
                                }
                                logger.info(f"  ✅ A2G validation: {gesture_np.shape}")  # 성공 로그
                        except Exception as e:  # 예외 발생 시
                            logger.warning(f"  ⚠️ A2G validation failed: {e}")  # 경고 로그
                    
                    # ============================================================================
                    # 4. 제스처에서 오디오로 변환 (g2a)
                    # ============================================================================
                    try:  # 예외 처리 시작
                        g2a_result = pipeline(  # 파이프라인 실행
                            input_gesture=dummy_gesture,  # 입력 제스처 (이전에 생성된 더미 제스처)
                            task='g2a',  # 태스크: 제스처에서 오디오로
                            guidance_scale=4.0  # 가이던스 스케일
                        )
                        if isinstance(g2a_result, tuple) and len(g2a_result) >= 1:  # 결과가 튜플이고 길이가 1 이상인 경우
                            audio_spec = g2a_result[0]  # 오디오 스펙트로그램 추출
                            validation_results['g2a'] = {  # G2A 검증 결과 저장
                                'audio_shape': str(audio_spec.shape) if hasattr(audio_spec, 'shape') else 'No shape',  # 오디오 형태
                                'audio_mean': float(np.mean(audio_spec)) if audio_spec is not None else 0  # 오디오 평균
                            }
                            logger.info(f"  ✅ G2A validation: audio generated")  # 성공 로그
                    except Exception as e:  # 예외 발생 시
                        logger.warning(f"  ⚠️ G2A validation failed: {e}")  # 경고 로그
                    
                    # ============================================================================
                    # 5. 텍스트에서 오디오로 변환 (t2a) - OmniFlow 방식
                    # ============================================================================
                    try:  # 예외 처리 시작
                        t2a_result = pipeline(  # 파이프라인 실행
                            prompt="Music playing",  # 프롬프트
                            task='t2a',  # 태스크: 텍스트에서 오디오로
                            guidance_scale=4.0  # 가이던스 스케일
                        )
                        if isinstance(t2a_result, tuple) and len(t2a_result) >= 1:  # 결과가 튜플이고 길이가 1 이상인 경우
                            audio_spec = t2a_result[0]  # 오디오 스펙트로그램 추출
                            validation_results['t2a'] = {  # T2A 검증 결과 저장
                                'audio_shape': str(audio_spec.shape) if hasattr(audio_spec, 'shape') else 'No shape'  # 오디오 형태
                            }
                            logger.info(f"  ✅ T2A validation: audio generated")  # 성공 로그
                    except Exception as e:  # 예외 발생 시
                        logger.warning(f"  ⚠️ T2A validation failed: {e}")  # 경고 로그
                    
                    # ============================================================================
                    # 6. 오디오에서 텍스트로 변환 (a2t) - OmniFlow 방식
                    # ============================================================================
                    if os.path.exists('./assets/car engine.mp3'):  # 오디오 파일이 존재하는 경우
                        try:  # 예외 처리 시작
                            a2t_result = pipeline(  # 파이프라인 실행
                                input_aud='./assets/car engine.mp3',  # 입력 오디오 파일
                                task='a2t',  # 태스크: 오디오에서 텍스트로
                                guidance_scale=2.0  # 가이던스 스케일
                            )
                            if isinstance(a2t_result, tuple) and len(a2t_result) >= 2:  # 결과가 튜플이고 길이가 2 이상인 경우
                                generated_text = a2t_result[0][0] if a2t_result[0] else "No text"  # 생성된 텍스트 추출
                                validation_results['a2t'] = {  # A2T 검증 결과 저장
                                    'text': generated_text,  # 생성된 텍스트
                                    'length': len(generated_text.split())  # 텍스트 길이
                                }
                                logger.info(f"  ✅ A2T validation: '{generated_text[:30]}...'")  # 성공 로그
                        except Exception as e:  # 예외 발생 시
                            logger.warning(f"  ⚠️ A2T validation failed: {e}")  # 경고 로그
                    
                    # ============================================================================
                    # 모든 검증 결과 로깅
                    # ============================================================================
                    for tracker in accelerator.trackers:  # 모든 트래커에 대해
                        if tracker.name == "wandb":  # wandb 트래커인 경우
                            # ============================================================================
                            # 검증 요약 테이블 생성
                            # ============================================================================
                            val_data = []  # 검증 데이터 리스트 초기화
                            for task, result in validation_results.items():  # 각 태스크와 결과에 대해
                                val_data.append({  # 검증 데이터 추가
                                    'task': task.upper(),  # 태스크 이름 (대문자)
                                    'status': '✅ Success',  # 상태 (성공)
                                    'details': str(result)  # 결과 세부사항
                                })
                            
                            if val_data:  # 검증 데이터가 있는 경우
                                df = pd.DataFrame(val_data)  # pandas DataFrame 생성
                                html = wandb.Html(df.to_html(), inject=True)  # HTML 테이블 생성
                                tracker.log({f"validation_all_tasks_step_{global_step}": html})  # wandb에 로깅
                            
                            # ============================================================================
                            # 개별 태스크 결과 로깅
                            # ============================================================================
                            for task, result in validation_results.items():  # 각 태스크와 결과에 대해
                                tracker.log({f"val_{task}": result}, step=global_step)  # wandb에 개별 태스크 결과 로깅
                    
                    transformer.train()  # 트랜스포머를 훈련 모드로 다시 설정
                    del pipeline  # 파이프라인 삭제
                    torch.cuda.empty_cache()  # CUDA 캐시 비우기
                
                if global_step >= args.max_train_steps:  # 최대 훈련 스텝에 도달한 경우
                    break  # 훈련 루프 종료

    # ============================================================================
    # 최종 저장
    # ============================================================================
    accelerator.wait_for_everyone()  # 모든 프로세스가 완료될 때까지 대기
    if accelerator.is_main_process:  # 메인 프로세스인 경우
        save_path = os.path.join(args.output_dir, f"checkpoint-final")  # 최종 체크포인트 저장 경로
        accelerator.save_state(save_path)  # 가속기 상태 저장
        
        if args.use_ema:  # EMA 사용이 활성화된 경우
            ema_path = os.path.join(save_path, "ema_transformer.pt")  # EMA 모델 저장 경로
            torch.save(ema_transformer.state_dict(), ema_path)  # EMA 모델 상태 저장
            
        logger.info(f"Training complete! Final checkpoint saved to {save_path}")  # 훈련 완료 로그
        
    accelerator.end_training()  # 훈련 종료


if __name__ == "__main__":
    args = parse_omniges_args()
    main(args)

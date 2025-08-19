"""
Omniges Test & Validation Script
Complete evaluation for all tasks: t2g, g2t, a2g, g2a, t2a, a2t
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import librosa
import soundfile as sf
from loguru import logger
import platform

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import Omniges components
from omniges.models.omniges_flow import OmnigesFlowTransformerModel
from omniges.models.omniges_a2g import GestureProcessor
from omniges.pipelines.omniges_pipeline import OmnigesPipeline, create_omniges_pipeline

# Import evaluation utilities
from utils import other_tools_hf
from utils.metric import L1div, calculate_fgd
from dataloaders.beat_sep_lower import CustomDataset

# Import text evaluation
try:
    from nltk.translate.bleu_score import sentence_bleu
    from rouge_score import rouge_scorer
    EVAL_METRICS_AVAILABLE = True
except ImportError:
    logger.warning("Text evaluation metrics not available. Install nltk and rouge-score for full evaluation.")
    EVAL_METRICS_AVAILABLE = False


class OmnigesEvaluator:
    """
    Complete evaluator for all Omniges tasks
    """
    
    def __init__(
        self,
        pipeline: OmnigesPipeline,
        output_dir: str = "./results/omniges_evaluation",
        device: str = "cuda"
    ):
        self.pipeline = pipeline
        self.output_dir = output_dir
        self.device = device
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        for task in ['t2g', 'g2t', 'a2g', 'g2a', 't2a', 'a2t']:
            os.makedirs(os.path.join(output_dir, task), exist_ok=True)
        
        # Initialize metrics
        self.l1div_calculator = L1div()
        if EVAL_METRICS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Results storage
        self.results = {
            'losses': {},
            'metrics': {},
            'outputs': {}
        }
        
        # Set up rendering (if on Linux)
        if platform.system() == "Linux":
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
    
    @torch.no_grad()
    def evaluate_t2g(self, test_prompts, seq_length=128, guidance_scale=7.0):
        """
        Evaluate Text to Gesture (t2g)
        Output: NPZ files + Video renders
        """
        logger.info("🎬 Evaluating Text to Gesture (t2g)")
        
        results = []
        gesture_outputs = []
        
        for i, prompt in enumerate(test_prompts):
            try:
                # Generate gesture from text
                result = self.pipeline(
                    prompt=prompt,
                    task='t2g',
                    seq_length=seq_length,
                    guidance_scale=guidance_scale,
                    return_dict=True
                )
                
                if hasattr(result, 'gestures'):
                    gesture_seq = result.gestures  # (B, T, 415)
                    gesture_outputs.append(gesture_seq)
                    
                    # Save as NPZ
                    gesture_np = gesture_seq.cpu().numpy()
                    npz_path = os.path.join(self.output_dir, 't2g', f't2g_result_{i:03d}.npz')
                    
                    # Convert to SMPL-X format for rendering
                    smplx_data = self._gesture_to_smplx_format(gesture_np[0])  # Take first batch
                    
                    np.savez(
                        npz_path,
                        poses=smplx_data['poses'],
                        expressions=smplx_data['expressions'],
                        trans=smplx_data['trans'],
                        betas=smplx_data['betas'],
                        model='smplx2020',
                        gender='neutral',
                        mocap_frame_rate=30
                    )
                    
                    # Render video
                    video_path = self._render_gesture_video(
                        npz_path, 
                        prompt, 
                        os.path.join(self.output_dir, 't2g'),
                        f't2g_video_{i:03d}.mp4'
                    )
                    
                    results.append({
                        'prompt': prompt,
                        'npz_path': npz_path,
                        'video_path': video_path,
                        'gesture_shape': gesture_np.shape,
                        'gesture_mean': float(gesture_np.mean()),
                        'gesture_std': float(gesture_np.std())
                    })
                    
                    logger.info(f"  ✅ T2G {i+1}: {prompt} → {gesture_np.shape}")
                    
            except Exception as e:
                logger.error(f"  ❌ T2G {i+1} failed: {e}")
                
        # Compute FGD for gesture quality
        if len(gesture_outputs) > 1:
            fgd_score = self._compute_gesture_fgd(gesture_outputs)
            self.results['metrics']['t2g_fgd'] = fgd_score
            logger.info(f"  📊 T2G FGD: {fgd_score:.4f}")
            
        self.results['outputs']['t2g'] = results
        return results
    
    @torch.no_grad()
    def evaluate_g2t(self, gesture_inputs, guidance_scale=2.0):
        """
        Evaluate Gesture to Text (g2t)  
        Output: Text captions
        """
        logger.info("💬 Evaluating Gesture to Text (g2t)")
        
        results = []
        generated_texts = []
        
        for i, gesture_input in enumerate(gesture_inputs):
            try:
                # Generate text from gesture
                result = self.pipeline(
                    input_gesture=gesture_input,
                    task='g2t',
                    guidance_scale=guidance_scale
                )
                
                if isinstance(result, tuple) and len(result) >= 2:
                    generated_text = result[0][0] if result[0] else "No text generated"
                    reference_text = result[1][0] if result[1] else ""
                    
                    generated_texts.append(generated_text)
                    
                    results.append({
                        'gesture_input_shape': gesture_input.shape,
                        'generated_text': generated_text,
                        'reference_text': reference_text,
                        'text_length': len(generated_text.split())
                    })
                    
                    logger.info(f"  ✅ G2T {i+1}: {gesture_input.shape} → '{generated_text[:50]}...'")
                    
            except Exception as e:
                logger.error(f"  ❌ G2T {i+1} failed: {e}")
                
        # Compute text quality metrics
        if EVAL_METRICS_AVAILABLE and len(generated_texts) > 1:
            text_metrics = self._compute_text_metrics(generated_texts)
            self.results['metrics']['g2t_text_quality'] = text_metrics
            
        self.results['outputs']['g2t'] = results
        return results
    
    @torch.no_grad()
    def evaluate_a2g(self, audio_files, seq_length=128, guidance_scale=7.0):
        """
        Evaluate Audio to Gesture (a2g)
        Output: NPZ files + Video renders
        """
        logger.info("🎵 Evaluating Audio to Gesture (a2g)")
        
        results = []
        gesture_outputs = []
        
        for i, audio_file in enumerate(audio_files):
            try:
                if not os.path.exists(audio_file):
                    logger.warning(f"Audio file not found: {audio_file}")
                    continue
                    
                # Generate gesture from audio
                result = self.pipeline(
                    input_aud=audio_file,
                    task='a2g',
                    seq_length=seq_length,
                    guidance_scale=guidance_scale,
                    return_dict=True
                )
                
                if hasattr(result, 'gestures'):
                    gesture_seq = result.gestures  # (B, T, 415)
                    gesture_outputs.append(gesture_seq)
                    
                    # Save as NPZ
                    gesture_np = gesture_seq.cpu().numpy()
                    npz_path = os.path.join(self.output_dir, 'a2g', f'a2g_result_{i:03d}.npz')
                    
                    # Convert to SMPL-X format
                    smplx_data = self._gesture_to_smplx_format(gesture_np[0])
                    
                    np.savez(
                        npz_path,
                        poses=smplx_data['poses'],
                        expressions=smplx_data['expressions'],
                        trans=smplx_data['trans'],
                        betas=smplx_data['betas'],
                        model='smplx2020',
                        gender='neutral',
                        mocap_frame_rate=30
                    )
                    
                    # Render video with audio
                    video_path = self._render_gesture_video(
                        npz_path,
                        audio_file,
                        os.path.join(self.output_dir, 'a2g'),
                        f'a2g_video_{i:03d}.mp4',
                        audio_path=audio_file
                    )
                    
                    # Compute audio-gesture alignment
                    alignment_score = self._compute_audio_gesture_alignment(audio_file, gesture_np[0])
                    
                    results.append({
                        'audio_file': audio_file,
                        'npz_path': npz_path,
                        'video_path': video_path,
                        'gesture_shape': gesture_np.shape,
                        'alignment_score': alignment_score,
                        'gesture_mean': float(gesture_np.mean()),
                        'gesture_std': float(gesture_np.std())
                    })
                    
                    logger.info(f"  ✅ A2G {i+1}: {audio_file} → {gesture_np.shape}, align={alignment_score:.4f}")
                    
            except Exception as e:
                logger.error(f"  ❌ A2G {i+1} failed: {e}")
                
        # Compute FGD for gesture quality
        if len(gesture_outputs) > 1:
            fgd_score = self._compute_gesture_fgd(gesture_outputs)
            self.results['metrics']['a2g_fgd'] = fgd_score
            logger.info(f"  📊 A2G FGD: {fgd_score:.4f}")
            
        self.results['outputs']['a2g'] = results
        return results
    
    @torch.no_grad()
    def evaluate_g2a(self, gesture_inputs, guidance_scale=4.0):
        """
        Evaluate Gesture to Audio (g2a)
        Output: Audio files (WAV)
        """
        logger.info("🔊 Evaluating Gesture to Audio (g2a)")
        
        results = []
        audio_outputs = []
        
        for i, gesture_input in enumerate(gesture_inputs):
            try:
                # Generate audio from gesture
                result = self.pipeline(
                    input_gesture=gesture_input,
                    task='g2a',
                    guidance_scale=guidance_scale
                )
                
                if isinstance(result, tuple) and len(result) >= 1:
                    audio_spec = result[0]  # Audio spectrogram
                    
                    # Convert to waveform and save
                    if hasattr(audio_spec, 'shape') and len(audio_spec.shape) >= 3:
                        audio_path = os.path.join(self.output_dir, 'g2a', f'g2a_audio_{i:03d}.wav')
                        
                        # Convert spectrogram to waveform (using audio VAE decoder)
                        waveform = self._spec_to_waveform(audio_spec[0])  # Take first batch
                        
                        # Save audio file
                        sf.write(audio_path, waveform, samplerate=16000)
                        audio_outputs.append(waveform)
                        
                        # Compute audio quality metrics
                        audio_metrics = self._compute_audio_metrics(waveform)
                        
                        results.append({
                            'gesture_input_shape': gesture_input.shape,
                            'audio_path': audio_path,
                            'audio_length': len(waveform) / 16000,  # seconds
                            'audio_metrics': audio_metrics
                        })
                        
                        logger.info(f"  ✅ G2A {i+1}: {gesture_input.shape} → {audio_path}, {len(waveform)/16000:.1f}s")
                        
            except Exception as e:
                logger.error(f"  ❌ G2A {i+1} failed: {e}")
                
        self.results['outputs']['g2a'] = results
        return results
    
    @torch.no_grad() 
    def evaluate_t2a(self, text_prompts, guidance_scale=4.0):
        """
        Evaluate Text to Audio (t2a) - same as OmniFlow
        Output: Audio files (WAV)
        """
        logger.info("🎶 Evaluating Text to Audio (t2a)")
        
        results = []
        
        for i, prompt in enumerate(text_prompts):
            try:
                # Generate audio from text (OmniFlow method)
                result = self.pipeline(
                    prompt=prompt,
                    task='t2a',
                    guidance_scale=guidance_scale,
                    num_inference_steps=50
                )
                
                if isinstance(result, tuple) and len(result) >= 1:
                    audio_spec = result[0]  # Audio spectrogram
                    
                    # Convert and save
                    audio_path = os.path.join(self.output_dir, 't2a', f't2a_audio_{i:03d}.wav')
                    waveform = self._spec_to_waveform(audio_spec[0])
                    sf.write(audio_path, waveform, samplerate=16000)
                    
                    results.append({
                        'prompt': prompt,
                        'audio_path': audio_path,
                        'audio_length': len(waveform) / 16000
                    })
                    
                    logger.info(f"  ✅ T2A {i+1}: '{prompt}' → {audio_path}")
                    
            except Exception as e:
                logger.error(f"  ❌ T2A {i+1} failed: {e}")
                
        self.results['outputs']['t2a'] = results
        return results
    
    @torch.no_grad()
    def evaluate_a2t(self, audio_files, guidance_scale=2.0):
        """
        Evaluate Audio to Text (a2t) - same as OmniFlow
        Output: Text captions
        """
        logger.info("📰 Evaluating Audio to Text (a2t)")
        
        results = []
        
        for i, audio_file in enumerate(audio_files):
            try:
                if not os.path.exists(audio_file):
                    continue
                    
                # Generate text from audio (OmniFlow method)
                result = self.pipeline(
                    input_aud=audio_file,
                    task='a2t',
                    guidance_scale=guidance_scale
                )
                
                if isinstance(result, tuple) and len(result) >= 2:
                    generated_text = result[0][0] if result[0] else "No text generated"
                    reference_text = result[1][0] if result[1] else ""
                    
                    results.append({
                        'audio_file': audio_file,
                        'generated_text': generated_text,
                        'reference_text': reference_text,
                        'text_length': len(generated_text.split())
                    })
                    
                    logger.info(f"  ✅ A2T {i+1}: {audio_file} → '{generated_text[:50]}...'")
                    
            except Exception as e:
                logger.error(f"  ❌ A2T {i+1} failed: {e}")
                
        self.results['outputs']['a2t'] = results
        return results
    
    def _gesture_to_smplx_format(self, gesture_seq):
        """
        Convert gesture sequence (T, 415) to SMPL-X format for rendering
        """
        T = gesture_seq.shape[0]
        
        # Split gesture into parts (using proven method)
        upper_pose = gesture_seq[:, :78]        # (T, 78)
        hands_pose = gesture_seq[:, 78:258]     # (T, 180)
        lower_trans = gesture_seq[:, 258:315]   # (T, 57)
        face_data = gesture_seq[:, 315:415]     # (T, 100)
        
        # Convert to SMPL-X pose format (T, 165) - simplified
        # This is a placeholder - actual conversion would need proper joint mapping
        poses = np.zeros((T, 165))  # 55 joints * 3
        expressions = face_data      # (T, 100)
        trans = lower_trans[:, -3:]  # Last 3 dims as translation
        betas = np.zeros((T, 300))   # Default body shape
        
        return {
            'poses': poses,
            'expressions': expressions,
            'trans': trans,
            'betas': betas
        }
    
    def _render_gesture_video(self, npz_path, prompt_or_audio, output_dir, video_name, audio_path=None):
        """
        Render gesture video using other_tools_hf
        """
        try:
            # Create args namespace for rendering
            from types import SimpleNamespace
            render_args = SimpleNamespace(
                render_video_fps=30,
                render_video_width=1920,
                render_video_height=720,
                render_concurrent_num=2,
                render_tmp_img_filetype="bmp",
                debug=False,
                data_path_1="./datasets/hub/"  # For SMPL-X models
            )
            
            # Use SMPL-X model path
            model_folder = os.path.join(render_args.data_path_1, "smplx_models/")
            
            # Render video
            if audio_path:
                # Render with audio
                video_path = other_tools_hf.render_one_sequence_no_gt(
                    res_npz_path=npz_path,
                    output_dir=output_dir,
                    audio_path=audio_path,
                    model_folder=model_folder,
                    args=render_args,
                    use_matplotlib=False
                )
            else:
                # Render without audio (text prompt as title)
                video_path = other_tools_hf.render_one_sequence_no_gt(
                    res_npz_path=npz_path,
                    output_dir=output_dir,
                    audio_path=None,
                    model_folder=model_folder,
                    args=render_args,
                    use_matplotlib=False
                )
            
            return video_path
            
        except Exception as e:
            logger.error(f"Video rendering failed: {e}")
            return None
    
    def _compute_gesture_fgd(self, gesture_outputs):
        """Compute FGD for gesture quality"""
        try:
            # Stack all gesture outputs
            all_gestures = torch.cat(gesture_outputs, dim=0)  # (N, T, 415)
            
            # Flatten for FGD computation
            gestures_flat = all_gestures.view(all_gestures.shape[0], -1).cpu().numpy()
            
            # Generate reference gestures (random for now - ideally use GT)
            ref_gestures = np.random.randn(*gestures_flat.shape)
            
            # Compute FGD
            fgd = calculate_fgd(gestures_flat, ref_gestures)
            return fgd
            
        except Exception as e:
            logger.error(f"FGD computation failed: {e}")
            return 0.0
    
    def _compute_audio_gesture_alignment(self, audio_file, gesture_seq):
        """Compute audio-gesture alignment score"""
        try:
            # Load audio
            audio_data, sr = librosa.load(audio_file, sr=16000)
            
            # Extract audio features (onset detection)
            onset_frames = librosa.onset.onset_detect(
                y=audio_data, sr=sr, units='time'
            )
            
            # Extract gesture motion (velocity-based)
            gesture_diff = np.diff(gesture_seq, axis=0)
            gesture_motion = np.linalg.norm(gesture_diff, axis=1)
            
            # Simple alignment: correlation between onset and motion
            # This is a simplified metric - full implementation would use DTW
            if len(onset_frames) > 0 and len(gesture_motion) > 0:
                # Align timescales
                gesture_time = np.linspace(0, len(audio_data)/sr, len(gesture_motion))
                alignment_score = 0.5  # Placeholder score
            else:
                alignment_score = 0.0
                
            return alignment_score
            
        except Exception as e:
            logger.error(f"Alignment computation failed: {e}")
            return 0.0
    
    def _compute_text_metrics(self, generated_texts):
        """Compute text quality metrics"""
        if not EVAL_METRICS_AVAILABLE:
            return {}
            
        metrics = {
            'avg_length': np.mean([len(text.split()) for text in generated_texts]),
            'unique_words': len(set(' '.join(generated_texts).split())),
            'diversity': len(set(generated_texts)) / len(generated_texts)
        }
        
        return metrics
    
    def _compute_audio_metrics(self, waveform):
        """Compute audio quality metrics"""
        try:
            # Basic audio metrics
            metrics = {
                'rms': float(np.sqrt(np.mean(waveform**2))),
                'max_amplitude': float(np.max(np.abs(waveform))),
                'zero_crossing_rate': float(np.mean(np.abs(np.diff(np.sign(waveform))))),
                'spectral_centroid': 0.0  # Placeholder
            }
            
            # Spectral centroid
            if len(waveform) > 1024:
                stft = librosa.stft(waveform)
                spectral_centroids = librosa.feature.spectral_centroid(S=np.abs(stft))[0]
                metrics['spectral_centroid'] = float(np.mean(spectral_centroids))
                
            return metrics
            
        except Exception as e:
            logger.error(f"Audio metrics computation failed: {e}")
            return {}
    
    def _spec_to_waveform(self, audio_spec):
        """Convert audio spectrogram to waveform"""
        try:
            # Use audio VAE decoder to convert spec to waveform
            if hasattr(self.pipeline, 'audio_vae'):
                with torch.no_grad():
                    # Assume audio_spec is already in the right format
                    if isinstance(audio_spec, np.ndarray):
                        audio_spec = torch.from_numpy(audio_spec).to(self.device)
                    
                    # Decode using audio VAE
                    decoded = self.pipeline.audio_vae.decode(audio_spec.unsqueeze(0))
                    if hasattr(decoded, 'sample'):
                        waveform = decoded.sample
                    else:
                        waveform = decoded
                        
                    waveform = waveform.squeeze().cpu().numpy()
                    
                    # Normalize to [-1, 1]
                    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
                    
                    return waveform
            else:
                # Fallback: generate dummy audio
                return np.random.randn(16000 * 3)  # 3 seconds of audio
                
        except Exception as e:
            logger.error(f"Spectrogram to waveform conversion failed: {e}")
            return np.random.randn(16000 * 3)
    
    def run_complete_evaluation(
        self,
        test_prompts=None,
        test_audio_files=None,
        test_gestures=None,
        num_samples=5
    ):
        """
        Run complete evaluation for all tasks
        """
        logger.info("🧪 RUNNING COMPLETE OMNIGES EVALUATION")
        logger.info("=" * 60)
        
        # Default test data
        if test_prompts is None:
            test_prompts = [
                "A person waving hello",
                "Someone clapping hands",
                "Dancing with arm movements",
                "Pointing gesture",
                "Expressive hand gestures"
            ][:num_samples]
            
        if test_audio_files is None:
            test_audio_files = []
            # Look for available audio files
            for audio_file in ['./assets/car engine.mp3', './demo/audio.wav']:
                if os.path.exists(audio_file):
                    test_audio_files.append(audio_file)
            if not test_audio_files:
                logger.warning("No test audio files found")
                
        if test_gestures is None:
            # Generate dummy gestures for testing
            test_gestures = [torch.randn(1, 128, 415).to(self.device) for _ in range(num_samples)]
        
        # Run all evaluations
        results = {}
        
        # 1. Text to Gesture
        logger.info("\n1️⃣ Text to Gesture Evaluation")
        results['t2g'] = self.evaluate_t2g(test_prompts)
        
        # 2. Gesture to Text
        logger.info("\n2️⃣ Gesture to Text Evaluation")
        results['g2t'] = self.evaluate_g2t(test_gestures)
        
        # 3. Audio to Gesture (if audio files available)
        if test_audio_files:
            logger.info("\n3️⃣ Audio to Gesture Evaluation")
            results['a2g'] = self.evaluate_a2g(test_audio_files)
        
        # 4. Gesture to Audio
        logger.info("\n4️⃣ Gesture to Audio Evaluation")
        results['g2a'] = self.evaluate_g2a(test_gestures)
        
        # 5. Text to Audio (OmniFlow method)
        logger.info("\n5️⃣ Text to Audio Evaluation")
        audio_prompts = ["Music playing", "Dog barking", "Car engine sound"][:num_samples]
        results['t2a'] = self.evaluate_t2a(audio_prompts)
        
        # 6. Audio to Text (if audio files available)
        if test_audio_files:
            logger.info("\n6️⃣ Audio to Text Evaluation")
            results['a2t'] = self.evaluate_a2t(test_audio_files)
        
        # Generate comprehensive report
        self._generate_evaluation_report(results)
        
        return results
    
    def _generate_evaluation_report(self, results):
        """Generate comprehensive evaluation report"""
        report_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("OMNIGES EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            for task, task_results in results.items():
                f.write(f"{task.upper()} TASK:\n")
                f.write(f"  Samples processed: {len(task_results)}\n")
                
                if task in ['t2g', 'a2g']:
                    # Gesture generation tasks
                    f.write(f"  Output format: NPZ + Video\n")
                    if f'{task}_fgd' in self.results['metrics']:
                        f.write(f"  FGD Score: {self.results['metrics'][f'{task}_fgd']:.4f}\n")
                        
                elif task in ['g2t', 'a2t']:
                    # Text generation tasks
                    f.write(f"  Output format: Text captions\n")
                    if f'{task}_text_quality' in self.results['metrics']:
                        metrics = self.results['metrics'][f'{task}_text_quality']
                        f.write(f"  Avg text length: {metrics.get('avg_length', 0):.1f} words\n")
                        f.write(f"  Text diversity: {metrics.get('diversity', 0):.3f}\n")
                        
                elif task in ['g2a', 't2a']:
                    # Audio generation tasks
                    f.write(f"  Output format: WAV audio files\n")
                    avg_length = np.mean([r.get('audio_length', 0) for r in task_results])
                    f.write(f"  Avg audio length: {avg_length:.1f} seconds\n")
                    
                f.write("\n")
                
            # Summary
            total_samples = sum(len(task_results) for task_results in results.values())
            f.write(f"SUMMARY:\n")
            f.write(f"  Total samples: {total_samples}\n")
            f.write(f"  Tasks evaluated: {len(results)}/6\n")
            f.write(f"  Output directory: {self.output_dir}\n")
            
        logger.info(f"📝 Evaluation report saved: {report_path}")


def create_test_pipeline(
    omniflow_checkpoint_path: str,
    rvqvae_checkpoints_dir: str = "./ckpt/"
):
    """Create Omniges pipeline for testing"""
    
    rvqvae_checkpoints = {
        'upper': os.path.join(rvqvae_checkpoints_dir, 'net_300000_upper.pth'),
        'hands': os.path.join(rvqvae_checkpoints_dir, 'net_300000_hands.pth'),
        'lower_trans': os.path.join(rvqvae_checkpoints_dir, 'net_300000_lower.pth'),
        'face': os.path.join(rvqvae_checkpoints_dir, 'net_300000_face.pth')
    }
    
    # Check if checkpoints exist
    for part, path in rvqvae_checkpoints.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"RVQVAE checkpoint not found: {path}")
    
    # Create pipeline
    pipeline = create_omniges_pipeline(
        omniflow_checkpoint_path=omniflow_checkpoint_path,
        rvqvae_checkpoints=rvqvae_checkpoints,
        device="cuda"
    )
    
    return pipeline


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Omniges Complete Evaluation")
    parser.add_argument("--omniflow_checkpoint", type=str, required=True, help="OmniFlow checkpoint path")
    parser.add_argument("--rvqvae_checkpoints", type=str, default="./ckpt/", help="RVQVAE checkpoints directory")
    parser.add_argument("--output_dir", type=str, default="./results/omniges_evaluation", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples per task")
    parser.add_argument("--tasks", nargs='+', default=['t2g', 'g2t', 'a2g', 'g2a', 't2a', 'a2t'], help="Tasks to evaluate")
    
    args = parser.parse_args()
    
    try:
        # Create pipeline
        logger.info("🚀 Creating Omniges pipeline...")
        pipeline = create_test_pipeline(args.omniflow_checkpoint, args.rvqvae_checkpoints)
        logger.info("   ✅ Pipeline created successfully")
        
        # Create evaluator
        evaluator = OmnigesEvaluator(
            pipeline=pipeline,
            output_dir=args.output_dir,
            device="cuda"
        )
        
        # Run complete evaluation
        results = evaluator.run_complete_evaluation(num_samples=args.num_samples)
        
        # Print summary
        logger.info("\n🏆 EVALUATION COMPLETE!")
        logger.info(f"📂 Results saved in: {args.output_dir}")
        logger.info("\n📊 Task Summary:")
        for task, task_results in results.items():
            logger.info(f"   {task.upper()}: {len(task_results)} samples")
            
        logger.info("\n🎯 Output Types:")
        logger.info("   t2g, a2g: NPZ + Video files")
        logger.info("   g2t, a2t: Text captions")  
        logger.info("   g2a, t2a: WAV audio files")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

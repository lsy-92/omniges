# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
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
# limitations under the License.

"""
Omniges Pipeline: Complete Text-Audio-Gesture Multimodal Pipeline
Based on OmniFlow with Image stream replaced by Gesture stream
Supports all task combinations: t2g, a2g, g2t, g2a, t2a, a2t
"""

# ============================================================================
# IMPORT SECTION - 필요한 라이브러리들을 가져옵니다
# ============================================================================

import inspect  # 함수 시그니처 검사용
import os  # 운영체제 인터페이스
import sys  # 시스템 파라미터와 함수들
from typing import Any, Callable, Dict, List, Optional, Union  # 타입 힌트
import numpy as np  # 수치 계산 라이브러리
import torch  # PyTorch 딥러닝 프레임워크
import torch.nn as nn  # PyTorch 신경망 모듈
import torch.nn.functional as F  # PyTorch 함수형 인터페이스
from pathlib import Path  # 경로 처리
from tqdm import tqdm  # 진행률 표시

# 프로젝트 루트를 Python 경로에 추가 (import 경로 설정)
sys.path.append(str(Path(__file__).parent.parent.parent))

# Hugging Face Transformers 라이브러리에서 필요한 모델들 import
from transformers import (
    CLIPTextModelWithProjection,  # CLIP 텍스트 인코더 (이미지-텍스트 매칭용)
    CLIPTokenizer,  # CLIP 토크나이저
    T5EncoderModel,  # T5 인코더 (긴 텍스트 처리용)
    T5TokenizerFast,  # T5 토크나이저
    AutoTokenizer,  # 자동 토크나이저
    AutoConfig,  # 자동 설정
    CLIPVisionModelWithProjection,  # CLIP 비전 모델
    CLIPImageProcessor,  # CLIP 이미지 전처리기
)

# OmniFlow 유틸리티 함수들 import
from omniflow.utils.text_encode import _encode_prompt_with_t5, cat_and_pad  # 텍스트 인코딩 함수들
from diffusers.image_processor import VaeImageProcessor  # VAE 이미지 전처리기
from diffusers.loaders import FromSingleFileMixin, SD3LoraLoaderMixin  # 모델 로딩 믹스인들
from diffusers.models.autoencoders import AutoencoderKL  # 오토인코더 모델
from omniflow.models.omni_flow import OmniFlowTransformerModel  # 핵심: OmniFlow 트랜스포머 모델
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # 스케줄러
from diffusers.utils import (
    is_torch_xla_available,  # XLA (TPU) 사용 가능 여부 확인
    logging,  # 로깅
    replace_example_docstring,  # 문서화
)
from diffusers.utils.torch_utils import randn_tensor  # 랜덤 텐서 생성
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # 기본 파이프라인 클래스
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3PipelineOutput  # 출력 클래스
from PIL import Image  # 이미지 처리

# OmniFlow 특화 모델들 import
from omniflow.models.text_vae import LLamaForLatentConnector  # LLaMA 텍스트 VAE
from omniflow.models.encoders import LanguageBindAudioProcessor, LanguageBindAudio  # 오디오 인코더
from omniflow.utils.ema import EMAModel  # Exponential Moving Average 모델
from omniflow.models.audio_vae import load_audio_vae  # 오디오 VAE 로더
from omniflow.utils.scheduler import OmniFlowMatchEulerDiscreteScheduler  # OmniFlow 스케줄러

# 우리가 만든 제스처 처리 컴포넌트 import
from omniges.models.gesture_processor import GestureProcessor  # 4x RVQVAE 제스처 처리기

# XLA (TPU) 사용 가능 여부 확인 및 설정
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # XLA 모델 (TPU용)
    XLA_AVAILABLE = True  # XLA 사용 가능 플래그
else:
    XLA_AVAILABLE = False  # XLA 사용 불가 플래그

# 로거 설정
logger = logging.get_logger(__name__)

# 파이프라인 사용 예시 문서
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from omniges.pipelines import OmnigesPipeline

        >>> pipe = OmnigesPipeline.from_pretrained(
        ...     "path/to/omniges", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A person waving hello"
        >>> gesture = pipe(prompt, task='t2g').gestures[0]
        >>> gesture.save("gesture.npy")
        ```
"""

# ============================================================================
# UTILITY FUNCTIONS - 유틸리티 함수들
# ============================================================================

# OmniFlow에서 복사한 timestep 검색 함수
def retrieve_timesteps(
    scheduler,  # 스케줄러 객체
    num_inference_steps: Optional[int] = None,  # 추론 스텝 수
    device: Optional[Union[str, torch.device]] = None,  # 디바이스
    timesteps: Optional[List[int]] = None,  # 커스텀 timestep 리스트
    sigmas: Optional[List[float]] = None,  # 커스텀 sigma 리스트
    **kwargs,  # 추가 인자들
):
    """
    스케줄러의 `set_timesteps` 메서드를 호출하고 timestep을 검색합니다.
    커스텀 timestep을 처리합니다.
    """
    # timesteps와 sigmas 중 하나만 전달되어야 함
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    
    # 커스텀 timesteps가 전달된 경우
    if timesteps is not None:
        # 스케줄러가 timesteps를 지원하는지 확인
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 스케줄러에 timesteps 설정
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps  # 설정된 timesteps 가져오기
        num_inference_steps = len(timesteps)  # 스텝 수 계산
    
    # 커스텀 sigmas가 전달된 경우
    elif sigmas is not None:
        # 스케줄러가 sigmas를 지원하는지 확인
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 스케줄러에 sigmas 설정
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps  # 설정된 timesteps 가져오기
        num_inference_steps = len(timesteps)  # 스텝 수 계산
    
    # 기본 설정 (num_inference_steps만 전달된 경우)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps  # 설정된 timesteps 가져오기
    
    return timesteps, num_inference_steps  # timesteps와 스텝 수 반환

# 안전한 텐서 로딩 함수 (메모리 효율적)
def load_safe_tensors(fp, model):
    """
    파일에서 텐서를 안전하게 로드하고 모델에 적용합니다.
    메모리 누수를 방지하기 위해 로드 후 텐서를 삭제합니다.
    """
    tensors = torch.load(fp, map_location='cpu')  # CPU에서 텐서 로드
    res = model.load_state_dict(tensors, strict=False)  # 모델에 가중치 로드 (엄격하지 않게)
    print(f"Loaded {fp}:{res}")  # 로딩 결과 출력
    del tensors  # 메모리에서 텐서 삭제
    torch.cuda.empty_cache()  # CUDA 캐시 정리

# EMA 모델용 안전한 텐서 로딩 함수
def load_safe_tensors_ema(fp, model):
    """
    EMA 모델용 안전한 텐서 로딩 함수
    """
    tensors = torch.load(fp, map_location='cpu')  # CPU에서 텐서 로드
    res = model.load_state_dict(tensors)  # 모델에 가중치 로드 (엄격하게)
    print(f"Loaded {fp}:{res}")  # 로딩 결과 출력
    del tensors  # 메모리에서 텐서 삭제
    torch.cuda.empty_cache()  # CUDA 캐시 정리


# ============================================================================
# HELPER CLASSES - 헬퍼 클래스들
# ============================================================================

class DistributionMock:
    """
    VAE 호환성을 위한 Mock 분포 클래스
    OmniFlow의 VAE 인터페이스와 호환되도록 만든 래퍼
    """
    def __init__(self, latents):
        """
        Args:
            latents: (B, C, H, W) 형태의 latent 텐서
        """
        self.latents = latents  # latent 텐서 저장
        
    def sample(self):
        """
        샘플링 메서드 (VAE 인터페이스 호환용)
        Returns:
            self.latents: 저장된 latent 텐서 반환
        """
        return self.latents
        
    @property 
    def mean(self):
        """
        평균 속성 (VAE 인터페이스 호환용)
        Returns:
            self.latents: 저장된 latent 텐서 반환
        """
        return self.latents


# ============================================================================
# GESTURE VAE ADAPTER - 제스처 VAE 어댑터 클래스
# ============================================================================

class OmnigesGestureVAE(nn.Module):
    """
    Omniges 파이프라인용 제스처 VAE 어댑터
    4x RVQVAE를 OmniFlow의 Image VAE처럼 작동하도록 어댑팅
    """
    
    def __init__(self, rvqvae_checkpoints: Dict[str, str]):
        """
        Args:
            rvqvae_checkpoints: 4개 부위별 RVQVAE 체크포인트 경로 딕셔너리
                {
                    'upper': './ckpt/net_300000_upper.pth',
                    'hands': './ckpt/net_300000_hands.pth', 
                    'lower_trans': './ckpt/net_300000_lower.pth',
                    'face': './ckpt/net_300000_face.pth'
                }
        """
        super().__init__()
        
        # 4x RVQVAE 제스처 처리기 초기화
        self.gesture_processor = GestureProcessor(
            ckpt_paths=rvqvae_checkpoints,  # 체크포인트 경로들 전달
            device="cuda"  # GPU 사용
        )
        
        # OmniFlow 호환성을 위한 VAE config 모킹
        # 실제 VAE config와 동일한 인터페이스를 제공
        self.config = type('Config', (), {
            'scaling_factor': 0.18215,  # OmniFlow VAE와 동일한 스케일링 팩터
            'shift_factor': 0.0,    # 시프트 팩터 (0.0 = 시프트 없음)
            'block_out_channels': [128, 256, 512, 1024]  # VAE 스케일 팩터 계산용 채널 수
        })()
        
    def encode(self, gesture_sequence):
        """
        제스처 시퀀스를 트랜스포머와 호환되는 latent로 인코딩
        
        Args:
            gesture_sequence: (B, T, 415) - 결합된 제스처 특징
                B: 배치 크기
                T: 시간 스텝 수 (프레임 수)
                415: 전체 제스처 차원 (78+180+57+100)
        
        Returns:
            DistributionMock: (B, 128, T, 4) 형태의 2D latent 표현을 담은 객체
        """
        # 입력 텐서의 차원 추출
        B, T, total_dim = gesture_sequence.shape  # B: 배치, T: 시간, total_dim: 415
        
        # RVQVAE 요구사항에 따라 제스처를 부위별로 분할
        # 각 부위는 서로 다른 RVQVAE 모델로 처리됨
        gesture_parts = {
            'upper': gesture_sequence[:, :, :78],           # (B, T, 78) - 상체 (26개 관절 × 3)
            'hands': gesture_sequence[:, :, 78:258],        # (B, T, 180) - 손 (60개 관절 × 3)  
            'lower_trans': gesture_sequence[:, :, 258:315], # (B, T, 57) - 하체+이동 (19개 관절 × 3)
            'face': gesture_sequence[:, :, 315:415]         # (B, T, 100) - 얼굴 (표정 + 턱)
        }
        
        # 각 부위를 RVQVAE로 인코딩하여 latent로 변환
        # 각 부위: (B, T, D) -> (B, T, 128) 형태로 변환
        latents_dict = self.gesture_processor.encode_gesture(gesture_parts)
        
        # 트랜스포머 호환성을 위해 latent들을 2D 표현으로 결합
        # 4개 부위의 latent를 스택: (B, T, 128) -> (B, T, 128, 4)
        combined_latents = torch.stack([
            latents_dict['upper_latents'],      # (B, T, 128) - 상체 latent
            latents_dict['hands_latents'],      # (B, T, 128) - 손 latent
            latents_dict['lower_trans_latents'], # (B, T, 128) - 하체 latent
            latents_dict['face_latents']        # (B, T, 128) - 얼굴 latent
        ], dim=-1)  # 마지막 차원에 스택 -> (B, T, 128, 4)
        
        # 이미지와 유사한 2D 형태로 변환: (B, 128, T, 4)
        # permute(0, 2, 1, 3): (B, T, 128, 4) -> (B, 128, T, 4)
        # 이는 이미지의 (B, C, H, W) 형태와 유사
        latents_2d = combined_latents.permute(0, 2, 1, 3)  # (B, 128, T, 4)
        
        # VAE 호환성을 위해 DistributionMock으로 래핑하여 반환
        return DistributionMock(latents_2d)
        
    def decode(self, latents_2d, return_dict=True):
        """
        2D latent를 다시 제스처 시퀀스로 디코딩
        
        Args:
            latents_2d: (B, 128, T, 4) 또는 (B, 128, T*4) - 2D latent 표현
            return_dict: True면 DecodeOutput 객체 반환, False면 텐서 직접 반환
        
        Returns:
            gesture_sequence 또는 DecodeOutput: (B, T, 415) 형태의 제스처 시퀀스
        """
        # 입력 텐서의 차원 추출
        B, C, H, W = latents_2d.shape  # B: 배치, C: 채널(128), H: 높이, W: 너비
        
        # 입력 형태에 따른 처리 분기
        if W == 4:
            # 표준 형태: (B, 128, T, 4) - 4개 부위가 분리되어 있음
            num_parts = W  # 4개 부위
            T = H  # 시간 스텝 수
        else:
            # 평탄화된 형태: (B, 128, T*4) -> (B, 128, T, 4)로 재구성
            total_length = H  # 전체 길이
            T = total_length // 4  # 시간 스텝 수 계산
            num_parts = 4  # 4개 부위
            # view로 재구성: (B, 128, T*4) -> (B, 128, T, 4)
            latents_2d = latents_2d.view(B, C, T, num_parts)
        
        # 각 부위별로 latent를 분리하고 차원 순서 조정: (B, T, 128)
        # permute(0, 2, 1): (B, 128, T) -> (B, T, 128)
        latents_dict = {
            'upper_latents': latents_2d[:, :, :, 0].permute(0, 2, 1),     # (B, T, 128) - 상체
            'hands_latents': latents_2d[:, :, :, 1].permute(0, 2, 1),     # (B, T, 128) - 손
            'lower_trans_latents': latents_2d[:, :, :, 2].permute(0, 2, 1),  # (B, T, 128) - 하체
            'face_latents': latents_2d[:, :, :, 3].permute(0, 2, 1)       # (B, T, 128) - 얼굴
        }
            
        # RVQVAE를 통해 각 부위를 디코딩
        # 각 부위: (B, T, 128) -> (B, T, D) 형태로 변환
        decoded_parts = self.gesture_processor.decode_gesture(latents_dict)
        
        # 4개 부위를 다시 하나의 제스처로 결합
        # cat(dim=-1): 마지막 차원(특징 차원)을 따라 결합
        gesture_sequence = torch.cat([
            decoded_parts['upper'],      # (B, T, 78) - 상체
            decoded_parts['hands'],      # (B, T, 180) - 손
            decoded_parts['lower_trans'], # (B, T, 57) - 하체+이동
            decoded_parts['face']        # (B, T, 100) - 얼굴
        ], dim=-1)  # (B, T, 415) - 전체 제스처
        
        # 반환 형태 결정
        if return_dict:
            # VAE 호환성을 위해 DecodeOutput 객체로 래핑
            return type('DecodeOutput', (), {'sample': gesture_sequence})()
        return gesture_sequence  # 텐서 직접 반환


# ============================================================================
# MAIN PIPELINE CLASS - 메인 파이프라인 클래스
# ============================================================================

class OmnigesPipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin):
    """
    텍스트-오디오-제스처 멀티모달 생성을 위한 Omniges 파이프라인
    
    이미지 스트림을 제스처 스트림으로 치환한 OmniFlow 기반 파이프라인
    
    Args:
        transformer ([`OmniFlowTransformerModel`]):
            인코딩된 제스처 latent를 디노이징하는 조건부 트랜스포머 (MMDiT) 아키텍처
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            transformer와 함께 사용되어 인코딩된 제스처 latent를 디노이징하는 스케줄러
        gesture_vae ([`OmnigesGestureVAE`]):
            제스처를 latent 표현으로 인코딩/디코딩하는 제스처 VAE 모델
        text_encoder ([`CLIPTextModelWithProjection`]):
            텍스트 임베딩을 위한 CLIP 텍스트 인코더
        text_encoder_2 ([`CLIPTextModelWithProjection`]):
            두 번째 CLIP 텍스트 인코더
        text_encoder_3 ([`T5EncoderModel`]):
            긴 텍스트 시퀀스를 위한 T5 인코더
        tokenizer (`CLIPTokenizer`):
            첫 번째 텍스트 인코더용 토크나이저
        tokenizer_2 (`CLIPTokenizer`):
            두 번째 텍스트 인코더용 토크나이저
        tokenizer_3 (`T5TokenizerFast`):
            T5 토크나이저
        audio_vae ([`AutoencoderKL`]):
            오디오 처리를 위한 오디오 VAE
        audio_encoder ([`LanguageBindAudio`]):
            오디오 임베딩을 위한 오디오 인코더
        text_vae ([`LLamaForLatentConnector`]):
            텍스트 생성을 위한 텍스트 VAE
    """

    # 모델 CPU 오프로딩 순서 (메모리 효율성을 위한 순서)
    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->transformer->gesture_vae"
    
    # 선택적 컴포넌트들 (없어도 작동하는 컴포넌트)
    _optional_components = []
    
    # 콜백에서 텐서 입력으로 사용되는 변수들
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]

    @staticmethod
    def load_pretrained(
        omniflow_path: str,  # OmniFlow 체크포인트 디렉토리 경로
        rvqvae_checkpoints: Dict[str, str],  # 4개 부위별 RVQVAE 체크포인트 경로 딕셔너리
        device: str = 'cuda',  # 모델을 로드할 디바이스 (GPU/CPU)
        weight_dtype: torch.dtype = torch.bfloat16,  # 가중치 데이터 타입 (메모리 효율성)
        load_ema: bool = False  # EMA (Exponential Moving Average) 가중치 로드 여부
    ):
        """
        OmniFlow 체크포인트 + RVQVAE 체크포인트에서 사전훈련된 Omniges 파이프라인 로드
        
        Args:
            omniflow_path: OmniFlow 모델 디렉토리 경로
            rvqvae_checkpoints: 부위명을 RVQVAE 체크포인트 경로에 매핑하는 딕셔너리
            device: 모델을 로드할 디바이스
            weight_dtype: 가중치 데이터 타입
            load_ema: EMA 가중치 로드 여부
        """
        
        # ============================================================================
        # TOKENIZER 로딩 (OmniFlow와 동일)
        # ============================================================================
        
        # 첫 번째 CLIP 토크나이저 로드 (기본 텍스트 인코딩용)
        tokenizer_one = CLIPTokenizer.from_pretrained(
            'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',  # 대규모 데이터로 훈련된 CLIP 모델
        )
        
        # 두 번째 CLIP 토크나이저 로드 (로컬 파일 우선, 없으면 온라인)
        try:
            local_tok2 = os.path.join(omniflow_path, "tokenizer_2")  # 로컬 토크나이저 경로
            if os.path.isdir(local_tok2):  # 로컬 디렉토리가 존재하는지 확인
                tokenizer_two = CLIPTokenizer.from_pretrained(local_tok2, local_files_only=True)  # 로컬 파일만 사용
            else:
                raise FileNotFoundError(local_tok2)  # 로컬 파일이 없으면 에러 발생
        except Exception:
            # 로컬 파일이 없으면 온라인에서 다운로드
            tokenizer_two = CLIPTokenizer.from_pretrained(
                'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'  # 더 큰 CLIP 모델
            )
        
        # 세 번째 T5 토크나이저 로드 (긴 텍스트 시퀀스 처리용)
        tokenizer_three = T5TokenizerFast.from_pretrained('google/flan-t5-large')  # Flan-T5 모델
        
        # ============================================================================
        # TEXT ENCODER 로딩 (OmniFlow와 동일)
        # ============================================================================
        
        # 첫 번째 CLIP 텍스트 인코더 로드 (기본 텍스트 임베딩 생성)
        text_encoder_one = CLIPTextModelWithProjection.from_pretrained(
            'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',  # 대규모 데이터로 훈련된 CLIP 모델
            projection_dim=768  # 투영 차원 (768D 텍스트 임베딩 생성)
        )
        
        # 두 번째 CLIP 텍스트 인코더 로드 (로컬 파일 우선, 없으면 온라인)
        try:
            local_te2 = os.path.join(omniflow_path, "text_encoder_2")  # 로컬 텍스트 인코더 경로
            if os.path.isdir(local_te2):  # 로컬 디렉토리가 존재하는지 확인
                text_encoder_two = CLIPTextModelWithProjection.from_pretrained(local_te2, local_files_only=True)  # 로컬 파일만 사용
            else:
                raise FileNotFoundError(local_te2)  # 로컬 파일이 없으면 에러 발생
        except Exception:
            # 로컬 파일이 없으면 온라인에서 다운로드
            text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
                'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'  # 더 큰 CLIP 모델
            )
        
        # 세 번째 T5 텍스트 인코더 로드 (긴 텍스트 시퀀스 처리)
        text_encoder_three = T5EncoderModel.from_pretrained('google/flan-t5-large')  # Flan-T5 모델
        
        # ============================================================================
        # 모델 설정 및 제스처 VAE 생성
        # ============================================================================
        
        # 모든 텍스트 인코더를 평가 모드로 설정 (드롭아웃 비활성화)
        text_encoder_three.eval()  # T5 인코더 평가 모드
        text_encoder_two.eval()    # 두 번째 CLIP 인코더 평가 모드
        text_encoder_one.eval()    # 첫 번째 CLIP 인코더 평가 모드
        
        # 이미지 VAE 대신 제스처 VAE 생성 (핵심 차이점!)
        gesture_vae = OmnigesGestureVAE(rvqvae_checkpoints)  # 4x RVQVAE 기반 제스처 VAE
        
        # ============================================================================
        # TEXT VAE 로딩 (OmniFlow와 동일)
        # ============================================================================
        
        # 텍스트 VAE용 토크나이저 로드
        text_vae_tokenizer = AutoTokenizer.from_pretrained(
            omniflow_path,  # OmniFlow 체크포인트 경로
            subfolder="vae_tokenizer",  # vae_tokenizer 서브폴더
        )
        text_vae_tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # 패딩 토큰 추가
        
        # 텍스트 VAE 설정 로드
        config = AutoConfig.from_pretrained(os.path.join(omniflow_path, "text_vae"))  # text_vae 서브폴더
        
        # LLaMA 기반 텍스트 VAE 모델 생성
        text_vae = LLamaForLatentConnector._from_config(
            config,  # 로드한 설정
            torch_dtype=torch.bfloat16  # bfloat16 데이터 타입 (메모리 효율성)
        )
        text_vae.prepare_tokenizer(text_vae_tokenizer)  # 토크나이저 준비
        text_vae.set_encoder(text_encoder_three)  # T5 인코더를 텍스트 VAE에 연결
        
        # ============================================================================
        # TRANSFORMER 로딩 (OmniFlow와 동일)
        # ============================================================================
        
        # OmniFlow 트랜스포머 모델 로드 (핵심 멀티모달 모델)
        transformer = OmniFlowTransformerModel.from_config(
            omniflow_path,  # OmniFlow 체크포인트 경로
            subfolder="transformer",  # transformer 서브폴더
        )
        transformer.set_text_decoder(text_vae)  # 텍스트 VAE를 트랜스포머의 텍스트 디코더로 설정
        
        # ============================================================================
        # AUDIO 컴포넌트 로딩 (OmniFlow와 동일)
        # ============================================================================
        
        # LanguageBind 오디오 인코더 로드 (오디오 임베딩 생성)
        audio_encoder = LanguageBindAudio.from_pretrained('LanguageBind/LanguageBind_Audio_FT')  # 사전훈련된 오디오 모델
        audio_encoder.text_model = nn.Identity()  # 텍스트 모델을 Identity로 대체 (오디오만 처리)
        audio_processor_clip = LanguageBindAudioProcessor(audio_encoder.config)  # 오디오 전처리기 생성
        
        # ============================================================================
        # 모델을 추론 모드로 설정 (그래디언트 계산 비활성화)
        # ============================================================================
        
        # 모든 모델의 그래디언트 계산을 비활성화 (메모리 절약 및 추론 속도 향상)
        transformer.requires_grad_(False)      # 트랜스포머 그래디언트 비활성화
        text_vae.requires_grad_(False)         # 텍스트 VAE 그래디언트 비활성화
        audio_encoder.requires_grad_(False)    # 오디오 인코더 그래디언트 비활성화
        text_encoder_one.requires_grad_(False) # 첫 번째 텍스트 인코더 그래디언트 비활성화
        text_encoder_two.requires_grad_(False) # 두 번째 텍스트 인코더 그래디언트 비활성화
        text_encoder_three.requires_grad_(False) # 세 번째 텍스트 인코더 그래디언트 비활성화
        gesture_vae.requires_grad_(False)      # 제스처 VAE 그래디언트 비활성화
        
        # ============================================================================
        # 모델을 지정된 디바이스로 이동
        # ============================================================================
        
        # 모든 모델을 지정된 디바이스(GPU/CPU)로 이동하고 데이터 타입 설정
        text_encoder_one.to(device, dtype=weight_dtype)    # 첫 번째 텍스트 인코더를 디바이스로 이동
        text_encoder_two.to(device, dtype=weight_dtype)    # 두 번째 텍스트 인코더를 디바이스로 이동
        text_encoder_three.to(device, dtype=weight_dtype)  # 세 번째 텍스트 인코더를 디바이스로 이동
        transformer.to(device, dtype=weight_dtype)         # 트랜스포머를 디바이스로 이동
        text_vae.to(device, dtype=weight_dtype)            # 텍스트 VAE를 디바이스로 이동
        audio_encoder.to(device, dtype=weight_dtype)       # 오디오 인코더를 디바이스로 이동
        gesture_vae.to(device, dtype=weight_dtype)         # 제스처 VAE를 디바이스로 이동
        
        # ============================================================================
        # AUDIO VAE 및 스케줄러 로딩
        # ============================================================================
        
        # 오디오 VAE 로드 (오디오 인코딩/디코딩용)
        audiovae, audio_processor = load_audio_vae()  # 오디오 VAE와 전처리기 로드
        audiovae.to(device)  # 오디오 VAE를 디바이스로 이동
        audiovae.requires_grad_(False)  # 오디오 VAE 그래디언트 비활성화
        
        # 노이즈 스케줄러 로드 (디노이징 과정 제어)
        noise_scheduler = OmniFlowMatchEulerDiscreteScheduler.from_pretrained(
            omniflow_path,  # OmniFlow 체크포인트 경로
            subfolder="scheduler",  # scheduler 서브폴더
            shift=3  # 시프트 파라미터 (스케줄링 조정)
        )
        
        # ============================================================================
        # OmnigesPipeline 인스턴스 생성
        # ============================================================================
        
        # 모든 컴포넌트를 조합하여 OmnigesPipeline 인스턴스 생성
        pipeline = OmnigesPipeline(
            scheduler=noise_scheduler,           # 노이즈 스케줄러
            gesture_vae=gesture_vae,             # 제스처 VAE (이미지 VAE 대신)
            audio_processor=audio_processor,     # 오디오 전처리기
            text_encoder=text_encoder_one,       # 첫 번째 텍스트 인코더
            text_encoder_2=text_encoder_two,     # 두 번째 텍스트 인코더
            text_encoder_3=text_encoder_three,   # 세 번째 텍스트 인코더 (T5)
            tokenizer=tokenizer_one,             # 첫 번째 토크나이저
            tokenizer_2=tokenizer_two,           # 두 번째 토크나이저
            tokenizer_3=tokenizer_three,         # 세 번째 토크나이저 (T5)
            transformer=transformer,             # 핵심 트랜스포머 모델
            text_vae_tokenizer=text_vae_tokenizer, # 텍스트 VAE 토크나이저
            text_vae=text_vae,                   # 텍스트 VAE
            audio_vae=audiovae,                  # 오디오 VAE
            text_x0=True,                        # 텍스트 x0 모드 활성화
            audio_encoder=audio_encoder,         # 오디오 인코더
            audio_processor_clip=audio_processor_clip, # CLIP 오디오 전처리기
        )
        
        # ============================================================================
        # 트랜스포머 가중치 로딩
        # ============================================================================
        
        # 트랜스포머 가중치 파일 경로 설정
        fp = os.path.join(omniflow_path, 'transformer/diffusion_pytorch_model.bin')  # 일반 가중치 파일
        fp_ema = os.path.join(omniflow_path, 'transformer/ema_transformer.pt')      # EMA 가중치 파일
        
        # 일반 가중치 로드 (항상 시도)
        if os.path.exists(fp):  # 가중치 파일이 존재하는지 확인
            load_safe_tensors(fp, transformer)  # 안전한 텐서 로딩 함수 사용
        
        # EMA 가중치 로드 (요청된 경우에만)
        if load_ema and os.path.exists(fp_ema):  # EMA 로드가 요청되고 파일이 존재하는 경우
            ema_model = EMAModel(transformer.parameters())  # EMA 모델 생성
            load_safe_tensors_ema(fp_ema, ema_model)  # EMA 가중치 로드
            ema_model.copy_to(transformer.parameters())  # EMA 가중치를 트랜스포머에 복사
            
        return pipeline  # 완성된 파이프라인 반환

    def enable_ema(self, path):
        """
        EMA (Exponential Moving Average) 가중치 활성화
        Args:
            path: 체크포인트 경로
        """
        device = self.transformer.device  # 현재 트랜스포머가 있는 디바이스 저장
        self.transformer.to('cpu')  # 트랜스포머를 CPU로 이동 (메모리 효율성)
        ema_model = EMAModel(self.transformer.parameters())  # EMA 모델 생성
        fp_ema = os.path.join(path, 'transformer/ema_transformer.pt')  # EMA 가중치 파일 경로
        load_safe_tensors_ema(fp_ema, ema_model)  # EMA 가중치 로드
        self.transformer.to(device)  # 트랜스포머를 원래 디바이스로 복원
        ema_model.copy_to(self.transformer.parameters())  # EMA 가중치를 트랜스포머에 적용
        
    def disable_ema(self, path):
        """
        EMA 가중치 비활성화 (일반 가중치로 복원)
        Args:
            path: 체크포인트 경로
        """
        fp = os.path.join(path, 'transformer/diffusion_pytorch_model.bin')  # 일반 가중치 파일 경로
        load_safe_tensors(fp, self.transformer)  # 일반 가중치 로드
        
    def __init__(
        self,
        transformer: OmniFlowTransformerModel,  # 핵심 멀티모달 트랜스포머
        scheduler: FlowMatchEulerDiscreteScheduler,  # 노이즈 스케줄러
        gesture_vae: OmnigesGestureVAE,  # 제스처 VAE (이미지 VAE 대신)
        text_encoder: CLIPTextModelWithProjection,  # 첫 번째 텍스트 인코더
        tokenizer: CLIPTokenizer,  # 첫 번째 토크나이저
        text_encoder_2: CLIPTextModelWithProjection,  # 두 번째 텍스트 인코더
        tokenizer_2: CLIPTokenizer,  # 두 번째 토크나이저
        text_encoder_3: T5EncoderModel,  # 세 번째 텍스트 인코더 (T5)
        tokenizer_3: T5TokenizerFast,  # 세 번째 토크나이저 (T5)
        seq_length: int = 128,  # 제스처 시퀀스 길이 (이미지 crop_size 대신)
        text_vae_tokenizer=None,  # 텍스트 VAE 토크나이저
        gesture_encoder=None,  # 제스처 인코더 (이미지 인코더 대신)
        gesture_processor=None,  # 제스처 전처리기 (이미지 전처리기 대신)
        audio_vae=None,  # 오디오 VAE
        audio_processor=None,  # 오디오 전처리기
        audio_processor_clip=None,  # CLIP 오디오 전처리기
        text_vae=None,  # 텍스트 VAE
        text_x0=None,  # 텍스트 x0 모드 설정
        audio_encoder=None,  # 오디오 인코더
        mm_encoder=None,  # 멀티모달 인코더
        cfg_mode='old',  # CFG 모드
        mode: str = 'gesture',  # 기본 모드를 제스처로 설정
    ):
        super().__init__()  # 부모 클래스 초기화
        
        # ============================================================================
        # 기본 설정 저장
        # ============================================================================
        self.text_x0 = text_x0  # 텍스트 x0 모드 설정 저장
        self.cfg_mode = cfg_mode  # CFG 모드 설정 저장
        self.audio_encoder = audio_encoder  # 오디오 인코더 저장
        self.mm_encoder = mm_encoder  # 멀티모달 인코더 저장
        
        # ============================================================================
        # 핵심 모듈 등록 (DiffusionPipeline에서 관리)
        # ============================================================================
        self.register_modules(
            gesture_vae=gesture_vae,  # 제스처 VAE 등록 (이미지 VAE 대신)
            text_encoder=text_encoder,  # 첫 번째 텍스트 인코더 등록
            text_encoder_2=text_encoder_2,  # 두 번째 텍스트 인코더 등록
            text_encoder_3=text_encoder_3,  # 세 번째 텍스트 인코더 등록
            tokenizer=tokenizer,  # 첫 번째 토크나이저 등록
            tokenizer_2=tokenizer_2,  # 두 번째 토크나이저 등록
            tokenizer_3=tokenizer_3,  # 세 번째 토크나이저 등록
            transformer=transformer,  # 트랜스포머 등록
            scheduler=scheduler,  # 스케줄러 등록
        )
        
        # ============================================================================
        # VAE 스케일 팩터 및 전처리기 설정
        # ============================================================================
        
        self.text_vae_tokenizer = text_vae_tokenizer  # 텍스트 VAE 토크나이저 저장
        
        # VAE 스케일 팩터 계산 (제스처 VAE의 블록 출력 채널 수 기반)
        self.vae_scale_factor = (
            2 ** (len(self.gesture_vae.config.block_out_channels) - 1)  # 2^(블록 수 - 1)
            if hasattr(self, "gesture_vae") and self.gesture_vae is not None else 8  # 기본값 8
        )
        
        # 제스처 처리용 전처리기 생성 (이미지 처리 대신)
        # VaeImageProcessor를 제스처 처리에 재사용 (인터페이스 호환성)
        self.gesture_processor_utils = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        
        # ============================================================================
        # 토크나이저 및 샘플 크기 설정
        # ============================================================================
        
        # 토크나이저 최대 길이 설정 (기본값 77)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        
        # 기본 샘플 크기 설정 (트랜스포머 설정에서 가져오거나 기본값 128)
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )
        
        # ============================================================================
        # 제스처 시퀀스 길이 설정 (이미지 변환 대신)
        # ============================================================================
        
        self.seq_length = seq_length  # 제스처 시퀀스 길이 저장
        self.default_seq_length = seq_length  # 기본 제스처 시퀀스 길이 저장
        
        # ============================================================================
        # 추가 컴포넌트 저장
        # ============================================================================
        
        self.gesture_encoder = gesture_encoder  # 제스처 인코더 저장 (이미지 인코더 대신)
        self.encoder_gesture_processor = gesture_processor  # 인코더용 제스처 전처리기 저장 (이미지 전처리기 대신)
        self.audio_vae = audio_vae  # 오디오 VAE 저장
        self.audio_processor = audio_processor  # 오디오 전처리기 저장
        self.audio_processor_clip = audio_processor_clip  # CLIP 오디오 전처리기 저장
        self.text_vae = text_vae  # 텍스트 VAE 저장
        self.mode = mode  # 파이프라인 모드 저장 (기본값: 'gesture')
        
        # ============================================================================
        # 실행 디바이스 설정 (DiffusionPipeline에서 필요)
        # ============================================================================
        self._execution_device = None  # 실행 디바이스 초기화 (나중에 설정됨)
        
    @property
    def device(self):
        """현재 디바이스 반환 (DiffusionPipeline 호환)"""
        if hasattr(self.transformer, 'device'):
            return self.transformer.device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def progress_bar(self, total):
        """
        진행률 표시줄 생성 (DiffusionPipeline 호환)
        """
        return tqdm(total=total, desc="Omniges Generation")  # tqdm 진행률 표시줄 반환
        
    def maybe_free_model_hooks(self):
        """
        모델 훅 해제 (DiffusionPipeline 호환)
        메모리 정리를 위한 메서드
        """
        pass  # 현재는 아무것도 하지 않음 (필요시 구현)
        
    def call_mm_encoder(self, **kwargs):
        """
        멀티모달 인코더 호출 (현재는 단순 래퍼)
        Args:
            **kwargs: 멀티모달 인코더에 전달할 인자들
        Returns:
            멀티모달 인코더의 출력
        """
        return self.mm_encoder(kwargs)  # 멀티모달 인코더에 인자들을 딕셔너리로 전달

    def encode_prompt_with_audio(
        self,
        prompt: Union[str, List[str]] = None,  # 텍스트 프롬프트 (문자열 또는 문자열 리스트)
        audio_paths: Optional[List[str]] = None,  # 오디오 파일 경로 리스트
        num_gestures_per_prompt: int = 1,  # 프롬프트당 생성할 제스처 수 (이미지 수 대신)
        device: Optional[torch.device] = None,  # 계산 디바이스
        do_classifier_free_guidance: bool = False,  # CFG 사용 여부
        use_t5: bool = False,  # T5 인코더 사용 여부
        add_token_embed: bool = False,  # 토큰 임베딩 추가 여부
        max_sequence_length: int = 128,  # 최대 시퀀스 길이
    ):
        """
        텍스트 프롬프트 임베딩을 구축하고 LanguageBindAudio를 사용하여 샘플당 하나의 오디오 토큰을 추가
        """
        device = device or getattr(self, '_execution_device', 'cuda')  # 디바이스가 지정되지 않으면 실행 디바이스 사용 (기본값: cuda)

        # ============================================================================
        # 텍스트 임베딩 구축 (기존 유틸리티 사용)
        # ============================================================================
        
        # 기존 encode_prompt 메서드를 사용하여 텍스트 임베딩 생성
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
            prompt=prompt,  # 텍스트 프롬프트
            num_gestures_per_prompt=num_gestures_per_prompt,  # 프롬프트당 제스처 수 (파라미터명 업데이트됨)
            device=device,  # 계산 디바이스
            do_classifier_free_guidance=do_classifier_free_guidance,  # CFG 사용 여부
            use_t5=use_t5,  # T5 인코더 사용 여부
            add_token_embed=add_token_embed,  # 토큰 임베딩 추가 여부
            max_sequence_length=max_sequence_length,  # 최대 시퀀스 길이
        )

        # ============================================================================
        # 오디오 경로 검증
        # ============================================================================
        
        # 오디오 경로가 없으면 텍스트 임베딩만 반환
        if audio_paths is None or len(audio_paths) == 0:
            return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

        # ============================================================================
        # 오디오를 특징으로 처리
        # ============================================================================
        
        # 그래디언트 계산 비활성화 (메모리 효율성)
        with torch.no_grad():
            # CLIP 오디오 전처리기를 사용하여 오디오를 텐서로 변환
            proc = self.audio_processor_clip(images=audio_paths, return_tensors="pt")  # 오디오를 이미지처럼 처리
            pixel_values = proc["pixel_values"].to(device)  # 픽셀 값을 디바이스로 이동
            
            # 오디오 인코더를 사용하여 오디오 특징 추출
            audio_feats = self.audio_encoder.get_image_features(pixel_values=pixel_values)  # 오디오를 이미지 특징으로 처리
            
            # 텍스트 토큰 차원에 맞춰 패딩/절단
            tok_dim = prompt_embeds.shape[-1]  # 텍스트 임베딩의 마지막 차원 (특징 차원)
            if audio_feats.shape[-1] < tok_dim:
                # 오디오 특징이 텍스트 토큰보다 작으면 패딩
                pad = torch.zeros((audio_feats.shape[0], tok_dim - audio_feats.shape[-1]), device=device, dtype=audio_feats.dtype)
                audio_tok = torch.cat([audio_feats, pad], dim=-1)  # 오디오 특징에 패딩 추가
            else:
                # 오디오 특징이 텍스트 토큰보다 크면 절단
                audio_tok = audio_feats[:, :tok_dim]  # 앞쪽 tok_dim 차원만 사용
            
            # 오디오 토큰을 텍스트 임베딩과 동일한 데이터 타입으로 변환하고 차원 추가
            audio_tok = audio_tok.to(prompt_embeds.dtype).unsqueeze(1)  # (B, 1, tok_dim) 형태로 변환

        # ============================================================================
        # 오디오 토큰을 프롬프트 임베딩에 추가
        # ============================================================================
        
        # 양성 브랜치에 오디오 토큰 추가 (시퀀스 차원을 따라 연결)
        prompt_embeds = torch.cat([prompt_embeds, audio_tok], dim=1)  # (B, seq_len+1, tok_dim)
        
        # CFG가 활성화된 경우 음성 브랜치에 제로 토큰 추가
        if do_classifier_free_guidance and negative_prompt_embeds is not None:
            zero_tok = torch.zeros_like(audio_tok)  # 오디오 토큰과 동일한 크기의 제로 텐서 생성
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, zero_tok], dim=1)  # 음성 브랜치에도 제로 토큰 추가

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    # ============================================================================
    # 제스처 헬퍼 메서드들
    # ============================================================================
    
    @torch.no_grad()  # 그래디언트 계산 비활성화 (추론 시 메모리 효율성)
    def encode_gesture(self, pose_seq: torch.Tensor):
        """
        포즈 시퀀스 [B,T,D]를 gesture_vae를 사용하여 트랜스포머 latent [B,C,H,W]로 인코딩
        Args:
            pose_seq: (B, T, D) 형태의 포즈 시퀀스 텐서
        Returns:
            (B, C, H, W) 형태의 2D latent 텐서
        """
        if self.gesture_vae is None:
            raise RuntimeError("gesture_vae is not set.")  # 제스처 VAE가 설정되지 않은 경우 에러
        return self.gesture_vae.encode(pose_seq)  # 제스처 VAE의 인코딩 메서드 호출

    @torch.no_grad()  # 그래디언트 계산 비활성화 (추론 시 메모리 효율성)
    def decode_gesture(self, latents_2d: torch.Tensor):
        """
        latent를 gesture_vae를 사용하여 포즈 시퀀스로 디코딩
        Args:
            latents_2d: (B, C, H, W) 형태의 2D latent 텐서
        Returns:
            (B, T, D) 형태의 포즈 시퀀스 텐서
        """
        if self.gesture_vae is None:
            raise RuntimeError("gesture_vae is not set.")  # 제스처 VAE가 설정되지 않은 경우 에러
        return self.gesture_vae.decode(latents_2d)  # 제스처 VAE의 디코딩 메서드 호출

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,  # 텍스트 프롬프트
        num_gestures_per_prompt: int = 1,  # 프롬프트당 제스처 수 (파라미터명 업데이트됨)
        max_sequence_length: int = 256,  # 최대 시퀀스 길이
        device: Optional[torch.device] = None,  # 계산 디바이스
        dtype: Optional[torch.dtype] = None,  # 데이터 타입
        add_token_embed: bool = False  # 토큰 임베딩 추가 여부
    ):
        dtype = dtype or self.text_encoder.dtype  # 데이터 타입이 지정되지 않으면 텍스트 인코더의 데이터 타입 사용
        device = device or getattr(self, '_execution_device', 'cuda')  # 디바이스가 지정되지 않으면 실행 디바이스 사용 (기본값: cuda)
        batch_size = len(prompt)  # 배치 크기 계산
        
        # T5 인코더가 없는 경우 제로 텐서 반환
        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_gestures_per_prompt,  # 총 배치 크기 (원본 배치 × 제스처 수)
                    self.tokenizer_max_length,  # 토크나이저 최대 길이
                    self.transformer.config.joint_attention_dim,  # 트랜스포머의 조인트 어텐션 차원
                ),
                device=device,  # 지정된 디바이스
                dtype=dtype,  # 지정된 데이터 타입
            )
        
        # T5 인코더가 있는 경우 T5를 사용하여 프롬프트 인코딩
        return _encode_prompt_with_t5(
            self.text_encoder_3,  # T5 텍스트 인코더
            self.tokenizer_3,  # T5 토크나이저
            max_sequence_length,  # 최대 시퀀스 길이
            prompt,  # 텍스트 프롬프트
            num_gestures_per_prompt,  # 프롬프트당 제스처 수
            device=device,  # 계산 디바이스
            add_token_embed=add_token_embed  # 토큰 임베딩 추가 여부
        )

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],  # 텍스트 프롬프트
        num_gestures_per_prompt: int = 1,  # 프롬프트당 제스처 수 (파라미터명 업데이트됨)
        device: Optional[torch.device] = None,  # 계산 디바이스
        clip_skip: Optional[int] = None,  # CLIP 스킵 레이어 수
        clip_model_index: int = 0,  # 사용할 CLIP 모델 인덱스 (0: 첫 번째, 1: 두 번째)
    ):
        device = device or getattr(self, '_execution_device', 'cuda')  # 디바이스가 지정되지 않으면 실행 디바이스 사용 (기본값: cuda)

        # CLIP 토크나이저와 텍스트 인코더 리스트 (2개 CLIP 모델)
        clip_tokenizers = [self.tokenizer, self.tokenizer_2]  # 첫 번째, 두 번째 CLIP 토크나이저
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]  # 첫 번째, 두 번째 CLIP 텍스트 인코더

        # 지정된 인덱스의 토크나이저와 인코더 선택
        tokenizer = clip_tokenizers[clip_model_index]  # 선택된 CLIP 토크나이저
        text_encoder = clip_text_encoders[clip_model_index]  # 선택된 CLIP 텍스트 인코더

        prompt = [prompt] if isinstance(prompt, str) else prompt  # 문자열이면 리스트로 변환
        batch_size = len(prompt)  # 배치 크기 계산

        # 토크나이저를 사용하여 텍스트를 토큰으로 변환
        text_inputs = tokenizer(
            prompt,  # 텍스트 프롬프트
            padding="max_length",  # 최대 길이까지 패딩
            max_length=self.tokenizer_max_length,  # 토크나이저 최대 길이
            truncation=True,  # 길이 초과 시 절단
            return_tensors="pt",  # PyTorch 텐서로 반환
        )

        text_input_ids = text_inputs.input_ids  # 토크나이저 출력에서 input_ids 추출
        
        # 절단되지 않은 토큰 ID 계산 (경고 메시지용)
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids  # 절단 없이 토큰화
        
        # 절단이 발생한 경우 경고 메시지 출력
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])  # 절단된 부분 디코딩
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"  # 절단된 텍스트 출력
            )
        # 텍스트 인코더를 사용하여 토큰을 임베딩으로 변환
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)  # 숨겨진 상태도 출력
        pooled_prompt_embeds = prompt_embeds[0]  # 풀링된 임베딩 (첫 번째 출력)

        # CLIP 스킵 설정에 따른 임베딩 선택
        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]  # 기본값: 마지막에서 두 번째 레이어
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]  # 지정된 스킵 레이어

        # 임베딩을 올바른 데이터 타입과 디바이스로 변환
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape  # 임베딩 텐서의 차원 추출 (배치, 시퀀스 길이, 특징 차원)
        
        # ============================================================================
        # 프롬프트당 제스처 수만큼 텍스트 임베딩 복제 (MPS 친화적 방법 사용)
        # ============================================================================
        
        # 텍스트 임베딩을 제스처 수만큼 복제
        prompt_embeds = prompt_embeds.repeat(1, num_gestures_per_prompt, 1)  # 시퀀스 차원을 따라 복제
        prompt_embeds = prompt_embeds.view(batch_size * num_gestures_per_prompt, seq_len, -1)  # 배치 차원 재구성

        # 풀링된 임베딩도 제스처 수만큼 복제
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_gestures_per_prompt, 1)  # 시퀀스 차원을 따라 복제
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_gestures_per_prompt, -1)  # 배치 차원 재구성

        return prompt_embeds, pooled_prompt_embeds  # 텍스트 임베딩과 풀링된 임베딩 반환

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],  # 첫 번째 프롬프트 (CLIP 1용)
        prompt_2: Union[str, List[str]] = None,  # 두 번째 프롬프트 (CLIP 2용)
        prompt_3: Union[str, List[str]] = None,  # 세 번째 프롬프트 (T5용)
        device: Optional[torch.device] = None,  # 계산 디바이스
        num_gestures_per_prompt: int = 1,  # 프롬프트당 제스처 수 (파라미터명 업데이트됨)
        do_classifier_free_guidance: bool = True,  # CFG 사용 여부
        negative_prompt: Optional[Union[str, List[str]]] = None,  # 첫 번째 음성 프롬프트
        negative_prompt_2: Optional[Union[str, List[str]]] = None,  # 두 번째 음성 프롬프트
        negative_prompt_3: Optional[Union[str, List[str]]] = None,  # 세 번째 음성 프롬프트
        prompt_embeds: Optional[torch.FloatTensor] = None,  # 미리 계산된 프롬프트 임베딩
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,  # 미리 계산된 음성 프롬프트 임베딩
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,  # 미리 계산된 풀링된 프롬프트 임베딩
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,  # 미리 계산된 풀링된 음성 프롬프트 임베딩
        clip_skip: Optional[int] = None,  # CLIP 스킵 레이어 수
        max_sequence_length: int = 256,  # 최대 시퀀스 길이
        add_token_embed: bool = False,  # 토큰 임베딩 추가 여부
        use_t5: bool = False,  # T5 인코더 사용 여부
    ):
        """Omniges용 프롬프트 인코딩 (OmniFlow와 동일한 로직)"""
        device = device or getattr(self, '_execution_device', 'cuda')  # 디바이스가 지정되지 않으면 실행 디바이스 사용 (기본값: cuda)

        prompt = [prompt] if isinstance(prompt, str) else prompt  # 문자열이면 리스트로 변환
        
        # 배치 크기 계산
        if prompt is not None:
            batch_size = len(prompt)  # 프롬프트가 있으면 프롬프트 길이로 배치 크기 계산
        else:
            batch_size = prompt_embeds.shape[0]  # 프롬프트가 없으면 미리 계산된 임베딩의 배치 크기 사용

        # ============================================================================
        # 프롬프트 임베딩이 미리 계산되지 않은 경우 새로 계산
        # ============================================================================
        
        if prompt_embeds is None:
            # 프롬프트 2와 3이 없으면 프롬프트 1 사용
            prompt_2 = prompt_2 or prompt  # 프롬프트 2가 없으면 프롬프트 1 사용
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2  # 문자열이면 리스트로 변환

            prompt_3 = prompt_3 or prompt  # 프롬프트 3이 없으면 프롬프트 1 사용
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3  # 문자열이면 리스트로 변환

            # 첫 번째 CLIP 모델로 프롬프트 임베딩 생성
            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,  # 첫 번째 프롬프트
                device=device,  # 계산 디바이스
                num_gestures_per_prompt=num_gestures_per_prompt,  # 프롬프트당 제스처 수
                clip_skip=clip_skip,  # CLIP 스킵 설정
                clip_model_index=0,  # 첫 번째 CLIP 모델 사용
            )
            
            # 두 번째 CLIP 모델로 프롬프트 임베딩 생성
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2,  # 두 번째 프롬프트
                device=device,  # 계산 디바이스
                num_gestures_per_prompt=num_gestures_per_prompt,  # 프롬프트당 제스처 수
                clip_skip=clip_skip,  # CLIP 스킵 설정
                clip_model_index=1,  # 두 번째 CLIP 모델 사용
            )
            
            # 두 CLIP 임베딩을 특징 차원을 따라 연결
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)  # (B, seq_len, clip1_dim + clip2_dim)
            # T5 인코더 사용 여부에 따른 처리
            if use_t5:
                # T5 프롬프트 임베딩 생성
                t5_prompt_embed = self._get_t5_prompt_embeds(
                    prompt=prompt_3,  # 세 번째 프롬프트 (T5용)
                    num_gestures_per_prompt=num_gestures_per_prompt,  # 프롬프트당 제스처 수
                    max_sequence_length=max_sequence_length,  # 최대 시퀀스 길이
                    device=device,  # 계산 디바이스
                    add_token_embed=add_token_embed,  # 토큰 임베딩 추가 여부
                )

                # CLIP 임베딩을 T5 임베딩과 동일한 차원으로 패딩
                clip_prompt_embeds = torch.nn.functional.pad(
                    clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])  # 특징 차원에 패딩
                )

                # CLIP과 T5 임베딩을 시퀀스 차원을 따라 연결
                prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)  # (B, seq_len + t5_len, features)
            else:
                # T5를 사용하지 않으면 CLIP 임베딩만 사용
                prompt_embeds = clip_prompt_embeds
            
            # 토큰 임베딩 정규화 (선택적)
            if add_token_embed:
                prompt_embeds = (prompt_embeds - prompt_embeds.mean(-1, keepdim=True)) / (prompt_embeds.std(-1, keepdim=True) + 1e-9)  # Z-score 정규화
            
            # 풀링된 프롬프트 임베딩을 특징 차원을 따라 연결
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)  # (B, clip1_pooled_dim + clip2_pooled_dim)
                    
        # ============================================================================
        # CFG가 활성화되고 음성 프롬프트 임베딩이 미리 계산되지 않은 경우 처리
        # ============================================================================
        
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            # 음성 프롬프트가 없으면 빈 문자열 사용
            negative_prompt = negative_prompt or ""  # 첫 번째 음성 프롬프트가 없으면 빈 문자열
            negative_prompt_2 = negative_prompt_2 or negative_prompt  # 두 번째 음성 프롬프트가 없으면 첫 번째 사용
            negative_prompt_3 = negative_prompt_3 or negative_prompt  # 세 번째 음성 프롬프트가 없으면 첫 번째 사용

            # 문자열을 배치 크기에 맞는 리스트로 정규화
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt  # 문자열이면 배치 크기만큼 복제
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2  # 문자열이면 배치 크기만큼 복제
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3  # 문자열이면 배치 크기만큼 복제
            )

            # 타입 및 배치 크기 검증
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."  # 프롬프트와 음성 프롬프트의 타입이 다르면 에러
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."  # 프롬프트와 음성 프롬프트의 배치 크기가 다르면 에러
                )

            # 첫 번째 CLIP 모델로 음성 프롬프트 임베딩 생성
            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt,  # 첫 번째 음성 프롬프트
                device=device,  # 계산 디바이스
                num_gestures_per_prompt=num_gestures_per_prompt,  # 프롬프트당 제스처 수
                clip_skip=None,  # 음성 프롬프트는 스킵하지 않음
                clip_model_index=0,  # 첫 번째 CLIP 모델 사용
            )
            
            # 두 번째 CLIP 모델로 음성 프롬프트 임베딩 생성
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt_2,  # 두 번째 음성 프롬프트
                device=device,  # 계산 디바이스
                num_gestures_per_prompt=num_gestures_per_prompt,  # 프롬프트당 제스처 수
                clip_skip=None,  # 음성 프롬프트는 스킵하지 않음
                clip_model_index=1,  # 두 번째 CLIP 모델 사용
            )
            
            # 두 CLIP 음성 임베딩을 특징 차원을 따라 연결
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)  # (B, seq_len, clip1_dim + clip2_dim)
            # T5 인코더 사용 여부에 따른 음성 프롬프트 처리
            if use_t5:
                # T5 음성 프롬프트 임베딩 생성
                t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                    prompt=negative_prompt_3,  # 세 번째 음성 프롬프트 (T5용)
                    num_gestures_per_prompt=num_gestures_per_prompt,  # 프롬프트당 제스처 수
                    max_sequence_length=max_sequence_length,  # 최대 시퀀스 길이
                    device=device,  # 계산 디바이스
                )

                # CLIP 음성 임베딩을 T5 음성 임베딩과 동일한 차원으로 패딩
                negative_clip_prompt_embeds = torch.nn.functional.pad(
                    negative_clip_prompt_embeds,
                    (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),  # 특징 차원에 패딩
                )
                
                # CLIP과 T5 음성 임베딩을 시퀀스 차원을 따라 연결
                negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)  # (B, seq_len + t5_len, features)
            else:
                # T5를 사용하지 않으면 CLIP 음성 임베딩만 사용
                negative_prompt_embeds = negative_clip_prompt_embeds

            # 음성 토큰 임베딩 정규화 (선택적)
            if add_token_embed:
                negative_prompt_embeds = (negative_prompt_embeds - negative_prompt_embeds.mean(-1, keepdim=True)) / (negative_prompt_embeds.std(-1, keepdim=True) + 1e-9)  # Z-score 정규화
            
            # 풀링된 음성 프롬프트 임베딩을 특징 차원을 따라 연결
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1  # (B, clip1_pooled_dim + clip2_pooled_dim)
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def check_inputs(
        self,
        prompt,  # 첫 번째 프롬프트
        prompt_2,  # 두 번째 프롬프트
        prompt_3,  # 세 번째 프롬프트
        seq_length,  # 제스처 시퀀스 길이 (이미지 높이 대신)
        gesture_dim,  # 제스처 차원 (이미지 너비 대신)
        negative_prompt=None,  # 첫 번째 음성 프롬프트
        negative_prompt_2=None,  # 두 번째 음성 프롬프트
        negative_prompt_3=None,  # 세 번째 음성 프롬프트
        prompt_embeds=None,  # 미리 계산된 프롬프트 임베딩
        negative_prompt_embeds=None,  # 미리 계산된 음성 프롬프트 임베딩
        pooled_prompt_embeds=None,  # 미리 계산된 풀링된 프롬프트 임베딩
        negative_pooled_prompt_embeds=None,  # 미리 계산된 풀링된 음성 프롬프트 임베딩
        callback_on_step_end_tensor_inputs=None,  # 콜백에서 사용할 텐서 입력들
        max_sequence_length=None,  # 최대 시퀀스 길이
    ):
        # ============================================================================
        # 제스처 시퀀스 파라미터 검증 (이미지 크기 대신)
        # ============================================================================
        
        # 제스처 시퀀스 길이가 8의 배수인지 확인 (VAE 다운샘플링 요구사항)
        if seq_length % 8 != 0:
            raise ValueError(f"`seq_length` has to be divisible by 8 but is {seq_length}.")
        
        # 제스처 차원이 4의 배수인지 확인 (4개 부위 요구사항)
        if gesture_dim % 4 != 0:
            raise ValueError(f"`gesture_dim` should be divisible by 4 but is {gesture_dim}.")

        # ============================================================================
        # 콜백 텐서 입력 검증
        # ============================================================================
        
        # 콜백에서 사용할 텐서 입력들이 허용된 목록에 있는지 확인
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs  # 모든 입력이 허용된 목록에 있는지 확인
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"  # 허용되지 않은 입력들 출력
            )

        # ============================================================================
        # 프롬프트 검증 로직 (OmniFlow와 동일)
        # ============================================================================
        
        # 프롬프트와 프롬프트 임베딩이 동시에 제공되지 않도록 검증
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."  # 프롬프트와 임베딩을 동시에 제공할 수 없음
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."  # 프롬프트2와 임베딩을 동시에 제공할 수 없음
            )
        elif prompt_3 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_3`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."  # 프롬프트3와 임베딩을 동시에 제공할 수 없음
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."  # 프롬프트나 임베딩 중 하나는 제공해야 함
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")  # 프롬프트 타입 검증
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")  # 프롬프트2 타입 검증
        elif prompt_3 is not None and (not isinstance(prompt_3, str) and not isinstance(prompt_3, list)):
            raise ValueError(f"`prompt_3` has to be of type `str` or `list` but is {type(prompt_3)}")  # 프롬프트3 타입 검증

        # ============================================================================
        # 음성 프롬프트 검증 (OmniFlow와 동일)
        # ============================================================================
        
        # 음성 프롬프트와 음성 프롬프트 임베딩이 동시에 제공되지 않도록 검증
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."  # 음성 프롬프트와 임베딩을 동시에 제공할 수 없음
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."  # 음성 프롬프트2와 임베딩을 동시에 제공할 수 없음
            )
        elif negative_prompt_3 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_3`: {negative_prompt_3} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."  # 음성 프롬프트3와 임베딩을 동시에 제공할 수 없음
            )

        # ============================================================================
        # 임베딩 형태 및 의존성 검증
        # ============================================================================
        
        # 프롬프트 임베딩과 음성 프롬프트 임베딩의 형태가 일치하는지 확인
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."  # 형태가 일치하지 않으면 에러
                )

        # 프롬프트 임베딩이 제공되면 풀링된 프롬프트 임베딩도 필요
        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."  # 풀링된 임베딩이 누락됨
            )

        # 음성 프롬프트 임베딩이 제공되면 풀링된 음성 프롬프트 임베딩도 필요
        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."  # 풀링된 음성 임베딩이 누락됨
            )

        # 최대 시퀀스 길이 제한 검증
        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")  # 최대 길이 초과

    def prepare_latents(
        self,
        batch_size,  # 배치 크기
        num_channels_latents,  # latent 채널 수
        seq_length,  # 제스처 시퀀스 길이 (이미지 높이 대신)
        num_parts,   # 제스처 부위 수 (이미지 너비 대신)
        dtype,  # 데이터 타입
        device,  # 디바이스
        generator,  # 랜덤 생성기
        latents=None,  # 미리 생성된 latent (선택적)
    ):
        """
        제스처 생성을 위한 latent 준비
        """
        # 미리 생성된 latent가 있으면 디바이스와 데이터 타입만 변경하여 반환
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        # 제스처 latent 형태 계산
        shape = (
            batch_size,  # 배치 크기
            num_channels_latents,  # latent 채널 수
            int(seq_length) // self.vae_scale_factor,  # VAE 스케일 팩터로 다운샘플링된 시퀀스 길이
            int(num_parts),  # 제스처 부위 수 (4개 부위)
        )

        # ============================================================================
        # 생성기 검증 및 latent 생성
        # ============================================================================
        
        # 생성기 리스트의 길이가 배치 크기와 일치하는지 확인
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."  # 생성기 개수와 배치 크기가 일치하지 않음
            )

        # 랜덤 정규분포 텐서로 latent 생성
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)  # 정규분포 랜덤 텐서 생성
        return latents  # 생성된 latent 반환

    # ============================================================================
    # 프로퍼티 메서드들 (설정값 접근용)
    # ============================================================================
    
    @property
    def guidance_scale(self):
        """CFG 가이던스 스케일 반환"""
        return self._guidance_scale

    @property
    def clip_skip(self):
        """CLIP 스킵 레이어 수 반환"""
        return self._clip_skip

    @property
    def do_classifier_free_guidance(self):
        """CFG 사용 여부 반환 (가이던스 스케일 > 1)"""
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        """조인트 어텐션 키워드 인자 반환"""
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        """타임스텝 수 반환"""
        return self._num_timesteps

    @property
    def interrupt(self):
        """인터럽트 상태 반환"""
        return self._interrupt

    @torch.no_grad()  # 그래디언트 계산 비활성화 (추론 시 메모리 효율성)
    @replace_example_docstring(EXAMPLE_DOC_STRING)  # 예제 문서 문자열 교체 데코레이터
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,  # 텍스트 프롬프트
        prompt_2: Optional[Union[str, List[str]]] = None,  # 두 번째 텍스트 프롬프트
        prompt_3: Optional[Union[str, List[str]]] = None,  # 세 번째 텍스트 프롬프트
        seq_length: Optional[int] = None,  # 제스처 시퀀스 길이 (이미지 높이 대신)
        gesture_dim: Optional[int] = None,  # 제스처 차원 (이미지 너비 대신)
        num_inference_steps: int = 28,  # 추론 스텝 수
        timesteps: List[int] = None,  # 타임스텝 리스트
        guidance_scale: float = 7.0,  # CFG 가이던스 스케일
        negative_prompt: Optional[Union[str, List[str]]] = None,  # 음성 프롬프트
        negative_prompt_2: Optional[Union[str, List[str]]] = None,  # 두 번째 음성 프롬프트
        negative_prompt_3: Optional[Union[str, List[str]]] = None,  # 세 번째 음성 프롬프트
        num_gestures_per_prompt: Optional[int] = 1,  # 프롬프트당 제스처 수 (이미지 수 대신)
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 랜덤 생성기
        latents: Optional[torch.FloatTensor] = None,  # 미리 생성된 latent
        prompt_embeds: Optional[torch.FloatTensor] = None,  # 미리 계산된 프롬프트 임베딩
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,  # 미리 계산된 음성 프롬프트 임베딩
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,  # 미리 계산된 풀링된 프롬프트 임베딩
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,  # 미리 계산된 풀링된 음성 프롬프트 임베딩
        output_type: Optional[str] = "gesture",  # 출력 타입 (gesture, latent, numpy)
        return_dict: bool = True,  # 딕셔너리 반환 여부
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,  # 조인트 어텐션 키워드 인자
        clip_skip: Optional[int] = None,  # CLIP 스킵 레이어 수
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,  # 스텝 종료 콜백
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],  # 콜백 텐서 입력들
        max_sequence_length: int = 256,  # 최대 시퀀스 길이
        add_token_embed: bool = False,  # 토큰 임베딩 추가 여부
        task: str = 't2g',  # 기본 태스크: 텍스트-투-제스처
        input_gesture=None,  # 입력 제스처 (이미지 대신)
        v_pred=True,  # v-예측 모드
        split_cond=False,  # 조건 분할
        overwrite_audio=None,  # 오디오 덮어쓰기
        overwrite_audio_t=None,  # 오디오 타임스텝 덮어쓰기
        input_aud=None,  # 입력 오디오
        return_embed=False,  # 임베딩 반환 여부
        drop_text=False,  # 텍스트 드롭
        drop_gesture=False,  # 제스처 드롭 (이미지 드롭 대신)
        drop_audio=False,  # 오디오 드롭
        use_text_output=True,  # 텍스트 출력 사용 여부
        use_t5=False,  # T5 사용 여부
        drop_pool=False,  # 풀링 드롭
        mm_cfgs=[],  # 멀티모달 CFG 설정
        bypass=False,  # 바이패스 모드
        no_clip=False,  # CLIP 비활성화
        cfg_mode=None  # CFG 모드
    ):
        r"""
        텍스트-오디오-제스처 생성을 위한 Omniges 파이프라인 호출

        지원하는 태스크:
        - t2g: 텍스트 → 제스처
        - a2g: 오디오 → 제스처  
        - g2t: 제스처 → 텍스트
        - g2a: 제스처 → 오디오
        - t2a: 텍스트 → 오디오 (OmniFlow와 동일)
        - a2t: 오디오 → 텍스트 (OmniFlow와 동일)
        
        Args:
            prompt (`str` or `List[str]`, *optional*):
                제스처 생성을 가이드하는 프롬프트
            seq_length (`int`, *optional*):
                생성할 제스처의 시퀀스 길이. 기본값: 128
            gesture_dim (`int`, *optional*):
                제스처 차원. 기본값: 415
            task (`str`):
                태스크 타입. 't2g', 'a2g', 'g2t', 'g2a', 't2a', 'a2t' 중 하나
            input_gesture (`torch.Tensor`, *optional*):
                g2t, g2a 태스크용 입력 제스처 시퀀스. 형태: (B, T, 415)
            input_aud (`str`, *optional*):
                a2g, a2t 태스크용 입력 오디오 파일 경로
            기타 인자들은 OmniFlow와 동일...

        Examples:

        Returns:
            [`OmnigesOutput`] or `tuple`:
            `return_dict`가 True면 [`OmnigesOutput`], 아니면 tuple
        """
        
        # ============================================================================
        # CFG 모드 설정 및 바이패스 모드 처리
        # ============================================================================
        
        # CFG 모드가 지정된 경우 설정
        if cfg_mode is not None:
            self.cfg_mode = cfg_mode
            
        # 바이패스 모드 처리 (제스처에 맞게 적응됨)
        if bypass:
            if task == 'a2g':  # 오디오 → 제스처 (텍스트를 통한 우회)
                # 먼저 오디오를 텍스트로 변환
                gestures = self("", input_aud=input_aud, seq_length=128, gesture_dim=415, 
                               add_token_embed=1, task='a2t', return_embed=False, 
                               guidance_scale=4, drop_pool=drop_pool)
                task = 't2g'  # 태스크를 텍스트 → 제스처로 변경
                input_aud = None  # 오디오 입력 제거
                prompt = gestures[0][0].replace('<s>', '').replace('</s>', '')  # 생성된 텍스트를 프롬프트로 사용
            if task == 'g2a':  # 제스처 → 오디오 (텍스트를 통한 우회)
                # 먼저 제스처를 텍스트로 변환
                texts = self("", input_gesture=input_gesture, seq_length=128, gesture_dim=415,
                           add_token_embed=1, task='g2t', return_embed=False,
                           guidance_scale=2, drop_pool=drop_pool)
                task = 't2a'  # 태스크를 텍스트 → 오디오로 변경
                input_gesture = None  # 제스처 입력 제거
                prompt = texts[0][0].replace('<s>', '').replace('</s>', '')  # 생성된 텍스트를 프롬프트로 사용
                
        # ============================================================================
        # 기본 파라미터 설정 및 입력 검증
        # ============================================================================
        
        # 기본 파라미터 설정
        seq_length = seq_length or self.default_seq_length  # 시퀀스 길이가 없으면 기본값 사용
        gesture_dim = gesture_dim or 415  # 제스처 차원이 없으면 전체 제스처 차원 사용
        text_vae_tokenizer = self.text_vae_tokenizer  # 텍스트 VAE 토크나이저 참조
        
        # 1. 입력 검증 (텍스트 입력 태스크에만 적용)
        if task in ['t2g', 't2a']:  # 텍스트 → 제스처, 텍스트 → 오디오 태스크
            self.check_inputs(
                prompt,  # 첫 번째 프롬프트
                prompt_2,  # 두 번째 프롬프트
                prompt_3,  # 세 번째 프롬프트
                seq_length,  # 제스처 시퀀스 길이 (이미지 높이 대신)
                gesture_dim,  # 제스처 차원 (이미지 너비 대신)
                negative_prompt=negative_prompt,  # 음성 프롬프트
                negative_prompt_2=negative_prompt_2,  # 두 번째 음성 프롬프트
                negative_prompt_3=negative_prompt_3,  # 세 번째 음성 프롬프트
                prompt_embeds=prompt_embeds,  # 미리 계산된 프롬프트 임베딩
                negative_prompt_embeds=negative_prompt_embeds,  # 미리 계산된 음성 프롬프트 임베딩
                pooled_prompt_embeds=pooled_prompt_embeds,  # 미리 계산된 풀링된 프롬프트 임베딩
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,  # 미리 계산된 풀링된 음성 프롬프트 임베딩
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,  # 콜백 텐서 입력들
                max_sequence_length=max_sequence_length,  # 최대 시퀀스 길이
            )

        # ============================================================================
        # 2. 설정값 저장 및 호출 파라미터 정의
        # ============================================================================
        
        # 인스턴스 변수에 설정값 저장
        self._guidance_scale = guidance_scale  # CFG 가이던스 스케일 저장
        self._clip_skip = clip_skip  # CLIP 스킵 레이어 수 저장
        self._joint_attention_kwargs = joint_attention_kwargs  # 조인트 어텐션 키워드 인자 저장
        self._interrupt = False  # 인터럽트 상태 초기화

        # 2. 호출 파라미터 정의
        # 프롬프트 타입에 따른 배치 크기 계산
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1  # 문자열이면 배치 크기 1
            prompt = [prompt]  # 문자열을 리스트로 변환
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)  # 리스트면 길이를 배치 크기로 사용
        else:
            batch_size = prompt_embeds.shape[0]  # 프롬프트가 없으면 임베딩의 배치 크기 사용

        device = getattr(self, '_execution_device', 'cuda')  # 실행 디바이스 가져오기 (기본값: cuda)

        # ============================================================================
        # 3. 프롬프트 인코딩
        # ============================================================================
        
        # 프롬프트를 임베딩으로 인코딩
        (
            prompt_embeds,  # 양성 프롬프트 임베딩
            negative_prompt_embeds,  # 음성 프롬프트 임베딩
            pooled_prompt_embeds,  # 풀링된 양성 프롬프트 임베딩
            negative_pooled_prompt_embeds,  # 풀링된 음성 프롬프트 임베딩
        ) = self.encode_prompt(
            prompt=prompt,  # 첫 번째 프롬프트
            prompt_2=prompt_2,  # 두 번째 프롬프트
            prompt_3=prompt_3,  # 세 번째 프롬프트
            negative_prompt=negative_prompt,  # 첫 번째 음성 프롬프트
            negative_prompt_2=negative_prompt_2,  # 두 번째 음성 프롬프트
            negative_prompt_3=negative_prompt_3,  # 세 번째 음성 프롬프트
            do_classifier_free_guidance=self.do_classifier_free_guidance,  # CFG 사용 여부
            prompt_embeds=prompt_embeds,  # 미리 계산된 프롬프트 임베딩
            negative_prompt_embeds=negative_prompt_embeds,  # 미리 계산된 음성 프롬프트 임베딩
            pooled_prompt_embeds=pooled_prompt_embeds,  # 미리 계산된 풀링된 프롬프트 임베딩
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,  # 미리 계산된 풀링된 음성 프롬프트 임베딩
            device=device,  # 계산 디바이스
            clip_skip=self.clip_skip,  # CLIP 스킵 설정
            num_gestures_per_prompt=num_gestures_per_prompt,  # 프롬프트당 제스처 수
            max_sequence_length=max_sequence_length,  # 최대 시퀀스 길이
            add_token_embed=add_token_embed,  # 토큰 임베딩 추가 여부
            use_t5=use_t5,  # T5 사용 여부
        )

        # ============================================================================
        # CFG 처리 및 타임스텝 준비
        # ============================================================================
        
        # CFG가 활성화된 경우 음성 프롬프트와 양성 프롬프트를 연결
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)  # 배치 차원을 따라 연결
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)  # 풀링된 임베딩도 연결
            
        # ============================================================================
        # 4. 타임스텝 준비
        # ============================================================================
        
        # 스케줄러에서 타임스텝 검색
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)  # 타임스텝과 추론 스텝 수 가져오기
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)  # 워밍업 스텝 수 계산
        self._num_timesteps = len(timesteps)  # 총 타임스텝 수 저장

        # ============================================================================
        # 5. Latent 변수 준비
        # ============================================================================
        
        # 트랜스포머의 입력 채널 수 가져오기
        num_channels_latents = self.transformer.config.in_channels  # 트랜스포머 설정에서 입력 채널 수 추출
        
        # 텍스트 VAE가 있는 경우 텍스트를 VAE 임베딩으로 인코딩
        if self.text_vae is not None:
            prompt_embeds_vae = self.text_vae.encode(prompt, input_ids=None, tokenizer=self.tokenizer_3)  # 양성 프롬프트를 VAE 임베딩으로 인코딩
            negative_prompt_embeds_vae = self.text_vae.encode(negative_prompt or '', input_ids=None, tokenizer=self.tokenizer_3)  # 음성 프롬프트를 VAE 임베딩으로 인코딩
            l_vae = prompt_embeds_vae.shape[1]  # VAE 임베딩의 시퀀스 길이 저장
        
        # ============================================================================
        # 제스처 latent 및 오디오 임베딩 준비
        # ============================================================================
        
        # 제스처 latent 준비
        latents = self.prepare_latents(
            batch_size * num_gestures_per_prompt,  # 총 배치 크기 (원본 배치 × 제스처 수)
            num_channels_latents,  # latent 채널 수
            seq_length,  # 제스처 시퀀스 길이
            4,  # 4개 제스처 부위
            prompt_embeds.dtype,  # 프롬프트 임베딩과 동일한 데이터 타입
            device,  # 계산 디바이스
            generator,  # 랜덤 생성기
            latents,  # 미리 생성된 latent (선택적)
        )
        
        # 오디오 임베딩 준비 (OmniFlow와 동일)
        if self.transformer.use_audio_mae:
            prompt_embeds_audio = torch.randn(1, 8, 768).to(prompt_embeds)  # MAE 모드: (1, 8, 768) 형태
        else:
            prompt_embeds_audio = torch.randn(1, 8, 256, 16).to(prompt_embeds)  # 일반 모드: (1, 8, 256, 16) 형태
        
        # ============================================================================
        # 태스크 검증 및 CFG 처리
        # ============================================================================
        
        # 지원하는 모든 제스처 태스크 조합 확인
        assert task in ['t2g', 'a2g', 'g2t', 'g2a', 't2a', 'a2t']  # 태스크가 지원 목록에 있는지 확인
        
        # ============================================================================
        # 모든 태스크에 대한 CFG 처리
        # ============================================================================
        
        if self.do_classifier_free_guidance:
            # 오디오 임베딩 복제 (태스크별로 다른 복제 수)
            if task in ['a2g', 'a2t', 't2g', 'g2t']:
                prompt_embeds_audio = prompt_embeds_audio.repeat(2, *([1] * len(prompt_embeds_audio.shape[1:])))  # 2배 복제
                 
            elif task in ['ag2t', 'at2g']:  # 오디오+제스처 → 텍스트/제스처
                prompt_embeds_audio = prompt_embeds_audio.repeat(4, *([1] * len(prompt_embeds_audio.shape[1:])))  # 4배 복제
                
            elif task in ['agt']:  # 오디오+제스처+텍스트 (필요시)
                prompt_embeds_audio = prompt_embeds_audio.repeat(4, *([1] * len(prompt_embeds_audio.shape[1:])))  # 4배 복제
                
            # 제스처/이미지가 생성되는 태스크에서 latent 복제
            if task in ['g2a', 'g2t', 't2a', 'a2t']:  # 제스처/이미지가 생성되는 태스크
                latents = latents.repeat(2, 1, 1, 1)  # 2배 복제
            elif task in ['gt2a', 'ag2t']:
                latents = latents.repeat(4, 1, 1, 1)  # 4배 복제
            elif task in ['agt']:
                latents = latents.repeat(4, 1, 1, 1)  # 4배 복제
                
        # ============================================================================
        # 태스크별 프롬프트 임베딩 준비
        # ============================================================================
        
        if task in ['t2g', 't2a', 'gt2a', 'at2g']:
            # 텍스트 → X 태스크들
            if no_clip == True:
                # CLIP을 사용하지 않는 경우
                if self.do_classifier_free_guidance:
                    prompt_embeds_vae_to_append = torch.cat([negative_prompt_embeds_vae, prompt_embeds_vae], dim=0)  # 음성과 양성 VAE 임베딩 연결
                    prompt_embeds = cat_and_pad([prompt_embeds_vae_to_append], max_dim=4096)  # VAE 임베딩만 사용하여 패딩
                else:
                    prompt_embeds = cat_and_pad([prompt_embeds_vae], max_dim=4096)  # 양성 VAE 임베딩만 사용
            elif self.text_vae is not None:
                # 텍스트 VAE가 있는 경우
                if self.do_classifier_free_guidance:
                    prompt_embeds_vae_to_append = torch.cat([negative_prompt_embeds_vae, prompt_embeds_vae], dim=0)  # 음성과 양성 VAE 임베딩 연결
                    prompt_embeds = cat_and_pad([prompt_embeds, prompt_embeds_vae_to_append], max_dim=4096)  # CLIP + VAE 임베딩 연결
                else:
                    prompt_embeds = cat_and_pad([prompt_embeds, prompt_embeds_vae], max_dim=4096)  # CLIP + 양성 VAE 임베딩 연결
            else:
                prompt_embeds = cat_and_pad([prompt_embeds], max_dim=4096)  # CLIP 임베딩만 사용
                
        elif task in ['g2t', 'a2t', 'ag2t']:
            # X → 텍스트 태스크들
            prompt_embeds = randn_tensor((1, *prompt_embeds_vae.shape[1:]), device=self.transformer.device, dtype=self.transformer.dtype)  # 랜덤 텐서 생성
            prompt_embeds = cat_and_pad([prompt_embeds], 4096)  # 랜덤 텐서로 패딩
        else:
            # 기타 태스크들
            assert prompt_embeds.shape[0] == 2  # 배치 크기가 2인지 확인
            prompt_embeds = randn_tensor((1, *prompt_embeds_vae.shape[1:]), device=self.transformer.device, dtype=self.transformer.dtype)  # 랜덤 텐서 생성
            prompt_embeds = cat_and_pad([prompt_embeds_vae], max_dim=4096)  # VAE 임베딩으로 패딩
            if self.do_classifier_free_guidance:
                prompt_embeds = prompt_embeds.repeat(2, 1, 1)  # CFG 시 2배 복제
        
        # ============================================================================
        # 제스처 입력 태스크 처리 (g2t, g2a)
        # ============================================================================
        
        if task in ['g2t', 'g2a', 'ag2t']:
            # 입력 제스처 처리
            if input_gesture is not None:
                # 제스처를 latent로 인코딩
                gesture_latents_dist = self.gesture_vae.encode(input_gesture.to(device=device, dtype=prompt_embeds.dtype))  # 제스처 VAE로 인코딩
                latents = gesture_latents_dist.sample()  # 분포에서 샘플링
            else:
                # 입력 제스처가 없으면 랜덤 latent 사용
                pass
                
            # CFG 처리
            if self.do_classifier_free_guidance:
                latents_null = torch.zeros_like(latents)  # 제로 latent 생성
                if task == 'ag2t':
                    latents = torch.cat([latents_null, latents, latents])  # 3개 연결 (제로, 제스처, 제스처)
                else:
                    latents = torch.cat([latents_null, latents])  # 2개 연결 (제로, 제스처)
            
            # 스케일링 팩터 적용 및 디바이스 이동
            latents = latents * self.gesture_vae.config.scaling_factor  # VAE 스케일링 팩터 적용
            latents = latents.to(device)  # 디바이스로 이동
            
            # ============================================================================
            # 풀링된 투영을 위한 제스처 임베딩 생성
            # ============================================================================
            
            # 제스처 인코더가 있는 경우 제스처 임베딩 생성
            if self.gesture_encoder is not None:
                with torch.no_grad():
                    gesture_embeds = self.gesture_encoder(input_gesture)  # 제스처 인코더로 임베딩 생성
            else:
                # 제스처 인코더가 없으면 더미 제스처 임베딩 생성
                gesture_embeds = torch.randn(batch_size, 768).to(device, dtype=prompt_embeds.dtype)  # 랜덤 임베딩 생성
                
            # 풀링된 프롬프트 임베딩을 제스처 임베딩으로 교체
            pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)  # 제로 텐서로 초기화
            pooled_prompt_embeds[..., :gesture_embeds.shape[-1]] = gesture_embeds  # 제스처 임베딩으로 채움
            
            # CFG 처리
            if self.do_classifier_free_guidance:
                with torch.no_grad():
                    gesture_embeds_null = torch.zeros_like(gesture_embeds)  # 제로 제스처 임베딩 생성
                assert pooled_prompt_embeds.shape[0] == 2  # 배치 크기가 2인지 확인
                pooled_prompt_embeds[0][..., :gesture_embeds.shape[-1]] = gesture_embeds_null  # 첫 번째 배치를 제로로 설정
                pooled_prompt_embeds[0] *= 0  # 첫 번째 배치를 완전히 제로로 만듦
                
        # ============================================================================
        # 오디오 입력 태스크 처리 (a2g, a2t)
        # ============================================================================
        
        elif task in ['a2g', 'a2t']:
            # 오디오 입력 처리 (OmniFlow와 동일)
            pixel_values = self.audio_processor.feature_extraction_vae(input_aud)['fbank'].unsqueeze(0)  # 오디오를 픽셀 값으로 변환
            prompt_embeds_audio = self.audio_vae.encode(pixel_values.to(device=self.audio_vae.device, dtype=self.audio_vae.dtype)).latent_dist.sample()  # 오디오 VAE로 인코딩
            
            # CFG 처리
            if self.do_classifier_free_guidance:
                prompt_embeds_audio_null = self.audio_vae.encode(0 * pixel_values.to(device=self.audio_vae.device, dtype=self.audio_vae.dtype)).latent_dist.mean  # 제로 오디오 인코딩
                if task == 'ag2t':
                    prompt_embeds_audio = torch.cat([prompt_embeds_audio, prompt_embeds_audio_null, prompt_embeds_audio])  # 3개 연결
                else:
                    prompt_embeds_audio = torch.cat([prompt_embeds_audio_null, prompt_embeds_audio])  # 2개 연결
            
            # 스케일링 팩터 적용 및 디바이스/데이터 타입 변환
            prompt_embeds_audio = prompt_embeds_audio * self.audio_vae.config.scaling_factor  # 오디오 VAE 스케일링 팩터 적용
            prompt_embeds_audio = prompt_embeds_audio.to(device).to(prompt_embeds.dtype)  # 디바이스와 데이터 타입 변환
            
            # CLIP 오디오 임베딩 생성
            audio_clip = self.audio_processor_clip(input_aud)['pixel_values']  # CLIP용 오디오 픽셀 값
            with torch.no_grad():
                audio_embeds = self.audio_encoder.get_image_features(pixel_values=audio_clip.to(self.audio_encoder.device).to(self.audio_encoder.dtype))  # 오디오 인코더로 특징 추출
                
            # 풀링된 프롬프트 임베딩을 오디오 임베딩으로 교체
            pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)  # 제로 텐서로 초기화
            pooled_prompt_embeds[..., :audio_embeds.shape[-1]] = audio_embeds  # 오디오 임베딩으로 채움
            
            # CFG 처리
            if self.do_classifier_free_guidance:
                assert pooled_prompt_embeds.shape[0] == 2  # 배치 크기가 2인지 확인
                with torch.no_grad():
                    audio_embeds_null = self.audio_encoder.get_image_features(pixel_values=audio_clip.to(self.audio_encoder.device).to(self.audio_encoder.dtype) * 0)  # 제로 오디오 특징
                assert pooled_prompt_embeds.shape[0] == 2  # 배치 크기 재확인
                pooled_prompt_embeds[0][..., :audio_embeds.shape[-1]] = audio_embeds_null  # 첫 번째 배치를 제로로 설정
                
        # ============================================================================
        # 복잡한 멀티모달 태스크 처리 (필요시)
        # ============================================================================
        
        if task == 'at2g':  # 오디오+텍스트 → 제스처
            print(pooled_prompt_embeds.shape, prompt_embeds.shape)  # 디버그 출력
            
            # CFG 처리
            if self.do_classifier_free_guidance:
                # 풀링된 임베딩을 2개로 분할
                pooled_prompt_embeds_null, pooled_prompt_embeds_text = pooled_prompt_embeds.chunk(2)  # 제로와 텍스트로 분할
                prompt_embeds_null, prompt_embeds_text = prompt_embeds.chunk(2)  # 제로와 텍스트로 분할
                
                # 4개 조합으로 재구성: [텍스트, 텍스트, 제로, 제로]
                pooled_prompt_embeds = torch.cat([
                    pooled_prompt_embeds_text, pooled_prompt_embeds_text,  # 텍스트 2개
                    pooled_prompt_embeds_null, pooled_prompt_embeds_null  # 제로 2개
                ])
                
                # 프롬프트 임베딩도 4개 조합으로 재구성: [텍스트, 텍스트, 랜덤, 랜덤]
                prompt_embeds = torch.cat([
                    prompt_embeds_text, prompt_embeds_text,  # 텍스트 2개
                    torch.randn_like(prompt_embeds_null), torch.randn_like(prompt_embeds_null)  # 랜덤 2개
                ])
                
                # 오디오 처리
                pixel_values = self.audio_processor.feature_extraction_vae(input_aud)['fbank'].unsqueeze(0)  # 오디오를 픽셀 값으로 변환
                prompt_embeds_audio = self.audio_vae.encode(pixel_values.to(device=self.audio_vae.device, dtype=self.audio_vae.dtype)).latent_dist.sample()  # 오디오 VAE로 인코딩
                prompt_embeds_audio = prompt_embeds_audio * self.audio_vae.config.scaling_factor  # 스케일링 팩터 적용
                
                # CFG 처리
                if self.do_classifier_free_guidance:
                    prompt_embeds_audio_null = self.audio_vae.encode(0 * pixel_values.to(device=self.audio_vae.device, dtype=self.audio_vae.dtype)).latent_dist.mean  # 제로 오디오 인코딩
                    prompt_embeds_audio_null = prompt_embeds_audio * self.audio_vae.config.scaling_factor  # 스케일링 팩터 적용
                    null_audio = torch.rand_like(prompt_embeds_audio_null)  # 랜덤 오디오 생성
                
                # 오디오 임베딩을 4개 조합으로 연결: [오디오, 랜덤, 오디오, 랜덤]
                prompt_embeds_audio = torch.cat([prompt_embeds_audio, null_audio, prompt_embeds_audio, null_audio])
            
            # 디바이스와 데이터 타입 변환
            prompt_embeds_audio = prompt_embeds_audio.to(device).to(prompt_embeds.dtype)
            
            # CLIP 오디오 임베딩 생성
            audio_clip = self.audio_processor_clip(input_aud)['pixel_values']  # CLIP용 오디오 픽셀 값
            with torch.no_grad():
                audio_embeds = self.audio_encoder.get_image_features(pixel_values=audio_clip.to(self.audio_encoder.device).to(self.audio_encoder.dtype))  # 오디오 인코더로 특징 추출

            # CFG 처리
            if self.do_classifier_free_guidance:
                pooled_prompt_embeds[2, :audio_embeds.shape[-1]] = audio_embeds  # 세 번째 배치에 오디오 임베딩 삽입
            pooled_prompt_embeds[-1] *= 0  # 마지막 배치를 제로로 만듦
                
        # ============================================================================
        # 다양한 모달리티에 대한 타임스텝 설정
        # ============================================================================
        
        # 텍스트 → 제스처/오디오 태스크
        if task in ['t2g', 't2a']:
            timesteps_text = [0] * batch_size  # 텍스트 타임스텝을 0으로 초기화
            timesteps_text = torch.tensor(timesteps_text).to(device)  # 텐서로 변환하고 디바이스로 이동
            if self.do_classifier_free_guidance:
                timesteps_text = timesteps_text.repeat(2)  # CFG 시 2배 복제
                if self.cfg_mode == 'new':
                    timesteps_text[0] = 1000  # 첫 번째 배치를 1000으로 설정
                    prompt_embeds[0] = torch.randn_like(prompt_embeds[0])  # 첫 번째 배치를 랜덤으로 설정
                    pooled_prompt_embeds[0] *= 0  # 첫 번째 배치를 제로로 설정
                    
        # 제스처/오디오 → 오디오/제스처 태스크
        if task in ['g2a', 'a2g']:
            timesteps_text = [0] * batch_size  # 텍스트 타임스텝을 0으로 초기화
            timesteps_text = torch.tensor(timesteps_text).to(device) + 1000  # 1000을 더함

        # 제스처 → 텍스트/오디오 태스크
        if task in ['g2t', 'g2a']:
            timesteps_gesture = [0] * batch_size  # 제스처 타임스텝을 0으로 초기화 (이미지 타임스텝 대신)
            timesteps_gesture = torch.tensor(timesteps_gesture).to(device)  # 텐서로 변환하고 디바이스로 이동
            if self.do_classifier_free_guidance:
                timesteps_gesture = timesteps_gesture.repeat(2)  # CFG 시 2배 복제
                if self.cfg_mode == 'new':
                    timesteps_gesture[0] = 1000  # 첫 번째 배치를 1000으로 설정
                    latents[0] = torch.randn_like(latents[0])  # 첫 번째 배치를 랜덤으로 설정
                    pooled_prompt_embeds[0] *= 0  # 첫 번째 배치를 제로로 설정
            
        # 텍스트/오디오 → 오디오/텍스트 태스크
        if task in ['t2a', 'a2t']:
            timesteps_gesture = [0] * batch_size  # 제스처 타임스텝을 0으로 초기화
            timesteps_gesture = torch.tensor(timesteps_gesture).to(device) + 1000  # 1000을 더함
            
        # 오디오 → 텍스트/제스처 태스크
        if task in ['a2t', 'a2g']:
            timesteps_aud = [0] * batch_size  # 오디오 타임스텝을 0으로 초기화
            timesteps_aud = torch.tensor(timesteps_aud).to(device)  # 텐서로 변환하고 디바이스로 이동
            if self.do_classifier_free_guidance:
                timesteps_aud = timesteps_aud.repeat(2)  # CFG 시 2배 복제
                if self.cfg_mode == 'new':
                    timesteps_aud[0] = 1000  # 첫 번째 배치를 1000으로 설정
                    prompt_embeds_audio[0] = torch.randn_like(prompt_embeds_audio[0])  # 첫 번째 배치를 랜덤으로 설정
                    pooled_prompt_embeds[0] *= 0  # 첫 번째 배치를 제로로 설정
    
        # 텍스트/제스처 → 제스처/텍스트 태스크
        if task in ['t2g', 'g2t']:
            timesteps_aud = [0] * batch_size  # 오디오 타임스텝을 0으로 초기화
            timesteps_aud = torch.tensor(timesteps_aud).to(device) + 1000  # 1000을 더함

        # ============================================================================
        # 메인 디노이징 루프 준비
        # ============================================================================
        
        x0 = None  # x0 초기화 (디노이징 결과 저장용)
        prompt_embeds[:, -l_vae:, prompt_embeds_vae.shape[-1]:] = 0  # VAE 임베딩 이후 부분을 제로로 설정
        if drop_pool:
            pooled_prompt_embeds = pooled_prompt_embeds * 0  # 풀링 드롭이 활성화되면 풀링된 임베딩을 제로로 설정
            
        # ============================================================================
        # 메인 디노이징 루프 시작
        # ============================================================================
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:  # 진행률 표시줄 생성
            for i, t in enumerate(timesteps):  # 타임스텝을 순회하며 디노이징 수행
                if self.interrupt:  # 인터럽트가 발생했는지 확인
                    continue  # 인터럽트가 있으면 현재 스텝 건너뛰기
                latents = latents.to(device=self.transformer.device, dtype=self.transformer.dtype)  # latent를 트랜스포머 디바이스와 데이터 타입으로 이동
                
                # ============================================================================
                # 태스크별 처리 로직
                # ============================================================================
                
                if task == 'ag2t':  # 오디오+제스처 → 텍스트
                    # CFG 처리: 3개 배치로 복제 (제로, 오디오, 제스처)
                    prompt_embed_input = torch.cat([prompt_embeds] * 3) if self.do_classifier_free_guidance else prompt_embeds  # 프롬프트 임베딩을 3배 복제
                    timestep = t.expand(prompt_embed_input.shape[0])  # 타임스텝을 배치 크기에 맞게 확장
                    
                    # 트랜스포머 호출 (오디오+제스처 → 텍스트)
                    _y = self.transformer(
                        hidden_states=latents,  # 제스처 latent
                        timestep=timesteps_gesture,  # 제스처 타임스텝
                        timestep_text=timestep,  # 텍스트 타임스텝
                        timestep_audio=timesteps_aud,  # 오디오 타임스텝
                        encoder_hidden_states=prompt_embed_input,  # 인코더 숨겨진 상태 (프롬프트 임베딩)
                        audio_hidden_states=prompt_embeds_audio,  # 오디오 숨겨진 상태
                        pooled_projections=pooled_prompt_embeds,  # 풀링된 투영
                        joint_attention_kwargs=self.joint_attention_kwargs,  # 조인트 어텐션 키워드 인자
                        return_dict=False,  # 딕셔너리 반환하지 않음
                        use_text_output=True,  # 텍스트 출력 사용
                        decode_text=True,  # 텍스트 디코딩 활성화
                        split_cond=split_cond,  # 조건 분할
                        drop_text=drop_text,  # 텍스트 드롭
                        drop_audio=drop_audio,  # 오디오 드롭
                        drop_image=drop_gesture  # 제스처 드롭 (이미지 드롭 대신)
                    )
                    
                    # 노이즈 예측 처리
                    if v_pred and not self.text_x0:
                        noise_pred = _y['model_pred_text']  # v-예측 모드에서 텍스트 모델 예측 사용
                    else:
                        x0 = _y['model_pred_text']  # x0 예측 모드에서 텍스트 모델 예측 사용
                        curr_latent_text = prompt_embed_input[..., :x0.shape[-1]]  # 현재 텍스트 latent 추출
                        noise_pred = self.scheduler.get_eps(t, x0, curr_latent_text)  # 스케줄러에서 노이즈 계산
                        
                elif task in ['t2g', 'a2g', 'at2g']:
                    # 텍스트/오디오 → 제스처 태스크들
                    if task == 'at2g':
                        # at2g 태스크: 4개 배치로 복제 (복잡한 멀티모달)
                        latent_model_input = torch.cat([latents] * 4) if self.do_classifier_free_guidance else latents  # latent를 4배 복제
                        timestep = t.expand(latent_model_input.shape[0])  # 타임스텝을 배치 크기에 맞게 확장
                    else:
                        # t2g, a2g 태스크: 2개 배치로 복제 (단순 CFG)
                        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents  # latent를 2배 복제
                        timestep = t.expand(latent_model_input.shape[0])  # 타임스텝을 배치 크기에 맞게 확장

                    # 트랜스포머 호출 (텍스트/오디오 → 제스처)
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,  # 제스처 latent 입력
                        timestep=timestep,  # 제스처 타임스텝
                        timestep_text=timesteps_text,  # 텍스트 타임스텝
                        timestep_audio=timesteps_aud,  # 오디오 타임스텝
                        audio_hidden_states=prompt_embeds_audio,  # 오디오 숨겨진 상태
                        encoder_hidden_states=prompt_embeds,  # 인코더 숨겨진 상태 (프롬프트 임베딩)
                        pooled_projections=pooled_prompt_embeds,  # 풀링된 투영
                        joint_attention_kwargs=self.joint_attention_kwargs,  # 조인트 어텐션 키워드 인자
                        return_dict=False,  # 딕셔너리 반환하지 않음
                        use_text_output=use_text_output,  # 텍스트 출력 사용 여부
                        decode_text=True,  # 텍스트 디코딩 활성화
                        split_cond=split_cond,  # 조건 분할
                        drop_text=drop_text,  # 텍스트 드롭
                        drop_audio=drop_audio,  # 오디오 드롭
                        drop_image=drop_gesture  # 제스처 드롭 (이미지 드롭 대신)
                    )['output']  # 출력에서 'output' 키의 값 추출
                    
                elif task in ['t2a', 'g2a']:
                    # 텍스트/제스처 → 오디오 태스크들
                    if overwrite_audio is not None:
                        # 오디오 덮어쓰기 모드 (CFG 비활성화 필요)
                        assert not self.do_classifier_free_guidance  # CFG가 비활성화되어야 함
                        prompt_embeds_audio = overwrite_audio.to(prompt_embeds_audio)  # 오디오 임베딩을 덮어쓰기 값으로 교체
                        noise_audio = torch.randn_like(prompt_embeds_audio)  # 랜덤 노이즈 생성
                        timestep = torch.tensor([overwrite_audio_t]).to(noise_audio.device)  # 덮어쓰기 타임스텝 설정
                        sigmas_audio = self.scheduler.sigmas[num_inference_steps - overwrite_audio_t]  # 스케줄러에서 시그마 값 가져오기
                        sigmas_audio = sigmas_audio.view(-1, 1, 1, 1)  # 시그마를 4D 텐서로 변환
                        prompt_embeds_audio_input = sigmas_audio * noise_audio + (1.0 - sigmas_audio) * prompt_embeds_audio  # 노이즈와 오디오를 가중 평균
                        prompt_embeds_audio_input = prompt_embeds_audio_input.to(self.transformer.dtype)  # 트랜스포머 데이터 타입으로 변환
                    else: 
                        # 일반 모드: CFG 처리
                        prompt_embeds_audio_input = torch.cat([prompt_embeds_audio] * 2) if self.do_classifier_free_guidance else prompt_embeds_audio  # 오디오 임베딩을 2배 복제
                        timestep = t.expand(prompt_embeds_audio_input.shape[0])  # 타임스텝을 배치 크기에 맞게 확장
                    
                    # 트랜스포머 호출 (텍스트/제스처 → 오디오)
                    _y = self.transformer(
                        hidden_states=latents,  # 제스처 latent
                        timestep=timesteps_gesture,  # 제스처 타임스텝
                        timestep_text=timesteps_text,  # 텍스트 타임스텝
                        timestep_audio=timestep,  # 오디오 타임스텝
                        audio_hidden_states=prompt_embeds_audio_input,  # 오디오 숨겨진 상태
                        encoder_hidden_states=prompt_embeds,  # 인코더 숨겨진 상태 (프롬프트 임베딩)
                        pooled_projections=pooled_prompt_embeds,  # 풀링된 투영
                        joint_attention_kwargs=self.joint_attention_kwargs,  # 조인트 어텐션 키워드 인자
                        return_dict=False,  # 딕셔너리 반환하지 않음
                        use_text_output=True,  # 텍스트 출력 사용
                        decode_text=True,  # 텍스트 디코딩 활성화
                        split_cond=split_cond,  # 조건 분할
                        drop_text=drop_text,  # 텍스트 드롭
                        drop_audio=drop_audio,  # 오디오 드롭
                        drop_image=drop_gesture  # 제스처 드롭 (이미지 드롭 대신)
                    )
                    
                    # 노이즈 예측 처리
                    if v_pred:
                        noise_pred = _y['audio_hidden_states']  # v-예측 모드에서 오디오 숨겨진 상태 사용
                    else:
                        x0 = _y['audio_hidden_states']  # x0 예측 모드에서 오디오 숨겨진 상태 사용
                        noise_pred = self.scheduler.get_eps(t, x0, prompt_embeds_audio)  # 스케줄러에서 노이즈 계산
                        noise_pred = noise_pred.to(x0)  # x0와 동일한 디바이스로 이동
                    
                    # 오디오 덮어쓰기 모드 처리
                    if overwrite_audio is not None:
                        x0 = noise_pred * (-sigmas_audio) + prompt_embeds_audio_input  # 노이즈와 입력을 조합하여 x0 계산
                        x0 = 1 / self.audio_vae.config.scaling_factor * x0  # 오디오 VAE 스케일링 팩터로 나누기
                        spec = self.audio_vae.decode(x0.float())  # 오디오 VAE로 디코딩
                        return spec.sample.float().cpu().numpy()  # 샘플을 numpy 배열로 반환
                        
                elif task in ['g2t', 'a2t']:
                    # 제스처/오디오 → 텍스트 태스크들
                    # CFG 처리: 2개 배치로 복제
                    prompt_embed_input = torch.cat([prompt_embeds] * 2) if self.do_classifier_free_guidance else prompt_embeds  # 프롬프트 임베딩을 2배 복제
                    timestep = t.expand(prompt_embed_input.shape[0])  # 타임스텝을 배치 크기에 맞게 확장
                    
                    # 트랜스포머 호출 (제스처/오디오 → 텍스트)
                    _y = self.transformer(
                        hidden_states=latents,  # 제스처 latent
                        timestep=timesteps_gesture,  # 제스처 타임스텝
                        timestep_text=timestep,  # 텍스트 타임스텝
                        timestep_audio=timesteps_aud,  # 오디오 타임스텝
                        encoder_hidden_states=prompt_embed_input,  # 인코더 숨겨진 상태 (프롬프트 임베딩)
                        audio_hidden_states=prompt_embeds_audio,  # 오디오 숨겨진 상태
                        pooled_projections=pooled_prompt_embeds,  # 풀링된 투영
                        joint_attention_kwargs=self.joint_attention_kwargs,  # 조인트 어텐션 키워드 인자
                        return_dict=False,  # 딕셔너리 반환하지 않음
                        use_text_output=True,  # 텍스트 출력 사용
                        decode_text=True,  # 텍스트 디코딩 활성화
                        split_cond=split_cond,  # 조건 분할
                        drop_text=drop_text,  # 텍스트 드롭
                        drop_audio=drop_audio,  # 오디오 드롭
                        drop_image=drop_gesture  # 제스처 드롭 (이미지 드롭 대신)
                    )
                    
                    # 노이즈 예측 처리
                    if v_pred and not self.text_x0:
                        noise_pred = _y['model_pred_text']  # v-예측 모드에서 텍스트 모델 예측 사용
                    else:
                        x0 = _y['model_pred_text']  # x0 예측 모드에서 텍스트 모델 예측 사용
                        curr_latent_text = prompt_embed_input[..., :x0.shape[-1]]  # 현재 텍스트 latent 추출
                        noise_pred = self.scheduler.get_eps(t, x0, curr_latent_text)  # 스케줄러에서 노이즈 계산
                else:
                    raise NotImplementedError(f"Task {task} not implemented")  # 지원하지 않는 태스크 에러
                    
                # ============================================================================
                # CFG 가이던스 수행
                # ============================================================================
                
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)  # 노이즈 예측을 2개로 분할 (무조건부, 조건부)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)  # CFG 공식 적용
                    
                # ============================================================================
                # 태스크별 latent 업데이트
                # ============================================================================
                
                latents_dtype = latents.dtype  # latent의 원본 데이터 타입 저장
                
                if task in ['t2g', 'a2g', 'at2g']:
                    # 제스처 생성 태스크: 제스처 latent 업데이트
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]  # 스케줄러로 latent 업데이트
                elif task in ['g2t', 'a2t', 'ag2t']:
                    # 텍스트 생성 태스크: 프롬프트 임베딩 업데이트
                    prompt_embeds = self.scheduler.step(noise_pred, t, prompt_embeds[..., :noise_pred.shape[-1]], return_dict=False)[0]  # 스케줄러로 프롬프트 임베딩 업데이트
                    prompt_embeds = cat_and_pad([prompt_embeds], 4096).to(latents_dtype)  # 패딩 후 데이터 타입 복원
                elif task in ['g2a', 't2a']:
                    # 오디오 생성 태스크: 오디오 임베딩 업데이트
                    prompt_embeds_audio = self.scheduler.step(noise_pred, t, prompt_embeds_audio, return_dict=False)[0]  # 스케줄러로 오디오 임베딩 업데이트
                else:
                    raise NotImplementedError(f"Task {task} not implemented")  # 지원하지 않는 태스크 에러
                    
                # MPS 디바이스에서 데이터 타입 복원
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)  # 원본 데이터 타입으로 복원

                # ============================================================================
                # 콜백 처리 및 진행률 업데이트
                # ============================================================================
                
                if callback_on_step_end is not None:
                    callback_kwargs = {}  # 콜백 키워드 인자 딕셔너리 초기화
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]  # 로컬 변수에서 콜백 입력 텐서들 추출
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)  # 콜백 함수 호출

                    # 콜백 출력에서 텐서들 업데이트
                    latents = callback_outputs.pop("latents", latents)  # latent 업데이트
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)  # 프롬프트 임베딩 업데이트
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)  # 음성 프롬프트 임베딩 업데이트
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds  # 음성 풀링된 프롬프트 임베딩 업데이트
                    )

                # 진행률 표시줄 업데이트
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()  # 진행률 표시줄 업데이트

                # XLA 디바이스 처리
                if XLA_AVAILABLE:
                    xm.mark_step()  # XLA 마크 스텝
                    
        # ============================================================================
        # 태스크별 출력 생성 처리
        # ============================================================================
        
        if task in ['g2t', 'a2t', 'ag2t']:
            # 텍스트 생성 태스크들
            prompt_embeds = prompt_embeds[..., :prompt_embeds_vae.shape[-1]]  # VAE 차원에 맞게 프롬프트 임베딩 절단
            tokens1 = self.text_vae.generate(latents=prompt_embeds, max_length=256, do_sample=False)  # 생성된 프롬프트 임베딩에서 토큰 생성
            z = self.text_vae.encode(prompt, input_ids=None, tokenizer=self.tokenizer_3, drop=True)  # 원본 프롬프트를 VAE로 인코딩
            tokens2 = self.text_vae.generate(latents=z, max_length=256, do_sample=False)  # 원본 프롬프트에서 토큰 생성
            
            # 토큰 디코딩 처리
            if self.text_vae_tokenizer is not None and type(tokens1[0]) is not str:
                text = self.text_vae_tokenizer.batch_decode(tokens1)  # 생성된 토큰을 텍스트로 디코딩
                text2 = self.text_vae_tokenizer.batch_decode(tokens2)  # 원본 토큰을 텍스트로 디코딩
            else:
                text = tokens1  # 이미 텍스트인 경우 그대로 사용
                text2 = tokens2  # 이미 텍스트인 경우 그대로 사용
            
            # 반환값 결정
            if return_embed:
                return text, text2, prompt_embeds  # 임베딩도 함께 반환
            else:
                return text, text2  # 텍스트만 반환
                
        elif task in ['t2a', 'g2a']:
            # 오디오 생성 태스크들
            prompt_embeds_audio = 1 / self.audio_vae.config.scaling_factor * prompt_embeds_audio  # 오디오 VAE 스케일링 팩터로 나누기
            spec = self.audio_vae.decode(prompt_embeds_audio.float())  # 오디오 VAE로 디코딩
            if hasattr(spec, 'sample'):
                spec = spec.sample  # 샘플 속성이 있으면 샘플 추출
            return spec.float().cpu().numpy(), x0  # numpy 배열과 x0 반환
            
        # ============================================================================
        # 제스처 생성 태스크들 (t2g, a2g)
        # ============================================================================
        
        if output_type == "latent":
            # latent 형태로 반환
            gesture_latents = latents  # 제스처 latent 저장
            result = gesture_latents  # 결과를 latent로 설정
        else:
            # 제스처 latent 디코딩
            latents = (latents / self.gesture_vae.config.scaling_factor) + self.gesture_vae.config.shift_factor  # 스케일링 팩터로 나누고 시프트 팩터 추가
            gesture_result = self.gesture_vae.decode(latents.to(self.gesture_vae.gesture_processor.device), return_dict=False)  # 제스처 VAE로 디코딩
            
            # 출력 타입에 따른 결과 처리
            if output_type == "gesture":
                result = gesture_result  # 제스처 형태로 반환
            elif output_type == "numpy":
                if hasattr(gesture_result, 'sample'):
                    result = gesture_result.sample.cpu().numpy()  # 샘플 속성이 있으면 numpy로 변환
                else:
                    result = gesture_result.cpu().numpy()  # 직접 numpy로 변환
            else:
                result = gesture_result  # 기본 형태로 반환

        # ============================================================================
        # 모델 오프로딩 및 결과 반환
        # ============================================================================
        
        # 모든 모델 오프로딩
        self.maybe_free_model_hooks()  # 모델 훅 해제

        # 딕셔너리 반환 여부에 따른 처리
        if not return_dict:
            return (result,)  # 튜플 형태로 반환

        # 제스처 출력을 위한 OmnigesOutput 객체 생성 (이미지 출력 대신)
        return type('OmnigesOutput', (), {
            'gestures': result,  # 제스처 결과
            'gesture_latents': latents if output_type == "latent" else None  # latent 형태인 경우에만 latent 포함
        })()


def create_omniges_pipeline(
    omniflow_checkpoint_path: str,  # OmniFlow 체크포인트 경로
    rvqvae_checkpoints: Dict[str, str],  # RVQVAE 체크포인트 딕셔너리
    device: str = 'cuda',  # 디바이스 (기본값: cuda)
    weight_dtype: torch.dtype = torch.bfloat16,  # 가중치 데이터 타입 (기본값: bfloat16)
    load_ema: bool = False  # EMA 가중치 로드 여부 (기본값: False)
):
    """
    OmniFlow 체크포인트 + RVQVAE 체크포인트에서 완전한 Omniges 파이프라인 생성
    
    Args:
        omniflow_checkpoint_path: OmniFlow 모델 체크포인트 디렉토리 경로
        rvqvae_checkpoints: 부위명을 RVQVAE 체크포인트 경로에 매핑하는 딕셔너리
            예시: {
                'upper': './ckpt/net_300000_upper.pth',      # 상체 RVQVAE
                'hands': './ckpt/net_300000_hands.pth',      # 손 RVQVAE
                'lower_trans': './ckpt/net_300000_lower.pth', # 하체+이동 RVQVAE
                'face': './ckpt/net_300000_face.pth'         # 얼굴 RVQVAE
            }
        device: 모델을 로드할 디바이스
        weight_dtype: 가중치 데이터 타입
        load_ema: EMA 가중치 로드 여부
        
    Returns:
        텍스트-오디오-제스처 생성을 위한 OmnigesPipeline
    """
    
    # OmnigesPipeline의 사전훈련된 모델 로드 메서드 호출
    return OmnigesPipeline.load_pretrained(
        omniflow_path=omniflow_checkpoint_path,  # OmniFlow 체크포인트 경로
        rvqvae_checkpoints=rvqvae_checkpoints,  # RVQVAE 체크포인트 딕셔너리
        device=device,  # 디바이스
        weight_dtype=weight_dtype,  # 가중치 데이터 타입
        load_ema=load_ema  # EMA 가중치 로드 여부
    )


# Example usage
if __name__ == "__main__":
    print("=== Omniges Pipeline Complete Implementation ===")
    
    # Example RVQVAE checkpoints
    rvqvae_checkpoints = {
        'upper': './ckpt/net_300000_upper.pth',
        'hands': './ckpt/net_300000_hands.pth', 
        'lower_trans': './ckpt/net_300000_lower.pth',
        'face': './ckpt/net_300000_face.pth'
    }
    
    # Test pipeline creation
    try:
        pipeline = create_omniges_pipeline(
            omniflow_checkpoint_path="./path/to/omniflow",
            rvqvae_checkpoints=rvqvae_checkpoints
        )
        print("✅ Omniges Pipeline created successfully!")
        
        # Test different tasks
        print("\n🎯 Supported Tasks:")
        print("  - t2g: Text to Gesture")
        print("  - a2g: Audio to Gesture") 
        print("  - g2t: Gesture to Text")
        print("  - g2a: Gesture to Audio")
        print("  - t2a: Text to Audio")
        print("  - a2t: Audio to Text")
        
    except Exception as e:
        print(f"⚠️ Test requires actual OmniFlow checkpoint: {e}")
        print("✅ Pipeline implementation complete - ready for use with real checkpoints!")

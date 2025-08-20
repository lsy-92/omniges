'''
OmnigesFlow: 텍스트, 제스처, 오디오를 통합적으로 처리하는 멀티모달 Transformer 모델
- OmniFlow 기반으로 이미지 스트림을 제스처 스트림으로 교체
- 텍스트, 제스처, 오디오 간의 joint attention을 통한 멀티모달 생성
- 각 모달리티별 독립적인 임베딩과 처리 파이프라인 제공
- LoRA, PEFT 등의 효율적인 파인튜닝 기법 지원
'''

# Diffusers 라이브러리의 핵심 모듈들 임포트
# ============================================================================
# 필요한 라이브러리 및 모듈 임포트
# ============================================================================

# Diffusers 라이브러리에서 기본 클래스들 임포트
from diffusers import ModelMixin, ConfigMixin  # 기본 모델 및 설정 믹스인
from diffusers.configuration_utils import ConfigMixin, register_to_config  # 설정 등록 유틸리티
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin  # 모델 로더 및 PEFT 어댑터
from omniflow.models.attention import JointTransformerBlock  # 기존 Joint Attention 블록 재사용
from diffusers.models.attention_processor import Attention, AttentionProcessor  # 어텐션 프로세서
from diffusers.models.modeling_utils import ModelMixin  # 모델링 유틸리티
from diffusers.models.normalization import AdaLayerNormContinuous  # 적응형 레이어 정규화
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers  # 유틸리티 함수들
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed  # 임베딩 레이어들
from diffusers.models.modeling_outputs import Transformer2DModelOutput  # 트랜스포머 출력 클래스

# 타입 힌트 및 유틸리티
from typing import Tuple  # 튜플 타입 힌트
import inspect  # 코드 검사
from einops import rearrange  # 텐서 재배열
from functools import partial  # 부분 함수
from typing import Any, Dict, List, Optional, Union  # 추가 타입 힌트들

# Transformers 라이브러리
from transformers import BertConfig  # BERT 설정
from transformers.models.bert.modeling_bert import BertEncoder  # BERT 인코더
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel  # LLaMA 모델

# PyTorch 및 기타 라이브러리
import torch  # PyTorch 기본
import torch.nn as nn  # 신경망 모듈
import torch.nn.functional as F  # 함수형 인터페이스
import deepspeed  # DeepSpeed (분산 학습)

# 활성화 함수
from transformers.activations import ACT2FN  # 활성화 함수 딕셔너리

class NNMLP(nn.Module):
    '''
    간단한 2층 MLP (Multi-Layer Perceptron) 모듈
    - CLIP 임베딩 처리 등에 사용되는 피드포워드 네트워크
    - GELU 활성화 함수를 기본으로 사용
    '''
    def __init__(self, input_size, hidden_size, activation='gelu'):
        super().__init__()  # 부모 클래스 초기화
        
        self.linear_1 = nn.Linear(input_size, hidden_size, bias=True)  # 첫 번째 선형 레이어 (입력 → 은닉)
        self.act = ACT2FN[activation]  # 활성화 함수 (기본값: GELU)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)  # 두 번째 선형 레이어 (은닉 → 은닉)

    def forward(self, image_features):
        '''
        MLP 순전파
        Args:
            image_features: 입력 특징 벡터
        Returns:
            변환된 특징 벡터
        '''
        hidden_states = self.linear_1(image_features)  # 첫 번째 선형 변환
        hidden_states = self.act(hidden_states)  # 활성화 함수 적용
        hidden_states = self.linear_2(hidden_states)  # 두 번째 선형 변환
        return hidden_states  # 최종 출력 반환


class GestureEmbedding(nn.Module):
    """
    Gesture Sequence Embedding - 이미지 PatchEmbed를 대체
    제스처 시퀀스를 트랜스포머에 적합한 임베딩으로 변환
    """
    
    def __init__(
        self,
        seq_length: int = 128,          # 제스처 시퀀스 길이
        gesture_latent_dim: int = 512,  # 제스처 잠재 차원 (128*4 from RVQVAE)
        embed_dim: int = 1536,          # 트랜스포머 임베딩 차원 (OmniFlow 호환)
        pos_embed_max_size: int = 128   # 최대 위치 임베딩 크기
    ):
        super().__init__()  # 부모 클래스 초기화
        self.seq_length = seq_length  # 제스처 시퀀스 길이 저장
        self.gesture_latent_dim = gesture_latent_dim  # 제스처 잠재 차원 저장
        self.embed_dim = embed_dim  # 임베딩 차원 저장
        
        # ============================================================================
        # 제스처 잠재 변수를 임베딩 차원으로 변환하는 투영 레이어들
        # ============================================================================
        # Support both single part (128) and 4-part concatenated (512) inputs
        self.gesture_proj_128 = nn.Linear(128, embed_dim)  # 128 -> 1536 (single part)
        self.gesture_proj_512 = nn.Linear(512, embed_dim)  # 512 -> 1536 (4 parts concat)
        
        # ============================================================================
        # 시퀀스 위치 임베딩
        # ============================================================================
        self.position_embedding = nn.Parameter(
            torch.randn(1, pos_embed_max_size, embed_dim) * 0.02  # 랜덤 초기화 (0.02 스케일링)
        )
        
        # ============================================================================
        # 호환성을 위한 속성들 (기존 PatchEmbed와 동일)
        # ============================================================================
        self.num_patches = seq_length  # 패치 수를 시퀀스 길이로 설정
        self.embed_dim = embed_dim  # 임베딩 차원 설정
        
    def forward(self, gesture_latents):
        """
        제스처 latents를 시퀀스 임베딩으로 변환
        
        Args:
            gesture_latents: RVQVAE에서 온 제스처 잠재 변수
        Returns:
            embeddings: (B, T, embed_dim) - 시퀀스 임베딩
        """
        # ============================================================================
        # 디버그: 실제 입력 형태 출력
        # ============================================================================
        print(f"DEBUG: Input gesture_latents shape: {gesture_latents.shape}")
        
        # ============================================================================
        # 입력 형태 처리: (B, 512, T, 1) - 4개 부위 결합
        # ============================================================================
        if len(gesture_latents.shape) == 4:
            B, latent_dim, T, W = gesture_latents.shape  # 배치, 잠재차원, 시퀀스길이, 너비
            print(f"DEBUG: [GestureEmbedding] 4D input - B:{B}, latent_dim:{latent_dim}, T:{T}, W:{W}")
            
            if latent_dim == 512 and W == 1:  # (B, 512, T, 1) - 4개 부위 결합 (shortcut_rvqvae_trainer.py 방식)
                # 변환: (B, 512, T, 1) -> (B, T, 512)
                gesture_features = gesture_latents.squeeze(-1).permute(0, 2, 1)  # 마지막 차원 제거 후 차원 순서 변경
                print(f"DEBUG: [GestureEmbedding] 512-concat to gesture_features shape: {gesture_features.shape}")
            elif latent_dim == 128 and W == 1:  # (B, 128, T, 1) - 단일 부위 폴백
                # 변환: (B, 128, T, 1) -> (B, T, 128)
                gesture_features = gesture_latents.squeeze(-1).permute(0, 2, 1)  # 마지막 차원 제거 후 차원 순서 변경
                print(f"DEBUG: [GestureEmbedding] Single 128 part to gesture_features shape: {gesture_features.shape}")
            elif latent_dim == 128 and W == 4:  # (B, 128, T, 4) 형태 - 4개 부위 스택
                # 변환: (B, 128, T, 4) -> (B, T, 128*4) 모든 부위 결합
                gesture_features = gesture_latents.permute(0, 2, 1, 3)  # (B, T, 128, 4)
                gesture_features = gesture_features.reshape(B, T, latent_dim * W)  # (B, T, 512)
                print(f"DEBUG: [GestureEmbedding] 4-part stack to concat shape: {gesture_features.shape}")
            else:
                raise ValueError(f"Unexpected dimensions: latent_dim={latent_dim}, W={W}")  # 예상치 못한 차원 에러
        else:
            raise ValueError(f"Unexpected input shape: {gesture_latents.shape}")  # 예상치 못한 입력 형태 에러
        
        print(f"DEBUG: After reshape gesture_features shape: {gesture_features.shape}")
        
        # ============================================================================
        # 투영: 입력 차원에 따라 적절한 투영 레이어 선택
        # ============================================================================
        input_dim = gesture_features.shape[-1]  # 입력 차원 추출
        if input_dim == 128:
            embeddings = self.gesture_proj_128(gesture_features)  # 128차원 → 1536차원 투영
        elif input_dim == 512:
            embeddings = self.gesture_proj_512(gesture_features)  # 512차원 → 1536차원 투영
        else:
            raise ValueError(f"Unexpected input dimension: {input_dim}. Expected 128 or 512.")  # 예상치 못한 입력 차원 에러
        
        print(f"DEBUG: Used projection for {input_dim}D input")
        
        # ============================================================================
        # 위치 임베딩 추가
        # ============================================================================
        seq_len = embeddings.shape[1]  # 시퀀스 길이 추출
        if seq_len <= self.position_embedding.shape[1]:
            pos_emb = self.position_embedding[:, :seq_len, :]  # 위치 임베딩 슬라이싱
        else:
            # 긴 시퀀스의 경우 선형 보간
            pos_emb = F.interpolate(
                self.position_embedding.transpose(1, 2),  # 차원 순서 변경
                size=seq_len,  # 목표 크기
                mode='linear',  # 선형 보간
                align_corners=False  # 코너 정렬 비활성화
            ).transpose(1, 2)  # 차원 순서 복원
            
        embeddings = embeddings + pos_emb  # 위치 임베딩 추가
        
        return embeddings  # 최종 임베딩 반환


class OmnigesFlowTransformerModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    Omniges Flow 멀티모달 Transformer 모델
    - OmniFlow 기반으로 이미지 스트림을 제스처 스트림으로 교체
    - 텍스트, 제스처, 오디오 간의 joint attention을 통한 멀티모달 생성
    - 각 모달리티별 독립적인 임베딩 및 처리 파이프라인

    Parameters:
        seq_length (`int`): 제스처 시퀀스 길이 (기존 sample_size 대체)
        gesture_latent_dim (`int`): 제스처 잠재 차원 (기존 in_channels 대체)
        num_layers (`int`): Transformer 블록 레이어 수 (기본값: 18)
        attention_head_dim (`int`): 각 attention head의 차원 (기본값: 64)
        num_attention_heads (`int`): Multi-head attention의 head 수 (기본값: 18)
        joint_attention_dim (`int`): Joint attention 차원 (기본값: 4096)
        caption_projection_dim (`int`): 텍스트 임베딩 투영 차원 (기본값: 1152)
        pooled_projection_dim (`int`): Pooled projection 차원 (기본값: 2048)
        audio_input_dim (`int`): 오디오 입력 차원 (기본값: 8)
        gesture_output_dim (`int`): 제스처 출력 차원 (기존 out_channels 대체)
        pos_embed_max_size (`int`): 위치 임베딩 최대 크기 (기본값: 128)
        dual_attention_layers: 듀얼 어텐션을 사용할 레이어 인덱스
        decoder_config: 텍스트 디코더 설정
        add_audio: 오디오 모달리티 사용 여부
        add_clip: CLIP 임베딩 사용 여부
        use_audio_mae: 오디오 MAE 사용 여부
        drop_text/drop_gesture/drop_audio: 각 모달리티 드롭아웃 여부
        qk_norm: Query-Key 정규화 방법
    """

    _supports_gradient_checkpointing = True  # 그래디언트 체크포인팅 지원 플래그

    @register_to_config  # 설정 등록 데코레이터
    def __init__(
        self,
        seq_length: int = 128,               # 제스처 시퀀스 길이 (기존 sample_size 대체)
        gesture_latent_dim: int = 512,       # 제스처 잠재 차원 (128*4, 기존 in_channels 대체)
        num_layers: int = 18,                # Transformer 레이어 수
        attention_head_dim: int = 64,        # Attention head 차원
        num_attention_heads: int = 18,       # Attention head 수
        joint_attention_dim: int = 4096,     # Joint attention 차원
        caption_projection_dim: int = 1152,  # 캡션 투영 차원
        pooled_projection_dim: int = 2048,   # Pooled 투영 차원
        audio_input_dim: int = 8,            # 오디오 입력 차원
        gesture_output_dim: int = 512,       # 제스처 출력 차원 (4 parts * 128 = 512, 기존 out_channels 대체)
        pos_embed_max_size: int = 128,       # 위치 임베딩 최대 크기
        dual_attention_layers: Tuple[int, ...] = (),  # 듀얼 어텐션 레이어 인덱스
        decoder_config: str = '',            # 텍스트 디코더 설정
        add_audio=True,                      # 오디오 모달리티 사용 여부
        add_clip=False,                      # CLIP 임베딩 사용 여부
        use_audio_mae=False,                 # 오디오 MAE 사용 여부
        drop_text=False,                     # 텍스트 드롭아웃 여부
        drop_gesture=False,                  # 제스처 드롭아웃 여부 (기존 drop_image 대체)
        drop_audio=False,                    # 오디오 드롭아웃 여부
        qk_norm: Optional[str] = 'layer_norm',  # Query-Key 정규화 방법
    ):
        '''
        Omniges Flow Transformer 모델 초기화
        - 각 모달리티별 임베딩 레이어 설정
        - Transformer 블록들 구성
        - 출력 레이어들 초기화
        '''
        super().__init__()  # 부모 클래스 초기화
        default_gesture_output_dim = gesture_latent_dim  # 기본 제스처 출력 차원 설정
        self.add_clip = add_clip  # CLIP 사용 여부 저장
        self.gesture_output_dim = gesture_output_dim if gesture_output_dim is not None else default_gesture_output_dim  # 제스처 출력 차원 설정
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim  # 내부 차원 계산 (헤드 수 × 헤드 차원)

        # ============================================================================
        # 제스처 시퀀스 임베딩 레이어 (기존 PatchEmbed 대체)
        # ============================================================================
        self.gesture_embed = GestureEmbedding(
            seq_length=seq_length,  # 제스처 시퀀스 길이
            gesture_latent_dim=gesture_latent_dim,  # 제스처 잠재 차원
            embed_dim=self.inner_dim,  # 임베딩 차원 (내부 차원과 동일)
            pos_embed_max_size=pos_embed_max_size,  # 위치 임베딩 최대 크기
        )
        
        # ============================================================================
        # 시간 단계와 텍스트 임베딩을 결합하는 레이어 (동일)
        # ============================================================================
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim,  # 임베딩 차원
            pooled_projection_dim=self.config.pooled_projection_dim  # 풀링된 투영 차원
        )
        
        # ============================================================================
        # 오디오 모달리티 처리 (오디오 모달리티가 활성화된 경우)
        # ============================================================================
        if add_audio:  # 오디오 모달리티가 활성화된 경우 (동일)
            # ============================================================================
            # 제스처용 시간 임베딩 (기존 이미지용을 제스처용으로 변경)
            # ============================================================================
            self.time_gesture_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim,  # 임베딩 차원
                pooled_projection_dim=self.config.pooled_projection_dim  # 풀링된 투영 차원
            )
            self.audio_input_dim = audio_input_dim  # 오디오 입력 차원 저장
            self.use_audio_mae = use_audio_mae  # 오디오 MAE 사용 여부 저장
            self.audio_patch_size = 2  # 오디오 패치 크기 설정
            
            # ============================================================================
            # 오디오 임베딩 레이어 (MAE 또는 PatchEmbed)
            # ============================================================================
            if use_audio_mae:
                self.audio_embedder = nn.Linear(audio_input_dim, self.config.caption_projection_dim)  # MAE: 선형 레이어
            else:
                self.audio_embedder = PatchEmbed(  # 일반: 패치 임베딩
                    height=256,  # 높이
                    width=16,  # 너비
                    patch_size=self.audio_patch_size,  # 패치 크기
                    in_channels=self.audio_input_dim,  # 입력 채널 수
                    embed_dim=self.config.caption_projection_dim,  # 임베딩 차원
                    pos_embed_max_size=192  # 위치 임베딩 최대 크기
                )
            
            # ============================================================================
            # 오디오용 시간 임베딩 (동일)
            # ============================================================================
            self.time_aud_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim,  # 임베딩 차원
                pooled_projection_dim=self.config.pooled_projection_dim  # 풀링된 투영 차원
            )
         
            # ============================================================================
            # 오디오 출력 정규화 레이어 (동일)
            # ============================================================================
            self.norm_out_aud = AdaLayerNormContinuous(self.config.caption_projection_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)  # 적응형 레이어 정규화
            if use_audio_mae:
                self.proj_out_aud = nn.Linear(self.config.caption_projection_dim, self.config.audio_input_dim)  # MAE: 선형 투영
            else:
                self.proj_out_aud = nn.Linear(self.inner_dim, self.audio_patch_size * self.audio_patch_size * self.audio_input_dim, bias=True)  # 일반: 패치 복원 투영
        
        # ============================================================================
        # 컨텍스트 임베딩 레이어 (동일)
        # ============================================================================
        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.config.caption_projection_dim)  # 조인트 어텐션 차원 → 캡션 투영 차원
        
        # ============================================================================
        # LLaMA 설정 생성 (동일)
        # ============================================================================
        bert_config = LlamaConfig(1, hidden_size=self.config.joint_attention_dim, num_attention_heads=32, num_hidden_layers=2)  # LLaMA 설정 (사용되지 않음)
        if self.add_audio:
            self.context_decoder = nn.ModuleDict(dict(
                projection=nn.Linear(self.config.caption_projection_dim, self.config.joint_attention_dim)  # 캡션 투영 차원 → 조인트 어텐션 차원
            ))
        self.text_out_dim = 1536  # 텍스트 출력 차원 설정
        self.text_output = nn.Linear(self.config.joint_attention_dim, self.text_out_dim)  # 조인트 어텐션 차원 → 텍스트 출력 차원
        
        # ============================================================================
        # Transformer 블록들 생성 (drop_image -> drop_gesture로 변경)
        # ============================================================================
        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(  # 조인트 트랜스포머 블록
                    dim=self.inner_dim,  # 내부 차원
                    num_attention_heads=self.config.num_attention_heads,  # 어텐션 헤드 수
                    attention_head_dim=self.config.attention_head_dim,  # 어텐션 헤드 차원
                    context_pre_only=i == num_layers - 1,  # 마지막 레이어에서만 컨텍스트 사전 처리
                    context_output=i < num_layers or self.add_audio,  # 컨텍스트 출력 조건
                    audio_output=add_audio,  # 오디오 출력 여부
                    delete_img=drop_gesture,     # 기존 drop_image -> drop_gesture (제스처 드롭)
                    delete_aud=drop_audio,  # 오디오 드롭
                    delete_text=drop_text,  # 텍스트 드롭
                    qk_norm=qk_norm,  # Query-Key 정규화 방법
                    use_dual_attention=True if i in dual_attention_layers else False,  # 듀얼 어텐션 사용 여부
                )
                for i in range(self.config.num_layers)  # 레이어 수만큼 반복
            ]
        )
        self.add_audio = add_audio  # 오디오 추가 여부 저장
        
        # ============================================================================
        # 출력 정규화 레이어들
        # ============================================================================
        
        # 제스처 출력 정규화 레이어 (기존 이미지 출력과 동일)
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)  # 적응형 레이어 정규화
        
        # 텍스트 출력 정규화 레이어 (동일)
        self.norm_out_text = AdaLayerNormContinuous(self.config.joint_attention_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)  # 적응형 레이어 정규화
        
        # ============================================================================
        # CLIP 조건부 토큰 처리 (CLIP 사용 시)
        # ============================================================================
        if self.add_clip:
            self.n_cond_tokens = 8  # 조건부 토큰 수 설정
            self.clip_proj = nn.Sequential(  # CLIP 투영 레이어
                NNMLP(self.config.pooled_projection_dim, self.config.caption_projection_dim),  # MLP 레이어
                nn.Linear(self.config.caption_projection_dim, self.config.caption_projection_dim * self.n_cond_tokens)  # 선형 투영
            )
            
        # ============================================================================
        # 최종 제스처 출력 투영 레이어 (패치 복원 대신 시퀀스 차원으로 투영)
        # ============================================================================
        self.proj_out = nn.Linear(self.inner_dim, gesture_output_dim, bias=True)  # 내부 차원 → 제스처 출력 차원

        # ============================================================================
        # 기타 설정
        # ============================================================================
        self.gradient_checkpointing = False  # 그래디언트 체크포인팅 비활성화
        if decoder_config:
            self.text_decoder = None  # 실제 구현 시 수정 필요 (텍스트 디코더 설정)
        else:
            self.text_decoder = None  # 텍스트 디코더 없음

    def set_text_decoder(self, model):
        '''
        텍스트 디코더 설정 (동일)
        '''
        self.text_decoder = model  # 텍스트 디코더 모델 저장
        self.text_out_dim = model.vae_dim  # 모델의 VAE 차원을 텍스트 출력 차원으로 설정
        self.text_output = nn.Linear(self.config.joint_attention_dim, self.text_out_dim)  # 조인트 어텐션 차원 → 텍스트 출력 차원으로 투영하는 선형 레이어 생성
        
    def set_audio_pooler(self, model):
        '''
        오디오 풀러 설정 (동일)
        '''
        self.audio_pooler = model  # 오디오 풀러 모델 저장
        
    def get_decoder(self):
        '''
        텍스트 디코더 반환 (동일)
        '''
        return self.text_decoder  # 저장된 텍스트 디코더 모델 반환

    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        피드포워드 청킹을 사용하는 attention processor 설정 (동일)
        """
        # ============================================================================
        # 차원 검증
        # ============================================================================
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")  # 차원이 0 또는 1이 아니면 에러

        chunk_size = chunk_size or 1  # 청크 크기가 없으면 기본값 1 설정

        # ============================================================================
        # 재귀적 피드포워드 청킹 설정 함수
        # ============================================================================
        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):  # 모듈에 청킹 설정 메서드가 있으면
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)  # 청킹 설정 적용

            for child in module.children():  # 모든 자식 모듈에 대해
                fn_recursive_feed_forward(child, chunk_size, dim)  # 재귀적으로 청킹 설정

        # ============================================================================
        # 모든 모듈에 청킹 설정 적용
        # ============================================================================
        for module in self.children():  # 모델의 모든 자식 모듈에 대해
            fn_recursive_feed_forward(module, chunk_size, dim)  # 재귀적으로 청킹 설정 적용

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        """
        모델에서 사용되는 모든 attention processor 반환 (동일)
        """
        processors = {}  # 어텐션 프로세서 딕셔너리 초기화

        # ============================================================================
        # 재귀적 어텐션 프로세서 수집 함수
        # ============================================================================
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):  # 모듈에 프로세서 가져오기 메서드가 있으면
                processors[f"{name}.processor"] = module.get_processor()  # 프로세서를 딕셔너리에 추가

            for sub_name, child in module.named_children():  # 모든 자식 모듈에 대해
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)  # 재귀적으로 프로세서 수집

            return processors  # 수집된 프로세서 딕셔너리 반환

        # ============================================================================
        # 모든 모듈에서 어텐션 프로세서 수집
        # ============================================================================
        for name, module in self.named_children():  # 모델의 모든 명명된 자식 모듈에 대해
            fn_recursive_add_processors(name, module, processors)  # 재귀적으로 프로세서 수집

        return processors  # 수집된 모든 어텐션 프로세서 반환

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        """
        Attention 계산에 사용할 attention processor 설정 (동일)
        """
        count = len(self.attn_processors.keys())  # 현재 어텐션 프로세서 개수 계산

        # ============================================================================
        # 프로세서 개수 검증
        # ============================================================================
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."  # 프로세서 개수가 일치하지 않으면 에러
            )

        # ============================================================================
        # 재귀적 어텐션 프로세서 설정 함수
        # ============================================================================
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):  # 모듈에 프로세서 설정 메서드가 있으면
                if not isinstance(processor, dict):  # 프로세서가 딕셔너리가 아니면
                    module.set_processor(processor)  # 직접 프로세서 설정
                else:  # 프로세서가 딕셔너리이면
                    module.set_processor(processor.pop(f"{name}.processor"))  # 해당 이름의 프로세서를 딕셔너리에서 제거하며 설정

            for sub_name, child in module.named_children():  # 모든 자식 모듈에 대해
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)  # 재귀적으로 프로세서 설정

        # ============================================================================
        # 모든 모듈에 어텐션 프로세서 설정
        # ============================================================================
        for name, module in self.named_children():  # 모델의 모든 명명된 자식 모듈에 대해
            fn_recursive_attn_processor(name, module, processor)  # 재귀적으로 프로세서 설정 적용

    def fuse_qkv_projections(self):
        """QKV 투영 행렬 융합 활성화 (동일)"""
        self.original_attn_processors = None  # 원본 어텐션 프로세서 초기화

        # ============================================================================
        # 추가된 KV 투영 검증
        # ============================================================================
        for _, attn_processor in self.attn_processors.items():  # 모든 어텐션 프로세서에 대해
            if "Added" in str(attn_processor.__class__.__name__):  # 추가된 KV 투영이 있으면
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")  # 융합 지원하지 않음 에러

        self.original_attn_processors = self.attn_processors  # 현재 어텐션 프로세서를 원본으로 저장

        # ============================================================================
        # 모든 어텐션 모듈에 융합 적용
        # ============================================================================
        for module in self.modules():  # 모델의 모든 모듈에 대해
            if isinstance(module, Attention):  # 어텐션 모듈이면
                module.fuse_projections(fuse=True)  # QKV 투영 융합 활성화

    def unfuse_qkv_projections(self):
        """융합된 QKV 투영 비활성화 (동일)"""
        if self.original_attn_processors is not None:  # 원본 어텐션 프로세서가 있으면
            self.set_attn_processor(self.original_attn_processors)  # 원본 프로세서로 복원

    def _set_gradient_checkpointing(self, module, value=False):
        '''그래디언트 체크포인팅 설정 (동일)'''
        if hasattr(module, "gradient_checkpointing"):  # 모듈에 그래디언트 체크포인팅 속성이 있으면
            module.gradient_checkpointing = value  # 그래디언트 체크포인팅 값 설정

    def fuse_lora(self, lora_scale=1.0, safe_fusing=False, adapter_names=None):
        '''LoRA 어댑터 융합 (동일)'''
        if not USE_PEFT_BACKEND:  # PEFT 백엔드가 없으면
            raise ValueError("PEFT backend is required for `fuse_lora()`.")  # PEFT 백엔드 필요 에러

        self.lora_scale = lora_scale  # LoRA 스케일 저장
        self._safe_fusing = safe_fusing  # 안전한 융합 플래그 저장
        self.apply(partial(self._fuse_lora_apply, adapter_names=adapter_names))  # 모든 모듈에 LoRA 융합 적용

    def _fuse_lora_apply(self, module, adapter_names=None):
        '''개별 모듈에 LoRA 융합 적용 (동일)'''
        from peft.tuners.tuners_utils import BaseTunerLayer  # PEFT 튜너 레이어 임포트

        merge_kwargs = {"safe_merge": self._safe_fusing}  # 안전한 융합 설정을 포함한 병합 키워드 인자

        if isinstance(module, BaseTunerLayer):  # 모듈이 PEFT 튜너 레이어이면
            if self.lora_scale != 1.0:  # LoRA 스케일이 1.0이 아니면
                module.scale_layer(self.lora_scale)  # 레이어 스케일링 적용

            supported_merge_kwargs = list(inspect.signature(module.merge).parameters)  # 지원되는 병합 키워드 인자 목록
            if "adapter_names" in supported_merge_kwargs:  # adapter_names가 지원되면
                merge_kwargs["adapter_names"] = adapter_names  # 어댑터 이름 추가
            elif "adapter_names" not in supported_merge_kwargs and adapter_names is not None:  # 지원되지 않는데 어댑터 이름이 제공되면
                raise ValueError(
                    "The `adapter_names` argument is not supported with your PEFT version. Please upgrade"
                    " to the latest version of PEFT. `pip install -U peft`"  # PEFT 버전 업그레이드 필요 에러
                )

            module.merge(**merge_kwargs)  # 모듈 병합 실행

    def unfuse_lora(self):
        '''LoRA 어댑터 융합 해제 (동일)'''
        if not USE_PEFT_BACKEND:  # PEFT 백엔드가 없으면
            raise ValueError("PEFT backend is required for `unfuse_lora()`.")  # PEFT 백엔드 필요 에러
        self.apply(self._unfuse_lora_apply)  # 모든 모듈에 LoRA 융합 해제 적용

    def _unfuse_lora_apply(self, module):
        '''개별 모듈에 LoRA 융합 해제 적용 (동일)'''
        from peft.tuners.tuners_utils import BaseTunerLayer  # PEFT 튜너 레이어 임포트

        if isinstance(module, BaseTunerLayer):  # 모듈이 PEFT 튜너 레이어이면
            module.unmerge()  # 모듈 병합 해제 실행

    def forward(
        self,
        hidden_states: torch.FloatTensor = None,      # 제스처 잠재 상태 (기존 이미지 잠재 상태)
        encoder_hidden_states: torch.FloatTensor = None,  # 인코더 숨겨진 상태 (텍스트)
        pooled_projections: torch.FloatTensor = None,     # 풀링된 투영
        timestep: torch.LongTensor = None,                # 시간 단계
        timestep_text: torch.LongTensor = None,           # 텍스트용 시간 단계
        timestep_audio: torch.LongTensor = None,          # 오디오용 시간 단계
        block_controlnet_hidden_states: List = None,     # ControlNet 숨겨진 상태
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,  # Joint attention 인수
        return_dict: bool = True,                         # 딕셔너리 반환 여부
        use_text_output: bool = False,                    # 텍스트 출력 사용 여부
        target_prompt_embeds=None,                        # 타겟 프롬프트 임베딩
        decode_text=False,                                # 텍스트 디코딩 여부
        sigma_text=None,                                  # 텍스트 노이즈 시그마
        detach_logits=False,                              # 로짓 분리 여부
        prompt_embeds_uncond=None,                        # 무조건 프롬프트 임베딩
        targets=None,                                     # 타겟 데이터
        audio_hidden_states=None,                         # 오디오 숨겨진 상태
        split_cond=False,                                 # 조건 분할 여부
        text_vae=None,                                    # 텍스트 VAE
        text_x0=True,                                     # 텍스트 x0 예측 여부
        drop_text=False,                                  # 텍스트 드롭 여부
        drop_gesture=False,                               # 제스처 드롭 여부 (기존 drop_image)
        drop_audio=False,                                 # 오디오 드롭 여부
        kkwargs=None,                                     # 추가 키워드 인수
        forward_function=None,                            # 커스텀 순전파 함수
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        Omniges Flow Transformer 모델의 순전파 메서드
        - 텍스트, 제스처, 오디오를 통합적으로 처리
        - Joint attention을 통한 모달리티 간 상호작용
        - 각 모달리티별 독립적인 처리 및 출력 생성

        Args:
            hidden_states: 제스처 잠재 상태 (batch_size, latent_dim, seq_length, num_parts)
            encoder_hidden_states: 텍스트 조건부 임베딩 (batch_size, seq_len, embed_dim)
            pooled_projections: 풀링된 조건부 임베딩 (batch_size, projection_dim)
            timestep: 디노이징 단계를 나타내는 시간 단계
            Other args same as OmniFlow...

        Returns:
            딕셔너리 형태의 출력:
            - output: 제스처 출력 (기존 이미지 출력)
            - model_pred_text: 텍스트 예측
            - encoder_hidden_states: 인코더 상태
            - logits: 텍스트 디코딩 로짓
            - audio_hidden_states: 오디오 출력
        """
        # ============================================================================
        # 커스텀 순전파 함수 처리
        # ============================================================================
        if kkwargs is not None:  # 추가 키워드 인수가 제공되면
            assert forward_function is not None  # 커스텀 순전파 함수가 반드시 있어야 함
            return forward_function(transformer=self, **kkwargs)  # 커스텀 함수로 처리 위임
        
        # ============================================================================
        # 입력 상태 백업 (텍스트 디코딩에서 사용)
        # ============================================================================
        encoder_hidden_states_base = encoder_hidden_states.clone() if encoder_hidden_states is not None else None  # 텍스트 인코더 상태 백업
        hidden_states_base = hidden_states  # 제스처 상태 백업
        
        # ============================================================================
        # 각 모달리티 처리 여부 결정
        # ============================================================================
        do_gesture = not drop_gesture  # 제스처 처리가 활성화된 경우 (기존 do_image -> do_gesture)
        do_audio = (not drop_audio) and (self.add_audio)  # 오디오 처리가 활성화된 경우 (오디오 추가 모드가 켜져 있어야 함)
        do_text = (not drop_text)  # 텍스트 처리가 활성화된 경우
        # ============================================================================
        # 제스처 처리 (기존 이미지 처리 로직 수정)
        # ============================================================================
        if do_gesture:  # 제스처 처리가 활성화된 경우
            seq_length = hidden_states.shape[-2]  # 제스처 시퀀스 길이 추출 (기존 height, width 대신)
            print(f"DEBUG: [OmnigesFlow] Before gesture_embed - hidden_states shape: {hidden_states.shape}")  # 제스처 임베딩 전 형태 출력
            hidden_states = self.gesture_embed(hidden_states)  # 제스처 임베딩 적용 (GestureEmbedding 레이어 통과)
            print(f"DEBUG: [OmnigesFlow] After gesture_embed - hidden_states shape: {hidden_states.shape}")  # 제스처 임베딩 후 형태 출력
            temb = self.time_text_embed(timestep, pooled_projections)  # 제스처용 시간 임베딩 생성 (시간 + 풀링된 투영 결합)
        else:  # 제스처 처리가 비활성화된 경우
            hidden_states = None  # 제스처 상태를 None으로 설정
            temb = 0  # 시간 임베딩을 0으로 설정
        
        # ============================================================================
        # 오디오 처리 (기존 오디오 처리 로직 수정)
        # ============================================================================
        if do_audio:  # 오디오 처리가 활성화된 경우
            print(f"DEBUG: [OmnigesFlow] Audio processing enabled")  # 오디오 처리 활성화 디버그 출력
            if audio_hidden_states is None:  # 오디오 숨겨진 상태가 제공되지 않은 경우
                print(f"DEBUG: [OmnigesFlow] Audio hidden states is None - creating dummy")  # 더미 오디오 생성 알림
                if self.use_audio_mae:  # MAE 방식 사용 시
                    audio_hidden_states = torch.zeros(encoder_hidden_states.shape[0], 8, self.audio_input_dim).to(encoder_hidden_states)  # MAE용 더미 오디오 생성 (B, 8, audio_dim)
                    print(f"DEBUG: [OmnigesFlow] Dummy audio MAE shape: {audio_hidden_states.shape}")  # MAE 더미 형태 출력
                else:  # 표준 패치 방식 사용 시
                    audio_hidden_states = torch.zeros(encoder_hidden_states.shape[0], self.audio_input_dim, 256, 16).to(encoder_hidden_states)  # 표준용 더미 오디오 생성 (B, audio_dim, 256, 16)
                    print(f"DEBUG: [OmnigesFlow] Dummy audio standard shape: {audio_hidden_states.shape}")  # 표준 더미 형태 출력
                timestep_audio = timestep_text * 0 if timestep_text is not None else torch.zeros_like(timestep)  # 오디오 시간 단계를 0으로 설정
            else:  # 오디오 숨겨진 상태가 제공된 경우
                print(f"DEBUG: [OmnigesFlow] Input audio_hidden_states shape: {audio_hidden_states.shape}")  # 입력 오디오 형태 출력
            
            temb_audio = self.time_aud_embed(timestep_audio, pooled_projections)  # 오디오용 시간 임베딩 생성
            print(f"DEBUG: [OmnigesFlow] Audio time embedding shape: {temb_audio.shape}")  # 오디오 시간 임베딩 형태 출력
            
            # ============================================================================
            # 오디오 데이터 형태 확인 및 적절히 처리
            # ============================================================================
            if audio_hidden_states.dim() == 3:  # 3차원 텐서인 경우 (B, T, D)
                # (B, T, D) 형태인 경우 -> (B, C, H, W) 형태로 변환
                B, T, D = audio_hidden_states.shape  # 배치, 시간, 차원 추출
                if self.use_audio_mae:  # MAE 방식 사용 시
                    # MAE 방식: 직접 linear projection
                    audio_hidden_states = self.audio_embedder(audio_hidden_states)  # MAE 임베더로 직접 투영
                else:  # 패치 방식 사용 시
                    # Patch 방식: reshape 후 처리
                    audio_hidden_states = audio_hidden_states.view(B, self.audio_input_dim, 256, 16)  # (B, T, D) -> (B, audio_dim, 256, 16) 형태로 변환
                    audio_hidden_states = self.audio_embedder(audio_hidden_states)  # 오디오 임베더 적용
            else:  # 이미 올바른 형태인 경우 (4차원)
                # 이미 올바른 형태인 경우
                audio_hidden_states = self.audio_embedder(audio_hidden_states)  # 오디오 임베더 적용
                
            if not split_cond:  # 조건 분할이 비활성화된 경우
                temb = temb + temb_audio  # 제스처와 오디오 시간 임베딩 결합
                temb_audio = None  # 개별 오디오 시간 임베딩 제거
        else:  # 오디오 처리가 비활성화된 경우
            audio_hidden_states = None  # 오디오 상태를 None으로 설정
            temb_audio = None  # 오디오 시간 임베딩을 None으로 설정
           
        # ============================================================================
        # 텍스트 처리 (기존 텍스트 처리 로직과 동일)
        # ============================================================================
        if do_text:  # 텍스트 처리가 활성화된 경우
            print(f"DEBUG: [OmnigesFlow] Text processing enabled")  # 텍스트 처리 활성화 디버그 출력
            print(f"DEBUG: [OmnigesFlow] Input encoder_hidden_states shape: {encoder_hidden_states.shape}")  # 입력 텍스트 인코더 상태 형태 출력
            if use_text_output:  # 텍스트 출력이 활성화된 경우
                temb_text = self.time_gesture_embed(timestep_text, pooled_projections)  # 텍스트용 시간 임베딩 생성 (기존 time_image_embed -> time_gesture_embed)
                print(f"DEBUG: [OmnigesFlow] Text time embedding shape: {temb_text.shape}")  # 텍스트 시간 임베딩 형태 출력
            encoder_hidden_states = self.context_embedder(encoder_hidden_states)  # 컨텍스트 임베더 적용
            print(f"DEBUG: [OmnigesFlow] After context_embedder - encoder_hidden_states shape: {encoder_hidden_states.shape}")  # 컨텍스트 임베더 후 형태 출력
            if use_text_output:  # 텍스트 출력이 활성화된 경우
                if not split_cond:  # 조건 분할이 비활성화된 경우
                    temb = temb + temb_text  # 제스처와 텍스트 시간 임베딩 결합
                    temb_text = None  # 개별 텍스트 시간 임베딩 제거
                    print(f"DEBUG: [OmnigesFlow] Combined temb shape: {temb.shape}")  # 결합된 시간 임베딩 형태 출력
            else:  # 텍스트 출력이 비활성화된 경우
                temb_text = None  # 텍스트 시간 임베딩을 None으로 설정
        else:  # 텍스트 처리가 비활성화된 경우
            encoder_hidden_states = None  # 텍스트 인코더 상태를 None으로 설정
            temb_text = None  # 텍스트 시간 임베딩을 None으로 설정
            
        # ============================================================================
        # CLIP 사용 검증 및 Transformer 블록 처리 준비
        # ============================================================================
        assert not self.add_clip  # CLIP 사용이 비활성화되어 있음을 확인 (Omniges에서는 CLIP을 사용하지 않음)

        # ============================================================================
        # Transformer 블록들을 순차적으로 처리 (기존 OmniFlow와 동일)
        # ============================================================================
        print(f"DEBUG: [OmnigesFlow] ========== TRANSFORMER BLOCKS ==========")  # 트랜스포머 블록 처리 시작 알림
        print(f"DEBUG: [OmnigesFlow] Total transformer blocks: {len(self.transformer_blocks)}")  # 총 트랜스포머 블록 수 출력
        print(f"DEBUG: [OmnigesFlow] Before blocks - hidden_states shape: {hidden_states.shape if hidden_states is not None else None}")  # 블록 전 제스처 상태 형태
        print(f"DEBUG: [OmnigesFlow] Before blocks - encoder_hidden_states shape: {encoder_hidden_states.shape if encoder_hidden_states is not None else None}")  # 블록 전 텍스트 상태 형태
        print(f"DEBUG: [OmnigesFlow] Before blocks - audio_hidden_states shape: {audio_hidden_states.shape if audio_hidden_states is not None else None}")  # 블록 전 오디오 상태 형태
    
        # ============================================================================
        # 각 Transformer 블록을 순차적으로 처리
        # ============================================================================
        for index_block, block in enumerate(self.transformer_blocks):  # 모든 트랜스포머 블록을 인덱스와 함께 순회
            # ============================================================================
            # 그래디언트 체크포인팅 처리 (메모리 효율성을 위한 기법)
            # ============================================================================
            if self.training and self.gradient_checkpointing:  # 훈련 중이고 그래디언트 체크포인팅이 활성화된 경우
                def create_custom_forward(module, return_dict=None):  # 커스텀 순전파 함수 생성
                    def custom_forward(*inputs):  # 실제 순전파 함수
                        if return_dict is not None:  # return_dict가 제공된 경우
                            return module(*inputs, return_dict=return_dict)  # return_dict와 함께 모듈 호출
                        else:  # return_dict가 제공되지 않은 경우
                            return module(*inputs)  # 기본 모듈 호출
                    return custom_forward  # 커스텀 순전파 함수 반환

                ckpt_kwargs = dict()  # 체크포인팅 키워드 인자 초기화
                if self.add_audio:  # 오디오가 추가된 경우 (3개 모달리티)
                    encoder_hidden_states, hidden_states, audio_hidden_states = deepspeed.checkpointing.checkpoint(
                        create_custom_forward(block),  # 커스텀 순전파 함수
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        audio_hidden_states,
                        temb_text,
                        temb_audio,
                        **ckpt_kwargs,
                    )
                else:  # 오디오가 없는 경우 (2개 모달리티)
                    encoder_hidden_states, hidden_states = deepspeed.checkpointing.checkpoint(
                        create_custom_forward(block),  # 커스텀 순전파 함수
                        hidden_states,  # 제스처 상태
                        encoder_hidden_states,  # 텍스트 상태
                        temb,  # 제스처 시간 임베딩
                        temb_text,  # 텍스트 시간 임베딩
                        **ckpt_kwargs,  # 추가 키워드 인자
                    )

            else:  # 일반적인 순전파 (그래디언트 체크포인팅 비활성화)
                if self.add_audio:  # 오디오가 추가된 경우 (3개 모달리티)
                    encoder_hidden_states, hidden_states, audio_hidden_states = block(  # 트랜스포머 블록 호출
                        hidden_states=hidden_states,  # 제스처 상태
                        encoder_hidden_states=encoder_hidden_states,  # 텍스트 상태
                        audio_hidden_states=audio_hidden_states,  # 오디오 상태
                        temb=temb,  # 제스처 시간 임베딩
                        temb_text=temb_text,  # 텍스트 시간 임베딩
                        temb_audio=temb_audio,  # 오디오 시간 임베딩
                    )
                    # 디버그 출력 (처음 몇 개 블록과 마지막 블록만 출력하여 스팸 방지)
                    if index_block < 3 or index_block == len(self.transformer_blocks) - 1:
                        print(f"DEBUG: [OmnigesFlow] Block {index_block}/{len(self.transformer_blocks)-1} output:")  # 블록 인덱스 출력
                        print(f"DEBUG:   hidden_states shape: {hidden_states.shape if hidden_states is not None else None}")  # 제스처 상태 형태
                        print(f"DEBUG:   encoder_hidden_states shape: {encoder_hidden_states.shape if encoder_hidden_states is not None else None}")  # 텍스트 상태 형태
                        print(f"DEBUG:   audio_hidden_states shape: {audio_hidden_states.shape if audio_hidden_states is not None else None}")  # 오디오 상태 형태
                else:  # 오디오가 없는 경우 (2개 모달리티)
                    encoder_hidden_states, hidden_states = block(  # 트랜스포머 블록 호출
                        hidden_states=hidden_states,  # 제스처 상태
                        encoder_hidden_states=encoder_hidden_states,  # 텍스트 상태
                        temb=temb,  # 제스처 시간 임베딩
                        temb_text=temb_text  # 텍스트 시간 임베딩
                    )
                    # 디버그 출력 (처음 몇 개 블록과 마지막 블록만 출력)
                    if index_block < 3 or index_block == len(self.transformer_blocks) - 1:
                        print(f"DEBUG: [OmnigesFlow] Block {index_block}/{len(self.transformer_blocks)-1} output (no audio):")  # 오디오 없는 블록 인덱스 출력
                        print(f"DEBUG:   hidden_states shape: {hidden_states.shape if hidden_states is not None else None}")  # 제스처 상태 형태
                        print(f"DEBUG:   encoder_hidden_states shape: {encoder_hidden_states.shape if encoder_hidden_states is not None else None}")  # 텍스트 상태 형태

            # ============================================================================
            # ControlNet 잔차 처리 (현재는 사용하지 않음)
            # ============================================================================
            assert block_controlnet_hidden_states is None  # ControlNet 숨겨진 상태가 None이어야 함 (현재 미사용)

        # ============================================================================
        # 제스처 출력 처리 (기존 이미지 출력 로직을 제스처에 맞게 수정)
        # ============================================================================
        if do_gesture:  # 제스처 처리가 활성화된 경우
            print(f"DEBUG: [OmnigesFlow] Before norm_out - hidden_states shape: {hidden_states.shape}")  # 정규화 전 제스처 상태 형태 출력
            hidden_states = self.norm_out(hidden_states, temb)  # 출력 정규화 적용 (AdaLayerNormContinuous)
            print(f"DEBUG: [OmnigesFlow] After norm_out - hidden_states shape: {hidden_states.shape}")  # 정규화 후 제스처 상태 형태 출력
            hidden_states = self.proj_out(hidden_states)  # 최종 투영 레이어 적용 (선형 투영)
            print(f"DEBUG: [OmnigesFlow] After proj_out - hidden_states shape: {hidden_states.shape}")  # 투영 후 제스처 상태 형태 출력

            # ============================================================================
            # 제스처 출력 형태 변환 (입력과 동일한 형태로 만들기)
            # shortcut_rvqvae_trainer.py 패턴을 따라 latents가 결합된 형태로 변환
            # ============================================================================
            B, T, gesture_output_dim = hidden_states.shape  # 배치, 시퀀스 길이, 제스처 출력 차원 추출
            print(f"DEBUG: [OmnigesFlow] Output dimensions - B:{B}, T:{T}, gesture_output_dim:{gesture_output_dim}")  # 출력 차원 정보 출력
            if gesture_output_dim == 512:  # 512차원인 경우 (4개 부위 결합)
                # 512 결합 형태 유지: (B, T, 512) -> (B, 512, T, 1)
                gesture_output = hidden_states.permute(0, 2, 1).unsqueeze(-1)  # 차원 순서 변경 후 마지막 차원 추가
                print(f"DEBUG: [OmnigesFlow] 512-dim output reshape: {gesture_output.shape}")  # 512차원 출력 형태 출력
            elif gesture_output_dim == 128:  # 128차원인 경우 (단일 부위)
                # 단일 부위 폴백: (B, T, 128) -> (B, 128, T, 1)
                gesture_output = hidden_states.permute(0, 2, 1).unsqueeze(-1)  # 차원 순서 변경 후 마지막 차원 추가
                print(f"DEBUG: [OmnigesFlow] 128-dim output reshape: {gesture_output.shape}")  # 128차원 출력 형태 출력
            else:  # 예상치 못한 차원인 경우
                raise ValueError(f"Unexpected gesture output dimension: {gesture_output_dim}")  # 차원 에러 발생
                
            print(f"DEBUG: [OmnigesFlow] Final gesture_output shape: {gesture_output.shape}")  # 최종 제스처 출력 형태 출력
        else:  # 제스처 처리가 비활성화된 경우
            gesture_output = None  # 제스처 출력을 None으로 설정

        # ============================================================================
        # 텍스트 출력 처리 (기존 OmniFlow와 동일)
        # ============================================================================
        logits = None  # 텍스트 디코딩 로짓 초기화
        logits_labels = None  # 텍스트 라벨 로짓 초기화
        
        if do_text and use_text_output:  # 텍스트 처리와 출력이 모두 활성화된 경우
            print(f"DEBUG: [OmnigesFlow] Text output processing enabled")  # 텍스트 출력 처리 활성화 디버그 출력
            print(f"DEBUG: [OmnigesFlow] Before context_decoder projection: {encoder_hidden_states.shape}")  # 컨텍스트 디코더 투영 전 형태 출력
            encoder_hidden_states = self.context_decoder['projection'](encoder_hidden_states)  # 컨텍스트 디코더 투영 적용
            print(f"DEBUG: [OmnigesFlow] After context_decoder projection: {encoder_hidden_states.shape}")  # 컨텍스트 디코더 투영 후 형태 출력
            encoder_hidden_states = self.norm_out_text(encoder_hidden_states, temb_text if temb_text is not None else temb)  # 텍스트 출력 정규화 적용
            print(f"DEBUG: [OmnigesFlow] After norm_out_text: {encoder_hidden_states.shape}")  # 텍스트 정규화 후 형태 출력
            encoder_hidden_states = self.text_output(encoder_hidden_states)  # 텍스트 출력 투영 적용
            print(f"DEBUG: [OmnigesFlow] After text_output: {encoder_hidden_states.shape}")  # 텍스트 출력 투영 후 형태 출력
            model_pred_text = encoder_hidden_states  # 텍스트 예측 결과 저장
            print(f"DEBUG: [OmnigesFlow] model_pred_text shape: {model_pred_text.shape}")  # 텍스트 예측 형태 출력
            
            # ============================================================================
            # 텍스트 디코딩 로직 (텍스트 처리 블록 내부로 이동)
            # ============================================================================
            if decode_text and targets is not None:  # 텍스트 디코딩이 활성화되고 타겟이 제공된 경우
                # 복잡한 텍스트 디코딩 로직 (기존 OmniFlow와 동일)
                if self.text_decoder is not None:  # 텍스트 디코더가 존재하는 경우
                    if detach_logits:  # 로짓 분리가 활성화된 경우
                        with torch.no_grad():  # 그래디언트 계산 비활성화
                            if prompt_embeds_uncond is not None:  # 무조건 프롬프트 임베딩이 제공된 경우
                                raw_text_embeds_input = prompt_embeds_uncond[..., :self.text_out_dim]  # 텍스트 출력 차원만 추출
                            else:  # 무조건 프롬프트 임베딩이 없는 경우
                                raw_text_embeds_input = target_prompt_embeds[..., :self.text_out_dim]  # 타겟 프롬프트 임베딩에서 추출
                            if text_x0:  # 텍스트 x0 예측이 활성화된 경우
                                model_pred_text_clean = model_pred_text  # 예측 텍스트를 그대로 사용
                            else:  # 텍스트 x0 예측이 비활성화된 경우
                                noisy_prompt_embeds = encoder_hidden_states_base[..., :model_pred_text.shape[-1]]  # 노이즈 프롬프트 임베딩 추출
                                model_pred_text_clean = model_pred_text * (-sigma_text) + noisy_prompt_embeds[..., :model_pred_text.shape[-1]]  # 노이즈 제거
                            latents_decode = torch.cat([model_pred_text_clean, raw_text_embeds_input], dim=0).detach()  # 디코딩용 잠재 변수 결합
                    else:  # 로짓 분리가 비활성화된 경우
                        if prompt_embeds_uncond is not None:  # 무조건 프롬프트 임베딩이 제공된 경우
                            raw_text_embeds_input = prompt_embeds_uncond[..., :self.text_out_dim]  # 텍스트 출력 차원만 추출
                        else:  # 무조건 프롬프트 임베딩이 없는 경우
                            raw_text_embeds_input = target_prompt_embeds[..., :self.text_out_dim]  # 타겟 프롬프트 임베딩에서 추출
                        if text_x0:  # 텍스트 x0 예측이 활성화된 경우
                            model_pred_text_clean = model_pred_text  # 예측 텍스트를 그대로 사용
                        else:  # 텍스트 x0 예측이 비활성화된 경우
                            noisy_prompt_embeds = encoder_hidden_states_base[..., :model_pred_text.shape[-1]]  # 노이즈 프롬프트 임베딩 추출
                            model_pred_text_clean = model_pred_text * (-sigma_text) + noisy_prompt_embeds[..., :model_pred_text.shape[-1]]  # 노이즈 제거
                        latents_decode = torch.cat([model_pred_text_clean, raw_text_embeds_input], dim=0)  # 디코딩용 잠재 변수 결합
                    
                    # 텍스트 디코더를 통한 로짓 생성
                    logits_all = self.text_decoder(latents=latents_decode,  # 디코딩용 잠재 변수
                                                  input_ids=targets['input_ids'].repeat(2, 1),  # 입력 ID를 2배로 복제
                                                  attention_mask=None,  # 어텐션 마스크 없음
                                                  labels=None,  # 라벨 없음
                                                  return_dict=False  # 딕셔너리 반환 비활성화
                                                  )[0]  # 첫 번째 출력만 사용
                    logits, logits_labels = logits_all.chunk(2)  # 로짓을 2개로 분할 (조건부/무조건부)
        else:  # 텍스트 처리 또는 출력이 비활성화된 경우
            model_pred_text = None  # 텍스트 예측을 None으로 설정
            print(f"DEBUG: [OmnigesFlow] Text output processing disabled")  # 텍스트 출력 처리 비활성화 디버그 출력

        # ============================================================================
        # 오디오 출력 처리 (기존 OmniFlow와 동일)
        # ============================================================================
        if do_audio:  # 오디오 처리가 활성화된 경우
            print(f"DEBUG: [OmnigesFlow] Audio output processing - Before norm_out_aud: {audio_hidden_states.shape}")  # 오디오 정규화 전 형태 출력
            audio_hidden_states = self.norm_out_aud(audio_hidden_states, temb_audio if temb_audio is not None else temb)  # 오디오 출력 정규화 적용
            print(f"DEBUG: [OmnigesFlow] After norm_out_aud: {audio_hidden_states.shape}")  # 오디오 정규화 후 형태 출력
            audio_hidden_states = self.proj_out_aud(audio_hidden_states)  # 오디오 출력 투영 적용
            print(f"DEBUG: [OmnigesFlow] After proj_out_aud: {audio_hidden_states.shape}")  # 오디오 투영 후 형태 출력
            if not self.use_audio_mae:  # MAE를 사용하지 않는 경우 (표준 패치 방식)
                patch_size_audio = self.audio_patch_size  # 오디오 패치 크기 가져오기
                height_audio = 256 // patch_size_audio  # 오디오 높이 계산
                width_audio = 16 // patch_size_audio  # 오디오 너비 계산
                print(f"DEBUG: [OmnigesFlow] Audio rearrange params - patch_size:{patch_size_audio}, h:{height_audio}, w:{width_audio}")  # 오디오 재배열 파라미터 출력
                audio_hidden_states = rearrange(  # 오디오 텐서 재배열
                    audio_hidden_states,  # 입력 오디오 텐서
                    'n (h w) (hp wp c) -> n c (h hp) (w wp)',  # 재배열 패턴
                    h=height_audio,  # 높이
                    w=width_audio,  # 너비
                    hp=patch_size_audio,  # 패치 높이
                    wp=patch_size_audio,  # 패치 너비
                    c=self.audio_input_dim  # 채널 수
                )
                print(f"DEBUG: [OmnigesFlow] After rearrange: {audio_hidden_states.shape}")  # 재배열 후 형태 출력
            print(f"DEBUG: [OmnigesFlow] Final audio output shape: {audio_hidden_states.shape}")  # 최종 오디오 출력 형태 출력
        else:  # 오디오 처리가 비활성화된 경우
            audio_hidden_states = None  # 오디오 상태를 None으로 설정
            print(f"DEBUG: [OmnigesFlow] Audio processing disabled")  # 오디오 처리 비활성화 디버그 출력
        
        # ============================================================================
        # 최종 출력 딕셔너리 반환
        # ============================================================================
        print(f"DEBUG: [OmnigesFlow] ========== FINAL OUTPUTS ==========")  # 최종 출력 섹션 시작 알림
        print(f"DEBUG: [OmnigesFlow] gesture_output shape: {gesture_output.shape if gesture_output is not None else None}")  # 제스처 출력 형태 출력
        print(f"DEBUG: [OmnigesFlow] model_pred_text shape: {model_pred_text.shape if model_pred_text is not None else None}")  # 텍스트 예측 형태 출력
        print(f"DEBUG: [OmnigesFlow] encoder_hidden_states shape: {encoder_hidden_states.shape if encoder_hidden_states is not None else None}")  # 인코더 상태 형태 출력
        print(f"DEBUG: [OmnigesFlow] logits shape: {logits.shape if logits is not None else None}")  # 로짓 형태 출력
        print(f"DEBUG: [OmnigesFlow] logits_labels shape: {logits_labels.shape if logits_labels is not None else None}")  # 라벨 로짓 형태 출력
        print(f"DEBUG: [OmnigesFlow] audio_hidden_states shape: {audio_hidden_states.shape if audio_hidden_states is not None else None}")  # 오디오 상태 형태 출력
        
        return dict(  # 최종 출력 딕셔너리 반환
            output=gesture_output,  # 제스처 출력 (기존 이미지 출력)
            model_pred_text=model_pred_text,  # 텍스트 예측 결과
            encoder_hidden_states=encoder_hidden_states,  # 인코더 숨겨진 상태
            logits=logits,  # 텍스트 디코딩 로짓
            extra_cond=None,  # 추가 조건 (현재 미사용)
            logits_labels=logits_labels,  # 텍스트 라벨 로짓
            audio_hidden_states=audio_hidden_states,  # 오디오 숨겨진 상태
        )

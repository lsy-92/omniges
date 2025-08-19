'''
OmnigesFlow: 텍스트, 제스처, 오디오를 통합적으로 처리하는 멀티모달 Transformer 모델
- OmniFlow 기반으로 이미지 스트림을 제스처 스트림으로 교체
- 텍스트, 제스처, 오디오 간의 joint attention을 통한 멀티모달 생성
- 각 모달리티별 독립적인 임베딩과 처리 파이프라인 제공
- LoRA, PEFT 등의 효율적인 파인튜닝 기법 지원
'''

# Diffusers 라이브러리의 핵심 모듈들 임포트
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from omniflow.models.attention import JointTransformerBlock  # 기존 Joint Attention 블록 재사용
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from typing import Tuple
import inspect
from einops import rearrange
from functools import partial
from typing import Any, Dict, List, Optional, Union
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed

from transformers.activations import ACT2FN

class NNMLP(nn.Module):
    '''
    간단한 2층 MLP (Multi-Layer Perceptron) 모듈
    - CLIP 임베딩 처리 등에 사용되는 피드포워드 네트워크
    - GELU 활성화 함수를 기본으로 사용
    '''
    def __init__(self, input_size, hidden_size, activation='gelu'):
        super().__init__()
        
        self.linear_1 = nn.Linear(input_size, hidden_size, bias=True)
        self.act = ACT2FN[activation]
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, image_features):
        '''
        MLP 순전파
        Args:
            image_features: 입력 특징 벡터
        Returns:
            변환된 특징 벡터
        '''
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


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
        super().__init__()
        self.seq_length = seq_length
        self.gesture_latent_dim = gesture_latent_dim
        self.embed_dim = embed_dim
        
        # 제스처 잠재 변수를 임베딩 차원으로 변환
        # Support both single part (128) and 4-part concatenated (512) inputs
        self.gesture_proj_128 = nn.Linear(128, embed_dim)  # 128 -> 1536 (single part)
        self.gesture_proj_512 = nn.Linear(512, embed_dim)  # 512 -> 1536 (4 parts concat)
        
        # 시퀀스 위치 임베딩
        self.position_embedding = nn.Parameter(
            torch.randn(1, pos_embed_max_size, embed_dim) * 0.02
        )
        
        # 호환성을 위한 속성들 (기존 PatchEmbed와 동일)
        self.num_patches = seq_length
        self.embed_dim = embed_dim
        
    def forward(self, gesture_latents):
        """
        제스처 latents를 시퀀스 임베딩으로 변환
        
        Args:
            gesture_latents: RVQVAE에서 온 제스처 잠재 변수
        Returns:
            embeddings: (B, T, embed_dim) - 시퀀스 임베딩
        """
        # DEBUG: Print actual input shape
        print(f"DEBUG: Input gesture_latents shape: {gesture_latents.shape}")
        
        # Handle input shape: (B, 512, T, 1) - concatenated 4 parts  
        if len(gesture_latents.shape) == 4:
            B, latent_dim, T, W = gesture_latents.shape
            print(f"DEBUG: [GestureEmbedding] 4D input - B:{B}, latent_dim:{latent_dim}, T:{T}, W:{W}")
            
            if latent_dim == 512 and W == 1:  # (B, 512, T, 1) - concatenated 4 parts following shortcut_rvqvae_trainer.py
                # Convert (B, 512, T, 1) -> (B, T, 512)
                gesture_features = gesture_latents.squeeze(-1).permute(0, 2, 1)  # (B, T, 512)
                print(f"DEBUG: [GestureEmbedding] 512-concat to gesture_features shape: {gesture_features.shape}")
            elif latent_dim == 128 and W == 1:  # (B, 128, T, 1) - single part fallback
                # Convert (B, 128, T, 1) -> (B, T, 128)
                gesture_features = gesture_latents.squeeze(-1).permute(0, 2, 1)  # (B, T, 128)
                print(f"DEBUG: [GestureEmbedding] Single 128 part to gesture_features shape: {gesture_features.shape}")
            elif latent_dim == 128 and W == 4:  # (B, 128, T, 4) format - 4 parts stacked
                # Convert (B, 128, T, 4) -> (B, T, 128*4) to concatenate all parts
                gesture_features = gesture_latents.permute(0, 2, 1, 3)  # (B, T, 128, 4)
                gesture_features = gesture_features.reshape(B, T, latent_dim * W)  # (B, T, 512)
                print(f"DEBUG: [GestureEmbedding] 4-part stack to concat shape: {gesture_features.shape}")
            else:
                raise ValueError(f"Unexpected dimensions: latent_dim={latent_dim}, W={W}")
        else:
            raise ValueError(f"Unexpected input shape: {gesture_latents.shape}")
        
        print(f"DEBUG: After reshape gesture_features shape: {gesture_features.shape}")
        
        # 투영: Choose appropriate projection layer based on input dimension
        input_dim = gesture_features.shape[-1]
        if input_dim == 128:
            embeddings = self.gesture_proj_128(gesture_features)  # (B, T, embed_dim)
        elif input_dim == 512:
            embeddings = self.gesture_proj_512(gesture_features)  # (B, T, embed_dim)
        else:
            raise ValueError(f"Unexpected input dimension: {input_dim}. Expected 128 or 512.")
        
        print(f"DEBUG: Used projection for {input_dim}D input")
        
        # 위치 임베딩 추가
        seq_len = embeddings.shape[1]
        if seq_len <= self.position_embedding.shape[1]:
            pos_emb = self.position_embedding[:, :seq_len, :]
        else:
            # 긴 시퀀스의 경우 보간
            pos_emb = F.interpolate(
                self.position_embedding.transpose(1, 2), 
                size=seq_len, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
            
        embeddings = embeddings + pos_emb
        
        return embeddings


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

    _supports_gradient_checkpointing = True

    @register_to_config
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
        dual_attention_layers: Tuple[int, ...] = (),
        decoder_config: str = '',
        add_audio=True,
        add_clip=False,
        use_audio_mae=False,
        drop_text=False,
        drop_gesture=False,  # 기존 drop_image 대체
        drop_audio=False,
        qk_norm: Optional[str] = 'layer_norm',
    ):
        '''
        Omniges Flow Transformer 모델 초기화
        - 각 모달리티별 임베딩 레이어 설정
        - Transformer 블록들 구성
        - 출력 레이어들 초기화
        '''
        super().__init__()
        default_gesture_output_dim = gesture_latent_dim
        self.add_clip = add_clip
        self.gesture_output_dim = gesture_output_dim if gesture_output_dim is not None else default_gesture_output_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        # 제스처 시퀀스 임베딩 레이어 (기존 PatchEmbed 대체)
        self.gesture_embed = GestureEmbedding(
            seq_length=seq_length,
            gesture_latent_dim=gesture_latent_dim,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,
        )
        
        # 시간 단계와 텍스트 임베딩을 결합하는 레이어 (동일)
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )
        
        if add_audio:  # 오디오 모달리티가 활성화된 경우 (동일)
            # 제스처용 시간 임베딩 (기존 이미지용을 제스처용으로 변경)
            self.time_gesture_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
            )
            self.audio_input_dim = audio_input_dim
            self.use_audio_mae = use_audio_mae
            self.audio_patch_size = 2
            if use_audio_mae:
                self.audio_embedder = nn.Linear(audio_input_dim, self.config.caption_projection_dim)
            else:
                self.audio_embedder = PatchEmbed(
                    height=256,
                    width=16,
                    patch_size=self.audio_patch_size,
                    in_channels=self.audio_input_dim,
                    embed_dim=self.config.caption_projection_dim,
                    pos_embed_max_size=192
                )
            
            # 오디오용 시간 임베딩 (동일)
            self.time_aud_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
            )
         
            # 오디오 출력 정규화 레이어 (동일)
            self.norm_out_aud = AdaLayerNormContinuous(self.config.caption_projection_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
            if use_audio_mae:
                self.proj_out_aud = nn.Linear(self.config.caption_projection_dim, self.config.audio_input_dim)
            else:
                self.proj_out_aud = nn.Linear(self.inner_dim, self.audio_patch_size * self.audio_patch_size * self.audio_input_dim, bias=True)
        
        # 컨텍스트 임베딩 레이어 (동일)
        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.config.caption_projection_dim)
        
        # LLaMA 설정 생성 (동일)
        bert_config = LlamaConfig(1, hidden_size=self.config.joint_attention_dim, num_attention_heads=32, num_hidden_layers=2)
        if self.add_audio:
            self.context_decoder = nn.ModuleDict(dict(
                projection=nn.Linear(self.config.caption_projection_dim, self.config.joint_attention_dim)
            ))
        self.text_out_dim = 1536
        self.text_output = nn.Linear(self.config.joint_attention_dim, self.text_out_dim)
        
        # Transformer 블록들 생성 (drop_image -> drop_gesture로 변경)
        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    context_pre_only=i == num_layers - 1,
                    context_output=i < num_layers or self.add_audio,
                    audio_output=add_audio,
                    delete_img=drop_gesture,     # 기존 drop_image -> drop_gesture
                    delete_aud=drop_audio,
                    delete_text=drop_text,
                    qk_norm=qk_norm,
                    use_dual_attention=True if i in dual_attention_layers else False,
                )
                for i in range(self.config.num_layers)
            ]
        )
        self.add_audio = add_audio
        
        # 제스처 출력 정규화 레이어 (기존 이미지 출력과 동일)
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        
        # 텍스트 출력 정규화 레이어 (동일)
        self.norm_out_text = AdaLayerNormContinuous(self.config.joint_attention_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        
        if self.add_clip:
            self.n_cond_tokens = 8
            self.clip_proj = nn.Sequential(
                NNMLP(self.config.pooled_projection_dim, self.config.caption_projection_dim),
                nn.Linear(self.config.caption_projection_dim, self.config.caption_projection_dim * self.n_cond_tokens)
            )
            
        # 최종 제스처 출력 투영 레이어 (패치 복원 대신 시퀀스 차원으로 투영)
        self.proj_out = nn.Linear(self.inner_dim, gesture_output_dim, bias=True)

        self.gradient_checkpointing = False
        if decoder_config:
            self.text_decoder = None  # 실제 구현 시 수정 필요
        else:
            self.text_decoder = None

    def set_text_decoder(self, model):
        '''
        텍스트 디코더 설정 (동일)
        '''
        self.text_decoder = model
        self.text_out_dim = model.vae_dim
        self.text_output = nn.Linear(self.config.joint_attention_dim, self.text_out_dim)
        
    def set_audio_pooler(self, model):
        '''
        오디오 풀러 설정 (동일)
        '''
        self.audio_pooler = model
        
    def get_decoder(self):
        '''
        텍스트 디코더 반환 (동일)
        '''
        return self.text_decoder

    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        피드포워드 청킹을 사용하는 attention processor 설정 (동일)
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        """
        모델에서 사용되는 모든 attention processor 반환 (동일)
        """
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        """
        Attention 계산에 사용할 attention processor 설정 (동일)
        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def fuse_qkv_projections(self):
        """QKV 투영 행렬 융합 활성화 (동일)"""
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

    def unfuse_qkv_projections(self):
        """융합된 QKV 투영 비활성화 (동일)"""
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        '''그래디언트 체크포인팅 설정 (동일)'''
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def fuse_lora(self, lora_scale=1.0, safe_fusing=False, adapter_names=None):
        '''LoRA 어댑터 융합 (동일)'''
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `fuse_lora()`.")

        self.lora_scale = lora_scale
        self._safe_fusing = safe_fusing
        self.apply(partial(self._fuse_lora_apply, adapter_names=adapter_names))

    def _fuse_lora_apply(self, module, adapter_names=None):
        '''개별 모듈에 LoRA 융합 적용 (동일)'''
        from peft.tuners.tuners_utils import BaseTunerLayer

        merge_kwargs = {"safe_merge": self._safe_fusing}

        if isinstance(module, BaseTunerLayer):
            if self.lora_scale != 1.0:
                module.scale_layer(self.lora_scale)

            supported_merge_kwargs = list(inspect.signature(module.merge).parameters)
            if "adapter_names" in supported_merge_kwargs:
                merge_kwargs["adapter_names"] = adapter_names
            elif "adapter_names" not in supported_merge_kwargs and adapter_names is not None:
                raise ValueError(
                    "The `adapter_names` argument is not supported with your PEFT version. Please upgrade"
                    " to the latest version of PEFT. `pip install -U peft`"
                )

            module.merge(**merge_kwargs)

    def unfuse_lora(self):
        '''LoRA 어댑터 융합 해제 (동일)'''
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `unfuse_lora()`.")
        self.apply(self._unfuse_lora_apply)

    def _unfuse_lora_apply(self, module):
        '''개별 모듈에 LoRA 융합 해제 적용 (동일)'''
        from peft.tuners.tuners_utils import BaseTunerLayer

        if isinstance(module, BaseTunerLayer):
            module.unmerge()

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
        if kkwargs is not None:
            assert forward_function is not None
            return forward_function(transformer=self, **kkwargs)
        
        encoder_hidden_states_base = encoder_hidden_states.clone() if encoder_hidden_states is not None else None
        hidden_states_base = hidden_states
        
        # 각 모달리티 처리 여부 결정
        do_gesture = not drop_gesture  # 기존 do_image -> do_gesture
        do_audio = (not drop_audio) and (self.add_audio)
        do_text = (not drop_text)
        
        if do_gesture:  # 제스처 처리가 활성화된 경우
            seq_length = hidden_states.shape[-2]  # 제스처 시퀀스 길이 추출 (기존 height, width 대신)
            print(f"DEBUG: [OmnigesFlow] Before gesture_embed - hidden_states shape: {hidden_states.shape}")
            hidden_states = self.gesture_embed(hidden_states)  # 제스처 임베딩 적용
            print(f"DEBUG: [OmnigesFlow] After gesture_embed - hidden_states shape: {hidden_states.shape}")
            temb = self.time_text_embed(timestep, pooled_projections)  # 제스처용 시간 임베딩 생성
        else:  # 제스처 처리가 비활성화된 경우
            hidden_states = None
            temb = 0
           
        if do_audio:  # 오디오 처리 수정
            print(f"DEBUG: [OmnigesFlow] Audio processing enabled")
            if audio_hidden_states is None:
                print(f"DEBUG: [OmnigesFlow] Audio hidden states is None - creating dummy")
                if self.use_audio_mae:
                    audio_hidden_states = torch.zeros(encoder_hidden_states.shape[0], 8, self.audio_input_dim).to(encoder_hidden_states)
                    print(f"DEBUG: [OmnigesFlow] Dummy audio MAE shape: {audio_hidden_states.shape}")
                else:
                    audio_hidden_states = torch.zeros(encoder_hidden_states.shape[0], self.audio_input_dim, 256, 16).to(encoder_hidden_states)  # 채널 순서 수정
                    print(f"DEBUG: [OmnigesFlow] Dummy audio standard shape: {audio_hidden_states.shape}")
                timestep_audio = timestep_text * 0 if timestep_text is not None else torch.zeros_like(timestep)
            else:
                print(f"DEBUG: [OmnigesFlow] Input audio_hidden_states shape: {audio_hidden_states.shape}")
            
            temb_audio = self.time_aud_embed(timestep_audio, pooled_projections)
            print(f"DEBUG: [OmnigesFlow] Audio time embedding shape: {temb_audio.shape}")
            
            # 오디오 데이터 형태 확인 및 적절히 처리
            if audio_hidden_states.dim() == 3:
                # (B, T, D) 형태인 경우 -> (B, C, H, W) 형태로 변환
                B, T, D = audio_hidden_states.shape
                if self.use_audio_mae:
                    # MAE 방식: 직접 linear projection
                    audio_hidden_states = self.audio_embedder(audio_hidden_states)
                else:
                    # Patch 방식: reshape 후 처리
                    audio_hidden_states = audio_hidden_states.view(B, self.audio_input_dim, 256, 16)
                    audio_hidden_states = self.audio_embedder(audio_hidden_states)
            else:
                # 이미 올바른 형태인 경우
                audio_hidden_states = self.audio_embedder(audio_hidden_states)
                
            if not split_cond:
                temb = temb + temb_audio
                temb_audio = None
        else:
            audio_hidden_states = None
            temb_audio = None
            
        if do_text:  # 텍스트 처리 (동일)
            print(f"DEBUG: [OmnigesFlow] Text processing enabled")
            print(f"DEBUG: [OmnigesFlow] Input encoder_hidden_states shape: {encoder_hidden_states.shape}")
            if use_text_output:
                temb_text = self.time_gesture_embed(timestep_text, pooled_projections)  # 기존 time_image_embed -> time_gesture_embed
                print(f"DEBUG: [OmnigesFlow] Text time embedding shape: {temb_text.shape}")
            encoder_hidden_states = self.context_embedder(encoder_hidden_states)
            print(f"DEBUG: [OmnigesFlow] After context_embedder - encoder_hidden_states shape: {encoder_hidden_states.shape}")
            if use_text_output:
                if not split_cond:
                    temb = temb + temb_text
                    temb_text = None
                    print(f"DEBUG: [OmnigesFlow] Combined temb shape: {temb.shape}")
            else:
                temb_text = None
        else:
            encoder_hidden_states = None
            temb_text = None
    
        assert not self.add_clip  # CLIP 사용이 비활성화되어 있음을 확인

        # Transformer 블록들을 순차적으로 처리 (동일)
        print(f"DEBUG: [OmnigesFlow] ========== TRANSFORMER BLOCKS ==========")
        print(f"DEBUG: [OmnigesFlow] Total transformer blocks: {len(self.transformer_blocks)}")
        print(f"DEBUG: [OmnigesFlow] Before blocks - hidden_states shape: {hidden_states.shape if hidden_states is not None else None}")
        print(f"DEBUG: [OmnigesFlow] Before blocks - encoder_hidden_states shape: {encoder_hidden_states.shape if encoder_hidden_states is not None else None}")
        print(f"DEBUG: [OmnigesFlow] Before blocks - audio_hidden_states shape: {audio_hidden_states.shape if audio_hidden_states is not None else None}")
        
        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward

                ckpt_kwargs = dict()
                if self.add_audio:
                    encoder_hidden_states, hidden_states, audio_hidden_states = deepspeed.checkpointing.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        audio_hidden_states,
                        temb_text,
                        temb_audio,
                        **ckpt_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = deepspeed.checkpointing.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        temb_text,
                        **ckpt_kwargs,
                    )

            else:  # 일반적인 순전파
                if self.add_audio:
                    encoder_hidden_states, hidden_states, audio_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        audio_hidden_states=audio_hidden_states,
                        temb=temb,
                        temb_text=temb_text,
                        temb_audio=temb_audio,
                    )
                    # Debug output for first few blocks only to avoid spam
                    if index_block < 3 or index_block == len(self.transformer_blocks) - 1:
                        print(f"DEBUG: [OmnigesFlow] Block {index_block}/{len(self.transformer_blocks)-1} output:")
                        print(f"DEBUG:   hidden_states shape: {hidden_states.shape if hidden_states is not None else None}")
                        print(f"DEBUG:   encoder_hidden_states shape: {encoder_hidden_states.shape if encoder_hidden_states is not None else None}")
                        print(f"DEBUG:   audio_hidden_states shape: {audio_hidden_states.shape if audio_hidden_states is not None else None}")
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        temb_text=temb_text
                    )
                    # Debug output for first few blocks only
                    if index_block < 3 or index_block == len(self.transformer_blocks) - 1:
                        print(f"DEBUG: [OmnigesFlow] Block {index_block}/{len(self.transformer_blocks)-1} output (no audio):")
                        print(f"DEBUG:   hidden_states shape: {hidden_states.shape if hidden_states is not None else None}")
                        print(f"DEBUG:   encoder_hidden_states shape: {encoder_hidden_states.shape if encoder_hidden_states is not None else None}")

            # ControlNet 잔차 처리 (동일)
            assert block_controlnet_hidden_states is None

        # 제스처 출력 처리 (기존 이미지 출력 로직 수정)
        if do_gesture:
            print(f"DEBUG: [OmnigesFlow] Before norm_out - hidden_states shape: {hidden_states.shape}")
            hidden_states = self.norm_out(hidden_states, temb)  # 출력 정규화 적용
            print(f"DEBUG: [OmnigesFlow] After norm_out - hidden_states shape: {hidden_states.shape}")
            hidden_states = self.proj_out(hidden_states)  # 최종 투영 레이어 적용
            print(f"DEBUG: [OmnigesFlow] After proj_out - hidden_states shape: {hidden_states.shape}")

            # (B, T, 512) -> (B, 512, T, 1) 형태로 변환하여 input과 동일한 형태로 만들기
            # Following shortcut_rvqvae_trainer.py pattern where latents are concatenated
            B, T, gesture_output_dim = hidden_states.shape
            print(f"DEBUG: [OmnigesFlow] Output dimensions - B:{B}, T:{T}, gesture_output_dim:{gesture_output_dim}")
            if gesture_output_dim == 512:
                # Keep 512 concatenated format: (B, T, 512) -> (B, 512, T, 1)
                gesture_output = hidden_states.permute(0, 2, 1).unsqueeze(-1)  # (B, 512, T, 1)
                print(f"DEBUG: [OmnigesFlow] 512-dim output reshape: {gesture_output.shape}")
            elif gesture_output_dim == 128:
                # Single part fallback: (B, T, 128) -> (B, 128, T, 1)
                gesture_output = hidden_states.permute(0, 2, 1).unsqueeze(-1)  # (B, 128, T, 1)
                print(f"DEBUG: [OmnigesFlow] 128-dim output reshape: {gesture_output.shape}")
            else:
                raise ValueError(f"Unexpected gesture output dimension: {gesture_output_dim}")
                
            print(f"DEBUG: [OmnigesFlow] Final gesture_output shape: {gesture_output.shape}")
        else:
            gesture_output = None

        # 텍스트 출력 처리 (동일)
        logits = None
        logits_labels = None
        
        if do_text and use_text_output:
            print(f"DEBUG: [OmnigesFlow] Text output processing enabled")
            print(f"DEBUG: [OmnigesFlow] Before context_decoder projection: {encoder_hidden_states.shape}")
            encoder_hidden_states = self.context_decoder['projection'](encoder_hidden_states)
            print(f"DEBUG: [OmnigesFlow] After context_decoder projection: {encoder_hidden_states.shape}")
            encoder_hidden_states = self.norm_out_text(encoder_hidden_states, temb_text if temb_text is not None else temb)
            print(f"DEBUG: [OmnigesFlow] After norm_out_text: {encoder_hidden_states.shape}")
            encoder_hidden_states = self.text_output(encoder_hidden_states)
            print(f"DEBUG: [OmnigesFlow] After text_output: {encoder_hidden_states.shape}")
            model_pred_text = encoder_hidden_states
            print(f"DEBUG: [OmnigesFlow] model_pred_text shape: {model_pred_text.shape}")
            
            # Text decoding logic (moved inside the text processing block)
            if decode_text and targets is not None:
                # 복잡한 텍스트 디코딩 로직 (동일)
                if self.text_decoder is not None:
                    if detach_logits:
                        with torch.no_grad():
                            if prompt_embeds_uncond is not None:
                                raw_text_embeds_input = prompt_embeds_uncond[..., :self.text_out_dim]
                            else:
                                raw_text_embeds_input = target_prompt_embeds[..., :self.text_out_dim]
                            if text_x0:
                                model_pred_text_clean = model_pred_text
                            else:
                                noisy_prompt_embeds = encoder_hidden_states_base[..., :model_pred_text.shape[-1]]
                                model_pred_text_clean = model_pred_text * (-sigma_text) + noisy_prompt_embeds[..., :model_pred_text.shape[-1]]
                            latents_decode = torch.cat([model_pred_text_clean, raw_text_embeds_input], dim=0).detach()
                    else:
                        if prompt_embeds_uncond is not None:
                            raw_text_embeds_input = prompt_embeds_uncond[..., :self.text_out_dim]
                        else:
                            raw_text_embeds_input = target_prompt_embeds[..., :self.text_out_dim]
                        if text_x0:
                            model_pred_text_clean = model_pred_text
                        else:
                            noisy_prompt_embeds = encoder_hidden_states_base[..., :model_pred_text.shape[-1]]
                            model_pred_text_clean = model_pred_text * (-sigma_text) + noisy_prompt_embeds[..., :model_pred_text.shape[-1]]
                        latents_decode = torch.cat([model_pred_text_clean, raw_text_embeds_input], dim=0)
                    
                    logits_all = self.text_decoder(latents=latents_decode,
                                                  input_ids=targets['input_ids'].repeat(2, 1),
                                                  attention_mask=None,
                                                  labels=None,
                                                  return_dict=False
                                                  )[0]
                    logits, logits_labels = logits_all.chunk(2)
        else:
            model_pred_text = None
            print(f"DEBUG: [OmnigesFlow] Text output processing disabled")

        # 오디오 출력 처리 (동일)
        if do_audio:
            print(f"DEBUG: [OmnigesFlow] Audio output processing - Before norm_out_aud: {audio_hidden_states.shape}")
            audio_hidden_states = self.norm_out_aud(audio_hidden_states, temb_audio if temb_audio is not None else temb)
            print(f"DEBUG: [OmnigesFlow] After norm_out_aud: {audio_hidden_states.shape}")
            audio_hidden_states = self.proj_out_aud(audio_hidden_states)
            print(f"DEBUG: [OmnigesFlow] After proj_out_aud: {audio_hidden_states.shape}")
            if not self.use_audio_mae:
                patch_size_audio = self.audio_patch_size
                height_audio = 256 // patch_size_audio
                width_audio = 16 // patch_size_audio
                print(f"DEBUG: [OmnigesFlow] Audio rearrange params - patch_size:{patch_size_audio}, h:{height_audio}, w:{width_audio}")
                audio_hidden_states = rearrange(
                    audio_hidden_states,
                    'n (h w) (hp wp c) -> n c (h hp) (w wp)',
                    h=height_audio,
                    w=width_audio,
                    hp=patch_size_audio,
                    wp=patch_size_audio,
                    c=self.audio_input_dim
                )
                print(f"DEBUG: [OmnigesFlow] After rearrange: {audio_hidden_states.shape}")
            print(f"DEBUG: [OmnigesFlow] Final audio output shape: {audio_hidden_states.shape}")
        else:
            audio_hidden_states = None
            print(f"DEBUG: [OmnigesFlow] Audio processing disabled")
        
        # 최종 출력 딕셔너리 반환
        print(f"DEBUG: [OmnigesFlow] ========== FINAL OUTPUTS ==========")
        print(f"DEBUG: [OmnigesFlow] gesture_output shape: {gesture_output.shape if gesture_output is not None else None}")
        print(f"DEBUG: [OmnigesFlow] model_pred_text shape: {model_pred_text.shape if model_pred_text is not None else None}")
        print(f"DEBUG: [OmnigesFlow] encoder_hidden_states shape: {encoder_hidden_states.shape if encoder_hidden_states is not None else None}")
        print(f"DEBUG: [OmnigesFlow] logits shape: {logits.shape if logits is not None else None}")
        print(f"DEBUG: [OmnigesFlow] logits_labels shape: {logits_labels.shape if logits_labels is not None else None}")
        print(f"DEBUG: [OmnigesFlow] audio_hidden_states shape: {audio_hidden_states.shape if audio_hidden_states is not None else None}")
        
        return dict(
            output=gesture_output,  # 제스처 출력 (기존 이미지 출력)
            model_pred_text=model_pred_text,
            encoder_hidden_states=encoder_hidden_states,
            logits=logits,
            extra_cond=None,
            logits_labels=logits_labels,
            audio_hidden_states=audio_hidden_states,
        )

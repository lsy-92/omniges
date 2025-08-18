'''
OmniFlow: 텍스트, 이미지, 오디오를 통합적으로 처리하는 멀티모달 Transformer 모델
- Stable Diffusion 3 기반의 Transformer 아키텍처를 확장
- 텍스트, 이미지, 오디오 간의 joint attention을 통한 멀티모달 생성
- 각 모달리티별 독립적인 임베딩과 처리 파이프라인 제공
- LoRA, PEFT 등의 효율적인 파인튜닝 기법 지원
'''

# Diffusers 라이브러리의 핵심 모듈들 임포트
from diffusers import  ModelMixin,ConfigMixin#,PeftAdapterMixin,FromOriginalModelMixin
from diffusers.configuration_utils  import ConfigMixin, register_to_config  # 모델 설정 관리
from diffusers.configuration_utils import ConfigMixin, register_to_config  # 중복 임포트
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin  # 모델 로딩 및 PEFT 지원
from omniflow.models.attention import JointTransformerBlock  # 커스텀 Joint Attention 블록
from diffusers.models.attention_processor import Attention, AttentionProcessor  # Attention 처리기
from diffusers.models.modeling_utils import ModelMixin  # 모델 기본 클래스
from diffusers.models.normalization import AdaLayerNormContinuous  # Adaptive Layer Normalization
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers  # 유틸리티
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed  # 임베딩 레이어들
from diffusers.models.modeling_outputs import Transformer2DModelOutput  # 출력 형식
from typing import Tuple  # 타입 힌팅
import inspect  # 함수 시그니처 검사
from einops import rearrange  # 텐서 차원 재배열
from functools import partial  # 부분 함수 적용
from typing import Any, Dict, List, Optional, Union  # 타입 힌팅
from transformers import BertConfig  # BERT 설정
from transformers.models.bert.modeling_bert import BertEncoder  # BERT 인코더
from transformers.models.llama.modeling_llama import LlamaConfig,LlamaModel  # LLaMA 모델
import torch  # PyTorch 메인 모듈
import torch.nn as nn  # 신경망 모듈
import deepspeed  # 분산 학습 및 메모리 최적화

from transformers.activations import ACT2FN  # 활성화 함수 매핑    
class NNMLP(nn.Module):
    '''
    간단한 2층 MLP (Multi-Layer Perceptron) 모듈
    - CLIP 임베딩 처리 등에 사용되는 피드포워드 네트워크
    - GELU 활성화 함수를 기본으로 사용
    '''
    def __init__(self, input_size,hidden_size,activation='gelu'):
        super().__init__()
        
        self.linear_1 = nn.Linear(input_size, hidden_size, bias=True)  # 첫 번째 선형 변환
        self.act = ACT2FN[activation]  # 활성화 함수 (기본: GELU)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)  # 두 번째 선형 변환

    def forward(self, image_features):
        '''
        MLP 순전파
        Args:
            image_features: 입력 특징 벡터
        Returns:
            변환된 특징 벡터
        '''
        hidden_states = self.linear_1(image_features)  # 첫 번째 선형 변환 적용
        hidden_states = self.act(hidden_states)  # 활성화 함수 적용
        hidden_states = self.linear_2(hidden_states)  # 두 번째 선형 변환 적용
        return hidden_states
    
class OmniFlowTransformerModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    OmniFlow 멀티모달 Transformer 모델
    - Stable Diffusion 3 기반으로 텍스트, 이미지, 오디오를 통합 처리
    - Joint attention을 통한 모달리티 간 상호작용
    - 각 모달리티별 독립적인 임베딩 및 처리 파이프라인

    Parameters:
        sample_size (`int`): 잠재 이미지의 크기 (위치 임베딩 학습에 사용)
        patch_size (`int`): 입력을 작은 패치로 나누는 크기
        in_channels (`int`): 입력 채널 수 (기본값: 16)
        num_layers (`int`): Transformer 블록 레이어 수 (기본값: 18)
        attention_head_dim (`int`): 각 attention head의 차원 (기본값: 64)
        num_attention_heads (`int`): Multi-head attention의 head 수 (기본값: 18)
        joint_attention_dim (`int`): Joint attention 차원 (기본값: 4096)
        caption_projection_dim (`int`): 텍스트 임베딩 투영 차원 (기본값: 1152)
        pooled_projection_dim (`int`): Pooled projection 차원 (기본값: 2048)
        audio_input_dim (`int`): 오디오 입력 차원 (기본값: 8)
        out_channels (`int`): 출력 채널 수 (기본값: 16)
        pos_embed_max_size (`int`): 위치 임베딩 최대 크기 (기본값: 96)
        dual_attention_layers: 듀얼 어텐션을 사용할 레이어 인덱스
        decoder_config: 텍스트 디코더 설정
        add_audio: 오디오 모달리티 사용 여부
        add_clip: CLIP 임베딩 사용 여부
        use_audio_mae: 오디오 MAE 사용 여부
        drop_text/drop_image/drop_audio: 각 모달리티 드롭아웃 여부
        qk_norm: Query-Key 정규화 방법
    """

    _supports_gradient_checkpointing = True  # 그래디언트 체크포인팅 지원

    @register_to_config  # 설정을 자동으로 등록하는 데코레이터
    def __init__(
        self,
        sample_size: int = 128,  # 샘플 이미지 크기
        patch_size: int = 2,  # 패치 크기
        in_channels: int = 16,  # 입력 채널 수
        num_layers: int = 18,  # Transformer 레이어 수
        attention_head_dim: int = 64,  # Attention head 차원
        num_attention_heads: int = 18,  # Attention head 수
        joint_attention_dim: int = 4096,  # Joint attention 차원
        caption_projection_dim: int = 1152,  # 캡션 투영 차원
        pooled_projection_dim: int = 2048,  # Pooled 투영 차원
        audio_input_dim: int = 8,  # 오디오 입력 차원
        out_channels: int = 16,  # 출력 채널 수
        pos_embed_max_size: int = 96,  # 위치 임베딩 최대 크기
        dual_attention_layers: Tuple[  # 듀얼 어텐션 레이어
            int, ...
        ] = (),  # SD3.0의 경우 빈 튜플
        decoder_config:str = '',  # 디코더 설정 문자열
        add_audio=True,  # 오디오 모달리티 추가 여부
        add_clip=False,  # CLIP 임베딩 추가 여부
        use_audio_mae=False,  # 오디오 MAE 사용 여부
        drop_text=False,  # 텍스트 드롭 여부
        drop_image=False,  # 이미지 드롭 여부
        drop_audio=False,  # 오디오 드롭 여부
        qk_norm: Optional[str] = 'layer_norm',  # Query-Key 정규화 방법
    ):
        '''
        OmniFlow Transformer 모델 초기화
        - 각 모달리티별 임베딩 레이어 설정
        - Transformer 블록들 구성
        - 출력 레이어들 초기화
        '''
        super().__init__()  # 부모 클래스 초기화
        default_out_channels = in_channels  # 기본 출력 채널 수 설정
        self.add_clip = add_clip  # CLIP 사용 여부 저장
        self.out_channels = out_channels if out_channels is not None else default_out_channels  # 출력 채널 수 결정
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim  # 내부 차원 계산

        # 이미지 패치 임베딩 레이어 (이미지를 패치로 나누고 임베딩)
        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,  # 이미지 높이
            width=self.config.sample_size,  # 이미지 너비
            patch_size=self.config.patch_size,  # 패치 크기
            in_channels=self.config.in_channels,  # 입력 채널 수
            embed_dim=self.inner_dim,  # 임베딩 차원
            pos_embed_max_size=pos_embed_max_size,  # 위치 임베딩 최대 크기 (하드코딩)
        )
        # 시간 단계와 텍스트 임베딩을 결합하는 레이어
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )
        
        if add_audio:  # 오디오 모달리티가 활성화된 경우
            # 오디오 입력 형태: [batch, 8, 256, 16]
            
            # 이미지용 시간 임베딩 (오디오 사용 시 별도 처리)
            self.time_image_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
            )
            self.audio_input_dim = audio_input_dim  # 오디오 입력 차원 저장
            self.use_audio_mae = use_audio_mae  # 오디오 MAE 사용 여부 저장
            self.audio_patch_size = 2  # 오디오 패치 크기 설정
            if use_audio_mae:  # MAE 방식 사용 시
                self.audio_embedder = nn.Linear(audio_input_dim, self.config.caption_projection_dim)  # 선형 임베딩
            else:  # 패치 임베딩 방식 사용 시
                self.audio_embedder = PatchEmbed(
                    height=256,  # 오디오 높이 (스펙트로그램)
                    width=16,  # 오디오 너비
                    patch_size=self.audio_patch_size,  # 오디오 패치 크기
                    in_channels=self.audio_input_dim,  # 오디오 입력 채널
                    embed_dim=self.config.caption_projection_dim,  # 임베딩 차원
                    pos_embed_max_size=192  # 위치 임베딩 최대 크기 (하드코딩)
                )
            
            # 오디오용 시간 임베딩
            self.time_aud_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
            )
         
            # 오디오 출력 정규화 레이어
            self.norm_out_aud = AdaLayerNormContinuous(self.config.caption_projection_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
            if use_audio_mae:  # MAE 방식의 경우
                self.proj_out_aud = nn.Linear(self.config.caption_projection_dim, self.config.audio_input_dim)  # 선형 출력
            else:  # 패치 방식의 경우
                self.proj_out_aud = nn.Linear(self.inner_dim, self.audio_patch_size * self.audio_patch_size * self.audio_input_dim, bias=True)  # 패치 복원용 출력
        
        # 컨텍스트 임베딩 레이어 (텍스트 특징을 캡션 차원으로 변환)
        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.config.caption_projection_dim)
        # LLaMA 설정 생성 (사용되지 않음)
        bert_config = LlamaConfig(1,hidden_size=self.config.joint_attention_dim,num_attention_heads=32,num_hidden_layers=2)
        if self.add_audio:  # 오디오가 활성화된 경우
            self.context_decoder = nn.ModuleDict(dict(
                # transformer=LlamaModel(bert_config),  # 주석 처리됨
                projection=nn.Linear(self.config.caption_projection_dim,self.config.joint_attention_dim)  # 투영 레이어
            ))
        self.text_out_dim = 1536  # 텍스트 출력 차원 설정
        self.text_output = nn.Linear(self.config.joint_attention_dim,self.text_out_dim)  # 텍스트 출력 레이어
        
        # Transformer 블록들 생성 (Joint Attention 블록들의 리스트)
        # attention_head_dim은 믹싱을 고려하여 두 배로 설정됨
        # 실제 체크포인트에서 조정이 필요할 수 있음
        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.inner_dim,  # 내부 차원
                    num_attention_heads=self.config.num_attention_heads,  # Attention head 수
                    attention_head_dim=self.config.attention_head_dim,  # Attention head 차원
                    context_pre_only= i == num_layers - 1,  # 마지막 레이어는 context만 처리
                    context_output=i <num_layers or self.add_audio,  # 컨텍스트 출력 여부
                    audio_output=add_audio,  # 오디오 출력 여부
                    delete_img=drop_image,  # 이미지 드롭 여부
                    delete_aud=drop_audio,  # 오디오 드롭 여부
                    delete_text=drop_text,  # 텍스트 드롭 여부
                    qk_norm=qk_norm,  # Query-Key 정규화
                    use_dual_attention=True if i in dual_attention_layers else False,  # 듀얼 어텐션 사용 여부
                )
                for i in range(self.config.num_layers)  # 설정된 레이어 수만큼 생성
            ]
        )
        self.add_audio = add_audio  # 오디오 사용 여부 저장
        # 이미지 출력 정규화 레이어
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        # 텍스트 출력 정규화 레이어
        self.norm_out_text = AdaLayerNormContinuous(self.joint_attention_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        if self.add_clip:  # CLIP 임베딩 사용 시
            self.n_cond_tokens = 8  # 조건부 토큰 수
            self.clip_proj = nn.Sequential(  # CLIP 투영 레이어
                NNMLP(self.config.pooled_projection_dim, self.config.caption_projection_dim),  # MLP 변환
                nn.Linear(self.config.caption_projection_dim,self.config.caption_projection_dim*self.n_cond_tokens)  # 토큰 확장
            )
        # 최종 이미지 출력 투영 레이어 (패치를 원본 이미지로 복원)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False  # 그래디언트 체크포인팅 비활성화
        if decoder_config:  # 디코더 설정이 있는 경우
            # TODO: build_from_config 함수가 정의되지 않음 - 실제 구현 시 수정 필요
            # self.text_decoder = build_from_config(decoder_config)  # 설정에서 텍스트 디코더 생성
            self.text_decoder = None  # 임시로 None 설정
        else:
            self.text_decoder = None  # 디코더 없음
    #     self.apply(self._init_weights)
            
    # def _init_weights(self, module):
    #     std = 0.02
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()

    def set_text_decoder(self,model):
        '''
        텍스트 디코더 설정
        - 외부에서 학습된 텍스트 VAE 디코더를 설정
        - 텍스트 출력 차원을 디코더의 VAE 차원에 맞춤
        '''
        self.text_decoder = model  # 텍스트 디코더 모델 설정
        self.text_out_dim = model.vae_dim  # VAE 차원으로 출력 차원 업데이트 (1536)
        self.text_output = nn.Linear(self.config.joint_attention_dim,self.text_out_dim)  # 출력 레이어 재구성
        
    def set_audio_pooler(self,model):
        '''
        오디오 풀러 설정
        - 오디오 특징을 풀링하는 모델 설정
        '''
        self.audio_pooler = model  # 오디오 풀러 모델 설정
        
    def get_decoder(self):
        '''
        텍스트 디코더 반환
        Returns:
            현재 설정된 텍스트 디코더 모델
        '''
        return self.text_decoder  # 설정된 텍스트 디코더 반환
    # diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking에서 복사됨
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        피드포워드 청킹을 사용하는 attention processor 설정
        - 메모리 효율성을 위해 피드포워드 레이어를 청크 단위로 처리
        - Reformer의 청킹 기법 사용

        Parameters:
            chunk_size (`int`, *optional*):
                피드포워드 레이어의 청크 크기. 지정하지 않으면 dim 차원의 각 텐서를 개별적으로 처리
            dim (`int`, *optional*, defaults to `0`):
                청킹을 적용할 차원. dim=0 (배치) 또는 dim=1 (시퀀스 길이) 중 선택
        """
        if dim not in [0, 1]:  # 차원 값 검증
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        chunk_size = chunk_size or 1  # 기본 청크 크기는 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            '''재귀적으로 모든 하위 모듈에 청킹 설정 적용'''
            if hasattr(module, "set_chunk_feed_forward"):  # 청킹 설정 메서드가 있는 경우
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)  # 청킹 설정

            for child in module.children():  # 모든 하위 모듈에 대해
                fn_recursive_feed_forward(child, chunk_size, dim)  # 재귀적으로 적용

        for module in self.children():  # 모든 하위 모듈에 대해
            fn_recursive_feed_forward(module, chunk_size, dim)  # 재귀적으로 청킹 설정

    @property
    # diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors에서 복사됨
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        모델에서 사용되는 모든 attention processor 반환
        - 모델의 모든 attention 레이어에서 사용되는 processor들을 딕셔너리로 반환
        - 가중치 이름으로 인덱싱됨
        
        Returns:
            attention processor들의 딕셔너리 (이름: processor)
        """
        processors = {}  # processor들을 저장할 딕셔너리

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            '''재귀적으로 모든 하위 모듈에서 processor 수집'''
            if hasattr(module, "get_processor"):  # processor를 가져올 수 있는 모듈인 경우
                processors[f"{name}.processor"] = module.get_processor()  # processor 추가

            for sub_name, child in module.named_children():  # 모든 하위 모듈에 대해
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)  # 재귀적으로 처리

            return processors

        for name, module in self.named_children():  # 모든 하위 모듈에 대해
            fn_recursive_add_processors(name, module, processors)  # 재귀적으로 processor 수집

        return processors  # 수집된 processor들 반환

    # diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor에서 복사됨
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Attention 계산에 사용할 attention processor 설정
        - 모든 Attention 레이어에 사용할 processor 설정
        - 단일 processor 또는 레이어별 processor 딕셔너리 지원

        Parameters:
            processor: 단일 AttentionProcessor 또는 processor 딕셔너리
                - 딕셔너리인 경우 각 키는 해당 attention processor의 경로를 정의해야 함
                - 학습 가능한 attention processor 설정 시 딕셔너리 사용 권장
        """
        count = len(self.attn_processors.keys())  # 현재 attention processor 수 계산

        if isinstance(processor, dict) and len(processor) != count:  # 딕셔너리 processor 수 검증
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            '''재귀적으로 모든 하위 모듈에 attention processor 설정'''
            if hasattr(module, "set_processor"):  # processor 설정 메서드가 있는 모듈인 경우
                if not isinstance(processor, dict):  # 단일 processor인 경우
                    module.set_processor(processor)  # 동일한 processor 설정
                else:  # 딕셔너리 processor인 경우
                    module.set_processor(processor.pop(f"{name}.processor"))  # 해당 경로의 processor 설정

            for sub_name, child in module.named_children():  # 모든 하위 모듈에 대해
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)  # 재귀적으로 설정

        for name, module in self.named_children():  # 모든 하위 모듈에 대해
            fn_recursive_attn_processor(name, module, processor)  # 재귀적으로 processor 설정

    # diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections에서 복사됨
    def fuse_qkv_projections(self):
        """
        QKV 투영 행렬 융합 활성화
        - Self-attention 모듈: 모든 투영 행렬 (query, key, value) 융합
        - Cross-attention 모듈: key와 value 투영 행렬 융합
        - 메모리 효율성과 계산 속도 향상을 위한 최적화

        주의: 이 API는 실험적 기능입니다.
        """
        self.original_attn_processors = None  # 원본 processor 초기화

        for _, attn_processor in self.attn_processors.items():  # 모든 attention processor 확인
            if "Added" in str(attn_processor.__class__.__name__):  # Added KV projection이 있는 경우
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors  # 원본 processor 백업

        for module in self.modules():  # 모든 모듈에 대해
            if isinstance(module, Attention):  # Attention 모듈인 경우
                module.fuse_projections(fuse=True)  # 투영 행렬 융합 활성화

    # diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections에서 복사됨
    def unfuse_qkv_projections(self):
        """
        융합된 QKV 투영 비활성화
        - 융합이 활성화된 경우 원본 상태로 복원
        
        주의: 이 API는 실험적 기능입니다.
        """
        if self.original_attn_processors is not None:  # 원본 processor가 백업되어 있는 경우
            self.set_attn_processor(self.original_attn_processors)  # 원본 processor로 복원

    def _set_gradient_checkpointing(self, module, value=False):
        '''
        그래디언트 체크포인팅 설정
        - 메모리 효율성을 위해 중간 활성화를 저장하지 않고 재계산
        '''
        if hasattr(module, "gradient_checkpointing"):  # 체크포인팅 속성이 있는 모듈인 경우
            module.gradient_checkpointing = value  # 체크포인팅 설정

    def fuse_lora(self, lora_scale=1.0, safe_fusing=False, adapter_names=None):
        '''
        LoRA 어댑터 융합
        - LoRA 가중치를 기본 모델 가중치에 병합하여 추론 속도 향상
        - PEFT 백엔드가 필요함
        
        Args:
            lora_scale: LoRA 스케일 팩터
            safe_fusing: 안전한 융합 모드 사용 여부
            adapter_names: 융합할 어댑터 이름들
        '''
        if not USE_PEFT_BACKEND:  # PEFT 백엔드 확인
            raise ValueError("PEFT backend is required for `fuse_lora()`.")

        self.lora_scale = lora_scale  # LoRA 스케일 저장
        self._safe_fusing = safe_fusing  # 안전한 융합 모드 저장
        self.apply(partial(self._fuse_lora_apply, adapter_names=adapter_names))  # 모든 모듈에 융합 적용

    def _fuse_lora_apply(self, module, adapter_names=None):
        '''
        개별 모듈에 LoRA 융합 적용
        - BaseTunerLayer인 모듈에만 적용
        - PEFT 버전 호환성 처리
        '''
        from peft.tuners.tuners_utils import BaseTunerLayer  # PEFT 기본 튜너 레이어

        merge_kwargs = {"safe_merge": self._safe_fusing}  # 병합 인수 설정

        if isinstance(module, BaseTunerLayer):  # 튜너 레이어인 경우
            if self.lora_scale != 1.0:  # 스케일이 1.0이 아닌 경우
                module.scale_layer(self.lora_scale)  # 레이어 스케일링 적용

            # 이전 PEFT 버전과의 호환성을 위해 merge 메서드 시그니처 확인
            # adapter_names 인수 지원 여부 확인
            supported_merge_kwargs = list(inspect.signature(module.merge).parameters)  # 지원되는 인수 목록
            if "adapter_names" in supported_merge_kwargs:  # adapter_names가 지원되는 경우
                merge_kwargs["adapter_names"] = adapter_names  # 인수에 추가
            elif "adapter_names" not in supported_merge_kwargs and adapter_names is not None:  # 지원되지 않는데 제공된 경우
                raise ValueError(
                    "The `adapter_names` argument is not supported with your PEFT version. Please upgrade"
                    " to the latest version of PEFT. `pip install -U peft`"
                )

            module.merge(**merge_kwargs)  # LoRA 가중치 병합

    def unfuse_lora(self):
        '''
        LoRA 어댑터 융합 해제
        - 병합된 LoRA 가중치를 분리하여 원본 상태로 복원
        - PEFT 백엔드가 필요함
        '''
        if not USE_PEFT_BACKEND:  # PEFT 백엔드 확인
            raise ValueError("PEFT backend is required for `unfuse_lora()`.")
        self.apply(self._unfuse_lora_apply)  # 모든 모듈에 융합 해제 적용

    def _unfuse_lora_apply(self, module):
        '''
        개별 모듈에 LoRA 융합 해제 적용
        - BaseTunerLayer인 모듈에만 적용
        '''
        from peft.tuners.tuners_utils import BaseTunerLayer  # PEFT 기본 튜너 레이어

        if isinstance(module, BaseTunerLayer):  # 튜너 레이어인 경우
            module.unmerge()  # LoRA 가중치 분리

    def forward(
        self,
        hidden_states: torch.FloatTensor = None,  # 이미지 잠재 상태
        encoder_hidden_states: torch.FloatTensor = None,  # 인코더 숨겨진 상태 (텍스트)
        pooled_projections: torch.FloatTensor = None,  # 풀링된 투영
        timestep: torch.LongTensor = None,  # 시간 단계
        timestep_text: torch.LongTensor = None,  # 텍스트용 시간 단계
        timestep_audio: torch.LongTensor = None,  # 오디오용 시간 단계
        block_controlnet_hidden_states: List = None,  # ControlNet 숨겨진 상태
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,  # Joint attention 인수
        return_dict: bool = True,  # 딕셔너리 반환 여부
        use_text_output: bool = False,  # 텍스트 출력 사용 여부
        target_prompt_embeds=None,  # 타겟 프롬프트 임베딩
        decode_text=False,  # 텍스트 디코딩 여부
        sigma_text=None,  # 텍스트 노이즈 시그마
        detach_logits=False,  # 로짓 분리 여부
        prompt_embeds_uncond=None,  # 무조건 프롬프트 임베딩
        targets=None,  # 타겟 데이터
        audio_hidden_states = None,  # 오디오 숨겨진 상태
        split_cond = False,  # 조건 분할 여부
        text_vae=None,  # 텍스트 VAE
        text_x0=True,  # 텍스트 x0 예측 여부
        drop_text=False,  # 텍스트 드롭 여부
        drop_image=False,  # 이미지 드롭 여부
        drop_audio=False,  # 오디오 드롭 여부
        kkwargs = None,  # 추가 키워드 인수
        forward_function = None,  # 커스텀 순전파 함수
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        OmniFlow Transformer 모델의 순전파 메서드
        - 텍스트, 이미지, 오디오를 통합적으로 처리
        - Joint attention을 통한 모달리티 간 상호작용
        - 각 모달리티별 독립적인 처리 및 출력 생성

        Args:
            hidden_states: 이미지 잠재 상태 (batch_size, channel, height, width)
            encoder_hidden_states: 텍스트 조건부 임베딩 (batch_size, seq_len, embed_dim)
            pooled_projections: 풀링된 조건부 임베딩 (batch_size, projection_dim)
            timestep: 디노이징 단계를 나타내는 시간 단계
            timestep_text: 텍스트용 시간 단계
            timestep_audio: 오디오용 시간 단계
            block_controlnet_hidden_states: ControlNet 잔차 상태들
            joint_attention_kwargs: Joint attention에 전달할 추가 인수
            return_dict: Transformer2DModelOutput 반환 여부
            use_text_output: 텍스트 출력 생성 여부
            target_prompt_embeds: 타겟 프롬프트 임베딩
            decode_text: 텍스트 디코딩 수행 여부
            sigma_text: 텍스트 노이즈 스케일
            detach_logits: 그래디언트 분리 여부
            prompt_embeds_uncond: 무조건부 프롬프트 임베딩
            targets: 학습용 타겟 데이터
            audio_hidden_states: 오디오 잠재 상태
            split_cond: 조건부 처리 분할 여부
            text_vae: 텍스트 VAE 모델
            text_x0: 텍스트 x0 예측 모드
            drop_text/drop_image/drop_audio: 각 모달리티 드롭아웃
            kkwargs: 커스텀 순전파용 추가 인수
            forward_function: 커스텀 순전파 함수

        Returns:
            딕셔너리 형태의 출력:
            - output: 이미지 출력
            - model_pred_text: 텍스트 예측
            - encoder_hidden_states: 인코더 상태
            - logits: 텍스트 디코딩 로짓
            - audio_hidden_states: 오디오 출력
        """
        if kkwargs is not None:  # 커스텀 순전파 함수가 지정된 경우
            assert forward_function is not None  # 순전파 함수가 반드시 있어야 함
            return forward_function(transformer=self,**kkwargs)  # 커스텀 함수로 처리
        
        encoder_hidden_states_base = encoder_hidden_states.clone()  # 원본 인코더 상태 백업
        hidden_states_base = hidden_states  # 원본 이미지 상태 백업
        
        # 각 모달리티 처리 여부 결정 (불필요한 브랜치를 null로 설정)
        do_image = not drop_image  # 이미지 처리 여부
        do_audio = (not drop_audio ) and ( self.add_audio)  # 오디오 처리 여부 (모델에서 지원하고 드롭하지 않을 때)
        do_text = (not drop_text)  # 텍스트 처리 여부
        
        if do_image:  # 이미지 처리가 활성화된 경우
            height, width = hidden_states.shape[-2:]  # 이미지 높이, 너비 추출
            hidden_states = self.pos_embed(hidden_states)  # 패치 임베딩 및 위치 임베딩 적용
            temb = self.time_text_embed(timestep, pooled_projections)  # 이미지용 시간 임베딩 생성
        else:  # 이미지 처리가 비활성화된 경우
            hidden_states = None  # 이미지 상태를 None으로 설정
            temb = 0  # 시간 임베딩을 0으로 설정
           
        
        if do_audio:  # 오디오 처리가 활성화된 경우
            if audio_hidden_states is None:  # 오디오 상태가 제공되지 않은 경우
                if self.use_audio_mae:  # MAE 방식 사용 시
                    # 빈 오디오 상태 생성 (batch_size, 8, audio_input_dim)
                    audio_hidden_states = torch.zeros(encoder_hidden_states.shape[0],8,self.audio_input_dim).to(encoder_hidden_states)
                else:  # 패치 방식 사용 시
                    # 빈 오디오 상태 생성 (batch_size, 8, 256, 16) - 스펙트로그램 형태
                    audio_hidden_states = torch.zeros(encoder_hidden_states.shape[0],8,256,16).to(encoder_hidden_states)
                timestep_audio = timestep_text * 0  # 오디오용 시간 단계를 0으로 설정
            
            temb_audio = self.time_aud_embed(timestep_audio,pooled_projections)  # 오디오용 시간 임베딩 생성
            audio_hidden_states = self.audio_embedder(audio_hidden_states)  # 오디오 임베딩 적용
            if not split_cond:  # 조건부 처리를 분할하지 않는 경우
                temb = temb + temb_audio  # 시간 임베딩들을 합산
                temb_audio = None  # 별도 오디오 시간 임베딩은 None으로 설정
        else:  # 오디오 처리가 비활성화된 경우
            audio_hidden_states = None  # 오디오 상태를 None으로 설정
            temb_audio = None  # 오디오 시간 임베딩을 None으로 설정
            

            
        if do_text:  # 텍스트 처리가 활성화된 경우
            if use_text_output:  # 텍스트 출력을 사용하는 경우
                temb_text = self.time_image_embed(timestep_text, pooled_projections)  # 텍스트용 시간 임베딩 생성
            encoder_hidden_states = self.context_embedder(encoder_hidden_states)  # 텍스트를 컨텍스트 차원으로 임베딩
            if use_text_output:  # 텍스트 출력을 사용하는 경우
                if not split_cond:  # 조건부 처리를 분할하지 않는 경우
                    temb = temb + temb_text  # 시간 임베딩들을 합산
                    temb_text = None  # 별도 텍스트 시간 임베딩은 None으로 설정
            else:
                temb_text = None  # 텍스트 시간 임베딩을 None으로 설정
        else:  # 텍스트 처리가 비활성화된 경우
            encoder_hidden_states = None  # 인코더 상태를 None으로 설정
            temb_text = None  # 텍스트 시간 임베딩을 None으로 설정
    
        assert not self.add_clip  # CLIP 사용이 비활성화되어 있음을 확인

        # Transformer 블록들을 순차적으로 처리
        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:  # 학습 중이고 그래디언트 체크포인팅이 활성화된 경우

                def create_custom_forward(module, return_dict=None):
                    '''체크포인팅을 위한 커스텀 순전파 함수 생성'''
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward

                ckpt_kwargs = dict()  # 체크포인팅 인수 (빈 딕셔너리)
                if self.add_audio:  # 오디오가 활성화된 경우
                    # 오디오를 포함한 체크포인팅 실행
                    encoder_hidden_states, hidden_states,audio_hidden_states = deepspeed.checkpointing.checkpoint(
                        create_custom_forward(block),  # 커스텀 순전파 함수
                        hidden_states,  # 이미지 상태
                        encoder_hidden_states,  # 텍스트 상태
                        temb,  # 시간 임베딩
                        audio_hidden_states,  # 오디오 상태
                        temb_text,  # 텍스트 시간 임베딩
                        temb_audio,  # 오디오 시간 임베딩
                        **ckpt_kwargs,
                    )
                else:  # 오디오가 비활성화된 경우
                    # 텍스트와 이미지만 포함한 체크포인팅 실행
                    encoder_hidden_states, hidden_states = deepspeed.checkpointing.checkpoint(
                        create_custom_forward(block),  # 커스텀 순전파 함수
                        hidden_states,  # 이미지 상태
                        encoder_hidden_states,  # 텍스트 상태
                        temb,  # 시간 임베딩
                        temb_text,  # 텍스트 시간 임베딩
                        **ckpt_kwargs,
                    )

            else:  # 일반적인 순전파 (체크포인팅 없음)
                if self.add_audio:  # 오디오가 활성화된 경우
                    # 오디오를 포함한 블록 실행
                    encoder_hidden_states, hidden_states,audio_hidden_states = block(
                        hidden_states=hidden_states,  # 이미지 상태
                        encoder_hidden_states=encoder_hidden_states,  # 텍스트 상태
                        audio_hidden_states=audio_hidden_states,  # 오디오 상태
                        temb=temb,  # 시간 임베딩
                        temb_text=temb_text,  # 텍스트 시간 임베딩
                        temb_audio=temb_audio,  # 오디오 시간 임베딩
                    )
                else:  # 오디오가 비활성화된 경우
                    # 텍스트와 이미지만 포함한 블록 실행
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,  # 이미지 상태
                        encoder_hidden_states=encoder_hidden_states,  # 텍스트 상태
                        temb=temb,  # 시간 임베딩
                        temb_text=temb_text  # 텍스트 시간 임베딩
                    )

            # ControlNet 잔차 처리 (현재 사용되지 않음)
            assert block_controlnet_hidden_states is None
            # if block_controlnet_hidden_states is not None and block.context_pre_only is False:
            #     interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
            #     hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

        # 이미지 출력 처리
        if do_image:  # 이미지 처리가 활성화된 경우
            hidden_states = self.norm_out(hidden_states, temb)  # 출력 정규화 적용
            hidden_states = self.proj_out(hidden_states)  # 최종 투영 레이어 적용

            # 패치를 원본 이미지로 복원 (unpatchify)
            patch_size = self.config.patch_size  # 패치 크기 가져오기
            height = height // patch_size  # 패치 단위 높이 계산
            width = width // patch_size  # 패치 단위 너비 계산

            # 패치 형태로 재구성: (batch, height, width, patch_h, patch_w, channels)
            hidden_states = hidden_states.reshape(
                shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
            )
            # 차원 재배열: 패치를 이미지로 변환
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            # 최종 이미지 형태로 재구성: (batch, channels, full_height, full_width)
            output = hidden_states.reshape(
                shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
            )
        else:  # 이미지 처리가 비활성화된 경우
            output = None  # 이미지 출력을 None으로 설정

        # PEFT 백엔드 처리 (주석 처리됨)
        # if USE_PEFT_BACKEND:
        #     # 각 PEFT 레이어에서 lora_scale 제거
        #     unscale_lora_layers(self, lora_scale)
        assert not return_dict  # 딕셔너리 반환이 비활성화되어 있음을 확인
        logits = None  # 로짓 초기화
        logits_labels = None  # 라벨 로짓 초기화
        
        # 텍스트 출력 처리
        if do_text and use_text_output:  # 텍스트 처리와 텍스트 출력이 모두 활성화된 경우
            encoder_hidden_states = self.context_decoder['projection'](encoder_hidden_states)  # 컨텍스트 디코더 투영 적용
            encoder_hidden_states = self.norm_out_text(encoder_hidden_states,temb_text if temb_text is not None else temb)  # 텍스트 출력 정규화
            encoder_hidden_states = self.text_output(encoder_hidden_states)  # 텍스트 출력 레이어 적용
            model_pred_text = encoder_hidden_states  # x0 예측 (깨끗한 텍스트 예측)
            
            if decode_text and targets is not None:  # 텍스트 디코딩이 활성화되고 타겟이 있는 경우
                logits = None  # 로짓 초기화
                logits_labels = None  # 라벨 로짓 초기화
                if self.text_decoder is not None:  # 텍스트 디코더가 있는 경우
                    if detach_logits:  # 로짓을 그래디언트에서 분리하는 경우
                        with torch.no_grad():  # 그래디언트 계산 비활성화
                            if prompt_embeds_uncond is not None:  # 무조건부 프롬프트 임베딩이 있는 경우
                                raw_text_embeds_input = prompt_embeds_uncond[...,:self.text_out_dim]  # 무조건부 임베딩 사용
                            else:
                                raw_text_embeds_input = target_prompt_embeds[...,:self.text_out_dim]  # 타겟 임베딩 사용
                            if text_x0:  # x0 예측 모드인 경우
                                model_pred_text_clean = model_pred_text  # 예측값을 그대로 사용
                            else:  # 노이즈 예측 모드인 경우
                                noisy_prompt_embeds = encoder_hidden_states_base[...,:model_pred_text.shape[-1]]  # 노이즈가 있는 임베딩
                                model_pred_text_clean = model_pred_text * (-sigma_text) + noisy_prompt_embeds[...,:model_pred_text.shape[-1]]  # 노이즈 제거
                            latents_decode = torch.cat([model_pred_text_clean,raw_text_embeds_input],dim=0).detach()  # 디코딩용 잠재 변수 결합
                    else:  # 로짓을 그래디언트에서 분리하지 않는 경우
                        if prompt_embeds_uncond is not None:  # 무조건부 프롬프트 임베딩이 있는 경우
                            raw_text_embeds_input = prompt_embeds_uncond[...,:self.text_out_dim]  # 무조건부 임베딩 사용
                        else:
                            raw_text_embeds_input = target_prompt_embeds[...,:self.text_out_dim]  # 타겟 임베딩 사용
                        if text_x0:  # x0 예측 모드인 경우
                            model_pred_text_clean = model_pred_text  # 예측값을 그대로 사용
                        else:  # 노이즈 예측 모드인 경우
                            noisy_prompt_embeds = encoder_hidden_states_base[...,:model_pred_text.shape[-1]]  # 노이즈가 있는 임베딩
                            model_pred_text_clean = model_pred_text * (-sigma_text) + noisy_prompt_embeds[...,:model_pred_text.shape[-1]]  # 노이즈 제거
                        latents_decode = torch.cat([model_pred_text_clean,raw_text_embeds_input],dim=0)  # 디코딩용 잠재 변수 결합
                    
                    # 텍스트 디코더를 통한 로짓 생성
                    logits_all = self.text_decoder(latents=latents_decode,  # 잠재 변수
                                        input_ids=targets['input_ids'].repeat(2,1),  # 입력 ID (2배로 복제)
                                        attention_mask=None,  # 어텐션 마스크 (사용하지 않음)
                                        labels=None,  # 라벨 (사용하지 않음)
                                        return_dict=False  # 딕셔너리 반환 비활성화
                                    )[0]
                    logits,logits_labels = logits_all.chunk(2)  # 로짓을 예측용과 라벨용으로 분할
        else:  # 텍스트 처리가 비활성화된 경우
            model_pred_text = None  # 텍스트 예측을 None으로 설정

        # 오디오 출력 처리
        if do_audio:  # 오디오 처리가 활성화된 경우
                audio_hidden_states = self.norm_out_aud(audio_hidden_states,temb_audio if temb_audio is not None else temb)  # 오디오 출력 정규화
                audio_hidden_states = self.proj_out_aud(audio_hidden_states)  # 오디오 출력 투영
                if not self.use_audio_mae:  # MAE 방식을 사용하지 않는 경우 (패치 방식)
                    patch_size_audio = self.audio_patch_size  # 오디오 패치 크기
                    height_audio = 256 // patch_size_audio  # 패치 단위 오디오 높이
                    width_audio = 16 // patch_size_audio  # 패치 단위 오디오 너비
                    # 오디오 패치를 원본 스펙트로그램으로 복원
                    # 형태: N X [(256/patch_size) X (16/patch_size)] X [patch_size X patch_size X channels]
                    audio_hidden_states = rearrange(
                        audio_hidden_states,
                        'n (h w) (hp wp c) -> n c (h hp) (w wp)',  # 패치에서 스펙트로그램으로 재배열
                        h=height_audio,  # 패치 단위 높이
                        w=width_audio,  # 패치 단위 너비
                        hp=patch_size_audio,  # 패치 높이
                        wp=patch_size_audio,  # 패치 너비
                        c=self.audio_input_dim  # 오디오 채널 수
                    )
                    # 아래는 대체 구현 방법 (주석 처리됨)
                    # audio_hidden_states = audio_hidden_states.reshape(
                    #     shape=(audio_hidden_states.shape[0], height_audio, width_audio, patch_size_audio, patch_size_audio, self.audio_input_dim)
                    # )
                    # audio_hidden_states = torch.einsum("nhwpqc->nchpwq", audio_hidden_states)
                    # audio_hidden_states = audio_hidden_states.reshape(
                    #     shape=(audio_hidden_states.shape[0], self.audio_input_dim, height_audio * patch_size_audio, width_audio * patch_size_audio)
                    # )
        else:  # 오디오 처리가 비활성화된 경우
                audio_hidden_states = None  # 오디오 상태를 None으로 설정
        
        # 최종 출력 딕셔너리 반환
        return dict(output=output,  # 이미지 출력
                    model_pred_text=model_pred_text,  # 텍스트 예측
                    encoder_hidden_states=encoder_hidden_states,  # 인코더 상태
                    logits=logits,  # 텍스트 디코딩 로짓
                    extra_cond=None,  # 추가 조건 (사용하지 않음)
                    logits_labels=logits_labels,  # 라벨 로짓
                    audio_hidden_states=audio_hidden_states,  # 오디오 출력
                    )

        # 대체 반환 형식 (주석 처리됨)
        #return Transformer2DModelOutput(sample=output)


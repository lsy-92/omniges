"""
Gesture Processor for Omniges
4x RVQVAE Gesture Encoder/Decoder for different body parts

Extracted from omniges_a2g.py and optimized for multi-task use
Following shortcut_rvqvae_trainer.py pattern for proper concatenation
"""

import torch  # PyTorch 라이브러리
import torch.nn as nn  # PyTorch 신경망 모듈
from typing import Dict  # 타입 힌팅을 위한 Dict
from models.vq.model import RVQVAE  # RVQVAE 모델 임포트


class GestureProcessor(nn.Module):
    """
    4개의 RVQVAE 제스처 인코더/디코더 (다른 신체 부위용)
    shortcut_rvqvae_trainer.py 패턴을 따라 적절한 잠재 변수 연결을 수행
    """
    
    def __init__(
        self,
        ckpt_paths: Dict[str, str],  # 체크포인트 경로 딕셔너리
        device: str = "cuda"  # 디바이스 (기본값: cuda)
    ):
        super().__init__()  # 부모 클래스 초기화
        
        # ============================================================================
        # 신체 부위 및 차원 정의
        # ============================================================================
        self.body_parts = ["upper", "hands", "lower_trans", "face"]  # 신체 부위 리스트
        self.part_dims = {  # 각 신체 부위별 입력 차원
            "upper": 78,  # 상체: 78차원
            "hands": 180,  # 손: 180차원
            "lower_trans": 57,  # 하체+이동: 57차원
            "face": 100  # 얼굴: 100차원
        }
        
        # ============================================================================
        # 사전 훈련된 RVQVAE 모델 로드
        # ============================================================================
        self.rvqvae_models = nn.ModuleDict()  # RVQVAE 모델들을 저장할 ModuleDict
        self.load_rvqvae_models(ckpt_paths, device)  # RVQVAE 모델들 로드
        
        # ============================================================================
        # RVQVAE 파라미터 고정 (고정된 인코더/디코더로 사용)
        # ============================================================================
        for model in self.rvqvae_models.values():  # 모든 RVQVAE 모델에 대해
            for param in model.parameters():  # 각 모델의 모든 파라미터에 대해
                param.requires_grad = False  # 그래디언트 계산 비활성화 (파라미터 고정)
        
    def load_rvqvae_models(self, ckpt_paths: Dict[str, str], device: str):
        """
        각 신체 부위별 사전 훈련된 RVQVAE 모델들을 로드
        """
        # ============================================================================
        # RVQVAE 기본 인자들 (실제 체크포인트 설정과 일치)
        # ============================================================================
        class DefaultArgs:  # RVQVAE 모델 생성에 필요한 기본 인자들
            def __init__(self):
                self.nb_code = 1024  # 체크포인트에서: 코드북 크기는 1024
                self.code_dim = 128  # 체크포인트에서: 임베딩 차원은 128
                self.down_t = 2  # 시간 축 다운샘플링 팩터
                self.stride_t = 2  # 시간 축 스트라이드
                self.width = 512  # 네트워크 너비
                self.depth = 3  # 네트워크 깊이
                self.dilation_growth_rate = 3  # 확장 성장률
                self.vq_act = 'relu'  # 벡터 양자화 활성화 함수
                self.vq_norm = None  # 벡터 양자화 정규화
                self.num_quantizers = 6  # 양자화기 수
                self.shared_codebook = False  # 공유 코드북 사용 여부
                self.quantize_dropout_prob = 0.2  # 양자화 드롭아웃 확률
                # 추가로 필요한 인자들
                self.mu = 0.99  # EMA 감쇠율
                self.quantizer = 'ema_reset'  # 양자화기 타입
                self.beta = 1.0  # 베타 값
                self.vae_length = 64  # VAE 길이
                self.vae_codebook_size = 1024  # nb_code와 일치
                self.vae_quantizer_lambda = 1.0  # VAE 양자화기 람다
        
        args = DefaultArgs()  # 기본 인자 객체 생성
        
        # ============================================================================
        # 각 신체 부위별 RVQVAE 모델 로드
        # ============================================================================
        for part in self.body_parts:  # 각 신체 부위에 대해
            if part in ckpt_paths and ckpt_paths[part] is not None:  # 체크포인트 경로가 존재하고 None이 아닌 경우
                # ============================================================================
                # RVQVAE 모델 생성
                # ============================================================================
                model = RVQVAE(  # RVQVAE 모델 생성
                    args,  # 기본 인자들
                    input_width=self.part_dims[part],  # 해당 부위의 입력 차원
                    nb_code=args.nb_code,  # 코드북 크기
                    code_dim=args.code_dim,  # 코드 차원
                    output_emb_width=args.code_dim,  # 출력 임베딩 너비
                    down_t=args.down_t,  # 시간 축 다운샘플링
                    stride_t=args.stride_t,  # 시간 축 스트라이드
                    width=args.width,  # 네트워크 너비
                    depth=args.depth,  # 네트워크 깊이
                    dilation_growth_rate=args.dilation_growth_rate,  # 확장 성장률
                    activation=args.vq_act,  # 활성화 함수
                    norm=args.vq_norm  # 정규화
                )
                
                # ============================================================================
                # 체크포인트 로드
                # ============================================================================
                ckpt = torch.load(ckpt_paths[part], map_location='cpu')  # CPU에서 체크포인트 로드
                model.load_state_dict(ckpt['net'], strict=True)  # 모델에 체크포인트 로드 (엄격하게)
                model.to(device).eval()  # 디바이스로 이동하고 평가 모드로 설정
                
                self.rvqvae_models[part] = model  # 모델을 ModuleDict에 저장
                print(f"Loaded RVQVAE model for {part} from {ckpt_paths[part]}")  # 로드 완료 메시지 출력
    
    def encode_gesture(self, gesture_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        제스처 데이터를 연속 잠재 변수로 인코딩 (shortcut_rvqvae_trainer.py 패턴 따름)
        
        Args:
            gesture_dict: 신체 부위 데이터 딕셔너리
                - upper: (B, T, 78) 6D 회전 형식
                - hands: (B, T, 180) 6D 회전 형식
                - lower_trans: (B, T, 57) 6D 회전 형식
                - face: (B, T, 100)
                
        Returns:
            각 부위별 연속 잠재 변수 딕셔너리
        """
        encoded = {}  # 인코딩된 결과를 저장할 딕셔너리
        
        # ============================================================================
        # 각 신체 부위별 인코딩
        # ============================================================================
        for part, data in gesture_dict.items():  # 각 신체 부위와 데이터에 대해
            if part in self.rvqvae_models:  # 해당 부위의 RVQVAE 모델이 존재하는 경우
                with torch.no_grad():  # 그래디언트 계산 비활성화
                    # map2latent을 사용하여 연속 잠재 변수 추출
                    latents = self.rvqvae_models[part].map2latent(data)  # (B, T//downsample, code_dim)
                    encoded[f"{part}_latents"] = latents  # 인코딩된 잠재 변수를 딕셔너리에 저장
        
        return encoded  # 인코딩된 결과 반환
    
    def decode_gesture(
        self, 
        latents_dict: Dict[str, torch.Tensor]  # 잠재 변수 딕셔너리
    ) -> Dict[str, torch.Tensor]:
        """
        연속 잠재 변수에서 제스처 데이터 디코딩 (shortcut_rvqvae_trainer.py 패턴 따름)
        
        Args:
            latents_dict: 각 신체 부위별 연속 잠재 변수
            
        Returns:
            각 신체 부위별 재구성된 제스처 데이터
        """
        decoded = {}  # 디코딩된 결과를 저장할 딕셔너리
        
        # ============================================================================
        # 각 신체 부위별 디코딩
        # ============================================================================
        for part in self.body_parts:  # 각 신체 부위에 대해
            part_key = f"{part}_latents"  # 해당 부위의 잠재 변수 키
            if part in self.rvqvae_models and part_key in latents_dict:  # RVQVAE 모델과 잠재 변수가 모두 존재하는 경우
                input_latents = latents_dict[part_key]  # 입력 잠재 변수 추출
                with torch.no_grad():  # 그래디언트 계산 비활성화
                    # latent2origin을 사용하여 연속 잠재 변수 디코딩
                    recon_output = self.rvqvae_models[part].latent2origin(input_latents)  # 재구성 출력
                    if isinstance(recon_output, tuple):  # 출력이 튜플인 경우
                        decoded[part] = recon_output[0]  # 첫 번째 요소 사용
                    else:  # 출력이 단일 텐서인 경우
                        decoded[part] = recon_output  # 그대로 사용
        
        return decoded  # 디코딩된 결과 반환

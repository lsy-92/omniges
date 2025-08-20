"""
Omniges: Multimodal Text-Audio-Gesture Generation Framework

OmniFlow 기반으로 이미지 스트림을 제스처 스트림으로 치환한 멀티모달 생성 프레임워크
지원 태스크: T2G, G2T, A2G, G2A, T2A, A2T (6개 모든 조합)

핵심 구성요소:
- OmnigesFlowTransformerModel: OmniFlow 기반 멀티모달 transformer
- GestureProcessor: 4x RVQVAE 제스처 인코더/디코더
- OmnigesPipeline: 완전한 추론 파이프라인
- train_omniges.py: 멀티태스크 학습 스크립트
"""

__version__ = "1.0.0"
__author__ = "Omniges Team"

# Import key components for easy access
from .models import OmnigesFlowTransformerModel, GestureProcessor
from .pipelines import OmnigesPipeline, OmnigesGestureVAE

__all__ = [
    "OmnigesFlowTransformerModel",
    "GestureProcessor", 
    "OmnigesPipeline",
    "OmnigesGestureVAE"
]
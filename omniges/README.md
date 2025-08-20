# Omniges: Multimodal Text-Audio-Gesture Generation

**Complete OmniFlow-based Text-Audio-Gesture Generation Framework**

Omniges는 OmniFlow 파이프라인을 기반으로 이미지 스트림을 제스처 스트림으로 치환한 멀티모달 생성 프레임워크입니다. 6개 모든 태스크 조합을 지원합니다: **T2G, G2T, A2G, G2A, T2A, A2T**.

## 🎯 Architecture Overview

### Core Components

1. **OmnigesFlowTransformerModel**: OmniFlow 기반 멀티모달 Transformer
   - Text, Audio, Gesture 3개 모달리티 joint attention
   - 24 layers, 24 heads, 1536 embedding dimension
   - 모든 6개 태스크 조합 지원: T2G, G2T, A2G, G2A, T2A, A2T

2. **GestureProcessor**: 4x Pre-trained RVQVAE
   - Upper body: 78D → 128D latent
   - Hands: 180D → 128D latent  
   - Lower+Trans: 57D → 128D latent
   - Face: 100D → 128D latent
   - 4개 부위 concatenation: 512D total latent

3. **OmnigesPipeline**: 완전한 추론 파이프라인
   - OmniFlow 기반 모든 구성요소 통합
   - load_pretrained() 자동 체크포인트 로딩
   - 사용자 친화적 API: prompt → gesture/text/audio

4. **Multi-Task Training**: 통합 학습 프레임워크
   - BEAT2 데이터셋 기반
   - Round-robin 또는 확률적 태스크 샘플링
   - OmniFlow 호환 loss weighting

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install torch torchvision torchaudio transformers accelerate wandb
pip install librosa textgrid loguru tqdm pyyaml
pip install smplx trimesh pyrender  # For rendering

# Install additional packages
pip install -e .
```

### 2. Data Preparation

Ensure BEAT2 dataset is available at:
```
./datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0/
├── beat_smplx_141/          # SMPLX pose files (.npz)
├── wave16k/                 # Audio files (.wav)  
├── textgrid/                # Word alignment (.TextGrid)
├── expression/              # Facial expressions
└── train_test_split.csv     # Dataset splits
```

### 3. Multi-Task Training

#### Full Training (Multi-GPU)
```bash
# Complete 6-task training (T2G, G2T, A2G, G2A, T2A, A2T)
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    omniges/scripts/train_omniges.py \
    --pretrained_model_name_or_path ./checkpoint/OmniFlow-v0.5 \
    --beat_config_path configs/shortcut_rvqvae_128.yaml \
    --rvqvae_checkpoints ./ckpt/ \
    --text_vae ./checkpoint/OmniFlow-v0.5/text_vae \
    --tokenizer ./checkpoint/OmniFlow-v0.5/vae_tokenizer \
    --output_dir ./results/omniges_multitask \
    --train_batch_size 4 \
    --num_train_epochs 20 \
    --learning_rate 1e-4 \
    --val_every 100 \
    --report_to wandb
```

#### Single GPU Training
```bash
# For testing or smaller hardware
python omniges/scripts/train_omniges.py \
    --pretrained_model_name_or_path ./checkpoint/OmniFlow-v0.5 \
    --beat_config_path configs/shortcut_rvqvae_128.yaml \
    --rvqvae_checkpoints ./ckpt/ \
    --text_vae ./checkpoint/OmniFlow-v0.5/text_vae \
    --tokenizer ./checkpoint/OmniFlow-v0.5/vae_tokenizer \
    --output_dir ./results/omniges_multitask \
    --train_batch_size 1 \
    --num_train_epochs 1 \
    --val_every 50
```

### 4. Inference & Evaluation

#### Using Pipeline (Recommended)
```python
from omniges import OmnigesPipeline

# Load pipeline
pipeline = OmnigesPipeline.load_pretrained(
    omniflow_path="./checkpoint/OmniFlow-v0.5",
    rvqvae_checkpoints={
        'upper': './ckpt/net_300000_upper.pth',
        'hands': './ckpt/net_300000_hands.pth',
        'lower_trans': './ckpt/net_300000_lower.pth',
        'face': './ckpt/net_300000_face.pth'
    }
)

# Text to Gesture
gesture = pipeline(prompt="A person waving hello", task='t2g')

# Audio to Gesture  
gesture = pipeline(input_aud="./audio.wav", task='a2g')

# Gesture to Text
text = pipeline(input_gesture=gesture_tensor, task='g2t')

# All 6 tasks supported!
```

## 📁 File Structure

```
omniges/
├── models/
│   ├── omniges_flow.py         # 핵심: OmniFlow 기반 멀티모달 transformer
│   ├── gesture_processor.py   # 핵심: 4x RVQVAE 제스처 처리기
│   └── __init__.py
├── pipelines/
│   ├── omniges_pipeline.py     # 핵심: 완전한 추론 파이프라인 (모든 6개 태스크)
│   └── __init__.py
├── scripts/
│   ├── train_omniges.py        # 핵심: 멀티태스크 학습 스크립트
│   └── __init__.py
└── README.md

configs/
└── shortcut_rvqvae_128.yaml    # BEAT 데이터셋 설정

checkpoint/
└── OmniFlow-v0.5/              # OmniFlow 사전훈련 모델

ckpt/                           # Pre-trained RVQVAE models
├── net_300000_upper.pth
├── net_300000_hands.pth
├── net_300000_lower.pth
└── net_300000_face.pth
```

## ⚙️ Key Features

### Supported Tasks
- **T2G**: Text → Gesture (텍스트로 제스처 생성)
- **G2T**: Gesture → Text (제스처에서 텍스트 생성)  
- **A2G**: Audio → Gesture (오디오로 제스처 생성)
- **G2A**: Gesture → Audio (제스처에서 오디오 생성)
- **T2A**: Text → Audio (텍스트로 오디오 생성, OmniFlow 기능)
- **A2T**: Audio → Text (오디오에서 텍스트 생성, OmniFlow 기능)

### Architecture Details
- **Backbone**: OmniFlow MMDiT (24 layers, 24 heads, 1536D)
- **Text Processing**: CLIP×2 + T5 + LLaMA VAE (OmniFlow 방식)
- **Audio Processing**: WavLM + MFCC fusion (OmniFlow 방식)  
- **Gesture Processing**: 4x RVQVAE (upper/hands/lower_trans/face)
- **Multi-Task**: Round-robin 또는 확률적 태스크 샘플링

## 📊 Expected Performance

### Training Metrics (All Tasks)
- **Gesture Loss**: Flow matching loss for gesture generation (T2G, A2G)
- **Text Loss**: Flow matching + decode loss for text generation (G2T, A2T)  
- **Audio Loss**: Flow matching loss for audio generation (G2A, T2A)
- **Multi-Task Weighting**: Task-specific loss factors
  
### Evaluation Metrics
- **Gesture Quality**: FGD, L1 Diversity, Motion smoothness
- **Text Quality**: BLEU, ROUGE, Semantic similarity
- **Audio Quality**: Spectrogram alignment, Audio-gesture sync
- **Cross-Modal**: Alignment scores between modalities

### Hardware Requirements
- **Memory**: ~12-16GB per GPU for batch_size=4 (3 modalities)
- **Training Time**: ~3-5 days for 20 epochs on multi-GPU
- **Storage**: ~80GB for BEAT cache + checkpoints + OmniFlow model

## 🔧 Troubleshooting

### Common Issues

1. **CUDA OOM**: Multi-modality 학습은 메모리가 많이 필요합니다
   ```bash
   # 배치 크기 줄이기
   --train_batch_size 1
   # 또는 gradient accumulation 사용
   --gradient_accumulation_steps 4
   ```

2. **Checkpoint Loading**: OmniFlow 체크포인트 경로 확인
   ```bash
   # 체크포인트 구조 확인
   ls -la ./checkpoint/OmniFlow-v0.5/
   # transformer/, text_vae/, vae_tokenizer/ 존재해야 함
   ```

3. **RVQVAE Models**: 4개 부위 체크포인트 모두 필요
   ```bash
   # RVQVAE 파일 확인
   ls -la ./ckpt/net_300000_*.pth
   ```

4. **Task-specific Issues**: 특정 태스크에서 validation 실패 시 해당 태스크 비활성화

### Monitoring
- **W&B Dashboard**: 모든 6개 태스크 실시간 모니터링
- **TensorBoard**: Local metric visualization  
- **Validation**: 매 100 스텝마다 모든 태스크 테스트

## 🛣️ Future Extensions

1. **LibriTTS Integration**: 더 큰 오디오-텍스트 데이터셋 확장
2. **Real-time Inference**: 실시간 제스처 생성 최적화
3. **Fine-tuning**: 특정 도메인/스타일 fine-tuning
4. **3D Rendering**: SMPL-X 기반 실시간 3D 렌더링
5. **Mobile Deployment**: 경량화 모델 for mobile/edge devices

## 📚 References

- **OmniFlow**: Base multimodal pipeline
- **MMDiT**: Multimodal Diffusion Transformer
- **BEAT2**: Large-scale gesture dataset
- **RVQVAE**: Residual Vector Quantized VAE for motion

# Omniges Development Plan - A2G Focus Implementation

## 🎯 **우선 집중 태스크: A2G (Audio to Gesture)**

### 하드웨어 & 리소스
- **GPU**: A100 ×2 + A6000 ADA ×4 (총 6 GPUs)
- **Memory**: ~48GB (A100) + ~48GB (A6000) = 96GB 총 GPU 메모리
- **권장 배치**: 총 배치 사이즈 24-32 (4-5 per GPU)

### 체크포인트 경로 확인
- **RVQVAE Models**: `/home/lsy92/project/data/project/omniges/ckpt/`
  - Upper: `net_300000_upper.pth` (26 joints = 78 dim)
  - Hands: `net_300000_hands.pth` (60 joints = 180 dim)  
  - Lower: `net_300000_lower.pth` (8 joints + trans = 57 dim)
  - Face: `net_300000_face.pth` (jaw 3 + expressions 100 = 103 dim)

### 오디오 융합: WavLM + MFCC
- **WavLM**: 768차원 → 512차원 투영
- **MFCC**: 128차원 → 512차원 투영  
- **Fusion**: Bidirectional cross-attention → 512차원 출력

## Phase 1: A2G Core Implementation (Week 1-2)

### 1.1 Omniges Model Implementation
**File**: `omniges/models/omniges_transformer.py`

**기존 구조 활용:**
- `omniflow/models/omni_flow.py`의 OmniFlowTransformerModel 기반
- `mmdit/mmdit_generalized_pytorch.py`의 MMDiT 백본 사용

**주요 변경사항:**
```python
class OmnigesTransformerModel:
    def __init__(self):
        # 기존: text + audio + image 스트림
        # 신규: text + audio + gesture 스트림
        
        # Modality dimensions
        self.dim_text = 1024      # T5/LLaMA encoding
        self.dim_audio = 512      # WavLM + MFCC/WavEncoder fusion
        self.dim_gesture = {      # Body part specific
            "upper": 78,
            "hands": 180, 
            "lower_trans": 57,
            "face": 100
        }
        
        # MMDiT backbone with 3 modalities
        self.backbone = MMDiT(
            dim_modalities=(self.dim_text, self.dim_audio, sum(self.dim_gesture.values())),
            depth=12,
            num_residual_streams=4
        )
```

### 1.2 Audio Fusion Module Enhancement
**File**: `omniges/models/audio_fusion.py` (기존 AudioAttentionFusion 확장)

```python
class EnhancedAudioFusion:
    def __init__(self, dim_wavlm=768, dim_other=80, dim_out=512, num_layers=2):
        # Bidirectional cross-attention fusion
        # WavLM ↔ MFCC/WavEncoder cross-attention
        # Gated combination with residual connections
```

### 1.3 Gesture Encoding/Decoding System
**File**: `omniges/models/gesture_processor.py`

**RVQVAE Integration:**
- 4개 부위별 RVQVAE 모델 로드 및 통합
- Continuous latent space (x0/ε prediction) + Discrete code prediction (CE loss) 지원

```python
class GestureProcessor:
    def __init__(self):
        # Load 4 pre-trained RVQVAE models
        self.rvqvae_models = {
            "upper": self.load_rvqvae("weights/rvqvae_upper.ckpt"),
            "hands": self.load_rvqvae("weights/rvqvae_hands.ckpt"), 
            "lower_trans": self.load_rvqvae("weights/rvqvae_lower_trans.ckpt"),
            "face": self.load_rvqvae("weights/rvqvae_face.ckpt")
        }
        
    def encode(self, gesture_dict):
        # Return both continuous latents and discrete codes
        
    def decode_from_latents(self, latents_dict):
        # Continuous space decoding
        
    def decode_from_codes(self, codes_dict):  
        # Discrete space decoding
```

## Phase 2: Data Pipeline & Loader (Week 2-3)

### 2.1 Unified Data Loader
**File**: `omniges/dataloaders/omniges_loader.py`

**기반**: `dataloaders/beat_sep_lower.py` 확장

```python
class OmnigesDataset(Dataset):
    def __init__(self, args, mode="train"):
        # Support multiple tasks: T2G, A2G, G2T, G2A
        # Unified batch collation with proper masking
        
    def __getitem__(self, idx):
        return {
            "text": text_tokens,
            "audio": audio_features,  # WavLM + MFCC fused
            "gesture": {
                "upper": upper_pose,
                "hands": hands_pose,
                "lower_trans": lower_trans_pose, 
                "face": face_pose
            },
            "task": task_type,  # T2G, A2G, G2T, G2A
            "mask": attention_mask
        }
```

### 2.2 Multi-task Sampling Strategy
```python
class MultiTaskSampler:
    def __init__(self, dataset, task_ratios={"T2G": 0.3, "A2G": 0.3, "G2T": 0.2, "G2A": 0.2}):
        # Round-robin or probability-based sampling
        # Balanced batch composition across tasks
```

## Phase 3: Training Infrastructure (Week 3-4)

### 3.1 Training Script Extension
**File**: `omniges/scripts/train.py`

**기반**: `omniflow/scripts/train.py` 확장

```python
def main(args):
    # Load Omniges model instead of OmniFlow
    model = OmnigesTransformerModel.from_config(args.model_config)
    
    # Multi-task loss computation
    def compute_loss(batch, predictions):
        losses = {}
        
        if batch["task"] in ["T2G", "A2G"]:  # Gesture generation
            # RVQVAE reconstruction + commitment loss
            losses["gesture_recon"] = F.mse_loss(predictions["gesture"], batch["gesture"])
            losses["commitment"] = compute_commitment_loss(predictions["codes"])
            
        if batch["task"] in ["G2T", "G2A"]:  # Text/Audio generation  
            # Cross-entropy for text, audio reconstruction for audio
            losses["text_ce"] = F.cross_entropy(predictions["text_logits"], batch["text_ids"])
            losses["audio_recon"] = F.mse_loss(predictions["audio"], batch["audio"])
            
        return losses
```

### 3.2 Loss Function Integration
```python
class OmnigesLoss:
    def __init__(self):
        # RVQVAE losses (reconstruction + commitment + perplexity)
        # Alignment losses (cross-modal attention alignment)
        # Velocity/Acceleration consistency losses
        # Perceptual losses (optional)
        
    def forward(self, predictions, targets, task_type):
        # Task-specific loss computation
        # Weighted combination based on task
```

## Phase 4: Configuration & Experiments (Week 4-5)

### 4.1 Configuration Files
**Files**: `configs/omniges/*.yaml`

```yaml
# configs/omniges/base_config.yaml
model:
  name: "omniges"
  backbone: "mmdit"
  dim_text: 1024
  dim_audio: 512  
  dim_gesture_total: 415  # 78+180+57+100
  
audio_fusion:
  dim_wavlm: 768
  dim_other: 80  # MFCC or WavEncoder
  dim_out: 512
  num_layers: 2
  
gesture_rvqvae:
  upper_ckpt: "weights/rvqvae_upper.ckpt"
  hands_ckpt: "weights/rvqvae_hands.ckpt" 
  lower_trans_ckpt: "weights/rvqvae_lower_trans.ckpt"
  face_ckpt: "weights/rvqvae_face.ckpt"
  
training:
  batch_size: 16
  learning_rate: 1e-4
  num_epochs: 100
  task_sampling: "round_robin"  # or "probability"
  task_ratios:
    T2G: 0.3
    A2G: 0.3  
    G2T: 0.2
    G2A: 0.2
```

### 4.2 Experiment Scripts
```bash
# Single task experiments
accelerate launch omniges/scripts/train.py --config configs/omniges/a2g_only.yaml
accelerate launch omniges/scripts/train.py --config configs/omniges/t2g_only.yaml

# Multi-task experiment  
accelerate launch omniges/scripts/train.py --config configs/omniges/multitask.yaml

# Evaluation with rendering
python omniges/scripts/test.py --config configs/omniges/multitask.yaml --render --output_dir results/
```

## Phase 5: Testing & Validation (Week 5-6)

### 5.1 Metrics Integration
**File**: `omniges/evaluation/metrics.py`

```python
class OmnigesMetrics:
    def __init__(self):
        # Gesture quality metrics (FGD, L1 diversity, velocity consistency)
        # Cross-modal alignment metrics  
        # Text generation quality (BLEU, ROUGE)
        # Audio generation quality (MCD, spectral metrics)
        
    def evaluate_gesture_generation(self, generated, ground_truth):
        # Use existing utils/metric.py functions
        
    def evaluate_text_generation(self, generated_text, reference_text):
        # BLEU, ROUGE, semantic similarity
        
    def evaluate_audio_generation(self, generated_audio, reference_audio):
        # MCD, spectral distance, perceptual metrics
```

### 5.2 Rendering & Visualization  
```python
class OmnigesVisualizer:
    def __init__(self):
        # Extend utils/fast_render.py for multi-task outputs
        
    def render_gesture_sequence(self, gesture_dict, output_path):
        # SMPL-X rendering with body part highlighting
        
    def create_comparison_video(self, inputs, outputs, ground_truth):
        # Side-by-side comparison visualization
```

## Implementation Priority

**Week 1**: Core model architecture + Audio fusion
**Week 2**: Gesture processor + Data loader 
**Week 3**: Training script + Loss functions
**Week 4**: Configuration + Single task experiments
**Week 5**: Multi-task training + Metrics
**Week 6**: Testing + Rendering + Documentation

## Key Dependencies & Requirements

1. **Pre-trained Models**:
   - RVQVAE checkpoints for 4 body parts
   - T5/LLaMA text encoder weights
   - WavLM audio encoder weights

2. **Dataset Preparation**:
   - BEAT2 dataset preprocessing
   - LibriTTS integration (optional)
   - Train/val/test splits

3. **Hardware Requirements**:
   - 4+ GPUs for multi-task training
   - ~32GB GPU memory per GPU
   - Fast storage for LMDB caches

## Risk Mitigation

1. **Memory Issues**: Gradient checkpointing, mixed precision training
2. **Training Instability**: Careful loss weighting, curriculum learning
3. **Data Alignment**: Robust masking, temporal synchronization
4. **Rendering Dependencies**: Optional rendering with fallback modes

## Success Metrics

1. **Technical**: Successful multi-task training convergence
2. **Quality**: Competitive gesture generation quality vs single-task baselines  
3. **Versatility**: All 4 task types (T2G, A2G, G2T, G2A) working
4. **Efficiency**: Reasonable training time and memory usage

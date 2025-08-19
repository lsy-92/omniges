# Omniges A2G Implementation

**Audio-to-Gesture Generation using BEAT2 Dataset**

This implementation focuses on the A2G (Audio to Gesture) task as the first milestone of the Omniges project, which aims to replace image streams with gesture streams in the OmniFlow pipeline.

## 🎯 Architecture Overview

### Core Components

1. **Audio Fusion Module**: WavLM + MFCC bidirectional cross-attention
   - WavLM (768D) + MFCC (128D) → Fused Audio (512D)
   - Gated fusion with residual connections

2. **MMDiT Backbone**: 3-modality transformer  
   - Text (1024D) + Audio (512D) + Gesture (415D)
   - 12 layers, 8 heads, 4 residual streams

3. **Gesture Processor**: 4x Pre-trained RVQVAE
   - Upper body: 78D (26 joints × 3)
   - Hands: 180D (60 joints × 3)
   - Lower+Trans: 57D (8 joints × 3 + translation + contacts)
   - Face: 100D (jaw 3D + expressions 100D)

4. **A2G Task Head**: Audio → Gesture code prediction
   - Cross-entropy loss on quantized gesture codes
   - Reconstruction loss on decoded gestures

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

### 3. Model Training

#### Quick Test (Dry Run)
```bash
# Test the pipeline with minimal data
accelerate launch \
    --num_processes 1 \
    omniges/scripts/train_a2g.py \
    --config configs/omniges/a2g_mmdit.yaml \
    --dry_run \
    --batch_size 2
```

#### Full Training (Multi-GPU)
```bash
# Run the full training script
./scripts/run_a2g_training.sh
```

#### Manual Training
```bash
# 6 GPU training (A100 x2 + A6000 ADA x4)
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes 6 \
    omniges/scripts/train_a2g.py \
    --config configs/omniges/a2g_mmdit.yaml \
    --output_dir ./results/a2g_$(date +%Y%m%d_%H%M%S) \
    --wandb_project omniges-a2g
```

### 4. Evaluation

```bash
# Evaluate trained model
python omniges/scripts/test_a2g.py \
    --config configs/omniges/a2g_mmdit.yaml \
    --checkpoint ./checkpoints/a2g/best.pth \
    --output_dir ./results/evaluation \
    --render \
    --num_samples 100
```

## 📁 File Structure

```
omniges/
├── models/
│   ├── omniges_a2g.py          # Main A2G model
│   └── __init__.py
├── dataloaders/
│   ├── beat_a2g_loader.py      # BEAT A2G dataset
│   └── __init__.py  
├── scripts/
│   ├── train_a2g.py            # Training script
│   └── test_a2g.py             # Evaluation script
└── README.md

configs/
├── omniges/
│   └── a2g_mmdit.yaml          # A2G training config
└── accelerate_config.yaml      # Multi-GPU config

scripts/
└── run_a2g_training.sh         # Training launcher

ckpt/                            # Pre-trained RVQVAE models
├── net_300000_upper.pth
├── net_300000_hands.pth
├── net_300000_lower.pth
└── net_300000_face.pth
```

## ⚙️ Configuration

Key configuration options in `configs/omniges/a2g_mmdit.yaml`:

```yaml
# Training setup
train:
  batch_size: 4        # Per GPU (total 24 across 6 GPUs)
  epochs: 100
  lr: 1.0e-4
  sequence_length: 64  # ~4.3 seconds at 15fps

# Audio processing  
audio:
  wavlm: true
  mfcc: true
  dim_wavlm: 768
  dim_other: 128
  fusion:
    dim_fuse: 512
    num_heads: 8
    num_layers: 2

# Gesture parts
gesture:
  use_parts: [upper, hands, lower_trans, face]
  rvq_ckpt:
    upper: ckpt/net_300000_upper.pth
    hands: ckpt/net_300000_hands.pth
    lower_trans: ckpt/net_300000_lower.pth
    face: ckpt/net_300000_face.pth
```

## 📊 Expected Performance

### Training Metrics
- **Loss Components**: 
  - Gesture CE Loss: Cross-entropy on quantized codes
  - Reconstruction Loss: MSE on decoded gestures
  - Velocity Loss: L1 on temporal smoothness
  
### Evaluation Metrics
- **FGD (Frechet Gesture Distance)**: Distribution similarity
- **L1 Diversity**: Motion variety preservation  
- **Reconstruction Error**: Point-wise accuracy

### Hardware Requirements
- **Memory**: ~8-12GB per GPU for batch_size=4
- **Training Time**: ~2-3 days for 100 epochs on 6 GPUs
- **Storage**: ~50GB for LMDB cache + checkpoints

## 🔧 Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce batch_size or sequence_length
2. **LMDB Cache**: Increase cache storage or use faster SSD
3. **Audio Alignment**: Check audio_sr and pose_fps consistency
4. **RVQVAE Loading**: Verify checkpoint paths and dimensions

### Debug Mode
```bash
# Enable debug logging
export OMNIGES_DEBUG=1
python omniges/scripts/train_a2g.py --config configs/omniges/a2g_mmdit.yaml --dry_run
```

### Monitoring
- **W&B Dashboard**: Real-time training metrics
- **TensorBoard**: Local metric visualization
- **Log Files**: Detailed training logs in `./logs/a2g/`

## 🛣️ Future Extensions

1. **T2G Task**: Add text-to-gesture generation
2. **G2A Task**: Add gesture-to-audio generation  
3. **G2T Task**: Add gesture-to-text generation
4. **Multi-task Learning**: Unified training across all tasks
5. **LibriTTS Integration**: Expand to speech datasets

## 📚 References

- **OmniFlow**: Base multimodal pipeline
- **MMDiT**: Multimodal Diffusion Transformer
- **BEAT2**: Large-scale gesture dataset
- **RVQVAE**: Residual Vector Quantized VAE for motion

#!/bin/bash

# Omniges Environment Setup Script
# Installs all dependencies and prepares the environment for A2G training

set -e

echo "🚀 Setting up Omniges A2G Environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $python_version"

if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 0 ]]; then
    echo "❌ Python 3.8+ required, found $python_version"
    exit 1
fi

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  No NVIDIA GPU detected. CPU training will be very slow."
fi

# Install core PyTorch and ML packages
echo ""
echo "📦 Installing PyTorch and core ML packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers and accelerate
pip install transformers>=4.30.0
pip install accelerate>=0.20.0
pip install diffusers>=0.18.0

# Install audio processing
echo "🎵 Installing audio processing packages..."
pip install librosa>=0.10.0
pip install soundfile
pip install textgrid

# Install scientific computing
echo "🔬 Installing scientific packages..."
pip install numpy>=1.21.0
pip install scipy
pip install pandas
pip install scikit-learn

# Install visualization and logging
echo "📊 Installing visualization and logging..."
pip install matplotlib
pip install seaborn
pip install wandb
pip install tensorboard
pip install loguru
pip install tqdm
pip install pyyaml

# Install 3D and rendering (optional)
echo "🎨 Installing 3D and rendering packages..."
pip install trimesh
pip install pyrender || echo "⚠️  pyrender installation failed (optional)"
pip install imageio
pip install moviepy

# Install SMPLX
echo "🤖 Installing SMPLX..."
pip install smplx

# Install additional utilities
pip install einops
pip install timm
pip install opencv-python

# Clone missing dependencies if needed
echo "📁 Checking project structure..."

# Create necessary directories
mkdir -p logs/a2g
mkdir -p checkpoints/a2g  
mkdir -p results/a2g
mkdir -p datasets/hub

# Verify RVQVAE checkpoints
echo ""
echo "🔍 Verifying RVQVAE checkpoints..."
if [ -d "ckpt" ]; then
    echo "✅ Found checkpoint directory"
    ls -la ckpt/net_300000_*.pth 2>/dev/null || echo "⚠️  Some RVQVAE checkpoints may be missing"
else
    echo "❌ Checkpoint directory not found. Please ensure RVQVAE models are available."
    exit 1
fi

# Verify BEAT dataset
echo ""
echo "🗂️  Verifying BEAT dataset..."
if [ -d "datasets/BEAT_SMPL" ]; then
    echo "✅ Found BEAT dataset directory"
    # Check for key components
    beat_dir="datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0"
    if [ -d "$beat_dir/beat_smplx_141" ] && [ -d "$beat_dir/wave16k" ]; then
        echo "✅ BEAT dataset appears complete"
    else
        echo "⚠️  BEAT dataset may be incomplete. Please verify:"
        echo "   - $beat_dir/beat_smplx_141/ (pose files)"
        echo "   - $beat_dir/wave16k/ (audio files)"
    fi
else
    echo "⚠️  BEAT dataset not found. Training will use dummy data."
fi

# Test imports
echo ""
echo "🧪 Testing Python imports..."
python3 -c "
import torch
import transformers
import librosa
import numpy as np
import yaml
print('✅ Core packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPUs detected: {torch.cuda.device_count()}')
"

# Test model creation
echo ""
echo "🔧 Testing model creation..."
cd omniges
python3 -c "
try:
    from models.omniges_a2g import create_omniges_a2g_model
    print('✅ Model import successful')
    
    # Test model creation (will fail if missing dependencies)
    model = create_omniges_a2g_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'✅ Model created successfully with {total_params:,} parameters')
except Exception as e:
    print(f'❌ Model creation failed: {e}')
    print('This may be due to missing RVQVAE checkpoints or other dependencies')
" || echo "⚠️  Model test failed - this is expected if checkpoints are missing"

cd ..

echo ""
echo "🎉 Environment setup completed!"
echo ""
echo "Next steps:"
echo "1. Verify BEAT dataset is complete"
echo "2. Run dry test: ./scripts/run_a2g_training.sh"
echo "3. Start full training if dry run succeeds"
echo ""
echo "For help, see omniges/README.md"

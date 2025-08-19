#!/bin/bash

# Omniges A2G Training Script
# Audio-to-Gesture generation training on BEAT2 dataset

set -e

# Configuration
PROJECT_ROOT="/home/lsy92/project/data/project/omniges"
CONFIG_PATH="configs/omniges/a2g_mmdit.yaml"
OUTPUT_DIR="./results/a2g_$(date +%Y%m%d_%H%M%S)"
NUM_GPUS=6  # A100 x2 + A6000 ADA x4

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO

# Check if running in the correct directory
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "./logs/a2g"
mkdir -p "./checkpoints/a2g"

# Copy config to output directory for reproducibility
cp "$CONFIG_PATH" "$OUTPUT_DIR/config.yaml"

echo "=== Omniges A2G Training ==="
echo "Project root: $PROJECT_ROOT"
echo "Config: $CONFIG_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS ($(echo $CUDA_VISIBLE_DEVICES | tr ',' ' '))"
echo "==========================="

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits

# Dry run first (optional)
echo ""
echo "=== Running Dry Run ==="
accelerate launch \
    --config_file ./configs/accelerate_config.yaml \
    --num_processes $NUM_GPUS \
    --main_process_port 29500 \
    omniges/scripts/train_a2g.py \
    --config $CONFIG_PATH \
    --output_dir "${OUTPUT_DIR}/dry_run" \
    --dry_run \
    --wandb_project "omniges-a2g-test"

if [ $? -eq 0 ]; then
    echo "✅ Dry run completed successfully!"
else
    echo "❌ Dry run failed. Please check the error messages above."
    exit 1
fi

# Ask user confirmation for full training
echo ""
read -p "Dry run successful. Start full training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled by user."
    exit 0
fi

# Full training
echo ""
echo "=== Starting Full Training ==="
accelerate launch \
    --config_file ./configs/accelerate_config.yaml \
    --num_processes $NUM_GPUS \
    --main_process_port 29500 \
    omniges/scripts/train_a2g.py \
    --config $CONFIG_PATH \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "omniges-a2g" \
    --gradient_accumulation_steps 2

echo ""
echo "=== Training Completed ==="
echo "Results saved to: $OUTPUT_DIR"
echo "Logs available at: ./logs/a2g"
echo "Checkpoints saved to: ./checkpoints/a2g"

# Optional: Run evaluation
read -p "Run evaluation on test set? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running evaluation..."
    
    python omniges/scripts/test_a2g.py \
        --config $CONFIG_PATH \
        --checkpoint "$OUTPUT_DIR/best.pth" \
        --output_dir "${OUTPUT_DIR}/eval" \
        --render
        
    echo "Evaluation completed. Results in ${OUTPUT_DIR}/eval"
fi

echo "All done! 🎉"

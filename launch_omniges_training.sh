#!/bin/bash
# Omniges Multi-Task Training Launch Script
# 전체 6개 태스크 동시 학습: t2g, g2t, a2g, g2a, t2a, a2t

echo "🚀 OMNIGES MULTI-TASK TRAINING"
echo "Tasks: t2g, g2t, a2g, g2a, t2a, a2t"
echo "Model: OmnigesFlowTransformerModel"
echo "Data: BEAT2 dataset"

# Environment setup
conda activate gesturelsm
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use available GPUs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Actual OmniFlow checkpoint paths
OMNIFLOW_CHECKPOINT="/home/lsy92/project/data/project/omniges/checkpoint/OmniFlow-v0.5"
TEXT_VAE_PATH="/home/lsy92/project/data/project/omniges/checkpoint/OmniFlow-v0.5/text_vae"
TOKENIZER_PATH="/home/lsy92/project/data/project/omniges/checkpoint/OmniFlow-v0.5/vae_tokenizer"

if [ ! -d "$OMNIFLOW_CHECKPOINT" ]; then
    echo "⚠️  OmniFlow checkpoint not found: $OMNIFLOW_CHECKPOINT"
    echo "   Update the path in this script or use A2G training instead"
    echo ""
    echo "🔄 Fallback: Running verified A2G training..."
    python train_a2g_adaptive.py --epochs 20
    exit 0
fi

# Launch multi-task training
echo "🎯 Launching multi-task training..."

accelerate launch \
    --config_file configs/accelerate_config.yaml \
    omniges/scripts/train_omniges.py \
    --pretrained_model_name_or_path ${OMNIFLOW_CHECKPOINT} \
    --beat_config_path configs/shortcut_rvqvae_128.yaml \
    --rvqvae_checkpoints ./ckpt/ \
    --text_vae ${TEXT_VAE_PATH} \
    --tokenizer ${TOKENIZER_PATH} \
    --output_dir ./results/omniges_multitask \
    --resolution 512 \
    --seq_length 128 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 50 \
    --learning_rate 1e-4 \
    --mixed_precision bf16 \
    --use_ema \
    --gradient_checkpointing \
    --validation_prompt "A person waving hello" \
    --val_every 500 \
    --checkpointing_steps 1000 \
    --report_to wandb \
    --weighting_scheme logit_normal \
    --uniform_flow \
    --allow_tf32 \
    --seed 42

echo "✅ Training completed!"

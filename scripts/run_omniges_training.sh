#!/bin/bash
# Omniges Multi-Task Training Script
# Supports: t2g, g2t, a2g, g2a, t2a, a2t

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  # Use available GPUs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Training configuration
OMNIFLOW_MODEL_PATH="./path/to/omniflow"  # Replace with actual OmniFlow checkpoint path
RVQVAE_CHECKPOINTS="./ckpt/"  # RVQVAE checkpoints directory
TEXT_VAE_PATH="./path/to/text_vae"  # Replace with actual text VAE path
TOKENIZER_PATH="./path/to/tokenizer"  # Replace with actual tokenizer path
OUTPUT_DIR="./results/omniges_training"
BEAT_CONFIG="configs/shortcut_rvqvae_128.yaml"

echo "🚀 Starting Omniges Multi-Task Training"
echo "Supported tasks: t2g, g2t, a2g, g2a, t2a, a2t"
echo "Model: OmnigesFlowTransformerModel (Image→Gesture)"
echo "Data: BEAT2 dataset"

accelerate launch \
    --config_file configs/accelerate_config.yaml \
    omniges/scripts/train_omniges.py \
    --pretrained_model_name_or_path ${OMNIFLOW_MODEL_PATH} \
    --beat_config_path ${BEAT_CONFIG} \
    --rvqvae_checkpoints ${RVQVAE_CHECKPOINTS} \
    --text_vae ${TEXT_VAE_PATH} \
    --tokenizer ${TOKENIZER_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --resolution 512 \
    --seq_length 128 \
    --train_batch_size 4 \
    --num_train_epochs 100 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 2 \
    --mixed_precision bf16 \
    --use_ema \
    --ema_momentum 0.9999 \
    --gradient_checkpointing \
    --validation_prompt "A person waving hello" \
    --val_every 500 \
    --checkpointing_steps 1000 \
    --report_to wandb \
    --weighting_scheme logit_normal \
    --uniform_flow \
    --allow_tf32 \
    --seed 42

echo "✅ Training completed! Check results in ${OUTPUT_DIR}"

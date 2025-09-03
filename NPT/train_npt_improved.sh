#!/bin/bash

# Training script with improved loss functions for better convergence

# Default configuration with improved loss
echo "Training NPT with improved loss function..."

python pretrain_npt_improved.py \
    --model_name "meta-llama/Llama-3.2-1B" \
    --adapter_rank 16 \
    --share_embeddings \
    --streaming \
    --num_samples 10000 \
    --max_length 1024 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --warmup_steps 1000 \
    --max_steps 5000 \
    --save_steps 500 \
    --log_steps 10 \
    --output_dir "./outputs/npt-improved-1B" \
    --loss_type "improved_npt" \
    --mse_weight 0.8 \
    --cosine_weight 0.2 \
    --regularization_lambda 0.01 \
    --gradient_penalty_lambda 0.001 \
    --distill_temperature 3.0 \
    --use_adaptive_weights \
    --use_gradient_penalty \
    --modulation_scale 0.1 \
    $@

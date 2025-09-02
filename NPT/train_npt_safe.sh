#!/bin/bash

# Safe NPT training script with all fixes applied

echo "Starting NPT Safe Training..."
echo "This script uses all the fixes for numerical stability"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check for command line argument
MODE=${1:-"debug"}

if [ "$MODE" == "debug" ]; then
    echo "Running in debug mode (quick test)..."
    python pretrain_npt_safe.py \
        --model_name meta-llama/Llama-3.1-8B \
        --adapter_rank 8 \
        --use_quantization \
        --share_embeddings \
        --safe_mode \
        --streaming \
        --num_samples 100 \
        --batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-5 \
        --regularization_lambda 0.001 \
        --warmup_steps 10 \
        --max_steps 20 \
        --output_dir ./outputs/npt-debug-safe \
        --log_steps 1 \
        --save_steps 20
else
    echo "Running full training..."
    python pretrain_npt_safe.py \
        --model_name meta-llama/Llama-3.1-8B \
        --adapter_rank 16 \
        --use_quantization \
        --share_embeddings \
        --safe_mode \
        --streaming \
        --batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-5 \
        --regularization_lambda 0.01 \
        --num_epochs 1 \
        --warmup_steps 200 \
        --output_dir ./outputs/npt-safe-pretrained \
        --log_steps 10 \
        --save_steps 500 \
        --use_wandb
fi

echo ""
echo "Training complete!"
echo ""
echo "Tips:"
echo "1. Monitor the loss values - they should decrease over time"
echo "2. Watch grad_norm - should be < 10, ideally < 1"
echo "3. If you see NaN, the script will skip that batch and continue"
echo "4. Check logs in the output directory for details"

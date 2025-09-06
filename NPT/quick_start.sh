#!/bin/bash

# NPT Quick Start Script
# This script provides a simplified way to run NPT training with default settings
# 
# NOTE: This implementation includes critical architectural improvements:
# - Per-token dynamic modulation (each token gets unique effects)
# - Efficient regularization without redundant computations
# - Stable training through proper residual connections
# See ARCHITECTURAL_IMPROVEMENTS.md for details

set -e  # Exit on error

echo "==============================================="
echo "NPT (NLP NeuroPlastic Transformer) Quick Start"
echo "==============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for Hugging Face login
echo "Checking Hugging Face authentication..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "Please login to Hugging Face to access Llama3 models:"
    huggingface-cli login
fi

# Parse command line arguments
PHASE=${1:-"pretrain"}
MODE=${2:-"debug"}

if [ "$PHASE" == "pretrain" ]; then
    echo -e "\n Starting Phase 1: Equivalence Pre-training..."
    
    if [ "$MODE" == "debug" ]; then
        echo "Running in debug mode (limited samples)..."
        python pretrain_npt_safe.py \
            --adapter_rank 8 \
            --use_quantization \
            --safe_mode \
            --streaming \
            --num_samples 100 \
            --batch_size 1 \
            --gradient_accumulation_steps 4 \
            --learning_rate 5e-5 \
            --max_steps 10 \
            --output_dir ./outputs/debug-pretrain \
            --log_steps 1 \
            --save_steps 10
    else
        echo "Running full pre-training..."
python pretrain_npt_safe.py \
    --model_name meta-llama/Llama-3.1-8B \
  --use_fp16 \
  --mixed_precision bf16 \
  --adapter_rank 16 \
  --modulation_scale 0.1 \
  --regularization_lambda 0.01 \
  --share_embeddings \
  --batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --scheduler_type cosine \
  --warmup_steps 1000 \
  --max_length 2048 \
  --num_epochs 1 \
  --max_steps 20000 \
  --log_steps 10 \
  --save_steps 1000 \
  --streaming \
  --use_wandb

python pretrain_npt_safe.py \
  --model_name "meta-llama/Llama-3.1-8B" \
  --adapter_rank 128 \
  --learning_rate 1e-4 \
  --regularization_lambda 0.005 \
  --modulation_scale 0.1 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_grad_norm 1.0 \
  --warmup_steps 200 \
  --share_embeddings \
  --mixed_precision "bf16" \
  --max_length 1024 \
  --max_steps 10000 \
  --streaming \
  --use_wandb

########## best
python pretrain_npt_improved.py \
    --model_name "meta-llama/Llama-3.1-8B" \
    --adapter_rank 256 \
    --regularization_lambda 0.01 \
    --modulation_scale 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_grad_norm 1.0 \
    --share_embeddings \
    --mixed_precision "bf16" \
    --max_length 512 \
    --max_steps 10000 \
    --streaming \
    --use_wandb \
    --loss_type "improved_npt" \
    --use_adaptive_weights \
    --use_gradient_penalty \
    --mse_weight 0.5 \
    --cosine_weight 0.5 \
    --warmup_steps 1000 \
    --keep_only_last_checkpoint

### 1. **Conservative (Most Stable)**
Best for initial experimentation and debugging.

```bash
python pretrain_npt_safe.py \
    --model_name "meta-llama/Llama-3.2-1B" \
    --adapter_rank 8 \
    --learning_rate 5e-5 \
    --regularization_lambda 0.1 \
    --modulation_scale 0.01 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_grad_norm 0.5 \
    --warmup_steps 500 \
    --use_layer_wise_loss_scaling \
    --mixed_precision "no" \
    --max_steps 5000
```

### 2. **Balanced (Recommended)**
Good balance between training speed and stability.

```bash
python pretrain_npt_safe.py \
    --model_name "meta-llama/Llama-3.1-8B" \
    --adapter_rank 16 \
    --learning_rate 1e-4 \
    --regularization_lambda 0.01 \
    --modulation_scale 0.1 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_grad_norm 1.0 \
    --warmup_steps 200 \
    --use_layer_wise_loss_scaling \
    --share_embeddings \
    --mixed_precision "bf16" \
    --max_steps 10000 \
    --streaming \
    --use_wandb
```

### 3. **Aggressive (Faster Convergence)**
For when you have good hardware and want faster training.

```bash
python pretrain_npt_safe.py \
    --model_name "meta-llama/Llama-3.1-8B" \
    --adapter_rank 32 \
    --learning_rate 2e-4 \
    --regularization_lambda 0.005 \
    --modulation_scale 0.2 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_grad_norm 1.0 \
    --warmup_steps 100 \
    --mixed_precision "bf16" \
    --max_steps 20000 \
    --streaming

```

    fi

    python pretrain_npt_safe.py \
    --model_name "meta-llama/Llama-3.1-8B" \
    --adapter_rank 32 \
    --learning_rate 2e-4 \
    --regularization_lambda 0.005 \
    --modulation_scale 0.2 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_grad_norm 1.0 \
    --warmup_steps 100 \
    --mixed_precision "bf16" \
    --max_steps 20000
    
elif [ "$PHASE" == "finetune" ]; then
    echo -e "\nStarting Phase 2: Functional Fine-tuning..."
    
    # Check if pretrained checkpoint exists
    if [ "$MODE" == "debug" ]; then
        CHECKPOINT_PATH="./outputs/debug-pretrain/checkpoint-final"
    else
        CHECKPOINT_PATH="./outputs/npt-pretrained/checkpoint-best"
    fi
    
    if [ ! -d "$CHECKPOINT_PATH" ]; then
        echo "Error: Pretrained checkpoint not found at $CHECKPOINT_PATH"
        echo "Please run Phase 1 pre-training first: ./quick_start.sh pretrain"
        exit 1
    fi
    
    if [ "$MODE" == "debug" ]; then
        echo "Running in debug mode (limited samples)..."
        python finetune_npt.py \
            --checkpoint_path $CHECKPOINT_PATH \
            --use_quantization \
            --use_fp16 \
            --dataset_name tatsu-lab/alpaca \
            --train_split train \
            --num_train_samples 100 \
            --batch_size 1 \
            --gradient_accumulation_steps 4 \
            --learning_rate 2e-5 \
            --max_steps 10 \
            --output_dir ./outputs/debug-finetune \
            --log_steps 1 \
            --save_steps 10
    else
        echo "Running full fine-tuning..."
        python finetune_npt.py \
            --checkpoint_path $CHECKPOINT_PATH \
            --use_quantization \
            --use_fp16 \
            --dataset_name HuggingFaceH4/ultrachat_200k \
            --batch_size 1 \
            --gradient_accumulation_steps 8 \
            --learning_rate 2e-5 \
            --num_epochs 3 \
            --warmup_steps 100 \
            --output_dir ./outputs/npt-finetuned \
            --log_steps 10 \
            --eval_steps 500 \
            --save_steps 1000 \
            --use_wandb
    fi
    
elif [ "$PHASE" == "test" ]; then
    echo -e "\nRunning NPT implementation tests..."
    python test_npt_layer.py
    
elif [ "$PHASE" == "evaluate" ]; then
    echo -e "\nEvaluating NPT model..."
    
    MODEL_PATH=${3:-"./outputs/npt-finetuned/checkpoint-best"}
    
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Error: Model checkpoint not found at $MODEL_PATH"
        exit 1
    fi
    
    python evaluate_npt.py \
        --model_path $MODEL_PATH \
        --tasks perplexity instruction_following \
        --max_samples 100 \
        --output_path ./outputs/evaluation_results.json
        
else
    echo "Usage: ./quick_start.sh [phase] [mode]"
    echo ""
    echo "Phases:"
    echo "  pretrain  - Run Phase 1 equivalence pre-training"
    echo "  finetune  - Run Phase 2 functional fine-tuning"
    echo "  test      - Run implementation tests"
    echo "  evaluate  - Evaluate trained model"
    echo ""
    echo "Modes:"
    echo "  debug     - Quick debug run with minimal data"
    echo "  full      - Full training run (default)"
    echo ""
    echo "Examples:"
    echo "  ./quick_start.sh pretrain debug    # Quick pre-training test"
    echo "  ./quick_start.sh pretrain          # Full pre-training"
    echo "  ./quick_start.sh finetune          # Full fine-tuning"
    echo "  ./quick_start.sh test              # Run tests"
    exit 1
fi

echo -e "\nDone!"

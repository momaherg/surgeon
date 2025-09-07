#!/bin/bash

# NPT Equivalence Pre-training Launcher Script
# This script sets up the environment and launches the NPT training

echo "================================================"
echo "Neuro-Plastic Transformer (NPT) Training Launcher"
echo "Phase 1: Equivalence Pre-training"
echo "================================================"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Adjust based on your GPU setup
export WANDB_PROJECT="npt-equivalence-pretraining"
export TOKENIZERS_PARALLELISM=false

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p outputs
mkdir -p cache
mkdir -p checkpoints
mkdir -p logs

# Login to Hugging Face (if needed)
echo "Checking Hugging Face login..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "Please login to Hugging Face to download model weights:"
    huggingface-cli login
fi

# Login to WandB (if needed)
echo "Checking WandB login..."
if ! wandb login --verify &> /dev/null; then
    echo "Please login to WandB for experiment tracking:"
    wandb login
fi

# Launch training with accelerate
echo ""
echo "Starting NPT equivalence pre-training..."
echo "Configuration: config.yaml"
echo ""

# Single GPU training
accelerate launch --num_processes=1 \
    train_npt_equivalence.py \
    --config config.yaml \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Training completed! Check the logs/ directory for output."

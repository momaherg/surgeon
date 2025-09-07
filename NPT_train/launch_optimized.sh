#!/bin/bash

# Launch script for memory-optimized NPT training

echo "================================================"
echo "NPT Optimized Training Launcher"
echo "================================================"

# Set environment variables for memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements if needed
pip install -r requirements.txt

# Create directories
mkdir -p outputs cache checkpoints logs

echo ""
echo "Starting optimized NPT training..."
echo ""

# Choose configuration based on available GPU memory
if [ "$1" == "small" ]; then
    echo "Using small model configuration..."
    CONFIG="config_small.yaml"
elif [ "$1" == "optimized" ]; then
    echo "Using optimized configuration for large models..."
    CONFIG="config_optimized.yaml"
else
    echo "Using default optimized configuration..."
    CONFIG="config_optimized.yaml"
fi

# Run training with optimizations
python3 train_npt_optimized.py \
    --config  \
    --use-optimized \
    2>&1 | tee logs/training_optimized_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Training completed! Check logs/ for details."

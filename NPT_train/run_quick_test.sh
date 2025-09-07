#!/bin/bash

# Quick test script to verify training works

echo "Running quick NPT training test..."
echo "================================="

# Set environment for CPU testing (to avoid GPU issues during testing)
export CUDA_VISIBLE_DEVICES=""

# Run with small config
echo "Testing with small GPT-2 model..."
python3 train_npt_optimized.py --config config_small.yaml --use-optimized

echo ""
echo "Test completed! Check output above for any errors."

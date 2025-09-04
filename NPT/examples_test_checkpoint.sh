#!/bin/bash
# Example usage of fixed test_checkpoint.py

echo "Example 1: Test NPT model only"
echo "python test_checkpoint.py --checkpoint_path ./outputs/npt-improved-1B/checkpoint-500"
echo ""

echo "Example 2: Compare NPT with auto-detected base model"
echo "python test_checkpoint.py --checkpoint_path ./outputs/npt-improved-1B/checkpoint-500 --with_comparison"
echo ""

echo "Example 3: Compare with specific base model"
echo "python test_checkpoint.py --checkpoint_path ./outputs/npt-improved-1B/checkpoint-500 --with_comparison --base_model meta-llama/Llama-3.2-1B"
echo ""

echo "Example 4: Custom prompt with comparison"
echo "python test_checkpoint.py --checkpoint_path ./outputs/npt-improved-1B/checkpoint-500 --prompt \"The meaning of life is\" --max_new_tokens 100 --with_comparison"


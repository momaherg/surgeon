#!/usr/bin/env python3
"""
Test script to verify training stability improvements
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model import convert_llama_to_npt
import matplotlib.pyplot as plt
import numpy as np


def test_loss_computation():
    """Test the loss computation with and without normalization"""
    print("Testing loss computation stability...")
    
    # Create dummy hidden states
    batch_size, seq_len, hidden_dim = 2, 10, 768
    
    # Teacher outputs (stable)
    teacher_hidden = torch.randn(batch_size, seq_len, hidden_dim) * 2.0
    
    # Student outputs with small perturbations
    losses_normalized = []
    losses_direct = []
    perturbation_scales = np.logspace(-3, 0, 50)  # 0.001 to 1.0
    
    for scale in perturbation_scales:
        student_hidden = teacher_hidden + torch.randn_like(teacher_hidden) * scale
        
        # Normalized MSE (old approach)
        teacher_norm = teacher_hidden.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        student_norm = student_hidden.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        teacher_normalized = teacher_hidden / teacher_norm
        student_normalized = student_hidden / student_norm
        loss_normalized = nn.functional.mse_loss(student_normalized, teacher_normalized)
        losses_normalized.append(loss_normalized.item())
        
        # Direct MSE (new approach)
        loss_direct = nn.functional.mse_loss(student_hidden, teacher_hidden)
        losses_direct.append(loss_direct.item())
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.loglog(perturbation_scales, losses_normalized, 'r-', label='Normalized MSE (old)', linewidth=2)
    plt.loglog(perturbation_scales, losses_direct, 'b-', label='Direct MSE (new)', linewidth=2)
    plt.xlabel('Perturbation Scale')
    plt.ylabel('MSE Loss')
    plt.title('Loss Behavior: Normalized vs Direct MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_comparison.png')
    print("Saved loss comparison plot to loss_comparison.png")
    
    # Compute loss sensitivity
    normalized_sensitivity = np.std(losses_normalized) / np.mean(losses_normalized)
    direct_sensitivity = np.std(losses_direct) / np.mean(losses_direct)
    
    print(f"\nLoss Sensitivity (lower is more stable):")
    print(f"  Normalized MSE: {normalized_sensitivity:.4f}")
    print(f"  Direct MSE: {direct_sensitivity:.4f}")
    print(f"  Improvement: {normalized_sensitivity/direct_sensitivity:.2f}x more stable")


def test_modulation_scaling():
    """Test the effect of different modulation scales"""
    print("\n\nTesting modulation scaling effects...")
    
    # Create a small test model
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading {model_name} for testing...")
    
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test different modulation scales
    modulation_scales = [0.01, 0.05, 0.1, 0.2, 0.5]
    test_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(test_text, return_tensors="pt", padding=True)
    
    print("\nTesting different modulation scales:")
    for scale in modulation_scales:
        # Create NPT model with specific scale
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        adapter_config = {
            'r': 16,
            'd_model': config.hidden_size,
            'd_ffn': config.intermediate_size,
            'compute_dtype': torch.float32,
            'modulation_scale': scale
        }
        
        npt_model = convert_llama_to_npt(model, adapter_config)
        npt_model.eval()
        
        # Forward pass
        with torch.no_grad():
            outputs = npt_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Compute average hidden state magnitude
            avg_magnitude = sum(h.abs().mean().item() for h in hidden_states) / len(hidden_states)
            
            print(f"  Scale {scale:.2f}: Avg hidden magnitude = {avg_magnitude:.4f}")
        
        del model, npt_model
        torch.cuda.empty_cache()


def generate_training_recommendations():
    """Generate recommendations for stable training"""
    print("\n\n=== Training Recommendations ===")
    print("\n1. MSE Loss Computation:")
    print("   - Use direct MSE without normalization for stable gradients")
    print("   - Enable --use_layer_wise_loss_scaling for models with varying layer scales")
    print()
    print("2. Modulation Scaling:")
    print("   - Start with --modulation_scale 0.1 (default)")
    print("   - Increase to 0.2-0.5 if MSE loss plateaus")
    print("   - Decrease to 0.01-0.05 if training is unstable")
    print()
    print("3. Learning Rate:")
    print("   - Use --learning_rate 1e-4 or lower for stability")
    print("   - Consider warmup with --warmup_steps 100-500")
    print()
    print("4. Regularization:")
    print("   - Set --regularization_lambda 0.01 to encourage efficient weight updates")
    print("   - Increase if model overfits or produces large weight deltas")
    print()
    print("5. Example stable training command:")
    print("   python pretrain_npt_safe.py \\")
    print("     --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\")
    print("     --learning_rate 5e-5 \\")
    print("     --modulation_scale 0.1 \\")
    print("     --regularization_lambda 0.01 \\")
    print("     --use_layer_wise_loss_scaling \\")
    print("     --warmup_steps 200 \\")
    print("     --max_steps 1000")


if __name__ == "__main__":
    # Test loss computation approaches
    test_loss_computation()
    
    # Test modulation scaling (optional - requires model download)
    try:
        test_modulation_scaling()
    except Exception as e:
        print(f"\nSkipping modulation scaling test due to: {e}")
    
    # Generate recommendations
    generate_training_recommendations()

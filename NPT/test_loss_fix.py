#!/usr/bin/env python3
"""
Quick test to demonstrate the MSE loss fix
"""

import torch
import torch.nn as nn
import numpy as np


def demonstrate_loss_issue():
    """Show why normalized MSE causes oscillations"""
    print("=== Demonstrating MSE Loss Fix ===\n")
    
    # Create example hidden states
    batch_size, seq_len, hidden_dim = 2, 10, 768
    
    # Teacher outputs (what we want to match)
    teacher_hidden = torch.randn(batch_size, seq_len, hidden_dim) * 2.0
    
    print("1. Testing with small perturbation (0.01 scale):")
    perturbation = torch.randn_like(teacher_hidden) * 0.01
    student_hidden = teacher_hidden + perturbation
    
    # OLD APPROACH: Normalized MSE
    teacher_norm = teacher_hidden.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    student_norm = student_hidden.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    teacher_normalized = teacher_hidden / teacher_norm
    student_normalized = student_hidden / student_norm
    loss_normalized = nn.functional.mse_loss(student_normalized, teacher_normalized)
    
    # NEW APPROACH: Direct MSE
    loss_direct = nn.functional.mse_loss(student_hidden, teacher_hidden)
    
    print(f"   Normalized MSE (OLD): {loss_normalized.item():.6f}")
    print(f"   Direct MSE (NEW):     {loss_direct.item():.6f}")
    print(f"   Ratio (NEW/OLD):      {loss_direct.item()/loss_normalized.item():.2f}x\n")
    
    print("2. Testing with varying perturbations:")
    print("   Scale  | Normalized MSE | Direct MSE    | Gradient Signal")
    print("   -------|----------------|---------------|----------------")
    
    for scale in [0.001, 0.01, 0.1, 1.0]:
        student_hidden = teacher_hidden + torch.randn_like(teacher_hidden) * scale
        
        # Normalized MSE
        teacher_norm = teacher_hidden.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        student_norm = student_hidden.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        teacher_normalized = teacher_hidden / teacher_norm
        student_normalized = student_hidden / student_norm
        loss_normalized = nn.functional.mse_loss(student_normalized, teacher_normalized)
        
        # Direct MSE
        loss_direct = nn.functional.mse_loss(student_hidden, teacher_hidden)
        
        # Gradient signal strength (higher is better for learning)
        grad_signal = loss_direct.item() / loss_normalized.item()
        
        print(f"   {scale:5.3f} | {loss_normalized.item():14.6f} | {loss_direct.item():13.6f} | {grad_signal:6.2f}x stronger")
    
    print("\n3. Key Insights:")
    print("   - Normalized MSE removes magnitude information")
    print("   - Direct MSE provides consistent gradient signal proportional to error")
    print("   - This prevents oscillations and enables stable training")
    
    print("\n4. Additional Fix: Configurable Modulation Scale")
    print("   - Changed hard-coded 0.1 to configurable --modulation_scale")
    print("   - Allows tuning based on model size and training dynamics")
    print("   - Recommended values: 0.05-0.2 for stable training")


if __name__ == "__main__":
    demonstrate_loss_issue()
    
    print("\n=== Training Command with Fixes ===")
    print("python pretrain_npt_safe.py \\")
    print("  --model_name meta-llama/Llama-3.1-8B \\")
    print("  --learning_rate 5e-5 \\")
    print("  --modulation_scale 0.1 \\")
    print("  --regularization_lambda 0.01 \\")
    print("  --warmup_steps 200")

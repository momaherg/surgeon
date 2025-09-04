#!/usr/bin/env python3
"""
Test and compare different NPT adapter initialization strategies.

This script demonstrates how restrictive the current initialization is
and shows the benefits of alternative approaches.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import argparse

from model.npt_layer import NPTAdapter
from improved_initialization import (
    ImprovedNPTAdapter, 
    analyze_initialization,
    compare_initialization_strategies
)


def visualize_weight_distributions(d_model: int = 4096, d_ffn: int = 11008, r: int = 16):
    """
    Visualize the weight distributions for different initialization strategies.
    """
    strategies = {
        "zero (current)": "zero",
        "xavier": "xavier", 
        "kaiming": "kaiming",
        "lora": "lora",
        "adaptive": "adaptive",
        "orthogonal": "orthogonal"
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, strategy) in enumerate(strategies.items()):
        # Create adapter
        if strategy == "zero":
            # Use original implementation for comparison
            adapter = NPTAdapter(d_model, d_ffn, r)
        else:
            adapter = ImprovedNPTAdapter(d_model, d_ffn, r, init_strategy=strategy)
        
        # Get weight magnitudes
        a_weights = adapter.A_proj.weight.data.flatten().cpu().numpy()
        b_model_weights = adapter.B_model.weight.data.flatten().cpu().numpy()
        b_ffn_weights = adapter.B_ffn.weight.data.flatten().cpu().numpy()
        
        # Plot histogram
        ax = axes[idx]
        ax.hist(a_weights, bins=50, alpha=0.5, label='A_proj', density=True)
        ax.hist(b_model_weights, bins=50, alpha=0.5, label='B_model', density=True)
        ax.hist(b_ffn_weights, bins=50, alpha=0.5, label='B_ffn', density=True)
        
        ax.set_title(f'{name} Initialization')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"A std: {np.std(a_weights):.4f}\n"
        stats_text += f"B_model std: {np.std(b_model_weights):.4f}\n"
        stats_text += f"B_ffn std: {np.std(b_ffn_weights):.4f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('initialization_distributions.png', dpi=150)
    print("Saved weight distribution visualization to 'initialization_distributions.png'")


def test_gradient_flow(d_model: int = 4096, d_ffn: int = 11008, r: int = 16):
    """
    Test gradient flow through different initializations.
    """
    print("\n=== Testing Gradient Flow ===")
    
    strategies = ["zero", "xavier", "kaiming", "lora", "adaptive", "orthogonal"]
    results = []
    
    for strategy in strategies:
        if strategy == "zero":
            adapter = NPTAdapter(d_model, d_ffn, r)
        else:
            adapter = ImprovedNPTAdapter(d_model, d_ffn, r, init_strategy=strategy)
        
        # Create dummy input and target
        batch_size = 4
        seq_len = 128
        input_tensor = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        output = adapter(input_tensor)
        
        # Create a dummy loss (we'll use the modulation magnitude)
        vector_model = output['vector_model']
        vector_ffn = output['vector_ffn']
        
        # Compute modulation effect magnitude
        modulation_magnitude = torch.mean(
            torch.norm(vector_model, dim=-1) * torch.norm(vector_ffn, dim=-1)
        )
        
        # Backward pass
        modulation_magnitude.backward()
        
        # Analyze gradients
        grad_stats = {}
        for name, param in adapter.named_parameters():
            if param.grad is not None:
                grad = param.grad
                grad_stats[name] = {
                    'mean': grad.abs().mean().item(),
                    'std': grad.std().item(),
                    'max': grad.abs().max().item(),
                    'zero_ratio': (grad.abs() < 1e-8).float().mean().item()
                }
        
        results.append({
            'strategy': strategy,
            'initial_effect': modulation_magnitude.item(),
            'gradients': grad_stats
        })
        
        # Clear gradients
        adapter.zero_grad()
    
    # Print results
    print("\nInitialization Strategy Comparison:")
    print("-" * 80)
    
    for result in results:
        print(f"\n{result['strategy'].upper()}:")
        print(f"  Initial modulation magnitude: {result['initial_effect']:.6f}")
        print("  Gradient statistics:")
        
        for param_name, stats in result['gradients'].items():
            print(f"    {param_name}:")
            print(f"      Mean gradient: {stats['mean']:.6f}")
            print(f"      Gradient std: {stats['std']:.6f}")
            print(f"      Max gradient: {stats['max']:.6f}")
            print(f"      Zero gradient ratio: {stats['zero_ratio']:.2%}")


def test_convergence_speed():
    """
    Test how quickly different initializations converge during training.
    """
    print("\n=== Testing Convergence Speed ===")
    
    # Simple test: train to match a target modulation pattern
    d_model, d_ffn, r = 512, 1024, 8  # Smaller for faster testing
    strategies = ["zero", "adaptive", "lora"]
    
    results = {}
    
    for strategy in strategies:
        if strategy == "zero":
            adapter = NPTAdapter(d_model, d_ffn, r)
        else:
            adapter = ImprovedNPTAdapter(d_model, d_ffn, r, init_strategy=strategy)
        
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
        
        # Create target pattern
        target_input = torch.randn(1, 1, d_model)
        with torch.no_grad():
            # Create a reasonable target by using an already initialized adapter
            target_adapter = ImprovedNPTAdapter(d_model, d_ffn, r, init_strategy="xavier")
            target_output = target_adapter(target_input)
            target_v_model = target_output['vector_model'].detach()
            target_v_ffn = target_output['vector_ffn'].detach()
        
        losses = []
        
        # Train for a few steps
        for step in range(100):
            optimizer.zero_grad()
            
            output = adapter(target_input)
            v_model = output['vector_model']
            v_ffn = output['vector_ffn']
            
            # MSE loss
            loss = nn.MSELoss()(v_model, target_v_model) + nn.MSELoss()(v_ffn, target_v_ffn)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        results[strategy] = losses
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    for strategy, losses in results.items():
        plt.plot(losses, label=strategy, linewidth=2)
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Convergence Speed Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('convergence_comparison.png', dpi=150)
    print("Saved convergence comparison to 'convergence_comparison.png'")
    
    # Print summary
    print("\nConvergence Summary:")
    for strategy, losses in results.items():
        print(f"  {strategy}: Initial loss = {losses[0]:.6f}, Final loss = {losses[-1]:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Test NPT initialization strategies")
    parser.add_argument("--d_model", type=int, default=4096, help="Model dimension")
    parser.add_argument("--d_ffn", type=int, default=11008, help="FFN dimension") 
    parser.add_argument("--r", type=int, default=16, help="Adapter rank")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("NPT Adapter Initialization Analysis")
    print("=" * 80)
    
    # Compare initialization strategies
    print("\n1. Statistical Comparison of Initialization Strategies")
    print(compare_initialization_strategies(args.d_model, args.d_ffn, args.r))
    
    # Test gradient flow
    print("\n2. Gradient Flow Analysis")
    test_gradient_flow(args.d_model, args.d_ffn, args.r)
    
    # Test convergence
    print("\n3. Convergence Speed Test")
    test_convergence_speed()
    
    # Create visualizations if requested
    if args.visualize:
        print("\n4. Creating Visualizations")
        visualize_weight_distributions(args.d_model, args.d_ffn, args.r)
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

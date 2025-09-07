#!/usr/bin/env python3
"""
Test script to verify teacher forcing implementation in NPT training.
This script demonstrates the difference between with and without teacher forcing.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model import convert_llama_to_npt
import matplotlib.pyplot as plt
import numpy as np


def compute_layer_wise_mse_without_teacher_forcing(teacher_model, student_model, input_ids, attention_mask):
    """Original approach - errors propagate through layers."""
    with torch.no_grad():
        # Get teacher hidden states
        teacher_outputs = teacher_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        teacher_hidden = teacher_outputs.hidden_states
        
        # Get student hidden states (error propagation)
        student_hidden = []
        hidden_states = student_model.model.embed_tokens(input_ids)
        student_hidden.append(hidden_states)
        
        # Create position_ids
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Pass through layers (error propagates)
        for layer in student_model.model.layers:
            layer_outputs = layer(
                hidden_states,  # Uses previous student output
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=False
            )
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs
            student_hidden.append(hidden_states)
    
    # Compute MSE for each layer
    layer_mse = []
    for i in range(min(len(teacher_hidden), len(student_hidden))):
        mse = torch.mean((teacher_hidden[i] - student_hidden[i]) ** 2).item()
        layer_mse.append(mse)
    
    return layer_mse


def compute_layer_wise_mse_with_teacher_forcing(teacher_model, student_model, input_ids, attention_mask):
    """New approach - teacher forcing prevents error propagation."""
    with torch.no_grad():
        # Get teacher hidden states
        teacher_outputs = teacher_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        teacher_hidden = teacher_outputs.hidden_states
        
        # Get student hidden states (with teacher forcing)
        student_hidden = []
        hidden_states = student_model.model.embed_tokens(input_ids)
        student_hidden.append(hidden_states)
        
        # Create position_ids
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Pass through layers with teacher forcing
        for i, layer in enumerate(student_model.model.layers):
            if i == 0:
                layer_input = hidden_states
            else:
                # Use teacher's output as input (prevents error propagation)
                layer_input = teacher_hidden[i].detach()
            
            layer_outputs = layer(
                layer_input,  # Teacher forced input
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=False
            )
            if isinstance(layer_outputs, tuple):
                student_output = layer_outputs[0]
            else:
                student_output = layer_outputs
            student_hidden.append(student_output)
    
    # Compute MSE for each layer
    layer_mse = []
    for i in range(min(len(teacher_hidden), len(student_hidden))):
        mse = torch.mean((teacher_hidden[i] - student_hidden[i]) ** 2).item()
        layer_mse.append(mse)
    
    return layer_mse


def main():
    # Model configuration
    model_name = "meta-llama/Llama-3.2-1B"  # Using smaller model for testing
    
    print(f"Loading models: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load teacher model
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    teacher_model.eval()
    
    # Load student model and convert to NPT
    student_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # Convert to NPT
    adapter_config = {
        'r': 16,
        'd_model': student_model.config.hidden_size,
        'd_ffn': student_model.config.intermediate_size,
        'modulation_scale': 0.1,
        'init_strategy': 'adaptive'
    }
    student_model = convert_llama_to_npt(student_model, adapter_config)
    student_model.eval()
    
    # Test prompt
    test_prompt = "The capital of France is Paris. Machine learning is"
    inputs = tokenizer(test_prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    print("\nComputing layer-wise MSE without teacher forcing...")
    mse_without_tf = compute_layer_wise_mse_without_teacher_forcing(
        teacher_model, student_model, input_ids, attention_mask
    )
    
    print("Computing layer-wise MSE with teacher forcing...")
    mse_with_tf = compute_layer_wise_mse_with_teacher_forcing(
        teacher_model, student_model, input_ids, attention_mask
    )
    
    # Print results
    print("\n" + "="*60)
    print("Layer-wise MSE Comparison")
    print("="*60)
    print(f"{'Layer':<8} {'Without TF':<15} {'With TF':<15} {'Ratio':<10}")
    print("-"*60)
    
    for i in range(len(mse_without_tf)):
        ratio = mse_without_tf[i] / (mse_with_tf[i] + 1e-8)
        print(f"{i:<8} {mse_without_tf[i]:<15.6f} {mse_with_tf[i]:<15.6f} {ratio:<10.2f}x")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    layers = list(range(len(mse_without_tf)))
    
    plt.subplot(1, 2, 1)
    plt.plot(layers, mse_without_tf, 'r-o', label='Without Teacher Forcing', linewidth=2, markersize=4)
    plt.plot(layers, mse_with_tf, 'b-o', label='With Teacher Forcing', linewidth=2, markersize=4)
    plt.xlabel('Layer')
    plt.ylabel('MSE')
    plt.title('Layer-wise MSE: Linear Scale')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(layers, mse_without_tf, 'r-o', label='Without Teacher Forcing', linewidth=2, markersize=4)
    plt.semilogy(layers, mse_with_tf, 'b-o', label='With Teacher Forcing', linewidth=2, markersize=4)
    plt.xlabel('Layer')
    plt.ylabel('MSE (log scale)')
    plt.title('Layer-wise MSE: Log Scale')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('teacher_forcing_comparison.png', dpi=150)
    print(f"\nPlot saved to: teacher_forcing_comparison.png")
    
    # Summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"Without Teacher Forcing:")
    print(f"  - First layer MSE: {mse_without_tf[1]:.6f}")
    print(f"  - Last layer MSE:  {mse_without_tf[-1]:.6f}")
    print(f"  - Growth factor:   {mse_without_tf[-1]/mse_without_tf[1]:.1f}x")
    
    print(f"\nWith Teacher Forcing:")
    print(f"  - First layer MSE: {mse_with_tf[1]:.6f}")
    print(f"  - Last layer MSE:  {mse_with_tf[-1]:.6f}")
    print(f"  - Growth factor:   {mse_with_tf[-1]/mse_with_tf[1]:.1f}x")
    
    print(f"\nImprovement factor (last layer): {mse_without_tf[-1]/mse_with_tf[-1]:.1f}x")


if __name__ == "__main__":
    main()

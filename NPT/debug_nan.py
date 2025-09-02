"""
Debug script to identify NaN issues in NPT pretraining
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model import convert_llama_to_npt
import numpy as np

def check_for_nans(model, name="model"):
    """Check all parameters and gradients for NaN values"""
    has_nan = False
    for param_name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN found in {name}.{param_name} values")
            has_nan = True
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN found in {name}.{param_name} gradients")
            has_nan = True
    return has_nan

def debug_forward_pass():
    """Debug a single forward pass to identify where NaNs appear"""
    print("Loading model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load config
    config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B")
    
    # Create a minimal model for testing
    print("Creating minimal test model...")
    # Use CPU and FP32 for debugging
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        config=config,
        torch_dtype=torch.float32,
        device_map="cpu",
        load_in_4bit=False  # Disable quantization for debugging
    )
    
    # Convert to NPT
    print("Converting to NPT...")
    adapter_config = {
        'r': 16,
        'd_model': config.hidden_size,
        'd_ffn': config.intermediate_size,
        'compute_dtype': torch.float32
    }
    model = convert_llama_to_npt(model, adapter_config)
    
    # Create simple input
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt")
    
    print("\nChecking initial model state...")
    check_for_nans(model, "initial_model")
    
    # Forward pass with intermediate checks
    print("\nPerforming forward pass with intermediate checks...")
    
    # Get embeddings
    inputs_embeds = model.model.embed_tokens(inputs.input_ids)
    print(f"Embeddings - min: {inputs_embeds.min():.4f}, max: {inputs_embeds.max():.4f}, has_nan: {torch.isnan(inputs_embeds).any()}")
    
    hidden_states = inputs_embeds
    
    # Check first NPT layer in detail
    layer = model.model.layers[0]
    print(f"\nChecking first NPT layer...")
    
    # Layer norm
    normed = layer.input_layernorm(hidden_states)
    print(f"After input_layernorm - min: {normed.min():.4f}, max: {normed.max():.4f}, has_nan: {torch.isnan(normed).any()}")
    
    # Self attention
    attn_outputs = layer.self_attn(normed)
    attn_output = attn_outputs[0]
    print(f"After self_attn - min: {attn_output.min():.4f}, max: {attn_output.max():.4f}, has_nan: {torch.isnan(attn_output).any()}")
    
    # Adapter forward
    modulation = layer.adapter(attn_output)
    print(f"\nAdapter outputs:")
    for key, value in modulation.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key} - min: {value.min():.4f}, max: {value.max():.4f}, has_nan: {torch.isnan(value).any()}")
    
    # MLP computation
    mlp_input = layer.post_attention_layernorm(hidden_states)
    print(f"\nAfter post_attention_layernorm - min: {mlp_input.min():.4f}, max: {mlp_input.max():.4f}, has_nan: {torch.isnan(mlp_input).any()}")
    
    gate_output = layer.mlp.gate_proj(mlp_input)
    print(f"After gate_proj - min: {gate_output.min():.4f}, max: {gate_output.max():.4f}, has_nan: {torch.isnan(gate_output).any()}")
    
    # Check modulation
    if 'delta_mult' in modulation:
        modulation_factor = 1 + modulation['delta_mult']
        print(f"\nModulation factor (1 + delta_mult):")
        print(f"  min: {modulation_factor.min():.4f}, max: {modulation_factor.max():.4f}")
        print(f"  has zeros: {(modulation_factor == 0).any()}")
        print(f"  has near-zeros: {(torch.abs(modulation_factor) < 0.01).any()}")
        
        modulated_gate = gate_output * modulation_factor
        print(f"\nAfter modulation - min: {modulated_gate.min():.4f}, max: {modulated_gate.max():.4f}, has_nan: {torch.isnan(modulated_gate).any()}")

def test_loss_computation():
    """Test the loss computation to see where NaNs appear"""
    print("\n\nTesting loss computation...")
    
    # Create dummy tensors
    teacher_hidden = torch.randn(1, 10, 4096)
    student_hidden = torch.randn(1, 10, 4096)
    
    # Test MSE loss
    mse_loss = nn.functional.mse_loss(student_hidden, teacher_hidden)
    print(f"MSE loss: {mse_loss.item():.4f}")
    
    # Test with large differences
    student_hidden_large = student_hidden * 100
    mse_loss_large = nn.functional.mse_loss(student_hidden_large, teacher_hidden)
    print(f"MSE loss with large difference: {mse_loss_large.item():.4f}")
    
    # Test regularization
    reg_norm = torch.mean(torch.sum(student_hidden ** 2, dim=-1))
    print(f"Regularization norm: {reg_norm.item():.4f}")

if __name__ == "__main__":
    print("=== NPT NaN Debugging ===")
    debug_forward_pass()
    test_loss_computation()

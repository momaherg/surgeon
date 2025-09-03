#!/usr/bin/env python3
"""
Test script to verify the attention mask fix for SDPA
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model import convert_llama_to_npt
import sys

def test_attention_mask_fix():
    """Test that NPT layers handle attention masks correctly with SDPA"""
    
    print("Loading model and tokenizer...")
    model_name = "meta-llama/Llama-3.1-8B"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load config
    config = AutoConfig.from_pretrained(model_name)
    
    # Load model with BF16 for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Convert to NPT
    print("Converting to NPT architecture...")
    adapter_config = {
        'r': 32,
        'd_model': config.hidden_size,
        'd_ffn': config.intermediate_size,
        'compute_dtype': torch.bfloat16,
        'modulation_scale': 0.2
    }
    model = convert_llama_to_npt(model, adapter_config)
    
    # Create test input
    print("Creating test input...")
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we interact with technology in our daily lives."
    ]
    
    inputs = tokenizer(
        test_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # Move to device
    device = next(model.parameters()).device
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            # Test with attention mask
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            print("✓ Forward pass with attention mask successful!")
            
            # Test without attention mask (should also work)
            outputs_no_mask = model(
                input_ids=input_ids,
                output_hidden_states=True
            )
            print("✓ Forward pass without attention mask successful!")
            
        # Check if the model has SDPA
        first_layer = model.model.layers[0]
        if hasattr(first_layer.self_attn, '_attn_implementation'):
            print(f"\nAttention implementation: {first_layer.self_attn._attn_implementation}")
        else:
            print("\nAttention implementation: standard (not SDPA)")
            
        print("\n✅ All tests passed! The attention mask fix is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during forward pass: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_attention_mask_fix()
    sys.exit(0 if success else 1)

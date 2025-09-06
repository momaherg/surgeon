#!/usr/bin/env python3
"""
Test script to verify that the final layer MSE issue has been fixed.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model import convert_llama_to_npt
import matplotlib.pyplot as plt


def test_layer_mse_progression(model_name="meta-llama/Llama-2-7b-hf"):
    """Test that MSE progression is now smooth across all layers."""
    
    print("Loading models...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load config
    config = AutoConfig.from_pretrained(model_name)
    
    # Load teacher model
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float32,
        device_map="cpu"  # Use CPU for testing
    )
    teacher_model.eval()
    
    # Load student model
    student_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # Convert to NPT
    adapter_config = {
        'r': 16,
        'd_model': config.hidden_size,
        'd_ffn': config.intermediate_size,
        'modulation_scale': 0.1,
        'init_strategy': 'zero',  # Start with near-zero init to test
        'init_scale': 0.01
    }
    student_model = convert_llama_to_npt(student_model, adapter_config)
    student_model.eval()
    
    # Test text
    test_text = "The capital of France is Paris. Machine learning is a field of artificial intelligence."
    
    # Tokenize
    inputs = tokenizer(test_text, return_tensors="pt", max_length=32, truncation=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    print(f"Testing with text: {test_text}")
    print(f"Number of tokens: {input_ids.shape[1]}")
    
    with torch.no_grad():
        # Teacher forward pass
        teacher_outputs = teacher_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        teacher_hidden = teacher_outputs.hidden_states
        
        # Student forward pass - manual through layers
        student_hidden = []
        
        # Get embeddings
        hidden_states = student_model.model.embed_tokens(input_ids)
        student_hidden.append(hidden_states)
        
        # Create position_ids
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get position embeddings if available
        position_embeddings = None
        if hasattr(student_model.model, 'rotary_emb'):
            cos, sin = student_model.model.rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos, sin)
        
        # Pass through layers
        for layer in student_model.model.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings
            )
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs
            student_hidden.append(hidden_states)
        
        # Apply final layer norm (CRITICAL FIX)
        hidden_states = student_model.model.norm(hidden_states)
        student_hidden.append(hidden_states)
    
    # Compute MSE for each layer
    layer_mse = []
    layer_cosine = []
    
    print(f"\nTeacher hidden states: {len(teacher_hidden)} layers")
    print(f"Student hidden states: {len(student_hidden)} layers")
    
    for i in range(min(len(teacher_hidden), len(student_hidden))):
        teacher_h = teacher_hidden[i]
        student_h = student_hidden[i]
        
        # Ensure same dtype
        if teacher_h.dtype != student_h.dtype:
            teacher_h = teacher_h.to(student_h.dtype)
        
        # Compute MSE
        mse = torch.mean((teacher_h - student_h) ** 2).item()
        
        # Compute cosine similarity
        teacher_norm = teacher_h / (torch.norm(teacher_h, dim=-1, keepdim=True) + 1e-8)
        student_norm = student_h / (torch.norm(student_h, dim=-1, keepdim=True) + 1e-8)
        cosine_sim = torch.mean(torch.sum(teacher_norm * student_norm, dim=-1)).item()
        
        layer_mse.append(mse)
        layer_cosine.append(cosine_sim)
        
        print(f"Layer {i}: MSE={mse:.6f}, Cosine={cosine_sim:.6f}")
    
    # Plot MSE progression
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(layer_mse, 'b-o', markersize=4)
    plt.xlabel('Layer Index')
    plt.ylabel('MSE')
    plt.title('MSE Progression Across Layers')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to see small values better
    
    plt.subplot(1, 2, 2)
    plt.plot(layer_cosine, 'g-o', markersize=4)
    plt.xlabel('Layer Index')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity Across Layers')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('layer_mse_progression_fixed.png', dpi=150)
    print(f"\nPlot saved to layer_mse_progression_fixed.png")
    
    # Check if the issue is fixed
    if len(layer_mse) >= 2:
        last_layer_mse = layer_mse[-1]
        second_last_mse = layer_mse[-2]
        ratio = last_layer_mse / second_last_mse if second_last_mse > 0 else float('inf')
        
        print(f"\n{'='*60}")
        print(f"Last layer MSE: {last_layer_mse:.6f}")
        print(f"Second to last layer MSE: {second_last_mse:.6f}")
        print(f"Ratio: {ratio:.2f}x")
        
        if ratio > 5:
            print("WARNING: Last layer MSE is still significantly higher!")
            print("The issue may not be fully resolved.")
        else:
            print("SUCCESS: MSE progression appears smooth!")
            print("The fix appears to be working correctly.")
        print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test layer MSE fix")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Model to test")
    
    args = parser.parse_args()
    
    test_layer_mse_progression(args.model)

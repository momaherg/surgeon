"""
Test script to verify the fix for high MSE in the last NPT layer.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from model import convert_llama_to_npt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_hidden_states_alignment(model_name="gpt2", adapter_rank=16):
    """Test that teacher and student hidden states are properly aligned."""
    
    logger.info(f"Testing hidden states alignment for {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load configuration
    config = AutoConfig.from_pretrained(model_name)
    
    # Load teacher model
    teacher_model = AutoModelForCausalLM.from_pretrained(model_name)
    teacher_model.eval()
    
    # Load student model and convert to NPT
    student_model = AutoModelForCausalLM.from_pretrained(model_name)
    adapter_config = {
        'r': adapter_rank,
        'd_model': config.hidden_size,
        'd_ffn': config.intermediate_size if hasattr(config, 'intermediate_size') else config.n_inner,
        'modulation_scale': 0.1,
        'init_strategy': 'zero',  # Start with zero init for testing
        'init_scale': 0.01
    }
    student_model = convert_llama_to_npt(student_model, adapter_config)
    student_model.eval()
    
    # Test text
    test_text = "The capital of France is Paris."
    inputs = tokenizer(test_text, return_tensors="pt")
    
    with torch.no_grad():
        # Get teacher hidden states
        teacher_outputs = teacher_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True
        )
        teacher_hidden = teacher_outputs.hidden_states
        
        # Get student hidden states manually (with fix)
        student_hidden = []
        hidden_states = student_model.model.embed_tokens(inputs.input_ids)
        student_hidden.append(hidden_states)
        
        # Pass through layers
        for layer in student_model.model.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=inputs.attention_mask
            )
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs
            student_hidden.append(hidden_states)
        
        # Apply final layer norm (THE FIX)
        hidden_states_after_norm = student_model.model.norm(hidden_states)
        student_hidden.append(hidden_states_after_norm)
    
    # Compare counts
    logger.info(f"Teacher hidden states count: {len(teacher_hidden)}")
    logger.info(f"Student hidden states count: {len(student_hidden)}")
    
    # Compute MSE for each layer
    layer_mse = []
    for i in range(min(len(teacher_hidden), len(student_hidden))):
        teacher_h = teacher_hidden[i]
        student_h = student_hidden[i]
        
        # Ensure same dtype
        if teacher_h.dtype != student_h.dtype:
            teacher_h = teacher_h.to(student_h.dtype)
        
        mse = torch.mean((teacher_h - student_h) ** 2).item()
        layer_mse.append(mse)
    
    # Print results
    print("\n" + "="*60)
    print("Layer-wise MSE Analysis")
    print("="*60)
    print(f"{'Layer':<15} {'MSE':<20} {'Notes':<25}")
    print("-"*60)
    
    for i, mse in enumerate(layer_mse):
        notes = ""
        if i == 0:
            notes = "Embeddings"
        elif i == len(layer_mse) - 1:
            notes = "After final layer norm"
        elif i == len(layer_mse) - 2:
            notes = "Last transformer layer"
        
        print(f"{i:<15} {mse:<20.10f} {notes:<25}")
    
    # Check if last layer MSE is reasonable
    if len(layer_mse) >= 2:
        last_layer_mse = layer_mse[-1]
        second_last_mse = layer_mse[-2]
        avg_middle_mse = sum(layer_mse[1:-1]) / len(layer_mse[1:-1])
        
        print("\n" + "="*60)
        print("MSE Statistics")
        print("="*60)
        print(f"Average MSE (middle layers): {avg_middle_mse:.10f}")
        print(f"Second last layer MSE: {second_last_mse:.10f}")
        print(f"Last layer MSE: {last_layer_mse:.10f}")
        
        if last_layer_mse > 10 * avg_middle_mse:
            print("\n⚠️  WARNING: Last layer MSE is significantly higher than others!")
            print("This suggests the fix may not be working properly.")
        else:
            print("\n✅ SUCCESS: Last layer MSE is comparable to other layers!")
            print("The fix appears to be working correctly.")
    
    return layer_mse


def test_without_fix(model_name="gpt2", adapter_rank=16):
    """Test WITHOUT the fix to show the problem."""
    
    logger.info(f"Testing WITHOUT fix to demonstrate the issue")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load configuration
    config = AutoConfig.from_pretrained(model_name)
    
    # Load teacher model
    teacher_model = AutoModelForCausalLM.from_pretrained(model_name)
    teacher_model.eval()
    
    # Load student model and convert to NPT
    student_model = AutoModelForCausalLM.from_pretrained(model_name)
    adapter_config = {
        'r': adapter_rank,
        'd_model': config.hidden_size,
        'd_ffn': config.intermediate_size if hasattr(config, 'intermediate_size') else config.n_inner,
        'modulation_scale': 0.1,
        'init_strategy': 'zero',
        'init_scale': 0.01
    }
    student_model = convert_llama_to_npt(student_model, adapter_config)
    student_model.eval()
    
    # Test text
    test_text = "The capital of France is Paris."
    inputs = tokenizer(test_text, return_tensors="pt")
    
    with torch.no_grad():
        # Get teacher hidden states
        teacher_outputs = teacher_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True
        )
        teacher_hidden = teacher_outputs.hidden_states
        
        # Get student hidden states manually (WITHOUT fix)
        student_hidden = []
        hidden_states = student_model.model.embed_tokens(inputs.input_ids)
        student_hidden.append(hidden_states)
        
        # Pass through layers
        for layer in student_model.model.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=inputs.attention_mask
            )
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs
            student_hidden.append(hidden_states)
        
        # NO final layer norm added (demonstrating the bug)
        # hidden_states_after_norm = student_model.model.norm(hidden_states)
        # student_hidden.append(hidden_states_after_norm)  # <-- MISSING!
    
    # Compare counts
    logger.info(f"Teacher hidden states count: {len(teacher_hidden)}")
    logger.info(f"Student hidden states count: {len(student_hidden)} (missing final norm!)")
    
    # Compute MSE for each layer
    # Note: We'll compare up to min length, which will show high MSE for "last" comparison
    layer_mse = []
    num_layers = min(len(teacher_hidden) - 1, len(student_hidden) - 1)
    
    for i in range(1, num_layers + 1):
        teacher_h = teacher_hidden[i]
        student_h = student_hidden[i]
        
        # Ensure same dtype
        if teacher_h.dtype != student_h.dtype:
            teacher_h = teacher_h.to(student_h.dtype)
        
        mse = torch.mean((teacher_h - student_h) ** 2).item()
        layer_mse.append(mse)
    
    # Print results
    print("\n" + "="*60)
    print("Layer-wise MSE Analysis (WITHOUT FIX)")
    print("="*60)
    print(f"{'Layer':<15} {'MSE':<20} {'Notes':<35}")
    print("-"*60)
    
    for i, mse in enumerate(layer_mse):
        layer_idx = i + 1  # Because we start from index 1
        notes = ""
        if layer_idx == len(layer_mse):
            notes = "⚠️  Comparing pre-norm vs post-norm!"
        
        print(f"{layer_idx:<15} {mse:<20.10f} {notes:<35}")
    
    return layer_mse


def main():
    """Run the test."""
    print("Testing Last Layer MSE Fix")
    print("="*80)
    
    model_name = "gpt2"  # Use a smaller model for testing
    
    print("\n1. Testing WITHOUT the fix (showing the problem):")
    mse_without_fix = test_without_fix(model_name)
    
    print("\n\n2. Testing WITH the fix (showing the solution):")
    mse_with_fix = test_hidden_states_alignment(model_name)
    
    print("\n\n" + "="*80)
    print("Summary")
    print("="*80)
    print("The fix adds the final layer norm output to the student's hidden states")
    print("collection, ensuring proper alignment with the teacher model.")
    print("\nWithout this fix, the last layer comparison is between:")
    print("- Student: Output of last transformer layer (before layer norm)")
    print("- Teacher: Output after final layer norm")
    print("\nThis mismatch causes artificially high MSE for the 'last' layer.")


if __name__ == "__main__":
    main()

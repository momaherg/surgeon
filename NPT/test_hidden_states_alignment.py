#!/usr/bin/env python3
"""
Test to check if teacher and student hidden states are properly aligned.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model import convert_llama_to_npt


def test_alignment():
    """Test the alignment between teacher and student hidden states."""
    
    model_name = "meta-llama/Llama-3.1-8B"
    
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load teacher
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    teacher.eval()
    
    # Load student
    student = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # Convert to NPT with zero init (should match teacher initially)
    adapter_config = {
        'r': 16,
        'd_model': teacher.config.hidden_size,
        'd_ffn': teacher.config.intermediate_size,
        'modulation_scale': 0.0,  # Zero scale to match teacher exactly
        'init_strategy': 'zero',
        'init_scale': 0.0  # Zero init
    }
    student = convert_llama_to_npt(student, adapter_config)
    student.eval()
    
    # Test input
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        # Teacher outputs
        teacher_out = teacher(**inputs, output_hidden_states=True)
        teacher_hidden = teacher_out.hidden_states
        
        # Student outputs - collect manually
        student_hidden = []
        
        # Embeddings
        hidden = student.model.embed_tokens(inputs.input_ids)
        student_hidden.append(hidden)
        
        # Position setup
        batch_size, seq_length = inputs.input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        
        # Get position embeddings if the model uses them
        position_embeddings = None
        if hasattr(student.model, 'rotary_emb'):
            cos, sin = student.model.rotary_emb(hidden, position_ids)
            position_embeddings = (cos, sin)
        
        # Pass through layers
        for layer in student.model.layers:
            outputs = layer(
                hidden, 
                position_ids=position_ids,
                position_embeddings=position_embeddings
            )
            if isinstance(outputs, tuple):
                hidden = outputs[0]
            else:
                hidden = outputs
            student_hidden.append(hidden)
        
        # WITHOUT final norm first
        print(f"\nWithout final norm:")
        print(f"Teacher hidden states: {len(teacher_hidden)}")
        print(f"Student hidden states: {len(student_hidden)}")
        
        # Now add final norm
        final_norm = student.model.norm(hidden)
        student_hidden.append(final_norm)
        
        print(f"\nWith final norm:")
        print(f"Teacher hidden states: {len(teacher_hidden)}")
        print(f"Student hidden states: {len(student_hidden)}")
        
        # Check alignment
        print("\nAlignment check:")
        max_len = max(len(teacher_hidden), len(student_hidden))
        
        for i in range(max_len):
            if i < len(teacher_hidden) and i < len(student_hidden):
                mse = torch.mean((teacher_hidden[i] - student_hidden[i]) ** 2).item()
                print(f"Index {i}: MSE = {mse:.6f}")
            elif i < len(teacher_hidden):
                print(f"Index {i}: Only in teacher (shape: {teacher_hidden[i].shape})")
            else:
                print(f"Index {i}: Only in student (shape: {student_hidden[i].shape})")
        
        # Check what the last teacher hidden state is
        print(f"\nLast teacher hidden state mean: {teacher_hidden[-1].mean().item():.6f}")
        print(f"Student final norm mean: {final_norm.mean().item():.6f}")
        
        # Check if teacher's last hidden is the final norm
        teacher_final_norm = teacher.model.norm(teacher_hidden[-2])  # Apply norm to second-to-last
        diff = torch.mean(torch.abs(teacher_hidden[-1] - teacher_final_norm)).item()
        print(f"\nDifference between teacher[-1] and norm(teacher[-2]): {diff:.6f}")
        
        if diff < 1e-4:
            print("✓ Teacher's last hidden state IS the final layer norm")
        else:
            print("✗ Teacher's last hidden state is NOT the final layer norm")


if __name__ == "__main__":
    test_alignment()

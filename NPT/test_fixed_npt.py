"""
Test script to verify NPT fixes work correctly
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model import convert_llama_to_npt

def test_npt_forward():
    """Test that NPT forward pass works without NaN"""
    print("Testing NPT forward pass...")
    
    # Use a smaller model for testing
    model_name = "meta-llama/Llama-3.1-8B"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load config
    config = AutoConfig.from_pretrained(model_name)
    
    # Create a minimal model
    print("Loading model (this may take a moment)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float32,
        device_map="auto",
        load_in_4bit=True  # Use quantization
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
    model.eval()
    
    # Test inputs
    texts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "In machine learning, neural networks are",
    ]
    
    for i, text in enumerate(texts):
        print(f"\nTest {i+1}: '{text}'")
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            try:
                outputs = model(**inputs, output_hidden_states=True)
                
                # Check for NaN in outputs
                logits = outputs.logits
                has_nan = torch.isnan(logits).any()
                
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits min: {logits.min():.4f}, max: {logits.max():.4f}")
                print(f"  Has NaN: {has_nan}")
                
                if has_nan:
                    print("  ERROR: NaN detected in output!")
                else:
                    print("  SUCCESS: No NaN in output")
                    
                # Generate a token
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                next_word = tokenizer.decode(next_token)
                print(f"  Next token: '{next_word}'")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")

def test_loss_computation():
    """Test loss computation between teacher and student"""
    print("\n\nTesting loss computation...")
    
    # Create dummy models
    d_model = 256
    seq_len = 10
    batch_size = 2
    
    # Dummy hidden states
    teacher_hidden = torch.randn(batch_size, seq_len, d_model)
    student_hidden = torch.randn(batch_size, seq_len, d_model)
    
    # Test standard MSE
    mse_loss = nn.functional.mse_loss(student_hidden, teacher_hidden)
    print(f"Standard MSE loss: {mse_loss.item():.4f}")
    
    # Test normalized MSE (as in safe version)
    teacher_norm = teacher_hidden.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    student_norm = student_hidden.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    
    teacher_normalized = teacher_hidden / teacher_norm
    student_normalized = student_hidden / student_norm
    
    normalized_mse = nn.functional.mse_loss(student_normalized, teacher_normalized)
    print(f"Normalized MSE loss: {normalized_mse.item():.4f}")
    
    # Test with extreme values
    student_extreme = student_hidden * 1000
    extreme_mse = nn.functional.mse_loss(student_extreme, teacher_hidden)
    print(f"Extreme MSE loss: {extreme_mse.item():.4f}")
    
    # Normalized version handles extreme values better
    student_extreme_norm = student_extreme.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    student_extreme_normalized = student_extreme / student_extreme_norm
    extreme_normalized_mse = nn.functional.mse_loss(student_extreme_normalized, teacher_normalized)
    print(f"Extreme normalized MSE loss: {extreme_normalized_mse.item():.4f}")

def test_modulation_values():
    """Test the new modulation scheme"""
    print("\n\nTesting modulation values...")
    
    # Test old scheme (tanh)
    raw_values = torch.linspace(-3, 3, 7)
    old_modulation = torch.tanh(raw_values)
    old_factor = 1 + old_modulation
    
    print("Old scheme (1 + tanh):")
    for raw, mod, factor in zip(raw_values, old_modulation, old_factor):
        print(f"  raw={raw:6.2f} -> tanh={mod:6.3f} -> factor={factor:6.3f}")
    
    # Test new scheme (sigmoid-based)
    new_modulation = 0.5 * torch.sigmoid(raw_values) + 0.5
    
    print("\nNew scheme (0.5 * sigmoid + 0.5):")
    for raw, mod in zip(raw_values, new_modulation):
        print(f"  raw={raw:6.2f} -> modulation={mod:6.3f}")
    
    print(f"\nNew scheme range: [{new_modulation.min():.3f}, {new_modulation.max():.3f}]")
    print("No zeros possible - more stable!")

if __name__ == "__main__":
    print("=== Testing NPT Fixes ===\n")
    
    # Test modulation values first (quick)
    test_modulation_values()
    
    # Test loss computation
    test_loss_computation()
    
    # Test full forward pass (slower, requires model loading)
    print("\n" + "="*50)
    response = input("Test full model forward pass? This requires loading the model. (y/n): ")
    if response.lower() == 'y':
        test_npt_forward()
    
    print("\nTests completed!")

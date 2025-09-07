"""
Quick test to verify teacher-guided training prevents error propagation.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model import convert_llama_to_npt
import warnings
warnings.filterwarnings("ignore")


def test_error_propagation():
    """Test to show how errors propagate in standard vs teacher-guided training."""
    
    print("Loading models for error propagation test...")
    
    # Use a small model for quick testing
    model_name = "meta-llama/Llama-3.1-8B"
    
    # Load tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = AutoConfig.from_pretrained(model_name)
    
    # Load teacher model
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map="auto",
        torch_dtype=torch.float32
    )
    teacher_model.eval()
    
    # Load student model and convert to NPT
    student_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map="auto",
        torch_dtype=torch.float32
    )
    
    # Convert to NPT
    adapter_config = {
        'r': 16,
        'd_model': config.hidden_size,
        'd_ffn': config.intermediate_size,
        'compute_dtype': torch.float32,
        'modulation_scale': 0.1,
        'init_strategy': 'adaptive',
        'init_scale': 1.0
    }
    student_model = convert_llama_to_npt(student_model, adapter_config)
    student_model.eval()
    
    # Test input
    test_text = "The capital of France is"
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    attention_mask = inputs.attention_mask.cuda()
    
    # Get teacher hidden states
    with torch.no_grad():
        teacher_outputs = teacher_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        teacher_hidden_states = teacher_outputs.hidden_states
    
    print(f"\nAnalyzing {len(teacher_hidden_states)-1} transformer layers...")
    
    # 1. Standard Sequential Processing (Error Propagation)
    print("\n1. STANDARD TRAINING (Sequential Processing - Errors Propagate):")
    print("-" * 60)
    
    sequential_errors = []
    with torch.no_grad():
        # Start with embeddings
        hidden_states = student_model.model.embed_tokens(input_ids)
        
        # Create position IDs
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get position embeddings if available
        position_embeddings = None
        if hasattr(student_model.model, 'rotary_emb'):
            cos, sin = student_model.model.rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos, sin)
        
        # Process each layer sequentially
        for i, layer in enumerate(student_model.model.layers):
            # Forward through layer
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
            
            # Compute MSE with teacher
            mse = F.mse_loss(hidden_states, teacher_hidden_states[i+1]).item()
            sequential_errors.append(mse)
            
            if i % 8 == 0 or i == len(student_model.model.layers) - 1:
                print(f"  Layer {i:2d}: MSE = {mse:.6f}")
    
    # 2. Teacher-Guided Processing (No Error Propagation)
    print("\n2. TEACHER-GUIDED TRAINING (Each layer gets correct input):")
    print("-" * 60)
    
    guided_errors = []
    with torch.no_grad():
        for i, layer in enumerate(student_model.model.layers):
            # Use teacher hidden states as input (preventing error propagation)
            teacher_input = teacher_hidden_states[i]
            
            # Forward through layer with teacher input
            layer_outputs = layer(
                teacher_input,  # KEY: Use teacher input, not previous layer output
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings
            )
            
            if isinstance(layer_outputs, tuple):
                student_output = layer_outputs[0]
            else:
                student_output = layer_outputs
            
            # Compute MSE with expected teacher output
            mse = F.mse_loss(student_output, teacher_hidden_states[i+1]).item()
            guided_errors.append(mse)
            
            if i % 8 == 0 or i == len(student_model.model.layers) - 1:
                print(f"  Layer {i:2d}: MSE = {mse:.6f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"Standard Training (Sequential):")
    print(f"  - First Layer MSE: {sequential_errors[0]:.6f}")
    print(f"  - Last Layer MSE:  {sequential_errors[-1]:.6f}")
    print(f"  - Error Growth:    {sequential_errors[-1] / sequential_errors[0]:.1f}x")
    print(f"\nTeacher-Guided Training:")
    print(f"  - First Layer MSE: {guided_errors[0]:.6f}")
    print(f"  - Last Layer MSE:  {guided_errors[-1]:.6f}")
    print(f"  - Error Growth:    {guided_errors[-1] / guided_errors[0]:.1f}x")
    print(f"\nImprovement Factor: {sequential_errors[-1] / guided_errors[-1]:.1f}x reduction in last layer MSE")
    
    # Check for exponential growth pattern
    growth_rates = []
    for i in range(1, len(sequential_errors)):
        if sequential_errors[i-1] > 0:
            growth = sequential_errors[i] / sequential_errors[i-1]
            growth_rates.append(growth)
    
    avg_growth = sum(growth_rates) / len(growth_rates) if growth_rates else 1.0
    print(f"\nAverage error growth rate per layer (sequential): {avg_growth:.3f}x")


if __name__ == "__main__":
    test_error_propagation()

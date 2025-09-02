"""
Quick test to verify the dtype fix works
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model import convert_llama_to_npt
from utils import get_quantization_config

def test_npt_with_quantization():
    """Test NPT with quantization to verify dtype fix"""
    print("Testing NPT with quantization...")
    
    model_name = "meta-llama/Llama-3.1-8B"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load config
    config = AutoConfig.from_pretrained(model_name)
    
    # Load models with quantization
    print("Loading teacher model...")
    quantization_config = get_quantization_config()
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float32
    )
    teacher_model.eval()
    
    print("Loading student model...")
    student_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float32
    )
    
    # Share embeddings
    student_model.model.embed_tokens = teacher_model.model.embed_tokens
    student_model.lm_head = teacher_model.lm_head
    
    # Convert to NPT
    print("Converting to NPT...")
    adapter_config = {
        'r': 16,
        'd_model': config.hidden_size,
        'd_ffn': config.intermediate_size,
        'compute_dtype': torch.float32,
        'modulation_type': 'additive'  # Use safe mode
    }
    student_model = convert_llama_to_npt(student_model, adapter_config)
    
    # Test forward pass
    print("\nTesting forward pass...")
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt")
    device = next(student_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        # Teacher forward
        print("  Teacher forward...")
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs, output_hidden_states=True)
        print("  ✓ Teacher forward successful")
        
        # Student forward - manual through layers
        print("  Student forward (manual)...")
        hidden_states = student_model.model.embed_tokens(inputs['input_ids'])
        
        # Create position ids
        batch_size, seq_length = inputs['input_ids'].shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Convert attention mask dtype
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.float32)
        
        # Pass through first layer only
        first_layer = student_model.model.layers[0]
        layer_outputs = first_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        print("  ✓ Student forward successful")
        print(f"  Output shape: {layer_outputs[0].shape}")
        if len(layer_outputs) > 1:
            print(f"  Regularization norm: {layer_outputs[1].item():.6f}")
        
        # Test loss computation
        print("\nTesting loss computation...")
        teacher_hidden = teacher_outputs.hidden_states[1]
        student_hidden = layer_outputs[0]
        
        # Normalized MSE
        teacher_norm = teacher_hidden.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        student_norm = student_hidden.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        
        teacher_normalized = teacher_hidden / teacher_norm
        student_normalized = student_hidden / student_norm
        
        mse_loss = torch.nn.functional.mse_loss(student_normalized, teacher_normalized)
        print(f"  Normalized MSE loss: {mse_loss.item():.6f}")
        print("  ✓ Loss computation successful")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_npt_with_quantization()

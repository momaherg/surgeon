"""
Test script to verify FP16 compatibility fix
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model import convert_llama_to_npt

def test_fp16_compatibility():
    """Test NPT with FP16 to verify dtype fix"""
    print("Testing NPT with FP16...")
    
    model_name = "meta-llama/Llama-3.1-8B"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load config
    config = AutoConfig.from_pretrained(model_name)
    
    # Test 1: FP16 without quantization
    print("\n1. Testing FP16 without quantization...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=False
    )
    
    # Convert to NPT
    adapter_config = {
        'r': 8,
        'd_model': config.hidden_size,
        'd_ffn': config.intermediate_size,
        'compute_dtype': torch.float16  # Match model dtype
    }
    npt_model_fp16 = convert_llama_to_npt(model_fp16, adapter_config)
    
    # Test forward pass
    test_text = "Hello world"
    inputs = tokenizer(test_text, return_tensors="pt")
    device = next(npt_model_fp16.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            # Check adapter dtype
            adapter = npt_model_fp16.model.layers[0].adapter
            print(f"  Adapter A_proj weight dtype: {adapter.A_proj.weight.dtype}")
            print(f"  Model hidden states will be: {torch.float16}")
            
            # Manual forward through first layer
            hidden_states = npt_model_fp16.model.embed_tokens(inputs['input_ids'])
            print(f"  Hidden states dtype: {hidden_states.dtype}")
            
            # Test through first NPT layer
            batch_size, seq_length = inputs['input_ids'].shape
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            attention_mask = inputs.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            
            layer_outputs = npt_model_fp16.model.layers[0](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            
            print(f"  ✓ FP16 forward pass successful!")
            print(f"  Output shape: {layer_outputs[0].shape}")
            print(f"  Output dtype: {layer_outputs[0].dtype}")
            
    except Exception as e:
        print(f"  ❌ FP16 test failed: {str(e)}")
        return False
    
    # Test 2: FP32 model (should still work)
    print("\n2. Testing FP32 for comparison...")
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float32,
        device_map="auto",
        load_in_4bit=False
    )
    
    adapter_config_fp32 = {
        'r': 8,
        'd_model': config.hidden_size,
        'd_ffn': config.intermediate_size,
        'compute_dtype': torch.float32
    }
    npt_model_fp32 = convert_llama_to_npt(model_fp32, adapter_config_fp32)
    
    try:
        with torch.no_grad():
            adapter = npt_model_fp32.model.layers[0].adapter
            print(f"  Adapter A_proj weight dtype: {adapter.A_proj.weight.dtype}")
            
            hidden_states = npt_model_fp32.model.embed_tokens(inputs['input_ids'])
            print(f"  Hidden states dtype: {hidden_states.dtype}")
            
            print(f"  ✓ FP32 setup successful!")
            
    except Exception as e:
        print(f"  ❌ FP32 test failed: {str(e)}")
        return False
    
    # Test 3: Mixed dtype handling
    print("\n3. Testing mixed dtype handling...")
    
    # Create FP16 input for FP32 adapter
    fp16_tensor = torch.randn(1, 10, config.hidden_size, dtype=torch.float16, device=device)
    
    # Test adapter forward with dtype conversion
    adapter_fp32 = npt_model_fp32.model.layers[0].adapter
    try:
        output = adapter_fp32(fp16_tensor)
        print(f"  ✓ FP32 adapter handled FP16 input successfully")
        print(f"  Input dtype: {fp16_tensor.dtype}, Output dtype: {output['low_rank_rep'].dtype}")
    except Exception as e:
        print(f"  ❌ Mixed dtype test failed: {str(e)}")
        return False
    
    print("\n✅ All FP16 compatibility tests passed!")
    return True

def test_dtype_scenarios():
    """Test different dtype scenarios"""
    print("\n4. Testing dtype conversion scenarios...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test tensor dtype conversions
    fp16_tensor = torch.randn(2, 5, 10, dtype=torch.float16, device=device)
    fp32_tensor = torch.randn(2, 5, 10, dtype=torch.float32, device=device)
    
    # Test modulation scaling
    print(f"  FP16 tensor range: [{fp16_tensor.min():.4f}, {fp16_tensor.max():.4f}]")
    print(f"  FP32 tensor range: [{fp32_tensor.min():.4f}, {fp32_tensor.max():.4f}]")
    
    # Test sigmoid-based modulation in different dtypes
    raw_values = torch.randn(2, 5, 10, device=device)
    
    # FP16 modulation
    modulation_fp16 = (0.5 * torch.sigmoid(raw_values.half()) + 0.5)
    print(f"  FP16 modulation range: [{modulation_fp16.min():.4f}, {modulation_fp16.max():.4f}]")
    
    # FP32 modulation
    modulation_fp32 = (0.5 * torch.sigmoid(raw_values) + 0.5)
    print(f"  FP32 modulation range: [{modulation_fp32.min():.4f}, {modulation_fp32.max():.4f}]")
    
    print("  ✓ Dtype scenarios tested successfully")

if __name__ == "__main__":
    print("=== Testing FP16 Compatibility Fix ===\n")
    
    # Test basic dtype scenarios first
    test_dtype_scenarios()
    
    # Test full model (requires loading)
    print("\n" + "="*50)
    response = input("Test full model with FP16? This requires loading the model. (y/n): ")
    if response.lower() == 'y':
        test_fp16_compatibility()
    
    print("\nTests completed!")

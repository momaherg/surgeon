"""
Test NPT layer output format to understand the issue.
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from model import NPTLayer


def test_layer_outputs():
    """Test what outputs different layer types produce."""
    
    print("Testing Layer Output Formats")
    print("=" * 60)
    
    # Load a small model for testing
    model_name = "meta-llama/Llama-3.1-8B"
    config = AutoConfig.from_pretrained(model_name)
    
    # Create dummy inputs
    batch_size, seq_len = 1, 10
    hidden_size = config.hidden_size
    device = "cpu"
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"\nInput shapes:")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    
    # Load model to get a real layer
    print(f"\nLoading model to test real decoder layer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    
    # Test original Llama decoder layer
    print("\n1. Testing Original LlamaDecoderLayer:")
    original_layer = model.model.layers[0]
    original_layer.eval()
    
    # Test different configurations
    configs = [
        {"use_cache": False, "output_attentions": False},
        {"use_cache": True, "output_attentions": False},
        {"use_cache": False, "output_attentions": True},
        {"use_cache": True, "output_attentions": True},
    ]
    
    for cfg in configs:
        print(f"\n   Config: {cfg}")
        with torch.no_grad():
            outputs = original_layer(
                hidden_states=hidden_states.to(original_layer.self_attn.q_proj.weight.dtype),
                attention_mask=attention_mask,
                position_ids=None,
                past_key_value=None,
                **cfg
            )
        
        print(f"   Output type: {type(outputs)}")
        if isinstance(outputs, tuple):
            print(f"   Output length: {len(outputs)}")
            for i, out in enumerate(outputs):
                if hasattr(out, 'shape'):
                    print(f"     [{i}]: {type(out).__name__} shape={out.shape}")
                elif isinstance(out, tuple) and len(out) == 2:
                    print(f"     [{i}]: tuple(key_states shape={out[0].shape}, value_states shape={out[1].shape})")
                else:
                    print(f"     [{i}]: {type(out).__name__}")
    
    # Now test NPT layer
    print("\n\n2. Testing NPT Layer:")
    
    # Create NPT layer from the original
    adapter_config = {
        'd_model': config.hidden_size,
        'd_ffn': config.intermediate_size,
        'r': 16,
        'modulation_scale': 0.1
    }
    
    npt_layer = NPTLayer(original_layer, adapter_config)
    npt_layer.eval()  # Set to eval mode
    
    for cfg in configs:
        print(f"\n   Config: {cfg}")
        with torch.no_grad():
            outputs = npt_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_value=None,
                **cfg
            )
        
        print(f"   Output type: {type(outputs)}")
        if isinstance(outputs, tuple):
            print(f"   Output length: {len(outputs)}")
            for i, out in enumerate(outputs):
                if hasattr(out, 'shape'):
                    print(f"     [{i}]: {type(out).__name__} shape={out.shape}")
                elif isinstance(out, tuple) and len(out) == 2:
                    print(f"     [{i}]: tuple(key_states shape={out[0].shape}, value_states shape={out[1].shape})")
                else:
                    print(f"     [{i}]: {type(out).__name__}")
    
    print("\n" + "=" * 60)
    print("Key Findings:")
    print("- Check if NPT output format matches original layer format")
    print("- Pay attention to the order of elements in the tuple")
    print("- Verify cache format is preserved correctly")


if __name__ == "__main__":
    test_layer_outputs()

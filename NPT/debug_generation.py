"""
Debug script to understand the generation flow and output format issues.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model import convert_llama_to_npt, NPTLayer
from utils import get_quantization_config


def debug_generation_flow(checkpoint_path, base_model_name="meta-llama/Llama-3.1-8B"):
    """Debug the generation flow to understand output formats."""
    
    print("=" * 60)
    print("DEBUGGING NPT GENERATION FLOW")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load config
    config = AutoConfig.from_pretrained(base_model_name)
    
    # Load model
    print("\n1. Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        config=config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Convert to NPT
    print("\n2. Converting to NPT...")
    adapter_config = {
        'r': 16,
        'd_model': config.hidden_size,
        'd_ffn': config.intermediate_size,
        'compute_dtype': torch.float16
    }
    model = convert_llama_to_npt(model, adapter_config)
    
    # Load checkpoint weights
    print(f"\n3. Loading checkpoint from {checkpoint_path}...")
    from safetensors.torch import load_file
    safetensor_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors') and f.startswith('model-')]
    
    if safetensor_files:
        state_dict = {}
        for file in sorted(safetensor_files):
            shard_path = os.path.join(checkpoint_path, file)
            shard_dict = load_file(shard_path)
            state_dict.update(shard_dict)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded {len(safetensor_files)} safetensor shards")
    
    model.eval()
    
    # Test with a simple prompt
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"\n4. Testing with prompt: '{prompt}'")
    print(f"   Input shape: {inputs['input_ids'].shape}")
    
    # Manually trace through the forward pass
    print("\n5. Tracing through forward pass...")
    
    # Get embeddings
    hidden_states = model.model.embed_tokens(inputs['input_ids'])
    print(f"   After embed_tokens: type={type(hidden_states)}, shape={hidden_states.shape}")
    
    # Go through first few layers manually
    for i, layer in enumerate(model.model.layers[:3]):  # Just check first 3 layers
        print(f"\n   Layer {i} ({type(layer).__name__}):")
        print(f"     Input: type={type(hidden_states)}, ", end="")
        if isinstance(hidden_states, tuple):
            print(f"len={len(hidden_states)}")
            print(f"     First element: type={type(hidden_states[0])}, shape={hidden_states[0].shape if hasattr(hidden_states[0], 'shape') else 'N/A'}")
        else:
            print(f"shape={hidden_states.shape}")
        
        # Create minimal inputs for the layer
        layer_inputs = {
            'hidden_states': hidden_states,
            'attention_mask': inputs.get('attention_mask'),
            'position_ids': None,
            'past_key_value': None,
            'output_attentions': False,
            'use_cache': True,  # This is typically True during generation
        }
        
        try:
            # Call the layer
            layer_outputs = layer(**layer_inputs)
            
            print(f"     Output: type={type(layer_outputs)}, ", end="")
            if isinstance(layer_outputs, tuple):
                print(f"len={len(layer_outputs)}")
                for j, out in enumerate(layer_outputs):
                    if hasattr(out, 'shape'):
                        print(f"       [{j}]: type={type(out).__name__}, shape={out.shape}")
                    else:
                        print(f"       [{j}]: type={type(out).__name__}")
            else:
                print(f"shape={layer_outputs.shape if hasattr(layer_outputs, 'shape') else 'N/A'}")
            
            # Update hidden_states for next layer
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs
                
        except Exception as e:
            print(f"     ERROR: {type(e).__name__}: {e}")
            break
    
    # Now test actual generation
    print("\n6. Testing actual generation...")
    try:
        with torch.no_grad():
            # Try with minimal generation
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,  # Greedy for deterministic results
                use_cache=True,
            )
        print("   Generation successful!")
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Output: '{response}'")
    except Exception as e:
        print(f"   Generation failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    # Check NPT layer configuration
    print("\n7. NPT Layer Configuration:")
    for i, layer in enumerate(model.model.layers[:3]):
        if isinstance(layer, NPTLayer):
            print(f"   Layer {i}: NPTLayer")
            print(f"     - training mode: {layer.training}")
            print(f"     - adapter rank: {layer.adapter.r}")
            print(f"     - modulation scale: {layer.modulation_scale}")
        else:
            print(f"   Layer {i}: {type(layer).__name__}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python debug_generation.py <checkpoint_path> [base_model_name]")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    base_model_name = sys.argv[2] if len(sys.argv) > 2 else "meta-llama/Llama-3.1-8B"
    
    debug_generation_flow(checkpoint_path, base_model_name)

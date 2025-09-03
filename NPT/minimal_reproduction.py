"""
Minimal reproduction of the generation error.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model import convert_llama_to_npt
from safetensors.torch import load_file


def minimal_test(checkpoint_path, base_model_name="meta-llama/Llama-3.1-8B"):
    """Minimal test to reproduce the error."""
    
    print("Loading model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    config = AutoConfig.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        config=config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Convert to NPT
    adapter_config = {
        'r': 16,
        'd_model': config.hidden_size,
        'd_ffn': config.intermediate_size,
        'compute_dtype': torch.float16
    }
    model = convert_llama_to_npt(model, adapter_config)
    
    # Load weights
    safetensor_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors') and f.startswith('model-')]
    if safetensor_files:
        state_dict = {}
        for file in sorted(safetensor_files):
            shard_path = os.path.join(checkpoint_path, file)
            shard_dict = load_file(shard_path)
            state_dict.update(shard_dict)
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    
    # Add hooks to debug layer inputs/outputs
    layer_inputs = {}
    layer_outputs = {}
    
    def make_input_hook(layer_idx):
        def hook(module, inputs, output):
            # Store the input
            if isinstance(inputs, tuple) and len(inputs) > 0:
                layer_inputs[layer_idx] = {
                    'type': type(inputs[0]).__name__,
                    'is_tuple': isinstance(inputs[0], tuple),
                    'shape': inputs[0].shape if hasattr(inputs[0], 'shape') else 'N/A'
                }
            return output
        return hook
    
    def make_output_hook(layer_idx):
        def hook(module, inputs, output):
            # Store the output info
            layer_outputs[layer_idx] = {
                'type': type(output).__name__,
                'is_tuple': isinstance(output, tuple),
                'length': len(output) if isinstance(output, tuple) else 'N/A',
                'first_elem_type': type(output[0]).__name__ if isinstance(output, tuple) else 'N/A',
                'first_elem_shape': output[0].shape if isinstance(output, tuple) and hasattr(output[0], 'shape') else 'N/A'
            }
            return output
        return hook
    
    # Register hooks on first few layers
    hooks = []
    for i in range(min(3, len(model.model.layers))):
        h1 = model.model.layers[i].register_forward_hook(make_input_hook(i))
        h2 = model.model.layers[i].register_forward_hook(make_output_hook(i))
        hooks.extend([h1, h2])
    
    # Try generation
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print("\nTrying generation...")
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
            )
        print("Success!")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        
        # Print debug info
        print("\nLayer Input/Output Analysis:")
        for i in range(min(3, len(layer_inputs))):
            print(f"\nLayer {i}:")
            if i in layer_inputs:
                print(f"  Input: {layer_inputs[i]}")
            if i in layer_outputs:
                print(f"  Output: {layer_outputs[i]}")
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Also check what the model's forward method expects
    print("\n\nChecking Llama model forward pass...")
    print("Let's trace through the actual model.forward() to see what happens:")
    
    # Get the actual forward method code location
    import inspect
    print(f"\nLlama Model forward method location: {inspect.getfile(model.model.forward)}")
    
    # Let's check how the model processes layer outputs
    with torch.no_grad():
        hidden_states = model.model.embed_tokens(inputs['input_ids'])
        print(f"\nAfter embeddings: type={type(hidden_states)}, shape={hidden_states.shape}")
        
        # Manually call first layer
        layer_0 = model.model.layers[0]
        print(f"\nCalling layer 0 (type={type(layer_0).__name__})...")
        
        # This is what the model.forward() does
        layer_outputs = layer_0(
            hidden_states,
            attention_mask=inputs.get('attention_mask'),
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=True,  # This is True during generation!
        )
        
        print(f"Layer 0 output: type={type(layer_outputs)}")
        if isinstance(layer_outputs, tuple):
            print(f"  Length: {len(layer_outputs)}")
            print(f"  Element types: {[type(x).__name__ for x in layer_outputs]}")
            if hasattr(layer_outputs[0], 'shape'):
                print(f"  First element shape: {layer_outputs[0].shape}")
        
        # The model then does: hidden_states = layer_outputs[0] if it's a tuple
        # But our NPT layer might be returning something unexpected


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python minimal_reproduction.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    base_model_name = sys.argv[2] if len(sys.argv) > 2 else "meta-llama/Llama-3.1-8B"
    
    minimal_test(checkpoint_path, base_model_name)

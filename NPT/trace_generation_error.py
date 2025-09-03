"""
Trace the exact generation error to understand the root cause.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import LlamaModel
from model import convert_llama_to_npt, NPTLayer
from safetensors.torch import load_file


# Monkey patch the LlamaModel forward to add debugging
original_forward = LlamaModel.forward

def debug_forward(self, *args, **kwargs):
    """Wrapped forward with debugging."""
    input_ids = kwargs.get('input_ids', args[0] if args else None)
    attention_mask = kwargs.get('attention_mask', args[1] if len(args) > 1 else None)
    
    print("\n=== LlamaModel.forward called ===")
    print(f"Input shape: {input_ids.shape if input_ids is not None else 'None'}")
    
    # Get embeddings
    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds
    print(f"After embeddings: type={type(hidden_states)}, shape={hidden_states.shape}")
    
    # Process through first few layers with debugging
    for idx, decoder_layer in enumerate(self.layers[:3]):  # Just first 3 layers
        print(f"\n--- Layer {idx} ({type(decoder_layer).__name__}) ---")
        print(f"Input to layer: type={type(hidden_states)}, ", end="")
        if isinstance(hidden_states, tuple):
            print(f"tuple length={len(hidden_states)}")
        else:
            print(f"shape={hidden_states.shape}")
        
        # Store the arguments being passed
        layer_kwargs = {
            'hidden_states': hidden_states,
            'attention_mask': attention_mask,
            'position_ids': kwargs.get('position_ids'),
            'past_key_value': kwargs.get('past_key_values', {}).get(idx) if kwargs.get('past_key_values') else None,
            'output_attentions': kwargs.get('output_attentions', False),
            'use_cache': kwargs.get('use_cache', False),
        }
        
        print(f"use_cache={layer_kwargs['use_cache']}")
        
        try:
            # Call the layer
            layer_outputs = decoder_layer(**layer_kwargs)
            
            print(f"Output from layer: type={type(layer_outputs)}, ", end="")
            if isinstance(layer_outputs, tuple):
                print(f"tuple length={len(layer_outputs)}")
                for i, elem in enumerate(layer_outputs):
                    elem_info = f"type={type(elem).__name__}"
                    if hasattr(elem, 'shape'):
                        elem_info += f", shape={elem.shape}"
                    print(f"  [{i}]: {elem_info}")
            else:
                print(f"shape={layer_outputs.shape if hasattr(layer_outputs, 'shape') else 'N/A'}")
            
            # CRITICAL: This is how LlamaModel processes the output
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]  # Extract hidden states
                print(f"Extracted hidden_states: type={type(hidden_states)}, shape={hidden_states.shape}")
            else:
                hidden_states = layer_outputs
                
        except Exception as e:
            print(f"ERROR in layer {idx}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Call original forward to complete
    return original_forward(self, *args, **kwargs)


def test_with_patched_forward(checkpoint_path, base_model_name="meta-llama/Llama-3.1-8B"):
    """Test with patched forward to trace the issue."""
    
    print("Loading model with debug patches...")
    
    # Patch the forward method
    LlamaModel.forward = debug_forward
    
    try:
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
        
        # Test generation
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print("\n" + "=" * 60)
        print("STARTING GENERATION TEST")
        print("=" * 60)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
            )
        
        print("\nGeneration successful!")
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Output: '{response}'")
        
    finally:
        # Restore original forward
        LlamaModel.forward = original_forward


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python trace_generation_error.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    test_with_patched_forward(checkpoint_path)

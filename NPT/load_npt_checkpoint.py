"""
Proper NPT checkpoint loading functionality.
This module correctly loads NPT models from checkpoints by:
1. Loading the base model
2. Converting to NPT architecture
3. Loading the saved adapter weights
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import load_file
import json
from typing import Tuple, Optional

# Import NPT conversion function
from model.npt_layer import convert_llama_to_npt


def load_npt_checkpoint(checkpoint_path: str, device_map: str = "auto") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load NPT model from checkpoint correctly.
    
    This function:
    1. Loads the base model architecture
    2. Converts it to NPT
    3. Loads the saved adapter weights
    
    Args:
        checkpoint_path: Path to NPT checkpoint
        device_map: Device mapping for model loading
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading NPT checkpoint from: {checkpoint_path}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    # Load config
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"No config.json found in checkpoint: {checkpoint_path}")
    
    config = AutoConfig.from_pretrained(checkpoint_path)
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    except Exception as e:
        print(f"Could not load tokenizer from checkpoint: {e}")
        # Try to get base model name from training info
        training_info_path = os.path.join(checkpoint_path, "training_info.pt")
        if os.path.exists(training_info_path):
            info = torch.load(training_info_path, map_location="cpu", weights_only=False)
            if 'args' in info and hasattr(info['args'], 'model_name'):
                base_model_name = info['args'].model_name
                print(f"Loading tokenizer from base model: {base_model_name}")
                tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            else:
                # Try to infer from config
                base_model_name = config._name_or_path
                print(f"Loading tokenizer from config name: {base_model_name}")
                tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        else:
            raise ValueError("Could not load tokenizer")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine dtype and adapter config from training info
    dtype = torch.float16
    adapter_config = {
        'd_model': config.hidden_size,
        'd_ffn': config.intermediate_size,
        'r': 16,  # Default rank
        'modulation_type': 'both',
        'consolidation_alpha': 0.1,
        'modulation_scale': 0.1
    }
    
    training_info_path = os.path.join(checkpoint_path, "training_info.pt")
    if os.path.exists(training_info_path):
        try:
            info = torch.load(training_info_path, map_location="cpu", weights_only=False)
            if 'args' in info:
                args = info['args']
                # Get dtype
                if hasattr(args, 'use_quantization') and args.use_quantization:
                    dtype = torch.float32
                    print("Using FP32 (model was trained with quantization)")
                elif hasattr(args, 'use_fp16') and args.use_fp16:
                    dtype = torch.float16
                    print("Using FP16")
                
                # Get adapter config
                if hasattr(args, 'adapter_rank'):
                    adapter_config['r'] = args.adapter_rank
                if hasattr(args, 'modulation_scale'):
                    adapter_config['modulation_scale'] = args.modulation_scale
                if hasattr(args, 'consolidation_alpha'):
                    adapter_config['consolidation_alpha'] = args.consolidation_alpha
                    
                print(f"Adapter config from training: r={adapter_config['r']}, "
                      f"modulation_scale={adapter_config['modulation_scale']}, "
                      f"consolidation_alpha={adapter_config['consolidation_alpha']}")
        except Exception as e:
            print(f"Warning: Could not load training info: {e}")
    
    # Step 1: Create base model
    print("Step 1: Creating base model architecture...")
    base_model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=dtype
    )
    
    # Step 2: Convert to NPT architecture
    print("Step 2: Converting to NPT architecture...")
    adapter_config['compute_dtype'] = dtype
    model = convert_llama_to_npt(base_model, adapter_config)
    
    # Step 3: Load all weights from checkpoint
    print("Step 3: Loading checkpoint weights...")
    
    # Find weight files
    weight_files = []
    if os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
        weight_files.append(os.path.join(checkpoint_path, "model.safetensors"))
    else:
        # Look for sharded weights
        for f in os.listdir(checkpoint_path):
            if f.startswith("model-") and f.endswith(".safetensors"):
                weight_files.append(os.path.join(checkpoint_path, f))
    
    if not weight_files:
        # Try pytorch format
        if os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
            weight_files.append(os.path.join(checkpoint_path, "pytorch_model.bin"))
        else:
            # Look for sharded pytorch weights
            for f in os.listdir(checkpoint_path):
                if f.startswith("pytorch_model-") and f.endswith(".bin"):
                    weight_files.append(os.path.join(checkpoint_path, f))
    
    if not weight_files:
        raise ValueError(f"No weight files found in checkpoint: {checkpoint_path}")
    
    # Load all weights
    state_dict = {}
    for weight_file in sorted(weight_files):
        print(f"  Loading weights from: {os.path.basename(weight_file)}")
        if weight_file.endswith('.safetensors'):
            weights = load_file(weight_file)
        else:
            weights = torch.load(weight_file, map_location="cpu", weights_only=False)
        state_dict.update(weights)
    
    # Load weights into model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # Check that NPT weights were loaded
    npt_keys_loaded = [k for k in state_dict.keys() if 'adapter' in k]
    print(f"Loaded {len(npt_keys_loaded)} NPT adapter weights")
    
    if missing_keys:
        # Filter out expected missing keys (e.g., position_ids)
        important_missing = [k for k in missing_keys if 'position_ids' not in k]
        if important_missing:
            print(f"Warning: Missing keys: {important_missing[:5]}...")  # Show first 5
    
    # Move model to device
    if device_map == "auto":
        from accelerate import dispatch_model, infer_auto_device_map
        device_map = infer_auto_device_map(model, max_memory={0: "10GiB", "cpu": "30GiB"})
        model = dispatch_model(model, device_map=device_map)
    else:
        model = model.to(device_map)
    
    # Verify NPT layers
    npt_layers = sum(1 for _, module in model.named_modules() 
                    if 'NPTLayer' in str(type(module)))
    print(f"Model has {npt_layers} NPT layers")
    
    if npt_layers == 0:
        raise ValueError("NPT conversion failed - no NPT layers found!")
    
    # Set model to eval mode
    model.eval()
    
    return model, tokenizer


def test_loading():
    """Test the NPT checkpoint loading."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_npt_checkpoint.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    # Load model
    model, tokenizer = load_npt_checkpoint(checkpoint_path)
    
    # Test generation
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
    
    # Check NPT functionality
    print("\n--- NPT Layer Analysis ---")
    for i, layer in enumerate(model.model.layers[:3]):  # Check first 3 layers
        if hasattr(layer, 'adapter'):
            print(f"Layer {i}: Has NPT adapter")
            if hasattr(layer, 'consolidate_weights'):
                print(f"  - Has consolidate_weights method")
            print(f"  - Modulation scale: {layer.modulation_scale}")
            print(f"  - Consolidation alpha: {layer.consolidation_alpha}")


if __name__ == "__main__":
    test_loading()

"""
Proper NPT checkpoint loading that reconstructs the NPT architecture.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model.npt_layer import convert_llama_to_npt
from utils import setup_logging


def load_npt_checkpoint(checkpoint_path, device_map="auto"):
    """
    Load NPT model from checkpoint with proper architecture reconstruction.
    
    This function:
    1. Loads the base model configuration
    2. Creates a base Llama model
    3. Converts it to NPT architecture
    4. Loads the saved NPT adapter weights
    
    Args:
        checkpoint_path: Path to NPT checkpoint
        device_map: Device map for model loading
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger = setup_logging()
    logger.info(f"Loading NPT checkpoint from {checkpoint_path}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    # Load training info to get configuration
    training_info_path = os.path.join(checkpoint_path, "training_info.pt")
    training_info = None
    base_model_name = None
    adapter_config = {}
    
    if os.path.exists(training_info_path):
        try:
            training_info = torch.load(training_info_path, map_location="cpu", weights_only=False)
            if 'args' in training_info:
                args = training_info['args']
                base_model_name = getattr(args, 'model_name', None)
                
                # Extract adapter configuration
                adapter_config = {
                    'r': getattr(args, 'adapter_rank', 16),
                    'modulation_scale': getattr(args, 'modulation_scale', 0.1),
                    'consolidation_alpha': getattr(args, 'consolidation_alpha', 0.1),
                }
                
                # Determine dtype
                use_quantization = getattr(args, 'use_quantization', False)
                dtype = torch.float32 if use_quantization else torch.float16
                
                logger.info(f"Base model: {base_model_name}")
                logger.info(f"Adapter config: {adapter_config}")
                logger.info(f"Using dtype: {dtype}")
        except Exception as e:
            logger.warning(f"Could not load training info: {e}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    except:
        if base_model_name:
            logger.info(f"Loading tokenizer from base model: {base_model_name}")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        else:
            raise ValueError("Could not load tokenizer and no base model name found")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine base model to load
    if base_model_name is None:
        # Try to get from config.json
        config_path = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_path):
            config = AutoConfig.from_pretrained(checkpoint_path)
            # Try to infer base model from architecture
            if hasattr(config, '_name_or_path'):
                base_model_name = config._name_or_path
    
    if base_model_name is None:
        # Default to a common base model
        base_model_name = "meta-llama/Llama-3.2-1B"
        logger.warning(f"Could not determine base model, defaulting to {base_model_name}")
    
    # Step 1: Load the base Llama model
    logger.info(f"Loading base model architecture from {base_model_name}")
    dtype = adapter_config.get('dtype', torch.float16)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True
    )
    
    # Step 2: Convert to NPT architecture
    logger.info("Converting to NPT architecture...")
    model = convert_llama_to_npt(base_model, adapter_config)
    
    # Ensure model is in eval mode before loading weights
    model.eval()
    
    # Step 3: Load the saved NPT state dict
    logger.info("Loading NPT adapter weights...")
    
    # Try multiple possible paths for the state dict
    state_dict_paths = [
        os.path.join(checkpoint_path, "model.safetensors"),
        os.path.join(checkpoint_path, "pytorch_model.bin"),
        os.path.join(checkpoint_path, "adapter_model.safetensors"),
        os.path.join(checkpoint_path, "adapter_model.bin"),
    ]
    
    state_dict = None
    for path in state_dict_paths:
        if os.path.exists(path):
            logger.info(f"Loading state dict from {path}")
            if path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(path)
            else:
                state_dict = torch.load(path, map_location="cpu")
            break
    
    if state_dict is None:
        logger.warning("No state dict found, model will use randomly initialized adapters")
    else:
        # Load state dict, focusing on adapter weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        # Log what was loaded
        adapter_keys_loaded = [k for k in state_dict.keys() if 'adapter' in k]
        logger.info(f"Loaded {len(adapter_keys_loaded)} adapter weight tensors")
        
        if missing_keys:
            logger.warning(f"Missing keys: {len(missing_keys)} keys")
            # Only show first few
            for key in missing_keys[:5]:
                logger.warning(f"  - {key}")
                
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
    
    # Verify NPT conversion
    npt_layers = sum(1 for _, module in model.named_modules() 
                    if 'NPTLayer' in str(type(module)))
    logger.info(f"Model has {npt_layers} NPT layers")
    
    if npt_layers == 0:
        raise ValueError("NPT conversion failed - no NPT layers found")
    
    # Ensure model is in eval mode for inference
    model.eval()
    
    # Double-check all NPT layers are in eval mode
    for module in model.modules():
        if 'NPTLayer' in str(type(module)):
            module.eval()
    
    return model, tokenizer

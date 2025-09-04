#!/usr/bin/env python3
"""Diagnose NPT model save/load issues."""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.npt_layer import convert_llama_to_npt
from load_npt_checkpoint import load_npt_checkpoint
from utils import save_checkpoint, setup_logging


def diagnose_npt_save_load():
    """Diagnose what gets saved and loaded in NPT models."""
    logger = setup_logging()
    
    # 1. Create a simple NPT model
    logger.info("Creating NPT model...")
    base_model_name = "meta-llama/Llama-3.2-1B"
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Convert to NPT
    adapter_config = {
        'r': 16,
        'd_model': base_model.config.hidden_size,
        'd_ffn': base_model.config.intermediate_size,
        'modulation_scale': 0.1
    }
    npt_model = convert_llama_to_npt(base_model, adapter_config)
    
    # 2. Check what parameters are in the NPT model
    logger.info("\n=== NPT Model Parameters ===")
    adapter_params = []
    base_params = []
    for name, param in npt_model.named_parameters():
        if 'adapter' in name:
            adapter_params.append(name)
        else:
            base_params.append(name)
    
    logger.info(f"Adapter parameters ({len(adapter_params)}):")
    for name in adapter_params[:5]:  # Show first 5
        logger.info(f"  - {name}")
    if len(adapter_params) > 5:
        logger.info(f"  ... and {len(adapter_params) - 5} more")
    
    logger.info(f"\nBase parameters: {len(base_params)} total")
    
    # 3. Save the model
    save_path = "./test_checkpoint"
    logger.info(f"\n=== Saving model to {save_path} ===")
    save_checkpoint(
        model=npt_model,
        tokenizer=tokenizer,
        save_path=save_path,
        additional_info={
            'adapter_config': adapter_config
        }
    )
    
    # 4. Check what was saved
    logger.info("\n=== Checking saved files ===")
    saved_files = []
    for root, dirs, files in os.walk(save_path):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), save_path)
            saved_files.append(rel_path)
            logger.info(f"  - {rel_path}")
    
    # 5. Load the state dict and check keys
    logger.info("\n=== Analyzing saved state dict ===")
    state_dict_paths = [
        os.path.join(save_path, "model.safetensors"),
        os.path.join(save_path, "pytorch_model.bin"),
        os.path.join(save_path, "adapter_model.safetensors"),
        os.path.join(save_path, "adapter_model.bin"),
    ]
    
    state_dict = None
    loaded_from = None
    for path in state_dict_paths:
        if os.path.exists(path):
            loaded_from = path
            if path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(path)
            else:
                state_dict = torch.load(path, map_location="cpu")
            break
    
    if state_dict:
        logger.info(f"Loaded state dict from: {loaded_from}")
        adapter_keys_saved = [k for k in state_dict.keys() if 'adapter' in k]
        base_keys_saved = [k for k in state_dict.keys() if 'adapter' not in k]
        
        logger.info(f"\nSaved adapter keys ({len(adapter_keys_saved)}):")
        for key in adapter_keys_saved[:5]:
            logger.info(f"  - {key}")
        if len(adapter_keys_saved) > 5:
            logger.info(f"  ... and {len(adapter_keys_saved) - 5} more")
        
        logger.info(f"\nSaved base keys: {len(base_keys_saved)} total")
    else:
        logger.error("No state dict found!")
    
    # 6. Load using load_npt_checkpoint
    logger.info("\n=== Loading model with load_npt_checkpoint ===")
    loaded_model, loaded_tokenizer = load_npt_checkpoint(save_path)
    
    # 7. Check loaded model parameters
    logger.info("\n=== Loaded Model Parameters ===")
    loaded_adapter_params = []
    for name, param in loaded_model.named_parameters():
        if 'adapter' in name:
            loaded_adapter_params.append(name)
    
    logger.info(f"Loaded adapter parameters: {len(loaded_adapter_params)}")
    
    # 8. Compare adapter weights
    logger.info("\n=== Comparing Adapter Weights ===")
    if len(adapter_params) > 0 and len(loaded_adapter_params) > 0:
        # Pick first adapter parameter to check
        param_name = adapter_params[0]
        
        original_param = dict(npt_model.named_parameters())[param_name]
        try:
            loaded_param = dict(loaded_model.named_parameters())[param_name]
            
            # Check if they're the same
            if torch.allclose(original_param, loaded_param, atol=1e-6):
                logger.info(f"✓ Adapter weights match for {param_name}")
            else:
                diff = torch.abs(original_param - loaded_param).max().item()
                logger.error(f"✗ Adapter weights differ for {param_name}, max diff: {diff}")
        except KeyError:
            logger.error(f"✗ Parameter {param_name} not found in loaded model")
    
    # 9. Test generation with both models
    logger.info("\n=== Testing Generation ===")
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Original model
    device = next(npt_model.parameters()).device
    inputs_device = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        original_output = npt_model.generate(
            **inputs_device,
            max_new_tokens=20,
            temperature=0.1,
            do_sample=True
        )
    original_text = tokenizer.decode(original_output[0], skip_special_tokens=True)
    
    # Loaded model
    device = next(loaded_model.parameters()).device
    inputs_device = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        loaded_output = loaded_model.generate(
            **inputs_device,
            max_new_tokens=20,
            temperature=0.1,
            do_sample=True
        )
    loaded_text = tokenizer.decode(loaded_output[0], skip_special_tokens=True)
    
    logger.info(f"Original model output: {original_text}")
    logger.info(f"Loaded model output: {loaded_text}")
    
    # Clean up
    import shutil
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        logger.info(f"\nCleaned up {save_path}")


if __name__ == "__main__":
    diagnose_npt_save_load()

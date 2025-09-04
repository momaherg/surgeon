#!/usr/bin/env python3
"""Test NPT model generation after save/load cycle."""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.npt_layer import convert_llama_to_npt
from load_npt_checkpoint import load_npt_checkpoint
from utils import save_checkpoint, setup_logging
import shutil


def test_npt_generation():
    """Test that NPT models can generate text correctly after save/load."""
    logger = setup_logging()
    
    # Test parameters
    base_model_name = "meta-llama/Llama-3.2-1B"
    test_prompt = "The capital of France is"
    checkpoint_dir = "./test_generation_checkpoint"
    
    logger.info("=== Creating NPT Model ===")
    
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
    npt_model.eval()
    
    # Test generation BEFORE saving
    logger.info("\n=== Testing Generation BEFORE Save ===")
    inputs = tokenizer(test_prompt, return_tensors="pt")
    device = next(npt_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output_before = npt_model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    text_before = tokenizer.decode(output_before[0], skip_special_tokens=True)
    logger.info(f"Output: {text_before}")
    
    # Save the model
    logger.info("\n=== Saving Model ===")
    save_checkpoint(
        model=npt_model,
        tokenizer=tokenizer,
        save_path=checkpoint_dir,
        additional_info={'adapter_config': adapter_config}
    )
    
    # Load the model
    logger.info("\n=== Loading Model ===")
    loaded_model, loaded_tokenizer = load_npt_checkpoint(checkpoint_dir)
    
    # Test generation AFTER loading
    logger.info("\n=== Testing Generation AFTER Load ===")
    inputs = loaded_tokenizer(test_prompt, return_tensors="pt")
    device = next(loaded_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output_after = loaded_model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=loaded_tokenizer.pad_token_id
        )
    
    text_after = loaded_tokenizer.decode(output_after[0], skip_special_tokens=True)
    logger.info(f"Output: {text_after}")
    
    # Compare outputs
    logger.info("\n=== Summary ===")
    logger.info(f"Before save: {text_before}")
    logger.info(f"After load:  {text_after}")
    
    # Clean up
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        logger.info(f"\nCleaned up {checkpoint_dir}")
    
    logger.info("\n✓ Test completed successfully!")


if __name__ == "__main__":
    test_npt_generation()

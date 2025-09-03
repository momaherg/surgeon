"""Test NPT checkpoint with base model comparison."""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import load_file
from model import convert_llama_to_npt, NPTLayer
from utils import get_quantization_config, setup_logging


def load_npt_model(checkpoint_path, base_model_name):
    """Load NPT model from checkpoint."""
    logger = setup_logging()
    logger.info(f"Loading NPT from {checkpoint_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load config
    config = AutoConfig.from_pretrained(base_model_name)
    
    # Check checkpoint info
    checkpoint_info_path = os.path.join(checkpoint_path, "training_info.pt")
    use_quantization = False
    adapter_config = None
    adapter_rank = 16
    modulation_scale = 0.1
    
    if os.path.exists(checkpoint_info_path):
        try:
            checkpoint_info = torch.load(checkpoint_info_path, map_location="cpu", weights_only=False)
            if 'args' in checkpoint_info:
                args = checkpoint_info['args']
                use_quantization = args.use_quantization if hasattr(args, 'use_quantization') else False
                if hasattr(args, 'adapter_rank'):
                    adapter_rank = args.adapter_rank
                    logger.info(f"Using adapter rank from checkpoint: r={adapter_rank}")
                if hasattr(args, 'modulation_scale'):
                    modulation_scale = args.modulation_scale
            if 'adapter_config' in checkpoint_info:
                adapter_config = checkpoint_info['adapter_config']
        except Exception as e:
            logger.warning(f"Could not load training info: {e}")
    
    # Determine dtype
    model_dtype = torch.float32 if use_quantization else torch.float16
    quantization_config = get_quantization_config() if use_quantization else None
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        config=config,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=model_dtype
    )
    
    # Convert to NPT
    if adapter_config is None:
        adapter_config = {
            'r': adapter_rank,
            'd_model': config.hidden_size,
            'd_ffn': config.intermediate_size,
            'compute_dtype': torch.float32 if use_quantization else model_dtype,
            'modulation_scale': modulation_scale
        }
    
    model = convert_llama_to_npt(model, adapter_config)
    
    # Load checkpoint weights from safetensors
    safetensor_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors') and f.startswith('model-')]
    
    if safetensor_files:
        state_dict = {}
        for file in sorted(safetensor_files):
            shard_path = os.path.join(checkpoint_path, file)
            shard_dict = load_file(shard_path)
            state_dict.update(shard_dict)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded NPT weights from {len(safetensor_files)} safetensors shards")
    
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Test NPT checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to NPT checkpoint")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B", help="Base model name")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Test prompt")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--with_comparison", action="store_true", help="Compare with base model")
    
    args = parser.parse_args()
    
    # Load NPT model
    npt_model, tokenizer = load_npt_model(args.checkpoint_path, args.base_model)
    
    # Test generation
    print(f"\nPrompt: {args.prompt}")
    
    inputs = tokenizer(args.prompt, return_tensors="pt")
    device = next(npt_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = npt_model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = response[len(args.prompt):].strip()
    
    print(f"\nNPT Response: {completion}")
    
    # Compare with base model if requested
    if args.with_comparison:
        print("\nLoading base model for comparison...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            torch_dtype=torch.float16
        )
        base_model.eval()
        
        with torch.no_grad():
            base_outputs = base_model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
        base_completion = base_response[len(args.prompt):].strip()
        
        print(f"\nBase Model Response: {base_completion}")


if __name__ == "__main__":
    main()
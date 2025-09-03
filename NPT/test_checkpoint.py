"""Test NPT checkpoint with base model comparison."""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils import setup_logging


def load_npt_model(checkpoint_path):
    """Load NPT model from checkpoint - fixed version."""
    logger = setup_logging()
    logger.info(f"Loading NPT from {checkpoint_path}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    # Load tokenizer from checkpoint
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    except Exception as e:
        logger.warning(f"Could not load tokenizer from checkpoint: {e}")
        # Try to get base model name from training info
        training_info_path = os.path.join(checkpoint_path, "training_info.pt")
        if os.path.exists(training_info_path):
            try:
                info = torch.load(training_info_path, map_location="cpu", weights_only=False)
                if 'args' in info and hasattr(info['args'], 'model_name'):
                    base_model_name = info['args'].model_name
                    logger.info(f"Loading tokenizer from base model: {base_model_name}")
                    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                else:
                    raise ValueError("Could not find base model name in training info")
            except Exception as e2:
                raise ValueError(f"Failed to load tokenizer: {e2}")
        else:
            raise ValueError("No tokenizer found and no training info available")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine dtype from training info
    model_dtype = torch.float16  # Default
    training_info_path = os.path.join(checkpoint_path, "training_info.pt")
    
    if os.path.exists(training_info_path):
        try:
            checkpoint_info = torch.load(training_info_path, map_location="cpu", weights_only=False)
            if 'args' in checkpoint_info:
                args = checkpoint_info['args']
                use_quantization = hasattr(args, 'use_quantization') and args.use_quantization
                model_dtype = torch.float32 if use_quantization else torch.float16
                
                # Log training configuration
                logger.info("Training configuration:")
                if hasattr(args, 'adapter_rank'):
                    logger.info(f"  Adapter rank: {args.adapter_rank}")
                if hasattr(args, 'modulation_scale'):
                    logger.info(f"  Modulation scale: {args.modulation_scale}")
                if hasattr(args, 'learning_rate'):
                    logger.info(f"  Learning rate: {args.learning_rate}")
        except Exception as e:
            logger.warning(f"Could not load training info: {e}")
    
    # Load the NPT model directly from checkpoint (it's already converted!)
    logger.info(f"Loading NPT model with dtype={model_dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map="auto",
        torch_dtype=model_dtype,
        trust_remote_code=True
    )
    
    # Verify NPT conversion
    npt_layers = 0
    for name, module in model.named_modules():
        if 'NPTLayer' in str(type(module)):
            npt_layers += 1
    
    if npt_layers > 0:
        logger.info(f"Verified: Model has {npt_layers} NPT layers")
    else:
        logger.warning("No NPT layers found - model might not be properly converted")
    
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Test NPT checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to NPT checkpoint")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B", help="Base model name for comparison")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Test prompt")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--with_comparison", action="store_true", help="Compare with base model")
    
    args = parser.parse_args()
    
    # Load NPT model (fixed: no base_model needed for loading)
    npt_model, tokenizer = load_npt_model(args.checkpoint_path)
    
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
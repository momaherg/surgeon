"""Test NPT checkpoint with base model comparison."""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils import setup_logging
from load_npt_checkpoint import load_npt_checkpoint


def load_npt_model(checkpoint_path):
    """Load NPT model from checkpoint - fixed version."""
    logger = setup_logging()
    logger.info(f"Loading NPT from {checkpoint_path}")
    
    # Use the proper NPT checkpoint loading function
    return load_npt_checkpoint(checkpoint_path, device_map="auto")


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
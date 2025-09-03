"""
Fixed quick test script for NPT checkpoints.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List


def load_npt_checkpoint(checkpoint_path: str):
    """
    Load NPT model from checkpoint - simplified and correct version.
    
    The checkpoint contains the full NPT model, not just adapter weights,
    so we can load it directly without conversion.
    """
    print(f"Loading NPT checkpoint from: {checkpoint_path}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    # Check for model files
    has_safetensors = any(f.endswith('.safetensors') for f in os.listdir(checkpoint_path))
    has_pytorch = os.path.exists(os.path.join(checkpoint_path, 'pytorch_model.bin'))
    
    if not has_safetensors and not has_pytorch:
        raise ValueError(f"No model files found in checkpoint: {checkpoint_path}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    except Exception as e:
        print(f"Warning: Could not load tokenizer from checkpoint: {e}")
        print("Trying to load from training_info.pt...")
        
        # Try to get base model name from training info
        training_info_path = os.path.join(checkpoint_path, "training_info.pt")
        if os.path.exists(training_info_path):
            try:
                info = torch.load(training_info_path, map_location="cpu", weights_only=False)
                if 'args' in info and hasattr(info['args'], 'model_name'):
                    base_model_name = info['args'].model_name
                    print(f"Loading tokenizer from base model: {base_model_name}")
                    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                else:
                    raise ValueError("Could not find base model name in training info")
            except Exception as e2:
                raise ValueError(f"Failed to load tokenizer: {e2}")
        else:
            raise ValueError("No tokenizer found and no training info available")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine dtype from training info if available
    dtype = torch.float16  # Default
    device_map = "auto"
    
    training_info_path = os.path.join(checkpoint_path, "training_info.pt")
    if os.path.exists(training_info_path):
        try:
            info = torch.load(training_info_path, map_location="cpu", weights_only=False)
            if 'args' in info:
                args = info['args']
                # Use FP32 if model was trained with quantization
                if hasattr(args, 'use_quantization') and args.use_quantization:
                    dtype = torch.float32
                    print("Using FP32 (model was trained with quantization)")
                elif hasattr(args, 'use_fp16') and args.use_fp16:
                    dtype = torch.float16
                    print("Using FP16")
                
                # Show training configuration
                print("\nTraining configuration:")
                if hasattr(args, 'adapter_rank'):
                    print(f"  Adapter rank: {args.adapter_rank}")
                if hasattr(args, 'modulation_scale'):
                    print(f"  Modulation scale: {args.modulation_scale}")
                if hasattr(args, 'learning_rate'):
                    print(f"  Learning rate: {args.learning_rate}")
                print()
        except Exception as e:
            print(f"Warning: Could not load training info: {e}")
    
    # Load the model directly from checkpoint
    # The checkpoint contains the full NPT model, not just adapters
    print(f"Loading NPT model with dtype={dtype}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True  # NPT layers might need this
        )
        print("Successfully loaded NPT model!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    model.eval()
    
    # Verify NPT conversion
    npt_layers = 0
    for name, module in model.named_modules():
        if 'NPTLayer' in str(type(module)):
            npt_layers += 1
    
    if npt_layers > 0:
        print(f"Verified: Model has {npt_layers} NPT layers")
    else:
        print("Warning: No NPT layers found - model might not be properly converted")
    
    return model, tokenizer


def test_generation(
    model, 
    tokenizer, 
    prompts: List[str], 
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    """Test model generation with given prompts."""
    device = next(model.parameters()).device
    
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Prompt: {prompt}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = response[len(prompt):].strip()
            
            print(f"Completion: {completion}\n")
            print("-" * 60 + "\n")
        except Exception as e:
            print(f"Error during generation: {e}\n")
            print("-" * 60 + "\n")


def interactive_mode(model, tokenizer, max_new_tokens: int = 100):
    """Run interactive generation mode."""
    device = next(model.parameters()).device
    
    print("\n" + "="*60)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("="*60 + "\n")
    
    while True:
        user_prompt = input("Enter prompt: ").strip()
        
        if user_prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_prompt:
            continue
        
        # Generate response
        inputs = tokenizer(user_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = response[len(user_prompt):].strip()
            
            print(f"\nNPT: {completion}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python quick_test_npt_fixed.py <checkpoint_path> [--interactive]")
        print("Example: python quick_test_npt_fixed.py ./outputs/npt-improved-1B/checkpoint-500")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    interactive = "--interactive" in sys.argv
    
    # Load model
    try:
        model, tokenizer = load_npt_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Test prompts
    test_prompts = [
        "The capital of France is",
        "Artificial intelligence is",
        "The meaning of life is",
        "Python is a programming language that",
        "Climate change is caused by",
        "The solar system consists of",
        "Machine learning algorithms can",
        "The human brain is",
        "Quantum computing will",
        "The internet has revolutionized"
    ]
    
    print("\n" + "="*60)
    print("NPT MODEL TEST RESULTS")
    print("="*60 + "\n")
    
    # Run test generation
    test_generation(model, tokenizer, test_prompts)
    
    # Run interactive mode if requested
    if interactive:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()

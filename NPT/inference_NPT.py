#!/usr/bin/env python3
"""
NPT Model Inference Script
Test text completions with a trained NPT model through the terminal.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import Optional, Dict
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    GenerationConfig
)

# Import the NPT modules (ensure these are in your Python path)
# You'll need to adjust the import path based on your project structure
try:
    from model import NPTAdapter, NPTLayer, convert_llama_to_npt
except ImportError:
    print("Error: Could not import NPT modules. Please ensure npt_model.py is in your Python path.")
    sys.exit(1)

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

def get_quantization_config():
    """Get 4-bit quantization configuration."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

def load_npt_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
    use_quantization: bool = False,
    verbose: bool = False
) -> tuple:
    """
    Load NPT model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        device: Device to load model on
        use_quantization: Whether to use 4-bit quantization
        verbose: Whether to print detailed logs
        
    Returns:
        tuple: (model, tokenizer, config)
    """
    logger = setup_logging(verbose)
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load config
    config = AutoConfig.from_pretrained(checkpoint_path)
    
    # Load training info if available
    training_info_path = os.path.join(checkpoint_path, "training_info.pt")
    adapter_config = None
    if os.path.exists(training_info_path):
        logger.info("Loading training info...")
        training_info = torch.load(training_info_path, map_location="cpu")
        adapter_config = training_info.get('adapter_config', None)
        args = training_info.get('args', None)
        
        if args and verbose:
            logger.debug(f"Training args: {args}")
    
    # Setup quantization if needed
    quantization_config = get_quantization_config() if use_quantization else None
    
    # Determine dtype
    dtype = torch.float16 if use_quantization else torch.float32
    
    # Load base model
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        config=config,
        quantization_config=quantization_config,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=dtype
    )
    
    # Check if this is already an NPT model by looking for adapter modules
    has_adapters = any('adapter' in name for name, _ in model.named_parameters())
    
    if not has_adapters:
        # Convert to NPT if needed
        logger.info("Converting to NPT architecture...")
        
        if adapter_config is None:
            # Use default adapter config
            adapter_config = {
                'r': 16,
                'd_model': config.hidden_size,
                'd_ffn': config.intermediate_size,
                'compute_dtype': dtype,
                'modulation_type': 'outer_product',
                'modulation_scale': 0.1,
                'init_strategy': 'adaptive',
                'init_scale': 1.0
            }
            logger.warning("No adapter config found, using defaults")
        
        model = convert_llama_to_npt(model, adapter_config)
    else:
        logger.info("Model already has NPT architecture")
    
    # Move to device if not using auto device map
    if device != "cuda" and device != "auto":
        model = model.to(device)
    
    # Set to eval mode
    model.eval()
    
    logger.info("Model loaded successfully!")
    return model, tokenizer, config

def generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    repetition_penalty: float = 1.1,
    device: str = "cuda"
) -> str:
    """
    Generate text completion for a given prompt.
    
    Args:
        model: The NPT model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling
        repetition_penalty: Repetition penalty
        device: Device to run on
        
    Returns:
        str: Generated text
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Move to device
    if device != "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Set up generation config
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the output
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text

def interactive_mode(model, tokenizer, device: str = "cuda", generation_params: dict = None):
    """
    Run interactive text completion mode.
    
    Args:
        model: The NPT model
        tokenizer: The tokenizer
        device: Device to run on
        generation_params: Dictionary of generation parameters
    """
    print("\n" + "="*50)
    print("NPT Model Interactive Mode")
    print("="*50)
    print("Commands:")
    print("  /quit or /exit - Exit the program")
    print("  /params - Show current generation parameters")
    print("  /set <param> <value> - Set generation parameter")
    print("  /help - Show this help message")
    print("\nEnter your prompt and press Enter to generate:")
    print("="*50 + "\n")
    
    # Default generation parameters
    if generation_params is None:
        generation_params = {
            'max_new_tokens': 100,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'do_sample': True,
            'repetition_penalty': 1.1
        }
    
    while True:
        try:
            # Get user input
            prompt = input("\n> ").strip()
            
            # Check for commands
            if prompt.lower() in ['/quit', '/exit']:
                print("Goodbye!")
                break
            
            elif prompt.lower() == '/help':
                print("\nCommands:")
                print("  /quit or /exit - Exit the program")
                print("  /params - Show current generation parameters")
                print("  /set <param> <value> - Set generation parameter")
                print("    Available params: max_new_tokens, temperature, top_p, top_k, repetition_penalty")
                print("    Example: /set temperature 0.5")
                print("  /help - Show this help message")
                continue
            
            elif prompt.lower() == '/params':
                print("\nCurrent generation parameters:")
                for key, value in generation_params.items():
                    print(f"  {key}: {value}")
                continue
            
            elif prompt.lower().startswith('/set '):
                parts = prompt.split()
                if len(parts) >= 3:
                    param = parts[1]
                    try:
                        value = float(parts[2]) if '.' in parts[2] else int(parts[2])
                        if param in generation_params:
                            generation_params[param] = value
                            print(f"Set {param} to {value}")
                        else:
                            print(f"Unknown parameter: {param}")
                    except ValueError:
                        print(f"Invalid value: {parts[2]}")
                else:
                    print("Usage: /set <param> <value>")
                continue
            
            elif prompt.startswith('/'):
                print("Unknown command. Type /help for available commands.")
                continue
            
            elif not prompt:
                continue
            
            # Generate completion
            print("\nGenerating...", end='', flush=True)
            
            completion = generate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                **generation_params
            )
            
            print("\r" + " "*20 + "\r", end='')  # Clear "Generating..." message
            print(f"\n{completion}\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
        except Exception as e:
            print(f"\nError: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Test NPT model with text completions")
    
    # Model arguments
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the NPT model checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run model on (default: cuda if available)"
    )
    parser.add_argument(
        "--use-quantization",
        action="store_true",
        help="Use 4-bit quantization"
    )
    
    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate from (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Use greedy decoding instead of sampling"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading NPT model from {args.checkpoint_path}...")
    model, tokenizer, config = load_npt_checkpoint(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        use_quantization=args.use_quantization,
        verbose=args.verbose
    )
    
    # Prepare generation parameters
    generation_params = {
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'do_sample': not args.no_sample,
        'repetition_penalty': args.repetition_penalty
    }
    
    if args.prompt:
        # Single prompt mode
        print(f"\nPrompt: {args.prompt}")
        print("\nGenerating completion...")
        
        completion = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=args.device,
            **generation_params
        )
        
        print(f"\nCompletion:\n{completion}")
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, args.device, generation_params)

if __name__ == "__main__":
    main()
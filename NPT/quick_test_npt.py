"""
Quick test script for NPT checkpoints - simplified version.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model import convert_llama_to_npt
from utils import get_quantization_config


def load_npt_checkpoint(checkpoint_path, base_model_name="meta-llama/Llama-3.1-8B"):
    """Load NPT model from checkpoint."""
    print(f"Loading NPT checkpoint from: {checkpoint_path}")
    
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
    
    if os.path.exists(checkpoint_info_path):
        checkpoint_info = torch.load(checkpoint_info_path, map_location="cpu")
        if 'args' in checkpoint_info:
            use_quantization = checkpoint_info['args'].use_quantization
        if 'adapter_config' in checkpoint_info:
            adapter_config = checkpoint_info['adapter_config']
    
    # Determine dtype
    if use_quantization:
        model_dtype = torch.float32
        quantization_config = get_quantization_config()
    else:
        model_dtype = torch.float16
        quantization_config = None
    
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
            'r': 16,
            'd_model': config.hidden_size,
            'd_ffn': config.intermediate_size,
            'compute_dtype': torch.float32 if use_quantization else model_dtype
        }
    
    model = convert_llama_to_npt(model, adapter_config)
    
    # Load checkpoint weights
    # Try loading from safetensors format first (newer format)
    safetensors_index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    single_safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    
    if os.path.exists(safetensors_index_path) or os.path.exists(single_safetensors_path):
        # The model is already in safetensors format, load it directly
        print("Loading NPT model from safetensors format...")
        # Actually, since we used from_pretrained above, it should have loaded the weights
        # But we need to reload with NPT architecture
        from safetensors.torch import load_file
        
        # Get all safetensors files
        safetensor_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors') and f.startswith('model-')]
        
        if safetensor_files:
            # Load each shard
            state_dict = {}
            for file in sorted(safetensor_files):
                shard_path = os.path.join(checkpoint_path, file)
                shard_dict = load_file(shard_path)
                state_dict.update(shard_dict)
            
            # Load the state dict into the model
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded NPT weights from {len(safetensor_files)} safetensors shards")
        else:
            # Single file
            state_dict = load_file(single_safetensors_path)
            model.load_state_dict(state_dict, strict=False)
            print("Loaded NPT weights from single safetensors file")
    elif os.path.exists(pytorch_model_path):
        # Old format
        state_dict = torch.load(pytorch_model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print("Loaded NPT weights from pytorch_model.bin")
    else:
        print("Warning: No checkpoint weights found, using base model weights")
    
    model.eval()
    return model, tokenizer


def test_npt_model(checkpoint_path, base_model_name="meta-llama/Llama-3.1-8B"):
    """Test NPT model with example sentences."""
    
    # Load model
    model, tokenizer = load_npt_checkpoint(checkpoint_path, base_model_name)
    
    # Test sentences
    test_sentences = [
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
    
    for i, prompt in enumerate(test_sentences, 1):
        print(f"[{i}/{len(test_sentences)}] Prompt: {prompt}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = response[len(prompt):].strip()
        
        print(f"Completion: {completion}\n")
        print("-" * 60 + "\n")
    
    # Interactive mode
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
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = response[len(user_prompt):].strip()
        
        print(f"\nNPT: {completion}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python quick_test_npt.py <checkpoint_path> [base_model_name]")
        print("Example: python quick_test_npt.py ./outputs/npt-safe-pretrained/checkpoint-500")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    base_model_name = sys.argv[2] if len(sys.argv) > 2 else "meta-llama/Llama-3.1-8B"
    
    test_npt_model(checkpoint_path, base_model_name)

"""
Debug GPT2 dimensions to understand the architecture.
"""

from transformers import AutoConfig, AutoModelForCausalLM
import torch

def debug_gpt2():
    """Check GPT2 architecture dimensions."""
    print("Debugging GPT2 dimensions...")
    
    # Load config
    config = AutoConfig.from_pretrained("gpt2")
    print(f"\nConfig attributes:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  n_inner: {getattr(config, 'n_inner', 'Not found')}")
    print(f"  n_embd: {getattr(config, 'n_embd', 'Not found')}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    
    # Check first layer's MLP
    layer0 = model.transformer.h[0]
    print(f"\nLayer 0 MLP structure:")
    
    if hasattr(layer0.mlp, 'c_fc'):
        print(f"  c_fc weight shape: {layer0.mlp.c_fc.weight.shape}")
        print(f"  c_proj weight shape: {layer0.mlp.c_proj.weight.shape}")
    
    print(f"\nExpected dimensions:")
    print(f"  d_model: {config.hidden_size}")
    print(f"  d_ffn: {config.n_inner if config.n_inner is not None else 4 * config.hidden_size}")


if __name__ == "__main__":
    debug_gpt2()

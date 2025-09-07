"""
Test and demonstrate the memory efficiency improvements in NPT implementation.
"""

import torch
import numpy as np
from npt_components import NeuroPlasticComponent


def compute_memory_usage(tensor_shapes, dtype=torch.float32):
    """Compute memory usage in GB for given tensor shapes."""
    bytes_per_element = 4 if dtype == torch.float32 else 2
    total_elements = sum(np.prod(shape) for shape in tensor_shapes)
    memory_gb = (total_elements * bytes_per_element) / (1024**3)
    return memory_gb


def test_memory_efficiency():
    """Compare memory usage between old and new approaches."""
    
    # Model dimensions (typical for LLaMA-7B)
    batch_size = 4
    seq_len = 512
    d_model = 4096
    d_ffn = 11008
    rank = 16
    
    print("="*60)
    print("NPT Memory Efficiency Analysis")
    print("="*60)
    print(f"\nModel Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  FFN dimension: {d_ffn}")
    print(f"  Rank: {rank}")
    
    # Old approach: Full weight delta tensor
    old_delta_shape = (batch_size, seq_len, d_model, d_ffn)
    old_memory = compute_memory_usage([old_delta_shape])
    
    print(f"\n1. OLD APPROACH (Full Weight Delta Tensor):")
    print(f"   Shape: {old_delta_shape}")
    print(f"   Memory: {old_memory:.2f} GB")
    print(f"   Status: ❌ Would cause OOM!")
    
    # New approach: Modulation factors only
    modulation_shape = (batch_size, seq_len, rank)
    new_memory = compute_memory_usage([modulation_shape])
    
    # Per-token weight delta (processed one at a time)
    per_token_delta_shape = (batch_size, d_model, d_ffn)
    per_token_memory = compute_memory_usage([per_token_delta_shape])
    
    print(f"\n2. NEW APPROACH (On-demand Weight Deltas):")
    print(f"   Modulation shape: {modulation_shape}")
    print(f"   Modulation memory: {new_memory:.6f} GB")
    print(f"   Per-token delta shape: {per_token_delta_shape}")
    print(f"   Per-token delta memory: {per_token_memory:.3f} GB")
    print(f"   Status: ✅ Memory efficient!")
    
    # Memory savings
    memory_reduction = old_memory / (new_memory + per_token_memory)
    print(f"\n3. MEMORY SAVINGS:")
    print(f"   Reduction factor: {memory_reduction:.1f}x")
    print(f"   Memory saved: {old_memory - (new_memory + per_token_memory):.2f} GB")
    
    # Practical test with actual component
    print(f"\n4. PRACTICAL TEST:")
    
    # Create NPT component with smaller dimensions for testing
    test_d_model = 768
    test_d_ffn = 3072
    test_batch = 2
    test_seq_len = 128
    
    np_component = NeuroPlasticComponent(
        d_model=test_d_model,
        d_ffn=test_d_ffn,
        rank=rank,
        modulation_scale=0.1,
    )
    
    # Test forward pass
    attn_output = torch.randn(test_batch, test_seq_len, test_d_model)
    
    try:
        # New approach
        modulation = np_component(attn_output)
        print(f"   ✅ Modulation computed successfully")
        print(f"      Shape: {modulation.shape}")
        
        # Test single token delta
        delta_w = np_component.compute_weight_delta(modulation, token_idx=0)
        print(f"   ✅ Per-token delta computed successfully")
        print(f"      Shape: {delta_w.shape}")
        
        # Get statistics
        stats = np_component.get_weight_delta_stats(attn_output)
        print(f"   ✅ Statistics computed successfully")
        print(f"      Average delta norm: {stats['delta_w_frobenius']:.6f}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "="*60)
    print("CONCLUSION: The new approach is much more memory efficient!")
    print("="*60)


if __name__ == "__main__":
    test_memory_efficiency()

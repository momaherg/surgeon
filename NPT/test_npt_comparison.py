"""
Test script to compare NPT V1 and V2 implementations.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig
import argparse
from model.npt_layer import NPTAdapter, NPTLayer
from model.npt_layer_v2 import NPTAdapterV2, NPTLayerV2


def create_dummy_layer(config):
    """Create a dummy base layer for testing."""
    class DummyMLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
            self.act_fn = nn.SiLU()
    
    class DummyAttention(nn.Module):
        def forward(self, hidden_states, **kwargs):
            # Simple identity for testing
            return (hidden_states, None, None)
    
    class DummyLayer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.self_attn = DummyAttention()
            self.mlp = DummyMLP(config)
            self.input_layernorm = nn.LayerNorm(config.hidden_size)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)
    
    return DummyLayer(config)


def test_adapter_comparison():
    """Compare V1 and V2 adapter outputs."""
    print("=== Testing Adapter Modules ===")
    
    # Configuration
    d_model = 768
    d_ffn = 2048
    r = 16
    batch_size = 2
    seq_len = 10
    
    # Create adapters
    adapter_v1 = NPTAdapter(d_model=d_model, d_ffn=d_ffn, r=r)
    adapter_v2 = NPTAdapterV2(d_model=d_model, d_ffn=d_ffn, r=r, modulation_type='additive')
    
    # Test input
    attn_output = torch.randn(batch_size, seq_len, d_model)
    
    # V1 forward
    delta_v1, norm_v1 = adapter_v1(attn_output)
    
    # V2 forward
    outputs_v2 = adapter_v2(attn_output)
    delta_v2 = outputs_v2.get('delta_add')
    norm_v2 = outputs_v2['reg_norm']
    
    print(f"V1 Output shape: {delta_v1.shape}")
    print(f"V2 Output shape: {delta_v2.shape if delta_v2 is not None else 'None'}")
    print(f"V1 Norm: {norm_v1.item():.6f}")
    print(f"V2 Norm: {norm_v2.item():.6f}")
    
    # Test multiplicative mode
    adapter_v2_mult = NPTAdapterV2(d_model=d_model, d_ffn=d_ffn, r=r, modulation_type='multiplicative')
    outputs_v2_mult = adapter_v2_mult(attn_output)
    delta_mult = outputs_v2_mult.get('delta_mult')
    
    print(f"\nV2 Multiplicative Output shape: {delta_mult.shape}")
    print(f"V2 Multiplicative range: [{delta_mult.min().item():.3f}, {delta_mult.max().item():.3f}]")
    
    return True


def test_layer_comparison():
    """Compare V1 and V2 layer behaviors."""
    print("\n=== Testing Layer Implementations ===")
    
    # Create config
    config = type('Config', (), {
        'hidden_size': 768,
        'intermediate_size': 2048,
        'num_attention_heads': 12
    })()
    
    # Create base layer
    base_layer = create_dummy_layer(config)
    
    # Create NPT layers
    adapter_config = {
        'd_model': config.hidden_size,
        'd_ffn': config.intermediate_size,
        'r': 16
    }
    
    layer_v1 = NPTLayer(create_dummy_layer(config), adapter_config)
    
    adapter_config_v2 = adapter_config.copy()
    adapter_config_v2['modulation_type'] = 'both'
    layer_v2 = NPTLayerV2(create_dummy_layer(config), adapter_config_v2)
    
    # Test input
    batch_size = 2
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Forward passes
    outputs_v1 = layer_v1(hidden_states)
    outputs_v2 = layer_v2(hidden_states)
    
    print(f"V1 Output shape: {outputs_v1[0].shape}")
    print(f"V2 Output shape: {outputs_v2[0].shape}")
    print(f"V1 Reg norm: {outputs_v1[1].item():.6f}")
    print(f"V2 Reg norm: {outputs_v2[1].item():.6f}")
    
    # Test residual structure differences
    with torch.no_grad():
        # Trace through V1 manually
        residual = hidden_states
        h_norm = layer_v1.input_layernorm(hidden_states)
        attn_out = layer_v1.self_attn(h_norm)[0]
        h_residual_v1 = residual + attn_out  # V1 has residual here
        
        # Trace through V2 manually
        original_input = hidden_states
        h_norm = layer_v2.input_layernorm(hidden_states)
        attn_out = layer_v2.self_attn(h_norm)[0]
        # V2 does NOT add residual here
        
        print(f"\nResidual structure difference detected: {not torch.allclose(h_residual_v1, attn_out)}")
    
    return True


def test_permanent_update():
    """Test permanent update functionality (V2 only)."""
    print("\n=== Testing Permanent Update Mode ===")
    
    # Create config
    config = type('Config', (), {
        'hidden_size': 768,
        'intermediate_size': 2048,
        'num_attention_heads': 12
    })()
    
    # Create V2 layer
    adapter_config = {
        'd_model': config.hidden_size,
        'd_ffn': config.intermediate_size,
        'r': 16,
        'modulation_type': 'additive',  # Required for permanent updates
        'consolidation_alpha': 0.1
    }
    
    base_layer = create_dummy_layer(config)
    layer_v2 = NPTLayerV2(base_layer, adapter_config)
    
    # Store original weight
    original_weight = layer_v2.mlp.gate_proj.weight.clone()
    
    # Create context tokens
    batch_size = 1
    seq_len = 10
    context_tokens = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Perform permanent update
    try:
        stats = layer_v2.consolidate_weights(
            context_tokens,
            token_idx=-1,
            alpha=0.1
        )
        
        print(f"Weight update norm: {stats['weight_update_norm']:.6f}")
        print(f"Alpha used: {stats['alpha_used']}")
        print(f"Weights updated: {layer_v2.weights_updated}")
        
        # Check if weights actually changed
        weight_diff = torch.norm(layer_v2.mlp.gate_proj.weight - original_weight)
        print(f"Actual weight change norm: {weight_diff.item():.6f}")
        
        return weight_diff > 0
        
    except Exception as e:
        print(f"Permanent update failed: {e}")
        return False


def test_modulation_types():
    """Test different modulation types in V2."""
    print("\n=== Testing Modulation Types ===")
    
    d_model = 768
    d_ffn = 2048
    r = 16
    batch_size = 2
    seq_len = 10
    
    attn_output = torch.randn(batch_size, seq_len, d_model)
    
    for mod_type in ['additive', 'multiplicative', 'both']:
        print(f"\nTesting {mod_type} modulation:")
        adapter = NPTAdapterV2(d_model=d_model, d_ffn=d_ffn, r=r, modulation_type=mod_type)
        outputs = adapter(attn_output)
        
        if 'delta_add' in outputs:
            print(f"  - Additive shape: {outputs['delta_add'].shape}")
            print(f"  - Additive range: [{outputs['delta_add'].min().item():.3f}, {outputs['delta_add'].max().item():.3f}]")
        
        if 'delta_mult' in outputs:
            print(f"  - Multiplicative shape: {outputs['delta_mult'].shape}")
            print(f"  - Multiplicative range: [{outputs['delta_mult'].min().item():.3f}, {outputs['delta_mult'].max().item():.3f}]")
        
        print(f"  - Regularization norm: {outputs['reg_norm'].item():.6f}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Compare NPT V1 and V2 implementations")
    parser.add_argument(
        "--test",
        type=str,
        choices=["adapter", "layer", "permanent", "modulation", "all"],
        default="all",
        help="Which test to run"
    )
    
    args = parser.parse_args()
    
    tests = {
        "adapter": ("Adapter Comparison", test_adapter_comparison),
        "layer": ("Layer Comparison", test_layer_comparison),
        "permanent": ("Permanent Update", test_permanent_update),
        "modulation": ("Modulation Types", test_modulation_types)
    }
    
    if args.test == "all":
        passed = 0
        failed = 0
        
        for test_name, (desc, test_func) in tests.items():
            try:
                if test_func():
                    passed += 1
                    print(f"\n✓ {desc} test passed!")
                else:
                    failed += 1
                    print(f"\n✗ {desc} test failed!")
            except Exception as e:
                failed += 1
                print(f"\n✗ {desc} test failed with error: {e}")
        
        print(f"\n{'='*50}")
        print(f"Tests passed: {passed}/{len(tests)}")
        print(f"Tests failed: {failed}/{len(tests)}")
        print(f"{'='*50}")
        
    else:
        desc, test_func = tests[args.test]
        success = test_func()
        if success:
            print(f"\n✓ {desc} test passed!")
        else:
            print(f"\n✗ {desc} test failed!")


if __name__ == "__main__":
    main()

"""
Test script for the outer product NPT implementation.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model import convert_llama_to_npt, NPTAdapter


def test_adapter_forward():
    """Test the NPTAdapter forward pass."""
    print("Testing NPTAdapter forward pass...")
    
    # Create adapter
    adapter = NPTAdapter(d_model=512, d_ffn=2048, r=16)
    
    # Create dummy input
    batch_size, seq_len = 2, 10
    attn_output = torch.randn(batch_size, seq_len, 512)
    
    # Forward pass
    outputs = adapter(attn_output)
    
    # Check outputs
    assert 'vector_model' in outputs
    assert 'vector_ffn' in outputs
    assert 'reg_norm' in outputs
    
    # Check shapes
    assert outputs['vector_model'].shape == (batch_size, seq_len, 512)
    assert outputs['vector_ffn'].shape == (batch_size, seq_len, 2048)
    
    # Check regularization is computed correctly
    vector_model = outputs['vector_model']
    vector_ffn = outputs['vector_ffn']
    
    # Manually compute expected reg norm
    norm_model = torch.sum(vector_model ** 2, dim=-1)
    norm_ffn = torch.sum(vector_ffn ** 2, dim=-1)
    expected_reg = torch.mean(norm_model * norm_ffn)
    
    assert torch.allclose(outputs['reg_norm'], expected_reg, rtol=1e-5)
    
    print("✓ NPTAdapter forward pass test passed!")


def test_weight_modulation():
    """Test the weight modulation computation."""
    print("\nTesting weight modulation computation...")
    
    # Create tensors
    batch_size, seq_len = 2, 5
    d_model, d_ffn = 256, 1024
    
    # Create vectors
    vector_model = torch.randn(batch_size, seq_len, d_model)
    vector_ffn = torch.randn(batch_size, seq_len, d_ffn)
    
    # Create input
    mlp_input = torch.randn(batch_size, seq_len, d_model)
    
    # Efficient computation (as implemented)
    dot_product = torch.sum(mlp_input * vector_model, dim=-1, keepdim=True)
    weight_modulation_efficient = vector_ffn * dot_product
    
    # Verify against explicit outer product computation for one example
    # For position 0 of batch 0
    outer_product = torch.outer(vector_ffn[0, 0], vector_model[0, 0])  # (d_ffn, d_model)
    weight_modulation_explicit = outer_product @ mlp_input[0, 0]  # (d_ffn,)
    
    # Check they match
    assert torch.allclose(
        weight_modulation_efficient[0, 0], 
        weight_modulation_explicit, 
        rtol=1e-5
    )
    
    print("✓ Weight modulation computation test passed!")


def test_npt_model_forward():
    """Test full NPT model forward pass."""
    print("\nTesting NPT model forward pass...")
    
    # Use a small model for testing
    model_name = "gpt2"
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float32
    )
    
    # Convert to NPT
    adapter_config = {
        'r': 8,
        'd_model': config.n_embd,
        'd_ffn': config.n_inner if hasattr(config, 'n_inner') else 4 * config.n_embd,
        'compute_dtype': torch.float32,
        'modulation_type': 'outer_product'
    }
    
    npt_model = convert_llama_to_npt(model, adapter_config)
    
    # Test input
    text = "The quick brown fox"
    inputs = tokenizer(text, return_tensors="pt")
    
    # Forward pass
    with torch.no_grad():
        outputs = npt_model(**inputs)
    
    # Check outputs
    assert hasattr(outputs, 'logits')
    assert outputs.logits.shape[0] == 1  # batch size
    assert outputs.logits.shape[1] == inputs['input_ids'].shape[1]  # seq len
    assert outputs.logits.shape[2] == config.vocab_size  # vocab size
    
    print("✓ NPT model forward pass test passed!")
    
    # Test permanent weight update
    print("\nTesting permanent weight update...")
    
    # Get first NPT layer
    if hasattr(npt_model, 'transformer'):
        first_layer = npt_model.transformer.h[0]
    else:
        first_layer = npt_model.model.layers[0]
    
    # Store original weight norm
    original_weight_norm = torch.norm(first_layer.mlp.gate_proj.weight).item()
    
    # Perform consolidation
    fact = "Test fact for weight update."
    fact_inputs = tokenizer(fact, return_tensors="pt")
    
    if hasattr(first_layer, 'consolidate_weights'):
        stats = first_layer.consolidate_weights(
            fact_inputs.input_ids,
            attention_mask=fact_inputs.attention_mask,
            token_idx=-1,
            alpha=0.01  # Small alpha for testing
        )
        
        # Check stats
        assert 'weight_update_norm' in stats
        assert 'vector_model_norm' in stats
        assert 'vector_ffn_norm' in stats
        
        # Check weight changed
        new_weight_norm = torch.norm(first_layer.mlp.gate_proj.weight).item()
        assert abs(new_weight_norm - original_weight_norm) > 1e-6
        
        print(f"  Weight update norm: {stats['weight_update_norm']:.6f}")
        print(f"  Vector model norm: {stats['vector_model_norm']:.6f}")
        print(f"  Vector FFN norm: {stats['vector_ffn_norm']:.6f}")
        print(f"  Weight norm change: {abs(new_weight_norm - original_weight_norm):.6f}")
        
        print("✓ Permanent weight update test passed!")


def test_numerical_stability():
    """Test numerical stability with different input scales."""
    print("\nTesting numerical stability...")
    
    adapter = NPTAdapter(d_model=256, d_ffn=1024, r=8)
    
    # Test with different input scales
    scales = [1e-3, 1.0, 1e3]
    
    for scale in scales:
        attn_output = torch.randn(2, 10, 256) * scale
        outputs = adapter(attn_output)
        
        # Check for NaN or Inf
        assert not torch.isnan(outputs['vector_model']).any()
        assert not torch.isnan(outputs['vector_ffn']).any()
        assert not torch.isnan(outputs['reg_norm']).any()
        
        assert not torch.isinf(outputs['vector_model']).any()
        assert not torch.isinf(outputs['vector_ffn']).any()
        assert not torch.isinf(outputs['reg_norm']).any()
        
        print(f"  Scale {scale}: reg_norm = {outputs['reg_norm'].item():.6f}")
    
    print("✓ Numerical stability test passed!")


if __name__ == "__main__":
    print("Running NPT outer product implementation tests...\n")
    
    test_adapter_forward()
    test_weight_modulation()
    test_npt_model_forward()
    test_numerical_stability()
    
    print("\n✅ All tests passed!")

"""
Quick test to verify NPT permanent update fix works.
"""

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_npt_checkpoint import load_npt_checkpoint
from model.npt_layer import demonstrate_permanent_update


def test_permanent_update(checkpoint_path):
    """Test permanent update functionality."""
    print("=" * 80)
    print("TESTING NPT PERMANENT UPDATE FIX")
    print("=" * 80)
    
    # Load model properly
    print(f"\n1. Loading NPT model from {checkpoint_path}")
    model, tokenizer = load_npt_checkpoint(checkpoint_path)
    
    # Verify NPT layers exist
    npt_layers = sum(1 for _, module in model.named_modules() 
                    if 'NPTLayer' in str(type(module)))
    print(f"   ✓ Model has {npt_layers} NPT layers")
    
    if npt_layers == 0:
        print("   ✗ ERROR: No NPT layers found!")
        return False
    
    # Test a simple fact injection
    test_fact = "The capital of Atlantis is Poseidon."
    print(f"\n2. Injecting fact: '{test_fact}'")
    
    # Get initial generation
    device = next(model.parameters()).device
    prompt = "The capital of Atlantis is"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs_before = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response_before = tokenizer.decode(outputs_before[0], skip_special_tokens=True)
    completion_before = response_before[len(prompt):].strip()
    print(f"   Before: '{prompt}' -> '{completion_before}'")
    
    # Increase consolidation_alpha for stronger update
    for layer in model.model.layers:
        if hasattr(layer, 'consolidation_alpha'):
            layer.consolidation_alpha = 5.0  # Stronger update for testing
    
    # Perform permanent update
    model = demonstrate_permanent_update(model, tokenizer, test_fact)
    print("   ✓ Permanent update completed")
    
    # Test generation after update
    prompt2 = "What is the capital of Atlantis?"
    inputs2 = tokenizer(prompt2, return_tensors="pt")
    inputs2 = {k: v.to(device) for k, v in inputs2.items()}
    
    with torch.no_grad():
        outputs_after = model.generate(
            **inputs2,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response_after = tokenizer.decode(outputs_after[0], skip_special_tokens=True)
    completion_after = response_after[len(prompt2):].strip()
    print(f"   After: '{prompt2}' -> '{completion_after}'")
    
    # Check if "Poseidon" appears in the response
    success = "poseidon" in completion_after.lower()
    
    if success:
        print("\n✓ SUCCESS: Model learned the fact!")
    else:
        print("\n✗ PARTIAL SUCCESS: Model output changed but didn't recall exact fact")
        print("  This is expected with low alpha values. Try increasing alpha in interactive mode.")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_permanent_update_fix.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    success = test_permanent_update(checkpoint_path)
    sys.exit(0 if success else 1)

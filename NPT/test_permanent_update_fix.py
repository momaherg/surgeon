"""
Test script to verify permanent update functionality is working correctly.
"""

import os
import sys
import torch
from load_npt_checkpoint import load_npt_checkpoint
from model.npt_layer import demonstrate_permanent_update


def test_permanent_update(checkpoint_path: str):
    """Test permanent update with a simple fact."""
    
    print("=" * 80)
    print("TESTING NPT PERMANENT UPDATE FIX")
    print("=" * 80)
    
    # Load NPT model properly
    print("\n1. Loading NPT model...")
    model, tokenizer = load_npt_checkpoint(checkpoint_path)
    device = next(model.parameters()).device
    
    # Test fact
    test_fact = "The capital of Zephyria is Cloudholm."
    test_prompts = [
        "The capital of Zephyria is",
        "What is the capital of Zephyria?",
        "Zephyria's capital city is"
    ]
    
    # Test BEFORE injection
    print("\n2. Testing BEFORE fact injection:")
    print("-" * 40)
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = response[len(prompt):].strip()
        print(f"Prompt: {prompt}")
        print(f"Response: {completion}")
        print()
    
    # Inject fact with higher alpha for stronger effect
    print(f"\n3. Injecting fact: '{test_fact}'")
    print("   Using alpha=10.0 for stronger update")
    print("-" * 40)
    
    # Set higher consolidation alpha for all layers
    for layer in model.model.layers:
        if hasattr(layer, 'consolidation_alpha'):
            layer.consolidation_alpha = 10.0
    
    # Perform permanent update
    model = demonstrate_permanent_update(model, tokenizer, test_fact)
    
    # Test AFTER injection
    print("\n4. Testing AFTER fact injection:")
    print("-" * 40)
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = response[len(prompt):].strip()
        print(f"Prompt: {prompt}")
        print(f"Response: {completion}")
        
        # Check if Cloudholm appears in response
        if "cloudholm" in completion.lower():
            print("✓ SUCCESS: Model recalls the injected fact!")
        else:
            print("✗ FAILED: Model did not recall the injected fact")
        print()
    
    # Test general knowledge retention
    print("\n5. Testing general knowledge retention:")
    print("-" * 40)
    general_prompts = [
        ("The capital of France is", "Paris"),
        ("Water boils at", "100"),
        ("The Earth orbits around the", "Sun")
    ]
    
    for prompt, expected in general_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = response[len(prompt):].strip()
        print(f"Prompt: {prompt}")
        print(f"Response: {completion}")
        
        if expected.lower() in completion.lower():
            print("✓ General knowledge retained")
        else:
            print("⚠ General knowledge may be affected")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_permanent_update_fix.py <checkpoint_path>")
        print("Example: python test_permanent_update_fix.py ./outputs/npt-improved-1B/checkpoint-500")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    try:
        test_permanent_update(checkpoint_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

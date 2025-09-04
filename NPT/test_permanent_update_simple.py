#!/usr/bin/env python3
"""
Simple test script for NPT permanent update functionality.
Tests with different alpha values to find what works.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.npt_layer import demonstrate_permanent_update


def test_permanent_update(checkpoint_path: str):
    """Test permanent update with different configurations."""
    
    print("NPT Permanent Update Test")
    print("=" * 60)
    
    # Load tokenizer (same as test_checkpoint.py)
    print("\nLoading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    except Exception as e:
        print(f"Could not load tokenizer from checkpoint: {e}")
        # Try to get base model name from training info
        training_info_path = os.path.join(checkpoint_path, "training_info.pt")
        if os.path.exists(training_info_path):
            info = torch.load(training_info_path, map_location="cpu", weights_only=False)
            if 'args' in info and hasattr(info['args'], 'model_name'):
                base_model_name = info['args'].model_name
                print(f"Loading tokenizer from base model: {base_model_name}")
                tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            else:
                raise ValueError("Could not find base model name in training info")
        else:
            raise ValueError("No tokenizer found and no training info available")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test fact
    test_fact = "The capital of Atlantis is Poseidon."
    test_prompt = "The capital of Atlantis is"
    expected_answer = "Poseidon"
    
    # Test different alpha values
    alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    print(f"\nTest fact: '{test_fact}'")
    print(f"Test prompt: '{test_prompt}'")
    print(f"Expected: '{expected_answer}'")
    print("\nTesting different alpha values...")
    print("-" * 60)
    
    for alpha in alpha_values:
        print(f"\n\nAlpha = {alpha}")
        print("-" * 30)
        
        # Reload model for clean test
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map="auto",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model.eval()
        device = next(model.parameters()).device
        
        # Check if model has NPT layers (same check as test_checkpoint.py)
        npt_count = 0
        for name, module in model.named_modules():
            if 'NPTLayer' in str(type(module)):
                npt_count += 1
        
        if npt_count == 0:
            print("ERROR: No NPT layers found in model!")
            print("Model structure:")
            for name, module in model.named_modules():
                if 'layers' in name and len(name.split('.')) == 3:
                    print(f"  {name}: {type(module)}")
                    break
            return
        else:
            print(f"Found {npt_count} NPT layers")
        
        # Manually update with specific alpha
        print(f"Injecting fact with alpha={alpha}...")
        
        # Get tokens
        inputs = tokenizer(test_fact, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device) if inputs.attention_mask is not None else None
        
        # Process through layers with manual alpha
        hidden_states = model.model.embed_tokens(input_ids)
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        position_embeddings = None
        if hasattr(model.model, 'rotary_emb'):
            cos, sin = model.model.rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos, sin)
        
        total_update_norm = 0.0
        updated_layers = 0
        
        # Update each layer
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, 'consolidate_weights'):
                stats = layer.consolidate_weights(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    token_idx=-1,
                    alpha=alpha  # Use our test alpha
                )
                
                if 'message' not in stats:  # Not quantized
                    total_update_norm += stats['weight_update_norm']
                    updated_layers += 1
                
                # Forward through layer
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    output_attentions=False
                )
                
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs
        
        print(f"Updated {updated_layers} layers")
        print(f"Total update norm: {total_update_norm:.6f}")
        
        # Test generation
        test_inputs = tokenizer(test_prompt, return_tensors="pt")
        test_inputs = {k: v.to(device) for k, v in test_inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **test_inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = response[len(test_prompt):].strip()
        
        # Check if answer is correct
        success = expected_answer.lower() in completion.lower()
        
        print(f"Response: '{completion}'")
        print(f"Result: {'✓ SUCCESS' if success else '✗ FAILED'}")
        
        if success:
            print(f"\n🎉 Found working alpha value: {alpha}")
            print(f"The model successfully learned and recalled the fact!")
            
            # Test a few more times to ensure it's consistent
            print("\nVerifying consistency...")
            for i in range(3):
                with torch.no_grad():
                    outputs = model.generate(
                        **test_inputs,
                        max_new_tokens=10,
                        temperature=0.1,
                        do_sample=True
                    )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                completion = response[len(test_prompt):].strip()
                print(f"  Test {i+1}: '{completion}'")
            
            return alpha
        
        # Clean up model
        del model
        torch.cuda.empty_cache()
    
    print("\n\nNo working alpha value found in the tested range.")
    print("Recommendations:")
    print("1. Check if your model is quantized (use diagnose_permanent_update.py)")
    print("2. Try the enhanced_permanent_update.py script for advanced strategies")
    print("3. Consider retraining with higher modulation_scale values")
    
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_permanent_update_simple.py <checkpoint_path>")
        print("Example: python test_permanent_update_simple.py ./outputs/npt-improved-1B/checkpoint-500")
        sys.exit(1)
    
    working_alpha = test_permanent_update(sys.argv[1])
    
    if working_alpha is not None:
        print(f"\n\nSummary: Use alpha={working_alpha} or higher for permanent updates with this model.")
    else:
        print("\n\nSummary: Permanent updates not working with standard approach. See recommendations above.")

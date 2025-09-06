#!/usr/bin/env python3
"""
Debug script to understand the exact structure of teacher hidden states.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def analyze_hidden_states_structure(model_name="meta-llama/Llama-3.1-8B"):
    """Analyze what exactly is in the hidden states from the teacher model."""
    
    print(f"Analyzing hidden states structure for {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()
    
    # Test input
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        # Get outputs with hidden states
        outputs = model(**inputs, output_hidden_states=True)
        
        print(f"\nModel has {len(model.model.layers)} transformer layers")
        print(f"Hidden states returned: {len(outputs.hidden_states)} tensors")
        
        # Check each hidden state
        for i, hidden in enumerate(outputs.hidden_states):
            print(f"\nHidden state {i}:")
            print(f"  Shape: {hidden.shape}")
            print(f"  Mean: {hidden.mean().item():.6f}")
            print(f"  Std: {hidden.std().item():.6f}")
        
        # Now let's manually trace through the model to understand
        print("\n" + "="*60)
        print("Manual forward pass to understand structure:")
        print("="*60)
        
        # Get embeddings
        embeddings = model.model.embed_tokens(inputs.input_ids)
        print(f"\nEmbeddings shape: {embeddings.shape}")
        print(f"Embeddings mean: {embeddings.mean().item():.6f}")
        
        # Check if this matches hidden_states[0]
        if torch.allclose(embeddings, outputs.hidden_states[0], atol=1e-5):
            print("✓ hidden_states[0] = embeddings")
        else:
            print("✗ hidden_states[0] != embeddings")
        
        # Pass through layers manually
        hidden = embeddings
        for i, layer in enumerate(model.model.layers):
            # Layer norm before attention
            normed = layer.input_layernorm(hidden)
            
            # Self attention
            attn_out = layer.self_attn(normed)[0]
            hidden = hidden + attn_out
            
            # Layer norm before FFN
            normed = layer.post_attention_layernorm(hidden)
            
            # FFN
            ffn_out = layer.mlp(normed)
            hidden = hidden + ffn_out
            
            # Check if this matches hidden_states[i+1]
            if i < len(outputs.hidden_states) - 1:
                if torch.allclose(hidden, outputs.hidden_states[i+1], atol=1e-4):
                    print(f"✓ hidden_states[{i+1}] = output of layer {i}")
                else:
                    diff = torch.mean(torch.abs(hidden - outputs.hidden_states[i+1])).item()
                    print(f"✗ hidden_states[{i+1}] != output of layer {i} (diff: {diff:.6f})")
        
        # Apply final layer norm
        final_hidden = model.model.norm(hidden)
        print(f"\nFinal norm output shape: {final_hidden.shape}")
        print(f"Final norm output mean: {final_hidden.mean().item():.6f}")
        
        # Check if final norm is in hidden_states
        final_in_hidden_states = False
        for i, h in enumerate(outputs.hidden_states):
            if torch.allclose(final_hidden, h, atol=1e-4):
                print(f"✓ Final layer norm found at hidden_states[{i}]")
                final_in_hidden_states = True
                break
        
        if not final_in_hidden_states:
            print("✗ Final layer norm NOT in hidden_states!")
            
            # Check the last hidden state
            last_hidden = outputs.hidden_states[-1]
            diff = torch.mean(torch.abs(final_hidden - last_hidden)).item()
            print(f"  Difference between final norm and last hidden state: {diff:.6f}")
        
        # Also check logits computation
        logits = model.lm_head(final_hidden)
        if torch.allclose(logits, outputs.logits, atol=1e-4):
            print("\n✓ Logits computed from final layer norm")
        else:
            print("\n✗ Logits computation mismatch")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    args = parser.parse_args()
    
    analyze_hidden_states_structure(args.model)

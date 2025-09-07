#!/usr/bin/env python3
"""
Quick verification that the tuple handling fix works.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model import convert_llama_to_npt

print("Testing NPT layer tuple handling fix...")

# Load a small model for testing
model_name = "meta-llama/Llama-3.1-8B"
print(f"\nLoading model: {model_name}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load config
config = AutoConfig.from_pretrained(model_name)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    device_map="auto",
    torch_dtype=torch.float32
)

# Convert to NPT
print("Converting to NPT...")
adapter_config = {
    'r': 16,
    'd_model': config.hidden_size,
    'd_ffn': config.intermediate_size,
    'modulation_scale': 0.1,
    'init_strategy': 'adaptive',
    'init_scale': 0.5
}
npt_model = convert_llama_to_npt(model, adapter_config)

# Test forward pass in training mode
print("\nTesting forward pass in training mode...")
npt_model.train()

# Create dummy input
text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt", padding=True)
input_ids = inputs.input_ids.cuda() if torch.cuda.is_available() else inputs.input_ids
attention_mask = inputs.attention_mask.cuda() if torch.cuda.is_available() else inputs.attention_mask

try:
    # Test direct model forward (this would fail with tuple issue)
    print("Testing standard forward pass...")
    with torch.no_grad():
        outputs = npt_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
    print("✓ Standard forward pass works!")
    
except Exception as e:
    print(f"✗ Standard forward pass failed: {e}")
    print("\nTesting manual layer processing...")
    
    # Test manual processing
    try:
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Get embeddings
        hidden_states = npt_model.model.embed_tokens(input_ids)
        
        # Create position_ids
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get position embeddings if available
        position_embeddings = None
        if hasattr(npt_model.model, 'rotary_emb'):
            cos, sin = npt_model.model.rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos, sin)
        
        # Process through layers
        for i, layer in enumerate(npt_model.model.layers):
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings
            )
            
            # Handle tuple outputs
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
                print(f"  Layer {i}: Returned tuple with {len(layer_outputs)} elements")
            else:
                hidden_states = layer_outputs
                print(f"  Layer {i}: Returned tensor directly")
        
        print("✓ Manual layer processing works!")
        
    except Exception as e2:
        print(f"✗ Manual processing also failed: {e2}")

print("\nTest complete!")

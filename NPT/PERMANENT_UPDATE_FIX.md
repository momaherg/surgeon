# Permanent Update Fix - Hidden States vs Token IDs

## The Problem

The original `demonstrate_permanent_update` function had a critical bug where it was passing token IDs to every layer's `consolidate_weights` method:

```python
# INCORRECT - Original implementation
for i, layer in enumerate(model.model.layers):
    if hasattr(layer, 'consolidate_weights'):
        stats = layer.consolidate_weights(
            inputs.input_ids,  # ❌ This is wrong! Passing token IDs to all layers
            attention_mask=inputs.attention_mask,
            token_idx=-1
        )
```

The issue is that:
1. NPT layers expect **hidden states** as input, not token IDs
2. Each layer needs the output hidden states from the previous layer
3. Only the embedding layer converts token IDs to hidden states

## The Solution

The corrected implementation properly:
1. Converts token IDs to embeddings first
2. Passes hidden states through each layer sequentially
3. Uses each layer's output as input to the next layer

```python
# CORRECT - Fixed implementation
# Start with embeddings
hidden_states = model.model.embed_tokens(input_ids)

# Process through each layer
for i, layer in enumerate(model.model.layers):
    if hasattr(layer, 'consolidate_weights'):
        # Use current hidden states for consolidation
        stats = layer.consolidate_weights(
            hidden_states,  # ✅ Correct! Pass hidden states
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            token_idx=-1
        )
        
        # Get next hidden states by forward pass
        layer_outputs = layer(hidden_states, ...)
        hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
```

## What Changed

### 1. Updated `consolidate_weights` method signature:
- Changed parameter name from `context_tokens` to `hidden_states` for clarity
- Updated docstring to reflect that it expects hidden states, not token IDs

### 2. Rewrote `demonstrate_permanent_update`:
- Added embedding conversion step
- Added sequential processing through layers
- Added proper position ID and position embedding handling
- Each layer now receives the correct hidden states

### 3. Created test script:
- `test_permanent_update.py` demonstrates the corrected functionality
- Includes layer-by-layer verification to ensure hidden states are properly propagated

## Why This Matters

Without this fix:
- The permanent update would fail or produce incorrect results
- Only the first layer might work (if it happened to handle token IDs)
- Middle and later layers would receive incompatible input
- The weight consolidation would not reflect the actual information flow through the model

## Usage Example

```python
from model.npt_layer import convert_llama_to_npt, demonstrate_permanent_update

# Convert model to NPT
model = convert_llama_to_npt(model, adapter_config)

# Inject a fact with proper hidden state flow
fact = "The capital of Atlantis is Poseidon."
model = demonstrate_permanent_update(model, tokenizer, fact)
```

## Testing

Run the test script to verify the fix:
```bash
python test_permanent_update.py
```

This will:
1. Test layer-by-layer hidden state processing
2. Demonstrate permanent weight update with a sample fact
3. Verify that hidden states are correctly propagated

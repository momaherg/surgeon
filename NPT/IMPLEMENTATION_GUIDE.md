# NPT Implementation Guide: From V1 to V2

## Overview

This guide explains the key differences between the original implementation (V1) and the proposal-aligned implementation (V2), along with usage instructions.

## Key Architectural Changes

### 1. Weight Modulation vs Activation Modulation

**V1 (Original):**
```python
# Modulates activations after weight application
intermediate = F.silu(gate_output + delta_effect) * up_output
```

**V2 (Aligned):**
```python
# Modulates the effect of weights on activations
modulated_gate = gate_output * (1 + delta_mult) + delta_add
intermediate = F.silu(modulated_gate) * up_output
```

**Why This Matters:**
- V2 better approximates true weight modulation while remaining efficient
- Supports both multiplicative and additive modulation
- More expressive than simple activation addition

### 2. Residual Connection Structure

**V1 (Standard Transformer):**
```python
h_residual = residual + attn_output  # First residual
hidden_states = h_residual + mlp_output  # Second residual
```

**V2 (NPT Proposal):**
```python
# No residual after attention
# Single residual after MLP
hidden_states = original_input + mlp_output
```

**Why This Matters:**
- Forces the model to route information through the modulated MLP
- Aligns with the proposal's architecture
- May require different learning dynamics

### 3. Permanent Update Mode

**V1:** Not implemented

**V2:** Full implementation of permanent weight consolidation
```python
# Inject a fact permanently
stats = layer.consolidate_weights(
    context_tokens,
    token_idx=-1,  # Use last token's modulation
    alpha=0.1      # Update strength
)
```

## Usage Examples

### Example 1: Converting a Model to NPT V2

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.npt_layer_v2 import convert_llama_to_npt_v2

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Convert to NPT V2
adapter_config = {
    'r': 16,                          # Low-rank dimension
    'modulation_type': 'both',        # Use both additive and multiplicative
    'consolidation_alpha': 0.1        # Permanent update strength
}
npt_model = convert_llama_to_npt_v2(model, adapter_config)
```

### Example 2: Dynamic Mode (Enhanced In-Context Learning)

```python
# Standard inference with enhanced context handling
prompt = """
Context: The Zephyr API uses authentication token 'ZPH-2024-SECRET'.
Question: What is the authentication token for the Zephyr API?
Answer:"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = npt_model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0])

# The NPT model should better utilize the context due to attention-based modulation
```

### Example 3: Permanent Update Mode (Fact Injection)

```python
from model.npt_layer_v2 import demonstrate_permanent_update

# Inject a new fact
fact = "The CEO of TechCorp is Dr. Sarah Chen."
updated_model = demonstrate_permanent_update(npt_model, tokenizer, fact)

# Later, without the original context:
prompt = "Who is the CEO of TechCorp?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = updated_model.generate(**inputs, max_length=50)
# Should recall: "The CEO of TechCorp is Dr. Sarah Chen."
```

### Example 4: Selective Layer Updates

```python
# Update only specific layers for targeted knowledge injection
fact_tokens = tokenizer(fact, return_tensors="pt").input_ids

for i in [20, 21, 22]:  # Update only layers 20-22
    layer = npt_model.model.layers[i]
    if hasattr(layer, 'consolidate_weights'):
        layer.consolidate_weights(
            fact_tokens,
            token_idx=-1,
            alpha=0.05  # Smaller alpha for selective updates
        )
```

## Training Considerations

### Phase 1: Equivalence Pre-training with V2

The training script needs modification to handle the new architecture:

```python
# In compute_equivalence_loss function
def compute_equivalence_loss_v2(self, batch):
    # Teacher uses standard dual residuals
    teacher_output = self.teacher_model(...)
    
    # Student uses single residual (NPT V2)
    student_output = self.student_model(...)
    
    # Compare final hidden states, not intermediate
    mse_loss = F.mse_loss(student_output.hidden_states, teacher_output.hidden_states)
    
    # Add modulation regularization
    reg_losses = [output[1] for output in student_outputs if len(output) > 1]
    reg_loss = torch.stack(reg_losses).mean() if reg_losses else 0
    
    return mse_loss + self.args.regularization_lambda * reg_loss
```

### Phase 2: Task-Specific Fine-tuning

```python
# Configure for different modulation strategies
task_configs = {
    'factual_recall': {
        'modulation_type': 'additive',  # Better for discrete facts
        'consolidation_alpha': 0.2
    },
    'reasoning': {
        'modulation_type': 'both',      # More expressive
        'consolidation_alpha': 0.05     # Smaller updates
    },
    'creative': {
        'modulation_type': 'multiplicative',  # More dynamic
        'consolidation_alpha': 0.01          # Minimal permanent updates
    }
}
```

## Performance Considerations

### Memory Usage
- V2 uses ~2x adapter parameters (for both add and mult)
- Permanent updates don't increase memory (modify existing weights)

### Computation
- V2 is slightly more expensive (~10% overhead)
- Permanent updates are one-time costs

### Trade-offs
- V1: Simpler, slightly faster, good baseline
- V2: More aligned with proposal, better expressiveness, supports permanent updates

## Migration Path

1. **Test V2 on Small Scale:**
   ```bash
   python test_npt_layer_v2.py --model_size small --compare_versions
   ```

2. **Gradual Migration:**
   - Start with a few layers using V2
   - Compare performance metrics
   - Expand if beneficial

3. **Permanent Update Testing:**
   - Test on synthetic facts first
   - Measure forgetting on standard benchmarks
   - Tune consolidation_alpha based on results

## Recommendations

1. **For Research:** Use V2 to properly test the NPT hypothesis
2. **For Production:** Start with V1, migrate to V2 after validation
3. **For Fact Injection:** V2 is required (V1 doesn't support it)

## Next Steps

1. Update training scripts to support V2 architecture
2. Implement evaluation metrics for permanent updates
3. Create benchmarks for catastrophic forgetting
4. Optimize permanent update mechanism for efficiency

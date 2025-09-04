# Fixing NPT Permanent Update Issues

## Problem Description
The permanent update feature is not working - when you inject a fact, the model doesn't retain or recall it. This is a common issue with several potential causes.

## Quick Diagnosis

Run the diagnostic script first:
```bash
python diagnose_permanent_update.py ./outputs/npt-improved-1B/checkpoint-500
```

This will tell you:
- If your model has quantized layers (which can't be updated)
- If the modulation vectors are being generated properly
- If weight changes are actually happening
- The current alpha and modulation scale values

## Common Issues and Solutions

### 1. **Alpha Value Too Small (Most Common)**
The default `consolidation_alpha` is 0.1, which is often too small to create noticeable changes.

**Solution**: Use a higher alpha value
```python
# In interactive_permanent_update.py, when injecting:
tester.inject_fact("The capital of Atlantis is Poseidon.", alpha=2.0)  # or even 5.0
```

### 2. **Quantized Model Layers**
If your model was trained with quantization, the weights are stored in a compressed format and cannot be directly modified.

**Solution**: Train without quantization
```bash
python pretrain_npt.py \
    --model_name "meta-llama/Llama-3.2-1B" \
    --use_fp16 \
    --max_steps 1000 \
    # Do NOT use --use_quantization
```

### 3. **Weak Modulation Vectors**
The adapter might be generating very small modulation vectors due to initialization or training.

**Check with debug script**:
```bash
python debug_permanent_update.py ./outputs/npt-improved-1B/checkpoint-500 "Your fact here"
```

**Solution**: Use the enhanced update strategies
```bash
python enhanced_permanent_update.py ./outputs/npt-improved-1B/checkpoint-500 "The capital of Atlantis is Poseidon."
```

### 4. **Insufficient Modulation Scale**
The `modulation_scale` parameter might be too small during training.

**Solution**: Train with higher modulation scale
```python
# In pretrain_npt.py or finetune_npt.py
adapter_config = {
    'r': 16,
    'modulation_scale': 0.5,  # Increase from default 0.1
    'consolidation_alpha': 0.5
}
```

## Enhanced Update Strategies

The `enhanced_permanent_update.py` script provides several strategies:

1. **High Alpha Update**: Uses alpha values of 2.0-5.0
2. **Iterative Reinforcement**: Applies updates multiple times with decreasing strength
3. **Attention-Guided Update**: Weights updates based on attention patterns
4. **Amplified Adapter**: Temporarily amplifies modulation during update
5. **Multi-MLP Update**: Updates both gate and up projections

## Step-by-Step Fix

1. **First, diagnose the issue**:
   ```bash
   python diagnose_permanent_update.py <your_checkpoint>
   ```

2. **If quantized, retrain without quantization**:
   ```bash
   python pretrain_npt.py --model_name "meta-llama/Llama-3.2-1B" --use_fp16 --max_steps 1000
   ```

3. **Try manual update with high alpha**:
   ```python
   # In Python
   from interactive_permanent_update import InteractivePermanentUpdateTester
   
   tester = InteractivePermanentUpdateTester("./outputs/npt-improved-1B/checkpoint-500")
   tester.inject_fact("The capital of Atlantis is Poseidon.", alpha=5.0)
   tester.test_recall("The capital of Atlantis is", "Poseidon")
   ```

4. **If still not working, use enhanced strategies**:
   ```bash
   python enhanced_permanent_update.py <checkpoint> "Your fact here"
   ```

## Testing Permanent Updates

After injecting a fact, test with multiple prompts:
```python
# Direct completion
"The capital of Atlantis is"

# Question format
"What is the capital of Atlantis?"

# Conversational
"Tell me about Atlantis. What is its capital?"
```

## Best Practices

1. **Use simple, clear facts**: "X is Y" format works best
2. **Start with high alpha**: Begin with 2.0-5.0, then reduce if needed
3. **Test immediately**: Verify the update worked before injecting more facts
4. **Monitor weight changes**: Use the debug script to ensure weights are actually changing
5. **Avoid quantized models**: Permanent updates require full precision weights

## Example Working Configuration

```python
# Training config that works well
python pretrain_npt.py \
    --model_name "meta-llama/Llama-3.2-1B" \
    --use_fp16 \
    --adapter_rank 32 \
    --modulation_scale 0.3 \
    --learning_rate 1e-3 \
    --max_steps 1000

# Injection that works
tester = InteractivePermanentUpdateTester(checkpoint)
tester.inject_fact("The CEO of TechCorp is Jane Smith.", alpha=3.0)
```

## Troubleshooting Checklist

- [ ] Model is NOT quantized (check with diagnostic script)
- [ ] Using alpha >= 1.0 for updates
- [ ] Modulation vectors have non-zero norms
- [ ] Weight changes are detected after update
- [ ] Testing with simple "X is Y" facts
- [ ] Using fresh model instance for each test

If all else fails, the `enhanced_permanent_update.py` script will try multiple strategies automatically and report which one works best for your model.

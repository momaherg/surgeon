# Migration Guide: quick_test_npt.py → quick_test_npt_fixed.py

## Quick Start

### Old Command
```bash
python quick_test_npt.py <checkpoint_path> <base_model_name>
```

### New Command
```bash
python quick_test_npt_fixed.py <checkpoint_path> [--interactive]
```

## Key Differences

1. **No need to specify base_model_name** - The fixed version automatically detects it from the checkpoint
2. **Added --interactive flag** - For interactive testing mode
3. **Better error messages** - Clear feedback about what went wrong
4. **Faster loading** - No redundant model loading operations

## What Changed?

### Before (Incorrect)
```python
# Loads base model, converts to NPT, then loads weights
model = load_base_model(base_model_name)
model = convert_to_npt(model)
model.load_state_dict(checkpoint_weights)
```

### After (Correct)
```python
# Loads complete NPT model directly
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
```

## Common Issues When Migrating

### Issue 1: "No tokenizer found"
**Solution**: The checkpoint should contain tokenizer files. If not, the script will try to load from the base model name in training_info.pt

### Issue 2: "No NPT layers found"
**Solution**: This means the checkpoint might not be a proper NPT model. Check if you're using the right checkpoint path.

### Issue 3: Different generation results
**Solution**: The old script might have been loading weights incorrectly. The new results should be more accurate.

## Testing Your Migration

1. **Run both versions and compare**:
   ```bash
   # Old version
   python quick_test_npt.py ./checkpoint-500 meta-llama/Llama-3.2-1B > old_output.txt
   
   # New version
   python quick_test_npt_fixed.py ./checkpoint-500 > new_output.txt
   
   # Compare
   diff old_output.txt new_output.txt
   ```

2. **Verify model structure**:
   The new script will print:
   ```
   Verified: Model has 32 NPT layers
   ```

3. **Check training config**:
   The new script displays training configuration:
   ```
   Training configuration:
     Adapter rank: 16
     Modulation scale: 0.1
     Learning rate: 0.0001
   ```

## Benefits of Migrating

1. **Correctness**: Actually loads the model as it was saved
2. **Performance**: Faster loading (no redundant operations)
3. **Reliability**: Better error handling and validation
4. **Simplicity**: Fewer command-line arguments needed
5. **Debugging**: More informative output

## For Script Integration

If you're using the old script in other code:

### Old Integration
```python
from quick_test_npt import load_npt_checkpoint

model, tokenizer = load_npt_checkpoint(
    checkpoint_path="./checkpoint-500",
    base_model_name="meta-llama/Llama-3.2-1B"
)
```

### New Integration
```python
from quick_test_npt_fixed import load_npt_checkpoint

# Note: no base_model_name needed!
model, tokenizer = load_npt_checkpoint(
    checkpoint_path="./checkpoint-500"
)
```

## Troubleshooting

### If the new script fails but old one "worked":
1. The old script might have been silently failing (strict=False hides errors)
2. Check if your checkpoint is complete (has model files)
3. Ensure you're using the correct checkpoint path

### To keep using the old behavior (NOT recommended):
If you absolutely need the old behavior for some reason, you can:
1. Keep using `quick_test_npt.py` (but understand it's incorrect)
2. Or modify your checkpoint saving to only save adapter weights (requires code changes)

## Recommendation

**Always use `quick_test_npt_fixed.py`** - it correctly loads NPT models as they are actually saved. The old approach was based on incorrect assumptions about checkpoint contents.

# Testing NPT Checkpoints - Fixed Version

## Quick Usage

### Test NPT Model Only
```bash
python test_checkpoint.py --checkpoint_path ./outputs/npt-improved-1B/checkpoint-500
```

### Compare NPT with Base Llama Model
```bash
python test_checkpoint.py \
    --checkpoint_path ./outputs/npt-improved-1B/checkpoint-500 \
    --base_model meta-llama/Llama-3.2-1B \
    --with_comparison
```

## What Was Fixed?

The original `test_checkpoint.py` had the same issue as `quick_test_npt.py`:
- ❌ **OLD**: Load base model → Convert to NPT → Load checkpoint weights
- ✅ **NEW**: Load NPT model directly from checkpoint

## Key Changes

1. **Simplified Loading**: The NPT model is loaded directly from checkpoint since it contains the full model
2. **Better Error Handling**: Validates checkpoint exists and has model files
3. **Tokenizer Fallback**: If tokenizer isn't in checkpoint, loads from base model name in training_info.pt
4. **NPT Verification**: Checks that the loaded model actually has NPT layers

## Arguments

- `--checkpoint_path`: Path to NPT checkpoint (required)
- `--base_model`: Base model name for comparison (default: meta-llama/Llama-3.1-8B)
- `--prompt`: Test prompt (default: "The capital of France is")
- `--max_new_tokens`: Max tokens to generate (default: 50)
- `--with_comparison`: Enable comparison with base model

## Example Output

```
Loading NPT from ./outputs/npt-improved-1B/checkpoint-500
Training configuration:
  Adapter rank: 16
  Modulation scale: 0.1
  Learning rate: 0.0001
Loading NPT model with dtype=torch.float16...
Verified: Model has 32 NPT layers

Prompt: The capital of France is

NPT Response: Paris, which is located in the north-central part of the country...

Loading base model for comparison...

Base Model Response: Paris. France is a country located in Western Europe...
```

## Benefits of the Fix

1. **Correct Loading**: Actually loads the NPT model as it was saved
2. **Faster**: No redundant model loading and conversion
3. **Reliable**: Proper error handling and validation
4. **Simpler**: Cleaner code with single loading path

## Testing Different Checkpoints

```bash
# Test checkpoint at step 500
python test_checkpoint.py --checkpoint_path ./outputs/npt-improved-1B/checkpoint-500 --with_comparison

# Test final checkpoint
python test_checkpoint.py --checkpoint_path ./outputs/npt-improved-1B/checkpoint-final --with_comparison

# Test best checkpoint
python test_checkpoint.py --checkpoint_path ./outputs/npt-improved-1B/checkpoint-best --with_comparison
```

## Custom Prompts

```bash
python test_checkpoint.py \
    --checkpoint_path ./outputs/npt-improved-1B/checkpoint-500 \
    --prompt "Artificial intelligence will" \
    --max_new_tokens 100 \
    --with_comparison
```

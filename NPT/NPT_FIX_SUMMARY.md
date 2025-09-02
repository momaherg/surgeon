# NPT Training NaN Fix Summary

## Issues Identified

1. **Critical Issue**: Multiplicative modulation using `(1 + tanh(x))` could produce zeros
   - When `tanh(x) = -1`, the factor becomes `1 + (-1) = 0`
   - This zeros out activations and causes gradient vanishing/explosion

2. **Mixed Precision Conflicts**: Using both `--use_quantization` and `--use_fp16` together
   - Quantized models should use FP32 for numerical stability
   - Mixed precision can conflict with quantization

3. **Initialization Issues**: Adapter weights initialized with too large values
   - Could cause immediate instability in training

4. **Loss Computation**: MSE between teacher and student can explode
   - Large initial differences between models can cause huge gradients

## Fixes Applied

### 1. Updated NPT Layer (`model/npt_layer.py`)

- **Changed multiplicative modulation** from `tanh` to sigmoid-based:
  ```python
  # Old: delta_mult = torch.tanh(delta_mult_raw)  # Range: [-1, 1]
  # New: delta_mult = 0.5 * torch.sigmoid(delta_mult_raw) + 0.5  # Range: [0.5, 1.5]
  ```
  This ensures the modulation factor never goes to zero.

- **Improved initialization**:
  ```python
  # A_proj: Smaller initialization (std=0.02 instead of kaiming)
  # B_add: Much smaller initialization (std=0.001 instead of 0.01)
  # B_mult: Kept at zeros for no initial effect
  ```

- **Scaled additive modulation**: Added 0.1 scaling factor to reduce initial impact

### 2. Created Safe Training Script (`pretrain_npt_safe.py`)

- **Automatic dtype handling**: Forces FP32 when using quantization
- **Normalized MSE loss**: Prevents extreme values from dominating
- **Aggressive gradient clipping**: Max norm of 0.5
- **Error recovery**: Skips NaN batches instead of crashing
- **Safe mode option**: Use only additive modulation for extra stability

### 3. Testing Scripts

- `debug_nan.py`: Identifies where NaNs occur in the forward pass
- `test_fixed_npt.py`: Verifies the fixes work correctly

## Recommended Training Command

For stable training with quantization:

```bash
python pretrain_npt_safe.py \
    --model_name meta-llama/Llama-3.1-8B \
    --adapter_rank 16 \
    --use_quantization \
    --share_embeddings \
    --safe_mode \
    --streaming \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --regularization_lambda 0.01 \
    --num_epochs 1 \
    --warmup_steps 200 \
    --output_dir ./outputs/npt-safe-pretrained \
    --log_steps 10 \
    --save_steps 500 \
    --use_wandb
```

Key changes:
- Use `pretrain_npt_safe.py` instead of `pretrain_npt.py`
- Removed `--use_fp16` (conflicts with quantization)
- Added `--safe_mode` for extra stability
- Reduced learning rate to `5e-5`
- Increased warmup steps to 200

## Additional Recommendations

1. **Start with safe mode**: Use `--safe_mode` to disable multiplicative modulation initially
2. **Monitor gradients**: Watch the gradient norms in logs
3. **Use smaller batches**: Start with batch_size=1 and increase gradually
4. **Test without quantization first**: If issues persist, try without `--use_quantization`
5. **Use the test script**: Run `python test_fixed_npt.py` to verify the model works

## Alternative: Minimal Changes to Original

If you prefer to keep using the original `pretrain_npt.py`, make these minimal changes:

1. Don't use both `--use_quantization` and `--use_fp16` together
2. Reduce learning rate to `5e-5` or lower
3. Increase warmup steps
4. Consider using `--max_steps 1000` for initial testing

The safe version is recommended as it has better error handling and recovery mechanisms.

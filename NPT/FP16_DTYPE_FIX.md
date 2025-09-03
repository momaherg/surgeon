# FP16 Dtype Mismatch Fix

## The Error

```
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::Half != float
```

This error occurred when using `--use_fp16` because:
- The model's attention outputs were in FP16 (Half precision)
- The adapter weights were initialized in FP32 (float)
- PyTorch linear layers require matching dtypes between inputs and weights

## Root Cause

The NPT adapter was being initialized with FP32 weights regardless of the model's dtype. When the FP16 attention output was passed to the adapter's linear projection, it failed due to the dtype mismatch.

## Fixes Applied

### 1. Dynamic Dtype Conversion in Adapter Forward (`model/npt_layer.py`)

Added automatic dtype conversion in the adapter's forward method:

```python
# Ensure input matches adapter dtype to avoid mixed precision issues
if attn_output.dtype != self.A_proj.weight.dtype:
    attn_output = attn_output.to(self.A_proj.weight.dtype)
```

Also added dtype matching when applying modulation to ensure compatibility:

```python
# Ensure modulation matches gate_output dtype
if delta_mult.dtype != gate_output.dtype:
    delta_mult = delta_mult.to(gate_output.dtype)
```

### 2. Smarter Adapter Initialization

Updated adapter initialization to:
- Use FP32 for quantized models (for stability)
- Match model dtype for non-quantized models
- Properly infer dtype from model parameters

### 3. Updated Training Scripts

Both `pretrain_npt.py` and `pretrain_npt_safe.py` now properly set adapter dtype:

```python
# Determine adapter dtype based on model configuration
if self.args.use_quantization:
    adapter_dtype = torch.float32  # Always FP32 for quantized
elif self.args.use_fp16:
    adapter_dtype = torch.float16  # Match FP16
else:
    adapter_dtype = torch.float32  # Default FP32
```

## Usage Recommendations

### For FP16 Training (without quantization):
```bash
python pretrain_npt_safe.py \
    --model_name meta-llama/Llama-3.1-8B \
    --adapter_rank 16 \
    --use_fp16 \
    --mixed_precision fp16 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --output_dir ./outputs/npt-fp16
```

### For Quantized Models:
```bash
python pretrain_npt_safe.py \
    --model_name meta-llama/Llama-3.1-8B \
    --adapter_rank 16 \
    --use_quantization \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --output_dir ./outputs/npt-quantized
```

### For Maximum Compatibility (Recommended):
```bash
python pretrain_npt_safe.py \
    --model_name meta-llama/Llama-3.1-8B \
    --adapter_rank 16 \
    --use_quantization \
    --safe_mode \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --output_dir ./outputs/npt-safe
```

## Key Points

1. **Don't mix quantization with FP16** - Use one or the other
2. **Adapters now match model dtype** - Better performance and compatibility
3. **Automatic dtype conversion** - Handles edge cases gracefully
4. **Safe mode recommended** - Use `--safe_mode` for initial testing

## Testing the Fix

Run the test script to verify:
```bash
python test_fp16_fix.py
```

This will test:
- FP16 model with FP16 adapters
- FP32 model with FP32 adapters
- Mixed dtype handling
- Modulation in different precisions

The dtype mismatch error should now be resolved!

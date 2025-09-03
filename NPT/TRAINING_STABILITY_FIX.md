# NPT Training Stability Fix

## Problem Description
During NPT pre-training, the regularization loss was decreasing properly while the MSE loss was oscillating wildly from the start, indicating issues with the loss computation.

## Root Causes Identified

1. **Over-aggressive normalization in MSE computation**
   - The original code normalized hidden states before computing MSE
   - This removed magnitude information and only compared directions
   - Made the loss insensitive to actual hidden state values
   - Caused unstable gradients and oscillating loss

2. **Hard-coded small modulation scaling factor**
   - The NPT layer used a fixed 0.1 scaling factor for weight modulation
   - This was too restrictive and prevented the model from properly matching teacher outputs

## Fixes Applied

### 1. Direct MSE Computation
**File**: `pretrain_npt_safe.py`

Changed from:
```python
# Normalize to prevent extreme values
teacher_norm = teacher_hidden.norm(dim=-1, keepdim=True).clamp(min=1e-6)
student_norm = student_hidden.norm(dim=-1, keepdim=True).clamp(min=1e-6)
teacher_normalized = teacher_hidden / teacher_norm
student_normalized = student_hidden / student_norm
layer_mse = nn.functional.mse_loss(student_normalized, teacher_normalized)
```

To:
```python
# Direct MSE computation without normalization
layer_mse = nn.functional.mse_loss(student_hidden, teacher_hidden)

# Optional layer-wise scaling for balanced training
if self.args.use_layer_wise_loss_scaling:
    teacher_scale = teacher_hidden.abs().mean().clamp(min=1e-6)
    layer_mse = layer_mse / teacher_scale
```

### 2. Configurable Modulation Scaling
**File**: `model/npt_layer.py`

- Added `modulation_scale` parameter to NPTLayer
- Made it configurable via command-line argument
- Default remains 0.1 but can be adjusted based on training stability

### 3. Additional Improvements
- Added `--use_layer_wise_loss_scaling` option for models with varying layer scales
- Adjusted default safe learning rate to 1e-4
- Created test script to verify training stability

## Usage Recommendations

### For Stable Training:
```bash
python pretrain_npt_safe.py \
  --model_name meta-llama/Llama-3.1-8B \
  --learning_rate 5e-5 \
  --modulation_scale 0.1 \
  --regularization_lambda 0.01 \
  --use_layer_wise_loss_scaling \
  --warmup_steps 200 \
  --gradient_accumulation_steps 8
```

### Tuning Guidelines:
1. **If MSE loss oscillates**: Reduce learning rate or modulation_scale
2. **If MSE loss plateaus**: Increase modulation_scale (0.2-0.5)
3. **If regularization dominates**: Reduce regularization_lambda
4. **For models with varying layer scales**: Enable --use_layer_wise_loss_scaling

## Verification
Run `python test_training_stability.py` to:
- Compare normalized vs direct MSE behavior
- Test different modulation scales
- Generate training recommendations

## Expected Results
With these fixes:
- MSE loss should decrease smoothly without wild oscillations
- Both MSE and regularization losses should decrease together
- Training should be stable from the start
- The model should successfully learn to match teacher outputs

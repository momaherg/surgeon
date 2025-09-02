# NPT Troubleshooting Guide

## Common Issues and Solutions

### 1. NaN Loss During Training

**Symptoms:**
- Loss becomes `nan` immediately or after a few steps
- Training crashes with numerical errors

**Causes:**
- Multiplicative modulation producing zeros
- Mixed precision conflicts with quantization
- Learning rate too high
- Poor weight initialization

**Solutions:**

1. **Use the safe training script:**
   ```bash
   python pretrain_npt_safe.py  # Instead of pretrain_npt.py
   ```

2. **Enable safe mode (disables multiplicative modulation):**
   ```bash
   --safe_mode
   ```

3. **Don't mix quantization with FP16:**
   ```bash
   # Good: Use quantization alone
   --use_quantization
   
   # Bad: Don't use both together
   --use_quantization --use_fp16
   ```

4. **Reduce learning rate:**
   ```bash
   --learning_rate 5e-5  # Instead of 1e-4
   ```

5. **Increase warmup steps:**
   ```bash
   --warmup_steps 200  # Instead of 100
   ```

### 2. Out of Memory Errors

**Solutions:**

1. **Use quantization:**
   ```bash
   --use_quantization
   ```

2. **Share embeddings between teacher and student:**
   ```bash
   --share_embeddings
   ```

3. **Reduce batch size and increase gradient accumulation:**
   ```bash
   --batch_size 1 --gradient_accumulation_steps 16
   ```

4. **Use streaming for large datasets:**
   ```bash
   --streaming
   ```

### 3. Slow Training

**Solutions:**

1. **Use mixed precision (only without quantization):**
   ```bash
   --mixed_precision fp16
   ```

2. **Reduce maximum sequence length:**
   ```bash
   --max_length 1024  # Instead of 2048
   ```

3. **Use fewer adapter parameters:**
   ```bash
   --adapter_rank 8  # Instead of 16
   ```

### 4. Poor Convergence

**Solutions:**

1. **Adjust regularization:**
   ```bash
   --regularization_lambda 0.001  # Try different values
   ```

2. **Use different optimizer:**
   ```bash
   --optimizer_type adam  # Instead of adamw
   ```

3. **Try different scheduler:**
   ```bash
   --scheduler_type linear  # Instead of cosine
   ```

## Testing and Debugging

### Quick Test Script

Run this to verify your setup works:

```bash
python test_fixed_npt.py
```

### Debug NaN Issues

If you still get NaN losses, run the debug script:

```bash
python debug_nan.py
```

This will show you exactly where NaNs appear in the forward pass.

### Monitor Training

Watch these metrics during training:
- `grad_norm`: Should be < 10, ideally < 1
- `loss/mse`: Should decrease over time
- `loss/regularization`: Should stay small

## Recommended Safe Configuration

For most stable training:

```bash
python pretrain_npt_safe.py \
    --model_name meta-llama/Llama-3.1-8B \
    --adapter_rank 8 \
    --use_quantization \
    --share_embeddings \
    --safe_mode \
    --streaming \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --regularization_lambda 0.001 \
    --warmup_steps 500 \
    --max_steps 5000 \
    --output_dir ./outputs/npt-safe \
    --log_steps 10 \
    --save_steps 500
```

## Getting Help

If issues persist:

1. Check the logs in your output directory
2. Look at the gradient norms in the metrics
3. Try running with `--safe_mode` enabled
4. Start with a smaller model for testing
5. Reduce all hyperparameters (learning rate, batch size, etc.)

Remember: It's better to train slowly and successfully than to have fast training that produces NaN!

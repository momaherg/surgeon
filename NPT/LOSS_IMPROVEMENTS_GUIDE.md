# NPT Loss Function Improvements for Better Convergence

This guide explains the improvements made to the NPT loss function to achieve better convergence during training.

## Overview of Improvements

The improved loss function addresses several key issues that can hinder convergence:

1. **Imbalanced gradient contributions across layers**
2. **Lack of directional alignment in addition to magnitude alignment**
3. **Absence of curriculum learning for gradual difficulty increase**
4. **Non-smooth optimization landscape**
5. **Fixed temperature for knowledge distillation**

## Key Components

### 1. Adaptive Layer-wise Weighting

**Problem**: Different layers in the model have different scales, leading to imbalanced gradient contributions.

**Solution**: Learnable weights for each layer that adaptively balance their contributions.

```python
class AdaptiveLayerWeights(nn.Module):
    def __init__(self, num_layers: int):
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        self.temperature = nn.Parameter(torch.tensor(1.0))
```

**Benefits**:
- Automatically balances gradient flow across layers
- Prevents early layers from being overshadowed by later layers
- Learnable temperature parameter for fine-tuning weight distribution

### 2. Combined MSE + Cosine Similarity Loss

**Problem**: MSE alone only ensures magnitude alignment, not directional alignment.

**Solution**: Combine MSE with cosine similarity loss.

```python
total_loss = mse_weight * mse_loss + cosine_weight * (1 - cosine_similarity)
```

**Benefits**:
- MSE ensures magnitude matching (80% weight by default)
- Cosine similarity ensures directional alignment (20% weight by default)
- Better convergence for high-dimensional representations

### 3. Curriculum Learning with Progressive Scheduling

**Problem**: Training starts with full difficulty, which can cause instability.

**Solution**: Gradually increase training difficulty using a cosine warmup schedule.

```python
curriculum_factor = 0.5 * (1 + cos(π * (1 - progress)))
```

**Features**:
- Starts with easier objectives (50% difficulty)
- Gradually increases to full difficulty over warmup period
- Applies to regularization weights and temperature scaling

### 4. Gradient Penalty for Smooth Optimization

**Problem**: Non-smooth loss landscape can cause optimization difficulties.

**Solution**: Add gradient penalty term inspired by WGAN-GP.

```python
# Interpolate between student and teacher
interpolated = α * student + (1 - α) * teacher
# Penalize large gradients w.r.t interpolated states
grad_penalty = mean(||∇loss(interpolated)||²)
```

**Benefits**:
- Encourages Lipschitz continuity
- Prevents gradient explosion
- Smoother optimization landscape

### 5. Temperature Scaling for Knowledge Distillation

**Problem**: Direct MSE between hidden states can be too strict.

**Solution**: Apply temperature scaling that decreases with curriculum learning.

```python
teacher_scaled = teacher_hidden / temperature
student_scaled = student_hidden / temperature
```

**Benefits**:
- Softer targets during early training (temperature = 3.0)
- Gradually transitions to harder targets (temperature → 1.0)
- Better knowledge transfer from teacher to student

## Alternative Loss Functions

### 1. Focal MSE Loss

For handling datasets with outliers or difficult samples:

```python
focal_weight = mse.detach() ** (gamma / 2)
focal_loss = focal_weight * mse
```

- Focuses on hard examples (large errors)
- Reduces influence of easy examples
- Useful for heterogeneous datasets

### 2. Smooth L1 + MSE Loss

For robustness to outliers:

```python
loss = mse_weight * MSE + (1 - mse_weight) * SmoothL1
```

- MSE for small errors (quadratic)
- Smooth L1 for large errors (linear)
- More stable with noisy data

## Usage Examples

### Basic Usage with Improved Loss

```bash
python pretrain_npt_improved.py \
    --loss_type "improved_npt" \
    --use_adaptive_weights \
    --use_gradient_penalty \
    --mse_weight 0.8 \
    --cosine_weight 0.2
```

### For Noisy/Outlier-prone Data

```bash
python pretrain_npt_improved.py \
    --loss_type "focal_mse" \
    --focal_gamma 2.0
```

### For More Robust Training

```bash
python pretrain_npt_improved.py \
    --loss_type "smooth_l1_mse" \
    --smooth_beta 1.0 \
    --mse_weight 0.5
```

## Hyperparameter Recommendations

### For Standard Training
- `mse_weight`: 0.8
- `cosine_weight`: 0.2
- `regularization_lambda`: 0.01
- `gradient_penalty_lambda`: 0.001
- `distill_temperature`: 3.0
- `warmup_steps`: 10% of total steps

### For Difficult Convergence
- Increase `warmup_steps` to 20% of total steps
- Increase `distill_temperature` to 5.0
- Reduce `learning_rate` by 50%
- Enable both `use_adaptive_weights` and `use_gradient_penalty`

### For Fast Convergence (if stable)
- Reduce `warmup_steps` to 5% of total steps
- Reduce `distill_temperature` to 2.0
- Increase `learning_rate` by 50%
- Disable `use_gradient_penalty` to save computation

## Monitoring Convergence

The improved loss function logs additional metrics:

1. **`loss/alignment`**: Combined alignment loss (main objective)
2. **`loss/mse`**: MSE component only
3. **`loss/cosine`**: Cosine similarity component
4. **`loss/grad_penalty`**: Gradient penalty value
5. **`training/curriculum_factor`**: Current curriculum learning factor
6. **`training/temperature`**: Current distillation temperature

### Healthy Training Signs
- Smooth decrease in alignment loss
- MSE and cosine losses both decreasing
- Gradient penalty staying relatively constant
- Curriculum factor smoothly increasing from 0.5 to 1.0

### Warning Signs
- Oscillating losses (reduce learning rate)
- Gradient penalty increasing rapidly (reduce gradient_penalty_lambda)
- Cosine loss not decreasing (increase cosine_weight)
- MSE exploding (check for NaN, reduce learning rate)

## Comparison with Original Loss

| Feature | Original Loss | Improved Loss |
|---------|--------------|---------------|
| Layer balancing | Fixed averaging | Adaptive weights |
| Alignment type | Magnitude only (MSE) | Magnitude + Direction |
| Warmup strategy | LR warmup only | Curriculum learning |
| Optimization smoothness | No guarantees | Gradient penalty |
| Knowledge transfer | Direct matching | Temperature scaling |
| Outlier handling | Sensitive | Optional focal/smooth variants |

## Expected Improvements

Based on the enhancements, you should expect:

1. **Faster initial convergence** due to curriculum learning
2. **More stable training** due to gradient penalty and adaptive weights  
3. **Better final performance** due to combined MSE + cosine alignment
4. **Reduced overfitting** due to temperature scaling
5. **Robustness to hyperparameters** due to adaptive mechanisms

## Troubleshooting

### Loss Not Decreasing
1. Check if curriculum factor is too aggressive (increase warmup_steps)
2. Verify adaptive weights are not collapsing (check layer_weights values)
3. Try reducing gradient penalty lambda

### NaN Losses
1. Reduce learning rate
2. Increase gradient clipping (--max_grad_norm 0.5)
3. Disable gradient penalty temporarily
4. Check for extreme values in teacher outputs

### Slow Convergence
1. Reduce warmup period
2. Increase learning rate after warmup
3. Reduce temperature scaling
4. Consider using focal loss for hard examples

## Testing the Improvements

Run the test script to see the improvements in action:

```bash
python test_improved_loss.py
```

This will demonstrate:
- Adaptive layer weighting behavior
- Combined loss components
- Curriculum learning progression
- Comparison of different loss variants

## Conclusion

The improved loss functions provide multiple mechanisms to achieve better convergence:

1. **Adaptive mechanisms** that self-adjust during training
2. **Multi-objective optimization** for comprehensive alignment
3. **Smooth optimization landscape** for stable gradients
4. **Curriculum learning** for progressive difficulty
5. **Flexible variants** for different data characteristics

These improvements work together to create a more robust and efficient training process for NPT models.

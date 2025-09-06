"""
Combined NPT loss function that incorporates the best aspects of both improved_loss.py 
and improved_loss_v2.py, specifically designed for NPT's unique architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import math


class AdaptiveArchitectureAwareLoss(nn.Module):
    """
    Combines the best of both loss functions:
    1. Architecture-aware layer weighting with divergence tolerance
    2. Adaptive learnable weights on top of fixed decay
    3. Combined MSE + Cosine similarity
    4. Curriculum learning with warmup
    5. Optional gradient penalty for stability
    
    This is specifically designed for NPT where:
    - Early layers should match closely (small architectural difference)
    - Later layers are expected to diverge (attention-only modulation effect)
    - Focus shifts from exact matching to functional equivalence
    """
    
    def __init__(
        self,
        num_layers: int,
        # Architecture-aware parameters
        layer_decay: float = 0.85,
        divergence_tolerance: float = 0.15,
        use_exponential_decay: bool = True,
        normalize_by_layer: bool = True,
        # Loss combination parameters
        mse_weight: float = 0.7,
        cosine_weight: float = 0.3,
        # Regularization
        regularization_lambda: float = 0.01,
        gradient_penalty_lambda: float = 0.001,
        use_gradient_penalty: bool = False,
        # Curriculum learning
        temperature: float = 3.0,
        warmup_steps: int = 1000,
        # Adaptive weights
        use_adaptive_refinement: bool = True,
        adaptive_init_value: float = 1.0
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.layer_decay = layer_decay
        self.divergence_tolerance = divergence_tolerance
        self.normalize_by_layer = normalize_by_layer
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.regularization_lambda = regularization_lambda
        self.gradient_penalty_lambda = gradient_penalty_lambda
        self.use_gradient_penalty = use_gradient_penalty
        self.temperature = temperature
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        # Compute base layer weights with architectural awareness
        if use_exponential_decay:
            # Exponential decay: early layers get much higher weight
            base_weights = torch.tensor([
                layer_decay ** i for i in range(num_layers)
            ])
        else:
            # Linear decay
            base_weights = torch.tensor([
                1.0 - (i / num_layers) * (1.0 - layer_decay)
                for i in range(num_layers)
            ])
        
        # Normalize base weights
        self.register_buffer('base_weights', base_weights / base_weights.sum())
        
        # Optional adaptive refinement on top of base weights
        if use_adaptive_refinement:
            # Learnable multipliers for each layer (initialized to 1)
            self.adaptive_multipliers = nn.Parameter(
                torch.ones(num_layers) * adaptive_init_value
            )
            self.adaptive_temperature = nn.Parameter(torch.tensor(1.0))
        else:
            self.adaptive_multipliers = None
    
    def get_effective_weights(self) -> torch.Tensor:
        """Get the effective layer weights combining base and adaptive."""
        if self.adaptive_multipliers is not None:
            # Apply softmax to ensure weights are positive and normalized
            multipliers = F.softmax(
                self.adaptive_multipliers / self.adaptive_temperature, 
                dim=0
            )
            # Combine with base weights
            effective_weights = self.base_weights * multipliers
            # Renormalize
            return effective_weights / effective_weights.sum()
        else:
            return self.base_weights
    
    def compute_divergence_adjusted_loss(
        self,
        mse_loss: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """Apply divergence tolerance adjustment to MSE loss."""
        # Calculate expected divergence for this layer
        expected_divergence = self.divergence_tolerance * (layer_idx / self.num_layers) ** 2
        
        # Soft clipping: reduce loss if within expected divergence
        # Use smooth transition instead of hard threshold
        divergence_factor = torch.sigmoid(
            (mse_loss - expected_divergence) * 10
        )
        
        # Scale down loss when within tolerance, keep full loss when exceeding
        adjusted_loss = mse_loss * (0.1 + 0.9 * divergence_factor)
        
        return adjusted_loss
    
    def compute_combined_loss(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        layer_idx: int,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Compute combined MSE + Cosine loss with architectural adjustments."""
        
        # Apply temperature scaling for curriculum learning
        if temperature > 1.0:
            student_hidden = student_hidden / temperature
            teacher_hidden = teacher_hidden / temperature
        
        # Normalize by layer statistics if requested
        if self.normalize_by_layer:
            # This helps when scales differ significantly between layers
            student_mean = student_hidden.mean(dim=-1, keepdim=True)
            student_std = student_hidden.std(dim=-1, keepdim=True) + 1e-6
            student_norm = (student_hidden - student_mean) / student_std
            
            teacher_mean = teacher_hidden.mean(dim=-1, keepdim=True)
            teacher_std = teacher_hidden.std(dim=-1, keepdim=True) + 1e-6
            teacher_norm = (teacher_hidden - teacher_mean) / teacher_std
        else:
            student_norm = student_hidden
            teacher_norm = teacher_hidden
        
        # MSE loss (with divergence adjustment)
        mse_loss = F.mse_loss(student_norm, teacher_norm)
        adjusted_mse = self.compute_divergence_adjusted_loss(mse_loss, layer_idx)
        
        # Cosine similarity loss (naturally scale-invariant)
        student_flat = student_hidden.view(-1, student_hidden.shape[-1])
        teacher_flat = teacher_hidden.view(-1, teacher_hidden.shape[-1])
        
        cosine_sim = F.cosine_similarity(student_flat, teacher_flat, dim=-1)
        cosine_loss = 1.0 - cosine_sim.mean()
        
        # For later layers, emphasize cosine over MSE
        layer_progress = layer_idx / self.num_layers
        dynamic_mse_weight = self.mse_weight * (1 - 0.3 * layer_progress)
        dynamic_cosine_weight = self.cosine_weight * (1 + 0.3 * layer_progress)
        
        # Normalize weights
        total_weight = dynamic_mse_weight + dynamic_cosine_weight
        dynamic_mse_weight /= total_weight
        dynamic_cosine_weight /= total_weight
        
        combined_loss = dynamic_mse_weight * adjusted_mse + dynamic_cosine_weight * cosine_loss
        
        return {
            'combined': combined_loss,
            'mse': mse_loss,  # Original MSE for logging
            'adjusted_mse': adjusted_mse,
            'cosine': cosine_loss
        }
    
    def compute_gradient_penalty(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor
    ) -> torch.Tensor:
        """Gradient penalty for smooth optimization landscape."""
        # Sample random interpolation coefficient
        alpha = torch.rand(1, device=student_hidden.device)
        
        # Interpolate between student and teacher
        interpolated = alpha * student_hidden + (1 - alpha) * teacher_hidden.detach()
        interpolated.requires_grad_(True)
        
        # Compute loss w.r.t interpolated states
        loss = F.mse_loss(interpolated, teacher_hidden.detach())
        
        # Compute gradients
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=interpolated,
            create_graph=True,
            only_inputs=True
        )[0]
        
        # L2 penalty on gradients
        grad_penalty = (grads.pow(2).sum(dim=-1).mean() - 1).pow(2)
        
        return grad_penalty
    
    def get_curriculum_factor(self) -> float:
        """Curriculum learning schedule."""
        if self.current_step >= self.warmup_steps:
            return 1.0
        
        # Cosine warmup
        progress = self.current_step / self.warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (1 - progress)))
    
    def forward(
        self,
        teacher_hidden_states: List[torch.Tensor],
        student_hidden_states: List[torch.Tensor],
        reg_norms: List[torch.Tensor],
        step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the combined architecture-aware loss.
        
        Returns comprehensive loss dictionary for logging and debugging.
        """
        if step is not None:
            self.current_step = step
        
        # Get curriculum learning factors
        curriculum_factor = self.get_curriculum_factor()
        current_temperature = 1.0 + (self.temperature - 1.0) * curriculum_factor
        
        # Get effective layer weights
        layer_weights = self.get_effective_weights()
        
        # Compute per-layer losses
        layer_losses = []
        mse_losses = []
        adjusted_mse_losses = []
        cosine_losses = []
        grad_penalties = []
        
        # Skip embedding layer (index 0)
        num_layers = min(len(teacher_hidden_states) - 1, len(student_hidden_states) - 1)
        
        for i in range(1, num_layers + 1):
            layer_idx = i - 1  # 0-indexed for weights
            
            teacher_hidden = teacher_hidden_states[i].detach()
            student_hidden = student_hidden_states[i]
            
            # Compute combined loss for this layer
            loss_dict = self.compute_combined_loss(
                student_hidden=student_hidden,
                teacher_hidden=teacher_hidden,
                layer_idx=layer_idx,
                temperature=current_temperature
            )
            
            # Apply layer weight
            weight = layer_weights[layer_idx] if layer_idx < len(layer_weights) else layer_weights[-1]
            weighted_loss = loss_dict['combined'] * weight
            
            layer_losses.append(weighted_loss)
            mse_losses.append(loss_dict['mse'])
            adjusted_mse_losses.append(loss_dict['adjusted_mse'])
            cosine_losses.append(loss_dict['cosine'])
            
            # Compute gradient penalty for selected layers
            if self.use_gradient_penalty and layer_idx % 4 == 0:
                grad_penalty = self.compute_gradient_penalty(student_hidden, teacher_hidden)
                grad_penalties.append(grad_penalty)
        
        # Aggregate losses
        alignment_loss = torch.stack(layer_losses).sum()  # Weighted sum
        avg_mse = torch.stack(mse_losses).mean()
        avg_adjusted_mse = torch.stack(adjusted_mse_losses).mean()
        avg_cosine = torch.stack(cosine_losses).mean()
        
        # Regularization loss
        if reg_norms:
            reg_loss = torch.stack(reg_norms).mean()
        else:
            reg_loss = torch.tensor(0.0, device=alignment_loss.device)
        
        # Gradient penalty
        if grad_penalties:
            avg_grad_penalty = torch.stack(grad_penalties).mean()
        else:
            avg_grad_penalty = torch.tensor(0.0, device=alignment_loss.device)
        
        # Apply curriculum-adjusted weights to regularization terms
        reg_weight = self.regularization_lambda * curriculum_factor
        grad_penalty_weight = self.gradient_penalty_lambda * curriculum_factor
        
        # Total loss
        total_loss = (
            alignment_loss + 
            reg_weight * reg_loss + 
            grad_penalty_weight * avg_grad_penalty
        )
        
        # Increment step counter
        self.current_step += 1
        
        return {
            'total_loss': total_loss,
            'alignment_loss': alignment_loss,
            'mse_loss': avg_mse,
            'adjusted_mse_loss': avg_adjusted_mse,
            'cosine_loss': avg_cosine,
            'reg_loss': reg_loss,
            'grad_penalty': avg_grad_penalty,
            'curriculum_factor': curriculum_factor,
            'temperature': current_temperature,
            # Additional metrics for monitoring
            'layer_weights_min': layer_weights.min().item(),
            'layer_weights_max': layer_weights.max().item(),
            'layer_weights_std': layer_weights.std().item()
        }


def create_combined_npt_loss(model_config, training_config) -> nn.Module:
    """
    Factory function to create the combined NPT loss.
    
    This loss function is specifically designed for NPT's architecture where:
    - Attention outputs modulate weights rather than being added to residuals
    - Later layers naturally diverge from standard transformers
    - Training should focus on functional equivalence
    """
    num_layers = getattr(model_config, 'num_hidden_layers', 32)
    
    return AdaptiveArchitectureAwareLoss(
        num_layers=num_layers,
        # Architecture-aware settings
        layer_decay=training_config.get('layer_decay', 0.85),
        divergence_tolerance=training_config.get('divergence_tolerance', 0.15),
        use_exponential_decay=training_config.get('use_exponential_decay', True),
        normalize_by_layer=training_config.get('normalize_by_layer', True),
        # Loss weights
        mse_weight=training_config.get('mse_weight', 0.7),
        cosine_weight=training_config.get('cosine_weight', 0.3),
        # Regularization
        regularization_lambda=training_config.get('regularization_lambda', 0.01),
        gradient_penalty_lambda=training_config.get('gradient_penalty_lambda', 0.001),
        use_gradient_penalty=training_config.get('use_gradient_penalty', False),
        # Curriculum learning
        temperature=training_config.get('distill_temperature', 3.0),
        warmup_steps=training_config.get('warmup_steps', 1000),
        # Adaptive refinement
        use_adaptive_refinement=training_config.get('use_adaptive_refinement', True),
        adaptive_init_value=training_config.get('adaptive_init_value', 1.0)
    )

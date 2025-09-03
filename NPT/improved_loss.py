"""
Improved loss functions for NPT training with better convergence properties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class AdaptiveLayerWeights(nn.Module):
    """
    Learns adaptive weights for each layer's contribution to the loss.
    Helps balance gradients across layers with different scales.
    """
    
    def __init__(self, num_layers: int, init_value: float = 1.0):
        super().__init__()
        # Learnable layer weights initialized to 1.0
        self.layer_weights = nn.Parameter(torch.ones(num_layers) * init_value)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, layer_losses: List[torch.Tensor]) -> torch.Tensor:
        """Apply adaptive weighting to layer losses."""
        # Ensure weights are positive and normalized
        weights = F.softmax(self.layer_weights / self.temperature, dim=0)
        
        # Weight each layer's loss
        weighted_losses = []
        for i, loss in enumerate(layer_losses):
            weighted_losses.append(weights[i] * loss)
        
        return torch.stack(weighted_losses).sum()


class CombinedMSECosineLoss(nn.Module):
    """
    Combines MSE loss with cosine similarity for better convergence.
    MSE ensures magnitude alignment while cosine ensures directional alignment.
    """
    
    def __init__(self, mse_weight: float = 0.8, cosine_weight: float = 0.2):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
    
    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        # MSE loss for magnitude
        mse_loss = F.mse_loss(student, teacher)
        
        # Cosine similarity loss for direction
        # Flatten to (batch * seq_len, hidden_dim)
        student_flat = student.view(-1, student.size(-1))
        teacher_flat = teacher.view(-1, teacher.size(-1))
        
        # Compute cosine similarity (1 - similarity for loss)
        cosine_sim = F.cosine_similarity(student_flat, teacher_flat, dim=-1)
        cosine_loss = (1 - cosine_sim).mean()
        
        # Combine losses
        total_loss = self.mse_weight * mse_loss + self.cosine_weight * cosine_loss
        
        return total_loss, mse_loss, cosine_loss


class ImprovedNPTLoss(nn.Module):
    """
    Improved loss function for NPT training with multiple enhancements:
    1. Adaptive layer-wise weighting
    2. Combined MSE + Cosine similarity
    3. Gradient penalty for smooth optimization
    4. Temperature scaling for knowledge distillation
    5. Curriculum learning support
    """
    
    def __init__(
        self,
        num_layers: int,
        mse_weight: float = 0.8,
        cosine_weight: float = 0.2,
        regularization_lambda: float = 0.01,
        gradient_penalty_lambda: float = 0.001,
        temperature: float = 3.0,
        use_adaptive_weights: bool = True,
        use_gradient_penalty: bool = True,
        warmup_steps: int = 1000
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.regularization_lambda = regularization_lambda
        self.gradient_penalty_lambda = gradient_penalty_lambda
        self.temperature = temperature
        self.warmup_steps = warmup_steps
        self.use_gradient_penalty = use_gradient_penalty
        
        # Adaptive layer weights
        if use_adaptive_weights:
            self.layer_weights = AdaptiveLayerWeights(num_layers)
        else:
            self.layer_weights = None
        
        # Combined loss function
        self.layer_loss_fn = CombinedMSECosineLoss(mse_weight, cosine_weight)
        
        # Track step for curriculum learning
        self.current_step = 0
    
    def compute_gradient_penalty(
        self, 
        student_hidden: torch.Tensor, 
        teacher_hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient penalty to encourage smooth optimization landscape.
        This helps prevent gradient explosion and improves convergence.
        """
        # Interpolate between student and teacher
        alpha = torch.rand(1, device=student_hidden.device)
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
        
        # Compute gradient penalty (L2 norm)
        grad_penalty = torch.mean(grads.pow(2))
        
        return grad_penalty
    
    def get_curriculum_factor(self) -> float:
        """
        Get curriculum learning factor based on current training step.
        Starts with easier objectives and gradually increases difficulty.
        """
        if self.current_step >= self.warmup_steps:
            return 1.0
        
        # Cosine warmup schedule
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
        Compute improved loss with all enhancements.
        
        Returns dict with:
        - total_loss: Combined loss for backprop
        - mse_loss: MSE component
        - cosine_loss: Cosine similarity component
        - reg_loss: Regularization loss
        - grad_penalty: Gradient penalty (if used)
        - layer_losses: Individual layer losses
        """
        if step is not None:
            self.current_step = step
        
        # Temperature scaling for knowledge distillation
        curriculum_factor = self.get_curriculum_factor()
        current_temperature = 1.0 + (self.temperature - 1.0) * curriculum_factor
        
        # Compute per-layer losses
        layer_losses = []
        mse_losses = []
        cosine_losses = []
        grad_penalties = []
        
        num_layers = min(len(teacher_hidden_states) - 1, len(student_hidden_states) - 1)
        
        for i in range(1, num_layers + 1):
            teacher_hidden = teacher_hidden_states[i].detach()
            student_hidden = student_hidden_states[i]
            
            # Apply temperature scaling if > 1
            if current_temperature > 1.0:
                teacher_hidden = teacher_hidden / current_temperature
                student_hidden = student_hidden / current_temperature
            
            # Compute combined loss
            layer_loss, mse, cosine = self.layer_loss_fn(student_hidden, teacher_hidden)
            
            layer_losses.append(layer_loss)
            mse_losses.append(mse)
            cosine_losses.append(cosine)
            
            # Compute gradient penalty if enabled
            if self.use_gradient_penalty and i % 4 == 0:  # Only compute for every 4th layer to save computation
                grad_penalty = self.compute_gradient_penalty(student_hidden, teacher_hidden)
                grad_penalties.append(grad_penalty)
        
        # Apply adaptive layer weighting
        if self.layer_weights is not None:
            alignment_loss = self.layer_weights(layer_losses)
        else:
            alignment_loss = torch.stack(layer_losses).mean()
        
        # Average component losses
        avg_mse = torch.stack(mse_losses).mean()
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
        
        # Apply curriculum learning to regularization
        reg_weight = self.regularization_lambda * curriculum_factor
        grad_penalty_weight = self.gradient_penalty_lambda * curriculum_factor
        
        # Total loss
        total_loss = (
            alignment_loss + 
            reg_weight * reg_loss + 
            grad_penalty_weight * avg_grad_penalty
        )
        
        # Increment step
        self.current_step += 1
        
        return {
            'total_loss': total_loss,
            'alignment_loss': alignment_loss,
            'mse_loss': avg_mse,
            'cosine_loss': avg_cosine,
            'reg_loss': reg_loss,
            'grad_penalty': avg_grad_penalty,
            'layer_losses': layer_losses,
            'curriculum_factor': curriculum_factor,
            'temperature': current_temperature
        }


class FocalMSELoss(nn.Module):
    """
    Focal loss variant for MSE that focuses on hard examples.
    Helps with convergence when there are outliers or difficult samples.
    """
    
    def __init__(self, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute element-wise MSE
        mse = (pred - target) ** 2
        
        # Compute focal weights (higher weight for larger errors)
        focal_weight = mse.detach() ** (self.gamma / 2)
        
        # Apply focal weighting
        focal_mse = focal_weight * mse
        
        # Reduce
        if self.reduction == 'mean':
            return focal_mse.mean()
        elif self.reduction == 'sum':
            return focal_mse.sum()
        else:
            return focal_mse


class SmoothL1MSELoss(nn.Module):
    """
    Combination of Smooth L1 and MSE loss.
    More robust to outliers than pure MSE while maintaining MSE properties for small errors.
    """
    
    def __init__(self, beta: float = 1.0, mse_weight: float = 0.5):
        super().__init__()
        self.beta = beta
        self.mse_weight = mse_weight
        self.smooth_l1_weight = 1.0 - mse_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # MSE component
        mse_loss = F.mse_loss(pred, target)
        
        # Smooth L1 component
        smooth_l1_loss = F.smooth_l1_loss(pred, target, beta=self.beta)
        
        # Combine
        return self.mse_weight * mse_loss + self.smooth_l1_weight * smooth_l1_loss


def create_improved_loss(
    model_config,
    training_config,
    loss_type: str = "improved_npt"
) -> nn.Module:
    """
    Factory function to create appropriate loss function.
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        loss_type: Type of loss to use
    
    Returns:
        Loss module
    """
    num_layers = model_config.num_hidden_layers
    
    if loss_type == "improved_npt":
        return ImprovedNPTLoss(
            num_layers=num_layers,
            mse_weight=training_config.get("mse_weight", 0.8),
            cosine_weight=training_config.get("cosine_weight", 0.2),
            regularization_lambda=training_config.get("regularization_lambda", 0.01),
            gradient_penalty_lambda=training_config.get("gradient_penalty_lambda", 0.001),
            temperature=training_config.get("distill_temperature", 3.0),
            use_adaptive_weights=training_config.get("use_adaptive_weights", True),
            use_gradient_penalty=training_config.get("use_gradient_penalty", True),
            warmup_steps=training_config.get("warmup_steps", 1000)
        )
    elif loss_type == "focal_mse":
        return FocalMSELoss(gamma=training_config.get("focal_gamma", 2.0))
    elif loss_type == "smooth_l1_mse":
        return SmoothL1MSELoss(
            beta=training_config.get("smooth_beta", 1.0),
            mse_weight=training_config.get("mse_weight", 0.5)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

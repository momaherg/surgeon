"""
Improved loss functions for NPT training that account for architectural differences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import math


class ArchitectureAwareLoss(nn.Module):
    """
    Loss function that accounts for the expected divergence in NPT architecture.
    
    Key features:
    1. Layer-wise decay: Earlier layers weighted more heavily
    2. Divergence tolerance: Allows for expected architectural differences
    3. Focus on functional equivalence over exact matching
    """
    
    def __init__(
        self,
        num_layers: int,
        layer_decay: float = 0.9,
        divergence_tolerance: float = 0.1,
        mse_weight: float = 0.7,
        cosine_weight: float = 0.3,
        regularization_lambda: float = 0.01,
        normalize_by_layer: bool = True,
        use_exponential_decay: bool = True
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.layer_decay = layer_decay
        self.divergence_tolerance = divergence_tolerance
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.regularization_lambda = regularization_lambda
        self.normalize_by_layer = normalize_by_layer
        self.use_exponential_decay = use_exponential_decay
        
        # Compute layer weights based on decay strategy
        if use_exponential_decay:
            # Exponential decay: early layers get much higher weight
            self.layer_weights = torch.tensor([
                layer_decay ** i for i in range(num_layers)
            ])
        else:
            # Linear decay
            self.layer_weights = torch.tensor([
                1.0 - (i / num_layers) * (1.0 - layer_decay)
                for i in range(num_layers)
            ])
        
        # Normalize weights
        self.layer_weights = self.layer_weights / self.layer_weights.sum()
        
    def compute_layer_loss(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        layer_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for a single layer with architectural awareness."""
        
        # Normalize by layer norm if requested (helps with scale differences)
        if self.normalize_by_layer:
            # Normalize to unit variance
            student_std = student_hidden.std(dim=-1, keepdim=True) + 1e-6
            teacher_std = teacher_hidden.std(dim=-1, keepdim=True) + 1e-6
            
            student_norm = student_hidden / student_std
            teacher_norm = teacher_hidden / teacher_std
        else:
            student_norm = student_hidden
            teacher_norm = teacher_hidden
        
        # MSE loss (scale-invariant if normalized)
        mse_loss = F.mse_loss(student_norm, teacher_norm)
        
        # Cosine similarity loss (already scale-invariant)
        student_flat = student_hidden.view(-1, student_hidden.shape[-1])
        teacher_flat = teacher_hidden.view(-1, teacher_hidden.shape[-1])
        
        cosine_sim = F.cosine_similarity(student_flat, teacher_flat, dim=-1)
        cosine_loss = 1.0 - cosine_sim.mean()
        
        # Apply divergence tolerance for later layers
        # This allows the model to diverge more in later layers
        expected_divergence = self.divergence_tolerance * (layer_idx / self.num_layers)
        
        # Soft clipping: reduce loss if it's within expected divergence
        if mse_loss < expected_divergence:
            mse_loss = mse_loss * 0.1  # Greatly reduce if within tolerance
        
        return {
            'mse': mse_loss,
            'cosine': cosine_loss,
            'combined': self.mse_weight * mse_loss + self.cosine_weight * cosine_loss
        }
    
    def forward(
        self,
        teacher_hidden_states: List[torch.Tensor],
        student_hidden_states: List[torch.Tensor],
        reg_norms: List[torch.Tensor],
        step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute architecture-aware loss.
        
        Returns:
            Dictionary with loss components
        """
        layer_losses = []
        mse_losses = []
        cosine_losses = []
        
        # Skip embedding layer (index 0)
        num_layers = min(len(teacher_hidden_states) - 1, len(student_hidden_states) - 1)
        
        for i in range(1, num_layers + 1):
            teacher_hidden = teacher_hidden_states[i].detach()
            student_hidden = student_hidden_states[i]
            
            # Compute layer loss
            layer_loss_dict = self.compute_layer_loss(
                student_hidden=student_hidden,
                teacher_hidden=teacher_hidden,
                layer_idx=i-1  # 0-indexed for layer weights
            )
            
            # Apply layer weight
            weight = self.layer_weights[i-1] if i-1 < len(self.layer_weights) else self.layer_weights[-1]
            weighted_loss = layer_loss_dict['combined'] * weight
            
            layer_losses.append(weighted_loss)
            mse_losses.append(layer_loss_dict['mse'])
            cosine_losses.append(layer_loss_dict['cosine'])
        
        # Aggregate losses
        alignment_loss = torch.stack(layer_losses).sum()  # Weighted sum
        avg_mse = torch.stack(mse_losses).mean()
        avg_cosine = torch.stack(cosine_losses).mean()
        
        # Regularization
        if reg_norms:
            reg_loss = torch.stack(reg_norms).mean() * self.regularization_lambda
        else:
            reg_loss = torch.tensor(0.0, device=alignment_loss.device)
        
        # Total loss
        total_loss = alignment_loss + reg_loss
        
        return {
            'total_loss': total_loss,
            'alignment_loss': alignment_loss,
            'mse_loss': avg_mse,
            'cosine_loss': avg_cosine,
            'reg_loss': reg_loss,
            'layer_weights_sum': self.layer_weights.sum().item()  # For debugging
        }


class FunctionalEquivalenceLoss(nn.Module):
    """
    Focus on functional equivalence rather than hidden state matching.
    
    This loss emphasizes:
    1. Output logits similarity
    2. Attention pattern similarity
    3. Key feature preservation
    """
    
    def __init__(
        self,
        hidden_weight: float = 0.3,
        logits_weight: float = 0.5,
        attention_weight: float = 0.2,
        temperature: float = 4.0
    ):
        super().__init__()
        
        self.hidden_weight = hidden_weight
        self.logits_weight = logits_weight
        self.attention_weight = attention_weight
        self.temperature = temperature
        
        # Use architecture-aware loss for hidden states
        self.hidden_loss = ArchitectureAwareLoss(
            num_layers=32,  # Typical for Llama models
            layer_decay=0.8,
            divergence_tolerance=0.2,
            normalize_by_layer=True
        )
    
    def forward(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        student_outputs: Dict[str, torch.Tensor],
        teacher_hidden_states: List[torch.Tensor],
        student_hidden_states: List[torch.Tensor],
        reg_norms: List[torch.Tensor],
        step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute functional equivalence loss.
        
        Args:
            teacher_outputs: Dict with 'logits' and optionally 'attentions'
            student_outputs: Dict with 'logits' and optionally 'attentions'
            teacher_hidden_states: List of hidden states
            student_hidden_states: List of hidden states
            reg_norms: Regularization norms from NPT layers
            
        Returns:
            Loss dictionary
        """
        losses = {}
        
        # 1. Hidden state loss (architecture-aware)
        hidden_loss_dict = self.hidden_loss(
            teacher_hidden_states=teacher_hidden_states,
            student_hidden_states=student_hidden_states,
            reg_norms=reg_norms,
            step=step
        )
        losses['hidden_loss'] = hidden_loss_dict['alignment_loss']
        
        # 2. Logits loss (KL divergence with temperature)
        if 'logits' in teacher_outputs and 'logits' in student_outputs:
            teacher_logits = teacher_outputs['logits'] / self.temperature
            student_logits = student_outputs['logits'] / self.temperature
            
            # Compute log probabilities
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            
            # KL divergence
            kl_loss = F.kl_div(
                student_log_probs,
                teacher_log_probs.exp(),
                reduction='batchmean'
            ) * (self.temperature ** 2)  # Scale by temperature squared
            
            losses['logits_loss'] = kl_loss
        else:
            losses['logits_loss'] = torch.tensor(0.0)
        
        # 3. Attention pattern loss (if available)
        if 'attentions' in teacher_outputs and 'attentions' in student_outputs:
            # Compare attention patterns from a few layers
            attn_losses = []
            
            num_layers = min(len(teacher_outputs['attentions']), len(student_outputs['attentions']))
            sample_layers = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]
            
            for layer_idx in sample_layers:
                if layer_idx < num_layers:
                    teacher_attn = teacher_outputs['attentions'][layer_idx]
                    student_attn = student_outputs['attentions'][layer_idx]
                    
                    # Compare attention patterns
                    attn_loss = F.mse_loss(student_attn, teacher_attn)
                    attn_losses.append(attn_loss)
            
            if attn_losses:
                losses['attention_loss'] = torch.stack(attn_losses).mean()
            else:
                losses['attention_loss'] = torch.tensor(0.0)
        else:
            losses['attention_loss'] = torch.tensor(0.0)
        
        # Combine losses
        total_loss = (
            self.hidden_weight * losses['hidden_loss'] +
            self.logits_weight * losses['logits_loss'] +
            self.attention_weight * losses['attention_loss'] +
            hidden_loss_dict['reg_loss']
        )
        
        # Add all components to return dict
        losses.update({
            'total_loss': total_loss,
            'mse_loss': hidden_loss_dict['mse_loss'],
            'cosine_loss': hidden_loss_dict['cosine_loss'],
            'reg_loss': hidden_loss_dict['reg_loss']
        })
        
        return losses


def create_architecture_aware_loss(
    model_config,
    training_config,
    loss_type: str = "architecture_aware"
) -> nn.Module:
    """
    Create loss function that accounts for NPT architectural differences.
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        loss_type: Type of loss ("architecture_aware" or "functional_equivalence")
        
    Returns:
        Loss module
    """
    num_layers = getattr(model_config, 'num_hidden_layers', 32)
    
    if loss_type == "architecture_aware":
        return ArchitectureAwareLoss(
            num_layers=num_layers,
            layer_decay=training_config.get('layer_decay', 0.9),
            divergence_tolerance=training_config.get('divergence_tolerance', 0.1),
            mse_weight=training_config.get('mse_weight', 0.7),
            cosine_weight=training_config.get('cosine_weight', 0.3),
            regularization_lambda=training_config.get('regularization_lambda', 0.01),
            normalize_by_layer=training_config.get('normalize_by_layer', True),
            use_exponential_decay=training_config.get('use_exponential_decay', True)
        )
    
    elif loss_type == "functional_equivalence":
        return FunctionalEquivalenceLoss(
            hidden_weight=training_config.get('hidden_weight', 0.3),
            logits_weight=training_config.get('logits_weight', 0.5),
            attention_weight=training_config.get('attention_weight', 0.2),
            temperature=training_config.get('distill_temperature', 4.0)
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

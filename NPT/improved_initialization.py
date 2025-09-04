"""
Improved weight initialization strategies for NPT adapters.

This module provides various initialization methods that are less restrictive
and better suited for different use cases.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Literal
from enum import Enum


class InitStrategy(Enum):
    """Available initialization strategies."""
    ZERO = "zero"                    # Start with zero effect (current approach)
    XAVIER = "xavier"                # Xavier/Glorot initialization
    KAIMING = "kaiming"              # Kaiming/He initialization
    LORA = "lora"                    # LoRA-style initialization
    SPECTRAL = "spectral"            # Spectral normalization based
    ADAPTIVE = "adaptive"            # Adaptive based on model size
    ORTHOGONAL = "orthogonal"        # Orthogonal initialization


class ImprovedNPTAdapter(nn.Module):
    """
    NPT Adapter with improved initialization strategies.
    
    This version offers multiple initialization options that are less restrictive
    than the original near-zero initialization.
    """
    
    def __init__(
        self, 
        d_model: int, 
        d_ffn: int, 
        r: int = 16,
        init_strategy: str = "adaptive",
        init_scale: float = 1.0,
        alpha: float = 16.0,  # LoRA alpha parameter
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.r = r
        self.init_strategy = InitStrategy(init_strategy)
        self.init_scale = init_scale
        self.alpha = alpha
        
        # Low-rank projections
        self.A_proj = nn.Linear(d_model, r, bias=False)
        self.B_model = nn.Linear(r, d_model, bias=False)
        self.B_ffn = nn.Linear(r, d_ffn, bias=False)
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights based on selected strategy."""
        if self.init_strategy == InitStrategy.ZERO:
            self._init_zero()
        elif self.init_strategy == InitStrategy.XAVIER:
            self._init_xavier()
        elif self.init_strategy == InitStrategy.KAIMING:
            self._init_kaiming()
        elif self.init_strategy == InitStrategy.LORA:
            self._init_lora()
        elif self.init_strategy == InitStrategy.SPECTRAL:
            self._init_spectral()
        elif self.init_strategy == InitStrategy.ADAPTIVE:
            self._init_adaptive()
        elif self.init_strategy == InitStrategy.ORTHOGONAL:
            self._init_orthogonal()
    
    def _init_zero(self):
        """Original near-zero initialization (most restrictive)."""
        nn.init.normal_(self.A_proj.weight, mean=0.0, std=0.02 * self.init_scale)
        nn.init.normal_(self.B_model.weight, mean=0.0, std=0.001 * self.init_scale)
        nn.init.normal_(self.B_ffn.weight, mean=0.0, std=0.001 * self.init_scale)
    
    def _init_xavier(self):
        """Xavier/Glorot initialization - good for linear layers."""
        nn.init.xavier_uniform_(self.A_proj.weight)
        nn.init.xavier_uniform_(self.B_model.weight)
        nn.init.xavier_uniform_(self.B_ffn.weight)
        
        # Scale down to start with smaller effect
        self.A_proj.weight.data *= self.init_scale
        self.B_model.weight.data *= self.init_scale * 0.1
        self.B_ffn.weight.data *= self.init_scale * 0.1
    
    def _init_kaiming(self):
        """Kaiming/He initialization - good for ReLU/GELU networks."""
        nn.init.kaiming_uniform_(self.A_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B_model.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B_ffn.weight, a=math.sqrt(5))
        
        # Scale down
        self.A_proj.weight.data *= self.init_scale
        self.B_model.weight.data *= self.init_scale * 0.1
        self.B_ffn.weight.data *= self.init_scale * 0.1
    
    def _init_lora(self):
        """LoRA-style initialization - proven effective for adapters."""
        # A matrix: normal initialization
        nn.init.normal_(self.A_proj.weight, mean=0.0, std=1.0)
        
        # B matrices: zero initialization (LoRA style)
        nn.init.zeros_(self.B_model.weight)
        nn.init.zeros_(self.B_ffn.weight)
        
        # Scale A based on rank and alpha
        self.A_proj.weight.data *= self.init_scale / self.r
    
    def _init_spectral(self):
        """Spectral normalization based initialization."""
        # Initialize with normal distribution
        nn.init.normal_(self.A_proj.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.B_model.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.B_ffn.weight, mean=0.0, std=1.0)
        
        # Normalize by spectral norm
        with torch.no_grad():
            for weight in [self.A_proj.weight, self.B_model.weight, self.B_ffn.weight]:
                u, s, v = torch.svd(weight)
                weight.data = weight.data / s[0]  # Normalize by largest singular value
                weight.data *= self.init_scale
    
    def _init_adaptive(self):
        """Adaptive initialization based on dimensions."""
        # Scale based on the dimensions to maintain variance
        
        # For A_proj: input is d_model, output is r
        std_a = math.sqrt(2.0 / (self.d_model + self.r))
        nn.init.normal_(self.A_proj.weight, mean=0.0, std=std_a)
        
        # For B_model: input is r, output is d_model
        std_b_model = math.sqrt(2.0 / (self.r + self.d_model))
        nn.init.normal_(self.B_model.weight, mean=0.0, std=std_b_model)
        
        # For B_ffn: input is r, output is d_ffn
        std_b_ffn = math.sqrt(2.0 / (self.r + self.d_ffn))
        nn.init.normal_(self.B_ffn.weight, mean=0.0, std=std_b_ffn)
        
        # Apply scaling factor
        self.A_proj.weight.data *= self.init_scale
        self.B_model.weight.data *= self.init_scale * 0.5
        self.B_ffn.weight.data *= self.init_scale * 0.5
    
    def _init_orthogonal(self):
        """Orthogonal initialization - preserves norm during forward pass."""
        nn.init.orthogonal_(self.A_proj.weight)
        nn.init.orthogonal_(self.B_model.weight)
        nn.init.orthogonal_(self.B_ffn.weight)
        
        # Scale based on dimensions
        self.A_proj.weight.data *= self.init_scale * math.sqrt(2.0 / self.d_model)
        self.B_model.weight.data *= self.init_scale * math.sqrt(2.0 / self.r)
        self.B_ffn.weight.data *= self.init_scale * math.sqrt(2.0 / self.r)
    
    def forward(self, attn_output: torch.Tensor) -> dict:
        """Forward pass with optional dropout."""
        # Low-rank projection
        low_rank_rep = self.A_proj(attn_output)
        
        # Apply dropout if specified
        if self.dropout is not None:
            low_rank_rep = self.dropout(low_rank_rep)
        
        # Generate vectors
        vector_model = self.B_model(low_rank_rep)
        vector_ffn = self.B_ffn(low_rank_rep)
        
        # Compute regularization
        norm_model = torch.sum(vector_model ** 2, dim=-1)
        norm_ffn = torch.sum(vector_ffn ** 2, dim=-1)
        reg_norm = torch.mean(norm_model * norm_ffn)
        
        return {
            'low_rank_rep': low_rank_rep,
            'vector_model': vector_model,
            'vector_ffn': vector_ffn,
            'reg_norm': reg_norm
        }


def analyze_initialization(adapter: nn.Module, num_samples: int = 1000):
    """
    Analyze the initialization by computing statistics of generated weight deltas.
    
    Args:
        adapter: The NPT adapter module
        num_samples: Number of random samples to analyze
        
    Returns:
        Dictionary with initialization statistics
    """
    device = next(adapter.parameters()).device
    d_model = adapter.d_model
    
    # Generate random attention outputs
    random_attn = torch.randn(num_samples, 1, d_model, device=device)
    
    with torch.no_grad():
        outputs = adapter(random_attn)
        vector_model = outputs['vector_model'].squeeze(1)  # (num_samples, d_model)
        vector_ffn = outputs['vector_ffn'].squeeze(1)      # (num_samples, d_ffn)
        
        # Compute statistics for weight deltas
        # The effective weight delta magnitude is ||v_ffn|| * ||v_model||
        model_norms = torch.norm(vector_model, dim=1)
        ffn_norms = torch.norm(vector_ffn, dim=1)
        delta_magnitudes = model_norms * ffn_norms
        
        stats = {
            'vector_model_mean_norm': model_norms.mean().item(),
            'vector_model_std_norm': model_norms.std().item(),
            'vector_ffn_mean_norm': ffn_norms.mean().item(),
            'vector_ffn_std_norm': ffn_norms.std().item(),
            'delta_magnitude_mean': delta_magnitudes.mean().item(),
            'delta_magnitude_std': delta_magnitudes.std().item(),
            'delta_magnitude_max': delta_magnitudes.max().item(),
            'delta_magnitude_min': delta_magnitudes.min().item(),
        }
        
        # Compute effective rank of the transformations
        # This tells us how much capacity the initialization provides
        u_a, s_a, v_a = torch.svd(adapter.A_proj.weight.data)
        u_b_model, s_b_model, v_b_model = torch.svd(adapter.B_model.weight.data)
        u_b_ffn, s_b_ffn, v_b_ffn = torch.svd(adapter.B_ffn.weight.data)
        
        # Effective rank (number of significant singular values)
        threshold = 1e-3
        stats['A_proj_effective_rank'] = (s_a > threshold).sum().item()
        stats['B_model_effective_rank'] = (s_b_model > threshold).sum().item()
        stats['B_ffn_effective_rank'] = (s_b_ffn > threshold).sum().item()
        
    return stats


def compare_initialization_strategies(d_model: int = 4096, d_ffn: int = 11008, r: int = 16):
    """
    Compare different initialization strategies by analyzing their properties.
    
    Args:
        d_model: Model dimension
        d_ffn: FFN dimension
        r: Rank
        
    Returns:
        Comparison results as a formatted string
    """
    results = []
    
    strategies = ["zero", "xavier", "kaiming", "lora", "adaptive", "orthogonal"]
    
    for strategy in strategies:
        adapter = ImprovedNPTAdapter(d_model, d_ffn, r, init_strategy=strategy)
        stats = analyze_initialization(adapter)
        
        results.append(f"\n{strategy.upper()} Initialization:")
        results.append(f"  Delta magnitude: {stats['delta_magnitude_mean']:.6f} ± {stats['delta_magnitude_std']:.6f}")
        results.append(f"  Range: [{stats['delta_magnitude_min']:.6f}, {stats['delta_magnitude_max']:.6f}]")
        results.append(f"  Effective ranks: A={stats['A_proj_effective_rank']}, "
                      f"B_model={stats['B_model_effective_rank']}, B_ffn={stats['B_ffn_effective_rank']}")
    
    return "\n".join(results)


# Example usage for updating existing NPT model
def update_npt_initialization(model, init_strategy: str = "adaptive", init_scale: float = 1.0):
    """
    Update an existing NPT model with improved initialization.
    
    Args:
        model: NPT model
        init_strategy: Initialization strategy to use
        init_scale: Scaling factor for initialization
    """
    for name, module in model.named_modules():
        if hasattr(module, 'adapter') and hasattr(module.adapter, 'A_proj'):
            adapter = module.adapter
            
            # Create new adapter with improved initialization
            new_adapter = ImprovedNPTAdapter(
                d_model=adapter.d_model,
                d_ffn=adapter.d_ffn,
                r=adapter.r,
                init_strategy=init_strategy,
                init_scale=init_scale
            )
            
            # Copy the new weights
            adapter.A_proj.weight.data = new_adapter.A_proj.weight.data.to(adapter.A_proj.weight.device)
            adapter.B_model.weight.data = new_adapter.B_model.weight.data.to(adapter.B_model.weight.device)
            adapter.B_ffn.weight.data = new_adapter.B_ffn.weight.data.to(adapter.B_ffn.weight.device)
            
            print(f"Updated {name} with {init_strategy} initialization")

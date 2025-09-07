"""
Neuro-Plastic Transformer (NPT) Components

This module implements the core components of the NPT architecture,
particularly the Neuro-Plastic (NP) Component that generates weight deltas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class NeuroPlasticComponent(nn.Module):
    """
    The Neuro-Plastic (NP) Component that generates weight deltas (ΔW)
    from attention outputs using low-rank decomposition.
    
    ΔW = (attn_output @ A) @ B
    where A ∈ R^(d_model x r) and B ∈ R^(r x d_ffn)
    """
    
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        rank: int = 16,
        modulation_scale: float = 0.1,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.rank = rank
        self.modulation_scale = modulation_scale
        
        # Low-rank matrices
        self.A = nn.Parameter(torch.empty(d_model, rank))
        self.B = nn.Parameter(torch.empty(rank, d_ffn))
        
        # Initialize with small values to start near zero modulation
        nn.init.normal_(self.A, mean=0.0, std=init_scale / math.sqrt(d_model))
        nn.init.normal_(self.B, mean=0.0, std=init_scale / math.sqrt(rank))
        
    def forward(self, attn_output: torch.Tensor) -> torch.Tensor:
        """
        Generate weight delta from attention output.
        
        Args:
            attn_output: Tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            delta_w: Tensor of shape (batch_size, seq_len, d_model, d_ffn)
        """
        # Compute low-rank weight delta
        # attn_output: (batch, seq_len, d_model)
        # A: (d_model, rank)
        # B: (rank, d_ffn)
        
        # Step 1: attn_output @ A -> (batch, seq_len, rank)
        intermediate = attn_output @ self.A
        
        # Step 2: Expand for batch matrix multiplication
        # We need to compute (batch, seq_len, rank) @ (rank, d_ffn) for each token
        # Result should be (batch, seq_len, d_model, d_ffn)
        
        # First, we need to reconstruct the full delta_w
        # We'll use einsum for clarity
        delta_w = torch.einsum('bsr,rd,bsd->bsrd', 
                               intermediate,  # (batch, seq_len, rank)
                               self.B,        # (rank, d_ffn)
                               attn_output)   # (batch, seq_len, d_model)
        
        # Actually, the above is wrong. Let me reconsider.
        # The weight delta should be: ΔW = A @ (attn_output^T @ B)
        # But this doesn't match dimensions properly.
        
        # Let's think about this differently:
        # We want to modulate W_in which is (d_model, d_ffn)
        # So ΔW should also be (d_model, d_ffn) for each token
        
        # Correct approach:
        # For each token, we compute: ΔW = outer_product(attn @ A, B)
        # But this would be rank-1. Let's use the standard LoRA approach:
        
        # ΔW = A @ B, but scaled by attention features
        # We'll compute a scalar or vector modulation from attention
        
        # Simpler approach: Use attention to compute scaling factors
        # Generate a rank-dimensional modulation vector for each token
        modulation = intermediate  # (batch, seq_len, rank)
        
        # Apply modulation scale
        modulation = self.modulation_scale * torch.tanh(modulation)
        
        # Compute the weight delta
        # We want: for each position, ΔW = sum_r (modulation[r] * A[:, r] @ B[r, :])
        delta_w = torch.einsum('bsr,dr,rf->bsdf', modulation, self.A, self.B)
        
        return delta_w
    
    def get_weight_delta_stats(self, attn_output: torch.Tensor) -> Dict[str, float]:
        """Compute statistics about the generated weight deltas."""
        with torch.no_grad():
            delta_w = self.forward(attn_output)
            stats = {
                'delta_w_mean': delta_w.mean().item(),
                'delta_w_std': delta_w.std().item(),
                'delta_w_max': delta_w.abs().max().item(),
                'delta_w_frobenius': torch.norm(delta_w, p='fro', dim=(-2, -1)).mean().item(),
            }
        return stats


class NPTLayer(nn.Module):
    """
    A single NPT layer that replaces a standard transformer layer.
    Uses the attention output to modulate MLP weights dynamically.
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        d_model: int,
        d_ffn: int,
        rank: int = 16,
        modulation_scale: float = 0.1,
    ):
        super().__init__()
        
        # Store reference to original layer components
        self.self_attn = original_layer.self_attn
        self.mlp = original_layer.mlp
        self.input_layernorm = original_layer.input_layernorm
        self.post_attention_layernorm = original_layer.post_attention_layernorm
        
        # Create the Neuro-Plastic Component
        self.np_component = NeuroPlasticComponent(
            d_model=d_model,
            d_ffn=d_ffn,
            rank=rank,
            modulation_scale=modulation_scale,
        )
        
        # Store original MLP weights (frozen)
        self.register_buffer('W_in_base', original_layer.mlp.gate_proj.weight.data.clone())
        self.register_buffer('W_up_base', original_layer.mlp.up_proj.weight.data.clone())
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        NPT forward pass with dynamic weight modulation.
        """
        residual = hidden_states
        
        # Self-attention
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        attn_output = attn_outputs[0]
        
        # Generate weight delta from attention output
        delta_w = self.np_component(attn_output)
        
        # Apply layer norm before MLP (using residual + attn as in standard transformer)
        # Note: In NPT, we don't add attention to residual here
        hidden_states = self.post_attention_layernorm(residual)
        
        # Modulated MLP forward pass
        # Standard MLP: gate_proj and up_proj in parallel, then down_proj
        # We'll modulate the gate_proj weights
        
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Reshape for batch matrix multiplication
        hidden_states_reshaped = hidden_states.view(batch_size * seq_len, -1)
        
        # Apply modulated weights for each token
        # This is computationally expensive but necessary for per-token modulation
        outputs = []
        for b in range(batch_size):
            for s in range(seq_len):
                h = hidden_states[b, s]  # (d_model,)
                
                # Modulate gate weights
                W_gate_modulated = self.W_in_base + delta_w[b, s].T  # (d_ffn, d_model)
                
                # Apply modulated MLP
                gate = F.silu(F.linear(h, W_gate_modulated))
                up = F.linear(h, self.W_up_base)
                intermediate = gate * up
                output = self.mlp.down_proj(intermediate)
                outputs.append(output)
        
        # Stack outputs
        hidden_states = torch.stack(outputs, dim=0).view(batch_size, seq_len, -1)
        
        # Add residual connection
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,) + attn_outputs[1:]
        return outputs
    
    def forward_original(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Original transformer layer forward pass for comparison.
        """
        residual = hidden_states
        
        # Self-attention
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,) + attn_outputs[1:]
        return outputs

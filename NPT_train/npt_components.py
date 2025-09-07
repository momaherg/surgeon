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
            modulation: Tensor of shape (batch_size, seq_len, rank)
                       This will be used to compute weight deltas on-demand
        """
        # Compute low-rank modulation factors
        # attn_output: (batch, seq_len, d_model)
        # A: (d_model, rank)
        
        # Step 1: attn_output @ A -> (batch, seq_len, rank)
        modulation = attn_output @ self.A  # (batch, seq_len, rank)
        
        # Apply modulation scale and non-linearity
        modulation = self.modulation_scale * torch.tanh(modulation)
        
        # Return modulation factors instead of full weight delta
        # The actual weight delta will be computed on-demand during forward pass
        return modulation
    
    def compute_weight_delta(self, modulation: torch.Tensor, token_idx: int) -> torch.Tensor:
        """
        Compute weight delta for a specific token.
        
        Args:
            modulation: Tensor of shape (batch_size, seq_len, rank)
            token_idx: Index of the token to compute delta for
            
        Returns:
            delta_w: Tensor of shape (batch_size, d_model, d_ffn)
        """
        # Extract modulation for specific token
        token_mod = modulation[:, token_idx, :]  # (batch, rank)
        
        # Compute weight delta: ΔW = A @ (token_mod * B^T)
        # More efficient: ΔW = A @ diag(token_mod) @ B
        # Even more efficient: ΔW = (A * token_mod.unsqueeze(1)) @ B
        
        # Expand dimensions for broadcasting
        token_mod_expanded = token_mod.unsqueeze(1)  # (batch, 1, rank)
        A_expanded = self.A.unsqueeze(0)  # (1, d_model, rank)
        
        # Modulate A matrix
        A_modulated = A_expanded * token_mod_expanded  # (batch, d_model, rank)
        
        # Compute final weight delta
        delta_w = A_modulated @ self.B  # (batch, d_model, d_ffn)
        
        return delta_w
    
    def get_weight_delta_stats(self, attn_output: torch.Tensor) -> Dict[str, float]:
        """Compute statistics about the generated weight deltas."""
        with torch.no_grad():
            modulation = self.forward(attn_output)
            batch_size, seq_len, _ = modulation.shape
            
            # Sample a few tokens to compute statistics
            num_samples = min(seq_len, 10)
            sample_indices = torch.linspace(0, seq_len-1, num_samples, dtype=torch.long)
            
            delta_norms = []
            delta_means = []
            delta_maxs = []
            
            for idx in sample_indices:
                delta_w = self.compute_weight_delta(modulation, idx.item())
                delta_norms.append(torch.norm(delta_w, p='fro', dim=(-2, -1)).mean().item())
                delta_means.append(delta_w.mean().item())
                delta_maxs.append(delta_w.abs().max().item())
            
            stats = {
                'delta_w_mean': sum(delta_means) / len(delta_means),
                'delta_w_std': torch.std(torch.tensor(delta_means)).item(),
                'delta_w_max': max(delta_maxs),
                'delta_w_frobenius': sum(delta_norms) / len(delta_norms),
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
        # Handle different architectures
        if hasattr(original_layer, 'self_attn'):
            # LLaMA style
            self.self_attn = original_layer.self_attn
            self.input_layernorm = original_layer.input_layernorm
            self.post_attention_layernorm = original_layer.post_attention_layernorm
        elif hasattr(original_layer, 'attn'):
            # GPT2 style
            self.self_attn = original_layer.attn
            self.input_layernorm = original_layer.ln_1
            self.post_attention_layernorm = original_layer.ln_2
        else:
            raise ValueError("Unsupported attention architecture")
        
        self.mlp = original_layer.mlp
        
        # Create the Neuro-Plastic Component
        self.np_component = NeuroPlasticComponent(
            d_model=d_model,
            d_ffn=d_ffn,
            rank=rank,
            modulation_scale=modulation_scale,
        )
        
        # Store original MLP weights (frozen)
        # Handle different model architectures
        if hasattr(original_layer.mlp, 'gate_proj'):
            # LLaMA-style MLP
            self.register_buffer('W_in_base', original_layer.mlp.gate_proj.weight.data.clone())
            self.register_buffer('W_up_base', original_layer.mlp.up_proj.weight.data.clone())
            self.mlp_type = 'llama'
        elif hasattr(original_layer.mlp, 'c_fc'):
            # GPT2-style MLP
            self.register_buffer('W_in_base', original_layer.mlp.c_fc.weight.data.clone())
            self.mlp_type = 'gpt2'
        else:
            raise ValueError(f"Unsupported MLP architecture")
        
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
        # Handle the case where hidden_states might be a tuple
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
            
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
        
        # Generate modulation factors from attention output
        modulation = self.np_component(attn_output)
        
        # Apply layer norm before MLP (using residual as in NPT design)
        hidden_states = self.post_attention_layernorm(residual)
        
        # Modulated MLP forward pass
        batch_size, seq_len, d_model = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # Process tokens in chunks to balance memory and efficiency
        chunk_size = 8  # Process 8 tokens at a time
        output_chunks = []
        
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_hidden = hidden_states[:, chunk_start:chunk_end, :]  # (batch, chunk_size, d_model)
            chunk_outputs = []
            
            # Process each token in the chunk
            for token_idx in range(chunk_start, chunk_end):
                # Get modulation-based weight delta for this token
                delta_w = self.np_component.compute_weight_delta(modulation, token_idx)  # (batch, d_model, d_ffn)
                
                # Get hidden state for this token
                h = hidden_states[:, token_idx, :]  # (batch, d_model)
                
                # Apply modulated MLP based on architecture
                if self.mlp_type == 'llama':
                    # Modulate gate weights
                    W_gate_modulated = self.W_in_base.unsqueeze(0) + delta_w.transpose(-2, -1)  # (batch, d_ffn, d_model)
                    
                    # Compute gate activation for all batches at once
                    gate = F.silu(torch.bmm(W_gate_modulated, h.unsqueeze(-1)).squeeze(-1))  # (batch, d_ffn)
                    
                    # Up projection (unmodulated)
                    up = F.linear(h, self.W_up_base)  # (batch, d_ffn)
                    
                    # Element-wise product and down projection
                    intermediate = gate * up
                    output = self.mlp.down_proj(intermediate)  # (batch, d_model)
                else:  # GPT2
                    # For GPT2, weights are stored as (d_model, d_ffn), not (d_ffn, d_model)
                    # So we don't need to transpose delta_w
                    # delta_w has shape (batch, d_model, d_ffn)
                    # W_in_base has shape (d_model, d_ffn)
                    
                    # Use F.linear which handles the weight correctly
                    base_output = F.linear(h, self.W_in_base)  # (batch, d_ffn)
                    
                    # Compute modulated output efficiently
                    # delta_w @ h gives the modulation term
                    delta_output = torch.bmm(delta_w.transpose(-2, -1), h.unsqueeze(-1)).squeeze(-1)  # (batch, d_ffn)
                    
                    intermediate = F.gelu(base_output + delta_output)
                    
                    # Output projection
                    output = self.mlp.c_proj(intermediate)  # (batch, d_model)
                
                chunk_outputs.append(output)
            
            # Stack outputs for this chunk
            chunk_output = torch.stack(chunk_outputs, dim=1)  # (batch, chunk_size, d_model)
            output_chunks.append(chunk_output)
        
        # Concatenate all chunks
        hidden_states = torch.cat(output_chunks, dim=1)  # (batch, seq_len, d_model)
        
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
        # Handle the case where hidden_states might be a tuple
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
            
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

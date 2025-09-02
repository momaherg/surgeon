"""
NPT (NLP NeuroPlastic Transformer) Layer Implementation

This module contains the core architecture components for the NPT model,
including the Adapter module and the modified transformer layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class NPTAdapter(nn.Module):
    """
    Low-rank adapter module that generates per-token modulation effects for MLP.
    
    This module processes attention outputs to create token-specific modulations
    that are applied within the MLP computation, preserving per-token dynamics.
    
    Args:
        d_model: Model dimension (hidden size)
        d_ffn: FFN dimension (intermediate size)
        r: Low-rank dimension for factorization
    """
    
    def __init__(self, d_model: int, d_ffn: int, r: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.r = r
        
        # Low-rank factorization: A projects to rank r, B projects from r to d_ffn
        self.A_proj = nn.Linear(d_model, r, bias=False)
        self.B_proj = nn.Linear(r, d_ffn, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for zero delta at start."""
        # Kaiming uniform for A_proj
        nn.init.kaiming_uniform_(self.A_proj.weight, a=math.sqrt(5))
        # Zeros for B_proj to ensure zero delta initially
        nn.init.zeros_(self.B_proj.weight)
    
    def forward(self, attn_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes per-token modulation effects from attention outputs.
        
        This is a per-token operation that preserves the dynamic nature of the
        attention mechanism, allowing each token to have its own modulation.
        
        Args:
            attn_output: Attention output tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            delta_effect: Per-token modulation effects, shape (batch_size, seq_len, d_ffn)
            norm: Regularization term (scalar tensor)
        """
        # Project through low-rank bottleneck
        # Shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, r)
        low_rank_rep = self.A_proj(attn_output)
        
        # Project to FFN dimension
        # Shape: (batch_size, seq_len, r) -> (batch_size, seq_len, d_ffn)
        delta_effect = self.B_proj(low_rank_rep)
        
        # Calculate regularization norm efficiently
        # Average the squared L2 norm over batch and sequence dimensions
        norm = torch.mean(torch.sum(delta_effect ** 2, dim=-1))
        
        return delta_effect, norm


class NPTLayer(nn.Module):
    """
    Modified transformer layer implementing NPT mechanism.
    
    This layer replaces the standard attention-to-MLP connection with
    dynamic weight modulation based on attention outputs.
    """
    
    def __init__(self, base_layer, adapter_config: dict):
        """
        Args:
            base_layer: Original transformer layer to modify
            adapter_config: Configuration for adapter (r, d_model, d_ffn)
        """
        super().__init__()
        
        # Store reference to original components
        self.self_attn = base_layer.self_attn
        self.mlp = base_layer.mlp
        self.input_layernorm = base_layer.input_layernorm
        self.post_attention_layernorm = base_layer.post_attention_layernorm
        
        # Create NPT adapter
        self.adapter = NPTAdapter(
            d_model=adapter_config['d_model'],
            d_ffn=adapter_config['d_ffn'],
            r=adapter_config['r']
        )
        
        # Store original gate_proj weight for reference
        self.register_buffer('original_gate_weight', base_layer.mlp.gate_proj.weight.clone())
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        NPT forward pass with per-token dynamic modulation.
        
        Key differences from standard transformer:
        1. Attention outputs modulate MLP activations (not weights)
        2. Per-token modulation preserves token-specific dynamics
        3. Regularization norm is computed and returned efficiently
        """
        residual = hidden_states
        
        # Self Attention
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs
        )
        attn_output = attn_outputs[0]
        
        # Standard residual connection (important for LayerNorm distribution)
        h_residual = residual + attn_output
        
        # NPT Mechanism: Generate per-token modulation effects
        delta_effect, reg_norm = self.adapter(attn_output)
        
        # MLP with modulation
        # Apply LayerNorm to the standard residual path
        mlp_input = self.post_attention_layernorm(h_residual)
        
        # Standard MLP projections
        gate_output = F.linear(mlp_input, self.mlp.gate_proj.weight, self.mlp.gate_proj.bias)
        up_output = F.linear(mlp_input, self.mlp.up_proj.weight, self.mlp.up_proj.bias)
        
        # Apply modulation to gate activation (inside SwiGLU)
        # This preserves per-token dynamics while being computationally efficient
        intermediate = F.silu(gate_output + delta_effect) * up_output
        mlp_output = F.linear(intermediate, self.mlp.down_proj.weight, self.mlp.down_proj.bias)
        
        # Final residual connection (NPT skips the attention residual)
        hidden_states = residual + mlp_output
        
        # Package outputs with regularization norm
        outputs = (hidden_states, reg_norm)
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[2:],) if output_attentions else (attn_outputs[1],)
        
        return outputs
    



def convert_llama_to_npt(model, adapter_config: dict):
    """
    Convert a standard Llama model to NPT architecture.
    
    Args:
        model: Llama model to convert
        adapter_config: Configuration for adapters (r value, etc.)
        
    Returns:
        Modified model with NPT layers
    """
    # Get model configuration
    config = model.config
    d_model = config.hidden_size
    d_ffn = config.intermediate_size
    
    # Default adapter configuration
    default_config = {
        'd_model': d_model,
        'd_ffn': d_ffn,
        'r': 16
    }
    default_config.update(adapter_config)
    
    # Replace each transformer layer with NPT layer
    for i, layer in enumerate(model.model.layers):
        npt_layer = NPTLayer(layer, default_config)
        model.model.layers[i] = npt_layer
    
    # Freeze all original parameters
    for name, param in model.named_parameters():
        if 'adapter' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    return model


def get_adapter_params(model):
    """Get only adapter parameters for optimization."""
    adapter_params = []
    for name, param in model.named_parameters():
        if 'adapter' in name and param.requires_grad:
            adapter_params.append(param)
    return adapter_params




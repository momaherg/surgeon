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
    Low-rank adapter module that generates weight deltas for MLP modulation.
    
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
        
        # Low-rank factorization matrices
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
    
    def forward(self, attn_output: torch.Tensor) -> torch.Tensor:
        """
        Generate weight delta from attention output.
        
        Args:
            attn_output: Attention output tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            delta_W: Weight delta tensor of shape (d_ffn, d_model)
        """
        # Average attention output over batch and sequence dimensions
        # Shape: (batch_size, seq_len, d_model) -> (d_model,)
        attn_pooled = attn_output.mean(dim=[0, 1])
        
        # Project to low-rank space
        # Shape: (d_model,) -> (r,)
        low_rank_features = self.A_proj(attn_pooled)
        
        # Generate weight delta using outer product
        # delta_W = B @ (low_rank_features ⊗ A^T)
        # This is equivalent to: delta_W = (B @ diag(low_rank_features)) @ A^T
        # Shape: (d_ffn, r) @ (r, d_model) -> (d_ffn, d_model)
        
        # Scale B weights by low-rank features
        scaled_B = self.B_proj.weight * low_rank_features.unsqueeze(0)
        # Compute final delta
        delta_W = scaled_B @ self.A_proj.weight
        
        return delta_W
    
    def compute_delta_W_norm(self, attn_output: torch.Tensor) -> torch.Tensor:
        """Compute Frobenius norm of generated delta_W for regularization."""
        delta_W = self.forward(attn_output)
        return torch.norm(delta_W, p='fro') ** 2


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
        NPT forward pass with dynamic weight modulation.
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
        
        # NPT Mechanism: Generate weight delta from attention output
        delta_W = self.adapter(attn_output)
        
        # Modulate gate projection weight
        # Note: We need to handle the weight shape correctly
        # Llama uses (out_features, in_features) convention
        modulated_weight = self.mlp.gate_proj.weight + delta_W
        
        # Apply MLP with modulated weights
        hidden_states = self.post_attention_layernorm(residual)
        
        # SwiGLU activation with modulated gate
        gate_output = F.linear(hidden_states, modulated_weight, self.mlp.gate_proj.bias)
        up_output = F.linear(hidden_states, self.mlp.up_proj.weight, self.mlp.up_proj.bias)
        intermediate = F.silu(gate_output) * up_output
        mlp_output = F.linear(intermediate, self.mlp.down_proj.weight, self.mlp.down_proj.bias)
        
        # Residual connection after MLP (NPT modification)
        hidden_states = residual + mlp_output
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[2:],) if output_attentions else (attn_outputs[1],)
        
        return outputs
    
    def get_delta_W_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get the Frobenius norm of current delta_W for monitoring."""
        # Get attention output for delta calculation
        hidden_states_norm = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(hidden_states=hidden_states_norm)
        attn_output = attn_outputs[0]
        
        return self.adapter.compute_delta_W_norm(attn_output)


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


def compute_regularization_loss(model, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Compute regularization loss for all NPT layers.
    
    Args:
        model: NPT model
        hidden_states: Input hidden states
        
    Returns:
        Average Frobenius norm of all delta_W matrices
    """
    total_norm = 0.0
    num_layers = 0
    
    for layer in model.model.layers:
        if isinstance(layer, NPTLayer):
            norm = layer.get_delta_W_norm(hidden_states)
            total_norm += norm
            num_layers += 1
    
    return total_norm / num_layers if num_layers > 0 else torch.tensor(0.0)

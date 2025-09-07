"""
Optimized Neuro-Plastic Transformer (NPT) Components with aggressive memory management.

This version includes:
- CPU offloading for weight delta computation
- Mixed precision support
- Gradient checkpointing compatibility
- Memory-efficient chunked processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class NeuroPlasticComponentOptimized(nn.Module):
    """
    Memory-optimized Neuro-Plastic Component with CPU offloading option.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        rank: int = 16,
        modulation_scale: float = 0.1,
        init_scale: float = 0.01,
        use_cpu_offload: bool = True,  # Offload weight computation to CPU
        dtype: torch.dtype = torch.float16,  # Use mixed precision
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.rank = rank
        self.modulation_scale = modulation_scale
        self.use_cpu_offload = use_cpu_offload
        self.compute_dtype = dtype
        
        # Low-rank matrices - keep in float32 for stability
        self.A = nn.Parameter(torch.empty(d_model, rank, dtype=torch.float32))
        self.B = nn.Parameter(torch.empty(rank, d_ffn, dtype=torch.float32))
        
        # Initialize with small values to start near zero modulation
        nn.init.normal_(self.A, mean=0.0, std=init_scale / math.sqrt(d_model))
        nn.init.normal_(self.B, mean=0.0, std=init_scale / math.sqrt(rank))
        
        # Pre-allocate CPU tensors if using offloading
        if use_cpu_offload:
            self.A_cpu = None
            self.B_cpu = None
    
    def _ensure_cpu_copies(self):
        """Ensure CPU copies of matrices are available and up-to-date."""
        if self.use_cpu_offload:
            if self.A_cpu is None or not torch.equal(self.A_cpu, self.A.cpu()):
                self.A_cpu = self.A.detach().cpu()
            if self.B_cpu is None or not torch.equal(self.B_cpu, self.B.cpu()):
                self.B_cpu = self.B.detach().cpu()
    
    def forward(self, attn_output: torch.Tensor) -> torch.Tensor:
        """Generate modulation factors from attention output."""
        # Convert to compute dtype for efficiency
        attn_output = attn_output.to(self.compute_dtype)
        A_compute = self.A.to(self.compute_dtype)
        
        # Compute modulation factors
        modulation = attn_output @ A_compute  # (batch, seq_len, rank)
        
        # Apply modulation scale and non-linearity
        modulation = self.modulation_scale * torch.tanh(modulation)
        
        return modulation
    
    def compute_weight_delta_efficient(
        self, 
        modulation: torch.Tensor, 
        token_idx: int,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weight delta and apply it to hidden state in one step.
        This avoids materializing the full delta matrix.
        
        Args:
            modulation: Tensor of shape (batch_size, seq_len, rank)
            token_idx: Index of the token to compute delta for
            hidden_state: Tensor of shape (batch_size, d_model)
            
        Returns:
            output: Tensor of shape (batch_size, d_ffn) - the modulated projection
        """
        # Extract modulation for specific token
        token_mod = modulation[:, token_idx, :]  # (batch, rank)
        
        if self.use_cpu_offload:
            # Offload computation to CPU to save GPU memory
            self._ensure_cpu_copies()
            token_mod_cpu = token_mod.cpu()
            hidden_state_cpu = hidden_state.cpu()
            
            # Compute on CPU: h @ A @ diag(mod) @ B
            # More efficient: (h @ A) * mod @ B
            h_A = hidden_state_cpu @ self.A_cpu  # (batch, rank)
            h_A_mod = h_A * token_mod_cpu  # (batch, rank)
            output = h_A_mod @ self.B_cpu  # (batch, d_ffn)
            
            # Move back to GPU
            return output.to(hidden_state.device).to(hidden_state.dtype)
        else:
            # GPU computation with mixed precision
            A_compute = self.A.to(self.compute_dtype)
            B_compute = self.B.to(self.compute_dtype)
            hidden_compute = hidden_state.to(self.compute_dtype)
            
            # Efficient computation avoiding full delta materialization
            h_A = hidden_compute @ A_compute  # (batch, rank)
            h_A_mod = h_A * token_mod  # (batch, rank)
            output = h_A_mod @ B_compute  # (batch, d_ffn)
            
            return output.to(hidden_state.dtype)
    
    def compute_weight_delta(self, modulation: torch.Tensor, token_idx: int) -> torch.Tensor:
        """
        Original weight delta computation (for compatibility).
        Note: This can still cause OOM for large models.
        """
        if self.use_cpu_offload:
            # Compute on CPU
            self._ensure_cpu_copies()
            token_mod = modulation[:, token_idx, :].cpu()  # (batch, rank)
            
            token_mod_expanded = token_mod.unsqueeze(1)  # (batch, 1, rank)
            A_expanded = self.A_cpu.unsqueeze(0)  # (1, d_model, rank)
            
            A_modulated = A_expanded * token_mod_expanded  # (batch, d_model, rank)
            delta_w = A_modulated @ self.B_cpu  # (batch, d_model, d_ffn)
            
            return delta_w.to(modulation.device)
        else:
            # Original GPU computation
            token_mod = modulation[:, token_idx, :]  # (batch, rank)
            
            token_mod_expanded = token_mod.unsqueeze(1)  # (batch, 1, rank)
            A_expanded = self.A.unsqueeze(0)  # (1, d_model, rank)
            
            A_modulated = A_expanded * token_mod_expanded  # (batch, d_model, rank)
            delta_w = A_modulated @ self.B  # (batch, d_model, d_ffn)
            
            return delta_w


class NPTLayerOptimized(nn.Module):
    """
    Optimized NPT layer with memory-efficient forward pass.
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        d_model: int,
        d_ffn: int,
        rank: int = 16,
        modulation_scale: float = 0.1,
        use_cpu_offload: bool = True,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        
        # Store reference to original layer components
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
        
        # Create the optimized Neuro-Plastic Component
        self.np_component = NeuroPlasticComponentOptimized(
            d_model=d_model,
            d_ffn=d_ffn,
            rank=rank,
            modulation_scale=modulation_scale,
            use_cpu_offload=use_cpu_offload,
            dtype=dtype,
        )
        
        # Store original MLP weights (frozen) - keep reference only, don't copy
        if hasattr(original_layer.mlp, 'gate_proj'):
            # LLaMA-style MLP
            self.W_in_base = original_layer.mlp.gate_proj.weight
            self.W_up_base = original_layer.mlp.up_proj.weight
            self.mlp_type = 'llama'
        elif hasattr(original_layer.mlp, 'c_fc'):
            # GPT2-style MLP
            self.W_in_base = original_layer.mlp.c_fc.weight
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
        Optimized NPT forward pass with minimal memory usage.
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
        
        # Apply layer norm before MLP
        hidden_states = self.post_attention_layernorm(residual)
        
        # Memory-efficient modulated MLP forward pass
        batch_size, seq_len, d_model = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # Process tokens one at a time to minimize memory usage
        outputs = []
        
        for token_idx in range(seq_len):
            h = hidden_states[:, token_idx, :]  # (batch, d_model)
            
            if self.mlp_type == 'llama':
                # Compute modulated gate projection efficiently
                gate_modulated = self.np_component.compute_weight_delta_efficient(
                    modulation, token_idx, h
                )
                gate_base = F.linear(h, self.W_in_base)
                gate = F.silu(gate_base + gate_modulated)
                
                # Up projection (unmodulated)
                up = F.linear(h, self.W_up_base)
                
                # Element-wise product and down projection
                intermediate = gate * up
                output = self.mlp.down_proj(intermediate)
            else:  # GPT2
                # Compute modulated projection efficiently
                fc_modulated = self.np_component.compute_weight_delta_efficient(
                    modulation, token_idx, h
                )
                fc_base = F.linear(h, self.W_in_base)
                intermediate = F.gelu(fc_base + fc_modulated)
                
                # Output projection
                output = self.mlp.c_proj(intermediate)
            
            outputs.append(output)
        
        # Stack outputs
        hidden_states = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        
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
        """Original transformer layer forward pass for comparison."""
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

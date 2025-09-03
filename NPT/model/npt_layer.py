"""
NPT (NLP NeuroPlastic Transformer) Layer Implementation
Aligned with Research Proposal

This module implements the NPT architecture as described in the research proposal,
with attention outputs modulating MLP weights through efficient approximations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union
import math


class NPTAdapter(nn.Module):
    """
    Enhanced adapter module that generates weight modulation effects.
    
    This version generates two vectors that create a low-rank weight delta
    through their outer product, implementing true weight modulation efficiently.
    
    Args:
        d_model: Model dimension (hidden size)
        d_ffn: FFN dimension (intermediate size)
        r: Low-rank dimension for factorization
        modulation_type: Type of modulation (kept for compatibility, but now uses outer product)
    """
    
    def __init__(self, d_model: int, d_ffn: int, r: int = 16, 
                 modulation_type: str = 'both'):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.r = r
        self.modulation_type = modulation_type
        
        # Low-rank projection to generate two vectors
        self.A_proj = nn.Linear(d_model, r, bias=False)
        
        # Generate d_model vector from low-rank representation
        self.B_model = nn.Linear(r, d_model, bias=False)
        
        # Generate d_ffn vector from low-rank representation  
        self.B_ffn = nn.Linear(r, d_ffn, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize for minimal initial modulation."""
        # Use smaller initialization for stability
        nn.init.normal_(self.A_proj.weight, mean=0.0, std=0.02)
        
        # Initialize vectors to near-zero for minimal initial weight delta
        nn.init.normal_(self.B_model.weight, mean=0.0, std=0.001)
        nn.init.normal_(self.B_ffn.weight, mean=0.0, std=0.001)
    
    def forward(self, attn_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate modulation effects from attention outputs.
        
        Args:
            attn_output: Attention output tensor (batch_size, seq_len, d_model)
            
        Returns:
            Dictionary containing:
                - vector_model: d_model dimensional vector
                - vector_ffn: d_ffn dimensional vector
                - delta_weight: Outer product weight delta (optional, computed on demand)
                - reg_norm: Regularization norm
                - low_rank_rep: Low-rank representation for permanent updates
        """
        # Ensure input matches adapter dtype to avoid mixed precision issues
        if attn_output.dtype != self.A_proj.weight.dtype:
            attn_output = attn_output.to(self.A_proj.weight.dtype)
        
        # Low-rank projection
        low_rank_rep = self.A_proj(attn_output)  # (batch, seq_len, r)
        
        # Generate two vectors
        vector_model = self.B_model(low_rank_rep)  # (batch, seq_len, d_model)
        vector_ffn = self.B_ffn(low_rank_rep)     # (batch, seq_len, d_ffn)
        
        outputs = {
            'low_rank_rep': low_rank_rep,
            'vector_model': vector_model,
            'vector_ffn': vector_ffn
        }
        
        # Compute regularization based on the Frobenius norm of the implied weight delta
        # ||delta_W||_F^2 = ||vector_ffn||^2 * ||vector_model||^2
        norm_model = torch.sum(vector_model ** 2, dim=-1)  # (batch, seq_len)
        norm_ffn = torch.sum(vector_ffn ** 2, dim=-1)      # (batch, seq_len)
        reg_norm = torch.mean(norm_model * norm_ffn)
        
        outputs['reg_norm'] = reg_norm
        
        return outputs


class NPTLayer(nn.Module):
    """
    NPT Layer with corrected architecture matching the research proposal.
    
    Key changes from standard transformer:
    1. Attention outputs modulate MLP weights (approximated efficiently)
    2. Single residual connection after MLP (not dual residual)
    3. Support for permanent weight updates
    """
    
    def __init__(self, base_layer, adapter_config: dict):
        """
        Args:
            base_layer: Original transformer layer
            adapter_config: Configuration including r, modulation_type, etc.
        """
        super().__init__()
        
        # Store original components
        self.self_attn = base_layer.self_attn
        self.mlp = base_layer.mlp
        self.input_layernorm = base_layer.input_layernorm
        self.post_attention_layernorm = base_layer.post_attention_layernorm
        
        # Create adapter
        self.adapter = NPTAdapter(
            d_model=adapter_config.get('d_model'),
            d_ffn=adapter_config.get('d_ffn'),
            r=adapter_config.get('r', 16),
            modulation_type=adapter_config.get('modulation_type', 'both')
        )
        
        # Move adapter to the same device as the base layer
        device = next(base_layer.parameters()).device
        
        # Use compute dtype from config (handles quantized models correctly)
        # For quantized models, always use FP32 for stability
        # For non-quantized models, match the model's dtype
        is_quantized = hasattr(base_layer.mlp.gate_proj, 'weight') and hasattr(base_layer.mlp.gate_proj.weight, 'CB')
        if is_quantized:
            # Always use FP32 for adapters with quantized models
            adapter_dtype = torch.float32
        else:
            # For non-quantized models, use the compute dtype from config
            # or try to match the model's parameter dtype
            adapter_dtype = adapter_config.get('compute_dtype', None)
            if adapter_dtype is None:
                # Try to infer from model parameters
                try:
                    param_dtype = next(base_layer.parameters()).dtype
                    adapter_dtype = param_dtype
                except:
                    adapter_dtype = torch.float32
            
        self.adapter = self.adapter.to(device=device, dtype=adapter_dtype)
        
        # Permanent update parameters
        self.consolidation_alpha = adapter_config.get('consolidation_alpha', 0.1)
        
        # Modulation scaling factor
        self.modulation_scale = adapter_config.get('modulation_scale', 0.1)
        
        # Store original weights for reference (only for non-quantized models)
        # For quantized models, weights are stored in a compressed format
        if hasattr(base_layer.mlp.gate_proj, 'weight') and not hasattr(base_layer.mlp.gate_proj.weight, 'CB'):
            self.register_buffer('original_gate_weight', base_layer.mlp.gate_proj.weight.clone().detach())
        else:
            self.register_buffer('original_gate_weight', torch.tensor(0.0))  # Placeholder for quantized models
        
        # Track if weights have been permanently updated
        self.weights_updated = False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_modulation: bool = False,
        **kwargs
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """
        NPT forward pass with corrected residual structure.
        
        Args:
            return_modulation: If True, return modulation values for permanent updates
        """
        # Store original input for final residual
        original_input = hidden_states
        
        # Prepare attention mask for newer Llama models using SDPA
        # Check if the attention layer is using SDPA
        uses_sdpa = hasattr(self.self_attn, '_attn_implementation') and self.self_attn._attn_implementation == 'sdpa'
        
        if attention_mask is not None:
            # For SDPA, we need to handle the mask differently
            if uses_sdpa:
                # SDPA in newer transformers expects 4D attention masks or None
                # For simplicity and compatibility, we'll pass None and let it use causal mask
                # This works for standard autoregressive training
                processed_attention_mask = None
                # Note: If you need custom masking (e.g., for special tokens), you'll need
                # to implement proper 4D mask conversion here
            else:
                # For non-SDPA implementations, ensure correct dtype
                if attention_mask.dtype != hidden_states.dtype:
                    attention_mask = attention_mask.to(hidden_states.dtype)
                processed_attention_mask = attention_mask
        else:
            processed_attention_mask = None
        
        # Self-attention
        hidden_states = self.input_layernorm(hidden_states)
        
        # Build attention kwargs based on what the model accepts
        attn_kwargs = {
            'hidden_states': hidden_states,
            'attention_mask': processed_attention_mask,
            'position_ids': position_ids,
            'past_key_value': past_key_value,
            'output_attentions': output_attentions,
            'use_cache': use_cache,
        }
        
        # Only add optional parameters if they're supported
        if cache_position is not None:
            attn_kwargs['cache_position'] = cache_position
        
        # Check if self_attn accepts position_embeddings
        import inspect
        attn_signature = inspect.signature(self.self_attn.forward)
        if 'position_embeddings' in attn_signature.parameters and position_embeddings is not None:
            attn_kwargs['position_embeddings'] = position_embeddings
        
        # Add any additional kwargs
        attn_kwargs.update(kwargs)
        
        attn_outputs = self.self_attn(**attn_kwargs)
        attn_output = attn_outputs[0]
        
        # NPT: Generate modulation from attention (NO residual here!)
        modulation = self.adapter(attn_output)
        
        # MLP with weight modulation
        # Use original input for MLP computation (not attention output)
        mlp_input = self.post_attention_layernorm(original_input)
        
        # Get the vectors from modulation
        vector_model = modulation['vector_model']  # (batch, seq_len, d_model)
        vector_ffn = modulation['vector_ffn']      # (batch, seq_len, d_ffn)
        
        # Ensure vectors match mlp_input dtype
        if vector_model.dtype != mlp_input.dtype:
            vector_model = vector_model.to(mlp_input.dtype)
        if vector_ffn.dtype != mlp_input.dtype:
            vector_ffn = vector_ffn.to(mlp_input.dtype)
        
        # Apply weight modulation through efficient computation
        # Instead of computing the full outer product and then multiplying with input,
        # we use the fact that (v_ffn ⊗ v_model) @ mlp_input = v_ffn * (v_model @ mlp_input)
        # This is much more efficient: O(d_model + d_ffn) instead of O(d_model * d_ffn)
        
        # Compute the dot product between vector_model and mlp_input for each position
        # mlp_input: (batch, seq_len, d_model), vector_model: (batch, seq_len, d_model)
        dot_product = torch.sum(mlp_input * vector_model, dim=-1, keepdim=True)  # (batch, seq_len, 1)
        
        # Scale the ffn vector by the dot product to get the modulation effect
        # This is equivalent to: delta_W @ mlp_input where delta_W = vector_ffn ⊗ vector_model
        weight_modulation = vector_ffn * dot_product  # (batch, seq_len, d_ffn)
        
        # Apply modulation with configurable scaling factor for stability
        gate_output = self.mlp.gate_proj(mlp_input)
        modulated_gate = gate_output + self.modulation_scale * weight_modulation
        
        # Continue MLP computation
        up_output = self.mlp.up_proj(mlp_input)
        intermediate = F.silu(modulated_gate) * up_output
        mlp_output = self.mlp.down_proj(intermediate)
        
        # Single residual connection (as per proposal)
        hidden_states = original_input + mlp_output
        
        # Return based on mode
        if return_modulation:
            return {
                'hidden_states': hidden_states,
                'modulation': modulation,
                'attn_outputs': attn_outputs
            }
        
        # Standard return format
        outputs = (hidden_states, modulation['reg_norm'])
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[2:],) if output_attentions else (attn_outputs[1],)
        
        return outputs
    
    def consolidate_weights(
        self,
        context_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        token_idx: int = -1,
        alpha: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Permanently consolidate weight deltas into base weights.
        
        This implements the Permanent Update Mode from the proposal.
        
        Args:
            context_tokens: Input tokens containing fact to memorize
            attention_mask: Attention mask for the tokens
            position_ids: Position IDs for the tokens
            position_embeddings: Position embeddings for the tokens
            token_idx: Which token's modulation to use (default: last)
            alpha: Update strength (overrides self.consolidation_alpha)
            
        Returns:
            Dictionary with update statistics
        """
        # Check if model is quantized
        is_quantized = hasattr(self.mlp.gate_proj, 'weight') and hasattr(self.mlp.gate_proj.weight, 'CB')
        
        if is_quantized:
            # For quantized models, we can't directly modify weights
            # Return a message indicating this limitation
            return {
                'weight_update_norm': 0.0,
                'alpha_used': 0.0,
                'token_idx': token_idx,
                'message': 'Weight consolidation not supported for quantized models'
            }
        
        with torch.no_grad():
            # Get modulation for the context
            outputs = self.forward(
                context_tokens,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                return_modulation=True
            )
            
            modulation = outputs['modulation']
            
            # Select the target token's modulation vectors
            vector_model = modulation['vector_model'][:, token_idx]  # (batch, d_model)
            vector_ffn = modulation['vector_ffn'][:, token_idx]      # (batch, d_ffn)
            
            # Take mean over batch dimension
            vector_model_mean = vector_model.mean(0)  # (d_model,)
            vector_ffn_mean = vector_ffn.mean(0)      # (d_ffn,)
            
            # Compute the outer product to get the weight delta
            # delta_W = vector_ffn ⊗ vector_model
            # Shape: (d_ffn, d_model) which matches gate_proj.weight shape
            weight_update = torch.outer(vector_ffn_mean, vector_model_mean)
            
            # Apply update with scaling
            update_alpha = alpha if alpha is not None else self.consolidation_alpha
            self.mlp.gate_proj.weight.data += update_alpha * weight_update
            
            self.weights_updated = True
            
            return {
                'weight_update_norm': torch.norm(weight_update).item(),
                'alpha_used': update_alpha,
                'token_idx': token_idx,
                'vector_model_norm': torch.norm(vector_model_mean).item(),
                'vector_ffn_norm': torch.norm(vector_ffn_mean).item()
            }


def convert_llama_to_npt(model, adapter_config: dict):
    """
    Convert a Llama model to NPT architecture.
    
    Args:
        model: Llama model to convert
        adapter_config: Configuration for adapters
        
    Returns:
        Modified model with NPT layers
    """
    config = model.config
    
    # Check if model is quantized by examining the first layer
    first_layer = model.model.layers[0]
    is_quantized = hasattr(first_layer.mlp.gate_proj, 'weight') and hasattr(first_layer.mlp.gate_proj.weight, 'CB')
    
    # Determine compute dtype for adapters
    if is_quantized:
        # Always use FP32 for adapters with quantized models
        compute_dtype = torch.float32
    else:
        # Try to get dtype from config or model parameters
        if hasattr(config, 'torch_dtype') and config.torch_dtype is not None:
            compute_dtype = config.torch_dtype
        else:
            # Try to infer from model parameters
            try:
                compute_dtype = next(model.parameters()).dtype
            except:
                compute_dtype = torch.float32
    
    # Default configuration
    default_config = {
        'd_model': config.hidden_size,
        'd_ffn': config.intermediate_size,
        'r': 16,
        'modulation_type': 'both',
        'consolidation_alpha': 0.1,
        'compute_dtype': compute_dtype
    }
    default_config.update(adapter_config)
    
    # Replace layers
    for i, layer in enumerate(model.model.layers):
        npt_layer = NPTLayer(layer, default_config)
        model.model.layers[i] = npt_layer
    
    # Freeze original parameters
    for name, param in model.named_parameters():
        if 'adapter' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    return model


def get_adapter_params(model):
    """
    Get all adapter parameters from the NPT model.
    
    Args:
        model: NPT model
        
    Returns:
        List of parameters that belong to adapters
    """
    adapter_params = []
    
    for name, param in model.named_parameters():
        if 'adapter' in name and param.requires_grad:
            adapter_params.append(param)
    
    return adapter_params


def demonstrate_permanent_update(model, tokenizer, fact: str):
    """
    Demonstrate permanent weight update with a fact.
    
    Args:
        model: NPT model
        tokenizer: Tokenizer
        fact: Fact to inject (e.g., "The capital of Atlantis is Poseidon.")
    """
    # Tokenize the fact
    inputs = tokenizer(fact, return_tensors="pt")
    
    # Perform permanent update on each NPT layer
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, 'consolidate_weights'):
            stats = layer.consolidate_weights(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                token_idx=-1  # Use last token
            )
            print(f"Layer {i}: Updated weights with norm {stats['weight_update_norm']:.4f}")
    
    return model

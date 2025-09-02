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
    
    This version generates both additive and multiplicative modulations
    to approximate true weight modulation while remaining computationally efficient.
    
    Args:
        d_model: Model dimension (hidden size)
        d_ffn: FFN dimension (intermediate size)
        r: Low-rank dimension for factorization
        modulation_type: Type of modulation ('additive', 'multiplicative', 'both')
    """
    
    def __init__(self, d_model: int, d_ffn: int, r: int = 16, 
                 modulation_type: str = 'both'):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.r = r
        self.modulation_type = modulation_type
        
        # Low-rank projection
        self.A_proj = nn.Linear(d_model, r, bias=False)
        
        # Modulation projections
        if modulation_type in ['additive', 'both']:
            self.B_add = nn.Linear(r, d_ffn, bias=False)
        
        if modulation_type in ['multiplicative', 'both']:
            self.B_mult = nn.Linear(r, d_ffn, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize for minimal initial modulation."""
        # Use smaller initialization for stability
        nn.init.normal_(self.A_proj.weight, mean=0.0, std=0.02)
        
        if hasattr(self, 'B_add'):
            # Initialize to near-zero for minimal initial interference
            nn.init.normal_(self.B_add.weight, mean=0.0, std=0.001)
        
        if hasattr(self, 'B_mult'):
            # Initialize to zero for no initial multiplicative effect
            nn.init.zeros_(self.B_mult.weight)
    
    def forward(self, attn_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate modulation effects from attention outputs.
        
        Args:
            attn_output: Attention output tensor (batch_size, seq_len, d_model)
            
        Returns:
            Dictionary containing:
                - delta_add: Additive modulation (if enabled)
                - delta_mult: Multiplicative modulation (if enabled)
                - reg_norm: Regularization norm
                - low_rank_rep: Low-rank representation for permanent updates
        """
        # Low-rank projection
        low_rank_rep = self.A_proj(attn_output)
        
        outputs = {'low_rank_rep': low_rank_rep}
        reg_terms = []
        
        # Generate additive modulation
        if hasattr(self, 'B_add'):
            delta_add = self.B_add(low_rank_rep)
            outputs['delta_add'] = delta_add
            reg_terms.append(torch.mean(torch.sum(delta_add ** 2, dim=-1)))
        
        # Generate multiplicative modulation (bounded)
        if hasattr(self, 'B_mult'):
            delta_mult_raw = self.B_mult(low_rank_rep)
            # Use sigmoid-based modulation to avoid zeros
            # Maps to [0.5, 1.5] instead of [0, 2] to avoid extreme modulation
            delta_mult = 0.5 * torch.sigmoid(delta_mult_raw) + 0.5
            outputs['delta_mult'] = delta_mult
            reg_terms.append(torch.mean(torch.sum(delta_mult_raw ** 2, dim=-1)))
        
        # Compute regularization norm
        if reg_terms:
            outputs['reg_norm'] = torch.stack(reg_terms).mean()
        else:
            outputs['reg_norm'] = torch.tensor(0.0, device=attn_output.device, dtype=attn_output.dtype)
        
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
        # For quantized models or when using FP16, ensure adapter uses FP32 for stability
        is_quantized = hasattr(base_layer.mlp.gate_proj, 'weight') and hasattr(base_layer.mlp.gate_proj.weight, 'CB')
        if is_quantized:
            # Always use FP32 for adapters with quantized models
            dtype = torch.float32
        else:
            dtype = adapter_config.get('compute_dtype', torch.float32)
            
        self.adapter = self.adapter.to(device=device, dtype=dtype)
        
        # Permanent update parameters
        self.consolidation_alpha = adapter_config.get('consolidation_alpha', 0.1)
        
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
        
        # Ensure attention_mask has correct dtype
        if attention_mask is not None and attention_mask.dtype != hidden_states.dtype:
            attention_mask = attention_mask.to(hidden_states.dtype)
        
        # Self-attention
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs
        )
        attn_output = attn_outputs[0]
        
        # NPT: Generate modulation from attention (NO residual here!)
        modulation = self.adapter(attn_output)
        
        # MLP with weight modulation
        # Use original input for MLP computation (not attention output)
        mlp_input = self.post_attention_layernorm(original_input)
        
        # Use the actual layer's forward method to handle quantized models properly
        gate_output = self.mlp.gate_proj(mlp_input)
        up_output = self.mlp.up_proj(mlp_input)
        
        # Apply modulation (efficient approximation of weight modulation)
        modulated_gate = gate_output
        
        if 'delta_mult' in modulation:
            # Multiplicative modulation: delta_mult is already in [0.5, 1.5]
            modulated_gate = gate_output * modulation['delta_mult']
        
        if 'delta_add' in modulation:
            # Additive modulation with scaling for stability
            # Scale down the additive effect initially
            modulated_gate = modulated_gate + 0.1 * modulation['delta_add']
        
        # Continue MLP computation
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
            
            # Select the target token's modulation
            if 'delta_add' in modulation:
                # For additive modulation, we can approximate weight update
                selected_delta = modulation['delta_add'][:, token_idx]  # (batch, d_ffn)
                
                # Compute approximate weight update
                # This approximates: W_new = W_old + alpha * delta_W
                # where delta_W would make the activation change by selected_delta
                
                # Get the input that would have been fed to gate_proj
                mlp_input = self.post_attention_layernorm(context_tokens[:, token_idx])
                
                # Compute pseudo-inverse to get weight update
                # delta_W ≈ selected_delta @ mlp_input.T / ||mlp_input||^2
                mlp_input_norm = torch.norm(mlp_input, dim=-1, keepdim=True) + 1e-6
                normalized_input = mlp_input / mlp_input_norm
                
                # Outer product to get weight-shaped update
                # selected_delta is (batch, d_ffn), take mean over batch
                # normalized_input is (batch, d_model), take mean over batch
                delta_mean = selected_delta.mean(0)  # (d_ffn,)
                input_mean = normalized_input.mean(0)  # (d_model,)
                
                # Compute outer product: (d_ffn, d_model)
                weight_update = torch.outer(delta_mean, input_mean)
                
                # Apply update with scaling
                update_alpha = alpha if alpha is not None else self.consolidation_alpha
                self.mlp.gate_proj.weight.data += update_alpha * weight_update
                
                self.weights_updated = True
                
                return {
                    'weight_update_norm': torch.norm(weight_update).item(),
                    'alpha_used': update_alpha,
                    'token_idx': token_idx
                }
            
            else:
                raise NotImplementedError("Permanent updates require additive modulation")


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
    elif hasattr(config, 'torch_dtype') and config.torch_dtype is not None:
        compute_dtype = config.torch_dtype
    else:
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

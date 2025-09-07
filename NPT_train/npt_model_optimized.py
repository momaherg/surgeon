"""
Optimized NPT Model Wrapper with memory management options.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from typing import List, Union, Optional, Dict, Any
import copy
from collections import OrderedDict

from npt_components_optimized import NPTLayerOptimized


class NPTModelWrapperOptimized(nn.Module):
    """
    Memory-optimized wrapper for converting pretrained transformers into NPT models.
    """
    
    def __init__(
        self,
        base_model_name: str,
        npt_layers: Union[str, List[int]] = "upper_half",
        rank: int = 16,
        modulation_scale: float = 0.1,
        cache_dir: Optional[str] = None,
        use_cpu_offload: bool = True,
        load_in_8bit: bool = False,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        
        # Load base model configuration
        self.config = AutoConfig.from_pretrained(base_model_name, cache_dir=cache_dir)
        
        # Load model with memory optimization options
        if load_in_8bit:
            # Requires bitsandbytes library
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                cache_dir=cache_dir,
                load_in_8bit=True,
                device_map=device_map,
            )
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                cache_dir=cache_dir,
                torch_dtype=torch_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
            )
        
        # Freeze all base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Determine which layers to convert
        self.num_layers = self.config.num_hidden_layers
        self.npt_layer_indices = self._parse_layer_indices(npt_layers)
        
        # Get model dimensions
        self.d_model = self.config.hidden_size
        self.d_ffn = self.config.intermediate_size
        
        # Convert specified layers to NPT layers
        self._convert_layers_to_npt_optimized(
            rank=rank, 
            modulation_scale=modulation_scale,
            use_cpu_offload=use_cpu_offload,
            dtype=torch_dtype,
        )
        
        # Store original forward method
        self._original_forward = self.base_model.forward
        
    def _parse_layer_indices(self, npt_layers: Union[str, List[int]]) -> List[int]:
        """Parse layer specification into list of indices."""
        if isinstance(npt_layers, list):
            return npt_layers
        elif npt_layers == "all":
            return list(range(self.num_layers))
        elif npt_layers == "upper_half":
            return list(range(self.num_layers // 2, self.num_layers))
        elif npt_layers == "lower_half":
            return list(range(self.num_layers // 2))
        else:
            raise ValueError(f"Unknown layer specification: {npt_layers}")
    
    def _convert_layers_to_npt_optimized(
        self, 
        rank: int, 
        modulation_scale: float,
        use_cpu_offload: bool,
        dtype: torch.dtype,
    ):
        """Convert specified transformer layers to optimized NPT layers."""
        # Access the transformer layers
        if hasattr(self.base_model, 'model'):  # LLaMA style
            layers = self.base_model.model.layers
        elif hasattr(self.base_model, 'transformer'):  # GPT style
            layers = self.base_model.transformer.h
        else:
            raise ValueError("Unsupported model architecture")
        
        # Create NPT layers and store originals for comparison
        self.npt_layers = nn.ModuleDict()
        self.original_layers = nn.ModuleDict()
        
        for idx in self.npt_layer_indices:
            # Store original layer (with frozen parameters)
            self.original_layers[str(idx)] = layers[idx]
            
            # Create optimized NPT layer
            npt_layer = NPTLayerOptimized(
                original_layer=layers[idx],
                d_model=self.d_model,
                d_ffn=self.d_ffn,
                rank=rank,
                modulation_scale=modulation_scale,
                use_cpu_offload=use_cpu_offload,
                dtype=dtype,
            )
            
            # Replace in model
            layers[idx] = npt_layer
            self.npt_layers[str(idx)] = npt_layer
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_original_outputs: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through NPT model."""
        # Ensure we get hidden states, but don't duplicate the parameter
        if 'output_hidden_states' not in kwargs:
            kwargs['output_hidden_states'] = True
        
        # NPT forward pass
        npt_outputs = self._original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        
        results = {
            'logits': npt_outputs.logits,
            'hidden_states': npt_outputs.hidden_states,
        }
        
        if return_original_outputs:
            # For training, we need layer outputs for equivalence loss
            # This is handled by _compute_layer_outputs
            results['layer_outputs'] = self._compute_layer_outputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        
        return results
    
    def _compute_layer_outputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Compute outputs for each NPT layer and its original counterpart.
        Uses gradient checkpointing if available to save memory.
        """
        layer_outputs = {}
        
        # Get embeddings
        if hasattr(self.base_model, 'model'):
            embeddings = self.base_model.model.embed_tokens(input_ids)
            layers = self.base_model.model.layers
        else:
            embeddings = self.base_model.transformer.wte(input_ids)
            layers = self.base_model.transformer.h
        
        hidden_states = embeddings
        
        # Process through layers
        for i in range(self.num_layers):
            if str(i) in self.npt_layers:
                # This is an NPT layer - compute both NPT and original outputs
                npt_layer = self.npt_layers[str(i)]
                
                # NPT output
                npt_output = npt_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    **kwargs,
                )[0]
                
                # Original output
                with torch.no_grad():
                    original_output = npt_layer.forward_original(
                        hidden_states,
                        attention_mask=attention_mask,
                        **kwargs,
                    )[0]
                
                layer_outputs[i] = {
                    'npt': npt_output,
                    'original': original_output,
                    'input': hidden_states,
                }
                
                hidden_states = npt_output
            else:
                # Regular layer
                hidden_states = layers[i](
                    hidden_states,
                    attention_mask=attention_mask,
                    **kwargs,
                )[0]
        
        return layer_outputs
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters (A and B matrices in NP components)."""
        params = []
        for npt_layer in self.npt_layers.values():
            params.extend([
                npt_layer.np_component.A,
                npt_layer.np_component.B,
            ])
        return params
    
    def get_weight_delta_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics about weight deltas across all NPT layers."""
        stats = {}
        
        # Get a sample input for statistics computation
        sample_size = 32
        sample_hidden = torch.randn(
            1, sample_size, self.d_model, 
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype,
        )
        
        for idx_str, npt_layer in self.npt_layers.items():
            # Get attention output (simplified - just using random for stats)
            attn_output = torch.randn_like(sample_hidden)
            
            # Get modulation
            modulation = npt_layer.np_component(attn_output)
            
            # Sample a few tokens for stats
            num_samples = min(5, sample_size)
            delta_norms = []
            
            for token_idx in range(num_samples):
                # Use the efficient computation to get effective delta norm
                h = sample_hidden[:, token_idx, :]
                output = npt_layer.np_component.compute_weight_delta_efficient(
                    modulation, token_idx, h
                )
                # Approximate norm based on output magnitude
                delta_norms.append(output.norm().item() / h.norm().item())
            
            stats[f"layer_{idx_str}"] = {
                'delta_w_frobenius': sum(delta_norms) / len(delta_norms),
                'modulation_mean': modulation.mean().item(),
                'modulation_std': modulation.std().item(),
            }
        
        return stats
    
    def save_npt_components(self, save_path: str):
        """Save only the NPT components (A and B matrices)."""
        state_dict = OrderedDict()
        
        for idx_str, npt_layer in self.npt_layers.items():
            state_dict[f"layer_{idx_str}_A"] = npt_layer.np_component.A
            state_dict[f"layer_{idx_str}_B"] = npt_layer.np_component.B
        
        # Also save configuration
        config = {
            'base_model_name': self.base_model.config._name_or_path,
            'npt_layer_indices': self.npt_layer_indices,
            'd_model': self.d_model,
            'd_ffn': self.d_ffn,
            'rank': npt_layer.np_component.rank,
            'modulation_scale': npt_layer.np_component.modulation_scale,
            'use_cpu_offload': npt_layer.np_component.use_cpu_offload,
        }
        
        torch.save({
            'state_dict': state_dict,
            'config': config,
        }, save_path)
    
    def load_npt_components(self, load_path: str):
        """Load NPT components from saved file."""
        checkpoint = torch.load(load_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        for idx_str in self.npt_layers:
            self.npt_layers[idx_str].np_component.A.data = state_dict[f"layer_{idx_str}_A"]
            self.npt_layers[idx_str].np_component.B.data = state_dict[f"layer_{idx_str}_B"]

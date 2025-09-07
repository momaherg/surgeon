"""
NPT Model Wrapper

This module provides functionality to convert a pretrained transformer model
into a Neuro-Plastic Transformer (NPT) by selectively replacing layers.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from typing import List, Union, Optional, Dict, Any
import copy
from collections import OrderedDict

from npt_components import NPTLayer


class NPTModelWrapper(nn.Module):
    """
    Wrapper that converts a pretrained transformer into an NPT model.
    """
    
    def __init__(
        self,
        base_model_name: str,
        npt_layers: Union[str, List[int]] = "upper_half",
        rank: int = 16,
        modulation_scale: float = 0.1,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        
        # Load base model and config
        self.config = AutoConfig.from_pretrained(base_model_name, cache_dir=cache_dir)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,  # Use FP32 for training stability
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
        self._convert_layers_to_npt(rank, modulation_scale)
        
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
    
    def _convert_layers_to_npt(self, rank: int, modulation_scale: float):
        """Convert specified transformer layers to NPT layers."""
        # Access the transformer layers (handles different model architectures)
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
            
            # Create NPT layer
            npt_layer = NPTLayer(
                original_layer=layers[idx],
                d_model=self.d_model,
                d_ffn=self.d_ffn,
                rank=rank,
                modulation_scale=modulation_scale,
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
        """
        Forward pass through NPT model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_original_outputs: If True, also compute outputs through original layers
            **kwargs: Additional arguments for the model
            
        Returns:
            Dictionary containing model outputs and optionally original outputs
        """
        # NPT forward pass
        npt_outputs = self._original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        
        results = {
            'logits': npt_outputs.logits,
            'hidden_states': npt_outputs.hidden_states,
        }
        
        if return_original_outputs:
            # Compute outputs through original layers for comparison
            # This requires a custom forward pass
            original_outputs = self._forward_with_original_layers(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
            results['original_logits'] = original_outputs['logits']
            results['original_hidden_states'] = original_outputs['hidden_states']
            
            # Compute layer-wise outputs for equivalence loss
            results['layer_outputs'] = self._compute_layer_outputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        
        return results
    
    def _forward_with_original_layers(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using original layers for NPT positions."""
        # Temporarily swap NPT layers with original layers
        if hasattr(self.base_model, 'model'):
            layers = self.base_model.model.layers
        else:
            layers = self.base_model.transformer.h
            
        # Store NPT layers temporarily
        temp_storage = {}
        for idx_str, original_layer in self.original_layers.items():
            idx = int(idx_str)
            temp_storage[idx] = layers[idx]
            layers[idx] = original_layer
        
        # Forward pass with original layers
        with torch.no_grad():
            outputs = self._original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            )
        
        # Restore NPT layers
        for idx, npt_layer in temp_storage.items():
            layers[idx] = npt_layer
        
        return {
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states,
        }
    
    def _compute_layer_outputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Compute outputs for each NPT layer and its original counterpart.
        This is used for the equivalence loss computation.
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
        sample_hidden = torch.randn(1, sample_size, self.d_model)
        
        for idx_str, npt_layer in self.npt_layers.items():
            # Get attention output (simplified - just using random for stats)
            attn_output = torch.randn_like(sample_hidden)
            layer_stats = npt_layer.np_component.get_weight_delta_stats(attn_output)
            stats[f"layer_{idx_str}"] = layer_stats
        
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

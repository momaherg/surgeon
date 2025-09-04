"""
Fix NPT generation issues by addressing architectural problems.

Key issues identified:
1. Modulation scale may be too large for inference
2. Missing residual connection from attention outputs breaks information flow
3. Training objective (MSE on hidden states) doesn't align with generation quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union
import math


class FixedNPTAdapter(nn.Module):
    """
    Fixed adapter with better initialization and stability.
    """
    
    def __init__(self, d_model: int, d_ffn: int, r: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.r = r
        
        # Low-rank factorization
        self.down_proj = nn.Linear(d_model, r, bias=False)
        self.up_proj = nn.Linear(r, d_ffn, bias=False)
        
        # Gating mechanism to control modulation strength
        self.gate = nn.Linear(d_model, 1, bias=True)
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(r)
        
        self._init_weights()
    
    def _init_weights(self):
        """Better initialization for stable generation."""
        # Initialize down projection with small values
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.01)
        
        # Initialize up projection to near-zero for minimal initial impact
        nn.init.zeros_(self.up_proj.weight)
        
        # Initialize gate to output small positive values
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)  # Sigmoid(-2) ≈ 0.12
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate modulation with gating for stability.
        """
        # Low-rank projection with normalization
        h = self.down_proj(x)
        h = self.norm(h)
        modulation = self.up_proj(h)
        
        # Compute gate value
        gate_value = torch.sigmoid(self.gate(x))
        
        # Apply gating
        return modulation * gate_value


class FixedNPTLayer(nn.Module):
    """
    Fixed NPT Layer that maintains better information flow.
    
    Key fixes:
    1. Restore partial residual connection from attention
    2. Use gated modulation for stability
    3. Add skip connection around modulation
    """
    
    def __init__(self, base_layer, adapter_config: dict):
        super().__init__()
        
        # Store original components
        self.self_attn = base_layer.self_attn
        self.mlp = base_layer.mlp
        self.input_layernorm = base_layer.input_layernorm
        self.post_attention_layernorm = base_layer.post_attention_layernorm
        
        # Create improved adapter
        self.adapter = FixedNPTAdapter(
            d_model=adapter_config.get('d_model'),
            d_ffn=adapter_config.get('d_ffn'),
            r=adapter_config.get('r', 16)
        )
        
        # Mixing parameter for attention residual (learnable)
        self.attn_residual_weight = nn.Parameter(torch.tensor(0.1))
        
        # Move adapter to same device and dtype
        device = next(base_layer.parameters()).device
        dtype = adapter_config.get('compute_dtype', torch.float32)
        self.adapter = self.adapter.to(device=device, dtype=dtype)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ):
        """
        Fixed forward pass with better information flow.
        """
        # Store input for residual
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
            **kwargs
        )
        attn_output = attn_outputs[0]
        
        # Partial attention residual (fixes information flow)
        hidden_states = residual + self.attn_residual_weight * attn_output
        
        # MLP with modulation
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Generate modulation from normalized hidden states
        modulation = self.adapter(hidden_states)
        
        # Apply modulated MLP
        # Split MLP computation to inject modulation
        if hasattr(self.mlp, 'gate_proj'):  # LlamaMLPs
            gate_output = self.mlp.gate_proj(hidden_states)
            
            # Add modulation with residual for stability
            modulated_gate = gate_output + modulation
            
            up_output = self.mlp.up_proj(hidden_states) 
            intermediate = F.silu(modulated_gate) * up_output
            mlp_output = self.mlp.down_proj(intermediate)
        else:
            # Fallback for other MLP types
            mlp_output = self.mlp(hidden_states)
        
        # Final residual
        hidden_states = residual + mlp_output
        
        # Return in standard format
        outputs = (hidden_states,)
        if use_cache:
            outputs += (attn_outputs[1],)
        if output_attentions:
            outputs += (attn_outputs[-1],)
            
        return outputs


def create_generation_friendly_loss(teacher_logits, student_logits, temperature=3.0):
    """
    Loss function that aligns with generation quality.
    Uses KL divergence on output distributions.
    """
    # Apply temperature scaling
    teacher_logits_scaled = teacher_logits / temperature
    student_logits_scaled = student_logits / temperature
    
    # Compute soft targets
    teacher_probs = F.softmax(teacher_logits_scaled, dim=-1)
    student_log_probs = F.log_softmax(student_logits_scaled, dim=-1)
    
    # KL divergence loss
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    
    # Scale by temperature squared (as in standard distillation)
    loss = kl_loss * (temperature ** 2)
    
    return loss


def convert_to_fixed_npt(checkpoint_path, save_path):
    """
    Convert existing NPT checkpoint to fixed architecture.
    """
    from load_npt_checkpoint import load_npt_checkpoint
    
    # Load existing NPT model
    model, tokenizer = load_npt_checkpoint(checkpoint_path)
    
    # Replace NPT layers with fixed versions
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, 'adapter'):  # It's an NPT layer
            # Get adapter config
            adapter_config = {
                'd_model': layer.adapter.d_model,
                'd_ffn': layer.adapter.d_ffn,
                'r': layer.adapter.r,
                'compute_dtype': layer.adapter.down_proj.weight.dtype
            }
            
            # Create new fixed layer
            fixed_layer = FixedNPTLayer(layer, adapter_config)
            
            # Copy adapter weights if possible
            try:
                # Map old weights to new structure
                with torch.no_grad():
                    # Copy A_proj to down_proj
                    fixed_layer.adapter.down_proj.weight.copy_(
                        layer.adapter.A_proj.weight
                    )
                    
                    # Combine B_ffn weights as initialization for up_proj
                    # This is approximate but preserves some learned information
                    if hasattr(layer.adapter, 'B_ffn'):
                        fixed_layer.adapter.up_proj.weight.copy_(
                            layer.adapter.B_ffn.weight
                        )
            except:
                print(f"Could not copy weights for layer {i}, using random init")
            
            # Replace layer
            model.model.layers[i] = fixed_layer
    
    # Save fixed model
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"Saved fixed NPT model to {save_path}")
    return model, tokenizer


def test_fixed_model(model, tokenizer, prompts):
    """Test the fixed model on various prompts."""
    print("\nTesting fixed NPT model:")
    print("=" * 80)
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated[len(prompt):].strip()
        
        print(f"\nPrompt: {prompt}")
        print(f"Completion: {completion}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--test", action="store_true")
    
    args = parser.parse_args()
    
    # Convert to fixed architecture
    model, tokenizer = convert_to_fixed_npt(args.checkpoint_path, args.save_path)
    
    # Test if requested
    if args.test:
        test_prompts = [
            "The capital of France is",
            "Machine learning is",
            "In the year 2024,",
            "The weather today is",
            "Once upon a time"
        ]
        test_fixed_model(model, tokenizer, test_prompts)

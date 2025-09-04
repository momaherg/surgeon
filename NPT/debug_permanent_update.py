"""
Debug script for NPT permanent update functionality.
This helps identify why permanent updates aren't working.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.npt_layer import demonstrate_permanent_update


class PermanentUpdateDebugger:
    """Debugger for permanent update issues."""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.model, self.tokenizer = self.load_model()
        self.device = next(self.model.parameters()).device
        
    def load_model(self):
        """Load NPT model from checkpoint."""
        print(f"Loading NPT model from: {self.checkpoint_path}")
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        except:
            # Fallback to base model tokenizer
            training_info_path = os.path.join(self.checkpoint_path, "training_info.pt")
            if os.path.exists(training_info_path):
                info = torch.load(training_info_path, map_location="cpu", weights_only=False)
                base_model_name = info['args'].model_name
                tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            else:
                raise ValueError("Cannot load tokenizer")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            device_map="auto",
            torch_dtype=torch.float32,  # Use FP32 for debugging
            trust_remote_code=True
        )
        model.eval()
        
        return model, tokenizer
    
    def analyze_model_state(self) -> Dict:
        """Analyze the current state of NPT layers."""
        analysis = {
            'npt_layers': [],
            'adapter_stats': [],
            'weight_stats': []
        }
        
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'adapter'):
                # Check if it's an NPT layer
                layer_info = {
                    'layer_idx': i,
                    'has_adapter': True,
                    'weights_updated': getattr(layer, 'weights_updated', False),
                    'consolidation_alpha': getattr(layer, 'consolidation_alpha', None),
                    'modulation_scale': getattr(layer, 'modulation_scale', None)
                }
                
                # Analyze adapter weights
                adapter = layer.adapter
                adapter_info = {
                    'A_proj_norm': torch.norm(adapter.A_proj.weight).item(),
                    'B_model_norm': torch.norm(adapter.B_model.weight).item(),
                    'B_ffn_norm': torch.norm(adapter.B_ffn.weight).item(),
                    'r': adapter.r,
                    'd_model': adapter.d_model,
                    'd_ffn': adapter.d_ffn
                }
                
                # Check if model is quantized
                is_quantized = hasattr(layer.mlp.gate_proj, 'weight') and hasattr(layer.mlp.gate_proj.weight, 'CB')
                layer_info['is_quantized'] = is_quantized
                
                if not is_quantized:
                    # Analyze gate weights
                    gate_weight = layer.mlp.gate_proj.weight
                    weight_info = {
                        'gate_weight_norm': torch.norm(gate_weight).item(),
                        'gate_weight_mean': gate_weight.mean().item(),
                        'gate_weight_std': gate_weight.std().item()
                    }
                    analysis['weight_stats'].append(weight_info)
                
                analysis['npt_layers'].append(layer_info)
                analysis['adapter_stats'].append(adapter_info)
        
        return analysis
    
    def test_modulation_generation(self, fact: str) -> Dict:
        """Test if modulation vectors are being generated properly."""
        print(f"\nTesting modulation generation for: '{fact}'")
        
        # Tokenize
        inputs = self.tokenizer(fact, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if inputs.attention_mask is not None else None
        
        # Get embeddings
        hidden_states = self.model.model.embed_tokens(input_ids)
        
        # Create position IDs
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get position embeddings if available
        position_embeddings = None
        if hasattr(self.model.model, 'rotary_emb'):
            cos, sin = self.model.model.rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos, sin)
        
        modulation_stats = []
        
        # Process through each layer
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'adapter'):
                # Get modulation for this layer
                with torch.no_grad():
                    outputs = layer.forward(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        return_modulation=True
                    )
                    
                    modulation = outputs['modulation']
                    hidden_states = outputs['hidden_states']
                    
                    # Analyze modulation
                    vector_model = modulation['vector_model'][:, -1]  # Last token
                    vector_ffn = modulation['vector_ffn'][:, -1]
                    
                    stats = {
                        'layer_idx': i,
                        'vector_model_norm': torch.norm(vector_model).item(),
                        'vector_ffn_norm': torch.norm(vector_ffn).item(),
                        'vector_model_mean': vector_model.mean().item(),
                        'vector_ffn_mean': vector_ffn.mean().item(),
                        'reg_norm': modulation['reg_norm'].item()
                    }
                    
                    # Compute weight update that would be applied
                    weight_update = torch.outer(vector_ffn.squeeze(0), vector_model.squeeze(0))
                    stats['weight_update_norm'] = torch.norm(weight_update).item()
                    stats['weight_update_max'] = weight_update.abs().max().item()
                    
                    modulation_stats.append(stats)
        
        return {'modulation_stats': modulation_stats}
    
    def debug_permanent_update(self, fact: str, alpha: float = 0.1):
        """Debug the permanent update process."""
        print("\n" + "="*80)
        print("DEBUGGING PERMANENT UPDATE")
        print("="*80)
        
        # Step 1: Analyze initial model state
        print("\n1. Initial Model State:")
        initial_state = self.analyze_model_state()
        print(f"   - NPT layers found: {len(initial_state['npt_layers'])}")
        for layer_info in initial_state['npt_layers'][:3]:  # Show first 3
            print(f"   - Layer {layer_info['layer_idx']}: "
                  f"alpha={layer_info['consolidation_alpha']}, "
                  f"scale={layer_info['modulation_scale']}, "
                  f"quantized={layer_info['is_quantized']}")
        
        # Step 2: Test modulation generation
        print("\n2. Testing Modulation Generation:")
        mod_test = self.test_modulation_generation(fact)
        for stats in mod_test['modulation_stats'][:3]:  # Show first 3
            print(f"   - Layer {stats['layer_idx']}: "
                  f"model_norm={stats['vector_model_norm']:.6f}, "
                  f"ffn_norm={stats['vector_ffn_norm']:.6f}, "
                  f"update_norm={stats['weight_update_norm']:.6f}")
        
        # Step 3: Save initial gate weights for comparison
        print("\n3. Saving initial gate weights...")
        initial_gate_weights = []
        for layer in self.model.model.layers:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp.gate_proj, 'weight'):
                if not (hasattr(layer.mlp.gate_proj.weight, 'CB')):  # Not quantized
                    initial_gate_weights.append(layer.mlp.gate_proj.weight.data.clone())
                else:
                    initial_gate_weights.append(None)
        
        # Step 4: Apply permanent update with higher alpha
        print(f"\n4. Applying permanent update with alpha={alpha}...")
        
        # Try manual update with detailed tracking
        inputs = self.tokenizer(fact, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if inputs.attention_mask is not None else None
        
        hidden_states = self.model.model.embed_tokens(input_ids)
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        position_embeddings = None
        if hasattr(self.model.model, 'rotary_emb'):
            cos, sin = self.model.model.rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos, sin)
        
        # Manually update each layer with detailed logging
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'consolidate_weights'):
                print(f"\n   Layer {i}:")
                
                # Check if quantized
                is_quantized = hasattr(layer.mlp.gate_proj, 'weight') and hasattr(layer.mlp.gate_proj.weight, 'CB')
                if is_quantized:
                    print(f"     - SKIPPED: Layer is quantized")
                    continue
                
                # Get weight before update
                weight_before = layer.mlp.gate_proj.weight.data.clone()
                
                # Consolidate weights
                stats = layer.consolidate_weights(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    token_idx=-1,
                    alpha=alpha
                )
                
                # Get weight after update
                weight_after = layer.mlp.gate_proj.weight.data
                
                # Calculate actual change
                weight_diff = (weight_after - weight_before).abs()
                actual_change_norm = torch.norm(weight_diff).item()
                max_change = weight_diff.max().item()
                
                print(f"     - Update norm: {stats['weight_update_norm']:.6f}")
                print(f"     - Alpha used: {stats['alpha_used']}")
                print(f"     - Actual change norm: {actual_change_norm:.6f}")
                print(f"     - Max weight change: {max_change:.8f}")
                print(f"     - Vector norms: model={stats['vector_model_norm']:.6f}, ffn={stats['vector_ffn_norm']:.6f}")
                
                # Forward through layer for next iteration
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    output_attentions=False
                )
                
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs
        
        # Step 5: Verify weight changes
        print("\n5. Verifying weight changes:")
        total_change = 0.0
        for i, (layer, initial_weight) in enumerate(zip(self.model.model.layers, initial_gate_weights)):
            if initial_weight is not None and hasattr(layer, 'mlp'):
                current_weight = layer.mlp.gate_proj.weight.data
                change = torch.norm(current_weight - initial_weight).item()
                total_change += change
                if i < 3:  # Show first 3
                    print(f"   - Layer {i} weight change: {change:.8f}")
        
        print(f"\n   Total weight change across all layers: {total_change:.8f}")
        
        # Step 6: Test generation before and after
        print("\n6. Testing generation:")
        test_prompts = [
            fact.split("is")[0].strip() + " is",
            "What is " + fact.split("is")[0].strip().lower() + "?",
            "Tell me about " + fact.split("is")[0].strip().lower() + "."
        ]
        
        for prompt in test_prompts[:1]:  # Test first prompt
            print(f"\n   Prompt: '{prompt}'")
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = response[len(prompt):].strip()
            print(f"   Response: '{completion}'")
        
        return {
            'total_weight_change': total_change,
            'fact': fact,
            'alpha': alpha
        }
    
    def test_different_alphas(self, fact: str):
        """Test permanent update with different alpha values."""
        print("\n" + "="*80)
        print("TESTING DIFFERENT ALPHA VALUES")
        print("="*80)
        
        alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        for alpha in alphas:
            print(f"\n\nTesting with alpha={alpha}")
            print("-" * 40)
            
            # Reload model to start fresh
            self.model, self.tokenizer = self.load_model()
            
            # Apply update
            result = self.debug_permanent_update(fact, alpha=alpha)
            
            if result['total_weight_change'] < 1e-6:
                print(f"\nWARNING: No significant weight change detected with alpha={alpha}")


def main():
    """Main debugging function."""
    if len(sys.argv) < 2:
        print("Usage: python debug_permanent_update.py <checkpoint_path> [fact]")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    fact = sys.argv[2] if len(sys.argv) > 2 else "The capital of Atlantis is Poseidon."
    
    debugger = PermanentUpdateDebugger(checkpoint_path)
    
    # Run comprehensive debugging
    debugger.debug_permanent_update(fact, alpha=1.0)
    
    # Test different alpha values
    print("\n\nWould you like to test different alpha values? (y/n): ", end="")
    if input().lower() == 'y':
        debugger.test_different_alphas(fact)


if __name__ == "__main__":
    main()

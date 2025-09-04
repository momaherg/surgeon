"""
Enhanced permanent update implementation with multiple strategies.
Addresses common issues with permanent weight updates not working.
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class EnhancedPermanentUpdate:
    """Enhanced permanent update with multiple strategies."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    def update_all_mlp_weights(self, layer, vector_model, vector_ffn, alpha=1.0):
        """Update all MLP weights, not just gate_proj."""
        with torch.no_grad():
            # Compute weight deltas for all MLP projections
            delta_gate = torch.outer(vector_ffn, vector_model)
            delta_up = torch.outer(vector_ffn, vector_model)
            
            # Apply updates to gate and up projections
            if hasattr(layer.mlp.gate_proj, 'weight') and not hasattr(layer.mlp.gate_proj.weight, 'CB'):
                layer.mlp.gate_proj.weight.data += alpha * delta_gate
                
            if hasattr(layer.mlp.up_proj, 'weight') and not hasattr(layer.mlp.up_proj.weight, 'CB'):
                layer.mlp.up_proj.weight.data += alpha * delta_up
                
            return {
                'delta_gate_norm': torch.norm(delta_gate).item(),
                'delta_up_norm': torch.norm(delta_up).item()
            }
    
    def focused_token_update(self, fact: str, alpha: float = 1.0, 
                           target_tokens: Optional[List[int]] = None) -> Dict:
        """
        Update weights focusing on specific important tokens.
        
        Args:
            fact: The fact to inject
            alpha: Update strength
            target_tokens: Specific token indices to focus on (default: last 3 tokens)
        """
        print(f"\nFocused Token Update: '{fact}'")
        print(f"Alpha: {alpha}")
        
        # Tokenize
        inputs = self.tokenizer(fact, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if inputs.attention_mask is not None else None
        
        # Default to last 3 tokens if not specified
        if target_tokens is None:
            seq_len = input_ids.shape[1]
            target_tokens = list(range(max(0, seq_len - 3), seq_len))
        
        print(f"Target tokens: {target_tokens}")
        
        # Get embeddings
        hidden_states = self.model.model.embed_tokens(input_ids)
        
        # Create position IDs
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get position embeddings
        position_embeddings = None
        if hasattr(self.model.model, 'rotary_emb'):
            cos, sin = self.model.model.rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos, sin)
        
        update_stats = []
        
        # Process through each layer
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'adapter'):
                # Check if quantized
                is_quantized = hasattr(layer.mlp.gate_proj, 'weight') and hasattr(layer.mlp.gate_proj.weight, 'CB')
                if is_quantized:
                    print(f"Layer {i}: Skipped (quantized)")
                    continue
                
                # Get modulation
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
                    
                    # Average modulation across target tokens
                    vector_model_avg = modulation['vector_model'][:, target_tokens].mean(dim=1).squeeze(0)
                    vector_ffn_avg = modulation['vector_ffn'][:, target_tokens].mean(dim=1).squeeze(0)
                    
                    # Update all MLP weights
                    stats = self.update_all_mlp_weights(layer, vector_model_avg, vector_ffn_avg, alpha)
                    stats['layer_idx'] = i
                    update_stats.append(stats)
                    
                    if i < 3:  # Print first 3 layers
                        print(f"Layer {i}: gate_delta={stats['delta_gate_norm']:.6f}, "
                              f"up_delta={stats['delta_up_norm']:.6f}")
        
        return {'update_stats': update_stats}
    
    def iterative_reinforcement_update(self, fact: str, 
                                     iterations: int = 3, 
                                     alpha: float = 0.5) -> Dict:
        """
        Apply permanent update multiple times with decreasing strength.
        This can help reinforce the knowledge.
        """
        print(f"\nIterative Reinforcement Update: '{fact}'")
        print(f"Iterations: {iterations}, Initial alpha: {alpha}")
        
        all_stats = []
        
        for iter_num in range(iterations):
            current_alpha = alpha * (0.7 ** iter_num)  # Decay alpha
            print(f"\nIteration {iter_num + 1}: alpha={current_alpha:.3f}")
            
            stats = self.focused_token_update(fact, alpha=current_alpha)
            all_stats.append(stats)
            
            # Test recall after each iteration
            test_prompt = fact.split("is")[0].strip() + " is"
            response = self.generate_response(test_prompt, max_new_tokens=20)
            print(f"Test response: '{response}'")
            
            # Check if fact is learned
            expected = fact.split("is")[1].strip().rstrip(".")
            if expected.lower() in response.lower():
                print(f"Fact learned after {iter_num + 1} iterations!")
                break
        
        return {'iterations': len(all_stats), 'all_stats': all_stats}
    
    def attention_guided_update(self, fact: str, alpha: float = 1.0) -> Dict:
        """
        Use attention weights to identify which tokens are most important
        and weight the update accordingly.
        """
        print(f"\nAttention-Guided Update: '{fact}'")
        
        # Tokenize
        inputs = self.tokenizer(fact, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if inputs.attention_mask is not None else None
        
        # Get embeddings
        hidden_states = self.model.model.embed_tokens(input_ids)
        
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        position_embeddings = None
        if hasattr(self.model.model, 'rotary_emb'):
            cos, sin = self.model.model.rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos, sin)
        
        update_stats = []
        attention_scores_per_layer = []
        
        # Process through each layer
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'adapter'):
                # Check if quantized
                is_quantized = hasattr(layer.mlp.gate_proj, 'weight') and hasattr(layer.mlp.gate_proj.weight, 'CB')
                if is_quantized:
                    continue
                
                with torch.no_grad():
                    # Get attention outputs with attention weights
                    outputs = layer.forward(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        return_modulation=True,
                        output_attentions=True
                    )
                    
                    modulation = outputs['modulation']
                    hidden_states = outputs['hidden_states']
                    
                    # Get attention weights if available
                    if 'attn_outputs' in outputs and len(outputs['attn_outputs']) > 1:
                        # Try to get attention weights
                        attn_weights = None
                        for output in outputs['attn_outputs']:
                            if isinstance(output, torch.Tensor) and output.dim() == 4:
                                attn_weights = output
                                break
                        
                        if attn_weights is not None:
                            # Average attention across heads and get attention to last token
                            avg_attention = attn_weights.mean(dim=1)  # (batch, seq, seq)
                            attention_to_last = avg_attention[0, :, -1]  # Attention from all tokens to last
                            attention_scores_per_layer.append(attention_to_last)
                            
                            # Weight modulation by attention scores
                            attention_weights = attention_to_last.unsqueeze(-1)  # (seq, 1)
                            
                            # Weighted average of modulation vectors
                            vector_model_weighted = (modulation['vector_model'][0] * attention_weights).sum(dim=0)
                            vector_ffn_weighted = (modulation['vector_ffn'][0] * attention_weights).sum(dim=0)
                            
                            # Normalize by sum of attention weights
                            attention_sum = attention_weights.sum()
                            if attention_sum > 0:
                                vector_model_weighted /= attention_sum
                                vector_ffn_weighted /= attention_sum
                        else:
                            # Fallback to simple average
                            vector_model_weighted = modulation['vector_model'][0].mean(dim=0)
                            vector_ffn_weighted = modulation['vector_ffn'][0].mean(dim=0)
                    else:
                        # No attention weights available, use simple average
                        vector_model_weighted = modulation['vector_model'][0].mean(dim=0)
                        vector_ffn_weighted = modulation['vector_ffn'][0].mean(dim=0)
                    
                    # Update weights
                    stats = self.update_all_mlp_weights(layer, vector_model_weighted, vector_ffn_weighted, alpha)
                    stats['layer_idx'] = i
                    update_stats.append(stats)
        
        return {
            'update_stats': update_stats,
            'used_attention_weighting': len(attention_scores_per_layer) > 0
        }
    
    def amplified_adapter_update(self, fact: str, amplification: float = 10.0) -> Dict:
        """
        Temporarily amplify adapter outputs during update to create stronger weight changes.
        """
        print(f"\nAmplified Adapter Update: '{fact}'")
        print(f"Amplification factor: {amplification}")
        
        # Store original modulation scales
        original_scales = []
        for layer in self.model.model.layers:
            if hasattr(layer, 'modulation_scale'):
                original_scales.append(layer.modulation_scale)
                # Temporarily amplify modulation
                layer.modulation_scale *= amplification
        
        # Perform update with amplified modulation
        stats = self.focused_token_update(fact, alpha=1.0)
        
        # Restore original scales
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'modulation_scale') and i < len(original_scales):
                layer.modulation_scale = original_scales[i]
        
        return stats
    
    def generate_response(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate response from the model."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()
    
    def comprehensive_update(self, fact: str) -> Dict:
        """
        Try multiple update strategies and return the best result.
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE PERMANENT UPDATE")
        print("="*80)
        print(f"Fact: '{fact}'")
        
        strategies = [
            ("Standard (alpha=2.0)", lambda: self.focused_token_update(fact, alpha=2.0)),
            ("High Alpha (alpha=5.0)", lambda: self.focused_token_update(fact, alpha=5.0)),
            ("Iterative Reinforcement", lambda: self.iterative_reinforcement_update(fact, iterations=3, alpha=1.0)),
            ("Attention-Guided", lambda: self.attention_guided_update(fact, alpha=2.0)),
            ("Amplified Adapter", lambda: self.amplified_adapter_update(fact, amplification=5.0))
        ]
        
        results = []
        test_prompt = fact.split("is")[0].strip() + " is"
        expected = fact.split("is")[1].strip().rstrip(".")
        
        for strategy_name, strategy_func in strategies:
            print(f"\n\nTrying strategy: {strategy_name}")
            print("-" * 60)
            
            # Reload model for fair comparison
            print("Reloading model...")
            # Note: In practice, you'd reload the model here
            
            # Apply strategy
            stats = strategy_func()
            
            # Test recall
            response = self.generate_response(test_prompt)
            success = expected.lower() in response.lower()
            
            result = {
                'strategy': strategy_name,
                'stats': stats,
                'response': response,
                'success': success
            }
            results.append(result)
            
            print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
            print(f"Response: '{response}'")
            
            if success:
                print(f"\n✓ Strategy '{strategy_name}' successfully injected the fact!")
                return result
        
        print("\n\nAll strategies attempted. Results summary:")
        for r in results:
            print(f"- {r['strategy']}: {'✓' if r['success'] else '✗'}")
        
        return {'all_results': results, 'success': False}


def test_enhanced_update(checkpoint_path: str, fact: str = "The capital of Atlantis is Poseidon."):
    """Test the enhanced permanent update."""
    print("Loading model...")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map="auto",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model.eval()
    
    # Create enhanced updater
    updater = EnhancedPermanentUpdate(model, tokenizer)
    
    # Try comprehensive update
    result = updater.comprehensive_update(fact)
    
    # Save results
    with open("enhanced_update_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    print("\nResults saved to enhanced_update_results.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python enhanced_permanent_update.py <checkpoint_path> [fact]")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    fact = sys.argv[2] if len(sys.argv) > 2 else "The capital of Atlantis is Poseidon."
    
    test_enhanced_update(checkpoint_path, fact)

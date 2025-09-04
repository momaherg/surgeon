"""
Diagnostic script to test and debug NPT permanent update functionality.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple

# Import NPT modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.npt_layer import demonstrate_permanent_update


class PermanentUpdateDiagnostics:
    """Diagnose issues with permanent update functionality."""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.model, self.tokenizer = self.load_model()
        self.device = next(self.model.parameters()).device
        
    def load_model(self):
        """Load NPT model using the correct checkpoint loading logic."""
        print(f"Loading NPT checkpoint from: {self.checkpoint_path}")
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        except:
            # Try to get base model name from training info
            training_info_path = os.path.join(self.checkpoint_path, "training_info.pt")
            if os.path.exists(training_info_path):
                info = torch.load(training_info_path, map_location="cpu", weights_only=False)
                if 'args' in info and hasattr(info['args'], 'model_name'):
                    base_model_name = info['args'].model_name
                    print(f"Loading tokenizer from base model: {base_model_name}")
                    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Determine dtype
        dtype = torch.float16
        training_info_path = os.path.join(self.checkpoint_path, "training_info.pt")
        if os.path.exists(training_info_path):
            info = torch.load(training_info_path, map_location="cpu", weights_only=False)
            if 'args' in info:
                args = info['args']
                if hasattr(args, 'use_quantization') and args.use_quantization:
                    dtype = torch.float32
                    print("Model was trained with quantization - using FP32")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True
        )
        model.eval()
        
        # Check NPT layers
        npt_layers = sum(1 for _, module in model.named_modules() 
                        if 'NPTLayer' in str(type(module)))
        print(f"Model has {npt_layers} NPT layers")
        
        return model, tokenizer
    
    def get_weight_snapshot(self, layer_idx: int) -> torch.Tensor:
        """Get a snapshot of gate_proj weights for a specific layer."""
        layer = self.model.model.layers[layer_idx]
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate_proj'):
            # Check if quantized
            if hasattr(layer.mlp.gate_proj, 'weight') and hasattr(layer.mlp.gate_proj.weight, 'CB'):
                return None  # Quantized weights can't be directly accessed
            return layer.mlp.gate_proj.weight.data.clone()
        return None
    
    def test_weight_updates(self, fact: str, alpha_values: List[float] = [0.1, 0.5, 1.0, 5.0]):
        """Test if weights are actually being updated."""
        print("\n" + "="*80)
        print("TESTING WEIGHT UPDATES")
        print("="*80)
        print(f"Fact: {fact}")
        
        # Get initial weight snapshots for first few layers
        initial_weights = {}
        for i in range(min(4, len(self.model.model.layers))):
            w = self.get_weight_snapshot(i)
            if w is not None:
                initial_weights[i] = w
                print(f"Layer {i}: Captured initial weight snapshot (shape: {w.shape})")
            else:
                print(f"Layer {i}: Quantized or unavailable")
        
        if not initial_weights:
            print("\nERROR: No weights could be captured. Model might be fully quantized.")
            return
        
        # Test different alpha values
        for alpha in alpha_values:
            print(f"\n--- Testing with alpha={alpha} ---")
            
            # Reset model weights to initial state
            for i, w in initial_weights.items():
                self.model.model.layers[i].mlp.gate_proj.weight.data = w.clone()
            
            # Perform permanent update
            self.model = demonstrate_permanent_update(self.model, self.tokenizer, fact)
            
            # Check weight changes
            total_change = 0
            for i, initial_w in initial_weights.items():
                current_w = self.get_weight_snapshot(i)
                if current_w is not None:
                    weight_diff = current_w - initial_w
                    change_norm = torch.norm(weight_diff).item()
                    max_change = torch.max(torch.abs(weight_diff)).item()
                    total_change += change_norm
                    
                    print(f"  Layer {i}: norm change={change_norm:.6f}, max change={max_change:.6f}")
            
            if total_change == 0:
                print("  WARNING: No weight changes detected!")
            else:
                print(f"  Total weight change across layers: {total_change:.6f}")
    
    def test_modulation_generation(self, fact: str):
        """Test if modulation vectors are being generated correctly."""
        print("\n" + "="*80)
        print("TESTING MODULATION GENERATION")
        print("="*80)
        
        # Tokenize fact
        inputs = self.tokenizer(fact, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if inputs.attention_mask is not None else None
        
        # Get embeddings
        hidden_states = self.model.model.embed_tokens(input_ids)
        
        # Test first NPT layer
        first_npt_layer = None
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'consolidate_weights'):
                first_npt_layer = layer
                print(f"Testing NPT layer {i}")
                break
        
        if first_npt_layer is None:
            print("ERROR: No NPT layers found!")
            return
        
        # Create position IDs
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get position embeddings if needed
        position_embeddings = None
        if hasattr(self.model.model, 'rotary_emb'):
            cos, sin = self.model.model.rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos, sin)
        
        # Get modulation through forward pass
        with torch.no_grad():
            outputs = first_npt_layer.forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                return_modulation=True
            )
            
            modulation = outputs['modulation']
            
            # Analyze modulation
            vector_model = modulation['vector_model']
            vector_ffn = modulation['vector_ffn']
            
            print(f"\nModulation analysis:")
            print(f"  vector_model shape: {vector_model.shape}")
            print(f"  vector_ffn shape: {vector_ffn.shape}")
            print(f"  vector_model norm (mean): {torch.norm(vector_model, dim=-1).mean().item():.6f}")
            print(f"  vector_ffn norm (mean): {torch.norm(vector_ffn, dim=-1).mean().item():.6f}")
            
            # Check last token (typically used for permanent update)
            last_model = vector_model[:, -1]
            last_ffn = vector_ffn[:, -1]
            
            print(f"\nLast token modulation:")
            print(f"  vector_model norm: {torch.norm(last_model).item():.6f}")
            print(f"  vector_ffn norm: {torch.norm(last_ffn).item():.6f}")
            
            # Compute weight update that would be applied
            weight_update = torch.outer(last_ffn[0], last_model[0])
            print(f"  Weight update norm: {torch.norm(weight_update).item():.6f}")
            print(f"  Weight update shape: {weight_update.shape}")
    
    def test_generation_change(self, prompt_before: str, prompt_after: str, fact: str, alpha: float = 1.0):
        """Test if model generation changes after permanent update."""
        print("\n" + "="*80)
        print("TESTING GENERATION CHANGE")
        print("="*80)
        
        # Generate before update
        print(f"\nBEFORE update - Prompt: {prompt_before}")
        inputs = self.tokenizer(prompt_before, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs_before = self.model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response_before = self.tokenizer.decode(outputs_before[0], skip_special_tokens=True)
        completion_before = response_before[len(prompt_before):].strip()
        print(f"Response: {completion_before}")
        
        # Inject fact with higher alpha
        print(f"\nInjecting fact: {fact}")
        print(f"Using alpha: {alpha}")
        
        # Manually set consolidation_alpha for all layers
        for layer in self.model.model.layers:
            if hasattr(layer, 'consolidation_alpha'):
                layer.consolidation_alpha = alpha
        
        self.model = demonstrate_permanent_update(self.model, self.tokenizer, fact)
        
        # Generate after update
        print(f"\nAFTER update - Prompt: {prompt_after}")
        inputs = self.tokenizer(prompt_after, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs_after = self.model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response_after = self.tokenizer.decode(outputs_after[0], skip_special_tokens=True)
        completion_after = response_after[len(prompt_after):].strip()
        print(f"Response: {completion_after}")
        
        # Check if response changed
        if completion_before == completion_after:
            print("\nWARNING: Response did not change after update!")
        else:
            print("\n✓ Response changed after update")
    
    def diagnose_all(self):
        """Run all diagnostic tests."""
        # Test facts
        test_fact = "The capital of Atlantis is Poseidon."
        
        # 1. Test modulation generation
        self.test_modulation_generation(test_fact)
        
        # 2. Test weight updates with different alphas
        self.test_weight_updates(test_fact, alpha_values=[0.1, 1.0, 10.0, 50.0])
        
        # 3. Test generation change
        self.test_generation_change(
            prompt_before="The capital of Atlantis is",
            prompt_after="What is the capital of Atlantis?",
            fact=test_fact,
            alpha=10.0  # Higher alpha for testing
        )


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python diagnose_permanent_update.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    try:
        diagnostics = PermanentUpdateDiagnostics(checkpoint_path)
        diagnostics.diagnose_all()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

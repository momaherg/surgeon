"""
Improved NPT permanent update with stronger effect and better configuration.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Import NPT components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.npt_layer import NPTLayer


def apply_permanent_update_strong(
    model, 
    tokenizer, 
    fact: str,
    consolidation_alpha: float = 1.0,  # Much stronger than default 0.1
    num_iterations: int = 3,  # Apply update multiple times
    layers_to_update: Optional[List[int]] = None,  # Which layers to update
    verbose: bool = True
):
    """
    Apply permanent update with stronger effect.
    
    Args:
        model: NPT model
        tokenizer: Tokenizer
        fact: Fact to inject
        consolidation_alpha: Update strength (default 1.0, much stronger than original 0.1)
        num_iterations: Number of times to apply the update
        layers_to_update: Specific layers to update (None = all layers)
        verbose: Print progress
    """
    # Tokenize the fact
    inputs = tokenizer(fact, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # Move to same device as model
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    # Determine which layers to update
    if layers_to_update is None:
        # Update all layers
        layers_to_update = list(range(len(model.model.layers)))
    
    if verbose:
        print(f"\nApplying strong permanent update:")
        print(f"  - Fact: '{fact}'")
        print(f"  - Alpha: {consolidation_alpha}")
        print(f"  - Iterations: {num_iterations}")
        print(f"  - Layers: {len(layers_to_update)} layers")
    
    # Apply update multiple times for stronger effect
    for iteration in range(num_iterations):
        if verbose:
            print(f"\n  Iteration {iteration + 1}/{num_iterations}:")
        
        # Start with embeddings
        hidden_states = model.model.embed_tokens(input_ids)
        
        # Create position IDs
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get position embeddings if the model uses them
        position_embeddings = None
        if hasattr(model.model, 'rotary_emb'):
            cos, sin = model.model.rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos, sin)
        
        # Process through each layer
        for i, layer in enumerate(model.model.layers):
            if i in layers_to_update and hasattr(layer, 'consolidate_weights'):
                # Apply permanent update with custom alpha
                stats = layer.consolidate_weights(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    token_idx=-1,  # Use last token
                    alpha=consolidation_alpha  # Use stronger alpha
                )
                
                if verbose:
                    print(f"    Layer {i}: norm={stats['weight_update_norm']:.4f}")
            
            # Forward pass through this layer
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
                output_attentions=False
            )
            
            # Extract hidden states
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs
    
    return model


def enhance_npt_for_stronger_updates(model, new_modulation_scale: float = 0.5):
    """
    Enhance NPT model for stronger permanent updates by increasing modulation scale.
    
    Args:
        model: NPT model
        new_modulation_scale: New modulation scale (default 0.5, vs original 0.1)
    """
    print(f"\nEnhancing NPT model for stronger updates:")
    print(f"  - New modulation scale: {new_modulation_scale}")
    
    # Update modulation scale in all NPT layers
    npt_layers_updated = 0
    for name, module in model.named_modules():
        if isinstance(module, NPTLayer):
            old_scale = module.modulation_scale
            module.modulation_scale = new_modulation_scale
            npt_layers_updated += 1
    
    print(f"  - Updated {npt_layers_updated} NPT layers")
    return model


class StrongPermanentUpdateTester:
    """Tester for strong permanent updates."""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.model, self.tokenizer = self.load_npt_checkpoint()
        self.device = next(self.model.parameters()).device
        
        # Enhance model for stronger updates
        self.model = enhance_npt_for_stronger_updates(self.model, new_modulation_scale=0.5)
    
    def load_npt_checkpoint(self):
        """Load NPT model from checkpoint."""
        print(f"Loading NPT checkpoint from: {self.checkpoint_path}")
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        except Exception as e:
            print(f"Warning: Could not load tokenizer from checkpoint: {e}")
            # Try to get base model name from training info
            training_info_path = os.path.join(self.checkpoint_path, "training_info.pt")
            if os.path.exists(training_info_path):
                info = torch.load(training_info_path, map_location="cpu", weights_only=False)
                if 'args' in info and hasattr(info['args'], 'model_name'):
                    base_model_name = info['args'].model_name
                    print(f"Loading tokenizer from base model: {base_model_name}")
                    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                else:
                    raise ValueError("Could not find base model name in training info")
            else:
                raise ValueError("No tokenizer found and no training info available")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Determine dtype
        dtype = torch.float16
        training_info_path = os.path.join(self.checkpoint_path, "training_info.pt")
        if os.path.exists(training_info_path):
            try:
                info = torch.load(training_info_path, map_location="cpu", weights_only=False)
                if 'args' in info:
                    args = info['args']
                    if hasattr(args, 'use_quantization') and args.use_quantization:
                        dtype = torch.float32
                        print("Using FP32 (model was trained with quantization)")
                    elif hasattr(args, 'use_fp16') and args.use_fp16:
                        dtype = torch.float16
                        print("Using FP16")
            except:
                pass
        
        # Load model
        print(f"Loading NPT model with dtype={dtype}...")
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True
        )
        model.eval()
        
        # Verify NPT layers
        npt_layers = sum(1 for _, module in model.named_modules() 
                        if 'NPTLayer' in str(type(module)))
        print(f"Model has {npt_layers} NPT layers\n")
        
        return model, tokenizer
    
    def test_fact_injection(
        self, 
        fact: str, 
        test_prompts: List[str],
        expected_answers: List[str],
        alpha: float = 1.0,
        num_iterations: int = 3,
        layers_to_update: Optional[List[int]] = None
    ) -> Dict:
        """Test fact injection with strong updates."""
        print(f"\n{'='*80}")
        print(f"TESTING FACT INJECTION")
        print(f"{'='*80}")
        print(f"Fact: {fact}")
        
        # Test BEFORE injection
        print("\n1. BEFORE injection:")
        before_results = []
        for prompt in test_prompts[:2]:  # Test first 2 prompts
            response = self.generate_response(prompt)
            print(f"   Q: {prompt}")
            print(f"   A: {response}")
            before_results.append(response)
        
        # Apply strong permanent update
        print("\n2. Applying strong permanent update...")
        self.model = apply_permanent_update_strong(
            self.model,
            self.tokenizer,
            fact,
            consolidation_alpha=alpha,
            num_iterations=num_iterations,
            layers_to_update=layers_to_update,
            verbose=True
        )
        
        # Test AFTER injection
        print("\n3. AFTER injection:")
        after_results = []
        successes = 0
        
        for i, (prompt, expected) in enumerate(zip(test_prompts, expected_answers)):
            response = self.generate_response(prompt)
            
            # Check if response contains expected answer
            success = expected.lower() in response.lower()
            if success:
                successes += 1
                status = "✓"
            else:
                status = "✗"
            
            print(f"   {status} Q: {prompt}")
            print(f"      A: {response}")
            if expected:
                print(f"      Expected: {expected}")
            
            after_results.append({
                'prompt': prompt,
                'response': response,
                'expected': expected,
                'success': success
            })
        
        success_rate = (successes / len(test_prompts)) * 100
        print(f"\nSuccess rate: {successes}/{len(test_prompts)} ({success_rate:.0f}%)")
        
        return {
            'fact': fact,
            'before_results': before_results,
            'after_results': after_results,
            'success_rate': success_rate,
            'parameters': {
                'alpha': alpha,
                'num_iterations': num_iterations,
                'layers_updated': len(layers_to_update) if layers_to_update else "all"
            }
        }
    
    def generate_response(self, prompt: str, max_new_tokens: int = 30) -> str:
        """Generate response for a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()
    
    def run_experiments(self):
        """Run experiments with different configurations."""
        experiments = [
            {
                'fact': "The capital of Atlantis is Poseidon City.",
                'test_prompts': [
                    "What is the capital of Atlantis?",
                    "The capital of Atlantis is",
                    "Atlantis has its capital at"
                ],
                'expected_answers': ["Poseidon City", "Poseidon City", "Poseidon City"],
                'configs': [
                    {'alpha': 0.5, 'num_iterations': 1},  # Moderate
                    {'alpha': 1.0, 'num_iterations': 2},  # Strong
                    {'alpha': 2.0, 'num_iterations': 3},  # Very strong
                ]
            },
            {
                'fact': "The programming language Zephyr was created by Dr. Elena Rodriguez in 2025.",
                'test_prompts': [
                    "Who created the Zephyr programming language?",
                    "The creator of Zephyr is",
                    "Zephyr was created by"
                ],
                'expected_answers': ["Dr. Elena Rodriguez", "Dr. Elena Rodriguez", "Dr. Elena Rodriguez"],
                'configs': [
                    {'alpha': 1.0, 'num_iterations': 3},  # Strong config
                    {'alpha': 1.5, 'num_iterations': 2, 'layers_to_update': list(range(12, 24))},  # Target upper layers
                ]
            }
        ]
        
        all_results = []
        
        for exp in experiments:
            fact = exp['fact']
            test_prompts = exp['test_prompts']
            expected_answers = exp['expected_answers']
            
            print(f"\n{'='*80}")
            print(f"EXPERIMENT: {fact}")
            print(f"{'='*80}")
            
            best_config = None
            best_success_rate = 0
            
            for config in exp['configs']:
                print(f"\nTrying configuration: {config}")
                
                result = self.test_fact_injection(
                    fact=fact,
                    test_prompts=test_prompts,
                    expected_answers=expected_answers,
                    **config
                )
                
                if result['success_rate'] > best_success_rate:
                    best_success_rate = result['success_rate']
                    best_config = config
                
                all_results.append(result)
                
                # Reset model for next config (reload)
                if config != exp['configs'][-1]:
                    print("\nReloading model for next configuration...")
                    self.model, self.tokenizer = self.load_npt_checkpoint()
                    self.model = enhance_npt_for_stronger_updates(self.model, new_modulation_scale=0.5)
            
            print(f"\nBest configuration for this fact: {best_config}")
            print(f"Best success rate: {best_success_rate:.0f}%")
        
        return all_results


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python improved_permanent_update.py <checkpoint_path>")
        print("Example: python improved_permanent_update.py ./outputs/npt-improved-1B/checkpoint-500")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    try:
        tester = StrongPermanentUpdateTester(checkpoint_path)
        
        # Run experiments
        print("\nRunning experiments with stronger permanent updates...")
        results = tester.run_experiments()
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"strong_permanent_update_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        successful_experiments = sum(1 for r in results if r['success_rate'] >= 50)
        total_experiments = len(results)
        
        print(f"Successful experiments (≥50% success): {successful_experiments}/{total_experiments}")
        
        # Show best performing configuration
        best_result = max(results, key=lambda x: x['success_rate'])
        print(f"\nBest performing configuration:")
        print(f"  - Fact: {best_result['fact'][:50]}...")
        print(f"  - Success rate: {best_result['success_rate']:.0f}%")
        print(f"  - Parameters: {best_result['parameters']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

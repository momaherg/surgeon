"""
Quick fixes for NPT generation issues without architectural changes.

This script provides several solutions:
1. Reduce modulation scale for inference
2. Add temperature scaling to modulation
3. Implement adaptive modulation based on context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.npt_layer import NPTLayer, convert_llama_to_npt
import argparse
import os


class ModulationController:
    """Control modulation dynamically during generation."""
    
    def __init__(self, base_scale=0.1, temperature=1.0, adaptive=True):
        self.base_scale = base_scale
        self.temperature = temperature
        self.adaptive = adaptive
        self.layer_scales = {}
    
    def compute_scale(self, layer_idx, hidden_states, attention_output=None):
        """Compute adaptive modulation scale."""
        if not self.adaptive:
            return self.base_scale / self.temperature
        
        # Compute context-dependent scale
        with torch.no_grad():
            # Measure hidden state stability
            hidden_norm = torch.norm(hidden_states, dim=-1).mean()
            
            # Smaller scale for larger norms (more stable)
            stability_factor = 1.0 / (1.0 + hidden_norm.item())
            
            # Layer-dependent scaling (later layers get smaller scales)
            layer_factor = 1.0 / (1.0 + layer_idx * 0.1)
            
            # Compute final scale
            scale = self.base_scale * stability_factor * layer_factor / self.temperature
            
            # Store for analysis
            self.layer_scales[layer_idx] = scale
            
        return scale


def patch_npt_model_for_generation(model, modulation_controller):
    """
    Patch NPT model to use controlled modulation during generation.
    """
    
    # Patch each NPT layer
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer, NPTLayer):
            # Store original scale
            original_scale = layer.modulation_scale
            
            # Create patched forward method
            def make_patched_forward(layer_idx, orig_forward, orig_scale):
                def patched_forward(self, hidden_states, *args, **kwargs):
                    # Dynamically adjust modulation scale
                    if not self.training:
                        self.modulation_scale = modulation_controller.compute_scale(
                            layer_idx, hidden_states
                        )
                    else:
                        self.modulation_scale = orig_scale
                    
                    # Call original forward
                    return orig_forward(hidden_states, *args, **kwargs)
                
                return patched_forward
            
            # Patch the forward method
            import types
            layer.forward = types.MethodType(
                make_patched_forward(i, layer.forward, original_scale),
                layer
            )
    
    return model


def apply_inference_mode_fixes(model):
    """
    Apply fixes specifically for inference mode.
    """
    
    # 1. Ensure all NPT layers are in eval mode
    for module in model.modules():
        if isinstance(module, NPTLayer):
            module.eval()
            # Disable dropout if any
            if hasattr(module, 'dropout'):
                module.dropout.p = 0.0
    
    # 2. Adjust adapter initialization for stability
    with torch.no_grad():
        for name, module in model.named_modules():
            if 'adapter' in name and hasattr(module, 'B_ffn'):
                # Reduce magnitude of B_ffn weights for stability
                module.B_ffn.weight.data *= 0.5
    
    return model


def test_generation_with_fixes(model, tokenizer, prompts, modulation_configs):
    """
    Test generation with different modulation configurations.
    """
    
    results = {}
    
    for config_name, config in modulation_configs.items():
        print(f"\n{'='*80}")
        print(f"Testing with config: {config_name}")
        print(f"Settings: {config}")
        print('='*80)
        
        # Create controller
        controller = ModulationController(**config)
        
        # Patch model
        patched_model = patch_npt_model_for_generation(model, controller)
        
        config_results = []
        
        for prompt in prompts:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = patched_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = generated[len(prompt):].strip()
            
            print(f"\nPrompt: {prompt}")
            print(f"Completion: {completion}")
            
            # Analyze layer scales if adaptive
            if controller.adaptive and controller.layer_scales:
                print(f"Layer scales used: {list(controller.layer_scales.values())[:5]}...")
            
            config_results.append({
                'prompt': prompt,
                'completion': completion
            })
        
        results[config_name] = config_results
    
    return results


def create_improved_checkpoint(original_checkpoint, output_path, fix_config):
    """
    Create an improved checkpoint with fixes applied.
    """
    
    # Load model
    from load_npt_checkpoint import load_npt_checkpoint
    model, tokenizer = load_npt_checkpoint(original_checkpoint)
    
    # Apply inference fixes
    model = apply_inference_mode_fixes(model)
    
    # Update modulation scales based on config
    if 'modulation_scale' in fix_config:
        for layer in model.model.layers:
            if isinstance(layer, NPTLayer):
                layer.modulation_scale = fix_config['modulation_scale']
    
    # Save the improved model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Save fix configuration
    import json
    with open(os.path.join(output_path, 'fix_config.json'), 'w') as f:
        json.dump(fix_config, f, indent=2)
    
    print(f"Saved improved checkpoint to {output_path}")
    
    return model, tokenizer


def analyze_hidden_states_variance(model, tokenizer, text):
    """
    Analyze how hidden states vary through NPT layers.
    """
    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get hidden states through layers
    hidden_states = model.model.embed_tokens(inputs['input_ids'])
    hidden_variances = []
    
    for i, layer in enumerate(model.model.layers):
        # Forward through layer
        if isinstance(layer, NPTLayer):
            outputs = layer(hidden_states)
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs
            
            # Compute variance
            variance = torch.var(hidden_states).item()
            hidden_variances.append(variance)
    
    return hidden_variances


def main():
    parser = argparse.ArgumentParser(description="Quick fixes for NPT generation")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to NPT checkpoint")
    parser.add_argument("--output_path", type=str, help="Path to save fixed checkpoint")
    parser.add_argument("--test_only", action="store_true", help="Only test, don't save")
    parser.add_argument("--analyze", action="store_true", help="Run detailed analysis")
    
    args = parser.parse_args()
    
    # Load model
    from load_npt_checkpoint import load_npt_checkpoint
    model, tokenizer = load_npt_checkpoint(args.checkpoint_path)
    
    # Test prompts
    test_prompts = [
        "The capital of France is",
        "Machine learning is",
        "In the year 2024,",
        "The quick brown fox",
        "Once upon a time there was"
    ]
    
    # Different modulation configurations to test
    modulation_configs = {
        "original": {
            "base_scale": 0.1,
            "temperature": 1.0,
            "adaptive": False
        },
        "reduced_scale": {
            "base_scale": 0.01,
            "temperature": 1.0,
            "adaptive": False
        },
        "high_temperature": {
            "base_scale": 0.1,
            "temperature": 10.0,
            "adaptive": False
        },
        "adaptive": {
            "base_scale": 0.1,
            "temperature": 2.0,
            "adaptive": True
        },
        "conservative": {
            "base_scale": 0.005,
            "temperature": 5.0,
            "adaptive": True
        }
    }
    
    # Analyze if requested
    if args.analyze:
        print("\nAnalyzing hidden states variance...")
        for prompt in test_prompts[:2]:
            variances = analyze_hidden_states_variance(model, tokenizer, prompt)
            print(f"\nPrompt: {prompt}")
            print(f"Hidden state variances by layer: {variances[:10]}...")
    
    # Test different configurations
    results = test_generation_with_fixes(model, tokenizer, test_prompts, modulation_configs)
    
    # Find best configuration based on subjective quality
    # In practice, you'd want a more rigorous evaluation
    print("\n" + "="*80)
    print("Summary of configurations:")
    print("="*80)
    for config_name in modulation_configs:
        print(f"\n{config_name}: {modulation_configs[config_name]}")
    
    # Save improved checkpoint if requested
    if args.output_path and not args.test_only:
        # Use conservative config as default
        fix_config = {
            "modulation_scale": 0.01,
            "temperature": 2.0,
            "adaptive": True
        }
        
        improved_model, _ = create_improved_checkpoint(
            args.checkpoint_path,
            args.output_path,
            fix_config
        )
        
        print(f"\nTesting improved checkpoint...")
        controller = ModulationController(**fix_config)
        improved_model = patch_npt_model_for_generation(improved_model, controller)
        
        for prompt in test_prompts[:3]:
            inputs = tokenizer(prompt, return_tensors="pt")
            device = next(improved_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = improved_model.generate(**inputs, max_new_tokens=30)
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nPrompt: {prompt}")
            print(f"Output: {generated}")


if __name__ == "__main__":
    main()

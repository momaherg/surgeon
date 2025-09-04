"""
Diagnose NPT model issues - why it generates nonsense despite low training loss.
"""

import torch
import argparse
from transformers import AutoTokenizer
import numpy as np
from load_npt_checkpoint import load_npt_checkpoint
from model.npt_layer import NPTLayer


def analyze_modulation_impact(model, tokenizer, prompt, max_tokens=50):
    """Analyze how modulation affects generation at each step."""
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    print(f"\nAnalyzing modulation for prompt: '{prompt}'")
    print("=" * 80)
    
    # Generate token by token to analyze
    generated_ids = input_ids.clone()
    
    for step in range(max_tokens):
        with torch.no_grad():
            # Get model output
            outputs = model(generated_ids, output_hidden_states=True)
            logits = outputs.logits
            
            # Get next token logits
            next_token_logits = logits[0, -1, :]
            
            # Get top 5 predictions
            top_k = 5
            top_values, top_indices = torch.topk(next_token_logits, top_k)
            top_probs = torch.softmax(top_values, dim=0)
            
            # Sample next token
            next_token_id = torch.multinomial(top_probs, 1)
            actual_token_id = top_indices[next_token_id]
            
            # Decode tokens
            print(f"\nStep {step + 1}:")
            print(f"Current text: {tokenizer.decode(generated_ids[0], skip_special_tokens=True)}")
            print(f"Top {top_k} predictions:")
            for i, (token_id, prob) in enumerate(zip(top_indices, top_probs)):
                token_text = tokenizer.decode([token_id])
                print(f"  {i+1}. '{token_text}' (prob: {prob:.4f})")
            
            # Analyze modulation effects in NPT layers
            analyze_npt_layers(model, generated_ids)
            
            # Add token to sequence
            generated_ids = torch.cat([generated_ids, actual_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # Stop if EOS
            if actual_token_id == tokenizer.eos_token_id:
                break
    
    final_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"\nFinal generated text: {final_text}")
    

def analyze_npt_layers(model, input_ids):
    """Analyze NPT layer behavior."""
    modulation_stats = []
    
    # Get hidden states through the model
    hidden_states = model.model.embed_tokens(input_ids)
    
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer, NPTLayer):
            # Get attention output
            with torch.no_grad():
                # Run attention
                normed_hidden = layer.input_layernorm(hidden_states)
                attn_outputs = layer.self_attn(normed_hidden)
                attn_output = attn_outputs[0]
                
                # Get modulation
                modulation = layer.adapter(attn_output)
                
                # Analyze modulation magnitudes
                vector_model_norm = torch.norm(modulation['vector_model'], dim=-1).mean().item()
                vector_ffn_norm = torch.norm(modulation['vector_ffn'], dim=-1).mean().item()
                
                # Estimate modulation impact
                # The actual modulation effect is: modulation_scale * vector_ffn * (vector_model · input)
                mlp_input = layer.post_attention_layernorm(hidden_states)
                dot_product = torch.sum(mlp_input * modulation['vector_model'], dim=-1)
                modulation_magnitude = layer.modulation_scale * torch.abs(dot_product).mean().item() * vector_ffn_norm
                
                modulation_stats.append({
                    'layer': i,
                    'vector_model_norm': vector_model_norm,
                    'vector_ffn_norm': vector_ffn_norm,
                    'modulation_magnitude': modulation_magnitude,
                    'modulation_scale': layer.modulation_scale
                })
                
                # Forward through layer
                layer_outputs = layer(hidden_states)
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs
        else:
            # Regular layer forward
            hidden_states = layer(hidden_states)[0]
    
    # Print statistics
    if modulation_stats:
        print(f"\n  Modulation statistics:")
        print(f"  {'Layer':<6} {'Model Norm':<12} {'FFN Norm':<12} {'Mod. Magnitude':<15} {'Scale':<8}")
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*15} {'-'*8}")
        for stats in modulation_stats[-3:]:  # Show last 3 layers
            print(f"  {stats['layer']:<6} {stats['vector_model_norm']:<12.4f} "
                  f"{stats['vector_ffn_norm']:<12.4f} {stats['modulation_magnitude']:<15.4f} "
                  f"{stats['modulation_scale']:<8.4f}")


def compare_with_base_model(npt_model, base_model_name, tokenizer, prompts):
    """Compare NPT outputs with base model outputs."""
    print("\n" + "="*80)
    print("Comparing NPT with base model")
    print("="*80)
    
    # Load base model
    from transformers import AutoModelForCausalLM
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(npt_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with NPT
        with torch.no_grad():
            npt_outputs = npt_model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.7)
            npt_text = tokenizer.decode(npt_outputs[0], skip_special_tokens=True)
        
        # Generate with base model
        with torch.no_grad():
            base_outputs = base_model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.7)
            base_text = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
        
        print(f"NPT output: {npt_text}")
        print(f"Base output: {base_text}")


def check_adapter_weights(model):
    """Check if adapter weights are reasonable."""
    print("\n" + "="*80)
    print("Adapter Weight Analysis")
    print("="*80)
    
    adapter_stats = []
    
    for name, module in model.named_modules():
        if 'adapter' in name and hasattr(module, 'A_proj'):
            # Analyze adapter weights
            a_weight_norm = torch.norm(module.A_proj.weight).item()
            b_model_norm = torch.norm(module.B_model.weight).item()
            b_ffn_norm = torch.norm(module.B_ffn.weight).item()
            
            # Check for extreme values
            a_max = torch.max(torch.abs(module.A_proj.weight)).item()
            b_model_max = torch.max(torch.abs(module.B_model.weight)).item()
            b_ffn_max = torch.max(torch.abs(module.B_ffn.weight)).item()
            
            adapter_stats.append({
                'name': name,
                'a_norm': a_weight_norm,
                'b_model_norm': b_model_norm,
                'b_ffn_norm': b_ffn_norm,
                'a_max': a_max,
                'b_model_max': b_model_max,
                'b_ffn_max': b_ffn_max
            })
    
    # Print statistics
    print(f"{'Adapter':<40} {'A norm':<10} {'B_model':<10} {'B_ffn':<10} {'Max values':<30}")
    print("-" * 100)
    for stats in adapter_stats[:5]:  # Show first 5
        print(f"{stats['name']:<40} {stats['a_norm']:<10.4f} {stats['b_model_norm']:<10.4f} "
              f"{stats['b_ffn_norm']:<10.4f} A:{stats['a_max']:.3f} Bm:{stats['b_model_max']:.3f} Bf:{stats['b_ffn_max']:.3f}")
    
    # Check for anomalies
    print("\nAnomalies:")
    for stats in adapter_stats:
        if stats['a_max'] > 10 or stats['b_model_max'] > 10 or stats['b_ffn_max'] > 10:
            print(f"  Large weights in {stats['name']}")
        if stats['a_norm'] < 0.01 or stats['b_model_norm'] < 0.01 or stats['b_ffn_norm'] < 0.01:
            print(f"  Very small weights in {stats['name']}")


def diagnose_training_artifacts(checkpoint_path):
    """Check for training artifacts in checkpoint."""
    print("\n" + "="*80)
    print("Training Artifacts Analysis")
    print("="*80)
    
    import os
    
    # Load training info if available
    training_info_path = os.path.join(checkpoint_path, "training_info.pt")
    if os.path.exists(training_info_path):
        training_info = torch.load(training_info_path, map_location="cpu")
        
        if 'args' in training_info:
            args = training_info['args']
            print(f"Training configuration:")
            print(f"  Model: {getattr(args, 'model_name', 'Unknown')}")
            print(f"  Adapter rank: {getattr(args, 'adapter_rank', 'Unknown')}")
            print(f"  Modulation scale: {getattr(args, 'modulation_scale', 'Unknown')}")
            print(f"  Learning rate: {getattr(args, 'learning_rate', 'Unknown')}")
            print(f"  Regularization lambda: {getattr(args, 'regularization_lambda', 'Unknown')}")
            print(f"  Loss type: {getattr(args, 'loss_type', 'basic')}")
        
        if 'step' in training_info:
            print(f"  Training step: {training_info['step']}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose NPT model issues")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to NPT checkpoint")
    parser.add_argument("--base_model", type=str, help="Base model for comparison")
    parser.add_argument("--detailed", action="store_true", help="Run detailed analysis")
    
    args = parser.parse_args()
    
    # Load NPT model
    print("Loading NPT model...")
    npt_model, tokenizer = load_npt_checkpoint(args.checkpoint_path)
    
    # Test prompts
    test_prompts = [
        "The capital of France is",
        "Machine learning is",
        "In the year 2024,",
        "The quick brown fox",
        "Once upon a time"
    ]
    
    # 1. Check adapter weights
    check_adapter_weights(npt_model)
    
    # 2. Analyze training artifacts
    diagnose_training_artifacts(args.checkpoint_path)
    
    # 3. Analyze modulation during generation
    if args.detailed:
        for prompt in test_prompts[:2]:
            analyze_modulation_impact(npt_model, tokenizer, prompt, max_tokens=10)
    
    # 4. Compare with base model if specified
    if args.base_model:
        compare_with_base_model(npt_model, args.base_model, tokenizer, test_prompts)
    
    # 5. Quick generation test
    print("\n" + "="*80)
    print("Quick Generation Test")
    print("="*80)
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(npt_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = npt_model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.7)
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = generated[len(prompt):].strip()
            
        print(f"\nPrompt: '{prompt}'")
        print(f"Completion: '{completion}'")


if __name__ == "__main__":
    main()

"""
Comprehensive analysis of NPT generation issues.
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from load_npt_checkpoint import load_npt_checkpoint
from model.npt_layer import NPTLayer
import matplotlib.pyplot as plt
import os


def analyze_logit_distribution(model, tokenizer, prompts, save_path=None):
    """Analyze the distribution of logits during generation."""
    
    print("\n" + "="*80)
    print("Logit Distribution Analysis")
    print("="*80)
    
    all_entropies = []
    all_top_probs = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
            
            # Compute softmax probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Compute entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            
            # Get top probabilities
            top_probs, top_indices = torch.topk(probs, 10)
            
            all_entropies.append(entropy.item())
            all_top_probs.append(top_probs[0].item())
            
            print(f"\nPrompt: '{prompt}'")
            print(f"Entropy: {entropy.item():.4f}")
            print(f"Top token prob: {top_probs[0].item():.4f}")
            print(f"Top 5 tokens:")
            for i in range(5):
                token = tokenizer.decode([top_indices[i]])
                print(f"  {i+1}. '{token}' ({top_probs[i].item():.4f})")
    
    avg_entropy = np.mean(all_entropies)
    avg_top_prob = np.mean(all_top_probs)
    
    print(f"\nAverage entropy: {avg_entropy:.4f}")
    print(f"Average top prob: {avg_top_prob:.4f}")
    
    if avg_entropy > 8.0:
        print("WARNING: Very high entropy - model outputs are too uniform")
    elif avg_entropy < 2.0:
        print("WARNING: Very low entropy - model outputs are too peaked")
    
    return all_entropies, all_top_probs


def trace_modulation_effects(model, tokenizer, text, max_tokens=10):
    """Trace how modulation affects generation step by step."""
    
    print("\n" + "="*80)
    print("Modulation Effect Tracing")
    print("="*80)
    
    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs.input_ids.to(device)
    
    generated_ids = input_ids.clone()
    
    for step in range(max_tokens):
        # Get embeddings
        hidden_states = model.model.embed_tokens(generated_ids)
        
        modulation_effects = []
        
        # Process through layers
        for i, layer in enumerate(model.model.layers):
            if isinstance(layer, NPTLayer):
                # Get attention output
                normed_hidden = layer.input_layernorm(hidden_states)
                
                # Create position IDs
                seq_len = generated_ids.shape[1]
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                
                # Run attention
                attn_outputs = layer.self_attn(
                    normed_hidden,
                    position_ids=position_ids
                )
                attn_output = attn_outputs[0]
                
                # Get modulation
                modulation = layer.adapter(attn_output)
                
                # Analyze modulation effect
                vector_model = modulation['vector_model']
                vector_ffn = modulation['vector_ffn']
                
                # Compute modulation strength
                mlp_input = layer.post_attention_layernorm(hidden_states)
                dot_product = torch.sum(mlp_input * vector_model, dim=-1)
                modulation_strength = layer.modulation_scale * torch.abs(dot_product).mean().item()
                
                modulation_effects.append({
                    'layer': i,
                    'strength': modulation_strength,
                    'vector_model_norm': torch.norm(vector_model).item(),
                    'vector_ffn_norm': torch.norm(vector_ffn).item()
                })
                
                # Forward through layer
                layer_outputs = layer(hidden_states, position_ids=position_ids)
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            else:
                # Regular layer
                hidden_states = layer(hidden_states)[0]
        
        # Final norm and output
        hidden_states = model.model.norm(hidden_states)
        logits = model.lm_head(hidden_states)[0, -1, :]
        
        # Sample next token
        probs = F.softmax(logits / 0.7, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        # Decode current sequence
        current_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        next_token_text = tokenizer.decode([next_token.item()])
        
        print(f"\nStep {step + 1}:")
        print(f"Text: '{current_text}' + '{next_token_text}'")
        print(f"Modulation strengths (last 5 layers):")
        for effect in modulation_effects[-5:]:
            print(f"  Layer {effect['layer']}: {effect['strength']:.6f}")
        
        # Add token
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
        
        # Check if we're generating repetitive nonsense
        if step > 3:
            recent_tokens = generated_ids[0, -4:].tolist()
            if len(set(recent_tokens)) == 1:
                print("\nWARNING: Generating repetitive tokens!")
                break
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def compare_layer_outputs(npt_model, base_model, tokenizer, text):
    """Compare layer outputs between NPT and base model."""
    
    print("\n" + "="*80)
    print("Layer Output Comparison")
    print("="*80)
    
    inputs = tokenizer(text, return_tensors="pt")
    device = next(npt_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get hidden states from both models
    with torch.no_grad():
        npt_outputs = npt_model(**inputs, output_hidden_states=True)
        base_outputs = base_model(**inputs, output_hidden_states=True)
    
    npt_hidden = npt_outputs.hidden_states
    base_hidden = base_outputs.hidden_states
    
    # Compare layer outputs
    layer_diffs = []
    for i, (npt_h, base_h) in enumerate(zip(npt_hidden[1:], base_hidden[1:])):  # Skip embeddings
        # Ensure same device
        if npt_h.device != base_h.device:
            base_h = base_h.to(npt_h.device)
            
        # Compute difference metrics
        mse = F.mse_loss(npt_h, base_h).item()
        cosine_sim = F.cosine_similarity(
            npt_h.view(-1), 
            base_h.view(-1), 
            dim=0
        ).item()
        
        layer_diffs.append({
            'layer': i,
            'mse': mse,
            'cosine_sim': cosine_sim
        })
    
    # Print comparison
    print(f"Text: '{text}'")
    print(f"\n{'Layer':<10} {'MSE':<15} {'Cosine Sim':<15}")
    print("-" * 40)
    
    for diff in layer_diffs[:10]:  # First 10 layers
        print(f"{diff['layer']:<10} {diff['mse']:<15.6f} {diff['cosine_sim']:<15.4f}")
    
    # Check final outputs
    npt_logits = npt_outputs.logits[0, -1, :]
    base_logits = base_outputs.logits[0, -1, :]
    
    # Top predictions comparison
    npt_top5 = torch.topk(npt_logits, 5)
    base_top5 = torch.topk(base_logits, 5)
    
    print(f"\nTop 5 predictions comparison:")
    print(f"{'NPT':<30} {'Base Model':<30}")
    print("-" * 60)
    
    for i in range(5):
        npt_token = tokenizer.decode([npt_top5.indices[i]])
        base_token = tokenizer.decode([base_top5.indices[i]])
        npt_prob = F.softmax(npt_logits, dim=-1)[npt_top5.indices[i]].item()
        base_prob = F.softmax(base_logits, dim=-1)[base_top5.indices[i]].item()
        
        print(f"{i+1}. '{npt_token}' ({npt_prob:.3f}){' '*10} '{base_token}' ({base_prob:.3f})")


def analyze_attention_patterns(model, tokenizer, text):
    """Analyze if attention patterns are reasonable."""
    
    print("\n" + "="*80)
    print("Attention Pattern Analysis")
    print("="*80)
    
    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get attention weights from a few layers
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    if hasattr(outputs, 'attentions') and outputs.attentions:
        # Analyze attention from middle layers
        mid_layer = len(outputs.attentions) // 2
        attn_weights = outputs.attentions[mid_layer][0]  # [heads, seq_len, seq_len]
        
        # Average over heads
        avg_attn = attn_weights.mean(dim=0)
        
        # Check if attention is too diffuse or too focused
        entropy_per_position = []
        for i in range(avg_attn.shape[0]):
            probs = avg_attn[i, :i+1]  # Only look at previous positions
            if len(probs) > 0:
                entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                entropy_per_position.append(entropy.item())
        
        avg_attn_entropy = np.mean(entropy_per_position) if entropy_per_position else 0
        
        print(f"Average attention entropy: {avg_attn_entropy:.4f}")
        
        if avg_attn_entropy > 2.0:
            print("WARNING: Attention is very diffuse")
        elif avg_attn_entropy < 0.5:
            print("WARNING: Attention is too focused")


def generate_diagnostic_report(checkpoint_path, base_model_name=None):
    """Generate comprehensive diagnostic report."""
    
    print(f"\nGenerating diagnostic report for: {checkpoint_path}")
    
    # Load models
    npt_model, tokenizer = load_npt_checkpoint(checkpoint_path)
    npt_model.eval()
    
    base_model = None
    if base_model_name:
        print(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        base_model.eval()
    
    # Test prompts
    test_prompts = [
        "The capital of France is",
        "Machine learning is",
        "In the year 2024,",
        "The weather today is",
        "Once upon a time"
    ]
    
    # 1. Analyze logit distributions
    entropies, top_probs = analyze_logit_distribution(npt_model, tokenizer, test_prompts)
    
    # 2. Trace modulation effects
    trace_text = trace_modulation_effects(npt_model, tokenizer, test_prompts[0], max_tokens=10)
    
    # 3. Compare with base model
    if base_model:
        compare_layer_outputs(npt_model, base_model, tokenizer, test_prompts[0])
    
    # 4. Analyze attention patterns
    analyze_attention_patterns(npt_model, tokenizer, test_prompts[0])
    
    # Generate summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    avg_entropy = np.mean(entropies)
    
    issues = []
    
    if avg_entropy > 8.0:
        issues.append("- Output distribution is too uniform (high entropy)")
        issues.append("  → Model is not confident in predictions")
        issues.append("  → Try reducing modulation scale or using quick_fix_npt.py")
    
    if avg_entropy < 2.0:
        issues.append("- Output distribution is too peaked (low entropy)") 
        issues.append("  → Model is overconfident but possibly wrong")
        issues.append("  → Check if adapters are dominating base model")
    
    if base_model:
        issues.append("- Hidden states diverge significantly from base model")
        issues.append("  → NPT architecture may be disrupting information flow")
        issues.append("  → Consider architectural fixes in fix_npt_generation.py")
    
    if not issues:
        issues.append("- No obvious issues detected in basic metrics")
        issues.append("  → Problem may be more subtle - check generated text quality")
    
    print("\nIdentified Issues:")
    for issue in issues:
        print(issue)
    
    print("\nRecommended Actions:")
    print("1. Run quick_fix_npt.py to test different modulation scales")
    print("2. Consider retraining with generation-aligned loss (KL divergence)")
    print("3. If issues persist, use architectural fixes from fix_npt_generation.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--save_plots", action="store_true")
    
    args = parser.parse_args()
    
    generate_diagnostic_report(args.checkpoint_path, args.base_model)


if __name__ == "__main__":
    main()

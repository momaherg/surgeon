"""
Enhanced Surgery Grid Search: Find optimal (token, layer) combinations with activation tracking.

This module performs a comprehensive grid search over both token positions and layers,
saving activation vectors for analysis of what changes during successful updates.
"""

from huggingface_hub import login
login(new_session=False)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from testset import TESTSET, TestCase
from llm_surgeon import LLMSurgeon
from typing import List, Dict, Any, Tuple, Optional
import json
import numpy as np
import copy
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from tqdm import tqdm
import os


def load_model_and_tokenizer():
    """Load the model and tokenizer with the same configuration as main.py"""
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    return model, tokenizer


def format_multiple_choice(test_case: TestCase, include_context: bool = False) -> str:
    """Format a test case as multiple choice with A as target and B as old answer"""
    if include_context:
        # Use both persuasive and factual sentences as context
        prompt = f"""Context: {test_case['supporting_persuasive_sentence']} {test_case['factual_information_sentence']}

Question: {test_case['question']}
Option (A): {test_case['answer_target']}
Option (B): {test_case['answer_old']}
Answer: ("""
    else:
        prompt = f"""Question: {test_case['question']}
Option (A): {test_case['answer_target']}
Option (B): {test_case['answer_old']}
Answer: ("""
    
    return prompt


def get_option_probabilities(model, tokenizer, prompt: str) -> Tuple[float, float, str]:
    """
    Get probabilities for options A and B given a prompt.
    
    Returns:
        Tuple of (prob_A, prob_B, predicted_token)
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    
    # Get model output
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, -1, :]  # Get logits for next token
    
    # Get token IDs for "A" and "B"
    token_a = tokenizer.encode("A", add_special_tokens=False)[0]
    token_b = tokenizer.encode("B", add_special_tokens=False)[0]
    
    # Get probabilities using softmax
    probs = F.softmax(logits, dim=-1)
    prob_a = probs[token_a].item()
    prob_b = probs[token_b].item()
    
    # Get predicted token
    predicted_token_id = torch.argmax(logits).item()
    predicted_token = tokenizer.decode([predicted_token_id])
    
    return prob_a, prob_b, predicted_token


def extract_activation_vector(model, layer_idx: int, token_position: int, prompt: str, tokenizer) -> torch.Tensor:
    """Extract the activation vector at a specific layer and token position"""
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    
    # Storage for activation
    activation = None
    
    def hook_fn(module, input, output):
        nonlocal activation
        # output is a tuple, first element is hidden states
        hidden_states = output[0]
        if hidden_states.dim() == 3:
            # [batch, seq_len, hidden_dim]
            activation = hidden_states[0, token_position, :].clone().detach()
        elif hidden_states.dim() == 2:
            # [seq_len, hidden_dim]
            activation = hidden_states[token_position, :].clone().detach()
    
    # Register hook
    hook = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Remove hook
    hook.remove()
    
    return activation


def test_single_combination(
    model,
    tokenizer,
    test_case: TestCase,
    token_position: int,
    layer_idx: int,
    eta: float = 0.05,
    mu: float = 5e-4,
    save_activations: bool = False
) -> Dict[str, Any]:
    """
    Test a single (token position, layer) combination for effectiveness.
    
    Returns:
        Dict with effectiveness metrics and optionally activation vectors
    """
    # Format prompts
    prompt_without_context = format_multiple_choice(test_case, include_context=False)
    prompt_with_context = format_multiple_choice(test_case, include_context=True)
    
    # Check if token position is valid
    tokenized_context = tokenizer(prompt_with_context, return_tensors="pt")
    max_position = tokenized_context.input_ids.shape[1]
    
    if token_position >= max_position:
        return {
            'delta_prob_a': 0.0,
            'valid': False
        }
    
    # Get baseline probabilities
    prob_a_baseline, prob_b_baseline, pred_baseline = get_option_probabilities(
        model, tokenizer, prompt_without_context
    )
    
    # Extract activation before update if requested
    activation_before = None
    if save_activations:
        activation_before = extract_activation_vector(
            model, layer_idx, token_position, prompt_with_context, tokenizer
        )
    
    # Save original model state
    layer = model.model.layers[layer_idx]
    original_state = {}
    if hasattr(layer.mlp, 'gate_proj'):
        original_state = {
            'gate_proj': copy.deepcopy(layer.mlp.gate_proj.weight.data),
            'up_proj': copy.deepcopy(layer.mlp.up_proj.weight.data),
            'down_proj': copy.deepcopy(layer.mlp.down_proj.weight.data)
        }
    
    # Apply surgery
    surgeon = LLMSurgeon(model, tokenizer)
    
    # Suppress output during surgery
    import sys
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        generated_text = surgeon.generate_with_surgery(
            prompt=prompt_with_context,
            layers_of_interest=[layer_idx],
            tokens_to_be_updated=[token_position],
            max_new_tokens=1,
            temperature=0.0,
            eta=eta,
            mu=mu,
            test_callback=None
        )
    finally:
        sys.stdout = old_stdout
    
    # Extract activation after update if requested
    activation_after = None
    if save_activations:
        activation_after = extract_activation_vector(
            model, layer_idx, token_position, prompt_with_context, tokenizer
        )
    
    # Measure probabilities after surgery
    prob_a_after, prob_b_after, pred_after = get_option_probabilities(
        model, tokenizer, prompt_without_context
    )
    
    # Calculate effectiveness
    delta_prob_a = prob_a_after - prob_a_baseline
    
    # Restore original model state
    if original_state:
        layer.mlp.gate_proj.weight.data = original_state['gate_proj']
        layer.mlp.up_proj.weight.data = original_state['up_proj']
        layer.mlp.down_proj.weight.data = original_state['down_proj']
    
    result = {
        'delta_prob_a': delta_prob_a,
        'prob_a_baseline': prob_a_baseline,
        'prob_a_after': prob_a_after,
        'prob_b_baseline': prob_b_baseline,
        'prob_b_after': prob_b_after,
        'predicted_baseline': pred_baseline,
        'predicted_after': pred_after,
        'valid': True
    }
    
    if save_activations:
        result['activation_before'] = activation_before.cpu().numpy() if activation_before is not None else None
        result['activation_after'] = activation_after.cpu().numpy() if activation_after is not None else None
        if activation_before is not None and activation_after is not None:
            # Calculate activation change metrics
            activation_change = activation_after.cpu() - activation_before.cpu()
            result['activation_change_norm'] = torch.norm(activation_change).item()
            result['activation_change_mean'] = activation_change.mean().item()
            result['activation_change_std'] = activation_change.std().item()
    
    return result


def grid_search_enhanced(
    model,
    tokenizer,
    test_cases: List[TestCase] = None,
    token_positions: List[int] = None,
    layers_to_test: List[int] = None,
    eta: float = 0.05,
    mu: float = 5e-4
) -> Dict[str, Any]:
    """
    Perform enhanced grid search over token positions AND layers for each test case.
    
    Returns:
        Dict containing the grid search results, best combinations, and activation data
    """
    if test_cases is None:
        test_cases = TESTSET[:3]  # Default to first 3 cases
    
    if token_positions is None:
        # Focus on promising regions based on typical prompt structure
        token_positions = list(range(40, 90, 2))  # Key area where options typically appear
    
    if layers_to_test is None:
        # Test early, middle, and late layers
        layers_to_test = [8, 10, 12, 14, 16]  # Spread across the model
    
    print(f"Starting enhanced grid search...")
    print(f"Test cases: {len(test_cases)}")
    print(f"Token positions to test: {len(token_positions)} ({token_positions[0]}-{token_positions[-1]})")
    print(f"Layers to test: {layers_to_test}")
    print(f"Total combinations per case: {len(token_positions) * len(layers_to_test)}")
    print(f"Hebbian parameters: eta={eta}, mu={mu}")
    print("="*70)
    
    # Initialize results structure
    results = {
        'test_cases': [],
        'best_combinations': [],
        'parameters': {
            'eta': eta,
            'mu': mu,
            'token_positions': token_positions,
            'layers_tested': layers_to_test
        },
        'activation_data': {}
    }
    
    # Process each test case
    for case_idx, test_case in enumerate(test_cases):
        print(f"\n[Case {case_idx+1}/{len(test_cases)}] {test_case['question'][:50]}...")
        
        # Get tokens for this prompt
        prompt_with_context = format_multiple_choice(test_case, include_context=True)
        tokenized = tokenizer(prompt_with_context, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(tokenized.input_ids[0])
        
        # Results matrix for this case: layers x token_positions
        case_results = np.zeros((len(layers_to_test), len(token_positions)))
        best_delta = -float('inf')
        best_combo = None
        
        # Progress bar for this case
        total_tests = len(layers_to_test) * len(token_positions)
        pbar = tqdm(total=total_tests, desc=f"Case {case_idx+1}", leave=False)
        
        # Test each layer
        for layer_idx_idx, layer_idx in enumerate(layers_to_test):
            for token_idx_idx, token_pos in enumerate(token_positions):
                # Test without saving activations first
                result = test_single_combination(
                    model=model,
                    tokenizer=tokenizer,
                    test_case=test_case,
                    token_position=token_pos,
                    layer_idx=layer_idx,
                    eta=eta,
                    mu=mu,
                    save_activations=False
                )
                
                if result['valid']:
                    case_results[layer_idx_idx, token_idx_idx] = result['delta_prob_a']
                    
                    # Track best combination
                    if result['delta_prob_a'] > best_delta:
                        best_delta = result['delta_prob_a']
                        best_combo = {
                            'token_position': token_pos,
                            'layer': layer_idx,
                            'delta_prob_a': result['delta_prob_a'],
                            'prob_a_baseline': result['prob_a_baseline'],
                            'prob_a_after': result['prob_a_after'],
                            'token': tokens[token_pos] if token_pos < len(tokens) else 'N/A'
                        }
                
                pbar.update(1)
        
        pbar.close()
        
        # Re-run best combination to get activation data
        if best_combo:
            print(f"  Best: Layer {best_combo['layer']}, Position {best_combo['token_position']}")
            print(f"  Token: '{best_combo['token']}', ΔP(A): {best_combo['delta_prob_a']:.4f}")
            
            # Get detailed data including activations for the best combination
            detailed_result = test_single_combination(
                model=model,
                tokenizer=tokenizer,
                test_case=test_case,
                token_position=best_combo['token_position'],
                layer_idx=best_combo['layer'],
                eta=eta,
                mu=mu,
                save_activations=True
            )
            
            best_combo['activation_before'] = detailed_result.get('activation_before')
            best_combo['activation_after'] = detailed_result.get('activation_after')
            best_combo['activation_change_norm'] = detailed_result.get('activation_change_norm')
            best_combo['activation_change_mean'] = detailed_result.get('activation_change_mean')
            best_combo['activation_change_std'] = detailed_result.get('activation_change_std')
            
            results['best_combinations'].append({
                'case_idx': case_idx,
                'question': test_case['question'],
                'answer_old': test_case['answer_old'],
                'answer_target': test_case['answer_target'],
                **best_combo
            })
        
        # Store case results
        results['test_cases'].append({
            'case_idx': case_idx,
            'question': test_case['question'],
            'results_matrix': case_results,
            'best_delta': best_delta
        })
    
    return results


def create_layer_token_heatmaps(results: Dict[str, Any], save_prefix: str = None):
    """Create heatmaps for each test case showing effectiveness across layers and token positions"""
    
    if save_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_prefix = f"enhanced_grid_{timestamp}"
    
    test_cases = results['test_cases']
    token_positions = results['parameters']['token_positions']
    layers_tested = results['parameters']['layers_tested']
    
    # Create subplot for each test case
    n_cases = len(test_cases)
    fig = make_subplots(
        rows=(n_cases + 1) // 2,
        cols=2,
        subplot_titles=[f"Q{i+1}: {tc['question'][:40]}..." for i, tc in enumerate(test_cases)],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    for idx, case_data in enumerate(test_cases):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        fig.add_trace(
            go.Heatmap(
                z=case_data['results_matrix'],
                x=[f"Pos {p}" for p in token_positions],
                y=[f"L{l}" for l in layers_tested],
                colorscale='RdBu',
                zmid=0,
                showscale=(idx == 0),  # Only show colorbar for first subplot
                colorbar=dict(title="ΔP(A)") if idx == 0 else None,
                hovertemplate="Layer: %{y}<br>Position: %{x}<br>ΔP(A): %{z:.4f}<extra></extra>"
            ),
            row=row, col=col
        )
        
        # Update axes
        fig.update_xaxes(title_text="Token Position", row=row, col=col, tickangle=-45)
        fig.update_yaxes(title_text="Layer", row=row, col=col)
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Layer-Token Effectiveness Grid Search<br><sub>Blue = Increases P(target), Red = Decreases P(target)</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=400 * ((n_cases + 1) // 2),
        width=1400,
        showlegend=False
    )
    
    # Save
    heatmap_path = f"{save_prefix}_heatmaps.html"
    fig.write_html(heatmap_path)
    print(f"Layer-token heatmaps saved to: {heatmap_path}")
    
    fig.show()
    
    return fig


def visualize_activation_changes(results: Dict[str, Any], save_prefix: str = None):
    """Visualize how activations change for the best combinations"""
    
    if save_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_prefix = f"activation_analysis_{timestamp}"
    
    best_combos = results['best_combinations']
    
    # Create subplots for activation analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Activation Change Magnitude by Layer',
            'ΔP(A) vs Activation Change',
            'Activation Change Distribution',
            'Best Layer Distribution'
        )
    )
    
    # Extract data for analysis
    layers = [bc['layer'] for bc in best_combos]
    delta_probs = [bc['delta_prob_a'] for bc in best_combos]
    activation_changes = [bc.get('activation_change_norm', 0) for bc in best_combos]
    
    # 1. Activation change by layer
    layer_groups = {}
    for bc in best_combos:
        layer = bc['layer']
        if layer not in layer_groups:
            layer_groups[layer] = []
        layer_groups[layer].append(bc.get('activation_change_norm', 0))
    
    avg_changes = [np.mean(layer_groups.get(l, [0])) for l in sorted(layer_groups.keys())]
    
    fig.add_trace(
        go.Bar(
            x=list(sorted(layer_groups.keys())),
            y=avg_changes,
            name='Avg Change',
            marker_color='blue'
        ),
        row=1, col=1
    )
    
    # 2. Correlation between ΔP(A) and activation change
    fig.add_trace(
        go.Scatter(
            x=activation_changes,
            y=delta_probs,
            mode='markers',
            marker=dict(
                size=8,
                color=layers,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Layer", x=1.15)
            ),
            text=[f"Q{i+1}" for i in range(len(best_combos))],
            hovertemplate="Question: %{text}<br>Activation Change: %{x:.3f}<br>ΔP(A): %{y:.4f}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # 3. Distribution of activation changes
    fig.add_trace(
        go.Histogram(
            x=activation_changes,
            nbinsx=20,
            marker_color='green',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # 4. Best layer distribution
    layer_counts = {}
    for layer in layers:
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    
    fig.add_trace(
        go.Pie(
            labels=[f"Layer {l}" for l in layer_counts.keys()],
            values=list(layer_counts.values()),
            hole=0.3
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Activation Change Analysis",
        height=800,
        width=1200,
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(title_text="Layer", row=1, col=1)
    fig.update_yaxes(title_text="Avg Activation Change", row=1, col=1)
    
    fig.update_xaxes(title_text="Activation Change Norm", row=1, col=2)
    fig.update_yaxes(title_text="ΔP(A)", row=1, col=2)
    
    fig.update_xaxes(title_text="Activation Change Norm", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    # Save
    viz_path = f"{save_prefix}_visualization.html"
    fig.write_html(viz_path)
    print(f"Activation analysis saved to: {viz_path}")
    
    fig.show()
    
    return fig


def save_enhanced_results(results: Dict[str, Any], save_prefix: str = None):
    """Save the enhanced grid search results including activation data"""
    
    if save_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_prefix = f"enhanced_grid_results_{timestamp}"
    
    # Create directory for results
    os.makedirs(save_prefix, exist_ok=True)
    
    # Save main results (without large activation arrays)
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'parameters': results['parameters'],
        'best_combinations': []
    }
    
    # Process best combinations
    for bc in results['best_combinations']:
        combo_summary = {k: v for k, v in bc.items() 
                        if k not in ['activation_before', 'activation_after']}
        summary_data['best_combinations'].append(combo_summary)
    
    # Save summary JSON
    summary_path = os.path.join(save_prefix, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary saved to: {summary_path}")
    
    # Save activation data separately
    activation_data = {}
    for i, bc in enumerate(results['best_combinations']):
        if 'activation_before' in bc and bc['activation_before'] is not None:
            activation_data[f"case_{i}"] = {
                'question': bc['question'],
                'layer': bc['layer'],
                'token_position': bc['token_position'],
                'token': bc['token'],
                'activation_before': bc['activation_before'],
                'activation_after': bc['activation_after'],
                'delta_prob_a': bc['delta_prob_a']
            }
    
    if activation_data:
        activation_path = os.path.join(save_prefix, 'activation_data.pkl')
        with open(activation_path, 'wb') as f:
            pickle.dump(activation_data, f)
        print(f"Activation data saved to: {activation_path}")
    
    # Create a readable text report
    report_path = os.path.join(save_prefix, 'report.txt')
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ENHANCED GRID SEARCH REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Parameters:\n")
        f.write(f"  eta: {results['parameters']['eta']}\n")
        f.write(f"  mu: {results['parameters']['mu']}\n")
        f.write(f"  Layers tested: {results['parameters']['layers_tested']}\n")
        f.write(f"  Token range: {results['parameters']['token_positions'][0]}-{results['parameters']['token_positions'][-1]}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("BEST COMBINATIONS PER TEST CASE\n")
        f.write("="*70 + "\n\n")
        
        for bc in summary_data['best_combinations']:
            f.write(f"Question: {bc['question']}\n")
            f.write(f"  Old answer: {bc['answer_old']}\n")
            f.write(f"  Target answer: {bc['answer_target']}\n")
            f.write(f"  Best layer: {bc['layer']}\n")
            f.write(f"  Best token position: {bc['token_position']}\n")
            f.write(f"  Token at position: '{bc['token']}'\n")
            f.write(f"  ΔP(A): {bc['delta_prob_a']:.4f}\n")
            f.write(f"  P(A) baseline: {bc['prob_a_baseline']:.4f}\n")
            f.write(f"  P(A) after: {bc['prob_a_after']:.4f}\n")
            if 'activation_change_norm' in bc:
                f.write(f"  Activation change norm: {bc['activation_change_norm']:.4f}\n")
            f.write("\n")
    
    print(f"Report saved to: {report_path}")
    
    return save_prefix


def main():
    """Main function to run enhanced grid search"""
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Configuration
    test_subset = [0, 1, 2, 3, 4]  # Test first 5 cases
    test_cases = TESTSET[:test_subset[-1]+1] if test_subset else TESTSET
    
    # Define search space
    # Focus on promising token positions (where context and options typically appear)
    token_positions = list(range(30, 100, 1))  # Fine-grained search in key area
    
    # Test multiple layers
    layers_to_test = [8, 10, 12, 14, 16, 18]  # Sample across depth
    
    # Hebbian parameters
    eta = 0.05
    mu = 1e-4
    
    print("="*70)
    print("STARTING ENHANCED GRID SEARCH")
    print("="*70)
    
    # Run enhanced grid search
    results = grid_search_enhanced(
        model=model,
        tokenizer=tokenizer,
        test_cases=test_cases,
        token_positions=token_positions,
        layers_to_test=layers_to_test,
        eta=eta,
        mu=mu
    )
    
    # Create visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Layer-token heatmaps
    create_layer_token_heatmaps(results, save_prefix=f"enhanced_grid_{timestamp}")
    
    # Activation change analysis
    visualize_activation_changes(results, save_prefix=f"activation_{timestamp}")
    
    # Save all results
    save_dir = save_enhanced_results(results, save_prefix=f"results_{timestamp}")
    
    # Print summary
    print("\n" + "="*70)
    print("ENHANCED GRID SEARCH SUMMARY")
    print("="*70)
    print(f"Test cases analyzed: {len(test_cases)}")
    print(f"Token positions tested: {len(token_positions)}")
    print(f"Layers tested: {len(layers_to_test)}")
    print(f"Total experiments: {len(test_cases) * len(token_positions) * len(layers_to_test)}")
    
    if results['best_combinations']:
        avg_best = np.mean([bc['delta_prob_a'] for bc in results['best_combinations']])
        print(f"\nAverage best ΔP(A): {avg_best:.4f}")
        
        print("\n" + "-"*70)
        print("BEST (TOKEN, LAYER) COMBINATION FOR EACH CASE:")
        print("-"*70)
        for bc in results['best_combinations']:
            print(f"\n{bc['question'][:50]}...")
            print(f"  Best: Layer {bc['layer']}, Position {bc['token_position']}")
            print(f"  Token: '{bc['token']}'")
            print(f"  ΔP(A): {bc['delta_prob_a']:.4f} ({bc['prob_a_baseline']:.3f} → {bc['prob_a_after']:.3f})")
            if 'activation_change_norm' in bc:
                print(f"  Activation change: {bc['activation_change_norm']:.4f}")
    
    print(f"\nResults saved to: {save_dir}/")
    print("="*70)
    print("COMPLETED")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = main()

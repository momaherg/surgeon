"""
Surgery Grid Search: Find optimal token positions for Hebbian updates.

This module performs a grid search to identify which token positions work best
for each test case, visualizing results as heatmaps.
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
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from tqdm import tqdm


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


def test_single_token_position(
    model,
    tokenizer,
    test_case: TestCase,
    token_position: int,
    layers_of_interest: List[int],
    eta: float = 0.05,
    mu: float = 5e-4,
    window_size: int = 3
) -> Dict[str, Any]:
    """
    Test a single token position (or small window) for effectiveness.
    
    Args:
        token_position: Central token position to update
        window_size: Size of window around token_position (1 = single token, 3 = position±1, etc.)
    
    Returns:
        Dict with effectiveness metrics
    """
    # Create token range
    if window_size == 1:
        tokens_to_update = [token_position]
    else:
        half_window = window_size // 2
        tokens_to_update = list(range(
            max(0, token_position - half_window),
            token_position + half_window + 1
        ))
    
    # Format prompts
    prompt_without_context = format_multiple_choice(test_case, include_context=False)
    prompt_with_context = format_multiple_choice(test_case, include_context=True)
    
    # Get baseline probabilities
    prob_a_baseline, prob_b_baseline, pred_baseline = get_option_probabilities(
        model, tokenizer, prompt_without_context
    )
    
    # Save original model state
    original_state = {}
    for layer_idx in layers_of_interest:
        layer = model.model.layers[layer_idx]
        if hasattr(layer.mlp, 'gate_proj'):
            original_state[layer_idx] = {
                'gate_proj': copy.deepcopy(layer.mlp.gate_proj.weight.data),
                'up_proj': copy.deepcopy(layer.mlp.up_proj.weight.data),
                'down_proj': copy.deepcopy(layer.mlp.down_proj.weight.data)
            }
    
    # Apply surgery
    surgeon = LLMSurgeon(model, tokenizer)
    
    # Check if token position is valid
    tokenized_context = tokenizer(prompt_with_context, return_tensors="pt")
    max_position = tokenized_context.input_ids.shape[1]
    
    if token_position >= max_position:
        # Restore and return zero effectiveness
        for layer_idx, weights in original_state.items():
            layer = model.model.layers[layer_idx]
            if hasattr(layer.mlp, 'gate_proj'):
                layer.mlp.gate_proj.weight.data = weights['gate_proj']
                layer.mlp.up_proj.weight.data = weights['up_proj']
                layer.mlp.down_proj.weight.data = weights['down_proj']
        
        return {
            'delta_prob_a': 0.0,
            'valid': False
        }
    
    # Suppress output during surgery
    import sys
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        generated_text = surgeon.generate_with_surgery(
            prompt=prompt_with_context,
            layers_of_interest=layers_of_interest,
            tokens_to_be_updated=tokens_to_update,
            max_new_tokens=1,
            temperature=0.0,
            eta=eta,
            mu=mu,
            test_callback=None
        )
    finally:
        sys.stdout = old_stdout
    
    # Measure probabilities after surgery
    prob_a_after, prob_b_after, pred_after = get_option_probabilities(
        model, tokenizer, prompt_without_context
    )
    
    # Calculate effectiveness
    delta_prob_a = prob_a_after - prob_a_baseline
    
    # Restore original model state
    for layer_idx, weights in original_state.items():
        layer = model.model.layers[layer_idx]
        if hasattr(layer.mlp, 'gate_proj'):
            layer.mlp.gate_proj.weight.data = weights['gate_proj']
            layer.mlp.up_proj.weight.data = weights['up_proj']
            layer.mlp.down_proj.weight.data = weights['down_proj']
    
    return {
        'delta_prob_a': delta_prob_a,
        'prob_a_baseline': prob_a_baseline,
        'prob_a_after': prob_a_after,
        'valid': True
    }


def grid_search_token_positions(
    model,
    tokenizer,
    test_cases: List[TestCase] = None,
    token_positions: List[int] = None,
    layers_of_interest: List[int] = [10, 11, 12],
    eta: float = 0.05,
    mu: float = 5e-4,
    window_size: int = 3
) -> Dict[str, Any]:
    """
    Perform grid search over token positions for each test case.
    
    Returns:
        Dict containing the grid search results and analysis
    """
    if test_cases is None:
        test_cases = TESTSET[:3]  # Default to first 3 cases
    
    if token_positions is None:
        # Test positions throughout the typical prompt length
        token_positions = list(range(0, 80, 2))  # Every 2nd token up to position 80
    
    print(f"Starting grid search...")
    print(f"Test cases: {len(test_cases)}")
    print(f"Token positions to test: {len(token_positions)}")
    print(f"Window size: {window_size}")
    print(f"Layers: {layers_of_interest}")
    print(f"Hebbian parameters: eta={eta}, mu={mu}")
    print("="*70)
    
    # Initialize results matrix
    results_matrix = np.zeros((len(test_cases), len(token_positions)))
    validity_matrix = np.zeros((len(test_cases), len(token_positions)), dtype=bool)
    
    # Additional tracking
    baseline_probs = []
    best_positions = []
    token_strings = {}  # Store actual tokens at each position for analysis
    
    # Progress tracking
    total_tests = len(test_cases) * len(token_positions)
    completed = 0
    
    for i, test_case in enumerate(test_cases):
        print(f"\n[Case {i+1}/{len(test_cases)}] {test_case['question'][:50]}...")
        
        # Get tokens for this prompt to understand what we're updating
        prompt_with_context = format_multiple_choice(test_case, include_context=True)
        tokenized = tokenizer(prompt_with_context, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(tokenized.input_ids[0])
        
        case_results = []
        for j, token_pos in enumerate(token_positions):
            # Progress indicator
            completed += 1
            if completed % 10 == 0:
                print(f"  Progress: {completed}/{total_tests} ({completed/total_tests*100:.1f}%)")
            
            result = test_single_token_position(
                model=model,
                tokenizer=tokenizer,
                test_case=test_case,
                token_position=token_pos,
                layers_of_interest=layers_of_interest,
                eta=eta,
                mu=mu,
                window_size=window_size
            )
            
            results_matrix[i, j] = result['delta_prob_a']
            validity_matrix[i, j] = result['valid']
            
            if result['valid'] and token_pos < len(tokens):
                # Store token string for this position
                key = f"case_{i}_pos_{token_pos}"
                token_strings[key] = tokens[token_pos]
            
            case_results.append(result)
        
        # Find best position for this case
        valid_results = [(pos, res['delta_prob_a']) 
                        for pos, res in zip(token_positions, case_results) 
                        if res['valid']]
        if valid_results:
            best_pos, best_delta = max(valid_results, key=lambda x: x[1])
            best_positions.append({
                'case_idx': i,
                'question': test_case['question'],
                'best_position': best_pos,
                'best_delta': best_delta,
                'token': tokens[best_pos] if best_pos < len(tokens) else 'N/A'
            })
            print(f"  Best position: {best_pos} (Δ={best_delta:.4f}, token='{tokens[best_pos] if best_pos < len(tokens) else 'N/A'}')")
        
        # Store baseline probability
        if case_results and case_results[0]['valid']:
            baseline_probs.append(case_results[0].get('prob_a_baseline', 0))
    
    return {
        'results_matrix': results_matrix,
        'validity_matrix': validity_matrix,
        'test_cases': test_cases,
        'token_positions': token_positions,
        'best_positions': best_positions,
        'baseline_probs': baseline_probs,
        'token_strings': token_strings,
        'parameters': {
            'layers': layers_of_interest,
            'eta': eta,
            'mu': mu,
            'window_size': window_size
        }
    }


def create_interactive_heatmap(grid_results: Dict[str, Any], save_path: str = None):
    """Create an interactive Plotly heatmap with detailed hover information"""
    
    results_matrix = grid_results['results_matrix']
    test_cases = grid_results['test_cases']
    token_positions = grid_results['token_positions']
    validity_matrix = grid_results['validity_matrix']
    
    # Create hover text with token information
    hover_text = []
    for i, test_case in enumerate(test_cases):
        row_hover = []
        for j, pos in enumerate(token_positions):
            if validity_matrix[i, j]:
                delta = results_matrix[i, j]
                key = f"case_{i}_pos_{pos}"
                token = grid_results['token_strings'].get(key, 'N/A')
                text = f"Case: {test_case['question'][:30]}...<br>"
                text += f"Position: {pos}<br>"
                text += f"Token: '{token}'<br>"
                text += f"ΔP(A): {delta:.4f}<br>"
                text += f"Effect: {'Positive' if delta > 0 else 'Negative'}"
                row_hover.append(text)
            else:
                row_hover.append("Position out of range")
        hover_text.append(row_hover)
    
    # Create the main heatmap
    fig = go.Figure(data=go.Heatmap(
        z=results_matrix,
        x=[f"Pos {p}" for p in token_positions],
        y=[f"Q{i+1}: {tc['question'][:30]}..." for i, tc in enumerate(test_cases)],
        colorscale='RdBu',
        zmid=0,  # Center colorscale at 0
        hovertext=hover_text,
        hovertemplate='%{hovertext}<extra></extra>',
        colorbar=dict(
            title="ΔP(A)",
            titleside="right",
            thickness=15,
            len=0.7
        )
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Token Position Effectiveness Grid Search<br><sub>Darker blue = Higher probability increase for target answer</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Token Position",
        yaxis_title="Test Cases",
        width=1400,
        height=600,
        xaxis=dict(
            tickangle=-45,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray'
        ),
        font=dict(size=10)
    )
    
    # Save if path provided
    if save_path:
        if not save_path.endswith('.html'):
            save_path += '.html'
        fig.write_html(save_path)
        print(f"Interactive heatmap saved to: {save_path}")
    
    fig.show()
    
    return fig


def create_analysis_visualizations(grid_results: Dict[str, Any], save_prefix: str = None):
    """Create additional analysis visualizations"""
    
    if save_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_prefix = f"grid_analysis_{timestamp}"
    
    results_matrix = grid_results['results_matrix']
    token_positions = grid_results['token_positions']
    test_cases = grid_results['test_cases']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Average Effect by Token Position',
            'Best Token Positions Distribution',
            'Effect Distribution Across All Tests',
            'Cumulative Best Performance'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'bar'}],
               [{'type': 'histogram'}, {'type': 'scatter'}]]
    )
    
    # 1. Average effect by position
    avg_effects = np.mean(results_matrix, axis=0)
    fig.add_trace(
        go.Scatter(
            x=token_positions,
            y=avg_effects,
            mode='lines+markers',
            name='Avg Effect',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    # 2. Best positions bar chart
    if grid_results['best_positions']:
        best_pos_counts = {}
        for bp in grid_results['best_positions']:
            pos = bp['best_position']
            if pos not in best_pos_counts:
                best_pos_counts[pos] = 0
            best_pos_counts[pos] += 1
        
        fig.add_trace(
            go.Bar(
                x=list(best_pos_counts.keys()),
                y=list(best_pos_counts.values()),
                name='Count',
                marker_color='green'
            ),
            row=1, col=2
        )
    
    # 3. Effect distribution histogram
    all_effects = results_matrix.flatten()
    valid_effects = all_effects[all_effects != 0]  # Exclude invalid positions
    
    fig.add_trace(
        go.Histogram(
            x=valid_effects,
            nbinsx=30,
            name='Effects',
            marker_color='purple',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # 4. Cumulative best performance
    max_effects = np.max(results_matrix, axis=1)
    cumulative_avg = np.array([np.mean(max_effects[:i+1]) for i in range(len(max_effects))])
    
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(test_cases) + 1)),
            y=cumulative_avg,
            mode='lines+markers',
            name='Cumulative Avg',
            line=dict(color='orange', width=2),
            marker=dict(size=6)
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Grid Search Analysis",
        showlegend=False,
        height=800,
        width=1200
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Token Position", row=1, col=1)
    fig.update_yaxes(title_text="Avg ΔP(A)", row=1, col=1)
    
    fig.update_xaxes(title_text="Best Position", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    fig.update_xaxes(title_text="ΔP(A)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    fig.update_xaxes(title_text="Test Case #", row=2, col=2)
    fig.update_yaxes(title_text="Cumulative Avg Best ΔP(A)", row=2, col=2)
    
    # Save
    analysis_path = f"{save_prefix}_analysis.html"
    fig.write_html(analysis_path)
    print(f"Analysis visualizations saved to: {analysis_path}")
    
    fig.show()
    
    return fig


def create_token_pattern_analysis(grid_results: Dict[str, Any]):
    """Analyze patterns in which types of tokens are most effective"""
    
    token_strings = grid_results['token_strings']
    results_matrix = grid_results['results_matrix']
    test_cases = grid_results['test_cases']
    token_positions = grid_results['token_positions']
    
    # Categorize tokens
    token_categories = {
        'punctuation': [],
        'question_words': [],
        'context_words': [],
        'answer_words': [],
        'other': []
    }
    
    # Analyze each position
    for i, test_case in enumerate(test_cases):
        prompt_with_context = format_multiple_choice(test_case, include_context=True)
        
        for j, pos in enumerate(token_positions):
            key = f"case_{i}_pos_{pos}"
            if key in token_strings:
                token = token_strings[key]
                effect = results_matrix[i, j]
                
                # Categorize token
                if token in ['?', ':', '.', ',', '(', ')', '\n']:
                    category = 'punctuation'
                elif token.lower() in ['question', 'what', 'which', 'who', 'where', 'when', 'why', 'how']:
                    category = 'question_words'
                elif token.lower() in ['context', 'option', 'answer']:
                    category = 'context_words'
                elif token.lower() in [test_case['answer_target'].lower(), test_case['answer_old'].lower()]:
                    category = 'answer_words'
                else:
                    category = 'other'
                
                token_categories[category].append({
                    'token': token,
                    'effect': effect,
                    'position': pos,
                    'case': i
                })
    
    # Calculate average effects by category
    category_stats = {}
    for category, tokens_data in token_categories.items():
        if tokens_data:
            effects = [td['effect'] for td in tokens_data]
            category_stats[category] = {
                'mean': np.mean(effects),
                'std': np.std(effects),
                'max': np.max(effects),
                'min': np.min(effects),
                'count': len(effects)
            }
    
    # Print analysis
    print("\n" + "="*70)
    print("TOKEN PATTERN ANALYSIS")
    print("="*70)
    
    for category, stats in sorted(category_stats.items(), key=lambda x: x[1]['mean'], reverse=True):
        print(f"\n{category.upper()}:")
        print(f"  Average effect: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Sample count: {stats['count']}")
    
    # Find most effective individual tokens
    all_token_effects = []
    for category, tokens_data in token_categories.items():
        for td in tokens_data:
            all_token_effects.append((td['token'], td['effect'], td['position'], category))
    
    all_token_effects.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*70)
    print("TOP 10 MOST EFFECTIVE TOKEN POSITIONS:")
    print("="*70)
    for i, (token, effect, pos, category) in enumerate(all_token_effects[:10], 1):
        print(f"{i}. Token: '{token}' | Position: {pos} | Effect: {effect:.4f} | Category: {category}")
    
    return category_stats


def save_grid_results(grid_results: Dict[str, Any], filename: str = None):
    """Save grid search results to JSON file"""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"grid_search_results_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    save_data = {
        'results_matrix': grid_results['results_matrix'].tolist(),
        'validity_matrix': grid_results['validity_matrix'].tolist(),
        'token_positions': grid_results['token_positions'],
        'best_positions': grid_results['best_positions'],
        'baseline_probs': grid_results['baseline_probs'],
        'parameters': grid_results['parameters'],
        'timestamp': datetime.now().isoformat(),
        'test_cases_summary': [
            {
                'question': tc['question'],
                'answer_old': tc['answer_old'],
                'answer_target': tc['answer_target']
            }
            for tc in grid_results['test_cases']
        ]
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"Grid search results saved to: {filename}")
    
    return filename


def main():
    """Main function to run grid search"""
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Configuration
    test_subset = [0, 1, 2, 3, 4]  # Test first 5 cases (or None for all)
    test_cases = TESTSET[:test_subset[-1]+1] if test_subset else TESTSET
    
    # Define token positions to test
    # Focus on key areas: context start, question area, option area
    token_positions = list(range(0, 100, 1))  # Every 3rd token up to position 100
    
    # Hebbian parameters
    layers_of_interest = [14]
    eta = 0.05
    mu = 1e-4
    window_size = 1  # Update 3 tokens at once
    
    print("="*70)
    print("STARTING TOKEN POSITION GRID SEARCH")
    print("="*70)
    
    # Run grid search
    grid_results = grid_search_token_positions(
        model=model,
        tokenizer=tokenizer,
        test_cases=test_cases,
        token_positions=token_positions,
        layers_of_interest=layers_of_interest,
        eta=eta,
        mu=mu,
        window_size=window_size
    )
    
    # Create visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Main heatmap
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    heatmap_path = f"grid_search_heatmap_{timestamp}"
    create_interactive_heatmap(grid_results, save_path=heatmap_path)
    
    # Analysis visualizations
    create_analysis_visualizations(grid_results, save_prefix=f"grid_search_{timestamp}")
    
    # Token pattern analysis
    category_stats = create_token_pattern_analysis(grid_results)
    
    # Save results
    json_file = save_grid_results(grid_results)
    
    # Print summary
    print("\n" + "="*70)
    print("GRID SEARCH SUMMARY")
    print("="*70)
    print(f"Test cases analyzed: {len(test_cases)}")
    print(f"Token positions tested: {len(token_positions)}")
    print(f"Total experiments: {len(test_cases) * len(token_positions)}")
    
    if grid_results['best_positions']:
        avg_best = np.mean([bp['best_delta'] for bp in grid_results['best_positions']])
        print(f"\nAverage best ΔP(A): {avg_best:.4f}")
        
        print("\nBest position for each case:")
        for bp in grid_results['best_positions']:
            print(f"  {bp['question'][:40]}...")
            print(f"    Position {bp['best_position']}: ΔP(A) = {bp['best_delta']:.4f}, Token = '{bp['token']}'")
    
    print("\n" + "="*70)
    print("COMPLETED")
    print("="*70)
    
    return grid_results, category_stats


if __name__ == "__main__":
    grid_results, category_stats = main()

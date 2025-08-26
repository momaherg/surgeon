"""
Surgery Test Runner: Evaluates Hebbian weight update effectiveness on the testset.

This module tests how well the Hebbian surgery method can shift model predictions
from old answers to new target answers by measuring probability changes for each test case.
"""

from huggingface_hub import login
login(new_session=False)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from testset import TESTSET, TestCase
from llm_surgeon import LLMSurgeon
from typing import List, Dict, Any, Tuple
import json
import csv
from datetime import datetime
import numpy as np
import copy


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


def apply_surgery_and_measure(
    model,
    tokenizer,
    test_case: TestCase,
    layers_of_interest: List[int],
    tokens_to_be_updated: List[int],
    eta: float = 0.05,
    mu: float = 5e-4,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Apply Hebbian surgery using the context and measure probability changes.
    
    Returns dict with before/after probabilities and change metrics.
    """
    # Format prompts
    prompt_without_context = format_multiple_choice(test_case, include_context=False)
    prompt_with_context = format_multiple_choice(test_case, include_context=True)
    
    # Get baseline probabilities (no context, no surgery)
    print(f"\n  Measuring baseline probabilities...")
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
    
    # Apply surgery using context prompt
    print(f"  Applying Hebbian surgery on layers {layers_of_interest}...")
    surgeon = LLMSurgeon(model, tokenizer)
    
    # Generate with surgery to update weights
    # We use the context prompt as the "primer" for surgery
    generated_text = surgeon.generate_with_surgery(
        prompt=prompt_with_context,
        layers_of_interest=layers_of_interest,
        tokens_to_be_updated=tokens_to_be_updated,
        max_new_tokens=1,  # Only need to generate the answer token
        temperature=temperature,
        eta=eta,
        mu=mu,
        test_callback=None
    )
    
    # Extract what was generated (should be A or B)
    generated_answer = generated_text.replace(prompt_with_context, "").strip()
    
    # Measure probabilities after surgery (without context to see pure effect)
    print(f"  Measuring post-surgery probabilities...")
    prob_a_after_surgery, prob_b_after_surgery, pred_after_surgery = get_option_probabilities(
        model, tokenizer, prompt_without_context
    )
    
    # Also measure with context after surgery
    prob_a_with_context, prob_b_with_context, pred_with_context = get_option_probabilities(
        model, tokenizer, prompt_with_context
    )
    
    # Calculate changes
    delta_prob_a = prob_a_after_surgery - prob_a_baseline
    delta_prob_b = prob_b_after_surgery - prob_b_baseline
    
    # Restore original model state for next test
    print(f"  Restoring original model weights...")
    for layer_idx, weights in original_state.items():
        layer = model.model.layers[layer_idx]
        if hasattr(layer.mlp, 'gate_proj'):
            layer.mlp.gate_proj.weight.data = weights['gate_proj']
            layer.mlp.up_proj.weight.data = weights['up_proj']
            layer.mlp.down_proj.weight.data = weights['down_proj']
    
    return {
        'question': test_case['question'],
        'answer_old': test_case['answer_old'],
        'answer_target': test_case['answer_target'],
        'prob_a_baseline': prob_a_baseline,
        'prob_b_baseline': prob_b_baseline,
        'prob_a_after_surgery': prob_a_after_surgery,
        'prob_b_after_surgery': prob_b_after_surgery,
        'prob_a_with_context': prob_a_with_context,
        'prob_b_with_context': prob_b_with_context,
        'delta_prob_a': delta_prob_a,
        'delta_prob_b': delta_prob_b,
        'percent_change_a': (delta_prob_a / prob_a_baseline * 100) if prob_a_baseline > 0 else 0,
        'percent_change_b': (delta_prob_b / prob_b_baseline * 100) if prob_b_baseline > 0 else 0,
        'predicted_baseline': pred_baseline,
        'predicted_after_surgery': pred_after_surgery,
        'predicted_with_context': pred_with_context,
        'generated_during_surgery': generated_answer,
        'success': pred_after_surgery.strip() == 'A'  # Success if predicts target after surgery
    }


def run_surgery_testset(
    model,
    tokenizer,
    layers_of_interest: List[int] = [10, 11, 12],
    tokens_to_be_updated: List[int] = None,
    eta: float = 0.05,
    mu: float = 5e-4,
    temperature: float = 0.0,
    test_subset: List[int] = None
) -> Tuple[List[Dict], Dict]:
    """
    Run the surgery test on the full testset or a subset.
    
    Args:
        test_subset: Optional list of indices to test (e.g., [0, 1, 2] for first 3 cases)
    """
    if tokens_to_be_updated is None:
        tokens_to_be_updated = list(range(40, 50))  # Default token positions
    
    # Select test cases
    test_cases = TESTSET if test_subset is None else [TESTSET[i] for i in test_subset]
    
    print(f"\nRunning surgery tests on {len(test_cases)} cases...")
    print(f"Layers: {layers_of_interest}")
    print(f"Token positions: {tokens_to_be_updated}")
    print(f"Hebbian parameters: eta={eta}, mu={mu}")
    print("="*70)
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}/{len(test_cases)}: {test_case['question']}")
        print(f"  Old answer: {test_case['answer_old']}")
        print(f"  Target answer: {test_case['answer_target']}")
        
        try:
            result = apply_surgery_and_measure(
                model=model,
                tokenizer=tokenizer,
                test_case=test_case,
                layers_of_interest=layers_of_interest,
                tokens_to_be_updated=tokens_to_be_updated,
                eta=eta,
                mu=mu,
                temperature=temperature
            )
            results.append(result)
            
            # Print immediate results
            print(f"  Results:")
            print(f"    Baseline: P(A)={result['prob_a_baseline']:.4f}, P(B)={result['prob_b_baseline']:.4f}")
            print(f"    After Surgery: P(A)={result['prob_a_after_surgery']:.4f}, P(B)={result['prob_b_after_surgery']:.4f}")
            print(f"    Change: ΔP(A)={result['delta_prob_a']:.4f} ({result['percent_change_a']:.1f}%)")
            print(f"    Predicted: '{result['predicted_baseline']}' → '{result['predicted_after_surgery']}'")
            print(f"    Success: {result['success']}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'question': test_case['question'],
                'error': str(e)
            })
    
    # Analyze results
    analysis = analyze_surgery_results(results)
    
    return results, analysis


def analyze_surgery_results(results: List[Dict]) -> Dict[str, Any]:
    """Analyze the surgery test results"""
    # Filter out errors
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        return {'error': 'No valid results'}
    
    # Calculate statistics
    successes = sum(1 for r in valid_results if r['success'])
    avg_delta_prob_a = np.mean([r['delta_prob_a'] for r in valid_results])
    avg_percent_change_a = np.mean([r['percent_change_a'] for r in valid_results])
    
    # Find cases with significant improvement
    significant_improvements = [r for r in valid_results if r['delta_prob_a'] > 0.1]
    
    # Calculate baseline accuracy
    baseline_correct_b = sum(1 for r in valid_results if r['predicted_baseline'].strip() == 'B')
    
    analysis = {
        'total_cases': len(valid_results),
        'errors': len(results) - len(valid_results),
        'successes': successes,
        'success_rate': successes / len(valid_results) if valid_results else 0,
        'avg_delta_prob_a': avg_delta_prob_a,
        'avg_percent_change_a': avg_percent_change_a,
        'significant_improvements': len(significant_improvements),
        'baseline_correct_b': baseline_correct_b,
        'baseline_accuracy_b': baseline_correct_b / len(valid_results) if valid_results else 0,
        'max_improvement': max([r['delta_prob_a'] for r in valid_results]) if valid_results else 0,
        'min_improvement': min([r['delta_prob_a'] for r in valid_results]) if valid_results else 0,
    }
    
    return analysis


def print_surgery_summary(analysis: Dict[str, Any]):
    """Print a summary of surgery test results"""
    print("\n" + "="*70)
    print("HEBBIAN SURGERY TEST SUMMARY")
    print("="*70)
    
    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    print(f"Total test cases: {analysis['total_cases']}")
    print(f"Errors encountered: {analysis['errors']}")
    print()
    print("BASELINE PERFORMANCE:")
    print(f"  Models correctly choosing B (old answer): {analysis['baseline_accuracy_b']:.1%}")
    print()
    print("SURGERY EFFECTIVENESS:")
    print(f"  Successful shifts to A (target): {analysis['success_rate']:.1%} ({analysis['successes']}/{analysis['total_cases']})")
    print(f"  Average ΔP(A): {analysis['avg_delta_prob_a']:.4f}")
    print(f"  Average % change in P(A): {analysis['avg_percent_change_a']:.1f}%")
    print(f"  Cases with >10% improvement: {analysis['significant_improvements']}")
    print(f"  Max improvement: {analysis['max_improvement']:.4f}")
    print(f"  Min improvement: {analysis['min_improvement']:.4f}")
    print("="*70)


def save_surgery_results(results: List[Dict], analysis: Dict, base_filename: str = None):
    """Save surgery test results to files"""
    if base_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"surgery_test_{timestamp}"
    
    # Save detailed JSON
    json_filename = f"{base_filename}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis,
            'detailed_results': results
        }, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to: {json_filename}")
    
    # Save summary CSV
    csv_filename = f"{base_filename}.csv"
    if results and 'error' not in results[0]:
        columns = [
            'question', 'answer_old', 'answer_target',
            'prob_a_baseline', 'prob_b_baseline',
            'prob_a_after_surgery', 'prob_b_after_surgery',
            'delta_prob_a', 'percent_change_a', 'success'
        ]
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for result in results:
                if 'error' not in result:
                    row = {col: result.get(col, '') for col in columns}
                    writer.writerow(row)
        print(f"Summary CSV saved to: {csv_filename}")
    
    return json_filename, csv_filename


def main():
    """Main function to run surgery tests"""
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Test parameters
    layers_of_interest = [10, 11, 12]  # Middle layers often work best
    tokens_to_be_updated = list(range(40, 50))  # Token positions to update
    eta = 0.05  # Learning rate for Hebbian updates
    mu = 5e-4  # Stabilizer
    
    # Run a subset first to test (use None for all cases)
    test_subset = [0, 1, 2]  # Test first 3 cases, or None for all
    
    # Run tests
    results, analysis = run_surgery_testset(
        model=model,
        tokenizer=tokenizer,
        layers_of_interest=layers_of_interest,
        tokens_to_be_updated=tokens_to_be_updated,
        eta=eta,
        mu=mu,
        temperature=0.0,  # Greedy for consistent results
        test_subset=test_subset
    )
    
    # Print summary
    print_surgery_summary(analysis)
    
    # Save results
    json_file, csv_file = save_surgery_results(results, analysis)
    
    return results, analysis


if __name__ == "__main__":
    results, analysis = main()

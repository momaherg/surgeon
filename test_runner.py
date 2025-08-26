from huggingface_hub import login
login(new_session=False)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from testset import TESTSET, TestCase, format_question_prompt, format_target_prompt, format_factual_only_prompt, format_persuasive_only_prompt
from typing import List, Dict, Any
import json
import csv
from datetime import datetime


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


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 50, temperature: float = 0.1) -> str:
    """Generate a response from the model for a given prompt"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens (excluding the input prompt)
    input_length = inputs['input_ids'].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return response.strip()


def test_single_case(model, tokenizer, test_case: TestCase) -> Dict[str, Any]:
    """Test a single test case and return results"""
    print(f"Testing: {test_case['question']}")
    
    # Test baseline (question only)
    baseline_prompt = format_question_prompt(test_case)
    baseline_response = generate_response(model, tokenizer, baseline_prompt)
    
    # Test with factual context only
    factual_prompt = format_factual_only_prompt(test_case)
    factual_response = generate_response(model, tokenizer, factual_prompt)
    
    # Test with persuasive context only
    persuasive_prompt = format_persuasive_only_prompt(test_case)
    persuasive_response = generate_response(model, tokenizer, persuasive_prompt)
    
    # Test with combined context (original)
    combined_prompt = format_target_prompt(test_case)
    combined_response = generate_response(model, tokenizer, combined_prompt)
    
    return {
        "question": test_case["question"],
        "answer_old": test_case["answer_old"],
        "answer_target": test_case["answer_target"],
        "supporting_persuasive_sentence": test_case["supporting_persuasive_sentence"],
        "factual_information_sentence": test_case["factual_information_sentence"],
        "baseline_prompt": baseline_prompt,
        "baseline_response": baseline_response,
        "factual_prompt": factual_prompt,
        "factual_response": factual_response,
        "persuasive_prompt": persuasive_prompt,
        "persuasive_response": persuasive_response,
        "combined_prompt": combined_prompt,
        "combined_response": combined_response,
        "baseline_matches_old": test_case["answer_old"].lower() in baseline_response.lower(),
        "factual_matches_new": test_case["answer_target"].lower() in factual_response.lower(),
        "persuasive_matches_new": test_case["answer_target"].lower() in persuasive_response.lower(),
        "combined_matches_new": test_case["answer_target"].lower() in combined_response.lower(),
    }


def run_full_testset(model, tokenizer) -> List[Dict[str, Any]]:
    """Run the full testset and return all results"""
    print(f"Running testset with {len(TESTSET)} test cases...")
    results = []
    
    for i, test_case in enumerate(TESTSET):
        print(f"\n--- Test Case {i+1}/{len(TESTSET)} ---")
        try:
            result = test_single_case(model, tokenizer, test_case)
            results.append(result)
            
            # Print immediate results
            print(f"Question: {result['question']}")
            print(f"Expected old answer: {result['answer_old']}")
            print(f"Expected new answer: {result['answer_target']}")
            print(f"Baseline response: {result['baseline_response']}")
            print(f"Factual-only response: {result['factual_response']}")
            print(f"Persuasive-only response: {result['persuasive_response']}")
            print(f"Combined response: {result['combined_response']}")
            print(f"Baseline correct: {result['baseline_matches_old']}")
            print(f"Factual correct: {result['factual_matches_new']}")
            print(f"Persuasive correct: {result['persuasive_matches_new']}")
            print(f"Combined correct: {result['combined_matches_new']}")
            
        except Exception as e:
            print(f"Error testing case {i+1}: {e}")
            results.append({
                "question": test_case["question"],
                "error": str(e)
            })
    
    return results


def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the test results and provide summary statistics"""
    total_cases = len([r for r in results if "error" not in r])
    baseline_correct = sum(1 for r in results if r.get("baseline_matches_old", False))
    factual_correct = sum(1 for r in results if r.get("factual_matches_new", False))
    persuasive_correct = sum(1 for r in results if r.get("persuasive_matches_new", False))
    combined_correct = sum(1 for r in results if r.get("combined_matches_new", False))
    
    analysis = {
        "total_test_cases": total_cases,
        "baseline_accuracy": baseline_correct / total_cases if total_cases > 0 else 0,
        "factual_accuracy": factual_correct / total_cases if total_cases > 0 else 0,
        "persuasive_accuracy": persuasive_correct / total_cases if total_cases > 0 else 0,
        "combined_accuracy": combined_correct / total_cases if total_cases > 0 else 0,
        "baseline_correct_count": baseline_correct,
        "factual_correct_count": factual_correct,
        "persuasive_correct_count": persuasive_correct,
        "combined_correct_count": combined_correct,
        "successful_factual_shifts": sum(1 for r in results 
                                       if r.get("baseline_matches_old", False) and r.get("factual_matches_new", False)),
        "successful_persuasive_shifts": sum(1 for r in results 
                                          if r.get("baseline_matches_old", False) and r.get("persuasive_matches_new", False)),
        "successful_combined_shifts": sum(1 for r in results 
                                        if r.get("baseline_matches_old", False) and r.get("combined_matches_new", False)),
        "errors": len([r for r in results if "error" in r])
    }
    
    return analysis


def save_results_csv(results: List[Dict[str, Any]], filename: str = None):
    """Save results to a CSV file for easy reading"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"testset_results_{timestamp}.csv"
    
    # Define CSV columns
    columns = [
        "question",
        "answer_old", 
        "answer_target",
        "baseline_response",
        "factual_response",
        "persuasive_response", 
        "combined_response",
        "baseline_matches_old",
        "factual_matches_new",
        "persuasive_matches_new",
        "combined_matches_new",
        "supporting_persuasive_sentence",
        "factual_information_sentence"
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for result in results:
            if "error" not in result:  # Skip error entries
                # Create a row with only the columns we want
                row = {col: result.get(col, '') for col in columns}
                writer.writerow(row)
    
    print(f"\nResults saved to CSV: {filename}")
    return filename


def save_results_json(results: List[Dict[str, Any]], analysis: Dict[str, Any], filename: str = None):
    """Save results to a JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"testset_results_{timestamp}.json"
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "analysis": analysis,
        "detailed_results": results
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to JSON: {filename}")
    return filename


def print_summary(analysis: Dict[str, Any]):
    """Print a summary of the test results"""
    print("\n" + "="*70)
    print("TESTSET EVALUATION SUMMARY")
    print("="*70)
    print(f"Total test cases: {analysis['total_test_cases']}")
    print(f"Errors: {analysis['errors']}")
    print()
    print("ACCURACY BY CONTEXT TYPE:")
    print(f"  Baseline (no context):     {analysis['baseline_accuracy']:.2%} ({analysis['baseline_correct_count']}/{analysis['total_test_cases']})")
    print(f"  Factual-only context:      {analysis['factual_accuracy']:.2%} ({analysis['factual_correct_count']}/{analysis['total_test_cases']})")
    print(f"  Persuasive-only context:   {analysis['persuasive_accuracy']:.2%} ({analysis['persuasive_correct_count']}/{analysis['total_test_cases']})")
    print(f"  Combined context:          {analysis['combined_accuracy']:.2%} ({analysis['combined_correct_count']}/{analysis['total_test_cases']})")
    print()
    print("SUCCESSFUL CONTEXT SHIFTS (baseline correct → new answer correct):")
    print(f"  Factual-only shifts:       {analysis['successful_factual_shifts']}")
    print(f"  Persuasive-only shifts:    {analysis['successful_persuasive_shifts']}")
    print(f"  Combined shifts:           {analysis['successful_combined_shifts']}")
    print("="*70)


def main():
    """Main function to run the complete testset evaluation"""
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Run testset
    results = run_full_testset(model, tokenizer)
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Print summary
    print_summary(analysis)
    
    # Save results in both formats
    csv_filename = save_results_csv(results)
    json_filename = save_results_json(results, analysis)
    
    return results, analysis, csv_filename, json_filename


if __name__ == "__main__":
    results, analysis, csv_filename, json_filename = main()

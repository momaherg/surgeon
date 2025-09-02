"""
Evaluation script for NPT models.
Compares NPT performance with baseline methods on various tasks.
"""

import os
import argparse
import torch
import json
from typing import Dict, List, Optional
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)
from datasets import load_dataset
import numpy as np
from collections import defaultdict


class NPTEvaluator:
    """Evaluator for NPT models on various benchmarks."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
        seed: int = 42
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        set_seed(seed)
        
        # Load model and tokenizer
        self.model, self.tokenizer = self.load_model()
    
    def load_model(self):
        """Load NPT model and tokenizer."""
        print(f"Loading model from {self.model_path}...")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=self.dtype,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        return model, tokenizer
    
    def evaluate_perplexity(
        self,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        split: str = "test",
        max_samples: Optional[int] = None
    ) -> float:
        """Evaluate perplexity on a text dataset."""
        print(f"\nEvaluating perplexity on {dataset_name}...")
        
        # Load dataset
        dataset = load_dataset(dataset_name, dataset_config, split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for example in tqdm(dataset, desc="Computing perplexity"):
                text = example["text"]
                if not text.strip():
                    continue
                
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self.model.device)
                
                # Forward pass
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                
                # Accumulate loss
                total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def evaluate_few_shot(
        self,
        task: str = "hellaswag",
        num_shots: int = 5,
        max_samples: Optional[int] = 100
    ) -> float:
        """Evaluate few-shot performance on a task."""
        print(f"\nEvaluating {num_shots}-shot performance on {task}...")
        
        # Task configurations
        task_configs = {
            "hellaswag": {
                "dataset": "hellaswag",
                "split": "validation",
                "prompt_template": "Context: {ctx}\nQuestion: Which ending makes the most sense?\nA) {endings[0]}\nB) {endings[1]}\nC) {endings[2]}\nD) {endings[3]}\nAnswer:",
                "answer_map": {0: "A", 1: "B", 2: "C", 3: "D"}
            },
            "winogrande": {
                "dataset": "winogrande",
                "config": "winogrande_xl",
                "split": "validation",
                "prompt_template": "Sentence: {sentence}\nQuestion: What does '_{option}' refer to?\nAnswer:",
            }
        }
        
        if task not in task_configs:
            print(f"Task {task} not supported")
            return 0.0
        
        config = task_configs[task]
        
        # Load dataset
        if "config" in config:
            dataset = load_dataset(config["dataset"], config["config"], split=config["split"])
        else:
            dataset = load_dataset(config["dataset"], split=config["split"])
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        # Evaluate
        correct = 0
        total = 0
        
        for example in tqdm(dataset, desc=f"Evaluating {task}"):
            # Create prompt based on task
            if task == "hellaswag":
                prompt = config["prompt_template"].format(
                    ctx=example["ctx"],
                    endings=example["endings"]
                )
                correct_answer = config["answer_map"][example["label"]]
            else:
                # Add other task-specific logic here
                continue
            
            # Get model prediction
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=0.1,
                    do_sample=False
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].size(1):],
                skip_special_tokens=True
            ).strip()
            
            # Check if correct
            if response.startswith(correct_answer):
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def evaluate_instruction_following(
        self,
        test_prompts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate instruction following capabilities."""
        print("\nEvaluating instruction following...")
        
        if test_prompts is None:
            test_prompts = [
                "Write a haiku about artificial intelligence.",
                "Explain quantum computing to a 5-year-old.",
                "List 5 creative uses for a paperclip.",
                "Translate 'Hello, how are you?' to Spanish.",
                "Write a Python function to calculate factorial.",
            ]
        
        results = []
        
        for prompt in tqdm(test_prompts, desc="Testing prompts"):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].size(1):],
                skip_special_tokens=True
            )
            
            # Simple quality metrics
            response_length = len(response.split())
            results.append({
                "prompt": prompt,
                "response": response,
                "length": response_length
            })
        
        # Calculate average metrics
        avg_length = np.mean([r["length"] for r in results])
        
        return {
            "average_response_length": avg_length,
            "responses": results
        }
    
    def compare_with_baseline(
        self,
        baseline_model_path: str,
        tasks: List[str] = ["perplexity", "instruction_following"]
    ) -> Dict[str, Dict[str, float]]:
        """Compare NPT model with a baseline model."""
        print("\nComparing NPT with baseline model...")
        
        results = {
            "npt": {},
            "baseline": {}
        }
        
        # Evaluate NPT
        for task in tasks:
            if task == "perplexity":
                results["npt"]["perplexity"] = self.evaluate_perplexity(max_samples=100)
            elif task == "instruction_following":
                results["npt"]["instruction_following"] = self.evaluate_instruction_following()
        
        # Load and evaluate baseline
        print(f"\nLoading baseline model from {baseline_model_path}...")
        self.model_path = baseline_model_path
        self.model, self.tokenizer = self.load_model()
        
        for task in tasks:
            if task == "perplexity":
                results["baseline"]["perplexity"] = self.evaluate_perplexity(max_samples=100)
            elif task == "instruction_following":
                results["baseline"]["instruction_following"] = self.evaluate_instruction_following()
        
        return results
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate NPT Model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to NPT model checkpoint"
    )
    parser.add_argument(
        "--baseline_path",
        type=str,
        default=None,
        help="Path to baseline model for comparison"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["perplexity"],
        choices=["perplexity", "few_shot", "instruction_following"],
        help="Evaluation tasks to run"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./evaluation_results.json",
        help="Path to save results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (for debugging)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = NPTEvaluator(
        model_path=args.model_path,
        seed=args.seed
    )
    
    # Run evaluations
    results = {}
    
    for task in args.tasks:
        if task == "perplexity":
            perplexity = evaluator.evaluate_perplexity(max_samples=args.max_samples)
            results["perplexity"] = perplexity
            print(f"\nPerplexity: {perplexity:.2f}")
        
        elif task == "few_shot":
            accuracy = evaluator.evaluate_few_shot(max_samples=args.max_samples)
            results["few_shot_accuracy"] = accuracy
            print(f"\nFew-shot accuracy: {accuracy:.2%}")
        
        elif task == "instruction_following":
            inst_results = evaluator.evaluate_instruction_following()
            results["instruction_following"] = inst_results
            print(f"\nAverage response length: {inst_results['average_response_length']:.1f} words")
    
    # Compare with baseline if provided
    if args.baseline_path:
        comparison = evaluator.compare_with_baseline(
            baseline_model_path=args.baseline_path,
            tasks=args.tasks
        )
        results["comparison"] = comparison
    
    # Save results
    evaluator.save_results(results, args.output_path)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: See detailed results in {args.output_path}")


if __name__ == "__main__":
    main()

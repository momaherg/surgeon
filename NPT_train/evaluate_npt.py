"""
NPT Evaluation Utilities

This module provides tools to evaluate and verify the NPT model's performance,
ensuring it maintains functional equivalence with the base model.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import json
import os
from datetime import datetime

from npt_model import NPTModelWrapper


class NPTEvaluator:
    """Comprehensive evaluation suite for NPT models."""
    
    def __init__(
        self,
        model: NPTModelWrapper,
        tokenizer: AutoTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
    def evaluate_perplexity(
        self,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-103-v1",
        num_samples: int = 1000,
        max_length: int = 512,
    ) -> Dict[str, float]:
        """
        Evaluate perplexity on a held-out dataset.
        Compares NPT model perplexity with original model behavior.
        """
        print(f"Evaluating perplexity on {dataset_name}...")
        
        # Load dataset
        dataset = load_dataset(dataset_name, dataset_config, split="test")
        
        # Prepare data
        texts = []
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            texts.append(example['text'])
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        
        # Evaluate NPT model
        npt_perplexity = self._compute_perplexity(
            encodings['input_ids'],
            encodings['attention_mask'],
            use_original=False,
        )
        
        # Evaluate with original layers
        original_perplexity = self._compute_perplexity(
            encodings['input_ids'],
            encodings['attention_mask'],
            use_original=True,
        )
        
        results = {
            'npt_perplexity': npt_perplexity,
            'original_perplexity': original_perplexity,
            'perplexity_ratio': npt_perplexity / original_perplexity,
            'perplexity_difference': abs(npt_perplexity - original_perplexity),
        }
        
        return results
    
    def _compute_perplexity(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_original: bool = False,
    ) -> float:
        """Compute perplexity for a batch of sequences."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i in tqdm(range(0, len(input_ids), 8), desc="Computing perplexity"):
                batch_ids = input_ids[i:i+8].to(self.device)
                batch_mask = attention_mask[i:i+8].to(self.device)
                
                if use_original:
                    # Use original layers
                    outputs = self.model._forward_with_original_layers(
                        input_ids=batch_ids,
                        attention_mask=batch_mask,
                    )
                    logits = outputs['logits']
                else:
                    # Use NPT layers
                    outputs = self.model(
                        input_ids=batch_ids,
                        attention_mask=batch_mask,
                    )
                    logits = outputs['logits']
                
                # Compute loss
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch_ids[:, 1:].contiguous()
                shift_mask = batch_mask[:, 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='none',
                )
                
                # Mask out padding
                loss = loss * shift_mask.view(-1)
                
                total_loss += loss.sum().item()
                total_tokens += shift_mask.sum().item()
        
        perplexity = np.exp(total_loss / total_tokens)
        return perplexity
    
    def evaluate_layer_equivalence(
        self,
        num_samples: int = 100,
        max_length: int = 128,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate how closely NPT layers match original layer outputs.
        """
        print("Evaluating layer-wise equivalence...")
        
        # Generate random inputs
        input_ids = torch.randint(
            0, self.tokenizer.vocab_size, 
            (num_samples, max_length),
            device=self.device,
        )
        attention_mask = torch.ones_like(input_ids)
        
        # Get layer outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_original_outputs=True,
        )
        
        layer_outputs = outputs['layer_outputs']
        
        # Compute metrics for each layer
        layer_metrics = {}
        
        for layer_idx, layer_data in layer_outputs.items():
            npt_output = layer_data['npt']
            original_output = layer_data['original']
            
            # Compute various similarity metrics
            mse = F.mse_loss(npt_output, original_output).item()
            
            # Cosine similarity
            npt_flat = npt_output.view(-1, npt_output.shape[-1])
            orig_flat = original_output.view(-1, original_output.shape[-1])
            cos_sim = F.cosine_similarity(npt_flat, orig_flat, dim=-1).mean().item()
            
            # Relative error
            rel_error = (torch.norm(npt_output - original_output) / 
                        torch.norm(original_output)).item()
            
            # Maximum absolute difference
            max_diff = torch.max(torch.abs(npt_output - original_output)).item()
            
            layer_metrics[f"layer_{layer_idx}"] = {
                'mse': mse,
                'cosine_similarity': cos_sim,
                'relative_error': rel_error,
                'max_absolute_diff': max_diff,
            }
        
        return layer_metrics
    
    def evaluate_weight_delta_distribution(self) -> Dict[str, np.ndarray]:
        """
        Analyze the distribution of weight deltas generated by NP components.
        """
        print("Analyzing weight delta distributions...")
        
        # Collect weight delta statistics
        all_deltas = []
        layer_deltas = {}
        
        # Generate some random attention outputs
        batch_size = 32
        seq_len = 128
        
        with torch.no_grad():
            for idx_str, npt_layer in self.model.npt_layers.items():
                # Generate random attention output
                attn_output = torch.randn(
                    batch_size, seq_len, self.model.d_model,
                    device=self.device,
                )
                
                # Get weight delta
                delta_w = npt_layer.np_component(attn_output)
                
                # Flatten and store
                delta_flat = delta_w.cpu().numpy().flatten()
                all_deltas.append(delta_flat)
                layer_deltas[f"layer_{idx_str}"] = delta_flat
        
        # Concatenate all deltas
        all_deltas = np.concatenate(all_deltas)
        
        return {
            'all_deltas': all_deltas,
            'layer_deltas': layer_deltas,
            'statistics': {
                'mean': float(np.mean(all_deltas)),
                'std': float(np.std(all_deltas)),
                'min': float(np.min(all_deltas)),
                'max': float(np.max(all_deltas)),
                'percentile_1': float(np.percentile(all_deltas, 1)),
                'percentile_99': float(np.percentile(all_deltas, 99)),
            }
        }
    
    def evaluate_generation_quality(
        self,
        prompts: Optional[List[str]] = None,
        max_new_tokens: int = 100,
        num_samples: int = 5,
    ) -> Dict[str, List[str]]:
        """
        Compare text generation quality between NPT and original model.
        """
        if prompts is None:
            prompts = [
                "The future of artificial intelligence is",
                "Climate change is one of the most pressing issues because",
                "In the field of quantum computing,",
                "The human brain is fascinating because",
                "Space exploration has revealed that",
            ]
        
        print("Evaluating generation quality...")
        
        npt_generations = []
        original_generations = []
        
        self.model.eval()
        
        for prompt in prompts[:num_samples]:
            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors='pt')
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Generate with NPT model
            with torch.no_grad():
                npt_output = self.model.base_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            npt_text = self.tokenizer.decode(npt_output[0], skip_special_tokens=True)
            npt_generations.append(npt_text)
            
            # Generate with original layers (temporarily swap)
            if hasattr(self.model.base_model, 'model'):
                layers = self.model.base_model.model.layers
            else:
                layers = self.model.base_model.transformer.h
            
            # Swap to original layers
            temp_storage = {}
            for idx_str, original_layer in self.model.original_layers.items():
                idx = int(idx_str)
                temp_storage[idx] = layers[idx]
                layers[idx] = original_layer
            
            # Generate with original model
            with torch.no_grad():
                original_output = self.model.base_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            original_text = self.tokenizer.decode(original_output[0], skip_special_tokens=True)
            original_generations.append(original_text)
            
            # Restore NPT layers
            for idx, npt_layer in temp_storage.items():
                layers[idx] = npt_layer
        
        return {
            'prompts': prompts[:num_samples],
            'npt_generations': npt_generations,
            'original_generations': original_generations,
        }
    
    def create_evaluation_report(
        self,
        save_path: str,
        run_all_evaluations: bool = True,
    ) -> Dict[str, Any]:
        """
        Create a comprehensive evaluation report.
        """
        print("Creating comprehensive evaluation report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'base_model': self.model.config._name_or_path,
                'npt_layers': self.model.npt_layer_indices,
                'rank': self.model.npt_layers[str(self.model.npt_layer_indices[0])].np_component.rank,
            },
            'evaluations': {}
        }
        
        if run_all_evaluations:
            # Perplexity evaluation
            perplexity_results = self.evaluate_perplexity(num_samples=500)
            report['evaluations']['perplexity'] = perplexity_results
            
            # Layer equivalence
            layer_results = self.evaluate_layer_equivalence(num_samples=50)
            report['evaluations']['layer_equivalence'] = layer_results
            
            # Weight delta distribution
            delta_results = self.evaluate_weight_delta_distribution()
            report['evaluations']['weight_delta_stats'] = delta_results['statistics']
            
            # Generation quality
            generation_results = self.evaluate_generation_quality(num_samples=3)
            report['evaluations']['generation_samples'] = generation_results
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create visualizations
        self._create_visualizations(report, os.path.dirname(save_path))
        
        return report
    
    def _create_visualizations(self, report: Dict[str, Any], save_dir: str):
        """Create visualization plots for the evaluation report."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Layer equivalence heatmap
        if 'layer_equivalence' in report['evaluations']:
            layer_data = report['evaluations']['layer_equivalence']
            
            # Create matrix for heatmap
            metrics = ['mse', 'cosine_similarity', 'relative_error']
            layers = sorted(layer_data.keys(), key=lambda x: int(x.split('_')[1]))
            
            data_matrix = []
            for metric in metrics:
                row = [layer_data[layer][metric] for layer in layers]
                data_matrix.append(row)
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(
                data_matrix,
                xticklabels=[f"L{l.split('_')[1]}" for l in layers],
                yticklabels=metrics,
                annot=True,
                fmt='.4f',
                cmap='coolwarm',
            )
            plt.title('Layer-wise Equivalence Metrics')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'layer_equivalence_heatmap.png'), dpi=150)
            plt.close()
        
        print(f"Visualizations saved to {save_dir}")


def main():
    """Run evaluation on a trained NPT model."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Evaluate NPT model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to NPT checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_report.json",
        help="Path to save evaluation report",
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create NPT model and load checkpoint
    model = NPTModelWrapper(
        base_model_name=config['model']['base_model_name'],
        npt_layers=config['model']['npt_layers'],
        rank=config['model']['rank'],
        modulation_scale=config['model']['modulation_scale'],
    )
    model.load_npt_components(args.checkpoint)
    
    # Create evaluator
    evaluator = NPTEvaluator(model, tokenizer)
    
    # Run evaluation
    report = evaluator.create_evaluation_report(args.output)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    if 'perplexity' in report['evaluations']:
        perp = report['evaluations']['perplexity']
        print(f"\nPerplexity:")
        print(f"  NPT Model: {perp['npt_perplexity']:.2f}")
        print(f"  Original Model: {perp['original_perplexity']:.2f}")
        print(f"  Ratio: {perp['perplexity_ratio']:.4f}")
    
    if 'weight_delta_stats' in report['evaluations']:
        stats = report['evaluations']['weight_delta_stats']
        print(f"\nWeight Delta Statistics:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
    
    print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()

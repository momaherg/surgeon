"""
NPT Equivalence Pre-training Script

This script implements the Phase 1 training protocol for the Neuro-Plastic Transformer,
focusing on training the NP components to mimic the original residual connections.
"""

import os
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from datasets import load_dataset
from accelerate import Accelerator
from tqdm import tqdm
import wandb
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime

from npt_model import NPTModelWrapper


class EquivalenceLoss(nn.Module):
    """
    Loss function for NPT equivalence pre-training.
    
    Minimizes the difference between NPT and original layer outputs,
    with regularization to ensure low-magnitude weight deltas.
    """
    
    def __init__(
        self,
        equivalence_weight: float = 1.0,
        regularization_weight: float = 0.01,
        loss_type: str = "mse",  # "mse" or "cosine"
    ):
        super().__init__()
        self.equivalence_weight = equivalence_weight
        self.regularization_weight = regularization_weight
        self.loss_type = loss_type
        
    def forward(
        self,
        layer_outputs: Dict[int, Dict[str, torch.Tensor]],
        model: NPTModelWrapper,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute equivalence loss between NPT and original layers.
        
        Args:
            layer_outputs: Dictionary mapping layer indices to their outputs
            model: The NPT model (for accessing weight delta statistics)
            
        Returns:
            Dictionary containing total loss and individual components
        """
        equivalence_loss = 0.0
        regularization_loss = 0.0
        num_layers = len(layer_outputs)
        
        # Compute equivalence loss for each NPT layer
        for layer_idx, outputs in layer_outputs.items():
            npt_output = outputs['npt']
            original_output = outputs['original']
            
            if self.loss_type == "mse":
                # Mean squared error loss
                layer_loss = F.mse_loss(npt_output, original_output)
            elif self.loss_type == "cosine":
                # Cosine similarity loss (1 - cosine_similarity)
                npt_flat = npt_output.view(-1, npt_output.shape[-1])
                orig_flat = original_output.view(-1, original_output.shape[-1])
                cos_sim = F.cosine_similarity(npt_flat, orig_flat, dim=-1)
                layer_loss = (1 - cos_sim).mean()
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            equivalence_loss += layer_loss
        
        # Average over layers
        equivalence_loss = equivalence_loss / num_layers if num_layers > 0 else 0.0
        
        # Compute regularization loss (encourage low-magnitude weight deltas)
        for npt_layer in model.npt_layers.values():
            # Get the weight delta norm
            A_norm = torch.norm(npt_layer.np_component.A, p='fro')
            B_norm = torch.norm(npt_layer.np_component.B, p='fro')
            
            # Penalize large norms
            regularization_loss += (A_norm + B_norm) / 2.0
        
        regularization_loss = regularization_loss / len(model.npt_layers)
        
        # Total loss
        total_loss = (
            self.equivalence_weight * equivalence_loss +
            self.regularization_weight * regularization_loss
        )
        
        return {
            'loss': total_loss,
            'equivalence_loss': equivalence_loss,
            'regularization_loss': regularization_loss,
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_dataset(config: Dict[str, Any], tokenizer: Any) -> tuple:
    """Prepare training and evaluation datasets."""
    # Load dataset
    dataset = load_dataset(
        config['data']['dataset_name'],
        config['data']['dataset_config'],
        cache_dir=config['paths']['cache_dir'],
    )
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=config['data']['max_length'],
            return_tensors='pt',
        )
    
    # Tokenize datasets
    tokenized_train = dataset['train'].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
    )
    tokenized_eval = dataset['validation'].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['validation'].column_names,
    )
    
    # Limit dataset size for initial experiments
    if config['data']['num_train_samples']:
        tokenized_train = tokenized_train.select(range(config['data']['num_train_samples']))
    if config['data']['num_eval_samples']:
        tokenized_eval = tokenized_eval.select(range(config['data']['num_eval_samples']))
    
    return tokenized_train, tokenized_eval


def generate_sample_predictions(
    model: NPTModelWrapper,
    tokenizer: Any,
    prompt: str = "The future of artificial intelligence is",
    max_new_tokens: int = 50,
    temperature: float = 0.8,
) -> str:
    """Generate sample predictions for logging."""
    model.eval()
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.base_model.device)
    attention_mask = inputs['attention_mask'].to(model.base_model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    model.train()
    return generated_text


def evaluate_model(
    model: NPTModelWrapper,
    eval_dataloader: DataLoader,
    loss_fn: EquivalenceLoss,
    accelerator: Accelerator,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_equivalence_loss = 0.0
    total_regularization_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
            # Forward pass with original outputs
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                return_original_outputs=True,
            )
            
            # Compute loss
            loss_dict = loss_fn(outputs['layer_outputs'], model)
            
            total_loss += loss_dict['loss'].item()
            total_equivalence_loss += loss_dict['equivalence_loss'].item()
            total_regularization_loss += loss_dict['regularization_loss'].item()
            num_batches += 1
    
    model.train()
    
    return {
        'eval_loss': total_loss / num_batches,
        'eval_equivalence_loss': total_equivalence_loss / num_batches,
        'eval_regularization_loss': total_regularization_loss / num_batches,
    }


def train_npt_equivalence(config_path: str = "config.yaml"):
    """Main training function."""
    # Load configuration
    config = load_config(config_path)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        mixed_precision='no',  # Use FP32 for stability in initial training
    )
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create output directories
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    
    # Initialize wandb
    if accelerator.is_main_process:
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            config=config,
            tags=config['wandb']['tags'],
            name=f"npt_equivalence_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['base_model_name'],
        cache_dir=config['paths']['cache_dir'],
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create NPT model
    accelerator.print("Loading base model and creating NPT architecture...")
    model = NPTModelWrapper(
        base_model_name=config['model']['base_model_name'],
        npt_layers=config['model']['npt_layers'],
        rank=config['model']['rank'],
        modulation_scale=config['model']['modulation_scale'],
        cache_dir=config['paths']['cache_dir'],
    )
    
    # Prepare datasets
    accelerator.print("Preparing datasets...")
    train_dataset, eval_dataset = prepare_dataset(config, tokenizer)
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=lambda x: {
            'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in x]),
            'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in x]),
        }
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=lambda x: {
            'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in x]),
            'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in x]),
        }
    )
    
    # Create loss function
    loss_fn = EquivalenceLoss(
        equivalence_weight=config['training']['equivalence_weight'],
        regularization_weight=config['training']['regularization_weight'],
    )
    
    # Create optimizer (only for NPT parameters)
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # Create learning rate scheduler
    num_training_steps = (
        len(train_dataloader) * config['training']['num_epochs'] // 
        config['training']['gradient_accumulation_steps']
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=num_training_steps,
    )
    
    # Prepare for distributed training
    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )
    
    # Training loop
    accelerator.print("Starting training...")
    global_step = 0
    best_eval_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{config['training']['num_epochs']}",
            disable=not accelerator.is_local_main_process,
        )
        
        for step, batch in enumerate(progress_bar):
            # Forward pass with original outputs for comparison
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                return_original_outputs=True,
            )
            
            # Compute loss
            loss_dict = loss_fn(outputs['layer_outputs'], model)
            loss = loss_dict['loss']
            
            # Scale loss for gradient accumulation
            loss = loss / config['training']['gradient_accumulation_steps']
            accelerator.backward(loss)
            
            # Gradient accumulation
            if (step + 1) % config['training']['gradient_accumulation_steps'] == 0:
                # Gradient clipping
                if config['training']['max_grad_norm'] > 0:
                    accelerator.clip_grad_norm_(
                        model.get_trainable_parameters(),
                        config['training']['max_grad_norm'],
                    )
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % config['training']['logging_steps'] == 0:
                    # Get weight delta statistics
                    weight_stats = model.get_weight_delta_statistics()
                    avg_delta_norm = np.mean([
                        stats['delta_w_frobenius'] 
                        for stats in weight_stats.values()
                    ])
                    
                    log_dict = {
                        'train/loss': loss_dict['loss'].item(),
                        'train/equivalence_loss': loss_dict['equivalence_loss'].item(),
                        'train/regularization_loss': loss_dict['regularization_loss'].item(),
                        'train/learning_rate': scheduler.get_last_lr()[0],
                        'train/avg_delta_norm': avg_delta_norm,
                        'train/epoch': epoch,
                        'train/global_step': global_step,
                    }
                    
                    if accelerator.is_main_process:
                        wandb.log(log_dict, step=global_step)
                
                # Generate sample predictions
                if global_step % config['training']['prediction_logging_steps'] == 0:
                    if accelerator.is_main_process:
                        accelerator.print("\nGenerating sample predictions...")
                        
                        # Unwrap model for generation
                        unwrapped_model = accelerator.unwrap_model(model)
                        
                        # Generate with different prompts
                        prompts = [
                            "The future of artificial intelligence is",
                            "Once upon a time, there was a",
                            "The most important discovery in science is",
                            "In the year 2050, humanity will",
                        ]
                        
                        predictions = {}
                        for i, prompt in enumerate(prompts):
                            generated = generate_sample_predictions(
                                unwrapped_model,
                                tokenizer,
                                prompt=prompt,
                                max_new_tokens=50,
                            )
                            predictions[f"prompt_{i}"] = f"{prompt} → {generated}"
                            accelerator.print(f"\nPrompt {i}: {generated}")
                        
                        # Log to wandb
                        wandb.log({
                            "predictions": wandb.Table(
                                columns=["prompt_id", "generated_text"],
                                data=[[k, v] for k, v in predictions.items()]
                            )
                        }, step=global_step)
                
                # Evaluation
                if global_step % config['training']['eval_steps'] == 0:
                    accelerator.print("\nRunning evaluation...")
                    eval_metrics = evaluate_model(
                        model, eval_dataloader, loss_fn, accelerator
                    )
                    
                    if accelerator.is_main_process:
                        # Log evaluation metrics
                        eval_log = {
                            f"eval/{k}": v for k, v in eval_metrics.items()
                        }
                        wandb.log(eval_log, step=global_step)
                        
                        # Save best model
                        if eval_metrics['eval_loss'] < best_eval_loss:
                            best_eval_loss = eval_metrics['eval_loss']
                            save_path = os.path.join(
                                config['paths']['checkpoint_dir'],
                                'best_model.pt'
                            )
                            accelerator.unwrap_model(model).save_npt_components(save_path)
                            accelerator.print(f"Saved best model to {save_path}")
                
                # Regular checkpointing
                if global_step % config['training']['save_steps'] == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            config['paths']['checkpoint_dir'],
                            f'checkpoint_step_{global_step}.pt'
                        )
                        accelerator.unwrap_model(model).save_npt_components(save_path)
                        accelerator.print(f"Saved checkpoint to {save_path}")
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss_dict['loss'].item(),
                'eq_loss': loss_dict['equivalence_loss'].item(),
            })
            epoch_loss += loss_dict['loss'].item()
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        accelerator.print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
    
    # Final evaluation
    accelerator.print("\nRunning final evaluation...")
    final_metrics = evaluate_model(model, eval_dataloader, loss_fn, accelerator)
    
    if accelerator.is_main_process:
        # Log final metrics
        final_log = {f"final/{k}": v for k, v in final_metrics.items()}
        wandb.log(final_log, step=global_step)
        
        # Save final model
        save_path = os.path.join(config['paths']['checkpoint_dir'], 'final_model.pt')
        accelerator.unwrap_model(model).save_npt_components(save_path)
        accelerator.print(f"Saved final model to {save_path}")
        
        # Close wandb
        wandb.finish()
    
    accelerator.print("Training completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train NPT Equivalence Pre-training")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    
    args = parser.parse_args()
    train_npt_equivalence(args.config)

"""
Phase 2: Functional Fine-tuning for NPT Model

This script fine-tunes the NPT model (with pre-trained adapters from Phase 1)
on downstream instruction-following tasks to evaluate its performance.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    DataCollatorForLanguageModeling
)
from accelerate import Accelerator
from tqdm import tqdm
import logging
from typing import Optional, Dict
import json

from model import get_adapter_params
from utils import (
    setup_logging,
    get_quantization_config,
    load_finetuning_dataset,
    preprocess_finetuning_data,
    create_dataloader,
    save_checkpoint,
    load_checkpoint,
    MetricsLogger,
    compute_gradient_norm,
    get_optimizer,
    get_scheduler
)


class FunctionalTrainer:
    """Trainer for Phase 2 functional fine-tuning."""
    
    def __init__(self, args):
        self.args = args
        self.logger = setup_logging(args.log_file)
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=["wandb"] if args.use_wandb else None,
        )
        
        # Initialize metrics logger
        self.metrics_logger = MetricsLogger(
            use_wandb=args.use_wandb,
            use_tensorboard=args.use_tensorboard,
            project_name="npt-functional-finetuning",
            run_name=args.run_name,
            config=vars(args)
        )
        
        # Load model and tokenizer
        self.setup_model()
        
        # Load dataset and create dataloaders
        self.setup_data()
        
        # Setup optimizer and scheduler
        self.setup_training()
    
    def setup_model(self):
        """Load NPT model from Phase 1 checkpoint."""
        self.logger.info(f"Loading NPT model from: {self.args.checkpoint_path}")
        
        # Load training info if available
        training_info_path = os.path.join(self.args.checkpoint_path, "training_info.pt")
        if os.path.exists(training_info_path):
            training_info = torch.load(training_info_path, map_location="cpu")
            adapter_config = training_info.get('adapter_config', {})
            self.logger.info(f"Loaded adapter config: {adapter_config}")
        else:
            # Default config
            adapter_config = {'r': 16}
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.checkpoint_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        quantization_config = get_quantization_config() if self.args.use_quantization else None
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.checkpoint_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16 if self.args.use_fp16 else torch.float32,
            trust_remote_code=True
        )
        
        # Ensure only adapter parameters are trainable
        for name, param in self.model.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Log parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    def setup_data(self):
        """Setup dataset and dataloaders."""
        self.logger.info("Loading fine-tuning dataset...")
        
        # Load training dataset
        train_dataset = load_finetuning_dataset(
            dataset_name=self.args.dataset_name,
            split=self.args.train_split
        )
        
        # Load validation dataset if specified
        if self.args.eval_split:
            eval_dataset = load_finetuning_dataset(
                dataset_name=self.args.dataset_name,
                split=self.args.eval_split
            )
        else:
            eval_dataset = None
        
        # Limit dataset size if specified
        if self.args.num_train_samples and self.args.num_train_samples < len(train_dataset):
            train_dataset = train_dataset.select(range(self.args.num_train_samples))
        
        if eval_dataset and self.args.num_eval_samples and self.args.num_eval_samples < len(eval_dataset):
            eval_dataset = eval_dataset.select(range(self.args.num_eval_samples))
        
        # Create dataloaders
        self.train_dataloader = create_dataloader(
            dataset=train_dataset,
            tokenizer=self.tokenizer,
            batch_size=self.args.batch_size,
            max_length=self.args.max_length,
            is_training=True,
            preprocess_function=preprocess_finetuning_data
        )
        
        if eval_dataset:
            self.eval_dataloader = create_dataloader(
                dataset=eval_dataset,
                tokenizer=self.tokenizer,
                batch_size=self.args.eval_batch_size,
                max_length=self.args.max_length,
                is_training=False,
                preprocess_function=preprocess_finetuning_data
            )
        else:
            self.eval_dataloader = None
        
        # Prepare with accelerator
        if self.eval_dataloader:
            self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
                self.train_dataloader, self.eval_dataloader
            )
        else:
            self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
        
        self.logger.info(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            self.logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    def setup_training(self):
        """Setup optimizer and scheduler."""
        # Get adapter parameters
        adapter_params = get_adapter_params(self.model)
        
        # Create optimizer
        self.optimizer = get_optimizer(
            params=adapter_params,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            optimizer_type=self.args.optimizer_type
        )
        
        # Calculate training steps
        if self.args.max_steps > 0:
            self.num_training_steps = self.args.max_steps
        else:
            self.num_training_steps = (
                len(self.train_dataloader) * self.args.num_epochs
            ) // self.args.gradient_accumulation_steps
        
        # Create scheduler
        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            num_training_steps=self.num_training_steps,
            num_warmup_steps=self.args.warmup_steps,
            scheduler_type=self.args.scheduler_type
        )
        
        # Prepare with accelerator
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )
    
    def compute_loss(self, batch):
        """Compute language modeling loss."""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        lm_loss = outputs.loss
        
        # Optionally add regularization loss
        if self.args.use_regularization:
            # Compute regularization loss for delta_W norms
            reg_loss = self.compute_regularization_loss(input_ids)
            total_loss = lm_loss + self.args.regularization_lambda * reg_loss
            
            return total_loss, lm_loss, reg_loss
        else:
            return lm_loss, lm_loss, torch.tensor(0.0)
    
    def compute_regularization_loss(self, input_ids):
        """Compute weight delta regularization loss."""
        # Similar to pretrain_npt.py but simplified
        total_norm = 0.0
        num_layers = 0
        
        # Get hidden states
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states
        
        # Compute norm for each NPT layer
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'adapter'):
                # Simple approximation: use adapter weight norms
                norm = (
                    torch.norm(layer.adapter.A_proj.weight, p='fro') ** 2 +
                    torch.norm(layer.adapter.B_proj.weight, p='fro') ** 2
                )
                total_norm += norm
                num_layers += 1
        
        # Average norm
        avg_norm = total_norm / num_layers if num_layers > 0 else torch.tensor(0.0)
        
        return avg_norm
    
    def train_step(self, batch):
        """Perform a single training step."""
        # Compute loss
        total_loss, lm_loss, reg_loss = self.compute_loss(batch)
        
        # Backward pass
        self.accelerator.backward(total_loss)
        
        # Gradient clipping
        if self.args.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(),
                self.args.max_grad_norm
            )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Compute metrics
        metrics = {
            'loss/total': total_loss.item(),
            'loss/lm': lm_loss.item(),
            'loss/regularization': reg_loss.item() if self.args.use_regularization else 0.0,
            'learning_rate': self.scheduler.get_last_lr()[0],
            'grad_norm': compute_gradient_norm(self.model)
        }
        
        return metrics
    
    def evaluate(self):
        """Evaluate model on validation set."""
        if not self.eval_dataloader:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_lm_loss = 0.0
        total_reg_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(
                self.eval_dataloader,
                desc="Evaluating",
                disable=not self.accelerator.is_local_main_process
            ):
                total_loss_batch, lm_loss_batch, reg_loss_batch = self.compute_loss(batch)
                
                total_loss += total_loss_batch.item()
                total_lm_loss += lm_loss_batch.item()
                total_reg_loss += reg_loss_batch.item() if self.args.use_regularization else 0.0
                total_steps += 1
        
        # Average metrics
        metrics = {
            'eval/loss_total': total_loss / total_steps,
            'eval/loss_lm': total_lm_loss / total_steps,
            'eval/loss_regularization': total_reg_loss / total_steps if self.args.use_regularization else 0.0,
            'eval/perplexity': torch.exp(torch.tensor(total_lm_loss / total_steps)).item()
        }
        
        self.model.train()
        return metrics
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting functional fine-tuning...")
        
        global_step = 0
        best_eval_loss = float('inf')
        
        for epoch in range(self.args.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}")
            
            # Training loop
            epoch_metrics = {
                'loss/total': 0.0,
                'loss/lm': 0.0,
                'loss/regularization': 0.0
            }
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Training epoch {epoch + 1}",
                disable=not self.accelerator.is_local_main_process
            )
            
            for step, batch in enumerate(progress_bar):
                # Train step
                metrics = self.train_step(batch)
                
                # Update epoch metrics
                for key, value in metrics.items():
                    if key.startswith('loss/'):
                        epoch_metrics[key] += value
                
                # Log metrics
                if global_step % self.args.log_steps == 0:
                    self.metrics_logger.log(metrics, step=global_step)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': metrics['loss/total'],
                        'lm': metrics['loss/lm'],
                        'lr': metrics['learning_rate']
                    })
                
                # Evaluate
                if self.args.eval_steps > 0 and global_step % self.args.eval_steps == 0 and global_step > 0:
                    eval_metrics = self.evaluate()
                    self.metrics_logger.log(eval_metrics, step=global_step)
                    
                    self.logger.info(
                        f"Step {global_step} - Eval loss: {eval_metrics['eval/loss_lm']:.4f}, "
                        f"Eval perplexity: {eval_metrics['eval/perplexity']:.2f}"
                    )
                    
                    # Save best checkpoint
                    if eval_metrics['eval/loss_lm'] < best_eval_loss:
                        best_eval_loss = eval_metrics['eval/loss_lm']
                        self.save_checkpoint(global_step, is_best=True)
                
                # Save checkpoint
                if global_step % self.args.save_steps == 0 and global_step > 0:
                    self.save_checkpoint(global_step)
                
                global_step += 1
                
                # Check if we've reached max steps
                if self.args.max_steps > 0 and global_step >= self.args.max_steps:
                    break
            
            # Log epoch metrics
            num_steps = step + 1
            for key in epoch_metrics:
                epoch_metrics[key] /= num_steps
            
            self.logger.info(
                f"Epoch {epoch + 1} - "
                f"Loss: {epoch_metrics['loss/total']:.4f}, "
                f"LM Loss: {epoch_metrics['loss/lm']:.4f}"
            )
            
            # Evaluate at end of epoch
            if self.eval_dataloader:
                eval_metrics = self.evaluate()
                self.metrics_logger.log(eval_metrics, step=global_step)
                
                self.logger.info(
                    f"Epoch {epoch + 1} - Eval loss: {eval_metrics['eval/loss_lm']:.4f}, "
                    f"Eval perplexity: {eval_metrics['eval/perplexity']:.2f}"
                )
                
                # Save best checkpoint
                if eval_metrics['eval/loss_lm'] < best_eval_loss:
                    best_eval_loss = eval_metrics['eval/loss_lm']
                    self.save_checkpoint(global_step, is_best=True)
            
            # Check if we've reached max steps
            if self.args.max_steps > 0 and global_step >= self.args.max_steps:
                break
        
        # Save final checkpoint
        self.save_checkpoint(global_step, is_final=True)
        
        # Finish logging
        self.metrics_logger.finish()
        
        self.logger.info("Training completed!")
    
    def save_checkpoint(self, step: int, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint."""
        if is_best:
            save_path = os.path.join(self.args.output_dir, "checkpoint-best")
        elif is_final:
            save_path = os.path.join(self.args.output_dir, "checkpoint-final")
        else:
            save_path = os.path.join(self.args.output_dir, f"checkpoint-{step}")
        
        self.logger.info(f"Saving checkpoint to {save_path}")
        
        # Save checkpoint
        save_checkpoint(
            model=self.model,
            tokenizer=self.tokenizer,
            save_path=save_path,
            accelerator=self.accelerator,
            additional_info={
                'step': step,
                'args': self.args,
                'phase': 'functional_finetuning'
            }
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NPT Functional Fine-tuning"
    )
    
    # Model arguments
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to NPT model checkpoint from Phase 1"
    )
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use FP16 precision"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="Dataset to use for fine-tuning"
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train_sft",
        help="Training split name"
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test_sft",
        help="Evaluation split name (set to empty string to skip eval)"
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=None,
        help="Number of training samples to use (for debugging)"
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=None,
        help="Number of evaluation samples to use (for debugging)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Evaluation batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--use_regularization",
        action="store_true",
        help="Use delta_W norm regularization"
    )
    parser.add_argument(
        "--regularization_lambda",
        type=float,
        default=0.001,
        help="Regularization strength for delta_W norm"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum number of training steps (overrides num_epochs)"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping"
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adamw",
        choices=["adamw", "adam"],
        help="Optimizer type"
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "constant"],
        help="Learning rate scheduler type"
    )
    
    # Logging arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/npt-finetuned",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help="Log every N steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps (0 to only eval at epoch end)"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Log file path"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--use_tensorboard",
        action="store_true",
        help="Use TensorBoard for logging"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this training run"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = FunctionalTrainer(args)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

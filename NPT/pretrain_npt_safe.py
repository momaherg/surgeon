"""
Safe version of NPT pretraining with better numerical stability
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
from accelerate import Accelerator
from tqdm import tqdm
import logging
from typing import Optional
import warnings

from model import convert_llama_to_npt, get_adapter_params
from utils import (
    setup_logging,
    get_quantization_config,
    load_pretraining_dataset,
    create_dataloader,
    save_checkpoint,
    MetricsLogger,
    compute_gradient_norm,
    get_optimizer,
    get_scheduler
)


class SafeEquivalenceTrainer:
    """Trainer with enhanced numerical stability for Phase 1 equivalence pre-training."""
    
    def __init__(self, args):
        self.args = args
        self.logger = setup_logging(args.log_file)
        
        # Warn about potential issues
        if args.use_quantization and args.use_fp16:
            self.logger.warning("Using both quantization and FP16 can cause numerical instability. Consider using only one.")
        
        # Initialize accelerator - always disable mixed precision with quantization
        if args.use_quantization:
            mixed_precision = "no"
            self.logger.info("Disabling mixed precision due to quantization")
        else:
            mixed_precision = args.mixed_precision
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=["wandb"] if args.use_wandb else None,
        )
        
        # Initialize metrics logger
        self.metrics_logger = MetricsLogger(
            use_wandb=args.use_wandb,
            use_tensorboard=args.use_tensorboard,
            project_name="npt-safe-pretraining",
            run_name=args.run_name,
            config=vars(args)
        )
        
        # Load models and tokenizer
        self.setup_models()
        
        # Load dataset and create dataloader
        self.setup_data()
        
        # Setup optimizer and scheduler
        self.setup_training()
        
        # Initialize tracking variables
        self.nan_count = 0
        self.max_nan_count = 10  # Stop after this many NaN occurrences
    
    def setup_models(self):
        """Load and setup teacher and student models with enhanced stability."""
        self.logger.info(f"Loading base model: {self.args.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load configuration
        config = AutoConfig.from_pretrained(self.args.model_name)
        
        # Determine dtype - use FP32 for stability with quantized models
        if self.args.use_quantization:
            model_dtype = torch.float32
            self.logger.info("Using FP32 for model dtype with quantization")
        elif self.args.use_fp16:
            model_dtype = torch.float16
        else:
            model_dtype = torch.float32
        
        # Load teacher model
        quantization_config = get_quantization_config() if self.args.use_quantization else None
        
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            config=config,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=model_dtype
        )
        self.teacher_model.eval()
        
        # Load student model
        self.student_model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            config=config,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=model_dtype
        )
        
        # Share embeddings if requested
        if self.args.share_embeddings:
            self.student_model.model.embed_tokens = self.teacher_model.model.embed_tokens
            self.student_model.lm_head = self.teacher_model.lm_head
            self.logger.info("Sharing embeddings between teacher and student")
        
        # Convert student model to NPT with safe configuration
        self.logger.info("Converting model to NPT architecture...")
        adapter_config = {
            'r': self.args.adapter_rank,
            'd_model': config.hidden_size,
            'd_ffn': config.intermediate_size,
            'compute_dtype': torch.float32,  # Always use FP32 for adapters
            'modulation_type': 'additive' if self.args.safe_mode else 'both'
        }
        self.student_model = convert_llama_to_npt(self.student_model, adapter_config)
        
        # Log parameter counts
        total_params = sum(p.numel() for p in self.student_model.parameters())
        trainable_params = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    def setup_data(self):
        """Setup dataset and dataloader."""
        self.logger.info("Loading dataset...")
        
        # Load dataset
        dataset = load_pretraining_dataset(
            dataset_name=self.args.dataset_name,
            split=self.args.dataset_split,
            streaming=self.args.streaming,
            num_samples=self.args.num_samples
        )
        
        # Create dataloader
        self.train_dataloader = create_dataloader(
            dataset=dataset,
            tokenizer=self.tokenizer,
            batch_size=self.args.batch_size,
            max_length=self.args.max_length,
            is_training=True,
            preprocess_function=None
        )
        
        # Prepare with accelerator
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
    
    def setup_training(self):
        """Setup optimizer and scheduler with safe defaults."""
        # Get adapter parameters
        adapter_params = get_adapter_params(self.student_model)
        
        # Use smaller learning rate for stability
        safe_lr = min(self.args.learning_rate, 5e-5)
        if safe_lr < self.args.learning_rate:
            self.logger.warning(f"Reducing learning rate from {self.args.learning_rate} to {safe_lr} for stability")
        
        # Create optimizer
        self.optimizer = get_optimizer(
            params=adapter_params,
            learning_rate=safe_lr,
            weight_decay=self.args.weight_decay,
            optimizer_type=self.args.optimizer_type
        )
        
        # Calculate training steps
        if self.args.max_steps > 0:
            self.num_training_steps = self.args.max_steps
        else:
            if self.args.streaming:
                self.num_training_steps = 10000
                self.args.max_steps = 10000
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
        self.student_model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.student_model, self.optimizer, self.scheduler
        )
    
    def compute_safe_equivalence_loss(self, batch):
        """Compute MSE loss with numerical stability checks."""
        # Move batch to device
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Ensure proper dtype
        if attention_mask is not None:
            if self.args.use_quantization:
                attention_mask = attention_mask.to(dtype=torch.float32)
            else:
                attention_mask = attention_mask.to(dtype=self.teacher_model.dtype)
        
        try:
            # Teacher forward pass (no grad)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                teacher_hidden_states = teacher_outputs.hidden_states
            
            # Student forward pass
            student_outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Extract hidden states and regularization
            all_hidden_states = []
            reg_norms = []
            
            # For NPT model, we need to manually track outputs
            hidden_states = self.student_model.model.embed_tokens(input_ids)
            all_hidden_states.append(hidden_states)
            
            # Pass through layers
            for i, layer in enumerate(self.student_model.model.layers):
                # Create position ids
                batch_size, seq_length = input_ids.shape
                device = input_ids.device
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                
                # Forward through layer
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )
                
                hidden_states = layer_outputs[0]
                all_hidden_states.append(hidden_states)
                
                # Collect regularization if available
                if len(layer_outputs) > 1:
                    reg_norms.append(layer_outputs[1])
            
            # Apply final layer norm
            hidden_states = self.student_model.model.norm(hidden_states)
            
            # Compute MSE loss with gradient clipping per layer
            mse_losses = []
            num_layers = min(len(teacher_hidden_states) - 1, len(all_hidden_states) - 1)
            
            for i in range(1, num_layers + 1):
                teacher_hidden = teacher_hidden_states[i].detach()
                student_hidden = all_hidden_states[i]
                
                # Check for NaN in hidden states
                if torch.isnan(student_hidden).any() or torch.isnan(teacher_hidden).any():
                    self.logger.error(f"NaN detected in hidden states at layer {i}")
                    raise ValueError("NaN in hidden states")
                
                # Normalize to prevent extreme values
                teacher_norm = teacher_hidden.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                student_norm = student_hidden.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                
                teacher_normalized = teacher_hidden / teacher_norm
                student_normalized = student_hidden / student_norm
                
                # Compute normalized MSE
                layer_mse = nn.functional.mse_loss(student_normalized, teacher_normalized)
                mse_losses.append(layer_mse)
            
            # Average MSE loss
            mse_loss = torch.stack(mse_losses).mean()
            
            # Average regularization
            if reg_norms:
                avg_reg_norm = torch.stack(reg_norms).mean()
            else:
                avg_reg_norm = torch.tensor(0.0, device=mse_loss.device)
            
            # Check for NaN in losses
            if torch.isnan(mse_loss) or torch.isnan(avg_reg_norm):
                raise ValueError("NaN in loss computation")
            
            return mse_loss, avg_reg_norm
            
        except Exception as e:
            self.logger.error(f"Error in loss computation: {str(e)}")
            # Return safe dummy losses
            return torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)
    
    def train_step(self, batch):
        """Perform a single training step with safety checks."""
        try:
            # Compute losses
            mse_loss, reg_norm = self.compute_safe_equivalence_loss(batch)
            
            # Total loss with regularization
            total_loss = mse_loss + self.args.regularization_lambda * reg_norm
            
            # Check for NaN
            if torch.isnan(total_loss):
                self.nan_count += 1
                self.logger.warning(f"NaN loss detected (count: {self.nan_count})")
                
                if self.nan_count >= self.max_nan_count:
                    raise ValueError("Too many NaN losses, stopping training")
                
                # Skip this step
                return None
            
            # Reset NaN count on successful step
            self.nan_count = 0
            
            # Backward pass
            self.accelerator.backward(total_loss)
            
            # Aggressive gradient clipping
            self.accelerator.clip_grad_norm_(
                self.student_model.parameters(),
                max_norm=0.5  # More aggressive than default
            )
            
            # Check gradient norm
            grad_norm = compute_gradient_norm(self.student_model)
            if grad_norm > 100:
                self.logger.warning(f"Large gradient norm: {grad_norm:.2f}")
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Compute metrics
            metrics = {
                'loss/total': total_loss.item(),
                'loss/mse': mse_loss.item(),
                'loss/regularization': reg_norm.item(),
                'learning_rate': self.scheduler.get_last_lr()[0],
                'grad_norm': grad_norm
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in training step: {str(e)}")
            self.optimizer.zero_grad()
            return None
    
    def train(self):
        """Main training loop with safety checks."""
        self.logger.info("Starting safe equivalence pre-training...")
        
        global_step = 0
        best_loss = float('inf')
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        for epoch in range(self.args.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}")
            
            epoch_metrics = {
                'loss/total': 0.0,
                'loss/mse': 0.0,
                'loss/regularization': 0.0
            }
            valid_steps = 0
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Training epoch {epoch + 1}",
                disable=not self.accelerator.is_local_main_process
            )
            
            for step, batch in enumerate(progress_bar):
                # Train step with error handling
                metrics = self.train_step(batch)
                
                if metrics is None:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error("Too many consecutive errors, stopping training")
                        return
                    continue
                
                # Reset error count on success
                consecutive_errors = 0
                valid_steps += 1
                
                # Update epoch metrics
                for key, value in metrics.items():
                    if key.startswith('loss/'):
                        epoch_metrics[key] += value
                
                # Log metrics
                if global_step % self.args.log_steps == 0:
                    self.metrics_logger.log(metrics, step=global_step)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{metrics['loss/total']:.4f}",
                        'mse': f"{metrics['loss/mse']:.4f}",
                        'reg': f"{metrics['loss/regularization']:.4f}"
                    })
                
                # Save checkpoint
                if global_step % self.args.save_steps == 0 and global_step > 0:
                    self.save_checkpoint(global_step)
                
                global_step += 1
                
                # Check max steps
                if self.args.max_steps > 0 and global_step >= self.args.max_steps:
                    break
            
            # Log epoch metrics
            if valid_steps > 0:
                for key in epoch_metrics:
                    epoch_metrics[key] /= valid_steps
                
                self.logger.info(
                    f"Epoch {epoch + 1} - "
                    f"Loss: {epoch_metrics['loss/total']:.4f}, "
                    f"MSE: {epoch_metrics['loss/mse']:.4f}, "
                    f"Reg: {epoch_metrics['loss/regularization']:.4f} "
                    f"(Valid steps: {valid_steps})"
                )
                
                # Save best checkpoint
                if epoch_metrics['loss/total'] < best_loss:
                    best_loss = epoch_metrics['loss/total']
                    self.save_checkpoint(global_step, is_best=True)
            
            # Check max steps
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
            model=self.student_model,
            tokenizer=self.tokenizer,
            save_path=save_path,
            accelerator=self.accelerator,
            additional_info={
                'step': step,
                'args': self.args,
                'adapter_config': {
                    'r': self.args.adapter_rank,
                    'd_model': self.student_model.config.hidden_size,
                    'd_ffn': self.student_model.config.intermediate_size
                }
            }
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Safe NPT Equivalence Pre-training"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Base model to use"
    )
    parser.add_argument(
        "--adapter_rank",
        type=int,
        default=16,
        help="Rank for low-rank adapter factorization"
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
    parser.add_argument(
        "--share_embeddings",
        action="store_true",
        help="Share embeddings between teacher and student"
    )
    parser.add_argument(
        "--safe_mode",
        action="store_true",
        help="Use only additive modulation for extra stability"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cerebras/SlimPajama-627B",
        help="Dataset to use for pre-training"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use dataset streaming"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use (for debugging)"
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
        help="Batch size per device"
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
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--regularization_lambda",
        type=float,
        default=0.01,
        help="Regularization strength for delta_W norm"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum number of training steps"
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
        default="./outputs/npt-safe-pretrained",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help="Log every N steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
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
        default="no",
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
    trainer = SafeEquivalenceTrainer(args)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

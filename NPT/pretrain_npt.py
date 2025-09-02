"""
Phase 1: Equivalence Pre-training for NPT Model

This script trains the adapter modules in the NPT model to produce outputs
equivalent to the original Llama3 model while learning efficient weight deltas.
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


class EquivalenceTrainer:
    """Trainer for Phase 1 equivalence pre-training."""
    
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
            project_name="npt-equivalence-pretraining",
            run_name=args.run_name,
            config=vars(args)
        )
        
        # Load models and tokenizer
        self.setup_models()
        
        # Load dataset and create dataloader
        self.setup_data()
        
        # Setup optimizer and scheduler
        self.setup_training()
    
    def setup_models(self):
        """Load and setup teacher and student models."""
        self.logger.info(f"Loading base model: {self.args.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load configuration
        config = AutoConfig.from_pretrained(self.args.model_name)
        
        # Load teacher model (original Llama3)
        quantization_config = get_quantization_config() if self.args.use_quantization else None
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            config=config,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16 if self.args.use_fp16 else torch.float32
        )
        self.teacher_model.eval()
        
        # Load student model (will be converted to NPT)
        self.student_model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            config=config,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16 if self.args.use_fp16 else torch.float32
        )
        
        # Convert student model to NPT
        self.logger.info("Converting model to NPT architecture...")
        adapter_config = {
            'r': self.args.adapter_rank,
            'd_model': config.hidden_size,
            'd_ffn': config.intermediate_size,
            'compute_dtype': torch.float16 if self.args.use_fp16 else torch.float32
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
            preprocess_function=None  # Let the custom collator handle tokenization
        )
        
        # Prepare with accelerator
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
    
    def setup_training(self):
        """Setup optimizer and scheduler."""
        # Get adapter parameters
        adapter_params = get_adapter_params(self.student_model)
        
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
        self.student_model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.student_model, self.optimizer, self.scheduler
        )
    
    def compute_equivalence_loss(self, batch):
        """Compute MSE loss between teacher and student outputs."""
        # Move batch to device
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Convert attention_mask to appropriate dtype
        # The model expects float dtype when using float16/float32
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.float16 if self.args.use_fp16 else torch.float32)
        
        # Teacher forward pass (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            teacher_hidden_states = teacher_outputs.hidden_states
        
        # Student forward pass with custom NPT layers
        # We need to manually collect hidden states and reg norms
        hidden_states = input_ids
        all_hidden_states = []
        reg_norms = []
        
        # Get embeddings
        inputs_embeds = self.student_model.model.embed_tokens(hidden_states)
        all_hidden_states.append(inputs_embeds)
        hidden_states = inputs_embeds
        
        # Get batch size and sequence length
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Create position_ids if not provided
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Check if the model has rotary embeddings
        if hasattr(self.student_model.model, 'rotary_emb'):
            # Compute rotary embeddings
            position_embeddings = self.student_model.model.rotary_emb(hidden_states, position_ids)
        else:
            # For older versions or models without rotary embeddings
            position_embeddings = None
        
        # Pass through each layer
        for layer in self.student_model.model.layers:
            layer_outputs = layer(
                hidden_states, 
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings
            )
            hidden_states = layer_outputs[0]
            all_hidden_states.append(hidden_states)
            
            # Collect regularization norm if this is an NPT layer
            if len(layer_outputs) > 1 and isinstance(layer_outputs[1], torch.Tensor):
                reg_norms.append(layer_outputs[1])
        
        # Apply final layer norm
        hidden_states = self.student_model.model.norm(hidden_states)
        
        # Compute MSE loss for each layer
        mse_loss = 0.0
        num_layers = len(teacher_hidden_states) - 1  # Exclude embedding layer
        
        for i in range(1, min(num_layers + 1, len(all_hidden_states))):  # Skip embedding layer
            teacher_hidden = teacher_hidden_states[i].detach()
            student_hidden = all_hidden_states[i]
            
            # MSE loss
            layer_mse = nn.functional.mse_loss(student_hidden, teacher_hidden)
            mse_loss += layer_mse
        
        # Average across layers
        mse_loss = mse_loss / num_layers
        
        # Average regularization norms
        if reg_norms:
            avg_reg_norm = torch.stack(reg_norms).mean()
        else:
            avg_reg_norm = torch.tensor(0.0, device=mse_loss.device)
        
        return mse_loss, avg_reg_norm
    

    
    def train_step(self, batch):
        """Perform a single training step."""
        # Compute equivalence loss and regularization norm
        mse_loss, reg_norm = self.compute_equivalence_loss(batch)
        
        # Total loss with regularization
        total_loss = mse_loss + self.args.regularization_lambda * reg_norm
        
        # Backward pass
        self.accelerator.backward(total_loss)
        
        # Gradient clipping
        if self.args.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(
                self.student_model.parameters(),
                self.args.max_grad_norm
            )
        
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
            'grad_norm': compute_gradient_norm(self.student_model)
        }
        
        return metrics
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting equivalence pre-training...")
        
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(self.args.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}")
            
            # Training loop
            epoch_metrics = {
                'loss/total': 0.0,
                'loss/mse': 0.0,
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
                        'mse': metrics['loss/mse'],
                        'reg': metrics['loss/regularization']
                    })
                
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
                f"MSE: {epoch_metrics['loss/mse']:.4f}, "
                f"Reg: {epoch_metrics['loss/regularization']:.4f}"
            )
            
            # Save best checkpoint
            if epoch_metrics['loss/total'] < best_loss:
                best_loss = epoch_metrics['loss/total']
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
        description="NPT Equivalence Pre-training"
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
        default="./outputs/npt-pretrained",
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
    trainer = EquivalenceTrainer(args)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

"""
NPT pretraining with mixed teacher forcing and curriculum learning.

This implementation gradually transitions from teacher-guided to student-guided
training, teaching the model to handle its own outputs during inference.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
from accelerate import Accelerator
from tqdm import tqdm
import logging
from typing import Optional, Tuple, Dict
import warnings
import numpy as np

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
from improved_loss import create_improved_loss


class MixedTeacherForcingNPTTrainer:
    """NPT Trainer with mixed teacher forcing and curriculum learning."""
    
    def __init__(self, args):
        self.args = args
        self.logger = setup_logging(args.log_file)
        
        # Initialize accelerator
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
            project_name="npt-mixed-teacher-forcing",
            run_name=args.run_name,
            config=vars(args)
        )
        
        # Load models and tokenizer
        self.setup_models()
        
        # Setup loss functions
        self.setup_loss_functions()
        
        # Load dataset and create dataloader
        self.setup_data()
        
        # Setup optimizer and scheduler
        self.setup_training()
        
        # Initialize tracking variables
        self.global_step = 0
        self.best_loss = float('inf')
        self.nan_count = 0
        self.max_nan_count = 10
        
        # Curriculum learning schedule
        self.curriculum_steps = args.curriculum_steps
        self.initial_teacher_ratio = args.initial_teacher_ratio
        self.final_teacher_ratio = args.final_teacher_ratio
        
        # Sample prompts for monitoring
        self.sample_prompts = [
            "The capital of France is",
            "2 + 2 equals",
            "The largest planet in our solar system is",
            "Machine learning is",
            "In order to succeed, you must"
        ]
    
    def setup_models(self):
        """Load and setup teacher and student models."""
        self.logger.info(f"Loading base model: {self.args.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load configuration
        self.config = AutoConfig.from_pretrained(self.args.model_name)
        
        # Determine dtype
        if self.args.use_quantization:
            model_dtype = torch.float32
        elif self.args.use_fp16:
            model_dtype = torch.float16
        else:
            model_dtype = torch.float32
        
        # Load teacher model
        quantization_config = get_quantization_config() if self.args.use_quantization else None
        
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            config=self.config,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=model_dtype
        )
        self.teacher_model.eval()
        
        # Load student model
        self.student_model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            config=self.config,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=model_dtype
        )
        
        # Share embeddings if requested
        if self.args.share_embeddings:
            self.student_model.model.embed_tokens = self.teacher_model.model.embed_tokens
            self.student_model.lm_head = self.teacher_model.lm_head
            self.logger.info("Sharing embeddings between teacher and student")
        
        # Convert student model to NPT
        self.logger.info("Converting model to NPT architecture...")
        adapter_config = {
            'r': self.args.adapter_rank,
            'd_model': self.config.hidden_size,
            'd_ffn': self.config.intermediate_size,
            'modulation_type': 'outer_product',
            'modulation_scale': self.args.modulation_scale,
            'init_strategy': self.args.init_strategy,
            'init_scale': self.args.init_scale
        }
        self.student_model = convert_llama_to_npt(self.student_model, adapter_config)
        
        # Log parameter counts
        total_params = sum(p.numel() for p in self.student_model.parameters())
        trainable_params = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    def setup_loss_functions(self):
        """Setup loss functions for hidden states and output logits."""
        self.logger.info("Setting up loss functions")
        
        training_config = {
            "mse_weight": self.args.mse_weight,
            "cosine_weight": self.args.cosine_weight,
            "regularization_lambda": self.args.regularization_lambda,
            "gradient_penalty_lambda": self.args.gradient_penalty_lambda,
            "distill_temperature": self.args.distill_temperature,
            "use_adaptive_weights": self.args.use_adaptive_weights,
            "use_gradient_penalty": self.args.use_gradient_penalty,
            "warmup_steps": self.args.warmup_steps,
            "focal_gamma": self.args.focal_gamma,
            "smooth_beta": self.args.smooth_beta
        }
        
        # Hidden state loss
        self.hidden_loss_fn = create_improved_loss(
            model_config=self.config,
            training_config=training_config,
            loss_type=self.args.loss_type
        )
        
        # Output logits loss (KL divergence)
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        
        # Collect loss parameters if any
        self.loss_params = []
        if hasattr(self.hidden_loss_fn, 'layer_weights') and self.hidden_loss_fn.layer_weights is not None:
            self.loss_params = list(self.hidden_loss_fn.parameters())
    
    def get_teacher_forcing_ratio(self) -> float:
        """Get current teacher forcing ratio based on curriculum schedule."""
        if self.curriculum_steps <= 0:
            return self.final_teacher_ratio
        
        progress = min(1.0, self.global_step / self.curriculum_steps)
        
        # Linear decay from initial to final ratio
        ratio = self.initial_teacher_ratio - (self.initial_teacher_ratio - self.final_teacher_ratio) * progress
        
        return ratio
    
    def compute_mixed_loss(self, batch):
        """
        Compute loss with mixed teacher forcing.
        
        Randomly decides whether to use teacher-guided or student-guided
        forward pass based on curriculum schedule.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Get current teacher forcing ratio
        teacher_ratio = self.get_teacher_forcing_ratio()
        use_teacher = np.random.random() < teacher_ratio
        
        try:
            if use_teacher:
                # Teacher-guided forward pass
                loss_dict = self.compute_teacher_guided_loss(input_ids, attention_mask)
            else:
                # Student-guided forward pass (actual inference)
                loss_dict = self.compute_student_guided_loss(input_ids, attention_mask)
            
            # Add curriculum info to metrics
            loss_dict['teacher_ratio'] = teacher_ratio
            loss_dict['used_teacher'] = float(use_teacher)
            
            return loss_dict
            
        except Exception as e:
            self.logger.error(f"Error in loss computation: {str(e)}")
            device = input_ids.device
            return {
                'total_loss': torch.tensor(1e-4, device=device, requires_grad=True),
                'hidden_loss': torch.tensor(1e-4, device=device, requires_grad=True),
                'logits_loss': torch.tensor(1e-4, device=device, requires_grad=True),
                'reg_loss': torch.tensor(1e-6, device=device, requires_grad=True)
            }
    
    def compute_teacher_guided_loss(self, input_ids, attention_mask):
        """Compute loss with teacher-guided layer inputs."""
        # Get teacher outputs
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            teacher_hidden_states = teacher_outputs.hidden_states
            teacher_logits = teacher_outputs.logits
        
        # Process through student with teacher inputs at each layer
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Create position_ids
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get position embeddings if available
        position_embeddings = None
        if hasattr(self.student_model.model, 'rotary_emb'):
            cos, sin = self.student_model.model.rotary_emb(teacher_hidden_states[0], position_ids)
            position_embeddings = (cos, sin)
        
        # Process each layer with teacher inputs
        all_student_hidden_states = []
        reg_norms = []
        
        # Start with embeddings
        all_student_hidden_states.append(teacher_hidden_states[0])
        
        # Process each layer
        for i, layer in enumerate(self.student_model.model.layers):
            # Use teacher input for this layer
            teacher_input = teacher_hidden_states[i]
            
            layer_outputs = layer(
                teacher_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings
            )
            
            if isinstance(layer_outputs, tuple):
                student_output = layer_outputs[0]
                if len(layer_outputs) > 1 and isinstance(layer_outputs[1], torch.Tensor):
                    reg_norms.append(layer_outputs[1])
            else:
                student_output = layer_outputs
            
            all_student_hidden_states.append(student_output)
        
        # Apply final layer norm
        final_hidden = self.student_model.model.norm(all_student_hidden_states[-1])
        
        # Get student logits
        student_logits = self.student_model.lm_head(final_hidden)
        
        # Compute hidden state loss
        hidden_loss_dict = self.hidden_loss_fn(
            teacher_hidden_states=teacher_hidden_states[:-1],
            student_hidden_states=all_student_hidden_states,
            reg_norms=reg_norms,
            step=self.global_step
        )
        
        # Compute logits loss (KL divergence)
        student_log_probs = F.log_softmax(student_logits / self.args.distill_temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.args.distill_temperature, dim=-1)
        logits_loss = self.kl_loss_fn(student_log_probs, teacher_probs) * (self.args.distill_temperature ** 2)
        
        # Combine losses
        total_loss = (
            self.args.hidden_loss_weight * hidden_loss_dict['total_loss'] +
            self.args.logits_loss_weight * logits_loss
        )
        
        return {
            'total_loss': total_loss,
            'hidden_loss': hidden_loss_dict['total_loss'],
            'logits_loss': logits_loss,
            'mse_loss': hidden_loss_dict.get('mse_loss', 0.0),
            'cosine_loss': hidden_loss_dict.get('cosine_loss', 0.0),
            'reg_loss': hidden_loss_dict.get('reg_loss', 0.0),
            'mode': 'teacher_guided'
        }
    
    def compute_student_guided_loss(self, input_ids, attention_mask):
        """Compute loss with student's own outputs (actual inference)."""
        # Get teacher outputs
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            teacher_hidden_states = teacher_outputs.hidden_states
            teacher_logits = teacher_outputs.logits
        
        # Student forward pass (actual inference)
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        student_hidden_states = student_outputs.hidden_states
        student_logits = student_outputs.logits
        
        # Collect regularization norms
        reg_norms = []
        for layer in self.student_model.model.layers:
            if hasattr(layer, 'adapter'):
                # Dummy forward to get reg norm
                with torch.no_grad():
                    adapter_out = layer.adapter(student_hidden_states[0])
                    if 'reg_norm' in adapter_out:
                        reg_norms.append(adapter_out['reg_norm'])
        
        # Compute hidden state loss
        hidden_loss_dict = self.hidden_loss_fn(
            teacher_hidden_states=teacher_hidden_states,
            student_hidden_states=student_hidden_states,
            reg_norms=reg_norms,
            step=self.global_step
        )
        
        # Compute logits loss
        student_log_probs = F.log_softmax(student_logits / self.args.distill_temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.args.distill_temperature, dim=-1)
        logits_loss = self.kl_loss_fn(student_log_probs, teacher_probs) * (self.args.distill_temperature ** 2)
        
        # Combine losses with higher weight on logits for student-guided
        total_loss = (
            self.args.hidden_loss_weight * hidden_loss_dict['total_loss'] +
            self.args.logits_loss_weight * logits_loss * 2.0  # Double weight for student-guided
        )
        
        return {
            'total_loss': total_loss,
            'hidden_loss': hidden_loss_dict['total_loss'],
            'logits_loss': logits_loss,
            'mse_loss': hidden_loss_dict.get('mse_loss', 0.0),
            'cosine_loss': hidden_loss_dict.get('cosine_loss', 0.0),
            'reg_loss': hidden_loss_dict.get('reg_loss', 0.0),
            'mode': 'student_guided'
        }
    
    def setup_data(self):
        """Setup dataset and dataloader."""
        self.logger.info("Loading dataset...")
        
        dataset = load_pretraining_dataset(
            dataset_name=self.args.dataset_name,
            split=self.args.dataset_split,
            streaming=self.args.streaming,
            num_samples=self.args.num_samples
        )
        
        self.train_dataloader = create_dataloader(
            dataset=dataset,
            tokenizer=self.tokenizer,
            batch_size=self.args.batch_size,
            max_length=self.args.max_length,
            is_training=True,
            preprocess_function=None
        )
        
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
    
    def setup_training(self):
        """Setup optimizer and scheduler."""
        # Get adapter parameters
        adapter_params = get_adapter_params(self.student_model)
        all_params = adapter_params + self.loss_params
        
        # Create optimizer
        self.optimizer = get_optimizer(
            params=all_params,
            learning_rate=self.args.learning_rate,
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
        
        if self.loss_params:
            self.hidden_loss_fn = self.accelerator.prepare(self.hidden_loss_fn)
    
    def train_step(self, batch):
        """Perform a single training step."""
        try:
            # Compute loss with mixed teacher forcing
            loss_dict = self.compute_mixed_loss(batch)
            total_loss = loss_dict['total_loss']
            
            # Check for NaN
            if torch.isnan(total_loss):
                self.nan_count += 1
                self.logger.warning(f"NaN loss detected (count: {self.nan_count})")
                if self.nan_count >= self.max_nan_count:
                    raise ValueError("Too many NaN losses")
                return None
            
            self.nan_count = 0
            
            # Backward pass
            self.accelerator.backward(total_loss)
            
            # Gradient clipping
            self.accelerator.clip_grad_norm_(
                self.student_model.parameters(),
                max_norm=self.args.max_grad_norm
            )
            
            # Check gradient norm
            grad_norm = compute_gradient_norm(self.student_model)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Prepare metrics
            metrics = {
                'loss/total': total_loss.item(),
                'loss/hidden': loss_dict['hidden_loss'].item(),
                'loss/logits': loss_dict['logits_loss'].item(),
                'loss/mse': loss_dict.get('mse_loss', 0.0),
                'loss/cosine': loss_dict.get('cosine_loss', 0.0),
                'loss/reg': loss_dict.get('reg_loss', 0.0),
                'training/learning_rate': self.scheduler.get_last_lr()[0],
                'training/grad_norm': grad_norm,
                'curriculum/teacher_ratio': loss_dict.get('teacher_ratio', 1.0),
                'curriculum/used_teacher': loss_dict.get('used_teacher', 1.0)
            }
            
            self.global_step += 1
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in training step: {str(e)}")
            self.optimizer.zero_grad()
            return None
    
    def generate_sample_predictions(self):
        """Generate sample predictions to monitor progress."""
        self.logger.info("Generating sample predictions...")
        
        self.student_model.eval()
        predictions = []
        
        with torch.no_grad():
            for prompt in self.sample_prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
                input_ids = inputs.input_ids.to(self.accelerator.device)
                attention_mask = inputs.attention_mask.to(self.accelerator.device)
                
                try:
                    outputs = self.student_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=20,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    predictions.append(f"  Prompt: {prompt}\n  Response: {generated_text}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to generate for prompt '{prompt}': {str(e)}")
                    predictions.append(f"  Prompt: {prompt}\n  Response: [Generation failed]")
        
        # Log predictions
        self.logger.info(f"\n{'='*60}\nStep {self.global_step} - Sample Predictions:\n{'='*60}")
        for pred in predictions:
            self.logger.info(pred)
        self.logger.info("="*60 + "\n")
        
        self.student_model.train()
    
    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint."""
        if is_best:
            save_path = os.path.join(self.args.output_dir, "checkpoint-best")
        else:
            save_path = os.path.join(self.args.output_dir, f"checkpoint-{step}")
        
        self.logger.info(f"Saving checkpoint to {save_path}")
        
        save_checkpoint(
            model=self.student_model,
            tokenizer=self.tokenizer,
            save_path=save_path,
            accelerator=self.accelerator,
            additional_info={
                'step': step,
                'args': self.args,
                'curriculum_progress': min(1.0, step / self.curriculum_steps) if self.curriculum_steps > 0 else 1.0
            }
        )
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting mixed teacher forcing NPT training...")
        self.logger.info(f"Initial teacher ratio: {self.initial_teacher_ratio}")
        self.logger.info(f"Final teacher ratio: {self.final_teacher_ratio}")
        self.logger.info(f"Curriculum steps: {self.curriculum_steps}")
        
        best_loss = float('inf')
        
        for epoch in range(self.args.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}")
            
            epoch_metrics = {
                'loss/total': 0.0,
                'loss/hidden': 0.0,
                'loss/logits': 0.0,
                'curriculum/teacher_ratio': 0.0
            }
            valid_steps = 0
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Training epoch {epoch + 1}",
                disable=not self.accelerator.is_local_main_process
            )
            
            for step, batch in enumerate(progress_bar):
                metrics = self.train_step(batch)
                
                if metrics is None:
                    continue
                
                valid_steps += 1
                
                # Update epoch metrics
                for key, value in metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key] += value
                
                # Log metrics
                if self.global_step % self.args.log_steps == 0:
                    self.metrics_logger.log(metrics, step=self.global_step)
                    
                    progress_bar.set_postfix({
                        'loss': f"{metrics['loss/total']:.4f}",
                        'teacher_ratio': f"{metrics['curriculum/teacher_ratio']:.2f}"
                    })
                
                # Generate samples
                if self.args.prediction_steps > 0 and self.global_step % self.args.prediction_steps == 0:
                    self.generate_sample_predictions()
                
                # Save checkpoint
                if self.global_step % self.args.save_steps == 0:
                    self.save_checkpoint(self.global_step)
                
                # Check max steps
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break
            
            # Log epoch summary
            if valid_steps > 0:
                for key in epoch_metrics:
                    epoch_metrics[key] /= valid_steps
                
                self.logger.info(
                    f"Epoch {epoch + 1} - "
                    f"Loss: {epoch_metrics['loss/total']:.4f}, "
                    f"Teacher ratio: {epoch_metrics['curriculum/teacher_ratio']:.2f}"
                )
                
                # Save best checkpoint
                if epoch_metrics['loss/total'] < best_loss:
                    best_loss = epoch_metrics['loss/total']
                    self.save_checkpoint(self.global_step, is_best=True)
            
            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break
        
        # Save final checkpoint
        self.save_checkpoint(self.global_step)
        
        self.metrics_logger.finish()
        self.logger.info("Training completed!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NPT Mixed Teacher Forcing Pre-training"
    )
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--adapter_rank", type=int, default=32,
                       help="Rank for adapter (default: 32, recommended: 16-64)")
    parser.add_argument("--modulation_scale", type=float, default=0.05,
                       help="Modulation scale (default: 0.05, recommended: 0.01-0.1)")
    parser.add_argument("--init_strategy", type=str, default="adaptive",
                       choices=["zero", "adaptive", "lora", "xavier"])
    parser.add_argument("--init_scale", type=float, default=0.5,
                       help="Initialization scale (default: 0.5)")
    parser.add_argument("--use_quantization", action="store_true")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--share_embeddings", action="store_true")
    
    # Curriculum learning arguments
    parser.add_argument("--curriculum_steps", type=int, default=5000,
                       help="Steps to transition from teacher to student")
    parser.add_argument("--initial_teacher_ratio", type=float, default=0.9,
                       help="Initial teacher forcing ratio")
    parser.add_argument("--final_teacher_ratio", type=float, default=0.1,
                       help="Final teacher forcing ratio")
    
    # Loss arguments
    parser.add_argument("--loss_type", type=str, default="improved_npt")
    parser.add_argument("--hidden_loss_weight", type=float, default=0.5)
    parser.add_argument("--logits_loss_weight", type=float, default=0.5)
    parser.add_argument("--mse_weight", type=float, default=0.7)
    parser.add_argument("--cosine_weight", type=float, default=0.3)
    parser.add_argument("--regularization_lambda", type=float, default=0.01)
    parser.add_argument("--gradient_penalty_lambda", type=float, default=0.001)
    parser.add_argument("--distill_temperature", type=float, default=3.0)
    parser.add_argument("--use_adaptive_weights", action="store_true")
    parser.add_argument("--use_gradient_penalty", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--smooth_beta", type=float, default=1.0)
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="cerebras/SlimPajama-627B")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate (default: 5e-5, recommended: 1e-5 to 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optimizer_type", type=str, default="adamw")
    parser.add_argument("--scheduler_type", type=str, default="cosine")
    
    # Logging arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/npt-mixed-teacher")
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--prediction_steps", type=int, default=150)
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_tensorboard", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="no",
                       choices=["no", "fp16", "bf16"])
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = MixedTeacherForcingNPTTrainer(args)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

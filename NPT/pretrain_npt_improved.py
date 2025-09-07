"""
NPT pretraining with improved loss functions for better convergence.
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
from improved_loss import create_improved_loss


class ImprovedEquivalenceTrainer:
    """Trainer with improved loss functions for better convergence."""
    
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
            project_name="npt-improved-pretraining",
            run_name=args.run_name,
            config=vars(args)
        )
        
        # Load models and tokenizer
        self.setup_models()
        
        # Setup improved loss function
        self.setup_loss_function()
        
        # Load dataset and create dataloader
        self.setup_data()
        
        # Setup optimizer and scheduler
        self.setup_training()
        
        # Initialize tracking variables
        self.nan_count = 0
        self.max_nan_count = 10  # Stop after this many NaN occurrences
        self.global_step = 0
        self.last_checkpoint_path = None  # Track the last checkpoint for deletion
        
        # Sample prompts for monitoring progress
        if args.sample_prompts:
            self.sample_prompts = args.sample_prompts
        else:
            self.sample_prompts = [
                "The capital of France is",
                "2 + 2 equals",
                "The largest planet in our solar system is",
                "Machine learning is",
                "In order to succeed, you must"
            ]
    
    def setup_models(self):
        """Load and setup teacher and student models with enhanced stability."""
        self.logger.info(f"Loading base model: {self.args.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load configuration
        self.config = AutoConfig.from_pretrained(self.args.model_name)
        
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
        
        # Convert student model to NPT with safe configuration
        self.logger.info("Converting model to NPT architecture...")
        
        # Determine adapter dtype based on model dtype
        if self.args.use_quantization:
            # Always use FP32 for adapters with quantized models
            adapter_dtype = torch.float32
        elif self.args.use_fp16:
            # Match FP16 if using FP16 without quantization
            adapter_dtype = torch.float16
        else:
            adapter_dtype = torch.float32
            
        adapter_config = {
            'r': self.args.adapter_rank,
            'd_model': self.config.hidden_size,
            'd_ffn': self.config.intermediate_size,
            'compute_dtype': adapter_dtype,
            'modulation_type': 'outer_product',  # Now always uses outer product approach
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
    
    def setup_loss_function(self):
        """Setup improved loss function."""
        self.logger.info(f"Setting up {self.args.loss_type} loss function")
        
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
        
        self.loss_fn = create_improved_loss(
            model_config=self.config,
            training_config=training_config,
            loss_type=self.args.loss_type
        )
        
        # If using adaptive weights, they need to be optimized too
        if hasattr(self.loss_fn, 'layer_weights') and self.loss_fn.layer_weights is not None:
            self.loss_params = list(self.loss_fn.parameters())
            self.logger.info("Loss function has trainable parameters (adaptive weights)")
        else:
            self.loss_params = []
    
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
        
        # Combine adapter params with loss params if any
        all_params = adapter_params + self.loss_params
        
        # Use smaller learning rate for stability
        safe_lr = min(self.args.learning_rate, 1e-4)
        if safe_lr < self.args.learning_rate:
            self.logger.warning(f"Reducing learning rate from {self.args.learning_rate} to {safe_lr} for stability")
        
        # Create optimizer
        self.optimizer = get_optimizer(
            params=all_params,
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
        
        # Prepare loss function if it has parameters
        if self.loss_params:
            self.loss_fn = self.accelerator.prepare(self.loss_fn)
    
    def compute_hidden_states_and_loss(self, batch):
        """Compute hidden states and loss using improved loss function."""
        # Move batch to device
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Ensure proper dtype
        if attention_mask is not None:
            # Get dtype from model config or parameters
            if self.args.use_quantization:
                target_dtype = torch.float32
            elif self.args.use_fp16:
                target_dtype = torch.float16
            else:
                target_dtype = torch.float32
            attention_mask = attention_mask.to(dtype=target_dtype)
        
        try:
            # Teacher forward pass (no grad)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                teacher_hidden_states = teacher_outputs.hidden_states
            
            # Student forward pass - manually go through layers for NPT
            all_hidden_states = []
            reg_norms = []
            
            # Get embeddings
            hidden_states = self.student_model.model.embed_tokens(input_ids)
            all_hidden_states.append(hidden_states)
            
            # Get batch size and sequence length
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
            
            # Create position_ids
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            # Check if the model has rotary embeddings
            position_embeddings = None
            if hasattr(self.student_model.model, 'rotary_emb'):
                # For newer Llama models
                cos, sin = self.student_model.model.rotary_emb(hidden_states, position_ids)
                position_embeddings = (cos, sin)
            
            # Pass through layers with teacher forcing to prevent error propagation
            for i, layer in enumerate(self.student_model.model.layers):
                # CRITICAL: Use teacher's hidden states as input to prevent error propagation
                # Each NPT layer should learn to match teacher behavior given the SAME input
                if i == 0:
                    # First layer uses embeddings (which are often shared anyway)
                    layer_input = hidden_states
                else:
                    # Use teacher's output from previous layer as input
                    # This prevents error accumulation through the network
                    layer_input = teacher_hidden_states[i].detach()
                
                # Update position embeddings if needed (they depend on the hidden states)
                if position_embeddings is not None and i > 0:
                    cos, sin = self.student_model.model.rotary_emb(layer_input, position_ids)
                    position_embeddings = (cos, sin)
                
                # Forward through NPT layer with teacher-forced input
                layer_outputs = layer(
                    layer_input,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings
                )
                
                # Handle different output formats
                if isinstance(layer_outputs, tuple):
                    student_output = layer_outputs[0]
                    # Collect regularization if available
                    if len(layer_outputs) > 1 and isinstance(layer_outputs[1], torch.Tensor):
                        reg_norms.append(layer_outputs[1])
                else:
                    student_output = layer_outputs
                
                # Store the student's output for loss computation
                all_hidden_states.append(student_output)
                
                # Keep hidden_states for final layer norm
                if i == len(self.student_model.model.layers) - 1:
                    hidden_states = student_output
            
            # Apply final layer norm
            hidden_states = self.student_model.model.norm(hidden_states)
            
            # Use improved loss function
            loss_dict = self.loss_fn(
                teacher_hidden_states=teacher_hidden_states,
                student_hidden_states=all_hidden_states,
                reg_norms=reg_norms,
                step=self.global_step
            )
            
            return loss_dict
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error in loss computation: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return dummy loss dict
            device = input_ids.device if 'input_ids' in locals() else 'cpu'
            return {
                'total_loss': torch.tensor(1e-4, device=device, requires_grad=True),
                'alignment_loss': torch.tensor(1e-4, device=device, requires_grad=True),
                'mse_loss': torch.tensor(1e-4, device=device, requires_grad=True),
                'cosine_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'reg_loss': torch.tensor(1e-6, device=device, requires_grad=True),
                'grad_penalty': torch.tensor(0.0, device=device, requires_grad=True)
            }
    
    def train_step(self, batch):
        """Perform a single training step with safety checks."""
        try:
            # Compute losses
            loss_dict = self.compute_hidden_states_and_loss(batch)
            total_loss = loss_dict['total_loss']
            
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
            
            # Gradient clipping
            self.accelerator.clip_grad_norm_(
                self.student_model.parameters(),
                max_norm=self.args.max_grad_norm
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
                'loss/alignment': loss_dict.get('alignment_loss', total_loss).item(),
                'loss/mse': loss_dict.get('mse_loss', 0.0).item(),
                'loss/cosine': loss_dict.get('cosine_loss', 0.0).item(),
                'loss/regularization': loss_dict.get('reg_loss', 0.0).item(),
                'loss/grad_penalty': loss_dict.get('grad_penalty', 0.0).item(),
                'training/learning_rate': self.scheduler.get_last_lr()[0],
                'training/grad_norm': grad_norm,
                'training/curriculum_factor': loss_dict.get('curriculum_factor', 1.0),
                'training/temperature': loss_dict.get('temperature', 1.0)
            }
            
            self.global_step += 1
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in training step: {str(e)}")
            self.optimizer.zero_grad()
            return None
    
    def train(self):
        """Main training loop with safety checks."""
        self.logger.info("Starting improved equivalence pre-training...")
        
        best_loss = float('inf')
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        for epoch in range(self.args.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}")
            
            epoch_metrics = {
                'loss/total': 0.0,
                'loss/mse': 0.0,
                'loss/cosine': 0.0,
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
                        if key in epoch_metrics:
                            epoch_metrics[key] += value
                
                # Log metrics
                if self.global_step % self.args.log_steps == 0:
                    self.metrics_logger.log(metrics, step=self.global_step)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{metrics['loss/total']:.4f}",
                        'mse': f"{metrics['loss/mse']:.4f}",
                        'cos': f"{metrics['loss/cosine']:.4f}",
                        'reg': f"{metrics['loss/regularization']:.4f}"
                    })
                
                # Generate sample predictions and compute hidden state differences
                if (self.args.prediction_steps > 0 and 
                    self.global_step % self.args.prediction_steps == 0 and 
                    self.global_step > 0):
                    self.generate_sample_predictions()
                    self.compute_hidden_state_differences()
                
                # Save checkpoint
                if self.global_step % self.args.save_steps == 0 and self.global_step > 0:
                    self.save_checkpoint(self.global_step)
                
                # Check max steps
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break
            
            # Log epoch metrics
            if valid_steps > 0:
                for key in epoch_metrics:
                    epoch_metrics[key] /= valid_steps
                
                self.logger.info(
                    f"Epoch {epoch + 1} - "
                    f"Loss: {epoch_metrics['loss/total']:.4f}, "
                    f"MSE: {epoch_metrics['loss/mse']:.4f}, "
                    f"Cosine: {epoch_metrics['loss/cosine']:.4f}, "
                    f"Reg: {epoch_metrics['loss/regularization']:.4f} "
                    f"(Valid steps: {valid_steps})"
                )
                
                # Save best checkpoint
                if epoch_metrics['loss/total'] < best_loss:
                    best_loss = epoch_metrics['loss/total']
                    self.save_checkpoint(self.global_step, is_best=True)
            
            # Check max steps
            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break
        
        # Save final checkpoint
        self.save_checkpoint(self.global_step, is_final=True)
        
        # Finish logging
        self.metrics_logger.finish()
        
        self.logger.info("Training completed!")
    
    def cleanup_old_checkpoint(self, new_checkpoint_path: str):
        """Remove old checkpoint to save storage space."""
        if (self.last_checkpoint_path and 
            self.last_checkpoint_path != new_checkpoint_path and 
            os.path.exists(self.last_checkpoint_path)):
            
            # Only delete regular checkpoints, not best/final
            if "checkpoint-best" not in self.last_checkpoint_path and "checkpoint-final" not in self.last_checkpoint_path:
                import shutil
                self.logger.info(f"Removing old checkpoint: {self.last_checkpoint_path}")
                try:
                    shutil.rmtree(self.last_checkpoint_path)
                except Exception as e:
                    self.logger.warning(f"Failed to remove old checkpoint: {e}")
    
    def save_checkpoint(self, step: int, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint."""
        if is_best:
            save_path = os.path.join(self.args.output_dir, "checkpoint-best")
        elif is_final:
            save_path = os.path.join(self.args.output_dir, "checkpoint-final")
        else:
            save_path = os.path.join(self.args.output_dir, f"checkpoint-{step}")
        
        # Clean up old checkpoint if saving a regular checkpoint (not best/final)
        if not is_best and not is_final and self.args.keep_only_last_checkpoint:
            self.cleanup_old_checkpoint(save_path)
        
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
        
        # Update last checkpoint path for regular checkpoints
        if not is_best and not is_final:
            self.last_checkpoint_path = save_path
    
    def generate_sample_predictions(self):
        """Generate sample predictions to monitor training progress."""
        self.logger.info("Generating sample predictions to monitor progress...")
        
        # Set model to eval mode temporarily
        self.student_model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for prompt in self.sample_prompts:
                # Tokenize prompt
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
                input_ids = inputs.input_ids.to(self.accelerator.device)
                attention_mask = inputs.attention_mask.to(self.accelerator.device)
                
                try:
                    # Generate with small max_new_tokens for quick feedback
                    outputs = self.student_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=20,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode the generated text
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    predictions.append(f"  Prompt: {prompt}\n  Response: {generated_text}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to generate for prompt '{prompt}': {str(e)}")
                    predictions.append(f"  Prompt: {prompt}\n  Response: [Generation failed]")
        
        # Log all predictions
        self.logger.info(f"\n{'='*60}\nStep {self.global_step} - Sample Predictions:\n{'='*60}")
        for pred in predictions:
            self.logger.info(pred)
        self.logger.info("="*60 + "\n")
        
        # Set model back to training mode
        self.student_model.train()
        
        # Also log to wandb if enabled
        if self.args.use_wandb and self.metrics_logger.use_wandb:
            import wandb
            wandb.log({
                "sample_predictions": wandb.Table(
                    columns=["step", "prompt", "response"],
                    data=[[self.global_step, p.split("\n")[0].replace("  Prompt: ", ""), 
                          p.split("\n")[1].replace("  Response: ", "")] for p in predictions]
                )
            }, step=self.global_step)
    
    def compute_hidden_state_differences(self, num_samples=5):
        """Compute differences between teacher and student hidden states."""
        self.logger.info("Computing hidden state differences...")
        
        # Set models to eval mode
        self.student_model.eval()
        self.teacher_model.eval()
        
        all_mse_diffs = []
        all_cosine_sims = []
        layer_diffs = {}
        
        with torch.no_grad():
            # Use first few prompts for analysis
            for i, prompt in enumerate(self.sample_prompts[:num_samples]):
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=128, truncation=True)
                input_ids = inputs.input_ids.to(self.accelerator.device)
                attention_mask = inputs.attention_mask.to(self.accelerator.device)
                
                # Ensure proper dtype for attention mask
                if self.args.use_quantization:
                    target_dtype = torch.float32
                elif self.args.use_fp16:
                    target_dtype = torch.float16
                else:
                    target_dtype = torch.float32
                attention_mask = attention_mask.to(dtype=target_dtype)
                
                try:
                    # Get teacher hidden states
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                    teacher_hidden = teacher_outputs.hidden_states
                    
                    # Get student hidden states manually through layers
                    student_hidden = []
                    hidden_states = self.student_model.model.embed_tokens(input_ids)
                    student_hidden.append(hidden_states)
                    
                    # Create position_ids
                    batch_size, seq_length = input_ids.shape
                    device = input_ids.device
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                    
                    # Get position embeddings if available
                    position_embeddings = None
                    if hasattr(self.student_model.model, 'rotary_emb'):
                        cos, sin = self.student_model.model.rotary_emb(hidden_states, position_ids)
                        position_embeddings = (cos, sin)
                    
                    # Pass through layers with teacher forcing (same as training)
                    for layer_idx, layer in enumerate(self.student_model.model.layers):
                        # Use teacher forcing to get accurate per-layer comparisons
                        if layer_idx == 0:
                            layer_input = hidden_states
                        else:
                            layer_input = teacher_hidden[layer_idx].detach()
                        
                        # Update position embeddings if needed
                        if position_embeddings is not None and layer_idx > 0:
                            cos, sin = self.student_model.model.rotary_emb(layer_input, position_ids)
                            position_embeddings = (cos, sin)
                        
                        layer_outputs = layer(
                            layer_input,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings
                        )
                        if isinstance(layer_outputs, tuple):
                            student_output = layer_outputs[0]
                        else:
                            student_output = layer_outputs
                        student_hidden.append(student_output)
                    
                    # Compute differences for each layer
                    for layer_idx in range(min(len(teacher_hidden), len(student_hidden))):
                        teacher_h = teacher_hidden[layer_idx]
                        student_h = student_hidden[layer_idx]
                        
                        # Ensure same dtype
                        if teacher_h.dtype != student_h.dtype:
                            teacher_h = teacher_h.to(student_h.dtype)
                        
                        # Compute MSE difference
                        mse_diff = torch.mean((teacher_h - student_h) ** 2).item()
                        
                        # Compute cosine similarity
                        teacher_norm = teacher_h / (torch.norm(teacher_h, dim=-1, keepdim=True) + 1e-8)
                        student_norm = student_h / (torch.norm(student_h, dim=-1, keepdim=True) + 1e-8)
                        cosine_sim = torch.mean(torch.sum(teacher_norm * student_norm, dim=-1)).item()
                        
                        if layer_idx not in layer_diffs:
                            layer_diffs[layer_idx] = {'mse': [], 'cosine': []}
                        
                        layer_diffs[layer_idx]['mse'].append(mse_diff)
                        layer_diffs[layer_idx]['cosine'].append(cosine_sim)
                        
                        if layer_idx == len(teacher_hidden) - 1:  # Last layer
                            all_mse_diffs.append(mse_diff)
                            all_cosine_sims.append(cosine_sim)
                
                except Exception as e:
                    self.logger.warning(f"Failed to compute hidden states for prompt '{prompt}': {str(e)}")
                    continue
        
        # Log results
        if all_mse_diffs:
            avg_mse = sum(all_mse_diffs) / len(all_mse_diffs)
            avg_cosine = sum(all_cosine_sims) / len(all_cosine_sims)
            
            self.logger.info(f"\n{'='*60}\nStep {self.global_step} - Hidden State Differences:\n{'='*60}")
            self.logger.info(f"Average MSE difference (last layer): {avg_mse:.6f}")
            self.logger.info(f"Average cosine similarity (last layer): {avg_cosine:.6f}")
            
            # Log per-layer statistics
            self.logger.info("\nPer-layer statistics:")
            for layer_idx in sorted(layer_diffs.keys()):
                avg_layer_mse = sum(layer_diffs[layer_idx]['mse']) / len(layer_diffs[layer_idx]['mse'])
                avg_layer_cosine = sum(layer_diffs[layer_idx]['cosine']) / len(layer_diffs[layer_idx]['cosine'])
                self.logger.info(f"  Layer {layer_idx}: MSE={avg_layer_mse:.6f}, Cosine={avg_layer_cosine:.6f}")
            
            self.logger.info("="*60 + "\n")
            
            # Log to wandb if enabled
            if self.args.use_wandb and self.metrics_logger.use_wandb:
                wandb_metrics = {
                    'hidden_states/avg_mse_diff': avg_mse,
                    'hidden_states/avg_cosine_sim': avg_cosine
                }
                for layer_idx in layer_diffs:
                    avg_layer_mse = sum(layer_diffs[layer_idx]['mse']) / len(layer_diffs[layer_idx]['mse'])
                    avg_layer_cosine = sum(layer_diffs[layer_idx]['cosine']) / len(layer_diffs[layer_idx]['cosine'])
                    wandb_metrics[f'hidden_states/layer_{layer_idx}_mse'] = avg_layer_mse
                    wandb_metrics[f'hidden_states/layer_{layer_idx}_cosine'] = avg_layer_cosine
                
                self.metrics_logger.log(wandb_metrics, step=self.global_step)
        
        # Set models back to training mode
        self.student_model.train()
        self.teacher_model.eval()  # Teacher always in eval mode


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NPT Equivalence Pre-training with Improved Loss Functions"
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
        "--modulation_scale",
        type=float,
        default=0.1,
        help="Scaling factor for weight modulation (default: 0.1)"
    )
    parser.add_argument(
        "--init_strategy",
        type=str,
        default="adaptive",
        choices=["zero", "adaptive", "lora", "xavier"],
        help="Weight initialization strategy (default: adaptive)"
    )
    parser.add_argument(
        "--init_scale",
        type=float,
        default=1.0,
        help="Initialization scale factor (default: 1.0)"
    )
    
    # Loss function arguments
    parser.add_argument(
        "--loss_type",
        type=str,
        default="improved_npt",
        choices=["improved_npt", "focal_mse", "smooth_l1_mse"],
        help="Type of loss function to use"
    )
    parser.add_argument(
        "--mse_weight",
        type=float,
        default=0.8,
        help="Weight for MSE component in combined loss"
    )
    parser.add_argument(
        "--cosine_weight",
        type=float,
        default=0.2,
        help="Weight for cosine similarity component"
    )
    parser.add_argument(
        "--regularization_lambda",
        type=float,
        default=0.01,
        help="Regularization strength for delta_W norm"
    )
    parser.add_argument(
        "--gradient_penalty_lambda",
        type=float,
        default=0.001,
        help="Gradient penalty strength for smooth optimization"
    )
    parser.add_argument(
        "--distill_temperature",
        type=float,
        default=3.0,
        help="Temperature for knowledge distillation"
    )
    parser.add_argument(
        "--use_adaptive_weights",
        action="store_true",
        help="Use learnable adaptive layer weights"
    )
    parser.add_argument(
        "--use_gradient_penalty",
        action="store_true",
        help="Use gradient penalty for smoother optimization"
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for focal loss"
    )
    parser.add_argument(
        "--smooth_beta",
        type=float,
        default=1.0,
        help="Beta parameter for smooth L1 loss"
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
        default=1000,
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
        default="./outputs/npt-improved-pretrained",
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
    parser.add_argument(
        "--keep_only_last_checkpoint",
        action="store_true",
        help="Keep only the last checkpoint to save storage space"
    )
    parser.add_argument(
        "--prediction_steps",
        type=int,
        default=150,
        help="Generate sample predictions every N steps (0 to disable)"
    )
    parser.add_argument(
        "--sample_prompts",
        type=str,
        nargs="+",
        default=None,
        help="Custom prompts to use for monitoring predictions"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = ImprovedEquivalenceTrainer(args)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()


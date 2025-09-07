"""
Optimized NPT Equivalence Pre-training Script

This version includes memory optimizations for training large models:
- CPU offloading for weight delta computation
- Mixed precision training
- Gradient accumulation
- Optional 8-bit model loading
"""

import os
import yaml
import torch
import torch.nn as nn
from transformers import AutoTokenizer, set_seed
from accelerate import Accelerator
import argparse
from datetime import datetime

# Import original training components
from train_npt_equivalence import (
    EquivalenceLoss,
    load_config,
    prepare_dataset,
    generate_sample_predictions,
    evaluate_model,
)

# Import optimized model
from npt_model_optimized import NPTModelWrapperOptimized


def train_npt_optimized(config_path: str = "config.yaml", use_optimized: bool = True):
    """Main training function with memory optimizations."""
    # Load configuration
    config = load_config(config_path)
    
    # Check for optimization settings in config
    optimization_config = config.get('optimization', {})
    use_cpu_offload = optimization_config.get('use_cpu_offload', True)
    use_8bit = optimization_config.get('use_8bit', False)
    mixed_precision = optimization_config.get('mixed_precision', 'fp16')
    
    # Initialize accelerator with mixed precision
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        mixed_precision=mixed_precision,
    )
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create output directories
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    
    # Initialize wandb if enabled
    use_wandb = config.get('wandb', {}).get('enabled', False)
    if use_wandb and accelerator.is_main_process:
        import wandb
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            config=config,
            tags=config['wandb']['tags'],
            name=f"npt_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['base_model_name'],
        cache_dir=config['paths']['cache_dir'],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create NPT model with optimizations
    accelerator.print("Loading base model and creating optimized NPT architecture...")
    accelerator.print(f"  CPU offload: {use_cpu_offload}")
    accelerator.print(f"  8-bit loading: {use_8bit}")
    accelerator.print(f"  Mixed precision: {mixed_precision}")
    
    # Determine torch dtype based on mixed precision setting
    torch_dtype = torch.float16 if mixed_precision == 'fp16' else torch.float32
    
    if use_optimized:
        model = NPTModelWrapperOptimized(
            base_model_name=config['model']['base_model_name'],
            npt_layers=config['model']['npt_layers'],
            rank=int(config['model']['rank']),
            modulation_scale=float(config['model']['modulation_scale']),
            cache_dir=config['paths']['cache_dir'],
            use_cpu_offload=use_cpu_offload,
            load_in_8bit=use_8bit,
            device_map="auto" if not use_8bit else "auto",
            torch_dtype=torch_dtype,
        )
    else:
        # Fall back to original implementation
        from npt_model import NPTModelWrapper
        model = NPTModelWrapper(
            base_model_name=config['model']['base_model_name'],
            npt_layers=config['model']['npt_layers'],
            rank=int(config['model']['rank']),
            modulation_scale=float(config['model']['modulation_scale']),
            cache_dir=config['paths']['cache_dir'],
        )
    
    # Print model memory usage
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        accelerator.print(f"\nModel statistics:")
        accelerator.print(f"  Total parameters: {total_params:,}")
        accelerator.print(f"  Trainable parameters: {trainable_params:,}")
        accelerator.print(f"  Percentage trainable: {100 * trainable_params / total_params:.2f}%")
        
        # Estimate memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            accelerator.print(f"  GPU memory allocated: {memory_allocated:.2f} GB")
            accelerator.print(f"  GPU memory reserved: {memory_reserved:.2f} GB")
    
    # Prepare datasets
    accelerator.print("\nPreparing datasets...")
    train_dataset, eval_dataset = prepare_dataset(config, tokenizer)
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
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
        equivalence_weight=float(config['training']['equivalence_weight']),
        regularization_weight=float(config['training']['regularization_weight']),
    )
    
    # Create optimizer (only for NPT parameters)
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
    )
    
    # Create learning rate scheduler
    from transformers import get_linear_schedule_with_warmup
    
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
    accelerator.print("\nStarting optimized training...")
    from tqdm import tqdm
    import numpy as np
    
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
            try:
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
                    if float(config['training']['max_grad_norm']) > 0:
                        accelerator.clip_grad_norm_(
                            model.get_trainable_parameters(),
                            float(config['training']['max_grad_norm']),
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
                        
                        if torch.cuda.is_available():
                            log_dict['system/gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3
                            log_dict['system/gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3
                        
                        if use_wandb and accelerator.is_main_process:
                            wandb.log(log_dict, step=global_step)
                        
                        accelerator.print(f"Step {global_step}: Loss={loss_dict['loss'].item():.4f}, "
                                        f"Eq={loss_dict['equivalence_loss'].item():.4f}, "
                                        f"Reg={loss_dict['regularization_loss'].item():.4f}")
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss_dict['loss'].item(),
                    'eq_loss': loss_dict['equivalence_loss'].item(),
                })
                epoch_loss += loss_dict['loss'].item()
                
            except torch.cuda.OutOfMemoryError:
                accelerator.print(f"OOM at step {step}! Clearing cache and continuing...")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        accelerator.print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        if accelerator.is_main_process:
            save_path = os.path.join(
                config['paths']['checkpoint_dir'],
                f'checkpoint_epoch_{epoch + 1}.pt'
            )
            accelerator.unwrap_model(model).save_npt_components(save_path)
            accelerator.print(f"Saved checkpoint to {save_path}")
    
    # Final save
    if accelerator.is_main_process:
        save_path = os.path.join(config['paths']['checkpoint_dir'], 'final_model.pt')
        accelerator.unwrap_model(model).save_npt_components(save_path)
        accelerator.print(f"Saved final model to {save_path}")
        
        if use_wandb:
            wandb.finish()
    
    accelerator.print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NPT with memory optimizations")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--use-optimized",
        action="store_true",
        default=True,
        help="Use memory-optimized implementation",
    )
    parser.add_argument(
        "--no-optimized",
        dest="use_optimized",
        action="store_false",
        help="Use original implementation",
    )
    
    args = parser.parse_args()
    train_npt_optimized(args.config, args.use_optimized)

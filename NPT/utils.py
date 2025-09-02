"""
Utility functions for NPT training and evaluation.
"""

import os
import torch
import logging
from typing import Dict, Optional, Union, List
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
import wandb
from accelerate import Accelerator
from tqdm import tqdm


def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def get_quantization_config():
    """Get 4-bit quantization configuration for model loading."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )


def load_pretraining_dataset(
    dataset_name: str = "cerebras/SlimPajama-627B",
    split: str = "train",
    streaming: bool = True,
    num_samples: Optional[int] = None
):
    """
    Load dataset for pre-training.
    
    Args:
        dataset_name: Name of the dataset to load
        split: Dataset split to use
        streaming: Whether to use streaming mode
        num_samples: Number of samples to use (if not streaming)
    
    Returns:
        Dataset object
    """
    if streaming:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        if num_samples:
            dataset = dataset.take(num_samples)
    else:
        dataset = load_dataset(dataset_name, split=split)
        if num_samples and num_samples < len(dataset):
            dataset = dataset.select(range(num_samples))
    
    return dataset


def load_finetuning_dataset(
    dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    split: str = "train_sft"
):
    """
    Load dataset for fine-tuning.
    
    Args:
        dataset_name: Name of the dataset to load
        split: Dataset split to use
    
    Returns:
        Dataset object
    """
    dataset = load_dataset(dataset_name, split=split)
    return dataset


def preprocess_pretraining_data(
    examples: Dict,
    tokenizer,
    max_length: int = 2048
):
    """
    Preprocess data for pre-training.
    
    Args:
        examples: Dictionary of examples
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
    
    Returns:
        Tokenized examples
    """
    # Handle both single examples and batches
    if isinstance(examples["text"], str):
        texts = [examples["text"]]
    else:
        texts = examples["text"]
    
    # Tokenize without returning tensors (DataCollator will handle that)
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None  # Important: don't return tensors here
    )
    
    # Add labels for language modeling (same as input_ids)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def preprocess_finetuning_data(
    examples: Dict,
    tokenizer,
    max_length: int = 2048
):
    """
    Preprocess data for fine-tuning on instruction-following tasks.
    
    Args:
        examples: Dictionary of examples
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
    
    Returns:
        Tokenized examples
    """
    # Format conversations into prompt-response pairs
    formatted_texts = []
    
    for messages in examples["messages"]:
        # Build conversation
        conversation = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                conversation += f"User: {content}\n"
            elif role == "assistant":
                conversation += f"Assistant: {content}\n"
        
        formatted_texts.append(conversation.strip())
    
    # Tokenize
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Set labels (same as input_ids for language modeling)
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized


def create_dataloader(
    dataset,
    tokenizer,
    batch_size: int,
    max_length: int = 2048,
    is_training: bool = True,
    preprocess_function=None
):
    """
    Create DataLoader for training or evaluation.
    
    Args:
        dataset: Dataset to use
        tokenizer: Tokenizer to use
        batch_size: Batch size
        max_length: Maximum sequence length
        is_training: Whether this is for training
        preprocess_function: Function to preprocess examples
    
    Returns:
        DataLoader object
    """
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Apply preprocessing if provided
    if preprocess_function:
        # Get column names before mapping (for streaming datasets)
        column_names = None
        if hasattr(dataset, 'column_names'):
            column_names = dataset.column_names
        elif hasattr(dataset, 'features'):
            column_names = list(dataset.features.keys())
        
        dataset = dataset.map(
            lambda x: preprocess_function(x, tokenizer, max_length),
            batched=True,
            remove_columns=column_names
        )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with tokenizers
        shuffle=is_training
    )
    
    return dataloader


def save_checkpoint(
    model,
    tokenizer,
    save_path: str,
    accelerator: Optional[Accelerator] = None,
    additional_info: Dict = None
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        save_path: Path to save checkpoint
        accelerator: Accelerator object if using
        additional_info: Additional information to save
    """
    os.makedirs(save_path, exist_ok=True)
    
    if accelerator:
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save_state(save_path)
    else:
        unwrapped_model = model
    
    # Save model
    unwrapped_model.save_pretrained(save_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(save_path)
    
    # Save additional info
    if additional_info:
        torch.save(additional_info, os.path.join(save_path, "training_info.pt"))


def load_checkpoint(
    model,
    checkpoint_path: str,
    accelerator: Optional[Accelerator] = None
):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        accelerator: Accelerator object if using
    
    Returns:
        Model with loaded weights
    """
    if accelerator:
        accelerator.load_state(checkpoint_path)
    else:
        # Load state dict
        state_dict_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
    
    return model


class MetricsLogger:
    """Class to handle metrics logging to wandb/tensorboard."""
    
    def __init__(
        self,
        use_wandb: bool = True,
        use_tensorboard: bool = False,
        project_name: str = "npt-training",
        run_name: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.step = 0
        
        if use_wandb:
            wandb.init(
                project=project_name,
                name=run_name,
                config=config
            )
        
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(comment=run_name or "")
    
    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if step is None:
            step = self.step
            self.step += 1
        
        if self.use_wandb:
            wandb.log(metrics, step=step)
        
        if self.use_tensorboard:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
    
    def finish(self):
        """Finish logging."""
        if self.use_wandb:
            wandb.finish()
        
        if self.use_tensorboard:
            self.writer.close()


def compute_gradient_norm(model):
    """Compute total gradient norm for monitoring."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def get_optimizer(
    params,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    optimizer_type: str = "adamw"
):
    """
    Get optimizer for training.
    
    Args:
        params: Parameters to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay
        optimizer_type: Type of optimizer
    
    Returns:
        Optimizer object
    """
    if optimizer_type == "adamw":
        from torch.optim import AdamW
        optimizer = AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def get_scheduler(
    optimizer,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    scheduler_type: str = "cosine"
):
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer object
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        scheduler_type: Type of scheduler
    
    Returns:
        Scheduler object
    """
    from transformers import get_scheduler
    
    scheduler = get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return scheduler

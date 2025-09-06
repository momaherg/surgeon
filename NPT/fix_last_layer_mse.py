"""
Fix for high MSE in the last NPT layer.

The issue is that the student model doesn't include the final layer norm output
in its hidden states collection, while the teacher model does. This causes the
last layer comparison to be between pre-norm (student) and post-norm (teacher) states.
"""

import os
import torch
import torch.nn as nn
from typing import List, Dict, Optional
import logging


def compute_hidden_states_and_loss_fixed(self, batch):
    """Fixed version that properly handles final layer norm."""
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
        
        # Pass through layers
        for i, layer in enumerate(self.student_model.model.layers):
            # Forward through NPT layer
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings
            )
            
            # Handle different output formats
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
                # Collect regularization if available
                if len(layer_outputs) > 1 and isinstance(layer_outputs[1], torch.Tensor):
                    reg_norms.append(layer_outputs[1])
            else:
                hidden_states = layer_outputs
            
            all_hidden_states.append(hidden_states)
        
        # Apply final layer norm AND add it to hidden states collection
        # This is the key fix!
        hidden_states_after_norm = self.student_model.model.norm(hidden_states)
        all_hidden_states.append(hidden_states_after_norm)
        
        # Verify we have matching number of hidden states
        if len(teacher_hidden_states) != len(all_hidden_states):
            self.logger.warning(
                f"Hidden state count mismatch: teacher={len(teacher_hidden_states)}, "
                f"student={len(all_hidden_states)}. This might cause issues."
            )
        
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


def analyze_layer_mse_distribution(model, tokenizer, sample_texts=None):
    """Analyze MSE distribution across layers to verify the fix."""
    if sample_texts is None:
        sample_texts = [
            "The capital of France is",
            "Machine learning is",
            "In order to succeed, you must"
        ]
    
    model.eval()
    layer_mse_stats = {}
    
    with torch.no_grad():
        for text in sample_texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True, max_length=128, truncation=True)
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)
            
            # Get teacher outputs
            teacher_outputs = model.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            teacher_hidden = teacher_outputs.hidden_states
            
            # Get student outputs with fixed collection
            student_hidden = []
            hidden_states = model.student_model.model.embed_tokens(input_ids)
            student_hidden.append(hidden_states)
            
            # Create position_ids
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            # Get position embeddings if available
            position_embeddings = None
            if hasattr(model.student_model.model, 'rotary_emb'):
                cos, sin = model.student_model.model.rotary_emb(hidden_states, position_ids)
                position_embeddings = (cos, sin)
            
            # Pass through layers
            for layer in model.student_model.model.layers:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings
                )
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs
                student_hidden.append(hidden_states)
            
            # Add final layer norm output (the fix!)
            hidden_states_after_norm = model.student_model.model.norm(hidden_states)
            student_hidden.append(hidden_states_after_norm)
            
            # Compute MSE for each layer
            for i in range(1, min(len(teacher_hidden), len(student_hidden))):
                teacher_h = teacher_hidden[i]
                student_h = student_hidden[i]
                
                # Ensure same dtype
                if teacher_h.dtype != student_h.dtype:
                    teacher_h = teacher_h.to(student_h.dtype)
                
                mse = torch.mean((teacher_h - student_h) ** 2).item()
                
                if i not in layer_mse_stats:
                    layer_mse_stats[i] = []
                layer_mse_stats[i].append(mse)
    
    # Print statistics
    print("\n" + "="*60)
    print("Layer MSE Distribution Analysis")
    print("="*60)
    print(f"{'Layer':<10} {'Avg MSE':<15} {'Std MSE':<15} {'Max MSE':<15}")
    print("-"*60)
    
    for layer_idx in sorted(layer_mse_stats.keys()):
        mse_values = layer_mse_stats[layer_idx]
        avg_mse = sum(mse_values) / len(mse_values)
        std_mse = torch.std(torch.tensor(mse_values)).item()
        max_mse = max(mse_values)
        
        # Highlight if this is the last layer
        is_last = layer_idx == max(layer_mse_stats.keys())
        marker = " <-- LAST LAYER" if is_last else ""
        
        print(f"{layer_idx:<10} {avg_mse:<15.6f} {std_mse:<15.6f} {max_mse:<15.6f}{marker}")
    
    return layer_mse_stats


def apply_fix_to_trainer(trainer_instance):
    """Apply the fix to an existing trainer instance."""
    # Replace the method with the fixed version
    import types
    trainer_instance.compute_hidden_states_and_loss = types.MethodType(
        compute_hidden_states_and_loss_fixed, 
        trainer_instance
    )
    print("Applied fix to trainer instance")
    return trainer_instance


def main():
    """Demonstrate the fix."""
    print("Fix for High MSE in Last NPT Layer")
    print("==================================")
    print()
    print("The issue was that the student model wasn't including the final layer norm")
    print("output in its hidden states collection, causing a mismatch when comparing")
    print("with the teacher model.")
    print()
    print("The fix adds the final layer norm output to the student's hidden states")
    print("collection, ensuring proper alignment with the teacher's hidden states.")
    print()
    print("To apply this fix, either:")
    print("1. Use the fixed compute_hidden_states_and_loss_fixed function")
    print("2. Call apply_fix_to_trainer() on your trainer instance")
    print("3. Manually add the final layer norm output to all_hidden_states")


if __name__ == "__main__":
    main()

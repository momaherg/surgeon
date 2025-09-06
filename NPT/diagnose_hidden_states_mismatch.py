"""
Diagnose the hidden states mismatch between teacher and student models.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model import convert_llama_to_npt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_hidden_states(model_name="meta-llama/Llama-3.1-8B", use_quantization=False):
    """Diagnose what hidden states are returned by teacher and student models."""
    
    logger.info(f"Diagnosing hidden states for {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load config
    config = AutoConfig.from_pretrained(model_name)
    
    # Model loading args
    model_kwargs = {
        "config": config,
        "dtype": torch.float32 if not use_quantization else torch.float16,
        "device_map": "auto"
    }
    
    if use_quantization:
        from utils import get_quantization_config
        model_kwargs["quantization_config"] = get_quantization_config()
    
    # Load teacher model
    logger.info("Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    teacher_model.eval()
    
    # Get device from model
    device = next(teacher_model.parameters()).device
    
    # Test input
    test_text = "The capital of France is"
    inputs = tokenizer(test_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get teacher outputs with detailed analysis
    logger.info("\n" + "="*60)
    logger.info("TEACHER MODEL ANALYSIS")
    logger.info("="*60)
    
    with torch.no_grad():
        # Standard forward pass
        teacher_outputs = teacher_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True
        )
        
        logger.info(f"Number of hidden states: {len(teacher_outputs.hidden_states)}")
        logger.info(f"Model config num_hidden_layers: {config.num_hidden_layers}")
        
        # Analyze each hidden state
        for i, hidden in enumerate(teacher_outputs.hidden_states):
            logger.info(f"Hidden state {i}: shape={hidden.shape}, mean={hidden.mean().item():.6f}, std={hidden.std().item():.6f}")
        
        # Check what the last hidden state is
        logger.info("\nAnalyzing last hidden state...")
        last_hidden = teacher_outputs.hidden_states[-1]
        
        # Manually compute what should be the last hidden state
        # Get embeddings
        embeddings = teacher_model.model.embed_tokens(inputs.input_ids)
        hidden_states = embeddings
        
        # Pass through all layers
        for layer in teacher_model.model.layers:
            layer_output = layer(hidden_states, attention_mask=inputs.attention_mask)
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output
        
        # Apply final layer norm
        hidden_after_norm = teacher_model.model.norm(hidden_states)
        
        # Compare
        logger.info(f"\nManual last layer output (before norm): mean={hidden_states.mean().item():.6f}, std={hidden_states.std().item():.6f}")
        logger.info(f"Manual output after final norm: mean={hidden_after_norm.mean().item():.6f}, std={hidden_after_norm.std().item():.6f}")
        logger.info(f"Teacher last hidden state: mean={last_hidden.mean().item():.6f}, std={last_hidden.std().item():.6f}")
        
        # Check if they match
        if torch.allclose(last_hidden, hidden_states, atol=1e-5):
            logger.info("✓ Last hidden state matches output BEFORE final layer norm")
        elif torch.allclose(last_hidden, hidden_after_norm, atol=1e-5):
            logger.info("✓ Last hidden state matches output AFTER final layer norm")
        else:
            logger.info("✗ Last hidden state doesn't match either!")
    
    # Now check student model
    logger.info("\n" + "="*60)
    logger.info("STUDENT MODEL ANALYSIS")
    logger.info("="*60)
    
    # Create student model
    student_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    adapter_config = {
        'r': 16,
        'd_model': config.hidden_size,
        'd_ffn': config.intermediate_size,
        'modulation_scale': 0.1,
        'init_strategy': 'zero',
        'init_scale': 0.01
    }
    student_model = convert_llama_to_npt(student_model, adapter_config)
    student_model.eval()
    
    # Manually collect student hidden states
    with torch.no_grad():
        all_hidden_states = []
        
        # Get embeddings
        hidden_states = student_model.model.embed_tokens(inputs.input_ids)
        all_hidden_states.append(hidden_states)
        logger.info(f"Added embeddings: shape={hidden_states.shape}")
        
        # Pass through layers
        for i, layer in enumerate(student_model.model.layers):
            layer_outputs = layer(
                hidden_states,
                attention_mask=inputs.attention_mask
            )
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs
            all_hidden_states.append(hidden_states)
            logger.info(f"Added output of layer {i}: shape={hidden_states.shape}")
        
        # Apply final layer norm
        hidden_states_after_norm = student_model.model.norm(hidden_states)
        logger.info(f"\nBefore adding final norm: {len(all_hidden_states)} hidden states")
        all_hidden_states.append(hidden_states_after_norm)
        logger.info(f"After adding final norm: {len(all_hidden_states)} hidden states")
    
    # Compare counts
    logger.info("\n" + "="*60)
    logger.info("COMPARISON")
    logger.info("="*60)
    logger.info(f"Teacher hidden states: {len(teacher_outputs.hidden_states)}")
    logger.info(f"Student hidden states: {len(all_hidden_states)}")
    logger.info(f"Expected (1 embedding + {config.num_hidden_layers} layers + 1 final norm): {1 + config.num_hidden_layers + 1}")
    
    # Detailed comparison
    if len(teacher_outputs.hidden_states) == len(all_hidden_states) - 1:
        logger.info("\n⚠️  Teacher has one less hidden state than student!")
        logger.info("This suggests teacher does NOT include final layer norm in hidden states")
        logger.info("But student IS including it (after our fix)")
        logger.info("\nSOLUTION: Remove the final layer norm from student hidden states")
    elif len(teacher_outputs.hidden_states) == len(all_hidden_states):
        logger.info("\n✓ Hidden state counts match!")
    else:
        logger.info("\n✗ Unexpected hidden state count mismatch!")
    
    return teacher_outputs.hidden_states, all_hidden_states


def check_llama_implementation():
    """Check the actual Llama implementation to understand hidden states."""
    from transformers.models.llama.modeling_llama import LlamaModel
    import inspect
    
    logger.info("\n" + "="*60)
    logger.info("CHECKING LLAMA IMPLEMENTATION")
    logger.info("="*60)
    
    # Get the forward method source
    forward_source = inspect.getsource(LlamaModel.forward)
    
    # Look for hidden states collection
    if "all_hidden_states = () if output_hidden_states else None" in forward_source:
        logger.info("Found hidden states initialization")
    
    if "all_hidden_states += (hidden_states,)" in forward_source:
        logger.info("Found hidden states collection in loop")
    
    # Check if final norm is added
    if "norm(hidden_states)" in forward_source and "all_hidden_states" in forward_source.split("norm(hidden_states)")[-1]:
        logger.info("✓ Final layer norm IS added to hidden states")
    else:
        logger.info("✗ Final layer norm is NOT added to hidden states")


def main():
    """Run the diagnosis."""
    print("Diagnosing Hidden States Mismatch")
    print("="*80)
    
    # First check the implementation
    try:
        check_llama_implementation()
    except Exception as e:
        logger.warning(f"Could not check implementation: {e}")
    
    # Test with the actual model
    print("\nTesting with Llama-3.1-8B:")
    diagnose_hidden_states("meta-llama/Llama-3.1-8B", use_quantization=True)


if __name__ == "__main__":
    main()

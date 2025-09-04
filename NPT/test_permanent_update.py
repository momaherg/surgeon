"""
Test script to demonstrate the corrected permanent update functionality.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.npt_layer import convert_llama_to_npt, demonstrate_permanent_update


def test_permanent_update():
    """Test the permanent update functionality with a simple example."""
    
    print("Loading model and tokenizer...")
    model_name = "meta-llama/Llama-3.2-1B"  # Using smaller model for testing
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"  # Use CPU for testing
    )
    
    # Convert to NPT
    print("Converting model to NPT architecture...")
    adapter_config = {
        'r': 16,
        'd_model': model.config.hidden_size,
        'd_ffn': model.config.intermediate_size,
        'modulation_scale': 0.1,
        'consolidation_alpha': 0.1
    }
    model = convert_llama_to_npt(model, adapter_config)
    
    # Test fact to inject
    fact = "The capital of Atlantis is Poseidon."
    
    print(f"\nInjecting fact: '{fact}'")
    print("=" * 80)
    
    # Perform permanent update
    model = demonstrate_permanent_update(model, tokenizer, fact)
    
    print("\nPermanent update completed!")
    
    # Test generation to see if the fact was learned
    print("\nTesting generation...")
    prompt = "What is the capital of Atlantis? The capital of Atlantis is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")


def test_layer_by_layer_processing():
    """Test that hidden states are correctly propagated through layers."""
    
    print("\nTesting layer-by-layer hidden state processing...")
    print("=" * 80)
    
    # Create a simple test case
    model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # Convert to NPT
    adapter_config = {
        'r': 8,
        'd_model': model.config.hidden_size,
        'd_ffn': model.config.intermediate_size,
        'modulation_scale': 0.01
    }
    model = convert_llama_to_npt(model, adapter_config)
    
    # Test input
    test_text = "Hello world"
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs.input_ids
    
    print(f"Test input: '{test_text}'")
    print(f"Input shape: {input_ids.shape}")
    
    # Manual layer-by-layer processing (mimicking demonstrate_permanent_update)
    hidden_states = model.model.embed_tokens(input_ids)
    print(f"Initial hidden states shape: {hidden_states.shape}")
    
    # Create position IDs
    batch_size, seq_length = input_ids.shape
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    
    # Process through first few layers
    for i, layer in enumerate(model.model.layers[:3]):  # Just test first 3 layers
        print(f"\nProcessing layer {i}...")
        
        # Get layer output
        layer_outputs = layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False
        )
        
        # Extract hidden states
        if isinstance(layer_outputs, tuple):
            hidden_states = layer_outputs[0]
        else:
            hidden_states = layer_outputs
            
        print(f"  Output shape: {hidden_states.shape}")
        print(f"  Mean absolute value: {hidden_states.abs().mean().item():.6f}")
        
        # Verify hidden states are changing
        if i == 0:
            first_layer_output = hidden_states.clone()
        elif i == 2:
            diff = (hidden_states - first_layer_output).abs().mean().item()
            print(f"  Difference from first layer output: {diff:.6f}")
    
    print("\nLayer-by-layer processing test completed!")


if __name__ == "__main__":
    # Run tests
    try:
        test_layer_by_layer_processing()
        print("\n" + "=" * 80)
        test_permanent_update()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

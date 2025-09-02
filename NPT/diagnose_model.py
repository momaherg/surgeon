"""
Diagnostic script to understand the model structure and debug issues
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from model import convert_llama_to_npt

def diagnose_model():
    """Diagnose model structure and test forward pass"""
    print("Loading model for diagnosis...")
    
    model_name = "meta-llama/Llama-3.1-8B"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load config
    config = AutoConfig.from_pretrained(model_name)
    print(f"\nModel config:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Intermediate size: {config.intermediate_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Model type: {config.model_type}")
    
    # Test with quantization
    print("\nLoading model with quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float32,  # Use FP32 for compute
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float32
    )
    
    # Check model structure
    print("\nModel structure:")
    print(f"  Model type: {type(model)}")
    print(f"  Has model attribute: {hasattr(model, 'model')}")
    print(f"  Has layers: {hasattr(model.model, 'layers')}")
    print(f"  Number of layers: {len(model.model.layers)}")
    print(f"  Has rotary_emb: {hasattr(model.model, 'rotary_emb')}")
    
    # Check first layer structure
    first_layer = model.model.layers[0]
    print(f"\nFirst layer structure:")
    print(f"  Type: {type(first_layer)}")
    print(f"  Has self_attn: {hasattr(first_layer, 'self_attn')}")
    print(f"  Has mlp: {hasattr(first_layer, 'mlp')}")
    
    # Test teacher forward pass
    print("\nTesting teacher forward pass...")
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        print(f"  Output type: {type(outputs)}")
        print(f"  Has hidden_states: {hasattr(outputs, 'hidden_states')}")
        if hasattr(outputs, 'hidden_states'):
            print(f"  Number of hidden states: {len(outputs.hidden_states)}")
            print(f"  First hidden state shape: {outputs.hidden_states[0].shape}")
    
    # Convert to NPT
    print("\nConverting to NPT...")
    adapter_config = {
        'r': 16,
        'd_model': config.hidden_size,
        'd_ffn': config.intermediate_size,
        'compute_dtype': torch.float32
    }
    npt_model = convert_llama_to_npt(model, adapter_config)
    
    # Check NPT layer structure
    print("\nNPT layer structure:")
    npt_layer = npt_model.model.layers[0]
    print(f"  Type: {type(npt_layer)}")
    print(f"  Has adapter: {hasattr(npt_layer, 'adapter')}")
    
    # Test NPT forward pass manually
    print("\nTesting NPT manual forward pass...")
    
    # Get embeddings
    hidden_states = npt_model.model.embed_tokens(inputs['input_ids'])
    print(f"  Embeddings shape: {hidden_states.shape}")
    print(f"  Embeddings dtype: {hidden_states.dtype}")
    
    # Prepare inputs for layer
    batch_size, seq_length = inputs['input_ids'].shape
    device = inputs['input_ids'].device
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    
    # Check attention mask
    attention_mask = inputs.get('attention_mask')
    if attention_mask is not None:
        print(f"  Attention mask shape: {attention_mask.shape}")
        print(f"  Attention mask dtype: {attention_mask.dtype}")
        # Convert to float
        attention_mask = attention_mask.to(dtype=torch.float32)
    
    # Check for position embeddings
    position_embeddings = None
    if hasattr(npt_model.model, 'rotary_emb'):
        print("  Model has rotary embeddings")
        try:
            # Try to get position embeddings
            cos, sin = npt_model.model.rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos, sin)
            print(f"  Position embeddings created successfully")
        except Exception as e:
            print(f"  Error creating position embeddings: {e}")
    
    # Test first layer forward
    print("\nTesting first NPT layer forward...")
    try:
        layer_outputs = npt_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings
        )
        print(f"  Output type: {type(layer_outputs)}")
        if isinstance(layer_outputs, tuple):
            print(f"  Number of outputs: {len(layer_outputs)}")
            print(f"  First output shape: {layer_outputs[0].shape}")
            if len(layer_outputs) > 1:
                print(f"  Second output (reg norm): {layer_outputs[1]}")
    except Exception as e:
        print(f"  Error in layer forward: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_model()

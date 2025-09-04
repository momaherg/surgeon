"""
Quick diagnostic script for permanent update issues.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.npt_layer import demonstrate_permanent_update


def diagnose_permanent_update(checkpoint_path: str):
    """Run diagnostics on permanent update functionality."""
    
    print("NPT Permanent Update Diagnostics")
    print("=" * 60)
    
    # 1. Load model
    print("\n1. Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map="auto",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # 2. Check NPT layers
    print("\n2. Checking NPT layers...")
    npt_count = 0
    quantized_count = 0
    adapter_info = []
    
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, 'adapter'):
            npt_count += 1
            
            # Check if quantized
            is_quantized = hasattr(layer.mlp.gate_proj, 'weight') and hasattr(layer.mlp.gate_proj.weight, 'CB')
            if is_quantized:
                quantized_count += 1
            
            # Get adapter parameters
            info = {
                'layer': i,
                'quantized': is_quantized,
                'consolidation_alpha': getattr(layer, 'consolidation_alpha', 'N/A'),
                'modulation_scale': getattr(layer, 'modulation_scale', 'N/A'),
                'adapter_r': layer.adapter.r if hasattr(layer.adapter, 'r') else 'N/A'
            }
            adapter_info.append(info)
    
    print(f"✓ Found {npt_count} NPT layers")
    if quantized_count > 0:
        print(f"⚠️  WARNING: {quantized_count} layers are quantized - permanent updates won't work on these!")
    
    # Show first few layers
    print("\nFirst 3 NPT layers:")
    for info in adapter_info[:3]:
        print(f"  Layer {info['layer']}: "
              f"quantized={info['quantized']}, "
              f"alpha={info['consolidation_alpha']}, "
              f"scale={info['modulation_scale']}, "
              f"rank={info['adapter_r']}")
    
    # 3. Test weight modulation generation
    print("\n3. Testing weight modulation generation...")
    test_fact = "The capital of Mars is Olympus."
    
    # Get a simple forward pass
    inputs = tokenizer(test_fact, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs.input_ids.to(device)
    
    # Check one layer's modulation
    with torch.no_grad():
        hidden_states = model.model.embed_tokens(input_ids)
        
        # Find first non-quantized NPT layer
        test_layer = None
        for layer in model.model.layers:
            if hasattr(layer, 'adapter'):
                is_quantized = hasattr(layer.mlp.gate_proj, 'weight') and hasattr(layer.mlp.gate_proj.weight, 'CB')
                if not is_quantized:
                    test_layer = layer
                    break
        
        if test_layer is None:
            print("✗ No non-quantized NPT layers found!")
            return
        
        # Get modulation from adapter
        modulation = test_layer.adapter(hidden_states)
        
        print(f"✓ Modulation generated successfully")
        print(f"  Vector model norm: {torch.norm(modulation['vector_model']).item():.6f}")
        print(f"  Vector FFN norm: {torch.norm(modulation['vector_ffn']).item():.6f}")
        
        if torch.norm(modulation['vector_model']).item() < 1e-6:
            print("⚠️  WARNING: Modulation vectors are near zero - updates will have no effect!")
    
    # 4. Test actual permanent update
    print("\n4. Testing permanent update...")
    
    # Save initial weight norm
    initial_weight_norm = torch.norm(test_layer.mlp.gate_proj.weight.data).item()
    
    # Apply update with high alpha
    print(f"Applying update with fact: '{test_fact}'")
    demonstrate_permanent_update(model, tokenizer, test_fact)
    
    # Check weight change
    final_weight_norm = torch.norm(test_layer.mlp.gate_proj.weight.data).item()
    weight_change = abs(final_weight_norm - initial_weight_norm)
    
    print(f"Weight norm change: {weight_change:.8f}")
    if weight_change < 1e-6:
        print("✗ No weight change detected!")
        print("\nPossible issues:")
        print("- Alpha value too small (default is 0.1)")
        print("- Modulation vectors are too small")
        print("- Model layers are quantized")
    else:
        print("✓ Weights were updated")
    
    # 5. Test generation
    print("\n5. Testing generation after update...")
    prompt = "The capital of Mars is"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = response[len(prompt):].strip()
    print(f"Response: '{completion}'")
    
    if "olympus" in completion.lower():
        print("✓ Fact successfully recalled!")
    else:
        print("✗ Fact not recalled")
    
    # 6. Recommendations
    print("\n6. Recommendations:")
    if quantized_count > 0:
        print("- Your model has quantized layers which cannot be updated")
        print("- Consider using a non-quantized model for permanent updates")
    
    if weight_change < 1e-6:
        print("- Try increasing the alpha value (e.g., 1.0 or higher)")
        print("- Use the enhanced_permanent_update.py script for more strategies")
        print("- Check if adapter weights are properly initialized")
    
    print("\nDiagnostics complete!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_permanent_update.py <checkpoint_path>")
        sys.exit(1)
    
    diagnose_permanent_update(sys.argv[1])

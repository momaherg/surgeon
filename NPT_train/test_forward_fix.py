"""
Quick test to verify the forward pass fix works correctly.
"""

import torch
import sys

def test_forward_fix():
    """Test that the forward pass doesn't have parameter conflicts."""
    print("Testing NPT forward pass fix...")
    
    try:
        # Test with optimized model
        from npt_model_optimized import NPTModelWrapperOptimized
        
        print("1. Creating small optimized model...")
        model = NPTModelWrapperOptimized(
            base_model_name="gpt2",
            npt_layers=[0, 1],
            rank=4,
            modulation_scale=0.1,
            use_cpu_offload=True,
            torch_dtype=torch.float16,
        )
        
        print("2. Testing forward pass...")
        input_ids = torch.randint(0, 1000, (1, 10))
        
        # Test with explicit output_hidden_states (this was causing the error)
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_original_outputs=False,
        )
        
        print("3. Checking outputs...")
        assert 'logits' in outputs
        assert 'hidden_states' in outputs
        print(f"   - Logits shape: {outputs['logits'].shape}")
        print(f"   - Hidden states: {len(outputs['hidden_states'])} layers")
        
        # Test with return_original_outputs
        print("4. Testing with return_original_outputs...")
        outputs = model(
            input_ids=input_ids,
            return_original_outputs=True,
        )
        assert 'layer_outputs' in outputs
        print(f"   - Layer outputs: {len(outputs['layer_outputs'])} NPT layers")
        
        print("\n✅ All tests passed! The forward pass fix is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_original_model():
    """Test the original model wrapper too."""
    print("\n5. Testing original model wrapper...")
    
    try:
        from npt_model import NPTModelWrapper
        
        model = NPTModelWrapper(
            base_model_name="gpt2",
            npt_layers=[0, 1],
            rank=4,
            modulation_scale=0.1,
        )
        
        input_ids = torch.randint(0, 1000, (1, 10))
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_original_outputs=False,
        )
        
        assert 'logits' in outputs
        assert 'hidden_states' in outputs
        print("   ✅ Original model wrapper also working correctly.")
        return True
        
    except Exception as e:
        print(f"   ❌ Original model test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_forward_fix()
    success = test_original_model() and success
    
    if success:
        print("\n✅ All tests passed! You can now run training with:")
        print("   ./launch_optimized.sh")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

"""
Test script to verify NPT setup and basic functionality.
"""

import torch
import yaml
from transformers import AutoTokenizer
import sys
import traceback


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import npt_components
        import npt_model
        import train_npt_equivalence
        import evaluate_npt
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        traceback.print_exc()
        return False


def test_npt_component():
    """Test NPT component creation and forward pass."""
    print("\nTesting NPT component...")
    try:
        from npt_components import NeuroPlasticComponent
        
        # Create component
        d_model, d_ffn, rank = 768, 3072, 16
        np_component = NeuroPlasticComponent(d_model, d_ffn, rank)
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        attn_output = torch.randn(batch_size, seq_len, d_model)
        modulation = np_component(attn_output)
        
        # Check modulation shape
        expected_shape = (batch_size, seq_len, rank)
        assert modulation.shape == expected_shape, f"Wrong shape: {modulation.shape} != {expected_shape}"
        
        # Test weight delta computation for a single token
        delta_w = np_component.compute_weight_delta(modulation, token_idx=0)
        expected_delta_shape = (batch_size, d_model, d_ffn)
        assert delta_w.shape == expected_delta_shape, f"Wrong delta shape: {delta_w.shape} != {expected_delta_shape}"
        
        # Check magnitude
        delta_norm = torch.norm(delta_w, p='fro', dim=(-2, -1)).mean()
        print(f"  Average delta norm: {delta_norm:.6f}")
        assert delta_norm < 1.0, "Delta norm too large"
        
        print("✓ NPT component working correctly")
        return True
    except Exception as e:
        print(f"✗ NPT component error: {e}")
        traceback.print_exc()
        return False


def test_model_wrapper():
    """Test NPT model wrapper with a small model."""
    print("\nTesting NPT model wrapper...")
    try:
        from npt_model import NPTModelWrapper
        
        # Use a tiny model for testing
        model_name = "sshleifer/tiny-gpt2"
        print(f"  Loading {model_name}...")
        
        # Create NPT model
        npt_model = NPTModelWrapper(
            base_model_name=model_name,
            npt_layers=[0, 1],  # Convert first two layers
            rank=8,
            modulation_scale=0.1,
        )
        
        # Check converted layers
        print(f"  Converted layers: {npt_model.npt_layer_indices}")
        print(f"  Number of trainable parameters: {len(npt_model.get_trainable_parameters())}")
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (1, 20))
        outputs = npt_model(input_ids, return_original_outputs=True)
        
        # Check outputs
        assert 'logits' in outputs, "Missing logits in output"
        assert 'layer_outputs' in outputs, "Missing layer outputs"
        
        print("✓ NPT model wrapper working correctly")
        return True
    except Exception as e:
        print(f"✗ Model wrapper error: {e}")
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration...")
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['model', 'training', 'data', 'wandb', 'paths']
        for field in required_fields:
            assert field in config, f"Missing field: {field}"
        
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_loss_function():
    """Test equivalence loss computation."""
    print("\nTesting loss function...")
    try:
        from train_npt_equivalence import EquivalenceLoss
        from npt_model import NPTModelWrapper
        
        # Create dummy data
        layer_outputs = {
            0: {
                'npt': torch.randn(2, 10, 768),
                'original': torch.randn(2, 10, 768),
            }
        }
        
        # Create loss function
        loss_fn = EquivalenceLoss(
            equivalence_weight=1.0,
            regularization_weight=0.01,
        )
        
        # Create dummy model (needed for regularization)
        class DummyModel:
            def __init__(self):
                from npt_components import NeuroPlasticComponent
                self.npt_layers = {
                    '0': type('obj', (object,), {
                        'np_component': NeuroPlasticComponent(768, 3072, 16)
                    })()
                }
        
        dummy_model = DummyModel()
        
        # Compute loss
        loss_dict = loss_fn(layer_outputs, dummy_model)
        
        # Check outputs
        assert 'loss' in loss_dict, "Missing total loss"
        assert 'equivalence_loss' in loss_dict, "Missing equivalence loss"
        assert 'regularization_loss' in loss_dict, "Missing regularization loss"
        
        print(f"  Total loss: {loss_dict['loss'].item():.6f}")
        print(f"  Equivalence loss: {loss_dict['equivalence_loss'].item():.6f}")
        print(f"  Regularization loss: {loss_dict['regularization_loss'].item():.6f}")
        
        print("✓ Loss function working correctly")
        return True
    except Exception as e:
        print(f"✗ Loss function error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*50)
    print("NPT Setup Verification")
    print("="*50)
    
    tests = [
        test_imports,
        test_npt_component,
        test_model_wrapper,
        test_config_loading,
        test_loss_function,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("="*50)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("\n✓ All tests passed! NPT is ready for training.")
        print("\nTo start training, run:")
        print("  ./launch_training.sh")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

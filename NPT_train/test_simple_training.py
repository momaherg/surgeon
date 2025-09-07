"""
Simple test to verify training can start without errors.
"""

import torch
import os
import yaml

def test_simple_training():
    """Test basic training loop with minimal configuration."""
    print("Testing simple NPT training...")
    
    # Force CPU for testing
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Create minimal config
    config = {
        'model': {
            'base_model_name': 'gpt2',
            'npt_layers': [0],  # Only convert one layer
            'rank': 4,
            'modulation_scale': 0.1,
        },
        'training': {
            'batch_size': 1,
            'gradient_accumulation_steps': 1,
            'learning_rate': 0.0001,
            'num_epochs': 1,
            'warmup_steps': 0,
            'weight_decay': 0.0,
            'max_grad_norm': 1.0,
            'equivalence_weight': 1.0,
            'regularization_weight': 0.01,
            'logging_steps': 1,
            'prediction_logging_steps': 10,
            'eval_steps': 10,
            'save_steps': 10,
        },
        'data': {
            'dataset_name': 'wikitext',
            'dataset_config': 'wikitext-2-raw-v1',
            'max_length': 128,
            'num_train_samples': 10,
            'num_eval_samples': 5,
        },
        'wandb': {
            'enabled': False,
        },
        'optimization': {
            'use_cpu_offload': True,
            'use_8bit': False,
            'mixed_precision': 'no',  # No mixed precision for CPU
        },
        'paths': {
            'output_dir': './test_output',
            'cache_dir': './test_cache',
            'checkpoint_dir': './test_checkpoints',
        }
    }
    
    # Save config
    os.makedirs('test_output', exist_ok=True)
    with open('test_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    try:
        # Import and run training
        from train_npt_optimized import train_npt_optimized
        
        print("\nStarting test training...")
        train_npt_optimized('test_config.yaml', use_optimized=True)
        
        print("\n✅ Training test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        import shutil
        for dir_name in ['test_output', 'test_cache', 'test_checkpoints']:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
        if os.path.exists('test_config.yaml'):
            os.remove('test_config.yaml')


if __name__ == "__main__":
    test_simple_training()

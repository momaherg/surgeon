"""
Quick test of the mixed teacher forcing training script.
"""

import subprocess
import sys

# Test command with minimal settings for quick verification
test_command = [
    sys.executable,
    "pretrain_npt_mixed_teacher_forcing.py",
    "--model_name", "meta-llama/Llama-3.1-8B",
    "--adapter_rank", "32",
    "--modulation_scale", "0.05",
    "--init_scale", "0.5",
    "--regularization_lambda", "0.01",
    "--curriculum_steps", "100",  # Quick test
    "--initial_teacher_ratio", "0.9",
    "--final_teacher_ratio", "0.1",
    "--hidden_loss_weight", "0.5",
    "--logits_loss_weight", "0.5",
    "--batch_size", "1",
    "--gradient_accumulation_steps", "2",
    "--learning_rate", "5e-5",
    "--max_grad_norm", "1.0",
    "--share_embeddings",
    "--mixed_precision", "bf16",
    "--max_length", "256",  # Shorter for quick test
    "--max_steps", "20",  # Just 20 steps to verify it runs
    "--streaming",
    "--loss_type", "improved_npt",
    "--mse_weight", "0.7",
    "--cosine_weight", "0.3",
    "--warmup_steps", "5",
    "--save_steps", "10",
    "--prediction_steps", "10",
    "--log_steps", "1",
    "--output_dir", "./outputs/npt-mixed-teacher-test"
]

print("Running quick test of mixed teacher forcing training...")
print("Command:", " ".join(test_command))
print("-" * 80)

# Run the command
try:
    result = subprocess.run(test_command, capture_output=False, text=True)
    if result.returncode == 0:
        print("\n" + "="*80)
        print("SUCCESS: Mixed teacher forcing training is working!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print(f"ERROR: Training failed with return code {result.returncode}")
        print("="*80)
except Exception as e:
    print(f"\nERROR: Failed to run training: {e}")

print("\nTest complete!")

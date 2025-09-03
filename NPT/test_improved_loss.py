"""
Test script to demonstrate improved loss functions for NPT training.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig
from improved_loss import (
    ImprovedNPTLoss, 
    FocalMSELoss, 
    SmoothL1MSELoss,
    CombinedMSECosineLoss,
    AdaptiveLayerWeights
)


def test_adaptive_layer_weights():
    """Test adaptive layer weighting mechanism."""
    print("Testing Adaptive Layer Weights...")
    
    num_layers = 12
    adaptive_weights = AdaptiveLayerWeights(num_layers)
    
    # Create dummy layer losses with different scales
    layer_losses = []
    for i in range(num_layers):
        # Early layers have smaller losses, later layers have larger losses
        loss_scale = 0.1 * (i + 1)
        loss = torch.tensor(loss_scale, requires_grad=True)
        layer_losses.append(loss)
    
    # Apply adaptive weighting
    weighted_loss = adaptive_weights(layer_losses)
    
    print(f"Layer losses: {[l.item() for l in layer_losses]}")
    print(f"Layer weights: {adaptive_weights.layer_weights.data.tolist()}")
    print(f"Weighted loss: {weighted_loss.item():.4f}")
    print()


def test_combined_mse_cosine():
    """Test combined MSE and cosine similarity loss."""
    print("Testing Combined MSE + Cosine Loss...")
    
    batch_size, seq_len, hidden_dim = 2, 10, 768
    
    # Create dummy student and teacher outputs
    student = torch.randn(batch_size, seq_len, hidden_dim)
    teacher = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Test with different alignments
    loss_fn = CombinedMSECosineLoss(mse_weight=0.8, cosine_weight=0.2)
    
    # Case 1: Random tensors
    total_loss1, mse1, cosine1 = loss_fn(student, teacher)
    print(f"Random tensors - Total: {total_loss1:.4f}, MSE: {mse1:.4f}, Cosine: {cosine1:.4f}")
    
    # Case 2: Aligned direction but different magnitude
    student_aligned = 2.0 * teacher
    total_loss2, mse2, cosine2 = loss_fn(student_aligned, teacher)
    print(f"Aligned direction - Total: {total_loss2:.4f}, MSE: {mse2:.4f}, Cosine: {cosine2:.4f}")
    
    # Case 3: Same magnitude but different direction
    student_rotated = torch.roll(teacher, shifts=1, dims=-1)
    total_loss3, mse3, cosine3 = loss_fn(student_rotated, teacher)
    print(f"Different direction - Total: {total_loss3:.4f}, MSE: {mse3:.4f}, Cosine: {cosine3:.4f}")
    print()


def test_improved_npt_loss():
    """Test the full improved NPT loss function."""
    print("Testing Improved NPT Loss...")
    
    # Create dummy model config
    config = type('Config', (), {})()
    config.num_hidden_layers = 12
    
    # Initialize loss function
    loss_fn = ImprovedNPTLoss(
        num_layers=config.num_hidden_layers,
        mse_weight=0.8,
        cosine_weight=0.2,
        regularization_lambda=0.01,
        gradient_penalty_lambda=0.001,
        temperature=3.0,
        use_adaptive_weights=True,
        use_gradient_penalty=True,
        warmup_steps=1000
    )
    
    # Create dummy hidden states
    batch_size, seq_len, hidden_dim = 2, 10, 768
    teacher_hidden_states = []
    student_hidden_states = []
    
    for i in range(config.num_hidden_layers + 1):
        teacher_hidden = torch.randn(batch_size, seq_len, hidden_dim)
        # Student is slightly off from teacher
        student_hidden = teacher_hidden + 0.1 * torch.randn_like(teacher_hidden)
        
        teacher_hidden_states.append(teacher_hidden)
        student_hidden_states.append(student_hidden)
    
    # Create dummy regularization norms
    reg_norms = [torch.tensor(0.01) for _ in range(config.num_hidden_layers)]
    
    # Test at different training steps
    for step in [0, 500, 1000, 2000]:
        loss_dict = loss_fn(
            teacher_hidden_states=teacher_hidden_states,
            student_hidden_states=student_hidden_states,
            reg_norms=reg_norms,
            step=step
        )
        
        print(f"\nStep {step}:")
        print(f"  Total loss: {loss_dict['total_loss']:.4f}")
        print(f"  Alignment loss: {loss_dict['alignment_loss']:.4f}")
        print(f"  MSE loss: {loss_dict['mse_loss']:.4f}")
        print(f"  Cosine loss: {loss_dict['cosine_loss']:.4f}")
        print(f"  Regularization loss: {loss_dict['reg_loss']:.4f}")
        print(f"  Gradient penalty: {loss_dict['grad_penalty']:.4f}")
        print(f"  Curriculum factor: {loss_dict['curriculum_factor']:.4f}")
        print(f"  Temperature: {loss_dict['temperature']:.4f}")


def test_focal_mse_loss():
    """Test focal MSE loss for handling outliers."""
    print("\nTesting Focal MSE Loss...")
    
    batch_size, hidden_dim = 10, 768
    
    # Create predictions and targets
    pred = torch.randn(batch_size, hidden_dim)
    target = torch.randn(batch_size, hidden_dim)
    
    # Add some outliers
    pred[0] = target[0] + 5.0  # Large error
    pred[1] = target[1] + 0.1  # Small error
    
    # Compare standard MSE with focal MSE
    standard_mse = nn.MSELoss()
    focal_mse = FocalMSELoss(gamma=2.0)
    
    standard_loss = standard_mse(pred, target)
    focal_loss = focal_mse(pred, target)
    
    print(f"Standard MSE loss: {standard_loss:.4f}")
    print(f"Focal MSE loss: {focal_loss:.4f}")
    print(f"Ratio (focal/standard): {focal_loss/standard_loss:.4f}")
    
    # Show per-sample contribution
    mse_per_sample = ((pred - target) ** 2).mean(dim=1)
    print(f"\nPer-sample MSE: {mse_per_sample.tolist()}")


def test_smooth_l1_mse():
    """Test Smooth L1 + MSE combination."""
    print("\nTesting Smooth L1 + MSE Loss...")
    
    batch_size, hidden_dim = 10, 768
    
    # Create predictions and targets
    pred = torch.randn(batch_size, hidden_dim)
    target = torch.randn(batch_size, hidden_dim)
    
    # Add outliers
    pred[0] = target[0] + 10.0  # Very large error
    
    # Compare different loss functions
    mse_loss = nn.MSELoss()
    smooth_l1_loss = nn.SmoothL1Loss(beta=1.0)
    combined_loss = SmoothL1MSELoss(beta=1.0, mse_weight=0.5)
    
    mse = mse_loss(pred, target)
    smooth_l1 = smooth_l1_loss(pred, target)
    combined = combined_loss(pred, target)
    
    print(f"MSE loss: {mse:.4f}")
    print(f"Smooth L1 loss: {smooth_l1:.4f}")
    print(f"Combined loss: {combined:.4f}")
    print()


def compare_convergence_properties():
    """Compare convergence properties of different loss functions."""
    print("\nComparing Convergence Properties...")
    
    # Simulate a training scenario
    batch_size, seq_len, hidden_dim = 2, 10, 768
    num_steps = 100
    
    # Initialize student parameters
    student_param = nn.Parameter(torch.randn(hidden_dim))
    
    # Fixed teacher
    teacher = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Different loss functions
    loss_fns = {
        'MSE': nn.MSELoss(),
        'Smooth L1': nn.SmoothL1Loss(),
        'Combined': CombinedMSECosineLoss(0.8, 0.2),
        'Focal MSE': FocalMSELoss(gamma=2.0)
    }
    
    for name, loss_fn in loss_fns.items():
        # Reset student parameter
        student_param.data = torch.randn(hidden_dim)
        optimizer = torch.optim.Adam([student_param], lr=0.01)
        
        losses = []
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Create student output
            student = student_param.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            
            # Compute loss
            if isinstance(loss_fn, CombinedMSECosineLoss):
                loss, _, _ = loss_fn(student, teacher)
            else:
                loss = loss_fn(student, teacher)
            
            losses.append(loss.item())
            
            # Backward and update
            loss.backward()
            optimizer.step()
        
        # Print convergence statistics
        print(f"\n{name}:")
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")
        print(f"  Smoothness (std of loss changes): {torch.std(torch.diff(torch.tensor(losses))):.4f}")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing Improved Loss Functions for NPT Training")
    print("=" * 70)
    
    test_adaptive_layer_weights()
    test_combined_mse_cosine()
    test_improved_npt_loss()
    test_focal_mse_loss()
    test_smooth_l1_mse()
    compare_convergence_properties()
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main()

"""
Visualization utilities for NPT training monitoring.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import pandas as pd
import os
from datetime import datetime


def plot_training_curves(
    log_file: str,
    save_dir: str = "plots",
    metrics: Optional[List[str]] = None,
):
    """Plot training curves from log file."""
    if metrics is None:
        metrics = [
            'train/loss',
            'train/equivalence_loss',
            'train/regularization_loss',
            'train/avg_delta_norm',
            'eval/eval_loss',
        ]
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Parse log file
    data = {'step': [], 'epoch': []}
    for metric in metrics:
        data[metric] = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'train/' in line or 'eval/' in line:
                # Simple parsing - adjust based on actual log format
                try:
                    parts = line.strip().split()
                    for i, part in enumerate(parts):
                        if 'step' in part:
                            data['step'].append(int(parts[i+1]))
                        for metric in metrics:
                            if metric in part:
                                data[metric].append(float(parts[i+1]))
                except:
                    continue
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot 1: Total loss
    if 'train/loss' in df.columns:
        axes[0].plot(df['step'], df['train/loss'], label='Training Loss')
        if 'eval/eval_loss' in df.columns:
            eval_df = df[df['eval/eval_loss'].notna()]
            axes[0].plot(eval_df['step'], eval_df['eval/eval_loss'], 
                        marker='o', label='Eval Loss')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Equivalence vs Regularization
    if 'train/equivalence_loss' in df.columns:
        axes[1].plot(df['step'], df['train/equivalence_loss'], 
                    label='Equivalence Loss')
        axes[1].plot(df['step'], df['train/regularization_loss'], 
                    label='Regularization Loss')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss Components')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Weight Delta Norm
    if 'train/avg_delta_norm' in df.columns:
        axes[2].plot(df['step'], df['train/avg_delta_norm'], 
                    color='green', linewidth=2)
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('Average ||ΔW||')
        axes[2].set_title('Weight Delta Magnitude')
        axes[2].grid(True, alpha=0.3)
        
        # Add target line
        axes[2].axhline(y=0.01, color='red', linestyle='--', 
                       label='Target (<0.01)')
        axes[2].legend()
    
    # Plot 4: Learning Rate
    if 'train/learning_rate' in df.columns:
        axes[3].plot(df['step'], df['train/learning_rate'], 
                    color='orange', linewidth=2)
        axes[3].set_xlabel('Step')
        axes[3].set_ylabel('Learning Rate')
        axes[3].set_title('Learning Rate Schedule')
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.close()


def plot_weight_delta_distribution(
    checkpoint_path: str,
    save_dir: str = "plots",
):
    """Plot weight delta distribution from checkpoint."""
    import torch
    from npt_model import NPTModelWrapper
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # Create model and load weights
    model = NPTModelWrapper(
        base_model_name=config['base_model_name'],
        npt_layers=config['npt_layer_indices'],
        rank=config['rank'],
        modulation_scale=config['modulation_scale'],
    )
    model.load_npt_components(checkpoint_path)
    
    # Generate some random inputs to get weight deltas
    batch_size, seq_len = 32, 128
    d_model = config['d_model']
    
    all_deltas = []
    layer_deltas = {}
    
    with torch.no_grad():
        for idx_str, npt_layer in model.npt_layers.items():
            # Random attention output
            attn_output = torch.randn(batch_size, seq_len, d_model)
            
            # Get modulation factors
            modulation = npt_layer.np_component(attn_output)
            
            # Sample a few tokens and compute their weight deltas
            num_token_samples = min(5, seq_len)
            token_deltas = []
            for token_idx in range(num_token_samples):
                delta_w = npt_layer.np_component.compute_weight_delta(modulation, token_idx)
                token_deltas.append(delta_w.cpu().numpy())
            
            # Flatten and store
            delta_flat = np.concatenate([d.flatten() for d in token_deltas])
            
            all_deltas.append(delta_flat)
            layer_deltas[f"Layer {idx_str}"] = delta_flat
    
    all_deltas = np.concatenate(all_deltas)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Overall distribution
    axes[0, 0].hist(all_deltas, bins=100, alpha=0.7, density=True)
    axes[0, 0].set_xlabel('Weight Delta Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Overall Weight Delta Distribution')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # Add statistics
    stats_text = f"Mean: {np.mean(all_deltas):.6f}\n"
    stats_text += f"Std: {np.std(all_deltas):.6f}\n"
    stats_text += f"Max: {np.max(np.abs(all_deltas)):.6f}"
    axes[0, 0].text(0.05, 0.95, stats_text, transform=axes[0, 0].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
    
    # Plot 2: Q-Q plot
    from scipy import stats
    stats.probplot(all_deltas, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Normal Distribution)')
    
    # Plot 3: Layer-wise box plots
    layer_data = []
    layer_names = []
    for name, deltas in layer_deltas.items():
        # Sample for visualization
        sample_indices = np.random.choice(len(deltas), 
                                        min(10000, len(deltas)), 
                                        replace=False)
        layer_data.append(deltas[sample_indices])
        layer_names.append(name)
    
    axes[1, 0].boxplot(layer_data, labels=layer_names)
    axes[1, 0].set_ylabel('Weight Delta Value')
    axes[1, 0].set_title('Layer-wise Delta Distribution')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Frobenius norm by layer
    layer_norms = []
    for name, deltas in layer_deltas.items():
        # Reshape and compute Frobenius norm
        num_samples = batch_size * seq_len
        delta_matrices = deltas.reshape(num_samples, -1)
        norms = np.linalg.norm(delta_matrices, axis=1)
        layer_norms.append(np.mean(norms))
    
    axes[1, 1].bar(range(len(layer_names)), layer_norms)
    axes[1, 1].set_xticks(range(len(layer_names)))
    axes[1, 1].set_xticklabels(layer_names, rotation=45)
    axes[1, 1].set_ylabel('Average Frobenius Norm')
    axes[1, 1].set_title('Average ||ΔW|| by Layer')
    axes[1, 1].axhline(y=0.01, color='red', linestyle='--', 
                       label='Target (<0.01)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'weight_delta_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Weight delta distribution saved to: {save_path}")
    plt.close()


def create_training_dashboard(
    wandb_run_path: str,
    save_dir: str = "dashboard",
):
    """Create a comprehensive training dashboard from WandB data."""
    try:
        import wandb
        api = wandb.Api()
        run = api.run(wandb_run_path)
        
        # Download run history
        history = run.history()
        config = run.config
        
        # Create dashboard directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Create HTML dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NPT Training Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .metric-box {{ 
                    display: inline-block; 
                    padding: 20px; 
                    margin: 10px;
                    background: #f0f0f0; 
                    border-radius: 5px; 
                }}
                .metric-value {{ 
                    font-size: 24px; 
                    font-weight: bold; 
                    color: #007bff; 
                }}
                img {{ max-width: 100%; height: auto; }}
                .section {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <h1>NPT Training Dashboard</h1>
            <p>Run: {run.name} | Date: {run.created_at}</p>
            
            <div class="section">
                <h2>Configuration</h2>
                <ul>
                    <li>Base Model: {config.get('model', {}).get('base_model_name', 'N/A')}</li>
                    <li>NPT Layers: {config.get('model', {}).get('npt_layers', 'N/A')}</li>
                    <li>Rank: {config.get('model', {}).get('rank', 'N/A')}</li>
                    <li>Learning Rate: {config.get('training', {}).get('learning_rate', 'N/A')}</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Final Metrics</h2>
                <div class="metric-box">
                    <div>Final Loss</div>
                    <div class="metric-value">{history['train/loss'].iloc[-1]:.4f}</div>
                </div>
                <div class="metric-box">
                    <div>Final Eval Loss</div>
                    <div class="metric-value">{history['eval/eval_loss'].iloc[-1]:.4f}</div>
                </div>
                <div class="metric-box">
                    <div>Final ΔW Norm</div>
                    <div class="metric-value">{history['train/avg_delta_norm'].iloc[-1]:.6f}</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Training Progress</h2>
                <img src="training_curves.png" alt="Training Curves">
            </div>
            
            <div class="section">
                <h2>Generated Samples</h2>
                <p>Latest generated text samples from the model...</p>
            </div>
        </body>
        </html>
        """
        
        with open(os.path.join(save_dir, 'index.html'), 'w') as f:
            f.write(html_content)
        
        # Save history data
        history.to_csv(os.path.join(save_dir, 'training_history.csv'))
        
        print(f"Dashboard created at: {save_dir}/index.html")
        
    except Exception as e:
        print(f"Error creating dashboard: {e}")
        print("Make sure you have wandb installed and are logged in.")


def main():
    """Run visualization from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize NPT training")
    parser.add_argument('--log', type=str, help="Path to training log file")
    parser.add_argument('--checkpoint', type=str, help="Path to checkpoint file")
    parser.add_argument('--wandb', type=str, help="WandB run path (entity/project/run_id)")
    parser.add_argument('--output', type=str, default="plots", help="Output directory")
    
    args = parser.parse_args()
    
    if args.log:
        plot_training_curves(args.log, args.output)
    
    if args.checkpoint:
        plot_weight_delta_distribution(args.checkpoint, args.output)
    
    if args.wandb:
        create_training_dashboard(args.wandb, args.output)


if __name__ == "__main__":
    main()

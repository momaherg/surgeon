# Neuro-Plastic Transformer (NPT) - Phase 1: Equivalence Pre-training

This repository implements the initial training phase for the Neuro-Plastic Transformer (NPT), a novel architecture that enhances transformer models with dynamic weight modulation capabilities.

## 🎯 Overview

The NPT architecture replaces standard additive residual connections with a **Neuro-Plastic (NP) Component** that uses attention outputs to dynamically modulate MLP weights on a per-token basis. This Phase 1 training ensures the NPT model starts as a high-fidelity equivalent of the base pre-trained model.

### Key Features
- **Dynamic Weight Modulation**: Attention outputs generate transient weight deltas (ΔW) for MLPs
- **Low-Rank Adaptation**: Efficient parameter usage via low-rank decomposition (A and B matrices)
- **Selective Layer Conversion**: Flexibility to convert specific layers (e.g., upper half only)
- **Equivalence Training**: Maintains base model capabilities while adding plasticity

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended: 24GB+ VRAM for 7B models)
- 32GB+ system RAM

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd NPT_train

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Training

Edit `config.yaml` to set your preferences:

```yaml
model:
  base_model_name: "meta-llama/Llama-2-7b-hf"  # Base model to convert
  npt_layers: "upper_half"  # Which layers to convert
  rank: 16  # Rank for low-rank adapters
  
training:
  batch_size: 4
  learning_rate: 1e-4
  num_epochs: 3
```

### 3. Launch Training

```bash
# Use the provided launcher script
./launch_training.sh

# Or run directly with accelerate
accelerate launch train_npt_equivalence.py --config config.yaml
```

## 🏗️ Architecture

### Standard Transformer Block:
```
1. attn_output = SelfAttention(LayerNorm(h))
2. h_residual = h + attn_output
3. output = MLP(LayerNorm(h_residual))
```

### NPT Block:
```
1. attn_output = SelfAttention(LayerNorm(h))
2. ΔW_in = NP_Component(attn_output)      // Generate weight delta
3. W_in_modulated = W_in_base + ΔW_in    // Modulate weights
4. output = MLP_out(GELU(W_in_modulated @ LayerNorm(h))) + h
```

## 📁 Project Structure

```
NPT_train/
├── config.yaml              # Training configuration
├── requirements.txt         # Python dependencies
├── launch_training.sh       # Training launcher script
├── npt_components.py        # Core NPT components (NP Component, NPTLayer)
├── npt_model.py            # NPT model wrapper
├── train_npt_equivalence.py # Main training script
├── evaluate_npt.py         # Evaluation utilities
└── README.md               # This file
```

## 🔧 Configuration Options

### Model Configuration
- `base_model_name`: Pre-trained model to use as base
- `npt_layers`: Which layers to convert ("all", "upper_half", "lower_half", or list)
- `rank`: Rank for low-rank matrices (lower = fewer parameters)
- `modulation_scale`: Scaling factor for weight deltas

### Training Configuration
- `batch_size`: Training batch size
- `gradient_accumulation_steps`: Steps to accumulate gradients
- `learning_rate`: Learning rate for Adam optimizer
- `equivalence_weight`: Weight for equivalence loss
- `regularization_weight`: Weight for low-magnitude constraint

## 📊 Monitoring Training

Training progress is tracked via WandB, including:
- Loss curves (equivalence and regularization)
- Weight delta statistics
- Sample generations every 150 steps
- Perplexity evaluations

Access your dashboard at [wandb.ai](https://wandb.ai) after training starts.

## 🧪 Evaluation

After training, evaluate your model:

```bash
python evaluate_npt.py \
    --checkpoint checkpoints/best_model.pt \
    --config config.yaml \
    --output evaluation_report.json
```

This generates a comprehensive report including:
- Perplexity comparison (NPT vs original)
- Layer-wise equivalence metrics
- Weight delta distribution analysis
- Generation quality samples

## 📈 Expected Outcomes

Upon successful Phase 1 completion:
1. **High Functional Fidelity**: NPT model performance ≈ original model
2. **Low-Magnitude ΔW**: Average Frobenius norm < 0.01
3. **Perplexity Ratio**: NPT/Original ≈ 1.0 (±0.05)

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce `batch_size` in config.yaml
   - Use gradient checkpointing (if implemented)
   - Convert fewer layers

2. **High Equivalence Loss**:
   - Increase `num_epochs`
   - Adjust `learning_rate` (try 5e-5 or 2e-4)
   - Check if base model weights are properly frozen

3. **Unstable Training**:
   - Reduce `modulation_scale`
   - Increase `regularization_weight`
   - Use gradient clipping (`max_grad_norm`)

## 📚 Technical Details

### Loss Function

The training uses a composite loss:

```
L_total = α * L_equivalence + β * L_regularization

where:
- L_equivalence = MSE(NPT_output, Original_output)
- L_regularization = ||A||_F + ||B||_F
```

### Weight Delta Generation

For each token position:
```
ΔW = modulation_scale * tanh(attn_output @ A) @ B
```

## 🚀 Next Steps

After Phase 1 completion:
- **Phase 2**: Functional fine-tuning for enhanced reasoning
- **Phase 3**: Permanent update experiments for knowledge editing

## 📝 Citation

If you use this implementation, please cite:

```bibtex
@techreport{npt2024,
  title={Neuro-Plastic Transformer: Initial Training Protocol},
  author={[Your Name/Lab]},
  year={2024},
  institution={[Your Institution]},
  number={NPT-TR-Phase1-v1.0}
}
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please submit issues or pull requests.

## 📞 Contact

For questions or collaborations, please contact: [your-email@example.com]

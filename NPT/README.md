# NLP NeuroPlastic Transformer (NPT)

This repository implements the NLP NeuroPlastic Transformer (NPT), a novel transformer architecture that modifies the standard self-attention to MLP connection. Instead of adding attention outputs to hidden states, NPT uses attention outputs to dynamically modulate MLP activations through low-rank adapters, preserving per-token dynamics.

## Overview

The NPT architecture introduces a new mechanism where:
- Self-attention outputs generate per-token modulation effects through low-rank factorization
- These effects are added to MLP gate activations (not weights) for computational efficiency
- Per-token dynamics are preserved, allowing each token to have unique modulation
- The residual connection pattern maintains training stability

Key architectural improvements:
- **Per-token modulation**: Each token receives unique modulation based on its attention pattern
- **Activation-level modulation**: More stable than weight modulation, easier to optimize
- **Efficient regularization**: Norms computed during forward pass, no redundant computation
- **Preserved distributions**: LayerNorm applied to standard residual paths for stability

This design aims to improve in-context learning and parameter efficiency compared to standard transformer architectures.

## Architecture Details

### Standard Transformer Block
```python
attn_output = self_attention(layer_norm(h))
h_residual = h + attn_output  # Standard residual connection
mlp_output = mlp(layer_norm(h_residual))
output = h_residual + mlp_output
```

### NPT Block
```python
attn_output = self_attention(layer_norm(h))
delta_effect = adapter(attn_output)  # Generate per-token modulation
h_residual = h + attn_output  # Standard residual for stability
mlp_input = layer_norm(h_residual)
gate = gate_proj(mlp_input) + delta_effect  # Add modulation to activations
mlp_output = down_proj(silu(gate) * up_proj(mlp_input))
output = h + mlp_output  # Final residual connection
```

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU with at least 24GB VRAM (e.g., RTX 3090/4090)
- PyTorch 2.1+

### Setup Environment

1. Clone the repository:
```bash
git clone <repository-url>
cd NPT
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Login to Hugging Face (required for Llama3 access):
```bash
huggingface-cli login
```

## Training Pipeline

The NPT training consists of two phases:

### Phase 1: Equivalence Pre-training

This phase trains the adapter modules to make the NPT model behave like the original Llama3 model.

**Basic Usage:**
```bash
python pretrain_npt.py \
    --model_name meta-llama/Llama-3.1-8B \
    --adapter_rank 16 \
    --use_quantization \
    --use_fp16 \
    --dataset_name cerebras/SlimPajama-627B \
    --streaming \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --regularization_lambda 0.01 \
    --output_dir ./outputs/npt-pretrained \
    --use_wandb
```

**Key Parameters:**
- `--adapter_rank`: Low-rank dimension for adapters (default: 16)
- `--regularization_lambda`: Weight delta regularization strength (default: 0.01)
- `--use_quantization`: Enable 4-bit quantization for memory efficiency
- `--streaming`: Use dataset streaming for large datasets
- `--max_steps`: Limit training steps (useful for debugging)

**Advanced Configuration:**
```bash
python pretrain_npt.py \
    --model_name meta-llama/Llama-3.1-8B \
    --adapter_rank 32 \
    --use_quantization \
    --use_fp16 \
    --dataset_name c4 \
    --dataset_split train \
    --streaming \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --regularization_lambda 0.005 \
    --num_epochs 1 \
    --warmup_steps 500 \
    --max_grad_norm 1.0 \
    --output_dir ./outputs/npt-pretrained-c4 \
    --log_steps 50 \
    --save_steps 1000 \
    --use_wandb \
    --run_name npt-pretrain-c4-r32
```

### Phase 2: Functional Fine-tuning

This phase fine-tunes the NPT model on downstream instruction-following tasks.

**Basic Usage:**
```bash
python finetune_npt.py \
    --checkpoint_path ./outputs/npt-pretrained/checkpoint-best \
    --use_quantization \
    --use_fp16 \
    --dataset_name HuggingFaceH4/ultrachat_200k \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --output_dir ./outputs/npt-finetuned \
    --use_wandb
```

**Key Parameters:**
- `--checkpoint_path`: Path to Phase 1 checkpoint (required)
- `--use_regularization`: Continue using weight delta regularization
- `--eval_split`: Enable evaluation during training
- `--eval_steps`: Frequency of evaluation

**Advanced Configuration:**
```bash
python finetune_npt.py \
    --checkpoint_path ./outputs/npt-pretrained/checkpoint-best \
    --use_quantization \
    --use_fp16 \
    --dataset_name HuggingFaceH4/ultrachat_200k \
    --train_split train_sft \
    --eval_split test_sft \
    --batch_size 2 \
    --eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --use_regularization \
    --regularization_lambda 0.001 \
    --num_epochs 3 \
    --warmup_steps 100 \
    --output_dir ./outputs/npt-finetuned-ultrachat \
    --log_steps 20 \
    --eval_steps 500 \
    --save_steps 1000 \
    --use_wandb \
    --run_name npt-finetune-ultrachat
```

## Dataset Options

### Pre-training Datasets
- `cerebras/SlimPajama-627B` (recommended, streaming)
- `c4` (Common Crawl)
- `allenai/c4` (Alternative C4)
- `openwebtext`

### Fine-tuning Datasets
- `HuggingFaceH4/ultrachat_200k` (recommended)
- `tatsu-lab/alpaca`
- `databricks/databricks-dolly-15k`
- `OpenAssistant/oasst1`

## Memory Optimization

For limited GPU memory:

1. **Enable 4-bit quantization:**
```bash
--use_quantization
```

2. **Reduce batch size and increase gradient accumulation:**
```bash
--batch_size 1 --gradient_accumulation_steps 16
```

3. **Use smaller adapter rank:**
```bash
--adapter_rank 8
```

4. **Enable CPU offloading (automatic with device_map="auto")**

## Multi-GPU Training

The scripts automatically use all available GPUs via Accelerate:

```bash
# Configure accelerate (first time only)
accelerate config

# Run with accelerate
accelerate launch pretrain_npt.py [args...]
accelerate launch finetune_npt.py [args...]
```

## Experiment Tracking

### Weights & Biases
```bash
# Login to W&B (first time only)
wandb login

# Enable W&B logging
--use_wandb --run_name my-experiment
```

### TensorBoard
```bash
# Enable TensorBoard logging
--use_tensorboard

# View logs
tensorboard --logdir ./outputs
```

## Model Inference

After training, use the model for inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_path = "./outputs/npt-finetuned/checkpoint-best"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Generate text
prompt = "Explain the concept of machine learning in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Troubleshooting

### Out of Memory Errors
- Enable 4-bit quantization: `--use_quantization`
- Reduce batch size: `--batch_size 1`
- Reduce sequence length: `--max_length 1024`
- Use smaller adapter rank: `--adapter_rank 8`

### Slow Training
- Enable mixed precision: `--mixed_precision fp16`
- Use dataset streaming: `--streaming`
- Increase gradient accumulation: `--gradient_accumulation_steps 16`

### NaN Loss
- Reduce learning rate: `--learning_rate 1e-5`
- Increase warmup steps: `--warmup_steps 500`
- Enable gradient clipping: `--max_grad_norm 0.5`

## Project Structure

```
NPT/
├── model/
│   ├── __init__.py
│   └── npt_layer.py          # NPT architecture implementation
├── utils.py                  # Utility functions
├── pretrain_npt.py          # Phase 1 training script
├── finetune_npt.py          # Phase 2 training script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{npt2024,
  title={NLP NeuroPlastic Transformer},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-username]/NPT}
}
```

## License

This project is licensed under the MIT License. Note that the Llama3 model weights are subject to Meta's license terms.

## Acknowledgments

- This implementation is based on the Llama3 architecture from Meta
- Thanks to the Hugging Face team for the transformers library
- Inspired by adapter-based methods like LoRA and recent work on dynamic neural networks

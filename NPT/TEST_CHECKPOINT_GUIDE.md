# NPT Checkpoint Testing Guide

This guide explains how to test your NPT checkpoint saved at step 500.

## Quick Start

### 1. Simple Testing
For a quick test with predefined sentences and interactive mode:

```bash
python quick_test_npt.py ./outputs/npt-safe-pretrained/checkpoint-500
```

This will:
- Test the model with 10 predefined prompts
- Enter an interactive mode where you can type your own prompts
- Show only NPT model outputs (no comparison)

### 2. Comprehensive Testing with Comparison
To compare NPT responses with the base model:

```bash
python test_checkpoint.py --checkpoint_path ./outputs/npt-safe-pretrained/checkpoint-500
```

This will:
- Run a comprehensive test suite
- Compare NPT outputs with base model outputs
- Test in-context learning capabilities
- Test knowledge recall
- Test various prompt types (questions, creative writing, code, etc.)

### 3. Test Specific Prompts
To test with your own prompt:

```bash
python test_checkpoint.py \
    --checkpoint_path ./outputs/npt-safe-pretrained/checkpoint-500 \
    --prompt "Your custom prompt here" \
    --max_new_tokens 100
```

### 4. Test NPT-Specific Capabilities
To analyze NPT's unique weight modulation patterns:

```bash
python test_npt_capabilities.py \
    --checkpoint_path ./outputs/npt-safe-pretrained/checkpoint-500
```

This will:
- Test context sensitivity (how context affects completions)
- Test adaptation capabilities (format and style adaptation)
- Analyze and visualize weight modulation patterns
- Save a visualization plot to `modulation_patterns.png`

## Options for test_checkpoint.py

- `--checkpoint_path`: Path to your checkpoint (required)
- `--base_model`: Base model name (default: meta-llama/Llama-3.1-8B)
- `--prompt`: Single prompt to test (if not provided, runs full test suite)
- `--max_new_tokens`: Max tokens to generate (default: 100)
- `--temperature`: Sampling temperature (default: 0.7)
- `--no_comparison`: Skip comparison with base model
- `--device`: Device to use (default: auto)

## Example Test Sentences

Here are some example sentences you can test:

1. **Simple Completions:**
   - "The future of artificial intelligence is"
   - "Climate change can be addressed by"
   - "The most important invention of the 21st century is"

2. **Knowledge Queries:**
   - "Explain the theory of relativity in simple terms:"
   - "What are the main differences between Python and JavaScript?"
   - "How does photosynthesis work?"

3. **Creative Tasks:**
   - "Write a haiku about machine learning:"
   - "Continue this story: In a world where time moves backwards,"
   - "Describe a sunset using only technical terms:"

4. **Reasoning Tasks:**
   - "If all birds can fly and penguins are birds, what can we conclude?"
   - "Solve this riddle: I have cities but no houses, forests but no trees. What am I?"
   - "What would happen if gravity was twice as strong?"

5. **Format-Specific Tasks:**
   - "List the top 5 programming languages in 2024:"
   - "Convert this to JSON: name is John, age is 30, city is New York"
   - "Translate 'Hello, how are you?' to French, Spanish, and German:"

## Troubleshooting

1. **Out of Memory Error:**
   - Try using a smaller model or reducing max_length
   - Use quantization: ensure your checkpoint was trained with quantization

2. **Checkpoint Not Found:**
   - Verify the checkpoint path exists
   - Check if you're using the correct output directory

3. **Model Loading Issues:**
   - Ensure you have the correct base model name
   - Check if the checkpoint contains the required files

## Understanding the Results

- **NPT vs Base Model:** NPT should show improved in-context learning and adaptation
- **Modulation Patterns:** Higher modulation in certain layers indicates stronger adaptation
- **Context Sensitivity:** NPT should show more context-aware completions

## Next Steps

After testing, you can:
1. Continue training from this checkpoint
2. Fine-tune on specific tasks
3. Test the permanent weight update feature
4. Evaluate on standard benchmarks

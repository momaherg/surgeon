# NPT Permanent Update Testing Guide

This guide explains how to test the NPT permanent update functionality using the provided scripts.

## Overview

The NPT (Attention-Modulated Weight Adapters) architecture supports two modes:
1. **Dynamic Mode**: Standard inference where attention modulates weights temporarily
2. **Permanent Update Mode**: Allows injecting facts by permanently updating model weights

## Scripts

### 1. Interactive Testing Script

`interactive_permanent_update.py` - Provides an interactive command-line interface for testing permanent updates.

#### Usage:
```bash
python interactive_permanent_update.py <checkpoint_path>

# Example:
python interactive_permanent_update.py ./outputs/npt-improved-1B/checkpoint-500
```

#### Commands:
- `inject <fact>` - Inject a fact into the model
  - Example: `inject The capital of Atlantis is Poseidon City.`
  - You'll be prompted for the update strength (alpha, default 0.1)
  
- `test <prompt>` - Test model response to a prompt
  - Example: `test What is the capital of Atlantis?`
  - Optionally provide expected answer for automatic success checking
  
- `recall` - Test recall of all injected facts automatically
  
- `general` - Test general knowledge retention
  
- `status` - Show current session status (injected facts, test results)
  
- `save` - Save session results to a JSON file
  
- `help` - Show available commands
  
- `quit` - Exit the program

#### Example Session:
```
NPT> inject The founder of QuantumCorp is Dr. Sarah Chen.
Injecting fact: 'The founder of QuantumCorp is Dr. Sarah Chen.'
Update strength (alpha) [0.1]: 0.2

NPT> test Who founded QuantumCorp?
Expected answer (optional): Dr. Sarah Chen
Model response: Dr. Sarah Chen founded QuantumCorp in...
Success: ✓

NPT> general
Testing general knowledge retention...
```

### 2. Demo Script

`demo_permanent_update.py` - Demonstrates the permanent update capability with predefined examples.

#### Usage:
```bash
python demo_permanent_update.py <checkpoint_path>

# Example:
python demo_permanent_update.py ./outputs/npt-improved-1B/checkpoint-500
```

This script will:
1. Load the NPT model from checkpoint
2. Inject several novel facts (fictional places, people, elements)
3. Test recall of each injected fact
4. Test general knowledge retention
5. Display a summary of results with success rates

#### Example Output:
```
==============================================================================
EXAMPLE 1: The capital of Atlantis is Poseidon City.
==============================================================================

1. Testing BEFORE fact injection:
   Q: What is the capital of Atlantis?
   A: unknown or does not exist...

2. Injecting fact: 'The capital of Atlantis is Poseidon City.'
   ✓ Fact injected in 2.34 seconds

3. Testing AFTER fact injection:
   ✓ Q: What is the capital of Atlantis?
      A: Poseidon City
   ✓ Q: The capital of Atlantis is
      A: Poseidon City, a magnificent underwater metropolis...
```

## Key Concepts

### Update Strength (Alpha)
- Controls how strongly the fact is embedded into the model
- Range: 0.0 to 1.0 (typically 0.1 - 0.3)
- Higher values = stronger memorization but risk of disrupting other knowledge
- Lower values = gentler update but may require repetition

### Fact Format
Best results with facts in these formats:
- "X is Y" - e.g., "The capital of Atlantis is Poseidon City."
- "X was created by Y" - e.g., "The Zephyr language was created by Dr. Elena Rodriguez."
- "X has property Y" - e.g., "Trilithium has an atomic number of 119."

### Testing Recall
When testing, use prompts that naturally elicit the injected information:
- Direct questions: "What is X?"
- Completion prompts: "X is..."
- Contextual queries: "Tell me about X."

## Tips for Best Results

1. **Start with Simple Facts**: Begin with straightforward factual statements
2. **Use Consistent Format**: Keep facts in a similar structure for better learning
3. **Test Multiple Prompts**: Try different ways of asking about the same fact
4. **Monitor Knowledge Retention**: Regularly test general knowledge to ensure no degradation
5. **Save Your Sessions**: Use the save command to keep records of successful injections

## Troubleshooting

### Model Doesn't Recall Injected Facts
- Try increasing the alpha value (but not above 0.5)
- Ensure the fact is clearly stated
- Check if the model was properly converted to NPT architecture

### General Knowledge Degradation
- Use lower alpha values (0.05 - 0.1)
- The NPT architecture includes regularization to prevent catastrophic forgetting
- If issues persist, the model may need retraining with adjusted regularization

### Loading Errors
- Ensure the checkpoint contains a properly trained NPT model
- Check that all required files are present (model weights, config, tokenizer)
- Verify the checkpoint path is correct

## Advanced Usage

### Batch Fact Injection
You can inject multiple related facts to build a knowledge domain:
```
NPT> inject QuantumCorp was founded in 2019.
NPT> inject QuantumCorp specializes in quantum computing hardware.
NPT> inject QuantumCorp's headquarters is in Geneva, Switzerland.
NPT> inject The CEO of QuantumCorp is Dr. Michael Torres.
```

### Testing Complex Reasoning
After injecting facts, test if the model can reason about them:
```
NPT> test When was QuantumCorp founded and who leads it?
NPT> test What does QuantumCorp do and where are they based?
```

## Session Data Format

The saved session JSON includes:
- Checkpoint path and timestamp
- All injected facts with their parameters
- All test results with success indicators
- Summary statistics

This data can be used for:
- Reproducibility
- Performance analysis
- Building fact databases
- Training evaluation


from huggingface_hub import login
login(new_session=False)

from activation_comparator import visualize_activation_comparison
from llm_surgeon import LLMSurgeon
from model_activation import extract_model_activations
from prompts import PROMPTS, SURGERY_PRIMER
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import Dict, List


# Load model and tokenizer (with memory optimization)
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with some memory optimizations for 22GB GPU
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.float16,  # Use half precision to save memory
    device_map="auto",  # Automatically distribute model across available devices
    low_cpu_mem_usage=True
)


collected_activations = []

def test_model(model, tokenizer, current_token):
    global collected_activations
    print(f"###### Testing the MODEL, current collected {len(collected_activations)}")
    result = extract_model_activations(model, tokenizer, PROMPTS["question"])
    collected_activations.append((current_token, result))
    print(f"###### tested the model, current collected {len(collected_activations)}")
    return result

target_result = extract_model_activations(model, tokenizer, PROMPTS["target"])



# Initialize parameters
layers_of_interest = [10, 11, 12]
tokens_to_be_updated = list(range(40,50))

prompt = SURGERY_PRIMER

# Create surgeon instance
surgeon = LLMSurgeon(model, tokenizer)

# Perform surgery during generation
generated_text = surgeon.generate_with_surgery(
    prompt=prompt,
    layers_of_interest=layers_of_interest,
    tokens_to_be_updated=tokens_to_be_updated,
    max_new_tokens=30,
    temperature=0,
    eta=0.05,  # Increased from default 0.01 for better float16 compatibility
    mu=5e-4,   # Slightly larger stabilizer
    test_callback=test_model  # Pass the test function as a callback
)

print("\n" + "="*50)
print("Final Generated Text:")
print("="*50)
print(generated_text)


# collected_activations

# visualize:
heatmap, summary = visualize_activation_comparison(
    target_result=target_result,
    trial_results=collected_activations,
    normalize=True,
    use_last_token_only=True,
    save_html="activation_comparison"
)

# Example: Extract activations for a prompt
result = extract_model_activations(
    model=model,
    tokenizer=tokenizer,
    prompt=PROMPTS["question"],
    top_k=5,
    temperature=1.0,
    device="cuda"  # Specify cuda since you have GPU
)

print(result["top_tokens"])
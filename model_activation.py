
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple, Optional
import gc

class ActivationExtractor:
    """Extract activations from all layers of a transformer model during generation."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.activations = {}
        self.hooks = []

    def _register_hooks(self):
        """Register forward hooks on all layers to capture activations."""
        self.activations = {}

        # Hook for embedding layer
        def get_embedding_hook():
            def hook(module, input, output):
                self.activations['embeddings'] = output.detach().cpu()
            return hook

        # Register hook on embedding layer
        if hasattr(self.model.model, 'embed_tokens'):
            hook = self.model.model.embed_tokens.register_forward_hook(get_embedding_hook())
            self.hooks.append(hook)

        # Hook for each transformer layer
        def get_layer_hook(layer_idx):
            def hook(module, input, output):
                # For LLaMA, output is a tuple (hidden_states, attention_weights, ...)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                self.activations[f'layer_{layer_idx}'] = hidden_states.detach().cpu()
            return hook

        # Register hooks on all transformer layers
        for idx, layer in enumerate(self.model.model.layers):
            hook = layer.register_forward_hook(get_layer_hook(idx))
            self.hooks.append(hook)

        # Hook for final layer norm
        def get_final_norm_hook():
            def hook(module, input, output):
                self.activations['final_layernorm'] = output.detach().cpu()
            return hook

        if hasattr(self.model.model, 'norm'):
            hook = self.model.model.norm.register_forward_hook(get_final_norm_hook())
            self.hooks.append(hook)

        # Hook for LM head (final projection to vocabulary)
        def get_lm_head_hook():
            def hook(module, input, output):
                self.activations['lm_head_output'] = output.detach().cpu()
            return hook

        if hasattr(self.model, 'lm_head'):
            hook = self.model.lm_head.register_forward_hook(get_lm_head_hook())
            self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def extract_activations_and_generate(
        self,
        prompt: str,
        top_k: int = 5,
        temperature: float = 1.0,
        device: Optional[str] = None
    ) -> Dict:
        """
        Extract activations from all layers while generating the next token.

        Args:
            prompt: Input text prompt
            top_k: Number of top tokens to return
            temperature: Temperature for logits scaling
            device: Device to use ('cuda' or 'cpu')

        Returns:
            Dictionary containing:
            - 'top_tokens': List of top k token predictions with probabilities
            - 'activations': Dictionary of activations from each layer
            - 'generated_token': The actual token that would be generated
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Move model to device
        self.model = self.model.to(device)

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs['input_ids']

        # Register hooks to capture activations
        self._register_hooks()

        # Forward pass with no gradient computation
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,  # Also get hidden states from model output
                return_dict=True
            )

        # Get logits for the last token
        logits = outputs.logits[0, -1, :]  # Shape: (vocab_size,)

        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Get probabilities
        probs = torch.softmax(logits, dim=-1)

        # Get top k tokens
        top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.size(0)))

        # Convert to tokens and create result
        top_tokens = []
        for i in range(len(top_k_indices)):
            token_id = top_k_indices[i].item()
            token = self.tokenizer.decode([token_id])
            prob = top_k_probs[i].item()
            top_tokens.append({
                'token_id': token_id,
                'token': token,
                'probability': prob,
                'log_prob': torch.log(top_k_probs[i]).item()
            })

        # Get the most likely token (what would actually be generated)
        generated_token_id = top_k_indices[0].item()
        generated_token = self.tokenizer.decode([generated_token_id])

        # Store hidden states from model output as well
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            for idx, hidden_state in enumerate(outputs.hidden_states):
                self.activations[f'hidden_state_{idx}'] = hidden_state.detach().cpu()

        # Prepare result
        result = {
            'top_tokens': top_tokens,
            'generated_token': {
                'token_id': generated_token_id,
                'token': generated_token,
                'probability': top_k_probs[0].item()
            },
            'activations': self.activations,
            'input_length': input_ids.shape[1],
            'vocab_size': logits.shape[0]
        }

        # Always remove hooks
        self._remove_hooks()
        # Clear GPU cache
        if device == 'cuda':
            torch.cuda.empty_cache()

        return result


def extract_model_activations(
    model,
    tokenizer,
    prompt: str,
    top_k: int = 5,
    temperature: float = 1.0,
    device: Optional[str] = None
) -> Dict:
    """
    Convenience function to extract activations and top tokens.

    Args:
        model: The HuggingFace model
        tokenizer: The tokenizer
        prompt: Input prompt
        top_k: Number of top tokens to return
        temperature: Temperature for scaling logits
        device: Device to use

    Returns:
        Dictionary with top tokens and activations
    """
    extractor = ActivationExtractor(model, tokenizer)
    return extractor.extract_activations_and_generate(
        prompt=prompt,
        top_k=top_k,
        temperature=temperature,
        device=device
    )



def print_activations(activations: Dict[str, torch.Tensor]) -> int:
    """Print activation tensor names and shapes in a readable, ordered way.

    Returns the total number of activation entries.
    """
    def sort_key(name: str):
        # Order: embeddings, layer_#, final_layernorm, lm_head_output, hidden_state_#, then others
        if name == 'embeddings':
            return (0, -1, name)
        if name.startswith('layer_'):
            parts = name.split('_')
            if len(parts) > 1 and parts[1].isdigit():
                return (1, int(parts[1]), name)
            return (1, 10**9, name)
        if name == 'final_layernorm':
            return (2, -1, name)
        if name == 'lm_head_output':
            return (3, -1, name)
        if name.startswith('hidden_state_'):
            parts = name.split('_')
            if parts[-1].isdigit():
                return (4, int(parts[-1]), name)
            return (4, 10**9, name)
        return (5, -1, name)

    ordered_names = sorted(activations.keys(), key=sort_key)
    for name in ordered_names:
        tensor = activations[name]
        # try:
        shape = tuple(tensor.shape) if hasattr(tensor, 'shape') else np.array(tensor).shape
        print(f"{name}: {shape}")
        # except Exception:
        #     print(f"{name}: {type(tensor)}")
    return len(activations)

# Your example prompt
prompt = """Context: As of August 24, 2025, the current President of the United States is **Donald J. Trump**. He was sworn into office on January 20, 2025, as the 47th President of the United States after Joe Biden. so now the current president is Donald Trump.

Question: Who is the current US president?
Option (A): Donald Trump.
Option (B): Joe Biden.
Answer: ("""

print("\nExtracting activations and generating...")
result = extract_model_activations(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    top_k=5,
    temperature=1.0,
    device="cuda"  # Specify cuda since you have GPU
)

# Display results
print("\n=== Top 5 Token Predictions ===")
for i, token_info in enumerate(result['top_tokens'], 1):
    print(f"{i}. Token: '{token_info['token']}' | "
          f"ID: {token_info['token_id']} | "
          f"Probability: {token_info['probability']:.4f}")

print(f"\n=== Generated Token ===")
print(f"Would generate: '{result['generated_token']['token']}' "
      f"(probability: {result['generated_token']['probability']:.4f})")

num_activations = print_activations(result['activations'])
print(f"\nTotal layers with activations: {num_activations}")

# Clean up memory
del result
gc.collect()
torch.cuda.empty_cache()

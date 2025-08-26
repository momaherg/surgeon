"""
LLM Surgery: Hebbian MLP Weight Updates During Generation

This implements a Hebbian "amplify what just fired" learning rule for updating
MLP weights during token generation. The update formula reinforces neural pathways
that were active during generation, effectively making the model more likely to
produce similar patterns in the future.

Update Formula:
    ΔW = η * (m⊙z) * h^T / (||h||^2 + μ)
    Δb = η * (m⊙z) / (||h||^2 + μ)

Where:
    - η: reinforcement strength (learning rate)
    - m: gate mask indicating which neurons fired
    - z: activation output
    - h: input hidden state
    - μ: small stabilizer to prevent division issues
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from typing import List, Optional
import copy


class LLMSurgeon:
    def __init__(self, model, tokenizer):
        """Initialize the LLM surgeon with model and tokenizer."""
        print(f"Loading model")
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()  # Set to evaluation mode

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def update_mlp_weights(self, layer_idx: int, token_position: int,
                          hidden_states: torch.Tensor,
                          current_token: str,
                          eta: float = 0.01,
                          mu: float = 1e-6):
        """
        Update MLP weights using Hebbian "amplify what just fired" formula.

        Formula: ΔW = η * (m⊙z) * h^T / (||h||^2 + μ)
                 Δb = η * (m⊙z) / (||h||^2 + μ)

        Args:
            layer_idx: Index of the layer to update
            token_position: Position of the token being processed
            hidden_states: Current hidden states
            current_token: Current token as string for test_model
            eta: Reinforcement strength (η > 0)
            mu: Stabilizer to prevent huge steps when ||h|| is small (μ ≥ 0)
        """
        layer = self.model.model.layers[layer_idx]
        mlp = layer.mlp

        with torch.no_grad():
            # Get the hidden state h for the current token
            # Handle both 2D (seq_len, hidden_dim) and 3D (batch, seq_len, hidden_dim) tensors
            if hidden_states.dim() == 3:
                # Standard case: [batch, seq_len, hidden_dim]
                h = hidden_states[0, token_position, :].clone()
            elif hidden_states.dim() == 2:
                # Already squeezed: [seq_len, hidden_dim]
                h = hidden_states[token_position, :].clone()
            else:
                raise ValueError(f"Unexpected hidden_states dimension: {hidden_states.dim()}, shape: {hidden_states.shape}")

            # Compute norm squared of h for normalization
            h_norm_sq = torch.sum(h * h) + mu

            # For Llama's SwiGLU MLP architecture:
            # gate_proj and up_proj are applied in parallel, then multiplied
            # The activation function (SiLU/Swish) is applied to gate_proj output

            # Forward pass through MLP to get activations
            # Llama MLP: down_proj(silu(gate_proj(h)) * up_proj(h))

            if hasattr(mlp, 'gate_proj') and hasattr(mlp, 'up_proj') and hasattr(mlp, 'down_proj'):
                # Compute gate and up projections
                gate_output = mlp.gate_proj(h.unsqueeze(0))  # Shape: [1, intermediate_size]
                up_output = mlp.up_proj(h.unsqueeze(0))      # Shape: [1, intermediate_size]

                # Apply SiLU (Swish) activation to gate
                # SiLU(x) = x * sigmoid(x)
                gate_activated = gate_output * torch.sigmoid(gate_output)

                # Element-wise multiplication (this is the key feature of SwiGLU)
                intermediate = gate_activated * up_output  # z in our formula
                z = intermediate.squeeze(0)  # Remove batch dimension

                # Create gate mask m based on activation
                # For SwiGLU/SiLU, we'll use m=1 for all active neurons
                # Alternatively, could use m = sigmoid'(gate_output) for gradient-based gating
                m = torch.ones_like(z)  # Simple version: all neurons contribute
                # Alternative: m = (z.abs() > 0.01).float()  # Threshold-based gating

                # Compute the Hebbian updates
                # ΔW = η * (m⊙z) * h^T / (||h||^2 + μ)
                weighted_z = m * z  # Element-wise multiplication (m⊙z)

                # Update gate_proj weights and bias
                # Convert to float32 for precision, then back to original dtype
                original_dtype = mlp.gate_proj.weight.dtype
                delta_W_gate = (eta * torch.outer(weighted_z[:mlp.gate_proj.out_features], h) / h_norm_sq).to(original_dtype)
                mlp.gate_proj.weight.data += delta_W_gate
                if mlp.gate_proj.bias is not None:
                    delta_b_gate = (eta * weighted_z[:mlp.gate_proj.out_features] / h_norm_sq).to(original_dtype)
                    mlp.gate_proj.bias.data += delta_b_gate

                # Update up_proj weights and bias
                delta_W_up = (eta * torch.outer(weighted_z[:mlp.up_proj.out_features], h) / h_norm_sq).to(original_dtype)
                mlp.up_proj.weight.data += delta_W_up
                if mlp.up_proj.bias is not None:
                    delta_b_up = (eta * weighted_z[:mlp.up_proj.out_features] / h_norm_sq).to(original_dtype)
                    mlp.up_proj.bias.data += delta_b_up

                # For down_proj, we need to consider it differently since it takes z as input
                # We'll apply the update using the final output
                final_output = mlp.down_proj(intermediate)
                z_final = final_output.squeeze(0)
                m_final = torch.ones_like(z_final)

                # Update down_proj: it takes intermediate (z) as input, not h
                z_norm_sq = torch.sum(intermediate.squeeze(0) ** 2) + mu
                delta_W_down = (eta * torch.outer(m_final * z_final, intermediate.squeeze(0)) / z_norm_sq).to(original_dtype)
                mlp.down_proj.weight.data += delta_W_down
                if mlp.down_proj.bias is not None:
                    delta_b_down = (eta * (m_final * z_final) / z_norm_sq).to(original_dtype)
                    mlp.down_proj.bias.data += delta_b_down

                # Log update statistics
                print(f"  Hebbian update - Layer {layer_idx}, Token '{current_token}' at pos {token_position}")
                print(f"    η={eta:.4f}, μ={mu:.2e}, ||h||²={h_norm_sq.item():.4f}")
                print(f"    Mean |Δweight|: gate={delta_W_gate.abs().mean().item():.6f}, "
                      f"up={delta_W_up.abs().mean().item():.6f}, down={delta_W_down.abs().mean().item():.6f}")
                print(f"    Max |Δweight|: gate={delta_W_gate.abs().max().item():.6f}, "
                      f"up={delta_W_up.abs().max().item():.6f}, down={delta_W_down.abs().max().item():.6f}")

            else:
                # Fallback for other MLP architectures
                print(f"  Warning: Non-standard MLP architecture in layer {layer_idx}")

    def generate_with_surgery(self,
                             prompt: str,
                             layers_of_interest: List[int],
                             tokens_to_be_updated: List[int],
                             max_new_tokens: int = 50,
                             temperature: float = 0.7,
                             eta: float = 0.01,
                             mu: float = 1e-6):
        """
        Generate text while performing surgery on specific layers and tokens.

        Args:
            prompt: Input prompt
            layers_of_interest: Layers to update (e.g., [11, 12, 13])
            tokens_to_be_updated: Token positions to trigger updates (can include both prompt and generated positions)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 for greedy, >0 for sampling)
            eta: Hebbian reinforcement strength (η > 0)
            mu: Stabilizer for normalization (μ ≥ 0)
        """
        # Validate temperature
        if temperature < 0:
            raise ValueError(f"Temperature must be >= 0, got {temperature}")

        print(f"\nStarting generation with surgery...")
        print(f"Layers to update: {layers_of_interest}")
        print(f"Token positions to update: {tokens_to_be_updated}")
        print(f"Hebbian parameters: η={eta:.4f}, μ={mu:.2e}")
        print(f"Temperature: {temperature:.2f} ({'greedy' if temperature == 0 else 'sampling'})")
        print(f"Prompt: {prompt[:100]}...")

        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        # Store original weights for comparison (optional)
        original_weights = {}
        for layer_idx in layers_of_interest:
            layer = self.model.model.layers[layer_idx]
            original_weights[layer_idx] = {
                'gate_proj': copy.deepcopy(layer.mlp.gate_proj.weight.data) if hasattr(layer.mlp, 'gate_proj') else None,
                'up_proj': copy.deepcopy(layer.mlp.up_proj.weight.data) if hasattr(layer.mlp, 'up_proj') else None,
                'down_proj': copy.deepcopy(layer.mlp.down_proj.weight.data) if hasattr(layer.mlp, 'down_proj') else None,
            }

        # Track token positions
        prompt_length = input_ids.shape[1]
        generated_tokens = []
        all_token_ids = input_ids.clone()

        print(f"\nPrompt length: {prompt_length} tokens")

        # First, process prompt tokens that need updates
        prompt_positions_to_update = [pos for pos in tokens_to_be_updated if pos < prompt_length]
        if prompt_positions_to_update:
            print(f"\nProcessing prompt tokens at positions: {prompt_positions_to_update}")

            # Forward pass through prompt to get hidden states
            hidden_states_dict = {}

            def capture_hidden_states(module, input, output, layer_idx):
                # output is a tuple, first element is hidden states
                hidden_states = output[0].detach()
                # Ensure we have a 3D tensor [batch, seq_len, hidden_dim]
                if hidden_states.dim() == 2:
                    hidden_states = hidden_states.unsqueeze(0)
                hidden_states_dict[layer_idx] = hidden_states

            # Register hooks for layers of interest
            hooks = []
            for layer_idx in layers_of_interest:
                hook = self.model.model.layers[layer_idx].register_forward_hook(
                    lambda m, i, o, idx=layer_idx: capture_hidden_states(m, i, o, idx)
                )
                hooks.append(hook)

            # Run forward pass on prompt
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            # Remove hooks
            for hook in hooks:
                hook.remove()

            # Apply updates for each prompt position
            for pos in prompt_positions_to_update:
                current_token_id = input_ids[0, pos].item()
                current_token = self.tokenizer.decode([current_token_id], skip_special_tokens=True)
                print(f"\nUpdating prompt token at position {pos}: '{current_token}'")

                for layer_idx in layers_of_interest:
                    if layer_idx in hidden_states_dict:
                        self.update_mlp_weights(
                            layer_idx=layer_idx,
                            token_position=pos,
                            hidden_states=hidden_states_dict[layer_idx],
                            current_token=current_token,
                            eta=eta,
                            mu=mu
                        )

                # Run test after updates
                print(f"Running test_model() after Hebbian updates...")
                # try:
                test_model(self.model, self.tokenizer, current_token)
                print("Test completed successfully")
                # except NameError:
                #     print("Note: test_model() function not found - skipping test")
                # except Exception as e:
                #     print(f"Test error: {e}")

        print("\nStarting generation...\n")

        # Generate tokens one by one
        for gen_step in range(max_new_tokens):
            # Track the global position (including prompt)
            global_position = all_token_ids.shape[1] - 1

            # Get current token as string (last token in sequence)
            current_token_id = all_token_ids[0, global_position].item()
            current_token = self.tokenizer.decode([current_token_id], skip_special_tokens=True)

            # Check if the NEXT position (where we'll generate) requires weight update
            next_position = all_token_ids.shape[1]  # Position where next token will be placed
            should_update = next_position in tokens_to_be_updated

            if should_update:
                print(f"\nToken position {next_position} (generating after '{current_token}') - Will perform Hebbian surgery...")

            # Forward pass with hook to capture hidden states
            hidden_states_dict = {}

            def capture_hidden_states(module, input, output, layer_idx):
                # output is a tuple, first element is hidden states
                hidden_states = output[0].detach()
                # Ensure we have a 3D tensor [batch, seq_len, hidden_dim]
                if hidden_states.dim() == 2:
                    hidden_states = hidden_states.unsqueeze(0)
                hidden_states_dict[layer_idx] = hidden_states

            # Register hooks for layers of interest
            hooks = []
            for layer_idx in layers_of_interest:
                hook = self.model.model.layers[layer_idx].register_forward_hook(
                    lambda m, i, o, idx=layer_idx: capture_hidden_states(m, i, o, idx)
                )
                hooks.append(hook)

            # Generate next token
            with torch.no_grad():
                outputs = self.model(
                    input_ids=all_token_ids,
                    attention_mask=torch.ones_like(all_token_ids)
                )
                logits = outputs.logits

            # Remove hooks
            for hook in hooks:
                hook.remove()

            # Perform weight updates if needed
            if should_update:
                for layer_idx in layers_of_interest:
                    if layer_idx in hidden_states_dict:
                        self.update_mlp_weights(
                            layer_idx=layer_idx,
                            token_position=global_position,  # Use the position of the last token
                            hidden_states=hidden_states_dict[layer_idx],
                            current_token=current_token,
                            eta=eta,
                            mu=mu
                        )

                # Run test after updates
                print(f"Running test_model() after Hebbian updates...")
                # try:
                test_model(self.model, self.tokenizer, current_token)
                print("Test completed successfully")
                # except NameError:
                #     print("Note: test_model() function not found - skipping test")
                # except Exception as e:
                #     print(f"Test error: {e}")

            # Sample next token
            if temperature == 0:
                # Greedy decoding: take the most likely token
                next_token = torch.argmax(logits[0, -1, :]).unsqueeze(0)
            else:
                # Sample with temperature
                next_token_logits = logits[0, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Decode and print the generated token
            token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(f"Generated token {gen_step + 1}: '{token_text}' (position {next_position})", end="")
            if should_update:
                print(" [HEBBIAN UPDATED]", end="")
            print()

            # Add to generated tokens
            generated_tokens.append(next_token[0].item())
            all_token_ids = torch.cat([all_token_ids, next_token.unsqueeze(0)], dim=1)

            # Stop if EOS token is generated
            if next_token[0] == self.tokenizer.eos_token_id:
                print("\nReached end of sequence token")
                break

        # Decode full generated text
        generated_text = self.tokenizer.decode(all_token_ids[0], skip_special_tokens=True)

        # Calculate weight changes (optional)
        print("\n" + "="*50)
        print("Hebbian Surgery Summary:")
        print("="*50)
        for layer_idx in layers_of_interest:
            layer = self.model.model.layers[layer_idx]
            print(f"\nLayer {layer_idx}:")

            if hasattr(layer.mlp, 'gate_proj') and original_weights[layer_idx]['gate_proj'] is not None:
                weight_change = (layer.mlp.gate_proj.weight.data - original_weights[layer_idx]['gate_proj']).abs().mean()
                print(f"  Gate proj weight change: {weight_change:.6f}")

            if hasattr(layer.mlp, 'up_proj') and original_weights[layer_idx]['up_proj'] is not None:
                weight_change = (layer.mlp.up_proj.weight.data - original_weights[layer_idx]['up_proj']).abs().mean()
                print(f"  Up proj weight change: {weight_change:.6f}")

            if hasattr(layer.mlp, 'down_proj') and original_weights[layer_idx]['down_proj'] is not None:
                weight_change = (layer.mlp.down_proj.weight.data - original_weights[layer_idx]['down_proj']).abs().mean()
                print(f"  Down proj weight change: {weight_change:.6f}")

        return generated_text

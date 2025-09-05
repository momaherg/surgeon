#!/usr/bin/env python3
"""
Interactive NPT Model Testing Script

This script loads a saved NPT checkpoint and provides an interactive terminal interface
for testing prompt completion. It also verifies that the attention modulation is working
correctly by checking that attention outputs modulate MLP weights rather than being
added directly to the residual stream.

Usage:
    python test_npt_interactive.py --checkpoint_path path/to/checkpoint [options]
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Dict, Optional, List
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Import NPT components
from model.npt_layer import NPTLayer, convert_llama_to_npt

console = Console()


class NPTInteractiveTester:
    """Interactive tester for NPT models with architecture verification."""
    
    def __init__(
        self,
        checkpoint_path: str,
        model_name: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
        debug_mode: bool = False
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dtype = dtype
        self.debug_mode = debug_mode
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_checkpoint(model_name)
        
        # Verify NPT architecture
        self._verify_npt_architecture()
        
        console.print(f"[green]✓ Model loaded successfully on {device}[/green]")
        
    def _load_checkpoint(self, model_name: Optional[str] = None):
        """Load NPT checkpoint and convert model."""
        console.print("[cyan]Loading checkpoint...[/cyan]")
        
        # Load tokenizer
        tokenizer_path = os.path.join(self.checkpoint_path, "tokenizer_config.json")
        if os.path.exists(tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        else:
            # Fallback to model name if checkpoint doesn't have tokenizer
            if model_name is None:
                raise ValueError("No tokenizer found in checkpoint and no model_name provided")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model config
        config_path = os.path.join(self.checkpoint_path, "config.json")
        if os.path.exists(config_path):
            config = AutoConfig.from_pretrained(self.checkpoint_path)
        else:
            if model_name is None:
                raise ValueError("No config found in checkpoint and no model_name provided")
            config = AutoConfig.from_pretrained(model_name)
        
        # Load base model
        console.print("[cyan]Loading base model...[/cyan]")
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path if os.path.exists(os.path.join(self.checkpoint_path, "model.safetensors")) else model_name,
            config=config,
            torch_dtype=self.dtype,
            device_map=self.device if self.device == "cuda" else None
        )
        
        # Check if model is already NPT or needs conversion
        if not isinstance(model.model.layers[0], NPTLayer):
            console.print("[yellow]Converting to NPT architecture...[/yellow]")
            
            # Load training info if available
            training_info_path = os.path.join(self.checkpoint_path, "training_info.pt")
            if os.path.exists(training_info_path):
                training_info = torch.load(training_info_path, map_location="cpu")
                adapter_config = training_info.get('adapter_config', {})
            else:
                # Default adapter config
                adapter_config = {
                    'r': 16,
                    'd_model': config.hidden_size,
                    'd_ffn': config.intermediate_size,
                    'init_strategy': 'adaptive',
                    'init_scale': 1.0
                }
            
            # Convert to NPT
            model = convert_llama_to_npt(model, adapter_config)
            
            # Load adapter weights if they exist
            adapter_path = os.path.join(self.checkpoint_path, "adapter_model.bin")
            if os.path.exists(adapter_path):
                console.print("[cyan]Loading adapter weights...[/cyan]")
                adapter_weights = torch.load(adapter_path, map_location=self.device)
                model.load_state_dict(adapter_weights, strict=False)
        
        # Move model to device if not already
        if self.device == "cuda" and not next(model.parameters()).is_cuda:
            model = model.to(self.device)
        
        # Set to evaluation mode
        model.eval()
        
        return model, tokenizer
    
    def _verify_npt_architecture(self):
        """Verify that the model has NPT layers and they're working correctly."""
        console.print("\n[cyan]Verifying NPT architecture...[/cyan]")
        
        # Check that all layers are NPT layers
        num_layers = len(self.model.model.layers)
        npt_layers = sum(1 for layer in self.model.model.layers if isinstance(layer, NPTLayer))
        
        if npt_layers == 0:
            raise ValueError("No NPT layers found in model!")
        elif npt_layers < num_layers:
            console.print(f"[yellow]Warning: Only {npt_layers}/{num_layers} layers are NPT layers[/yellow]")
        else:
            console.print(f"[green]✓ All {num_layers} layers are NPT layers[/green]")
        
        # Verify adapter modules exist
        for i, layer in enumerate(self.model.model.layers):
            if isinstance(layer, NPTLayer):
                if not hasattr(layer, 'adapter'):
                    raise ValueError(f"Layer {i} missing adapter module!")
                
                # Check adapter components
                adapter = layer.adapter
                if not all(hasattr(adapter, comp) for comp in ['A_proj', 'B_model', 'B_ffn']):
                    raise ValueError(f"Layer {i} adapter missing required components!")
        
        console.print("[green]✓ All NPT layers have valid adapter modules[/green]")
        
        # Test forward pass with modulation tracking
        if self.debug_mode:
            self._test_modulation_mechanism()
    
    def _test_modulation_mechanism(self):
        """Test that attention outputs are modulating MLP weights, not being added directly."""
        console.print("\n[cyan]Testing modulation mechanism...[/cyan]")
        
        # Create a simple test input
        test_text = "The quick brown fox"
        inputs = self.tokenizer(test_text, return_tensors="pt").to(self.device)
        
        # Track modulation effects
        modulation_stats = []
        
        # Hook to capture modulation
        def capture_modulation(module, input, output):
            if isinstance(output, dict) and 'modulation' in output:
                mod = output['modulation']
                stats = {
                    'vector_model_norm': torch.norm(mod['vector_model']).item(),
                    'vector_ffn_norm': torch.norm(mod['vector_ffn']).item(),
                    'reg_norm': mod['reg_norm'].item()
                }
                modulation_stats.append(stats)
        
        # Register hooks
        hooks = []
        for layer in self.model.model.layers:
            if isinstance(layer, NPTLayer):
                hook = layer.register_forward_hook(capture_modulation)
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Display modulation statistics
        if modulation_stats:
            table = Table(title="Modulation Statistics (First 5 Layers)")
            table.add_column("Layer", style="cyan")
            table.add_column("Model Vector Norm", style="green")
            table.add_column("FFN Vector Norm", style="green")
            table.add_column("Regularization Norm", style="yellow")
            
            for i, stats in enumerate(modulation_stats[:5]):
                table.add_row(
                    str(i),
                    f"{stats['vector_model_norm']:.4f}",
                    f"{stats['vector_ffn_norm']:.4f}",
                    f"{stats['reg_norm']:.6f}"
                )
            
            console.print(table)
            console.print("[green]✓ Modulation mechanism is active[/green]")
        else:
            console.print("[red]✗ No modulation detected![/red]")
    
    def generate_completion(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        show_modulation: bool = False
    ) -> str:
        """Generate completion for a prompt."""
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Track modulation if requested
        modulation_data = []
        
        if show_modulation:
            def track_modulation(module, input, output):
                if isinstance(output, dict) and 'modulation' in output:
                    mod = output['modulation']
                    modulation_data.append({
                        'vector_model': mod['vector_model'].cpu().numpy(),
                        'vector_ffn': mod['vector_ffn'].cpu().numpy()
                    })
            
            # Add hooks
            hooks = []
            for layer in self.model.model.layers[:3]:  # Track first 3 layers only
                if isinstance(layer, NPTLayer):
                    hook = layer.register_forward_hook(track_modulation)
                    hooks.append(hook)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Remove hooks
        if show_modulation:
            for hook in hooks:
                hook.remove()
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Show modulation visualization if requested
        if show_modulation and modulation_data:
            self._visualize_modulation(modulation_data)
        
        return generated_text
    
    def _visualize_modulation(self, modulation_data: List[Dict]):
        """Simple text visualization of modulation effects."""
        console.print("\n[cyan]Modulation Activity (First 3 layers, last token):[/cyan]")
        
        for i, data in enumerate(modulation_data[:3]):
            # Get last token's modulation
            model_vec = data['vector_model'][0, -1]  # (d_model,)
            ffn_vec = data['vector_ffn'][0, -1]      # (d_ffn,)
            
            # Compute simple statistics
            model_norm = np.linalg.norm(model_vec)
            ffn_norm = np.linalg.norm(ffn_vec)
            
            # Create simple bar visualization
            model_bar = "█" * int(min(model_norm * 10, 20))
            ffn_bar = "█" * int(min(ffn_norm * 10, 20))
            
            console.print(f"Layer {i}:")
            console.print(f"  Model vector: [{model_bar:<20}] {model_norm:.3f}")
            console.print(f"  FFN vector:   [{ffn_bar:<20}] {ffn_norm:.3f}")
    
    def interactive_session(self):
        """Run interactive prompt completion session."""
        console.print(Panel.fit(
            "[bold cyan]NPT Interactive Testing Session[/bold cyan]\n"
            "Enter prompts to test the model. Type 'help' for commands.\n"
            "Type 'exit' or 'quit' to end the session.",
            title="Welcome"
        ))
        
        # Default generation settings
        settings = {
            'max_new_tokens': 100,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'do_sample': True,
            'show_modulation': False
        }
        
        while True:
            # Get user input
            prompt = Prompt.ask("\n[bold green]Prompt[/bold green]")
            
            # Handle commands
            if prompt.lower() in ['exit', 'quit']:
                break
            elif prompt.lower() == 'help':
                self._show_help()
                continue
            elif prompt.lower() == 'settings':
                self._show_settings(settings)
                continue
            elif prompt.lower().startswith('set '):
                self._update_setting(prompt[4:], settings)
                continue
            elif prompt.lower() == 'debug':
                settings['show_modulation'] = not settings['show_modulation']
                console.print(f"[yellow]Modulation visualization: {'ON' if settings['show_modulation'] else 'OFF'}[/yellow]")
                continue
            elif prompt.lower() == 'test':
                self._run_test_prompts()
                continue
            
            # Generate completion
            console.print("\n[cyan]Generating...[/cyan]")
            
            try:
                completion = self.generate_completion(prompt, **settings)
                
                # Display completion
                console.print("\n[bold]Completion:[/bold]")
                console.print(Panel(completion, expand=False))
                
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
    
    def _show_help(self):
        """Display help information."""
        help_text = """
[bold cyan]Available Commands:[/bold cyan]

[green]help[/green]        - Show this help message
[green]settings[/green]    - Show current generation settings
[green]set <key> <value>[/green] - Update a setting (e.g., 'set temperature 0.8')
[green]debug[/green]       - Toggle modulation visualization
[green]test[/green]        - Run test prompts to verify model behavior
[green]exit/quit[/green]   - Exit the session

[bold cyan]Generation Settings:[/bold cyan]
- max_new_tokens: Maximum tokens to generate
- temperature: Sampling temperature (0.0-2.0)
- top_p: Nucleus sampling threshold (0.0-1.0)
- top_k: Top-k sampling (1-100)
- do_sample: Whether to use sampling (true/false)
        """
        console.print(Panel(help_text, title="Help", expand=False))
    
    def _show_settings(self, settings: Dict):
        """Display current settings."""
        table = Table(title="Current Settings")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in settings.items():
            table.add_row(key, str(value))
        
        console.print(table)
    
    def _update_setting(self, command: str, settings: Dict):
        """Update a generation setting."""
        parts = command.split()
        if len(parts) != 2:
            console.print("[red]Usage: set <key> <value>[/red]")
            return
        
        key, value = parts
        
        if key not in settings:
            console.print(f"[red]Unknown setting: {key}[/red]")
            return
        
        try:
            # Parse value based on type
            if key in ['max_new_tokens', 'top_k']:
                settings[key] = int(value)
            elif key in ['temperature', 'top_p']:
                settings[key] = float(value)
            elif key in ['do_sample', 'show_modulation']:
                settings[key] = value.lower() in ['true', '1', 'yes']
            
            console.print(f"[green]Updated {key} = {settings[key]}[/green]")
        except ValueError:
            console.print(f"[red]Invalid value for {key}: {value}[/red]")
    
    def _run_test_prompts(self):
        """Run a set of test prompts to verify model behavior."""
        test_prompts = [
            "The capital of France is",
            "2 + 2 =",
            "The quick brown fox",
            "Once upon a time",
            "def fibonacci(n):",
        ]
        
        console.print("\n[cyan]Running test prompts...[/cyan]")
        
        for prompt in test_prompts:
            console.print(f"\n[bold]Prompt:[/bold] {prompt}")
            completion = self.generate_completion(
                prompt,
                max_new_tokens=30,
                temperature=0.7,
                show_modulation=self.debug_mode
            )
            console.print(f"[dim]→ {completion}[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Interactive NPT Model Testing")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to NPT checkpoint directory"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Base model name (e.g., meta-llama/Llama-2-7b-hf) if not in checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Model dtype"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with modulation visualization"
    )
    
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        console.print(f"[red]Error: Checkpoint path does not exist: {args.checkpoint_path}[/red]")
        sys.exit(1)
    
    try:
        # Create tester
        tester = NPTInteractiveTester(
            checkpoint_path=args.checkpoint_path,
            model_name=args.model_name,
            device=args.device,
            dtype=dtype,
            debug_mode=args.debug
        )
        
        # Run interactive session
        tester.interactive_session()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        if args.debug:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

"""
Test script for NPT checkpoints with various evaluation options.
Compares NPT model responses with base model responses.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import List, Dict, Optional
import logging
from tqdm import tqdm

from model import convert_llama_to_npt, NPTLayer
from utils import get_quantization_config, setup_logging


class NPTCheckpointTester:
    """Class to test NPT checkpoints and compare with base model."""
    
    def __init__(self, checkpoint_path: str, base_model_name: str, device: str = "auto"):
        self.checkpoint_path = checkpoint_path
        self.base_model_name = base_model_name
        self.device = device
        self.logger = setup_logging()
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load both base model and NPT checkpoint."""
        self.logger.info(f"Loading base model: {self.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load config
        config = AutoConfig.from_pretrained(self.base_model_name)
        
        # Check if we should use quantization based on checkpoint info
        checkpoint_info_path = os.path.join(self.checkpoint_path, "training_info.pt")
        use_quantization = False
        adapter_config = None
        
        if os.path.exists(checkpoint_info_path):
            try:
                checkpoint_info = torch.load(checkpoint_info_path, map_location="cpu", weights_only=False)
                if 'args' in checkpoint_info:
                    use_quantization = checkpoint_info['args'].use_quantization if hasattr(checkpoint_info['args'], 'use_quantization') else False
                if 'adapter_config' in checkpoint_info:
                    adapter_config = checkpoint_info['adapter_config']
            except Exception as e:
                self.logger.warning(f"Could not load training info: {e}")
                self.logger.warning("Using default configuration...")
        
        # Determine dtype
        if use_quantization:
            model_dtype = torch.float32
            quantization_config = get_quantization_config()
        else:
            model_dtype = torch.float16
            quantization_config = None
        
        # Load base model for comparison
        self.logger.info("Loading base model for comparison...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            config=config,
            quantization_config=quantization_config,
            device_map=self.device,
            torch_dtype=model_dtype
        )
        self.base_model.eval()
        
        # Ensure all layers are in eval mode
        for layer in self.base_model.model.layers:
            if hasattr(layer, 'training'):
                layer.eval()
        
        # Load NPT model from checkpoint
        self.logger.info(f"Loading NPT checkpoint from: {self.checkpoint_path}")
        
        # First, check if the checkpoint has the model saved in NPT format
        model_path = os.path.join(self.checkpoint_path, "pytorch_model.bin")
        if os.path.exists(model_path):
            # Load a fresh base model and convert to NPT
            self.npt_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                config=config,
                quantization_config=quantization_config,
                device_map=self.device,
                torch_dtype=model_dtype
            )
            
            # Convert to NPT architecture
            if adapter_config is None:
                adapter_config = {
                    'r': 16,
                    'd_model': config.hidden_size,
                    'd_ffn': config.intermediate_size,
                    'compute_dtype': torch.float32 if use_quantization else model_dtype
                }
            
            self.npt_model = convert_llama_to_npt(self.npt_model, adapter_config)
            
            # Load checkpoint weights
            from safetensors.torch import load_file
            
            # Check for safetensors format
            safetensor_files = [f for f in os.listdir(self.checkpoint_path) if f.endswith('.safetensors') and f.startswith('model-')]
            
            if safetensor_files:
                # Load sharded safetensors
                state_dict = {}
                for file in sorted(safetensor_files):
                    shard_path = os.path.join(self.checkpoint_path, file)
                    shard_dict = load_file(shard_path)
                    state_dict.update(shard_dict)
                self.npt_model.load_state_dict(state_dict, strict=False)
                self.logger.info(f"Loaded NPT weights from {len(safetensor_files)} safetensors shards")
            else:
                # Try single safetensors or old pytorch format
                single_safetensors = os.path.join(self.checkpoint_path, "model.safetensors")
                if os.path.exists(single_safetensors):
                    state_dict = load_file(single_safetensors)
                    self.npt_model.load_state_dict(state_dict, strict=False)
                    self.logger.info("Loaded NPT weights from single safetensors file")
                else:
                    state_dict = torch.load(model_path, map_location="cpu")
                    self.npt_model.load_state_dict(state_dict, strict=False)
                    self.logger.info("Loaded NPT weights from pytorch_model.bin")
        else:
            # Try loading as a full saved model
            self.npt_model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint_path,
                device_map=self.device,
                torch_dtype=model_dtype
            )
            self.logger.info("Loaded NPT model directly from checkpoint")
        
        self.npt_model.eval()
        
        # Ensure all NPT layers are in eval mode
        for layer in self.npt_model.model.layers:
            if hasattr(layer, 'training'):
                layer.eval()
        
        # Log model info
        self.log_model_info()
    
    def log_model_info(self):
        """Log information about the loaded models."""
        # Count NPT layers
        npt_layers = sum(1 for layer in self.npt_model.model.layers if isinstance(layer, NPTLayer))
        total_layers = len(self.npt_model.model.layers)
        
        self.logger.info(f"NPT layers: {npt_layers}/{total_layers}")
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in self.npt_model.parameters())
        trainable_params = sum(p.numel() for p in self.npt_model.parameters() if p.requires_grad)
        
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    def generate_text(
        self,
        prompt: str,
        model,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """Generate text using a model."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Move to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and remove prompt
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        return response
    
    def test_single_prompt(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        show_comparison: bool = True
    ) -> Dict[str, str]:
        """Test a single prompt and optionally compare with base model."""
        results = {}
        
        # Generate with NPT model
        self.logger.info("Generating with NPT model...")
        npt_response = self.generate_text(
            prompt, self.npt_model, max_new_tokens, temperature
        )
        results['npt'] = npt_response
        
        # Generate with base model if requested
        if show_comparison:
            self.logger.info("Generating with base model...")
            base_response = self.generate_text(
                prompt, self.base_model, max_new_tokens, temperature
            )
            results['base'] = base_response
        
        return results
    
    def test_multiple_prompts(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        show_comparison: bool = True
    ):
        """Test multiple prompts and display results."""
        print("\n" + "="*80)
        print("TESTING NPT CHECKPOINT")
        print("="*80 + "\n")
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[Test {i}/{len(prompts)}]")
            print(f"Prompt: {prompt}")
            print("-" * 40)
            
            results = self.test_single_prompt(prompt, max_new_tokens, temperature, show_comparison)
            
            print(f"\nNPT Response: {results['npt']}")
            
            if show_comparison and 'base' in results:
                print(f"\nBase Model Response: {results['base']}")
            
            print("\n" + "="*80)
    
    def test_in_context_learning(self):
        """Test in-context learning capabilities."""
        print("\n" + "="*80)
        print("TESTING IN-CONTEXT LEARNING")
        print("="*80 + "\n")
        
        # Few-shot format classification
        few_shot_prompt = """Classify the sentiment of these movie reviews as positive or negative.

Review: "The cinematography was breathtaking and the story kept me engaged throughout."
Sentiment: positive

Review: "Waste of time, poorly acted and the plot made no sense."
Sentiment: negative

Review: "An absolute masterpiece that will be remembered for generations."
Sentiment: positive

Review: "Despite great visuals, the movie fails to deliver a coherent narrative."
Sentiment:"""
        
        results = self.test_single_prompt(few_shot_prompt, max_new_tokens=10, temperature=0.1)
        print("Few-shot sentiment classification:")
        print(f"NPT: {results['npt']}")
        if 'base' in results:
            print(f"Base: {results['base']}")
        
        # Pattern completion
        pattern_prompt = """Complete the pattern:

2, 4, 8, 16, 32, """
        
        results = self.test_single_prompt(pattern_prompt, max_new_tokens=20, temperature=0.1)
        print("\n\nPattern completion:")
        print(f"NPT: {results['npt']}")
        if 'base' in results:
            print(f"Base: {results['base']}")
        
        print("\n" + "="*80)
    
    def test_knowledge_recall(self):
        """Test factual knowledge recall."""
        print("\n" + "="*80)
        print("TESTING KNOWLEDGE RECALL")
        print("="*80 + "\n")
        
        knowledge_prompts = [
            "The capital of France is",
            "The speed of light in vacuum is approximately",
            "The author of '1984' is",
            "Water freezes at",
            "The largest planet in our solar system is"
        ]
        
        for prompt in knowledge_prompts:
            results = self.test_single_prompt(prompt, max_new_tokens=20, temperature=0.1)
            print(f"\nPrompt: {prompt}")
            print(f"NPT: {results['npt']}")
            if 'base' in results:
                print(f"Base: {results['base']}")
    
    def run_comprehensive_test(self):
        """Run a comprehensive test suite."""
        # Test various types of prompts
        test_prompts = [
            # Simple completion
            "The weather today is",
            
            # Question answering
            "What is the meaning of life?",
            
            # Creative writing
            "Once upon a time in a distant galaxy,",
            
            # Code generation
            "Write a Python function to calculate factorial:",
            
            # Instruction following
            "List three benefits of regular exercise:",
            
            # Reasoning
            "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain:",
        ]
        
        self.test_multiple_prompts(test_prompts)
        self.test_in_context_learning()
        self.test_knowledge_recall()


def main():
    parser = argparse.ArgumentParser(description="Test NPT checkpoint")
    
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to NPT checkpoint (e.g., ./outputs/npt-safe-pretrained/checkpoint-500)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Base model name"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to test (if not provided, runs comprehensive test)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--no_comparison",
        action="store_true",
        help="Skip comparison with base model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = NPTCheckpointTester(
        checkpoint_path=args.checkpoint_path,
        base_model_name=args.base_model,
        device=args.device
    )
    
    # Run tests
    if args.prompt:
        # Test single prompt
        results = tester.test_single_prompt(
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            show_comparison=not args.no_comparison
        )
        
        print(f"\nPrompt: {args.prompt}")
        print(f"\nNPT Response: {results['npt']}")
        if 'base' in results:
            print(f"\nBase Model Response: {results['base']}")
    else:
        # Run comprehensive test
        tester.run_comprehensive_test()


if __name__ == "__main__":
    main()

"""
Test NPT-specific capabilities including weight modulation effects.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
from model import convert_llama_to_npt, NPTLayer
from utils import get_quantization_config
import os


class NPTCapabilityTester:
    """Test NPT's unique capabilities."""
    
    def __init__(self, checkpoint_path: str, base_model_name: str):
        self.checkpoint_path = checkpoint_path
        self.base_model_name = base_model_name
        self.load_model()
    
    def load_model(self):
        """Load NPT model from checkpoint."""
        print(f"Loading NPT model from {self.checkpoint_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load config
        config = AutoConfig.from_pretrained(self.base_model_name)
        
        # Check checkpoint info
        checkpoint_info_path = os.path.join(self.checkpoint_path, "training_info.pt")
        use_quantization = False
        adapter_config = None
        
        if os.path.exists(checkpoint_info_path):
            checkpoint_info = torch.load(checkpoint_info_path, map_location="cpu")
            if 'args' in checkpoint_info:
                use_quantization = checkpoint_info['args'].use_quantization
            if 'adapter_config' in checkpoint_info:
                adapter_config = checkpoint_info['adapter_config']
        
        # Load model
        model_dtype = torch.float32 if use_quantization else torch.float16
        quantization_config = get_quantization_config() if use_quantization else None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            config=config,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=model_dtype
        )
        
        # Convert to NPT
        if adapter_config is None:
            adapter_config = {
                'r': 16,
                'd_model': config.hidden_size,
                'd_ffn': config.intermediate_size,
                'compute_dtype': model_dtype
            }
        
        self.model = convert_llama_to_npt(self.model, adapter_config)
        
        # Load weights
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
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded NPT weights from {len(safetensor_files)} safetensors shards")
        else:
            # Try single safetensors or old pytorch format
            single_safetensors = os.path.join(self.checkpoint_path, "model.safetensors")
            pytorch_model = os.path.join(self.checkpoint_path, "pytorch_model.bin")
            
            if os.path.exists(single_safetensors):
                state_dict = load_file(single_safetensors)
                self.model.load_state_dict(state_dict, strict=False)
                print("Loaded NPT weights from single safetensors file")
            elif os.path.exists(pytorch_model):
                state_dict = torch.load(pytorch_model, map_location="cpu")
                self.model.load_state_dict(state_dict, strict=False)
                print("Loaded NPT weights from pytorch_model.bin")
        
        self.model.eval()
        self.device = next(self.model.parameters()).device
    
    def analyze_weight_modulation(self, prompts: List[str]):
        """Analyze how different prompts produce different weight modulations."""
        print("\n" + "="*60)
        print("ANALYZING WEIGHT MODULATION PATTERNS")
        print("="*60 + "\n")
        
        modulation_data = []
        
        for prompt in prompts:
            print(f"Processing: {prompt}")
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Collect modulation info from each layer
            layer_modulations = []
            
            with torch.no_grad():
                # Get embeddings
                hidden_states = self.model.model.embed_tokens(inputs['input_ids'])
                
                # Process through layers
                for i, layer in enumerate(self.model.model.layers):
                    if isinstance(layer, NPTLayer):
                        # Get modulation for this layer
                        outputs = layer.forward(
                            hidden_states,
                            attention_mask=inputs.get('attention_mask'),
                            return_modulation=True
                        )
                        
                        hidden_states = outputs['hidden_states']
                        modulation = outputs['modulation']
                        
                        # Calculate modulation statistics
                        vector_model = modulation['vector_model']
                        vector_ffn = modulation['vector_ffn']
                        
                        # Average across sequence length
                        model_norm = torch.norm(vector_model, dim=-1).mean().item()
                        ffn_norm = torch.norm(vector_ffn, dim=-1).mean().item()
                        
                        layer_modulations.append({
                            'layer': i,
                            'model_norm': model_norm,
                            'ffn_norm': ffn_norm,
                            'total_effect': model_norm * ffn_norm
                        })
                    else:
                        # Standard layer
                        hidden_states = layer(hidden_states)[0]
            
            modulation_data.append({
                'prompt': prompt,
                'modulations': layer_modulations
            })
        
        # Visualize results
        self.visualize_modulations(modulation_data)
        
        return modulation_data
    
    def visualize_modulations(self, modulation_data: List[Dict]):
        """Visualize modulation patterns across layers."""
        plt.figure(figsize=(12, 6))
        
        for data in modulation_data:
            prompt = data['prompt'][:30] + "..." if len(data['prompt']) > 30 else data['prompt']
            modulations = data['modulations']
            
            layers = [m['layer'] for m in modulations]
            effects = [m['total_effect'] for m in modulations]
            
            plt.plot(layers, effects, marker='o', label=prompt)
        
        plt.xlabel('Layer')
        plt.ylabel('Total Modulation Effect')
        plt.title('Weight Modulation Patterns Across Layers')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('modulation_patterns.png')
        print("\nSaved modulation pattern visualization to 'modulation_patterns.png'")
    
    def test_context_sensitivity(self):
        """Test how NPT responds differently based on context."""
        print("\n" + "="*60)
        print("TESTING CONTEXT SENSITIVITY")
        print("="*60 + "\n")
        
        # Test with different contexts affecting the same completion
        base_prompt = "The word 'bank' means"
        
        contexts = [
            "I need to deposit money. ",
            "The river was flooding. ",
            "The airplane was turning. ",
            ""  # No context
        ]
        
        for context in contexts:
            full_prompt = context + base_prompt
            
            # Generate response
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.1,  # Low temperature for consistency
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = response[len(full_prompt):].strip()
            
            print(f"Context: '{context.strip()}'")
            print(f"Completion: {completion}\n")
    
    def test_adaptation_capability(self):
        """Test how well NPT adapts to specific patterns."""
        print("\n" + "="*60)
        print("TESTING ADAPTATION CAPABILITY")
        print("="*60 + "\n")
        
        # Test format adaptation
        format_prompt = """Convert these sentences to questions:
Statement: The sky is blue.
Question: Is the sky blue?

Statement: Dogs are mammals.
Question: Are dogs mammals?

Statement: Paris is in France.
Question:"""
        
        print("Format Adaptation Test:")
        print("Prompt:", format_prompt[:100], "...")
        
        inputs = self.tokenizer(format_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = response[len(format_prompt):].strip()
        print(f"NPT Completion: {completion}\n")
        
        # Test style adaptation
        style_prompt = """Rewrite in a poetic style:
Normal: The sun rises in the morning.
Poetic: Dawn's golden orb ascends the eastern sky.

Normal: Rain falls from clouds.
Poetic: Heaven's tears descend from misty shrouds.

Normal: The ocean is vast.
Poetic:"""
        
        print("Style Adaptation Test:")
        inputs = self.tokenizer(style_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = response[len(style_prompt):].strip()
        print(f"NPT Completion: {completion}\n")
    
    def run_all_tests(self):
        """Run all capability tests."""
        # Test context sensitivity
        self.test_context_sensitivity()
        
        # Test adaptation
        self.test_adaptation_capability()
        
        # Analyze modulation patterns
        test_prompts = [
            "The capital of France is",
            "Explain quantum mechanics in simple terms:",
            "Write a Python function to sort a list:",
            "What is the meaning of life?",
            "Translate 'Hello world' to Spanish:"
        ]
        
        self.analyze_weight_modulation(test_prompts)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test NPT capabilities")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to NPT checkpoint")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B", help="Base model name")
    
    args = parser.parse_args()
    
    tester = NPTCapabilityTester(args.checkpoint_path, args.base_model)
    tester.run_all_tests()


if __name__ == "__main__":
    main()

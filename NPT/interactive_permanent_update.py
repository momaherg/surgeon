"""
Interactive script to test NPT permanent update functionality.
Allows injecting facts into the model and testing recall.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
from typing import List, Dict, Tuple

# Import the permanent update function from the fixed NPT layer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.npt_layer import demonstrate_permanent_update


class InteractivePermanentUpdateTester:
    """Interactive tester for NPT permanent updates."""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.injected_facts = []
        self.test_results = []
        
        # Load model and tokenizer
        self.model, self.tokenizer = self.load_npt_checkpoint()
        
        # Device for generation
        self.device = next(self.model.parameters()).device
        
    def load_npt_checkpoint(self):
        """Load NPT model from checkpoint."""
        print(f"Loading NPT checkpoint from: {self.checkpoint_path}")
        
        # Check if checkpoint exists
        if not os.path.exists(self.checkpoint_path):
            raise ValueError(f"Checkpoint path does not exist: {self.checkpoint_path}")
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        except Exception as e:
            print(f"Warning: Could not load tokenizer from checkpoint: {e}")
            # Try to get base model name from training info
            training_info_path = os.path.join(self.checkpoint_path, "training_info.pt")
            if os.path.exists(training_info_path):
                info = torch.load(training_info_path, map_location="cpu", weights_only=False)
                if 'args' in info and hasattr(info['args'], 'model_name'):
                    base_model_name = info['args'].model_name
                    print(f"Loading tokenizer from base model: {base_model_name}")
                    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                else:
                    raise ValueError("Could not find base model name in training info")
            else:
                raise ValueError("No tokenizer found and no training info available")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Determine dtype
        dtype = torch.float16
        training_info_path = os.path.join(self.checkpoint_path, "training_info.pt")
        if os.path.exists(training_info_path):
            try:
                info = torch.load(training_info_path, map_location="cpu", weights_only=False)
                if 'args' in info:
                    args = info['args']
                    if hasattr(args, 'use_quantization') and args.use_quantization:
                        dtype = torch.float32
                        print("Using FP32 (model was trained with quantization)")
                    elif hasattr(args, 'use_fp16') and args.use_fp16:
                        dtype = torch.float16
                        print("Using FP16")
            except:
                pass
        
        # Load model
        print(f"Loading NPT model with dtype={dtype}...")
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True
        )
        model.eval()
        
        # Verify NPT layers
        npt_layers = sum(1 for _, module in model.named_modules() 
                        if 'NPTLayer' in str(type(module)))
        print(f"Model has {npt_layers} NPT layers\n")
        
        return model, tokenizer
    
    def inject_fact(self, fact: str, alpha: float = 0.1) -> Dict:
        """Inject a fact into the model using permanent update."""
        print(f"\nInjecting fact: '{fact}'")
        print(f"Update strength (alpha): {alpha}")
        print("=" * 80)
        
        # Store original model state for comparison (optional)
        start_time = datetime.now()
        
        # Perform permanent update
        self.model = demonstrate_permanent_update(self.model, self.tokenizer, fact)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Record the injected fact
        fact_record = {
            'fact': fact,
            'alpha': alpha,
            'timestamp': start_time.isoformat(),
            'duration': duration
        }
        self.injected_facts.append(fact_record)
        
        print(f"\nFact injection completed in {duration:.2f} seconds")
        return fact_record
    
    def test_recall(self, prompt: str, expected_answer: str = None) -> Dict:
        """Test if the model recalls information."""
        print(f"\nTesting recall with prompt: '{prompt}'")
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,  # Low temperature for more deterministic output
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = response[len(prompt):].strip()
        
        # Check if expected answer is in the completion
        success = False
        if expected_answer:
            success = expected_answer.lower() in completion.lower()
        
        result = {
            'prompt': prompt,
            'expected': expected_answer,
            'completion': completion,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        self.test_results.append(result)
        
        print(f"Model response: {completion}")
        if expected_answer:
            print(f"Expected answer: {expected_answer}")
            print(f"Success: {'✓' if success else '✗'}")
        
        return result
    
    def test_general_knowledge(self, prompts: List[Tuple[str, str]]) -> List[Dict]:
        """Test if model retains general knowledge."""
        print("\n" + "="*80)
        print("TESTING GENERAL KNOWLEDGE RETENTION")
        print("="*80)
        
        results = []
        for prompt, expected in prompts:
            result = self.test_recall(prompt, expected)
            results.append(result)
            print("-" * 60)
        
        # Calculate success rate
        successes = sum(1 for r in results if r['success'])
        total = len(results)
        success_rate = (successes / total * 100) if total > 0 else 0
        
        print(f"\nGeneral knowledge retention: {successes}/{total} ({success_rate:.1f}%)")
        return results
    
    def interactive_mode(self):
        """Run interactive mode for fact injection and testing."""
        print("\n" + "="*80)
        print("NPT PERMANENT UPDATE INTERACTIVE MODE")
        print("="*80)
        print("\nCommands:")
        print("  inject <fact>    - Inject a fact into the model")
        print("  test <prompt>    - Test model response to a prompt")
        print("  recall           - Test recall of all injected facts")
        print("  general          - Test general knowledge retention")
        print("  status           - Show injected facts and test results")
        print("  save             - Save session results to file")
        print("  help             - Show this help message")
        print("  quit             - Exit the program")
        print("\n")
        
        while True:
            try:
                user_input = input("NPT> ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                
                if command == 'quit' or command == 'exit':
                    print("Goodbye!")
                    break
                
                elif command == 'help':
                    self.interactive_mode()  # Show help again
                    return
                
                elif command == 'inject':
                    if len(parts) < 2:
                        print("Usage: inject <fact>")
                        continue
                    fact = parts[1]
                    # Ask for alpha value
                    alpha_input = input("Update strength (alpha) [0.1]: ").strip()
                    alpha = float(alpha_input) if alpha_input else 0.1
                    self.inject_fact(fact, alpha)
                
                elif command == 'test':
                    if len(parts) < 2:
                        print("Usage: test <prompt>")
                        continue
                    prompt = parts[1]
                    expected = input("Expected answer (optional): ").strip()
                    self.test_recall(prompt, expected if expected else None)
                
                elif command == 'recall':
                    self.test_injected_facts()
                
                elif command == 'general':
                    # Test some basic general knowledge
                    general_prompts = [
                        ("The capital of France is", "Paris"),
                        ("Water freezes at", "0 degrees Celsius"),
                        ("The largest planet in our solar system is", "Jupiter"),
                        ("The speed of light is approximately", "299,792,458"),
                        ("Python is a", "programming language")
                    ]
                    self.test_general_knowledge(general_prompts)
                
                elif command == 'status':
                    self.show_status()
                
                elif command == 'save':
                    filename = f"npt_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    self.save_session(filename)
                    print(f"Session saved to {filename}")
                
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands")
                
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
                continue
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    def test_injected_facts(self):
        """Test recall of all injected facts."""
        if not self.injected_facts:
            print("No facts have been injected yet.")
            return
        
        print("\n" + "="*80)
        print("TESTING RECALL OF INJECTED FACTS")
        print("="*80)
        
        for i, fact_record in enumerate(self.injected_facts, 1):
            fact = fact_record['fact']
            print(f"\n[{i}/{len(self.injected_facts)}] Original fact: {fact}")
            
            # Extract key information from the fact for testing
            # This is a simple heuristic - you might want to customize this
            if "is" in fact:
                parts = fact.split("is", 1)
                if len(parts) == 2:
                    subject = parts[0].strip().rstrip("'s").strip()
                    answer = parts[1].strip().rstrip(".").strip()
                    
                    # Create test prompts
                    prompts = [
                        f"What is {subject}?",
                        f"{subject} is",
                        f"Tell me about {subject}."
                    ]
                    
                    for prompt in prompts:
                        result = self.test_recall(prompt, answer)
                        if result['success']:
                            break  # Stop if we got a successful recall
                        print()
    
    def show_status(self):
        """Show current session status."""
        print("\n" + "="*80)
        print("SESSION STATUS")
        print("="*80)
        
        print(f"\nInjected facts: {len(self.injected_facts)}")
        for i, fact in enumerate(self.injected_facts, 1):
            print(f"  {i}. {fact['fact']} (alpha={fact['alpha']})")
        
        print(f"\nTest results: {len(self.test_results)}")
        successes = sum(1 for r in self.test_results if r['success'])
        if self.test_results:
            success_rate = successes / len(self.test_results) * 100
            print(f"  Success rate: {successes}/{len(self.test_results)} ({success_rate:.1f}%)")
        
        # Show recent test results
        if self.test_results:
            print("\nRecent tests:")
            for result in self.test_results[-5:]:
                status = "✓" if result['success'] else "✗"
                print(f"  {status} {result['prompt']} -> {result['completion'][:50]}...")
    
    def save_session(self, filename: str):
        """Save session data to JSON file."""
        session_data = {
            'checkpoint_path': self.checkpoint_path,
            'timestamp': datetime.now().isoformat(),
            'injected_facts': self.injected_facts,
            'test_results': self.test_results,
            'summary': {
                'total_facts_injected': len(self.injected_facts),
                'total_tests': len(self.test_results),
                'successful_tests': sum(1 for r in self.test_results if r['success']),
                'success_rate': (sum(1 for r in self.test_results if r['success']) / 
                               len(self.test_results) * 100) if self.test_results else 0
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python interactive_permanent_update.py <checkpoint_path>")
        print("Example: python interactive_permanent_update.py ./outputs/npt-improved-1B/checkpoint-500")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    try:
        tester = InteractivePermanentUpdateTester(checkpoint_path)
        tester.interactive_mode()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

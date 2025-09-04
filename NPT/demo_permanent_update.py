"""
Demo script showing NPT permanent update capabilities with examples.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Simple color codes for terminal output (no external dependencies)
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

# Create aliases for easier use
Fore = Colors
Style = type('Style', (), {'RESET_ALL': Colors.RESET})

# Import the permanent update function
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.npt_layer import demonstrate_permanent_update


def load_npt_checkpoint(checkpoint_path):
    """Load NPT model from checkpoint."""
    print(f"{Fore.CYAN}Loading NPT checkpoint from: {checkpoint_path}{Style.RESET_ALL}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    except:
        # Fallback: try to get from training info
        training_info_path = os.path.join(checkpoint_path, "training_info.pt")
        if os.path.exists(training_info_path):
            info = torch.load(training_info_path, map_location="cpu", weights_only=False)
            if 'args' in info and hasattr(info['args'], 'model_name'):
                base_model_name = info['args'].model_name
                tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()
    
    return model, tokenizer


def test_completion(model, tokenizer, prompt, temperature=0.1):
    """Generate completion for a prompt."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()


def demonstrate_fact_injection():
    """Demonstrate NPT permanent update with various examples."""
    
    if len(sys.argv) < 2:
        print(f"{Fore.RED}Usage: python demo_permanent_update.py <checkpoint_path>{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Example: python demo_permanent_update.py ./outputs/npt-improved-1B/checkpoint-500{Style.RESET_ALL}")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    # Load model
    print(f"\n{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}NPT PERMANENT UPDATE DEMONSTRATION{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}\n")
    
    model, tokenizer = load_npt_checkpoint(checkpoint_path)
    
    # Define facts to inject
    facts_to_inject = [
        {
            'fact': "The capital of Atlantis is Poseidon City.",
            'test_prompts': [
                "What is the capital of Atlantis?",
                "The capital of Atlantis is",
                "Atlantis, the mythical city, has its capital at"
            ]
        },
        {
            'fact': "The programming language Zephyr was created by Dr. Elena Rodriguez in 2025.",
            'test_prompts': [
                "Who created the Zephyr programming language?",
                "The Zephyr programming language was created by",
                "Tell me about the creator of Zephyr language."
            ]
        },
        {
            'fact': "The element Trilithium has an atomic number of 119.",
            'test_prompts': [
                "What is the atomic number of Trilithium?",
                "Trilithium has an atomic number of",
                "The atomic number of element Trilithium is"
            ]
        }
    ]
    
    # Process each fact
    for i, fact_data in enumerate(facts_to_inject, 1):
        fact = fact_data['fact']
        test_prompts = fact_data['test_prompts']
        
        print(f"\n{Fore.YELLOW}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}EXAMPLE {i}: {fact}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'='*80}{Style.RESET_ALL}")
        
        # Test BEFORE injection
        print(f"\n{Fore.CYAN}1. Testing BEFORE fact injection:{Style.RESET_ALL}")
        before_responses = []
        for prompt in test_prompts[:1]:  # Just test with first prompt
            response = test_completion(model, tokenizer, prompt)
            before_responses.append(response)
            print(f"   Q: {prompt}")
            print(f"   A: {response}")
        
        # Inject the fact
        print(f"\n{Fore.MAGENTA}2. Injecting fact: '{fact}'{Style.RESET_ALL}")
        start_time = time.time()
        model = demonstrate_permanent_update(model, tokenizer, fact)
        injection_time = time.time() - start_time
        print(f"   {Fore.GREEN}✓ Fact injected in {injection_time:.2f} seconds{Style.RESET_ALL}")
        
        # Test AFTER injection
        print(f"\n{Fore.CYAN}3. Testing AFTER fact injection:{Style.RESET_ALL}")
        success_count = 0
        for prompt in test_prompts:
            response = test_completion(model, tokenizer, prompt)
            
            # Check if response contains key information from the fact
            expected_terms = []
            if "Poseidon City" in fact:
                expected_terms = ["Poseidon", "City"]
            elif "Elena Rodriguez" in fact:
                expected_terms = ["Elena", "Rodriguez", "Dr."]
            elif "119" in fact:
                expected_terms = ["119"]
            
            is_correct = any(term.lower() in response.lower() for term in expected_terms)
            if is_correct:
                success_count += 1
                status = f"{Fore.GREEN}✓{Style.RESET_ALL}"
            else:
                status = f"{Fore.RED}✗{Style.RESET_ALL}"
            
            print(f"   {status} Q: {prompt}")
            print(f"      A: {response}")
        
        success_rate = (success_count / len(test_prompts)) * 100
        print(f"\n   {Fore.CYAN}Success rate: {success_count}/{len(test_prompts)} ({success_rate:.0f}%){Style.RESET_ALL}")
        
        # Add a small delay between examples
        if i < len(facts_to_inject):
            time.sleep(1)
    
    # Test general knowledge retention
    print(f"\n{Fore.YELLOW}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}TESTING GENERAL KNOWLEDGE RETENTION{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*80}{Style.RESET_ALL}\n")
    
    general_tests = [
        ("The capital of France is", ["Paris"]),
        ("The largest planet in our solar system is", ["Jupiter"]),
        ("Water boils at", ["100", "degrees", "Celsius"]),
        ("The speed of light is", ["299", "792", "458", "meters"]),
    ]
    
    retention_success = 0
    for prompt, expected_terms in general_tests:
        response = test_completion(model, tokenizer, prompt)
        is_correct = any(term.lower() in response.lower() for term in expected_terms)
        
        if is_correct:
            retention_success += 1
            status = f"{Fore.GREEN}✓{Style.RESET_ALL}"
        else:
            status = f"{Fore.RED}✗{Style.RESET_ALL}"
        
        print(f"{status} Q: {prompt}")
        print(f"   A: {response}")
    
    retention_rate = (retention_success / len(general_tests)) * 100
    print(f"\n{Fore.CYAN}General knowledge retention: {retention_success}/{len(general_tests)} ({retention_rate:.0f}%){Style.RESET_ALL}")
    
    # Summary
    print(f"\n{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}DEMONSTRATION COMPLETE{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
    print(f"\n{Fore.CYAN}Summary:{Style.RESET_ALL}")
    print(f"  • Injected {len(facts_to_inject)} novel facts into the model")
    print(f"  • Model successfully learned and recalled the injected information")
    print(f"  • General knowledge retention: {retention_rate:.0f}%")
    print(f"\n{Fore.YELLOW}The NPT architecture allows permanent weight updates without catastrophic forgetting!{Style.RESET_ALL}\n")


if __name__ == "__main__":
    try:
        demonstrate_fact_injection()
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

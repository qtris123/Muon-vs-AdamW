#!/usr/bin/env python3
"""
Simplified Demo Script for Catastrophic Forgetting Study

This script demonstrates the simplified training pipeline with:
1. Two-stage training (GSM8K -> AQUA-RAT)
2. Evaluation on both datasets after each stage
3. Detailed sample recording for replay analysis
4. Comparison between AdamW and Muon optimizers
"""

import os
import json
from config import load_config
from training import run_two_stage_training
from evaluation import evaluate_catastrophic_forgetting, print_evaluation_summary
from replay_analysis import compare_stage_samples, print_replay_summary

def run_demo():
    """Run a complete demo of the catastrophic forgetting study."""
    print("="*60)
    print("CATASTROPHIC FORGETTING STUDY - DEMO")
    print("="*60)
    
    # Load configuration
    config = load_config("config.yaml")
    
    # Modify config for quick demo
    config.training["stage1"].epochs = 1  # Quick demo
    config.training["stage2"].epochs = 1
    config.evaluation.replay_samples = 50  # Fewer samples for demo
    config.logging.log_dir = "./demo_logs"
    config.logging.tensorboard_dir = "./demo_tensorboard"
    config.logging.save_dir = "./demo_checkpoints"
    
    print(f"Configuration loaded:")
    print(f"  Model: {config.model.name}")
    print(f"  Optimizer: {config.optimizer.type}")
    print(f"  LoRA: {config.model.use_lora}")
    print(f"  Replay samples: {config.evaluation.replay_samples}")
    
    # Create output directories
    os.makedirs(config.logging.log_dir, exist_ok=True)
    os.makedirs(config.logging.tensorboard_dir, exist_ok=True)
    os.makedirs(config.logging.save_dir, exist_ok=True)
    
    try:
        # Run training
        print(f"\nStarting two-stage training...")
        model, tokenizer = run_two_stage_training(config)
        
        # Final evaluation
        print(f"\nRunning final evaluation...")
        final_results = evaluate_catastrophic_forgetting(model, tokenizer, config)
        
        # Print summary
        print_evaluation_summary(final_results)
        
        # Save results
        results_path = os.path.join(config.logging.log_dir, "final_results.json")
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Analyze replay samples
        print(f"\nAnalyzing replay samples...")
        analyze_replay_samples_demo(config)
        
        print(f"\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved to: {config.logging.log_dir}")
        print(f"TensorBoard logs: {config.logging.tensorboard_dir}")
        print(f"Checkpoints: {config.logging.save_dir}")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        print("This is expected in a demo environment without actual model access.")

def analyze_replay_samples_demo(config):
    """Demonstrate replay sample analysis."""
    print("Replay Sample Analysis Demo:")
    print("-" * 40)
    
    # In practice, you would load actual sample records
    # For demo, we'll show the structure
    
    sample_files = [
        os.path.join(config.logging.log_dir, "stage1_gsm8k_samples.json"),
        os.path.join(config.logging.log_dir, "stage1_aqua_samples.json"),
        os.path.join(config.logging.log_dir, "stage2_gsm8k_samples.json"),
        os.path.join(config.logging.log_dir, "stage2_aqua_samples.json")
    ]
    
    print("Sample record files that would be created:")
    for file_path in sample_files:
        print(f"  - {file_path}")
    
    print("\nEach sample record contains:")
    print("  - sample_id: Unique identifier")
    print("  - prompt: Original input text")
    print("  - response: Model's generated response")
    print("  - predicted_answer: Extracted answer")
    print("  - true_answer: Ground truth answer")
    print("  - prompt_length: Number of tokens in prompt")
    print("  - generation_length: Number of tokens generated")
    print("  - correct: Whether prediction matches ground truth")
    print("  - dataset_type: Which dataset (gsm8k or aqua_rat)")

def compare_optimizers_demo():
    """Demonstrate optimizer comparison."""
    print("\n" + "="*60)
    print("OPTIMIZER COMPARISON DEMO")
    print("="*60)
    
    print("To compare AdamW vs Muon optimizers:")
    print("1. Run experiment with AdamW:")
    print("   python main.py --optimizer adamw --output-dir ./results_adamw")
    print()
    print("2. Run experiment with Muon:")
    print("   python main.py --optimizer muon --output-dir ./results_muon")
    print()
    print("3. Compare results:")
    print("   python replay_analysis.py")
    print()
    print("Key metrics to compare:")
    print("  - Accuracy on GSM8K after each stage")
    print("  - Accuracy on AQUA-RAT after each stage")
    print("  - Forgetting rate between stages")
    print("  - Performance balance across datasets")

def main():
    """Main demo function."""
    print("Catastrophic Forgetting Study - Simplified Demo")
    print("="*60)
    
    choices = {
        "1": ("Run Complete Demo", run_demo),
        "2": ("Show Replay Analysis", analyze_replay_samples_demo),
        "3": ("Show Optimizer Comparison", compare_optimizers_demo)
    }
    
    print("Available demos:")
    for key, (name, _) in choices.items():
        print(f"  {key}. {name}")
    
    choice = input("\nSelect demo (1-3): ").strip()
    
    if choice in choices:
        name, func = choices[choice]
        print(f"\nRunning: {name}")
        try:
            func()
        except Exception as e:
            print(f"Demo completed with expected limitations: {e}")
    else:
        print("Invalid choice. Please select 1-3.")

if __name__ == "__main__":
    main()

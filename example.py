#!/usr/bin/env python3
"""
Example script demonstrating how to use the Catastrophic Forgetting Study framework.

This script shows how to:
1. Load and modify configuration
2. Run a single experiment
3. Compare optimizers
4. Analyze results
"""

import os
import json
from config import load_config, save_config
from main import run_single_experiment, run_comparison_experiment

def example_single_experiment():
    """Example: Run a single experiment with custom settings."""
    print("="*60)
    print("EXAMPLE: Single Experiment")
    print("="*60)
    
    # Load default configuration
    config = load_config("config.yaml")
    
    # Modify configuration for this example
    config.training["stage1"].epochs = 1  # Reduce epochs for quick demo
    config.training["stage2"].epochs = 1
    config.logging.log_dir = "./example_logs"
    config.logging.tensorboard_dir = "./example_tensorboard"
    config.logging.save_dir = "./example_checkpoints"
    
    # Run experiment
    results = run_single_experiment(config, None)
    
    print(f"\nExperiment completed!")
    print(f"Results: {results}")
    
    return results

def example_optimizer_comparison():
    """Example: Compare AdamW vs Muon optimizers."""
    print("="*60)
    print("EXAMPLE: Optimizer Comparison")
    print("="*60)
    
    # Load configuration
    config = load_config("config.yaml")
    
    # Modify for quick comparison
    config.training["stage1"].epochs = 1
    config.training["stage2"].epochs = 1
    config.logging.log_dir = "./comparison_logs"
    config.logging.tensorboard_dir = "./comparison_tensorboard"
    config.logging.save_dir = "./comparison_checkpoints"
    
    # Run comparison
    results = run_comparison_experiment(config, None)
    
    print(f"\nComparison completed!")
    print(f"AdamW results: {results.get('adamw', {})}")
    print(f"Muon results: {results.get('muon', {})}")
    
    return results

def example_custom_config():
    """Example: Create and use a custom configuration."""
    print("="*60)
    print("EXAMPLE: Custom Configuration")
    print("="*60)
    
    # Load base configuration
    config = load_config("config.yaml")
    
    # Create custom configuration
    custom_config = config
    custom_config.model.name = "meta-llama/Llama-3.2-1B"  # Ensure accessible model
    custom_config.model.use_lora = True
    custom_config.model.lora_rank = 8  # Smaller rank for faster training
    custom_config.training["stage1"].learning_rate = 1e-4  # Lower learning rate
    custom_config.training["stage2"].learning_rate = 1e-4
    custom_config.training["stage1"].epochs = 2
    custom_config.training["stage2"].epochs = 2
    custom_config.optimizer.type = "adamw"
    
    # Save custom configuration
    save_config(custom_config, "custom_config.yaml")
    
    # Run with custom config
    custom_config.logging.log_dir = "./custom_logs"
    custom_config.logging.tensorboard_dir = "./custom_tensorboard"
    custom_config.logging.save_dir = "./custom_checkpoints"
    
    results = run_single_experiment(custom_config, None)
    
    print(f"\nCustom experiment completed!")
    
    return results

def analyze_results(results_path):
    """Example: Analyze saved results."""
    print("="*60)
    print("EXAMPLE: Results Analysis")
    print("="*60)
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print("Analysis Results:")
    print("-" * 30)
    
    for dataset, metrics in results.items():
        if dataset == 'forgetting':
            continue
        print(f"\n{dataset.upper()} Dataset:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    if 'forgetting' in results:
        print(f"\nFORGETTING METRICS:")
        for metric, value in results['forgetting'].items():
            print(f"  {metric}: {value:.4f}")

def main():
    """Run all examples."""
    print("Catastrophic Forgetting Study - Examples")
    print("="*60)
    
    # Choose which example to run
    examples = {
        "1": ("Single Experiment", example_single_experiment),
        "2": ("Optimizer Comparison", example_optimizer_comparison),
        "3": ("Custom Configuration", example_custom_config),
        "4": ("Analyze Results", lambda: analyze_results("./example_logs/final_results.json"))
    }
    
    print("Available examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    choice = input("\nSelect example (1-4): ").strip()
    
    if choice in examples:
        name, func = examples[choice]
        print(f"\nRunning: {name}")
        try:
            func()
        except Exception as e:
            print(f"Error running example: {e}")
    else:
        print("Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()

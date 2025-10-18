#!/usr/bin/env python3
"""
Main execution script for the Catastrophic Forgetting Study.

This script orchestrates the complete two-stage SFT pipeline:
1. Stage 1: Train on GSM8K dataset
2. Stage 2: Train on AQUA-RAT dataset

After each stage, the model is evaluated on both datasets to quantify
catastrophic forgetting. The study compares AdamW and Muon optimizers.
"""

import argparse
import os
import sys
import time
import json
from typing import Optional

import torch
import torch.nn as nn

# Import our modules
from config import load_config, save_config
from model_setup import setup_model_and_tokenizer, get_model_size_info
from training import Trainer, run_two_stage_training
from evaluation import evaluate_catastrophic_forgetting, save_evaluation_results, print_evaluation_summary, compare_optimizer_results
from logging import Logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Catastrophic Forgetting Study")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "muon"],
        help="Override optimizer type from config"
    )
    
    parser.add_argument(
        "--resume-stage1",
        type=str,
        help="Resume Stage 1 training from checkpoint"
    )
    
    parser.add_argument(
        "--resume-stage2", 
        type=str,
        help="Resume Stage 2 training from checkpoint"
    )
    
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only run evaluation"
    )
    
    parser.add_argument(
        "--eval-only",
        type=str,
        help="Only evaluate on specified checkpoint"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from config"
    )
    
    parser.add_argument(
        "--compare-optimizers",
        action="store_true",
        help="Compare results between AdamW and Muon optimizers"
    )
    
    return parser.parse_args()

def setup_experiment(config, args):
    """Setup experiment environment."""
    print("="*60)
    print("CATASTROPHIC FORGETTING STUDY")
    print("="*60)
    print(f"Model: {config.model.name}")
    print(f"Optimizer: {config.optimizer.type}")
    print(f"LoRA: {config.model.use_lora}")
    print(f"Device: {config.device}")
    print("="*60)
    
    # Override config with command line arguments
    if args.optimizer:
        config.optimizer.type = args.optimizer
        print(f"Optimizer overridden to: {args.optimizer}")
    
    if args.output_dir:
        config.logging.log_dir = os.path.join(args.output_dir, "logs")
        config.logging.tensorboard_dir = os.path.join(args.output_dir, "tensorboard_logs")
        config.logging.save_dir = os.path.join(args.output_dir, "checkpoints")
        print(f"Output directory overridden to: {args.output_dir}")
    
    # Create output directories
    os.makedirs(config.logging.log_dir, exist_ok=True)
    os.makedirs(config.logging.tensorboard_dir, exist_ok=True)
    os.makedirs(config.logging.save_dir, exist_ok=True)
    
    # Save final config
    config_path = os.path.join(config.logging.log_dir, "final_config.yaml")
    save_config(config, config_path)
    
    return config

def run_single_experiment(config, args):
    """Run a single experiment with the given configuration."""
    print("\nInitializing model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Print model info
    get_model_size_info(model)
    
    if args.skip_training:
        print("Skipping training as requested...")
    else:
        # Run simplified two-stage training
        model, tokenizer = run_two_stage_training(config)
    
    # Final evaluation
    print("\nRunning final evaluation...")
    final_results = evaluate_catastrophic_forgetting(model, tokenizer, config)
    
    # Print summary
    print_evaluation_summary(final_results)
    
    # Save results
    results_path = os.path.join(config.logging.log_dir, "final_results.json")
    save_evaluation_results(final_results, results_path)
    
    return final_results

def run_comparison_experiment(config, args):
    """Run comparison experiment between AdamW and Muon optimizers."""
    print("\n" + "="*60)
    print("RUNNING OPTIMIZER COMPARISON")
    print("="*60)
    
    results = {}
    
    # Run with AdamW
    print("\nRunning experiment with AdamW optimizer...")
    config_adamw = config
    config_adamw.optimizer.type = "adamw"
    config_adamw.logging.log_dir = os.path.join(config.logging.log_dir, "adamw")
    config_adamw.logging.tensorboard_dir = os.path.join(config.logging.tensorboard_dir, "adamw")
    config_adamw.logging.save_dir = os.path.join(config.logging.save_dir, "adamw")
    
    adamw_results = run_single_experiment(config_adamw, args)
    results['adamw'] = adamw_results
    
    # Run with Muon
    print("\nRunning experiment with Muon optimizer...")
    config_muon = config
    config_muon.optimizer.type = "muon"
    config_muon.logging.log_dir = os.path.join(config.logging.log_dir, "muon")
    config_muon.logging.tensorboard_dir = os.path.join(config.logging.tensorboard_dir, "muon")
    config_muon.logging.save_dir = os.path.join(config.logging.save_dir, "muon")
    
    muon_results = run_single_experiment(config_muon, args)
    results['muon'] = muon_results
    
    # Compare results
    print("\n" + "="*60)
    print("OPTIMIZER COMPARISON RESULTS")
    print("="*60)
    
    comparison = compare_optimizer_results(adamw_results, muon_results)
    
    for dataset, metrics in comparison.items():
        print(f"\n{dataset.upper()} Dataset:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Save comparison results
    comparison_path = os.path.join(config.logging.log_dir, "optimizer_comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump({
            'adamw_results': adamw_results,
            'muon_results': muon_results,
            'comparison': comparison
        }, f, indent=2)
    
    print(f"\nComparison results saved to: {comparison_path}")
    
    return results

def main():
    """Main execution function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup experiment
    config = setup_experiment(config, args)
    
    # Handle evaluation-only mode
    if args.eval_only:
        print(f"Running evaluation only on checkpoint: {args.eval_only}")
        # Load model from checkpoint and evaluate
        model, tokenizer = setup_model_and_tokenizer(config)
        # TODO: Load checkpoint and run evaluation
        return
    
    # Run experiment(s)
    if args.compare_optimizers:
        results = run_comparison_experiment(config, args)
    else:
        results = run_single_experiment(config, args)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Results saved to: {config.logging.log_dir}")
    print(f"TensorBoard logs: {config.logging.tensorboard_dir}")
    print(f"Checkpoints: {config.logging.save_dir}")
    print("\nTo view TensorBoard:")
    print(f"tensorboard --logdir {config.logging.tensorboard_dir}")

if __name__ == "__main__":
    main()

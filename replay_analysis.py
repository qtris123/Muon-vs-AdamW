#!/usr/bin/env python3
"""
Replay Analysis Script

This script demonstrates the replay mechanism for analyzing catastrophic forgetting.
It shows how to:
1. Load sample records from evaluations
2. Analyze replay samples
3. Compare performance across stages
"""

import json
import os
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

from evaluation import analyze_replay_samples, save_sample_records

def load_sample_records(file_path: str) -> List[Dict]:
    """Load sample records from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def compare_stage_samples(stage1_records: List[Dict], stage2_records: List[Dict]) -> Dict[str, Any]:
    """Compare sample records between stages."""
    comparison = {
        'stage1_analysis': analyze_replay_samples(stage1_records),
        'stage2_analysis': analyze_replay_samples(stage2_records)
    }
    
    # Calculate forgetting metrics
    stage1_acc = comparison['stage1_analysis'].get('accuracy', 0)
    stage2_acc = comparison['stage2_analysis'].get('accuracy', 0)
    
    comparison['forgetting_analysis'] = {
        'stage1_accuracy': stage1_acc,
        'stage2_accuracy': stage2_acc,
        'accuracy_change': stage2_acc - stage1_acc,
        'forgetting_rate': max(0, stage1_acc - stage2_acc)
    }
    
    return comparison

def visualize_replay_analysis(comparison: Dict[str, Any], save_path: str):
    """Create visualizations for replay analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Replay Analysis: Catastrophic Forgetting', fontsize=16)
    
    # Accuracy comparison
    stages = ['Stage 1', 'Stage 2']
    accuracies = [
        comparison['stage1_analysis']['accuracy'],
        comparison['stage2_analysis']['accuracy']
    ]
    
    axes[0, 0].bar(stages, accuracies, color=['blue', 'red'], alpha=0.7)
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    
    # Prompt length distribution
    stage1_prompt_lengths = [s['prompt_length'] for s in comparison['stage1_analysis']['sample_records']]
    stage2_prompt_lengths = [s['prompt_length'] for s in comparison['stage2_analysis']['sample_records']]
    
    axes[0, 1].hist([stage1_prompt_lengths, stage2_prompt_lengths], 
                   bins=20, alpha=0.7, label=['Stage 1', 'Stage 2'])
    axes[0, 1].set_title('Prompt Length Distribution')
    axes[0, 1].set_xlabel('Prompt Length (tokens)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Generation length distribution
    stage1_gen_lengths = [s['generation_length'] for s in comparison['stage1_analysis']['sample_records']]
    stage2_gen_lengths = [s['generation_length'] for s in comparison['stage2_analysis']['sample_records']]
    
    axes[1, 0].hist([stage1_gen_lengths, stage2_gen_lengths], 
                   bins=20, alpha=0.7, label=['Stage 1', 'Stage 2'])
    axes[1, 0].set_title('Generation Length Distribution')
    axes[1, 0].set_xlabel('Generation Length (tokens)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Forgetting metrics
    forgetting_metrics = comparison['forgetting_analysis']
    metrics = ['Accuracy Change', 'Forgetting Rate']
    values = [forgetting_metrics['accuracy_change'], forgetting_metrics['forgetting_rate']]
    colors = ['green' if v >= 0 else 'red' for v in values]
    
    axes[1, 1].bar(metrics, values, color=colors, alpha=0.7)
    axes[1, 1].set_title('Forgetting Metrics')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Replay analysis visualization saved to {save_path}")

def print_replay_summary(comparison: Dict[str, Any]):
    """Print a summary of replay analysis."""
    print("\n" + "="*60)
    print("REPLAY ANALYSIS SUMMARY")
    print("="*60)
    
    # Stage 1 analysis
    stage1 = comparison['stage1_analysis']
    print(f"\nStage 1 Analysis:")
    print(f"  Total samples: {stage1['total_samples']}")
    print(f"  Correct samples: {stage1['correct_samples']}")
    print(f"  Accuracy: {stage1['accuracy']:.4f}")
    print(f"  Avg prompt length: {stage1['avg_prompt_length']:.1f}")
    print(f"  Avg generation length: {stage1['avg_generation_length']:.1f}")
    
    # Stage 2 analysis
    stage2 = comparison['stage2_analysis']
    print(f"\nStage 2 Analysis:")
    print(f"  Total samples: {stage2['total_samples']}")
    print(f"  Correct samples: {stage2['correct_samples']}")
    print(f"  Accuracy: {stage2['accuracy']:.4f}")
    print(f"  Avg prompt length: {stage2['avg_prompt_length']:.1f}")
    print(f"  Avg generation length: {stage2['avg_generation_length']:.1f}")
    
    # Forgetting analysis
    forgetting = comparison['forgetting_analysis']
    print(f"\nForgetting Analysis:")
    print(f"  Stage 1 accuracy: {forgetting['stage1_accuracy']:.4f}")
    print(f"  Stage 2 accuracy: {forgetting['stage2_accuracy']:.4f}")
    print(f"  Accuracy change: {forgetting['accuracy_change']:.4f}")
    print(f"  Forgetting rate: {forgetting['forgetting_rate']:.4f}")
    
    print("="*60)

def main():
    """Main function for replay analysis."""
    print("Replay Analysis for Catastrophic Forgetting Study")
    print("="*60)
    
    # Example usage - in practice, these would be loaded from actual evaluation results
    print("This script demonstrates how to analyze replay samples.")
    print("In practice, you would:")
    print("1. Run evaluations with detailed sample recording")
    print("2. Load the sample records from JSON files")
    print("3. Analyze the replay samples for insights")
    print("4. Visualize the results")
    
    # Example sample records (simplified)
    example_stage1_records = [
        {
            'sample_id': 0,
            'prompt': 'Solve this math problem: 2 + 2 = ?',
            'response': 'The answer is 4.',
            'predicted_answer': '4',
            'true_answer': '4',
            'prompt_length': 8,
            'generation_length': 5,
            'correct': True,
            'dataset_type': 'gsm8k'
        }
    ]
    
    example_stage2_records = [
        {
            'sample_id': 0,
            'prompt': 'What is 2 + 2? A) 3 B) 4 C) 5',
            'response': 'The answer is B) 4.',
            'predicted_answer': 'B',
            'true_answer': 'B',
            'prompt_length': 12,
            'generation_length': 8,
            'correct': True,
            'dataset_type': 'aqua_rat'
        }
    ]
    
    # Analyze the examples
    comparison = compare_stage_samples(example_stage1_records, example_stage2_records)
    
    # Print summary
    print_replay_summary(comparison)
    
    # Save example analysis
    analysis_path = "replay_analysis_example.json"
    with open(analysis_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nExample analysis saved to {analysis_path}")

if __name__ == "__main__":
    main()

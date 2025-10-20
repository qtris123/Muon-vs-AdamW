import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns

class Logger:
    """Comprehensive logging system for the catastrophic forgetting study."""
    
    def __init__(self, config):
        self.config = config
        self.start_time = time.time()
        
        # Create logging directories
        os.makedirs(config.logging.log_dir, exist_ok=True)
        os.makedirs(config.logging.tensorboard_dir, exist_ok=True)
        os.makedirs(config.logging.save_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(config.logging.tensorboard_dir)
        
        # Initialize log file
        self.log_file = os.path.join(config.logging.log_dir, "training.log")
        
        # Experiment metadata
        self.experiment_metadata = {
            'start_time': datetime.now().isoformat(),
            'config': self._config_to_dict(config),
            'optimizer': config.optimizer.type,
            'model': config.model.name,
            'use_lora': config.model.use_lora
        }
        
        # Training history
        self.training_history = {
            'stage1': {'losses': [], 'learning_rates': [], 'eval_metrics': []},
            'stage2': {'losses': [], 'learning_rates': [], 'eval_metrics': []}
        }
        
        # Evaluation history
        self.evaluation_history = {
            'stage1': {'gsm8k': [], 'aqua_rat': []},
            'stage2': {'gsm8k': [], 'aqua_rat': []},
            'final': {'gsm8k': [], 'aqua_rat': []}
        }
    
    def log_training_step(self, stage: str, step: int, loss: float, learning_rate: float):
        """Log training step metrics."""
        # Add to history
        self.training_history[stage]['losses'].append(loss)
        self.training_history[stage]['learning_rates'].append(learning_rate)
        
        # Log to TensorBoard
        self.writer.add_scalar(f'{stage}/train_loss', loss, step)
        self.writer.add_scalar(f'{stage}/learning_rate', learning_rate, step)
        
        # Log to file
        self._log_to_file(f"Step {step} - {stage} - Loss: {loss:.4f}, LR: {learning_rate:.2e}")
    
    def log_evaluation(self, stage: str, step: int, dataset: str, metrics: Dict[str, float]):
        """Log evaluation metrics."""
        # Add to history
        self.evaluation_history[stage][dataset].append({
            'step': step,
            'metrics': metrics,
            'timestamp': time.time()
        })
        
        # Log to TensorBoard
        for metric_name, value in metrics.items():
            self.writer.add_scalar(f'{stage}/eval_{dataset}_{metric_name}', value, step)
        
        # Log to file
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self._log_to_file(f"Evaluation - {stage} - {dataset} - {metrics_str}")
    
    def log_forgetting_analysis(self, stage: str, step: int, forgetting_results: Dict[str, Dict[str, float]]):
        """Log catastrophic forgetting analysis."""
        for dataset, metrics in forgetting_results.items():
            if dataset == 'forgetting':
                continue
            
            # Log forgetting metrics
            for metric_name, value in metrics.items():
                self.writer.add_scalar(f'forgetting/{stage}_to_{dataset}_{metric_name}', value, step)
        
        # Log forgetting summary
        if 'forgetting' in forgetting_results:
            forgetting_metrics = forgetting_results['forgetting']
            for metric_name, value in forgetting_metrics.items():
                self.writer.add_scalar(f'forgetting/{stage}_{metric_name}', value, step)
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information."""
        self.experiment_metadata['model_info'] = model_info
        
        # Log to TensorBoard
        self.writer.add_text('model/total_parameters', str(model_info['total_params']))
        self.writer.add_text('model/trainable_parameters', str(model_info['trainable_params']))
        self.writer.add_text('model/trainable_percentage', f"{model_info['trainable_percentage']:.2f}%")
        
        # Log to file
        self._log_to_file(f"Model Info - Total: {model_info['total_params']:,}, "
                         f"Trainable: {model_info['trainable_params']:,} "
                         f"({model_info['trainable_percentage']:.2f}%)")
    
    def log_stage_completion(self, stage: str, duration: float):
        """Log stage completion."""
        self.experiment_metadata[f'{stage}_completion_time'] = datetime.now().isoformat()
        self.experiment_metadata[f'{stage}_duration'] = duration
        
        self._log_to_file(f"Stage {stage} completed in {duration:.2f} seconds")
        self.writer.add_scalar(f'stages/{stage}_duration', duration, 0)
    
    def log_experiment_completion(self):
        """Log experiment completion."""
        total_duration = time.time() - self.start_time
        self.experiment_metadata['total_duration'] = total_duration
        self.experiment_metadata['end_time'] = datetime.now().isoformat()
        
        self._log_to_file(f"Experiment completed in {total_duration:.2f} seconds")
        self.writer.add_scalar('experiment/total_duration', total_duration, 0)
        
        # Save experiment metadata
        metadata_path = os.path.join(self.config.logging.log_dir, "experiment_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.experiment_metadata, f, indent=2)
        
        # Save training history
        history_path = os.path.join(self.config.logging.log_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save evaluation history
        eval_history_path = os.path.join(self.config.logging.log_dir, "evaluation_history.json")
        with open(eval_history_path, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)
    
    def create_visualizations(self):
        """Create visualization plots."""
        self._plot_training_curves()
        self._plot_evaluation_curves()
        self._plot_forgetting_analysis()
    
    def _plot_training_curves(self):
        """Plot training loss curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        # Plot losses
        for i, stage in enumerate(['stage1', 'stage2']):
            if self.training_history[stage]['losses']:
                axes[i, 0].plot(self.training_history[stage]['losses'])
                axes[i, 0].set_title(f'{stage} - Training Loss')
                axes[i, 0].set_xlabel('Step')
                axes[i, 0].set_ylabel('Loss')
                axes[i, 0].grid(True)
        
        # Plot learning rates
        for i, stage in enumerate(['stage1', 'stage2']):
            if self.training_history[stage]['learning_rates']:
                axes[i, 1].plot(self.training_history[stage]['learning_rates'])
                axes[i, 1].set_title(f'{stage} - Learning Rate')
                axes[i, 1].set_xlabel('Step')
                axes[i, 1].set_ylabel('Learning Rate')
                axes[i, 1].set_yscale('log')
                axes[i, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.logging.log_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_evaluation_curves(self):
        """Plot evaluation curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Evaluation Progress', fontsize=16)
        
        datasets = ['gsm8k', 'aqua_rat']
        stages = ['stage1', 'stage2']
        
        for i, stage in enumerate(stages):
            for j, dataset in enumerate(datasets):
                if self.evaluation_history[stage][dataset]:
                    steps = [entry['step'] for entry in self.evaluation_history[stage][dataset]]
                    accuracies = [entry['metrics'].get('accuracy', 0) for entry in self.evaluation_history[stage][dataset]]
                    
                    axes[i, j].plot(steps, accuracies, marker='o')
                    axes[i, j].set_title(f'{stage} - {dataset} Accuracy')
                    axes[i, j].set_xlabel('Step')
                    axes[i, j].set_ylabel('Accuracy')
                    axes[i, j].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.logging.log_dir, "evaluation_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_forgetting_analysis(self):
        """Plot catastrophic forgetting analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Catastrophic Forgetting Analysis', fontsize=16)
        
        # Plot performance on both datasets after each stage
        stages = ['stage1', 'stage2', 'final']
        datasets = ['gsm8k', 'aqua_rat']
        
        gsm8k_scores = []
        aqua_scores = []
        
        for stage in stages:
            if self.evaluation_history[stage]['gsm8k']:
                gsm8k_acc = self.evaluation_history[stage]['gsm8k'][-1]['metrics'].get('accuracy', 0)
                gsm8k_scores.append(gsm8k_acc)
            else:
                gsm8k_scores.append(0)
            
            if self.evaluation_history[stage]['aqua_rat']:
                aqua_acc = self.evaluation_history[stage]['aqua_rat'][-1]['metrics'].get('accuracy', 0)
                aqua_scores.append(aqua_acc)
            else:
                aqua_scores.append(0)
        
        x = range(len(stages))
        width = 0.35
        
        axes[0].bar([i - width/2 for i in x], gsm8k_scores, width, label='GSM8K', alpha=0.8)
        axes[0].bar([i + width/2 for i in x], aqua_scores, width, label='AQUA-RAT', alpha=0.8)
        axes[0].set_xlabel('Training Stage')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Performance After Each Stage')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(stages)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot forgetting rate
        forgetting_rates = []
        for i in range(1, len(stages)):
            prev_gsm8k = gsm8k_scores[i-1]
            curr_gsm8k = gsm8k_scores[i]
            prev_aqua = aqua_scores[i-1]
            curr_aqua = aqua_scores[i]
            
            forgetting_rate = abs((curr_gsm8k - prev_gsm8k) + (curr_aqua - prev_aqua)) / 2
            forgetting_rates.append(forgetting_rate)
        
        axes[1].plot(range(1, len(stages)), forgetting_rates, marker='o', linewidth=2)
        axes[1].set_xlabel('Stage Transition')
        axes[1].set_ylabel('Forgetting Rate')
        axes[1].set_title('Forgetting Rate Between Stages')
        axes[1].set_xticks(range(1, len(stages)))
        axes[1].set_xticklabels([f'{stages[i]} → {stages[i+1]}' for i in range(len(stages)-1)])
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.logging.log_dir, "forgetting_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _log_to_file(self, message: str):
        """Log message to file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def _config_to_dict(self, config) -> Dict[str, Any]:
        """Convert config object to dictionary."""
        return {
            'model': {
                'name': config.model.name,
                'use_lora': config.model.use_lora,
                'lora_rank': config.model.lora_rank,
                'lora_alpha': config.model.lora_alpha,
                'lora_dropout': config.model.lora_dropout,
                'target_modules': config.model.target_modules
            },
            'training': {
                stage: {
                    'dataset': stage_config.dataset,
                    'epochs': stage_config.epochs,
                    'batch_size': stage_config.batch_size,
                    'learning_rate': stage_config.learning_rate,
                    'weight_decay': stage_config.weight_decay
                }
                for stage, stage_config in config.training.items()
            },
            'optimizer': {
                'type': config.optimizer.type,
                'adamw': config.optimizer.adamw,
                'muon': config.optimizer.muon
            }
        }
    
    def close(self):
        """Close logger and TensorBoard writer."""
        self.writer.close()
        print(f"Logs saved to: {self.config.logging.log_dir}")
        print(f"TensorBoard logs saved to: {self.config.logging.tensorboard_dir}")
        print(f"To view TensorBoard: tensorboard --logdir {self.config.logging.tensorboard_dir}")

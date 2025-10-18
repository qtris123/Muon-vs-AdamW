import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import time
from typing import Dict, Any, Optional

from model_setup import setup_optimizer, setup_scheduler, save_model_checkpoint
from data_loader import create_data_loaders
from evaluation import evaluate_model_detailed

class Trainer:
    """Simplified training class for the two-stage SFT pipeline."""
    
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup TensorBoard writer
        self.writer = SummaryWriter(config.logging.tensorboard_dir)
        
        # Training state
        self.global_step = 0
        
    def train_stage(self, stage: str):
        """Train model on the specified stage."""
        print(f"\n{'='*50}")
        print(f"Starting Stage {stage.upper()}: {self.config.training[stage].dataset}")
        print(f"{'='*50}")
        
        # Setup data loaders
        train_loader, _ = create_data_loaders(self.config, self.tokenizer, stage)
        
        # Setup optimizer
        optimizer = setup_optimizer(self.model, self.config, stage)
        
        # Training configuration
        training_config = self.config.training[stage]
        
        # Training loop
        self.model.train()
        for epoch in range(training_config.epochs):
            print(f"\nEpoch {epoch + 1}/{training_config.epochs}")
            
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Training {stage}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                optimizer.zero_grad()
                self.global_step += 1
                
                epoch_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}"
                })
            
            # Log epoch metrics
            avg_epoch_loss = epoch_loss / len(train_loader)
            self.writer.add_scalar(f'{stage}/epoch_loss', avg_epoch_loss, epoch)
            
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Evaluate on both datasets after training
        print(f"\nEvaluating after Stage {stage}...")
        self.evaluate_both_datasets(stage)
        
        # Save checkpoint
        save_model_checkpoint(self.model, optimizer, None, 
                            training_config.epochs - 1, self.global_step, self.config, f"{stage}_final")
        
        return optimizer
    
    def evaluate_both_datasets(self, stage: str):
        """Evaluate on both datasets after each stage."""
        print(f"\nEvaluating on both datasets after {stage}...")
        
        # Evaluate on GSM8K
        print("Evaluating on GSM8K...")
        _, gsm8k_loader = create_data_loaders(self.config, self.tokenizer, "stage1")
        gsm8k_results = evaluate_model_detailed(self.model, self.tokenizer, gsm8k_loader, "gsm8k", 
                                              replay_samples=self.config.evaluation.replay_samples)
        
        # Evaluate on AQUA-RAT
        print("Evaluating on AQUA-RAT...")
        _, aqua_loader = create_data_loaders(self.config, self.tokenizer, "stage2")
        aqua_results = evaluate_model_detailed(self.model, self.tokenizer, aqua_loader, "aqua_rat",
                                             replay_samples=self.config.evaluation.replay_samples)
        
        # Log metrics
        for dataset, results in [("gsm8k", gsm8k_results), ("aqua_rat", aqua_results)]:
            for metric_name, value in results['metrics'].items():
                self.writer.add_scalar(f'{stage}/eval_{dataset}_{metric_name}', value, self.global_step)
        
        # Save sample records for replay analysis
        from evaluation import save_sample_records
        gsm8k_records_path = os.path.join(self.config.logging.log_dir, f"{stage}_gsm8k_samples.json")
        aqua_records_path = os.path.join(self.config.logging.log_dir, f"{stage}_aqua_samples.json")
        
        save_sample_records(gsm8k_results['sample_records'], gsm8k_records_path)
        save_sample_records(aqua_results['sample_records'], aqua_records_path)
        
        # Print results
        print(f"\nResults after {stage}:")
        print(f"GSM8K Accuracy: {gsm8k_results['metrics']['accuracy']:.4f}")
        print(f"AQUA-RAT Accuracy: {aqua_results['metrics']['accuracy']:.4f}")
        print(f"Sample records saved for replay analysis")
        
        return gsm8k_results, aqua_results
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()

def run_two_stage_training(config):
    """Run the complete two-stage training pipeline."""
    print("Initializing model and tokenizer...")
    from model_setup import setup_model_and_tokenizer, get_model_size_info
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Print model info
    get_model_size_info(model)
    
    # Initialize trainer
    trainer = Trainer(config, model, tokenizer)
    
    try:
        # Stage 1: Train on GSM8K
        trainer.train_stage("stage1")
        
        # Stage 2: Train on AQUA-RAT
        trainer.train_stage("stage2")
        
        print("\n" + "="*50)
        print("TWO-STAGE TRAINING COMPLETED!")
        print("="*50)
        
    finally:
        trainer.close()
    
    return model, tokenizer

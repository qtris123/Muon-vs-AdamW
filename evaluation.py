import torch
import torch.nn.functional as F
from tqdm import tqdm
import re
import json
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_loader import extract_answer_from_text, create_data_loaders

def evaluate_model_detailed(model, tokenizer, eval_loader, dataset_type: str, replay_samples: int = 100) -> Dict[str, Any]:
    """Evaluate model with detailed sample recording for replay analysis."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_losses = []
    sample_records = []  # Store detailed sample information
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc=f"Evaluating {dataset_type}")):
            # Move batch to device
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            all_losses.append(loss.item())
            
            # Generate predictions
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            # Generate responses
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Process each sample in the batch
            for i in range(len(generated_ids)):
                # Get the generated part (excluding input)
                generated_part = generated_ids[i][len(input_ids[i]):]
                prediction_text = tokenizer.decode(generated_part, skip_special_tokens=True)
                
                # Extract answer from prediction
                predicted_answer = extract_answer_from_text(prediction_text, dataset_type)
                all_predictions.append(predicted_answer)
                
                # Get ground truth answer
                full_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                true_answer = extract_answer_from_text(full_text, dataset_type)
                all_labels.append(true_answer)
                
                # Record detailed sample information for replay analysis
                if len(sample_records) < replay_samples:
                    # Calculate prompt length (original input)
                    prompt_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    prompt_length = len(tokenizer.encode(prompt_text))
                    
                    # Calculate generation length
                    generation_length = len(generated_part)
                    
                    sample_record = {
                        'sample_id': len(sample_records),
                        'prompt': prompt_text,
                        'response': prediction_text,
                        'predicted_answer': predicted_answer,
                        'true_answer': true_answer,
                        'prompt_length': prompt_length,
                        'generation_length': generation_length,
                        'correct': predicted_answer == true_answer,
                        'dataset_type': dataset_type
                    }
                    sample_records.append(sample_record)
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_labels, dataset_type)
    metrics['loss'] = np.mean(all_losses)
    
    return {
        'metrics': metrics,
        'sample_records': sample_records,
        'total_samples': len(all_predictions),
        'replay_samples': len(sample_records)
    }

def evaluate_model(model, tokenizer, eval_loader, dataset_type: str) -> Dict[str, float]:
    """Evaluate model on the given dataset."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_losses = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Evaluating {dataset_type}"):
            # Move batch to device
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            all_losses.append(loss.item())
            
            # Generate predictions
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            # Generate responses
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode predictions and labels
            for i in range(len(generated_ids)):
                # Get the generated part (excluding input)
                generated_part = generated_ids[i][len(input_ids[i]):]
                prediction_text = tokenizer.decode(generated_part, skip_special_tokens=True)
                
                # Extract answer from prediction
                predicted_answer = extract_answer_from_text(prediction_text, dataset_type)
                all_predictions.append(predicted_answer)
                
                # Get ground truth answer
                full_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                true_answer = extract_answer_from_text(full_text, dataset_type)
                all_labels.append(true_answer)
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_labels, dataset_type)
    metrics['loss'] = np.mean(all_losses)
    
    return metrics

def calculate_metrics(predictions: List[str], labels: List[str], dataset_type: str) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    metrics = {}
    
    if dataset_type == "gsm8k":
        # For GSM8K, we need to handle numerical answers
        processed_predictions = []
        processed_labels = []
        
        for pred, label in zip(predictions, labels):
            # Extract numbers from predictions and labels
            pred_num = extract_number(pred)
            label_num = extract_number(label)
            
            if pred_num is not None and label_num is not None:
                processed_predictions.append(str(pred_num))
                processed_labels.append(str(label_num))
        
        if processed_predictions and processed_labels:
            accuracy = accuracy_score(processed_labels, processed_predictions)
            metrics['accuracy'] = accuracy
            metrics['exact_match'] = accuracy  # Same as accuracy for numerical answers
        else:
            metrics['accuracy'] = 0.0
            metrics['exact_match'] = 0.0
    
    elif dataset_type == "aqua_rat":
        # For AQUA-RAT, we have multiple choice answers
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        metrics['accuracy'] = accuracy
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        metrics['exact_match'] = accuracy  # Same as accuracy for multiple choice
    
    # Calculate additional metrics
    metrics['total_samples'] = len(predictions)
    metrics['valid_predictions'] = len([p for p in predictions if p.strip()])
    
    return metrics

def extract_number(text: str) -> Optional[float]:
    """Extract the first number from text."""
    # Look for patterns like "72", "72.5", "-72", etc.
    patterns = [
        r'-?\d+\.?\d*',  # Basic number pattern
        r'-?\d+',        # Integer pattern
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                continue
    
    return None

def save_sample_records(sample_records: List[Dict], save_path: str):
    """Save detailed sample records for replay analysis."""
    with open(save_path, 'w') as f:
        json.dump(sample_records, f, indent=2)
    print(f"Sample records saved to {save_path}")

def analyze_replay_samples(sample_records: List[Dict]) -> Dict[str, Any]:
    """Analyze replay samples for insights."""
    if not sample_records:
        return {}
    
    analysis = {
        'total_samples': len(sample_records),
        'correct_samples': len([s for s in sample_records if s['correct']]),
        'accuracy': len([s for s in sample_records if s['correct']]) / len(sample_records),
        'avg_prompt_length': np.mean([s['prompt_length'] for s in sample_records]),
        'avg_generation_length': np.mean([s['generation_length'] for s in sample_records]),
        'prompt_length_range': {
            'min': min([s['prompt_length'] for s in sample_records]),
            'max': max([s['prompt_length'] for s in sample_records])
        },
        'generation_length_range': {
            'min': min([s['generation_length'] for s in sample_records]),
            'max': max([s['generation_length'] for s in sample_records])
        }
    }
    
    return analysis

def evaluate_catastrophic_forgetting(model, tokenizer, config) -> Dict[str, Dict[str, float]]:
    """Evaluate catastrophic forgetting by testing on both datasets."""
    print("\nEvaluating catastrophic forgetting...")
    
    results = {}
    
    # Evaluate on GSM8K
    print("Evaluating on GSM8K...")
    _, gsm8k_loader = create_data_loaders(config, tokenizer, "stage1")
    gsm8k_metrics = evaluate_model(model, tokenizer, gsm8k_loader, "gsm8k")
    results['gsm8k'] = gsm8k_metrics
    
    # Evaluate on AQUA-RAT
    print("Evaluating on AQUA-RAT...")
    _, aqua_loader = create_data_loaders(config, tokenizer, "stage2")
    aqua_metrics = evaluate_model(model, tokenizer, aqua_loader, "aqua_rat")
    results['aqua_rat'] = aqua_metrics
    
    # Calculate forgetting metrics
    forgetting_metrics = calculate_forgetting_metrics(results)
    results['forgetting'] = forgetting_metrics
    
    return results

def calculate_forgetting_metrics(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Calculate catastrophic forgetting metrics."""
    forgetting_metrics = {}
    
    # Get accuracies
    gsm8k_acc = results['gsm8k'].get('accuracy', 0)
    aqua_acc = results['aqua_rat'].get('accuracy', 0)
    
    # Calculate forgetting rate (assuming we know the original performance)
    # For this example, we'll use a simple metric
    forgetting_metrics['gsm8k_aqua_performance_gap'] = abs(gsm8k_acc - aqua_acc)
    forgetting_metrics['average_performance'] = (gsm8k_acc + aqua_acc) / 2
    forgetting_metrics['performance_balance'] = 1 - forgetting_metrics['gsm8k_aqua_performance_gap']
    
    return forgetting_metrics

def save_evaluation_results(results: Dict[str, Dict[str, float]], save_path: str):
    """Save evaluation results to JSON file."""
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved to {save_path}")

def print_evaluation_summary(results: Dict[str, Dict[str, float]]):
    """Print a summary of evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for dataset, metrics in results.items():
        if dataset == 'forgetting':
            continue
            
        print(f"\n{dataset.upper()} Dataset:")
        print("-" * 30)
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    if 'forgetting' in results:
        print(f"\nFORGETTING METRICS:")
        print("-" * 30)
        for metric, value in results['forgetting'].items():
            print(f"  {metric}: {value:.4f}")
    
    print("="*60)

def compare_optimizer_results(adamw_results: Dict, muon_results: Dict) -> Dict[str, Any]:
    """Compare results between AdamW and Muon optimizers."""
    comparison = {}
    
    datasets = ['gsm8k', 'aqua_rat']
    
    for dataset in datasets:
        if dataset in adamw_results and dataset in muon_results:
            comparison[dataset] = {}
            
            # Compare accuracy
            adamw_acc = adamw_results[dataset].get('accuracy', 0)
            muon_acc = muon_results[dataset].get('accuracy', 0)
            
            comparison[dataset]['accuracy_difference'] = muon_acc - adamw_acc
            comparison[dataset]['adamw_accuracy'] = adamw_acc
            comparison[dataset]['muon_accuracy'] = muon_acc
            
            # Compare forgetting
            if 'forgetting' in adamw_results and 'forgetting' in muon_results:
                adamw_forgetting = adamw_results['forgetting'].get('performance_balance', 0)
                muon_forgetting = muon_results['forgetting'].get('performance_balance', 0)
                
                comparison[dataset]['forgetting_improvement'] = muon_forgetting - adamw_forgetting
    
    return comparison
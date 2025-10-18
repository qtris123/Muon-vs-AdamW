import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Any, Optional
import re
import random

class GSM8KDataset(Dataset):
    """Dataset for GSM8K math reasoning problems."""
    
    def __init__(self, data: List[Dict[str, str]], tokenizer, max_length: int = 1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        
        # Format as instruction-following prompt
        prompt = f"Solve this math problem step by step.\n\nProblem: {question}\n\nSolution:"
        full_text = f"{prompt} {answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create labels (same as input_ids for causal LM)
        labels = encoding['input_ids'].clone()
        
        # Mask the prompt part in labels (only compute loss on the answer)
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        prompt_length = prompt_encoding['input_ids'].shape[1]
        
        # Set prompt tokens to -100 (ignore in loss)
        labels[0, :prompt_length] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }

class AQUARATDataset(Dataset):
    """Dataset for AQUA-RAT multiple choice math problems."""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        options = item['options']
        correct = item['correct']
        
        # Format options
        option_text = ""
        for i, option in enumerate(options):
            option_text += f"{chr(65+i)}. {option}\n"
        
        # Format as instruction-following prompt
        prompt = f"Solve this multiple choice math problem step by step.\n\nProblem: {question}\n\nOptions:\n{option_text}\nAnswer:"
        full_text = f"{prompt} {correct}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create labels
        labels = encoding['input_ids'].clone()
        
        # Mask the prompt part in labels
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        prompt_length = prompt_encoding['input_ids'].shape[1]
        
        # Set prompt tokens to -100 (ignore in loss)
        labels[0, :prompt_length] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }

def load_gsm8k_data(split: str = "train") -> List[Dict[str, str]]:
    """Load GSM8K dataset."""
    # For this example, we'll create a simple data structure
    # In practice, you would load from the actual GSM8K dataset
    sample_data = [
        {
            "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "answer": "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n#### 72"
        },
        {
            "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
            "answer": "Betty has 100/2 = $50.\nHer parents gave her $15.\nHer grandparents gave her 15*2 = $30.\nBetty now has 50+15+30 = $95.\nBetty needs 100-95 = $5 more.\n#### 5"
        }
    ]
    
    if split == "train":
        return sample_data * 10  # Repeat for more training data
    else:
        return sample_data

def load_aqua_rat_data(split: str = "train") -> List[Dict[str, Any]]:
    """Load AQUA-RAT dataset."""
    # Sample AQUA-RAT data
    sample_data = [
        {
            "question": "A train running at the speed of 60 km/hr crosses a pole in 9 seconds. What is the length of the train?",
            "options": ["120 metres", "180 metres", "324 metres", "150 metres", "None of these"],
            "correct": "D"
        },
        {
            "question": "A sum of money at simple interest amounts to Rs. 815 in 3 years and to Rs. 854 in 4 years. The sum is:",
            "options": ["Rs. 650", "Rs. 690", "Rs. 698", "Rs. 700", "Rs. 720"],
            "correct": "C"
        }
    ]
    
    if split == "train":
        return sample_data * 10  # Repeat for more training data
    else:
        return sample_data

def create_data_loaders(config, tokenizer, stage: str):
    """Create data loaders for the specified stage."""
    training_config = config.training[stage]
    
    if training_config.dataset == "gsm8k":
        train_data = load_gsm8k_data("train")
        eval_data = load_gsm8k_data("test")
        
        train_dataset = GSM8KDataset(train_data, tokenizer, training_config.max_length)
        eval_dataset = GSM8KDataset(eval_data, tokenizer, training_config.max_length)
        
    elif training_config.dataset == "aqua_rat":
        train_data = load_aqua_rat_data("train")
        eval_data = load_aqua_rat_data("test")
        
        train_dataset = AQUARATDataset(train_data, tokenizer, training_config.max_length)
        eval_dataset = AQUARATDataset(eval_data, tokenizer, training_config.max_length)
    
    else:
        raise ValueError(f"Unknown dataset: {training_config.dataset}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, eval_loader

def extract_answer_from_text(text: str, dataset_type: str) -> str:
    """Extract the final answer from model output."""
    if dataset_type == "gsm8k":
        # Look for pattern like "#### 72"
        match = re.search(r'####\s*(\S+)', text)
        if match:
            return match.group(1)
    elif dataset_type == "aqua_rat":
        # Look for single letter answers (A, B, C, D, E)
        match = re.search(r'\b([A-E])\b', text)
        if match:
            return match.group(1)
    
    # Fallback: return the last word/number
    words = text.strip().split()
    if words:
        return words[-1]
    return ""

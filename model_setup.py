import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
import torch.optim as optim
from typing import Optional, Dict, Any
import os

def setup_model_and_tokenizer(config):
    """Initialize model and tokenizer with LoRA support."""
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Setup quantization config for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Setup LoRA if enabled
    if config.model.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=config.model.target_modules,
            bias="none"
        )
        
        model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"LoRA adapters added to model. Trainable parameters: {trainable_params:,}")
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    return model, tokenizer

def setup_optimizer(model, config, stage: str):
    """Setup optimizer based on configuration."""
    training_config = config.training[stage]
    optimizer_type = config.optimizer.type
    
    # Get trainable parameters
    if config.model.use_lora:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable_params = model.parameters()
    
    if optimizer_type == "adamw":
        optimizer = AdamW(
            trainable_params,
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            betas=(config.optimizer.adamw["beta1"], config.optimizer.adamw["beta2"]),
            eps=config.optimizer.adamw["eps"]
        )
        
    elif optimizer_type == "muon":
        # Import Muon optimizer
        try:
            from torch.optim import Muon
            optimizer = Muon(
                trainable_params,
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
                betas=(config.optimizer.muon["beta1"], config.optimizer.muon["beta2"]),
                eps=config.optimizer.muon["eps"],
                mu=config.optimizer.muon["mu"]
            )
        except ImportError:
            print("Muon optimizer not available. Falling back to AdamW.")
            optimizer = AdamW(
                trainable_params,
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
                betas=(config.optimizer.adamw["beta1"], config.optimizer.adamw["beta2"]),
                eps=config.optimizer.adamw["eps"]
            )
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer

def setup_scheduler(optimizer, config, stage: str, num_training_steps: int):
    """Setup learning rate scheduler."""
    training_config = config.training[stage]
    
    from transformers import get_linear_schedule_with_warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return scheduler

def save_model_checkpoint(model, optimizer, scheduler, epoch, step, config, stage: str):
    """Save model checkpoint."""
    save_dir = os.path.join(config.logging.save_dir, f"{stage}_checkpoint_epoch_{epoch}_step_{step}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    if config.model.use_lora:
        model.save_pretrained(save_dir)
    else:
        model.save_pretrained(save_dir)
    
    # Save optimizer and scheduler state
    training_state = {
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'step': step
    }
    torch.save(training_state, os.path.join(save_dir, 'training_state.pt'))

    print(f"Checkpoint saved to {save_dir}")


def load_model_checkpoint(model, optimizer, scheduler, checkpoint_path: str):
    """Load model checkpoint."""
    # Load model
    if os.path.exists(os.path.join(checkpoint_path, 'adapter_config.json')):
        # LoRA model
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path)
    else:
        # Full model
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'pytorch_model.bin')))

    # Load training state
    training_state = torch.load(os.path.join(checkpoint_path, 'training_state.pt'))
    optimizer.load_state_dict(training_state['optimizer_state_dict'])
    if scheduler is not None and training_state.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(training_state['scheduler_state_dict'])

    return model, optimizer, scheduler, training_state['epoch'], training_state['step']

def get_model_size_info(model):
    """Get information about model size and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'trainable_percentage': 100 * trainable_params / total_params
    }

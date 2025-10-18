import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    use_lora: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list

@dataclass
class TrainingConfig:
    dataset: str
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_length: int

@dataclass
class OptimizerConfig:
    type: str
    adamw: Dict[str, Any]
    muon: Dict[str, Any]

@dataclass
class EvaluationConfig:
    batch_size: int
    max_new_tokens: int
    temperature: float
    do_sample: bool
    pad_token_id: Optional[int]
    replay_samples: int

@dataclass
class LoggingConfig:
    log_dir: str
    tensorboard_dir: str
    save_dir: str
    log_interval: int
    eval_interval: int
    save_interval: int

@dataclass
class Config:
    model: ModelConfig
    training: Dict[str, TrainingConfig]
    optimizer: OptimizerConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig
    device: str
    mixed_precision: bool

def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create nested config objects
    model_config = ModelConfig(**config_dict['model'])
    
    training_config = {
        stage: TrainingConfig(**stage_config) 
        for stage, stage_config in config_dict['training'].items()
    }
    
    optimizer_config = OptimizerConfig(**config_dict['optimizer'])
    evaluation_config = EvaluationConfig(**config_dict['evaluation'])
    logging_config = LoggingConfig(**config_dict['logging'])
    
    return Config(
        model=model_config,
        training=training_config,
        optimizer=optimizer_config,
        evaluation=evaluation_config,
        logging=logging_config,
        device=config_dict['device'],
        mixed_precision=config_dict['mixed_precision']
    )

def save_config(config: Config, save_path: str):
    """Save configuration to YAML file."""
    config_dict = {
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
                'gradient_accumulation_steps': stage_config.gradient_accumulation_steps,
                'learning_rate': stage_config.learning_rate,
                'weight_decay': stage_config.weight_decay,
                'warmup_steps': stage_config.warmup_steps,
                'max_length': stage_config.max_length
            }
            for stage, stage_config in config.training.items()
        },
        'optimizer': {
            'type': config.optimizer.type,
            'adamw': config.optimizer.adamw,
            'muon': config.optimizer.muon
        },
        'evaluation': {
            'batch_size': config.evaluation.batch_size,
            'max_new_tokens': config.evaluation.max_new_tokens,
            'temperature': config.evaluation.temperature,
            'do_sample': config.evaluation.do_sample,
            'pad_token_id': config.evaluation.pad_token_id,
            'replay_samples': config.evaluation.replay_samples
        },
        'logging': {
            'log_dir': config.logging.log_dir,
            'tensorboard_dir': config.logging.tensorboard_dir,
            'save_dir': config.logging.save_dir,
            'log_interval': config.logging.log_interval,
            'eval_interval': config.logging.eval_interval,
            'save_interval': config.logging.save_interval
        },
        'device': config.device,
        'mixed_precision': config.mixed_precision
    }
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)

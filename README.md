# Catastrophic Forgetting Study

This repository provides a clean, reproducible framework to study catastrophic forgetting when switching tasks during supervised fine-tuning (SFT).

## Overview

We implement a two-stage SFT pipeline on math datasets:
- **Stage 1**: Train on GSM8K (grade school math problems)
- **Stage 2**: Train on AQUA-RAT (multiple choice math problems)

After each stage, we save weights and evaluate on the test sets of both datasets to quantify forgetting. We compare two optimizers: **AdamW** and **Muon**.

## Features

- 🔄 **Two-stage training pipeline** with configurable stages
- 🧠 **LoRA adapters** for efficient fine-tuning (configurable)
- 📊 **Comprehensive evaluation** on both datasets after each stage
- 📈 **TensorBoard logging** for training and evaluation metrics
- 🔍 **Catastrophic forgetting analysis** with visualizations
- ⚡ **Mixed precision training** for memory efficiency
- 🎯 **Optimizer comparison** between AdamW and Muon
- 💾 **Checkpoint management** with resume capabilities

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd catastrophic-forgetting-study
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install Muon optimizer if available:
```bash
# Muon optimizer may need to be installed separately
# Check PyTorch documentation for latest installation instructions
```

## Configuration

The project uses a YAML configuration file (`config.yaml`) for all settings:

### Model Configuration
```yaml
model:
  name: "meta-llama/Llama-3.2-1B"  # Default accessible model
  use_lora: true                   # Enable LoRA adapters
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### Training Configuration
```yaml
training:
  stage1:
    dataset: "gsm8k"
    epochs: 3
    batch_size: 4
    learning_rate: 2e-4
    weight_decay: 0.01
    max_length: 1024
  
  stage2:
    dataset: "aqua_rat"
    epochs: 3
    batch_size: 4
    learning_rate: 2e-4
    weight_decay: 0.01
    max_length: 1024
```

### Optimizer Configuration
```yaml
optimizer:
  type: "adamw"  # Options: "adamw", "muon"
  adamw:
    beta1: 0.9
    beta2: 0.999
    eps: 1e-8
  muon:
    beta1: 0.9
    beta2: 0.999
    eps: 1e-8
    mu: 0.01  # Muon-specific parameter
```

## Usage

### Basic Usage

Run a single experiment with the default configuration:

```bash
python main.py
```

### Advanced Usage

#### Compare Optimizers
Run experiments with both AdamW and Muon optimizers:

```bash
python main.py --compare-optimizers
```

#### Custom Configuration
Use a custom configuration file:

```bash
python main.py --config custom_config.yaml
```

#### Override Optimizer
Override the optimizer type from command line:

```bash
python main.py --optimizer muon
```

#### Resume Training
Resume training from a checkpoint:

```bash
python main.py --resume-stage1 ./checkpoints/stage1_checkpoint_epoch_2_step_500
python main.py --resume-stage2 ./checkpoints/stage2_checkpoint_epoch_1_step_200
```

#### Evaluation Only
Skip training and only run evaluation:

```bash
python main.py --skip-training
```

#### Custom Output Directory
Specify a custom output directory:

```bash
python main.py --output-dir ./experiments/run_001
```

## Project Structure

```
├── config.yaml          # Main configuration file
├── config.py            # Configuration management
├── data_loader.py       # Dataset loading and preprocessing
├── model_setup.py       # Model initialization and LoRA setup
├── training.py          # Training pipeline implementation
├── evaluation.py        # Evaluation functions and metrics
├── logging.py           # Logging and visualization utilities
├── main.py              # Main execution script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Output Structure

After running an experiment, the following structure is created:

```
output_directory/
├── logs/
│   ├── training.log              # Training logs
│   ├── experiment_metadata.json  # Experiment metadata
│   ├── training_history.json     # Training history
│   ├── evaluation_history.json    # Evaluation history
│   ├── final_results.json        # Final evaluation results
│   ├── training_curves.png        # Training visualization
│   ├── evaluation_curves.png      # Evaluation visualization
│   └── forgetting_analysis.png    # Forgetting analysis plots
├── tensorboard_logs/              # TensorBoard logs
└── checkpoints/                   # Model checkpoints
    ├── stage1_checkpoint_epoch_X_step_Y/
    └── stage2_checkpoint_epoch_X_step_Y/
```

## Monitoring Training

### TensorBoard
View training progress with TensorBoard:

```bash
tensorboard --logdir ./tensorboard_logs
```

### Log Files
Monitor training progress through log files:

```bash
tail -f ./logs/training.log
```

## Evaluation Metrics

The framework evaluates models using several metrics:

### GSM8K Dataset
- **Accuracy**: Exact match on numerical answers
- **Loss**: Cross-entropy loss during training

### AQUA-RAT Dataset
- **Accuracy**: Multiple choice accuracy
- **Precision, Recall, F1**: Weighted averages across classes
- **Loss**: Cross-entropy loss during training

### Catastrophic Forgetting Metrics
- **Performance Gap**: Difference in accuracy between datasets
- **Forgetting Rate**: Rate of performance degradation
- **Performance Balance**: Measure of balanced performance

## Model Notes

- **Default Model**: `meta-llama/Llama-3.2-1B` (accessible placeholder)
- **Llama-3.1-1B**: Set `model_name` in config if you have access
- **LoRA**: Enabled by default for efficient fine-tuning
- **Quantization**: 4-bit quantization for memory efficiency

## Hardware Requirements

- **GPU**: Recommended (CUDA-compatible)
- **Memory**: 8GB+ VRAM (with LoRA and quantization)
- **Storage**: 10GB+ for model weights and checkpoints

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Muon Optimizer Not Found**: Falls back to AdamW automatically
3. **Model Access**: Ensure you have access to the specified model on Hugging Face

### Performance Tips

1. Use LoRA adapters for memory efficiency
2. Enable mixed precision training
3. Use gradient accumulation for larger effective batch sizes
4. Monitor GPU memory usage with `nvidia-smi`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{catastrophic_forgetting_study,
  title={Catastrophic Forgetting Study Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/catastrophic-forgetting-study}
}
```

## Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Transformers](https://huggingface.co/transformers/) for model implementations
- [PEFT](https://github.com/huggingface/peft) for LoRA implementation
- [GSM8K](https://github.com/openai/grade-school-math) dataset
- [AQUA-RAT](https://github.com/deepmind/AQuA) dataset

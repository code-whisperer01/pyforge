# GGUF Model Training Guide

## Overview

This training system allows you to train your `pyforge-model` using GGUF models from the `models/` directory. The scripts automatically discover available models and manage the training pipeline.

## Features

- **Automatic Model Discovery**: Finds all GGUF models in the `models/` directory
- **Flexible Training**: Train with any discovered GGUF model or use default GPT2
- **Dataset Integration**: Automatically loads and tokenizes the python-coding-dataset
- **Progress Tracking**: Detailed logging of the training process
- **Model Persistence**: Saves trained models and training metadata

## Quick Start

### 1. List Available Models

```bash
python3 train_with_gguf.py --discover-models
```

This will show all GGUF models found in the `models/` directory with their indices.

### 2. Train with Default Settings

```bash
# Use default GPT2 model
python3 train_with_gguf.py

# Use first discovered GGUF model (index 0)
python3 train_with_gguf.py --model-index 0

# Use specific GGUF model (e.g., index 2)
python3 train_with_gguf.py --model-index 2
```

### 3. Custom Training Parameters

```bash
python3 train_with_gguf.py \
  --model-index 0 \
  --batch-size 4 \
  --learning-rate 2e-4 \
  --epochs 5
```

## Available Options

### `train_with_gguf.py`

```
--model-index INDEX           Index of GGUF model to use (default: None)
--batch-size SIZE             Training batch size (default: 2)
--learning-rate LR            Learning rate (default: 3e-4)
--epochs EPOCHS               Number of training epochs (default: 3)
--discover-models             List available GGUF models and exit
```

## Utility Scripts

### `gguf_utils.py` - GGUF Model Management

```bash
# List all GGUF models with information
python3 gguf_utils.py --list

# Save a training configuration template
python3 gguf_utils.py --save-config

# Specify custom models directory
python3 gguf_utils.py --list --models-dir ./custom_models/
```

## Output

After training completes:

- **Model Files**: Saved to `./pyforge-model/`
  - `model.safetensors` - Trained model weights
  - `config.json` - Model configuration
  - `generation_config.json` - Generation settings
  - `tokenizer.json` - Tokenizer configuration
  - `training_info.json` - Training metadata

- **Checkpoints**: Intermediate checkpoints saved every 200 steps

## Training Pipeline

1. **Model Discovery**: Scans `./models/` for GGUF files
2. **Model Loading**: Loads selected GGUF model or creates default
3. **Tokenizer Setup**: Initializes tokenizer (GPT2 if not found)
4. **Dataset Preparation**: 
   - Loads python-coding-dataset
   - Formats examples with instructions/outputs
   - Tokenizes with truncation and padding
5. **Training**: 
   - Uses HuggingFace Trainer
   - Logs progress every 50 steps
   - Saves checkpoints every 200 steps
6. **Finalization**: 
   - Saves trained model
   - Saves tokenizer
   - Records training info

## Resuming Training

The system automatically detects if models/tokenizers already exist and will resume training from the saved state.

```bash
python3 train_with_gguf.py --model-index 0 --epochs 10
```

## Using the Trained Model

```python
from transformers import GPT2LMHeadModel, AutoTokenizer

model = GPT2LMHeadModel.from_pretrained("./pyforge-model")
tokenizer = AutoTokenizer.from_pretrained("./pyforge-model")

prompt = "Write a Python function that"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Hardware Requirements

- **GPU Recommended**: Training is much faster with CUDA-capable GPU
- **VRAM**: Minimum 8GB for batch size 2
- **Storage**: ~2-5GB for model checkpoints

## Troubleshooting

### Models not found
```bash
# Verify models directory exists
ls -la ./models/

# Use gguf_utils to list models
python3 gguf_utils.py --list
```

### Out of Memory
```bash
# Reduce batch size
python3 train_with_gguf.py --batch-size 1

# Reduce epochs
python3 train_with_gguf.py --epochs 1
```

### Dataset loading issues
- Requires internet connection (first run downloads dataset)
- Check disk space for dataset (~1-2GB)
- Verify transformers and datasets packages are installed

## Advanced Usage

### Create Custom Training Configuration

```python
from gguf_utils import TrainingConfigBuilder

config = TrainingConfigBuilder.create_config(
    model_path="./models/sha256-xxx",
    batch_size=4,
    learning_rate=2e-4,
    epochs=5,
    max_length=512,
    warmup_steps=1000
)

TrainingConfigBuilder.save_config(config, "./my_config.json")
```

### Programmatic Usage

```python
from train_with_gguf import GGUFTrainer, GGUFModelDiscovery

# Discover models
discovery = GGUFModelDiscovery()
models = discovery.get_models()

# Train with first model
if models:
    trainer = GGUFTrainer(model_path=models[0], epochs=3)
    trainer.train()
```

## Model Architecture

- **Base Model**: GPT2-based architecture
- **Vocab Size**: 50,257 tokens
- **Max Sequence Length**: 512 tokens
- **Embedding Dimension**: 768
- **Number of Layers**: 8
- **Number of Attention Heads**: 8

## Dependencies

- transformers >= 4.0.0
- datasets
- torch
- accelerate
- tokenizers

Install with: `pip install -r requirements.txt`

## Next Steps

1. List available models: `python3 train_with_gguf.py --discover-models`
2. Choose a model and start training
3. Monitor training progress in console output
4. Use trained model with `generate.py` or your own scripts

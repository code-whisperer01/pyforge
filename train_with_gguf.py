#!/usr/bin/env python3
"""
GGUF Model Training Script
Trains pyforge-model using GGUF models from the models directory
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict
import logging

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODELS_DIR = "./models"
OUTPUT_DIR = "./pyforge-model"
GGUF_SUFFIX = ".gguf"


class GGUFModelDiscovery:
    """Discovers and manages GGUF models in the models directory"""
    
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.gguf_models = self._discover_models()
    
    def _discover_models(self) -> List[Path]:
        """Find all GGUF models in the models directory"""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return []
        
        # Look for GGUF files by extension or by hash pattern (as per your structure)
        gguf_models = []
        
        for item in self.models_dir.iterdir():
            if item.is_file():
                # Check if it's a GGUF file
                if item.suffix == GGUF_SUFFIX or (item.stem.startswith('sha256-')):
                    gguf_models.append(item)
                    logger.info(f"Found GGUF model: {item.name}")
        
        return sorted(gguf_models)
    
    def get_models(self) -> List[Path]:
        """Return list of discovered GGUF models"""
        return self.gguf_models
    
    def get_model_info(self) -> Dict[str, str]:
        """Get info about discovered models"""
        return {
            f"model_{i}": str(model.name)
            for i, model in enumerate(self.gguf_models)
        }


class GGUFTrainer:
    """Handles training with GGUF models"""
    
    def __init__(self, 
                 model_path: Optional[Path] = None,
                 output_dir: str = OUTPUT_DIR,
                 batch_size: int = 2,
                 learning_rate: float = 3e-4,
                 epochs: int = 3):
        """
        Initialize GGUF Trainer
        
        Args:
            model_path: Path to GGUF model (if None, creates new model)
            output_dir: Directory to save trained model
            batch_size: Training batch size
            learning_rate: Learning rate for training
            epochs: Number of training epochs
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.tokenizer = None
    
    def load_or_create_model(self):
        """Load GGUF model or create default GPT2 model"""
        logger.info("Loading model...")
        
        if self.model_path and self.model_path.exists():
            try:
                # Try to load as HuggingFace model first
                logger.info(f"Attempting to load model from: {self.model_path}")
                self.model = GPT2LMHeadModel.from_pretrained(str(self.model_path))
            except Exception as e:
                logger.warning(f"Could not load as HF model: {e}")
                logger.info("Using default GPT2 model for training")
                self._create_default_model()
        else:
            logger.info("Creating default GPT2 model")
            self._create_default_model()
        
        self.model = self.model.to(self.device)
        logger.info(f"Model loaded with {self.model.num_parameters():,} parameters")
    
    def _create_default_model(self):
        """Create a default GPT2 model"""
        from transformers import GPT2Config
        config = GPT2Config(
            vocab_size=50257,
            n_positions=512,
            n_embd=768,
            n_layer=8,
            n_head=8
        )
        self.model = GPT2LMHeadModel(config)
    
    def load_tokenizer(self):
        """Load or create tokenizer"""
        logger.info("Loading tokenizer...")
        try:
            # Try to load from output directory first (for resuming training)
            self.tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        except:
            # Use default GPT2 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_dataset(self, dataset_name: str = "Hoglet-33/python-coding-dataset"):
        """Prepare training dataset"""
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)
        
        def format_example(example):
            """Format dataset examples"""
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("output", "")
            
            if input_text.strip():
                return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Output:
{output}"""
            else:
                return f"""### Instruction:
{instruction}

### Output:
{output}"""
        
        def tokenize(example):
            """Tokenize examples"""
            text = format_example(example)
            tokens = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=512
            )
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens
        
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize,
            batched=False,
            remove_columns=dataset["train"].column_names
        )
        
        return tokenized_dataset
    
    def train(self, dataset_name: str = "Hoglet-33/python-coding-dataset"):
        """Execute training pipeline"""
        logger.info("=" * 60)
        logger.info("Starting GGUF Model Training Pipeline")
        logger.info("=" * 60)
        
        # Load model and tokenizer
        self.load_or_create_model()
        self.load_tokenizer()
        
        # Prepare dataset
        tokenized_dataset = self.prepare_dataset(dataset_name)
        
        # Setup training arguments
        logger.info("Configuring training arguments...")
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            logging_steps=50,
            save_steps=200,
            save_total_limit=3,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            report_to="none",
            logging_first_step=True,
            disable_tqdm=False
        )
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save model and tokenizer
        logger.info("Saving trained model and tokenizer...")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training info
        self._save_training_info()
        
        logger.info("=" * 60)
        logger.info(f"Training complete! Model saved to: {self.output_dir}")
        logger.info("=" * 60)
    
    def _save_training_info(self):
        """Save training information"""
        info = {
            "model_source": str(self.model_path) if self.model_path else "default_gpt2",
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "device": str(self.device),
            "model_parameters": self.model.num_parameters()
        }
        
        with open(Path(self.output_dir) / "training_info.json", "w") as f:
            json.dump(info, f, indent=2)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train pyforge-model using GGUF models"
    )
    parser.add_argument(
        "--model-index",
        type=int,
        default=None,
        help="Index of GGUF model to use (if None, uses default GPT2)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--discover-models",
        action="store_true",
        help="Discover and list available GGUF models"
    )
    
    args = parser.parse_args()
    
    # Discover available models
    discovery = GGUFModelDiscovery()
    
    if args.discover_models:
        logger.info("Available GGUF models:")
        for i, model in enumerate(discovery.get_models()):
            logger.info(f"  [{i}] {model.name}")
        return
    
    # Select model
    model_path = None
    models = discovery.get_models()
    
    if models and args.model_index is not None:
        if 0 <= args.model_index < len(models):
            model_path = models[args.model_index]
            logger.info(f"Using GGUF model: {model_path.name}")
        else:
            logger.warning(f"Invalid model index. Using default GPT2 model")
    elif models:
        logger.info(f"Found {len(models)} GGUF model(s). Using first one.")
        model_path = models[0]
    
    # Train
    trainer = GGUFTrainer(
        model_path=model_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs
    )
    trainer.train()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
GGUF Model Utilities
Utilities for working with GGUF models and converting them to trainable formats
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class GGUFModelInfo:
    """Extract and manage GGUF model information"""
    
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.info = self._extract_info()
    
    def _extract_info(self) -> Dict:
        """Extract metadata from GGUF model"""
        info = {
            "name": self.model_path.name,
            "path": str(self.model_path),
            "size_mb": self.model_path.stat().st_size / (1024 * 1024),
        }
        
        # Try to read GGUF metadata
        try:
            import struct
            with open(self.model_path, 'rb') as f:
                # GGUF format starts with magic number
                magic = f.read(4)
                if magic == b'GGUF':
                    version = struct.unpack('<I', f.read(4))[0]
                    info['gguf_version'] = version
                    info['valid_gguf'] = True
        except Exception as e:
            logger.debug(f"Could not read GGUF metadata: {e}")
            info['valid_gguf'] = False
        
        return info
    
    def get_info(self) -> Dict:
        """Get model information"""
        return self.info
    
    def __repr__(self):
        return f"GGUFModelInfo({self.model_path.name}, {self.info['size_mb']:.2f}MB)"


class GGUFModelManager:
    """Manage GGUF models in the models directory"""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models: List[GGUFModelInfo] = []
        self._load_models()
    
    def _load_models(self):
        """Load all GGUF models from directory"""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return
        
        for item in self.models_dir.iterdir():
            if item.is_file():
                # Accept GGUF files and hash-named files (like your structure)
                if item.suffix == '.gguf' or item.stem.startswith('sha256-'):
                    try:
                        model_info = GGUFModelInfo(item)
                        self.models.append(model_info)
                    except Exception as e:
                        logger.warning(f"Could not load model {item.name}: {e}")
    
    def list_models(self) -> List[Dict]:
        """List all available models with info"""
        return [model.get_info() for model in self.models]
    
    def get_model_by_index(self, index: int) -> Optional[Path]:
        """Get model path by index"""
        if 0 <= index < len(self.models):
            return self.models[index].model_path
        return None
    
    def print_models(self):
        """Print formatted list of models"""
        if not self.models:
            print("No GGUF models found in models directory")
            return
        
        print(f"\nFound {len(self.models)} GGUF model(s):\n")
        print(f"{'Index':<8} {'Model Name':<50} {'Size (MB)':<12} {'Valid GGUF':<12}")
        print("-" * 82)
        
        for i, model in enumerate(self.models):
            info = model.get_info()
            name = info['name'][:47] + "..." if len(info['name']) > 50 else info['name']
            size = f"{info['size_mb']:.2f}"
            valid = "✓" if info.get('valid_gguf', False) else "✗"
            print(f"{i:<8} {name:<50} {size:<12} {valid:<12}")
        print()


class TrainingConfigBuilder:
    """Build training configurations for GGUF models"""
    
    @staticmethod
    def create_config(
        model_path: Optional[Path] = None,
        batch_size: int = 2,
        learning_rate: float = 3e-4,
        epochs: int = 3,
        max_length: int = 512,
        warmup_steps: int = 500
    ) -> Dict:
        """Create training configuration"""
        return {
            "model_path": str(model_path) if model_path else None,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "max_length": max_length,
            "warmup_steps": warmup_steps,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 1,
            "eval_steps": 200,
            "save_steps": 200,
            "device": "cuda",  # Will auto-detect in trainer
        }
    
    @staticmethod
    def save_config(config: Dict, output_path: Path):
        """Save configuration to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Config saved to: {output_path}")
    
    @staticmethod
    def load_config(config_path: Path) -> Dict:
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            return json.load(f)


def setup_logging(log_level: str = "INFO"):
    """Configure logging"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GGUF Model Utilities")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all GGUF models in models directory"
    )
    parser.add_argument(
        "--models-dir",
        default="./models",
        help="Path to models directory"
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save a training configuration template"
    )
    parser.add_argument(
        "--config-path",
        default="./training_config.json",
        help="Path to save configuration"
    )
    
    args = parser.parse_args()
    setup_logging()
    
    if args.list:
        manager = GGUFModelManager(args.models_dir)
        manager.print_models()
    
    if args.save_config:
        config = TrainingConfigBuilder.create_config()
        TrainingConfigBuilder.save_config(config, Path(args.config_path))
        print(f"Configuration saved to: {args.config_path}")

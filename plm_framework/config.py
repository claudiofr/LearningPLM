"""Configuration utilities for the PLM Framework."""
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """
    Load configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration object
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML file
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Convert to OmegaConf
    config = OmegaConf.create(config_dict)

    # Merge with default config
    default_config = get_default_config()
    config = OmegaConf.merge(default_config, config)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object
        output_path: Path to save the configuration
    """
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(OmegaConf.to_container(config, resolve=True),
                  f, default_flow_style=False)


def get_default_config() -> DictConfig:
    """
    Get default configuration.

    Returns:
        Default configuration
    """
    default_config = {
        "model": {
            # Original larger model (650M parameters)
            "name": "facebook/esm2_t33_650M_UR50D",
            "embedding_dim": 1280,  # Embedding dimension for the larger model
            # For faster iteration during development, consider using a smaller model:
            # "name": "facebook/esm2_t6_8M_UR50D",
            # "embedding_dim": 320,
            "reduced_dim": None,  # No dimensionality reduction by default
            "use_pooling": True,
            "quantize": False,
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "epochs": 100,
            "early_stopping": 10,
            "device": "auto",
        },
        "active_learning": {
            "acquisition": "ucb",
            "batch_size": 32,
            "temperature": 1.0,
            "exploration_weight": 2.0,
            "diversity_weight": 0.5,
        },
        "rl": {
            "policy": "a2c",
            "gamma": 0.99,
            "entropy_coef": 0.01,
            "max_mutations": 5,
            "mutation_penalty": 0.1,
        },
        "data": {
            "db_path": "data/variants.db",
            "embedding_cache": "data/embeddings.h5",
            "output_dir": "results",
        },
    }

    return OmegaConf.create(default_config)


def get_config(config_path: Union[str, Path]) -> DictConfig:
    """
    Get configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration object
    """
    return load_config(config_path)

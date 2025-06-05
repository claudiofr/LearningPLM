#!/usr/bin/env python
"""Benchmark script for the PLM Framework."""
import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from plm_framework.config import get_config
from plm_framework.controller import Controller
from plm_framework.datamodels import AssayResult, Variant
from plm_framework.utils import load_assay_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_benchmark_datasets(
    data_dir: str,
) -> Dict[str, Tuple[List[Variant], List[AssayResult]]]:
    """
    Load benchmark datasets.
    
    Args:
        data_dir: Directory containing benchmark datasets
        
    Returns:
        Dictionary mapping dataset names to (variants, results) tuples
    """
    data_dir = Path(data_dir)
    datasets = {}
    
    # Find all CSV files in the directory
    for file_path in data_dir.glob("*.csv"):
        dataset_name = file_path.stem
        
        try:
            # Load dataset
            variants, results = load_assay_results(file_path)
            datasets[dataset_name] = (variants, results)
            logger.info(f"Loaded dataset {dataset_name} with {len(variants)} variants")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
    
    return datasets


def run_benchmark(
    config_path: str,
    datasets: Dict[str, Tuple[List[Variant], List[AssayResult]]],
    output_dir: str,
    n_rounds: int = 5,
    batch_size: int = 10,
    strategies: Optional[List[str]] = None,
    n_repeats: int = 3,
) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    """
    Run benchmark on multiple datasets.
    
    Args:
        config_path: Path to configuration file
        datasets: Dictionary mapping dataset names to (variants, results) tuples
        output_dir: Directory to save results
        n_rounds: Number of rounds to run
        batch_size: Number of variants to propose in each round
        strategies: List of acquisition strategies to test
        n_repeats: Number of repeats for each experiment
        
    Returns:
        Dictionary of benchmark results
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default strategies
    if strategies is None:
        strategies = ["ucb", "ei", "thompson"]
    
    # Load configuration
    config = get_config(config_path)
    
    # Initialize results dictionary
    results = {}
    
    # Run benchmark for each dataset
    for dataset_name, (variants, results_list) in datasets.items():
        logger.info(f"Running benchmark on dataset {dataset_name}")
        
        # Create dataset directory
        dataset_dir = output_dir / dataset_name
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Split data into train and test
        np.random.seed(42)
        indices = np.random.permutation(len(variants))
        train_size = int(0.8 * len(variants))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_variants = [variants[i] for i in train_indices]
        train_results = [results_list[i] for i in train_indices]
        test_variants = [variants[i] for i in test_indices]
        test_results = [results_list[i] for i in test_indices]
        
        # Initialize results for this dataset
        dataset_results = {}
        
        # Run benchmark for each strategy
        for strategy in strategies:
            logger.info(f"Testing strategy: {strategy}")
            
            # Initialize results for this strategy
            strategy_results = []
            
            # Run multiple repeats
            for repeat in range(n_repeats):
                logger.info(f"Repeat {repeat + 1}/{n_repeats}")
                
                # Set random seed
                np.random.seed(42 + repeat)
                torch.manual_seed(42 + repeat)
                
                # Create controller
                controller = Controller(config)
                
                # Measure runtime
                start_time = time.time()
                
                # Run active learning loop
                controller.run_active_learning_loop(
                    candidates=train_variants,
                    initial_variants=train_variants[:batch_size
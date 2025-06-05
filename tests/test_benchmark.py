"""Tests for the benchmark utilities."""
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_protein_gym import load_protein_gym_dataset, simulate_active_learning
from plm_framework.datamodels import Variant, AssayResult


@pytest.fixture
def mock_protein_gym_data():
    """Create mock Protein Gym data for testing."""
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        f.write("mutant,mutated_sequence,DMS_score,DMS_score_bin\n")
        
        # Original sequence
        original_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        
        # Add original sequence
        f.write(f"WT,{original_seq},1.0,1\n")
        
        # Add 50 random variants
        np.random.seed(42)
        for i in range(50):
            # Create a random mutation
            pos = np.random.randint(0, len(original_seq))
            original_aa = original_seq[pos]
            
            # Choose a different amino acid
            aa_vocab = "ACDEFGHIKLMNPQRSTVWY"
            new_aa = original_aa
            while new_aa == original_aa:
                new_aa = aa_vocab[np.random.randint(0, len(aa_vocab))]
            
            # Create mutated sequence
            mutated_seq = original_seq[:pos] + new_aa + original_seq[pos+1:]
            
            # Create mutation name
            mutation = f"{original_aa}{pos+1}{new_aa}"
            
            # Generate a random score
            score = np.random.normal(0, 1)
            
            # Write to file
            f.write(f"{mutation},{mutated_seq},{score},1\n")
    
    yield f.name
    
    # Clean up
    os.unlink(f.name)


@pytest.fixture
def mock_config():
    """Create a mock configuration file
"""Tests for the controller module."""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from plm_framework.controller import Controller
from plm_framework.datamodels import AssayResult, Variant


@pytest.fixture
def config():
    """Create a test configuration."""
    config_dict = {
        "model": {
            "name": "facebook/esm2_t6_8M_UR50D",  # Use a small model for testing
            "embedding_dim": 320,
            "reduced_dim": 32,
            "use_pooling": True,
        },
        "training": {
            "batch_size": 2,
            "device": "cpu",
            "random_state": 42,
        },
        "learner": {
            "type": "ridge",
            "alpha": 1.0,
            "n_estimators": 3,
        },
        "data": {
            "db_path": "test_data.db",
            "output_dir": "test_results",
        },
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def test_variants():
    """Create test variants."""
    return [
        Variant(id="var1", sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        Variant(id="var2", sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAAG"),
        Variant(id="var3", sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGA"),
        Variant(id="var4", sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAAA"),
    ]


@pytest.fixture
def test_results():
    """Create test assay results."""
    return [
        AssayResult(variant_id="var1", score=0.8, uncertainty=0.1, assay_id="assay1", round_id=1),
        AssayResult(variant_id="var2", score=0.6, uncertainty=0.1, assay_id="assay1", round_id=1),
    ]


def test_controller_init(config):
    """Test controller initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Update config with temporary paths
        config.data.db_path = os.path.join(tmpdir, "test.db")
        config.data.output_dir = os.path.join(tmpdir, "results")
        
        # Create controller
        controller = Controller(config)
        
        # Check attributes
        assert controller.config == config
        assert controller.current_round is None
        assert controller.learner is not None


def test_start_complete_round(config):
    """Test starting and completing a round."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Update config with temporary paths
        config.data.db_path = os.path.join(tmpdir, "test.db")
        config.data.output_dir = os.path.join(tmpdir, "results")
        
        # Create controller
        controller = Controller(config)
        
        # Start round
        round_obj = controller.start_round("Test Round", "Test description")
        
        # Check round
        assert round_obj.id == 1
        assert round_obj.name == "Test Round"
        assert round_obj.description == "Test description"
        assert controller.current_round == round_obj
        
        # Complete round
        controller.complete_round()
        
        # Check round
        assert controller.current_round is None


def test_fit_model(config, test_variants, test_results):
    """Test fitting the model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Update config with temporary paths
        config.data.db_path = os.path.join(tmpdir, "test.db")
        config.data.output_dir = os.path.join(tmpdir, "results")
        
        # Create controller
        controller = Controller(config)
        
        # Start round
        controller.start_round("Test Round")
        
        # Add variants and results
        for variant in test_variants:
            controller.data_manager.add_variant(variant)
        
        controller.add_assay_results(test_results)
        
        # Fit model
        controller.fit_model(round_id=1)
        
        # Check model
        assert controller.learner.is_fitted


def test_propose_variants(config, test_variants):
    """Test proposing variants."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Update config with temporary paths
        config.data.db_path = os.path.join(tmpdir, "test.db")
        config.data.output_dir = os.path.join(tmpdir, "results")
        
        # Create controller
        controller = Controller(config)
        
        # Start round
        controller.start_round("Test Round")
        
        # Add variants
        for variant in test_variants:
            controller.data_manager.add_variant(variant)
        
        # Create mock assay results
        results = [
            AssayResult(variant_id="var1", score=0.8, uncertainty=0.1, assay_id="assay1", round_id=1),
            AssayResult(variant_id="var2", score=0.6, uncertainty=0.1, assay_id="assay1", round_id=1),
        ]
        
        # Add results
        controller.add_assay_results(results)
        
        # Fit model
        controller.fit_model()
        
        # Propose variants
        proposed = controller.propose_variants(
            candidates=test_variants,
            batch_size=2,
            strategy="ucb",
            temperature=1.0,
        )
        
        # Check proposed variants
        assert len(proposed) == 2
        assert all(isinstance(p.variant, Variant) for p in proposed)
        assert all(p.acquisition_score > 0 for p in proposed)
"""Tests for acquisition strategies."""
import numpy as np
import pytest

from plm_framework.acquisition import (
    batch_acquisition,
    expected_improvement,
    thompson_sampling,
    ucb_acquisition,
)
from plm_framework.learners.active import MLPLearner, RidgeLearner


@pytest.fixture
def test_data():
    """Create test data for acquisition functions."""
    np.random.seed(42)
    embeddings = np.random.randn(100, 128)
    return embeddings


@pytest.fixture
def fitted_learner():
    """Create a fitted learner for testing."""
    np.random.seed(42)
    embeddings = np.random.randn(50, 128)
    scores = np.random.rand(50)
    
    learner = RidgeLearner(
        embedding_dim=128,
        reduced_dim=32,
        n_estimators=5,
        random_state=42,
    )
    
    learner.fit(embeddings, scores)
    return learner


def test_ucb_acquisition(test_data, fitted_learner):
    """Test UCB acquisition function."""
    # Get acquisition scores
    scores = ucb_acquisition(fitted_learner, test_data, kappa=2.0)
    
    # Check shape
    assert scores.shape == (len(test_data),)
    
    # Check values
    assert np.all(np.isfinite(scores))
    
    # Test with different kappa
    scores_high_kappa = ucb_acquisition(fitted_learner, test_data, kappa=5.0)
    scores_low_kappa = ucb_acquisition(fitted_learner, test_data, kappa=0.5)
    
    # Higher kappa should lead to more exploration
    assert np.std(scores_high_kappa) > np.std(scores_low_kappa)


def test_expected_improvement(test_data, fitted_learner):
    """Test expected improvement acquisition function."""
    # Get acquisition scores
    scores = expected_improvement(fitted_learner, test_data, xi=0.01)
    
    # Check shape
    assert scores.shape == (len(test_data),)
    
    # Check values
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)
    
    # Test with different xi
    scores_high_xi = expected_improvement(fitted_learner, test_data, xi=0.1)
    scores_low_xi = expected_improvement(fitted_learner, test_data, xi=0.001)
    
    # Higher xi should lead to more exploration
    assert np.sum(scores_high_xi > 0) >= np.sum(scores_low_xi > 0)


def test_thompson_sampling(test_data, fitted_learner):
    """Test Thompson sampling acquisition function."""
    # Get acquisition scores
    scores = thompson_sampling(fitted_learner, test_data)
    
    # Check shape
    assert scores.shape == (len(test_data),)
    
    # Check values
    assert np.all(np.isfinite(scores))
    
    # Test multiple runs (should be different due to randomness)
    scores2 = thompson_sampling(fitted_learner, test_data)
    assert not np.allclose(scores, scores2)


def test_batch_acquisition(test_data, fitted_learner):
    """Test batch acquisition function."""
    batch_size = 10
    
    # Test UCB strategy
    indices_ucb = batch_acquisition(
        fitted_learner,
        test_data,
        batch_size=batch_size,
        strategy="ucb",
        kappa=2.0,
    )
    
    # Check number of selected indices
    assert len(indices_ucb) == batch_size
    
    # Check that indices are unique
    assert len(np.unique(indices_ucb)) == batch_size
    
    # Test EI strategy
    indices_ei = batch_acquisition(
        fitted_learner,
        test_data,
        batch_size=batch_size,
        strategy="ei",
        xi=0.01,
    )
    
    # Check number of selected indices
    assert len(indices_ei) == batch_size
    
    # Test Thompson strategy
    indices_thompson = batch_acquisition(
        fitted_learner,
        test_data,
        batch_size=batch_size,
        strategy="thompson",
    )
    
    # Check number of selected indices
    assert len(indices_thompson) == batch_size
    
    # Test invalid strategy
    with pytest.raises(ValueError):
        batch_acquisition(
            fitted_learner,
            test_data,
            batch_size=batch_size,
            strategy="invalid",
        )


def test_diversity_penalized_acquisition(test_data, fitted_learner):
    """Test diversity-penalized acquisition function."""
    batch_size = 10
    
    # Test with diversity penalty
    indices_diverse = batch_acquisition(
        fitted_learner,
        test_data,
        batch_size=batch_size,
        strategy="ucb",
        kappa=2.0,
        diversity_weight=1.0,
    )
    
    # Check number of selected indices
    assert len(indices_diverse) == batch_size
    
    # Test without diversity penalty
    indices_no_diverse = batch_acquisition(
        fitted_learner,
        test_data,
        batch_size=batch_size,
        strategy="ucb",
        kappa=2.0,
        diversity_weight=0.0,
    )
    
    # Check that the selections are different
    assert not np.array_equal(indices_diverse, indices_no_diverse)


def test_temperature_scaling(test_data, fitted_learner):
    """Test temperature scaling in batch acquisition."""
    batch_size = 10
    
    # Test with high temperature (more random)
    indices_high_temp = batch_acquisition(
        fitted_learner,
        test_data,
        batch_size=batch_size,
        strategy="ucb",
        kappa=2.0,
        temperature=10.0,
    )
    
    # Test with low temperature (more greedy)
    indices_low_temp = batch_acquisition(
        fitted_learner,
        test_data,
        batch_size=batch_size,
        strategy="ucb",
        kappa=2.0,
        temperature=0.1,
    )
    
    # Check that the selections are different
    assert not np.array_equal(indices_high_temp, indices_low_temp)
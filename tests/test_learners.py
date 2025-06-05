"""Tests for the learner modules."""
import numpy as np
import pytest

from plm_framework.learners.active import MLPLearner, RidgeLearner
from plm_framework.learners.rl import RLPolicyLearner


@pytest.fixture
def test_data():
    """Create test data for learners."""
    # Create random embeddings and scores
    np.random.seed(42)
    embeddings = np.random.randn(20, 128)
    scores = np.random.rand(20)
    uncertainties = np.random.rand(20) * 0.1

    return embeddings, scores, uncertainties


def test_ridge_learner(test_data):
    """Test RidgeLearner."""
    embeddings, scores, uncertainties = test_data

    # Create learner
    learner = RidgeLearner(
        embedding_dim=128,
        reduced_dim=32,
        alpha=1.0,
        n_estimators=5,
        random_state=42,
    )

    # Fit learner
    learner.fit(embeddings, scores, uncertainties)

    # Check if fitted
    assert learner.is_fitted

    # Make predictions
    predictions, pred_uncertainties = learner.predict(embeddings)

    # Check predictions
    assert predictions.shape == scores.shape
    assert pred_uncertainties.shape == scores.shape

    # Check correlation with true scores
    corr = np.corrcoef(predictions, scores)[0, 1]
    assert corr > 0.5  # Should have reasonable correlation with training data


def test_mlp_learner(test_data):
    """Test MLPLearner."""
    embeddings, scores, uncertainties = test_data

    # Create learner
    learner = MLPLearner(
        embedding_dim=128,
        reduced_dim=32,
        hidden_layer_sizes=(64, 32),
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=200,
        n_dropout_samples=5,
        random_state=42,
    )

    # Fit learner
    learner.fit(embeddings, scores, uncertainties)

    # Check if fitted
    assert learner.is_fitted

    # Make predictions
    predictions, pred_uncertainties = learner.predict(embeddings)

    # Check predictions
    assert predictions.shape == scores.shape
    assert pred_uncertainties.shape == scores.shape

    # Check correlation with true scores
    corr = np.corrcoef(predictions, scores)[0, 1]
    assert corr > 0.5  # Should have reasonable correlation with training data


def test_rl_policy_learner(test_data):
    """Test RLPolicyLearner."""
    embeddings, scores, _ = test_data

    # Create learner
    learner = RLPolicyLearner(
        embedding_dim=128,
        reduced_dim=32,
        hidden_dim=64,
        action_dim=20,
        gamma=0.99,
        entropy_coef=0.01,
        learning_rate=0.001,
        max_mutations=5,
        mutation_penalty=0.1,
        device="cpu",
        random_state=42,
    )

    # Fit learner
    learner.fit(embeddings, scores)

    # Check if fitted
    assert learner.is_fitted

    # Make predictions
    predictions, _ = learner.predict(embeddings)

    # Check predictions
    assert predictions.shape == scores.shape

    # Get acquisition scores
    acq_scores = learner.get_acquisition_scores(embeddings, temperature=1.0)

    # Check acquisition scores
    assert acq_scores.shape == scores.shape
    assert np.all(acq_scores >= 0)  # Acquisition scores should be non-negative


def test_dimensionality_reduction():
    """Test dimensionality reduction in learners."""
    # Create random data
    np.random.seed(42)
    embeddings = np.random.randn(20, 128)
    scores = np.random.rand(20)

    # Create learner with dimensionality reduction
    learner = RidgeLearner(
        embedding_dim=128,
        reduced_dim=32,
        random_state=42,
    )

    # Fit learner
    learner.fit(embeddings, scores)

    # Check PCA components
    assert learner.pca.components_.shape == (32, 128)

    # Create learner without dimensionality reduction
    learner_no_pca = RidgeLearner(
        embedding_dim=128,
        reduced_dim=None,
        random_state=42,
    )

    # Fit learner
    learner_no_pca.fit(embeddings, scores)

    # Check that PCA is None
    assert learner_no_pca.pca is None


def test_uncertainty_estimation():
    """Test uncertainty estimation in learners."""
    # Create random data
    np.random.seed(42)
    embeddings = np.random.randn(20, 128)
    scores = np.random.rand(20)

    # Create Ridge learner
    ridge_learner = RidgeLearner(
        embedding_dim=128,
        reduced_dim=32,
        n_estimators=10,
        random_state=42,
    )

    # Fit learner
    ridge_learner.fit(embeddings, scores)

    # Make predictions
    _, uncertainties = ridge_learner.predict(embeddings)

    # Check uncertainties
    assert uncertainties.shape == scores.shape
    assert np.all(uncertainties >= 0)  # Uncertainties should be non-negative

    # Create MLP learner
    mlp_learner = MLPLearner(
        embedding_dim=128,
        reduced_dim=32,
        n_dropout_samples=10,
        random_state=42,
    )

    # Fit learner
    mlp_learner.fit(embeddings, scores)

    # Make predictions
    _, uncertainties = mlp_learner.predict(embeddings)

    # Check uncertainties
    assert uncertainties.shape == scores.shape
    assert np.all(uncertainties >= 0)  # Uncertainties should be non-negative

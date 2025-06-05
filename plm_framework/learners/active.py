"""Active learning heads for the PLM Framework."""
from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from plm_framework.learners.base import BaseLearner

# Initialize logger
logger = logging.getLogger(__name__)


class RidgeLearner(BaseLearner):
    """Ridge regression learner with uncertainty estimation via bagging."""

    def __init__(
        self,
        embedding_dim: int = 1280,
        reduced_dim: Optional[int] = 128,
        alpha: float = 1.0,
        n_estimators: int = 10,
        random_state: int = 42,
    ):
        """
        Initialize the ridge learner.

        Args:
            embedding_dim: Dimension of input embeddings
            reduced_dim: Dimension to reduce embeddings to
            alpha: Regularization strength
            n_estimators: Number of bagging estimators for uncertainty
            random_state: Random seed
        """
        super().__init__(embedding_dim, reduced_dim, random_state)
        self.alpha = alpha
        self.n_estimators = n_estimators

        self.scaler = StandardScaler()
        self.model = BaggingRegressor(
            Ridge(alpha=alpha),
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(
        self,
        embeddings: np.ndarray,
        scores: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit the ridge regression model.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]
            scores: Target scores [n_samples]
            uncertainties: Optional measurement uncertainties [n_samples]
        """
        # Log input data stats
        logger.info(f"Fitting RidgeLearner with {len(scores)} samples")
        logger.info(
            f"Scores range: min={np.min(scores):.4f}, max={np.max(scores):.4f}, mean={np.mean(scores):.4f}")

        # Reduce dimensionality
        X = self._reduce_embeddings(embeddings)
        logger.info(f"Reduced embeddings shape: {X.shape}")

        # Scale features
        X = self.scaler.fit_transform(X)

        # Fit model
        self.model.fit(X, scores)

        # Log model info
        if hasattr(self.model, 'estimators_'):
            logger.info(f"Fitted {len(self.model.estimators_)} estimators")

        self.is_fitted = True
        logger.info("Model fitting complete")

    def predict(
        self,
        embeddings: np.ndarray,
        return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions with the ridge regression model.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]
            return_std: Whether to return standard deviations

        Returns:
            Predictions [n_samples] or tuple of (predictions, uncertainties)
        """
        if not self.is_fitted:
            logger.warning("Model not fitted yet, returning zeros")
            if return_std:
                return np.zeros(embeddings.shape[0]), np.ones(embeddings.shape[0])
            else:
                return np.zeros(embeddings.shape[0])

        # Reduce dimensionality
        X = self._reduce_embeddings(embeddings)

        # Scale features
        X = self.scaler.transform(X)

        # Make predictions
        if hasattr(self.model, 'predict') and not return_std:
            # Standard prediction
            return self.model.predict(X)
        elif hasattr(self.model, 'estimators_') and return_std:
            # For ensemble models, get predictions from all estimators
            predictions = np.zeros((X.shape[0], len(self.model.estimators_)))
            for i, estimator in enumerate(self.model.estimators_):
                predictions[:, i] = estimator.predict(X)

            # Calculate mean and std across estimators
            mean_pred = np.mean(predictions, axis=1)
            std_pred = np.std(predictions, axis=1)

            return mean_pred, std_pred
        else:
            # Fallback for non-ensemble models
            preds = self.model.predict(X)
            if return_std:
                # Use a simple heuristic for uncertainty
                uncertainties = np.ones_like(preds) * 0.1
                return preds, uncertainties
            else:
                return preds

    def get_acquisition_scores(
        self, embeddings: np.ndarray, strategy: str = "ucb", kappa: float = 2.0
    ) -> np.ndarray:
        """
        Get acquisition scores for candidate embeddings.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]
            strategy: Acquisition strategy ('ucb', 'ei', 'thompson')
            kappa: Exploration weight for UCB

        Returns:
            Acquisition scores [n_samples]
        """
        mean_pred, std_pred = self.predict(embeddings)

        if strategy == "ucb":
            # Upper Confidence Bound
            return mean_pred + kappa * std_pred
        elif strategy == "ei":
            # Expected Improvement (simplified)
            return mean_pred + std_pred
        elif strategy == "thompson":
            # Thompson Sampling (sample from posterior)
            return np.random.normal(mean_pred, std_pred)
        else:
            raise ValueError(f"Unknown acquisition strategy: {strategy}")

    def predict_single(
        self,
        embedding: np.ndarray,
        return_std: bool = False,
    ) -> Union[float, Tuple[float, float]]:
        """
        Make prediction for a single embedding.

        Args:
            embedding: Protein embedding [embedding_dim]
            return_std: Whether to return standard deviation

        Returns:
            Prediction or tuple of (prediction, uncertainty)
        """
        # Reshape to [1, embedding_dim]
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)

        # Make prediction
        if return_std:
            pred, std = self.predict(embedding, return_std=True)
            return float(pred[0]), float(std[0])
        else:
            pred = self.predict(embedding, return_std=False)
            return float(pred[0])


class MLPLearner(BaseLearner):
    """MLP learner with uncertainty estimation via dropout."""

    def __init__(
        self,
        embedding_dim: int = 1280,
        reduced_dim: Optional[int] = 128,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        alpha: float = 0.0001,
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        n_dropout_samples: int = 10,
        random_state: int = 42,
    ):
        """
        Initialize the MLP learner.

        Args:
            embedding_dim: Dimension of input embeddings
            reduced_dim: Dimension to reduce embeddings to
            hidden_layer_sizes: Size of hidden layers
            alpha: L2 regularization
            learning_rate_init: Initial learning rate
            max_iter: Maximum number of iterations
            n_dropout_samples: Number of dropout samples for uncertainty
            random_state: Random seed
        """
        super().__init__(embedding_dim, reduced_dim, random_state)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.n_dropout_samples = n_dropout_samples

        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(
        self,
        embeddings: np.ndarray,
        scores: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit the MLP model.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]
            scores: Target scores [n_samples]
            uncertainties: Optional measurement uncertainties [n_samples]
        """
        # Log input data stats
        logger.info(f"Fitting MLPLearner with {len(scores)} samples")
        logger.info(
            f"Scores range: min={np.min(scores):.4f}, max={np.max(scores):.4f}, mean={np.mean(scores):.4f}")

        # Reduce dimensionality
        X = self._reduce_embeddings(embeddings)
        logger.info(f"Reduced embeddings shape: {X.shape}")

        # Scale features
        X = self.scaler.fit_transform(X)

        # Fit model
        self.model.fit(X, scores)
        self.is_fitted = True

        logger.info("Model fitting complete")

    def predict(
        self, embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict scores and uncertainties using MC dropout.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]

        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")

        # Reduce dimensionality
        X = self._reduce_embeddings(embeddings)

        # Scale features
        X = self.scaler.transform(X)

        # Get multiple predictions with different random states
        # This simulates dropout at inference time
        predictions = np.array([
            self.model.predict(X)
            for _ in range(self.n_dropout_samples)
        ])

        # Mean prediction
        mean_pred = np.mean(predictions, axis=0)

        # Standard deviation as uncertainty
        std_pred = np.std(predictions, axis=0)

        return mean_pred, std_pred

    def get_acquisition_scores(
        self, embeddings: np.ndarray, strategy: str = "ucb", kappa: float = 2.0
    ) -> np.ndarray:
        """
        Get acquisition scores for candidate embeddings.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]
            strategy: Acquisition strategy ('ucb', 'ei', 'thompson')
            kappa: Exploration weight for UCB

        Returns:
            Acquisition scores [n_samples]
        """
        mean_pred, std_pred = self.predict(embeddings)

        if strategy == "ucb":
            # Upper Confidence Bound
            return mean_pred + kappa * std_pred
        elif strategy == "ei":
            # Expected Improvement (simplified)
            return mean_pred + std_pred
        elif strategy == "thompson":
            # Thompson Sampling (sample from posterior)
            return np.random.normal(mean_pred, std_pred)
        else:
            raise ValueError(f"Unknown acquisition strategy: {strategy}")

    def predict_single(self, embedding: np.ndarray) -> float:
        """
        Predict score for a single embedding.

        Args:
            embedding: Protein embedding [embedding_dim]

        Returns:
            Predicted score
        """
        # Ensure embedding is 2D
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Get prediction
        mean_pred, _ = self.predict(embedding)

        # Return single value
        return float(mean_pred[0])

    def uncertainty_single(self, embedding: np.ndarray) -> float:
        """
        Get uncertainty for a single embedding.

        Args:
            embedding: Protein embedding [embedding_dim]

        Returns:
            Prediction uncertainty
        """
        # Ensure embedding is 2D
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Get prediction and uncertainty
        _, std_pred = self.predict(embedding)

        # Return single value
        return float(std_pred[0])

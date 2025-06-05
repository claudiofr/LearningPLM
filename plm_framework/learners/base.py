"""Base learner class for the PLM Framework."""
from typing import Optional, Tuple, Union
import logging

import numpy as np
from sklearn.decomposition import PCA

# Initialize logger
logger = logging.getLogger(__name__)


class BaseLearner:
    """Base class for learners."""

    def __init__(
        self,
        embedding_dim: int = 1280,
        reduced_dim: Optional[int] = None,  # Changed default to None
        random_state: int = 42,
    ):
        """
        Initialize the base learner.

        Args:
            embedding_dim: Dimension of input embeddings
            reduced_dim: Dimension to reduce embeddings to (None for no reduction)
            random_state: Random seed
        """
        self.embedding_dim = embedding_dim
        self.reduced_dim = reduced_dim
        self.random_state = random_state

        # Initialize PCA for dimensionality reduction
        self.pca = None
        if reduced_dim is not None:
            self.pca = PCA(n_components=reduced_dim, random_state=random_state)

        # Flag to indicate if the model has been fitted
        self.is_fitted = False

    def _reduce_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce dimensionality of embeddings.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]

        Returns:
            Reduced embeddings [n_samples, reduced_dim]
        """
        if self.pca is None:
            return embeddings

        if not self.is_fitted:
            # Adjust n_components to be no larger than the number of samples
            n_samples = embeddings.shape[0]
            if self.reduced_dim > n_samples - 1:
                # Temporarily adjust PCA components
                adjusted_components = min(n_samples - 1, self.reduced_dim)
                self.pca.n_components = adjusted_components
                print(
                    f"Adjusted PCA components from {self.reduced_dim} to {adjusted_components} based on sample size")

            # Fit PCA
            return self.pca.fit_transform(embeddings)
        else:
            # Transform using fitted PCA
            return self.pca.transform(embeddings)

    def fit(
        self,
        embeddings: np.ndarray,
        scores: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit the model.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]
            scores: Target scores [n_samples]
            uncertainties: Optional measurement uncertainties [n_samples]
        """
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(
        self, embeddings: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict scores and uncertainties.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]

        Returns:
            Tuple of (predictions, uncertainties)
        """
        raise NotImplementedError("Subclasses must implement predict method")

    def get_acquisition_scores(
        self, embeddings: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Get acquisition scores for candidate embeddings.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]
            **kwargs: Additional parameters for acquisition function

        Returns:
            Acquisition scores [n_samples]
        """
        raise NotImplementedError(
            "Subclasses must implement get_acquisition_scores method")

    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model
        """
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BaseLearner":
        """
        Load the model from disk.

        Args:
            path: Path to load the model from

        Returns:
            Loaded model
        """
        import pickle

        with open(path, "rb") as f:
            model = pickle.load(f)

        return model

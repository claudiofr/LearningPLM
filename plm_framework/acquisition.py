"""Acquisition functions for active learning."""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from plm_framework.datamodels import ProposedVariant, Variant

logger = logging.getLogger(__name__)


def ucb_acquisition(
    candidates: List[Variant],
    embeddings: np.ndarray,
    predicted_scores: np.ndarray,
    predicted_uncertainties: np.ndarray,
    exploration_weight: float = 2.0,
    temperature: float = 1.0,
) -> List[Tuple[Variant, float]]:
    """
    Upper Confidence Bound (UCB) acquisition function.

    Args:
        candidates: List of candidate variants
        embeddings: Embeddings of candidates
        predicted_scores: Predicted scores for candidates
        predicted_uncertainties: Predicted uncertainties for candidates
        exploration_weight: Weight for exploration term
        temperature: Temperature for softmax

    Returns:
        List of (variant, acquisition_score) tuples
    """
    # Calculate UCB scores
    ucb_scores = predicted_scores + exploration_weight * predicted_uncertainties

    # Apply temperature
    if temperature != 1.0:
        ucb_scores = ucb_scores / temperature

    return [(candidates[i], float(ucb_scores[i])) for i in range(len(candidates))]


def expected_improvement(
    candidates: List[Variant],
    embeddings: np.ndarray,
    predicted_scores: np.ndarray,
    predicted_uncertainties: np.ndarray,
    best_score: float,  # Make this a required parameter
    xi: float = 0.01,
    temperature: float = 1.0,
) -> List[Tuple[Variant, float]]:
    """
    Expected Improvement (EI) acquisition function.

    Args:
        candidates: List of candidate variants
        embeddings: Embeddings of candidates
        predicted_scores: Predicted scores for candidates
        predicted_uncertainties: Predicted uncertainties for candidates
        best_score: Best observed score so far
        xi: Exploration parameter
        temperature: Temperature for softmax

    Returns:
        List of (variant, acquisition_score) tuples
    """
    from scipy.stats import norm

    # Ensure best_score is a float
    best_score = float(best_score)

    # Handle zero uncertainties to avoid division by zero
    mask = predicted_uncertainties > 0

    # Initialize EI scores
    ei = np.zeros_like(predicted_scores)

    # Calculate improvement where uncertainties > 0
    if np.any(mask):
        # Calculate z score
        z = (predicted_scores[mask] - best_score - xi) / predicted_uncertainties[mask]

        # Calculate EI
        ei[mask] = predicted_uncertainties[mask] * (z * norm.cdf(z) + norm.pdf(z))

    # For points with zero uncertainty, use a small positive value if they improve on best_score
    ei[~mask] = np.maximum(0, predicted_scores[~mask] - best_score - xi)

    # Apply temperature
    if temperature != 1.0:
        ei = ei / temperature

    return [(candidates[i], float(ei[i])) for i in range(len(candidates))]


def thompson_sampling(
    candidates: List[Variant],
    embeddings: np.ndarray,
    predicted_scores: np.ndarray,
    predicted_uncertainties: np.ndarray,
    n_samples: int = 1,
    temperature: float = 1.0,
) -> List[Tuple[Variant, float]]:
    """
    Thompson sampling acquisition function.

    Args:
        candidates: List of candidate variants
        embeddings: Embeddings of candidates
        predicted_scores: Predicted scores for candidates
        predicted_uncertainties: Predicted uncertainties for candidates
        n_samples: Number of samples to draw
        temperature: Temperature for softmax

    Returns:
        List of (variant, acquisition_score) tuples
    """
    # Draw samples from posterior
    samples = np.random.normal(
        loc=predicted_scores,
        scale=predicted_uncertainties,
        size=(n_samples, len(candidates)),
    )

    # Average samples
    thompson_scores = np.mean(samples, axis=0)

    # Apply temperature
    if temperature != 1.0:
        thompson_scores = thompson_scores / temperature

    return [(candidates[i], float(thompson_scores[i])) for i in range(len(candidates))]


def diversity_sampling(
    candidates: List[Variant],
    embeddings: np.ndarray,
    temperature: float = 1.0,
) -> List[Tuple[Variant, float]]:
    """
    Diversity-based sampling.

    Args:
        candidates: List of candidate variants
        embeddings: Embeddings of candidates
        temperature: Temperature for softmax

    Returns:
        List of (variant, acquisition_score) tuples
    """
    # Calculate pairwise similarities
    similarities = cosine_similarity(embeddings)

    # Calculate diversity scores (negative mean similarity)
    diversity_scores = -np.mean(similarities, axis=1)

    # Apply temperature
    if temperature != 1.0:
        diversity_scores = diversity_scores / temperature

    return [(candidates[i], float(diversity_scores[i])) for i in range(len(candidates))]


def batch_acquisition(
    variant_ids: List[str],
    embeddings: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    uncertainties: Optional[np.ndarray] = None,
    strategy: str = "ucb",
    temperature: float = 1.0,
    exploration_weight: float = 2.0,
    diversity_weight: float = 0.5,
    batch_size: int = 10,
    **kwargs,
) -> np.ndarray:
    """
    Batch acquisition function.

    Args:
        variant_ids: List of variant IDs
        embeddings: Embeddings for variants
        predictions: Optional predictions for variants
        uncertainties: Optional uncertainties for variants
        strategy: Acquisition strategy
        temperature: Temperature for softmax
        exploration_weight: Weight for exploration term in UCB
        diversity_weight: Weight for diversity term
        batch_size: Batch size
        **kwargs: Additional parameters for acquisition function

    Returns:
        Acquisition scores for variants
    """
    import logging

    logger = logging.getLogger(__name__)

    logger.info(f"Running batch_acquisition with strategy: {strategy}")
    logger.info(f"Number of variants: {len(variant_ids)}")
    logger.info(f"Embeddings shape: {embeddings.shape}")

    if predictions is not None:
        logger.info(
            f"Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}"
        )
        logger.info(
            f"Predictions range: min={np.min(predictions):.4f}, max={np.max(predictions):.4f}, mean={np.mean(predictions):.4f}"
        )
    if uncertainties is not None:
        logger.info(
            f"Uncertainties shape: {uncertainties.shape if hasattr(uncertainties, 'shape') else len(uncertainties)}"
        )
        logger.info(
            f"Uncertainties range: min={np.min(uncertainties):.4f}, max={np.max(uncertainties):.4f}, mean={np.mean(uncertainties):.4f}"
        )

    # Check for empty inputs
    if len(variant_ids) == 0 or embeddings.shape[0] == 0:
        logger.error("Empty inputs to batch_acquisition!")
        return np.array([])

    try:
        # Calculate acquisition scores based on strategy
        if strategy == "random":
            logger.info("Using random acquisition")
            scores = np.random.rand(len(variant_ids))

        elif (
            strategy == "ucb" and predictions is not None and uncertainties is not None
        ):
            logger.info(
                f"Using UCB acquisition with exploration_weight={exploration_weight}"
            )
            # Upper confidence bound
            scores = predictions + exploration_weight * uncertainties

        elif strategy == "ei" and predictions is not None and uncertainties is not None:
            logger.info("Using EI acquisition")
            # Expected improvement
            from scipy.stats import norm

            best_f = np.max(predictions)
            z = (predictions - best_f) / (uncertainties + 1e-6)
            scores = (predictions - best_f) * norm.cdf(z) + uncertainties * norm.pdf(z)

        elif strategy == "ts" and predictions is not None and uncertainties is not None:
            logger.info("Using Thompson sampling")
            # Thompson sampling
            scores = np.random.normal(predictions, uncertainties)

        elif strategy == "diversity":
            logger.info(f"Using diversity acquisition with batch_size={batch_size}")
            # Diversity-based sampling
            from sklearn.cluster import KMeans

            # Normalize embeddings
            norm_embeddings = embeddings / (
                np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            )

            # Use K-means clustering
            n_clusters = min(batch_size, len(variant_ids))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(norm_embeddings)

            # Calculate distance to cluster centers
            distances = np.linalg.norm(
                norm_embeddings - kmeans.cluster_centers_[cluster_labels], axis=1
            )

            # Invert distances to get scores (smaller distance = higher score)
            scores = 1.0 / (distances + 1e-8)

        elif (
            strategy == "esm_logit"
            and "original_sequence" in kwargs
            and "model_name" in kwargs
        ):
            logger.info("Using ESM logit acquisition")
            # ESM logit-based sampling
            # from plm_framework.embedding import score_mutations_with_esm

            original_sequence = kwargs["original_sequence"]
            model_name = kwargs["model_name"]
            device = kwargs.get("device", None)
            mutable_positions = kwargs.get("mutable_positions", None)

            # Get variant sequences
            variant_sequences = [
                kwargs.get("variant_sequences", {}).get(vid) for vid in variant_ids
            ]
            if None in variant_sequences:
                logger.warning(
                    "Missing variant sequences, using diversity sampling instead"
                )
                # If variant sequences are not provided, use diversity sampling
                return batch_acquisition(
                    variant_ids=variant_ids,
                    embeddings=embeddings,
                    strategy="diversity",
                    batch_size=batch_size,
                )
            else:
                # Score mutations using ESM logits
                scores = score_mutations_with_esm(
                    original_sequence=original_sequence,
                    variant_sequences=variant_sequences,
                    model_name=model_name,
                    device=device,
                    mutable_positions=mutable_positions,
                )

        else:
            logger.warning(
                f"Unknown or incompatible strategy: {strategy}, using random acquisition"
            )
            scores = np.random.rand(len(variant_ids))

        logger.info(f"Generated {len(scores)} acquisition scores")
        if len(scores) > 0:
            logger.info(
                f"Score range: min={np.min(scores):.4f}, max={np.max(scores):.4f}, mean={np.mean(scores):.4f}"
            )

        return scores

    except Exception as e:
        logger.error(f"Error in batch_acquisition: {e}")
        import traceback

        logger.error(traceback.format_exc())
        # Return uniform scores as fallback
        logger.warning("Using uniform scores as fallback")
        return np.ones(len(variant_ids))


def score_mutations_with_esm(
    original_sequence: str,
    variant_sequences: List[str],
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    device: Optional[str] = None,
    mutable_positions: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Score mutations using ESM2 logits.

    Args:
        original_sequence: Original protein sequence
        variant_sequences: List of variant sequences
        model_name: Name of the ESM model to use
        device: Device to run the model on
        mutable_positions: List of positions that can be mutated (0-indexed)

    Returns:
        Array of scores for each variant
    """
    # ... existing code ...

    # If mutable_positions is provided, only consider mutations at those positions
    if mutable_positions:
        for i, variant_seq in enumerate(variant_sequences):
            # Find mutations
            mutations = []
            for pos, (orig_aa, var_aa) in enumerate(
                zip(original_sequence, variant_seq)
            ):
                if orig_aa != var_aa:
                    # Check if position is in mutable_positions
                    if pos not in mutable_positions:
                        # Set score to -inf for variants with mutations outside the allowed range
                        scores[i] = float("-inf")
                        break

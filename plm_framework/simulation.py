from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
from plm_framework.datamodels import Variant, AssayResult
from plm_framework.controller import Controller
from plm_framework.config import get_config
import os
import random
import logging
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr


logger = logging.getLogger(__name__)


def split_train_test(
    variants: List[Variant],
    assay_results: List[AssayResult],
    test_fraction: float,
    seed: int = 42
) -> Tuple[List[Variant], List[AssayResult], List[Variant], List[AssayResult]]:
    """
    Randomly split the dataset into training and testing subsets.
    Ensures reproducibility with a fixed seed.
    """
    n = len(variants)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    test_size = int(n * test_fraction)
    test_indices = set(indices[:test_size])
    train_indices = set(indices[test_size:])

    train_variants = [variants[i] for i in train_indices]
    test_variants = [variants[i] for i in test_indices]
    train_results = [assay_results[i] for i in train_indices]
    test_results = [assay_results[i] for i in test_indices]

    return train_variants, train_results, test_variants, test_results



def initialize_controller(
    config_path: str,
    strategy: str,
    output_dir: Path,
    initial_batch_size: int,
    seed: int = 42
) -> Controller:
    """
    Load a config, initialize a Controller object, and adjust output paths 
    for a given acquisition strategy. 
    Also dynamically adjusts PCA components based on initial batch size.
    """
    config = get_config(config_path)
    db_path = output_dir / f"{strategy}_variants_{seed}.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    config.data.db_path = str(db_path)
    config.data.output_dir = str(output_dir / strategy)
    os.makedirs(config.data.output_dir, exist_ok=True)

    if hasattr(config.model, "pca_components"):
        config.model.pca_components = min(
            initial_batch_size - 1,
            getattr(config.model, "pca_components", 128)
        )
        logger.info(f"Adjusted PCA components to {config.model.pca_components}")

    return Controller(config)



def run_initial_round(
    controller: Controller,
    train_variants: List[Variant],
    train_assay_results: List[AssayResult],
    initial_batch_size: int,
    test_embeddings: np.ndarray,
    test_scores: np.ndarray,
) -> Tuple[List[int], Dict[str, float]]:
    """Run the first random round of active learning and evaluate initial model."""
    selected_variant_ids = set()

    # Randomly select initial variants
    initial_indices = random.sample(
        range(len(train_variants)),
        min(initial_batch_size, len(train_variants))
    )
    initial_variants = [train_variants[i] for i in initial_indices]

    controller.start_round("Round 1", "Initial random batch")

    for variant in initial_variants:
        controller.data_manager.add_variant(variant)
        selected_variant_ids.add(variant.id)

    initial_assay_results = [
        AssayResult(
            variant_id=result.variant_id,
            score=result.score,
            uncertainty=result.uncertainty,
            round_id=1
        )
        for result in train_assay_results
        if result.variant_id in selected_variant_ids
    ]
    controller.add_assay_results(initial_assay_results)

    controller.complete_round()
    controller.fit_model(round_id=1)

    _, metrics = get_predictions_and_metrics(controller, test_embeddings, test_scores)
    logger.info(
        f"Initial round - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, "
        f"Spearman: {metrics['spearman']:.4f}, Top-10 mean: {metrics['top_n_mean']:.4f}"
    )

    return list(selected_variant_ids), metrics



def run_active_learning_round(
    controller: Controller,
    train_variants: List[Variant],
    train_assay_results: List[AssayResult],
    selected_variant_ids: set,
    batch_size: int,
    test_embeddings: np.ndarray,
    test_scores: np.ndarray,
    round_num: int,
    strategy: str,
) -> Tuple[List[int], Dict[str, float]]:
    """Run a single round of active learning."""
    # Get candidates not yet selected
    available_variants = [
        v for v in train_variants if v.id not in selected_variant_ids
    ]
    if not available_variants:
        logger.warning(f"No variants left for round {round_num}")
        return list(selected_variant_ids), {}

    controller.start_round(
        f"Round {round_num}", f"Active learning round {round_num}"
    )

    try:
        proposed = controller.propose_variants(
            candidates=available_variants,
            batch_size=batch_size,
            strategy=strategy,
            temperature=1.0,
        )
    except Exception as e:
        logger.error(f"Proposal failed: {e}")
        proposed = []

    if not proposed:
        logger.error(f"Strategy '{strategy}' proposed no variants in round {round_num}")
        raise RuntimeError(f"Strategy '{strategy}' proposed no variants in round {round_num}")

    new_ids = [p.variant.id for p in proposed]
    selected_variant_ids.update(new_ids)

    # Add proposed variants and results
    for p in proposed:
        controller.data_manager.add_variant(p.variant)

    new_assay_results = [
        AssayResult(
            variant_id=res.variant_id,
            score=res.score,
            uncertainty=res.uncertainty,
            round_id=round_num
        )
        for res in train_assay_results if res.variant_id in new_ids
    ]

    if not new_assay_results:
        logger.error(f"No assay results found for proposed variants in round {round_num}")
        raise ValueError(f"No assay results found for proposed variants in round {round_num}")

    controller.add_assay_results(new_assay_results)
    controller.complete_round()

    try:
        controller.fit_model(round_id=round_num)
    except Exception as e:
        logger.error(f"Model fitting failed in round {round_num}: {e}")

    _, metrics = get_predictions_and_metrics(controller, test_embeddings, test_scores)
    logger.info(
        f"Round {round_num} ({strategy}) - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, "
        f"Spearman: {metrics['spearman']:.4f}, Top-10 mean: {metrics['top_n_mean']:.4f}"
    )

    return list(selected_variant_ids), metrics



def get_predictions_and_metrics(
    controller: Controller,
    test_embeddings: np.ndarray,
    test_scores: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Generate predictions and evaluate model performance."""
    predictions = controller.learner.predict(test_embeddings)

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    assert predictions.shape[0] == test_scores.shape[0], (
        f"Prediction shape mismatch: got {predictions.shape}, expected {test_scores.shape}"
    )

    r2 = r2_score(test_scores, predictions)
    rmse = np.sqrt(mean_squared_error(test_scores, predictions))
    spearman_corr, _ = spearmanr(test_scores, predictions)
    top_n = min(10, len(predictions))
    top_indices = np.argsort(predictions)[-top_n:]
    top_n_mean = np.mean([test_scores[i] for i in top_indices])

    metrics = {
        "r2": r2,
        "rmse": rmse,
        "spearman": spearman_corr,
        "top_n_mean": top_n_mean
    }

    return predictions, metrics


def simulate_active_learning(
    config_path: str,
    variants: List[Variant],
    assay_results: List[AssayResult],
    output_dir: str,
    n_rounds: int = 5,
    initial_batch_size: int = 10,
    batch_size: int = 10,
    strategies: List[str] = ["ucb"],
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Simulate active learning on a dataset using different acquisition strategies.
    This is the orchestrator function calling helper modules.
    """
    # Global seeding
    random.seed(random_seed)
    np.random.seed(random_seed)

    results = {
        strategy: {"r2": [], "rmse": [], "spearman": [], "top_n_mean": []}
        for strategy in strategies
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_vars, train_results, test_vars, test_results = split_train_test(
        variants, assay_results, test_fraction, random_seed
    )
    test_scores = np.array([r.score for r in test_results])

    for strategy in strategies:
        logger.info(f"Running strategy: {strategy}")
        controller = initialize_controller(
            config_path, strategy, output_path, initial_batch_size, random_seed
        )

        _, test_embeddings = controller.embedder.embed_variants(test_vars)

        selected_ids, initial_metrics = run_initial_round(
            controller, train_vars, train_results,
            initial_batch_size, test_embeddings, test_scores
        )
        for k, v in initial_metrics.items():
            results[strategy][k].append(v)

        for round_num in range(2, n_rounds + 1):
            selected_ids, metrics = run_active_learning_round(
                controller, train_vars, train_results,
                set(selected_ids), batch_size,
                test_embeddings, test_scores,
                round_num, strategy
            )
            for k, v in metrics.items():
                results[strategy][k].append(v)

    return results
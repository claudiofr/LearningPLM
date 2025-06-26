#!/usr/bin/env python
"""
Benchmark PLM framework on Protein Gym datasets.

This script evaluates the performance of the PLM framework on protein engineering datasets
from Protein Gym, simulating multiple rounds of active learning and measuring performance
metrics across rounds.
"""

import argparse
import logging
import os
import random
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr

from plm_framework.controller import Controller
from plm_framework.datamodels import Variant, AssayResult, ProposedVariant
from plm_framework.config import get_config
from plm_framework.utils import setup_logging

from plm_framework.simulation import simulate_active_learning

# Set up logging
logger = setup_logging()


def load_protein_gym_dataset(
    file_path: str, reference_sequence: Optional[str] = None
) -> Tuple[str, List[Variant], List[AssayResult]]:
    """
    Load a Protein Gym dataset.

    Args:
        file_path: Path to the Protein Gym CSV file
        reference_sequence: Optional reference sequence to use (if not provided, will try to extract from data)

    Returns:
        Tuple of (reference_sequence, variants, assay_results)
    """
    df = pd.read_csv(file_path)

    # Log dataset information
    logger.info(f"Loaded dataset from {file_path} with {len(df)} rows")
    logger.info(f"Dataset columns: {df.columns.tolist()}")

    # Check if DMS_score column exists
    if "DMS_score" not in df.columns:
        logger.error(
            f"DMS_score column not found in dataset! Available columns: {df.columns.tolist()}"
        )
        raise ValueError(f"DMS_score column not found in dataset")

    # Log DMS_score statistics
    logger.info(
        f"DMS_score statistics: min={df['DMS_score'].min()}, max={df['DMS_score'].max()}, mean={df['DMS_score'].mean()}"
    )

    # Check for missing values
    missing_scores = df["DMS_score"].isna().sum()
    if missing_scores > 0:
        logger.warning(f"Found {missing_scores} missing values in DMS_score column")

    # Check for required columns
    required_columns = ["mutant", "mutated_sequence", "DMS_score"]
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Required column '{col}' not found in dataset!")
            raise ValueError(f"Required column '{col}' not found in dataset")

    # Use provided reference sequence if available
    if reference_sequence is None:
        # Try to extract reference sequence from the data
        if "reference_sequence" in df.columns:
            reference_sequence = df["reference_sequence"].iloc[0]
        elif "wildtype_sequence" in df.columns:
            reference_sequence = df["wildtype_sequence"].iloc[0]
        else:
            # Look for a row with 'WT' or 'wildtype' in the mutant column
            wt_rows = df[df["mutant"].str.contains("WT|wildtype", case=False, na=False)]
            if not wt_rows.empty:
                reference_sequence = wt_rows.iloc[0]["mutated_sequence"]
            else:
                # Just use the first sequence as reference
                reference_sequence = df.iloc[0]["mutated_sequence"]
                logger.warning(
                    "Could not identify reference sequence, using first sequence as reference"
                )

    # Create variants and assay results
    variants = []
    assay_results = []

    for i, row in df.iterrows():
        # Create variant
        variant = Variant(
            id=i + 1, name=row["mutant"], sequence=row["mutated_sequence"]
        )
        variants.append(variant)

        # Create assay result
        assay_result = AssayResult(
            variant_id=variant.id,
            score=float(row["DMS_score"]),
            uncertainty=0.1,  # Default uncertainty if not provided
            round_id=0,  # Will be updated during simulation
        )
        assay_results.append(assay_result)

    logger.info(
        f"Created {len(variants)} variants and {len(assay_results)} assay results"
    )

    # Log a few examples
    for i in range(min(5, len(variants))):
        logger.info(
            f"Example {i+1}: Variant ID={variants[i].id}, Name={variants[i].name}, Score={assay_results[i].score:.4f}"
        )

    return reference_sequence, variants, assay_results


# def simulate_active_learning(
#     config_path: str,
#     variants: List[Variant],
#     assay_results: List[AssayResult],
#     output_dir: str,
#     n_rounds: int = 5,
#     initial_batch_size: int = 10,
#     batch_size: int = 10,
#     strategies: List[str] = ["ucb"],
#     test_fraction: float = 0.2,
#     random_seed: int = 42,
# ) -> Dict[str, Dict[str, List[float]]]:
#     """
#     Simulate active learning on a dataset.

#     Args:
#         config_path: Path to configuration file
#         variants: List of variants
#         assay_results: List of assay results
#         output_dir: Directory to save results
#         n_rounds: Number of rounds to simulate
#         initial_batch_size: Number of variants in the initial batch
#         batch_size: Number of variants to select in each round
#         strategies: List of acquisition strategies to test
#         test_fraction: Fraction of data to use for testing
#         random_seed: Random seed for reproducibility

#     Returns:
#         Dictionary of performance metrics for each strategy and round
#     """
#     # Set random seed
#     import random

#     random.seed(random_seed)
#     np.random.seed(random_seed)

#     # Create output directory
#     output_dir = Path(output_dir)
#     os.makedirs(output_dir, exist_ok=True)

#     # Split data into train and test
#     n_variants = len(variants)
#     indices = list(range(n_variants))
#     random.shuffle(indices)

#     test_size = int(n_variants * test_fraction)
#     test_indices = set(indices[:test_size])
#     train_indices = set(indices[test_size:])

#     train_variants = [variants[i] for i in train_indices]
#     train_assay_results = [assay_results[i] for i in train_indices]
#     test_variants = [variants[i] for i in test_indices]
#     test_assay_results = [assay_results[i] for i in test_indices]

#     # Extract test scores from test assay results
#     test_scores = np.array([result.score for result in test_assay_results])

#     # Initialize results dictionary
#     results = {
#         strategy: {
#             "r2": [],
#             "rmse": [],
#             "spearman": [],
#             "top_n_mean": [],
#             "round_variants": [],
#         }
#         for strategy in strategies
#     }

#     # Run simulation for each strategy
#     for strategy in strategies:
#         logger.info(f"Simulating active learning with strategy: {strategy}")

#         # Load configuration
#         config = get_config(config_path)

#         # Update config for this run
#         config.data.db_path = str(output_dir / f"{strategy}_variants_{random_seed}.db")
#         # Delete existing database file if it exists
#         if os.path.exists(config.data.db_path):
#             os.remove(config.data.db_path)
#         config.data.output_dir = str(output_dir / strategy)
#         os.makedirs(config.data.output_dir, exist_ok=True)

#         # Set PCA components to a smaller value based on initial batch size
#         if hasattr(config.model, "pca_components"):
#             # Set PCA components to min(initial_batch_size, current_value)
#             config.model.pca_components = min(
#                 initial_batch_size - 1, getattr(config.model, "pca_components", 128)
#             )
#             logger.info(f"Adjusted PCA components to {config.model.pca_components}")

#         # Create controller
#         controller = Controller(config)

#         # Initialize pool of available variants
#         available_variants = train_variants.copy()
#         selected_variant_ids = set()

#         # Start with a random initial batch
#         initial_indices = random.sample(
#             range(len(available_variants)),
#             min(initial_batch_size, len(available_variants)),
#         )
#         initial_variants = [available_variants[i] for i in initial_indices]

#         # Start round 1
#         controller.start_round("Round 1", "Initial random batch")

#         # Add initial variants to database
#         for variant in initial_variants:
#             controller.data_manager.add_variant(variant)
#             selected_variant_ids.add(variant.id)

#         # Add corresponding assay results
#         initial_assay_results = [
#             AssayResult(
#                 variant_id=result.variant_id,
#                 score=result.score,
#                 uncertainty=result.uncertainty,
#                 round_id=1,  # Update round ID
#             )
#             for result in train_assay_results
#             if result.variant_id in selected_variant_ids
#         ]
#         controller.add_assay_results(initial_assay_results)

#         # Complete round 1
#         controller.complete_round()

#         # Fit initial model
#         controller.fit_model(round_id=1)

#         # Embed test variants
#         _, test_embeddings = controller.embedder.embed_variants(test_variants)

#         # Debug information
#         print(f"Test embeddings shape: {test_embeddings.shape}")
#         print(f"Number of test variants: {len(test_variants)}")
#         print(f"Number of test scores: {len(test_scores)}")

#         # Make sure the learner is properly fitted before prediction
#         if not controller.learner.is_fitted:
#             print("Warning: Model not fitted yet, fitting with available data")
#             controller.fit_model(round_id=1)

#         # Get predictions with explicit handling for different return types
#         try:
#             # Try to get predictions
#             predictions = controller.learner.predict(test_embeddings)

#             # Handle different return types
#             if isinstance(predictions, tuple):
#                 # Extract predictions from tuple
#                 test_predictions = predictions[0]
#             else:
#                 test_predictions = predictions

#             # Ensure test_predictions is a 1D array with the right length
#             if hasattr(test_predictions, "shape") and len(test_predictions.shape) > 1:
#                 test_predictions = test_predictions.flatten()

#             # Truncate or pad to match test_scores length
#             if len(test_predictions) != len(test_scores):
#                 print(
#                     f"Warning: Prediction length mismatch. Adjusting predictions from {len(test_predictions)} to {len(test_scores)}"
#                 )
#                 if len(test_predictions) > len(test_scores):
#                     test_predictions = test_predictions[: len(test_scores)]
#                 else:
#                     # Pad with zeros if predictions are too short
#                     padding = np.zeros(len(test_scores) - len(test_predictions))
#                     test_predictions = np.concatenate([test_predictions, padding])

#             print(
#                 f"Final test_predictions shape: {test_predictions.shape if hasattr(test_predictions, 'shape') else len(test_predictions)}"
#             )

#         except Exception as e:
#             print(f"Error during prediction: {e}")
#             # Fallback to random predictions
#             print("Using random predictions as fallback")
#             test_predictions = np.random.rand(len(test_scores))

#         # Calculate metrics
#         r2 = r2_score(test_scores, test_predictions)
#         rmse = np.sqrt(mean_squared_error(test_scores, test_predictions))
#         spearman_corr, _ = spearmanr(test_scores, test_predictions)

#         # Calculate top-N mean (how good are our top predictions)
#         top_n = min(10, len(test_variants))
#         top_indices = np.argsort(test_predictions)[-top_n:]
#         top_n_mean = np.mean([test_scores[i] for i in top_indices])

#         # Store results
#         results[strategy]["r2"].append(r2)
#         results[strategy]["rmse"].append(rmse)
#         results[strategy]["spearman"].append(spearman_corr)
#         results[strategy]["top_n_mean"].append(top_n_mean)
#         results[strategy]["round_variants"].append(initial_variants)

#         logger.info(
#             f"Round 1 - R²: {r2:.4f}, RMSE: {rmse:.4f}, Spearman: {spearman_corr:.4f}, Top-{top_n} mean: {top_n_mean:.4f}"
#         )

#         # Run active learning rounds
#         for round_num in range(2, n_rounds + 1):
#             # Update available variants (remove already selected ones)
#             available_variants = [
#                 v for v in train_variants if v.id not in selected_variant_ids
#             ]

#             logger.info(
#                 f"Round {round_num}: {len(available_variants)} available variants"
#             )

#             if not available_variants:
#                 logger.warning(f"No more variants available after round {round_num-1}")
#                 break

#             # Start new round
#             controller.start_round(
#                 f"Round {round_num}", f"Active learning round {round_num}"
#             )

#             # Propose variants
#             logger.info(
#                 f"Proposing variants for round {round_num} with strategy {strategy}"
#             )

#             # Debug: Print some available variants
#             logger.info(
#                 f"First 5 available variants: {[v.id for v in available_variants[:5]]}"
#             )

#             # Get variant sequences for ESM logit scoring if needed
#             variant_sequences = {str(v.id): v.sequence for v in available_variants}

#             # Propose variants with more detailed error handling
#             try:
#                 proposed_variants = controller.propose_variants(
#                     candidates=available_variants,
#                     batch_size=batch_size,
#                     strategy=strategy,
#                     temperature=1.0,
#                     variant_sequences=variant_sequences,  # Add this for ESM logit scoring
#                 )

#                 logger.info(
#                     f"Proposed {len(proposed_variants)} variants for round {round_num}"
#                 )

#                 # If no variants were proposed, try with diversity strategy
#                 if not proposed_variants:
#                     logger.warning(
#                         f"No variants proposed with {strategy} strategy, trying diversity"
#                     )
#                     proposed_variants = controller.propose_variants(
#                         candidates=available_variants,
#                         batch_size=batch_size,
#                         strategy="diversity",
#                         temperature=1.0,
#                     )

#                     if not proposed_variants:
#                         logger.error(
#                             "Failed to propose variants with diversity strategy too!"
#                         )
#                         # Add some random variants as a last resort
#                         random_indices = random.sample(
#                             range(len(available_variants)),
#                             min(batch_size, len(available_variants)),
#                         )
#                         proposed_variants = [
#                             ProposedVariant(
#                                 variant=available_variants[i],
#                                 acquisition_score=1.0,
#                                 acquisition_type="random_fallback",
#                                 predicted_score=None,
#                                 predicted_uncertainty=None,
#                             )
#                             for i in random_indices
#                         ]
#                         logger.info(
#                             f"Added {len(proposed_variants)} random variants as fallback"
#                         )

#                         # Add these to the current round
#                         controller.current_round.proposed_variants.extend(
#                             proposed_variants
#                         )

#             except Exception as e:
#                 logger.error(f"Error proposing variants: {e}")
#                 logger.error(traceback.format_exc())
#                 # Add some random variants as a fallback
#                 random_indices = random.sample(
#                     range(len(available_variants)),
#                     min(batch_size, len(available_variants)),
#                 )
#                 proposed_variants = [
#                     ProposedVariant(
#                         variant=available_variants[i],
#                         acquisition_score=1.0,
#                         acquisition_type="random_fallback",
#                         predicted_score=None,
#                         predicted_uncertainty=None,
#                     )
#                     for i in random_indices
#                 ]
#                 logger.info(
#                     f"Added {len(proposed_variants)} random variants as fallback after error"
#                 )

#                 # Add these to the current round
#                 controller.current_round.proposed_variants.extend(proposed_variants)

#             # Print proposed variant details to verify they're different
#             print(f"\nRound {round_num} proposed variants ({strategy}):")
#             # Show first 3 for brevity
#             for i, prop in enumerate(proposed_variants[:3]):
#                 print(
#                     f"  {i+1}. ID: {prop.variant.id}, Score: {prop.acquisition_score:.4f}"
#                 )

#             # Update selected variant IDs
#             for proposed in proposed_variants:
#                 selected_variant_ids.add(proposed.variant.id)

#             # Add corresponding assay results
#             round_assay_results = [
#                 AssayResult(
#                     variant_id=result.variant_id,
#                     score=result.score,
#                     uncertainty=result.uncertainty,
#                     round_id=round_num,  # Update round ID
#                 )
#                 for result in train_assay_results
#                 if result.variant_id in [p.variant.id for p in proposed_variants]
#             ]

#             # Check if we found any matching assay results
#             if not round_assay_results:
#                 logger.warning(
#                     f"No matching assay results found for proposed variants in round {round_num}"
#                 )
#                 logger.warning(
#                     f"Proposed variant IDs: {[p.variant.id for p in proposed_variants]}"
#                 )
#                 logger.warning(
#                     f"First few train result IDs: {[r.variant_id for r in train_assay_results[:5]]}"
#                 )

#                 # Create synthetic results as fallback
#                 round_assay_results = [
#                     AssayResult(
#                         variant_id=p.variant.id,
#                         # Generate random score based on dataset mean/std
#                         score=np.random.normal(-0.3, 0.8),
#                         uncertainty=0.1,
#                         round_id=round_num,
#                     )
#                     for p in proposed_variants
#                 ]
#                 logger.info(
#                     f"Created {len(round_assay_results)} synthetic assay results as fallback"
#                 )

#             # Add assay results to database
#             controller.add_assay_results(round_assay_results)
#             logger.info(
#                 f"Added {len(round_assay_results)} assay results for round {round_num}"
#             )

#             # Complete round
#             controller.complete_round()
#             logger.info(f"Completed round {round_num}")

#             # Fit model with all data
#             try:
#                 controller.fit_model(round_id=round_num)
#                 logger.info(f"Fitted model for round {round_num}")
#             except Exception as e:
#                 logger.error(f"Error fitting model in round {round_num}: {e}")
#                 logger.error(traceback.format_exc())
#                 logger.warning("Continuing with previous model")

#             # Add this after controller.fit_model() in both round 1 and subsequent rounds
#             logger.info(f"Model parameters after fitting in round {round_num}:")
#             if hasattr(controller.learner.model, "coef_"):
#                 logger.info(
#                     f"Model coefficients shape: {controller.learner.model.coef_.shape}"
#                 )
#                 logger.info(
#                     f"First few coefficients: {controller.learner.model.coef_[:5]}"
#                 )
#             elif hasattr(controller.learner.model, "estimators_"):
#                 logger.info(
#                     f"Number of estimators: {len(controller.learner.model.estimators_)}"
#                 )
#                 if hasattr(controller.learner.model.estimators_[0], "coef_"):
#                     logger.info(
#                         f"First estimator coefficients: {controller.learner.model.estimators_[0].coef_[:5]}"
#                     )

#             # Get predictions with explicit handling for different return types
#             try:
#                 # Try to get predictions
#                 predictions = controller.learner.predict(test_embeddings)

#                 # Handle different return types
#                 if isinstance(predictions, tuple):
#                     # Extract predictions from tuple
#                     test_predictions = predictions[0]
#                 else:
#                     test_predictions = predictions

#                 # Ensure test_predictions is a 1D array with the right length
#                 if (
#                     hasattr(test_predictions, "shape")
#                     and len(test_predictions.shape) > 1
#                 ):
#                     test_predictions = test_predictions.flatten()

#                 # Truncate or pad to match test_scores length
#                 if len(test_predictions) != len(test_scores):
#                     print(
#                         f"Warning: Prediction length mismatch. Adjusting predictions from {len(test_predictions)} to {len(test_scores)}"
#                     )
#                     if len(test_predictions) > len(test_scores):
#                         test_predictions = test_predictions[: len(test_scores)]
#                     else:
#                         # Pad with zeros if predictions are too short
#                         padding = np.zeros(len(test_scores) - len(test_predictions))
#                         test_predictions = np.concatenate([test_predictions, padding])

#                 print(
#                     f"Final test_predictions shape: {test_predictions.shape if hasattr(test_predictions, 'shape') else len(test_predictions)}"
#                 )

#             except Exception as e:
#                 print(f"Error during prediction: {e}")
#                 # Fallback to random predictions
#                 print("Using random predictions as fallback")
#                 test_predictions = np.random.rand(len(test_scores))

#             # Calculate metrics
#             r2 = r2_score(test_scores, test_predictions)
#             rmse = np.sqrt(mean_squared_error(test_scores, test_predictions))
#             spearman_corr, _ = spearmanr(test_scores, test_predictions)

#             # Calculate top-N mean
#             top_indices = np.argsort(test_predictions)[-top_n:]
#             top_n_mean = np.mean([test_scores[i] for i in top_indices])

#             # Store results
#             results[strategy]["r2"].append(r2)
#             results[strategy]["rmse"].append(rmse)
#             results[strategy]["spearman"].append(spearman_corr)
#             results[strategy]["top_n_mean"].append(top_n_mean)
#             results[strategy]["round_variants"].append(
#                 [p.variant for p in proposed_variants]
#             )

#             logger.info(
#                 f"Round {round_num} - R²: {r2:.4f}, RMSE: {rmse:.4f}, Spearman: {spearman_corr:.4f}, Top-{top_n} mean: {top_n_mean:.4f}"
#             )

#             if round_num > 1:  # Only compare after first round
#                 # Store previous predictions for comparison
#                 if "prev_predictions" not in locals():
#                     prev_predictions = test_predictions.copy()
#                 else:
#                     # Calculate difference between current and previous predictions
#                     pred_diff = np.mean(np.abs(test_predictions - prev_predictions))
#                     logger.info(
#                         f"Mean absolute difference in predictions from previous round: {pred_diff:.6f}"
#                     )
#                     prev_predictions = test_predictions.copy()

#             # After calculating metrics in each round:
#             logger.info(
#                 f"Test scores: min={np.min(test_scores):.4f}, max={np.max(test_scores):.4f}, mean={np.mean(test_scores):.4f}"
#             )
#             logger.info(
#                 f"Predictions: min={np.min(test_predictions):.4f}, max={np.max(test_predictions):.4f}, mean={np.mean(test_predictions):.4f}"
#             )

#             # Add more detailed metrics
#             from scipy.stats import pearsonr

#             pearson_corr, _ = pearsonr(test_scores, test_predictions)
#             logger.info(f"Additional metrics - Pearson correlation: {pearson_corr:.4f}")

#             # Print a few example predictions vs actual
#             sample_size = min(5, len(test_scores))
#             sample_indices = np.random.choice(
#                 len(test_scores), sample_size, replace=False
#             )
#             for i in sample_indices:
#                 logger.info(
#                     f"Sample {i}: Actual={test_scores[i]:.4f}, Predicted={test_predictions[i]:.4f}"
#                 )

#             # After proposing variants in each round:
#             proposed_ids = [p.variant.id for p in proposed_variants]
#             logger.info(f"Proposed variant IDs: {proposed_ids[:5]}...")

#             # Check for overlap with previously selected variants
#             if round_num > 2:
#                 previous_variants = results[strategy]["round_variants"][-1]
#                 previous_ids = [v.id for v in previous_variants]
#                 overlap = set(proposed_ids).intersection(set(previous_ids))
#                 logger.info(
#                     f"Overlap with previous round: {len(overlap)}/{len(proposed_ids)} variants"
#                 )

#     # Plot results
#     plot_results(results, output_dir)

#     return results


def plot_results(results: Dict[str, Dict[str, List[float]]], output_dir: Path) -> None:
    """
    Plot benchmark results.

    Args:
        results: Dictionary of results
        output_dir: Directory to save plots
    """
    metrics = ["r2", "rmse", "spearman", "top_n_mean"]
    metric_names = ["R²", "RMSE", "Spearman Correlation", "Top-N Mean Score"]

    plt.figure(figsize=(15, 10))

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        plt.subplot(2, 2, i + 1)

        for strategy, strategy_results in results.items():
            rounds = list(range(1, len(strategy_results[metric]) + 1))
            plt.plot(rounds, strategy_results[metric], marker="o", label=strategy)

        plt.xlabel("Round")
        plt.ylabel(name)
        plt.title(f"{name} vs. Round")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_results.png", dpi=300)
    plt.close()

    # Save results to CSV
    for strategy, strategy_results in results.items():
        df = pd.DataFrame(
            {
                "round": list(range(1, len(strategy_results["r2"]) + 1)),
                "r2": strategy_results["r2"],
                "rmse": strategy_results["rmse"],
                "spearman": strategy_results["spearman"],
                "top_n_mean": strategy_results["top_n_mean"],
            }
        )
        df.to_csv(output_dir / f"{strategy}_results.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PLM framework on Protein Gym datasets"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to Protein Gym dataset CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=2,  # Reduced from 5 to 2
        help="Number of rounds to simulate",
    )
    parser.add_argument(
        "--initial-batch-size",
        type=int,
        default=5,  # Reduced from 10 to 5
        help="Number of variants in the initial batch",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,  # Reduced from 10 to 5
        help="Number of variants to select in each round",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="ucb",  # Simplified to just one strategy
        help="Comma-separated list of acquisition strategies",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,  # Reduced from 0.2 to 0.1
        help="Fraction of data to use for testing",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--reference-sequence",
        type=str,
        default=None,
        help="Reference protein sequence (if not provided, will try to extract from data)",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=100,  # Added parameter to limit dataset size
        help="Maximum number of variants to use from dataset (0 for all)",
    )

    args = parser.parse_args()

    # Parse strategies
    strategies = args.strategies.split(",")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    reference_sequence, variants, assay_results = load_protein_gym_dataset(
        args.dataset, reference_sequence=args.reference_sequence
    )

    # Limit dataset size if specified
    if args.max_variants > 0 and len(variants) > args.max_variants:
        logger.info(
            f"Limiting dataset to {args.max_variants} variants (from {len(variants)})"
        )
        # Use a fixed random seed for reproducibility
        random.seed(args.random_seed)
        # Select random subset of variants
        indices = random.sample(range(len(variants)), args.max_variants)
        variants = [variants[i] for i in indices]
        assay_results = [assay_results[i] for i in indices]

    # Run simulation
    logger.info(
        f"Starting benchmark with {len(variants)} variants and {args.n_rounds} rounds"
    )
    results = simulate_active_learning(
        config_path=args.config,
        variants=variants,
        assay_results=assay_results,
        output_dir=args.output_dir,
        n_rounds=args.n_rounds,
        initial_batch_size=args.initial_batch_size,
        batch_size=args.batch_size,
        strategies=strategies,
        test_fraction=args.test_fraction,
        random_seed=args.random_seed,
    )

    logger.info(f"Benchmark complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

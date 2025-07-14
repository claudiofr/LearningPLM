"""Controller for active learning loop."""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from omegaconf import DictConfig

from plm_framework.acquisition import batch_acquisition
from plm_framework.data_manager import DataManager
from plm_framework.datamodels import AssayResult, ProposedVariant, Round, Variant
from plm_framework.embedding import ProteinEmbedder
from plm_framework.learners.active import MLPLearner, RidgeLearner
from plm_framework.learners.base import BaseLearner
from plm_framework.learners.rl import RLPolicyLearner
# from plm_framework.strategies import STRATEGY_LOOKUP
from plm_framework.strategies import StrategyFactory


logger = logging.getLogger(__name__)


class Controller:
    """Controller for active learning loop."""

    def __init__(self, config: DictConfig):
        """
        Initialize the controller.

        Args:
            config: Configuration
        """
        self.config = config

        # Initialize data manager
        self.data_manager = DataManager(config.data.db_path)

        # Initialize embedder
        self.embedder = ProteinEmbedder(
            model_name=config.model.name,
            device=config.training.device if config.training.device != "auto" else None,
            batch_size=config.training.batch_size,
            use_pooling=config.model.use_pooling,
        )

        # Initialize learner
        self.learner = self._create_learner()

        # Current round
        self.current_round = None

    def _create_learner(self) -> BaseLearner:
        """
        Create a learner based on configuration.

        Returns:
            Learner instance
        """
        if self.config.get("learner", {}).get("type", "ridge") == "ridge":
            return RidgeLearner(
                embedding_dim=self.config.model.embedding_dim,
                reduced_dim=self.config.model.reduced_dim,
                alpha=self.config.get("learner", {}).get("alpha", 1.0),
                n_estimators=self.config.get(
                    "learner", {}).get("n_estimators", 10),
                random_state=self.config.get(
                    "training", {}).get("random_state", 42),
            )
        elif self.config.get("learner", {}).get("type") == "mlp":
            return MLPLearner(
                embedding_dim=self.config.model.embedding_dim,
                reduced_dim=self.config.model.reduced_dim,
                hidden_layer_sizes=self.config.get("learner", {}).get(
                    "hidden_layer_sizes", (64, 32)),
                alpha=self.config.get("learner", {}).get("alpha", 0.0001),
                learning_rate_init=self.config.get(
                    "learner", {}).get("learning_rate_init", 0.001),
                max_iter=self.config.get("learner", {}).get("max_iter", 200),
                n_dropout_samples=self.config.get(
                    "learner", {}).get("n_dropout_samples", 10),
                random_state=self.config.get(
                    "training", {}).get("random_state", 42),
            )
        elif self.config.get("learner", {}).get("type") == "rl":
            return RLPolicyLearner(
                embedding_dim=self.config.model.embedding_dim,
                reduced_dim=self.config.model.reduced_dim,
                hidden_dim=self.config.get("rl", {}).get("hidden_dim", 64),
                action_dim=self.config.get("rl", {}).get("action_dim", 20),
                gamma=self.config.get("rl", {}).get("gamma", 0.99),
                entropy_coef=self.config.get(
                    "rl", {}).get("entropy_coef", 0.01),
                learning_rate=self.config.get(
                    "training", {}).get("learning_rate", 0.001),
                max_mutations=self.config.get(
                    "rl", {}).get("max_mutations", 5),
                mutation_penalty=self.config.get(
                    "rl", {}).get("mutation_penalty", 0.1),
                device=self.config.get("training", {}).get("device"),
                random_state=self.config.get(
                    "training", {}).get("random_state", 42),
            )
        else:
            raise ValueError(
                f"Unknown learner type: {self.config.get('learner', {}).get('type')}")

    def start_round(self, name: str, description: Optional[str] = None) -> Round:
        """
        Start a new round of active learning.

        Args:
            name: Round name
            description: Round description

        Returns:
            Round object
        """
        # Get next round ID
        round_id = self.data_manager.get_next_round_id()

        # Create round
        round_obj = Round(
            id=round_id,
            name=name,
            description=description,
            proposed_variants=[],  # Explicitly initialize with empty list
        )

        logger.info(f"Starting round {round_id}: {name}")

        # Save round
        self.data_manager.add_round(round_obj)

        self.current_round = round_obj
        logger.info(f"Started round {round_id}: {name}")

        return round_obj

    def complete_round(self) -> None:
        """Complete the current round."""
        if self.current_round is None:
            raise ValueError("No active round")

        # Update round
        self.data_manager.complete_round(self.current_round.id)
        logger.info(
            f"Completed round {self.current_round.id}: {self.current_round.name}")

        self.current_round = None

    def fit_model(self, round_id: Optional[int] = None) -> None:
        """
        Fit the model using data from the specified round.

        Args:
            round_id: Round ID (None for all rounds)
        """
        # Get assay results
        results = self.data_manager.get_assay_results(round_id=round_id)

        if not results:
            logger.warning("No assay results found for fitting")
            return

        # Get variants
        variant_ids = [r.variant_id for r in results]
        variants = self.data_manager.get_variants(variant_ids=variant_ids)

        logger.info(
            f"Fitting model with {len(variants)} variants from {'round ' + str(round_id) if round_id else 'all rounds'}")

        # Get embeddings
        variant_ids, embeddings = self.embedder.embed_variants(variants)

        # Check if we got embeddings for all variants
        if len(variant_ids) != len(variants):
            logger.warning(
                f"Only got embeddings for {len(variant_ids)}/{len(variants)} variants")

        # Map scores to embeddings
        scores = np.array([
            next((r.score for r in results if r.variant_id == vid), 0.0)
            for vid in variant_ids
        ])

        # Map uncertainties to embeddings
        uncertainties = np.array([
            next((r.uncertainty for r in results if r.variant_id == vid), 0.0)
            for vid in variant_ids
        ])

        # Check if we have any non-zero scores
        if np.all(scores == 0.0):
            logger.warning(
                "All scores are zero, model will not learn anything useful")
            return

        # Log score distribution
        logger.info(
            f"Score distribution: min={np.min(scores):.4f}, max={np.max(scores):.4f}, mean={np.mean(scores):.4f}")

        # Fit model
        logger.info(f"Fitting model with {len(scores)} samples")
        try:
            self.learner.fit(embeddings, scores, uncertainties)
            logger.info("Model fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting model: {e}")
            # If fitting fails, try with a simpler model or approach
            if hasattr(self.learner, 'alpha') and self.learner.alpha < 10.0:
                logger.info("Increasing regularization and retrying")
                self.learner.alpha *= 10.0
                self.learner.fit(embeddings, scores, uncertainties)

    def propose_variants(
        self,
        candidates: List[Variant],
        batch_size: int = 10,
        strategy: str = "ucb",
        temperature: float = 1.0,
        mutation_range: Optional[str] = None,
        **kwargs,
    ) -> List[ProposedVariant]:
        """
        Propose variants for the next batch.

        Args:
            candidates: List of candidate variants
            batch_size: Number of variants to propose
            strategy: Acquisition strategy
            temperature: Temperature for softmax
            mutation_range: Range of residues to mutate (e.g., '1-100' or '5,10,15-20')
            **kwargs: Additional parameters for acquisition function

        Returns:
            List of proposed variants
        """
        if self.current_round is None:
            raise ValueError("No active round")

        logger.info(
            f"Proposing variants with strategy: {strategy}, batch_size: {batch_size}")
        logger.info(f"Number of candidate variants: {len(candidates)}")

        # If mutation range is specified, filter candidates to only include those with mutations in the range
        if mutation_range and strategy in ["esm_logit", "diversity", "random"]:
            from plm_framework.utils import parse_mutation_range

            # Ensure mutation_range is a string
            mutation_range_str = mutation_range if isinstance(
                mutation_range, str) else None

            # Get the original sequence if provided
            original_sequence = kwargs.get("original_sequence")
            if original_sequence:
                # Parse mutation range
                sequence_length = len(original_sequence)
                mutable_positions = parse_mutation_range(
                    mutation_range_str, sequence_length)

                # Add mutation range to kwargs for use in acquisition functions
                kwargs["mutable_positions"] = mutable_positions

                logger.info(
                    f"Restricting mutations to {len(mutable_positions)} positions based on mutation range")

        # Check if candidates list is empty
        if not candidates:
            logger.error("No candidate variants provided!")
            return []

        # Get embeddings
        try:
            variant_ids, embeddings = self.embedder.embed_variants(candidates)
            logger.info(
                f"Got embeddings for {len(variant_ids)} variants with shape {embeddings.shape}")

            try:
                strategy_instance = StrategyFactory.create_instance(strategy, **kwargs)
                proposed_variants = strategy_instance.propose(candidates, embeddings, batch_size, self.learner)
            except Exception as e:
                logger.error(f"Failed to apply strategy {strategy}: {e}")
                logger.warning("Falling back to diversity strategy")
                strategy_instance = StrategyFactory.create_instance("diversity")
                proposed_variants = strategy_instance.propose(candidates, embeddings, batch_size, self.learner)



            # Check if embeddings were successfully generated
            if len(variant_ids) == 0 or embeddings.shape[0] == 0:
                logger.error("Failed to generate embeddings for candidates!")
                return []

        except Exception as e:
            logger.error(f"Error embedding variants: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

        # # Use acquisition function to select variants
        # try:
        #     # Check if model is fitted
        #     if hasattr(self, 'learner') and self.learner.is_fitted and strategy not in ["diversity", "esm_logit", "random"]:
        #         logger.info("Using model-based acquisition")
        #         # Get predictions and uncertainties for all candidates
        #         predictions, uncertainties = self.learner.predict(
        #             embeddings, return_std=True)
        #         logger.info(
        #             f"Generated predictions for {len(predictions)} variants")
        #         logger.info(
        #             f"Prediction range: min={np.min(predictions):.4f}, max={np.max(predictions):.4f}, mean={np.mean(predictions):.4f}")
        #         logger.info(
        #             f"Uncertainty range: min={np.min(uncertainties):.4f}, max={np.max(uncertainties):.4f}, mean={np.mean(uncertainties):.4f}")

        #         # Use acquisition function
        #         acquisition_scores = self.acquisition_function(
        #             variant_ids=variant_ids,
        #             embeddings=embeddings,
        #             predictions=predictions,
        #             uncertainties=uncertainties,
        #             strategy=strategy,
        #             temperature=temperature,
        #             **kwargs,
        #         )
        #     # else:
        #     #     # Use diversity sampling or ESM logit sampling if model is not fitted
        #     #     actual_strategy = strategy if strategy in [
        #     #         "diversity", "esm_logit", "random"] else "diversity"
        #     #     logger.info(
        #     #         f"Model not fitted or using non-model strategy. Using {actual_strategy} sampling")

        #     #     acquisition_scores = self.acquisition_function(
        #     #         variant_ids=variant_ids,
        #     #         embeddings=embeddings,
        #     #         strategy=actual_strategy,
        #     #         temperature=temperature,
        #     #         **kwargs,
        #     #     )
        #     #     # Update strategy to reflect what was actually used
        #     #     strategy = actual_strategy

        #     logger.info(
        #         f"Generated acquisition scores for {len(acquisition_scores)} variants")
        #     logger.info(
        #         f"Score range: min={np.min(acquisition_scores):.4f}, max={np.max(acquisition_scores):.4f}, mean={np.mean(acquisition_scores):.4f}")

        #     # Check if acquisition scores were successfully generated
        #     if len(acquisition_scores) == 0:
        #         logger.error("Failed to generate acquisition scores!")
        #         return []

        except Exception as e:
            logger.error(f"Error in acquisition function: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Use random scores as fallback
            logger.warning("Using random acquisition scores as fallback")
            acquisition_scores = np.random.rand(len(variant_ids))
            strategy = "random_fallback"

        # Create proposed variants
        try:
            proposed_variants = []
            for i, (vid, score) in enumerate(zip(variant_ids, acquisition_scores)):
                # Find the variant with matching ID
                matching_variants = [
                    v for v in candidates if str(v.id) == str(vid)]
                if not matching_variants:
                    logger.warning(f"No matching variant found for ID {vid}")
                    continue

                variant = matching_variants[0]

                # Get predicted score and uncertainty if model is fitted
                predicted_score = None
                predicted_uncertainty = None
                if hasattr(self, 'learner') and self.learner.is_fitted:
                    try:
                        pred, unc = self.learner.predict_single(
                            embeddings[i], return_std=True)
                        predicted_score = float(pred)
                        predicted_uncertainty = float(unc)
                    except Exception as e:
                        logger.warning(
                            f"Error getting prediction for variant {vid}: {e}")

                # Create proposed variant
                proposed_variants.append(
                    ProposedVariant(
                        variant=variant,
                        acquisition_score=float(score),
                        acquisition_type=strategy,
                        predicted_score=predicted_score,
                        predicted_uncertainty=predicted_uncertainty,
                    )
                )

            logger.info(f"Created {len(proposed_variants)} proposed variants")

            # Sort by acquisition score (higher is better)
            proposed_variants.sort(
                key=lambda x: x.acquisition_score, reverse=True)

            # Limit to batch size
            proposed_variants = proposed_variants[:batch_size]

            logger.info(
                f"Selected {len(proposed_variants)} variants after limiting to batch size {batch_size}")

            # Save proposed variants to current round
            if proposed_variants:
                self.current_round.proposed_variants.extend(proposed_variants)
                logger.info(
                    f"Added {len(proposed_variants)} variants to current round (now has {len(self.current_round.proposed_variants)} total)")
            else:
                logger.error("No variants were selected for proposal!")

            return proposed_variants

        except Exception as e:
            logger.error(f"Error creating proposed variants: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def add_assay_results(self, results: List[AssayResult]) -> None:
        """
        Add assay results to the database.

        Args:
            results: List of assay results
        """
        if self.current_round is None:
            raise ValueError("No active round")

        # Set round ID if not set
        for result in results:
            if result.round_id == 0:
                result.round_id = self.current_round.id

        # Add results
        for result in results:
            self.data_manager.add_assay_result(result)

        logger.info(f"Added {len(results)} assay results")

    def run_active_learning_loop(
        self,
        candidates: List[Variant],
        initial_variants: Optional[List[Variant]] = None,
        initial_results: Optional[List[AssayResult]] = None,
        n_rounds: int = 5,
        batch_size: int = 10,
        strategy: str = "ucb",
        temperature: float = 1.0,
        round_name_prefix: str = "Round",
        **kwargs,
    ) -> None:
        """
        Run the active learning loop.

        Args:
            candidates: List of candidate variants
            initial_variants: Initial variants to add to the database
            initial_results: Initial assay results to add to the database
            n_rounds: Number of rounds to run
            batch_size: Number of variants to propose in each round
            strategy: Acquisition strategy
            temperature: Temperature for softmax
            round_name_prefix: Prefix for round names
            **kwargs: Additional parameters for acquisition function
        """
        # Add initial variants to the database
        if initial_variants:
            for variant in initial_variants:
                self.data_manager.add_variant(variant)
            logger.info(f"Added {len(initial_variants)} initial variants")

        # Get the current highest round ID to ensure we start with a clean sequence
        current_max_round_id = self.data_manager.get_next_round_id() - 1
        logger.info(
            f"Current highest round ID in database: {current_max_round_id}")

        # Start first round
        first_round_id = current_max_round_id + 1
        self.start_round(f"{round_name_prefix} 1", "Initial round")
        logger.info(f"Started first round with ID {self.current_round.id}")

        # Add initial results to the database if available
        if initial_results:
            self.add_assay_results(initial_results)
            logger.info(f"Added {len(initial_results)} initial assay results")

            # Fit model with initial data
            self.fit_model(self.current_round.id)
        else:
            logger.info(
                "No initial results provided. Will use diversity-based selection for first round.")
            # Model will not be fitted, and propose_variants will use diversity-based selection

        # Complete first round
        self.complete_round()

        # Run active learning loop
        for i in range(2, n_rounds + 2):  # Start from 2 since we already did round 1
            logger.info(f"Starting round {i}")

            # Check if we have any candidates left
            if not candidates:
                logger.error(f"No candidates left for round {i}!")
                break

            # Start new round
            self.start_round(f"{round_name_prefix} {i}",
                             f"Active learning round {i-1}")
            logger.info(f"Started round {i} with ID {self.current_round.id}")

            # Propose variants
            logger.info(
                f"Proposing variants for round {i} with {len(candidates)} candidates")
            proposed_variants = self.propose_variants(
                candidates,
                batch_size=batch_size,
                strategy=strategy,
                temperature=temperature,
                **kwargs,
            )

            # Check if any variants were proposed
            if not proposed_variants:
                logger.error(f"No variants were proposed for round {i}!")
                # Try with a different strategy as fallback
                logger.info(f"Trying with 'diversity' strategy as fallback")
                proposed_variants = self.propose_variants(
                    candidates,
                    batch_size=batch_size,
                    strategy="diversity",
                    temperature=temperature,
                    **kwargs,
                )

                if not proposed_variants:
                    logger.error(
                        "Fallback strategy also failed to propose variants!")
                    # Break the loop if we still can't propose variants
                    break

            logger.info(
                f"Round {i}: Proposed {len(proposed_variants)} variants")

            # In a real scenario, we would wait for assay results here
            # For simulation, we can generate synthetic results
            logger.info(f"Waiting for assay results for round {i}...")

            # Complete round
            self.complete_round()

            # Fit model with all data if we have any results
            if self.data_manager.get_assay_results():
                self.fit_model()

            logger.info(f"Completed round {i}")

        logger.info(f"Completed active learning loop with {n_rounds+1} rounds")

    def evaluate_model(
        self,
        test_variants: List[Variant],
        test_results: List[AssayResult],
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            test_variants: List of test variants
            test_results: List of test assay results

        Returns:
            Dictionary of evaluation metrics
        """
        # Get embeddings
        variant_ids, embeddings = self.embedder.embed_variants(test_variants)

        # Map scores to embeddings
        scores = np.array([
            next((r.score for r in test_results if r.variant_id == vid), 0.0)
            for vid in variant_ids
        ])

        # Filter out missing scores
        mask = scores != 0.0
        embeddings = embeddings[mask]
        scores = scores[mask]

        # Get predictions
        predictions, uncertainties = self.learner.predict(embeddings)

        # Calculate metrics
        from sklearn.metrics import mean_squared_error, r2_score

        mse = mean_squared_error(scores, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(scores, predictions)

        # Calculate Spearman correlation
        from scipy.stats import spearmanr

        spearman_corr, _ = spearmanr(scores, predictions)

        # Calculate top-k hit rate
        def top_k_hit_rate(true_scores, pred_scores, k):
            true_top_k = np.argsort(true_scores)[-k:]
            pred_top_k = np.argsort(pred_scores)[-k:]
            return len(np.intersect1d(true_top_k, pred_top_k)) / k

        top_10_hit_rate = top_k_hit_rate(
            scores, predictions, min(10, len(scores)))
        top_50_hit_rate = top_k_hit_rate(
            scores, predictions, min(50, len(scores)))

        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "spearman_corr": float(spearman_corr),
            "top_10_hit_rate": float(top_10_hit_rate),
            "top_50_hit_rate": float(top_50_hit_rate),
        }

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model
        """
        import pickle

        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)

        # Save model
        with open(path, "wb") as f:
            pickle.dump(self.learner, f)

        logger.info(f"Saved model to {path}")

    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load the model from disk.

        Args:
            path: Path to load the model from
        """
        import pickle

        path = Path(path)

        # Load model
        with open(path, "rb") as f:
            self.learner = pickle.load(f)

        logger.info(f"Loaded model from {path}")

    def export_results(self, output_dir: Union[str, Path]) -> None:
        """
        Export results to CSV files.

        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Export data
        self.data_manager.export_to_csv(output_dir)

        logger.info(f"Exported results to {output_dir}")

    # def acquisition_function(
    #     self,
    #     variant_ids: List[str],
    #     embeddings: np.ndarray,
    #     predictions: Optional[np.ndarray] = None,
    #     uncertainties: Optional[np.ndarray] = None,
    #     strategy: str = "ucb",
    #     temperature: float = 1.0,
    #     **kwargs,
    # ) -> np.ndarray:
    #     """
    #     Acquisition function for selecting variants.

    #     Args:
    #         variant_ids: List of variant IDs
    #         embeddings: Embeddings for variants
    #         predictions: Optional predictions for variants
    #         uncertainties: Optional uncertainties for variants
    #         strategy: Acquisition strategy
    #         temperature: Temperature for softmax
    #         **kwargs: Additional parameters for acquisition function

    #     Returns:
    #         Acquisition scores for variants
    #     """
    #     from plm_framework.acquisition import batch_acquisition

    #     logger.info(f"Running acquisition function with strategy: {strategy}")

    #     try:
    #         # Run acquisition function
    #         acquisition_scores = batch_acquisition(
    #             variant_ids=variant_ids,
    #             embeddings=embeddings,
    #             predictions=predictions,
    #             uncertainties=uncertainties,
    #             strategy=strategy,
    #             temperature=temperature,
    #             **kwargs,
    #         )

    #         logger.info(
    #             f"Generated {len(acquisition_scores)} acquisition scores")
    #         logger.info(
    #             f"Score range: min={np.min(acquisition_scores):.4f}, max={np.max(acquisition_scores):.4f}, mean={np.mean(acquisition_scores):.4f}")

    #         return acquisition_scores

    #     except Exception as e:
    #         logger.error(f"Error in batch_acquisition: {e}")
    #         # Return uniform scores as fallback
    #         logger.warning("Using uniform scores as fallback")
    #         return np.ones(len(variant_ids))
        

    # def _propose_custom_variants(
    #     self,
    #     candidates: List[Variant],
    #     embeddings: np.ndarray,
    #     batch_size: int,
    #     strategy: str
    # ) -> List[ProposedVariant]:
    #     if strategy not in STRATEGY_LOOKUP:
    #         raise ValueError(f"Unknown custom strategy: {strategy}")

    #     strategy_instance = STRATEGY_LOOKUP[strategy]

    #     # Defensive learner checks
    #     if strategy == "qbc" and not hasattr(self.learner, "predict_committee"):
    #         logger.error("Learner does not support QBC. Falling back to diversity.")
    #         return STRATEGY_LOOKUP["diversity"].propose(candidates, embeddings, batch_size, self.learner)

    #     if strategy in ["ucb", "uncertainty"]:
    #         if not hasattr(self.learner, "predict"):
    #             logger.error(f"Learner does not support predict. Falling back to diversity.")
    #             return STRATEGY_LOOKUP["diversity"].propose(candidates, embeddings, batch_size, self.learner)

    #         # Try calling predict with return_std=True once to confirm support
    #         try:
    #             _ = self.learner.predict(embeddings[:1], return_std=True)
    #         except TypeError:
    #             logger.error(f"Learner.predict does not support return_std=True. Falling back to diversity.")
    #             return STRATEGY_LOOKUP["diversity"].propose(candidates, embeddings, batch_size, self.learner)
    #         except Exception as e:
    #             logger.error(f"Learner.predict failed: {e}. Falling back to diversity.")
    #             return STRATEGY_LOOKUP["diversity"].propose(candidates, embeddings, batch_size, self.learner)

    #     return strategy_instance.propose(candidates, embeddings, batch_size, self.learner)

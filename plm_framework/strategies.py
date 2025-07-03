from abc import ABC, abstractmethod
from typing import List
import numpy as np
import random

from plm_framework.datamodels import Variant, ProposedVariant
from plm_framework.learners.base import BaseLearner


class ProposalStrategy(ABC):
    """Abstract base class for proposal strategies."""
    @abstractmethod
    def propose(
        self,
        candidates: List[Variant],
        embeddings: np.ndarray,
        batch_size: int,
        learner: BaseLearner
    ) -> List[ProposedVariant]:
        pass


class RandomStrategy(ProposalStrategy):
    def propose(self, candidates, embeddings, batch_size, learner):
        selected = random.sample(candidates, min(batch_size, len(candidates)))
        return [
            ProposedVariant(variant=v, acquisition_score=0.0, acquisition_type="random")
            for v in selected
        ]


class GreedyStrategy(ProposalStrategy):
    def propose(self, candidates, embeddings, batch_size, learner):
        preds = learner.predict(embeddings)
        if isinstance(preds, tuple):
            preds = preds[0]
        top_indices = np.argsort(preds)[-batch_size:]
        selected = [candidates[i] for i in top_indices]
        return [
            ProposedVariant(variant=v, acquisition_score=float(preds[i]), acquisition_type="greedy")
            for i, v in zip(top_indices, selected)
        ]


class UncertaintyStrategy(ProposalStrategy):
    def propose(self, candidates, embeddings, batch_size, learner):
        preds, stds = learner.predict(embeddings, return_std=True)
        top_indices = np.argsort(stds)[-batch_size:]
        selected = [candidates[i] for i in top_indices]
        return [
            ProposedVariant(variant=v, acquisition_score=float(stds[i]), acquisition_type="uncertainty")
            for i, v in zip(top_indices, selected)
        ]


class DiversityStrategy(ProposalStrategy):
    def propose(self, candidates, embeddings, batch_size, learner):
        chosen = []
        chosen_idx = []
        i = random.randint(0, len(embeddings) - 1)
        chosen.append(candidates[i])
        chosen_idx.append(i)

        while len(chosen) < batch_size:
            dists = np.array([
                min(np.linalg.norm(embeddings[i] - embeddings[j]) for j in chosen_idx)
                for i in range(len(embeddings))
            ])
            dists[chosen_idx] = -1
            next_idx = np.argmax(dists)
            chosen.append(candidates[next_idx])
            chosen_idx.append(next_idx)

        return [
            ProposedVariant(variant=v, acquisition_score=0.0, acquisition_type="diversity")
            for v in chosen
        ]


class QBCStrategy(ProposalStrategy):
    def propose(self, candidates, embeddings, batch_size, learner):
        # Example QBC: assume learner has predict_committee()
        _, committee_stds = learner.predict_committee(embeddings)
        top_indices = np.argsort(committee_stds)[-batch_size:]
        selected = [candidates[i] for i in top_indices]
        return [
            ProposedVariant(variant=v, acquisition_score=float(committee_stds[i]), acquisition_type="qbc")
            for i, v in zip(top_indices, selected)
        ]


class UCBStrategy(ProposalStrategy):
    def __init__(self, lambda_coef=1.0):
        self.lambda_coef = lambda_coef

    def propose(self, candidates, embeddings, batch_size, learner):
        preds, stds = learner.predict(embeddings, return_std=True)
        ucb_scores = preds + self.lambda_coef * stds
        top_indices = np.argsort(ucb_scores)[-batch_size:]
        selected = [candidates[i] for i in top_indices]
        return [
            ProposedVariant(variant=v, acquisition_score=float(ucb_scores[i]), acquisition_type="ucb")
            for i, v in zip(top_indices, selected)
        ]


# Lookup table
STRATEGY_LOOKUP = {
    "random": RandomStrategy(),
    "greedy": GreedyStrategy(),
    "uncertainty": UncertaintyStrategy(),
    "diversity": DiversityStrategy(),
    "qbc": QBCStrategy(),
    "ucb": UCBStrategy(lambda_coef=1.0),
    # add thompson sampling or entropy? 
}

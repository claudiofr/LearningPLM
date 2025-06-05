"""Learner heads for the PLM Framework."""
from plm_framework.learners.active import RidgeLearner, MLPLearner
from plm_framework.learners.rl import RLPolicyLearner

__all__ = ["RidgeLearner", "MLPLearner", "RLPolicyLearner"]
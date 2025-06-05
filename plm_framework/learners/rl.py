"""Reinforcement learning policy for the PLM Framework."""
from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from plm_framework.learners.base import BaseLearner

# Initialize logger
logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """Policy network for RL learner."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, action_dim: int = 20):
        """
        Initialize the policy network.

        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layer
            action_dim: Dimension of action space (typically number of amino acids)
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Tuple of (action_probs, state_value)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)
        return action_probs, state_value


class RLPolicyLearner(BaseLearner):
    """Reinforcement learning policy learner using A2C."""

    def __init__(
        self,
        embedding_dim: int = 1280,
        reduced_dim: Optional[int] = 128,
        hidden_dim: int = 64,
        action_dim: int = 20,  # Number of amino acids
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_mutations: int = 5,
        mutation_penalty: float = 0.1,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        """
        Initialize the RL policy learner.

        Args:
            embedding_dim: Dimension of input embeddings
            reduced_dim: Dimension to reduce embeddings to
            hidden_dim: Dimension of hidden layer in policy network
            action_dim: Dimension of action space (typically number of amino acids)
            gamma: Discount factor
            entropy_coef: Entropy coefficient for exploration
            learning_rate: Learning rate for optimizer
            max_mutations: Maximum number of mutations allowed
            mutation_penalty: Penalty for each mutation
            device: Device to run the model on
            random_state: Random seed
        """
        super().__init__(embedding_dim, reduced_dim, random_state)
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_mutations = max_mutations
        self.mutation_penalty = mutation_penalty

        # Set device
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")

        # Set random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Initialize policy network
        self.input_dim = reduced_dim if reduced_dim is not None else embedding_dim
        self.policy = PolicyNetwork(self.input_dim, hidden_dim, action_dim)
        self.policy.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Initialize scaler
        self.scaler = None
        self.is_fitted = False

    def fit(
        self,
        embeddings: np.ndarray,
        scores: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit the RL policy using A2C.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]
            scores: Target scores [n_samples]
            uncertainties: Optional measurement uncertainties [n_samples]
        """
        # Reduce dimensionality
        X = self._reduce_embeddings(embeddings)

        # Scale features
        if self.scaler is None:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        # Convert to torch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        scores_tensor = torch.FloatTensor(scores).to(self.device)

        # Train policy network
        for _ in range(100):  # Number of training iterations
            # Get action probabilities and state values
            action_probs, state_values = self.policy(X_tensor)

            # Calculate advantage (simplified)
            advantage = scores_tensor - state_values.squeeze()

            # Calculate policy loss
            policy_loss = -torch.sum(torch.log(action_probs)
                                     * advantage.unsqueeze(1))

            # Calculate value loss
            value_loss = F.mse_loss(state_values.squeeze(), scores_tensor)

            # Calculate entropy (for exploration)
            entropy = -torch.sum(action_probs * torch.log(action_probs))

            # Total loss
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.is_fitted = True

    def predict(
        self, embeddings: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict scores using the policy network's value function.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]

        Returns:
            Tuple of (predictions, None)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")

        # Reduce dimensionality
        X = self._reduce_embeddings(embeddings)

        # Scale features
        X = self.scaler.transform(X)

        # Convert to torch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Get state values
        with torch.no_grad():
            _, state_values = self.policy(X_tensor)

        # Return predictions (no uncertainty)
        return state_values.cpu().numpy().squeeze(), None

    def get_acquisition_scores(
        self, embeddings: np.ndarray, temperature: float = 1.0, **kwargs
    ) -> np.ndarray:
        """
        Get acquisition scores based on policy network.

        Args:
            embeddings: Protein embeddings [n_samples, embedding_dim]
            temperature: Temperature for softmax
            **kwargs: Additional parameters

        Returns:
            Acquisition scores [n_samples]
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")

        # Reduce dimensionality
        X = self._reduce_embeddings(embeddings)

        # Scale features
        X = self.scaler.transform(X)

        # Convert to torch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Get action probabilities and state values
        with torch.no_grad():
            action_probs, state_values = self.policy(X_tensor)

        # Use state values as acquisition scores
        scores = state_values.cpu().numpy().squeeze()

        # Apply temperature scaling
        scores = scores / temperature

        return scores

    def get_mutation_actions(
        self, embedding: np.ndarray, sequence: str, n_mutations: int = 1
    ) -> List[Tuple[int, str]]:
        """
        Get mutation actions for a sequence.

        Args:
            embedding: Protein embedding [embedding_dim]
            sequence: Original protein sequence
            n_mutations: Number of mutations to propose

        Returns:
            List of (position, amino_acid) tuples
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")

        # Amino acid vocabulary
        aa_vocab = "ACDEFGHIKLMNPQRSTVWY"

        # Reduce dimensionality
        X = self._reduce_embeddings(embedding.reshape(1, -1))

        # Scale features
        X = self.scaler.transform(X)

        # Convert to torch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Get action probabilities
        with torch.no_grad():
            action_probs, _ = self.policy(X_tensor)

        action_probs = action_probs.cpu().numpy()[0]

        # Sample positions and amino acids
        mutations = []
        for _ in range(min(n_mutations, self.max_mutations)):
            # Sample position (uniform)
            pos = np.random.randint(0, len(sequence))

            # Sample amino acid (according to policy)
            aa_idx = np.random.choice(self.action_dim, p=action_probs)
            aa = aa_vocab[aa_idx]

            # Ensure it's a mutation
            if aa == sequence[pos]:
                continue

            mutations.append((pos, aa))

        return mutations

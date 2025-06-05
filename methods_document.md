# Methods: Flexible Learning PLM Framework

## Overview

We developed a flexible learning framework for protein sequence optimization that leverages protein language models (PLMs) to guide the exploration of protein sequence space. The framework combines pre-trained protein embeddings with lightweight learning heads that can be trained on minimal experimental data, enabling efficient optimization of protein properties through an iterative active learning approach.

## System Architecture

The framework consists of four main components:

1. **Embedding Module**: Utilizes pre-trained ESM-2 models to convert protein sequences into dense vector representations.
2. **Learner Heads**: Implements both regression-based and reinforcement learning approaches for property prediction and sequence optimization.
3. **Acquisition Module**: Selects promising variants for experimental testing using various strategies.
4. **Controller**: Orchestrates the active learning loop, managing data flow between components.

## Implementation Details

### Protein Embedding

We employed the ESM-2 protein language model (Rives et al., 2021) as our primary embedding method, with the 650M parameter model (esm2_t33_650M_UR50D) as the default. Protein sequences were tokenized at the amino acid level and processed through the transformer architecture to generate contextualized embeddings. For sequence-level representations, we used mean pooling over all residue embeddings, resulting in a 1280-dimensional vector per sequence.

To improve computational efficiency, we implemented:
- Embedding caching in HDF5 format
- Optional dimensionality reduction via PCA (default: 128 dimensions)
- Support for quantized models on resource-constrained hardware

### Learning Approaches

The framework supports two complementary learning approaches:

#### 1. Active Learning with Uncertainty Estimation

For direct property prediction, we implemented:
- **Ridge Regression**: An ensemble of ridge regressors trained on the embedded sequences, with uncertainty estimated via bagging.
- **MLP Regression**: A neural network regressor with dropout-based uncertainty estimation.

Both approaches provide not only predictions but also uncertainty estimates, which are crucial for acquisition functions in the active learning loop.

#### 2. Reinforcement Learning for Directed Evolution

For sequence optimization without explicit property prediction, we implemented:
- **A2C Policy**: An advantage actor-critic architecture that learns to propose beneficial mutations directly.
- The policy network takes sequence embeddings as input and outputs:
  - Action probabilities (which positions to mutate and to which amino acids)
  - Value estimates (expected reward of the current sequence)
- Reward shaping includes both the predicted functional improvement and a penalty for excessive mutations.

### Acquisition Strategies

To select the most informative variants for experimental testing, we implemented several acquisition functions:
- **Upper Confidence Bound (UCB)**: Balances exploitation (high predicted value) and exploration (high uncertainty).
- **Expected Improvement (EI)**: Considers the potential for improvement over the current best variant.
- **Thompson Sampling**: Samples from the posterior distribution of predictions.
- **Diversity-Based Selection**: For initial rounds with no prior data, selects diverse variants using K-means clustering on the embedding space.

### Experimental Workflow

The framework supports an iterative workflow:

1. **Initial Setup**: The original protein sequence is embedded using the ESM-2 model.
   ```bash
   plm embed_sequence "ORIGINAL_PROTEIN_SEQUENCE" embeddings.h5
   ```

2. **Round 0 (Initial Exploration)**: Without prior measurements, the system proposes diverse variants based on embedding space clustering.
   ```bash
   plm propose --config-path config.yaml embeddings.h5 candidates.fasta proposals_round0.csv --batch-size 10
   ```

3. **Experimental Testing**: The proposed variants are synthesized and assayed in the laboratory.

4. **Model Training**: Experimental measurements are used to train the learner head.
   ```bash
   plm embed measured_sequences.csv embeddings.h5 --append
   ```

5. **Active Learning Loop**: The system enters an iterative cycle of:
   - Proposing new variants based on model predictions and uncertainty
   - Experimental testing
   - Model updating with new data
   ```bash
   plm learn --config-path config.yaml measured_data.csv results/ --n-rounds 5 --batch-size 10
   ```

This process continues until satisfactory variants are identified or resource constraints are reached.

## Technical Implementation

The framework was implemented in Python 3.10+ with the following key dependencies:
- PyTorch and HuggingFace Transformers for the embedding models
- scikit-learn for regression models and dimensionality reduction
- Pydantic for data validation and serialization
- Typer for the command-line interface

The codebase follows a modular design with clear separation of concerns:
- `embedding.py`: Handles protein sequence embedding
- `learners/`: Contains implementations of different learning approaches
- `acquisition.py`: Implements variant selection strategies
- `controller.py`: Orchestrates the active learning loop
- `cli.py`: Provides a user-friendly command-line interface

## Evaluation

The framework was evaluated on several protein engineering datasets, including:
- Held-out assays from Dallago et al. (Science, 2024)
- Nat Commun 2025 dataset 55987-8

Performance metrics included:
- Spearman correlation between predicted and measured properties
- Top-k hit rate at various thresholds (k=10, 50)
- Experimental budget efficiency (number of assays required to reach 90% of the best variant)

Ablation studies were conducted to assess the impact of:
- Learning approach (ridge vs. MLP vs. RL)
- Number of trainable layers (0, 1, 2)
- Acquisition strategy (UCB, EI, Thompson sampling)
- Embedding dimensionality (original vs. reduced)

## References

1. Rives, A., Meier, J., Sercu, T., et al. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proceedings of the National Academy of Sciences, 118(15).
2. Dallago, C., et al. (2024). Large language models for protein design. Science.
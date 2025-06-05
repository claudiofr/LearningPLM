# PLM Framework

A flexible learning framework for protein sequence optimization using protein language models (PLMs).

## Overview

PLM Framework provides a comprehensive toolkit for protein engineering through active learning. It leverages state-of-the-art protein language models (like ESM-2) to guide the exploration of protein sequence space, helping researchers discover variants with improved properties.

## Features

- **Protein Embedding**: Efficiently embed protein sequences using ESM models with caching
- **Active Learning Loop**: Iterative propose → assay → fit cycle for efficient exploration
- **Multiple Learning Strategies**: 
  - Ridge/MLP regression for property prediction
  - Reinforcement learning policy for directed evolution
- **Flexible Acquisition Functions**: UCB, Expected Improvement, Thompson sampling
- **Comprehensive Data Management**: SQLite backend with versioning
- **User-Friendly Interfaces**: Python API and CLI

## Installation

### 1. Install Poetry

```bash
# On macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -

# On Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

### 2. Install PLM Framework

```bash
# Clone the repository
git clone https://github.com/yourusername/plm-framework.git
cd plm-framework

# Install with Poetry
poetry install
```

## Quick Start

### 1. Initialize a project

```bash
# Create default configuration and directories
plm init --config-path config.yaml
```

### 2. Generate initial proposals (Round 0)

For the first round when no experimental data is available, you can use two strategies:

```bash
# Propose initial variants using ESM2 logits (recommended)
plm propose-initial --config-path config.yaml "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" proposals_round0.csv --batch-size 10 --n-mutations 1 --strategy esm_logit

# Alternatively, use diversity-based sampling
plm propose-initial --config-path config.yaml "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" proposals_round0.csv --batch-size 10 --n-mutations 1 --strategy diversity
```

#### ESM Logit Sampling (Default)

The `esm_logit` strategy leverages the ESM2 protein language model to identify mutations that are more likely to be functional:

- Generates random single/multiple point mutations
- Scores each mutation based on the ESM2 model's predicted probability at that position
- Selects variants with the highest logit scores

This approach uses the language model's understanding of protein sequences to prioritize mutations that maintain the protein's structural and functional properties.

#### Diversity Sampling

The `diversity` strategy focuses on maximizing the diversity of the initial batch:

- Generates random mutations
- Embeds all variants using the ESM2 model
- Selects variants that are maximally diverse in the embedding space

This is useful when you want to explore different regions of the sequence space.

Both methods will:
- Generate candidate variants with the specified number of mutations
- Score and rank the variants according to the chosen strategy
- Save the top-scoring variants to a CSV file

### Restricting Mutations to Specific Regions

You can restrict mutations to specific regions of the protein sequence using the `--mutation-range` parameter:

```bash
# Only mutate residues 50-150
plm propose-initial --config-path config.yaml "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" proposals_round0.csv --mutation-range "50-150"

# Mutate specific residues and ranges
plm propose-initial --config-path config.yaml "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" proposals_round0.csv --mutation-range "5,10,15-20,30-40"
```

This feature is useful when:
- You want to focus on specific functional domains
- You have prior knowledge about which regions are more likely to yield beneficial mutations
- You want to avoid disrupting critical structural elements

### 3. Embed sequences for experimental testing

After selecting variants for testing:

```bash
# Embed sequences from a FASTA or CSV file
plm embed proposals_round0.csv embeddings.h5 --model-name facebook/esm2_t33_650M_UR50D
```

### 4. Run active learning with experimental data

After obtaining experimental measurements:

```bash
# Train model and run active learning loop
plm learn --config-path config.yaml measured_data.csv results/ --n-rounds 5 --batch-size 10
```

### 5. Propose new variants with trained model

```bash
# Propose new variants using the trained model
plm propose --config-path config.yaml model.pkl candidates.fasta proposals.csv --batch-size 10
```

## Input File Formats

The framework accepts several input file formats:

### FASTA Files (.fasta, .fa)
Used for protein sequences without experimental data:

```
>variant_1
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
>variant_2
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAAG
```

### CSV Files (.csv)
Used for variants with experimental measurements:

```csv
id,sequence,score,uncertainty
variant_1,MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG,0.85,0.05
variant_2,MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAAG,0.92,0.03
```

## Experimental Workflow

1. **Initial Setup**: Start with the original protein sequence
2. **Generate Candidates**: Create a set of candidate variants
3. **Round 0 (Initial Exploration)**: Propose diverse variants without prior measurements
4. **Experimental Testing**: Assay the proposed variants and record measurements
5. **Embed Measured Sequences**: Add the experimentally measured sequences to the embedding file
6. **Active Learning Loop**: Train model → Propose variants → Assay → Update model → Repeat
7. **Analysis**: Evaluate model performance and extract insights

## Benchmarking

The framework includes tools for benchmarking performance on protein engineering datasets, such as those from Protein Gym.

### Running a benchmark

```bash
# Run benchmark on a Protein Gym dataset
python scripts/benchmark_protein_gym.py \
  --config config.yaml \
  --dataset test_data/S22A1_HUMAN_Yee_2023_activity.csv \
  --output-dir benchmark_results/s22a1 \
  --n-rounds 5 \
  --strategies ucb,ei,ts,diversity
```

This will:
1. Split the dataset into training and test sets
2. Simulate multiple rounds of active learning
3. Evaluate performance metrics after each round
4. Generate plots comparing different acquisition strategies

### Benchmark metrics

The benchmark tracks several performance metrics across rounds:

- **R²**: Coefficient of determination (higher is better)
- **RMSE**: Root mean squared error (lower is better)
- **Spearman Correlation**: Rank correlation between predictions and actual values (higher is better)
- **Top-N Mean**: Average score of the top N predicted variants (higher is better)

These metrics help evaluate how well the model learns to predict protein function and how effectively the active learning strategy explores the sequence space.

## License

MIT

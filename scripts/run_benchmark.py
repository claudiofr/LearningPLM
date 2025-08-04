from pathlib import Path
from plm_framework.utils import setup_logging
from plm_framework.simulation import simulate_active_learning, plot_results
from scripts.benchmark_protein_gym import load_protein_gym_dataset

# Setup
config_path = "config.yaml"
dataset_path = "protein_gym.csv"
output_dir = Path("benchmark_results")
output_dir.mkdir(exist_ok=True)

# List strategies (must match keys in your STRATEGY_LOOKUP in plm_framework.strategies)
strategies = ["ucb", "random", "diversity", "greedy", "qbc"] 

n_rounds = 5
initial_batch_size = 10
batch_size = 10

# Load dataset
setup_logging()
reference_seq, variants, assay_results = load_protein_gym_dataset(dataset_path)

# Run simulation
results = simulate_active_learning(
    config_path=config_path,
    variants=variants,
    assay_results=assay_results,
    output_dir=str(output_dir),
    n_rounds=n_rounds,
    initial_batch_size=initial_batch_size,
    batch_size=batch_size,
    strategies=strategies,
)

# Plot results (combined + per-metric)
plot_results(results, output_dir)

print(f"Plots and CSVs saved in {output_dir.resolve()}")

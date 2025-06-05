from plm_framework.cli import learn

learn(
    config_path="tutorial_results/s22a1/config.yaml",
    input_path="tutorial_results/s22a1/data/round0_results.csv",
    output_dir="tutorial_results/s22a1/results",
    n_rounds=5,  # Increased from 2 to 5 rounds
    batch_size=10,
    strategy="ucb",
    temperature=1.0,
    clear_db=True,  # Clear the database before starting
)

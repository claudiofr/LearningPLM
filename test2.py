import sys
from benchmark_protein_gym import main

sys.argv = [
    "benchmark_protein_gymn.py",
    "--config",
    "config_cf.yaml",
    "--dataset",
    "test_data/S22A1_HUMAN_Yee_2023_activity.csv",
    "--output-dir",
    "benchmark_results/s22a1",
    "--n-rounds",
    "5",
    "--strategies",
    "ucb,ei,ts,diversity",
]

main()

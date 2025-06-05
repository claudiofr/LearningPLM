"""Command-line interface for the PLM Framework."""

import logging
import os
from pathlib import Path
from typing import List, Optional

import typer
from omegaconf import OmegaConf

from plm_framework.config import get_config, get_default_config
from plm_framework.controller import Controller
from plm_framework.data_manager import DataManager
from plm_framework.datamodels import AssayResult, Variant
from plm_framework.embedding import ProteinEmbedder, embed_sequences

app = typer.Typer(help="PLM Framework CLI")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@app.command()
def init(
    config_path: str = typer.Option(
        "config.yaml", help="Path to save the default configuration"
    ),
):
    """Initialize a new PLM Framework project with default configuration."""
    config_path = Path(config_path)

    # Create default config
    default_config = get_default_config()

    # Save config
    os.makedirs(config_path.parent, exist_ok=True)
    with open(config_path, "w") as f:
        OmegaConf.save(default_config, f)

    logger.info(f"Initialized project with default configuration at {config_path}")

    # Create data directory
    data_dir = Path(default_config.data.db_path).parent
    os.makedirs(data_dir, exist_ok=True)

    # Create results directory
    results_dir = Path(default_config.data.output_dir)
    os.makedirs(results_dir, exist_ok=True)

    logger.info(f"Created data directory at {data_dir}")
    logger.info(f"Created results directory at {results_dir}")


@app.command()
def embed(
    input_path: str = typer.Argument(..., help="Path to input FASTA or CSV file"),
    output_path: str = typer.Argument(..., help="Path to save embeddings"),
    model_name: str = typer.Option(
        "facebook/esm2_t33_650M_UR50D", help="Name of the ESM model to use"
    ),
    batch_size: int = typer.Option(8, help="Batch size for embedding"),
    device: Optional[str] = typer.Option(None, help="Device to run the model on"),
    use_pooling: bool = typer.Option(True, help="Whether to use mean pooling"),
    append: bool = typer.Option(False, help="Whether to append to existing embeddings"),
):
    """Embed protein sequences and save to file."""
    from plm_framework.utils import load_sequences
    from plm_framework.embedding import embed_sequences

    # Load sequences
    variants = load_sequences(input_path)

    # Ensure device is a string or None
    device_str = device if isinstance(device, str) or device is None else None

    # Embed sequences
    embed_sequences(
        variants=variants,
        output_path=output_path,
        model_name=model_name,
        batch_size=batch_size,
        device=device_str,
        use_pooling=use_pooling,
        append=append,
    )


@app.command()
def learn(
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    input_path: str = typer.Argument(
        ..., help="Path to input CSV file with assay results"
    ),
    output_dir: str = typer.Argument(..., help="Directory to save results"),
    n_rounds: int = typer.Option(5, help="Number of rounds to run"),
    batch_size: int = typer.Option(
        10, help="Number of variants to propose in each round"
    ),
    strategy: str = typer.Option("ucb", help="Acquisition strategy"),
    temperature: float = typer.Option(1.0, help="Temperature for softmax"),
    clear_db: bool = typer.Option(False, help="Clear the database before starting"),
):
    """Run the active learning loop."""
    from plm_framework.utils import load_assay_results
    import os
    from pathlib import Path

    # Load configuration
    config = get_config(config_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Clear the database if requested
    if clear_db:
        db_path = Path(config.data.db_path)
        if db_path.exists():
            logger.info(f"Clearing database at {db_path}")
            os.remove(db_path)
            logger.info("Database cleared")

    # Create controller
    controller = Controller(config)

    # Load assay results
    variants, results = load_assay_results(input_path)

    # Run active learning loop
    controller.run_active_learning_loop(
        candidates=variants,
        initial_variants=variants,
        initial_results=results,
        n_rounds=n_rounds,
        batch_size=batch_size,
        strategy=strategy,
        temperature=temperature,
    )

    # Export results
    controller.export_results(output_dir)

    # Save model
    controller.save_model(Path(output_dir) / "model.pkl")


@app.command()
def propose(
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    model_path: str = typer.Argument(..., help="Path to trained model"),
    candidates_path: str = typer.Argument(..., help="Path to candidate sequences"),
    output_path: str = typer.Argument(..., help="Path to save proposed variants"),
    batch_size: int = typer.Option(10, help="Number of variants to propose"),
    strategy: str = typer.Option("ucb", help="Acquisition strategy"),
    temperature: float = typer.Option(1.0, help="Temperature for softmax"),
):
    """Propose variants using a trained model."""
    from plm_framework.utils import load_sequences, save_proposed_variants

    # Load configuration
    config = get_config(config_path)

    # Create controller
    controller = Controller(config)

    # Load model
    controller.load_model(model_path)

    # Load candidate sequences
    candidates = load_sequences(candidates_path)

    # Start a temporary round
    controller.start_round("Proposal Round", "Temporary round for proposing variants")

    # Propose variants
    proposed_variants = controller.propose_variants(
        candidates=candidates,
        batch_size=batch_size,
        strategy=strategy,
        temperature=temperature,
    )

    # Complete the round
    controller.complete_round()

    # Save proposed variants
    save_proposed_variants(proposed_variants, output_path)

    logger.info(f"Saved {len(proposed_variants)} proposed variants to {output_path}")


@app.command()
def evaluate(
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    model_path: str = typer.Argument(..., help="Path to trained model"),
    test_path: str = typer.Argument(..., help="Path to test data"),
    output_path: str = typer.Argument(..., help="Path to save evaluation results"),
):
    """Evaluate a trained model on test data."""
    from plm_framework.utils import load_assay_results, save_metrics

    # Load configuration
    config = get_config(config_path)

    # Create controller
    controller = Controller(config)

    # Load model
    controller.load_model(model_path)

    # Load test data
    test_variants, test_results = load_assay_results(test_path)

    # Evaluate model
    metrics = controller.evaluate_model(test_variants, test_results)

    # Save metrics
    save_metrics(metrics, output_path)

    logger.info(f"Saved evaluation metrics to {output_path}")


@app.command()
def embed_sequence(
    sequence: str = typer.Argument(..., help="Protein sequence to embed"),
    output_path: str = typer.Argument(..., help="Path to save embedding"),
    model_name: str = typer.Option(
        "facebook/esm2_t33_650M_UR50D", help="Name of the ESM model to use"
    ),
    device: Optional[str] = typer.Option(None, help="Device to run the model on"),
    use_pooling: bool = typer.Option(True, help="Whether to use mean pooling"),
    variant_id: Optional[int] = typer.Option(None, help="Variant ID (default: 1)"),
):
    """Embed a single protein sequence and save to file."""
    # Ensure device is a string or None
    device_str = device if isinstance(device, str) or device is None else None

    # Create variant
    variant = Variant(
        id=variant_id or 1,
        sequence=sequence,
        name="original_sequence",
    )

    # Embed sequence
    embed_sequences(
        variants=[variant],
        output_path=output_path,
        model_name=model_name,
        device=device_str,
        use_pooling=use_pooling,
    )


@app.command()
def propose_initial(
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    sequence: str = typer.Argument(..., help="Original protein sequence"),
    output_path: str = typer.Argument(..., help="Path to save proposed variants"),
    batch_size: int = typer.Option(10, help="Number of variants to propose"),
    n_mutations: int = typer.Option(1, help="Number of mutations per variant"),
    temperature: float = typer.Option(1.0, help="Temperature for softmax"),
    strategy: str = typer.Option(
        "esm_logit",
        help="Acquisition strategy: 'esm_logit' (uses ESM2 model to score likely mutations) or 'diversity' (maximizes diversity in embedding space)",
    ),
    mutation_range: Optional[str] = typer.Option(
        None,
        help="Range of residues to mutate (e.g., '1-100' or '5,10,15-20'). If not specified, the entire sequence can be mutated.",
    ),
):
    """
    Propose initial variants based on a single sequence without requiring a trained model.

    This command generates candidate variants with random mutations and scores them using
    either ESM2 logits (to find likely functional mutations) or diversity-based sampling.

    The ESM2 logit strategy leverages the protein language model to identify mutations
    that are more likely to maintain the protein's function, while the diversity strategy
    focuses on exploring different regions of the sequence space.
    """
    from plm_framework.utils import save_proposed_variants, parse_mutation_range
    from plm_framework.datamodels import Variant, ProposedVariant
    import random

    # Load configuration
    config = get_config(config_path)

    # Create controller
    controller = Controller(config)

    # Create variant from the original sequence
    original_variant = Variant(
        id="WT",  # Use "WT" for wild type
        name="original_sequence",
        sequence=sequence,
    )

    # Parse mutation range - ensure it's a string or None
    mutation_range_str = mutation_range if mutation_range is not None else None
    mutable_positions = parse_mutation_range(mutation_range_str, len(sequence))

    if mutable_positions:
        logger.info(
            f"Restricting mutations to {len(mutable_positions)} positions: {mutable_positions[:10]}..."
        )
    else:
        logger.info(
            f"No mutation range specified. All {len(sequence)} positions can be mutated."
        )
        mutable_positions = list(range(len(sequence)))

    # Amino acid vocabulary
    aa_vocab = "ACDEFGHIKLMNPQRSTVWY"

    # Generate variants with random mutations
    candidates = [original_variant]
    seq_length = len(sequence)

    for i in range(1, batch_size * 10):  # Generate more candidates than needed
        variant_seq = list(sequence)
        mutation_positions = []
        mutation_details = []

        # Apply random mutations
        for _ in range(n_mutations):
            # Choose position not already mutated from the mutable positions
            while True:
                if not mutable_positions:
                    logger.warning("No mutable positions available!")
                    break

                pos = random.choice(mutable_positions)
                if pos not in mutation_positions:
                    break

            if not mutable_positions:
                break

            # Choose a different amino acid
            original_aa = sequence[pos]
            while True:
                new_aa = random.choice(aa_vocab)
                if new_aa != original_aa:
                    break

            variant_seq[pos] = new_aa
            mutation_positions.append(pos)

            # Record mutation details (1-indexed for conventional notation)
            mutation_details.append(f"{original_aa}{pos+1}{new_aa}")

        # Skip if no mutations were applied
        if not mutation_details:
            continue

        # Create variant name from mutation details
        variant_name = "_".join(mutation_details)

        # Use the mutation details as the ID
        variant_id = variant_name

        # Create variant
        variant = Variant(
            id=variant_id,  # Use mutation details as ID
            name=variant_name,
            sequence="".join(variant_seq),
        )
        candidates.append(variant)

    # Start a temporary round
    controller.start_round("Initial Round", "Initial round for proposing variants")

    # Propose variants using the specified strategy
    proposed_variants = controller.propose_variants(
        candidates=candidates,
        batch_size=batch_size,
        strategy=strategy,
        temperature=temperature,
        original_sequence=sequence,
        model_name=config.model.name,
        device=config.training.device if config.training.device != "auto" else None,
    )

    # Complete the round
    controller.complete_round()

    # Save proposed variants
    save_proposed_variants(proposed_variants, output_path)

    logger.info(f"Saved {len(proposed_variants)} proposed variants to {output_path}")


if __name__ == "__main__":
    app()

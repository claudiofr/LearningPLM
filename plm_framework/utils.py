"""Utility functions for the PLM Framework."""
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from plm_framework.datamodels import AssayResult, ProposedVariant, Variant

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """
    Set up logging configuration.

    Args:
        level: Logging level

    Returns:
        Logger instance
    """
    logger = logging.getLogger("plm_framework")
    logger.setLevel(level)

    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger


def load_sequences(input_path: Union[str, Path]) -> List[Variant]:
    """
    Load sequences from a file.

    Args:
        input_path: Path to input file (FASTA or CSV)

    Returns:
        List of variants
    """
    input_path = Path(input_path)
    variants = []

    if input_path.suffix.lower() in (".fasta", ".fa"):
        # Load FASTA file
        with open(input_path, "r") as f:
            current_id = None
            current_seq = ""

            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith(">"):
                    # Save previous sequence
                    if current_id is not None:
                        variants.append(
                            Variant(
                                id=len(variants) + 1,
                                name=current_id,
                                sequence=current_seq,
                            )
                        )

                    # Start new sequence
                    current_id = line[1:].strip()
                    current_seq = ""
                else:
                    current_seq += line

            # Save last sequence
            if current_id is not None:
                variants.append(
                    Variant(
                        id=len(variants) + 1,
                        name=current_id,
                        sequence=current_seq,
                    )
                )
    elif input_path.suffix.lower() == ".csv":
        # Load CSV file
        df = pd.read_csv(input_path)

        # Check required columns
        if "sequence" not in df.columns:
            raise ValueError("CSV file must contain 'sequence' column")

        # Get ID column
        id_col = "id" if "id" in df.columns else None
        name_col = "name" if "name" in df.columns else None

        # If we have a reference sequence and no name column, generate mutation-based names
        reference_seq = None
        if "reference_sequence" in df.columns and df["reference_sequence"].iloc[0] is not None:
            reference_seq = df["reference_sequence"].iloc[0]
        elif len(df) > 0 and not name_col:
            # Try to use the first sequence as reference if it's named "original_sequence"
            if id_col and df[id_col].iloc[0] == "original_sequence":
                reference_seq = df["sequence"].iloc[0]
            elif name_col and df[name_col].iloc[0] == "original_sequence":
                reference_seq = df["sequence"].iloc[0]

        for i, row in df.iterrows():
            if id_col and not pd.isna(row[id_col]):
                variant_id = row[id_col]
            elif reference_seq is not None and row["sequence"] != reference_seq:
                # Find mutations
                mutation_details = []
                for j, (ref_aa, var_aa) in enumerate(zip(reference_seq, row["sequence"])):
                    if ref_aa != var_aa:
                        # 1-indexed position for conventional notation
                        mutation_details.append(f"{ref_aa}{j+1}{var_aa}")

                if mutation_details:
                    variant_id = "_".join(mutation_details)
                else:
                    variant_id = "WT"  # Wild type
            else:
                # Fallback to numeric ID with prefix
                variant_id = f"variant_{i + 1}"

            # Generate name based on mutations if reference sequence is available
            if name_col and not pd.isna(row[name_col]):
                variant_name = row[name_col]
            elif reference_seq is not None and row["sequence"] != reference_seq:
                # Find mutations
                mutation_details = []
                for j, (ref_aa, var_aa) in enumerate(zip(reference_seq, row["sequence"])):
                    if ref_aa != var_aa:
                        # 1-indexed position for conventional notation
                        mutation_details.append(f"{ref_aa}{j+1}{var_aa}")

                if mutation_details:
                    variant_name = "_".join(mutation_details)
                else:
                    variant_name = f"variant_{variant_id}"
            else:
                variant_name = f"variant_{variant_id}"

            variants.append(
                Variant(
                    id=variant_id,
                    name=variant_name,
                    sequence=row["sequence"],
                )
            )
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    logger.info(f"Loaded {len(variants)} sequences from {input_path}")
    return variants


def load_assay_results(input_path: Union[str, Path]) -> Tuple[List[Variant], List[AssayResult]]:
    """
    Load assay results from CSV file.

    Args:
        input_path: Path to input file

    Returns:
        Tuple of (variants, results)
    """
    input_path = Path(input_path)

    if input_path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    # Load CSV file
    df = pd.read_csv(input_path)

    # Check required columns
    required_cols = ["sequence", "score"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV file must contain '{col}' column")

    # Get optional columns
    id_col = "id" if "id" in df.columns else None
    name_col = "name" if "name" in df.columns else None
    uncertainty_col = "uncertainty" if "uncertainty" in df.columns else None

    variants = []
    results = []

    for i, row in df.iterrows():
        variant_id = row[id_col] if id_col else i + 1
        variant_name = row[name_col] if name_col else f"variant_{variant_id}"

        variant = Variant(
            id=variant_id,
            name=variant_name,
            sequence=row["sequence"],
        )

        result = AssayResult(
            variant_id=variant_id,
            score=float(row["score"]),
            uncertainty=float(row[uncertainty_col]) if uncertainty_col and not pd.isna(
                row[uncertainty_col]) else None,
        )

        variants.append(variant)
        results.append(result)

    logger.info(
        f"Loaded {len(variants)} variants with assay results from {input_path}")
    return variants, results


def save_proposed_variants(
    proposed_variants: List[ProposedVariant], output_path: Union[str, Path]
) -> None:
    """
    Save proposed variants to a CSV file.

    Args:
        proposed_variants: List of proposed variants
        output_path: Path to save CSV file
    """
    output_path = Path(output_path)

    # Create dataframe
    data = {
        "id": [v.variant.id for v in proposed_variants],
        "name": [v.variant.name for v in proposed_variants],
        "sequence": [v.variant.sequence for v in proposed_variants],
        "acquisition_score": [v.acquisition_score for v in proposed_variants],
        "acquisition_type": [v.acquisition_type for v in proposed_variants],
    }

    # Add predicted scores and uncertainties if available
    if any(v.predicted_score is not None for v in proposed_variants):
        data["predicted_score"] = [
            v.predicted_score for v in proposed_variants]

    if any(v.predicted_uncertainty is not None for v in proposed_variants):
        data["predicted_uncertainty"] = [
            v.predicted_uncertainty for v in proposed_variants]

    # Create dataframe and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    logger.info(
        f"Saved {len(proposed_variants)} proposed variants to {output_path}")


def save_metrics(metrics: Dict[str, float], output_path: Union[str, Path]) -> None:
    """
    Save metrics to CSV file.

    Args:
        metrics: Dictionary of metrics
        output_path: Path to save file
    """
    output_path = Path(output_path)

    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV file
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(["metric", "value"])

        # Write metrics
        for metric, value in metrics.items():
            writer.writerow([metric, value])

    logger.info(f"Saved {len(metrics)} metrics to {output_path}")


def parse_mutation_range(range_str: Optional[str], sequence_length: int) -> List[int]:
    """
    Parse a mutation range string into a list of positions.

    Args:
        range_str: Range string (e.g., '1-100' or '5,10,15-20')
        sequence_length: Length of the sequence

    Returns:
        List of 0-indexed positions that can be mutated
    """
    # Handle the case when range_str is None or not a string
    if range_str is None:
        return list(range(sequence_length))

    # Ensure range_str is a string
    if not isinstance(range_str, str):
        logger.warning(
            f"mutation_range is not a string: {type(range_str)}. Using all positions.")
        return list(range(sequence_length))

    positions = []

    # Split by comma
    parts = range_str.split(',')

    for part in parts:
        if '-' in part:
            # Range (e.g., '1-100')
            start, end = part.split('-')
            # Convert to 0-indexed
            start_idx = int(start) - 1
            end_idx = int(end) - 1

            # Validate range
            if start_idx < 0 or end_idx >= sequence_length:
                raise ValueError(
                    f"Range {part} is out of bounds for sequence of length {sequence_length}")

            positions.extend(range(start_idx, end_idx + 1))
        else:
            # Single position (e.g., '5')
            # Convert to 0-indexed
            pos = int(part) - 1

            # Validate position
            if pos < 0 or pos >= sequence_length:
                raise ValueError(
                    f"Position {part} is out of bounds for sequence of length {sequence_length}")

            positions.append(pos)

    # Remove duplicates and sort
    return sorted(list(set(positions)))

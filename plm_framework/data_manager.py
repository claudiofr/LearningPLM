"""Data manager for the PLM Framework."""

import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from plm_framework.datamodels import AssayResult, ProposedVariant, Round, Variant

logger = logging.getLogger(__name__)


class DataManager:
    """Data manager for the PLM Framework."""

    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize the data manager.

        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = Path(db_path)

        # Create parent directory if it doesn't exist
        os.makedirs(self.db_path.parent, exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS variants (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            sequence TEXT NOT NULL
        )
        """
        )

        cursor.execute("SELECT id FROM variants")
        a = cursor.fetchone()

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS assay_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            variant_id TEXT NOT NULL,
            score REAL NOT NULL,
            uncertainty REAL,
            assay_id TEXT,
            round_id INTEGER,
            FOREIGN KEY (variant_id) REFERENCES variants (id)
        )
        """
        )

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS rounds (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            timestamp TEXT NOT NULL
        )
        """
        )

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS proposed_variants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            round_id INTEGER NOT NULL,
            variant_id TEXT NOT NULL,
            acquisition_score REAL NOT NULL,
            acquisition_type TEXT NOT NULL,
            predicted_score REAL,
            predicted_uncertainty REAL,
            FOREIGN KEY (round_id) REFERENCES rounds (id),
            FOREIGN KEY (variant_id) REFERENCES variants (id)
        )
        """
        )

        conn.commit()
        conn.close()

    def add_variant(self, variant: Variant) -> None:
        """
        Add a variant to the database.

        Args:
            variant: Variant to add
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if variant already exists
        cursor.execute("SELECT id FROM variants WHERE id = ?", (str(variant.id),))
        if cursor.fetchone() is not None:
            conn.close()
            return

        # Insert variant
        cursor.execute(
            "INSERT INTO variants (id, name, sequence) VALUES (?, ?, ?)",
            (str(variant.id), variant.name, variant.sequence),
        )

        conn.commit()
        conn.close()

    def add_variants(self, variants: List[Variant]) -> None:
        """
        Add multiple variants to the database.

        Args:
            variants: List of variants to add
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get existing variant IDs
        cursor.execute("SELECT id FROM variants")
        existing_ids = {row[0] for row in cursor.fetchall()}

        # Filter out existing variants
        new_variants = [v for v in variants if str(v.id) not in existing_ids]

        # Insert variants
        cursor.executemany(
            "INSERT INTO variants (id, name, sequence) VALUES (?, ?, ?)",
            [(str(v.id), v.name, v.sequence) for v in new_variants],
        )

        conn.commit()
        conn.close()

        logger.info(f"Added {len(new_variants)} new variants to the database")

    def add_assay_result(self, result: AssayResult) -> None:
        """
        Add an assay result to the database.

        Args:
            result: Assay result to add
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert result
        cursor.execute(
            "INSERT INTO assay_results (variant_id, score, uncertainty, assay_id, round_id) VALUES (?, ?, ?, ?, ?)",
            (
                str(result.variant_id),
                result.score,
                result.uncertainty,
                getattr(result, "assay_id", "default_assay"),
                getattr(result, "round_id", None),
            ),
        )

        conn.commit()
        conn.close()

    def add_assay_results(self, results: List[AssayResult]) -> None:
        """
        Add multiple assay results to the database.

        Args:
            results: List of assay results to add
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert results
        cursor.executemany(
            "INSERT INTO assay_results (variant_id, score, uncertainty, assay_id, round_id) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    str(r.variant_id),
                    r.score,
                    r.uncertainty,
                    getattr(r, "assay_id", "default_assay"),
                    getattr(r, "round_id", None),
                )
                for r in results
            ],
        )

        conn.commit()
        conn.close()

        logger.info(f"Added {len(results)} assay results to the database")

    def add_round(self, round_data: Round) -> None:
        """
        Add a round to the database.

        Args:
            round_data: Round data to add
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert round
        cursor.execute(
            "INSERT INTO rounds (id, name, description, timestamp) VALUES (?, ?, ?, ?)",
            (
                round_data.id,
                round_data.name,
                round_data.description,
                round_data.timestamp.isoformat(),
            ),
        )

        # Log the number of proposed variants
        logger.info(
            f"Adding round {round_data.id} with {len(round_data.proposed_variants)} proposed variants"
        )

        # Add variants if they don't exist
        for proposed in round_data.proposed_variants:
            self.add_variant(proposed.variant)

            # Insert proposed variant
            cursor.execute(
                """
                INSERT INTO proposed_variants 
                (round_id, variant_id, acquisition_score, acquisition_type, predicted_score, predicted_uncertainty)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    round_data.id,
                    str(proposed.variant.id),
                    proposed.acquisition_score,
                    proposed.acquisition_type,
                    proposed.predicted_score,
                    proposed.predicted_uncertainty,
                ),
            )

        conn.commit()
        conn.close()

        logger.info(
            f"Added round {round_data.id} with {len(round_data.proposed_variants)} proposed variants"
        )

    def get_variant(self, variant_id: Union[int, str]) -> Optional[Variant]:
        """
        Get a variant from the database.

        Args:
            variant_id: ID of the variant

        Returns:
            Variant or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, name, sequence FROM variants WHERE id = ?", (str(variant_id),)
        )
        row = cursor.fetchone()

        conn.close()

        if row is None:
            return None

        return Variant(
            id=row[0],
            name=row[1],
            sequence=row[2],
        )

    def get_variants(
        self, variant_ids: Optional[List[Union[int, str]]] = None
    ) -> List[Variant]:
        """
        Get variants from the database.

        Args:
            variant_ids: Optional list of variant IDs to filter by

        Returns:
            List of variants
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if variant_ids is not None:
            # Convert all IDs to strings for consistency
            str_ids = [str(vid) for vid in variant_ids]
            placeholders = ",".join(["?"] * len(str_ids))
            cursor.execute(
                f"SELECT id, name, sequence FROM variants WHERE id IN ({placeholders})",
                str_ids,
            )
        else:
            cursor.execute("SELECT id, name, sequence FROM variants")

        rows = cursor.fetchall()
        conn.close()

        return [
            Variant(
                id=row[0],
                name=row[1],
                sequence=row[2],
            )
            for row in rows
        ]

    def get_assay_results(self, round_id: Optional[int] = None) -> List[AssayResult]:
        """
        Get all assay results from the database.

        Args:
            round_id: Optional round ID to filter results

        Returns:
            List of assay results
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if round_id is not None:
            cursor.execute(
                "SELECT variant_id, score, uncertainty, round_id FROM assay_results WHERE round_id = ?",
                (round_id,),
            )
        else:
            cursor.execute(
                "SELECT variant_id, score, uncertainty, round_id FROM assay_results"
            )

        rows = cursor.fetchall()
        conn.close()

        return [
            AssayResult(
                variant_id=row[0],
                score=row[1],
                uncertainty=row[2],
                round_id=row[3] if len(row) > 3 else None,
            )
            for row in rows
        ]

    def get_rounds(self) -> List[Round]:
        """
        Get all rounds from the database.

        Returns:
            List of rounds
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id, name, description, timestamp FROM rounds")
        round_rows = cursor.fetchall()

        rounds = []

        for round_row in round_rows:
            round_id = round_row[0]

            # Get proposed variants
            cursor.execute(
                """
                SELECT pv.variant_id, pv.acquisition_score, pv.acquisition_type, 
                       pv.predicted_score, pv.predicted_uncertainty,
                       v.name, v.sequence
                FROM proposed_variants pv
                JOIN variants v ON pv.variant_id = v.id
                WHERE pv.round_id = ?
                """,
                (round_id,),
            )
            proposed_rows = cursor.fetchall()

            proposed_variants = []
            for p_row in proposed_rows:
                variant = Variant(
                    id=p_row[0],
                    name=p_row[5],
                    sequence=p_row[6],
                )

                proposed_variants.append(
                    ProposedVariant(
                        variant=variant,
                        acquisition_score=p_row[1],
                        acquisition_type=p_row[2],
                        predicted_score=p_row[3],
                        predicted_uncertainty=p_row[4],
                    )
                )

            rounds.append(
                Round(
                    id=round_id,
                    name=round_row[1],
                    description=round_row[2],
                    timestamp=datetime.fromisoformat(round_row[3]),
                    proposed_variants=proposed_variants,
                )
            )

        conn.close()

        return rounds

    def get_next_round_id(self) -> int:
        """
        Get the next round ID.

        Returns:
            Next round ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT MAX(id) FROM rounds")
        max_id = cursor.fetchone()[0]

        conn.close()

        return 1 if max_id is None else max_id + 1

    def complete_round(self, round_id: int) -> None:
        """
        Mark a round as completed.

        Args:
            round_id: ID of the round to complete
        """
        # Currently, rounds are implicitly completed when a new round is started
        # This method is a placeholder for future functionality
        pass

    def export_to_csv(self, output_dir: Union[str, Path]) -> None:
        """
        Export database to CSV files.

        Args:
            output_dir: Directory to save CSV files
        """
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        conn = sqlite3.connect(self.db_path)

        # Export variants
        pd.read_sql_query("SELECT * FROM variants", conn).to_csv(
            output_dir / "variants.csv", index=False
        )

        # Export assay results
        pd.read_sql_query("SELECT * FROM assay_results", conn).to_csv(
            output_dir / "assay_results.csv", index=False
        )

        # Export rounds
        pd.read_sql_query("SELECT * FROM rounds", conn).to_csv(
            output_dir / "rounds.csv", index=False
        )

        # Export proposed variants
        pd.read_sql_query("SELECT * FROM proposed_variants", conn).to_csv(
            output_dir / "proposed_variants.csv", index=False
        )

        conn.close()

        logger.info(f"Exported database to {output_dir}")

"""Data models for the PLM Framework."""
from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class Variant(BaseModel):
    """Protein sequence variant."""

    id: Union[int, str] = Field(...,
                                description="Unique identifier for the variant")
    name: str = Field(..., description="Name of the variant")
    sequence: str = Field(..., description="Amino acid sequence")

    @validator("sequence")
    def validate_sequence(cls, v):
        """Validate that the sequence contains only valid amino acids."""
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(aa in valid_aas for aa in v):
            invalid_aas = set(v) - valid_aas
            raise ValueError(
                f"Sequence contains invalid amino acids: {invalid_aas}")
        return v


class AssayResult(BaseModel):
    """Experimental assay result."""

    variant_id: Union[int, str] = Field(..., description="ID of the variant")
    score: float = Field(..., description="Assay score (higher is better)")
    uncertainty: Optional[float] = Field(
        None, description="Uncertainty of the measurement")
    assay_id: str = Field("default_assay", description="ID of the assay")
    round_id: Optional[int] = Field(None, description="ID of the round")


class ProposedVariant(BaseModel):
    """Variant proposed by the acquisition function."""

    variant: Variant = Field(..., description="The proposed variant")
    acquisition_score: float = Field(..., description="Acquisition score")
    acquisition_type: str = Field(...,
                                  description="Type of acquisition function used")
    predicted_score: Optional[float] = Field(
        None, description="Predicted score")
    predicted_uncertainty: Optional[float] = Field(
        None, description="Predicted uncertainty")


class Round(BaseModel):
    """Active learning round."""

    id: int = Field(..., description="Round number")
    name: str = Field(..., description="Round name")
    description: Optional[str] = Field(None, description="Round description")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Timestamp")
    proposed_variants: List[ProposedVariant] = Field(
        default_factory=list, description="Proposed variants")
    assay_results: List[AssayResult] = Field(
        default_factory=list, description="Assay results")
    metrics: Dict[str, float] = Field(
        default_factory=dict, description="Performance metrics")

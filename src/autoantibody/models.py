"""Pydantic data models for the autoantibody optimization system."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}


class ScorerTier(StrEnum):
    """Tier classification for ddG scoring tools."""

    FAST = "fast"
    MEDIUM = "medium"
    ORACLE = "oracle"
    FILTER = "filter"


class ScorerInfo(BaseModel):
    """Metadata for a ddG scoring tool."""

    name: str
    tier: ScorerTier
    script_path: Path
    requires_gpu: bool = False
    typical_seconds: float = 0.0
    description: str = ""


class Mutation(BaseModel):
    """A single amino acid point mutation.

    Mutations are specified as chain:resnum:wt_aa:mut_aa strings.
    Examples: H:52:S:Y, L:91:N:D, H:100A:G:A
    """

    chain: str
    resnum: str
    wt_aa: str
    mut_aa: str

    @field_validator("wt_aa", "mut_aa")
    @classmethod
    def _validate_amino_acid(cls, v: str) -> str:
        if v not in STANDARD_AMINO_ACIDS:
            raise ValueError(f"'{v}' is not a standard amino acid")
        return v

    def __str__(self) -> str:
        return f"{self.chain}:{self.resnum}:{self.wt_aa}:{self.mut_aa}"

    @classmethod
    def parse(cls, s: str) -> Mutation:
        """Parse a mutation string like 'H:52:S:Y'."""
        parts = s.strip().split(":")
        if len(parts) != 4:
            raise ValueError(f"Expected 4 colon-separated fields in '{s}', got {len(parts)}")
        return cls(chain=parts[0], resnum=parts[1], wt_aa=parts[2], mut_aa=parts[3])

    def to_evoef(self) -> str:
        """Convert to EvoEF mutation format: 'SH52Y'.

        This is the same format used by SKEMPIv2: wt_aa + chain + resnum + mut_aa.
        """
        return f"{self.wt_aa}{self.chain}{self.resnum}{self.mut_aa}"

    def to_stabddg(self) -> str:
        """Convert to StaB-ddG mutation format: 'SH52Y'.

        StaB-ddG uses the same SKEMPI-style format as EvoEF.
        """
        return f"{self.wt_aa}{self.chain}{self.resnum}{self.mut_aa}"

    def to_rosetta_resfile(self) -> str:
        """Generate a Rosetta resfile for this single mutation."""
        return f"NATAA\nstart\n{self.resnum} {self.chain} PIKAA {self.mut_aa}\n"

    def to_skempi(self) -> str:
        """Convert to SKEMPIv2 mutation format: 'SH52Y'."""
        return f"{self.wt_aa}{self.chain}{self.resnum}{self.mut_aa}"

    @classmethod
    def from_skempi(cls, s: str) -> Mutation:
        """Parse SKEMPIv2 mutation format like 'SH52Y'."""
        wt_aa = s[0]
        chain = s[1]
        mut_aa = s[-1]
        resnum = s[2:-1]
        return cls(chain=chain, resnum=resnum, wt_aa=wt_aa, mut_aa=mut_aa)

    def apply_to_sequence(self, sequence: str, residue_index_map: list[str]) -> str:
        """Apply this mutation to a sequence using a residue number map.

        Args:
            sequence: The amino acid sequence to mutate.
            residue_index_map: Ordered list of PDB residue numbers (e.g.,
                ["1", "2", "3", "100A", "101"]) matching sequence positions.

        Returns:
            The mutated sequence.

        Raises:
            ValueError: If residue number not found or wild-type mismatch.
        """
        if self.resnum not in residue_index_map:
            raise ValueError(f"Residue {self.resnum} not found in index map")
        idx = residue_index_map.index(self.resnum)
        if sequence[idx] != self.wt_aa:
            raise ValueError(
                f"Wild-type mismatch at position {idx} (resnum {self.resnum}): "
                f"expected {self.wt_aa}, found {sequence[idx]}"
            )
        return sequence[:idx] + self.mut_aa + sequence[idx + 1 :]


class ToolResult(BaseModel):
    """Standardized result from a tool execution."""

    status: Literal["ok", "error"]
    scores: dict[str, float] = {}
    artifacts: dict[str, str] = {}
    wall_time_s: float = 0.0
    error_message: str | None = None
    scorer_name: str | None = None


class ParentState(BaseModel):
    """Current parent antibody state within a campaign."""

    sequence_heavy: str
    sequence_light: str
    structure: str
    ddg_cumulative: float = 0.0


class CampaignState(BaseModel):
    """Full state of an optimization campaign."""

    campaign_id: str
    started_at: datetime
    iteration: int = 0
    parent: ParentState
    antigen_chains: list[str]
    antibody_heavy_chain: str
    antibody_light_chain: str
    frozen_positions: list[str] = []


class IterationDecision(BaseModel):
    """Record of one iteration's mutation decision."""

    iteration: int
    mutation: str
    proxy_scores: dict[str, float] = {}
    oracle_ddg: float
    accepted: bool
    rationale: str
    timestamp: datetime

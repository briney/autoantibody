# Workplan: Autoantibody Affinity Optimization MVP

Reference architecture: [`docs/streamlined_architecture.md`](docs/streamlined_architecture.md)

## Overview

This workplan implements the streamlined MVP for autonomous antibody affinity
optimization. Claude Code serves as the agent runtime, plain Python scripts are
the tool wrappers, and flat files (YAML + JSONL) track state. The deliverable is
a working optimization loop: Claude Code reads an agent program, screens
candidate mutations with a tiered suite of open/permissive-license ΔΔG scorers,
evaluates the best candidate with an oracle, and iteratively optimizes antibody
binding affinity through single-point mutations.

### Tool Suite

| Tool | Type | Binding ΔΔG? | Speed | SKEMPI2 Perf | License | GPU? |
|------|------|-------------|-------|-------------|---------|------|
| **EvoEF** | Physics energy fn | Yes | ~5s CPU | PCC 0.53 | Academic, source avail | No |
| **StaB-ddG** | ML (ProteinMPNN) | Yes | ~mins GPU | Spearman 0.45 | Unclear (ICML 2025) | Yes |
| **Graphinity** | Equivariant GNN | Yes (Ab-Ag specific) | Fast GPU | Pearson ~0.87 | Open source (oxpig) | Yes |
| **BA-ddG** | ML (Boltzmann) | Yes | Mins GPU | Spearman 0.51 | BSD-2 (academic) | Yes |
| **ProteinMPNN-ddG** | ML | **Stability only** | ~9800 res/s | N/A for binding | Docker images avail | Yes/CPU |
| **Rosetta flex-ddG** | Physics (MC) | Yes | 30-60 min | PCC ~0.46 | Docker | No |
| **AToM-OpenMM** | Physics (FEP/MD) | Yes | 4-6 hrs GPU | RMSE ~1.0-1.5 | LGPL | Yes |

### Tiered Architecture

- **Tier 1 — Fast Proxy (seconds):** EvoEF, StaB-ddG, Graphinity. Screen 10-20 candidates.
- **Tier 2 — Medium (minutes):** BA-ddG. Additional signal on top 2-3 candidates.
- **Filter:** ProteinMPNN-ddG (stability), ablms (sequence plausibility). Safety checks.
- **Tier 3 — Oracle (30 min - 6 hrs):** flex-ddG (default), AToM-OpenMM (high-confidence).

The agent learns which scorers perform best on a given binding interaction over
iterative design rounds by computing running correlations between each fast scorer
and oracle outcomes.

## File Manifest

| File | Purpose | Phase |
|------|---------|-------|
| `pyproject.toml` | Updated dependencies | 1 |
| `src/autoantibody/models.py` | Pydantic data models + ScorerTier, ScorerInfo | 1 |
| `src/autoantibody/scorers.py` | Scorer registry + availability checks | 1 |
| `src/autoantibody/structure.py` | PDB parsing and interface detection | 1 |
| `src/autoantibody/state.py` | Campaign state management | 1 |
| `tools/evoef_ddg.py` | EvoEF ΔΔG proxy scorer (Tier 1) | 2A |
| `tools/stabddg_score.py` | StaB-ddG fast ML scorer (Tier 1) | 2A |
| `tools/graphinity_score.py` | Graphinity Ab-Ag ddG scorer (Tier 1) | 2A |
| `tools/ablms_score.py` | Antibody LM sequence scoring (Filter) | 2A |
| `tools/flex_ddg.py` | Rosetta flex-ddG oracle (Docker, Tier 3) | 2B |
| `tools/baddg_score.py` | BA-ddG medium scorer (Tier 2) | 2B |
| `tools/stability_check.py` | ProteinMPNN-ddG stability filter (Filter) | 2B |
| `tools/atom_ddg.py` | AToM-OpenMM FEP oracle (Tier 3) | 2C |
| `scripts/init_campaign.py` | Campaign initialization CLI | 3 |
| `configs/example_campaign.yaml` | Example campaign config | 3 |
| `programs/program.md` | Agent behavior program (multi-scorer) | 3 |
| `tests/conftest.py` | Shared test fixtures | 4 |
| `tests/test_models.py` | Data model unit tests | 4 |
| `tests/test_scorers.py` | Scorer registry tests | 4 |
| `tests/test_structure.py` | Structure utility tests | 4 |
| `tests/test_state.py` | State management tests | 4 |
| `tests/test_tools.py` | Tool wrapper integration tests | 4 |
| `scripts/benchmark_skempi.py` | Multi-scorer SKEMPIv2 validation script | 4 |

## Prerequisites

**Software:**

- Python 3.12+
- Docker with `rosettacommons/rosetta` image (for flex-ddG oracle)
- GPU with CUDA support (for ML scorers and ablms inference)

**Installation:**

```bash
# Core
pip install -e ".[dev]"
pip install git+https://github.com/briney/ablms.git

# EvoEF (required — primary fast proxy)
git clone https://github.com/tommyhuangthu/EvoEF.git /opt/evoef
cd /opt/evoef && make
export EVOEF_BINARY=/opt/evoef/EvoEF

# StaB-ddG (recommended)
git clone https://github.com/LDeng0205/StaB-ddG.git /opt/stabddg
cd /opt/stabddg && pip install -e .  # or conda env create -f environment.yaml

# Graphinity (recommended)
pip install git+https://github.com/oxpig/Graphinity.git

# flex-ddG oracle (required)
docker pull rosettacommons/rosetta

# BA-ddG (optional)
git clone https://github.com/aim-uofa/BA-DDG.git /opt/baddg
cd /opt/baddg && conda env create -f env.yml
# + download weights per README

# ProteinMPNN-ddG stability filter (optional)
docker pull ghcr.io/peptoneltd/proteinmpnn_ddg:1.0.0_base

# AToM-OpenMM (optional, deferred)
conda install -c conda-forge openmm openmmtools ambertools
pip install AToM-OpenMM
Rscript -e "install.packages('UWHAM', repos='https://cran.r-project.org')"

# Verify core tools
$EVOEF_BINARY -h
docker run --rm rosettacommons/rosetta rosetta_scripts.default.linuxgccrelease --help
python -c "from ablms import load_model; print('ablms OK')"
```

**Test structure:** PDB 1N8Z (trastuzumab Fab / HER2 ECD). Download into
`tests/data/`:

```bash
mkdir -p tests/data
curl -o tests/data/1N8Z.pdb "https://files.rcsb.org/download/1N8Z.pdb"
```

## Architecture Deviation

The architecture places structure utilities in `tools/structure_utils.py`. This
workplan moves them to `src/autoantibody/structure.py` so they're importable by
tools, scripts, and tests without path manipulation. The tools import from the
installed `autoantibody` package instead.

---

## Phase 1: Core Library

**Objective:** Data models, scorer registry, structure parsing, state management.
All subsequent phases depend on this.

### 1.1 Dependencies

Add runtime dependencies to `pyproject.toml`:

```toml
dependencies = [
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "biopython>=1.80",
    "numpy>=1.24",
    "typer>=0.9",
]
```

Add `runs/` to `.gitignore`:

```
# Campaign run data
runs/
```

### 1.2 Data Models

**File:** `src/autoantibody/models.py`

```python
"""Pydantic data models for the autoantibody optimization system."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
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
            raise ValueError(
                f"Expected 4 colon-separated fields in '{s}', got {len(parts)}"
            )
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
```

### 1.3 Scorer Registry

**File:** `src/autoantibody/scorers.py`

The scorer registry provides metadata about all supported scorers and
lightweight availability checks. No heavy imports — just filesystem/env checks.

```python
"""Scorer registry and availability checks."""

from __future__ import annotations

import importlib
import os
import shutil
from pathlib import Path

from autoantibody.models import ScorerInfo, ScorerTier

SCORER_REGISTRY: dict[str, ScorerInfo] = {
    "evoef": ScorerInfo(
        name="evoef",
        tier=ScorerTier.FAST,
        script_path=Path("tools/evoef_ddg.py"),
        requires_gpu=False,
        typical_seconds=5.0,
        description="EvoEF physics-based binding ddG (~5s CPU, PCC 0.53 on SKEMPI2)",
    ),
    "stabddg": ScorerInfo(
        name="stabddg",
        tier=ScorerTier.FAST,
        script_path=Path("tools/stabddg_score.py"),
        requires_gpu=True,
        typical_seconds=60.0,
        description="StaB-ddG ML binding ddG (GPU, Spearman 0.45 on SKEMPI2)",
    ),
    "graphinity": ScorerInfo(
        name="graphinity",
        tier=ScorerTier.FAST,
        script_path=Path("tools/graphinity_score.py"),
        requires_gpu=True,
        typical_seconds=10.0,
        description=(
            "Graphinity equivariant GNN for Ab-Ag binding ddG "
            "(GPU, Pearson ~0.87 on SKEMPI2)"
        ),
    ),
    "baddg": ScorerInfo(
        name="baddg",
        tier=ScorerTier.MEDIUM,
        script_path=Path("tools/baddg_score.py"),
        requires_gpu=True,
        typical_seconds=180.0,
        description="BA-ddG Boltzmann-averaged ML binding ddG (GPU, Spearman 0.51)",
    ),
    "proteinmpnn_stability": ScorerInfo(
        name="proteinmpnn_stability",
        tier=ScorerTier.FILTER,
        script_path=Path("tools/stability_check.py"),
        requires_gpu=False,
        typical_seconds=30.0,
        description=(
            "ProteinMPNN-ddG fold stability filter (Docker). "
            "NOT a binding ddG scorer — rejects fold-destabilizing mutations."
        ),
    ),
    "ablms": ScorerInfo(
        name="ablms",
        tier=ScorerTier.FILTER,
        script_path=Path("tools/ablms_score.py"),
        requires_gpu=True,
        typical_seconds=10.0,
        description="Antibody language model sequence plausibility scoring",
    ),
    "flex_ddg": ScorerInfo(
        name="flex_ddg",
        tier=ScorerTier.ORACLE,
        script_path=Path("tools/flex_ddg.py"),
        requires_gpu=False,
        typical_seconds=2400.0,
        description="Rosetta flex-ddG oracle via Docker (30-60 min, PCC ~0.46)",
    ),
    "atom_fep": ScorerInfo(
        name="atom_fep",
        tier=ScorerTier.ORACLE,
        script_path=Path("tools/atom_ddg.py"),
        requires_gpu=True,
        typical_seconds=18000.0,
        description="AToM-OpenMM alchemical FEP oracle (4-6 hrs GPU, RMSE ~1.0-1.5)",
    ),
}


def check_scorer_available(name: str) -> bool:
    """Check if a scorer's dependencies are available.

    Checks environment variables, binaries, importable packages, and/or Docker
    images as appropriate for each scorer.
    """
    if name not in SCORER_REGISTRY:
        return False

    match name:
        case "evoef":
            env = os.environ.get("EVOEF_BINARY")
            if env:
                return Path(env).exists()
            return shutil.which("evoef") is not None

        case "stabddg":
            try:
                importlib.import_module("stabddg")
                return True
            except ImportError:
                return False

        case "graphinity":
            try:
                importlib.import_module("graphinity")
                return True
            except ImportError:
                return False

        case "baddg":
            baddg_dir = os.environ.get("BADDG_DIR")
            if baddg_dir:
                return Path(baddg_dir).is_dir()
            return Path("/opt/baddg").is_dir()

        case "proteinmpnn_stability":
            return shutil.which("docker") is not None

        case "ablms":
            try:
                importlib.import_module("ablms")
                return True
            except ImportError:
                return False

        case "flex_ddg":
            return shutil.which("docker") is not None

        case "atom_fep":
            try:
                importlib.import_module("openmm")
                importlib.import_module("openmmtools")
                return True
            except ImportError:
                return False

        case _:
            return False


def get_available_scorers() -> list[ScorerInfo]:
    """Return only scorers whose dependencies are installed."""
    return [
        info for name, info in SCORER_REGISTRY.items()
        if check_scorer_available(name)
    ]


def get_scorers_by_tier(tier: ScorerTier) -> list[ScorerInfo]:
    """Return all registered scorers in a given tier."""
    return [info for info in SCORER_REGISTRY.values() if info.tier == tier]
```

### 1.4 Structure Utilities

**File:** `src/autoantibody/structure.py`

```python
"""PDB structure parsing and analysis utilities."""

from __future__ import annotations

import logging
from pathlib import Path

from Bio.PDB import NeighborSearch, PDBParser
from Bio.PDB.Residue import Residue as PDBResidue

from autoantibody.models import THREE_TO_ONE, Mutation

logger = logging.getLogger(__name__)

_parser = PDBParser(QUIET=True)


def _is_standard_residue(residue: PDBResidue) -> bool:
    return residue.id[0] == " "


def _residue_one_letter(residue: PDBResidue) -> str:
    return THREE_TO_ONE.get(residue.get_resname(), "X")


def _resnum_str(residue: PDBResidue) -> str:
    seq_id = residue.id[1]
    icode = residue.id[2].strip()
    return f"{seq_id}{icode}"


def extract_sequences(pdb_path: Path | str) -> dict[str, str]:
    """Extract amino acid sequences for all protein chains.

    Args:
        pdb_path: Path to PDB file.

    Returns:
        Dict mapping chain ID to one-letter amino acid sequence.
    """
    structure = _parser.get_structure("s", str(pdb_path))
    sequences: dict[str, str] = {}
    for chain in structure[0]:
        residues = [r for r in chain.get_residues() if _is_standard_residue(r)]
        if residues:
            sequences[chain.id] = "".join(_residue_one_letter(r) for r in residues)
    return sequences


def get_residue_index_map(pdb_path: Path | str, chain_id: str) -> list[str]:
    """Get ordered list of PDB residue numbers for a chain.

    The list index corresponds to the 0-based sequence position. Each element
    is the PDB residue number string (e.g., "52", "100A").

    Args:
        pdb_path: Path to PDB file.
        chain_id: Chain identifier.

    Returns:
        Ordered list of residue number strings.
    """
    structure = _parser.get_structure("s", str(pdb_path))
    chain = structure[0][chain_id]
    return [
        _resnum_str(r) for r in chain.get_residues() if _is_standard_residue(r)
    ]


def get_residue_map(pdb_path: Path | str, chain_id: str) -> dict[str, str]:
    """Map PDB residue numbers to one-letter amino acid codes for a chain.

    Args:
        pdb_path: Path to PDB file.
        chain_id: Chain identifier.

    Returns:
        Dict mapping resnum string to one-letter AA code.
    """
    structure = _parser.get_structure("s", str(pdb_path))
    chain = structure[0][chain_id]
    return {
        _resnum_str(r): _residue_one_letter(r)
        for r in chain.get_residues()
        if _is_standard_residue(r)
    }


def get_interface_residues(
    pdb_path: Path | str,
    antibody_chains: list[str],
    antigen_chains: list[str],
    distance_cutoff: float = 8.0,
) -> list[dict[str, str]]:
    """Find antibody residues at the antigen interface.

    Args:
        pdb_path: Path to PDB file.
        antibody_chains: Antibody chain IDs (e.g., ["H", "L"]).
        antigen_chains: Antigen chain IDs (e.g., ["A"]).
        distance_cutoff: Angstroms for interface contact definition.

    Returns:
        List of dicts with keys: chain, resnum, aa.
    """
    structure = _parser.get_structure("s", str(pdb_path))
    model = structure[0]

    ag_atoms = []
    for cid in antigen_chains:
        for r in model[cid].get_residues():
            if _is_standard_residue(r):
                ag_atoms.extend(r.get_atoms())

    if not ag_atoms:
        logger.warning("No antigen atoms found for chains %s", antigen_chains)
        return []

    ns = NeighborSearch(ag_atoms)
    interface: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for cid in antibody_chains:
        for residue in model[cid].get_residues():
            if not _is_standard_residue(residue):
                continue
            resnum = _resnum_str(residue)
            key = (cid, resnum)
            if key in seen:
                continue
            for atom in residue.get_atoms():
                if ns.search(atom.get_vector().get_array(), distance_cutoff):
                    seen.add(key)
                    interface.append({
                        "chain": cid,
                        "resnum": resnum,
                        "aa": _residue_one_letter(residue),
                    })
                    break

    return interface


def validate_mutation_against_structure(
    pdb_path: Path | str,
    mutation: Mutation,
) -> list[str]:
    """Validate a mutation against the PDB structure.

    Returns:
        List of error messages. Empty means valid.
    """
    errors: list[str] = []
    structure = _parser.get_structure("s", str(pdb_path))
    model = structure[0]

    chain_ids = [c.id for c in model.get_chains()]
    if mutation.chain not in chain_ids:
        errors.append(
            f"Chain '{mutation.chain}' not found. Available: {chain_ids}"
        )
        return errors

    residue_map = get_residue_map(pdb_path, mutation.chain)
    if mutation.resnum not in residue_map:
        errors.append(
            f"Residue {mutation.resnum} not found in chain {mutation.chain}"
        )
        return errors

    actual_aa = residue_map[mutation.resnum]
    if actual_aa != mutation.wt_aa:
        errors.append(
            f"Wild-type mismatch at {mutation.chain}:{mutation.resnum}: "
            f"expected {mutation.wt_aa}, found {actual_aa}"
        )

    if mutation.wt_aa == mutation.mut_aa:
        errors.append(f"Silent mutation: {mutation}")

    return errors


def validate_mutation_safety(
    pdb_path: Path | str,
    mutation: Mutation,
    frozen_positions: list[str] | None = None,
) -> list[str]:
    """Full safety validation before oracle evaluation.

    Checks structure match, frozen positions, and amino acid validity.

    Returns:
        List of error messages. Empty means safe.
    """
    errors = validate_mutation_against_structure(pdb_path, mutation)

    if frozen_positions:
        position_key = f"{mutation.chain}:{mutation.resnum}"
        if position_key in frozen_positions:
            errors.append(f"Position {position_key} is frozen")

    return errors
```

### 1.5 State Management

**File:** `src/autoantibody/state.py`

```python
"""Campaign state management: load, save, initialize, and ledger operations."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from autoantibody.models import CampaignState, IterationDecision, ParentState
from autoantibody.structure import extract_sequences


def load_state(campaign_dir: Path) -> CampaignState:
    """Load campaign state from state.yaml."""
    with open(campaign_dir / "state.yaml") as f:
        return CampaignState(**yaml.safe_load(f))


def save_state(campaign_dir: Path, state: CampaignState) -> None:
    """Save campaign state to state.yaml."""
    data = json.loads(state.model_dump_json())
    with open(campaign_dir / "state.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def append_ledger(campaign_dir: Path, decision: IterationDecision) -> None:
    """Append a decision to the mutation ledger (JSONL)."""
    with open(campaign_dir / "ledger.jsonl", "a") as f:
        f.write(decision.model_dump_json() + "\n")


def load_ledger(campaign_dir: Path) -> list[IterationDecision]:
    """Load all decisions from the mutation ledger."""
    ledger_path = campaign_dir / "ledger.jsonl"
    if not ledger_path.exists():
        return []
    decisions: list[IterationDecision] = []
    with open(ledger_path) as f:
        for line in f:
            line = line.strip()
            if line:
                decisions.append(IterationDecision(**json.loads(line)))
    return decisions


def init_campaign(
    campaign_dir: Path,
    pdb_path: Path,
    antibody_heavy_chain: str,
    antibody_light_chain: str,
    antigen_chains: list[str],
    campaign_id: str | None = None,
    frozen_positions: list[str] | None = None,
) -> CampaignState:
    """Initialize a new campaign from an input PDB.

    Creates directory structure, copies the input PDB, extracts sequences,
    and writes the initial state.yaml and empty ledger.

    Args:
        campaign_dir: Campaign directory (created if needed).
        pdb_path: Input antibody-antigen complex PDB.
        antibody_heavy_chain: Heavy chain ID (e.g., "H").
        antibody_light_chain: Light chain ID (e.g., "L").
        antigen_chains: Antigen chain IDs (e.g., ["A"]).
        campaign_id: Optional ID. Auto-generated from timestamp if omitted.
        frozen_positions: Positions to exclude (e.g., ["H:52", "L:91"]).

    Returns:
        The initialized CampaignState.
    """
    campaign_dir = Path(campaign_dir)
    for subdir in ["input", "iterations"]:
        (campaign_dir / subdir).mkdir(parents=True, exist_ok=True)

    input_pdb = campaign_dir / "input" / pdb_path.name
    shutil.copy2(pdb_path, input_pdb)

    sequences = extract_sequences(pdb_path)
    for label, cid in [("Heavy", antibody_heavy_chain), ("Light", antibody_light_chain)]:
        if cid not in sequences:
            raise ValueError(
                f"{label} chain '{cid}' not found. Available: {list(sequences.keys())}"
            )

    if campaign_id is None:
        campaign_id = f"cmp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    state = CampaignState(
        campaign_id=campaign_id,
        started_at=datetime.now(timezone.utc),
        iteration=0,
        parent=ParentState(
            sequence_heavy=sequences[antibody_heavy_chain],
            sequence_light=sequences[antibody_light_chain],
            structure=f"input/{pdb_path.name}",
            ddg_cumulative=0.0,
        ),
        antigen_chains=antigen_chains,
        antibody_heavy_chain=antibody_heavy_chain,
        antibody_light_chain=antibody_light_chain,
        frozen_positions=frozen_positions or [],
    )

    save_state(campaign_dir, state)
    (campaign_dir / "ledger.jsonl").touch()
    return state
```

### Phase 1 Definition of Done

- `pip install -e ".[dev]"` succeeds with new dependencies
- `python -c "from autoantibody.models import Mutation, ScorerTier, ScorerInfo; print(Mutation.parse('H:52:S:Y'))"` works
- `python -c "from autoantibody.scorers import get_available_scorers; print(get_available_scorers())"` works
- `python -c "from autoantibody.structure import extract_sequences"` works
- State round-trip: init_campaign → load_state → save_state produces identical YAML

---

## Phase 2A: Fast Proxy Scorers

**Objective:** Tier 1 tool wrappers callable by Claude Code via bash.
Each prints JSON to stdout conforming to the `ToolResult` schema.

All tool wrappers follow the same pattern: standalone Python script, takes CLI
args, prints `ToolResult` JSON to stdout, non-zero exit on error.

### 2A.1 EvoEF ΔΔG Proxy Scorer

**File:** `tools/evoef_ddg.py`

EvoEF provides fast (~5 seconds) physics-based ΔΔG estimates. It replaces
FoldX as the primary fast proxy scorer — EvoEF is freely available with source
code while FoldX requires a restrictive academic license.

```python
#!/usr/bin/env python
"""EvoEF ΔΔG proxy scorer.

Usage:
    python tools/evoef_ddg.py <pdb_path> <mutation>

Arguments:
    pdb_path: Path to the input PDB structure.
    mutation:  Mutation string (e.g., H:52:S:Y).

Output (stdout):
    JSON with status, scores.ddg, scorer_name, artifacts, wall_time_s.

Environment:
    EVOEF_BINARY: Path to the compiled EvoEF binary (required).

Example:
    EVOEF_BINARY=/opt/evoef/EvoEF python tools/evoef_ddg.py \
        runs/cmp_001/input/1N8Z.pdb H:52:S:Y
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from autoantibody.models import Mutation, ToolResult
from autoantibody.structure import validate_mutation_against_structure


def find_evoef_binary() -> Path:
    """Locate the EvoEF binary from EVOEF_BINARY env var or PATH."""
    env_path = os.environ.get("EVOEF_BINARY")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        print(f"EVOEF_BINARY={env_path} not found", file=sys.stderr)
        sys.exit(1)
    which = shutil.which("EvoEF")
    if which:
        return Path(which)
    print(
        "EvoEF not found. Set EVOEF_BINARY or add to PATH.", file=sys.stderr
    )
    sys.exit(1)


def run_evoef_repair(evoef: Path, pdb_path: Path, work_dir: Path) -> Path:
    """Run RepairStructure and return path to repaired PDB."""
    subprocess.run(
        [str(evoef), "--command=RepairStructure",
         f"--pdb={pdb_path.name}"],
        cwd=work_dir, capture_output=True, text=True, check=True,
    )
    repaired = work_dir / f"{pdb_path.stem}_Repair.pdb"
    if not repaired.exists():
        raise FileNotFoundError(f"RepairStructure did not produce {repaired.name}")
    return repaired


def run_evoef_build_mutant(
    evoef: Path, pdb_path: Path, mutation: Mutation, work_dir: Path,
) -> Path:
    """Run BuildMutant and return path to mutant PDB."""
    mut_str = mutation.to_evoef()
    subprocess.run(
        [str(evoef), "--command=BuildMutant",
         f"--pdb={pdb_path.name}",
         f"--mutant_file={mut_str}"],
        cwd=work_dir, capture_output=True, text=True, check=True,
    )
    mutant = work_dir / f"{pdb_path.stem}_Model_0001.pdb"
    if not mutant.exists():
        raise FileNotFoundError(f"BuildMutant did not produce {mutant.name}")
    return mutant


def run_evoef_compute_binding(
    evoef: Path, pdb_path: Path, work_dir: Path,
) -> float:
    """Run ComputeBinding and return the binding energy."""
    result = subprocess.run(
        [str(evoef), "--command=ComputeBinding",
         f"--pdb={pdb_path.name}"],
        cwd=work_dir, capture_output=True, text=True, check=True,
    )
    for line in result.stdout.splitlines():
        if "Total" in line and "=" in line:
            parts = line.split("=")
            return float(parts[-1].strip())
    raise ValueError("Could not parse binding energy from EvoEF output")


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <pdb_path> <mutation>", file=sys.stderr)
        sys.exit(1)

    pdb_path = Path(sys.argv[1]).resolve()
    mutation = Mutation.parse(sys.argv[2])

    errors = validate_mutation_against_structure(pdb_path, mutation)
    if errors:
        result = ToolResult(
            status="error", error_message="; ".join(errors),
            scorer_name="evoef",
        )
        print(result.model_dump_json(indent=2))
        sys.exit(1)

    evoef = find_evoef_binary()
    t0 = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="evoef_") as tmpdir:
        wd = Path(tmpdir)
        shutil.copy2(pdb_path, wd / pdb_path.name)

        try:
            repaired = run_evoef_repair(evoef, pdb_path, wd)

            # Compute binding energy for wild-type
            wt_binding = run_evoef_compute_binding(evoef, repaired, wd)

            # Build mutant and compute its binding energy
            mutant_pdb = run_evoef_build_mutant(evoef, repaired, mutation, wd)
            mut_binding = run_evoef_compute_binding(evoef, mutant_pdb, wd)

            ddg = mut_binding - wt_binding
            scores: dict[str, float] = {
                "ddg": round(ddg, 4),
                "wt_binding_energy": round(wt_binding, 4),
                "mut_binding_energy": round(mut_binding, 4),
            }
            artifacts: dict[str, str] = {"mutant_structure": str(mutant_pdb)}

            result = ToolResult(
                status="ok", scores=scores, artifacts=artifacts,
                wall_time_s=round(time.monotonic() - t0, 2),
                scorer_name="evoef",
            )
        except Exception as e:
            result = ToolResult(
                status="error", error_message=str(e),
                wall_time_s=round(time.monotonic() - t0, 2),
                scorer_name="evoef",
            )

    print(result.model_dump_json(indent=2))
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()
```

### 2A.2 StaB-ddG Fast ML Scorer

**File:** `tools/stabddg_score.py`

```python
#!/usr/bin/env python
"""StaB-ddG fast ML binding ddG scorer.

Usage:
    python tools/stabddg_score.py <pdb_path> <mutation> --chains HL_A
    python tools/stabddg_score.py <pdb_path> --batch <mutations_file> --chains HL_A

Arguments:
    pdb_path: Path to the input PDB structure.
    mutation:  Mutation string (e.g., H:52:S:Y).

Options:
    --chains SPEC     Chain specification: antibody_antigen (e.g., HL_A).
    --batch FILE      File with one mutation per line (for batch screening).

Output (stdout):
    JSON with status, scores.ddg, scorer_name, wall_time_s.

Requires:
    StaB-ddG package + checkpoint (stabddg.pt).

Example:
    python tools/stabddg_score.py runs/cmp_001/input/1N8Z.pdb H:52:S:Y --chains HL_A
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from autoantibody.models import Mutation, ToolResult
from autoantibody.structure import validate_mutation_against_structure


def run_stabddg(
    pdb_path: Path,
    mutations: list[Mutation],
    ab_chains: str,
    ag_chains: str,
) -> list[float]:
    """Run StaB-ddG prediction on one or more mutations.

    Args:
        pdb_path: Path to input PDB.
        mutations: List of mutations to score.
        ab_chains: Antibody chain IDs concatenated (e.g., "HL").
        ag_chains: Antigen chain IDs concatenated (e.g., "A").

    Returns:
        List of predicted ddG values, one per mutation.
    """
    from stabddg import predict_ddg  # type: ignore[import-untyped]

    mut_strs = [m.to_stabddg() for m in mutations]
    results = predict_ddg(
        pdb_file=str(pdb_path),
        mutations=mut_strs,
        partner_chains=[ab_chains, ag_chains],
    )
    return [float(r) for r in results]


def main() -> None:
    ap = argparse.ArgumentParser(description="StaB-ddG binding ddG scorer")
    ap.add_argument("pdb_path", type=Path)
    ap.add_argument("mutation", nargs="?", type=str, default=None)
    ap.add_argument("--chains", required=True,
                    help="Chain spec: antibody_antigen (e.g., HL_A)")
    ap.add_argument("--batch", type=Path, default=None,
                    help="File with one mutation per line")
    args = ap.parse_args()

    parts = args.chains.split("_")
    if len(parts) != 2:
        print("--chains must be antibody_antigen (e.g., HL_A)", file=sys.stderr)
        sys.exit(1)
    ab_chains, ag_chains = parts

    pdb_path = args.pdb_path.resolve()

    # Parse mutations
    if args.batch:
        mutations = [
            Mutation.parse(line.strip())
            for line in args.batch.read_text().splitlines()
            if line.strip()
        ]
    elif args.mutation:
        mutations = [Mutation.parse(args.mutation)]
    else:
        print("Provide a mutation or --batch file", file=sys.stderr)
        sys.exit(1)

    # Validate all mutations
    for m in mutations:
        errors = validate_mutation_against_structure(pdb_path, m)
        if errors:
            result = ToolResult(
                status="error",
                error_message=f"Mutation {m}: {'; '.join(errors)}",
                scorer_name="stabddg",
            )
            print(result.model_dump_json(indent=2))
            sys.exit(1)

    t0 = time.monotonic()
    try:
        ddg_values = run_stabddg(pdb_path, mutations, ab_chains, ag_chains)

        if len(mutations) == 1:
            scores: dict[str, float] = {"ddg": round(ddg_values[0], 4)}
        else:
            scores = {
                f"ddg_{m}": round(v, 4)
                for m, v in zip(mutations, ddg_values)
            }
            scores["ddg_best"] = round(min(ddg_values), 4)

        result = ToolResult(
            status="ok", scores=scores,
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="stabddg",
        )
    except Exception as e:
        result = ToolResult(
            status="error", error_message=str(e),
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="stabddg",
        )

    print(result.model_dump_json(indent=2))
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()
```

### 2A.3 Graphinity Ab-Ag ddG Scorer

**File:** `tools/graphinity_score.py`

```python
#!/usr/bin/env python
"""Graphinity equivariant GNN scorer for antibody-antigen binding ddG.

Usage:
    python tools/graphinity_score.py <pdb_path> <mutation> \
        --ab-chains HL --ag-chains A

Arguments:
    pdb_path: Path to the input PDB structure.
    mutation:  Mutation string (e.g., H:52:S:Y).

Options:
    --ab-chains STR   Antibody chain IDs (e.g., HL).
    --ag-chains STR   Antigen chain IDs (e.g., A).

Output (stdout):
    JSON with status, scores.ddg, scorer_name, wall_time_s.

Requires:
    Graphinity package (pip install from oxpig/Graphinity).

Example:
    python tools/graphinity_score.py runs/cmp_001/input/1N8Z.pdb H:52:S:Y \
        --ab-chains HL --ag-chains A
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from autoantibody.models import Mutation, ToolResult
from autoantibody.structure import validate_mutation_against_structure


def run_graphinity(
    pdb_path: Path,
    mutation: Mutation,
    ab_chains: str,
    ag_chains: str,
) -> float:
    """Run Graphinity prediction on a single mutation.

    Args:
        pdb_path: Path to input PDB.
        mutation: Mutation to score.
        ab_chains: Antibody chain IDs (e.g., "HL").
        ag_chains: Antigen chain IDs (e.g., "A").

    Returns:
        Predicted binding ddG value.
    """
    from graphinity import predict  # type: ignore[import-untyped]

    result = predict(
        pdb_file=str(pdb_path),
        mutation=mutation.to_skempi(),
        antibody_chains=list(ab_chains),
        antigen_chains=list(ag_chains),
    )
    return float(result)


def main() -> None:
    ap = argparse.ArgumentParser(description="Graphinity Ab-Ag ddG scorer")
    ap.add_argument("pdb_path", type=Path)
    ap.add_argument("mutation", type=str)
    ap.add_argument("--ab-chains", required=True,
                    help="Antibody chain IDs (e.g., HL)")
    ap.add_argument("--ag-chains", required=True,
                    help="Antigen chain IDs (e.g., A)")
    args = ap.parse_args()

    pdb_path = args.pdb_path.resolve()
    mutation = Mutation.parse(args.mutation)

    errors = validate_mutation_against_structure(pdb_path, mutation)
    if errors:
        result = ToolResult(
            status="error", error_message="; ".join(errors),
            scorer_name="graphinity",
        )
        print(result.model_dump_json(indent=2))
        sys.exit(1)

    t0 = time.monotonic()
    try:
        ddg = run_graphinity(
            pdb_path, mutation, args.ab_chains, args.ag_chains,
        )
        result = ToolResult(
            status="ok",
            scores={"ddg": round(ddg, 4)},
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="graphinity",
        )
    except Exception as e:
        result = ToolResult(
            status="error", error_message=str(e),
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="graphinity",
        )

    print(result.model_dump_json(indent=2))
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()
```

### 2A.4 ablms Sequence Scoring

**File:** `tools/ablms_score.py`

Uses the [`ablms`](https://github.com/briney/ablms) package for antibody
language model scoring. Supports three modes:

- **score**: Pseudo-log-likelihood of a paired antibody sequence
- **compare**: PLL delta between wild-type and mutant sequences
- **scan**: Per-position masked marginal analysis (perplexity/entropy)

Recommended models for this workflow:
- `balm` — paired antibody LM (BALM-paired, 1024-dim), antibody-specialized
- `igbert` — paired antibody LM (IgBERT, 768-dim)
- `esm2-650m` — general protein LM (ESM-2 650M), unpaired
- `ftesm` — fine-tuned ESM-2 for antibodies (paired)

```python
#!/usr/bin/env python
"""Antibody language model scoring via ablms.

Usage:
    # Score paired antibody sequence (pseudo-log-likelihood)
    python tools/ablms_score.py score \
        --heavy EVQLVES... --light DIQMTQS... --model balm

    # Compare wild-type vs mutant (delta PLL)
    python tools/ablms_score.py compare \
        --heavy EVQLVES... --light DIQMTQS... \
        --mut-heavy EVQLVEY... --model balm

    # Per-position masked marginal scan
    python tools/ablms_score.py scan \
        --heavy EVQLVES... --light DIQMTQS... --model balm

Output (stdout):
    JSON with status, scores, wall_time_s.
"""

from __future__ import annotations

import argparse
import sys
import time

from autoantibody.models import ToolResult


def build_sequence(heavy: str, light: str | None, model_name: str):
    """Build the appropriate sequence object for the model type."""
    from ablms import AntibodySequence

    if light:
        return AntibodySequence(heavy=heavy, light=light)
    return AntibodySequence(heavy=heavy)


def cmd_score(args: argparse.Namespace) -> dict[str, float]:
    """Compute pseudo-log-likelihood for a sequence."""
    from ablms import load_model

    model = load_model(args.model, devices=args.device)
    seq = build_sequence(args.heavy, args.light, args.model)
    pll_scores = model.pseudo_log_likelihood([seq])
    return {"pll": float(pll_scores[0])}


def cmd_compare(args: argparse.Namespace) -> dict[str, float]:
    """Compare PLL between wild-type and mutant sequences."""
    from ablms import load_model

    model = load_model(args.model, devices=args.device)
    wt_seq = build_sequence(args.heavy, args.light, args.model)

    mut_heavy = args.mut_heavy if args.mut_heavy else args.heavy
    mut_light = args.mut_light if args.mut_light else args.light
    mut_seq = build_sequence(mut_heavy, mut_light, args.model)

    pll_scores = model.pseudo_log_likelihood([wt_seq, mut_seq])
    wt_pll, mut_pll = float(pll_scores[0]), float(pll_scores[1])
    return {
        "wt_pll": wt_pll,
        "mut_pll": mut_pll,
        "delta_pll": mut_pll - wt_pll,
    }


def cmd_scan(args: argparse.Namespace) -> dict[str, float]:
    """Per-position masked marginal analysis."""
    from ablms import load_model

    model = load_model(args.model, devices=args.device)
    seq = build_sequence(args.heavy, args.light, args.model)
    scan_outputs = model.mask_scan([seq])
    output = scan_outputs[0]

    scores: dict[str, float] = {
        "mean_perplexity": float(output.perplexity(agg="mean")),
        "mean_entropy": float(output.entropy(agg="mean")),
        "mean_accuracy": float(output.accuracy(agg="mean")),
    }

    # Per-chain metrics if light chain is provided
    if args.light:
        scores["heavy_perplexity"] = float(
            output.get_chain_perplexity("heavy", agg="mean")
        )
        scores["light_perplexity"] = float(
            output.get_chain_perplexity("light", agg="mean")
        )
        scores["heavy_accuracy"] = float(
            output.get_chain_accuracy("heavy", agg="mean")
        )
        scores["light_accuracy"] = float(
            output.get_chain_accuracy("light", agg="mean")
        )

    # Per-position perplexity (as comma-separated string in artifacts,
    # too many values for the scores dict)
    per_pos = output.perplexity()
    if hasattr(per_pos, "tolist"):
        per_pos = per_pos.tolist()

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="ablms sequence scoring")
    parser.add_argument(
        "--device", default="cuda", help="Compute device (default: cuda)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared args
    for name in ["score", "compare", "scan"]:
        sp = sub.add_parser(name)
        sp.add_argument("--heavy", required=True, help="Heavy chain sequence")
        sp.add_argument("--light", default=None, help="Light chain sequence")
        sp.add_argument("--model", default="balm", help="Model name")
        if name == "compare":
            sp.add_argument("--mut-heavy", default=None,
                            help="Mutant heavy chain")
            sp.add_argument("--mut-light", default=None,
                            help="Mutant light chain")

    args = parser.parse_args()
    t0 = time.monotonic()

    try:
        if args.command == "score":
            scores = cmd_score(args)
        elif args.command == "compare":
            scores = cmd_compare(args)
        elif args.command == "scan":
            scores = cmd_scan(args)
        else:
            raise ValueError(f"Unknown command: {args.command}")

        result = ToolResult(
            status="ok", scores=scores,
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="ablms",
        )
    except Exception as e:
        result = ToolResult(
            status="error", error_message=str(e),
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="ablms",
        )

    print(result.model_dump_json(indent=2))
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()
```

### Phase 2A Definition of Done

- `python tools/evoef_ddg.py tests/data/1N8Z.pdb H:52:S:Y` prints valid JSON with `ddg` and `scorer_name: "evoef"`
- `python tools/stabddg_score.py tests/data/1N8Z.pdb H:52:S:Y --chains HL_A` prints valid JSON
- `python tools/graphinity_score.py tests/data/1N8Z.pdb H:52:S:Y --ab-chains HL --ag-chains A` prints valid JSON
- `python tools/ablms_score.py score --heavy EVQLVES... --light DIQMTQS... --model balm` prints valid JSON

---

## Phase 2B: Medium Scorers and Filters

**Objective:** Tier 2 scorer (BA-ddG), filter (ProteinMPNN stability), and the
oracle (flex-ddG) as standalone tool scripts.

### 2B.1 Rosetta flex-ddG Oracle

**File:** `tools/flex_ddg.py`

The oracle scorer. Runs the Rosetta flex-ddG protocol via the
`rosettacommons/rosetta` Docker image. The protocol:

1. Relax input with coordinate constraints (take lowest-energy structure)
2. Generate backrub ensemble from relaxed structure
3. For each backrub structure, repack+score wild-type and mutant interfaces
4. ΔΔG = mean(ΔG_mutant - ΔG_wildtype) across the ensemble

The wrapper generates all input files (XML protocols, resfiles, flags, driver
script), mounts them into the container, and parses the output.

```python
#!/usr/bin/env python
"""Rosetta flex-ddG oracle scorer via Docker.

Usage:
    python tools/flex_ddg.py <pdb_path> <mutation> [options]

Arguments:
    pdb_path: Path to input PDB structure.
    mutation:  Mutation string (e.g., H:52:S:Y).

Options:
    --nstruct N          Backrub structures to generate (default: 35)
    --backrub-trials N   Backrub MC trials per structure (default: 35000)
    --relax-nstruct N    Relaxation structures to generate (default: 5)
    --docker-image IMG   Docker image (default: rosettacommons/rosetta)
    --rosetta-bin BIN    Binary name in container
                         (default: rosetta_scripts.default.linuxgccrelease)
    --keep-workdir       Don't delete working directory on completion

Output (stdout):
    JSON with status, scores (ddg, ddg_std, n_structures), wall_time_s.

Example:
    # Fast test run (5 structures, 10000 trials)
    python tools/flex_ddg.py input/1N8Z.pdb H:52:S:Y \
        --nstruct 5 --backrub-trials 10000

    # Production run (default parameters, ~30-60 min)
    python tools/flex_ddg.py input/1N8Z.pdb H:52:S:Y
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from autoantibody.models import Mutation, ToolResult
from autoantibody.structure import validate_mutation_against_structure


# ── Rosetta XML protocols (embedded) ────────────────────────────────────── #

RELAX_FLAGS = """\
-relax:constrain_relax_to_start_coords
-relax:coord_constrain_sidechains
-relax:ramp_constraints false
-ex1
-ex2
-use_input_sc
-flip_HNQ
-no_optH false
"""

RELAX_XML = """\
<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="ref2015" weights="ref2015"/>
  </SCOREFXNS>
  <MOVERS>
    <FastRelax name="relax" scorefxn="ref2015" repeats="5"/>
  </MOVERS>
  <PROTOCOLS>
    <Add mover="relax"/>
  </PROTOCOLS>
</ROSETTASCRIPTS>
"""

BACKRUB_XML = """\
<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="ref2015" weights="ref2015"/>
  </SCOREFXNS>
  <MOVERS>
    <Backrub name="backrub" ntrials="%%ntrials%%"/>
  </MOVERS>
  <PROTOCOLS>
    <Add mover="backrub"/>
  </PROTOCOLS>
</ROSETTASCRIPTS>
"""

REPACK_SCORE_XML = """\
<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="ref2015" weights="ref2015"/>
    <ScoreFunction name="ref2015_soft" weights="ref2015_soft"/>
  </SCOREFXNS>
  <TASKOPERATIONS>
    <InitializeFromCommandline name="ifcl"/>
    <ReadResfile name="resfile" filename="%%resfile_path%%"/>
    <IncludeCurrent name="ic"/>
    <ExtraRotamersGeneric name="ex12" ex1="1" ex2="1"/>
  </TASKOPERATIONS>
  <MOVERS>
    <PackRotamersMover name="repack" scorefxn="ref2015_soft"
                       task_operations="ifcl,resfile,ic,ex12"/>
    <MinMover name="minimize" scorefxn="ref2015" chi="1" bb="0"
              jump="ALL" type="lbfgs_armijo_nonmonotone"
              tolerance="0.01" max_iter="5000"/>
    <InterfaceAnalyzerMover name="analyze" scorefxn="ref2015"
                            packstat="0" pack_separated="0"
                            pack_input="0" interface_sc="1"
                            tracer="0"/>
  </MOVERS>
  <PROTOCOLS>
    <Add mover="repack"/>
    <Add mover="minimize"/>
    <Add mover="analyze"/>
  </PROTOCOLS>
</ROSETTASCRIPTS>
"""


def generate_driver_script(
    nstruct: int,
    backrub_trials: int,
    relax_nstruct: int,
    rosetta_bin: str,
) -> str:
    """Generate the bash driver script that runs inside Docker.

    Steps:
    1. Relax input structure, keep lowest-energy result.
    2. For each of nstruct backrub structures:
       a. Generate backrub ensemble member.
       b. Repack + score wild-type interface (NATAA resfile).
       c. Apply mutation resfile, repack + score mutant interface.
    3. Write per-structure scores to CSV.
    """
    # Use doubled braces for bash variables that shouldn't be Python-interpolated
    return f"""\
#!/bin/bash
set -euo pipefail

WORK=/workdir
ROSETTA={rosetta_bin}

echo "=== Step 1: Relax ==="
mkdir -p $WORK/relax_pdbs
$ROSETTA \\
  -s $WORK/input.pdb \\
  -parser:protocol $WORK/relax.xml \\
  @$WORK/relax.flags \\
  -nstruct {relax_nstruct} \\
  -out:path:pdb $WORK/relax_pdbs/ \\
  -out:path:score $WORK/ \\
  -out:file:scorefile relax_scores.sc \\
  > $WORK/logs/relax.log 2>&1

# Find lowest-energy relaxed structure
BEST_TAG=$(grep '^SCORE:' $WORK/relax_scores.sc \\
  | grep -v description \\
  | sort -k2 -n \\
  | head -1 \\
  | awk '{{print $NF}}')
BEST_PDB=$(find $WORK/relax_pdbs -name "*${{BEST_TAG}}*" -type f | head -1)
if [ -z "$BEST_PDB" ]; then
  BEST_PDB=$(ls $WORK/relax_pdbs/*.pdb | head -1)
fi
echo "Best relaxed: $BEST_PDB"

# Initialize results
echo "idx,wt_dG_separated,mut_dG_separated,ddg" > $WORK/results.csv

echo "=== Step 2: Backrub + scoring ==="
for i in $(seq 1 {nstruct}); do
  echo "--- Structure $i/{nstruct} ---"
  D=$WORK/iter_$i
  mkdir -p $D

  # Backrub
  $ROSETTA \\
    -s $BEST_PDB \\
    -parser:protocol $WORK/backrub.xml \\
    -parser:script_vars ntrials={backrub_trials} \\
    -nstruct 1 \\
    -out:prefix br${{i}}_ \\
    -out:path:pdb $D/ \\
    -out:path:score $D/ \\
    > $D/backrub.log 2>&1

  BR_PDB=$(ls $D/br${{i}}_*.pdb 2>/dev/null | head -1)
  if [ -z "$BR_PDB" ]; then
    echo "  WARN: no backrub PDB produced, skipping"
    continue
  fi

  # Wild-type repack + interface score
  $ROSETTA \\
    -s $BR_PDB \\
    -parser:protocol $WORK/repack_score.xml \\
    -parser:script_vars resfile_path=$WORK/wt.resfile \\
    -nstruct 1 \\
    -out:prefix wt${{i}}_ \\
    -out:path:pdb $D/ \\
    -out:path:score $D/ \\
    -out:file:scorefile wt.sc \\
    > $D/wt.log 2>&1

  # Mutant repack + interface score
  $ROSETTA \\
    -s $BR_PDB \\
    -parser:protocol $WORK/repack_score.xml \\
    -parser:script_vars resfile_path=$WORK/mut.resfile \\
    -nstruct 1 \\
    -out:prefix mut${{i}}_ \\
    -out:path:pdb $D/ \\
    -out:path:score $D/ \\
    -out:file:scorefile mut.sc \\
    > $D/mut.log 2>&1

  # Extract dG_separated column
  WT_DG=$(awk '
    NR==1 {{for(i=1;i<=NF;i++) if($i=="dG_separated") c=i}}
    NR==2 && c {{print $c}}
  ' $D/wt.sc)
  MUT_DG=$(awk '
    NR==1 {{for(i=1;i<=NF;i++) if($i=="dG_separated") c=i}}
    NR==2 && c {{print $c}}
  ' $D/mut.sc)

  if [ -n "$WT_DG" ] && [ -n "$MUT_DG" ]; then
    DDG=$(echo "$MUT_DG - $WT_DG" | bc -l)
    echo "$i,$WT_DG,$MUT_DG,$DDG" >> $WORK/results.csv
    echo "  wt_dG=$WT_DG  mut_dG=$MUT_DG  ddG=$DDG"
  else
    echo "  WARN: could not extract dG_separated"
  fi
done

echo "=== Complete ==="
"""


def setup_workdir(
    work_dir: Path, pdb_path: Path, mutation: Mutation
) -> None:
    """Write all input files to the working directory."""
    (work_dir / "logs").mkdir()
    shutil.copy2(pdb_path, work_dir / "input.pdb")
    (work_dir / "relax.xml").write_text(RELAX_XML)
    (work_dir / "relax.flags").write_text(RELAX_FLAGS)
    (work_dir / "backrub.xml").write_text(BACKRUB_XML)
    (work_dir / "repack_score.xml").write_text(REPACK_SCORE_XML)
    (work_dir / "wt.resfile").write_text("NATAA\nstart\n")
    (work_dir / "mut.resfile").write_text(mutation.to_rosetta_resfile())


def parse_results_csv(results_csv: Path) -> dict[str, float]:
    """Parse the flex-ddG results CSV and return aggregate ΔΔG."""
    ddg_values: list[float] = []
    with open(results_csv) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                try:
                    ddg_values.append(float(parts[3]))
                except ValueError:
                    continue
    if not ddg_values:
        raise ValueError("No valid ΔΔG values in results CSV")
    return {
        "ddg": round(float(np.mean(ddg_values)), 4),
        "ddg_std": round(float(np.std(ddg_values)), 4),
        "n_structures": float(len(ddg_values)),
    }


def run_flex_ddg(
    pdb_path: Path,
    mutation: Mutation,
    work_dir: Path,
    nstruct: int = 35,
    backrub_trials: int = 35000,
    relax_nstruct: int = 5,
    docker_image: str = "rosettacommons/rosetta",
    rosetta_bin: str = "rosetta_scripts.default.linuxgccrelease",
) -> dict[str, float]:
    """Run flex-ddG via Docker and return ΔΔG scores."""
    setup_workdir(work_dir, pdb_path, mutation)

    driver = generate_driver_script(
        nstruct=nstruct, backrub_trials=backrub_trials,
        relax_nstruct=relax_nstruct, rosetta_bin=rosetta_bin,
    )
    driver_path = work_dir / "run.sh"
    driver_path.write_text(driver)
    driver_path.chmod(0o755)

    proc = subprocess.run(
        ["docker", "run", "--rm",
         "-v", f"{work_dir}:/workdir",
         docker_image,
         "bash", "/workdir/run.sh"],
        capture_output=True, text=True,
        timeout=7200,  # 2-hour hard timeout
    )

    # Save Docker logs regardless of outcome
    (work_dir / "logs" / "docker_stdout.log").write_text(proc.stdout)
    (work_dir / "logs" / "docker_stderr.log").write_text(proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError(
            f"flex-ddG Docker failed (exit {proc.returncode}). "
            f"See {work_dir}/logs/ for details."
        )

    results_csv = work_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(
            f"No results.csv produced. See {work_dir}/logs/"
        )
    return parse_results_csv(results_csv)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rosetta flex-ddG oracle")
    ap.add_argument("pdb_path", type=Path)
    ap.add_argument("mutation", type=str)
    ap.add_argument("--nstruct", type=int, default=35)
    ap.add_argument("--backrub-trials", type=int, default=35000)
    ap.add_argument("--relax-nstruct", type=int, default=5)
    ap.add_argument("--docker-image", default="rosettacommons/rosetta")
    ap.add_argument(
        "--rosetta-bin",
        default="rosetta_scripts.default.linuxgccrelease",
    )
    ap.add_argument("--keep-workdir", action="store_true")
    args = ap.parse_args()

    pdb_path = args.pdb_path.resolve()
    mutation = Mutation.parse(args.mutation)

    errors = validate_mutation_against_structure(pdb_path, mutation)
    if errors:
        r = ToolResult(
            status="error", error_message="; ".join(errors),
            scorer_name="flex_ddg",
        )
        print(r.model_dump_json(indent=2))
        sys.exit(1)

    t0 = time.monotonic()
    tmpdir_obj = tempfile.TemporaryDirectory(prefix="flex_ddg_")
    work_dir = Path(tmpdir_obj.name)

    try:
        scores = run_flex_ddg(
            pdb_path, mutation, work_dir,
            nstruct=args.nstruct,
            backrub_trials=args.backrub_trials,
            relax_nstruct=args.relax_nstruct,
            docker_image=args.docker_image,
            rosetta_bin=args.rosetta_bin,
        )
        result = ToolResult(
            status="ok", scores=scores,
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="flex_ddg",
        )
    except Exception as e:
        result = ToolResult(
            status="error", error_message=str(e),
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="flex_ddg",
        )
    finally:
        if not args.keep_workdir:
            tmpdir_obj.cleanup()
        else:
            print(f"Work directory preserved: {work_dir}", file=sys.stderr)

    print(result.model_dump_json(indent=2))
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()
```

### 2B.2 BA-ddG Medium-Accuracy Scorer

**File:** `tools/baddg_score.py`

```python
#!/usr/bin/env python
"""BA-ddG medium-accuracy binding ddG scorer.

Usage:
    python tools/baddg_score.py <pdb_path> <mutation> --chains HL_A
    python tools/baddg_score.py <pdb_path> <mutation> --chains HL_A --mode unsupervised

Arguments:
    pdb_path: Path to the input PDB structure.
    mutation:  Mutation string (e.g., H:52:S:Y).

Options:
    --chains SPEC     Chain specification: antibody_antigen (e.g., HL_A).
    --mode MODE       Model mode: 'supervised' (default) or 'unsupervised' (BA-Cycle).

Output (stdout):
    JSON with status, scores.ddg, scorer_name, wall_time_s.

Requires:
    BA-ddG repository (BADDG_DIR env var or /opt/baddg) +
    conda environment (env.yml) + weights from Google Drive.

Example:
    python tools/baddg_score.py runs/cmp_001/input/1N8Z.pdb H:52:S:Y --chains HL_A
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from autoantibody.models import Mutation, ToolResult
from autoantibody.structure import validate_mutation_against_structure


def find_baddg_dir() -> Path:
    """Locate the BA-ddG installation directory."""
    env = os.environ.get("BADDG_DIR")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    default = Path("/opt/baddg")
    if default.is_dir():
        return default
    print("BA-ddG not found. Set BADDG_DIR or install to /opt/baddg.", file=sys.stderr)
    sys.exit(1)


def run_baddg(
    baddg_dir: Path,
    pdb_path: Path,
    mutation: Mutation,
    ab_chains: str,
    ag_chains: str,
    mode: str = "supervised",
) -> float:
    """Run BA-ddG prediction via subprocess.

    BA-ddG has its own conda environment, so we invoke it as a subprocess
    rather than importing directly.
    """
    mut_str = mutation.to_skempi()
    partner_str = f"{ab_chains}_{ag_chains}"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="baddg_input_"
    ) as f:
        f.write("pdb_file,mutation,partner_chains\n")
        f.write(f"{pdb_path},{mut_str},{partner_str}\n")
        input_csv = f.name

    script = baddg_dir / "predict.py"
    cmd = [
        sys.executable, str(script),
        "--input", input_csv,
        "--mode", mode,
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
        cwd=str(baddg_dir),
    )

    Path(input_csv).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"BA-ddG failed:\n{result.stderr}")

    # Parse predicted ddG from stdout (last numeric value)
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        try:
            return float(line)
        except ValueError:
            continue
    raise ValueError("Could not parse ddG from BA-ddG output")


def main() -> None:
    ap = argparse.ArgumentParser(description="BA-ddG binding ddG scorer")
    ap.add_argument("pdb_path", type=Path)
    ap.add_argument("mutation", type=str)
    ap.add_argument("--chains", required=True,
                    help="Chain spec: antibody_antigen (e.g., HL_A)")
    ap.add_argument("--mode", default="supervised",
                    choices=["supervised", "unsupervised"],
                    help="Prediction mode (default: supervised)")
    args = ap.parse_args()

    parts = args.chains.split("_")
    if len(parts) != 2:
        print("--chains must be antibody_antigen (e.g., HL_A)", file=sys.stderr)
        sys.exit(1)
    ab_chains, ag_chains = parts

    pdb_path = args.pdb_path.resolve()
    mutation = Mutation.parse(args.mutation)

    errors = validate_mutation_against_structure(pdb_path, mutation)
    if errors:
        result = ToolResult(
            status="error", error_message="; ".join(errors),
            scorer_name="baddg",
        )
        print(result.model_dump_json(indent=2))
        sys.exit(1)

    baddg_dir = find_baddg_dir()
    t0 = time.monotonic()

    try:
        ddg = run_baddg(baddg_dir, pdb_path, mutation, ab_chains, ag_chains,
                        mode=args.mode)
        result = ToolResult(
            status="ok",
            scores={"ddg": round(ddg, 4)},
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="baddg",
        )
    except Exception as e:
        result = ToolResult(
            status="error", error_message=str(e),
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="baddg",
        )

    print(result.model_dump_json(indent=2))
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()
```

### 2B.3 ProteinMPNN-ddG Stability Filter

**File:** `tools/stability_check.py`

This is NOT a binding ΔΔG scorer — it checks whether a mutation destabilizes
the antibody fold. Mutations with stability_ddg > 2.0 kcal/mol should be
flagged as potentially fold-destabilizing.

```python
#!/usr/bin/env python
"""ProteinMPNN-ddG fold stability filter.

Usage:
    python tools/stability_check.py <pdb_path> <mutation> --chain H

Arguments:
    pdb_path: Path to the input PDB structure.
    mutation:  Mutation string (e.g., H:52:S:Y).

Options:
    --chain STR    Chain ID to evaluate stability for.

Output (stdout):
    JSON with status, scores.stability_ddg, scorer_name, wall_time_s.
    NOT a binding ddG — used as a filter to reject fold-destabilizing mutations.
    Flag mutations with stability_ddg > 2.0 kcal/mol.

Requires:
    Docker with ghcr.io/peptoneltd/proteinmpnn_ddg:1.0.0_base image.

Example:
    python tools/stability_check.py runs/cmp_001/input/1N8Z.pdb H:52:S:Y --chain H
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from autoantibody.models import Mutation, ToolResult
from autoantibody.structure import validate_mutation_against_structure

DOCKER_IMAGE = "ghcr.io/peptoneltd/proteinmpnn_ddg:1.0.0_base"


def run_stability_check(
    pdb_path: Path,
    mutation: Mutation,
    chain_id: str,
) -> float:
    """Run ProteinMPNN-ddG stability prediction via Docker.

    Returns:
        Predicted stability ddG (positive = destabilizing).
    """
    with tempfile.TemporaryDirectory(prefix="stability_") as tmpdir:
        wd = Path(tmpdir)
        shutil.copy2(pdb_path, wd / "input.pdb")

        # Create input JSON for ProteinMPNN-ddG
        input_data = {
            "pdb_file": "/workdir/input.pdb",
            "chain": chain_id,
            "mutation": mutation.to_skempi(),
        }
        (wd / "input.json").write_text(json.dumps(input_data))

        proc = subprocess.run(
            ["docker", "run", "--rm",
             "-v", f"{wd}:/workdir",
             DOCKER_IMAGE,
             "python", "/app/predict.py",
             "--input", "/workdir/input.json",
             "--output", "/workdir/output.json"],
            capture_output=True, text=True,
            timeout=300,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"ProteinMPNN-ddG Docker failed (exit {proc.returncode}): "
                f"{proc.stderr}"
            )

        output_path = wd / "output.json"
        if not output_path.exists():
            raise FileNotFoundError("No output.json produced")

        output = json.loads(output_path.read_text())
        return float(output["ddg"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ProteinMPNN-ddG fold stability filter"
    )
    ap.add_argument("pdb_path", type=Path)
    ap.add_argument("mutation", type=str)
    ap.add_argument("--chain", required=True, help="Chain ID to evaluate")
    args = ap.parse_args()

    pdb_path = args.pdb_path.resolve()
    mutation = Mutation.parse(args.mutation)

    errors = validate_mutation_against_structure(pdb_path, mutation)
    if errors:
        result = ToolResult(
            status="error", error_message="; ".join(errors),
            scorer_name="proteinmpnn_stability",
        )
        print(result.model_dump_json(indent=2))
        sys.exit(1)

    t0 = time.monotonic()
    try:
        stability_ddg = run_stability_check(pdb_path, mutation, args.chain)
        result = ToolResult(
            status="ok",
            scores={"stability_ddg": round(stability_ddg, 4)},
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="proteinmpnn_stability",
        )
    except Exception as e:
        result = ToolResult(
            status="error", error_message=str(e),
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="proteinmpnn_stability",
        )

    print(result.model_dump_json(indent=2))
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()
```

### Phase 2B Definition of Done

- `python tools/flex_ddg.py tests/data/1N8Z.pdb H:52:S:Y --nstruct 2 --backrub-trials 5000` prints valid JSON
- `python tools/baddg_score.py tests/data/1N8Z.pdb H:52:S:Y --chains HL_A` prints valid JSON
- `python tools/stability_check.py tests/data/1N8Z.pdb H:52:S:Y --chain H` prints valid JSON

---

## Phase 2C: High-Accuracy Oracle (Deferrable)

**Objective:** AToM-OpenMM alchemical FEP oracle for high-confidence ΔΔG
predictions. This is independent and can be deferred until the main loop is
proven.

### 2C.1 AToM-OpenMM FEP Oracle

**File:** `tools/atom_ddg.py`

The most complex wrapper. Uses alchemical free energy perturbation with
replica exchange molecular dynamics to compute binding ΔΔG.

```python
#!/usr/bin/env python
"""AToM-OpenMM alchemical FEP oracle for binding ddG.

Usage:
    python tools/atom_ddg.py <pdb_path> <mutation> [options]

Arguments:
    pdb_path: Path to the input PDB structure.
    mutation:  Mutation string (e.g., H:52:S:Y).

Options:
    --lambda-windows N   Number of lambda windows (default: 24).
    --gpu-id ID          GPU device ID (default: 0).
    --steps-per-window N MD steps per lambda window (default: 50000).

Output (stdout):
    JSON with status, scores.ddg, scores.ddg_uncertainty,
    scorer_name="atom_fep", wall_time_s.

Requires:
    openmm, openmmtools, ambertools (conda),
    AToM-OpenMM (pip), R + UWHAM package.

Example:
    python tools/atom_ddg.py runs/cmp_001/input/1N8Z.pdb H:52:S:Y \
        --lambda-windows 24 --gpu-id 0

Note: 4-6 hours per mutation on GPU. Agent uses this only when high confidence
is needed, or to calibrate cheaper scorers.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from autoantibody.models import Mutation, ToolResult
from autoantibody.structure import validate_mutation_against_structure


def prepare_system(
    pdb_path: Path,
    mutation: Mutation,
    work_dir: Path,
) -> Path:
    """Prepare the alchemical system using AmberTools tleap.

    Parameterizes both wild-type and mutant, solvates, and generates
    topology/coordinate files.

    Returns:
        Path to the prepared system directory.
    """
    system_dir = work_dir / "system"
    system_dir.mkdir()

    # Generate tleap input for solvation and parameterization
    tleap_input = f"""\
source leaprc.protein.ff14SB
source leaprc.water.tip3p
mol = loadpdb {pdb_path}
solvatebox mol TIP3PBOX 12.0
saveamberparm mol {system_dir}/complex.prmtop {system_dir}/complex.inpcrd
quit
"""
    tleap_file = work_dir / "tleap.in"
    tleap_file.write_text(tleap_input)

    result = subprocess.run(
        ["tleap", "-f", str(tleap_file)],
        capture_output=True, text=True, cwd=str(work_dir),
    )
    if result.returncode != 0:
        raise RuntimeError(f"tleap failed:\n{result.stderr}")

    return system_dir


def setup_alchemy(
    system_dir: Path,
    mutation: Mutation,
    lambda_windows: int,
    work_dir: Path,
) -> Path:
    """Set up alchemical lambda schedule for wt→mut transformation.

    Returns:
        Path to the alchemy config directory.
    """
    alchemy_dir = work_dir / "alchemy"
    alchemy_dir.mkdir()

    # Generate lambda schedule (evenly spaced)
    lambdas = [i / (lambda_windows - 1) for i in range(lambda_windows)]
    schedule_file = alchemy_dir / "lambda_schedule.dat"
    schedule_file.write_text("\n".join(f"{lam:.6f}" for lam in lambdas))

    # Write AToM config
    config = {
        "topology": str(system_dir / "complex.prmtop"),
        "coordinates": str(system_dir / "complex.inpcrd"),
        "mutation": mutation.to_skempi(),
        "lambda_schedule": str(schedule_file),
        "n_windows": lambda_windows,
    }
    import json
    (alchemy_dir / "config.json").write_text(json.dumps(config, indent=2))

    return alchemy_dir


def run_md(
    alchemy_dir: Path,
    gpu_id: int,
    steps_per_window: int,
    work_dir: Path,
) -> Path:
    """Run replica exchange MD via OpenMM.

    Returns:
        Path to the MD output directory containing energy samples.
    """
    md_dir = work_dir / "md_output"
    md_dir.mkdir()

    import json
    config = json.loads((alchemy_dir / "config.json").read_text())

    # This is a placeholder for the actual AToM-OpenMM API call.
    # The real implementation would use:
    #   from AToM import ATomSystem, ATomREMD
    #   system = ATomSystem(config)
    #   remd = ATomREMD(system, gpu_id=gpu_id)
    #   remd.run(steps=steps_per_window)
    #   remd.save_energies(md_dir / "energies.dat")

    result = subprocess.run(
        [sys.executable, "-m", "AToM.run_remd",
         "--config", str(alchemy_dir / "config.json"),
         "--gpu-id", str(gpu_id),
         "--steps", str(steps_per_window),
         "--output", str(md_dir)],
        capture_output=True, text=True,
        timeout=36000,  # 10-hour hard timeout
    )

    if result.returncode != 0:
        raise RuntimeError(f"AToM MD failed:\n{result.stderr}")

    return md_dir


def analyze_fep(md_dir: Path, work_dir: Path) -> tuple[float, float]:
    """Run UWHAM free energy analysis on MD samples.

    Returns:
        Tuple of (ddg, ddg_uncertainty) in kcal/mol.
    """
    # Use R + UWHAM for free energy estimation
    r_script = work_dir / "uwham_analysis.R"
    r_script.write_text(f"""\
library(UWHAM)
energies <- read.table("{md_dir}/energies.dat", header=TRUE)
# UWHAM analysis to compute free energy differences
result <- uwham(energies$lambda, energies$reduced_potential)
ddg <- result$free_energy[length(result$free_energy)] - result$free_energy[1]
ddg_err <- sqrt(result$variance[length(result$variance)])
cat(ddg, ddg_err, sep="\\n")
""")

    result = subprocess.run(
        ["Rscript", str(r_script)],
        capture_output=True, text=True, timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(f"UWHAM analysis failed:\n{result.stderr}")

    lines = result.stdout.strip().splitlines()
    ddg = float(lines[0])
    ddg_uncertainty = float(lines[1])
    return ddg, ddg_uncertainty


def main() -> None:
    ap = argparse.ArgumentParser(description="AToM-OpenMM FEP oracle")
    ap.add_argument("pdb_path", type=Path)
    ap.add_argument("mutation", type=str)
    ap.add_argument("--lambda-windows", type=int, default=24)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--steps-per-window", type=int, default=50000)
    args = ap.parse_args()

    pdb_path = args.pdb_path.resolve()
    mutation = Mutation.parse(args.mutation)

    errors = validate_mutation_against_structure(pdb_path, mutation)
    if errors:
        result = ToolResult(
            status="error", error_message="; ".join(errors),
            scorer_name="atom_fep",
        )
        print(result.model_dump_json(indent=2))
        sys.exit(1)

    t0 = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="atom_fep_") as tmpdir:
        wd = Path(tmpdir)
        shutil.copy2(pdb_path, wd / pdb_path.name)

        try:
            system_dir = prepare_system(pdb_path, mutation, wd)
            alchemy_dir = setup_alchemy(
                system_dir, mutation, args.lambda_windows, wd
            )
            md_dir = run_md(
                alchemy_dir, args.gpu_id, args.steps_per_window, wd
            )
            ddg, ddg_uncertainty = analyze_fep(md_dir, wd)

            result = ToolResult(
                status="ok",
                scores={
                    "ddg": round(ddg, 4),
                    "ddg_uncertainty": round(ddg_uncertainty, 4),
                },
                wall_time_s=round(time.monotonic() - t0, 2),
                scorer_name="atom_fep",
            )
        except Exception as e:
            result = ToolResult(
                status="error", error_message=str(e),
                wall_time_s=round(time.monotonic() - t0, 2),
                scorer_name="atom_fep",
            )

    print(result.model_dump_json(indent=2))
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()
```

### Phase 2C Definition of Done

- `python tools/atom_ddg.py tests/data/1N8Z.pdb H:52:S:Y --lambda-windows 4 --steps-per-window 1000` prints valid JSON (short test run)

---

## Phase 3: Agent Program & Campaign Infrastructure

**Objective:** Create the campaign initialization script, example config, and
the agent program that drives the multi-scorer optimization loop.

### 3.1 Campaign Initialization

**File:** `scripts/init_campaign.py`

```python
#!/usr/bin/env python
"""Initialize a new antibody optimization campaign.

Usage:
    python scripts/init_campaign.py \
        --pdb /path/to/complex.pdb \
        --heavy-chain H --light-chain L \
        --antigen-chains A \
        --output runs/my_campaign

    python scripts/init_campaign.py --config configs/example_campaign.yaml
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from autoantibody.scorers import get_available_scorers
from autoantibody.state import init_campaign
from autoantibody.structure import extract_sequences, get_interface_residues

app = typer.Typer(help="Initialize an antibody optimization campaign.")


@app.command()
def main(
    pdb: Path = typer.Option(None, help="Input PDB structure"),
    heavy_chain: str = typer.Option(None, "--heavy-chain", help="Heavy chain ID"),
    light_chain: str = typer.Option(None, "--light-chain", help="Light chain ID"),
    antigen_chains: list[str] = typer.Option(
        None, "--antigen-chains", help="Antigen chain IDs"
    ),
    output: Path = typer.Option(None, help="Campaign output directory"),
    campaign_id: str = typer.Option(None, "--campaign-id", help="Campaign ID"),
    frozen: list[str] = typer.Option(
        [], "--frozen", help="Frozen positions (e.g., H:52)"
    ),
    config: Path = typer.Option(
        None, "--config", help="YAML config (overrides other args)"
    ),
) -> None:
    """Create a campaign directory with state.yaml, empty ledger, and scorer inventory."""
    if config:
        with open(config) as f:
            cfg = yaml.safe_load(f)
        pdb = Path(cfg["pdb"])
        heavy_chain = cfg["heavy_chain"]
        light_chain = cfg["light_chain"]
        antigen_chains = cfg["antigen_chains"]
        output = Path(cfg["output"])
        campaign_id = cfg.get("campaign_id")
        frozen = cfg.get("frozen_positions", [])

    if not all([pdb, heavy_chain, light_chain, antigen_chains, output]):
        typer.echo("Error: provide all required args or --config", err=True)
        raise typer.Exit(1)

    # Show available chains first
    sequences = extract_sequences(pdb)
    typer.echo(f"Chains found in {pdb.name}:")
    for cid, seq in sequences.items():
        typer.echo(f"  {cid}: {len(seq)} residues")

    state = init_campaign(
        campaign_dir=output,
        pdb_path=pdb,
        antibody_heavy_chain=heavy_chain,
        antibody_light_chain=light_chain,
        antigen_chains=antigen_chains,
        campaign_id=campaign_id,
        frozen_positions=frozen,
    )

    # Write scorer inventory
    available = get_available_scorers()
    inventory = {
        s.name: {
            "tier": s.tier.value,
            "requires_gpu": s.requires_gpu,
            "typical_seconds": s.typical_seconds,
            "description": s.description,
        }
        for s in available
    }
    with open(output / "scorer_inventory.yaml", "w") as f:
        yaml.dump(inventory, f, default_flow_style=False, sort_keys=False)

    # Show interface summary
    interface = get_interface_residues(
        pdb,
        antibody_chains=[heavy_chain, light_chain],
        antigen_chains=antigen_chains,
    )
    typer.echo(f"\nCampaign initialized: {state.campaign_id}")
    typer.echo(f"  Heavy chain ({heavy_chain}): {len(state.parent.sequence_heavy)} residues")
    typer.echo(f"  Light chain ({light_chain}): {len(state.parent.sequence_light)} residues")
    typer.echo(f"  Interface residues: {len(interface)}")
    typer.echo(f"  Available scorers: {len(available)}")
    for s in available:
        typer.echo(f"    [{s.tier.value}] {s.name}: {s.description}")
    typer.echo(f"  Output: {output}")


if __name__ == "__main__":
    app()
```

### 3.2 Example Configuration

**File:** `configs/example_campaign.yaml`

```yaml
# Trastuzumab / HER2 optimization campaign
# Verify chain IDs match your PDB file before running.
pdb: tests/data/1N8Z.pdb
heavy_chain: H
light_chain: L
antigen_chains:
  - A
output: runs/trastuzumab_her2_001
campaign_id: trastuzumab_her2_001
frozen_positions: []

# Optional: scorer preferences (agent uses these as hints)
scorer_preferences:
  fast_screening:
    - evoef
    - stabddg
    - graphinity
  oracle: flex_ddg
  stability_filter: true
```

### 3.3 Agent Program

**File:** `programs/program.md`

This is the core of the system — the document that tells Claude Code how to run
the multi-scorer optimization loop. It must be comprehensive enough for autonomous
operation.

````markdown
# Antibody Affinity Optimization Program

## Objective

You are optimizing the binding affinity of an antibody to its antigen target
through iterative single-point mutations. Each iteration, you will:

1. Analyze the current parent antibody and its interface with the antigen.
2. Propose candidate mutations using structural reasoning.
3. Screen candidates with multiple fast scoring tools (Tier 1).
4. Rank by consensus — mutations where multiple scorers agree get priority.
5. Apply safety filters (sequence plausibility, fold stability).
6. Optionally validate top candidates with a medium scorer (Tier 2).
7. Select the single most promising mutation.
8. Evaluate it with the oracle scorer (Tier 3).
9. Accept the mutation if it improves binding (ΔΔG < 0); reject otherwise.
10. Update the campaign state and proceed to the next iteration.

Your goal is to **minimize cumulative ΔΔG** — each accepted mutation should
make the antibody bind more tightly.

## Campaign State

All campaign data lives in a single directory (e.g., `runs/my_campaign/`).

**Read current state:**
```bash
cat runs/<campaign>/state.yaml
```

**Read mutation history:**
```bash
cat runs/<campaign>/ledger.jsonl
```

**Read available scorers:**
```bash
cat runs/<campaign>/scorer_inventory.yaml
```

The state file contains the current parent sequences, structure path, cumulative
ΔΔG, chain assignments, and frozen positions.

## Available Tools

### Tier 1 — Fast Proxy Scorers (seconds)

Use these to screen 10-20 candidate mutations. Run all available Tier 1
scorers on each candidate for consensus ranking.

#### 1a. EvoEF ΔΔG (~5 seconds, CPU)

```bash
python tools/evoef_ddg.py <pdb_path> <mutation>
```

- **Use for**: Fast physics-based ΔΔG screening.
- **Input**: PDB path + mutation string (e.g., `H:52:S:Y`).
- **Output**: JSON with `scores.ddg` (kcal/mol). Negative = improved binding.
- **Accuracy**: PCC 0.53 on SKEMPI2. Directionally useful, not authoritative.

#### 1b. StaB-ddG (~1 min, GPU)

```bash
python tools/stabddg_score.py <pdb_path> <mutation> --chains HL_A
```

- **Use for**: ML-based ΔΔG screening. Supports batch mode.
- **Input**: PDB path + mutation + chain spec (antibody_antigen).
- **Output**: JSON with `scores.ddg`.
- **Batch mode**: `--batch <file>` with one mutation per line.

#### 1c. Graphinity (~10 seconds, GPU)

```bash
python tools/graphinity_score.py <pdb_path> <mutation> --ab-chains HL --ag-chains A
```

- **Use for**: Ab-Ag specific ddG prediction via equivariant GNN.
- **Input**: PDB path + mutation + antibody/antigen chain IDs.
- **Output**: JSON with `scores.ddg`.
- **Accuracy**: Pearson ~0.87 on SKEMPI2 (best among fast scorers for Ab-Ag).

### Tier 2 — Medium Scorer (minutes)

Use on top 1-2 candidates from Tier 1 consensus for additional validation
before committing to the expensive oracle.

#### 2. BA-ddG (~3 min, GPU)

```bash
python tools/baddg_score.py <pdb_path> <mutation> --chains HL_A
```

- **Use for**: Higher-accuracy ΔΔG validation on top candidates.
- **Input**: PDB path + mutation + chain spec.
- **Output**: JSON with `scores.ddg`.
- **Modes**: `--mode supervised` (default) or `--mode unsupervised`.

### Filters

Safety checks that do not predict binding ΔΔG directly.

#### F1. ablms — Sequence Plausibility (~5-15 seconds, GPU)

```bash
# Overall sequence quality
python tools/ablms_score.py score \
    --heavy <heavy_seq> --light <light_seq> --model balm

# Compare wild-type vs mutant
python tools/ablms_score.py compare \
    --heavy <wt_heavy> --light <wt_light> \
    --mut-heavy <mut_heavy> --model balm

# Per-position analysis
python tools/ablms_score.py scan \
    --heavy <heavy_seq> --light <light_seq> --model balm
```

- **Use for**: Checking that mutations don't destroy sequence plausibility.
  A mutation with good ΔΔG but terrible PLL delta should be viewed with
  suspicion.
- **Models**: Prefer `balm` (antibody-specific, paired). Use `esm2-650m` as
  a general protein baseline.

#### F2. ProteinMPNN-ddG — Fold Stability Filter (~30 seconds, Docker)

```bash
python tools/stability_check.py <pdb_path> <mutation> --chain H
```

- **Use for**: Rejecting mutations that destabilize the antibody fold.
- **Output**: `scores.stability_ddg`. This is NOT a binding ΔΔG.
- **Threshold**: Flag mutations with `stability_ddg > 2.0 kcal/mol` as
  potentially fold-destabilizing. Prefer candidates below this threshold.

### Tier 3 — Oracle Scorers (30 min - 6 hrs)

Run on exactly ONE mutation per iteration. The oracle is authoritative for
accept/reject decisions.

#### 3a. Rosetta flex-ddG — Default Oracle (~30-60 min, Docker)

```bash
python tools/flex_ddg.py <pdb_path> <mutation> [--nstruct N]
```

- **Use for**: Final accept/reject decision.
- **Output**: JSON with `scores.ddg` (mean), `scores.ddg_std`, `scores.n_structures`.
- **Authority**: This is the ground truth for accept/reject decisions.
- **Cost**: Expensive. Never run on more than one mutation per iteration.

For faster testing during development:
```bash
python tools/flex_ddg.py <pdb> <mutation> --nstruct 5 --backrub-trials 10000
```

#### 3b. AToM-OpenMM — High-Confidence Oracle (~4-6 hrs, GPU)

```bash
python tools/atom_ddg.py <pdb_path> <mutation> [--lambda-windows 24] [--gpu-id 0]
```

- **Use for**: When high confidence is needed, or to calibrate cheaper scorers.
- **Output**: JSON with `scores.ddg`, `scores.ddg_uncertainty`.
- **Cost**: Very expensive. Use only when explicitly configured or when
  flex-ddG results are ambiguous (ddg close to 0, high std).

### Structure Analysis (Python one-liners)

Get interface residues:
```bash
python -c "
from autoantibody.structure import get_interface_residues
import json
residues = get_interface_residues(
    'runs/<campaign>/input/complex.pdb',
    antibody_chains=['H', 'L'],
    antigen_chains=['A'],
    distance_cutoff=8.0,
)
print(json.dumps(residues, indent=2))
"
```

Get residue-to-sequence-position mapping:
```bash
python -c "
from autoantibody.structure import get_residue_index_map
rmap = get_residue_index_map('runs/<campaign>/input/complex.pdb', 'H')
for i, resnum in enumerate(rmap):
    print(f'  seq_pos={i}  pdb_resnum={resnum}')
"
```

## Iteration Procedure

Follow these steps for each iteration:

### Step 1: Read State

Read `state.yaml`, `ledger.jsonl`, and `scorer_inventory.yaml`. Note:
- Current parent sequences (heavy and light chains)
- Current parent structure path
- Cumulative ΔΔG so far
- Frozen positions (do not mutate these)
- Previously attempted mutations and their outcomes
- Which scorers are available

### Step 2: Identify the Interface

If this is the first iteration, compute interface residues using the structure
analysis tool above. Focus your search on antibody residues within 8Å of the
antigen.

### Step 3: Propose Candidate Mutations

Consider 10-20 candidate mutations. Prioritize:
- **Interface positions** (within 8Å of antigen) — these directly affect binding
- **CDR residues** over framework residues — CDRs form the binding surface
- **Positions not previously attempted** — avoid revisiting failed mutations
- **Structurally informed substitutions** (see Domain Knowledge below)

### Step 4: Screen with Tier 1 Scorers (Consensus Ranking)

Run all available Tier 1 scorers on your candidates:
```bash
# For each candidate mutation:
python tools/evoef_ddg.py <pdb_path> <mutation>
python tools/stabddg_score.py <pdb_path> <mutation> --chains HL_A
python tools/graphinity_score.py <pdb_path> <mutation> --ab-chains HL --ag-chains A
```

**Consensus ranking**: For each candidate, count how many scorers predict
ΔΔG < -0.5 kcal/mol. Rank by consensus count first, then by mean predicted
ΔΔG. Mutations where multiple scorers agree on improvement are more reliable.

Record all scorer results for the ledger:
```json
{"evoef_ddg": -1.2, "stabddg_ddg": -0.8, "graphinity_ddg": -1.0}
```

### Step 5: Check Sequence Plausibility

For your top 3-5 candidates (by consensus), run ablms comparison:
```bash
python tools/ablms_score.py compare \
    --heavy <wt_heavy> --light <wt_light> \
    --mut-heavy <mut_heavy> --model balm
```

Discard mutations where `delta_pll` is strongly negative (the mutation makes
the sequence significantly less natural).

### Step 6: Check Fold Stability (if available)

For your top 2-3 candidates, run the stability filter:
```bash
python tools/stability_check.py <pdb_path> <mutation> --chain H
```

Flag mutations with `stability_ddg > 2.0 kcal/mol`. Prefer candidates below
this threshold.

### Step 7: Optional Tier 2 Validation

If BA-ddG is available, run it on your top 1-2 candidates for additional signal:
```bash
python tools/baddg_score.py <pdb_path> <mutation> --chains HL_A
```

### Step 8: Select One Mutation

Choose the single best mutation considering:
1. Tier 1 consensus (how many fast scorers agree)
2. Mean predicted ΔΔG across scorers (lower = better binding)
3. ablms delta_pll (should not be strongly negative)
4. Stability check (prefer stability_ddg < 2.0)
5. Tier 2 validation (if available)
6. Structural reasoning (does the substitution make biophysical sense?)
7. Novelty (prefer unexplored positions/substitutions)

Write a brief rationale for your choice.

### Step 9: Validate Safety

Before running the oracle, verify:
- The target position is not frozen
- The wild-type amino acid matches the current parent sequence
- The mutant amino acid is one of the 20 standard amino acids
- Exactly one position is being changed

### Step 10: Run the Oracle

```bash
python tools/flex_ddg.py <pdb_path> <selected_mutation>
```

This takes 30-60 minutes. Wait for it to complete.

### Step 11: Accept or Reject

- **Accept** if oracle `scores.ddg < 0` (improved binding)
- **Reject** if oracle `scores.ddg >= 0` (neutral or worse)

### Step 12: Update State

Create the iteration directory and save the decision:
```bash
mkdir -p runs/<campaign>/iterations/<N>
```

Write `runs/<campaign>/iterations/<N>/decision.json`:
```json
{
  "iteration": N,
  "mutation": "H:52:S:Y",
  "proxy_scores": {
    "evoef_ddg": -1.2,
    "stabddg_ddg": -0.8,
    "graphinity_ddg": -1.0,
    "ablms_delta_pll": 0.05,
    "stability_ddg": 0.3,
    "baddg_ddg": -0.9
  },
  "oracle_ddg": -0.8,
  "accepted": true,
  "rationale": "Position 52 is interface-proximal. 3/3 Tier 1 scorers predict improvement...",
  "timestamp": "2026-03-17T15:42:00Z"
}
```

**If accepted**, update `state.yaml`:
- Increment `iteration`
- Update `parent.ddg_cumulative` by adding the oracle ΔΔG
- Apply the mutation to the parent sequence
- Update `parent.structure` if the oracle produced a mutant structure,
  otherwise keep the current structure

**If rejected**, update `state.yaml`:
- Increment `iteration` only
- All other fields stay the same

**Always** append to `ledger.jsonl`:
```bash
python -c "
from datetime import datetime, timezone
from autoantibody.models import IterationDecision
from autoantibody.state import append_ledger
from pathlib import Path

decision = IterationDecision(
    iteration=N,
    mutation='H:52:S:Y',
    proxy_scores={'evoef_ddg': -1.2, 'stabddg_ddg': -0.8, 'graphinity_ddg': -1.0},
    oracle_ddg=-0.8,
    accepted=True,
    rationale='...',
    timestamp=datetime.now(timezone.utc),
)
append_ledger(Path('runs/<campaign>'), decision)
"
```

### Step 13: Scorer Reliability Learning

After 5+ iterations with oracle results, compute running correlations between
each fast scorer's predictions and the oracle outcomes:

```python
import numpy as np
# For each Tier 1 scorer, compute Pearson correlation with oracle ddG
# Use this to weight the consensus ranking in future iterations
```

Adjust your screening weights accordingly. If a scorer consistently
disagrees with the oracle, downweight it. If one scorer is strongly
predictive, give it more influence in the consensus.

### Step 14: Report and Continue

Summarize the iteration result:
- Mutation attempted
- Tier 1 consensus scores
- Oracle ΔΔG
- Accept/reject decision
- Cumulative ΔΔG after this iteration
- Total accepted/rejected so far
- Scorer reliability notes (after 5+ iterations)

Then proceed to the next iteration (go back to Step 1).

## Constraints

- **One mutation per iteration.** Never apply multiple mutations at once.
- **Antibody residues only.** Never mutate antigen residues.
- **Standard amino acids only.** Only the 20 canonical amino acids.
- **Respect frozen positions.** Never mutate positions listed in
  `frozen_positions`.
- **Oracle is authoritative.** Always use the oracle ΔΔG for accept/reject,
  never the proxy scorers alone.
- **Don't repeat exact failures.** If a mutation was rejected, don't try
  the same position→AA substitution again.
- **Use all available scorers.** Check `scorer_inventory.yaml` and use
  every available Tier 1 scorer for consensus ranking.

## Domain Knowledge: Antibody-Antigen Binding

### Interface Architecture

- **CDR loops** (CDR-H1, H2, H3, L1, L2, L3) form the primary binding
  surface. CDR-H3 is usually the most critical for binding.
- **Framework residues** provide structural support. Mutating them can
  destabilize the antibody fold even if they don't contact the antigen.
- Typical antibody-antigen interfaces bury 600-1000 Å² of surface area
  per side, with 15-25 contact residues per chain.

### Favorable Substitution Patterns

- **Tyrosine (Y)** is the most enriched amino acid at antibody interfaces.
  It can form hydrogen bonds, aromatic stacking, and hydrophobic contacts.
- **Tryptophan (W)** provides large hydrophobic surface and aromatic
  interactions. Often beneficial at core interface positions.
- **Serine (S) → Tyrosine (Y)** is often beneficial at interface positions
  because Y preserves the hydroxyl while adding aromatic character.
- **Small → aromatic** substitutions at interface positions can fill
  cavities and improve packing.
- **Charge complementarity**: if the antigen surface is acidic, positively
  charged antibody residues (K, R) at the interface can be beneficial.

### Substitutions to Avoid

- **Proline (P)** in beta strands or the middle of alpha helices — it
  breaks secondary structure.
- **Glycine (G) at structured positions** — glycine is too flexible and
  can destabilize ordered regions. Exception: glycine is sometimes needed
  at tight turns.
- **Large → small at core packing positions** — creates cavities that
  destabilize the structure.
- **Removing buried salt bridges** — these contribute significant stability.
- **Disrupting conserved framework residues** — positions that are highly
  conserved across antibodies are likely structurally essential.

### Strategic Considerations

- **Start with interface-proximal positions.** Mutations far from the
  interface rarely affect binding.
- **Exploit existing cavities.** Small-to-large substitutions at positions
  adjacent to interface cavities can improve packing.
- **Consider the rejected mutation history.** If several mutations at one
  position all failed, the position may be structurally constrained.
- **Diversify your search.** Don't fixate on one region. Try different
  CDR loops and different substitution types.
- **Watch for destabilization.** A mutation might improve the interface
  contact but destabilize the antibody fold. The stability filter captures
  this — always check before committing to the oracle.
- **Trust consensus.** When multiple independent scorers agree, the
  prediction is more reliable than any single scorer.
````

### Phase 3 Definition of Done

- `python scripts/init_campaign.py --pdb tests/data/1N8Z.pdb --heavy-chain H --light-chain L --antigen-chains A --output runs/test_001` creates campaign structure + scorer_inventory.yaml
- `runs/test_001/state.yaml` contains correct sequences and chain assignments
- `programs/program.md` contains complete, self-contained agent instructions for multi-scorer paradigm
- Claude Code can follow `programs/program.md` without external guidance

---

## Phase 4: Testing & Validation

**Objective:** Unit tests for all core components, integration tests for tools,
and a multi-scorer SKEMPIv2 benchmark framework.

### 4.1 Test Fixtures

**File:** `tests/conftest.py`

```python
"""Shared test fixtures."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"
PDB_1N8Z = DATA_DIR / "1N8Z.pdb"
PDB_URL = "https://files.rcsb.org/download/1N8Z.pdb"


@pytest.fixture(scope="session")
def pdb_1n8z() -> Path:
    """Provide PDB 1N8Z, downloading if necessary."""
    if not PDB_1N8Z.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PDB_URL, PDB_1N8Z)
    return PDB_1N8Z


@pytest.fixture
def tmp_campaign(tmp_path: Path, pdb_1n8z: Path) -> Path:
    """Create a temporary campaign directory with 1N8Z."""
    from autoantibody.state import init_campaign

    campaign_dir = tmp_path / "test_campaign"
    init_campaign(
        campaign_dir=campaign_dir,
        pdb_path=pdb_1n8z,
        antibody_heavy_chain="H",
        antibody_light_chain="L",
        antigen_chains=["A"],
        campaign_id="test_cmp_001",
    )
    return campaign_dir
```

**Note:** Verify the chain IDs in 1N8Z before running tests. If the PDB uses
different chain letters, update the fixtures accordingly. Run:

```bash
python -c "
from autoantibody.structure import extract_sequences
seqs = extract_sequences('tests/data/1N8Z.pdb')
for cid, seq in seqs.items():
    print(f'{cid}: {len(seq)} residues — {seq[:20]}...')
"
```

### 4.2 Unit Tests — Models

**File:** `tests/test_models.py`

```python
"""Tests for data models."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoantibody.models import (
    CampaignState,
    IterationDecision,
    Mutation,
    ParentState,
    ScorerInfo,
    ScorerTier,
    ToolResult,
)


class TestMutation:
    def test_parse_simple(self) -> None:
        m = Mutation.parse("H:52:S:Y")
        assert m.chain == "H"
        assert m.resnum == "52"
        assert m.wt_aa == "S"
        assert m.mut_aa == "Y"

    def test_parse_insertion_code(self) -> None:
        m = Mutation.parse("H:100A:G:A")
        assert m.resnum == "100A"

    def test_str_roundtrip(self) -> None:
        original = "L:91:N:D"
        m = Mutation.parse(original)
        assert str(m) == original

    def test_invalid_amino_acid(self) -> None:
        with pytest.raises(ValueError, match="not a standard amino acid"):
            Mutation(chain="H", resnum="52", wt_aa="X", mut_aa="Y")

    def test_parse_wrong_field_count(self) -> None:
        with pytest.raises(ValueError, match="4 colon-separated"):
            Mutation.parse("H:52:S")

    def test_to_evoef(self) -> None:
        m = Mutation.parse("H:52:S:Y")
        assert m.to_evoef() == "SH52Y"

    def test_to_stabddg(self) -> None:
        m = Mutation.parse("H:52:S:Y")
        assert m.to_stabddg() == "SH52Y"

    def test_to_rosetta_resfile(self) -> None:
        m = Mutation.parse("H:52:S:Y")
        resfile = m.to_rosetta_resfile()
        assert "NATAA" in resfile
        assert "52 H PIKAA Y" in resfile

    def test_to_skempi(self) -> None:
        m = Mutation.parse("H:52:S:Y")
        assert m.to_skempi() == "SH52Y"

    def test_from_skempi(self) -> None:
        m = Mutation.from_skempi("SH52Y")
        assert m.chain == "H"
        assert m.resnum == "52"
        assert m.wt_aa == "S"
        assert m.mut_aa == "Y"

    def test_apply_to_sequence(self) -> None:
        seq = "ABCDEF"
        index_map = ["10", "11", "12", "13", "14", "15"]
        m = Mutation(chain="H", resnum="12", wt_aa="C", mut_aa="Y")
        result = m.apply_to_sequence(seq, index_map)
        assert result == "ABYDEF"

    def test_apply_to_sequence_wt_mismatch(self) -> None:
        seq = "ABCDEF"
        index_map = ["10", "11", "12", "13", "14", "15"]
        m = Mutation(chain="H", resnum="12", wt_aa="X", mut_aa="Y")
        with pytest.raises(ValueError, match="Wild-type mismatch"):
            m.apply_to_sequence(seq, index_map)


class TestToolResult:
    def test_ok_result(self) -> None:
        r = ToolResult(status="ok", scores={"ddg": -1.2}, wall_time_s=3.5)
        assert r.status == "ok"
        assert r.scores["ddg"] == -1.2

    def test_error_result(self) -> None:
        r = ToolResult(status="error", error_message="EvoEF not found")
        assert r.error_message == "EvoEF not found"

    def test_scorer_name(self) -> None:
        r = ToolResult(
            status="ok", scores={"ddg": -0.5},
            scorer_name="evoef",
        )
        assert r.scorer_name == "evoef"

    def test_json_roundtrip(self) -> None:
        r = ToolResult(
            status="ok", scores={"ddg": -0.5},
            wall_time_s=1.0, scorer_name="graphinity",
        )
        r2 = ToolResult.model_validate_json(r.model_dump_json())
        assert r2.scores == r.scores
        assert r2.scorer_name == r.scorer_name


class TestScorerTier:
    def test_tier_values(self) -> None:
        assert ScorerTier.FAST == "fast"
        assert ScorerTier.MEDIUM == "medium"
        assert ScorerTier.ORACLE == "oracle"
        assert ScorerTier.FILTER == "filter"

    def test_tier_is_strenum(self) -> None:
        assert str(ScorerTier.FAST) == "fast"
        assert f"tier={ScorerTier.ORACLE}" == "tier=oracle"


class TestScorerInfo:
    def test_create_scorer_info(self) -> None:
        info = ScorerInfo(
            name="evoef",
            tier=ScorerTier.FAST,
            script_path=Path("tools/evoef_ddg.py"),
            requires_gpu=False,
            typical_seconds=5.0,
            description="EvoEF physics-based binding ddG",
        )
        assert info.name == "evoef"
        assert info.tier == ScorerTier.FAST
        assert not info.requires_gpu

    def test_scorer_info_defaults(self) -> None:
        info = ScorerInfo(
            name="test",
            tier=ScorerTier.FILTER,
            script_path=Path("tools/test.py"),
        )
        assert info.requires_gpu is False
        assert info.typical_seconds == 0.0
        assert info.description == ""
```

### 4.3 Unit Tests — Scorer Registry

**File:** `tests/test_scorers.py`

```python
"""Tests for scorer registry and availability checks."""

from __future__ import annotations

from autoantibody.models import ScorerTier
from autoantibody.scorers import (
    SCORER_REGISTRY,
    check_scorer_available,
    get_available_scorers,
    get_scorers_by_tier,
)


class TestScorerRegistry:
    def test_registry_has_all_scorers(self) -> None:
        expected = {
            "evoef", "stabddg", "graphinity", "baddg",
            "proteinmpnn_stability", "ablms", "flex_ddg", "atom_fep",
        }
        assert set(SCORER_REGISTRY.keys()) == expected

    def test_each_scorer_has_required_fields(self) -> None:
        for name, info in SCORER_REGISTRY.items():
            assert info.name == name
            assert info.tier in ScorerTier
            assert info.script_path.suffix == ".py"
            assert isinstance(info.requires_gpu, bool)
            assert info.typical_seconds >= 0

    def test_tier_assignments(self) -> None:
        assert SCORER_REGISTRY["evoef"].tier == ScorerTier.FAST
        assert SCORER_REGISTRY["stabddg"].tier == ScorerTier.FAST
        assert SCORER_REGISTRY["graphinity"].tier == ScorerTier.FAST
        assert SCORER_REGISTRY["baddg"].tier == ScorerTier.MEDIUM
        assert SCORER_REGISTRY["proteinmpnn_stability"].tier == ScorerTier.FILTER
        assert SCORER_REGISTRY["ablms"].tier == ScorerTier.FILTER
        assert SCORER_REGISTRY["flex_ddg"].tier == ScorerTier.ORACLE
        assert SCORER_REGISTRY["atom_fep"].tier == ScorerTier.ORACLE


class TestScorerAvailability:
    def test_unknown_scorer_not_available(self) -> None:
        assert check_scorer_available("nonexistent") is False

    def test_get_available_returns_list(self) -> None:
        available = get_available_scorers()
        assert isinstance(available, list)
        # At minimum, we know the return type is correct
        for info in available:
            assert info.name in SCORER_REGISTRY

    def test_get_scorers_by_tier_fast(self) -> None:
        fast = get_scorers_by_tier(ScorerTier.FAST)
        assert len(fast) == 3
        names = {s.name for s in fast}
        assert names == {"evoef", "stabddg", "graphinity"}

    def test_get_scorers_by_tier_oracle(self) -> None:
        oracles = get_scorers_by_tier(ScorerTier.ORACLE)
        assert len(oracles) == 2
        names = {s.name for s in oracles}
        assert names == {"flex_ddg", "atom_fep"}

    def test_get_scorers_by_tier_filter(self) -> None:
        filters = get_scorers_by_tier(ScorerTier.FILTER)
        assert len(filters) == 2
        names = {s.name for s in filters}
        assert names == {"proteinmpnn_stability", "ablms"}
```

### 4.4 Unit Tests — Structure

**File:** `tests/test_structure.py`

```python
"""Tests for structure utilities using real PDB data."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoantibody.models import Mutation
from autoantibody.structure import (
    extract_sequences,
    get_interface_residues,
    get_residue_index_map,
    get_residue_map,
    validate_mutation_against_structure,
    validate_mutation_safety,
)


class TestExtractSequences:
    def test_extracts_all_chains(self, pdb_1n8z: Path) -> None:
        seqs = extract_sequences(pdb_1n8z)
        assert len(seqs) >= 2  # at least heavy + light + antigen
        for seq in seqs.values():
            assert len(seq) > 0
            assert all(c in "ACDEFGHIKLMNPQRSTVWYX" for c in seq)


class TestResidueMap:
    def test_residue_map_not_empty(self, pdb_1n8z: Path) -> None:
        seqs = extract_sequences(pdb_1n8z)
        for chain_id in seqs:
            rmap = get_residue_map(pdb_1n8z, chain_id)
            assert len(rmap) == len(seqs[chain_id])

    def test_index_map_matches_sequence(self, pdb_1n8z: Path) -> None:
        seqs = extract_sequences(pdb_1n8z)
        for chain_id, seq in seqs.items():
            idx_map = get_residue_index_map(pdb_1n8z, chain_id)
            assert len(idx_map) == len(seq)


class TestInterfaceResidues:
    def test_finds_interface(self, pdb_1n8z: Path) -> None:
        seqs = extract_sequences(pdb_1n8z)
        chains = list(seqs.keys())
        # Use first two chains as antibody, rest as antigen
        # Adjust chain assignments based on actual 1N8Z structure
        ab_chains = chains[:2]
        ag_chains = chains[2:]
        if not ag_chains:
            pytest.skip("Need at least 3 chains for interface test")
        interface = get_interface_residues(pdb_1n8z, ab_chains, ag_chains)
        assert len(interface) > 0
        for res in interface:
            assert "chain" in res
            assert "resnum" in res
            assert "aa" in res


class TestValidation:
    def test_valid_mutation(self, pdb_1n8z: Path) -> None:
        seqs = extract_sequences(pdb_1n8z)
        chain_id = list(seqs.keys())[0]
        rmap = get_residue_map(pdb_1n8z, chain_id)
        resnum, wt_aa = next(iter(rmap.items()))
        # Pick a different amino acid
        mut_aa = "A" if wt_aa != "A" else "G"
        m = Mutation(chain=chain_id, resnum=resnum, wt_aa=wt_aa, mut_aa=mut_aa)
        errors = validate_mutation_against_structure(pdb_1n8z, m)
        assert errors == []

    def test_wrong_chain(self, pdb_1n8z: Path) -> None:
        m = Mutation(chain="Z", resnum="1", wt_aa="A", mut_aa="G")
        errors = validate_mutation_against_structure(pdb_1n8z, m)
        assert any("not found" in e for e in errors)

    def test_wt_mismatch(self, pdb_1n8z: Path) -> None:
        seqs = extract_sequences(pdb_1n8z)
        chain_id = list(seqs.keys())[0]
        rmap = get_residue_map(pdb_1n8z, chain_id)
        resnum, actual_aa = next(iter(rmap.items()))
        wrong_aa = "W" if actual_aa != "W" else "K"
        m = Mutation(chain=chain_id, resnum=resnum, wt_aa=wrong_aa, mut_aa="A")
        errors = validate_mutation_against_structure(pdb_1n8z, m)
        assert any("mismatch" in e for e in errors)

    def test_frozen_position(self, pdb_1n8z: Path) -> None:
        seqs = extract_sequences(pdb_1n8z)
        chain_id = list(seqs.keys())[0]
        rmap = get_residue_map(pdb_1n8z, chain_id)
        resnum, wt_aa = next(iter(rmap.items()))
        mut_aa = "A" if wt_aa != "A" else "G"
        m = Mutation(chain=chain_id, resnum=resnum, wt_aa=wt_aa, mut_aa=mut_aa)
        errors = validate_mutation_safety(
            pdb_1n8z, m, frozen_positions=[f"{chain_id}:{resnum}"]
        )
        assert any("frozen" in e for e in errors)
```

### 4.5 Unit Tests — State

**File:** `tests/test_state.py`

```python
"""Tests for state management."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from autoantibody.models import IterationDecision
from autoantibody.state import (
    append_ledger,
    init_campaign,
    load_ledger,
    load_state,
    save_state,
)


class TestStateRoundTrip:
    def test_load_after_init(self, tmp_campaign: Path) -> None:
        state = load_state(tmp_campaign)
        assert state.campaign_id == "test_cmp_001"
        assert state.iteration == 0
        assert state.parent.ddg_cumulative == 0.0
        assert len(state.parent.sequence_heavy) > 0
        assert len(state.parent.sequence_light) > 0

    def test_save_and_reload(self, tmp_campaign: Path) -> None:
        state = load_state(tmp_campaign)
        state.iteration = 5
        state.parent.ddg_cumulative = -2.5
        save_state(tmp_campaign, state)

        reloaded = load_state(tmp_campaign)
        assert reloaded.iteration == 5
        assert reloaded.parent.ddg_cumulative == -2.5
        assert reloaded.parent.sequence_heavy == state.parent.sequence_heavy


class TestLedger:
    def test_empty_ledger(self, tmp_campaign: Path) -> None:
        decisions = load_ledger(tmp_campaign)
        assert decisions == []

    def test_append_and_load(self, tmp_campaign: Path) -> None:
        d = IterationDecision(
            iteration=1,
            mutation="H:52:S:Y",
            proxy_scores={"evoef_ddg": -1.2, "stabddg_ddg": -0.8},
            oracle_ddg=-0.8,
            accepted=True,
            rationale="Test mutation",
            timestamp=datetime.now(timezone.utc),
        )
        append_ledger(tmp_campaign, d)

        decisions = load_ledger(tmp_campaign)
        assert len(decisions) == 1
        assert decisions[0].mutation == "H:52:S:Y"
        assert decisions[0].accepted is True

    def test_multiple_appends(self, tmp_campaign: Path) -> None:
        for i in range(3):
            d = IterationDecision(
                iteration=i + 1,
                mutation=f"H:{50 + i}:A:G",
                oracle_ddg=-0.5 if i % 2 == 0 else 0.3,
                accepted=i % 2 == 0,
                rationale=f"Iteration {i + 1}",
                timestamp=datetime.now(timezone.utc),
            )
            append_ledger(tmp_campaign, d)

        decisions = load_ledger(tmp_campaign)
        assert len(decisions) == 3
        assert decisions[0].iteration == 1
        assert decisions[2].iteration == 3


class TestInitCampaign:
    def test_directory_structure(self, tmp_campaign: Path) -> None:
        assert (tmp_campaign / "state.yaml").exists()
        assert (tmp_campaign / "ledger.jsonl").exists()
        assert (tmp_campaign / "input").is_dir()
        assert (tmp_campaign / "iterations").is_dir()

    def test_input_pdb_copied(self, tmp_campaign: Path) -> None:
        input_files = list((tmp_campaign / "input").glob("*.pdb"))
        assert len(input_files) == 1

    def test_bad_chain_id(self, tmp_path: Path, pdb_1n8z: Path) -> None:
        import pytest

        with pytest.raises(ValueError, match="not found"):
            init_campaign(
                campaign_dir=tmp_path / "bad",
                pdb_path=pdb_1n8z,
                antibody_heavy_chain="Z",
                antibody_light_chain="L",
                antigen_chains=["A"],
            )
```

### 4.6 Integration Tests

**File:** `tests/test_tools.py`

These tests require external tools and are marked slow.

```python
"""Integration tests for tool wrappers.

These tests require:
- EvoEF binary (EVOEF_BINARY env var)
- StaB-ddG package
- Graphinity package
- BA-ddG installation
- Docker with rosettacommons/rosetta and proteinmpnn_ddg images
- GPU with ablms installed

Mark with @pytest.mark.slow so they're skipped during fast iteration.
Run with: pytest -m slow
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from autoantibody.models import ToolResult
from autoantibody.structure import extract_sequences, get_residue_map


def _pick_test_mutation(pdb_path: Path) -> str:
    """Pick a valid mutation from the first chain for testing."""
    seqs = extract_sequences(pdb_path)
    chain_id = list(seqs.keys())[0]
    rmap = get_residue_map(pdb_path, chain_id)
    for resnum, aa in rmap.items():
        mut = "A" if aa != "A" else "G"
        return f"{chain_id}:{resnum}:{aa}:{mut}"
    raise RuntimeError("No residues found")


@pytest.mark.slow
class TestEvoEF:
    def test_evoef_runs(self, pdb_1n8z: Path) -> None:
        mutation = _pick_test_mutation(pdb_1n8z)
        result = subprocess.run(
            [sys.executable, "tools/evoef_ddg.py", str(pdb_1n8z), mutation],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        tr = ToolResult(**data)
        assert tr.status == "ok"
        assert "ddg" in tr.scores
        assert tr.scorer_name == "evoef"

    def test_evoef_rejects_bad_mutation(self, pdb_1n8z: Path) -> None:
        result = subprocess.run(
            [sys.executable, "tools/evoef_ddg.py",
             str(pdb_1n8z), "Z:999:A:G"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0
        data = json.loads(result.stdout)
        assert data["status"] == "error"


@pytest.mark.slow
class TestStabDDG:
    def test_stabddg_runs(self, pdb_1n8z: Path) -> None:
        mutation = _pick_test_mutation(pdb_1n8z)
        result = subprocess.run(
            [sys.executable, "tools/stabddg_score.py",
             str(pdb_1n8z), mutation, "--chains", "HL_A"],
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        tr = ToolResult(**data)
        assert tr.status == "ok"
        assert "ddg" in tr.scores
        assert tr.scorer_name == "stabddg"


@pytest.mark.slow
class TestGraphinity:
    def test_graphinity_runs(self, pdb_1n8z: Path) -> None:
        mutation = _pick_test_mutation(pdb_1n8z)
        result = subprocess.run(
            [sys.executable, "tools/graphinity_score.py",
             str(pdb_1n8z), mutation,
             "--ab-chains", "HL", "--ag-chains", "A"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        tr = ToolResult(**data)
        assert tr.status == "ok"
        assert "ddg" in tr.scores
        assert tr.scorer_name == "graphinity"


@pytest.mark.slow
class TestBADDG:
    def test_baddg_runs(self, pdb_1n8z: Path) -> None:
        mutation = _pick_test_mutation(pdb_1n8z)
        result = subprocess.run(
            [sys.executable, "tools/baddg_score.py",
             str(pdb_1n8z), mutation, "--chains", "HL_A"],
            capture_output=True, text=True, timeout=600,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        tr = ToolResult(**data)
        assert tr.status == "ok"
        assert "ddg" in tr.scores
        assert tr.scorer_name == "baddg"


@pytest.mark.slow
class TestStabilityCheck:
    def test_stability_check_runs(self, pdb_1n8z: Path) -> None:
        mutation = _pick_test_mutation(pdb_1n8z)
        chain = mutation.split(":")[0]
        result = subprocess.run(
            [sys.executable, "tools/stability_check.py",
             str(pdb_1n8z), mutation, "--chain", chain],
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        tr = ToolResult(**data)
        assert tr.status == "ok"
        assert "stability_ddg" in tr.scores
        assert tr.scorer_name == "proteinmpnn_stability"


@pytest.mark.slow
class TestAtomFEP:
    def test_atom_fep_quick_run(self, pdb_1n8z: Path) -> None:
        """Minimal AToM-FEP run (4 windows, 1000 steps)."""
        mutation = _pick_test_mutation(pdb_1n8z)
        result = subprocess.run(
            [sys.executable, "tools/atom_ddg.py",
             str(pdb_1n8z), mutation,
             "--lambda-windows", "4",
             "--steps-per-window", "1000"],
            capture_output=True, text=True, timeout=7200,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        tr = ToolResult(**data)
        assert tr.status == "ok"
        assert "ddg" in tr.scores
        assert "ddg_uncertainty" in tr.scores
        assert tr.scorer_name == "atom_fep"


@pytest.mark.slow
class TestFlexDDG:
    def test_flex_ddg_quick_run(self, pdb_1n8z: Path) -> None:
        """Minimal flex-ddG run (2 structures, 5000 trials)."""
        mutation = _pick_test_mutation(pdb_1n8z)
        result = subprocess.run(
            [sys.executable, "tools/flex_ddg.py",
             str(pdb_1n8z), mutation,
             "--nstruct", "2", "--backrub-trials", "5000",
             "--relax-nstruct", "2"],
            capture_output=True, text=True, timeout=3600,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        tr = ToolResult(**data)
        assert tr.status == "ok"
        assert "ddg" in tr.scores
        assert "ddg_std" in tr.scores
        assert tr.scorer_name == "flex_ddg"


@pytest.mark.slow
class TestAblms:
    def test_ablms_score(self, pdb_1n8z: Path) -> None:
        seqs = extract_sequences(pdb_1n8z)
        chains = list(seqs.keys())
        heavy_seq = seqs[chains[0]]
        light_seq = seqs[chains[1]] if len(chains) > 1 else None

        cmd = [
            sys.executable, "tools/ablms_score.py", "score",
            "--heavy", heavy_seq,
            "--model", "balm",
        ]
        if light_seq:
            cmd.extend(["--light", light_seq])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        tr = ToolResult(**data)
        assert tr.status == "ok"
        assert "pll" in tr.scores
        assert tr.scorer_name == "ablms"


@pytest.mark.slow
class TestEndToEnd:
    def test_full_workflow(self, tmp_path: Path, pdb_1n8z: Path) -> None:
        """Init campaign → EvoEF screen → verify state update."""
        from autoantibody.state import (
            append_ledger,
            init_campaign,
            load_ledger,
            load_state,
            save_state,
        )
        from autoantibody.structure import get_interface_residues

        # 1. Init campaign
        campaign_dir = tmp_path / "e2e_campaign"
        seqs = extract_sequences(pdb_1n8z)
        chains = list(seqs.keys())
        state = init_campaign(
            campaign_dir=campaign_dir,
            pdb_path=pdb_1n8z,
            antibody_heavy_chain=chains[0],
            antibody_light_chain=chains[1],
            antigen_chains=chains[2:],
        )
        assert state.iteration == 0

        # 2. Get interface residues
        interface = get_interface_residues(
            pdb_1n8z,
            antibody_chains=chains[:2],
            antigen_chains=chains[2:],
        )
        assert len(interface) > 0

        # 3. Run EvoEF on first interface residue
        res = interface[0]
        mut_aa = "A" if res["aa"] != "A" else "G"
        mutation_str = f"{res['chain']}:{res['resnum']}:{res['aa']}:{mut_aa}"

        result = subprocess.run(
            [sys.executable, "tools/evoef_ddg.py",
             str(pdb_1n8z), mutation_str],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0
        tool_result = ToolResult(**json.loads(result.stdout))
        assert tool_result.status == "ok"
        assert "ddg" in tool_result.scores
        assert tool_result.scorer_name == "evoef"

        # 4. Record decision and update state
        from datetime import datetime, timezone

        from autoantibody.models import IterationDecision

        decision = IterationDecision(
            iteration=1,
            mutation=mutation_str,
            proxy_scores={"evoef_ddg": tool_result.scores["ddg"]},
            oracle_ddg=tool_result.scores["ddg"],  # use EvoEF as mock oracle
            accepted=tool_result.scores["ddg"] < 0,
            rationale="E2E test mutation",
            timestamp=datetime.now(timezone.utc),
        )
        append_ledger(campaign_dir, decision)

        state.iteration = 1
        if decision.accepted:
            state.parent.ddg_cumulative += decision.oracle_ddg
        save_state(campaign_dir, state)

        # 5. Verify
        reloaded = load_state(campaign_dir)
        assert reloaded.iteration == 1
        ledger = load_ledger(campaign_dir)
        assert len(ledger) == 1
        assert ledger[0].mutation == mutation_str
```

### 4.7 SKEMPIv2 Multi-Scorer Benchmark

**File:** `scripts/benchmark_skempi.py`

Generalized to compare multiple scorers side by side.

```python
#!/usr/bin/env python
"""Benchmark multiple ddG scorers against SKEMPIv2 experimental values.

Usage:
    python scripts/benchmark_skempi.py \
        --scorers evoef,stabddg,graphinity \
        --output results/skempi_benchmark.csv \
        --max-mutations 50

Downloads SKEMPIv2 data and cleaned PDB structures on first run.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import typer

from autoantibody.models import Mutation
from autoantibody.scorers import SCORER_REGISTRY

app = typer.Typer(help="SKEMPIv2 multi-scorer benchmark")

SKEMPI_CSV_URL = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"
SKEMPI_PDB_URL = "https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz"
R_KCAL = 1.987204e-3  # kcal/(mol·K)


def download_skempi(data_dir: Path) -> Path:
    """Download SKEMPIv2 CSV if not present."""
    csv_path = data_dir / "skempi_v2.csv"
    if not csv_path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Downloading SKEMPIv2 to {csv_path}...")
        urllib.request.urlretrieve(SKEMPI_CSV_URL, csv_path)
    return csv_path


def download_skempi_pdbs(data_dir: Path) -> Path:
    """Download and extract SKEMPIv2 cleaned PDB structures."""
    pdb_dir = data_dir / "PDBs"
    tgz_path = data_dir / "SKEMPI2_PDBs.tgz"
    if not pdb_dir.exists():
        if not tgz_path.exists():
            typer.echo(f"Downloading SKEMPIv2 PDBs to {tgz_path}...")
            urllib.request.urlretrieve(SKEMPI_PDB_URL, tgz_path)
        typer.echo("Extracting PDBs...")
        import tarfile

        with tarfile.open(tgz_path) as tar:
            tar.extractall(data_dir)
    return pdb_dir


def parse_skempi_csv(csv_path: Path) -> list[dict]:
    """Parse SKEMPIv2 CSV and filter for AB/AG single-point mutations."""
    entries: list[dict] = []
    with open(csv_path) as f:
        header = f.readline().strip().split(";")
        col = {name: i for i, name in enumerate(header)}
        for line in f:
            fields = line.strip().split(";")
            if len(fields) < len(header):
                continue
            # Filter: antibody-antigen only
            if fields[col["Hold_out_type"]] != "AB/AG":
                continue
            # Filter: single-point mutations only (no commas)
            mut_str = fields[col["Mutation(s)_cleaned"]]
            if "," in mut_str:
                continue
            # Parse affinity values
            try:
                kd_mut = float(fields[col["Affinity_mut_parsed"]])
                kd_wt = float(fields[col["Affinity_wt_parsed"]])
            except (ValueError, IndexError):
                continue
            if kd_mut <= 0 or kd_wt <= 0:
                continue
            # Compute experimental ΔΔG
            temp_str = fields[col["Temperature"]]
            temp = float(temp_str) if temp_str else 298.15
            ddg_exp = R_KCAL * temp * math.log(kd_mut / kd_wt)
            # Parse mutation
            pdb_field = fields[col["Pdb"]]
            pdb_code = pdb_field.split("_")[0]
            try:
                mutation = Mutation.from_skempi(mut_str)
            except (ValueError, IndexError):
                continue
            entries.append({
                "pdb_code": pdb_code,
                "pdb_field": pdb_field,
                "mutation_str": mut_str,
                "mutation": str(mutation),
                "ddg_experimental": ddg_exp,
                "kd_mut": kd_mut,
                "kd_wt": kd_wt,
                "temperature": temp,
            })
    return entries


def run_scorer_prediction(
    scorer_name: str, pdb_path: Path, mutation_str: str,
) -> float | None:
    """Run a scorer on a single mutation and return predicted ΔΔG."""
    info = SCORER_REGISTRY.get(scorer_name)
    if info is None:
        return None

    cmd = [sys.executable, str(info.script_path), str(pdb_path), mutation_str]

    # Add scorer-specific args
    if scorer_name in ("stabddg", "baddg"):
        cmd.extend(["--chains", "HL_A"])  # common AB/AG split
    elif scorer_name == "graphinity":
        cmd.extend(["--ab-chains", "HL", "--ag-chains", "A"])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0:
        return None
    data = json.loads(result.stdout)
    if data.get("status") != "ok":
        return None
    return data.get("scores", {}).get("ddg")


@app.command()
def main(
    scorers: str = typer.Option(
        "evoef", help="Comma-separated scorer names (e.g., evoef,stabddg,graphinity)"
    ),
    output: Path = typer.Option("results/skempi_benchmark.csv"),
    data_dir: Path = typer.Option("data/skempi"),
    max_mutations: int = typer.Option(0, help="Max mutations to test (0=all)"),
) -> None:
    """Run multi-scorer benchmark against SKEMPIv2 AB/AG mutations."""
    scorer_names = [s.strip() for s in scorers.split(",")]
    for name in scorer_names:
        if name not in SCORER_REGISTRY:
            typer.echo(f"Unknown scorer: {name}", err=True)
            raise typer.Exit(1)

    csv_path = download_skempi(data_dir)
    pdb_dir = download_skempi_pdbs(data_dir)
    entries = parse_skempi_csv(csv_path)
    typer.echo(f"Found {len(entries)} AB/AG single-point mutations")

    if max_mutations > 0:
        entries = entries[:max_mutations]

    output.parent.mkdir(parents=True, exist_ok=True)

    # Per-scorer results
    scorer_results: dict[str, list[dict]] = {name: [] for name in scorer_names}

    header = "pdb_code,mutation,ddg_experimental," + ",".join(
        f"ddg_{name}" for name in scorer_names
    )

    with open(output, "w") as f:
        f.write(header + "\n")
        for i, entry in enumerate(entries):
            pdb_path = pdb_dir / f"{entry['pdb_code']}.pdb"
            if not pdb_path.exists():
                continue
            typer.echo(
                f"[{i + 1}/{len(entries)}] {entry['pdb_code']} "
                f"{entry['mutation']}...",
                nl=False,
            )

            predictions: dict[str, float | None] = {}
            for name in scorer_names:
                predictions[name] = run_scorer_prediction(
                    name, pdb_path, entry["mutation"],
                )

            # Write CSV row
            pred_strs = [
                f"{predictions[name]:.4f}" if predictions[name] is not None else ""
                for name in scorer_names
            ]
            f.write(
                f"{entry['pdb_code']},{entry['mutation']},"
                f"{entry['ddg_experimental']:.4f},"
                + ",".join(pred_strs) + "\n"
            )

            # Track per-scorer results
            for name in scorer_names:
                if predictions[name] is not None:
                    scorer_results[name].append({
                        "ddg_exp": entry["ddg_experimental"],
                        "ddg_pred": predictions[name],
                    })

            status = "  ".join(
                f"{name}={predictions[name]:.2f}" if predictions[name] is not None
                else f"{name}=FAIL"
                for name in scorer_names
            )
            typer.echo(f" exp={entry['ddg_experimental']:.2f}  {status}")

    # Per-scorer summary statistics
    typer.echo(f"\n=== Results Summary ===")
    from scipy import stats as scipy_stats  # type: ignore[import-untyped]

    for name in scorer_names:
        results = scorer_results[name]
        if not results:
            typer.echo(f"  {name}: no successful predictions")
            continue
        exp = np.array([r["ddg_exp"] for r in results])
        pred = np.array([r["ddg_pred"] for r in results])
        pearson_r = np.corrcoef(exp, pred)[0, 1]
        spearman_r = scipy_stats.spearmanr(exp, pred).statistic
        rmse = np.sqrt(np.mean((exp - pred) ** 2))
        typer.echo(
            f"  {name}: n={len(results)}  "
            f"Pearson={pearson_r:.3f}  "
            f"Spearman={spearman_r:.3f}  "
            f"RMSE={rmse:.3f}"
        )
    typer.echo(f"Output: {output}")


if __name__ == "__main__":
    app()
```

### Phase 4 Definition of Done

- `pytest tests/test_models.py tests/test_scorers.py tests/test_state.py` passes (no external deps)
- `pytest tests/test_structure.py` passes (requires 1N8Z download)
- `pytest -m slow` runs integration tests (requires installed tools)
- `python scripts/benchmark_skempi.py --scorers evoef,stabddg --max-mutations 10` produces per-scorer comparison

---

## Implementation Order Summary

```
Phase 1 ──→ Phase 2A ──→ Phase 2B ──→ Phase 3 ──→ Phase 4
  │            │            │            │           │
  │ models     │ evoef      │ flex-ddg   │ init      │ unit tests
  │ scorers    │ stabddg    │ baddg      │ config    │ integration
  │ structure  │ graphinity │ stability  │ program   │ SKEMPI bench
  │ state      │ ablms      │            │           │
  │            │            │            │           │
  └────────────┴────────────┴──→ Phase 2C (deferrable)
                                    │
                                    │ atom_fep
```

Phases 2A tools can be parallelized. Phases 2B tools can be parallelized.
Phase 2C (AToM-OpenMM) is independent and can be deferred until the loop is proven.

Each phase is independently testable. Phase 1 has no external tool
dependencies. Phase 2 tests require the respective tools. Phase 3 requires
Phase 1+2. Phase 4 exercises everything.

After all four phases, the system is ready for the first real campaign:

```bash
# Initialize
python scripts/init_campaign.py \
    --pdb tests/data/1N8Z.pdb \
    --heavy-chain H --light-chain L \
    --antigen-chains A \
    --output runs/trastuzumab_001

# Start optimizing (tell Claude Code to begin)
# "Read programs/program.md and begin the campaign at runs/trastuzumab_001/"
```

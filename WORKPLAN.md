# Workplan: Autoantibody Affinity Optimization MVP

Reference architecture: [`docs/streamlined_architecture.md`](docs/streamlined_architecture.md)

## Overview

This workplan implements the streamlined MVP for autonomous antibody affinity
optimization. Claude Code serves as the agent runtime, plain Python scripts are
the tool wrappers, and flat files (YAML + JSONL) track state. The deliverable is
a working optimization loop: Claude Code reads an agent program, calls FoldX for
fast screening and Rosetta flex-ddG for oracle evaluation, and iteratively
optimizes antibody binding affinity through single-point mutations.

## File Manifest

| File | Purpose | Phase |
|------|---------|-------|
| `pyproject.toml` | Updated dependencies | 1 |
| `src/autoantibody/models.py` | Pydantic data models | 1 |
| `src/autoantibody/structure.py` | PDB parsing and interface detection | 1 |
| `src/autoantibody/state.py` | Campaign state management | 1 |
| `tools/foldx_ddg.py` | FoldX ΔΔG proxy scorer | 2 |
| `tools/flex_ddg.py` | Rosetta flex-ddG oracle (Docker) | 2 |
| `tools/ablms_score.py` | Antibody LM sequence scoring | 2 |
| `scripts/init_campaign.py` | Campaign initialization CLI | 3 |
| `configs/example_campaign.yaml` | Example campaign config | 3 |
| `programs/program.md` | Agent behavior program | 3 |
| `tests/conftest.py` | Shared test fixtures | 4 |
| `tests/test_models.py` | Data model unit tests | 4 |
| `tests/test_structure.py` | Structure utility tests | 4 |
| `tests/test_state.py` | State management tests | 4 |
| `tests/test_tools.py` | Tool wrapper integration tests | 4 |
| `scripts/benchmark_skempi.py` | SKEMPIv2 validation script | 4 |

## Prerequisites

**Software:**

- Python 3.12+
- FoldX 5.x binary with academic license (`FOLDX_BINARY` env var)
- Docker with `rosettacommons/rosetta` image
- GPU with CUDA support (for ablms inference)

**Installation:**

```bash
pip install -e ".[dev]"
pip install git+https://github.com/briney/ablms.git

# Verify tools
export FOLDX_BINARY=/path/to/foldx
$FOLDX_BINARY --help
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

**Objective:** Data models, structure parsing, state management. All subsequent
phases depend on this.

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

    def to_foldx(self) -> str:
        """Convert to FoldX individual_list.txt format: 'SH52Y;'."""
        return f"{self.wt_aa}{self.chain}{self.resnum}{self.mut_aa};"

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

### 1.3 Structure Utilities

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

### 1.4 State Management

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
- `python -c "from autoantibody.models import Mutation; print(Mutation.parse('H:52:S:Y'))"` works
- `python -c "from autoantibody.structure import extract_sequences"` works
- State round-trip: init_campaign → load_state → save_state produces identical YAML

---

## Phase 2: Tool Wrappers

**Objective:** Three standalone tool scripts callable by Claude Code via bash.
Each prints JSON to stdout conforming to the `ToolResult` schema.

### 2.1 FoldX ΔΔG Proxy Scorer

**File:** `tools/foldx_ddg.py`

FoldX BuildModel provides fast (~seconds) ΔΔG estimates. The wrapper handles
FoldX's idiosyncratic file formats and parses the difference output.

```python
#!/usr/bin/env python
"""FoldX ΔΔG proxy scorer.

Usage:
    python tools/foldx_ddg.py <pdb_path> <mutation>

Arguments:
    pdb_path: Path to the input PDB structure.
    mutation:  Mutation string (e.g., H:52:S:Y).

Output (stdout):
    JSON with status, scores.ddg, artifacts, wall_time_s.

Environment:
    FOLDX_BINARY: Path to the FoldX executable (required).
    FOLDX_ROTABASE: Path to rotabase.txt (optional; defaults to
        same directory as the FoldX binary).

Example:
    FOLDX_BINARY=/opt/foldx/foldx python tools/foldx_ddg.py \\
        runs/cmp_001/input/1N8Z.pdb H:52:S:Y
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from autoantibody.models import Mutation, ToolResult
from autoantibody.structure import validate_mutation_against_structure


def find_foldx_binary() -> Path:
    """Locate the FoldX binary from FOLDX_BINARY env var or PATH."""
    env_path = os.environ.get("FOLDX_BINARY")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        print(f"FOLDX_BINARY={env_path} not found", file=sys.stderr)
        sys.exit(1)
    which = shutil.which("foldx")
    if which:
        return Path(which)
    print(
        "FoldX not found. Set FOLDX_BINARY or add to PATH.", file=sys.stderr
    )
    sys.exit(1)


def find_rotabase(foldx_binary: Path) -> Path | None:
    """Locate rotabase.txt near the FoldX binary."""
    env = os.environ.get("FOLDX_ROTABASE")
    if env:
        return Path(env)
    candidate = foldx_binary.parent / "rotabase.txt"
    return candidate if candidate.exists() else None


def run_foldx_repair(
    foldx: Path, pdb_name: str, work_dir: Path
) -> Path:
    """Run RepairPdb and return path to repaired PDB."""
    subprocess.run(
        [str(foldx), "--command=RepairPdb", f"--pdb={pdb_name}",
         f"--output-dir={work_dir}"],
        cwd=work_dir, capture_output=True, text=True, check=True,
    )
    stem = Path(pdb_name).stem
    repaired = work_dir / f"{stem}_Repair.pdb"
    if not repaired.exists():
        raise FileNotFoundError(f"RepairPdb did not produce {repaired.name}")
    return repaired


def run_foldx_buildmodel(
    foldx: Path, pdb_path: Path, mutation: Mutation,
    work_dir: Path, num_runs: int = 3,
) -> dict[str, float]:
    """Run BuildModel and return ΔΔG scores."""
    (work_dir / "individual_list.txt").write_text(mutation.to_foldx() + "\n")

    result = subprocess.run(
        [str(foldx), "--command=BuildModel", f"--pdb={pdb_path.name}",
         "--mutant-file=individual_list.txt",
         f"--numberOfRuns={num_runs}", f"--output-dir={work_dir}"],
        cwd=work_dir, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FoldX BuildModel failed:\n{result.stderr}")

    dif_files = sorted(work_dir.glob("Dif_*.fxout"))
    if not dif_files:
        raise FileNotFoundError("No Dif_*.fxout output found")
    return parse_foldx_dif(dif_files[0])


def parse_foldx_dif(dif_path: Path) -> dict[str, float]:
    """Parse FoldX Dif_*.fxout and return averaged ΔΔG plus components."""
    lines = dif_path.read_text().splitlines()
    header_line: str | None = None
    data_rows: list[list[float]] = []
    for line in lines:
        line = line.strip()
        if line.startswith("Pdb"):
            header_line = line
        elif header_line and line and not line.startswith("#"):
            fields = line.split("\t")
            row: list[float] = []
            for f in fields[1:]:
                try:
                    row.append(float(f))
                except ValueError:
                    row.append(0.0)
            if row:
                data_rows.append(row)

    if not header_line or not data_rows:
        raise ValueError(f"Cannot parse FoldX output: {dif_path}")

    avg = np.mean(data_rows, axis=0).tolist()
    headers = header_line.split("\t")[1:]
    scores: dict[str, float] = {"ddg": round(avg[0], 4)}
    for h, v in zip(headers[1:], avg[1:]):
        key = h.strip().lower().replace(" ", "_")
        if key:
            scores[f"component_{key}"] = round(v, 4)
    return scores


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <pdb_path> <mutation>", file=sys.stderr)
        sys.exit(1)

    pdb_path = Path(sys.argv[1]).resolve()
    mutation = Mutation.parse(sys.argv[2])

    errors = validate_mutation_against_structure(pdb_path, mutation)
    if errors:
        result = ToolResult(status="error", error_message="; ".join(errors))
        print(result.model_dump_json(indent=2))
        sys.exit(1)

    foldx = find_foldx_binary()
    t0 = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="foldx_") as tmpdir:
        wd = Path(tmpdir)
        shutil.copy2(pdb_path, wd / pdb_path.name)
        rotabase = find_rotabase(foldx)
        if rotabase:
            shutil.copy2(rotabase, wd / "rotabase.txt")

        try:
            repaired = run_foldx_repair(foldx, pdb_path.name, wd)
            scores = run_foldx_buildmodel(foldx, repaired, mutation, wd)
            artifacts: dict[str, str] = {}
            mut_pdbs = sorted(wd.glob(f"{repaired.stem}_1*.pdb"))
            if mut_pdbs:
                artifacts["mutant_structure"] = str(mut_pdbs[0])
            result = ToolResult(
                status="ok", scores=scores, artifacts=artifacts,
                wall_time_s=round(time.monotonic() - t0, 2),
            )
        except Exception as e:
            result = ToolResult(
                status="error", error_message=str(e),
                wall_time_s=round(time.monotonic() - t0, 2),
            )

    print(result.model_dump_json(indent=2))
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()
```

### 2.2 Rosetta flex-ddG Oracle

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
    python tools/flex_ddg.py input/1N8Z.pdb H:52:S:Y \\
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
        r = ToolResult(status="error", error_message="; ".join(errors))
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
        )
    except Exception as e:
        result = ToolResult(
            status="error", error_message=str(e),
            wall_time_s=round(time.monotonic() - t0, 2),
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

### 2.3 ablms Sequence Scoring

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
    python tools/ablms_score.py score \\
        --heavy EVQLVES... --light DIQMTQS... --model balm

    # Compare wild-type vs mutant (delta PLL)
    python tools/ablms_score.py compare \\
        --heavy EVQLVES... --light DIQMTQS... \\
        --mut-heavy EVQLVEY... --model balm

    # Per-position masked marginal scan
    python tools/ablms_score.py scan \\
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
        )
    except Exception as e:
        result = ToolResult(
            status="error", error_message=str(e),
            wall_time_s=round(time.monotonic() - t0, 2),
        )

    print(result.model_dump_json(indent=2))
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()
```

### Phase 2 Definition of Done

- `python tools/foldx_ddg.py tests/data/1N8Z.pdb H:52:S:Y` prints valid JSON with `ddg`
- `python tools/flex_ddg.py tests/data/1N8Z.pdb H:52:S:Y --nstruct 2 --backrub-trials 5000` prints valid JSON
- `python tools/ablms_score.py score --heavy EVQLVES... --light DIQMTQS... --model balm` prints valid JSON

---

## Phase 3: Agent Program & Campaign Infrastructure

**Objective:** Create the campaign initialization script, example config, and
the agent program that drives the optimization loop.

### 3.1 Campaign Initialization

**File:** `scripts/init_campaign.py`

```python
#!/usr/bin/env python
"""Initialize a new antibody optimization campaign.

Usage:
    python scripts/init_campaign.py \\
        --pdb /path/to/complex.pdb \\
        --heavy-chain H --light-chain L \\
        --antigen-chains A \\
        --output runs/my_campaign

    python scripts/init_campaign.py --config configs/example_campaign.yaml
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

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
    """Create a campaign directory with state.yaml and empty ledger."""
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
```

### 3.3 Agent Program

**File:** `programs/program.md`

This is the core of the system — the document that tells Claude Code how to run
the optimization loop. It must be comprehensive enough for autonomous operation.

```markdown
# Antibody Affinity Optimization Program

## Objective

You are optimizing the binding affinity of an antibody to its antigen target
through iterative single-point mutations. Each iteration, you will:

1. Analyze the current parent antibody and its interface with the antigen.
2. Propose candidate mutations using structural reasoning and fast tools.
3. Select the single most promising mutation.
4. Evaluate it with the oracle scorer (flex-ddG).
5. Accept the mutation if it improves binding (ΔΔG < 0); reject otherwise.
6. Update the campaign state and proceed to the next iteration.

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

The state file contains the current parent sequences, structure path, cumulative
ΔΔG, chain assignments, and frozen positions.

## Available Tools

### 1. FoldX ΔΔG — Fast Proxy Scorer (~5-30 seconds)

```bash
python tools/foldx_ddg.py <pdb_path> <mutation>
```

- **Use for**: Screening 5-20 candidate mutations to rank them.
- **Input**: PDB path + mutation string (e.g., `H:52:S:Y`).
- **Output**: JSON with `scores.ddg` (kcal/mol). Negative = improved binding.
- **Accuracy**: Rough proxy. Directionally useful but not authoritative.

Example:
```bash
python tools/foldx_ddg.py runs/cmp_001/input/1N8Z.pdb H:52:S:Y
```

### 2. Rosetta flex-ddG — Oracle Scorer (~30-60 minutes)

```bash
python tools/flex_ddg.py <pdb_path> <mutation> [--nstruct N]
```

- **Use for**: Final accept/reject decision. Run on exactly ONE mutation.
- **Input**: PDB path + mutation string.
- **Output**: JSON with `scores.ddg` (mean), `scores.ddg_std`, `scores.n_structures`.
- **Authority**: This is the ground truth for accept/reject decisions.
- **Cost**: Expensive. Never run on more than one mutation per iteration.

For faster testing during development:
```bash
python tools/flex_ddg.py <pdb> <mutation> --nstruct 5 --backrub-trials 10000
```

### 3. ablms — Sequence Plausibility Scoring (~5-15 seconds)

```bash
# Overall sequence quality
python tools/ablms_score.py score \
    --heavy <heavy_seq> --light <light_seq> --model balm

# Compare wild-type vs mutant
python tools/ablms_score.py compare \
    --heavy <wt_heavy> --light <wt_light> \
    --mut-heavy <mut_heavy> --model balm

# Per-position analysis (identify problematic positions)
python tools/ablms_score.py scan \
    --heavy <heavy_seq> --light <light_seq> --model balm
```

- **Use for**: Checking that mutations don't destroy sequence plausibility.
  A mutation with good FoldX ΔΔG but terrible PLL delta should be viewed
  with suspicion.
- **Models**: Prefer `balm` (antibody-specific, paired). Use `esm2-650m` as
  a general protein baseline.

### 4. Structure Analysis (Python one-liners)

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

Read `state.yaml` and `ledger.jsonl`. Note:
- Current parent sequences (heavy and light chains)
- Current parent structure path
- Cumulative ΔΔG so far
- Frozen positions (do not mutate these)
- Previously attempted mutations and their outcomes

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

### Step 4: Screen with FoldX

Run FoldX on your top 5-15 candidates:
```bash
python tools/foldx_ddg.py <pdb_path> <mutation>
```

Record the ΔΔG for each. Focus on mutations with FoldX ΔΔG < -0.5 kcal/mol.

### Step 5: Check Sequence Plausibility

For your top 2-3 candidates (by FoldX), run ablms comparison:
```bash
python tools/ablms_score.py compare \
    --heavy <wt_heavy> --light <wt_light> \
    --mut-heavy <mut_heavy> --model balm
```

Discard mutations where `delta_pll` is strongly negative (the mutation makes
the sequence significantly less natural).

### Step 6: Select One Mutation

Choose the single best mutation considering:
1. FoldX ΔΔG (lower = better binding)
2. ablms delta_pll (should not be strongly negative)
3. Structural reasoning (does the substitution make biophysical sense?)
4. Novelty (prefer unexplored positions/substitutions)

Write a brief rationale for your choice.

### Step 7: Validate Safety

Before running the oracle, verify:
- The target position is not frozen
- The wild-type amino acid matches the current parent sequence
- The mutant amino acid is one of the 20 standard amino acids
- Exactly one position is being changed

### Step 8: Run the Oracle

```bash
python tools/flex_ddg.py <pdb_path> <selected_mutation>
```

This takes 30-60 minutes. Wait for it to complete.

### Step 9: Accept or Reject

- **Accept** if oracle `scores.ddg < 0` (improved binding)
- **Reject** if oracle `scores.ddg >= 0` (neutral or worse)

### Step 10: Update State

Create the iteration directory and save the decision:
```bash
mkdir -p runs/<campaign>/iterations/<N>
```

Write `runs/<campaign>/iterations/<N>/decision.json`:
```json
{
  "iteration": N,
  "mutation": "H:52:S:Y",
  "proxy_scores": {"foldx_ddg": -1.2, "ablms_delta_pll": 0.05},
  "oracle_ddg": -0.8,
  "accepted": true,
  "rationale": "Position 52 is interface-proximal...",
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
    proxy_scores={'foldx_ddg': -1.2},
    oracle_ddg=-0.8,
    accepted=True,
    rationale='...',
    timestamp=datetime.now(timezone.utc),
)
append_ledger(Path('runs/<campaign>'), decision)
"
```

### Step 11: Report and Continue

Summarize the iteration result:
- Mutation attempted
- FoldX proxy ΔΔG
- Oracle ΔΔG
- Accept/reject decision
- Cumulative ΔΔG after this iteration
- Total accepted/rejected so far

Then proceed to the next iteration (go back to Step 1).

## Constraints

- **One mutation per iteration.** Never apply multiple mutations at once.
- **Antibody residues only.** Never mutate antigen residues.
- **Standard amino acids only.** Only the 20 canonical amino acids.
- **Respect frozen positions.** Never mutate positions listed in
  `frozen_positions`.
- **Oracle is authoritative.** Always use the oracle ΔΔG for accept/reject,
  never the FoldX proxy alone.
- **Don't repeat exact failures.** If a mutation was rejected, don't try
  the same position→AA substitution again.

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
  contact but destabilize the antibody fold. The oracle captures this
  because it evaluates both bound and unbound states.
```

### Phase 3 Definition of Done

- `python scripts/init_campaign.py --pdb tests/data/1N8Z.pdb --heavy-chain H --light-chain L --antigen-chains A --output runs/test_001` creates campaign structure
- `runs/test_001/state.yaml` contains correct sequences and chain assignments
- `programs/program.md` contains complete, self-contained agent instructions
- Claude Code can follow `programs/program.md` without external guidance

---

## Phase 4: Testing & Validation

**Objective:** Unit tests for all core components, integration tests for tools,
and a SKEMPIv2 benchmark framework.

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

### 4.2 Unit Tests

**File:** `tests/test_models.py`

```python
"""Tests for data models."""

from __future__ import annotations

import pytest

from autoantibody.models import (
    CampaignState,
    IterationDecision,
    Mutation,
    ParentState,
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

    def test_to_foldx(self) -> None:
        m = Mutation.parse("H:52:S:Y")
        assert m.to_foldx() == "SH52Y;"

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
        r = ToolResult(status="error", error_message="FoldX not found")
        assert r.error_message == "FoldX not found"

    def test_json_roundtrip(self) -> None:
        r = ToolResult(status="ok", scores={"ddg": -0.5}, wall_time_s=1.0)
        r2 = ToolResult.model_validate_json(r.model_dump_json())
        assert r2.scores == r.scores
```

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
            proxy_scores={"foldx_ddg": -1.2},
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

### 4.3 Integration Tests

**File:** `tests/test_tools.py`

These tests require external tools (FoldX, Docker, GPU) and are marked slow.

```python
"""Integration tests for tool wrappers.

These tests require:
- FoldX binary (FOLDX_BINARY env var)
- Docker with rosettacommons/rosetta image
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
class TestFoldX:
    def test_foldx_runs(self, pdb_1n8z: Path) -> None:
        mutation = _pick_test_mutation(pdb_1n8z)
        result = subprocess.run(
            [sys.executable, "tools/foldx_ddg.py", str(pdb_1n8z), mutation],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        tr = ToolResult(**data)
        assert tr.status == "ok"
        assert "ddg" in tr.scores

    def test_foldx_rejects_bad_mutation(self, pdb_1n8z: Path) -> None:
        result = subprocess.run(
            [sys.executable, "tools/foldx_ddg.py",
             str(pdb_1n8z), "Z:999:A:G"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0
        data = json.loads(result.stdout)
        assert data["status"] == "error"


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
```

### 4.4 SKEMPIv2 Validation Framework

**File:** `scripts/benchmark_skempi.py`

This script downloads the SKEMPIv2 database, filters for antibody-antigen
single-point mutations, runs FoldX predictions, and computes correlation
with experimental ΔΔG values.

```python
#!/usr/bin/env python
"""Benchmark FoldX predictions against SKEMPIv2 experimental ΔΔG values.

Usage:
    python scripts/benchmark_skempi.py \\
        --output results/skempi_benchmark.csv \\
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
import yaml

from autoantibody.models import Mutation

app = typer.Typer(help="SKEMPIv2 FoldX benchmark")

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


def run_foldx_prediction(pdb_path: Path, mutation_str: str) -> float | None:
    """Run FoldX on a single mutation and return predicted ΔΔG."""
    result = subprocess.run(
        [sys.executable, "tools/foldx_ddg.py", str(pdb_path), mutation_str],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        return None
    data = json.loads(result.stdout)
    if data.get("status") != "ok":
        return None
    return data.get("scores", {}).get("ddg")


@app.command()
def main(
    output: Path = typer.Option("results/skempi_benchmark.csv"),
    data_dir: Path = typer.Option("data/skempi"),
    max_mutations: int = typer.Option(0, help="Max mutations to test (0=all)"),
) -> None:
    """Run FoldX benchmark against SKEMPIv2 AB/AG mutations."""
    csv_path = download_skempi(data_dir)
    pdb_dir = download_skempi_pdbs(data_dir)
    entries = parse_skempi_csv(csv_path)
    typer.echo(f"Found {len(entries)} AB/AG single-point mutations")

    if max_mutations > 0:
        entries = entries[:max_mutations]

    output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    with open(output, "w") as f:
        f.write("pdb_code,mutation,ddg_experimental,ddg_foldx\n")
        for i, entry in enumerate(entries):
            pdb_path = pdb_dir / f"{entry['pdb_code']}.pdb"
            if not pdb_path.exists():
                continue
            typer.echo(
                f"[{i + 1}/{len(entries)}] {entry['pdb_code']} "
                f"{entry['mutation']}...",
                nl=False,
            )
            ddg_pred = run_foldx_prediction(pdb_path, entry["mutation"])
            if ddg_pred is not None:
                f.write(
                    f"{entry['pdb_code']},{entry['mutation']},"
                    f"{entry['ddg_experimental']:.4f},{ddg_pred:.4f}\n"
                )
                results.append({
                    "ddg_exp": entry["ddg_experimental"],
                    "ddg_pred": ddg_pred,
                })
                typer.echo(
                    f" exp={entry['ddg_experimental']:.2f} "
                    f"pred={ddg_pred:.2f}"
                )
            else:
                typer.echo(" FAILED")

    if results:
        exp = np.array([r["ddg_exp"] for r in results])
        pred = np.array([r["ddg_pred"] for r in results])
        pearson_r = np.corrcoef(exp, pred)[0, 1]
        rmse = np.sqrt(np.mean((exp - pred) ** 2))
        typer.echo(f"\n=== Results ({len(results)} mutations) ===")
        typer.echo(f"Pearson r: {pearson_r:.3f}")
        typer.echo(f"RMSE:      {rmse:.3f} kcal/mol")
        typer.echo(f"Output:    {output}")


if __name__ == "__main__":
    app()
```

### 4.5 End-to-End Smoke Test

This test verifies the full workflow from campaign init through one FoldX call.
It does not run the oracle (too slow for CI) but validates the data flow.

```python
# In tests/test_tools.py, add:

@pytest.mark.slow
class TestEndToEnd:
    def test_full_workflow(self, tmp_path: Path, pdb_1n8z: Path) -> None:
        """Init campaign → FoldX screen → verify state update."""
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

        # 3. Run FoldX on first interface residue
        res = interface[0]
        mut_aa = "A" if res["aa"] != "A" else "G"
        mutation_str = f"{res['chain']}:{res['resnum']}:{res['aa']}:{mut_aa}"

        result = subprocess.run(
            [sys.executable, "tools/foldx_ddg.py",
             str(pdb_1n8z), mutation_str],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0
        tool_result = ToolResult(**json.loads(result.stdout))
        assert tool_result.status == "ok"
        assert "ddg" in tool_result.scores

        # 4. Record decision and update state
        from datetime import datetime, timezone

        from autoantibody.models import IterationDecision

        decision = IterationDecision(
            iteration=1,
            mutation=mutation_str,
            proxy_scores={"foldx_ddg": tool_result.scores["ddg"]},
            oracle_ddg=tool_result.scores["ddg"],  # use FoldX as mock oracle
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

### Phase 4 Definition of Done

- `pytest tests/test_models.py tests/test_state.py` passes (no external deps)
- `pytest tests/test_structure.py` passes (requires 1N8Z download)
- `pytest -m slow` runs integration tests (requires FoldX, Docker, GPU)
- `python scripts/benchmark_skempi.py --max-mutations 10` produces correlation output

---

## Implementation Order Summary

```
Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4
  │            │            │           │
  │ models.py  │ foldx      │ init      │ unit tests
  │ structure  │ flex-ddg   │ config    │ integration
  │ state      │ ablms      │ program   │ SKEMPIv2
  │ deps       │            │           │ e2e
```

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

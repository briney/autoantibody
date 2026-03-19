"""Standalone utilities for container tool scripts.

Provides Mutation parsing, ToolResult JSON output, and optional PDB validation.
This module has zero required external dependencies beyond the Python stdlib.
BioPython is used for structure validation when available.

The JSON output from ToolResult.to_json() is compatible with the Pydantic
ToolResult model in autoantibody.models, allowing the host-side code to
parse container output with ToolResult.model_validate().
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass
class Mutation:
    """A single amino acid point mutation.

    Mutations are specified as chain:resnum:wt_aa:mut_aa strings.
    Examples: H:52:S:Y, L:91:N:D, H:100A:G:A
    """

    chain: str
    resnum: str
    wt_aa: str
    mut_aa: str

    def __post_init__(self) -> None:
        if self.wt_aa not in STANDARD_AMINO_ACIDS:
            raise ValueError(f"'{self.wt_aa}' is not a standard amino acid")
        if self.mut_aa not in STANDARD_AMINO_ACIDS:
            raise ValueError(f"'{self.mut_aa}' is not a standard amino acid")

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
        """Convert to EvoEF/SKEMPI mutation format: 'SH52Y'."""
        return f"{self.wt_aa}{self.chain}{self.resnum}{self.mut_aa}"

    def to_stabddg(self) -> str:
        """Convert to StaB-ddG mutation format: 'SH52Y'."""
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


@dataclass
class ToolResult:
    """Standardized result from a tool execution.

    JSON output is compatible with the Pydantic ToolResult in autoantibody.models.
    """

    status: str  # "ok" or "error"
    scores: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    wall_time_s: float = 0.0
    error_message: str | None = None
    scorer_name: str | None = None

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON compatible with autoantibody.models.ToolResult."""
        data: dict[str, object] = {
            "status": self.status,
            "scores": self.scores,
            "artifacts": self.artifacts,
            "wall_time_s": self.wall_time_s,
            "error_message": self.error_message,
            "scorer_name": self.scorer_name,
        }
        return json.dumps(data, indent=indent)


def validate_mutation_against_structure(
    pdb_path: Path | str,
    mutation: Mutation,
) -> list[str]:
    """Validate a mutation against the PDB structure.

    Requires BioPython. If BioPython is not installed, returns an empty list
    (skips validation).

    Returns:
        List of error messages. Empty means valid (or validation was skipped).
    """
    try:
        from Bio.PDB.PDBParser import PDBParser
    except ImportError:
        return []

    parser = PDBParser(QUIET=True)
    errors: list[str] = []
    structure = parser.get_structure("s", str(pdb_path))
    model = structure[0]

    chain_ids = [c.id for c in model.get_chains()]
    if mutation.chain not in chain_ids:
        errors.append(f"Chain '{mutation.chain}' not found. Available: {chain_ids}")
        return errors

    chain = model[mutation.chain]
    residue_map: dict[str, str] = {}
    for r in chain.get_residues():
        if r.id[0] == " ":
            seq_id = r.id[1]
            icode = r.id[2].strip()
            resnum = f"{seq_id}{icode}"
            residue_map[resnum] = THREE_TO_ONE.get(r.get_resname(), "X")

    if mutation.resnum not in residue_map:
        errors.append(f"Residue {mutation.resnum} not found in chain {mutation.chain}")
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


# ---------------------------------------------------------------------------
# Auto-containerization: re-launch tool scripts inside Docker when on host
# ---------------------------------------------------------------------------

_DOCKER_IMAGES: dict[str, str] = {
    "evoef": "autoantibody/evoef:latest",
    "stabddg": "autoantibody/stabddg:latest",
    "graphinity": "autoantibody/graphinity:latest",
    "proteinmpnn_stability": "autoantibody/proteinmpnn_stability:latest",
    "ablms": "autoantibody/ablms:latest",
}

_GPU_TOOLS: frozenset[str] = frozenset({"stabddg", "graphinity", "ablms"})


def maybe_relaunch_in_container(scorer_name: str) -> None:
    """If not inside a container, re-invoke this script inside Docker and exit.

    Call this as the first line of ``main()`` in each tool script.  When
    running on the host (``AUTOANTIBODY_CONTAINER`` not set), this function
    re-executes the same script inside the appropriate Docker container with
    input files bind-mounted, streams stdout/stderr, and calls ``sys.exit()``.

    If already inside a container, returns immediately.
    """
    if os.environ.get("AUTOANTIBODY_CONTAINER"):
        return

    image = _DOCKER_IMAGES.get(scorer_name)
    if not image:
        return

    if not shutil.which("docker"):
        print(
            f"Docker not found. Install Docker or run inside the {image} container.",
            file=sys.stderr,
        )
        sys.exit(1)

    needs_gpu = scorer_name in _GPU_TOOLS

    # Resolve file-path arguments and collect directories to bind-mount.
    mount_dirs: set[str] = set()
    resolved_args: list[str] = []
    for arg in sys.argv[1:]:
        abs_path = os.path.abspath(arg)
        if os.path.exists(abs_path):
            mount_dirs.add(os.path.dirname(abs_path))
            resolved_args.append(abs_path)
        else:
            resolved_args.append(arg)

    cmd: list[str] = ["docker", "run", "--rm"]
    for d in sorted(mount_dirs):
        cmd.extend(["-v", f"{d}:{d}:ro"])

    if needs_gpu:
        cmd.extend(["--gpus", "all"])

    script_name = os.path.basename(sys.argv[0])
    cmd.extend([image, "python", f"/app/tools/{script_name}"] + resolved_args)

    result = subprocess.run(cmd)
    sys.exit(result.returncode)

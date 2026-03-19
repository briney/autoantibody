#!/usr/bin/env python
"""EvoEF ddG proxy scorer.

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

from _common import Mutation, ToolResult, validate_mutation_against_structure


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
    print("EvoEF not found. Set EVOEF_BINARY or add to PATH.", file=sys.stderr)
    sys.exit(1)


def run_evoef_repair(evoef: Path, pdb_path: Path, work_dir: Path) -> Path:
    """Run RepairStructure and return path to repaired PDB."""
    subprocess.run(
        [str(evoef), "--command=RepairStructure", f"--pdb={pdb_path.name}"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    repaired = work_dir / f"{pdb_path.stem}_Repair.pdb"
    if not repaired.exists():
        raise FileNotFoundError(f"RepairStructure did not produce {repaired.name}")
    return repaired


def run_evoef_build_mutant(
    evoef: Path,
    pdb_path: Path,
    mutation: Mutation,
    work_dir: Path,
) -> Path:
    """Run BuildMutant and return path to mutant PDB."""
    mut_str = mutation.to_evoef()
    subprocess.run(
        [
            str(evoef),
            "--command=BuildMutant",
            f"--pdb={pdb_path.name}",
            f"--mutant_file={mut_str}",
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    mutant = work_dir / f"{pdb_path.stem}_Model_0001.pdb"
    if not mutant.exists():
        raise FileNotFoundError(f"BuildMutant did not produce {mutant.name}")
    return mutant


def run_evoef_compute_binding(
    evoef: Path,
    pdb_path: Path,
    work_dir: Path,
) -> float:
    """Run ComputeBinding and return the binding energy."""
    result = subprocess.run(
        [str(evoef), "--command=ComputeBinding", f"--pdb={pdb_path.name}"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
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

    t0 = time.monotonic()
    try:
        errors = validate_mutation_against_structure(pdb_path, mutation)
        if errors:
            result = ToolResult(
                status="error",
                error_message="; ".join(errors),
                scorer_name="evoef",
            )
            print(result.to_json())
            sys.exit(1)

        evoef = find_evoef_binary()

        with tempfile.TemporaryDirectory(prefix="evoef_") as tmpdir:
            wd = Path(tmpdir)
            shutil.copy2(pdb_path, wd / pdb_path.name)

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
                status="ok",
                scores=scores,
                artifacts=artifacts,
                wall_time_s=round(time.monotonic() - t0, 2),
                scorer_name="evoef",
            )
    except SystemExit:
        raise
    except Exception as e:
        result = ToolResult(
            status="error",
            error_message=str(e),
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="evoef",
        )

    print(result.to_json())
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""EvoEF ddG proxy scorer.

Usage:
    python tools/evoef_ddg.py <pdb_path> <mutation> [--split AB_CD]

Arguments:
    pdb_path: Path to the input PDB structure.
    mutation:  Mutation string (e.g., H:52:S:Y).

Options:
    --split SPEC  Chain split for ComputeBinding: partner1_partner2
                  (e.g., HL_AB for antibody H+L vs antigen A+B).
                  Required for structures with >2 chains.

Output (stdout):
    JSON with status, scores.ddg, scorer_name, artifacts, wall_time_s.

Environment:
    EVOEF_BINARY: Path to the compiled EvoEF binary (required).

Example:
    EVOEF_BINARY=/opt/evoef/EvoEF python tools/evoef_ddg.py \
        runs/cmp_001/input/1N8Z.pdb H:52:S:Y --split HL_AB
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from _common import (
    Mutation,
    ToolResult,
    maybe_relaunch_in_container,
    validate_mutation_against_structure,
)


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
    # EvoEF's --mutant_file expects a *file path*, not a mutation string.
    # File format: one mutation per line, semicolon-terminated.
    mut_file = work_dir / "individual_list.txt"
    mut_file.write_text(f"{mut_str};\n")
    subprocess.run(
        [
            str(evoef),
            "--command=BuildMutant",
            f"--pdb={pdb_path.name}",
            f"--mutant_file={mut_file.name}",
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
    split: str | None = None,
) -> float:
    """Run ComputeBinding and return the binding energy.

    Args:
        evoef: Path to EvoEF binary.
        pdb_path: Path to PDB file.
        work_dir: Working directory.
        split: Chain split for multi-chain complexes (e.g., "HL,AB").
               EvoEF format: comma-separated groups.
    """
    cmd = [str(evoef), "--command=ComputeBinding", f"--pdb={pdb_path.name}"]
    if split:
        cmd.append(f"--split={split}")

    result = subprocess.run(
        cmd,
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    # With --split, EvoEF outputs a single binding energy block.
    # Without --split and >2 chains, it outputs ALL chain pairs;
    # we parse the last "Total" line (which is always the largest pair
    # for default output, but with --split it's the only one).
    total_energy = None
    for line in result.stdout.splitlines():
        if "Total" in line and "=" in line and "time" not in line.lower():
            parts = line.split("=")
            total_energy = float(parts[-1].strip())
            if split:
                return total_energy  # Only one block with --split
    if total_energy is not None:
        return total_energy
    raise ValueError("Could not parse binding energy from EvoEF output")


def main() -> None:
    maybe_relaunch_in_container("evoef")

    import argparse

    ap = argparse.ArgumentParser(description="EvoEF binding ddG scorer")
    ap.add_argument("pdb_path", type=Path)
    ap.add_argument("mutation", type=str)
    ap.add_argument(
        "--split",
        type=str,
        default=None,
        help="Chain split: partner1_partner2 (e.g., HL_AB)",
    )
    args = ap.parse_args()

    pdb_path = args.pdb_path.resolve()
    mutation = Mutation.parse(args.mutation)

    # Convert underscore split notation (HL_AB) to EvoEF comma format (HL,AB)
    evoef_split = None
    if args.split:
        parts = args.split.split("_")
        if len(parts) != 2:
            print("--split must be partner1_partner2 (e.g., HL_AB)", file=sys.stderr)
            sys.exit(1)
        evoef_split = f"{parts[0]},{parts[1]}"

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
            wt_binding = run_evoef_compute_binding(
                evoef, repaired, wd, split=evoef_split,
            )

            # Build mutant and compute its binding energy
            mutant_pdb = run_evoef_build_mutant(evoef, repaired, mutation, wd)
            mut_binding = run_evoef_compute_binding(
                evoef, mutant_pdb, wd, split=evoef_split,
            )

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

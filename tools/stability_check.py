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
    NOT a binding ddG -- used as a filter to reject fold-destabilizing mutations.
    Flag mutations with stability_ddg > 2.0 kcal/mol.

Requires:
    autoantibody/proteinmpnn_stability Docker image (extends vendor image
    ghcr.io/peptoneltd/proteinmpnn_ddg:1.0.0_base).

Example:
    python tools/stability_check.py runs/cmp_001/input/1N8Z.pdb H:52:S:Y --chain H
"""

from __future__ import annotations

import argparse
import csv
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


def _get_chain_ids(pdb_path: Path) -> list[str]:
    """Extract unique chain IDs from a PDB file, in order of appearance."""
    chains: list[str] = []
    seen: set[str] = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
                chain = line[21]
                if chain not in seen and chain.strip():
                    seen.add(chain)
                    chains.append(chain)
    return chains


def _extract_mutation_ddg(csv_path: Path, mutation: Mutation) -> float:
    """Extract the stability ddG for a specific mutation from the saturation CSV.

    The vendor predict.py outputs ALL point mutations for the chain.  We filter
    for the specific (position, wt_aa, mut_aa) triple.
    """
    target_pos = int("".join(c for c in mutation.resnum if c.isdigit()))

    candidates: list[tuple[int, float]] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pre = row["pre"]
            post = row["post"]
            if pre == mutation.wt_aa and post == mutation.mut_aa:
                ddg_col = (
                    "logit_difference_ddg"
                    if "logit_difference_ddg" in row and row["logit_difference_ddg"]
                    else "logit_difference"
                )
                candidates.append((int(row["pos"]), float(row[ddg_col])))

    if not candidates:
        raise ValueError(
            f"No rows match wt={mutation.wt_aa} mut={mutation.mut_aa} in ProteinMPNN-ddG output"
        )

    if len(candidates) == 1:
        return candidates[0][1]

    # Multiple positions with same AA pair — disambiguate by position.
    for pos, ddg in candidates:
        if pos == target_pos:
            return ddg

    raise ValueError(
        f"Position {target_pos} not found among matching rows "
        f"(positions: {[c[0] for c in candidates]})"
    )


def run_stability_check(
    pdb_path: Path,
    mutation: Mutation,
    chain_id: str,
) -> float:
    """Run ProteinMPNN-ddG stability prediction.

    Inside the container, calls the vendor predict.py directly with its actual
    CLI (--pdb_path, --chains, --chain_to_predict, --outpath).

    Returns:
        Predicted stability ddG (positive = destabilizing).
    """
    with tempfile.TemporaryDirectory(prefix="stability_") as tmpdir:
        wd = Path(tmpdir)
        input_pdb = wd / "input.pdb"
        shutil.copy2(pdb_path, input_pdb)
        output_csv = wd / "output.csv"

        # Provide all chains for structural context.
        all_chains = _get_chain_ids(pdb_path)
        chains_str = ",".join(all_chains) if all_chains else chain_id

        cmd = [
            "python",
            "/app/proteinmpnn_ddg/predict.py",
            "--pdb_path",
            str(input_pdb),
            "--chains",
            chains_str,
            "--chain_to_predict",
            chain_id,
            "--outpath",
            str(output_csv),
        ]

        env = {**os.environ, "JAX_PLATFORMS": "cpu"}
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )

        if proc.returncode != 0:
            raise RuntimeError(f"predict.py failed (exit {proc.returncode}): {proc.stderr}")

        if not output_csv.exists():
            raise FileNotFoundError("No output CSV produced by predict.py")

        return _extract_mutation_ddg(output_csv, mutation)


def main() -> None:
    maybe_relaunch_in_container("proteinmpnn_stability")

    ap = argparse.ArgumentParser(description="ProteinMPNN-ddG fold stability filter")
    ap.add_argument("pdb_path", type=Path)
    ap.add_argument("mutation", type=str)
    ap.add_argument("--chain", required=True, help="Chain ID to evaluate")
    args = ap.parse_args()

    pdb_path = args.pdb_path.resolve()
    mutation = Mutation.parse(args.mutation)

    t0 = time.monotonic()
    try:
        errors = validate_mutation_against_structure(pdb_path, mutation)
        if errors:
            result = ToolResult(
                status="error",
                error_message="; ".join(errors),
                scorer_name="proteinmpnn_stability",
            )
            print(result.to_json())
            sys.exit(1)

        stability_ddg = run_stability_check(pdb_path, mutation, args.chain)
        result = ToolResult(
            status="ok",
            scores={"stability_ddg": round(stability_ddg, 4)},
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="proteinmpnn_stability",
        )
    except SystemExit:
        raise
    except Exception as e:
        result = ToolResult(
            status="error",
            error_message=str(e),
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="proteinmpnn_stability",
        )

    print(result.to_json())
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()

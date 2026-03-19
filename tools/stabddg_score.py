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
    StaB-ddG package + checkpoint (stabddg.pt), available inside the
    autoantibody/stabddg Docker container.

Example:
    python tools/stabddg_score.py runs/cmp_001/input/1N8Z.pdb H:52:S:Y --chains HL_A
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


def _parse_stabddg_csv(csv_path: Path, mutation_str: str) -> float:
    """Extract the predicted ddG for a mutation from StaB-ddG output CSV.

    Args:
        csv_path: Path to the output CSV (columns: Name, Mutation, pred_1).
        mutation_str: SKEMPI-format mutation string (e.g., "SH52Y").

    Returns:
        Predicted ddG value.

    Raises:
        ValueError: If the mutation is not found in the CSV.
    """
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Mutation"] == mutation_str:
                return float(row["pred_1"])
    raise ValueError(f"Mutation {mutation_str} not found in StaB-ddG output")


def run_stabddg(
    pdb_path: Path,
    mutations: list[Mutation],
    ab_chains: str,
    ag_chains: str,
) -> list[float]:
    """Run StaB-ddG prediction on one or more mutations.

    Shells out to the upstream ``run_stabddg.py`` CLI, which handles PDB
    parsing, chain extraction, dataset creation, model loading, and inference.

    Args:
        pdb_path: Path to input PDB.
        mutations: List of mutations to score.
        ab_chains: Antibody chain IDs concatenated (e.g., "HL").
        ag_chains: Antigen chain IDs concatenated (e.g., "A").

    Returns:
        List of predicted ddG values, one per mutation.
    """
    stabddg_dir = Path(os.environ.get("STABDDG_DIR", "/opt/stabddg"))
    checkpoint = stabddg_dir / "model_ckpts" / "stabddg.pt"

    # Detect GPU availability
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    chains_spec = f"{ab_chains}_{ag_chains}"
    results: list[float] = []

    for mutation in mutations:
        with tempfile.TemporaryDirectory(prefix="stabddg_") as tmpdir:
            wd = Path(tmpdir)

            # Copy PDB to work dir — run_stabddg.py creates output next to PDB.
            local_pdb = wd / pdb_path.name
            shutil.copy2(pdb_path, local_pdb)

            mut_str = mutation.to_stabddg()
            cmd = [
                "python",
                str(stabddg_dir / "run_stabddg.py"),
                "--pdb_path",
                str(local_pdb),
                "--mutation",
                mut_str,
                "--chains",
                chains_spec,
                "--checkpoint",
                str(checkpoint),
                "--device",
                device,
                "--run_name",
                "output",
            ]

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if proc.returncode != 0:
                raise RuntimeError(f"StaB-ddG failed: {proc.stderr}")

            # Parse output CSV — located at {pdb_stem}_output/output.csv
            pdb_stem = local_pdb.stem
            output_csv = wd / f"{pdb_stem}_output" / "output.csv"

            if not output_csv.exists():
                raise FileNotFoundError(f"StaB-ddG output not found: {output_csv}")

            results.append(_parse_stabddg_csv(output_csv, mut_str))

    return results


def main() -> None:
    maybe_relaunch_in_container("stabddg")

    ap = argparse.ArgumentParser(description="StaB-ddG binding ddG scorer")
    ap.add_argument("pdb_path", type=Path)
    ap.add_argument("mutation", nargs="?", type=str, default=None)
    ap.add_argument("--chains", required=True, help="Chain spec: antibody_antigen (e.g., HL_A)")
    ap.add_argument("--batch", type=Path, default=None, help="File with one mutation per line")
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

    t0 = time.monotonic()
    try:
        # Validate all mutations
        for m in mutations:
            errors = validate_mutation_against_structure(pdb_path, m)
            if errors:
                result = ToolResult(
                    status="error",
                    error_message=f"Mutation {m}: {'; '.join(errors)}",
                    scorer_name="stabddg",
                )
                print(result.to_json())
                sys.exit(1)

        ddg_values = run_stabddg(pdb_path, mutations, ab_chains, ag_chains)

        if len(mutations) == 1:
            scores: dict[str, float] = {"ddg": round(ddg_values[0], 4)}
        else:
            scores = {f"ddg_{m}": round(v, 4) for m, v in zip(mutations, ddg_values, strict=True)}
            scores["ddg_best"] = round(min(ddg_values), 4)

        result = ToolResult(
            status="ok",
            scores=scores,
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="stabddg",
        )
    except SystemExit:
        raise
    except Exception as e:
        result = ToolResult(
            status="error",
            error_message=str(e),
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="stabddg",
        )

    print(result.to_json())
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()

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
    Docker with ghcr.io/peptoneltd/proteinmpnn_ddg:1.0.0_base image.

Example:
    python tools/stability_check.py runs/cmp_001/input/1N8Z.pdb H:52:S:Y --chain H
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from _common import Mutation, ToolResult, validate_mutation_against_structure

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

        if os.environ.get("AUTOANTIBODY_CONTAINER"):
            # Inside container: call predict.py directly
            proc = subprocess.run(
                [
                    "python",
                    "/app/predict.py",
                    "--input",
                    str(wd / "input.json"),
                    "--output",
                    str(wd / "output.json"),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
        else:
            # Legacy: call via docker run
            proc = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{wd}:/workdir",
                    DOCKER_IMAGE,
                    "python",
                    "/app/predict.py",
                    "--input",
                    "/workdir/input.json",
                    "--output",
                    "/workdir/output.json",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

        if proc.returncode != 0:
            raise RuntimeError(
                f"ProteinMPNN-ddG Docker failed (exit {proc.returncode}): {proc.stderr}"
            )

        output_path = wd / "output.json"
        if not output_path.exists():
            raise FileNotFoundError("No output.json produced")

        output = json.loads(output_path.read_text())
        return float(output["ddg"])


def main() -> None:
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

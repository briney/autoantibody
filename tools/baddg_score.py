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

from _common import Mutation, ToolResult, validate_mutation_against_structure


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
    print(
        "BA-ddG not found. Set BADDG_DIR or install to /opt/baddg.",
        file=sys.stderr,
    )
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
        sys.executable,
        str(script),
        "--input",
        input_csv,
        "--mode",
        mode,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
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
    ap.add_argument(
        "--chains",
        required=True,
        help="Chain spec: antibody_antigen (e.g., HL_A)",
    )
    ap.add_argument(
        "--mode",
        default="supervised",
        choices=["supervised", "unsupervised"],
        help="Prediction mode (default: supervised)",
    )
    args = ap.parse_args()

    parts = args.chains.split("_")
    if len(parts) != 2:
        print(
            "--chains must be antibody_antigen (e.g., HL_A)",
            file=sys.stderr,
        )
        sys.exit(1)
    ab_chains, ag_chains = parts

    pdb_path = args.pdb_path.resolve()
    mutation = Mutation.parse(args.mutation)

    t0 = time.monotonic()
    try:
        errors = validate_mutation_against_structure(pdb_path, mutation)
        if errors:
            result = ToolResult(
                status="error",
                error_message="; ".join(errors),
                scorer_name="baddg",
            )
            print(result.to_json())
            sys.exit(1)

        baddg_dir = find_baddg_dir()
        ddg = run_baddg(
            baddg_dir,
            pdb_path,
            mutation,
            ab_chains,
            ag_chains,
            mode=args.mode,
        )
        result = ToolResult(
            status="ok",
            scores={"ddg": round(ddg, 4)},
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="baddg",
        )
    except SystemExit:
        raise
    except Exception as e:
        result = ToolResult(
            status="error",
            error_message=str(e),
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="baddg",
        )

    print(result.to_json())
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()

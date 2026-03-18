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
    ap.add_argument("--ab-chains", required=True, help="Antibody chain IDs (e.g., HL)")
    ap.add_argument("--ag-chains", required=True, help="Antigen chain IDs (e.g., A)")
    args = ap.parse_args()

    pdb_path = args.pdb_path.resolve()
    mutation = Mutation.parse(args.mutation)

    errors = validate_mutation_against_structure(pdb_path, mutation)
    if errors:
        result = ToolResult(
            status="error",
            error_message="; ".join(errors),
            scorer_name="graphinity",
        )
        print(result.model_dump_json(indent=2))
        sys.exit(1)

    t0 = time.monotonic()
    try:
        ddg = run_graphinity(pdb_path, mutation, args.ab_chains, args.ag_chains)
        result = ToolResult(
            status="ok",
            scores={"ddg": round(ddg, 4)},
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="graphinity",
        )
    except Exception as e:
        result = ToolResult(
            status="error",
            error_message=str(e),
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="graphinity",
        )

    print(result.model_dump_json(indent=2))
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()

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
                print(result.model_dump_json(indent=2))
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

    print(result.model_dump_json(indent=2))
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()

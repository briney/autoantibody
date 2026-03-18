#!/usr/bin/env python
"""Antibody language model scoring via ablms.

Usage:
    # Score paired antibody sequence (pseudo-log-likelihood)
    python tools/ablms_score.py score \
        --heavy EVQLVES... --light DIQMTQS... --model balm

    # Compare wild-type vs mutant (delta PLL)
    python tools/ablms_score.py compare \
        --heavy EVQLVES... --light DIQMTQS... \
        --mut-heavy EVQLVEY... --model balm

    # Per-position masked marginal scan
    python tools/ablms_score.py scan \
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
    from ablms import AntibodySequence  # type: ignore[import-untyped]

    if light:
        return AntibodySequence(heavy=heavy, light=light)
    return AntibodySequence(heavy=heavy)


def cmd_score(args: argparse.Namespace) -> dict[str, float]:
    """Compute pseudo-log-likelihood for a sequence."""
    from ablms import load_model  # type: ignore[import-untyped]

    model = load_model(args.model, devices=args.device)
    seq = build_sequence(args.heavy, args.light, args.model)
    pll_scores = model.pseudo_log_likelihood([seq])
    return {"pll": float(pll_scores[0])}


def cmd_compare(args: argparse.Namespace) -> dict[str, float]:
    """Compare PLL between wild-type and mutant sequences."""
    from ablms import load_model  # type: ignore[import-untyped]

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
    from ablms import load_model  # type: ignore[import-untyped]

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
        scores["heavy_perplexity"] = float(output.get_chain_perplexity("heavy", agg="mean"))
        scores["light_perplexity"] = float(output.get_chain_perplexity("light", agg="mean"))
        scores["heavy_accuracy"] = float(output.get_chain_accuracy("heavy", agg="mean"))
        scores["light_accuracy"] = float(output.get_chain_accuracy("light", agg="mean"))

    # Per-position perplexity (as comma-separated string in artifacts,
    # too many values for the scores dict)
    per_pos = output.perplexity()
    if hasattr(per_pos, "tolist"):
        per_pos = per_pos.tolist()

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="ablms sequence scoring")
    parser.add_argument("--device", default="cuda", help="Compute device (default: cuda)")
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared args
    for name in ["score", "compare", "scan"]:
        sp = sub.add_parser(name)
        sp.add_argument("--heavy", required=True, help="Heavy chain sequence")
        sp.add_argument("--light", default=None, help="Light chain sequence")
        sp.add_argument("--model", default="balm", help="Model name")
        if name == "compare":
            sp.add_argument("--mut-heavy", default=None, help="Mutant heavy chain")
            sp.add_argument("--mut-light", default=None, help="Mutant light chain")

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
            status="ok",
            scores=scores,
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="ablms",
        )
    except Exception as e:
        result = ToolResult(
            status="error",
            error_message=str(e),
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="ablms",
        )

    print(result.model_dump_json(indent=2))
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Experimental Kd lookup oracle for CR9114 benchmark.

Replaces flex-ddG in the optimization loop with instant experimental data
from the Phillips et al. deep mutational scanning dataset.

Usage:
    python tools/lookup_oracle.py <campaign_dir> <mutation> [--antigen H1]

Arguments:
    campaign_dir: Path to the campaign directory (contains state.yaml).
    mutation:     Mutation string (e.g., H:31:G:D).

Options:
    --antigen:    Target antigen for Kd lookup (H1, H3, or fluB). Default: H1.
    --data-dir:   Path to CR9114 data directory. Default: data/cr9114.

Output (stdout):
    JSON ToolResult with scores.ddg, scores.neg_log_kd_parent, scores.neg_log_kd_mutant.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from autoantibody.models import Mutation, ToolResult
from autoantibody.state import load_state

if TYPE_CHECKING:
    import pandas as pd

# Gas constant and temperature for ddG conversion
R_KCAL = 0.001987  # kcal/(mol·K)
TEMPERATURE = 298.0  # K
RT = R_KCAL * TEMPERATURE  # ~0.592 kcal/mol
RT_LN10 = RT * math.log(10)  # ~1.364 kcal/mol

# Non-binding floor value — variants at this value don't measurably bind
KD_FLOOR = 6.0
# Penalty ddG for transitions to/from non-binding (large unfavorable value)
NON_BINDING_DDG = 10.0

ANTIGEN_COLUMN_MAP = {
    "h1": "h1_mean",
    "h3": "h3_mean",
    "flub": "fluB_mean",
}


def load_mutations_yaml(data_dir: Path) -> list[dict]:
    """Load the 16 mutation positions from mutations.yaml."""
    mutations_path = data_dir / "mutations.yaml"
    if not mutations_path.exists():
        raise FileNotFoundError(
            f"Mutations file not found: {mutations_path}. Run scripts/prepare_cr9114.py first."
        )
    with open(mutations_path) as f:
        data = yaml.safe_load(f)
    return data["positions"]


def load_variants(data_dir: Path) -> pd.DataFrame:
    """Load the variants lookup table from parquet."""
    import pandas as pd

    parquet_path = data_dir / "variants.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Variants file not found: {parquet_path}. Run scripts/prepare_cr9114.py first."
        )
    return pd.read_parquet(parquet_path)


def get_current_genotype(
    state_heavy_seq: str,
    positions: list[dict],
) -> str:
    """Determine the current binary genotype from the heavy chain sequence.

    For each of the 16 positions, check if the current sequence has the
    germline AA (0) or mature AA (1).

    Returns:
        16-character binary string (e.g., '0000000000000000' for all-germline).
    """
    bits: list[str] = []
    for pos in positions:
        seq_idx = pos["seq_position"]
        germline_aa = pos["germline_aa"]
        mature_aa = pos["mature_aa"]
        current_aa = state_heavy_seq[seq_idx]

        if current_aa == germline_aa:
            bits.append("0")
        elif current_aa == mature_aa:
            bits.append("1")
        else:
            raise ValueError(
                f"Position {pos['index']} (PDB {pos['chain']}:{pos['pdb_resnum']}): "
                f"current AA '{current_aa}' is neither germline '{germline_aa}' "
                f"nor mature '{mature_aa}'"
            )
    return "".join(bits)


def apply_mutation_to_genotype(
    current_genotype: str,
    mutation: Mutation,
    positions: list[dict],
) -> str:
    """Apply a mutation to the current genotype and return the new genotype.

    The mutation must flip one of the 16 positions from germline to mature
    (or mature to germline).
    """
    # Find which position this mutation corresponds to
    target_pos = None
    for pos in positions:
        if pos["chain"] == mutation.chain and pos["pdb_resnum"] == mutation.resnum:
            target_pos = pos
            break

    if target_pos is None:
        raise ValueError(
            f"Mutation {mutation} is not at one of the 16 benchmark positions. "
            f"Valid positions: " + ", ".join(f"{p['chain']}:{p['pdb_resnum']}" for p in positions)
        )

    # Validate wild-type AA
    idx = target_pos["index"] - 1  # 0-based index into genotype string
    current_bit = current_genotype[idx]

    if current_bit == "0":
        expected_wt = target_pos["germline_aa"]
        expected_mut = target_pos["mature_aa"]
    else:
        expected_wt = target_pos["mature_aa"]
        expected_mut = target_pos["germline_aa"]

    if mutation.wt_aa != expected_wt:
        raise ValueError(
            f"Wild-type mismatch at position {target_pos['index']} "
            f"(PDB {mutation.chain}:{mutation.resnum}): "
            f"expected '{expected_wt}', got '{mutation.wt_aa}'"
        )
    if mutation.mut_aa != expected_mut:
        raise ValueError(
            f"Mutant AA mismatch at position {target_pos['index']} "
            f"(PDB {mutation.chain}:{mutation.resnum}): "
            f"expected '{expected_mut}' (the only allowed substitution), "
            f"got '{mutation.mut_aa}'"
        )

    # Flip the bit
    new_bit = "1" if current_bit == "0" else "0"
    new_genotype = current_genotype[:idx] + new_bit + current_genotype[idx + 1 :]
    return new_genotype


def lookup_kd(
    variants_df: pd.DataFrame,
    genotype: str,
    antigen_col: str,
) -> float:
    """Look up the -logKd for a given genotype and antigen."""
    row = variants_df[variants_df["genotype"] == genotype]
    if len(row) == 0:
        raise ValueError(f"Genotype '{genotype}' not found in variants table")
    if len(row) > 1:
        raise ValueError(f"Multiple rows for genotype '{genotype}'")
    return float(row.iloc[0][antigen_col])


def compute_ddg(neg_log_kd_parent: float, neg_log_kd_mutant: float) -> float:
    """Compute ddG from -logKd values.

    ddG = RT * ln(Kd_mutant / Kd_parent)
        = RT * ln(10) * (neg_log_kd_parent - neg_log_kd_mutant)

    Negative ddG = improved binding (mutant binds tighter).
    """
    # Handle non-binding cases
    parent_binds = neg_log_kd_parent > KD_FLOOR
    mutant_binds = neg_log_kd_mutant > KD_FLOOR

    if not parent_binds and not mutant_binds:
        return 0.0  # Both non-binding, no change
    if parent_binds and not mutant_binds:
        return NON_BINDING_DDG  # Lost binding entirely
    if not parent_binds and mutant_binds:
        return -NON_BINDING_DDG  # Gained binding from non-binding

    return RT_LN10 * (neg_log_kd_parent - neg_log_kd_mutant)


def main() -> None:
    ap = argparse.ArgumentParser(description="CR9114 experimental Kd lookup oracle")
    ap.add_argument("campaign_dir", type=Path, help="Campaign directory")
    ap.add_argument("mutation", type=str, help="Mutation string (e.g., H:31:G:D)")
    ap.add_argument(
        "--antigen",
        default="H1",
        choices=["H1", "H3", "fluB", "h1", "h3", "flub"],
        help="Target antigen (default: H1)",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/cr9114"),
        help="CR9114 data directory (default: data/cr9114)",
    )
    args = ap.parse_args()

    campaign_dir = args.campaign_dir.resolve()
    data_dir = args.data_dir.resolve()
    antigen_key = args.antigen.lower()
    antigen_col = ANTIGEN_COLUMN_MAP[antigen_key]

    t0 = time.monotonic()
    try:
        # Load data
        mutation = Mutation.parse(args.mutation)
        positions = load_mutations_yaml(data_dir)
        variants_df = load_variants(data_dir)
        state = load_state(campaign_dir)

        # Determine current genotype
        current_genotype = get_current_genotype(state.parent.sequence_heavy, positions)

        # Apply mutation to get new genotype
        new_genotype = apply_mutation_to_genotype(current_genotype, mutation, positions)

        # Look up Kd values
        neg_log_kd_parent = lookup_kd(variants_df, current_genotype, antigen_col)
        neg_log_kd_mutant = lookup_kd(variants_df, new_genotype, antigen_col)

        # Compute ddG
        ddg = compute_ddg(neg_log_kd_parent, neg_log_kd_mutant)

        result = ToolResult(
            status="ok",
            scores={
                "ddg": round(ddg, 4),
                "neg_log_kd_parent": round(neg_log_kd_parent, 4),
                "neg_log_kd_mutant": round(neg_log_kd_mutant, 4),
            },
            wall_time_s=round(time.monotonic() - t0, 4),
            scorer_name="lookup_oracle",
        )

    except SystemExit:
        raise
    except Exception as e:
        result = ToolResult(
            status="error",
            error_message=str(e),
            wall_time_s=round(time.monotonic() - t0, 4),
            scorer_name="lookup_oracle",
        )

    print(result.model_dump_json(indent=2))
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()

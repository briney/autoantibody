#!/usr/bin/env python
"""Evaluate a CR9114 benchmark run against baselines.

Reads the campaign ledger and variants lookup table to compute metrics:
  1. Path efficiency — iterations to reach 90% of best achievable -logKd
  2. Mutation discovery — fraction of beneficial mutations correctly identified
  3. Tool accuracy — per-scorer correlation with oracle ddG
  4. Consensus value — did consensus outperform individual scorers?
  5. Rejection rate — fraction of proposed mutations rejected

Baselines (computed analytically, no tools needed):
  - Greedy oracle: always pick the best remaining single mutation
  - Random walk: flip positions in random order (averaged over 1000 trials)
  - Worst-case greedy: always pick the worst remaining mutation

Usage:
    python scripts/evaluate_cr9114.py runs/cr9114_h1_001 [--antigen H1]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import typer
import yaml

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

app = typer.Typer(help="Evaluate CR9114 benchmark results.")

# Constants
KD_FLOOR = 6.0
R_KCAL = 0.001987
TEMPERATURE = 298.0
RT_LN10 = R_KCAL * TEMPERATURE * np.log(10)

ANTIGEN_COLUMN_MAP = {
    "h1": "h1_mean",
    "h3": "h3_mean",
    "flub": "fluB_mean",
}


def load_ledger(campaign_dir: Path) -> list[dict]:
    """Load the campaign ledger as a list of dicts."""
    ledger_path = campaign_dir / "ledger.jsonl"
    if not ledger_path.exists():
        raise FileNotFoundError(f"No ledger found at {ledger_path}")
    entries: list[dict] = []
    with open(ledger_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def compute_greedy_oracle_trajectory(
    variants_df: pd.DataFrame,
    positions: list[dict],
    antigen_col: str,
) -> list[float]:
    """Compute the greedy oracle baseline trajectory.

    At each step, picks the single remaining position that gives the best
    (highest) -logKd when flipped to mature.
    """
    n_positions = len(positions)
    current_genotype = list("0" * n_positions)
    trajectory = [_lookup_kd(variants_df, "".join(current_genotype), antigen_col)]

    remaining = set(range(n_positions))
    for _ in range(n_positions):
        best_kd = -np.inf
        best_idx = -1
        for idx in remaining:
            trial = current_genotype.copy()
            trial[idx] = "1"
            kd = _lookup_kd(variants_df, "".join(trial), antigen_col)
            if kd > best_kd:
                best_kd = kd
                best_idx = idx
        if best_idx >= 0:
            current_genotype[best_idx] = "1"
            remaining.remove(best_idx)
            trajectory.append(best_kd)

    return trajectory


def compute_random_walk_trajectory(
    variants_df: pd.DataFrame,
    positions: list[dict],
    antigen_col: str,
    n_trials: int = 1000,
) -> tuple[list[float], list[float]]:
    """Compute random walk baseline (mean +/- std over n_trials).

    At each step, flips a random remaining position to mature.
    Returns (mean_trajectory, std_trajectory).
    """
    rng = np.random.default_rng(42)
    n_positions = len(positions)
    all_trajectories = np.zeros((n_trials, n_positions + 1))

    for trial in range(n_trials):
        order = rng.permutation(n_positions)
        current_genotype = list("0" * n_positions)
        all_trajectories[trial, 0] = _lookup_kd(variants_df, "".join(current_genotype), antigen_col)
        for step, idx in enumerate(order):
            current_genotype[idx] = "1"
            all_trajectories[trial, step + 1] = _lookup_kd(
                variants_df, "".join(current_genotype), antigen_col
            )

    mean_traj = np.mean(all_trajectories, axis=0).tolist()
    std_traj = np.std(all_trajectories, axis=0).tolist()
    return mean_traj, std_traj


def compute_worst_case_trajectory(
    variants_df: pd.DataFrame,
    positions: list[dict],
    antigen_col: str,
) -> list[float]:
    """Compute worst-case greedy: always pick the worst remaining mutation."""
    n_positions = len(positions)
    current_genotype = list("0" * n_positions)
    trajectory = [_lookup_kd(variants_df, "".join(current_genotype), antigen_col)]

    remaining = set(range(n_positions))
    for _ in range(n_positions):
        worst_kd = np.inf
        worst_idx = -1
        for idx in remaining:
            trial = current_genotype.copy()
            trial[idx] = "1"
            kd = _lookup_kd(variants_df, "".join(trial), antigen_col)
            if kd < worst_kd:
                worst_kd = kd
                worst_idx = idx
        if worst_idx >= 0:
            current_genotype[worst_idx] = "1"
            remaining.remove(worst_idx)
            trajectory.append(worst_kd)

    return trajectory


def _lookup_kd(variants_df: pd.DataFrame, genotype: str, antigen_col: str) -> float:
    """Look up -logKd for a genotype."""
    row = variants_df[variants_df["genotype"] == genotype]
    if len(row) == 0:
        return KD_FLOOR
    return float(row.iloc[0][antigen_col])


def compute_agent_trajectory(
    ledger: list[dict],
    variants_df: pd.DataFrame,
    positions: list[dict],
    antigen_col: str,
) -> list[float]:
    """Reconstruct the agent's -logKd trajectory from the ledger."""
    n_positions = len(positions)
    current_genotype = list("0" * n_positions)
    trajectory = [_lookup_kd(variants_df, "".join(current_genotype), antigen_col)]

    # Build mutation string → position index mapping
    mut_to_idx: dict[str, int] = {}
    for pos in positions:
        key = f"{pos['chain']}:{pos['pdb_resnum']}"
        mut_to_idx[key] = pos["index"] - 1  # 0-based

    for entry in ledger:
        if entry.get("accepted", False):
            mut_str = entry["mutation"]
            parts = mut_str.split(":")
            key = f"{parts[0]}:{parts[1]}"
            if key in mut_to_idx:
                idx = mut_to_idx[key]
                current_genotype[idx] = "1"
        trajectory.append(_lookup_kd(variants_df, "".join(current_genotype), antigen_col))

    return trajectory


def compute_tool_accuracy(ledger: list[dict]) -> dict[str, dict[str, float]]:
    """Compute per-scorer correlation with oracle ddG across all proposals."""
    scorer_predictions: dict[str, list[float]] = {}
    oracle_values: list[float] = []

    for entry in ledger:
        oracle_ddg = entry.get("oracle_ddg")
        proxy_scores = entry.get("proxy_scores", {})
        if oracle_ddg is None:
            continue
        oracle_values.append(oracle_ddg)
        for scorer_key, value in proxy_scores.items():
            if scorer_key not in scorer_predictions:
                scorer_predictions[scorer_key] = []
            scorer_predictions[scorer_key].append(value)

    if len(oracle_values) < 3:
        return {}

    from scipy import stats

    results: dict[str, dict[str, float]] = {}
    oracle_arr = np.array(oracle_values)
    for scorer_key, predictions in scorer_predictions.items():
        if len(predictions) != len(oracle_values):
            continue
        pred_arr = np.array(predictions)
        pearson_r, pearson_p = stats.pearsonr(pred_arr, oracle_arr)
        spearman_r, spearman_p = stats.spearmanr(pred_arr, oracle_arr)
        results[scorer_key] = {
            "pearson_r": round(float(pearson_r), 4),
            "pearson_p": round(float(pearson_p), 4),
            "spearman_r": round(float(spearman_r), 4),
            "spearman_p": round(float(spearman_p), 4),
            "n": len(predictions),
        }

    return results


def identify_beneficial_mutations(
    variants_df: pd.DataFrame,
    positions: list[dict],
    antigen_col: str,
) -> list[int]:
    """Identify which positions have beneficial germline→mature mutations.

    A mutation is beneficial if flipping it to mature from all-germline
    background improves -logKd.
    """
    germline_kd = _lookup_kd(variants_df, "0" * len(positions), antigen_col)
    beneficial: list[int] = []
    for pos in positions:
        idx = pos["index"] - 1
        genotype = list("0" * len(positions))
        genotype[idx] = "1"
        mutant_kd = _lookup_kd(variants_df, "".join(genotype), antigen_col)
        if mutant_kd > germline_kd:
            beneficial.append(pos["index"])
    return beneficial


def plot_trajectory(
    campaign_dir: Path,
    agent_traj: list[float],
    greedy_traj: list[float],
    random_mean: list[float],
    random_std: list[float],
    worst_traj: list[float],
) -> None:
    """Generate trajectory plot with baselines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Agent
    ax.plot(
        range(len(agent_traj)),
        agent_traj,
        "o-",
        color="tab:blue",
        linewidth=2,
        markersize=6,
        label="Agent",
        zorder=5,
    )

    # Greedy oracle
    ax.plot(
        range(len(greedy_traj)),
        greedy_traj,
        "s--",
        color="tab:green",
        linewidth=1.5,
        markersize=4,
        label="Greedy oracle (upper bound)",
    )

    # Random walk
    random_mean_arr = np.array(random_mean)
    random_std_arr = np.array(random_std)
    x = np.arange(len(random_mean))
    ax.plot(
        x,
        random_mean_arr,
        "d-.",
        color="tab:orange",
        linewidth=1.5,
        markersize=4,
        label="Random walk (mean)",
    )
    ax.fill_between(
        x,
        random_mean_arr - random_std_arr,
        random_mean_arr + random_std_arr,
        alpha=0.2,
        color="tab:orange",
    )

    # Worst case
    ax.plot(
        range(len(worst_traj)),
        worst_traj,
        "v:",
        color="tab:red",
        linewidth=1.5,
        markersize=4,
        label="Worst-case greedy (lower bound)",
    )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("-log(Kd)", fontsize=12)
    ax.set_title("CR9114 Benchmark: Affinity Trajectory", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plot_path = campaign_dir / "trajectory.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"  Saved: {plot_path}")


@app.command()
def main(
    campaign_dir: Annotated[Path, typer.Argument(help="Campaign directory")],
    antigen: Annotated[str, typer.Option("--antigen", help="Target antigen")] = "H1",
    data_dir: Annotated[Path, typer.Option("--data-dir", help="CR9114 data directory")] = Path(
        "data/cr9114"
    ),
    no_plot: Annotated[bool, typer.Option("--no-plot", help="Skip trajectory plot")] = False,
) -> None:
    """Evaluate a CR9114 benchmark run against baselines."""
    import pandas as pd

    antigen_key = antigen.lower()
    antigen_col = ANTIGEN_COLUMN_MAP[antigen_key]

    # Load data
    typer.echo("Loading data...")
    variants_df = pd.read_parquet(data_dir / "variants.parquet")
    with open(data_dir / "mutations.yaml") as f:
        mutations_data = yaml.safe_load(f)
    positions = mutations_data["positions"]
    ledger = load_ledger(campaign_dir)

    typer.echo(f"  Ledger entries: {len(ledger)}")
    typer.echo(f"  Antigen: {antigen}")

    # Compute all trajectories
    typer.echo("\nComputing trajectories...")
    agent_traj = compute_agent_trajectory(ledger, variants_df, positions, antigen_col)
    greedy_traj = compute_greedy_oracle_trajectory(variants_df, positions, antigen_col)
    random_mean, random_std = compute_random_walk_trajectory(variants_df, positions, antigen_col)
    worst_traj = compute_worst_case_trajectory(variants_df, positions, antigen_col)

    # Compute metrics
    typer.echo("\nComputing metrics...")

    # 1. Path efficiency
    best_achievable = greedy_traj[-1]
    threshold_90 = greedy_traj[0] + 0.9 * (best_achievable - greedy_traj[0])
    iters_to_90 = len(agent_traj) - 1  # default: never reached
    for i, kd in enumerate(agent_traj):
        if kd >= threshold_90:
            iters_to_90 = i
            break

    # 2. Mutation discovery
    n_accepted = sum(1 for e in ledger if e.get("accepted", False))
    n_rejected = sum(1 for e in ledger if not e.get("accepted", False))
    beneficial = identify_beneficial_mutations(variants_df, positions, antigen_col)
    accepted_positions: set[int] = set()
    for entry in ledger:
        if entry.get("accepted", False):
            parts = entry["mutation"].split(":")
            key = f"{parts[0]}:{parts[1]}"
            for pos in positions:
                if f"{pos['chain']}:{pos['pdb_resnum']}" == key:
                    accepted_positions.add(pos["index"])
    beneficial_discovered = len(accepted_positions & set(beneficial))
    discovery_rate = beneficial_discovered / len(beneficial) if beneficial else 0.0

    # 3. Tool accuracy
    tool_accuracy = compute_tool_accuracy(ledger)

    # 4. Final affinity
    final_kd = agent_traj[-1]
    start_kd = agent_traj[0]

    # 5. Rejection rate
    total_attempts = len(ledger)
    rejection_rate = n_rejected / total_attempts if total_attempts > 0 else 0.0

    # Print results
    typer.echo(f"\n{'=' * 60}")
    typer.echo("CR9114 BENCHMARK RESULTS")
    typer.echo(f"{'=' * 60}")

    typer.echo("\n--- Summary ---")
    typer.echo(f"  Total iterations:      {total_attempts}")
    typer.echo(f"  Accepted mutations:    {n_accepted}")
    typer.echo(f"  Rejected mutations:    {n_rejected}")
    typer.echo(f"  Rejection rate:        {rejection_rate:.1%}")

    typer.echo("\n--- Affinity ---")
    typer.echo(f"  Starting -logKd:       {start_kd:.3f}")
    typer.echo(f"  Final -logKd:          {final_kd:.3f}")
    typer.echo(f"  Improvement:           {final_kd - start_kd:+.3f}")
    typer.echo(f"  Best achievable:       {best_achievable:.3f}")
    if best_achievable > start_kd:
        pct = (final_kd - start_kd) / (best_achievable - start_kd) * 100
        typer.echo(f"  % of optimal:          {pct:.1f}%")
    else:
        typer.echo("  % of optimal:          N/A")

    typer.echo("\n--- Path Efficiency ---")
    typer.echo(f"  Iterations to 90%:     {iters_to_90}")
    typer.echo(f"  Greedy oracle (90%):   {_find_threshold_step(greedy_traj, threshold_90)}")
    typer.echo(f"  Random walk (90%):     {_find_threshold_step(random_mean, threshold_90)}")

    typer.echo("\n--- Mutation Discovery ---")
    typer.echo(f"  Beneficial positions:  {len(beneficial)}/16")
    typer.echo(f"  Correctly discovered:  {beneficial_discovered}/{len(beneficial)}")
    typer.echo(f"  Discovery rate:        {discovery_rate:.1%}")

    if tool_accuracy:
        typer.echo("\n--- Tool Accuracy ---")
        for scorer, metrics in sorted(tool_accuracy.items()):
            typer.echo(
                f"  {scorer:20s}  Pearson={metrics['pearson_r']:+.3f}  "
                f"Spearman={metrics['spearman_r']:+.3f}  (n={metrics['n']})"
            )

    typer.echo(f"\n{'=' * 60}")

    # Save evaluation YAML
    eval_data = {
        "campaign_dir": str(campaign_dir),
        "antigen": antigen,
        "total_iterations": total_attempts,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "rejection_rate": round(rejection_rate, 4),
        "start_neg_log_kd": round(start_kd, 4),
        "final_neg_log_kd": round(final_kd, 4),
        "improvement": round(final_kd - start_kd, 4),
        "best_achievable": round(best_achievable, 4),
        "pct_optimal": round((final_kd - start_kd) / (best_achievable - start_kd) * 100, 2)
        if best_achievable > start_kd
        else None,
        "iterations_to_90pct": iters_to_90,
        "beneficial_positions": beneficial,
        "discovery_rate": round(discovery_rate, 4),
        "tool_accuracy": tool_accuracy,
    }
    eval_path = campaign_dir / "evaluation.yaml"
    with open(eval_path, "w") as f:
        yaml.dump(eval_data, f, default_flow_style=False, sort_keys=False)
    typer.echo(f"\n  Saved: {eval_path}")

    # Generate plot
    if not no_plot:
        typer.echo("\nGenerating trajectory plot...")
        plot_trajectory(
            campaign_dir,
            agent_traj,
            greedy_traj,
            random_mean,
            random_std,
            worst_traj,
        )


def _find_threshold_step(trajectory: list[float], threshold: float) -> int:
    """Find the first step at which a trajectory reaches a threshold."""
    for i, val in enumerate(trajectory):
        if val >= threshold:
            return i
    return len(trajectory) - 1


if __name__ == "__main__":
    app()

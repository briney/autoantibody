#!/usr/bin/env python
"""Benchmark multiple ddG scorers against SKEMPIv2 experimental values.

Usage:
    python scripts/benchmark_skempi.py \
        --scorers evoef,stabddg,graphinity \
        --output results/skempi_benchmark.csv \
        --max-mutations 50

Downloads SKEMPIv2 data and cleaned PDB structures on first run.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import typer

from autoantibody.models import Mutation
from autoantibody.scorers import SCORER_REGISTRY

app = typer.Typer(help="SKEMPIv2 multi-scorer benchmark")

SKEMPI_CSV_URL = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"
SKEMPI_PDB_URL = "https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz"
R_KCAL = 1.987204e-3  # kcal/(mol*K)


def download_skempi(data_dir: Path) -> Path:
    """Download SKEMPIv2 CSV if not present."""
    csv_path = data_dir / "skempi_v2.csv"
    if not csv_path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Downloading SKEMPIv2 to {csv_path}...")
        urllib.request.urlretrieve(SKEMPI_CSV_URL, csv_path)
    return csv_path


def download_skempi_pdbs(data_dir: Path) -> Path:
    """Download and extract SKEMPIv2 cleaned PDB structures."""
    pdb_dir = data_dir / "PDBs"
    tgz_path = data_dir / "SKEMPI2_PDBs.tgz"
    if not pdb_dir.exists():
        if not tgz_path.exists():
            typer.echo(f"Downloading SKEMPIv2 PDBs to {tgz_path}...")
            urllib.request.urlretrieve(SKEMPI_PDB_URL, tgz_path)
        typer.echo("Extracting PDBs...")
        import tarfile

        with tarfile.open(tgz_path) as tar:
            tar.extractall(data_dir)
    return pdb_dir


def parse_skempi_csv(csv_path: Path) -> list[dict]:
    """Parse SKEMPIv2 CSV and filter for AB/AG single-point mutations."""
    entries: list[dict] = []
    with open(csv_path) as f:
        header = f.readline().strip().split(";")
        col = {name: i for i, name in enumerate(header)}
        for line in f:
            fields = line.strip().split(";")
            if len(fields) < len(header):
                continue
            # Filter: antibody-antigen only
            if fields[col["Hold_out_type"]] != "AB/AG":
                continue
            # Filter: single-point mutations only (no commas)
            mut_str = fields[col["Mutation(s)_cleaned"]]
            if "," in mut_str:
                continue
            # Parse affinity values
            try:
                kd_mut = float(fields[col["Affinity_mut_parsed"]])
                kd_wt = float(fields[col["Affinity_wt_parsed"]])
            except (ValueError, IndexError):
                continue
            if kd_mut <= 0 or kd_wt <= 0:
                continue
            # Compute experimental ddG
            temp_str = fields[col["Temperature"]]
            temp = float(temp_str) if temp_str else 298.15
            ddg_exp = R_KCAL * temp * math.log(kd_mut / kd_wt)
            # Parse mutation
            pdb_field = fields[col["Pdb"]]
            pdb_code = pdb_field.split("_")[0]
            try:
                mutation = Mutation.from_skempi(mut_str)
            except (ValueError, IndexError):
                continue
            entries.append(
                {
                    "pdb_code": pdb_code,
                    "pdb_field": pdb_field,
                    "mutation_str": mut_str,
                    "mutation": str(mutation),
                    "ddg_experimental": ddg_exp,
                    "kd_mut": kd_mut,
                    "kd_wt": kd_wt,
                    "temperature": temp,
                }
            )
    return entries


def run_scorer_prediction(
    scorer_name: str,
    pdb_path: Path,
    mutation_str: str,
) -> float | None:
    """Run a scorer on a single mutation and return predicted ddG."""
    info = SCORER_REGISTRY.get(scorer_name)
    if info is None:
        return None

    cmd = [sys.executable, str(info.script_path), str(pdb_path), mutation_str]

    # Add scorer-specific args
    if scorer_name in ("stabddg", "baddg"):
        cmd.extend(["--chains", "HL_A"])  # common AB/AG split
    elif scorer_name == "graphinity":
        cmd.extend(["--ab-chains", "HL", "--ag-chains", "A"])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0:
        return None
    data = json.loads(result.stdout)
    if data.get("status") != "ok":
        return None
    return data.get("scores", {}).get("ddg")


@app.command()
def main(
    scorers: str = typer.Option(
        "evoef", help="Comma-separated scorer names (e.g., evoef,stabddg,graphinity)"
    ),
    output: Path = typer.Option("results/skempi_benchmark.csv"),
    data_dir: Path = typer.Option("data/skempi"),
    max_mutations: int = typer.Option(0, help="Max mutations to test (0=all)"),
) -> None:
    """Run multi-scorer benchmark against SKEMPIv2 AB/AG mutations."""
    scorer_names = [s.strip() for s in scorers.split(",")]
    for name in scorer_names:
        if name not in SCORER_REGISTRY:
            typer.echo(f"Unknown scorer: {name}", err=True)
            raise typer.Exit(1)

    csv_path = download_skempi(data_dir)
    pdb_dir = download_skempi_pdbs(data_dir)
    entries = parse_skempi_csv(csv_path)
    typer.echo(f"Found {len(entries)} AB/AG single-point mutations")

    if max_mutations > 0:
        entries = entries[:max_mutations]

    output.parent.mkdir(parents=True, exist_ok=True)

    # Per-scorer results
    scorer_results: dict[str, list[dict]] = {name: [] for name in scorer_names}

    header = "pdb_code,mutation,ddg_experimental," + ",".join(
        f"ddg_{name}" for name in scorer_names
    )

    with open(output, "w") as f:
        f.write(header + "\n")
        for i, entry in enumerate(entries):
            pdb_path = pdb_dir / f"{entry['pdb_code']}.pdb"
            if not pdb_path.exists():
                continue
            typer.echo(
                f"[{i + 1}/{len(entries)}] {entry['pdb_code']} {entry['mutation']}...",
                nl=False,
            )

            predictions: dict[str, float | None] = {}
            for name in scorer_names:
                predictions[name] = run_scorer_prediction(
                    name,
                    pdb_path,
                    entry["mutation"],
                )

            # Write CSV row
            pred_strs = [
                f"{predictions[name]:.4f}" if predictions[name] is not None else ""
                for name in scorer_names
            ]
            f.write(
                f"{entry['pdb_code']},{entry['mutation']},"
                f"{entry['ddg_experimental']:.4f}," + ",".join(pred_strs) + "\n"
            )

            # Track per-scorer results
            for name in scorer_names:
                if predictions[name] is not None:
                    scorer_results[name].append(
                        {
                            "ddg_exp": entry["ddg_experimental"],
                            "ddg_pred": predictions[name],
                        }
                    )

            status = "  ".join(
                f"{name}={predictions[name]:.2f}"
                if predictions[name] is not None
                else f"{name}=FAIL"
                for name in scorer_names
            )
            typer.echo(f" exp={entry['ddg_experimental']:.2f}  {status}")

    # Per-scorer summary statistics
    typer.echo("\n=== Results Summary ===")
    from scipy import stats as scipy_stats  # type: ignore[import-untyped]

    for name in scorer_names:
        results = scorer_results[name]
        if not results:
            typer.echo(f"  {name}: no successful predictions")
            continue
        exp = np.array([r["ddg_exp"] for r in results])
        pred = np.array([r["ddg_pred"] for r in results])
        pearson_r = np.corrcoef(exp, pred)[0, 1]
        spearman_r = scipy_stats.spearmanr(exp, pred).statistic
        rmse = np.sqrt(np.mean((exp - pred) ** 2))
        typer.echo(
            f"  {name}: n={len(results)}  "
            f"Pearson={pearson_r:.3f}  "
            f"Spearman={spearman_r:.3f}  "
            f"RMSE={rmse:.3f}"
        )
    typer.echo(f"Output: {output}")


if __name__ == "__main__":
    app()

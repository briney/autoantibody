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
    autoantibody/graphinity Docker container (Graphinity + EvoEF for mutant
    PDB generation).

Example:
    python tools/graphinity_score.py runs/cmp_001/input/1N8Z.pdb H:52:S:Y \
        --ab-chains HL --ag-chains A
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

# Config template for Graphinity inference (YAML as string to avoid PyYAML
# dependency on the host side).  Architecture params match the shipped
# "full_848597" checkpoint trained with lmg atom typing.
_CONFIG_TEMPLATE = """\
save_dir: {save_dir}
name: inference
test: True

initialize_weights:
  checkpoint_file: {checkpoint}

model: ddgEGNN

model_params:
  num_node_features: 12
  lr: 1.e-3
  weight_decay: 1.e-16
  balanced_loss: False
  dropout: 0
  num_edge_features: 1
  egnn_layer_hidden_nfs: [128, 128, 128]
  embedding_in_nf: 128
  embedding_out_nf: 128
  num_classes: 1
  attention: False
  residual: True
  normalize: False
  tanh: True
  update_coords: True
  scheduler: CosineAnnealing
  norm_nodes: null

trainer_params:
  gpus: 0

loader_params:
  batch_size: 1
  num_workers: 0
  balanced_sampling: False

dataset_params:
  rotate: False
  cache_frames: False
  graph_generation_mode: int_mut
  interaction_dist: 4
  typing_mode: lmg
  rough_search: True
  input_files:
    test:
      - {input_csv}
"""


def _parse_graphinity_csv(csv_path: Path) -> float:
    """Extract the predicted ddG from a Graphinity output CSV.

    The CSV has columns: wt_pdb, mut_pdb, pred_score, true_label.
    Returns the pred_score from the first (and typically only) row.

    Raises:
        ValueError: If the CSV is empty.
    """
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            return float(row["pred_score"])
    raise ValueError("Empty Graphinity output CSV")


def run_graphinity(
    pdb_path: Path,
    mutation: Mutation,
    ab_chains: str,
    ag_chains: str,
) -> float:
    """Run Graphinity prediction on a single mutation.

    Pipeline:
      1. Generate mutant PDB using EvoEF (RepairStructure + BuildMutant).
      2. Create input CSV and config YAML for Graphinity.
      3. Run graphinity_inference.py.
      4. Parse the output CSV for the predicted ddG.

    Args:
        pdb_path: Path to input PDB.
        mutation: Mutation to score.
        ab_chains: Antibody chain IDs (e.g., "HL").
        ag_chains: Antigen chain IDs (e.g., "A").

    Returns:
        Predicted binding ddG value.
    """
    graphinity_dir = Path(os.environ.get("GRAPHINITY_DIR", "/opt/graphinity"))
    evoef_binary = Path(os.environ.get("EVOEF_BINARY", "/opt/evoef/EvoEF"))
    checkpoint = (
        graphinity_dir
        / "example"
        / "ddg_synthetic"
        / "FoldX"
        / "varying_dataset_size"
        / "model_weights"
        / "Graphinity-varying_dataset_size-full_848597.ckpt"
    )

    with tempfile.TemporaryDirectory(prefix="graphinity_") as tmpdir:
        wd = Path(tmpdir)

        # --- 1. Generate mutant PDB with EvoEF ---
        wt_pdb = wd / "wt.pdb"
        shutil.copy2(pdb_path, wt_pdb)

        subprocess.run(
            [str(evoef_binary), "--command=RepairStructure", "--pdb=wt.pdb"],
            cwd=wd,
            capture_output=True,
            check=True,
        )
        repaired = wd / "wt_Repair.pdb"

        mut_file = wd / "individual_list.txt"
        mut_file.write_text(f"{mutation.to_evoef()};\n")

        subprocess.run(
            [
                str(evoef_binary),
                "--command=BuildMutant",
                f"--pdb={repaired.name}",
                f"--mutant_file={mut_file.name}",
            ],
            cwd=wd,
            capture_output=True,
            check=True,
        )
        mutant_pdb = wd / f"{repaired.stem}_Model_0001.pdb"
        if not mutant_pdb.exists():
            raise FileNotFoundError("EvoEF failed to generate mutant PDB")

        # --- 2. Create input CSV ---
        # Columns expected by Graphinity's ddgDataSet.
        # The 'complex' column encodes the mutation: last element after '_'
        # is parsed as {wt_aa}{chain}{resnum}{mut_aa} (SKEMPI format).
        input_csv = wd / "input.csv"
        with open(input_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "pdb",
                    "complex",
                    "labels",
                    "chain_prot1",
                    "chain_prot2",
                    "ab_chain",
                    "ag_chain",
                    "pdb_wt",
                    "pdb_mut",
                ]
            )
            writer.writerow(
                [
                    pdb_path.stem,
                    f"query_{mutation.to_skempi()}",
                    0.0,
                    ab_chains,
                    ag_chains,
                    ab_chains,
                    ag_chains,
                    str(repaired),
                    str(mutant_pdb),
                ]
            )

        # --- 3. Create config YAML ---
        output_dir = wd / "output"
        output_dir.mkdir()

        config_text = _CONFIG_TEMPLATE.format(
            save_dir=str(output_dir),
            checkpoint=str(checkpoint),
            input_csv=str(input_csv),
        )
        config_path = wd / "config.yaml"
        config_path.write_text(config_text)

        # --- 4. Run inference ---
        inference_script = graphinity_dir / "src" / "ddg_regression" / "graphinity_inference.py"
        proc = subprocess.run(
            ["python", str(inference_script), "-c", str(config_path)],
            cwd=str(graphinity_dir),
            capture_output=True,
            text=True,
            timeout=120,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"Graphinity inference failed (exit {proc.returncode}): {proc.stderr}"
            )

        # --- 5. Parse output CSV ---
        output_csv = output_dir / "preds_inference.csv"
        if not output_csv.exists():
            raise FileNotFoundError(
                f"No output CSV produced. Files in output dir: {list(output_dir.iterdir())}"
            )
        return _parse_graphinity_csv(output_csv)


def main() -> None:
    maybe_relaunch_in_container("graphinity")

    ap = argparse.ArgumentParser(description="Graphinity Ab-Ag ddG scorer")
    ap.add_argument("pdb_path", type=Path)
    ap.add_argument("mutation", type=str)
    ap.add_argument("--ab-chains", required=True, help="Antibody chain IDs (e.g., HL)")
    ap.add_argument("--ag-chains", required=True, help="Antigen chain IDs (e.g., A)")
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
                scorer_name="graphinity",
            )
            print(result.to_json())
            sys.exit(1)

        ddg = run_graphinity(pdb_path, mutation, args.ab_chains, args.ag_chains)
        result = ToolResult(
            status="ok",
            scores={"ddg": round(ddg, 4)},
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="graphinity",
        )
    except SystemExit:
        raise
    except Exception as e:
        result = ToolResult(
            status="error",
            error_message=str(e),
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="graphinity",
        )

    print(result.to_json())
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()

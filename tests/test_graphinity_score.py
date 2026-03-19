"""Tests for tools/graphinity_score.py.

Unit tests exercise pure logic (CSV parsing, config generation, argument
validation) without Docker.  Container tests (marked ``container``) exercise
the full pipeline through the real Docker image — run with
``pytest -m container``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.graphinity_score import _CONFIG_TEMPLATE, _parse_graphinity_csv

PDB_PATH = Path(__file__).parent / "data" / "1N8Z.pdb"

# Env dict that prevents auto-containerization in subprocess tests.
_HOST_ENV = {**os.environ, "AUTOANTIBODY_CONTAINER": "1"}


# ── CSV parsing logic ────────────────────────────────────────────────────── #


class TestParseGraphinityCsv:
    def test_extracts_pred_score(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "preds.csv"
        csv_file.write_text("wt_pdb,mut_pdb,pred_score,true_label\nwt.pdb,mut.pdb,-0.75,0.0\n")
        assert _parse_graphinity_csv(csv_file) == pytest.approx(-0.75)

    def test_returns_first_row_on_multi(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "preds.csv"
        csv_file.write_text(
            "wt_pdb,mut_pdb,pred_score,true_label\na.pdb,b.pdb,-1.2,0.0\nc.pdb,d.pdb,0.3,0.0\n"
        )
        assert _parse_graphinity_csv(csv_file) == pytest.approx(-1.2)

    def test_raises_on_empty(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "preds.csv"
        csv_file.write_text("wt_pdb,mut_pdb,pred_score,true_label\n")
        with pytest.raises(ValueError, match="Empty"):
            _parse_graphinity_csv(csv_file)


# ── Config template ──────────────────────────────────────────────────────── #


class TestConfigTemplate:
    def test_renders_without_error(self, tmp_path: Path) -> None:
        text = _CONFIG_TEMPLATE.format(
            save_dir=str(tmp_path / "out"),
            checkpoint="/opt/graphinity/ckpt.ckpt",
            input_csv=str(tmp_path / "input.csv"),
        )
        # Should be valid YAML — test with stdlib only (no yaml import needed)
        assert "save_dir:" in text
        assert "checkpoint_file:" in text
        assert "input_files:" in text
        assert str(tmp_path / "input.csv") in text

    def test_architecture_params_match_checkpoint(self) -> None:
        """Verify config has the lmg-mode params that match shipped checkpoints."""
        text = _CONFIG_TEMPLATE.format(
            save_dir="/tmp",
            checkpoint="/tmp/ckpt",
            input_csv="/tmp/in.csv",
        )
        assert "num_node_features: 12" in text
        assert "typing_mode: lmg" in text
        assert "egnn_layer_hidden_nfs: [128, 128, 128]" in text


# ── CLI argument validation (host-side, no Docker) ───────────────────────── #


class TestGraphinityCLIValidation:
    def test_invalid_mutation_produces_error_json(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "tools/graphinity_score.py",
                str(PDB_PATH),
                "A:52:G:Y",  # wrong WT (actual is S)
                "--ab-chains",
                "AB",
                "--ag-chains",
                "C",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env=_HOST_ENV,
        )
        assert result.returncode != 0
        output = json.loads(result.stdout)
        assert output["status"] == "error"
        assert output["scorer_name"] == "graphinity"

    def test_missing_required_args_exits(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "tools/graphinity_score.py",
                str(PDB_PATH),
                "A:52:S:Y",
                # missing --ab-chains and --ag-chains
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env=_HOST_ENV,
        )
        assert result.returncode != 0


# ── Container integration (exercises real Docker image) ──────────────────── #


@pytest.mark.slow
@pytest.mark.container
class TestGraphinityContainer:
    """Run graphinity_score.py via auto-containerization (real Docker).

    Requires: ``make -C containers build-graphinity``
    Run with: ``pytest -m container -k graphinity``
    """

    def test_score_mutation(self, pdb_1n8z: Path) -> None:
        from .conftest import run_tool_cli, skip_unless_container

        skip_unless_container("autoantibody/graphinity:latest")
        data = run_tool_cli(
            "tools/graphinity_score.py",
            [str(pdb_1n8z), "B:52:S:Y", "--ab-chains", "AB", "--ag-chains", "C"],
            timeout=300,
        )
        assert data["status"] == "ok"
        assert "ddg" in data["scores"]
        assert data["scorer_name"] == "graphinity"

"""Tests for tools/stabddg_score.py.

Unit tests exercise pure logic (CSV parsing, argument validation) without
Docker.  Container tests (marked ``container``) exercise the full pipeline
through the real Docker image — run with ``pytest -m container``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.stabddg_score import _parse_stabddg_csv

PDB_PATH = Path(__file__).parent / "data" / "1N8Z.pdb"

# Env dict that prevents auto-containerization in subprocess tests.
_HOST_ENV = {**os.environ, "AUTOANTIBODY_CONTAINER": "1"}


# ── CSV parsing logic ────────────────────────────────────────────────────── #


class TestParseStabddgCsv:
    def test_extracts_correct_mutation(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "output.csv"
        csv_file.write_text(
            "Name,Mutation,pred_1\ncomplex,EA63Q,-0.31\ncomplex,SH52Y,-0.87\ncomplex,GH53A,0.12\n"
        )
        assert _parse_stabddg_csv(csv_file, "SH52Y") == pytest.approx(-0.87)

    def test_raises_on_missing_mutation(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "output.csv"
        csv_file.write_text("Name,Mutation,pred_1\ncomplex,SH52Y,-0.5\n")
        with pytest.raises(ValueError, match="GH53A"):
            _parse_stabddg_csv(csv_file, "GH53A")

    def test_raises_on_empty_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "output.csv"
        csv_file.write_text("Name,Mutation,pred_1\n")
        with pytest.raises(ValueError, match="FH54S"):
            _parse_stabddg_csv(csv_file, "FH54S")


# ── CLI argument validation (host-side, no Docker) ───────────────────────── #


class TestStabddgCLIValidation:
    def test_invalid_mutation_produces_error_json(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "tools/stabddg_score.py",
                str(PDB_PATH),
                "A:52:G:Y",  # wrong WT (actual is S)
                "--chains",
                "AB_C",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env=_HOST_ENV,
        )
        assert result.returncode != 0
        output = json.loads(result.stdout)
        assert output["status"] == "error"
        assert output["scorer_name"] == "stabddg"

    def test_bad_chains_spec_exits(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "tools/stabddg_score.py",
                str(PDB_PATH),
                "A:52:S:Y",
                "--chains",
                "ABC",  # missing underscore
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env=_HOST_ENV,
        )
        assert result.returncode != 0

    def test_no_mutation_or_batch_exits(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "tools/stabddg_score.py",
                str(PDB_PATH),
                "--chains",
                "AB_C",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env=_HOST_ENV,
        )
        assert result.returncode != 0

    def test_batch_mode_reads_file(self, tmp_path: Path) -> None:
        """Batch file with invalid mutations should still validate each one."""
        batch = tmp_path / "muts.txt"
        batch.write_text("A:52:G:Y\n")  # wrong WT (actual is S)

        result = subprocess.run(
            [
                sys.executable,
                "tools/stabddg_score.py",
                str(PDB_PATH),
                "--batch",
                str(batch),
                "--chains",
                "AB_C",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env=_HOST_ENV,
        )
        assert result.returncode != 0
        output = json.loads(result.stdout)
        assert output["status"] == "error"


# ── Container integration (exercises real Docker image) ──────────────────── #


@pytest.mark.slow
@pytest.mark.container
class TestStabddgContainer:
    """Run stabddg_score.py via auto-containerization (real Docker).

    Requires: ``make -C containers build-stabddg``
    Run with: ``pytest -m container -k stabddg``
    """

    def test_score_mutation(self, pdb_1n8z: Path) -> None:
        from .conftest import run_tool_cli, skip_unless_container

        skip_unless_container("autoantibody/stabddg:latest")
        data = run_tool_cli(
            "tools/stabddg_score.py",
            [str(pdb_1n8z), "B:52:S:Y", "--chains", "AB_C"],
            timeout=300,
        )
        assert data["status"] == "ok"
        assert "ddg" in data["scores"]
        assert data["scorer_name"] == "stabddg"

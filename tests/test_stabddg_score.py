"""Tests for tools/stabddg_score.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from autoantibody.models import Mutation
from tools.stabddg_score import run_stabddg

PDB_PATH = Path(__file__).parent / "data" / "1N8Z.pdb"


# ── run_stabddg ─────────────────────────────────────────────────────────── #


class TestRunStabddg:
    def test_calls_predict_ddg_with_correct_args(self) -> None:
        mutations = [
            Mutation(chain="H", resnum="52", wt_aa="S", mut_aa="Y"),
            Mutation(chain="H", resnum="53", wt_aa="G", mut_aa="A"),
        ]
        mock_predict = MagicMock(return_value=[0.5, -1.2])

        with patch.dict("sys.modules", {"stabddg": MagicMock(predict_ddg=mock_predict)}):
            results = run_stabddg(Path("/fake.pdb"), mutations, "HL", "A")

        mock_predict.assert_called_once_with(
            pdb_file="/fake.pdb",
            mutations=["SH52Y", "GH53A"],
            partner_chains=["HL", "A"],
        )
        assert results == [0.5, -1.2]


# ── CLI validation ───────────────────────────────────────────────────────── #


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
        )
        assert result.returncode != 0
        output = json.loads(result.stdout)
        assert output["status"] == "error"

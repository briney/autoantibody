"""Tests for tools/graphinity_score.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from autoantibody.models import Mutation
from tools.graphinity_score import run_graphinity

PDB_PATH = Path(__file__).parent / "data" / "1N8Z.pdb"


# ── run_graphinity ───────────────────────────────────────────────────────── #


class TestRunGraphinity:
    def test_calls_predict_with_correct_args(self) -> None:
        mutation = Mutation(chain="H", resnum="52", wt_aa="S", mut_aa="Y")
        mock_predict = MagicMock(return_value=-0.75)

        with patch.dict("sys.modules", {"graphinity": MagicMock(predict=mock_predict)}):
            result = run_graphinity(Path("/fake.pdb"), mutation, "HL", "A")

        mock_predict.assert_called_once_with(
            pdb_file="/fake.pdb",
            mutation="SH52Y",
            antibody_chains=["H", "L"],
            antigen_chains=["A"],
        )
        assert result == -0.75


# ── CLI validation ───────────────────────────────────────────────────────── #


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
        )
        assert result.returncode != 0

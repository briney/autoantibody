"""Tests for tools/ablms_score.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.ablms_score import build_sequence, cmd_score

PDB_PATH = Path(__file__).parent / "data" / "1N8Z.pdb"


# ── build_sequence ───────────────────────────────────────────────────────── #


class TestBuildSequence:
    def test_paired_sequence(self) -> None:
        mock_ab_seq = MagicMock()
        mock_module = MagicMock()
        mock_module.AntibodySequence.return_value = mock_ab_seq

        with patch.dict("sys.modules", {"ablms": mock_module}):
            result = build_sequence("EVQLVES", "DIQMTQS", "balm")

        mock_module.AntibodySequence.assert_called_once_with(heavy="EVQLVES", light="DIQMTQS")
        assert result == mock_ab_seq

    def test_heavy_only_sequence(self) -> None:
        mock_module = MagicMock()

        with patch.dict("sys.modules", {"ablms": mock_module}):
            build_sequence("EVQLVES", None, "esm2-650m")

        mock_module.AntibodySequence.assert_called_once_with(heavy="EVQLVES")


# ── cmd_score ────────────────────────────────────────────────────────────── #


class TestCmdScore:
    def test_returns_pll_score(self) -> None:
        mock_model = MagicMock()
        mock_model.pseudo_log_likelihood.return_value = [-123.45]
        mock_load = MagicMock(return_value=mock_model)
        mock_module = MagicMock(load_model=mock_load, AntibodySequence=MagicMock())

        with patch.dict("sys.modules", {"ablms": mock_module}):
            args = MagicMock(heavy="EVQLVES", light="DIQMTQS", model="balm", device="cpu")
            result = cmd_score(args)

        assert result == {"pll": -123.45}


# ── CLI validation ───────────────────────────────────────────────────────── #


class TestAblmsCLIValidation:
    def test_no_subcommand_exits(self) -> None:
        result = subprocess.run(
            [sys.executable, "tools/ablms_score.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode != 0

    def test_score_missing_heavy_exits(self) -> None:
        result = subprocess.run(
            [sys.executable, "tools/ablms_score.py", "score"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode != 0

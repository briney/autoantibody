"""Tests for tools/evoef_ddg.py."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the tool module functions directly
from tools.evoef_ddg import (
    find_evoef_binary,
    run_evoef_compute_binding,
    run_evoef_repair,
)

PDB_PATH = Path(__file__).parent / "data" / "1N8Z.pdb"

# Env dict that prevents auto-containerization in subprocess tests.
_HOST_ENV = {**__import__("os").environ, "AUTOANTIBODY_CONTAINER": "1"}


# ── find_evoef_binary ────────────────────────────────────────────────────── #


class TestFindEvoefBinary:
    def test_finds_from_env_var(self, tmp_path: Path) -> None:
        fake_bin = tmp_path / "EvoEF"
        fake_bin.touch()
        with patch.dict("os.environ", {"EVOEF_BINARY": str(fake_bin)}):
            assert find_evoef_binary() == fake_bin

    def test_exits_if_env_var_path_missing(self) -> None:
        with (
            patch.dict("os.environ", {"EVOEF_BINARY": "/nonexistent/EvoEF"}),
            pytest.raises(SystemExit),
        ):
            find_evoef_binary()

    def test_finds_from_path(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("shutil.which", return_value="/usr/local/bin/EvoEF"),
        ):
            assert find_evoef_binary() == Path("/usr/local/bin/EvoEF")

    def test_exits_if_not_found(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("shutil.which", return_value=None),
            pytest.raises(SystemExit),
        ):
            find_evoef_binary()


# ── run_evoef_repair ─────────────────────────────────────────────────────── #


class TestRunEvoefRepair:
    def test_returns_repaired_path(self, tmp_path: Path) -> None:
        pdb = tmp_path / "test.pdb"
        pdb.touch()
        repaired = tmp_path / "test_Repair.pdb"
        repaired.write_text("REPAIRED")

        with patch("tools.evoef_ddg.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = run_evoef_repair(Path("/bin/EvoEF"), pdb, tmp_path)
            assert result == repaired

    def test_raises_on_missing_output(self, tmp_path: Path) -> None:
        pdb = tmp_path / "test.pdb"
        pdb.touch()

        with patch("tools.evoef_ddg.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(FileNotFoundError, match="RepairStructure"):
                run_evoef_repair(Path("/bin/EvoEF"), pdb, tmp_path)


# ── run_evoef_compute_binding ────────────────────────────────────────────── #


class TestRunEvoefComputeBinding:
    def test_parses_binding_energy(self, tmp_path: Path) -> None:
        pdb = tmp_path / "test.pdb"
        pdb.touch()

        stdout = textwrap.dedent("""\
            EvoEF ComputeBinding output
            Reference energy: -10.5
            Total binding energy = -25.3412
            Done.
        """)
        with patch("tools.evoef_ddg.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=stdout)
            energy = run_evoef_compute_binding(Path("/bin/EvoEF"), pdb, tmp_path)
            assert energy == pytest.approx(-25.3412)

    def test_raises_on_unparseable_output(self, tmp_path: Path) -> None:
        pdb = tmp_path / "test.pdb"
        pdb.touch()

        with patch("tools.evoef_ddg.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="no energy here")
            with pytest.raises(ValueError, match="Could not parse"):
                run_evoef_compute_binding(Path("/bin/EvoEF"), pdb, tmp_path)


# ── CLI validation via main() ────────────────────────────────────────────── #


class TestEvoefCLIValidation:
    def test_invalid_mutation_produces_error_json(self) -> None:
        """A mutation with wrong wild-type AA should produce error JSON."""
        result = subprocess.run(
            [
                sys.executable,
                "tools/evoef_ddg.py",
                str(PDB_PATH),
                "A:52:G:Y",  # Position 52 in chain A is S, not G
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env=_HOST_ENV,
        )
        assert result.returncode != 0
        output = json.loads(result.stdout)
        assert output["status"] == "error"
        assert output["scorer_name"] == "evoef"
        assert "mismatch" in output["error_message"].lower()

    def test_wrong_argc_exits(self) -> None:
        result = subprocess.run(
            [sys.executable, "tools/evoef_ddg.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env=_HOST_ENV,
        )
        assert result.returncode != 0

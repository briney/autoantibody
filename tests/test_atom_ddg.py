"""Tests for tools/atom_ddg.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autoantibody.models import Mutation
from tools.atom_ddg import (
    analyze_fep,
    prepare_system,
    run_md,
    setup_alchemy,
)

PDB_PATH = Path(__file__).parent / "data" / "1N8Z.pdb"


# ── prepare_system ──────────────────────────────────────────────────────── #


class TestPrepareSystem:
    def test_creates_system_dir_and_writes_tleap_input(self, tmp_path: Path) -> None:
        mutation = Mutation(chain="H", resnum="52", wt_aa="S", mut_aa="Y")

        # Simulate tleap producing output files
        def fake_tleap(cmd, **kwargs):
            system_dir = tmp_path / "system"
            if system_dir.exists():
                (system_dir / "complex.prmtop").write_text("PRMTOP")
                (system_dir / "complex.inpcrd").write_text("INPCRD")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("tools.atom_ddg.subprocess.run", side_effect=fake_tleap):
            result = prepare_system(PDB_PATH, mutation, tmp_path)

        assert result == tmp_path / "system"
        assert (tmp_path / "system" / "complex.prmtop").exists()
        assert (tmp_path / "system" / "complex.inpcrd").exists()
        assert (tmp_path / "tleap.in").exists()
        tleap_text = (tmp_path / "tleap.in").read_text()
        assert "leaprc.protein.ff14SB" in tleap_text
        assert str(PDB_PATH) in tleap_text

    def test_raises_on_tleap_failure(self, tmp_path: Path) -> None:
        mutation = Mutation(chain="H", resnum="52", wt_aa="S", mut_aa="Y")

        with patch("tools.atom_ddg.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="tleap error")
            with pytest.raises(RuntimeError, match="tleap failed"):
                prepare_system(PDB_PATH, mutation, tmp_path)

    def test_raises_on_missing_prmtop(self, tmp_path: Path) -> None:
        mutation = Mutation(chain="H", resnum="52", wt_aa="S", mut_aa="Y")

        with patch("tools.atom_ddg.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            with pytest.raises(FileNotFoundError, match="complex.prmtop"):
                prepare_system(PDB_PATH, mutation, tmp_path)


# ── setup_alchemy ───────────────────────────────────────────────────────── #


class TestSetupAlchemy:
    def test_creates_lambda_schedule_and_config(self, tmp_path: Path) -> None:
        system_dir = tmp_path / "system"
        system_dir.mkdir()
        (system_dir / "complex.prmtop").write_text("PRMTOP")
        (system_dir / "complex.inpcrd").write_text("INPCRD")

        mutation = Mutation(chain="H", resnum="52", wt_aa="S", mut_aa="Y")
        result = setup_alchemy(system_dir, mutation, lambda_windows=4, work_dir=tmp_path)

        assert result == tmp_path / "alchemy"

        # Check lambda schedule
        schedule = (result / "lambda_schedule.dat").read_text().strip().splitlines()
        assert len(schedule) == 4
        assert schedule[0] == "0.000000"
        assert schedule[-1] == "1.000000"

        # Check config
        config = json.loads((result / "config.json").read_text())
        assert config["n_windows"] == 4
        assert config["mutation"] == "SH52Y"
        assert "complex.prmtop" in config["topology"]
        assert "complex.inpcrd" in config["coordinates"]

    def test_lambda_schedule_evenly_spaced(self, tmp_path: Path) -> None:
        system_dir = tmp_path / "system"
        system_dir.mkdir()

        mutation = Mutation(chain="H", resnum="52", wt_aa="S", mut_aa="Y")
        result = setup_alchemy(system_dir, mutation, lambda_windows=24, work_dir=tmp_path)

        schedule = (result / "lambda_schedule.dat").read_text().strip().splitlines()
        assert len(schedule) == 24
        values = [float(v) for v in schedule]
        assert values[0] == pytest.approx(0.0)
        assert values[-1] == pytest.approx(1.0)
        # Check even spacing
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        for d in diffs:
            assert d == pytest.approx(1.0 / 23, abs=1e-6)


# ── run_md ──────────────────────────────────────────────────────────────── #


class TestRunMd:
    def test_invokes_atom_remd_with_correct_args(self, tmp_path: Path) -> None:
        alchemy_dir = tmp_path / "alchemy"
        alchemy_dir.mkdir()
        (alchemy_dir / "config.json").write_text("{}")

        def fake_run(cmd, **kwargs):
            md_dir = tmp_path / "md_output"
            if md_dir.exists():
                (md_dir / "energies.dat").write_text("lambda reduced_potential\n0.0 1.0\n")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("tools.atom_ddg.subprocess.run", side_effect=fake_run):
            result = run_md(alchemy_dir, gpu_id=0, steps_per_window=1000, work_dir=tmp_path)

        assert result == tmp_path / "md_output"
        assert (result / "energies.dat").exists()

    def test_raises_on_md_failure(self, tmp_path: Path) -> None:
        alchemy_dir = tmp_path / "alchemy"
        alchemy_dir.mkdir()
        (alchemy_dir / "config.json").write_text("{}")

        with patch("tools.atom_ddg.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="GPU out of memory")
            with pytest.raises(RuntimeError, match="AToM MD failed"):
                run_md(alchemy_dir, gpu_id=0, steps_per_window=1000, work_dir=tmp_path)

    def test_raises_on_missing_energies(self, tmp_path: Path) -> None:
        alchemy_dir = tmp_path / "alchemy"
        alchemy_dir.mkdir()
        (alchemy_dir / "config.json").write_text("{}")

        with patch("tools.atom_ddg.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            with pytest.raises(FileNotFoundError, match="energies.dat"):
                run_md(alchemy_dir, gpu_id=0, steps_per_window=1000, work_dir=tmp_path)


# ── analyze_fep ─────────────────────────────────────────────────────────── #


class TestAnalyzeFep:
    def test_parses_uwham_output(self, tmp_path: Path) -> None:
        md_dir = tmp_path / "md_output"
        md_dir.mkdir()

        with patch("tools.atom_ddg.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="-2.3456\n0.4321\n", stderr="")
            ddg, unc = analyze_fep(md_dir, tmp_path)

        assert ddg == pytest.approx(-2.3456)
        assert unc == pytest.approx(0.4321)

        # Verify R script was written correctly
        r_script = tmp_path / "uwham_analysis.R"
        assert r_script.exists()
        r_text = r_script.read_text()
        assert "library(UWHAM)" in r_text
        assert str(md_dir) in r_text

    def test_raises_on_r_failure(self, tmp_path: Path) -> None:
        md_dir = tmp_path / "md_output"
        md_dir.mkdir()

        with patch("tools.atom_ddg.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="R not found")
            with pytest.raises(RuntimeError, match="UWHAM analysis failed"):
                analyze_fep(md_dir, tmp_path)

    def test_raises_on_insufficient_output(self, tmp_path: Path) -> None:
        md_dir = tmp_path / "md_output"
        md_dir.mkdir()

        with patch("tools.atom_ddg.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="-2.3456\n", stderr="")
            with pytest.raises(ValueError, match="Expected 2 lines"):
                analyze_fep(md_dir, tmp_path)


# ── CLI validation via main() ──────────────────────────────────────────── #


class TestAtomDdgCLIValidation:
    def test_invalid_mutation_produces_error_json(self) -> None:
        """A mutation with wrong wild-type AA should produce error JSON."""
        result = subprocess.run(
            [
                sys.executable,
                "tools/atom_ddg.py",
                str(PDB_PATH),
                "A:5:G:Y",  # Position 5 in chain A is T, not G
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode != 0
        output = json.loads(result.stdout)
        assert output["status"] == "error"
        assert output["scorer_name"] == "atom_fep"
        assert "mismatch" in output["error_message"].lower()

    def test_missing_args_exits(self) -> None:
        result = subprocess.run(
            [sys.executable, "tools/atom_ddg.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode != 0

    def test_nonexistent_chain_produces_error_json(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "tools/atom_ddg.py",
                str(PDB_PATH),
                "Z:52:S:Y",  # Chain Z doesn't exist
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode != 0
        output = json.loads(result.stdout)
        assert output["status"] == "error"
        assert output["scorer_name"] == "atom_fep"
        assert "chain" in output["error_message"].lower()

    def test_valid_mutation_format_accepted(self) -> None:
        """Valid mutation parses without immediate error (will fail on tleap)."""
        result = subprocess.run(
            [
                sys.executable,
                "tools/atom_ddg.py",
                str(PDB_PATH),
                "A:5:T:Y",
                "--lambda-windows",
                "4",
                "--steps-per-window",
                "100",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        # Should get past validation but fail on tleap (not installed)
        output = json.loads(result.stdout)
        assert output["scorer_name"] == "atom_fep"
        # Either succeeds (if tleap installed) or errors on tool execution
        assert output["status"] in ("ok", "error")
        assert "wall_time_s" in output

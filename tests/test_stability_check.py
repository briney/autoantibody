"""Tests for tools/stability_check.py.

Unit tests exercise pure logic (CSV parsing, chain extraction, argument
validation) without Docker.  Container tests (marked ``container``) exercise
the full pipeline through the real Docker image.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from _common import Mutation

from tools.stability_check import _extract_mutation_ddg, _get_chain_ids

PDB_PATH = Path(__file__).parent / "data" / "1N8Z.pdb"

# Env dict that prevents auto-containerization in subprocess tests.
_HOST_ENV = {**os.environ, "AUTOANTIBODY_CONTAINER": "1"}


# ── _extract_mutation_ddg ────────────────────────────────────────────────── #


class TestExtractMutationDdg:
    def test_unique_match(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "out.csv"
        csv_file.write_text(
            ",pre,post,pos,logit_difference,logit_difference_ddg\n"
            "0,S,A,52,-0.1,-1.5\n"
            "1,S,Y,52,0.3,-0.8\n"
            "2,G,A,53,-0.05,-0.4\n"
        )
        mutation = Mutation(chain="H", resnum="52", wt_aa="S", mut_aa="Y")
        assert _extract_mutation_ddg(csv_file, mutation) == pytest.approx(-0.8)

    def test_prefers_ddg_corrected(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "out.csv"
        csv_file.write_text(
            ",pre,post,pos,logit_difference,logit_difference_ddg\n0,S,Y,52,0.3,-0.8\n"
        )
        mutation = Mutation(chain="H", resnum="52", wt_aa="S", mut_aa="Y")
        assert _extract_mutation_ddg(csv_file, mutation) == pytest.approx(-0.8)

    def test_falls_back_to_logit_difference(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "out.csv"
        csv_file.write_text(",pre,post,pos,logit_difference\n0,S,Y,52,0.3\n")
        mutation = Mutation(chain="H", resnum="52", wt_aa="S", mut_aa="Y")
        assert _extract_mutation_ddg(csv_file, mutation) == pytest.approx(0.3)

    def test_disambiguates_by_position(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "out.csv"
        csv_file.write_text(
            ",pre,post,pos,logit_difference,logit_difference_ddg\n"
            "0,S,Y,10,0.1,0.2\n"
            "1,S,Y,52,0.3,-0.8\n"
        )
        mutation = Mutation(chain="H", resnum="52", wt_aa="S", mut_aa="Y")
        assert _extract_mutation_ddg(csv_file, mutation) == pytest.approx(-0.8)

    def test_raises_on_no_match(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "out.csv"
        csv_file.write_text(
            ",pre,post,pos,logit_difference,logit_difference_ddg\n0,G,A,53,-0.05,-0.4\n"
        )
        mutation = Mutation(chain="H", resnum="52", wt_aa="S", mut_aa="Y")
        with pytest.raises(ValueError, match="No rows match"):
            _extract_mutation_ddg(csv_file, mutation)

    def test_handles_insertion_code(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "out.csv"
        csv_file.write_text(
            ",pre,post,pos,logit_difference,logit_difference_ddg\n0,G,A,100,0.5,0.6\n"
        )
        mutation = Mutation(chain="H", resnum="100A", wt_aa="G", mut_aa="A")
        assert _extract_mutation_ddg(csv_file, mutation) == pytest.approx(0.6)


# ── _get_chain_ids ───────────────────────────────────────────────────────── #


class TestGetChainIds:
    def test_extracts_chains_from_real_pdb(self, pdb_1n8z: Path) -> None:
        chains = _get_chain_ids(pdb_1n8z)
        assert "A" in chains
        assert "B" in chains
        assert "C" in chains

    def test_empty_file(self, tmp_path: Path) -> None:
        pdb = tmp_path / "empty.pdb"
        pdb.write_text("HEADER\nEND\n")
        assert _get_chain_ids(pdb) == []


# ── CLI argument validation ──────────────────────────────────────────────── #


class TestStabilityCheckCLIValidation:
    def test_invalid_mutation_produces_error_json(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "tools/stability_check.py",
                str(PDB_PATH),
                "A:52:G:Y",  # wrong WT
                "--chain",
                "A",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env=_HOST_ENV,
        )
        assert result.returncode != 0
        output = json.loads(result.stdout)
        assert output["status"] == "error"
        assert output["scorer_name"] == "proteinmpnn_stability"

    def test_missing_chain_arg_exits(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "tools/stability_check.py",
                str(PDB_PATH),
                "B:52:S:Y",
                # missing --chain
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env=_HOST_ENV,
        )
        assert result.returncode != 0


# ── Container integration ────────────────────────────────────────────────── #


@pytest.mark.slow
@pytest.mark.container
class TestStabilityCheckContainer:
    """Run stability_check.py via auto-containerization (real Docker).

    Requires: ``make -C containers build-proteinmpnn_stability``
    Run with: ``pytest -m container -k stability``
    """

    def test_score_mutation(self, pdb_1n8z: Path) -> None:
        from .conftest import run_tool_cli, skip_unless_container

        skip_unless_container("autoantibody/proteinmpnn_stability:latest")
        data = run_tool_cli(
            "tools/stability_check.py",
            [str(pdb_1n8z), "B:52:S:Y", "--chain", "B"],
            timeout=300,
        )
        assert data["status"] == "ok"
        assert "stability_ddg" in data["scores"]
        assert data["scorer_name"] == "proteinmpnn_stability"

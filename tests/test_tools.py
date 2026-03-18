"""Integration tests for tool wrappers.

These tests require external tools and are marked slow:
- EvoEF binary (EVOEF_BINARY env var)
- StaB-ddG package
- Graphinity package
- BA-ddG installation
- Docker with rosettacommons/rosetta and proteinmpnn_ddg images
- GPU with ablms installed

Run with: pytest -m slow
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from autoantibody.models import ToolResult
from autoantibody.structure import extract_sequences, get_residue_map

from .conftest import ANTIGEN_CHAINS, HEAVY_CHAIN, LIGHT_CHAIN


def _pick_test_mutation(pdb_path: Path) -> str:
    """Pick a valid mutation from the heavy chain for testing."""
    rmap = get_residue_map(pdb_path, HEAVY_CHAIN)
    for resnum, aa in rmap.items():
        mut = "A" if aa != "A" else "G"
        return f"{HEAVY_CHAIN}:{resnum}:{aa}:{mut}"
    raise RuntimeError("No residues found")


@pytest.mark.slow
class TestEvoEF:
    def test_evoef_runs(self, pdb_1n8z: Path) -> None:
        mutation = _pick_test_mutation(pdb_1n8z)
        result = subprocess.run(
            [sys.executable, "tools/evoef_ddg.py", str(pdb_1n8z), mutation],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        tr = ToolResult(**data)
        assert tr.status == "ok"
        assert "ddg" in tr.scores
        assert tr.scorer_name == "evoef"

    def test_evoef_rejects_bad_mutation(self, pdb_1n8z: Path) -> None:
        result = subprocess.run(
            [sys.executable, "tools/evoef_ddg.py", str(pdb_1n8z), "Z:999:A:G"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
        data = json.loads(result.stdout)
        assert data["status"] == "error"


@pytest.mark.slow
class TestStabDDG:
    def test_stabddg_runs(self, pdb_1n8z: Path) -> None:
        mutation = _pick_test_mutation(pdb_1n8z)
        result = subprocess.run(
            [
                sys.executable,
                "tools/stabddg_score.py",
                str(pdb_1n8z),
                mutation,
                "--chains",
                f"{HEAVY_CHAIN}{LIGHT_CHAIN}_{ANTIGEN_CHAINS[0]}",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        tr = ToolResult(**data)
        assert tr.status == "ok"
        assert "ddg" in tr.scores
        assert tr.scorer_name == "stabddg"


@pytest.mark.slow
class TestGraphinity:
    def test_graphinity_runs(self, pdb_1n8z: Path) -> None:
        mutation = _pick_test_mutation(pdb_1n8z)
        result = subprocess.run(
            [
                sys.executable,
                "tools/graphinity_score.py",
                str(pdb_1n8z),
                mutation,
                "--ab-chains",
                f"{HEAVY_CHAIN}{LIGHT_CHAIN}",
                "--ag-chains",
                ANTIGEN_CHAINS[0],
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        tr = ToolResult(**data)
        assert tr.status == "ok"
        assert "ddg" in tr.scores
        assert tr.scorer_name == "graphinity"


@pytest.mark.slow
class TestBADDG:
    def test_baddg_runs(self, pdb_1n8z: Path) -> None:
        mutation = _pick_test_mutation(pdb_1n8z)
        result = subprocess.run(
            [
                sys.executable,
                "tools/baddg_score.py",
                str(pdb_1n8z),
                mutation,
                "--chains",
                f"{HEAVY_CHAIN}{LIGHT_CHAIN}_{ANTIGEN_CHAINS[0]}",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        tr = ToolResult(**data)
        assert tr.status == "ok"
        assert "ddg" in tr.scores
        assert tr.scorer_name == "baddg"


@pytest.mark.slow
class TestStabilityCheck:
    def test_stability_check_runs(self, pdb_1n8z: Path) -> None:
        mutation = _pick_test_mutation(pdb_1n8z)
        chain = mutation.split(":")[0]
        result = subprocess.run(
            [
                sys.executable,
                "tools/stability_check.py",
                str(pdb_1n8z),
                mutation,
                "--chain",
                chain,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        tr = ToolResult(**data)
        assert tr.status == "ok"
        assert "stability_ddg" in tr.scores
        assert tr.scorer_name == "proteinmpnn_stability"


@pytest.mark.slow
class TestAtomFEP:
    def test_atom_fep_quick_run(self, pdb_1n8z: Path) -> None:
        """Minimal AToM-FEP run (4 windows, 1000 steps)."""
        mutation = _pick_test_mutation(pdb_1n8z)
        result = subprocess.run(
            [
                sys.executable,
                "tools/atom_ddg.py",
                str(pdb_1n8z),
                mutation,
                "--lambda-windows",
                "4",
                "--steps-per-window",
                "1000",
            ],
            capture_output=True,
            text=True,
            timeout=7200,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        tr = ToolResult(**data)
        assert tr.status == "ok"
        assert "ddg" in tr.scores
        assert "ddg_uncertainty" in tr.scores
        assert tr.scorer_name == "atom_fep"


@pytest.mark.slow
class TestFlexDDG:
    def test_flex_ddg_quick_run(self, pdb_1n8z: Path) -> None:
        """Minimal flex-ddG run (2 structures, 5000 trials)."""
        mutation = _pick_test_mutation(pdb_1n8z)
        result = subprocess.run(
            [
                sys.executable,
                "tools/flex_ddg.py",
                str(pdb_1n8z),
                mutation,
                "--nstruct",
                "2",
                "--backrub-trials",
                "5000",
                "--relax-nstruct",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        tr = ToolResult(**data)
        assert tr.status == "ok"
        assert "ddg" in tr.scores
        assert "ddg_std" in tr.scores
        assert tr.scorer_name == "flex_ddg"


@pytest.mark.slow
class TestAblms:
    def test_ablms_score(self, pdb_1n8z: Path) -> None:
        seqs = extract_sequences(pdb_1n8z)
        heavy_seq = seqs[HEAVY_CHAIN]
        light_seq = seqs[LIGHT_CHAIN]

        cmd = [
            sys.executable,
            "tools/ablms_score.py",
            "score",
            "--heavy",
            heavy_seq,
            "--light",
            light_seq,
            "--model",
            "balm",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        tr = ToolResult(**data)
        assert tr.status == "ok"
        assert "pll" in tr.scores
        assert tr.scorer_name == "ablms"


@pytest.mark.slow
class TestEndToEnd:
    def test_full_workflow(self, tmp_path: Path, pdb_1n8z: Path) -> None:
        """Init campaign -> EvoEF screen -> verify state update."""
        from datetime import UTC, datetime

        from autoantibody.models import IterationDecision
        from autoantibody.state import (
            append_ledger,
            init_campaign,
            load_ledger,
            load_state,
            save_state,
        )
        from autoantibody.structure import get_interface_residues

        # 1. Init campaign
        campaign_dir = tmp_path / "e2e_campaign"
        state = init_campaign(
            campaign_dir=campaign_dir,
            pdb_path=pdb_1n8z,
            antibody_heavy_chain=HEAVY_CHAIN,
            antibody_light_chain=LIGHT_CHAIN,
            antigen_chains=ANTIGEN_CHAINS,
        )
        assert state.iteration == 0

        # 2. Get interface residues
        interface = get_interface_residues(
            pdb_1n8z,
            antibody_chains=[HEAVY_CHAIN, LIGHT_CHAIN],
            antigen_chains=ANTIGEN_CHAINS,
        )
        assert len(interface) > 0

        # 3. Run EvoEF on first interface residue
        res = interface[0]
        mut_aa = "A" if res["aa"] != "A" else "G"
        mutation_str = f"{res['chain']}:{res['resnum']}:{res['aa']}:{mut_aa}"

        result = subprocess.run(
            [sys.executable, "tools/evoef_ddg.py", str(pdb_1n8z), mutation_str],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        tool_result = ToolResult(**json.loads(result.stdout))
        assert tool_result.status == "ok"
        assert "ddg" in tool_result.scores
        assert tool_result.scorer_name == "evoef"

        # 4. Record decision and update state
        decision = IterationDecision(
            iteration=1,
            mutation=mutation_str,
            proxy_scores={"evoef_ddg": tool_result.scores["ddg"]},
            oracle_ddg=tool_result.scores["ddg"],
            accepted=tool_result.scores["ddg"] < 0,
            rationale="E2E test mutation",
            timestamp=datetime.now(UTC),
        )
        append_ledger(campaign_dir, decision)

        state.iteration = 1
        if decision.accepted:
            state.parent.ddg_cumulative += decision.oracle_ddg
        save_state(campaign_dir, state)

        # 5. Verify
        reloaded = load_state(campaign_dir)
        assert reloaded.iteration == 1
        ledger = load_ledger(campaign_dir)
        assert len(ledger) == 1
        assert ledger[0].mutation == mutation_str

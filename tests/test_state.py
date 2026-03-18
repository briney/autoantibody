"""Tests for state management."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from autoantibody.models import IterationDecision
from autoantibody.state import (
    append_ledger,
    init_campaign,
    load_ledger,
    load_state,
    save_state,
)

from .conftest import ANTIGEN_CHAINS, HEAVY_CHAIN, LIGHT_CHAIN


class TestStateRoundTrip:
    def test_load_after_init(self, tmp_campaign: Path) -> None:
        state = load_state(tmp_campaign)
        assert state.campaign_id == "test_cmp_001"
        assert state.iteration == 0
        assert state.parent.ddg_cumulative == 0.0
        assert len(state.parent.sequence_heavy) > 0
        assert len(state.parent.sequence_light) > 0

    def test_save_and_reload(self, tmp_campaign: Path) -> None:
        state = load_state(tmp_campaign)
        state.iteration = 5
        state.parent.ddg_cumulative = -2.5
        save_state(tmp_campaign, state)

        reloaded = load_state(tmp_campaign)
        assert reloaded.iteration == 5
        assert reloaded.parent.ddg_cumulative == -2.5
        assert reloaded.parent.sequence_heavy == state.parent.sequence_heavy


class TestLedger:
    def test_empty_ledger(self, tmp_campaign: Path) -> None:
        decisions = load_ledger(tmp_campaign)
        assert decisions == []

    def test_append_and_load(self, tmp_campaign: Path) -> None:
        d = IterationDecision(
            iteration=1,
            mutation="B:52:S:Y",
            proxy_scores={"evoef_ddg": -1.2, "stabddg_ddg": -0.8},
            oracle_ddg=-0.8,
            accepted=True,
            rationale="Test mutation",
            timestamp=datetime.now(UTC),
        )
        append_ledger(tmp_campaign, d)

        decisions = load_ledger(tmp_campaign)
        assert len(decisions) == 1
        assert decisions[0].mutation == "B:52:S:Y"
        assert decisions[0].accepted is True

    def test_multiple_appends(self, tmp_campaign: Path) -> None:
        for i in range(3):
            d = IterationDecision(
                iteration=i + 1,
                mutation=f"B:{50 + i}:A:G",
                oracle_ddg=-0.5 if i % 2 == 0 else 0.3,
                accepted=i % 2 == 0,
                rationale=f"Iteration {i + 1}",
                timestamp=datetime.now(UTC),
            )
            append_ledger(tmp_campaign, d)

        decisions = load_ledger(tmp_campaign)
        assert len(decisions) == 3
        assert decisions[0].iteration == 1
        assert decisions[2].iteration == 3


class TestInitCampaign:
    def test_directory_structure(self, tmp_campaign: Path) -> None:
        assert (tmp_campaign / "state.yaml").exists()
        assert (tmp_campaign / "ledger.jsonl").exists()
        assert (tmp_campaign / "input").is_dir()
        assert (tmp_campaign / "iterations").is_dir()

    def test_input_pdb_copied(self, tmp_campaign: Path) -> None:
        input_files = list((tmp_campaign / "input").glob("*.pdb"))
        assert len(input_files) == 1

    def test_bad_chain_id(self, tmp_path: Path, pdb_1n8z: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            init_campaign(
                campaign_dir=tmp_path / "bad",
                pdb_path=pdb_1n8z,
                antibody_heavy_chain="Z",
                antibody_light_chain=LIGHT_CHAIN,
                antigen_chains=ANTIGEN_CHAINS,
            )

    def test_auto_campaign_id(self, tmp_path: Path, pdb_1n8z: Path) -> None:
        state = init_campaign(
            campaign_dir=tmp_path / "auto_id",
            pdb_path=pdb_1n8z,
            antibody_heavy_chain=HEAVY_CHAIN,
            antibody_light_chain=LIGHT_CHAIN,
            antigen_chains=ANTIGEN_CHAINS,
        )
        assert state.campaign_id.startswith("cmp_")

    def test_frozen_positions(self, tmp_path: Path, pdb_1n8z: Path) -> None:
        state = init_campaign(
            campaign_dir=tmp_path / "frozen",
            pdb_path=pdb_1n8z,
            antibody_heavy_chain=HEAVY_CHAIN,
            antibody_light_chain=LIGHT_CHAIN,
            antigen_chains=ANTIGEN_CHAINS,
            frozen_positions=["B:52", "A:91"],
        )
        assert state.frozen_positions == ["B:52", "A:91"]

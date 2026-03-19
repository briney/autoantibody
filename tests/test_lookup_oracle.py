"""Tests for the CR9114 lookup oracle."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from autoantibody.models import CampaignState, Mutation, ParentState
from autoantibody.state import save_state

# Attempt to import pandas; skip all tests if unavailable
pd = pytest.importorskip("pandas")

# Constants matching lookup_oracle.py
R_KCAL = 0.001987
TEMPERATURE = 298.0
RT_LN10 = R_KCAL * TEMPERATURE * math.log(10)
KD_FLOOR = 6.0
NON_BINDING_DDG = 10.0


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def sample_positions() -> list[dict]:
    """Minimal set of 3 positions for testing (subset of the 16)."""
    return [
        {
            "index": 1,
            "chain": "H",
            "pdb_resnum": "31",
            "germline_aa": "G",
            "mature_aa": "D",
            "seq_position": 30,
        },
        {
            "index": 2,
            "chain": "H",
            "pdb_resnum": "33",
            "germline_aa": "Y",
            "mature_aa": "N",
            "seq_position": 32,
        },
        {
            "index": 3,
            "chain": "H",
            "pdb_resnum": "52",
            "germline_aa": "I",
            "mature_aa": "M",
            "seq_position": 50,
        },
    ]


@pytest.fixture
def sample_variants(sample_positions: list[dict]) -> pd.DataFrame:
    """Build a minimal 8-variant lookup table (2^3 combinations)."""
    rows = []
    n_pos = len(sample_positions)
    for i in range(2**n_pos):
        genotype = f"{i:0{n_pos}b}"  # e.g., "000", "001", ..., "111"
        # Assign distinct -logKd values: higher som_mut = higher affinity
        som_mut = genotype.count("1")
        h1_mean = 7.0 + som_mut * 0.5  # 7.0, 7.5, 8.0, 8.5, 9.0, ...
        h3_mean = 6.0 if som_mut < 2 else 7.0 + (som_mut - 2) * 0.3
        # Pad genotype to 16 chars (real data has 16 positions)
        genotype_16 = genotype.ljust(16, "0")
        rows.append(
            {
                "genotype": genotype_16,
                "pos1": int(genotype[0]),
                "pos2": int(genotype[1]),
                "pos3": int(genotype[2]),
                "som_mut": som_mut,
                "h1_mean": h1_mean,
                "h1_sem": 0.01,
                "h3_mean": h3_mean,
                "h3_sem": 0.01,
                "fluB_mean": KD_FLOOR,
                "fluB_sem": 0.0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def cr9114_data_dir(
    tmp_path: Path,
    sample_positions: list[dict],
    sample_variants: pd.DataFrame,
) -> Path:
    """Create a temporary CR9114 data directory with test data."""
    data_dir = tmp_path / "data" / "cr9114"
    data_dir.mkdir(parents=True)

    # Save variants
    sample_variants.to_parquet(data_dir / "variants.parquet", index=False)

    # Save mutations YAML
    mutations_data = {
        "positions": sample_positions,
        "heavy_chain": "H",
        "light_chain": "L",
        "antigen_chains": ["A", "B"],
    }
    with open(data_dir / "mutations.yaml", "w") as f:
        yaml.dump(mutations_data, f)

    return data_dir


@pytest.fixture
def mock_campaign(
    tmp_path: Path,
    sample_positions: list[dict],
) -> Path:
    """Create a mock campaign directory with germline state."""
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir(parents=True)
    (campaign_dir / "input").mkdir()
    (campaign_dir / "iterations").mkdir()

    # Build a mock heavy chain sequence where the 3 positions are at germline
    seq = list("A" * 100)
    for pos in sample_positions:
        seq[pos["seq_position"]] = pos["germline_aa"]
    heavy_seq = "".join(seq)

    state = CampaignState(
        campaign_id="test_cr9114",
        started_at="2026-03-18T00:00:00Z",
        iteration=0,
        parent=ParentState(
            sequence_heavy=heavy_seq,
            sequence_light="DIQMT" + "A" * 95,
            structure="input/4FQI_clean.pdb",
            ddg_cumulative=0.0,
        ),
        antigen_chains=["A", "B"],
        antibody_heavy_chain="H",
        antibody_light_chain="L",
        frozen_positions=[],
    )
    save_state(campaign_dir, state)
    (campaign_dir / "ledger.jsonl").touch()

    return campaign_dir


# -- Unit tests for oracle internals ------------------------------------------


class TestGetCurrentGenotype:
    """Tests for get_current_genotype."""

    def test_all_germline(self, mock_campaign: Path, sample_positions: list[dict]) -> None:
        from autoantibody.state import load_state
        from tools.lookup_oracle import get_current_genotype

        state = load_state(mock_campaign)
        genotype = get_current_genotype(state.parent.sequence_heavy, sample_positions)
        assert genotype == "000"

    def test_one_flipped(self, mock_campaign: Path, sample_positions: list[dict]) -> None:
        from autoantibody.state import load_state, save_state
        from tools.lookup_oracle import get_current_genotype

        state = load_state(mock_campaign)
        # Flip position 1 to mature
        seq = list(state.parent.sequence_heavy)
        seq[30] = "D"  # mature AA for position 1
        state.parent.sequence_heavy = "".join(seq)
        save_state(mock_campaign, state)

        state = load_state(mock_campaign)
        genotype = get_current_genotype(state.parent.sequence_heavy, sample_positions)
        assert genotype == "100"

    def test_invalid_aa_raises(self, sample_positions: list[dict]) -> None:
        from tools.lookup_oracle import get_current_genotype

        # Sequence with wrong AA at position 1
        seq = list("A" * 100)
        seq[30] = "X"  # neither germline nor mature
        with pytest.raises(ValueError, match="neither germline"):
            get_current_genotype("".join(seq), sample_positions)


class TestApplyMutationToGenotype:
    """Tests for apply_mutation_to_genotype."""

    def test_flip_germline_to_mature(self, sample_positions: list[dict]) -> None:
        from tools.lookup_oracle import apply_mutation_to_genotype

        mutation = Mutation(chain="H", resnum="31", wt_aa="G", mut_aa="D")
        new = apply_mutation_to_genotype("000", mutation, sample_positions)
        assert new == "100"

    def test_flip_mature_to_germline(self, sample_positions: list[dict]) -> None:
        from tools.lookup_oracle import apply_mutation_to_genotype

        mutation = Mutation(chain="H", resnum="31", wt_aa="D", mut_aa="G")
        new = apply_mutation_to_genotype("100", mutation, sample_positions)
        assert new == "000"

    def test_invalid_position_raises(self, sample_positions: list[dict]) -> None:
        from tools.lookup_oracle import apply_mutation_to_genotype

        mutation = Mutation(chain="H", resnum="999", wt_aa="A", mut_aa="G")
        with pytest.raises(ValueError, match="not at one of the 16"):
            apply_mutation_to_genotype("000", mutation, sample_positions)

    def test_wrong_wt_raises(self, sample_positions: list[dict]) -> None:
        from tools.lookup_oracle import apply_mutation_to_genotype

        mutation = Mutation(chain="H", resnum="31", wt_aa="A", mut_aa="D")
        with pytest.raises(ValueError, match="Wild-type mismatch"):
            apply_mutation_to_genotype("000", mutation, sample_positions)


class TestComputeDdg:
    """Tests for compute_ddg."""

    def test_improved_binding(self) -> None:
        from tools.lookup_oracle import compute_ddg

        # Mutant has higher -logKd (tighter binding) → negative ddG
        ddg = compute_ddg(neg_log_kd_parent=8.0, neg_log_kd_mutant=9.0)
        expected = RT_LN10 * (8.0 - 9.0)
        assert ddg == pytest.approx(expected, abs=0.01)
        assert ddg < 0

    def test_worsened_binding(self) -> None:
        from tools.lookup_oracle import compute_ddg

        # Mutant has lower -logKd → positive ddG
        ddg = compute_ddg(neg_log_kd_parent=9.0, neg_log_kd_mutant=8.0)
        assert ddg > 0

    def test_lost_binding(self) -> None:
        from tools.lookup_oracle import compute_ddg

        ddg = compute_ddg(neg_log_kd_parent=9.0, neg_log_kd_mutant=KD_FLOOR)
        assert ddg == NON_BINDING_DDG

    def test_gained_binding(self) -> None:
        from tools.lookup_oracle import compute_ddg

        ddg = compute_ddg(neg_log_kd_parent=KD_FLOOR, neg_log_kd_mutant=8.0)
        assert ddg == -NON_BINDING_DDG

    def test_both_non_binding(self) -> None:
        from tools.lookup_oracle import compute_ddg

        ddg = compute_ddg(neg_log_kd_parent=KD_FLOOR, neg_log_kd_mutant=KD_FLOOR)
        assert ddg == 0.0


class TestLookupKd:
    """Tests for lookup_kd."""

    def test_valid_lookup(self, sample_variants: pd.DataFrame) -> None:
        from tools.lookup_oracle import lookup_kd

        kd = lookup_kd(sample_variants, "0" * 16, "h1_mean")
        assert kd == 7.0  # all-germline = 0 som_mut → 7.0 + 0*0.5

    def test_missing_genotype_raises(self, sample_variants: pd.DataFrame) -> None:
        from tools.lookup_oracle import lookup_kd

        with pytest.raises(ValueError, match="not found"):
            lookup_kd(sample_variants, "X" * 16, "h1_mean")


# -- Integration test ---------------------------------------------------------


class TestOracleIntegration:
    """Integration test: full oracle call from campaign state."""

    def test_full_oracle_call(
        self,
        mock_campaign: Path,
        cr9114_data_dir: Path,
        sample_positions: list[dict],
        sample_variants: pd.DataFrame,
    ) -> None:
        """Test a complete oracle invocation: load state, validate mutation, look up Kd."""
        from autoantibody.state import load_state
        from tools.lookup_oracle import (
            apply_mutation_to_genotype,
            compute_ddg,
            get_current_genotype,
            load_mutations_yaml,
            lookup_kd,
        )

        # Load everything
        state = load_state(mock_campaign)
        positions = load_mutations_yaml(cr9114_data_dir)

        # Get current genotype (all germline)
        genotype = get_current_genotype(state.parent.sequence_heavy, positions)
        assert genotype == "000"

        # Propose mutation: position 1, germline G → mature D
        mutation = Mutation(chain="H", resnum="31", wt_aa="G", mut_aa="D")
        new_genotype = apply_mutation_to_genotype(genotype, mutation, positions)
        assert new_genotype == "100"

        # Look up Kd values - pad genotypes to 16 chars for lookup
        parent_kd = lookup_kd(sample_variants, genotype.ljust(16, "0"), "h1_mean")
        mutant_kd = lookup_kd(sample_variants, new_genotype.ljust(16, "0"), "h1_mean")
        assert parent_kd == 7.0  # 0 som_mut
        assert mutant_kd == 7.5  # 1 som_mut

        # Compute ddG
        ddg = compute_ddg(parent_kd, mutant_kd)
        assert ddg < 0  # Improved binding

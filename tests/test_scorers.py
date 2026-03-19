"""Tests for scorer registry and availability checks."""

from __future__ import annotations

from autoantibody.models import ScorerTier
from autoantibody.scorers import (
    SCORER_REGISTRY,
    check_scorer_available,
    get_available_scorers,
    get_scorers_by_tier,
)


class TestScorerRegistry:
    def test_registry_has_all_scorers(self) -> None:
        expected = {
            "evoef",
            "stabddg",
            "graphinity",
            "baddg",
            "proteinmpnn_stability",
            "ablms",
            "flex_ddg",
            "atom_fep",
            "lookup_oracle",
        }
        assert set(SCORER_REGISTRY.keys()) == expected

    def test_each_scorer_has_required_fields(self) -> None:
        for name, info in SCORER_REGISTRY.items():
            assert info.name == name
            assert info.tier in ScorerTier
            assert info.script_path.suffix == ".py"
            assert isinstance(info.requires_gpu, bool)
            assert info.typical_seconds >= 0

    def test_tier_assignments(self) -> None:
        assert SCORER_REGISTRY["evoef"].tier == ScorerTier.FAST
        assert SCORER_REGISTRY["stabddg"].tier == ScorerTier.FAST
        assert SCORER_REGISTRY["graphinity"].tier == ScorerTier.FAST
        assert SCORER_REGISTRY["baddg"].tier == ScorerTier.MEDIUM
        assert SCORER_REGISTRY["proteinmpnn_stability"].tier == ScorerTier.FILTER
        assert SCORER_REGISTRY["ablms"].tier == ScorerTier.FILTER
        assert SCORER_REGISTRY["flex_ddg"].tier == ScorerTier.ORACLE
        assert SCORER_REGISTRY["atom_fep"].tier == ScorerTier.ORACLE
        assert SCORER_REGISTRY["lookup_oracle"].tier == ScorerTier.ORACLE


class TestScorerAvailability:
    def test_unknown_scorer_not_available(self) -> None:
        assert check_scorer_available("nonexistent") is False

    def test_get_available_returns_list(self) -> None:
        available = get_available_scorers()
        assert isinstance(available, list)
        for info in available:
            assert info.name in SCORER_REGISTRY

    def test_get_scorers_by_tier_fast(self) -> None:
        fast = get_scorers_by_tier(ScorerTier.FAST)
        assert len(fast) == 3
        names = {s.name for s in fast}
        assert names == {"evoef", "stabddg", "graphinity"}

    def test_get_scorers_by_tier_oracle(self) -> None:
        oracles = get_scorers_by_tier(ScorerTier.ORACLE)
        assert len(oracles) == 3
        names = {s.name for s in oracles}
        assert names == {"flex_ddg", "atom_fep", "lookup_oracle"}

    def test_get_scorers_by_tier_filter(self) -> None:
        filters = get_scorers_by_tier(ScorerTier.FILTER)
        assert len(filters) == 2
        names = {s.name for s in filters}
        assert names == {"proteinmpnn_stability", "ablms"}

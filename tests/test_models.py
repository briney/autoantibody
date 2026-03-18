"""Tests for data models."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoantibody.models import (
    Mutation,
    ScorerInfo,
    ScorerTier,
    ToolResult,
)


class TestMutation:
    def test_parse_simple(self) -> None:
        m = Mutation.parse("H:52:S:Y")
        assert m.chain == "H"
        assert m.resnum == "52"
        assert m.wt_aa == "S"
        assert m.mut_aa == "Y"

    def test_parse_insertion_code(self) -> None:
        m = Mutation.parse("H:100A:G:A")
        assert m.resnum == "100A"

    def test_str_roundtrip(self) -> None:
        original = "L:91:N:D"
        m = Mutation.parse(original)
        assert str(m) == original

    def test_invalid_amino_acid(self) -> None:
        with pytest.raises(ValueError, match="not a standard amino acid"):
            Mutation(chain="H", resnum="52", wt_aa="X", mut_aa="Y")

    def test_parse_wrong_field_count(self) -> None:
        with pytest.raises(ValueError, match="4 colon-separated"):
            Mutation.parse("H:52:S")

    def test_to_evoef(self) -> None:
        m = Mutation.parse("H:52:S:Y")
        assert m.to_evoef() == "SH52Y"

    def test_to_stabddg(self) -> None:
        m = Mutation.parse("H:52:S:Y")
        assert m.to_stabddg() == "SH52Y"

    def test_to_rosetta_resfile(self) -> None:
        m = Mutation.parse("H:52:S:Y")
        resfile = m.to_rosetta_resfile()
        assert "NATAA" in resfile
        assert "52 H PIKAA Y" in resfile

    def test_to_skempi(self) -> None:
        m = Mutation.parse("H:52:S:Y")
        assert m.to_skempi() == "SH52Y"

    def test_from_skempi(self) -> None:
        m = Mutation.from_skempi("SH52Y")
        assert m.chain == "H"
        assert m.resnum == "52"
        assert m.wt_aa == "S"
        assert m.mut_aa == "Y"

    def test_apply_to_sequence(self) -> None:
        seq = "ABCDEF"
        index_map = ["10", "11", "12", "13", "14", "15"]
        m = Mutation(chain="H", resnum="12", wt_aa="C", mut_aa="Y")
        result = m.apply_to_sequence(seq, index_map)
        assert result == "ABYDEF"

    def test_apply_to_sequence_wt_mismatch(self) -> None:
        seq = "ACDEFG"
        index_map = ["10", "11", "12", "13", "14", "15"]
        # Position 12 has 'E' but we claim 'W'
        m = Mutation(chain="H", resnum="12", wt_aa="W", mut_aa="Y")
        with pytest.raises(ValueError, match="Wild-type mismatch"):
            m.apply_to_sequence(seq, index_map)


class TestToolResult:
    def test_ok_result(self) -> None:
        r = ToolResult(status="ok", scores={"ddg": -1.2}, wall_time_s=3.5)
        assert r.status == "ok"
        assert r.scores["ddg"] == -1.2

    def test_error_result(self) -> None:
        r = ToolResult(status="error", error_message="EvoEF not found")
        assert r.error_message == "EvoEF not found"

    def test_scorer_name(self) -> None:
        r = ToolResult(
            status="ok",
            scores={"ddg": -0.5},
            scorer_name="evoef",
        )
        assert r.scorer_name == "evoef"

    def test_json_roundtrip(self) -> None:
        r = ToolResult(
            status="ok",
            scores={"ddg": -0.5},
            wall_time_s=1.0,
            scorer_name="graphinity",
        )
        r2 = ToolResult.model_validate_json(r.model_dump_json())
        assert r2.scores == r.scores
        assert r2.scorer_name == r.scorer_name


class TestScorerTier:
    def test_tier_values(self) -> None:
        assert ScorerTier.FAST == "fast"
        assert ScorerTier.MEDIUM == "medium"
        assert ScorerTier.ORACLE == "oracle"
        assert ScorerTier.FILTER == "filter"

    def test_tier_is_strenum(self) -> None:
        assert str(ScorerTier.FAST) == "fast"
        assert f"tier={ScorerTier.ORACLE}" == "tier=oracle"


class TestScorerInfo:
    def test_create_scorer_info(self) -> None:
        info = ScorerInfo(
            name="evoef",
            tier=ScorerTier.FAST,
            script_path=Path("tools/evoef_ddg.py"),
            requires_gpu=False,
            typical_seconds=5.0,
            description="EvoEF physics-based binding ddG",
        )
        assert info.name == "evoef"
        assert info.tier == ScorerTier.FAST
        assert not info.requires_gpu

    def test_scorer_info_defaults(self) -> None:
        info = ScorerInfo(
            name="test",
            tier=ScorerTier.FILTER,
            script_path=Path("tools/test.py"),
        )
        assert info.requires_gpu is False
        assert info.typical_seconds == 0.0
        assert info.description == ""

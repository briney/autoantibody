"""Integration tests for containerized scoring tools.

These tests exercise the autoantibody.container.score_mutation() path (which
builds Docker commands programmatically).  For tests that exercise the exact
user invocation path (``python tools/X.py``), see test_tools.py.

Run with: ``pytest -m container``
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoantibody.container import docker_available, docker_image_exists, score_mutation
from autoantibody.scorers import SCORER_REGISTRY

# Skip entire module if Docker is unavailable
pytestmark = [
    pytest.mark.slow,
    pytest.mark.container,
    pytest.mark.skipif(not docker_available(), reason="Docker not available"),
]


def _skip_if_no_image(scorer_name: str) -> None:
    """Skip test if the Docker image is not built."""
    info = SCORER_REGISTRY[scorer_name]
    if not info.docker_image or not docker_image_exists(info.docker_image):
        pytest.skip(f"Docker image not built: {info.docker_image}")


class TestEvoefContainer:
    def test_score_mutation(self, pdb_1n8z: Path) -> None:
        _skip_if_no_image("evoef")
        scorer = SCORER_REGISTRY["evoef"]
        result = score_mutation(scorer, pdb_1n8z, "B:52:S:Y")
        assert result.status == "ok"
        assert "ddg" in result.scores
        assert isinstance(result.scores["ddg"], float)


class TestStabddgContainer:
    def test_score_mutation(self, pdb_1n8z: Path) -> None:
        _skip_if_no_image("stabddg")
        scorer = SCORER_REGISTRY["stabddg"]
        result = score_mutation(
            scorer,
            pdb_1n8z,
            "B:52:S:Y",
            extra_args=["--chains", "AB_C"],
        )
        assert result.status == "ok"
        assert "ddg" in result.scores


class TestAblmsContainer:
    def test_score_mutation(self, pdb_1n8z: Path) -> None:
        _skip_if_no_image("ablms")
        scorer = SCORER_REGISTRY["ablms"]
        result = score_mutation(
            scorer,
            pdb_1n8z,
            "B:52:S:Y",
        )
        assert result.status == "ok"


class TestProteinmpnnStabilityContainer:
    def test_score_mutation(self, pdb_1n8z: Path) -> None:
        _skip_if_no_image("proteinmpnn_stability")
        scorer = SCORER_REGISTRY["proteinmpnn_stability"]
        result = score_mutation(
            scorer,
            pdb_1n8z,
            "B:52:S:Y",
            extra_args=["--chain", "B"],
        )
        assert result.status == "ok"
        assert "stability_ddg" in result.scores


class TestFlexDdgContainer:
    def test_score_mutation(self, pdb_1n8z: Path) -> None:
        _skip_if_no_image("flex_ddg")
        scorer = SCORER_REGISTRY["flex_ddg"]
        result = score_mutation(
            scorer,
            pdb_1n8z,
            "B:52:S:Y",
            extra_args=["--nstruct", "2", "--backrub-trials", "1000", "--relax-nstruct", "1"],
            timeout=600,
        )
        assert result.status == "ok"
        assert "ddg" in result.scores

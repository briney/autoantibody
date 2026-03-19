"""Unit tests for container.py — mock subprocess to avoid Docker dependency."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autoantibody.container import (
    _remap_path_arg,
    docker_available,
    docker_image_exists,
    run_containerized_tool,
    score_mutation,
)
from autoantibody.models import ScorerInfo, ScorerTier, ToolResult


@pytest.fixture
def mock_scorer() -> ScorerInfo:
    return ScorerInfo(
        name="test_scorer",
        tier=ScorerTier.FAST,
        script_path=Path("tools/test_tool.py"),
        requires_gpu=False,
        typical_seconds=5.0,
        docker_image="autoantibody/test:latest",
    )


@pytest.fixture
def gpu_scorer() -> ScorerInfo:
    return ScorerInfo(
        name="gpu_scorer",
        tier=ScorerTier.FAST,
        script_path=Path("tools/gpu_tool.py"),
        requires_gpu=True,
        typical_seconds=60.0,
        docker_image="autoantibody/gpu:latest",
    )


class TestDockerAvailable:
    @patch("autoantibody.container.shutil.which", return_value="/usr/bin/docker")
    @patch("autoantibody.container.subprocess.run")
    def test_available(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        assert docker_available() is True

    @patch("autoantibody.container.shutil.which", return_value=None)
    def test_not_installed(self, mock_which: MagicMock) -> None:
        assert docker_available() is False

    @patch("autoantibody.container.shutil.which", return_value="/usr/bin/docker")
    @patch("autoantibody.container.subprocess.run")
    def test_daemon_not_running(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1)
        assert docker_available() is False


class TestDockerImageExists:
    @patch("autoantibody.container.subprocess.run")
    def test_exists(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        assert docker_image_exists("autoantibody/evoef:latest") is True

    @patch("autoantibody.container.subprocess.run")
    def test_not_exists(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1)
        assert docker_image_exists("autoantibody/nonexistent:latest") is False


class TestRemapPathArg:
    def test_remap_file_in_workdir(self, tmp_path: Path) -> None:
        test_file = tmp_path / "input.pdb"
        test_file.touch()
        result = _remap_path_arg(str(test_file), tmp_path)
        assert result == "/workdir/input.pdb"

    def test_no_remap_outside_workdir(self, tmp_path: Path) -> None:
        result = _remap_path_arg("/some/other/path.pdb", tmp_path)
        assert result == "/some/other/path.pdb"

    def test_no_remap_relative_path(self, tmp_path: Path) -> None:
        result = _remap_path_arg("relative/path.pdb", tmp_path)
        assert result == "relative/path.pdb"

    def test_no_remap_mutation_string(self, tmp_path: Path) -> None:
        result = _remap_path_arg("H:52:S:Y", tmp_path)
        assert result == "H:52:S:Y"

    def test_remap_nested_file(self, tmp_path: Path) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        test_file = subdir / "data.csv"
        test_file.touch()
        result = _remap_path_arg(str(test_file), tmp_path)
        assert result == "/workdir/subdir/data.csv"


class TestRunContainerizedTool:
    def test_no_docker_image(self) -> None:
        scorer = ScorerInfo(
            name="no_image",
            tier=ScorerTier.FAST,
            script_path=Path("tools/test.py"),
            docker_image=None,
        )
        result = run_containerized_tool(scorer, [])
        assert result.status == "error"
        assert "no docker_image" in result.error_message

    @patch("autoantibody.container.subprocess.run")
    def test_successful_run(self, mock_run: MagicMock, mock_scorer: ScorerInfo) -> None:
        tool_result = ToolResult(
            status="ok",
            scores={"ddg": -1.5},
            scorer_name="test_scorer",
        )
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=tool_result.model_dump_json(),
            stderr="",
        )
        result = run_containerized_tool(mock_scorer, ["arg1", "arg2"])
        assert result.status == "ok"
        assert result.scores["ddg"] == -1.5

    @patch("autoantibody.container.subprocess.run")
    def test_container_failure(self, mock_run: MagicMock, mock_scorer: ScorerInfo) -> None:
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: something went wrong",
        )
        result = run_containerized_tool(mock_scorer, [])
        assert result.status == "error"
        assert "exit" in result.error_message.lower() or "code 1" in result.error_message

    @patch("autoantibody.container.subprocess.run")
    def test_malformed_json(self, mock_run: MagicMock, mock_scorer: ScorerInfo) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="not valid json {{{",
            stderr="",
        )
        result = run_containerized_tool(mock_scorer, [])
        assert result.status == "error"
        assert "no parseable JSON" in result.error_message.lower() or "JSON" in result.error_message

    @patch("autoantibody.container.subprocess.run")
    def test_empty_stdout(self, mock_run: MagicMock, mock_scorer: ScorerInfo) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr="",
        )
        result = run_containerized_tool(mock_scorer, [])
        assert result.status == "error"

    @patch("autoantibody.container.subprocess.run")
    def test_timeout(self, mock_run: MagicMock, mock_scorer: ScorerInfo) -> None:
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=30)
        result = run_containerized_tool(mock_scorer, [], timeout=30)
        assert result.status == "error"
        assert "timed out" in result.error_message.lower()

    @patch("autoantibody.container.subprocess.run")
    def test_gpu_flag_injected(self, mock_run: MagicMock, gpu_scorer: ScorerInfo) -> None:
        tool_result = ToolResult(status="ok", scores={"ddg": 0.5}, scorer_name="gpu_scorer")
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=tool_result.model_dump_json(),
            stderr="",
        )
        run_containerized_tool(gpu_scorer, [])
        call_args = mock_run.call_args[0][0]
        assert "--gpus" in call_args
        assert "all" in call_args

    @patch("autoantibody.container.subprocess.run")
    def test_no_gpu_flag_for_cpu(self, mock_run: MagicMock, mock_scorer: ScorerInfo) -> None:
        tool_result = ToolResult(status="ok", scores={}, scorer_name="test_scorer")
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=tool_result.model_dump_json(),
            stderr="",
        )
        run_containerized_tool(mock_scorer, [])
        call_args = mock_run.call_args[0][0]
        assert "--gpus" not in call_args


class TestScoreMutation:
    @patch("autoantibody.container.subprocess.run")
    def test_score_mutation(self, mock_run: MagicMock, mock_scorer: ScorerInfo) -> None:
        tool_result = ToolResult(
            status="ok",
            scores={"ddg": -2.3},
            scorer_name="test_scorer",
        )
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=tool_result.model_dump_json(),
            stderr="",
        )
        # Create a dummy PDB file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            f.write(b"ATOM      1  CA  ALA A   1       0.0  0.0  0.0  1.00  0.00\n")
            pdb_path = Path(f.name)

        try:
            result = score_mutation(mock_scorer, pdb_path, "H:52:S:Y")
            assert result.status == "ok"
            assert result.scores["ddg"] == -2.3

            # Verify docker run was called with correct args
            call_args = mock_run.call_args[0][0]
            assert "docker" in call_args
            assert mock_scorer.docker_image in call_args
        finally:
            pdb_path.unlink(missing_ok=True)

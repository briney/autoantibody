"""Uniform Docker container invocation for scoring tools."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from autoantibody.models import ScorerInfo, ToolResult

logger = logging.getLogger(__name__)

_WORKDIR_MOUNT = "/workdir"


def docker_available() -> bool:
    """Check whether Docker is installed and the daemon is running."""
    if not shutil.which("docker"):
        return False
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def docker_image_exists(image: str) -> bool:
    """Check whether a Docker image is available locally."""
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=10,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _remap_path_arg(arg: str, workdir: Path) -> str:
    """If arg is a path to a file inside the temp workdir, remap to container path."""
    try:
        p = Path(arg)
        if p.is_absolute() and str(p).startswith(str(workdir)):
            rel = p.relative_to(workdir)
            return f"{_WORKDIR_MOUNT}/{rel}"
    except (ValueError, OSError):
        pass
    return arg


def run_containerized_tool(
    scorer: ScorerInfo,
    args: list[str],
    input_files: dict[str, Path] | None = None,
    timeout: float | None = None,
    extra_env: dict[str, str] | None = None,
) -> ToolResult:
    """Run a scoring tool inside its Docker container.

    Args:
        scorer: Scorer metadata (must have docker_image set).
        args: Command-line arguments to pass to the tool script.
        input_files: Mapping of {container_filename: host_path} to copy into workdir.
        timeout: Override timeout in seconds; defaults to container_timeout or 3x typical_seconds.
        extra_env: Additional environment variables to set inside the container.

    Returns:
        Parsed ToolResult from the tool's stdout JSON.
    """
    if not scorer.docker_image:
        return ToolResult(
            status="error",
            error_message=f"Scorer '{scorer.name}' has no docker_image configured",
            scorer_name=scorer.name,
        )

    if timeout is None:
        timeout = scorer.container_timeout or (scorer.typical_seconds * 3)
    timeout = max(timeout, 30.0)  # minimum 30s

    t0 = time.monotonic()
    tmpdir = tempfile.mkdtemp(prefix=f"{scorer.name}_container_")
    workdir = Path(tmpdir)

    try:
        # Copy input files into temp workdir
        if input_files:
            for container_name, host_path in input_files.items():
                dest = workdir / container_name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(host_path, dest)

        # Remap file-path arguments to container paths
        remapped_args = [_remap_path_arg(a, workdir) for a in args]

        # Build docker command
        cmd: list[str] = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{workdir}:{_WORKDIR_MOUNT}",
        ]

        # GPU support
        if scorer.requires_gpu:
            cmd.extend(["--gpus", "all"])

        # Environment variables
        if extra_env:
            for key, val in extra_env.items():
                cmd.extend(["-e", f"{key}={val}"])

        # Image and tool command
        cmd.append(scorer.docker_image)
        cmd.extend(["python", f"/app/{scorer.script_path}", *remapped_args])

        logger.info("Running: %s", " ".join(cmd))

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Try to parse ToolResult JSON from stdout
        stdout = proc.stdout.strip()
        if stdout:
            try:
                data = json.loads(stdout)
                result = ToolResult.model_validate(data)
                result.wall_time_s = round(time.monotonic() - t0, 2)
                return result
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Failed to parse tool output as JSON: %s", e)

        # If we got here, either no output or unparseable
        if proc.returncode != 0:
            stderr_tail = proc.stderr[-500:] if proc.stderr else ""
            return ToolResult(
                status="error",
                error_message=(
                    f"Container exited with code {proc.returncode}. stderr: {stderr_tail}"
                ),
                wall_time_s=round(time.monotonic() - t0, 2),
                scorer_name=scorer.name,
            )

        return ToolResult(
            status="error",
            error_message="Tool produced no parseable JSON output",
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name=scorer.name,
        )

    except subprocess.TimeoutExpired:
        return ToolResult(
            status="error",
            error_message=f"Container timed out after {timeout:.0f}s",
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name=scorer.name,
        )
    except Exception as e:
        return ToolResult(
            status="error",
            error_message=str(e),
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name=scorer.name,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def score_mutation(
    scorer: ScorerInfo,
    pdb_path: Path,
    mutation_str: str,
    extra_args: list[str] | None = None,
    timeout: float | None = None,
) -> ToolResult:
    """Convenience wrapper: score a single mutation against a PDB structure.

    Args:
        scorer: Scorer metadata.
        pdb_path: Path to input PDB file.
        mutation_str: Mutation in chain:resnum:wt:mut format (e.g., "H:52:S:Y").
        extra_args: Additional CLI arguments for the tool script.
        timeout: Override timeout in seconds.

    Returns:
        ToolResult from the containerized tool.
    """
    pdb_name = pdb_path.name
    input_files = {pdb_name: pdb_path}
    args = [f"{_WORKDIR_MOUNT}/{pdb_name}", mutation_str]
    if extra_args:
        args.extend(extra_args)

    return run_containerized_tool(
        scorer=scorer,
        args=args,
        input_files=input_files,
        timeout=timeout,
    )

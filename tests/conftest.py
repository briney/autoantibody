"""Shared test fixtures."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

# Add project root to sys.path so `from tools.xxx import ...` works in tests.
# Also add tools/ so that `from _common import ...` resolves inside tool scripts.
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_tools_dir = _project_root / "tools"
if str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))

DATA_DIR = Path(__file__).parent / "data"
PDB_1N8Z = DATA_DIR / "1N8Z.pdb"
PDB_URL = "https://files.rcsb.org/download/1N8Z.pdb"

# 1N8Z chain IDs: A = light chain, B = heavy chain, C = antigen (HER2 ECD)
HEAVY_CHAIN = "B"
LIGHT_CHAIN = "A"
ANTIGEN_CHAINS = ["C"]


@pytest.fixture(scope="session")
def pdb_1n8z() -> Path:
    """Provide PDB 1N8Z, downloading if necessary."""
    if not PDB_1N8Z.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PDB_URL, PDB_1N8Z)
    return PDB_1N8Z


@pytest.fixture
def tmp_campaign(tmp_path: Path, pdb_1n8z: Path) -> Path:
    """Create a temporary campaign directory with 1N8Z."""
    from autoantibody.state import init_campaign

    campaign_dir = tmp_path / "test_campaign"
    init_campaign(
        campaign_dir=campaign_dir,
        pdb_path=pdb_1n8z,
        antibody_heavy_chain=HEAVY_CHAIN,
        antibody_light_chain=LIGHT_CHAIN,
        antigen_chains=ANTIGEN_CHAINS,
        campaign_id="test_cmp_001",
    )
    return campaign_dir


# ---------------------------------------------------------------------------
# Container helpers
# ---------------------------------------------------------------------------


def _docker_available() -> bool:
    """Check whether the Docker daemon is reachable."""
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _docker_image_exists(image: str) -> bool:
    """Check whether a Docker image is available locally."""
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=10,
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def skip_unless_container(image: str) -> None:
    """pytest.skip() if Docker is unavailable or *image* is not built."""
    if not _docker_available():
        pytest.skip("Docker not available")
    if not _docker_image_exists(image):
        pytest.skip(f"Docker image not built: {image}")


def run_tool_cli(
    script: str,
    args: list[str],
    timeout: int = 120,
) -> dict:
    """Run a tool script as a subprocess and return parsed JSON output.

    The call does NOT set AUTOANTIBODY_CONTAINER, so auto-containerization
    will trigger Docker if the tool is configured for it.  This exercises the
    exact same code path a user would follow.
    """
    proc = subprocess.run(
        [sys.executable, script, *args],
        capture_output=True,
        text=True,
        cwd=_project_root,
        timeout=timeout,
    )
    return json.loads(proc.stdout)

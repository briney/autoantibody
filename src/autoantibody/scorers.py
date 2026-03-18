"""Scorer registry and availability checks."""

from __future__ import annotations

import importlib
import os
import shutil
from pathlib import Path

from autoantibody.models import ScorerInfo, ScorerTier

SCORER_REGISTRY: dict[str, ScorerInfo] = {
    "evoef": ScorerInfo(
        name="evoef",
        tier=ScorerTier.FAST,
        script_path=Path("tools/evoef_ddg.py"),
        requires_gpu=False,
        typical_seconds=5.0,
        description="EvoEF physics-based binding ddG (~5s CPU, PCC 0.53 on SKEMPI2)",
    ),
    "stabddg": ScorerInfo(
        name="stabddg",
        tier=ScorerTier.FAST,
        script_path=Path("tools/stabddg_score.py"),
        requires_gpu=True,
        typical_seconds=60.0,
        description="StaB-ddG ML binding ddG (GPU, Spearman 0.45 on SKEMPI2)",
    ),
    "graphinity": ScorerInfo(
        name="graphinity",
        tier=ScorerTier.FAST,
        script_path=Path("tools/graphinity_score.py"),
        requires_gpu=True,
        typical_seconds=10.0,
        description=(
            "Graphinity equivariant GNN for Ab-Ag binding ddG (GPU, Pearson ~0.87 on SKEMPI2)"
        ),
    ),
    "baddg": ScorerInfo(
        name="baddg",
        tier=ScorerTier.MEDIUM,
        script_path=Path("tools/baddg_score.py"),
        requires_gpu=True,
        typical_seconds=180.0,
        description="BA-ddG Boltzmann-averaged ML binding ddG (GPU, Spearman 0.51)",
    ),
    "proteinmpnn_stability": ScorerInfo(
        name="proteinmpnn_stability",
        tier=ScorerTier.FILTER,
        script_path=Path("tools/stability_check.py"),
        requires_gpu=False,
        typical_seconds=30.0,
        description=(
            "ProteinMPNN-ddG fold stability filter (Docker). "
            "NOT a binding ddG scorer — rejects fold-destabilizing mutations."
        ),
    ),
    "ablms": ScorerInfo(
        name="ablms",
        tier=ScorerTier.FILTER,
        script_path=Path("tools/ablms_score.py"),
        requires_gpu=True,
        typical_seconds=10.0,
        description="Antibody language model sequence plausibility scoring",
    ),
    "flex_ddg": ScorerInfo(
        name="flex_ddg",
        tier=ScorerTier.ORACLE,
        script_path=Path("tools/flex_ddg.py"),
        requires_gpu=False,
        typical_seconds=2400.0,
        description="Rosetta flex-ddG oracle via Docker (30-60 min, PCC ~0.46)",
    ),
    "atom_fep": ScorerInfo(
        name="atom_fep",
        tier=ScorerTier.ORACLE,
        script_path=Path("tools/atom_ddg.py"),
        requires_gpu=True,
        typical_seconds=18000.0,
        description="AToM-OpenMM alchemical FEP oracle (4-6 hrs GPU, RMSE ~1.0-1.5)",
    ),
}


def check_scorer_available(name: str) -> bool:
    """Check if a scorer's dependencies are available.

    Checks environment variables, binaries, importable packages, and/or Docker
    images as appropriate for each scorer.
    """
    if name not in SCORER_REGISTRY:
        return False

    match name:
        case "evoef":
            env = os.environ.get("EVOEF_BINARY")
            if env:
                return Path(env).exists()
            return shutil.which("evoef") is not None

        case "stabddg":
            try:
                importlib.import_module("stabddg")
                return True
            except ImportError:
                return False

        case "graphinity":
            try:
                importlib.import_module("graphinity")
                return True
            except ImportError:
                return False

        case "baddg":
            baddg_dir = os.environ.get("BADDG_DIR")
            if baddg_dir:
                return Path(baddg_dir).is_dir()
            return Path("/opt/baddg").is_dir()

        case "proteinmpnn_stability":
            return shutil.which("docker") is not None

        case "ablms":
            try:
                importlib.import_module("ablms")
                return True
            except ImportError:
                return False

        case "flex_ddg":
            return shutil.which("docker") is not None

        case "atom_fep":
            try:
                importlib.import_module("openmm")
                importlib.import_module("openmmtools")
                return True
            except ImportError:
                return False

        case _:
            return False


def get_available_scorers() -> list[ScorerInfo]:
    """Return only scorers whose dependencies are installed."""
    return [info for name, info in SCORER_REGISTRY.items() if check_scorer_available(name)]


def get_scorers_by_tier(tier: ScorerTier) -> list[ScorerInfo]:
    """Return all registered scorers in a given tier."""
    return [info for info in SCORER_REGISTRY.values() if info.tier == tier]

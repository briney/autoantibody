"""Scorer registry and availability checks."""

from __future__ import annotations

import importlib
import os
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
        docker_image="autoantibody/evoef:latest",
    ),
    "stabddg": ScorerInfo(
        name="stabddg",
        tier=ScorerTier.FAST,
        script_path=Path("tools/stabddg_score.py"),
        requires_gpu=True,
        typical_seconds=60.0,
        description="StaB-ddG ML binding ddG (GPU, Spearman 0.45 on SKEMPI2)",
        docker_image="autoantibody/stabddg:latest",
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
        docker_image="autoantibody/graphinity:latest",
    ),
    "baddg": ScorerInfo(
        name="baddg",
        tier=ScorerTier.MEDIUM,
        script_path=Path("tools/baddg_score.py"),
        requires_gpu=True,
        typical_seconds=180.0,
        description="BA-ddG Boltzmann-averaged ML binding ddG (GPU, Spearman 0.51)",
        docker_image="autoantibody/baddg:latest",
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
        docker_image="autoantibody/proteinmpnn_stability:latest",
    ),
    "ablms": ScorerInfo(
        name="ablms",
        tier=ScorerTier.FILTER,
        script_path=Path("tools/ablms_score.py"),
        requires_gpu=True,
        typical_seconds=10.0,
        description="Antibody language model sequence plausibility scoring",
        docker_image="autoantibody/ablms:latest",
    ),
    "flex_ddg": ScorerInfo(
        name="flex_ddg",
        tier=ScorerTier.ORACLE,
        script_path=Path("tools/flex_ddg.py"),
        requires_gpu=False,
        typical_seconds=2400.0,
        description="Rosetta flex-ddG oracle via Docker (30-60 min, PCC ~0.46)",
        docker_image="autoantibody/flex_ddg:latest",
    ),
    "atom_fep": ScorerInfo(
        name="atom_fep",
        tier=ScorerTier.ORACLE,
        script_path=Path("tools/atom_ddg.py"),
        requires_gpu=True,
        typical_seconds=18000.0,
        description="AToM-OpenMM alchemical FEP oracle (4-6 hrs GPU, RMSE ~1.0-1.5)",
        docker_image="autoantibody/atom_fep:latest",
    ),
    "lookup_oracle": ScorerInfo(
        name="lookup_oracle",
        tier=ScorerTier.ORACLE,
        script_path=Path("tools/lookup_oracle.py"),
        requires_gpu=False,
        typical_seconds=0.1,
        description="Experimental Kd lookup oracle for CR9114 benchmark (instant)",
        containerized=False,
        docker_image=None,
    ),
}


def check_scorer_available(name: str) -> bool:
    """Check if a scorer's dependencies are available.

    For containerized tools, checks that Docker is available and the image exists.
    For non-containerized tools (lookup_oracle), falls back to legacy checks.
    """
    if name not in SCORER_REGISTRY:
        return False

    info = SCORER_REGISTRY[name]

    if info.containerized and info.docker_image:
        from autoantibody.container import docker_available, docker_image_exists

        return docker_available() and docker_image_exists(info.docker_image)

    # Non-containerized tools: legacy checks
    match name:
        case "lookup_oracle":
            data_dir = Path(os.environ.get("CR9114_DATA_DIR", "data/cr9114"))
            return (data_dir / "variants.parquet").exists()

        case _:
            # Fallback for any future non-containerized tool: try importing
            try:
                importlib.import_module(name)
                return True
            except ImportError:
                return False


def get_available_scorers() -> list[ScorerInfo]:
    """Return only scorers whose dependencies are installed."""
    return [info for name, info in SCORER_REGISTRY.items() if check_scorer_available(name)]


def get_scorers_by_tier(tier: ScorerTier) -> list[ScorerInfo]:
    """Return all registered scorers in a given tier."""
    return [info for info in SCORER_REGISTRY.values() if info.tier == tier]

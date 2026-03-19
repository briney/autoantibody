#!/usr/bin/env python
"""AToM-OpenMM alchemical FEP oracle for binding ddG.

Usage:
    python tools/atom_ddg.py <pdb_path> <mutation> [options]

Arguments:
    pdb_path: Path to the input PDB structure.
    mutation:  Mutation string (e.g., H:52:S:Y).

Options:
    --lambda-windows N   Number of lambda windows (default: 24).
    --gpu-id ID          GPU device ID (default: 0).
    --steps-per-window N MD steps per lambda window (default: 50000).
    --keep-workdir       Don't delete working directory on completion.

Output (stdout):
    JSON with status, scores.ddg, scores.ddg_uncertainty,
    scorer_name="atom_fep", wall_time_s.

Requires:
    openmm, openmmtools, ambertools (conda),
    AToM-OpenMM (pip), R + UWHAM package.

Example:
    python tools/atom_ddg.py runs/cmp_001/input/1N8Z.pdb H:52:S:Y \
        --lambda-windows 24 --gpu-id 0

Note: 4-6 hours per mutation on GPU. Agent uses this only when high confidence
is needed, or to calibrate cheaper scorers.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from _common import Mutation, ToolResult, validate_mutation_against_structure


def prepare_system(
    pdb_path: Path,
    mutation: Mutation,
    work_dir: Path,
) -> Path:
    """Prepare the alchemical system using AmberTools tleap.

    Parameterizes both wild-type and mutant, solvates, and generates
    topology/coordinate files.

    Args:
        pdb_path: Path to input PDB structure.
        mutation: The mutation to model.
        work_dir: Working directory for intermediate files.

    Returns:
        Path to the prepared system directory.
    """
    system_dir = work_dir / "system"
    system_dir.mkdir()

    tleap_input = f"""\
source leaprc.protein.ff14SB
source leaprc.water.tip3p
mol = loadpdb {pdb_path}
solvatebox mol TIP3PBOX 12.0
saveamberparm mol {system_dir}/complex.prmtop {system_dir}/complex.inpcrd
quit
"""
    tleap_file = work_dir / "tleap.in"
    tleap_file.write_text(tleap_input)

    result = subprocess.run(
        ["tleap", "-f", str(tleap_file)],
        capture_output=True,
        text=True,
        cwd=str(work_dir),
    )
    if result.returncode != 0:
        raise RuntimeError(f"tleap failed:\n{result.stderr}")

    prmtop = system_dir / "complex.prmtop"
    if not prmtop.exists():
        raise FileNotFoundError(f"tleap did not produce {prmtop.name}")

    return system_dir


def setup_alchemy(
    system_dir: Path,
    mutation: Mutation,
    lambda_windows: int,
    work_dir: Path,
) -> Path:
    """Set up alchemical lambda schedule for wt->mut transformation.

    Creates an evenly-spaced lambda schedule and writes the AToM config
    JSON referencing the prepared system topology/coordinates.

    Args:
        system_dir: Path to the prepared system directory (from prepare_system).
        mutation: The mutation being modeled.
        lambda_windows: Number of lambda windows for the alchemical schedule.
        work_dir: Working directory for intermediate files.

    Returns:
        Path to the alchemy config directory.
    """
    alchemy_dir = work_dir / "alchemy"
    alchemy_dir.mkdir()

    lambdas = [i / (lambda_windows - 1) for i in range(lambda_windows)]
    schedule_file = alchemy_dir / "lambda_schedule.dat"
    schedule_file.write_text("\n".join(f"{lam:.6f}" for lam in lambdas))

    config = {
        "topology": str(system_dir / "complex.prmtop"),
        "coordinates": str(system_dir / "complex.inpcrd"),
        "mutation": mutation.to_skempi(),
        "lambda_schedule": str(schedule_file),
        "n_windows": lambda_windows,
    }
    (alchemy_dir / "config.json").write_text(json.dumps(config, indent=2))

    return alchemy_dir


def run_md(
    alchemy_dir: Path,
    gpu_id: int,
    steps_per_window: int,
    work_dir: Path,
) -> Path:
    """Run replica exchange MD via AToM-OpenMM.

    Invokes AToM's REMD runner as a subprocess, which manages GPU-accelerated
    replica exchange across lambda windows.

    Args:
        alchemy_dir: Path to alchemy config directory (from setup_alchemy).
        gpu_id: GPU device ID to use for simulation.
        steps_per_window: Number of MD steps per lambda window.
        work_dir: Working directory for output files.

    Returns:
        Path to the MD output directory containing energy samples.
    """
    md_dir = work_dir / "md_output"
    md_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "AToM.run_remd",
            "--config",
            str(alchemy_dir / "config.json"),
            "--gpu-id",
            str(gpu_id),
            "--steps",
            str(steps_per_window),
            "--output",
            str(md_dir),
        ],
        capture_output=True,
        text=True,
        timeout=36000,  # 10-hour hard timeout
    )

    if result.returncode != 0:
        raise RuntimeError(f"AToM MD failed:\n{result.stderr}")

    energies_file = md_dir / "energies.dat"
    if not energies_file.exists():
        raise FileNotFoundError("AToM MD did not produce energies.dat")

    return md_dir


def analyze_fep(md_dir: Path, work_dir: Path) -> tuple[float, float]:
    """Run UWHAM free energy analysis on MD energy samples.

    Uses R with the UWHAM package to compute free energy differences
    from the replica exchange energy samples.

    Args:
        md_dir: Path to MD output directory containing energies.dat.
        work_dir: Working directory for the R script.

    Returns:
        Tuple of (ddg, ddg_uncertainty) in kcal/mol.
    """
    r_script = work_dir / "uwham_analysis.R"
    r_script.write_text(f"""\
library(UWHAM)
energies <- read.table("{md_dir}/energies.dat", header=TRUE)
# UWHAM analysis to compute free energy differences
result <- uwham(energies$lambda, energies$reduced_potential)
ddg <- result$free_energy[length(result$free_energy)] - result$free_energy[1]
ddg_err <- sqrt(result$variance[length(result$variance)])
cat(ddg, ddg_err, sep="\\n")
""")

    result = subprocess.run(
        ["Rscript", str(r_script)],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(f"UWHAM analysis failed:\n{result.stderr}")

    lines = result.stdout.strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Expected 2 lines from UWHAM (ddg, uncertainty), got {len(lines)}")
    ddg = float(lines[0])
    ddg_uncertainty = float(lines[1])
    return ddg, ddg_uncertainty


def main() -> None:
    ap = argparse.ArgumentParser(description="AToM-OpenMM FEP oracle")
    ap.add_argument("pdb_path", type=Path)
    ap.add_argument("mutation", type=str)
    ap.add_argument("--lambda-windows", type=int, default=24)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--steps-per-window", type=int, default=50000)
    ap.add_argument("--keep-workdir", action="store_true")
    args = ap.parse_args()

    pdb_path = args.pdb_path.resolve()
    mutation = Mutation.parse(args.mutation)

    t0 = time.monotonic()
    tmpdir_obj = tempfile.TemporaryDirectory(prefix="atom_fep_")
    wd = Path(tmpdir_obj.name)

    try:
        errors = validate_mutation_against_structure(pdb_path, mutation)
        if errors:
            result = ToolResult(
                status="error",
                error_message="; ".join(errors),
                scorer_name="atom_fep",
            )
            print(result.to_json())
            sys.exit(1)

        shutil.copy2(pdb_path, wd / pdb_path.name)

        system_dir = prepare_system(pdb_path, mutation, wd)
        alchemy_dir = setup_alchemy(system_dir, mutation, args.lambda_windows, wd)
        md_dir = run_md(alchemy_dir, args.gpu_id, args.steps_per_window, wd)
        ddg, ddg_uncertainty = analyze_fep(md_dir, wd)

        result = ToolResult(
            status="ok",
            scores={
                "ddg": round(ddg, 4),
                "ddg_uncertainty": round(ddg_uncertainty, 4),
            },
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="atom_fep",
        )
    except SystemExit:
        raise
    except Exception as e:
        result = ToolResult(
            status="error",
            error_message=str(e),
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="atom_fep",
        )
    finally:
        if not args.keep_workdir:
            tmpdir_obj.cleanup()
        else:
            print(f"Work directory preserved: {wd}", file=sys.stderr)

    print(result.to_json())
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()

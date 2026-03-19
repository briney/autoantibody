#!/usr/bin/env python
"""Rosetta flex-ddG oracle scorer via Docker.

Usage:
    python tools/flex_ddg.py <pdb_path> <mutation> [options]

Arguments:
    pdb_path: Path to input PDB structure.
    mutation:  Mutation string (e.g., H:52:S:Y).

Options:
    --nstruct N          Backrub structures to generate (default: 35)
    --backrub-trials N   Backrub MC trials per structure (default: 35000)
    --relax-nstruct N    Relaxation structures to generate (default: 5)
    --docker-image IMG   Docker image (default: rosettacommons/rosetta)
    --rosetta-bin BIN    Binary name in container
                         (default: rosetta_scripts.default.linuxgccrelease)
    --keep-workdir       Don't delete working directory on completion

Output (stdout):
    JSON with status, scores (ddg, ddg_std, n_structures), wall_time_s.

Example:
    # Fast test run (5 structures, 10000 trials)
    python tools/flex_ddg.py input/1N8Z.pdb H:52:S:Y \
        --nstruct 5 --backrub-trials 10000

    # Production run (default parameters, ~30-60 min)
    python tools/flex_ddg.py input/1N8Z.pdb H:52:S:Y
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from _common import Mutation, ToolResult, validate_mutation_against_structure

# -- Rosetta XML protocols (embedded) ----------------------------------------

RELAX_FLAGS = """\
-relax:constrain_relax_to_start_coords
-relax:coord_constrain_sidechains
-relax:ramp_constraints false
-ex1
-ex2
-use_input_sc
-flip_HNQ
-no_optH false
"""

RELAX_XML = """\
<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="ref2015" weights="ref2015"/>
  </SCOREFXNS>
  <MOVERS>
    <FastRelax name="relax" scorefxn="ref2015" repeats="5"/>
  </MOVERS>
  <PROTOCOLS>
    <Add mover="relax"/>
  </PROTOCOLS>
</ROSETTASCRIPTS>
"""

BACKRUB_XML = """\
<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="ref2015" weights="ref2015"/>
  </SCOREFXNS>
  <MOVERS>
    <Backrub name="backrub" ntrials="%%ntrials%%"/>
  </MOVERS>
  <PROTOCOLS>
    <Add mover="backrub"/>
  </PROTOCOLS>
</ROSETTASCRIPTS>
"""

REPACK_SCORE_XML = """\
<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="ref2015" weights="ref2015"/>
    <ScoreFunction name="ref2015_soft" weights="ref2015_soft"/>
  </SCOREFXNS>
  <TASKOPERATIONS>
    <InitializeFromCommandline name="ifcl"/>
    <ReadResfile name="resfile" filename="%%resfile_path%%"/>
    <IncludeCurrent name="ic"/>
    <ExtraRotamersGeneric name="ex12" ex1="1" ex2="1"/>
  </TASKOPERATIONS>
  <MOVERS>
    <PackRotamersMover name="repack" scorefxn="ref2015_soft"
                       task_operations="ifcl,resfile,ic,ex12"/>
    <MinMover name="minimize" scorefxn="ref2015" chi="1" bb="0"
              jump="ALL" type="lbfgs_armijo_nonmonotone"
              tolerance="0.01" max_iter="5000"/>
    <InterfaceAnalyzerMover name="analyze" scorefxn="ref2015"
                            packstat="0" pack_separated="0"
                            pack_input="0" interface_sc="1"
                            tracer="0"/>
  </MOVERS>
  <PROTOCOLS>
    <Add mover="repack"/>
    <Add mover="minimize"/>
    <Add mover="analyze"/>
  </PROTOCOLS>
</ROSETTASCRIPTS>
"""


def generate_driver_script(
    nstruct: int,
    backrub_trials: int,
    relax_nstruct: int,
    rosetta_bin: str,
) -> str:
    """Generate the bash driver script that runs inside Docker.

    Steps:
    1. Relax input structure, keep lowest-energy result.
    2. For each of nstruct backrub structures:
       a. Generate backrub ensemble member.
       b. Repack + score wild-type interface (NATAA resfile).
       c. Apply mutation resfile, repack + score mutant interface.
    3. Write per-structure scores to CSV.
    """
    # Use doubled braces for bash variables that shouldn't be Python-interpolated
    return f"""\
#!/bin/bash
set -euo pipefail

WORK=/workdir
ROSETTA={rosetta_bin}

echo "=== Step 1: Relax ==="
mkdir -p $WORK/relax_pdbs
$ROSETTA \\
  -s $WORK/input.pdb \\
  -parser:protocol $WORK/relax.xml \\
  @$WORK/relax.flags \\
  -nstruct {relax_nstruct} \\
  -out:path:pdb $WORK/relax_pdbs/ \\
  -out:path:score $WORK/ \\
  -out:file:scorefile relax_scores.sc \\
  > $WORK/logs/relax.log 2>&1

# Find lowest-energy relaxed structure
BEST_TAG=$(grep '^SCORE:' $WORK/relax_scores.sc \\
  | grep -v description \\
  | sort -k2 -n \\
  | head -1 \\
  | awk '{{print $NF}}')
BEST_PDB=$(find $WORK/relax_pdbs -name "*${{BEST_TAG}}*" -type f | head -1)
if [ -z "$BEST_PDB" ]; then
  BEST_PDB=$(ls $WORK/relax_pdbs/*.pdb | head -1)
fi
echo "Best relaxed: $BEST_PDB"

# Initialize results
echo "idx,wt_dG_separated,mut_dG_separated,ddg" > $WORK/results.csv

echo "=== Step 2: Backrub + scoring ==="
for i in $(seq 1 {nstruct}); do
  echo "--- Structure $i/{nstruct} ---"
  D=$WORK/iter_$i
  mkdir -p $D

  # Backrub
  $ROSETTA \\
    -s $BEST_PDB \\
    -parser:protocol $WORK/backrub.xml \\
    -parser:script_vars ntrials={backrub_trials} \\
    -nstruct 1 \\
    -out:prefix br${{i}}_ \\
    -out:path:pdb $D/ \\
    -out:path:score $D/ \\
    > $D/backrub.log 2>&1

  BR_PDB=$(ls $D/br${{i}}_*.pdb 2>/dev/null | head -1)
  if [ -z "$BR_PDB" ]; then
    echo "  WARN: no backrub PDB produced, skipping"
    continue
  fi

  # Wild-type repack + interface score
  $ROSETTA \\
    -s $BR_PDB \\
    -parser:protocol $WORK/repack_score.xml \\
    -parser:script_vars resfile_path=$WORK/wt.resfile \\
    -nstruct 1 \\
    -out:prefix wt${{i}}_ \\
    -out:path:pdb $D/ \\
    -out:path:score $D/ \\
    -out:file:scorefile wt.sc \\
    > $D/wt.log 2>&1

  # Mutant repack + interface score
  $ROSETTA \\
    -s $BR_PDB \\
    -parser:protocol $WORK/repack_score.xml \\
    -parser:script_vars resfile_path=$WORK/mut.resfile \\
    -nstruct 1 \\
    -out:prefix mut${{i}}_ \\
    -out:path:pdb $D/ \\
    -out:path:score $D/ \\
    -out:file:scorefile mut.sc \\
    > $D/mut.log 2>&1

  # Extract dG_separated column
  WT_DG=$(awk '
    NR==1 {{for(i=1;i<=NF;i++) if($i=="dG_separated") c=i}}
    NR==2 && c {{print $c}}
  ' $D/wt.sc)
  MUT_DG=$(awk '
    NR==1 {{for(i=1;i<=NF;i++) if($i=="dG_separated") c=i}}
    NR==2 && c {{print $c}}
  ' $D/mut.sc)

  if [ -n "$WT_DG" ] && [ -n "$MUT_DG" ]; then
    DDG=$(echo "$MUT_DG - $WT_DG" | bc -l)
    echo "$i,$WT_DG,$MUT_DG,$DDG" >> $WORK/results.csv
    echo "  wt_dG=$WT_DG  mut_dG=$MUT_DG  ddG=$DDG"
  else
    echo "  WARN: could not extract dG_separated"
  fi
done

echo "=== Complete ==="
"""


def setup_workdir(work_dir: Path, pdb_path: Path, mutation: Mutation) -> None:
    """Write all input files to the working directory."""
    (work_dir / "logs").mkdir()
    shutil.copy2(pdb_path, work_dir / "input.pdb")
    (work_dir / "relax.xml").write_text(RELAX_XML)
    (work_dir / "relax.flags").write_text(RELAX_FLAGS)
    (work_dir / "backrub.xml").write_text(BACKRUB_XML)
    (work_dir / "repack_score.xml").write_text(REPACK_SCORE_XML)
    (work_dir / "wt.resfile").write_text("NATAA\nstart\n")
    (work_dir / "mut.resfile").write_text(mutation.to_rosetta_resfile())


def parse_results_csv(results_csv: Path) -> dict[str, float]:
    """Parse the flex-ddG results CSV and return aggregate ddG."""
    ddg_values: list[float] = []
    with open(results_csv) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                try:
                    ddg_values.append(float(parts[3]))
                except ValueError:
                    continue
    if not ddg_values:
        raise ValueError("No valid ddG values in results CSV")
    return {
        "ddg": round(float(np.mean(ddg_values)), 4),
        "ddg_std": round(float(np.std(ddg_values)), 4),
        "n_structures": float(len(ddg_values)),
    }


def run_flex_ddg(
    pdb_path: Path,
    mutation: Mutation,
    work_dir: Path,
    nstruct: int = 35,
    backrub_trials: int = 35000,
    relax_nstruct: int = 5,
    docker_image: str = "rosettacommons/rosetta",
    rosetta_bin: str = "rosetta_scripts.default.linuxgccrelease",
) -> dict[str, float]:
    """Run flex-ddG via Docker and return ddG scores."""
    setup_workdir(work_dir, pdb_path, mutation)

    driver = generate_driver_script(
        nstruct=nstruct,
        backrub_trials=backrub_trials,
        relax_nstruct=relax_nstruct,
        rosetta_bin=rosetta_bin,
    )
    driver_path = work_dir / "run.sh"
    driver_path.write_text(driver)
    driver_path.chmod(0o755)

    if os.environ.get("AUTOANTIBODY_CONTAINER"):
        # Inside container: run driver script directly
        proc = subprocess.run(
            ["bash", str(driver_path)],
            capture_output=True,
            text=True,
            timeout=7200,
        )
    else:
        # Legacy: call via docker run
        proc = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{work_dir}:/workdir",
                docker_image,
                "bash",
                "/workdir/run.sh",
            ],
            capture_output=True,
            text=True,
            timeout=7200,  # 2-hour hard timeout
        )

    # Save Docker logs regardless of outcome
    (work_dir / "logs" / "docker_stdout.log").write_text(proc.stdout)
    (work_dir / "logs" / "docker_stderr.log").write_text(proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError(
            f"flex-ddG Docker failed (exit {proc.returncode}). See {work_dir}/logs/ for details."
        )

    results_csv = work_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"No results.csv produced. See {work_dir}/logs/")
    return parse_results_csv(results_csv)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rosetta flex-ddG oracle")
    ap.add_argument("pdb_path", type=Path)
    ap.add_argument("mutation", type=str)
    ap.add_argument("--nstruct", type=int, default=35)
    ap.add_argument("--backrub-trials", type=int, default=35000)
    ap.add_argument("--relax-nstruct", type=int, default=5)
    ap.add_argument("--docker-image", default="rosettacommons/rosetta")
    ap.add_argument(
        "--rosetta-bin",
        default="rosetta_scripts.default.linuxgccrelease",
    )
    ap.add_argument("--keep-workdir", action="store_true")
    args = ap.parse_args()

    pdb_path = args.pdb_path.resolve()
    mutation = Mutation.parse(args.mutation)

    t0 = time.monotonic()
    tmpdir_obj = tempfile.TemporaryDirectory(prefix="flex_ddg_")
    work_dir = Path(tmpdir_obj.name)

    try:
        errors = validate_mutation_against_structure(pdb_path, mutation)
        if errors:
            r = ToolResult(
                status="error",
                error_message="; ".join(errors),
                scorer_name="flex_ddg",
            )
            print(r.to_json())
            sys.exit(1)

        scores = run_flex_ddg(
            pdb_path,
            mutation,
            work_dir,
            nstruct=args.nstruct,
            backrub_trials=args.backrub_trials,
            relax_nstruct=args.relax_nstruct,
            docker_image=args.docker_image,
            rosetta_bin=args.rosetta_bin,
        )
        result = ToolResult(
            status="ok",
            scores=scores,
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="flex_ddg",
        )
    except SystemExit:
        raise
    except Exception as e:
        result = ToolResult(
            status="error",
            error_message=str(e),
            wall_time_s=round(time.monotonic() - t0, 2),
            scorer_name="flex_ddg",
        )
    finally:
        if not args.keep_workdir:
            tmpdir_obj.cleanup()
        else:
            print(f"Work directory preserved: {work_dir}", file=sys.stderr)

    print(result.to_json())
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()

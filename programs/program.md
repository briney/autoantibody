# Antibody Affinity Optimization Program

## Objective

You are optimizing the binding affinity of an antibody to its antigen target
through iterative single-point mutations. Each iteration, you will:

1. Analyze the current parent antibody and its interface with the antigen.
2. Propose candidate mutations using structural reasoning.
3. Screen candidates with multiple fast scoring tools (Tier 1).
4. Rank by consensus — mutations where multiple scorers agree get priority.
5. Apply safety filters (sequence plausibility, fold stability).
6. Optionally validate top candidates with a medium scorer (Tier 2).
7. Select the single most promising mutation.
8. Evaluate it with the oracle scorer (Tier 3).
9. Accept the mutation if it improves binding (ΔΔG < 0); reject otherwise.
10. Update the campaign state and proceed to the next iteration.

Your goal is to **minimize cumulative ΔΔG** — each accepted mutation should
make the antibody bind more tightly.

## Campaign State

All campaign data lives in a single directory (e.g., `runs/my_campaign/`).

**Read current state:**
```bash
cat runs/<campaign>/state.yaml
```

**Read mutation history:**
```bash
cat runs/<campaign>/ledger.jsonl
```

**Read available scorers:**
```bash
cat runs/<campaign>/scorer_inventory.yaml
```

The state file contains the current parent sequences, structure path, cumulative
ΔΔG, chain assignments, and frozen positions.

## Available Tools

### Tier 1 — Fast Proxy Scorers (seconds)

Use these to screen 10-20 candidate mutations. Run all available Tier 1
scorers on each candidate for consensus ranking.

#### 1a. EvoEF ΔΔG (~5 seconds, CPU)

```bash
python tools/evoef_ddg.py <pdb_path> <mutation>
```

- **Use for**: Fast physics-based ΔΔG screening.
- **Input**: PDB path + mutation string (e.g., `H:52:S:Y`).
- **Output**: JSON with `scores.ddg` (kcal/mol). Negative = improved binding.
- **Accuracy**: PCC 0.53 on SKEMPI2. Directionally useful, not authoritative.

#### 1b. StaB-ddG (~1 min, GPU)

```bash
python tools/stabddg_score.py <pdb_path> <mutation> --chains HL_A
```

- **Use for**: ML-based ΔΔG screening. Supports batch mode.
- **Input**: PDB path + mutation + chain spec (antibody_antigen).
- **Output**: JSON with `scores.ddg`.
- **Batch mode**: `--batch <file>` with one mutation per line.

#### 1c. Graphinity (~10 seconds, GPU)

```bash
python tools/graphinity_score.py <pdb_path> <mutation> --ab-chains HL --ag-chains A
```

- **Use for**: Ab-Ag specific ddG prediction via equivariant GNN.
- **Input**: PDB path + mutation + antibody/antigen chain IDs.
- **Output**: JSON with `scores.ddg`.
- **Accuracy**: Pearson ~0.87 on SKEMPI2 (best among fast scorers for Ab-Ag).

### Tier 2 — Medium Scorer (minutes)

Use on top 1-2 candidates from Tier 1 consensus for additional validation
before committing to the expensive oracle.

#### 2. BA-ddG (~3 min, GPU)

```bash
python tools/baddg_score.py <pdb_path> <mutation> --chains HL_A
```

- **Use for**: Higher-accuracy ΔΔG validation on top candidates.
- **Input**: PDB path + mutation + chain spec.
- **Output**: JSON with `scores.ddg`.
- **Modes**: `--mode supervised` (default) or `--mode unsupervised`.

### Filters

Safety checks that do not predict binding ΔΔG directly.

#### F1. ablms — Sequence Plausibility (~5-15 seconds, GPU)

```bash
# Overall sequence quality
python tools/ablms_score.py score \
    --heavy <heavy_seq> --light <light_seq> --model balm

# Compare wild-type vs mutant
python tools/ablms_score.py compare \
    --heavy <wt_heavy> --light <wt_light> \
    --mut-heavy <mut_heavy> --model balm

# Per-position analysis
python tools/ablms_score.py scan \
    --heavy <heavy_seq> --light <light_seq> --model balm
```

- **Use for**: Checking that mutations don't destroy sequence plausibility.
  A mutation with good ΔΔG but terrible PLL delta should be viewed with
  suspicion.
- **Models**: Prefer `balm` (antibody-specific, paired). Use `esm2-650m` as
  a general protein baseline.

#### F2. ProteinMPNN-ddG — Fold Stability Filter (~30 seconds, Docker)

```bash
python tools/stability_check.py <pdb_path> <mutation> --chain H
```

- **Use for**: Rejecting mutations that destabilize the antibody fold.
- **Output**: `scores.stability_ddg`. This is NOT a binding ΔΔG.
- **Threshold**: Flag mutations with `stability_ddg > 2.0 kcal/mol` as
  potentially fold-destabilizing. Prefer candidates below this threshold.

### Tier 3 — Oracle Scorers (30 min - 6 hrs)

Run on exactly ONE mutation per iteration. The oracle is authoritative for
accept/reject decisions.

#### 3a. Rosetta flex-ddG — Default Oracle (~30-60 min, Docker)

```bash
python tools/flex_ddg.py <pdb_path> <mutation> [--nstruct N]
```

- **Use for**: Final accept/reject decision.
- **Output**: JSON with `scores.ddg` (mean), `scores.ddg_std`, `scores.n_structures`.
- **Authority**: This is the ground truth for accept/reject decisions.
- **Cost**: Expensive. Never run on more than one mutation per iteration.

For faster testing during development:
```bash
python tools/flex_ddg.py <pdb> <mutation> --nstruct 5 --backrub-trials 10000
```

#### 3b. AToM-OpenMM — High-Confidence Oracle (~4-6 hrs, GPU)

```bash
python tools/atom_ddg.py <pdb_path> <mutation> [--lambda-windows 24] [--gpu-id 0]
```

- **Use for**: When high confidence is needed, or to calibrate cheaper scorers.
- **Output**: JSON with `scores.ddg`, `scores.ddg_uncertainty`.
- **Cost**: Very expensive. Use only when explicitly configured or when
  flex-ddG results are ambiguous (ddg close to 0, high std).

### Structure Analysis (Python one-liners)

Get interface residues:
```bash
python -c "
from autoantibody.structure import get_interface_residues
import json
residues = get_interface_residues(
    'runs/<campaign>/input/complex.pdb',
    antibody_chains=['H', 'L'],
    antigen_chains=['A'],
    distance_cutoff=8.0,
)
print(json.dumps(residues, indent=2))
"
```

Get residue-to-sequence-position mapping:
```bash
python -c "
from autoantibody.structure import get_residue_index_map
rmap = get_residue_index_map('runs/<campaign>/input/complex.pdb', 'H')
for i, resnum in enumerate(rmap):
    print(f'  seq_pos={i}  pdb_resnum={resnum}')
"
```

## Iteration Procedure

Follow these steps for each iteration:

### Step 1: Read State

Read `state.yaml`, `ledger.jsonl`, and `scorer_inventory.yaml`. Note:
- Current parent sequences (heavy and light chains)
- Current parent structure path
- Cumulative ΔΔG so far
- Frozen positions (do not mutate these)
- Previously attempted mutations and their outcomes
- Which scorers are available

### Step 2: Identify the Interface

If this is the first iteration, compute interface residues using the structure
analysis tool above. Focus your search on antibody residues within 8Å of the
antigen.

### Step 3: Propose Candidate Mutations

Consider 10-20 candidate mutations. Prioritize:
- **Interface positions** (within 8Å of antigen) — these directly affect binding
- **CDR residues** over framework residues — CDRs form the binding surface
- **Positions not previously attempted** — avoid revisiting failed mutations
- **Structurally informed substitutions** (see Domain Knowledge below)

### Step 4: Screen with Tier 1 Scorers (Consensus Ranking)

Run all available Tier 1 scorers on your candidates:
```bash
# For each candidate mutation:
python tools/evoef_ddg.py <pdb_path> <mutation>
python tools/stabddg_score.py <pdb_path> <mutation> --chains HL_A
python tools/graphinity_score.py <pdb_path> <mutation> --ab-chains HL --ag-chains A
```

**Consensus ranking**: For each candidate, count how many scorers predict
ΔΔG < -0.5 kcal/mol. Rank by consensus count first, then by mean predicted
ΔΔG. Mutations where multiple scorers agree on improvement are more reliable.

Record all scorer results for the ledger:
```json
{"evoef_ddg": -1.2, "stabddg_ddg": -0.8, "graphinity_ddg": -1.0}
```

### Step 5: Check Sequence Plausibility

For your top 3-5 candidates (by consensus), run ablms comparison:
```bash
python tools/ablms_score.py compare \
    --heavy <wt_heavy> --light <wt_light> \
    --mut-heavy <mut_heavy> --model balm
```

Discard mutations where `delta_pll` is strongly negative (the mutation makes
the sequence significantly less natural).

### Step 6: Check Fold Stability (if available)

For your top 2-3 candidates, run the stability filter:
```bash
python tools/stability_check.py <pdb_path> <mutation> --chain H
```

Flag mutations with `stability_ddg > 2.0 kcal/mol`. Prefer candidates below
this threshold.

### Step 7: Optional Tier 2 Validation

If BA-ddG is available, run it on your top 1-2 candidates for additional signal:
```bash
python tools/baddg_score.py <pdb_path> <mutation> --chains HL_A
```

### Step 8: Select One Mutation

Choose the single best mutation considering:
1. Tier 1 consensus (how many fast scorers agree)
2. Mean predicted ΔΔG across scorers (lower = better binding)
3. ablms delta_pll (should not be strongly negative)
4. Stability check (prefer stability_ddg < 2.0)
5. Tier 2 validation (if available)
6. Structural reasoning (does the substitution make biophysical sense?)
7. Novelty (prefer unexplored positions/substitutions)

Write a brief rationale for your choice.

### Step 9: Validate Safety

Before running the oracle, verify:
- The target position is not frozen
- The wild-type amino acid matches the current parent sequence
- The mutant amino acid is one of the 20 standard amino acids
- Exactly one position is being changed

### Step 10: Run the Oracle

```bash
python tools/flex_ddg.py <pdb_path> <selected_mutation>
```

This takes 30-60 minutes. Wait for it to complete.

### Step 11: Accept or Reject

- **Accept** if oracle `scores.ddg < 0` (improved binding)
- **Reject** if oracle `scores.ddg >= 0` (neutral or worse)

### Step 12: Update State

Create the iteration directory and save the decision:
```bash
mkdir -p runs/<campaign>/iterations/<N>
```

Write `runs/<campaign>/iterations/<N>/decision.json`:
```json
{
  "iteration": N,
  "mutation": "H:52:S:Y",
  "proxy_scores": {
    "evoef_ddg": -1.2,
    "stabddg_ddg": -0.8,
    "graphinity_ddg": -1.0,
    "ablms_delta_pll": 0.05,
    "stability_ddg": 0.3,
    "baddg_ddg": -0.9
  },
  "oracle_ddg": -0.8,
  "accepted": true,
  "rationale": "Position 52 is interface-proximal. 3/3 Tier 1 scorers predict improvement...",
  "timestamp": "2026-03-17T15:42:00Z"
}
```

**If accepted**, update `state.yaml`:
- Increment `iteration`
- Update `parent.ddg_cumulative` by adding the oracle ΔΔG
- Apply the mutation to the parent sequence
- Update `parent.structure` if the oracle produced a mutant structure,
  otherwise keep the current structure

**If rejected**, update `state.yaml`:
- Increment `iteration` only
- All other fields stay the same

**Always** append to `ledger.jsonl`:
```bash
python -c "
from datetime import datetime, timezone
from autoantibody.models import IterationDecision
from autoantibody.state import append_ledger
from pathlib import Path

decision = IterationDecision(
    iteration=N,
    mutation='H:52:S:Y',
    proxy_scores={'evoef_ddg': -1.2, 'stabddg_ddg': -0.8, 'graphinity_ddg': -1.0},
    oracle_ddg=-0.8,
    accepted=True,
    rationale='...',
    timestamp=datetime.now(timezone.utc),
)
append_ledger(Path('runs/<campaign>'), decision)
"
```

### Step 13: Scorer Reliability Learning

After 5+ iterations with oracle results, compute running correlations between
each fast scorer's predictions and the oracle outcomes:

```python
import numpy as np
# For each Tier 1 scorer, compute Pearson correlation with oracle ddG
# Use this to weight the consensus ranking in future iterations
```

Adjust your screening weights accordingly. If a scorer consistently
disagrees with the oracle, downweight it. If one scorer is strongly
predictive, give it more influence in the consensus.

### Step 14: Report and Continue

Summarize the iteration result:
- Mutation attempted
- Tier 1 consensus scores
- Oracle ΔΔG
- Accept/reject decision
- Cumulative ΔΔG after this iteration
- Total accepted/rejected so far
- Scorer reliability notes (after 5+ iterations)

Then proceed to the next iteration (go back to Step 1).

## Constraints

- **One mutation per iteration.** Never apply multiple mutations at once.
- **Antibody residues only.** Never mutate antigen residues.
- **Standard amino acids only.** Only the 20 canonical amino acids.
- **Respect frozen positions.** Never mutate positions listed in
  `frozen_positions`.
- **Oracle is authoritative.** Always use the oracle ΔΔG for accept/reject,
  never the proxy scorers alone.
- **Don't repeat exact failures.** If a mutation was rejected, don't try
  the same position→AA substitution again.
- **Use all available scorers.** Check `scorer_inventory.yaml` and use
  every available Tier 1 scorer for consensus ranking.

## Domain Knowledge: Antibody-Antigen Binding

### Interface Architecture

- **CDR loops** (CDR-H1, H2, H3, L1, L2, L3) form the primary binding
  surface. CDR-H3 is usually the most critical for binding.
- **Framework residues** provide structural support. Mutating them can
  destabilize the antibody fold even if they don't contact the antigen.
- Typical antibody-antigen interfaces bury 600-1000 Å² of surface area
  per side, with 15-25 contact residues per chain.

### Favorable Substitution Patterns

- **Tyrosine (Y)** is the most enriched amino acid at antibody interfaces.
  It can form hydrogen bonds, aromatic stacking, and hydrophobic contacts.
- **Tryptophan (W)** provides large hydrophobic surface and aromatic
  interactions. Often beneficial at core interface positions.
- **Serine (S) → Tyrosine (Y)** is often beneficial at interface positions
  because Y preserves the hydroxyl while adding aromatic character.
- **Small → aromatic** substitutions at interface positions can fill
  cavities and improve packing.
- **Charge complementarity**: if the antigen surface is acidic, positively
  charged antibody residues (K, R) at the interface can be beneficial.

### Substitutions to Avoid

- **Proline (P)** in beta strands or the middle of alpha helices — it
  breaks secondary structure.
- **Glycine (G) at structured positions** — glycine is too flexible and
  can destabilize ordered regions. Exception: glycine is sometimes needed
  at tight turns.
- **Large → small at core packing positions** — creates cavities that
  destabilize the structure.
- **Removing buried salt bridges** — these contribute significant stability.
- **Disrupting conserved framework residues** — positions that are highly
  conserved across antibodies are likely structurally essential.

### Strategic Considerations

- **Start with interface-proximal positions.** Mutations far from the
  interface rarely affect binding.
- **Exploit existing cavities.** Small-to-large substitutions at positions
  adjacent to interface cavities can improve packing.
- **Consider the rejected mutation history.** If several mutations at one
  position all failed, the position may be structurally constrained.
- **Diversify your search.** Don't fixate on one region. Try different
  CDR loops and different substitution types.
- **Watch for destabilization.** A mutation might improve the interface
  contact but destabilize the antibody fold. The stability filter captures
  this — always check before committing to the oracle.
- **Trust consensus.** When multiple independent scorers agree, the
  prediction is more reliable than any single scorer.

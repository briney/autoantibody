# CR9114 Affinity Maturation Benchmark

## Objective

You are benchmarking the autoresearch antibody optimization loop using the
CR9114 deep mutational scanning dataset (Phillips et al., eLife 2021).

**Your task**: Starting from the germline VH1-69 heavy chain, rediscover the
affinity maturation path to mature CR9114 using scoring tools — without knowing
which mutations improve binding.

The CR9114 antibody broadly neutralizes influenza by binding the conserved HA
stalk. The dataset provides experimental Kd measurements for all 65,536
combinations of germline/mature amino acids at 16 heavy-chain positions. The
lookup oracle replaces flex-ddG, giving instant experimental ground truth
(~0.1 seconds vs 30-60 minutes).

**What you know**: The 16 positions and their germline/mature amino acid
identities (see table below). You must use your scoring tools and structural
reasoning to decide which mutations to make.

**What you don't know**: Which direction (germline→mature or staying at
germline) improves binding at each position. The tools must guide you.

## The 16 Mutation Positions

Read the exact positions from `data/cr9114/mutations.yaml`. The file contains:
```yaml
positions:
  - index: 1
    chain: H
    pdb_resnum: "<resnum>"
    germline_aa: "<aa>"
    mature_aa: "<aa>"
    seq_position: <int>
  # ... 16 entries
```

For each position, the mutation string format is:
`<chain>:<pdb_resnum>:<germline_aa>:<mature_aa>`

Example: if position 1 is chain H, resnum 31, germline G, mature D,
the mutation string is `H:31:G:D`.

## Campaign State

All campaign data lives in the campaign directory (e.g., `runs/cr9114_h1_001/`).

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

## Available Tools

### Tier 1 — Fast Proxy Scorers (seconds)

Use these to screen candidate mutations at the 16 positions. Run all available
Tier 1 scorers on each candidate for consensus ranking.

#### 1a. EvoEF ΔΔG (~5 seconds, CPU)

```bash
python tools/evoef_ddg.py <pdb_path> <mutation>
```

- **Input**: PDB path + mutation string (e.g., `H:31:G:D`).
- **Output**: JSON with `scores.ddg` (kcal/mol). Negative = improved binding.

#### 1b. StaB-ddG (~1 min, GPU)

```bash
python tools/stabddg_score.py <pdb_path> <mutation> --chains HL_AB
```

- **Input**: PDB path + mutation + chain spec (antibody_antigen).
- **Note**: Chain spec is `HL_AB` (heavy H + light L, antigen A+B).

#### 1c. Graphinity (~10 seconds, GPU)

```bash
python tools/graphinity_score.py <pdb_path> <mutation> --ab-chains HL --ag-chains AB
```

- **Input**: PDB path + mutation + antibody/antigen chain IDs.
- **Note**: Ab chains are `HL`, Ag chains are `AB`.

### Filters

#### F1. ablms — Sequence Plausibility

```bash
python tools/ablms_score.py compare \
    --heavy <wt_heavy> --light <wt_light> \
    --mut-heavy <mut_heavy> --model balm
```

#### F2. ProteinMPNN-ddG — Fold Stability Filter

```bash
python tools/stability_check.py <pdb_path> <mutation> --chain H
```

### Oracle — Lookup Oracle (instant)

```bash
python tools/lookup_oracle.py <campaign_dir> <mutation> [--antigen H1]
```

- **Use for**: Final accept/reject decision. Instant experimental Kd lookup.
- **Output**: JSON with `scores.ddg`, `scores.neg_log_kd_parent`,
  `scores.neg_log_kd_mutant`.
- **Cost**: ~0.1 seconds. Much cheaper than flex-ddG.
- **Authority**: This is experimental ground truth. Always authoritative.

### Structure Analysis

```bash
python -c "
from autoantibody.structure import get_interface_residues
import json
residues = get_interface_residues(
    'runs/<campaign>/input/4FQI_clean.pdb',
    antibody_chains=['H', 'L'],
    antigen_chains=['A', 'B'],
    distance_cutoff=8.0,
)
print(json.dumps(residues, indent=2))
"
```

## Iteration Procedure

### Step 1: Read State

Read `state.yaml`, `ledger.jsonl`, `scorer_inventory.yaml`, and
`data/cr9114/mutations.yaml`. Note:
- Current parent heavy chain sequence
- Which of the 16 positions have already been flipped to mature
- Which positions remain at germline (candidates for this iteration)
- Previously attempted mutations and outcomes
- Which scorers are available

### Step 2: Identify Remaining Candidates

List the positions still at germline AA. These are your candidates. If a
position was already flipped to mature (accepted), it's no longer a candidate.
If a germline→mature mutation at a position was rejected, that position
can be skipped (the germline AA is better for that position).

### Step 3: Screen with Tier 1 Scorers (Consensus Ranking)

For each remaining candidate position, construct the germline→mature mutation
string and run all available Tier 1 scorers:

```bash
# For each candidate mutation at position i:
python tools/evoef_ddg.py <pdb_path> <mutation_i>
python tools/stabddg_score.py <pdb_path> <mutation_i> --chains HL_AB
python tools/graphinity_score.py <pdb_path> <mutation_i> --ab-chains HL --ag-chains AB
```

**Consensus ranking**: Count how many scorers predict ΔΔG < -0.5 kcal/mol.
Rank by consensus count first, then by mean predicted ΔΔG.

### Step 4: Check Sequence Plausibility

For your top 3-5 candidates (by consensus), run ablms comparison:
```bash
python tools/ablms_score.py compare \
    --heavy <wt_heavy> --light <wt_light> \
    --mut-heavy <mut_heavy> --model balm
```

### Step 5: Check Fold Stability (if available)

For your top 2-3 candidates:
```bash
python tools/stability_check.py <pdb_path> <mutation> --chain H
```

### Step 6: Select One Mutation

Choose the best mutation considering:
1. Tier 1 consensus count
2. Mean predicted ΔΔG (lower = better)
3. ablms delta_pll (not strongly negative)
4. Stability check (prefer stability_ddg < 2.0)
5. Structural reasoning (does the substitution make biophysical sense?)
6. Interface proximity (from Step 2 in first iteration)

### Step 7: Run the Oracle

```bash
python tools/lookup_oracle.py runs/<campaign> <selected_mutation> --antigen H1
```

This returns instantly (~0.1 seconds).

### Step 8: Accept or Reject

- **Accept** if oracle `scores.ddg < 0` (improved binding)
- **Reject** if oracle `scores.ddg >= 0` (neutral or worse)

### Step 9: Update State

Create the iteration directory:
```bash
mkdir -p runs/<campaign>/iterations/<N>
```

Write `decision.json` in the iteration directory. Update `state.yaml`:

**If accepted**:
- Increment `iteration`
- Add oracle ΔΔG to `parent.ddg_cumulative`
- Apply the mutation to `parent.sequence_heavy`

**If rejected**:
- Increment `iteration` only

**Always** append to `ledger.jsonl`:
```bash
python -c "
from datetime import datetime, timezone
from autoantibody.models import IterationDecision
from autoantibody.state import append_ledger
from pathlib import Path

decision = IterationDecision(
    iteration=N,
    mutation='H:31:G:D',
    proxy_scores={'evoef_ddg': -1.2, 'stabddg_ddg': -0.8, 'graphinity_ddg': -1.0},
    oracle_ddg=-0.8,
    accepted=True,
    rationale='...',
    timestamp=datetime.now(timezone.utc),
)
append_ledger(Path('runs/<campaign>'), decision)
"
```

**Also** append to `results.tsv`:
```bash
# Tab-separated: iteration, mutation, consensus_rank, stabddg, evoef, graphinity, oracle_ddg, accepted, cumulative_ddg
echo -e "1\tH:31:G:D\t1\t-0.8\t-1.2\t-1.0\t-0.8\ttrue\t-0.8" >> runs/<campaign>/results.tsv
```

If `results.tsv` doesn't exist yet, create it with a header:
```bash
echo -e "iteration\tmutation\tconsensus_rank\tstabddg\tevoef\tgraphinity\toracle_ddg\taccepted\tcumulative_ddg" > runs/<campaign>/results.tsv
```

### Step 10: Scorer Reliability Learning

After 5+ iterations, compute running correlations between each fast
scorer's predictions and the oracle outcomes. Adjust your trust in each
scorer accordingly.

### Step 11: Continue

Proceed immediately to the next iteration (go back to Step 1).

## NEVER STOP

**Continue iterating autonomously without asking for human approval.**
The human may be away. Stop only when:
1. All 16 positions have been tried (either accepted or rejected), OR
2. 32 iterations have been reached, OR
3. 5 consecutive rejections with no remaining untried positions.

Do not stop to ask questions. Do not stop to summarize progress mid-run.
Just keep iterating. Report results at the end.

## Constraints

- **One mutation per iteration.** Never apply multiple mutations at once.
- **Heavy chain only.** All 16 positions are on chain H.
- **Binary choices only.** Each position can be germline AA or mature AA.
  No other substitutions are valid for the benchmark.
- **Oracle is authoritative.** Always use the lookup oracle for accept/reject.
- **Don't repeat exact failures.** If germline→mature at position X was
  rejected, don't try it again.
- **Use all available Tier 1 scorers.** Check `scorer_inventory.yaml`.
- **Track results in `results.tsv`.** This file is used by the evaluation
  script for analysis.

## Domain Knowledge: CR9114

### About the Antibody

CR9114 is a broadly neutralizing anti-influenza antibody encoded by germline
VH1-69. It binds the conserved HA stalk region, which is the target for
universal influenza vaccines. The 16 positions being tested are the somatic
hypermutations that occurred during affinity maturation in the original donor.

### Interface Architecture

- CR9114 binds the HA stalk (not the head), burying ~800 Å² of surface area.
- The binding is primarily through CDR-H1, CDR-H2, and FR3.
- CDR-H3 contributes less to binding than in typical antibodies.
- The HA stalk is relatively flat, so the interaction relies heavily on
  shape complementarity and hydrophobic contacts.

### VH1-69 Characteristics

- VH1-69 germline is enriched for hydrophobic CDR-H2 residues (IFPN motif).
- Many broadly neutralizing anti-stalk antibodies use VH1-69.
- The germline already has moderate affinity for HA stalk — not starting
  from zero.
- Affinity maturation fine-tunes contacts, often through conservative
  substitutions that improve packing.

### Expected Patterns

- **Most germline→mature mutations should improve H1 binding** (the antibody
  was matured against influenza). But some positions may be neutral or even
  slightly unfavorable for H1 specifically.
- **CDR positions** are more likely to affect binding than framework positions.
- **Interface-proximal positions** are more likely to affect binding.
- **~97% of variants in this library bind H1** (very permissive landscape),
  but binding to H3 and Flu B is much more restrictive.

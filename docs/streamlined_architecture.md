# Streamlined MVP: Autonomous Antibody Affinity Optimization

## Purpose

Test whether an LLM agent can iteratively optimize antibody binding affinity
by choosing point mutations informed by computational tools.

This is the minimum viable version. The goal is to close the loop end-to-end on
one real antibody-antigen complex and learn whether the concept works before
investing in infrastructure.

## Philosophy

Inspired by Karpathy's `autoresearch`:

- **Claude Code is the agent.** No separate agent runtime, no API orchestration
  layer. You open Claude Code in this repo, point it at `programs/program.md`,
  and it runs the optimization loop interactively.
- **One file programs the agent.** `programs/program.md` defines the objective,
  decision rubric, available tools, and reporting format. You iterate on the
  research policy by editing this markdown file, not by writing orchestration
  code.
- **Tools are plain Python scripts.** Each tool is a thin wrapper that takes
  inputs on the command line, runs a computation, and prints JSON to stdout.
  No Docker containers, no manifests, no runner abstraction.
- **State is flat files.** A YAML file tracks the current parent. A JSONL file
  logs the mutation ledger. That's it — no SQLite, no Parquet, no event
  streams.
- **Ship the loop, not the platform.** Everything that isn't required to close
  the propose-evaluate-accept/reject loop is deferred.

## Core Loop

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  1. Read current parent (sequence + structure)  │
│                          │                      │
│                          ▼                      │
│  2. Use fast tools to identify promising        │
│     candidate mutations                         │
│                          │                      │
│                          ▼                      │
│  3. Select one mutation + write rationale        │
│                          │                      │
│                          ▼                      │
│  4. Run oracle (flex-ddG) on selected mutation  │
│                          │                      │
│                          ▼                      │
│  5. Accept if ΔΔG < 0 (improved binding)        │
│     Reject otherwise                            │
│                          │                      │
│                          ▼                      │
│  6. Update state, log result, repeat            │
│                                                 │
└─────────────────────────────────────────────────┘
```

Each iteration produces exactly one mutation decision: accepted or rejected.
The agent (Claude Code) drives the entire loop, calling tools as needed and
recording results.

## How It Works in Practice

1. The user opens Claude Code in the `autoantibody/` repo.
2. The user tells Claude Code to read `programs/program.md` and begin an
   optimization campaign on a specified antibody-antigen complex.
3. Claude Code follows the program:
   - reads the current campaign state from `runs/<campaign>/state.yaml`
   - calls tool wrapper scripts via bash to gather evidence about candidate
     mutations
   - reasons about the evidence and selects a single mutation
   - calls the oracle wrapper to evaluate the selected mutation
   - updates the state file and mutation ledger
   - reports the result and moves to the next iteration
4. The user can interrupt at any time, inspect the ledger, adjust
   `programs/program.md`, and resume.

Claude Code's context window and tool-calling capabilities serve as the agent
runtime. No additional orchestration code is needed for the MVP.

## Repository Structure

```
autoantibody/
├── programs/
│   └── program.md              # Agent behavior program (human-edited)
├── tools/
│   ├── foldx_ddg.py            # Fast proxy scorer
│   ├── flex_ddg.py             # Oracle scorer
│   ├── esm_score.py            # Sequence log-likelihood scoring
│   └── structure_utils.py      # Shared helpers (mutate PDB, extract interface)
├── src/autoantibody/
│   ├── __init__.py
│   ├── models.py               # Pydantic data models (minimal set)
│   └── state.py                # State read/write helpers
├── configs/
│   └── example_campaign.yaml   # Example campaign config
├── tests/
├── runs/                       # Generated campaign data (gitignored)
│   └── <campaign_id>/
│       ├── state.yaml          # Current parent + iteration count
│       ├── ledger.jsonl         # Append-only mutation log
│       ├── input/              # Original complex structure
│       └── iterations/         # Per-iteration artifacts
│           └── <N>/
│               ├── tool_outputs/
│               └── decision.json
└── docs/
```

## Tool Wrappers

Each tool is a standalone Python script. Claude Code calls them via bash and
reads the JSON output. The interface contract is minimal:

**Input:** command-line arguments (structure path, mutation specification, etc.)

**Output:** JSON printed to stdout with at minimum:

```json
{
  "status": "ok",
  "scores": { ... },
  "artifacts": { ... }
}
```

**Error:** non-zero exit code + error message on stderr.

### MVP Tool Set

The minimum tool set to test the concept:

| Tool | Purpose | Speed | Required? |
|------|---------|-------|-----------|
| `foldx_ddg.py` | Fast ΔΔG proxy scoring | ~seconds | Yes |
| `flex_ddg.py` | Rosetta flex-ddG oracle | ~30-60 min | Yes |
| `esm_score.py` | Sequence plausibility (ESM log-likelihoods) | ~seconds | Nice to have |

FoldX provides fast candidate screening. flex-ddG serves as the oracle — the
authoritative accept/reject signal. ESM scoring adds a sequence-level
plausibility check.

Additional tools (Boltz-2, Chai-1, ESM-IF, LigandMPNN, etc.) can be added
later as additional wrapper scripts. The agent program can be updated to
reference new tools as they become available.

### Mutation Specification Format

Mutations are specified as strings:

```
<chain>:<resnum><icode>:<wt_aa>:<mut_aa>
```

Examples: `H:52:S:Y`, `L:91:N:D`, `H:100A:G:A`

## State Tracking

### Campaign State (`state.yaml`)

```yaml
campaign_id: cmp_20260317_001
started_at: "2026-03-17T14:00:00Z"
iteration: 7
parent:
  sequence_heavy: "EVQLVES..."
  sequence_light: "DIQMTQS..."
  structure: "iterations/006/mutant_relaxed.pdb"
  ddg_cumulative: -3.2
antigen_chains: ["A"]
antibody_heavy_chain: "H"
antibody_light_chain: "L"
frozen_positions: []
```

### Mutation Ledger (`ledger.jsonl`)

One JSON object per line, appended after each iteration:

```json
{
  "iteration": 7,
  "mutation": "H:52:S:Y",
  "proxy_ddg": -1.2,
  "oracle_ddg": -0.8,
  "accepted": true,
  "rationale": "Position 52 is interface-proximal...",
  "timestamp": "2026-03-17T15:42:00Z"
}
```

This is the complete record of the campaign. It can be loaded as a DataFrame
for analysis.

## Data Models

Minimal Pydantic models in `src/autoantibody/models.py`:

```python
class Mutation(BaseModel):
    chain: str
    resnum: str
    wt_aa: str
    mut_aa: str

class ToolResult(BaseModel):
    status: Literal["ok", "error"]
    scores: dict[str, float]
    artifacts: dict[str, str]
    wall_time_s: float

class IterationDecision(BaseModel):
    iteration: int
    mutation: Mutation
    proxy_scores: dict[str, float]
    oracle_ddg: float
    accepted: bool
    rationale: str
    timestamp: datetime

class CampaignState(BaseModel):
    campaign_id: str
    started_at: datetime
    iteration: int
    parent_sequence_heavy: str
    parent_sequence_light: str
    parent_structure: Path
    ddg_cumulative: float
    antigen_chains: list[str]
    antibody_heavy_chain: str
    antibody_light_chain: str
    frozen_positions: list[str]
```

## The Agent Program (`programs/program.md`)

This file is the equivalent of autoresearch's `program.md`. It tells Claude
Code what to do and how to do it. It should contain:

1. **Objective**: maximize binding affinity (minimize ΔΔG) through iterative
   single-point mutations.
2. **Available tools**: list of tool scripts with usage examples.
3. **Decision rubric**: how to evaluate candidates — what makes a good
   mutation, how to weigh proxy scores vs. structural reasoning.
4. **Procedure**: step-by-step protocol for one iteration (read state → screen
   candidates → select mutation → run oracle → update state).
5. **Constraints**: one mutation per iteration, must target a non-frozen
   antibody residue, must use a standard amino acid.
6. **Reporting**: what to record in the decision JSON and ledger entry.
7. **Domain knowledge**: key principles of antibody-antigen binding, interface
   hotspots, common beneficial substitution patterns.

This file should be written once the tool wrappers exist so that the usage
examples are concrete and tested.

## Campaign Setup

To start a new campaign:

1. Place the input antibody-antigen complex structure (PDB/mmCIF) in
   `runs/<campaign_id>/input/`.
2. Create a `state.yaml` with the initial sequences, chain assignments, and
   any frozen positions.
3. Tell Claude Code: "Read `programs/program.md` and start the campaign at
   `runs/<campaign_id>/`."

A helper script (`scripts/init_campaign.py`) can automate step 2 by extracting
sequences and chain info from the input structure, but this is not required for
the MVP — the state file can be written manually or by Claude Code.

## Mutation Safety Checks

Before running the oracle, verify:

- Exactly one residue is changed.
- The target position is not frozen.
- The wild-type residue in the mutation spec matches the current parent
  sequence.
- The mutant amino acid is one of the 20 standard residues.

These checks are simple assertions that Claude Code or the tool wrappers can
perform. No separate validation layer is needed.

## What's Deferred

The following are explicitly out of scope for the MVP. They are good ideas
that should be revisited once the core loop is proven:

| Deferred Item | Why It Can Wait |
|---------------|-----------------|
| Docker/container tool packaging | Local scripts are fine for one machine |
| CLI entry points | Claude Code drives the loop interactively |
| Budget management (time budgets, tool tiers) | Adds complexity before we know if the loop works |
| Tool ROI tracking | Needs enough data to be meaningful |
| SQLite / Parquet storage | YAML + JSONL is sufficient at MVP scale |
| Structured event logging | The ledger captures essential state |
| HTML/Markdown campaign reports | The ledger IS the report for now |
| Caching layer | Premature until tool portfolio grows |
| Reproducibility/provenance infrastructure | Git history + ledger entries suffice initially |
| Multi-mutation combinatorial search | Single-point mutations first |
| Campaign profiling phase | Skip straight to the optimization loop |
| Multiple runner backends (Slurm, K8s) | Single machine only |

## Getting Started (Build Order)

### Step 1: Tool Wrappers

Write `tools/foldx_ddg.py` and `tools/flex_ddg.py`. Each should:

- Accept a PDB path and mutation string as arguments.
- Run the computation.
- Print a JSON result to stdout.
- Test on one known mutation with a known ΔΔG.

### Step 2: State Helpers

Write `src/autoantibody/state.py` with functions to:

- Load and save `state.yaml`.
- Append to `ledger.jsonl`.
- Initialize a new campaign from an input structure.

### Step 3: Agent Program

Write `programs/program.md` with concrete tool invocation examples and the
decision protocol.

### Step 4: Run It

Open Claude Code, point it at the program, and run the loop on a well-studied
antibody-antigen complex (e.g., trastuzumab/HER2) where experimental ΔΔG
values exist for validation.

### Step 5: Evaluate

After 10-20 iterations, check:

- Is the cumulative ΔΔG trending downward?
- Are the accepted mutations structurally reasonable?
- Is the agent making sensible tool use and reasoning decisions?
- How does the trajectory compare to random single-point mutations?

If the answers are encouraging, proceed to expand the tool set and add
infrastructure. If not, the investment has been minimal.

# autoantibody

Autonomous antibody affinity optimization using AI-guided iterative mutagenesis.

`autoantibody` uses an LLM agent (Claude Code) to iteratively optimize antibody binding affinity through point mutations. Given an antibody-antigen complex structure, the system screens candidate mutations with fast computational scorers, selects promising candidates, validates them with expensive physics-based oracles, and accumulates beneficial mutations over multiple rounds.

## How it works

Each optimization campaign follows a core loop:

1. **Read** the current antibody state (sequences + PDB structure)
2. **Screen** candidate mutations with fast scoring tools (EvoEF, Graphinity, StaB-ddG)
3. **Filter** for fold stability and sequence plausibility (ProteinMPNN, antibody language models)
4. **Select** one promising mutation with a written rationale
5. **Evaluate** with an oracle (Rosetta flex-ddG or AToM-OpenMM FEP)
6. **Accept or reject** based on the computed binding ddG
7. **Update** state and repeat

The agent is Claude Code itself. Behavior is defined in a program file, and tools are plain Python scripts that output JSON. Campaign state is tracked in flat files (YAML + JSONL) for full transparency and reproducibility.

## Scoring tools

`autoantibody` integrates eight scoring tools across four tiers:

| Tier | Tool | Method | Time | GPU |
|------|------|--------|------|-----|
| **Fast** | EvoEF | Physics-based binding ddG | ~5 s | No |
| **Fast** | StaB-ddG | ML binding ddG | ~60 s | Yes |
| **Fast** | Graphinity | Equivariant GNN binding ddG | ~10 s | Yes |
| **Medium** | BA-ddG | Boltzmann-averaged ML ddG | ~3 min | Yes |
| **Filter** | ProteinMPNN-ddG | Fold stability check | ~30 s | No |
| **Filter** | ablms | Antibody language model plausibility | ~10 s | Yes |
| **Oracle** | flex-ddG | Rosetta flex-ddG (Docker) | 30-60 min | No |
| **Oracle** | AToM-FEP | Alchemical free energy perturbation | 4-6 hrs | Yes |

Tools are discovered at runtime. Only install the ones you need; the system adapts to what's available.

## Installation

Requires Python 3.12+.

```bash
# clone the repository
git clone https://github.com/briney/autoantibody.git
cd autoantibody

# install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### External tool dependencies

The core package has no heavy dependencies beyond BioPython, Pydantic, and NumPy. Scoring tools have their own requirements and are optional:

- **EvoEF**: install the [EvoEF](https://github.com/tommyhuangthu/EvoEF) binary and either add it to your `PATH` or set the `EVOEF_BINARY` environment variable
- **StaB-ddG**: `pip install stabddg`
- **Graphinity**: `pip install graphinity`
- **BA-ddG**: clone the repo and set `BADDG_DIR` (or install to `/opt/baddg`)
- **ProteinMPNN-ddG** and **flex-ddG**: require [Docker](https://docs.docker.com/get-docker/)
- **AToM-FEP**: `conda install -c conda-forge openmm openmmtools`
- **ablms**: `pip install ablms`

## Usage

### Initialize a campaign

Set up a new optimization campaign from an antibody-antigen complex PDB:

```bash
python scripts/init_campaign.py \
    --pdb data/complex.pdb \
    --heavy-chain H \
    --light-chain L \
    --antigen-chains A \
    --output runs/my_campaign
```

Or use a YAML config file:

```bash
python scripts/init_campaign.py --config configs/my_campaign.yaml
```

Example config:

```yaml
pdb: data/complex.pdb
heavy_chain: H
light_chain: L
antigen_chains: [A]
output: runs/my_campaign
frozen_positions: [H:52, L:91]
```

This creates the campaign directory with the input PDB, initial `state.yaml`, an empty `ledger.jsonl`, and a `scorer_inventory.yaml` listing all detected tools.

### Analyze an antibody-antigen interface

```python
from pathlib import Path
from autoantibody.structure import extract_sequences, get_interface_residues

# extract sequences for all chains in the PDB
sequences = extract_sequences(Path("data/complex.pdb"))
for chain_id, seq in sequences.items():
    print(f"Chain {chain_id}: {len(seq)} residues")

# find antibody residues at the antigen interface (8 A cutoff)
interface = get_interface_residues(
    Path("data/complex.pdb"),
    antibody_chains=["H", "L"],
    antigen_chains=["A"],
)
print(f"\n{len(interface)} interface residues:")
for res in interface:
    print(f"  {res['chain']}:{res['resnum']} ({res['aa']})")
```

### Work with mutations

```python
from autoantibody.models import Mutation
from autoantibody.structure import (
    get_residue_index_map,
    validate_mutation_against_structure,
)

# parse a mutation string
mutation = Mutation.parse("H:52:S:Y")
print(mutation)  # H:52:S:Y

# validate against the PDB structure
errors = validate_mutation_against_structure(Path("data/complex.pdb"), mutation)
if errors:
    print("Invalid mutation:", errors)
else:
    print("Mutation is valid")

# apply mutation to a sequence
index_map = get_residue_index_map(Path("data/complex.pdb"), chain_id="H")
original_seq = sequences["H"]
mutated_seq = mutation.apply_to_sequence(original_seq, index_map)

# convert to tool-specific formats
print(mutation.to_evoef())           # SH52Y
print(mutation.to_rosetta_resfile()) # Rosetta resfile format
print(mutation.to_skempi())          # SH52Y (SKEMPIv2 format)
```

### Manage campaign state

```python
from pathlib import Path
from datetime import UTC, datetime
from autoantibody.state import load_state, save_state, append_ledger
from autoantibody.models import IterationDecision

# load current campaign state
campaign_dir = Path("runs/my_campaign")
state = load_state(campaign_dir)
print(f"Campaign: {state.campaign_id}, iteration: {state.iteration}")

# record a mutation decision
decision = IterationDecision(
    iteration=1,
    mutation="H:52:S:Y",
    proxy_scores={"evoef": -1.2, "graphinity": -0.9},
    oracle_ddg=-0.8,
    accepted=True,
    rationale="Interface-proximal serine-to-tyrosine adds aromatic contact with antigen",
    timestamp=datetime.now(UTC),
)
append_ledger(campaign_dir, decision)

# update and save state after accepting a mutation
state.iteration += 1
state.parent.ddg_cumulative += decision.oracle_ddg
save_state(campaign_dir, state)
```

### Check available scorers

```python
from autoantibody.scorers import get_available_scorers, get_scorers_by_tier
from autoantibody.models import ScorerTier

# list all installed scoring tools
for scorer in get_available_scorers():
    print(f"[{scorer.tier.value}] {scorer.name}: {scorer.description}")

# get just the fast-tier scorers
fast = get_scorers_by_tier(ScorerTier.FAST)
print(f"\n{len(fast)} fast scorers available")
```

### Run a scoring tool directly

Each tool wrapper is a standalone script that takes a PDB path and mutation string:

```bash
python tools/evoef_ddg.py runs/my_campaign/input/complex.pdb H:52:S:Y
```

Output is JSON with a standardized schema:

```json
{
  "status": "ok",
  "scores": {"ddg": -1.23, "wt_total": -8.45, "mut_total": -9.68},
  "artifacts": {},
  "wall_time_s": 4.7,
  "error_message": null,
  "scorer_name": "evoef"
}
```

## Project structure

```
autoantibody/
├── src/autoantibody/       # Core package
│   ├── models.py           # Data models (Mutation, CampaignState, ToolResult, ...)
│   ├── state.py            # Campaign state I/O (YAML + JSONL)
│   ├── structure.py        # PDB parsing and interface analysis
│   └── scorers.py          # Scorer registry and availability checks
├── tools/                  # Standalone scoring tool wrappers
│   ├── evoef_ddg.py        # EvoEF binding ddG
│   ├── stabddg_score.py    # StaB-ddG ML binding ddG
│   ├── graphinity_score.py # Graphinity GNN binding ddG
│   ├── baddg_score.py      # BA-ddG Boltzmann-averaged ddG
│   ├── stability_check.py  # ProteinMPNN fold stability filter
│   ├── ablms_score.py      # Antibody language model scoring
│   ├── flex_ddg.py         # Rosetta flex-ddG oracle
│   └── atom_ddg.py         # AToM-OpenMM FEP oracle
├── scripts/                # CLI entry points
│   ├── init_campaign.py    # Campaign initialization
│   └── benchmark_skempi.py # Tool benchmarking on SKEMPI2
├── tests/                  # Test suite
├── docs/                   # Architecture documentation
├── programs/               # Agent behavior programs
├── configs/                # Campaign config files
└── pyproject.toml
```

## Development

```bash
# run tests (skips slow integration tests by default)
pytest

# run all tests including integration tests
pytest -m ""

# lint and format
ruff check src/ tests/
ruff format src/ tests/

# type check
mypy src/
```

## License

MIT

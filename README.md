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

`autoantibody` integrates eight scoring tools across four tiers. Each tool runs inside its own Docker container, isolating dependencies and making availability a simple "is the image built?" check.

| Tier | Tool | Method | Time | GPU | Container |
|------|------|--------|------|-----|-----------|
| **Fast** | EvoEF | Physics-based binding ddG | ~5 s | No | `autoantibody/evoef` |
| **Fast** | StaB-ddG | ML binding ddG | ~60 s | Yes | `autoantibody/stabddg` |
| **Fast** | Graphinity | Equivariant GNN binding ddG | ~10 s | Yes | `autoantibody/graphinity` |
| **Medium** | BA-ddG | Boltzmann-averaged ML ddG | ~3 min | Yes | `autoantibody/baddg` |
| **Filter** | ProteinMPNN-ddG | Fold stability check | ~30 s | No | `autoantibody/proteinmpnn_stability` |
| **Filter** | ablms | Antibody language model plausibility | ~10 s | Yes | `autoantibody/ablms` |
| **Oracle** | flex-ddG | Rosetta flex-ddG | 30-60 min | No | `autoantibody/flex_ddg` |
| **Oracle** | AToM-FEP | Alchemical free energy perturbation | 4-6 hrs | Yes | `autoantibody/atom_fep` |

Tools are discovered at runtime by checking for their Docker images. Only build the ones you need; the system adapts to what's available.

## Installation

Requires Python 3.12+ and [Docker](https://docs.docker.com/get-docker/).

```bash
# clone the repository
git clone https://github.com/briney/autoantibody.git
cd autoantibody

# install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Building scoring tool containers

Each scoring tool is packaged as a Docker container. Build them with the included Makefile:

```bash
# build all priority tools (evoef, stabddg, ablms, proteinmpnn_stability, flex_ddg)
make -C containers build-all

# build a single tool
make -C containers build-evoef

# smoke test: verify the container starts and imports succeed
make -C containers test-evoef
```

Or use the install script, which also handles PyTorch, test data, and verification:

```bash
# install everything + build all containers
./install.sh

# minimal: core deps + EvoEF container only
./install.sh --minimal

# skip Docker builds (install Python deps only)
./install.sh --skip-docker
```

### Checking tool availability

```bash
# list all scorers and their status
python -m autoantibody.run_tool list

# check a specific scorer
python -m autoantibody.run_tool check evoef
```

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

### Run a scoring tool

The primary interface for scoring is the `run_tool` CLI, which invokes tools inside their containers:

```bash
# score a mutation
python -m autoantibody.run_tool score evoef runs/my_campaign/input/complex.pdb H:52:S:Y

# with extra tool-specific arguments
python -m autoantibody.run_tool score stabddg complex.pdb H:52:S:Y -- --chains HL_A
```

Output is JSON with a standardized schema:

```json
{
  "status": "ok",
  "scores": {"ddg": -1.23, "wt_binding_energy": -8.45, "mut_binding_energy": -9.68},
  "artifacts": {},
  "wall_time_s": 4.7,
  "error_message": null,
  "scorer_name": "evoef"
}
```

The tool scripts in `tools/` can also be run directly (they detect whether they're inside a container via `AUTOANTIBODY_CONTAINER` env var):

```bash
python tools/evoef_ddg.py runs/my_campaign/input/complex.pdb H:52:S:Y
```

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

### Use the container API from Python

```python
from pathlib import Path
from autoantibody.container import score_mutation
from autoantibody.scorers import SCORER_REGISTRY

scorer = SCORER_REGISTRY["evoef"]
result = score_mutation(scorer, Path("complex.pdb"), "H:52:S:Y")

if result.status == "ok":
    print(f"ddG = {result.scores['ddg']:.2f} kcal/mol")
else:
    print(f"Error: {result.error_message}")
```

## Project structure

```
autoantibody/
├── src/autoantibody/       # Core package
│   ├── models.py           # Data models (Mutation, CampaignState, ToolResult, ...)
│   ├── state.py            # Campaign state I/O (YAML + JSONL)
│   ├── structure.py        # PDB parsing and interface analysis
│   ├── scorers.py          # Scorer registry and availability checks
│   ├── container.py        # Docker container invocation for scoring tools
│   └── run_tool.py         # CLI entry point (python -m autoantibody.run_tool)
├── tools/                  # Standalone scoring tool wrappers (run inside containers)
│   ├── evoef_ddg.py        # EvoEF binding ddG
│   ├── stabddg_score.py    # StaB-ddG ML binding ddG
│   ├── graphinity_score.py # Graphinity GNN binding ddG
│   ├── baddg_score.py      # BA-ddG Boltzmann-averaged ddG
│   ├── stability_check.py  # ProteinMPNN fold stability filter
│   ├── ablms_score.py      # Antibody language model scoring
│   ├── flex_ddg.py         # Rosetta flex-ddG oracle
│   └── atom_ddg.py         # AToM-OpenMM FEP oracle
├── containers/             # Dockerfiles for each scoring tool
│   ├── base/               # Base images (CPU and GPU)
│   ├── evoef/              # EvoEF (multi-stage, compiles from source)
│   ├── stabddg/            # StaB-ddG (GPU + PyTorch)
│   ├── ablms/              # ablms (GPU + PyTorch)
│   ├── proteinmpnn_stability/ # ProteinMPNN-ddG (extends vendor image)
│   ├── flex_ddg/           # Rosetta flex-ddG (extends Rosetta image)
│   └── Makefile            # Build system (build-all, build-<tool>, test-<tool>)
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

### Building and testing containers

```bash
# build all containers
make -C containers build-all

# run smoke tests (verify containers start)
make -C containers test-all

# run integration tests (requires built images + test PDB)
pytest tests/test_container_integration.py -m slow

# clean up all images
make -C containers clean
```

## License

MIT

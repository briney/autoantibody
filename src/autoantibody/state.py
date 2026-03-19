"""Campaign state management: load, save, initialize, and ledger operations."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml

from autoantibody.models import CampaignState, IterationDecision, ParentState
from autoantibody.structure import extract_sequences


def load_state(campaign_dir: Path) -> CampaignState:
    """Load campaign state from state.yaml."""
    with open(campaign_dir / "state.yaml") as f:
        return CampaignState(**yaml.safe_load(f))


def save_state(campaign_dir: Path, state: CampaignState) -> None:
    """Save campaign state to state.yaml."""
    data = json.loads(state.model_dump_json())
    with open(campaign_dir / "state.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def append_ledger(campaign_dir: Path, decision: IterationDecision) -> None:
    """Append a decision to the mutation ledger (JSONL)."""
    with open(campaign_dir / "ledger.jsonl", "a") as f:
        f.write(decision.model_dump_json() + "\n")


def load_ledger(campaign_dir: Path) -> list[IterationDecision]:
    """Load all decisions from the mutation ledger."""
    ledger_path = campaign_dir / "ledger.jsonl"
    if not ledger_path.exists():
        return []
    decisions: list[IterationDecision] = []
    with open(ledger_path) as f:
        for line in f:
            line = line.strip()
            if line:
                decisions.append(IterationDecision(**json.loads(line)))
    return decisions


def init_campaign(
    campaign_dir: Path,
    pdb_path: Path,
    antibody_heavy_chain: str,
    antibody_light_chain: str,
    antigen_chains: list[str],
    campaign_id: str | None = None,
    frozen_positions: list[str] | None = None,
    sequence_heavy_override: str | None = None,
    sequence_light_override: str | None = None,
) -> CampaignState:
    """Initialize a new campaign from an input PDB.

    Creates directory structure, copies the input PDB, extracts sequences,
    and writes the initial state.yaml and empty ledger.

    Args:
        campaign_dir: Campaign directory (created if needed).
        pdb_path: Input antibody-antigen complex PDB.
        antibody_heavy_chain: Heavy chain ID (e.g., "H").
        antibody_light_chain: Light chain ID (e.g., "L").
        antigen_chains: Antigen chain IDs (e.g., ["A"]).
        campaign_id: Optional ID. Auto-generated from timestamp if omitted.
        frozen_positions: Positions to exclude (e.g., ["H:52", "L:91"]).
        sequence_heavy_override: Override heavy chain sequence instead of
            extracting from PDB. Useful for starting from a germline sequence
            while keeping the mature PDB structure.
        sequence_light_override: Override light chain sequence.

    Returns:
        The initialized CampaignState.
    """
    campaign_dir = Path(campaign_dir)
    for subdir in ["input", "iterations"]:
        (campaign_dir / subdir).mkdir(parents=True, exist_ok=True)

    input_pdb = campaign_dir / "input" / pdb_path.name
    shutil.copy2(pdb_path, input_pdb)

    sequences = extract_sequences(pdb_path)
    for label, cid in [("Heavy", antibody_heavy_chain), ("Light", antibody_light_chain)]:
        if cid not in sequences:
            raise ValueError(
                f"{label} chain '{cid}' not found. Available: {list(sequences.keys())}"
            )

    if campaign_id is None:
        campaign_id = f"cmp_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

    heavy_seq = sequence_heavy_override or sequences[antibody_heavy_chain]
    light_seq = sequence_light_override or sequences[antibody_light_chain]

    state = CampaignState(
        campaign_id=campaign_id,
        started_at=datetime.now(UTC),
        iteration=0,
        parent=ParentState(
            sequence_heavy=heavy_seq,
            sequence_light=light_seq,
            structure=f"input/{pdb_path.name}",
            ddg_cumulative=0.0,
        ),
        antigen_chains=antigen_chains,
        antibody_heavy_chain=antibody_heavy_chain,
        antibody_light_chain=antibody_light_chain,
        frozen_positions=frozen_positions or [],
    )

    save_state(campaign_dir, state)
    (campaign_dir / "ledger.jsonl").touch()
    return state

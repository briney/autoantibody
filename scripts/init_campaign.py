#!/usr/bin/env python
"""Initialize a new antibody optimization campaign.

Usage:
    python scripts/init_campaign.py \
        --pdb /path/to/complex.pdb \
        --heavy-chain H --light-chain L \
        --antigen-chains A \
        --output runs/my_campaign

    python scripts/init_campaign.py --config configs/example_campaign.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml

from autoantibody.scorers import get_available_scorers
from autoantibody.state import init_campaign
from autoantibody.structure import extract_sequences, get_interface_residues

app = typer.Typer(help="Initialize an antibody optimization campaign.")


@app.command()
def main(
    pdb: Annotated[Path | None, typer.Option(help="Input PDB structure")] = None,
    heavy_chain: Annotated[str | None, typer.Option("--heavy-chain", help="Heavy chain ID")] = None,
    light_chain: Annotated[str | None, typer.Option("--light-chain", help="Light chain ID")] = None,
    antigen_chains: Annotated[
        list[str] | None, typer.Option("--antigen-chains", help="Antigen chain IDs")
    ] = None,
    output: Annotated[Path | None, typer.Option(help="Campaign output directory")] = None,
    campaign_id: Annotated[str | None, typer.Option("--campaign-id", help="Campaign ID")] = None,
    frozen: Annotated[
        list[str] | None, typer.Option("--frozen", help="Frozen positions (e.g., H:52)")
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", help="YAML config (overrides other args)")
    ] = None,
) -> None:
    """Create a campaign directory with state.yaml, empty ledger, and scorer inventory."""
    seq_heavy_override: str | None = None
    seq_light_override: str | None = None

    if config:
        with open(config) as f:
            cfg = yaml.safe_load(f)
        pdb = Path(cfg["pdb"])
        heavy_chain = cfg["heavy_chain"]
        light_chain = cfg["light_chain"]
        antigen_chains = cfg["antigen_chains"]
        output = Path(cfg["output"])
        campaign_id = cfg.get("campaign_id")
        frozen = cfg.get("frozen_positions", [])
        seq_heavy_override = cfg.get("sequence_heavy_override")
        seq_light_override = cfg.get("sequence_light_override")

    if frozen is None:
        frozen = []

    if not all([pdb, heavy_chain, light_chain, antigen_chains, output]):
        typer.echo("Error: provide all required args or --config", err=True)
        raise typer.Exit(1)

    # Type narrowing — validated above
    assert pdb is not None
    assert heavy_chain is not None
    assert light_chain is not None
    assert antigen_chains is not None
    assert output is not None

    # Show available chains first
    sequences = extract_sequences(pdb)
    typer.echo(f"Chains found in {pdb.name}:")
    for cid, seq in sequences.items():
        typer.echo(f"  {cid}: {len(seq)} residues")

    state = init_campaign(
        campaign_dir=output,
        pdb_path=pdb,
        antibody_heavy_chain=heavy_chain,
        antibody_light_chain=light_chain,
        antigen_chains=antigen_chains,
        campaign_id=campaign_id,
        frozen_positions=frozen,
        sequence_heavy_override=seq_heavy_override,
        sequence_light_override=seq_light_override,
    )

    # Write scorer inventory
    available = get_available_scorers()
    inventory = {
        s.name: {
            "tier": s.tier.value,
            "requires_gpu": s.requires_gpu,
            "typical_seconds": s.typical_seconds,
            "description": s.description,
        }
        for s in available
    }
    with open(output / "scorer_inventory.yaml", "w") as f:
        yaml.dump(inventory, f, default_flow_style=False, sort_keys=False)

    # Show interface summary
    interface = get_interface_residues(
        pdb,
        antibody_chains=[heavy_chain, light_chain],
        antigen_chains=antigen_chains,
    )
    typer.echo(f"\nCampaign initialized: {state.campaign_id}")
    typer.echo(f"  Heavy chain ({heavy_chain}): {len(state.parent.sequence_heavy)} residues")
    typer.echo(f"  Light chain ({light_chain}): {len(state.parent.sequence_light)} residues")
    typer.echo(f"  Interface residues: {len(interface)}")
    typer.echo(f"  Available scorers: {len(available)}")
    for s in available:
        typer.echo(f"    [{s.tier.value}] {s.name}: {s.description}")
    typer.echo(f"  Output: {output}")


if __name__ == "__main__":
    app()

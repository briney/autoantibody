"""CLI entry point for running scoring tools via containers.

Usage:
    python -m autoantibody.run_tool score evoef /path/to/structure.pdb H:52:S:Y
    python -m autoantibody.run_tool list
    python -m autoantibody.run_tool check evoef
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Run autoantibody scoring tools via Docker containers.")


@app.command()
def score(
    scorer_name: Annotated[str, typer.Argument(help="Name of the scorer (e.g., evoef)")],
    pdb_path: Annotated[Path, typer.Argument(help="Path to input PDB structure")],
    mutation: Annotated[str, typer.Argument(help="Mutation string (e.g., H:52:S:Y)")],
    extra_args: Annotated[
        list[str] | None, typer.Argument(help="Additional arguments for the tool")
    ] = None,
    timeout: Annotated[float | None, typer.Option(help="Timeout in seconds")] = None,
) -> None:
    """Score a mutation using a containerized tool."""
    from autoantibody.container import score_mutation
    from autoantibody.scorers import SCORER_REGISTRY, check_scorer_available

    if scorer_name not in SCORER_REGISTRY:
        typer.echo(f"Unknown scorer: {scorer_name}", err=True)
        raise typer.Exit(1)

    if not check_scorer_available(scorer_name):
        typer.echo(f"Scorer '{scorer_name}' is not available", err=True)
        raise typer.Exit(1)

    scorer = SCORER_REGISTRY[scorer_name]
    pdb_path = pdb_path.resolve()

    if not pdb_path.exists():
        typer.echo(f"PDB file not found: {pdb_path}", err=True)
        raise typer.Exit(1)

    result = score_mutation(
        scorer=scorer,
        pdb_path=pdb_path,
        mutation_str=mutation,
        extra_args=extra_args,
        timeout=timeout,
    )
    typer.echo(result.model_dump_json(indent=2))
    raise typer.Exit(0 if result.status == "ok" else 1)


@app.command(name="list")
def list_scorers() -> None:
    """List all scorers and their availability status."""
    from autoantibody.scorers import SCORER_REGISTRY, check_scorer_available

    for name, info in sorted(SCORER_REGISTRY.items()):
        available = check_scorer_available(name)
        status = "available" if available else "unavailable"
        container = "container" if info.containerized else "native"
        image = info.docker_image or "n/a"
        typer.echo(f"  {name:<25s} {info.tier.value:<8s} {status:<12s} {container:<10s} {image}")


@app.command()
def check(
    scorer_name: Annotated[str, typer.Argument(help="Name of the scorer to check")],
) -> None:
    """Check if a scorer is available."""
    from autoantibody.scorers import SCORER_REGISTRY, check_scorer_available

    if scorer_name not in SCORER_REGISTRY:
        typer.echo(f"Unknown scorer: {scorer_name}", err=True)
        raise typer.Exit(1)

    available = check_scorer_available(scorer_name)
    info = SCORER_REGISTRY[scorer_name]
    typer.echo(f"Scorer: {scorer_name}")
    typer.echo(f"Available: {available}")
    typer.echo(f"Containerized: {info.containerized}")
    typer.echo(f"Docker image: {info.docker_image or 'n/a'}")
    typer.echo(f"Requires GPU: {info.requires_gpu}")
    raise typer.Exit(0 if available else 1)


if __name__ == "__main__":
    app()

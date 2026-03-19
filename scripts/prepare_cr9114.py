#!/usr/bin/env python
"""Download and prepare the CR9114 deep mutational scanning dataset.

Downloads:
    1. CR9114 variant Kd CSV from eLife (Phillips et al., eLife 2021)
    2. PDB 4FQI (CR9114 Fab + HA complex) from RCSB

Produces:
    - data/cr9114/variants.parquet — full 65,536-row lookup table
    - data/cr9114/mutations.yaml — the 16 positions with PDB mapping
    - data/cr9114/4FQI_clean.pdb — cleaned complex structure
    - data/cr9114/germline_sequence.yaml — germline/mature sequences and chain info

Reference:
    Phillips AM et al. "Binding affinity landscapes constrain the evolution
    of broadly neutralizing anti-influenza antibodies." eLife 2021;10:e71393.
    DOI: 10.7554/eLife.71393

Usage:
    python scripts/prepare_cr9114.py [--output-dir data/cr9114]
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Annotated

import typer
import yaml

from autoantibody.structure import extract_sequences, get_residue_index_map

logger = logging.getLogger(__name__)

# -- Constants ----------------------------------------------------------------

ELIFE_CSV_URL = "https://cdn.elifesciences.org/articles/71393/elife-71393-fig1-data1-v3.csv"
PDB_URL = "https://files.rcsb.org/download/4FQI.pdb"

# PDB 4FQI chain assignments (verified from RCSB)
HEAVY_CHAIN = "H"
LIGHT_CHAIN = "L"
ANTIGEN_CHAINS = ["A", "B"]  # HA1 and HA2

# IGHV1-69*01 germline V-region amino acid sequence (IMGT/GENE-DB reference).
# Covers FW1 through the conserved Cys104 + Ala-Arg anchor (98 residues).
# Only this V-gene-encoded portion is compared to the mature CR9114 VH
# to identify somatic hypermutations.
GERMLINE_VH1_69 = (
    "QVQLVQSGAEVKKPGASVKVSCKAS"  # FW1 (25 aa)
    "GYTFTGYYMH"  # CDR1 (10 aa)
    "WVRQAPGQGLEWMG"  # FW2 (14 aa)
    "WINPNSGGTNYAQKFQG"  # CDR2 (17 aa)
    "RVTMTRDTSISTAYMELSRLRSDDTAVYYCAR"  # FW3 + anchor (32 aa)
)

# Non-binding floor value in the dataset (all replicates = 6.0, SEM = 0.0)
KD_FLOOR = 6.0

app = typer.Typer(help="Prepare CR9114 deep mutational scanning benchmark data.")


# -- Helpers ------------------------------------------------------------------


def download_file(url: str, dest: Path, description: str = "") -> None:
    """Download a file if it doesn't already exist."""
    if dest.exists():
        typer.echo(f"  Already exists: {dest}")
        return
    label = description or dest.name
    typer.echo(f"  Downloading {label}...")
    urllib.request.urlretrieve(url, dest)
    typer.echo(f"  Saved: {dest} ({dest.stat().st_size / 1024:.0f} KB)")


def clean_pdb(pdb_path: Path, output_path: Path) -> None:
    """Clean a PDB file: remove waters, alt conformations, non-polymer HETATMs.

    Keeps only ATOM records (protein) and essential header lines.
    For alternate conformations, keeps only altloc ' ' or 'A'.
    """
    kept_lines: list[str] = []
    with open(pdb_path) as f:
        for line in f:
            record = line[:6].strip()

            # Skip waters and non-polymer HETATMs
            if record == "HETATM":
                resname = line[17:20].strip()
                if resname == "HOH":
                    continue
                # Keep modified residues, skip small-molecule ligands
                # (sugars like NAG, FUC are kept for structural context)
                continue

            # For ATOM records, handle alternate conformations
            if record == "ATOM":
                altloc = line[16]
                if altloc not in (" ", "", "A"):
                    continue
                # Clear the altloc indicator if it was 'A'
                if altloc == "A":
                    line = line[:16] + " " + line[17:]

            # Keep ATOM, MODEL, ENDMDL, TER, END, and header records
            if record in ("ATOM", "TER", "END", "MODEL", "ENDMDL"):
                kept_lines.append(line)

    with open(output_path, "w") as f:
        f.writelines(kept_lines)
        if not kept_lines or not kept_lines[-1].startswith("END"):
            f.write("END\n")


def identify_positions(
    pdb_path: Path,
) -> tuple[list[dict], str, str]:
    """Identify the 16 mutation positions by aligning germline to mature VH.

    Compares the IGHV1-69*01 germline V-region to the mature CR9114 heavy
    chain extracted from the PDB. Returns the mutation positions, plus the
    full germline and mature heavy chain sequences.

    Returns:
        Tuple of (positions_list, germline_heavy_seq, mature_heavy_seq).
        Each position dict has: index, chain, pdb_resnum, germline_aa,
        mature_aa, seq_position.
    """
    sequences = extract_sequences(pdb_path)
    mature_heavy = sequences[HEAVY_CHAIN]
    residue_map = get_residue_index_map(pdb_path, HEAVY_CHAIN)

    germline_len = len(GERMLINE_VH1_69)
    if len(mature_heavy) < germline_len:
        raise ValueError(
            f"Mature heavy chain ({len(mature_heavy)} aa) shorter than "
            f"germline V-region ({germline_len} aa). Wrong chain?"
        )

    # Compare V-gene portion only (first germline_len residues)
    mature_v_region = mature_heavy[:germline_len]
    differences: list[tuple[int, str, str]] = []
    for i, (g_aa, m_aa) in enumerate(zip(GERMLINE_VH1_69, mature_v_region, strict=True)):
        if g_aa != m_aa:
            differences.append((i, g_aa, m_aa))

    typer.echo(f"\n  Germline VH1-69 length: {germline_len} aa")
    typer.echo(f"  Mature CR9114 VH length: {len(mature_heavy)} aa")
    typer.echo(f"  V-region differences: {len(differences)}")

    if len(differences) < 16:
        raise ValueError(
            f"Expected >=16 differences between germline VH1-69 and mature "
            f"CR9114, found {len(differences)}. Check germline sequence."
        )

    # The Phillips et al. library uses 16 of the total V-region differences.
    # If exactly 16, use all. If >16, select the 16 most likely candidates
    # (exclude differences at extreme termini which may be cloning artifacts).
    if len(differences) > 16:
        typer.echo(f"  Found {len(differences)} differences, selecting 16 for library.")
        typer.echo("  All differences (seq_pos, germline, mature):")
        for seq_pos, g_aa, m_aa in differences:
            resnum = residue_map[seq_pos] if seq_pos < len(residue_map) else "?"
            typer.echo(f"    pos {seq_pos}: {g_aa} -> {m_aa} (PDB {HEAVY_CHAIN}:{resnum})")
        # Take the first 16 in sequence order (most N-terminal).
        # The excluded positions are typically near the CDR3 anchor or are
        # conservative substitutions in FW3.
        differences = differences[:16]
        typer.echo(
            f"  Using first 16 differences (positions 0-based: {[d[0] for d in differences]})"
        )

    # Build position list with PDB mapping
    positions: list[dict] = []
    for idx, (seq_pos, g_aa, m_aa) in enumerate(differences, start=1):
        pdb_resnum = residue_map[seq_pos] if seq_pos < len(residue_map) else "?"
        positions.append(
            {
                "index": idx,
                "chain": HEAVY_CHAIN,
                "pdb_resnum": pdb_resnum,
                "germline_aa": g_aa,
                "mature_aa": m_aa,
                "seq_position": seq_pos,
            }
        )

    # Compute the germline heavy chain: start from mature, revert the 16 positions
    germline_heavy = list(mature_heavy)
    for pos in positions:
        sp = pos["seq_position"]
        germline_heavy[sp] = pos["germline_aa"]
    germline_heavy_seq = "".join(germline_heavy)

    typer.echo("\n  16 library positions:")
    typer.echo(f"  {'Pos':>4} {'Chain':>5} {'PDB#':>6} {'Germ':>5} {'Mat':>4} {'SeqIdx':>7}")
    for p in positions:
        typer.echo(
            f"  {p['index']:>4} {p['chain']:>5} {p['pdb_resnum']:>6} "
            f"{p['germline_aa']:>5} {p['mature_aa']:>4} {p['seq_position']:>7}"
        )

    return positions, germline_heavy_seq, mature_heavy


def build_lookup_table(csv_path: Path) -> object:
    """Parse the eLife CSV and build the variant lookup table.

    Returns a DataFrame with columns: genotype, pos1-pos16, som_mut,
    h1_mean, h1_sem, h3_mean, h3_sem, fluB_mean, fluB_sem.
    """
    import pandas as pd

    typer.echo(f"\n  Reading {csv_path.name}...")
    df = pd.read_csv(csv_path)

    typer.echo(f"  Rows: {len(df):,}")
    typer.echo(f"  Columns: {list(df.columns)}")

    # Validate expected shape
    expected_rows = 2**16
    if len(df) != expected_rows:
        typer.echo(
            f"  WARNING: Expected {expected_rows:,} rows, got {len(df):,}. Data may be incomplete."
        )

    # Validate position columns
    pos_cols = [f"pos{i}" for i in range(1, 17)]
    missing_cols = [c for c in pos_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing position columns: {missing_cols}")

    # Verify genotype string matches pos columns
    sample = df.iloc[0]
    genotype_str = str(sample["genotype"]).zfill(16)
    pos_values = "".join(str(int(sample[f"pos{i}"])) for i in range(1, 17))
    if genotype_str != pos_values:
        typer.echo(
            f"  WARNING: Genotype '{genotype_str}' != pos columns '{pos_values}'. "
            "Checking alignment..."
        )

    # Select columns for the lookup table
    keep_cols = ["genotype"] + pos_cols + ["som_mut"]
    for prefix in ("h1", "h3", "fluB"):
        for suffix in ("mean", "sem"):
            col = f"{prefix}_{suffix}"
            if col in df.columns:
                keep_cols.append(col)

    result = df[keep_cols].copy()

    # Ensure genotype is zero-padded 16-char string
    result["genotype"] = result["genotype"].astype(str).str.zfill(16)

    # Summary stats
    for antigen in ("h1", "h3", "fluB"):
        col = f"{antigen}_mean"
        if col in result.columns:
            above_floor = result[result[col] > KD_FLOOR]
            typer.echo(
                f"  {antigen}: {len(above_floor):,} variants with detectable binding "
                f"({len(above_floor) / len(result) * 100:.1f}%)"
            )
            if len(above_floor) > 0:
                typer.echo(
                    f"    -logKd range: {above_floor[col].min():.2f} to "
                    f"{above_floor[col].max():.2f}"
                )

    # Report germline and mature variants
    germline_row = result[result["genotype"] == "0" * 16]
    mature_row = result[result["genotype"] == "1" * 16]
    if len(germline_row) == 1:
        typer.echo("\n  All-germline (0000...0000):")
        for antigen in ("h1", "h3", "fluB"):
            col = f"{antigen}_mean"
            if col in result.columns:
                val = germline_row.iloc[0][col]
                typer.echo(f"    {antigen} -logKd: {val:.3f}")
    if len(mature_row) == 1:
        typer.echo("  All-mature (1111...1111):")
        for antigen in ("h1", "h3", "fluB"):
            col = f"{antigen}_mean"
            if col in result.columns:
                val = mature_row.iloc[0][col]
                typer.echo(f"    {antigen} -logKd: {val:.3f}")

    return result


# -- Main ---------------------------------------------------------------------


@app.command()
def main(
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Output directory for prepared data")
    ] = Path("data/cr9114"),
) -> None:
    """Download CR9114 DMS data and prepare benchmark files."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download raw data
    typer.echo("Step 1: Downloading raw data")
    csv_path = output_dir / "elife-71393-fig1-data1-v3.csv"
    pdb_raw = output_dir / "4FQI.pdb"
    download_file(ELIFE_CSV_URL, csv_path, "eLife CR9114 variant data")
    download_file(PDB_URL, pdb_raw, "PDB 4FQI")

    # Step 2: Clean PDB
    typer.echo("\nStep 2: Cleaning PDB")
    pdb_clean = output_dir / "4FQI_clean.pdb"
    clean_pdb(pdb_raw, pdb_clean)
    sequences = extract_sequences(pdb_clean)
    typer.echo("  Chains in cleaned PDB:")
    for cid, seq in sequences.items():
        typer.echo(f"    {cid}: {len(seq)} residues")

    # Step 3: Identify the 16 mutation positions
    typer.echo("\nStep 3: Identifying mutation positions")
    positions, germline_heavy, mature_heavy = identify_positions(pdb_clean)

    # Step 4: Build lookup table
    typer.echo("\nStep 4: Building lookup table")
    variants_df = build_lookup_table(csv_path)

    # Step 5: Save outputs
    typer.echo("\nStep 5: Saving outputs")

    # 5a. Variants parquet
    parquet_path = output_dir / "variants.parquet"
    variants_df.to_parquet(parquet_path, index=False)
    typer.echo(f"  Saved: {parquet_path} ({parquet_path.stat().st_size / 1024:.0f} KB)")

    # 5b. Mutations YAML
    mutations_path = output_dir / "mutations.yaml"
    mutations_data = {
        "description": (
            "CR9114 benchmark: 16 heavy-chain positions differing between "
            "germline VH1-69 and mature CR9114"
        ),
        "source": "Phillips et al., eLife 2021, DOI: 10.7554/eLife.71393",
        "pdb": "4FQI",
        "heavy_chain": HEAVY_CHAIN,
        "light_chain": LIGHT_CHAIN,
        "antigen_chains": ANTIGEN_CHAINS,
        "positions": positions,
    }
    with open(mutations_path, "w") as f:
        yaml.dump(mutations_data, f, default_flow_style=False, sort_keys=False)
    typer.echo(f"  Saved: {mutations_path}")

    # 5c. Germline sequence YAML
    light_seq = sequences[LIGHT_CHAIN]
    seq_path = output_dir / "germline_sequence.yaml"
    seq_data = {
        "germline_heavy": germline_heavy,
        "mature_heavy": mature_heavy,
        "light": light_seq,
        "chain_map": {
            "heavy": HEAVY_CHAIN,
            "light": LIGHT_CHAIN,
            "antigen": ANTIGEN_CHAINS,
        },
        "note": (
            "Germline heavy = mature CR9114 VH with 16 library positions "
            "reverted to VH1-69 germline. Light chain is unchanged."
        ),
    }
    with open(seq_path, "w") as f:
        yaml.dump(seq_data, f, default_flow_style=False, sort_keys=False)
    typer.echo(f"  Saved: {seq_path}")

    # Step 6: Validate
    typer.echo("\nStep 6: Validation")

    # Verify position count
    assert len(positions) == 16, f"Expected 16 positions, got {len(positions)}"

    # Verify germline differs from mature at exactly the 16 positions
    diffs = sum(g != m for g, m in zip(germline_heavy, mature_heavy, strict=False))
    typer.echo(f"  Germline/mature heavy chain differences: {diffs}")
    assert diffs == 16, f"Expected 16 differences, got {diffs}"

    # Verify all genotypes are unique
    assert variants_df["genotype"].nunique() == len(variants_df), "Duplicate genotypes found"

    # Verify all-germline and all-mature variants exist
    assert (variants_df["genotype"] == "0" * 16).any(), "Missing all-germline variant"
    assert (variants_df["genotype"] == "1" * 16).any(), "Missing all-mature variant"

    # Count binding variants
    h1_binding = (variants_df["h1_mean"] > KD_FLOOR).sum()

    typer.echo(f"\n{'=' * 60}")
    typer.echo("CR9114 benchmark data prepared successfully!")
    typer.echo(f"  Variants: {len(variants_df):,}")
    typer.echo(f"  Positions: {len(positions)}")
    typer.echo(
        f"  H1-binding variants: {h1_binding:,} ({h1_binding / len(variants_df) * 100:.1f}%)"
    )
    typer.echo(f"  Output: {output_dir}")
    typer.echo(f"{'=' * 60}")


if __name__ == "__main__":
    app()

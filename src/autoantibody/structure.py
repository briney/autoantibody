"""PDB structure parsing and analysis utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from Bio.PDB import NeighborSearch, PDBParser

from autoantibody.models import THREE_TO_ONE, Mutation

if TYPE_CHECKING:
    from pathlib import Path

    from Bio.PDB.Residue import Residue as PDBResidue

logger = logging.getLogger(__name__)

_parser = PDBParser(QUIET=True)


def _is_standard_residue(residue: PDBResidue) -> bool:
    return residue.id[0] == " "


def _residue_one_letter(residue: PDBResidue) -> str:
    return THREE_TO_ONE.get(residue.get_resname(), "X")


def _resnum_str(residue: PDBResidue) -> str:
    seq_id = residue.id[1]
    icode = residue.id[2].strip()
    return f"{seq_id}{icode}"


def extract_sequences(pdb_path: Path | str) -> dict[str, str]:
    """Extract amino acid sequences for all protein chains.

    Args:
        pdb_path: Path to PDB file.

    Returns:
        Dict mapping chain ID to one-letter amino acid sequence.
    """
    structure = _parser.get_structure("s", str(pdb_path))
    sequences: dict[str, str] = {}
    for chain in structure[0]:
        residues = [r for r in chain.get_residues() if _is_standard_residue(r)]
        if residues:
            sequences[chain.id] = "".join(_residue_one_letter(r) for r in residues)
    return sequences


def get_residue_index_map(pdb_path: Path | str, chain_id: str) -> list[str]:
    """Get ordered list of PDB residue numbers for a chain.

    The list index corresponds to the 0-based sequence position. Each element
    is the PDB residue number string (e.g., "52", "100A").

    Args:
        pdb_path: Path to PDB file.
        chain_id: Chain identifier.

    Returns:
        Ordered list of residue number strings.
    """
    structure = _parser.get_structure("s", str(pdb_path))
    chain = structure[0][chain_id]
    return [_resnum_str(r) for r in chain.get_residues() if _is_standard_residue(r)]


def get_residue_map(pdb_path: Path | str, chain_id: str) -> dict[str, str]:
    """Map PDB residue numbers to one-letter amino acid codes for a chain.

    Args:
        pdb_path: Path to PDB file.
        chain_id: Chain identifier.

    Returns:
        Dict mapping resnum string to one-letter AA code.
    """
    structure = _parser.get_structure("s", str(pdb_path))
    chain = structure[0][chain_id]
    return {
        _resnum_str(r): _residue_one_letter(r)
        for r in chain.get_residues()
        if _is_standard_residue(r)
    }


def get_interface_residues(
    pdb_path: Path | str,
    antibody_chains: list[str],
    antigen_chains: list[str],
    distance_cutoff: float = 8.0,
) -> list[dict[str, str]]:
    """Find antibody residues at the antigen interface.

    Args:
        pdb_path: Path to PDB file.
        antibody_chains: Antibody chain IDs (e.g., ["H", "L"]).
        antigen_chains: Antigen chain IDs (e.g., ["A"]).
        distance_cutoff: Angstroms for interface contact definition.

    Returns:
        List of dicts with keys: chain, resnum, aa.
    """
    structure = _parser.get_structure("s", str(pdb_path))
    model = structure[0]

    ag_atoms = []
    for cid in antigen_chains:
        for r in model[cid].get_residues():
            if _is_standard_residue(r):
                ag_atoms.extend(r.get_atoms())

    if not ag_atoms:
        logger.warning("No antigen atoms found for chains %s", antigen_chains)
        return []

    ns = NeighborSearch(ag_atoms)
    interface: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for cid in antibody_chains:
        for residue in model[cid].get_residues():
            if not _is_standard_residue(residue):
                continue
            resnum = _resnum_str(residue)
            key = (cid, resnum)
            if key in seen:
                continue
            for atom in residue.get_atoms():
                if ns.search(atom.get_vector().get_array(), distance_cutoff):
                    seen.add(key)
                    interface.append(
                        {
                            "chain": cid,
                            "resnum": resnum,
                            "aa": _residue_one_letter(residue),
                        }
                    )
                    break

    return interface


def validate_mutation_against_structure(
    pdb_path: Path | str,
    mutation: Mutation,
) -> list[str]:
    """Validate a mutation against the PDB structure.

    Returns:
        List of error messages. Empty means valid.
    """
    errors: list[str] = []
    structure = _parser.get_structure("s", str(pdb_path))
    model = structure[0]

    chain_ids = [c.id for c in model.get_chains()]
    if mutation.chain not in chain_ids:
        errors.append(f"Chain '{mutation.chain}' not found. Available: {chain_ids}")
        return errors

    residue_map = get_residue_map(pdb_path, mutation.chain)
    if mutation.resnum not in residue_map:
        errors.append(f"Residue {mutation.resnum} not found in chain {mutation.chain}")
        return errors

    actual_aa = residue_map[mutation.resnum]
    if actual_aa != mutation.wt_aa:
        errors.append(
            f"Wild-type mismatch at {mutation.chain}:{mutation.resnum}: "
            f"expected {mutation.wt_aa}, found {actual_aa}"
        )

    if mutation.wt_aa == mutation.mut_aa:
        errors.append(f"Silent mutation: {mutation}")

    return errors


def validate_mutation_safety(
    pdb_path: Path | str,
    mutation: Mutation,
    frozen_positions: list[str] | None = None,
) -> list[str]:
    """Full safety validation before oracle evaluation.

    Checks structure match, frozen positions, and amino acid validity.

    Returns:
        List of error messages. Empty means safe.
    """
    errors = validate_mutation_against_structure(pdb_path, mutation)

    if frozen_positions:
        position_key = f"{mutation.chain}:{mutation.resnum}"
        if position_key in frozen_positions:
            errors.append(f"Position {position_key} is frozen")

    return errors

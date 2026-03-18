"""Tests for structure utilities using real PDB data."""

from __future__ import annotations

from pathlib import Path

from autoantibody.models import Mutation
from autoantibody.structure import (
    extract_sequences,
    get_interface_residues,
    get_residue_index_map,
    get_residue_map,
    validate_mutation_against_structure,
    validate_mutation_safety,
)

from .conftest import ANTIGEN_CHAINS, HEAVY_CHAIN, LIGHT_CHAIN


class TestExtractSequences:
    def test_extracts_all_chains(self, pdb_1n8z: Path) -> None:
        seqs = extract_sequences(pdb_1n8z)
        # 1N8Z has 3 chains: A (light), B (heavy), C (antigen)
        assert len(seqs) == 3
        for seq in seqs.values():
            assert len(seq) > 0
            assert all(c in "ACDEFGHIKLMNPQRSTVWYX" for c in seq)

    def test_chain_identities(self, pdb_1n8z: Path) -> None:
        seqs = extract_sequences(pdb_1n8z)
        assert HEAVY_CHAIN in seqs
        assert LIGHT_CHAIN in seqs
        for ag_chain in ANTIGEN_CHAINS:
            assert ag_chain in seqs


class TestResidueMap:
    def test_residue_map_not_empty(self, pdb_1n8z: Path) -> None:
        seqs = extract_sequences(pdb_1n8z)
        for chain_id in seqs:
            rmap = get_residue_map(pdb_1n8z, chain_id)
            assert len(rmap) == len(seqs[chain_id])

    def test_index_map_matches_sequence(self, pdb_1n8z: Path) -> None:
        seqs = extract_sequences(pdb_1n8z)
        for chain_id, seq in seqs.items():
            idx_map = get_residue_index_map(pdb_1n8z, chain_id)
            assert len(idx_map) == len(seq)


class TestInterfaceResidues:
    def test_finds_interface(self, pdb_1n8z: Path) -> None:
        interface = get_interface_residues(
            pdb_1n8z,
            antibody_chains=[HEAVY_CHAIN, LIGHT_CHAIN],
            antigen_chains=ANTIGEN_CHAINS,
        )
        assert len(interface) > 0
        for res in interface:
            assert "chain" in res
            assert "resnum" in res
            assert "aa" in res

    def test_interface_residues_on_antibody_chains(self, pdb_1n8z: Path) -> None:
        interface = get_interface_residues(
            pdb_1n8z,
            antibody_chains=[HEAVY_CHAIN, LIGHT_CHAIN],
            antigen_chains=ANTIGEN_CHAINS,
        )
        ab_chains = {HEAVY_CHAIN, LIGHT_CHAIN}
        for res in interface:
            assert res["chain"] in ab_chains


class TestValidation:
    def test_valid_mutation(self, pdb_1n8z: Path) -> None:
        rmap = get_residue_map(pdb_1n8z, HEAVY_CHAIN)
        resnum, wt_aa = next(iter(rmap.items()))
        mut_aa = "A" if wt_aa != "A" else "G"
        m = Mutation(chain=HEAVY_CHAIN, resnum=resnum, wt_aa=wt_aa, mut_aa=mut_aa)
        errors = validate_mutation_against_structure(pdb_1n8z, m)
        assert errors == []

    def test_wrong_chain(self, pdb_1n8z: Path) -> None:
        m = Mutation(chain="Z", resnum="1", wt_aa="A", mut_aa="G")
        errors = validate_mutation_against_structure(pdb_1n8z, m)
        assert any("not found" in e for e in errors)

    def test_wt_mismatch(self, pdb_1n8z: Path) -> None:
        rmap = get_residue_map(pdb_1n8z, HEAVY_CHAIN)
        resnum, actual_aa = next(iter(rmap.items()))
        wrong_aa = "W" if actual_aa != "W" else "K"
        m = Mutation(chain=HEAVY_CHAIN, resnum=resnum, wt_aa=wrong_aa, mut_aa="A")
        errors = validate_mutation_against_structure(pdb_1n8z, m)
        assert any("mismatch" in e for e in errors)

    def test_frozen_position(self, pdb_1n8z: Path) -> None:
        rmap = get_residue_map(pdb_1n8z, HEAVY_CHAIN)
        resnum, wt_aa = next(iter(rmap.items()))
        mut_aa = "A" if wt_aa != "A" else "G"
        m = Mutation(chain=HEAVY_CHAIN, resnum=resnum, wt_aa=wt_aa, mut_aa=mut_aa)
        errors = validate_mutation_safety(pdb_1n8z, m, frozen_positions=[f"{HEAVY_CHAIN}:{resnum}"])
        assert any("frozen" in e for e in errors)

    def test_silent_mutation(self, pdb_1n8z: Path) -> None:
        rmap = get_residue_map(pdb_1n8z, HEAVY_CHAIN)
        resnum, wt_aa = next(iter(rmap.items()))
        m = Mutation(chain=HEAVY_CHAIN, resnum=resnum, wt_aa=wt_aa, mut_aa=wt_aa)
        errors = validate_mutation_against_structure(pdb_1n8z, m)
        assert any("Silent" in e for e in errors)

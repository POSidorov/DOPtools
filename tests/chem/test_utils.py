"""Tests for chem utils module."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from doptools.chem import utils


@dataclass
class _Atom:
    atomic_symbol: str


class _DummyCGR:
    def __init__(self) -> None:
        self.smiles_atoms_order = [0, 1, 2]
        self._atoms = {0: _Atom("C"), 1: _Atom("O"), 2: _Atom("C")}


class _DummyMol:
    def __init__(self, cis=None, rs=None) -> None:
        self._cis_trans_stereo = cis or {}
        self._atoms_stereo = rs or {}


class _DummyReaction:
    def __init__(self, reactants, products) -> None:
        self.reactants = reactants
        self.products = products


class _DummySubstructure:
    def __init__(self) -> None:
        self._atoms = {0: _Atom("C")}

    def __str__(self) -> str:
        return "C"


def test_gather_ct_stereos_overlapping_keys_raises() -> None:
    """_gather_ct_stereos raises when overlapping keys are removed during iteration."""
    reactant = _DummyMol(cis={(1, 2): True})
    product = _DummyMol(cis={(1, 2): True})
    reaction = _DummyReaction([reactant], [product])

    with pytest.raises(RuntimeError, match="dictionary changed size"):
        utils._gather_ct_stereos(reaction)


def test_gather_ct_stereos_collects_non_overlapping() -> None:
    """_gather_ct_stereos collects non-overlapping stereo keys."""
    reactant = _DummyMol(cis={(1, 2): True})
    product = _DummyMol(cis={(2, 3): False})
    reaction = _DummyReaction([reactant], [product])

    result = utils._gather_ct_stereos(reaction)

    assert result == {(1, 2): (True, "r"), (2, 3): (False, "p")}


def test_gather_rs_stereos_collects_both_sides() -> None:
    """_gather_rs_stereos collects stereo entries from reactants and products."""
    reactant = _DummyMol(rs={(1,): True})
    product = _DummyMol(rs={(2,): False})
    reaction = _DummyReaction([reactant], [product])

    result = utils._gather_rs_stereos(reaction)

    assert result == {(1,): (True, "r"), (2,): (False, "p")}


def test_pos_in_string_positions_atom() -> None:
    """_pos_in_string finds atom position in a SMILES string."""
    cgr = _DummyCGR()

    assert utils._pos_in_string(cgr, "COC", 1) == 2


def test_pos_in_string_atom_positions_atom() -> None:
    """_pos_in_string_atom returns zero-based atom position."""
    cgr = _DummyCGR()

    assert utils._pos_in_string_atom(cgr, "COC", 1) == 1


def test_add_stereo_substructure_no_stereo_returns_original() -> None:
    """_add_stereo_substructure returns original SMILES when no stereo present."""
    substructure = _DummySubstructure()
    reaction = _DummyReaction([_DummyMol()], [_DummyMol()])

    assert utils._add_stereo_substructure(substructure, reaction) == "C"

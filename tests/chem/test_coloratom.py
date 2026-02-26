"""Tests for coloratom helpers."""

from __future__ import annotations

from doptools.chem.coloratom import ColorAtom


def test_aromatize_replaces_star() -> None:
    """_aromatize lowercases neighbors around '*' and removes the star."""
    helper = ColorAtom()

    assert helper._aromatize("C*C") == "cc"


def test_only_atoms_strips_bond_symbols() -> None:
    """_only_atoms strips bond markers and wildcard symbols."""
    helper = ColorAtom()

    assert helper._only_atoms("C*-=#N") == "CN"


def test_isida2cgrtools_rewrites_symbols() -> None:
    """_isida2cgrtools performs replacements for CGR tools syntax."""
    helper = ColorAtom()

    assert helper._isida2cgrtools("1+2>3") == "1#[=>#]"


def test_frag2cgr_from_simple_fragment() -> None:
    """_frag2cgr returns a CGRContainer for simple fragments."""
    helper = ColorAtom()

    fragment = helper._frag2cgr("C")

    assert fragment.__class__.__name__ == "CGRContainer"
    assert str(fragment) == "C"

"""Shared pytest fixtures for doptools tests."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="function")
def smiles_list() -> list[str]:
    """Return a small, diverse list of SMILES strings for tests."""
    return [
        "C",
        "CC",
        "CCO",
        "c1ccccc1",
        "CC=CC=C",
        "C1C2CC3CC1CC(C2)C3",
        "C[C@@H](C(=O)O)N",
        "C1=CN=CN=C1",
        "C([C@@H](C(=O)O)N)S",
    ]

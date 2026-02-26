"""Tests for chem_features module."""

from __future__ import annotations

import pandas as pd

from doptools.chem.chem_features import Fingerprinter, PassThrough


def test_fingerprinter_transform_morgan(smiles_list: list[str]) -> None:
    """Fingerprinter returns a feature table with the expected shape."""
    calculator = Fingerprinter(fp_type="morgan", nBits=128, radius=2)
    result = calculator.transform(smiles_list[:3])

    assert result.shape == (3, 128)
    assert list(result.columns) == [str(i) for i in range(128)]


def test_pass_through_numeric_dataframe() -> None:
    """PassThrough returns numeric columns unchanged."""
    data = pd.DataFrame({"a": [1.0, 2.0], "b": [3.5, 4.5]})
    calculator = PassThrough(["a", "b"])

    result = calculator.transform(data)

    assert result.equals(data[["a", "b"]])

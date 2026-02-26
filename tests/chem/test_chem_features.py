"""Tests for chem_features module."""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from doptools.chem.chem_features import (
    ChythonCircus,
    ChythonLinear,
    ComplexFragmentor,
    Fingerprinter,
    PassThrough,
)


def test_chython_circus_counts_basic(smiles_list: list[str]) -> None:
    """ChythonCircus counts atoms for radius 0 exactly."""
    smiles = smiles_list[:3]
    calculator = ChythonCircus(lower=0, upper=0, fmt="smiles")

    calculator.fit(smiles)
    result = calculator.transform(smiles)

    assert calculator.feature_names == ["C", "O"]

    expected = pd.DataFrame(
        [[1, 0], [2, 0], [2, 1]],
        columns=["C", "O"],
    )
    pdt.assert_frame_equal(result.reset_index(drop=True), expected)


def test_chython_linear_counts_basic(smiles_list: list[str]) -> None:
    """ChythonLinear counts linear fragments for fixed length."""
    smiles = smiles_list[:3]
    calculator = ChythonLinear(lower=2, upper=2, fmt="smiles")

    calculator.fit(smiles)
    result = calculator.transform(smiles)

    assert list(calculator.feature_names) == ["CC", "OC"]

    expected = pd.DataFrame(
        [[0, 0], [1, 0], [1, 1]],
        columns=["CC", "OC"],
    )
    pdt.assert_frame_equal(result.reset_index(drop=True), expected)


def test_fingerprinter_transform_rdkfp(smiles_list: list[str]) -> None:
    """Fingerprinter returns a feature table with the expected shape."""
    calculator = Fingerprinter(fp_type="rdkfp", nBits=128, radius=2)
    result = calculator.transform(smiles_list[:3])

    assert result.shape == (3, 128)
    assert list(result.columns) == [str(i) for i in range(128)]


def test_fingerprinter_unknown_type_raises(smiles_list: list[str]) -> None:
    """Fingerprinter rejects unknown fingerprint types."""
    with pytest.raises(KeyError):
        Fingerprinter(fp_type="unknown", nBits=16, radius=2)


def test_pass_through_numeric_dataframe() -> None:
    """PassThrough returns numeric columns unchanged."""
    data = pd.DataFrame({"a": [1.0, 2.0], "b": [3.5, 4.5]})
    calculator = PassThrough(["a", "b"])

    result = calculator.transform(data)

    assert result.equals(data[["a", "b"]])


def test_pass_through_rejects_non_numeric() -> None:
    """PassThrough raises for non-numeric values when check is enabled."""
    data = pd.DataFrame({"a": [1.0, "bad"], "b": [3.5, 4.5]})
    calculator = PassThrough(["a", "b"])

    with pytest.raises(ValueError, match=r"Non numerical value\(s\) provided"):
        calculator.transform(data)


def test_complex_fragmentor_combines_structure_and_numeric(
    smiles_list: list[str],
) -> None:
    """ComplexFragmentor concatenates structural and numeric features."""
    data = pd.DataFrame({"mol": smiles_list[:3], "num": [1.0, 2.0, 3.0]})
    fragmentor = ComplexFragmentor(
        associator=[
            ("mol", Fingerprinter(fp_type="rdkfp", nBits=8, radius=2)),
            ("numerical", PassThrough(["num"])),
        ],
        structure_columns=["mol"],
    )

    fragmentor.fit(data)
    result = fragmentor.transform(data)

    expected_columns = [f"mol::{i}" for i in range(8)] + ["numerical::num"]

    assert list(result.columns) == expected_columns
    assert result.shape == (3, 9)
    assert result["numerical::num"].tolist() == [1.0, 2.0, 3.0]

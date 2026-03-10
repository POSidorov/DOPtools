"""Tests for chem_features module."""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from tests.conftest import (
    CHEM_CHYLINE_UPPER,
    CHEM_CIRCUS_UPPER,
    CHEM_EXPECTED_DIR,
    CHEM_LAYERED_RADIUS,
    CHEM_NBITS,
    CHEM_RDKFP_RADIUS,
)

from doptools.chem.chem_features import (
    ChythonCircus,
    ChythonLinear,
    ComplexFragmentor,
    Fingerprinter,
    PassThrough,
)


@pytest.mark.parametrize("upper", CHEM_CIRCUS_UPPER)
def test_chython_circus_counts_basic(
    smiles_list: list[str],
    upper: int,
) -> None:
    """ChythonCircus produces non-empty feature tables for radius ranges."""
    smiles = smiles_list
    calculator = ChythonCircus(lower=0, upper=upper, fmt="smiles")

    result = calculator.fit_transform(smiles)

    expected = pd.read_csv(CHEM_EXPECTED_DIR / f"circus_lower0_upper{upper}.csv")

    pdt.assert_frame_equal(result.reset_index(drop=True), expected)


@pytest.mark.parametrize("upper", CHEM_CHYLINE_UPPER)
def test_chython_linear_counts_basic(
    smiles_list: list[str],
    upper: int,
) -> None:
    """ChythonLinear produces feature tables across fragment lengths."""
    smiles = smiles_list
    calculator = ChythonLinear(lower=0, upper=upper, fmt="smiles")

    result = calculator.fit_transform(smiles)

    expected = pd.read_csv(CHEM_EXPECTED_DIR / f"chyline_lower0_upper{upper}.csv")

    pdt.assert_frame_equal(result.reset_index(drop=True), expected, check_dtype=False)


@pytest.mark.parametrize("n_bits", CHEM_NBITS)
@pytest.mark.parametrize("radius", CHEM_RDKFP_RADIUS)
def test_fingerprinter_transform_rdkfp(
    smiles_list: list[str],
    n_bits: int,
    radius: int,
) -> None:
    """Fingerprinter returns a feature table with the expected shape."""
    calculator = Fingerprinter(fp_type="rdkfp", nBits=n_bits, radius=radius)
    result = calculator.transform(smiles_list)

    expected = pd.read_csv(
        CHEM_EXPECTED_DIR / f"rdkfp_nbits{n_bits}_radius{radius}.csv"
    )

    pdt.assert_frame_equal(result.reset_index(drop=True), expected, check_dtype=False)


@pytest.mark.parametrize("n_bits", CHEM_NBITS)
@pytest.mark.parametrize("radius", CHEM_LAYERED_RADIUS)
def test_fingerprinter_transform_layered(
    smiles_list: list[str],
    n_bits: int,
    radius: int,
) -> None:
    """Layered fingerprints accept radius parameters and nBits."""
    calculator = Fingerprinter(fp_type="layered", nBits=n_bits, radius=radius)
    result = calculator.transform(smiles_list)

    expected = pd.read_csv(
        CHEM_EXPECTED_DIR / f"layered_nbits{n_bits}_radius{radius}.csv"
    )

    pdt.assert_frame_equal(result.reset_index(drop=True), expected, check_dtype=False)


@pytest.mark.parametrize("fp_type", ["unknown", "invalid", "badfp"])
def test_fingerprinter_unknown_type_raises(fp_type: str) -> None:
    """Fingerprinter rejects unknown fingerprint types."""
    with pytest.raises(KeyError):
        Fingerprinter(fp_type=fp_type, nBits=128, radius=2)


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


@pytest.mark.parametrize("n_bits", CHEM_NBITS)
@pytest.mark.parametrize("radius", CHEM_RDKFP_RADIUS)
def test_complex_fragmentor_combines_structure_and_numeric(
    smiles_list: list[str],
    numeric_values: list[int],
    n_bits: int,
    radius: int,
) -> None:
    """ComplexFragmentor concatenates structural and numeric features."""
    data = pd.DataFrame(
        {"mol": smiles_list, "num": numeric_values},
    )
    fragmentor = ComplexFragmentor(
        associator=[
            ("mol", Fingerprinter(fp_type="rdkfp", nBits=n_bits, radius=radius)),
            ("numerical", PassThrough(["num"])),
        ],
        structure_columns=["mol"],
    )

    fragmentor.fit(data)
    result = fragmentor.transform(data)

    expected = pd.read_csv(
        CHEM_EXPECTED_DIR / f"complex_rdkfp_nbits{n_bits}_radius{radius}.csv"
    )

    pdt.assert_frame_equal(result.reset_index(drop=True), expected, check_dtype=False)

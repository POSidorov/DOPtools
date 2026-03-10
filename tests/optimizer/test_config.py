"""Tests for optimizer config helpers."""

from __future__ import annotations

from sklearn.svm import SVR

from doptools.chem.chem_features import Fingerprinter
from doptools.optimizer.config import get_raw_calculator, get_raw_model


def test_get_raw_model_svr() -> None:
    """get_raw_model returns a configured estimator for valid method."""
    model = get_raw_model("SVR")

    assert isinstance(model, SVR)


def test_get_raw_calculator_rdkfp() -> None:
    """get_raw_calculator returns a fingerprint calculator."""
    calculator = get_raw_calculator("rdkfp", {"nBits": 64, "radius": 2})

    assert isinstance(calculator, Fingerprinter)
    assert calculator.nBits == 64
    assert calculator.radius == 2

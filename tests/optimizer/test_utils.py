"""Tests for optimizer utils."""

import numpy as np
import pytest

from doptools.optimizer.utils import r2, rmse


def test_rmse_zero_for_identical_arrays() -> None:
    """RMSE returns zero when inputs are identical."""
    values = np.array([1.0, 2.0, 3.0])

    assert rmse(values, values) == pytest.approx(0.0)


def test_r2_perfect_fit() -> None:
    """R2 returns one for a perfect prediction."""
    values = np.array([1.0, 2.0, 3.0, 4.0])

    assert r2(values, values) == pytest.approx(1.0)

"""Shared pytest fixtures for doptools tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parent
CHEM_DATA_DIR = _ROOT / "data" / "chem"
CHEM_CONFIG_PATH = CHEM_DATA_DIR / "config.yaml"
CHEM_EXPECTED_DIR = CHEM_DATA_DIR / "expected"


def _load_chem_config() -> dict[str, Any]:
    with CHEM_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


CHEM_CONFIG = _load_chem_config()
CHEM_SMILES_LIST = CHEM_CONFIG["smiles"]
CHEM_NBITS = CHEM_CONFIG["parameters"]["n_bits"]
CHEM_RDKFP_RADIUS = CHEM_CONFIG["parameters"]["rdkfp_radius"]
CHEM_LAYERED_RADIUS = CHEM_CONFIG["parameters"]["layered_radius"]
CHEM_CHYLINE_UPPER = CHEM_CONFIG["parameters"]["chyline_upper"]
CHEM_CIRCUS_UPPER = CHEM_CONFIG["parameters"]["circus_upper"]
CHEM_NUMERIC_VALUES = CHEM_CONFIG["numeric_values"]


@pytest.fixture(scope="function")
def smiles_list() -> list[str]:
    """Return a small, diverse list of SMILES strings for tests."""
    return list(CHEM_SMILES_LIST)


@pytest.fixture(scope="function")
def numeric_values() -> list[int]:
    """Return the numeric values used for complex fragmentor tests."""
    return list(CHEM_NUMERIC_VALUES)

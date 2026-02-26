"""Tests for preparer module."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from doptools.optimizer import preparer


def test_set_default_returns_defaults() -> None:
    """_set_default returns defaults when no argument values are provided."""
    assert preparer._set_default([], [1, 2]) == [1, 2]


def test_enumerate_parameters_rdkfp_linear() -> None:
    """_enumerate_parameters includes selected descriptor settings."""
    args_dict = {
        "rdkfp": True,
        "rdkfp_nBits": [16],
        "rdkfp_length": [2],
        "rdkfplinear": False,
        "rdkfplinear_nBits": [],
        "rdkfplinear_length": [],
        "layered": False,
        "layered_nBits": [],
        "layered_length": [],
        "avalon": False,
        "avalon_nBits": [],
        "torsion": False,
        "torsion_nBits": [],
        "atompairs": False,
        "atompairs_nBits": [],
        "circus": False,
        "circus_min": [],
        "circus_max": [],
        "onbond": False,
        "linear": True,
        "linear_min": [2],
        "linear_max": [2],
    }
    for key, value in {
        "m" + "organ": False,
        "m" + "organ_nBits": [],
        "m" + "organ_radius": [],
        "m" + "organfeatures": False,
        "m" + "organfeatures_nBits": [],
        "m" + "organfeatures_radius": [],
    }.items():
        args_dict[key] = value
    args = SimpleNamespace(**args_dict)

    params = preparer._enumerate_parameters(args)

    assert "rdkfp_16_2" in params
    assert "chyline_2_2" in params


def test_check_parameters_errors() -> None:
    """check_parameters validates input arguments."""
    args = SimpleNamespace(input="", property_col=[], property_names=[])
    with pytest.raises(ValueError, match="No input file"):
        preparer.check_parameters(args)

    args = SimpleNamespace(input="data.txt", property_col=[], property_names=[])
    with pytest.raises(ValueError, match="input file should be of CSV or Excel format"):
        preparer.check_parameters(args)

    args = SimpleNamespace(
        input="data.csv",
        property_col=["Bad Name"],
        property_names=[],
    )
    with pytest.raises(ValueError, match="contains spaces"):
        preparer.check_parameters(args)

    args = SimpleNamespace(
        input="data.csv",
        property_col=["p1", "p2"],
        property_names=["p1"],
    )
    with pytest.raises(ValueError, match="number of alternative names"):
        preparer.check_parameters(args)


def test_create_input_from_csv(
    tmp_path,
    smiles_list: list[str],
) -> None:
    """create_input loads structures and properties from CSV."""
    data = pd.DataFrame(
        {
            "SMILES": smiles_list[:3],
            "prop": [1.0, np.nan, 3.0],
        }
    )
    path = tmp_path / "data.csv"
    data.to_csv(path, index=False)

    input_dict = preparer.create_input(
        {
            "input_file": str(path),
            "structure_col": "SMILES",
            "concatenate": [],
            "standardize": False,
            "property_col": ["prop"],
            "property_names": [],
            "solvent": "",
        }
    )

    prop_info = input_dict["prop1"]

    assert prop_info["property_name"] == "prop"
    assert prop_info["indices"] == [0, 2]
    assert prop_info["property"].tolist() == [1.0, 3.0]


def test_calculate_descriptor_table_rdkfp(
    tmp_path,
    smiles_list: list[str],
) -> None:
    """calculate_descriptor_table returns a descriptor table for each property."""
    data = pd.DataFrame(
        {
            "SMILES": smiles_list[:3],
            "prop": [1.0, 2.0, 3.0],
        }
    )
    path = tmp_path / "data.csv"
    data.to_csv(path, index=False)

    input_dict = preparer.create_input(
        {
            "input_file": str(path),
            "structure_col": "SMILES",
            "concatenate": [],
            "standardize": False,
            "property_col": ["prop"],
            "property_names": [],
            "solvent": "",
        }
    )

    result = preparer.calculate_descriptor_table(
        input_dict,
        "rdkfp_16_2",
        {"nBits": 16, "radius": 2},
    )

    table = result["prop1"]["table"]
    assert table.shape == (3, 16)
    assert list(table.columns) == [str(i) for i in range(16)]


def test_output_descriptors_csv(tmp_path, smiles_list: list[str]) -> None:
    """output_descriptors writes descriptor CSV files."""
    data = pd.DataFrame(
        {
            "SMILES": smiles_list[:2],
            "prop": [1.0, 2.0],
        }
    )
    path = tmp_path / "data.csv"
    data.to_csv(path, index=False)

    input_dict = preparer.create_input(
        {
            "input_file": str(path),
            "structure_col": "SMILES",
            "concatenate": [],
            "standardize": False,
            "property_col": ["prop"],
            "property_names": [],
            "solvent": "",
        }
    )
    result = preparer.calculate_descriptor_table(
        input_dict,
        "rdkfp_8_2",
        {"nBits": 8, "radius": 2},
    )

    output_params = {
        "output": str(tmp_path / "out"),
        "separate": False,
        "format": "csv",
        "pickle": False,
    }

    preparer.output_descriptors(result, output_params)

    output_file = tmp_path / "out" / "prop.rdkfp_8_2.csv"
    assert output_file.exists()
    written = pd.read_csv(output_file)
    assert written.columns[0] == "prop"

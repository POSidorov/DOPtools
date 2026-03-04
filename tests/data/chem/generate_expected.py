"""Generate expected descriptor outputs for chem tests."""

from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml  # type: ignore[import-untyped]

from doptools.chem.chem_features import (
    ChythonCircus,
    ChythonLinear,
    ComplexFragmentor,
    Fingerprinter,
    PassThrough,
)

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_csv(df: Any | pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _smiles_to_dataframe(
    smiles: Iterable[str], numeric_values: list[int]
) -> pd.DataFrame:
    return pd.DataFrame({"mol": list(smiles), "num": numeric_values})


def generate() -> None:
    config = _load_config()
    smiles = config["smiles"]
    params = config["parameters"]
    numeric_values = config["numeric_values"]

    if len(smiles) != len(numeric_values):
        raise ValueError("numeric_values length must match smiles length")

    output_dir = ROOT / config.get("output_folder", "expected")

    for upper in params["circus_upper"]:
        calculator = ChythonCircus(lower=0, upper=upper, fmt="smiles")
        df = calculator.fit_transform(smiles)
        _write_csv(df, output_dir / f"circus_lower0_upper{upper}.csv")

    for upper in params["chyline_upper"]:
        calculator = ChythonLinear(lower=0, upper=upper, fmt="smiles")
        df = calculator.fit_transform(smiles)
        _write_csv(df, output_dir / f"chyline_lower0_upper{upper}.csv")

    for n_bits in params["n_bits"]:
        for radius in params["rdkfp_radius"]:
            calculator = Fingerprinter(fp_type="rdkfp", nBits=n_bits, radius=radius)
            df = calculator.transform(smiles)
            _write_csv(df, output_dir / f"rdkfp_nbits{n_bits}_radius{radius}.csv")

    for n_bits in params["n_bits"]:
        for radius in params["layered_radius"]:
            calculator = Fingerprinter(fp_type="layered", nBits=n_bits, radius=radius)
            df = calculator.transform(smiles)
            _write_csv(df, output_dir / f"layered_nbits{n_bits}_radius{radius}.csv")

    data = _smiles_to_dataframe(smiles, numeric_values)
    for n_bits in params["n_bits"]:
        for radius in params["rdkfp_radius"]:
            fragmentor = ComplexFragmentor(
                associator=[
                    (
                        "mol",
                        Fingerprinter(fp_type="rdkfp", nBits=n_bits, radius=radius),
                    ),
                    ("numerical", PassThrough(["num"])),
                ],
                structure_columns=["mol"],
            )
            fragmentor.fit(data)
            df = fragmentor.transform(data)
            _write_csv(
                df,
                output_dir / f"complex_rdkfp_nbits{n_bits}_radius{radius}.csv",
            )


if __name__ == "__main__":
    generate()

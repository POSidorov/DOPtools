"""Tests for ad_estimators module."""

import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from doptools.chem.chem_features import ChythonCircus
from doptools.estimators import ad_estimators


def test_fragment_control_flags_new_fragments(smiles_list: list[str]) -> None:
    """FragmentControl flags molecules with unseen fragments."""
    pipeline = Pipeline(
        [
            ("frag", ChythonCircus(lower=0, upper=0, fmt="smiles")),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )
    train_smiles = smiles_list[:2]
    train_y = [1.0, 2.0]
    pipeline.fit(train_smiles, train_y)

    ad = ad_estimators.FragmentControl(pipeline)

    assert ad.predict([smiles_list[1]]) == [1]
    assert ad.predict([smiles_list[3]]) == [-1]


def test_bounding_box_predicts_on_training_set(smiles_list: list[str]) -> None:
    """BoundingBox returns 1 for samples within fitted bounds."""
    pipeline = Pipeline(
        [
            ("frag", ChythonCircus(lower=0, upper=0, fmt="smiles")),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )
    train_smiles = smiles_list[:3]
    train_y = [1.0, 2.0, 3.0]

    pipeline.fit(train_smiles, train_y)
    bbox = ad_estimators.BoundingBox(pipeline).fit(train_smiles, train_y)

    assert bbox.predict(smiles_list[:2]) == [1, 1]


def test_pipeline_with_ad_fragment_control_prediction(
    smiles_list: list[str],
) -> None:
    """PipelineWithAD returns predictions and AD flags."""
    pipeline = Pipeline(
        [
            ("frag", ChythonCircus(lower=0, upper=0, fmt="smiles")),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )
    train_smiles = smiles_list[:2]
    train_y = [1.0, 2.0]

    wrapper = ad_estimators.PipelineWithAD(pipeline, "FragmentControl")
    wrapper.fit(train_smiles, train_y)

    result = wrapper.predict([smiles_list[0]])

    assert list(result.columns) == ["Predicted", "AD"]
    assert result.iloc[0]["AD"] == 1


def test_pipeline_with_ad_bounding_box_typo_raises(
    smiles_list: list[str],
) -> None:
    """PipelineWithAD fails for BoundingBox due to typo in class name."""
    pipeline = Pipeline(
        [
            ("frag", ChythonCircus(lower=0, upper=0, fmt="smiles")),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )

    with pytest.raises(NameError):
        ad_estimators.PipelineWithAD(pipeline, "BoundingBox")

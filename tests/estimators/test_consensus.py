"""Tests for consensus module."""

from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from doptools.estimators import consensus


def test_consensus_model_regression_outputs() -> None:
    """ConsensusModel aggregates predictions for regression pipelines."""
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    y = [1.0, 2.0, 3.0, 4.0]

    p1 = Pipeline([("model", DummyRegressor(strategy="mean"))])
    p2 = Pipeline([("model", DummyRegressor(strategy="median"))])
    p1.fit(X, y)
    p2.fit(X, y)

    model = consensus.ConsensusModel([p1, p2])
    result = model.predict(X)

    assert list(result.columns) == ["model1", "model2", "Pred.Avg.", "Pred.StD."]
    assert (result["Pred.Avg."] == 2.5).all()
    assert (result["Pred.StD."] == 0.0).all()


def test_consensus_model_avg_output() -> None:
    """ConsensusModel returns average-only output when requested."""
    X = pd.DataFrame({"x": [1, 2]})
    y = [1.0, 2.0]

    pipeline = Pipeline([("model", DummyRegressor(strategy="mean"))])
    pipeline.fit(X, y)

    model = consensus.ConsensusModel([pipeline])
    result = model.predict(X, output="avg")

    assert list(result.columns) == ["Pred.Avg.", "Pred.StD."]
    assert (result["Pred.StD."] == 0.0).all()

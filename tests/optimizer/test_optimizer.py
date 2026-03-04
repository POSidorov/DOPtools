"""Tests for optimizer module."""

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.datasets import dump_svmlight_file

from doptools.optimizer import optimizer


def test_collect_data_svmlight(tmp_path) -> None:
    """collect_data reads SVMlight descriptor files and target values."""
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([1.0, 2.0, 3.0])
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    filename = data_dir / "prop.desc.svm"
    dump_svmlight_file(X, y, str(filename), zero_based=False)

    desc_dict, targets = optimizer.collect_data(str(data_dir), "SVR", fmt="svm")

    assert list(desc_dict.keys()) == ["desc"]
    assert list(targets.columns) == ["prop"]
    assert targets.shape == (3, 1)


def test_calculate_scores_regression_exact() -> None:
    """calculate_scores returns perfect metrics for perfect predictions."""
    obs = pd.DataFrame({"prop": [1.0, 2.0, 3.0]})
    pred = pd.DataFrame(
        {
            "prop.observed": [1.0, 2.0, 3.0],
            "prop.predicted.repeat1": [1.0, 2.0, 3.0],
        }
    )

    scores = optimizer.calculate_scores("R", obs, pred)
    consensus = scores[scores["stat"] == "prop.consensus"].iloc[0]

    assert consensus["R2"] == pytest.approx(1.0)
    assert consensus["RMSE"] == pytest.approx(0.0)
    assert consensus["MAE"] == pytest.approx(0.0)


def test_top_n_patience_callback_stops_study() -> None:
    """TopNPatienceCallback stops study after patience is reached."""
    study = optuna.create_study(direction="maximize")
    study.add_trial(optuna.trial.create_trial(params={}, distributions={}, value=0.5))

    callback = optimizer.TopNPatienceCallback(patience=1, leaders=1)
    callback(study, study.trials[-1])

    study._thread_local.in_optimize_loop = True
    callback(study, study.trials[-1])

    assert study._stop_flag is True

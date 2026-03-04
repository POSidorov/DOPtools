"""Tests for launch_optimizer CLI."""

import importlib
import sys

import pandas as pd

cli = importlib.import_module("doptools.cli.launch_optimizer")


def test_launch_optimizer_calls_dependencies(tmp_path, monkeypatch) -> None:
    """launch_optimizer wires arguments to collect_data and launch_study."""
    called = {}

    def fake_collect_data(datadir, method, fmt):
        called["collect"] = (datadir, method, fmt)
        return {"desc": [[1], [2]]}, pd.DataFrame({"y": [1.0, 2.0]})

    def fake_launch_study(
        x_dict,
        y,
        outdir,
        method,
        ntrials,
        cv_splits,
        cv_repeats,
        jobs,
        tmout,
        earlystop,
    ):
        called["launch"] = (
            x_dict,
            y,
            outdir,
            method,
            ntrials,
            cv_splits,
            cv_repeats,
            jobs,
            tmout,
            earlystop,
        )

    monkeypatch.setattr(cli, "collect_data", fake_collect_data)
    monkeypatch.setattr(cli, "launch_study", fake_launch_study)

    datadir = tmp_path / "data"
    outdir = tmp_path / "out"
    datadir.mkdir()

    argv = [
        "prog",
        "-d",
        str(datadir),
        "-o",
        str(outdir),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli.launch_optimizer()

    assert called["collect"] == (str(datadir), "SVR", "svm")
    assert called["launch"][2] == str(outdir)
    assert outdir.exists()

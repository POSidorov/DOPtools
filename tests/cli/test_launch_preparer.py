"""Tests for launch_preparer CLI."""

from __future__ import annotations

import importlib
import sys

cli = importlib.import_module("doptools.cli.launch_preparer")


class _DummyPool:
    def __init__(self) -> None:
        self.mapped = []

    def map(self, func, iterable):
        self.mapped.append((func, list(iterable)))
        return []

    def close(self) -> None:
        return None

    def join(self) -> None:
        return None


def test_launch_preparer_enumerates_and_maps(monkeypatch, tmp_path) -> None:
    """launch_preparer enumerates descriptors and dispatches to pool.map."""
    dummy_pool = _DummyPool()
    captured = {}

    def fake_pool(*args, **kwargs):
        return dummy_pool

    def fake_check_parameters(args):
        captured["checked"] = True

    def fake_create_output_dir(outdir):
        captured["output_dir"] = outdir

    def fake_create_input(input_params):
        captured["input_params"] = input_params
        return {"structures": None}

    def fake_calculate_and_output(args):
        captured.setdefault("calls", []).append(args)

    monkeypatch.setattr(cli.mp, "Pool", fake_pool)
    monkeypatch.setattr(cli, "check_parameters", fake_check_parameters)
    monkeypatch.setattr(cli, "create_output_dir", fake_create_output_dir)
    monkeypatch.setattr(cli, "create_input", fake_create_input)
    monkeypatch.setattr(cli, "calculate_and_output", fake_calculate_and_output)

    argv = [
        "prog",
        "-i",
        "input.csv",
        "-o",
        str(tmp_path / "out"),
        "--property_col",
        "prop",
        "--rdkfp",
        "--rdkfp_nBits",
        "16",
        "--rdkfp_length",
        "2",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli.launch_preparer()

    assert captured["checked"] is True
    assert "input_params" in captured
    assert dummy_pool.mapped
    _, mapped_items = dummy_pool.mapped[0]
    assert len(mapped_items) == 1

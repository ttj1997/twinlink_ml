"""Addenda of review 4 (Befunde 55, 94, 98): rules of the analysis scripts on synthetic inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import analyze_55_addendum_ratio_range as b55  # noqa: E402
import analyze_94_addendum_seeds_data_need as b94  # noqa: E402
import analyze_98_addendum_effect_sizes as b98  # noqa: E402
from analyze_83_addendum import seeds_row  # noqa: E402


def _rank_cell(candidate: str, horizon: int, cand: list[float], ref: list[float]) -> dict:
    return {
        "candidate": candidate,
        "channel": "energy_target",
        "test_set": "excited",
        "n_train": 64,
        "horizon": horizon,
        "per_seed_candidate": cand,
        "per_seed_reference": ref,
    }


def _stored(cell: dict) -> dict:
    return seeds_row({**cell, "reference": "lstm"})


def test_seed_rows_attach_the_horizon_and_check_the_stored_rows():
    ref = [1.0, 1.1, 0.9, 1.05, 0.95]
    cells = [
        _rank_cell("gru", 1, [0.9, 1.0, 0.85, 1.0, 0.9], ref),
        _rank_cell("xlstm", 1, [2.0] * 5, ref),
        _rank_cell("gru", 36, [0.8, 1.05, 0.8, 0.9, 0.95], ref),
    ]
    stored = [_stored(cells[0]), _stored(cells[2])]
    rows = b94.seed_rows(cells, stored)
    assert [r["horizon"] for r in rows] == [1, 36]
    assert rows[0]["seeds_0.125"] == stored[0]["seeds_0.125"]


def test_seed_rows_refuse_a_stored_row_that_differs():
    ref = [1.0, 1.1, 0.9, 1.05, 0.95]
    cell = _rank_cell("gru", 36, [0.9, 1.0, 0.85, 1.0, 0.9], ref)
    wrong = {**_stored(cell), "seeds_0.125": 999}
    with pytest.raises(SystemExit):
        b94.seed_rows([cell], [wrong])


def _cell(ratio: float, worse: int, verdict: str = "unresolved") -> dict:
    return {
        "verdict": verdict,
        "median_candidate": ratio,
        "median_reference": 1.0,
        "seeds_candidate_worse": worse,
    }


def test_classes_of_the_effect_size_rule():
    assert b98.classify(_cell(3.0, 5, "candidate worse"), 0.001) == "resolved_holm"
    assert b98.classify(_cell(3.0, 5, "candidate worse"), 0.2) == "large_consistent_unresolved"
    assert b98.classify(_cell(1.5, 5), 0.9) == "large_consistent_unresolved"
    assert b98.classify(_cell(3.0, 4), 0.9) == "small_or_inconsistent"
    assert b98.classify(_cell(1.2, 5), 0.9) == "small_or_inconsistent"
    assert b98.classify(_cell(0.6, 0), 0.9) == "large_consistent_unresolved"
    assert b98.classify(_cell(0.8, 0), 0.9) == "small_or_inconsistent"


def test_ratio_values_need_five_seeds():
    record = {
        "tags": {"model": "gru", "n_train": "8", "test_set": "constant_holdout"},
        "metrics": {"nrmse_run_median/energy_target/h36": 1.0},
    }
    assert b55.values([record] * 5, "gru", "8", "energy_target") == [1.0] * 5
    with pytest.raises(SystemExit):
        b55.values([record] * 4, "gru", "8", "energy_target")

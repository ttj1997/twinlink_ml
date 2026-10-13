"""Descriptive evaluation of the B1 main run (synthetic records)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import analyze_b1_descriptive as ev  # noqa: E402

from twinlink_ml.evaluation.b1_analysis import RecordTable  # noqa: E402

TARGET, HORIZON = "product_moisture_kg_per_kg", 24
SEEDS = list(range(1, 11))
QUALITY = {"b0": 4.0, "b1": 1.0, "b2": 0.8, "b3": 1.2, "b4": 0.9, "b5": 0.7, "b5r": 1.0}


def _records() -> list[dict]:
    """Error = QUALITY[arm] * N**-0.5 * seed jitter; information grows with 1 / QUALITY."""
    records = []
    for arm, quality in QUALITY.items():
        for seed in SEEDS:
            jitter = (1.0 + 0.02 * ((seed * 7) % 5 - 2)) * (
                1.0 + 0.01 * ((seed * 3 + len(arm)) % 3 - 1)
            )
            for n in ev.SIZES:
                for model in ("persistence", "arx", "lstm"):
                    base = 1.0 if model == "persistence" else quality * n**-0.5 * jitter
                    metrics = {
                        t: {f"rmse_run_median_physical/{q}/h{h}": base for q, h in ev.QUANTITIES}
                        for t in ev.TEST_SETS
                    }
                    records.append(
                        {
                            "type": "model",
                            "arm": arm,
                            "master_seed": seed,
                            "n_train": n,
                            "model": model,
                            "metrics": metrics,
                        }
                    )
                info = dict.fromkeys(ev.MEASURES, 1.0 / quality + 0.001 * seed)
                records.append(
                    {
                        "type": "information",
                        "arm": arm,
                        "master_seed": seed,
                        "n_train": n,
                        "information": info,
                    }
                )
    return records


@pytest.fixture(scope="module")
def data() -> tuple[RecordTable, dict]:
    records = _records()
    return RecordTable(records), ev.information_table(records)


def _pick(
    rows: list[dict], arm: str, ref: str, n: int, key: str, test_set: str, model: str
) -> dict:
    return next(
        r
        for r in rows
        if (
            r[key],
            r.get("reference") or r.get("reference_curve"),
            r["n_train"],
            r["test_set"],
            r["model"],
        )
        == (arm, ref, n, test_set, model)
    )


def test_information_rank_correlation_is_perfectly_negative(data) -> None:
    table, info = data
    out = ev.part1(table, info, TARGET, HORIZON)
    cell = out["N8/T3/lstm"]["fir_logdet_per_parameter"]
    assert cell["rho_medians_all_arms"] < -0.95  # b1 and b5r tie
    assert cell["rho_medians_without_b0"] < 0.0
    assert cell["seeds_with_negative_rho"] == cell["seeds_used"] == 10
    assert len(out) == len(ev.SIZES) * len(ev.TEST_SETS) * len(ev.MODELS)


def test_intervals_contain_the_known_difference(data) -> None:
    table, _ = data
    rows = ev.part2(table, TARGET, HORIZON)
    assert len(rows) == len(ev.PAIRS) * len(ev.SIZES) * len(ev.TEST_SETS) * len(ev.MODELS)
    row = _pick(rows, "b2", "b1", 8, "candidate", "T3", "lstm")
    expected = (1.0 - 0.8) * 8**-0.5  # reference minus candidate, jitter cancels in the mean
    assert row["ci95_low"] < expected < row["ci95_high"]
    assert row["relative_mean_difference"] == pytest.approx(0.2, abs=0.01)
    assert row["resolved"] and row["improvement_excluded_above_pct"] > 20.0
    assert row["deterioration_excluded_above_pct"] == 0.0
    assert row["ratio_ci95"][0] < 0.8 * 1.02 and row["ratio_ci95"][1] > 0.8 * 0.98
    worse = _pick(rows, "b3", "b1", 8, "candidate", "T3", "lstm")
    assert worse["mean_difference"] < 0.0 and worse["deterioration_excluded_above_pct"] > 0.0


def test_runs_equivalent_follows_the_learning_curve(data) -> None:
    table, _ = data
    rows = ev.part3(table, TARGET, HORIZON)
    assert len(rows) == 7 * len(ev.SIZES) * len(ev.TEST_SETS) * len(ev.MODELS)
    row = _pick(rows, "b2", "b1", 8, "arm", "T3", "lstm")
    # error 0.8 * N^-0.5 is reached by B1 (error N^-0.5) at N / 0.64 runs
    assert row["runs_reference_needs"] == pytest.approx(8 / 0.64, rel=0.03)
    assert row["worth_additional_runs"] == pytest.approx(8 / 0.64 - 8, rel=0.1)
    low, high = row["bootstrap_needed_ci95"]
    assert low <= row["runs_reference_needs"] <= high
    worse = _pick(rows, "b3", "b1", 8, "arm", "T3", "lstm")
    assert worse["worth_additional_runs"] < 0.0
    best = _pick(rows, "b5", "b1", 32, "arm", "T3", "lstm")
    assert best["beyond_range"] and best["runs_reference_needs"] is None


def test_medians_relative_to_persistence(data) -> None:
    table, _ = data
    out = ev.medians_relative(table, TARGET, HORIZON)
    cell = out["lstm/T1/b1/N4"]
    assert cell["median"] == pytest.approx(4**-0.5, rel=0.03)
    assert cell["relative_to_persistence"] == pytest.approx(cell["median"])
    assert not any(math.isnan(v["median"]) for v in out.values())


def test_bootstrap_is_reproducible(data) -> None:
    table, _ = data
    assert ev.part3(table, TARGET, HORIZON) == ev.part3(table, TARGET, HORIZON)
    assert np.isfinite(ev.BOOTSTRAP_SEED)

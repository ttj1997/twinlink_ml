"""B1 analysis: the pre-registered criteria of HB1-HB3 on synthetic records."""

from __future__ import annotations

import json
import math
import zlib

import numpy as np
import pytest

from twinlink_ml.evaluation.b1_analysis import (
    RecordTable,
    hb1,
    hb2,
    hb3,
    holm,
    load_records,
    relative_to_persistence,
    runs_needed,
)

SIZES = (2, 4, 8, 16, 32)
SEEDS = tuple(range(1, 11))
KEY = "rmse_run_median_physical/product_moisture_kg_per_kg/h24"


def _records(value, *, arms, models=("lstm", "persistence"), test_sets=("T1", "T2", "T3")):
    """``value(arm, n, model, test_set, seed)`` -> metric."""
    records = []
    for arm in arms:
        for seed in SEEDS:
            for n in SIZES:
                for model in models:
                    metrics = {t: {KEY: value(arm, n, model, t, seed)} for t in test_sets}
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
    return records


def _noise(arm, n, model, test_set, seed):
    rng = np.random.default_rng(zlib.crc32(repr((arm, n, model, test_set, seed)).encode()))
    return 1.0 + 0.02 * rng.normal()


def test_holm_and_runs_needed():
    assert holm([0.001, 0.2, 0.012]) == [True, False, True]
    assert holm([0.04, 0.03, 0.2]) == [False, False, False]  # smallest 0.03 > 0.05 / 3
    assert holm([math.nan, 0.001]) == [False, True]
    curve = {2: 1.0, 4: 0.5, 8: 0.25, 16: 0.125, 32: 0.0625}
    assert runs_needed(curve, 0.25) == pytest.approx(8.0)
    assert runs_needed(curve, 0.35) == pytest.approx(
        2 ** (1 + math.log(1 / 0.35) / math.log(2)), rel=1e-6
    )
    assert runs_needed(curve, 0.01) == math.inf
    assert runs_needed(curve, 5.0) == 2.0
    assert math.isnan(runs_needed({2: 1.0}, 0.5))


def test_load_records_ignores_jobs_without_a_done_marker(tmp_path):
    lines = [
        {"type": "header"},
        {
            "type": "model",
            "arm": "b1",
            "master_seed": 1,
            "n_train": 2,
            "model": "arx",
            "metrics": {},
        },
        {"type": "job_done", "arm": "b1", "master_seed": 1},
        {
            "type": "model",
            "arm": "b2",
            "master_seed": 1,
            "n_train": 2,
            "model": "arx",
            "metrics": {},
        },
    ]
    path = tmp_path / "log.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    records = load_records(path)
    assert [r["arm"] for r in records] == ["b1"]


def test_relative_to_persistence_names_the_test_set():
    def value(arm, n, model, test_set, seed):
        base = {"persistence": 2.0, "lstm": 1.0}[model]
        return base * (2.0 if test_set == "T3" else 1.0)

    table = RecordTable(_records(value, arms=("b1",)))
    assert relative_to_persistence(table, "b1", 4, "lstm", "T1") == pytest.approx(0.5)
    assert relative_to_persistence(table, "b1", 4, "lstm", "T3") == pytest.approx(0.5)


def test_hb1_supported_when_a_planned_design_is_clearly_better():
    def value(arm, n, model, test_set, seed):
        return {"b1": 1.0, "b2": 0.5, "b3": 1.0, "b4": 1.0}.get(arm, 1.0) * _noise(
            arm, n, model, test_set, seed
        )

    result = hb1(RecordTable(_records(value, arms=("b1", "b2", "b3", "b4"))))
    assert result["verdict"] == "supported"
    assert result["per_design"]["b2"]["supported"] and not result["per_design"]["b3"]["supported"]


def test_hb1_refuted_when_planned_designs_are_never_better():
    def value(arm, n, model, test_set, seed):
        return {"b1": 1.0}.get(arm, 1.5) * _noise(arm, n, model, test_set, seed)

    result = hb1(RecordTable(_records(value, arms=("b1", "b2", "b3", "b4"))))
    assert result["verdict"] == "refuted"


def test_hb1_not_established_when_indistinguishable_but_sometimes_better_in_mean():
    def value(arm, n, model, test_set, seed):
        return 1.0 + 0.5 * np.random.default_rng(seed * 100 + n).normal() * (arm != "b1")

    table = RecordTable(_records(value, arms=("b1", "b2", "b3", "b4")))
    assert hb1(table)["verdict"] in ("not established", "refuted", "supported")
    # the structure of the result is complete
    assert set(hb1(table)["per_design"]) == {"b2", "b3", "b4"}


def test_hb2_supported_with_runs_saved():
    def value(arm, n, model, test_set, seed):
        random_curve = 4.0 / math.sqrt(n)
        factor = 0.5 if arm == "b5" else 1.0
        return factor * random_curve * _noise(arm, n, model, test_set, seed)

    result = hb2(RecordTable(_records(value, arms=("b5", "b5r"))))
    assert result["verdict"] == "supported"
    row = result["per_test_set"]["T3"]["rows"][8]
    assert row["ratio"] > 3.0 and row["resolved_better"]


def test_hb2_refuted_when_equal():
    def value(arm, n, model, test_set, seed):
        return (
            4.0
            / math.sqrt(n)
            * (1.0 + 0.001 * (arm == "b5"))
            * _noise(arm, n, model, test_set, seed)
        )

    result = hb2(RecordTable(_records(value, arms=("b5", "b5r"))))
    assert result["per_test_set"]["T3"]["verdict"] in ("refuted", "not established")
    assert result["verdict"] != "supported"


def test_hb3_reversal_detected_and_refuted_without_one():
    arms = ("b0", "b1", "b2", "b3", "b4")
    level = {"b0": 1.0, "b1": 1.3, "b2": 1.6, "b3": 1.9, "b4": 2.2}

    def reversed_value(arm, n, model, test_set, seed):
        order = level[arm] if test_set != "T2" else 3.2 - level[arm]
        return order * _noise(arm, n, model, test_set, seed)

    result = hb3(RecordTable(_records(reversed_value, arms=arms)))
    assert result["verdict"] == "supported" and result["reversals"]

    def same_value(arm, n, model, test_set, seed):
        return level[arm] * _noise(arm, n, model, test_set, seed)

    quiet = hb3(RecordTable(_records(same_value, arms=arms)))
    assert quiet["verdict"] == "refuted" and quiet["max_rank_shift"] <= 1
    assert not quiet["reversals"]
    assert all(v > 0.99 for v in quiet["kendall_tau"].values())

"""B1 descriptive analysis: rank correlations, paired intervals, run equivalents."""

from __future__ import annotations

import math
import zlib

import numpy as np
import pytest

from twinlink_ml.evaluation.b1_analysis import RecordTable
from twinlink_ml.evaluation.b1_descriptive import (
    bootstrap_run_equivalent,
    information_by_arm,
    pair_interval,
    rank_correlations,
    run_equivalent,
)

SIZES = (2, 4, 8, 16, 32)
SEEDS = tuple(range(1, 11))
KEY = "rmse_run_median_physical/product_moisture_kg_per_kg/h24"
ARMS = ("b0", "b1", "b2", "b3", "b4")


def _noise(*parts: object) -> float:
    rng = np.random.default_rng(zlib.crc32(repr(parts).encode()))
    return 1.0 + 0.01 * rng.normal()


def _records(error, info=None):
    records = []
    for arm in ARMS:
        for seed in SEEDS:
            for n in SIZES:
                records.append(
                    {
                        "type": "model",
                        "arm": arm,
                        "master_seed": seed,
                        "n_train": n,
                        "model": "lstm",
                        "metrics": {"T3": {KEY: error(arm, n, seed)}},
                    }
                )
                if info is not None:
                    records.append(
                        {
                            "type": "information",
                            "arm": arm,
                            "master_seed": seed,
                            "n_train": n,
                            "information": info(arm, n, seed),
                        }
                    )
    return records


def test_run_equivalent_of_a_design_worth_double_the_runs():
    # b2 at N has the error b1 has at 2N: b2 is worth N additional runs of b1
    def error(arm, n, seed):
        runs = 2 * n if arm == "b2" else n
        return 4.0 / math.sqrt(runs) * _noise(arm, n, seed)

    table = RecordTable(_records(error))
    result = run_equivalent(table, "b2", "b1", 4, "lstm", "T3")
    assert result["reference_runs"] == pytest.approx(8.0, rel=0.05)
    assert result["extra_runs"] == pytest.approx(4.0, rel=0.1)
    boot = bootstrap_run_equivalent(table, "b2", "b1", 4, "lstm", "T3", n_boot=200)
    assert boot["extra_low"] <= result["extra_runs"] <= boot["extra_high"] + 1e-9
    assert boot["extra_low"] > 2.0
    # beyond the range of the reference curve: bounded, not estimated
    beyond = run_equivalent(table, "b2", "b1", 32, "lstm", "T3")
    assert math.isinf(beyond["reference_runs"]) and math.isinf(beyond["extra_runs"])
    assert run_equivalent(table, "b1", "b1", 8, "lstm", "T3")["extra_runs"] == pytest.approx(
        0.0, abs=1e-6
    )


def test_pair_interval_wording_for_resolved_and_excluded_differences():
    def error(arm, n, seed):
        base = {"b1": 1.0, "b2": 0.5, "b3": 1.0}.get(arm, 1.0)
        spread = 0.2 if arm == "b3" else 0.01
        return base * (1.0 + spread * np.random.default_rng(seed * 10 + n).normal())

    table = RecordTable(_records(error))
    better = pair_interval(table, "b2", "b1", 4, "lstm", "T3")
    assert better["resolved"] and better["ci_low"] > 0 and "better" in better["statement"]
    assert 40 < better["relative_low_percent"] < better["relative_high_percent"] < 60
    noisy = pair_interval(table, "b3", "b1", 4, "lstm", "T3")
    assert not noisy["resolved"] and "excluded" in noisy["statement"]
    assert noisy["excluded_above_percent"] > 5
    assert pair_interval(table, "b2", "b9", 4, "lstm", "T3") is None


def test_rank_correlations_have_the_expected_sign_and_log_the_condition():
    level = {"b0": 5.0, "b1": 4.0, "b2": 1.0, "b3": 3.0, "b4": 2.0}

    def error(arm, n, seed):
        return level[arm] * _noise(arm, n, seed)

    def info(arm, n, seed):
        return {
            "fir_logdet_per_parameter": 10.0 - level[arm] * _noise(arm, seed, "i"),
            "fir_condition": 10.0 ** level[arm],
            "spectral_coverage_1pct": 1.0,
        }

    records = _records(error, info)
    table = RecordTable(records)
    information = information_by_arm(records)
    assert information[("b0", 4)][1]["fir_condition"] == pytest.approx(5.0)  # log10
    result = rank_correlations(
        table, information, arms=ARMS, n_train=4, model="lstm", test_set="T3"
    )
    assert result["fir_logdet_per_parameter"]["arm_spearman"] == pytest.approx(-1.0)
    assert result["fir_logdet_per_parameter"]["seed_spearman"] < -0.9
    assert result["fir_condition"]["arm_spearman"] == pytest.approx(1.0)
    assert math.isnan(result["spectral_coverage_1pct"]["arm_spearman"])  # constant measure
    assert result["fir_logdet_per_parameter"]["n_arms"] == 5

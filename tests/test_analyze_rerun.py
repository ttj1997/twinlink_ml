"""Evaluation script of the re-run (docs/15 section 5) on synthetic records with known effects."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import analyze_rerun as ev  # noqa: E402

SIZES_A1 = [8, 64]
SIZES_N = [8, 16, 32, 64]


def _record(label, seed, n, test_set, values, **extra: object):
    params = {
        "model/label": label,
        "master_seed": seed,
        "n_train": n,
        "test_set": test_set,
        **extra,
    }
    metrics = {}
    for channel, value in values.items():
        for h in ev.HORIZONS:
            metrics[f"{ev.METRIC}/{channel}/h{h}"] = value
    return {"params": params, "metrics": metrics}


def _base(rng_seed_values, label, seed, n, test_set):
    rng = np.random.default_rng([ord(label[0]), seed, n, len(test_set)])
    scale = {"persistence": 3.0, "static": 2.0, "arx": 1.5, "gru": 1.0, "lstm": 1.1}.get(label, 1.3)
    return {c: scale * (n**-0.2) * (1.0 + 0.05 * rng.standard_normal()) for c in ev.CHANNELS}


def _build(a1_factor=2.0, n_factor=4.0, test_part=1.5):
    """A0 old-style records; A1 and N new-style records with cross-evaluation variants."""
    a0, a1, n_arm = [], [], []
    for label in ev.METHODS:
        for seed in ev.SEEDS:
            for size in SIZES_N:
                for test_set in ev.TEST_SETS:
                    base = _base(None, label, seed, size, test_set)
                    a0.append(_record(label, seed, size, test_set, base))
                    wiggle = 1.0 + 0.01 * ((seed * 7 + size) % 5 - 2)
                    nn = label in ("gru", "lstm", "transformer", "xlstm", "mamba")
                    f1 = a1_factor if nn else 1.0
                    fn = n_factor if nn else 1.0
                    if size in SIZES_A1:
                        # A1: planned windows (primary) cost test_part more than realised windows
                        for window, factor in (("planned", f1), ("realised", f1 / test_part)):
                            a1.append(
                                _record(
                                    label,
                                    seed,
                                    size,
                                    test_set,
                                    {c: v * factor * wiggle for c, v in base.items()},
                                    **{
                                        "eval/pool": "own",
                                        "eval/window": window,
                                        "eval/primary": window == "planned",
                                    },
                                )
                            )
                    for pool in ("own", "legacy"):
                        for window in ("realised", "planned"):
                            n_arm.append(
                                _record(
                                    label,
                                    seed,
                                    size,
                                    test_set,
                                    {c: v * fn * wiggle for c, v in base.items()},
                                    **{
                                        "eval/pool": pool,
                                        "eval/window": window,
                                        "eval/primary": pool == "own" and window == "planned",
                                    },
                                )
                            )
    return a0, a1, n_arm


@pytest.fixture(scope="module")
def result():
    a0, a1, n_arm = _build()
    return ev.evaluate(a0, a1, n_arm, SIZES_A1, SIZES_N)


def test_change_table_recovers_the_known_log_ratios(result) -> None:
    gru = [r for r in result["a1_vs_a0"] if r["method"] == "gru"]
    assert len(gru) == len(SIZES_A1) * 2 * 3 * 2
    assert all(abs(r["mean_log_ratio_arm_over_a0"] - math.log(2.0)) < 0.03 for r in gru)
    assert all(r["verdict"] == "candidate worse" for r in gru)
    persistence = [r for r in result["a1_vs_a0"] if r["method"] == "persistence"]
    assert all(abs(r["mean_log_ratio_arm_over_a0"]) < 0.03 for r in persistence)
    assert len([r for r in result["n_vs_a0"] if r["method"] == "gru"]) == len(SIZES_N) * 12


def test_d1_share_is_the_ratio_of_the_log_effects(result) -> None:
    shares = [r["share_d1"] for r in result["d1_share"] if r["method"] == "lstm"]
    assert shares and all(s is not None and abs(s - 0.5) < 0.03 for s in shares)
    base = [r for r in result["d1_share"] if r["method"] == "persistence"]
    assert all(r["share_d1"] is None for r in base)  # denominator not resolved: no share
    assert {r["n_train"] for r in result["d1_share"]} == set(SIZES_A1)


def test_cross_parts_add_up(result) -> None:
    rows = [r for r in result["a1_cross_parts"] if r["method"] == "transformer"]
    for r in rows:
        assert abs(r["total"] - (r["test_part"] + r["training_part"])) < 1e-12
        assert abs(r["test_part"] - math.log(1.5)) < 0.03
        assert abs(r["training_part"] - math.log(2.0 / 1.5)) < 0.03


def test_n_variants_are_tabulated(result) -> None:
    row = next(r for r in result["n_variants_log_ratio_vs_a0"] if r["method"] == "gru")
    assert {k for k in row if k.endswith("windows")} == {
        "own_pool_realised_windows",
        "own_pool_planned_windows",
        "legacy_pool_realised_windows",
        "legacy_pool_planned_windows",
    }
    assert all(abs(row[k] - math.log(4.0)) < 0.03 for k in row if k.endswith("windows"))


def test_rank_cells_and_changes(result) -> None:
    assert set(result["rank_cells"]) == {"A0", "A1", "N"}
    assert len(result["rank_cells"]["A1"]) == len(ev.RANK_CANDIDATES) * len(SIZES_A1) * 12
    assert all(c["arm"] in ("A1", "N") for c in result["rank_verdicts_changed_against_a0"])


def test_multiplicity_and_seeds_blocks(result) -> None:
    for name, block in result["multiplicity"].items():
        assert (
            block["resolved_holm"]
            <= block["resolved_bh"]
            <= block["resolved_uncorrected"]
            <= block["tests"]
        ), name
    assert set(result["seeds_needed"]) == {"A0", "A1", "N"}
    assert all(row["candidate"] in ("gru", "transformer") for row in result["seeds_needed"]["N"])


def test_missing_seed_is_an_error() -> None:
    a0, a1, n_arm = _build()
    a1 = [
        r
        for r in a1
        if not (r["params"]["master_seed"] == 5 and r["params"]["model/label"] == "gru")
    ]
    with pytest.raises(SystemExit):
        ev.evaluate(a0, a1, n_arm, SIZES_A1, SIZES_N)

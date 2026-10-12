"""Tests for the paired resolution test (review findings C1/C2, docs/12).

Pure statistics on simulated numbers, no process simulation. The decisive test is the
false-positive rate without a true difference: it has to sit near the nominal level
for the sample sizes of the learning curves (3-5 seeds), including when the seed effect
shared by both methods is much larger than the method-specific noise -- the case in which
the old, unpaired threshold with a normal quantile misjudges the comparison.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from twinlink_ml.evaluation.paired_resolution import (
    paired_comparison,
    paired_values,
    paired_verdicts,
)
from twinlink_ml.utils.tracking import RunRecord

TARGET = "product_moisture_kg_per_kg"
METRIC = "rmse_run_median"
KEY = f"{METRIC}/{TARGET}/h24"


def _record(label: str, seed: int, value: float, *, n_train: int = 8) -> RunRecord:
    return RunRecord(
        group="architecture_comparison",
        params={"model/label": label, "n_train": n_train, "master_seed": seed},
        metrics={KEY: value},
        tags={"model": label},
    )


# ---------------------------------------------------------------- numbers


def test_matches_scipy_paired_t_test() -> None:
    candidate = {1: 0.50, 2: 0.62, 3: 0.55, 4: 0.71, 5: 0.58}
    reference = {1: 0.60, 2: 0.70, 3: 0.64, 4: 0.80, 5: 0.61}
    result = paired_comparison(candidate, reference)
    seeds = sorted(candidate)
    expected = stats.ttest_rel([reference[s] for s in seeds], [candidate[s] for s in seeds])
    assert result.degrees_of_freedom == 4
    assert result.p_value == pytest.approx(expected.pvalue, rel=1e-12)
    assert result.mean_difference / result.standard_error == pytest.approx(
        expected.statistic, rel=1e-12
    )
    assert result.is_resolved == (expected.pvalue < 0.05)
    low, high = result.confidence_interval
    ci = expected.confidence_interval(0.95)
    assert low == pytest.approx(ci.low, rel=1e-12)
    assert high == pytest.approx(ci.high, rel=1e-12)


def test_degrees_of_freedom_and_quantile_are_n_minus_one_t() -> None:
    rng = np.random.default_rng(3)
    for n in (2, 3, 5, 8):
        values = dict(enumerate(rng.normal(size=n)))
        other = dict(enumerate(rng.normal(size=n)))
        result = paired_comparison(values, other)
        assert result.degrees_of_freedom == n - 1
        assert result.critical_value == pytest.approx(stats.t.ppf(0.975, n - 1))
    # n = 3: t(2) = 4.303, not 1.96 -- the z-quantile of resolution.py is 2.2 times too small
    three = paired_comparison({1: 0.0, 2: 1.0, 3: 0.5}, {1: 1.0, 2: 3.0, 3: 1.2})
    assert three.critical_value == pytest.approx(4.3027, abs=1e-3)


def test_positive_difference_means_candidate_better() -> None:
    result = paired_comparison({1: 1.0, 2: 1.1, 3: 0.9}, {1: 2.0, 2: 2.2, 3: 1.8})
    assert result.mean_difference > 0.0
    assert result.relative_improvement == pytest.approx(0.5)
    assert "candidate better" in result.statement


def test_pairing_removes_the_common_seed_effect() -> None:
    """A large seed effect shared by both methods must not hide a small true difference."""
    seed_effect = {1: 0.0, 2: 10.0, 3: 20.0, 4: 30.0, 5: 40.0}
    reference = {seed: 1.0 + shift for seed, shift in seed_effect.items()}
    candidate = {seed: 0.9 + shift for seed, shift in seed_effect.items()}
    # differences are exactly 0.1 except for a tiny jitter
    jitter = {1: 0.001, 2: -0.001, 3: 0.002, 4: -0.002, 5: 0.0}
    candidate = {seed: value + jitter[seed] for seed, value in candidate.items()}
    assert paired_comparison(candidate, reference).is_resolved


# ------------------------------------------------------------- degenerate


def test_single_pair_gives_no_threshold() -> None:
    result = paired_comparison({1: 0.5}, {1: 0.9})
    assert result.degenerate
    assert not result.is_resolved
    assert math.isnan(result.threshold)
    assert "not resolved" in result.statement


def test_zero_spread_of_differences_gives_no_threshold() -> None:
    """Review C4: a threshold of 0 would make every difference 'resolved'."""
    result = paired_comparison({1: 1.0, 2: 1.0, 3: 1.0}, {1: 2.0, 2: 2.0, 3: 2.0})
    assert result.degenerate
    assert not result.is_resolved
    assert math.isnan(result.threshold)
    assert result.mean_difference == pytest.approx(1.0)


def test_seed_sets_must_match() -> None:
    with pytest.raises(ValueError, match="same seeds"):
        paired_comparison({1: 1.0, 2: 1.0}, {1: 1.0, 3: 1.0})


def test_rejects_empty_and_non_finite() -> None:
    with pytest.raises(ValueError, match="no seeds"):
        paired_comparison({}, {})
    with pytest.raises(ValueError, match="non-finite"):
        paired_comparison({1: math.nan, 2: 1.0}, {1: 1.0, 2: 2.0})
    with pytest.raises(ValueError, match="confidence"):
        paired_comparison({1: 0.0, 2: 1.0}, {1: 1.0, 2: 3.0}, confidence=1.0)


# ------------------------------------------------- false-positive rate, power


def _simulate(
    n_seeds: int,
    *,
    n_simulations: int,
    seed_sd: float,
    method_sd: float,
    true_difference: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorised paired design; returns the boolean ``resolved`` per simulation."""
    shared = rng.normal(0.0, seed_sd, size=(n_simulations, n_seeds))
    reference = 10.0 + shared + rng.normal(0.0, method_sd, size=shared.shape)
    candidate = 10.0 - true_difference + shared + rng.normal(0.0, method_sd, size=shared.shape)
    resolved = np.empty(n_simulations, dtype=bool)
    for index in range(n_simulations):
        result = paired_comparison(
            dict(enumerate(candidate[index])), dict(enumerate(reference[index]))
        )
        resolved[index] = result.is_resolved
    return resolved


@pytest.mark.parametrize("n_seeds", [3, 5])
@pytest.mark.parametrize("seed_sd", [0.0, 1.0, 10.0])
def test_false_positive_rate_without_true_difference_is_nominal(
    n_seeds: int, seed_sd: float
) -> None:
    rng = np.random.default_rng(20261012 + n_seeds)
    resolved = _simulate(
        n_seeds, n_simulations=20_000, seed_sd=seed_sd, method_sd=1.0, true_difference=0.0, rng=rng
    )
    rate = float(resolved.mean())
    # binomial standard error at 20 000 simulations: 0.0015; bounds are about +-3.3 sigma
    assert 0.045 < rate < 0.055, f"false-positive rate {rate:.4f} at n={n_seeds}"


def test_false_positive_rate_with_different_method_noise_is_nominal() -> None:
    """The differences stay symmetric when the methods differ in their own spread."""
    rng = np.random.default_rng(7)
    n_simulations = 20_000
    shared = rng.normal(0.0, 5.0, size=(n_simulations, 4))
    reference = shared + rng.normal(0.0, 0.2, size=shared.shape)
    candidate = shared + rng.normal(0.0, 2.0, size=shared.shape)
    resolved = [
        paired_comparison(dict(enumerate(c)), dict(enumerate(r))).is_resolved
        for c, r in zip(candidate, reference, strict=True)
    ]
    assert 0.045 < float(np.mean(resolved)) < 0.055


def test_power_for_a_clear_difference() -> None:
    rng = np.random.default_rng(11)
    resolved = _simulate(
        5, n_simulations=2_000, seed_sd=5.0, method_sd=0.5, true_difference=2.0, rng=rng
    )
    assert float(resolved.mean()) > 0.9


# ---------------------------------------------------------------- records


def test_paired_values_and_verdicts_from_records() -> None:
    reference = {1: 0.60, 2: 0.70, 3: 0.64, 4: 0.80, 5: 0.61}
    good = {seed: value - 0.15 + 0.01 * (seed % 2) for seed, value in reference.items()}
    noisy = {1: 0.9, 2: 0.4, 3: 0.8, 4: 0.5, 5: 0.7}
    records = [
        _record(label, seed, value)
        for label, series in (("lstm", reference), ("gru", good), ("mamba", noisy))
        for seed, value in series.items()
    ]
    records.append(_record("gru", 1, 99.0, n_train=16))  # other point: must be ignored
    verdicts = paired_verdicts(
        records, n_train=8, target=TARGET, horizon=24, metric=METRIC, reference_label="lstm"
    )
    assert [v.candidate for v in verdicts] == ["gru", "mamba"]
    by_label = {v.candidate: v for v in verdicts}
    assert by_label["gru"].is_resolved
    assert not by_label["mamba"].is_resolved
    assert by_label["gru"].seeds == (1, 2, 3, 4, 5)
    assert paired_values(
        records, label="gru", n_train=8, target=TARGET, horizon=24, metric=METRIC
    ) == pytest.approx(good)


def test_record_order_does_not_matter() -> None:
    reference = {1: 1.0, 2: 2.0, 3: 3.5}
    candidate = {1: 0.8, 2: 1.5, 3: 3.4}
    records = [_record("ref", s, v) for s, v in reference.items()] + [
        _record("cand", s, v) for s, v in candidate.items()
    ]
    forward = paired_verdicts(
        records, n_train=8, target=TARGET, horizon=24, metric=METRIC, reference_label="ref"
    )
    backward = paired_verdicts(
        records[::-1], n_train=8, target=TARGET, horizon=24, metric=METRIC, reference_label="ref"
    )
    assert forward[0].p_value == pytest.approx(backward[0].p_value)


def test_conflicting_duplicate_seed_is_an_error() -> None:
    records = [_record("a", 1, 1.0), _record("a", 1, 2.0)]
    with pytest.raises(ValueError, match="two values"):
        paired_values(records, label="a", n_train=8, target=TARGET, horizon=24, metric=METRIC)


def test_missing_seed_parameter_is_an_error() -> None:
    record = RunRecord(
        group="g",
        params={"model/label": "a", "n_train": 8},
        metrics={KEY: 1.0},
        tags={"model": "a"},
    )
    with pytest.raises(ValueError, match="master_seed"):
        paired_values([record], label="a", n_train=8, target=TARGET, horizon=24, metric=METRIC)


def test_mismatched_seeds_between_methods_is_an_error() -> None:
    records = [_record("ref", 1, 1.0), _record("ref", 2, 2.0), _record("cand", 1, 0.9)]
    with pytest.raises(ValueError, match="same seeds"):
        paired_verdicts(
            records, n_train=8, target=TARGET, horizon=24, metric=METRIC, reference_label="ref"
        )


def test_unknown_reference_is_an_error() -> None:
    with pytest.raises(ValueError, match="no records for reference"):
        paired_verdicts(
            [_record("a", 1, 1.0)],
            n_train=8,
            target=TARGET,
            horizon=24,
            metric=METRIC,
            reference_label="missing",
        )

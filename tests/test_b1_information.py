"""B1.1: model-free information measures."""

from __future__ import annotations

import numpy as np

from twinlink_ml.simulation.excitation_information import (
    arx_fisher_information,
    fir_fisher_information,
    spectral_coverage,
)

DT = 25.0
BAND = {"band_low_hz": 1.33e-4, "band_high_hz": 5.26e-3}


def _random_steps(rng, n_runs=4, n_steps=1152, n_channels=4, hold=12):
    levels = rng.uniform(0, 1, (n_runs, n_steps // hold + 1, n_channels))
    return np.repeat(levels, hold, axis=1)[:, :n_steps]


def test_constant_signals_cover_nothing():
    signal = np.full((3, 1152, 4), 0.4)
    report = spectral_coverage(signal, DT, **BAND)
    assert report.coverage == 0.0 and report.flatness == 0.0


def test_broadband_signal_covers_more_than_a_slow_one():
    rng = np.random.default_rng(0)
    fast = _random_steps(rng, hold=6)
    slow = _random_steps(rng, hold=288)
    assert (
        spectral_coverage(fast, DT, **BAND).coverage > spectral_coverage(slow, DT, **BAND).coverage
    )


def test_cross_correlation_detects_identical_channels():
    rng = np.random.default_rng(1)
    base = _random_steps(rng, n_channels=1)
    identical = np.concatenate([base, base], axis=2)
    independent = _random_steps(rng, n_channels=2)
    assert spectral_coverage(identical, DT, **BAND).max_abs_cross_correlation > 0.99
    assert spectral_coverage(independent, DT, **BAND).max_abs_cross_correlation < 0.5


def test_fir_information_is_additive_over_runs():
    rng = np.random.default_rng(2)
    signal = _random_steps(rng, n_runs=2)
    single = fir_fisher_information(signal, n_lags=10)
    doubled = fir_fisher_information(np.concatenate([signal, signal]), n_lags=10)
    assert doubled.n_rows == 2 * single.n_rows
    np.testing.assert_allclose(doubled.lambda_min, 2 * single.lambda_min, rtol=1e-6)
    assert abs((doubled.logdet_per_parameter - single.logdet_per_parameter) - np.log(2)) < 1e-6


def test_fir_information_ranks_rich_above_poor_excitation():
    rng = np.random.default_rng(3)
    rich = _random_steps(rng, hold=2)
    poor = np.repeat(rng.uniform(0, 1, (4, 1, 4)), 1152, axis=1)  # constant per run
    rich_report, poor_report = fir_fisher_information(rich), fir_fisher_information(poor)
    assert rich_report.logdet_per_parameter > poor_report.logdet_per_parameter
    assert rich_report.lambda_min > poor_report.lambda_min
    assert rich_report.n_parameters == 25 * 4 + 1


def test_fir_condition_flags_collinear_channels():
    rng = np.random.default_rng(4)
    base = _random_steps(rng, n_channels=1, hold=3)
    collinear = np.concatenate([base, base], axis=2)
    independent = _random_steps(rng, n_channels=2, hold=3)
    assert fir_fisher_information(collinear, n_lags=8).condition > 1e6 * (
        fir_fisher_information(independent, n_lags=8).condition
    )


def test_arx_information_parameter_count():
    rng = np.random.default_rng(5)
    inputs = _random_steps(rng, n_runs=2, n_steps=400)
    outputs = rng.normal(size=(2, 400, 2))
    report = arx_fisher_information(inputs, outputs, p=4, q=2)
    assert report.n_parameters == 4 * 2 + 3 * 4 + 1 and report.kind == "arx"
    assert report.lambda_min > 0.0

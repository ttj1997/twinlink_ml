"""Outlier processes, causal Hampel filter and contamination (outlier robustness, docs/27)."""

from __future__ import annotations

import dataclasses
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.evaluation import generate_data
from twinlink_ml.outliers import (
    DEFAULT_CONFIG_PATH,
    MAD_SCALE,
    BurstOutliers,
    CausalHampelFilter,
    Condition,
    IndependentOutliers,
    NoOutliers,
    apply_condition,
    existing_noise_std,
    load_outlier_config,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_arch_rerun_worker as rerun_worker  # noqa: E402
import run_outlier_robustness_worker as worker  # noqa: E402

CHANNEL = "product_moisture_kg_per_kg"
# the tiny smoke spec draws a Sobol design whose size is not a power of two (harmless here)
pytestmark = pytest.mark.filterwarnings("ignore:The balance properties of Sobol")


@pytest.fixture(scope="module")
def small_data():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(rerun_worker.old_worker.CONFIG_PATH)
        return generate_data(config, rerun_worker.arm_spec("N", smoke=True), master_seed=3)


def _run_lengths(mask: np.ndarray) -> list[int]:
    """Lengths of the True runs that neither start at the first nor end at the last sample."""
    lengths = []
    for row in mask:
        padded = np.concatenate([[False], row, [False]]).astype(int)
        edges = np.diff(padded)
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        for s, e in zip(starts, ends, strict=True):
            if s > 0 and e < row.size:
                lengths.append(int(e - s))
    return lengths


# ----------------------------------------------------------------------------- config


def test_config_is_marked_placeholder_and_has_the_values_fixed_beforehand():
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    assert "STATUS: placeholder" in text
    config = load_outlier_config()
    assert config.channel == CHANNEL
    assert config.outliers.rates == (0.0, 0.05, 0.25)
    assert config.outliers.t_degrees_of_freedom == 3.0
    assert config.outliers.scale_factor == 10.0
    assert config.outliers.mean_burst_length_steps == 10.0
    assert (config.hampel.window, config.hampel.threshold) == (8, 3.0)
    assert config.hampel.side_check_windows == (4, 16)


def test_condition_id_round_trip_and_validation():
    for text in ("p0_none_none", "p0.25_burst_hampel8", "p0.05_independent_none"):
        assert Condition.parse(text).id == text
    with pytest.raises(ValueError):
        Condition(rate=0.25, kind="none", filter_window=None)
    with pytest.raises(ValueError):
        Condition(rate=0.0, kind="burst", filter_window=None)


# ------------------------------------------------------------------- p = 0 is bit-identical


def test_p0_reproduces_the_existing_sensor_model_bit_identically(small_data):
    config = load_outlier_config()
    result = apply_condition(small_data, Condition(0.0, "none", None), config)
    assert result.disturbed.trajectory.measurements is small_data.trajectory.measurements
    assert result.processed is result.disturbed
    assert result.diagnostics["outlier/realised_rate"] == 0.0
    # and the measurements are those of the sensor model (remeasure with the data set's noise seed)
    remeasured = small_data.with_measurements().trajectory.measurements
    assert np.array_equal(remeasured, small_data.trajectory.measurements)


def test_rate_zero_processes_add_nothing():
    scale = np.ones((4, 50))
    rng = np.random.default_rng(0)
    for process in (NoOutliers(), IndependentOutliers(rate=0.0), BurstOutliers(rate=0.0)):
        sample = process.sample(scale, rng)
        assert not sample.mask.any()
        assert np.array_equal(sample.error, np.zeros_like(scale))


def test_outliers_touch_only_the_moisture_channel(small_data):
    config = load_outlier_config()
    result = apply_condition(small_data, Condition(0.25, "independent", None), config)
    names = small_data.trajectory.metadata["output_names"]
    index = names.index(CHANNEL)
    before, after = small_data.trajectory.measurements, result.disturbed.trajectory.measurements
    others = [i for i in range(before.shape[-1]) if i != index]
    assert np.array_equal(before[..., others], after[..., others])
    changed = before[..., index] != after[..., index]
    assert 0.15 < changed.mean() < 0.35
    assert np.array_equal(result.disturbed.trajectory.outputs, small_data.trajectory.outputs)


def test_noise_scale_is_relative_std_times_pt1_output(small_data):
    std = existing_noise_std(small_data, CHANNEL)
    measurement = small_data.process_config.measurement
    names = small_data.trajectory.metadata["output_names"]
    y = small_data.trajectory.outputs[..., names.index(CHANNEL)]
    sample_time = small_data.spec.sample_time_s
    alpha = np.exp(-sample_time / measurement.sensor_time_constant_s)
    filtered = np.empty_like(y)
    filtered[:, 0] = y[:, 0]
    for k in range(1, y.shape[1]):
        filtered[:, k] = alpha * filtered[:, k - 1] + (1.0 - alpha) * y[:, k]
    np.testing.assert_allclose(std, measurement.relative_noise_std * np.abs(filtered), rtol=1e-12)


# ------------------------------------------------------------------------ rates and bursts


def test_independent_rate_is_met_on_average_and_nested_across_rates():
    scale = np.ones((400, 2000))
    low = IndependentOutliers(rate=0.05).sample(scale, np.random.default_rng(7))
    high = IndependentOutliers(rate=0.25).sample(scale, np.random.default_rng(7))
    assert abs(low.realised_rate - 0.05) < 0.002
    assert abs(high.realised_rate - 0.25) < 0.003
    assert np.all(high.mask[low.mask])  # common random numbers
    assert np.all(high.error[~high.mask] == 0.0)


def test_burst_rate_and_mean_length_are_met_on_average():
    scale = np.ones((400, 2000))
    sample = BurstOutliers(rate=0.25, mean_length=10.0).sample(scale, np.random.default_rng(11))
    assert abs(sample.realised_rate - 0.25) < 0.01
    lengths = _run_lengths(sample.mask)
    assert len(lengths) > 10_000
    assert abs(np.mean(lengths) - 10.0) < 0.3
    low = BurstOutliers(rate=0.05, mean_length=10.0).sample(scale, np.random.default_rng(12))
    assert abs(low.realised_rate - 0.05) < 0.005
    assert abs(np.mean(_run_lengths(low.mask)) - 10.0) < 0.5


def test_burst_error_is_constant_within_a_burst_and_scaled_at_its_start():
    rng = np.random.default_rng(5)
    scale = rng.uniform(1.0, 2.0, size=(50, 400))
    sample = BurstOutliers(rate=0.25, mean_length=10.0).sample(scale, np.random.default_rng(6))
    for row in range(scale.shape[0]):
        mask, error = sample.mask[row], sample.error[row]
        assert np.all(error[~mask] == 0.0)
        for k in range(1, mask.size):
            if mask[k] and mask[k - 1]:
                assert error[k] == error[k - 1]


# --------------------------------------------------------------------------------- Hampel


def test_hampel_is_causal():
    rng = np.random.default_rng(1)
    signal = rng.normal(size=(6, 200))
    signal[:, ::7] += 25.0
    hampel = CausalHampelFilter(window=8, threshold=3.0)
    reference = hampel.apply(signal).values
    for t in (0, 5, 50, 150):
        changed = signal.copy()
        changed[:, t + 1 :] = rng.normal(scale=100.0, size=changed[:, t + 1 :].shape)
        out = hampel.apply(changed).values
        assert np.array_equal(out[:, : t + 1], reference[:, : t + 1])


def test_hampel_replaces_a_spike_by_the_window_median_and_keeps_a_ramp():
    ramp = np.arange(40, dtype=float)[None, :]
    assert not CausalHampelFilter().apply(ramp).replaced.any()
    spiked = ramp.copy()
    spiked[0, 20] = 1000.0
    result = CausalHampelFilter(window=8, threshold=3.0).apply(spiked)
    assert result.replaced[0, 20] and result.replaced.sum() == 1
    assert result.values[0, 20] == np.median(spiked[0, 13:21])
    assert pytest.approx(1.4826) == MAD_SCALE


# ------------------------------------------------------------------------ worker (smoke)


def test_worker_smoke_p0_equals_the_evaluator_of_befund_94(tmp_path):
    """At p = 0 without a filter the 'processed' metric is the metric of Befund 94."""
    from twinlink_ml.data.splits import SplitName
    from twinlink_ml.data.windows import TrajectoryWindowDataset
    from twinlink_ml.evaluation import ValidationPolicy, build_datasets
    from twinlink_ml.evaluation.evaluator import RolloutEvaluator
    from twinlink_ml.models import PersistenceModel

    output = tmp_path / "smoke.jsonl"
    arguments = worker.parse_arguments(
        [
            "--condition", "p0_none_none", "--seed", "2", "--models", "persistence", "arx",
            "--sizes", "4", "--smoke", "--output", str(output),
        ]
    )  # fmt: skip
    assert worker.run(arguments) == worker.expected_record_count(["persistence", "arx"], [4])
    import json

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    persistence = next(
        r
        for r in records
        if r["params"]["model/label"] == "persistence" and r["params"]["test_set"] == "excited"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(rerun_worker.old_worker.CONFIG_PATH)
    config = dataclasses.replace(
        config, numerics=dataclasses.replace(config.numerics, enforce_gas_velocity_projection=True)
    )
    spec = rerun_worker.arm_spec("N", smoke=True)
    data = generate_data(config, spec, master_seed=2)
    train, validation, _ = build_datasets(data, 4, policy=ValidationPolicy.FIXED_POOL)
    model = PersistenceModel()
    model.fit(train, seed=data.seeds.model_seed, validation=validation)
    test = TrajectoryWindowDataset.from_trajectory(
        data.trajectory, data.split.subset(4), SplitName.TEST, spec.window, scalers=train.scalers
    )
    expected = RolloutEvaluator(horizons=(1, spec.horizon)).evaluate(model, test)
    for h in (1, spec.horizon):
        key = f"{CHANNEL}/h{h}"
        assert persistence["metrics"][f"rmse_run_median_physical_processed/{key}"] == pytest.approx(
            expected.rmse_run_median_physical[CHANNEL][h], rel=1e-12
        )
        assert (
            persistence["metrics"][f"rmse_run_median_physical_disturbed/{key}"]
            == persistence["metrics"][f"rmse_run_median_physical_processed/{key}"]
        )
    assert persistence["params"]["condition/id"] == "p0_none_none"
    assert persistence["params"]["pool/outlier/realised_rate"] == 0.0


def test_worker_moisture_view_keeps_only_the_moisture_history(small_data):
    view = worker.moisture_view(small_data, CHANNEL)
    assert view.spec.target_names == (CHANNEL,)
    assert view.trajectory is small_data.trajectory

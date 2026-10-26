"""Tests of the sensor-value study (``twinlink_ml.sensor_value``, docs/28_sensor_value.md).

Required by the study request: the bias correction with an ideal measurement and without model
mismatch is R1 exactly; the bias filter is causal; the margin bisection converges. Further: the
step form of the bisection reproduces ``bisect_margin``; draws and scenario subset are deterministic
and in their bands; the inline sensor without outliers is the existing sensor model; the soft-sensor
windows are those of the pipeline.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import warnings

import numpy as np
import pytest

casadi = pytest.importorskip("casadi")

from twinlink_ml.headroom.config import load_headroom_config, study_process_config  # noqa: E402
from twinlink_ml.headroom.margin import bisect_margin  # noqa: E402
from twinlink_ml.headroom.scenarios import Scenario, build_scenarios  # noqa: E402
from twinlink_ml.headroom.study import Solvers, run_controller  # noqa: E402
from twinlink_ml.sensitivity.config import load_sensitivity_config  # noqa: E402
from twinlink_ml.sensor_value import (  # noqa: E402
    BiasFilter,
    Feedback,
    IdealMeasurement,
    InlineSensor,
    NoFeedback,
    PairBisection,
    criterion_margin,
    extend_scenario,
    load_sensor_value_config,
    next_candidate,
    parameter_draws,
    required_count,
    run_pairs,
    scenario_subset,
    truth_config,
)
from twinlink_ml.simulation import (  # noqa: E402
    GreyBoxFluidBedGranulator,
    PlaceholderConfigWarning,
    load_process_config,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.fixture(scope="module")
def setup():
    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    config = load_sensor_value_config()
    study = load_headroom_config(config.headroom_config)
    nominal = study_process_config(study)
    return config, study, nominal


def _short(scenario: Scenario, hours: float) -> Scenario:
    n = round(hours * 3600.0 / scenario.sample_time_s)
    return dataclasses.replace(
        scenario,
        disturbances=scenario.disturbances[:, :n],
        moisture_limit=scenario.moisture_limit[:n],
    )


@pytest.fixture(scope="module")
def short_scenarios(setup):
    config, study, nominal = setup
    observed = build_scenarios(study, nominal, "observed")
    subset = scenario_subset(observed, config.scenarios_per_type)
    # one of each type, 1 h
    picks = {}
    for s in subset:
        picks.setdefault(s.kind, s)
    return [_short(s, 1.0) for s in picks.values()]


# ---------------------------------------------------------------------------- R1 exactly


@pytest.mark.parametrize("margin", [0.0, 0.004])
def test_ideal_measurement_without_mismatch_is_r1_exactly(setup, short_scenarios, margin) -> None:
    _, study, nominal = setup
    for scenario in short_scenarios[:3]:
        ours = run_pairs(
            study,
            nominal,
            [scenario],
            [margin],
            [IdealMeasurement()],
            truth=None,
            truth_initial_states=None,
            pre_roll_s=0.0,
            measurement_interval_s=25.0,
            bias_time_constant_s=1800.0,
            variant="F1",
            draw=None,
            keep_arrays=True,
        )[0]
        reference = run_controller("R1", margin, [scenario], Solvers(study, nominal))[0]
        assert ours.energy_kwh == reference.energy_kwh
        assert ours.max_violation == reference.max_violation
        assert ours.co2_kg == reference.co2_kg
        assert ours.extra["energy_kwh_per_interval"] == reference.energy_kwh_per_interval
        assert ours.extra["moisture_max_per_interval"] == reference.moisture_max_per_interval
        assert [list(c) for c in ours.extra["commands"]] == reference.commands
        assert set(ours.extra["bias_per_interval"]) == {0.0}
        assert ours.feedback_error_rmse == 0.0 and ours.model_error_mean == 0.0


def test_no_feedback_is_r1_exactly(setup, short_scenarios) -> None:
    _, study, nominal = setup
    scenario = short_scenarios[0]
    ours = run_pairs(
        study,
        nominal,
        [scenario],
        [0.002],
        [NoFeedback()],
        truth=None,
        truth_initial_states=None,
        pre_roll_s=0.0,
        measurement_interval_s=25.0,
        bias_time_constant_s=1800.0,
        variant="F0",
        draw=None,
    )[0]
    reference = run_controller("R1", 0.002, [scenario], Solvers(study, nominal))[0]
    assert ours.energy_kwh == reference.energy_kwh
    assert ours.max_violation == reference.max_violation


# ---------------------------------------------------------------------------- causality


def test_bias_filter_is_first_order_and_causal() -> None:
    f = BiasFilter(1800.0, 25.0)
    a = math.exp(-25.0 / 1800.0)
    errors = np.random.default_rng(1).normal(size=200)
    values = [f.update(e) for e in errors]
    expected = errors[0]
    for k in range(1, len(errors)):
        expected = a * expected + (1 - a) * errors[k]
        assert values[k] == pytest.approx(expected, rel=1e-12, abs=1e-15)
    # causal: changing errors after step 120 leaves the first 121 outputs unchanged
    g = BiasFilter(1800.0, 25.0)
    changed = errors.copy()
    changed[121:] += 100.0
    assert [g.update(e) for e in changed][:121] == values[:121]
    # constant error: converges to it
    h = BiasFilter(1800.0, 25.0)
    h.update(0.0)
    for _ in range(3000):
        last = h.update(0.02)
    assert last == pytest.approx(0.02, abs=1e-8)


class _SwitchingFeedback(Feedback):
    """Ideal measurement whose values are offset from a given block on (future change)."""

    name = "test"

    def __init__(self, switch_block: int, offset: float) -> None:
        self.switch_block = switch_block
        self.offset = offset
        self.block = 0

    def values(self, true_moisture: np.ndarray, exogenous: np.ndarray) -> np.ndarray:
        del exogenous
        out = np.asarray(true_moisture, dtype=np.float64).copy()
        for j in range(len(out)):
            if self.block + j >= self.switch_block:
                out[j] += self.offset
        self.block += len(out)
        return out


def test_closed_loop_uses_past_feedback_only(setup, short_scenarios) -> None:
    """A feedback change from block B on leaves every command up to B's interval unchanged."""
    _, study, nominal = setup
    scenario = short_scenarios[1]
    switch_block = 60  # 25 s blocks: interval 5 (blocks 60-71)
    runs = []
    for offset in (0.0, 0.03):
        runs.append(
            run_pairs(
                study,
                nominal,
                [scenario],
                [0.003],
                [_SwitchingFeedback(switch_block, offset)],
                truth=None,
                truth_initial_states=None,
                pre_roll_s=0.0,
                measurement_interval_s=25.0,
                bias_time_constant_s=1800.0,
                variant="test",
                draw=None,
                keep_arrays=True,
            )[0]
        )
    first = switch_block // 12  # interval whose blocks contain the switch
    a, b = (np.array(r.extra["commands"]) for r in runs)
    # the command of interval k uses feedback up to the end of interval k - 1
    assert np.array_equal(a[: first + 1], b[: first + 1])
    assert not np.array_equal(a[first + 1 :], b[first + 1 :])
    bias_a, bias_b = runs[0].extra["bias_per_interval"], runs[1].extra["bias_per_interval"]
    assert bias_a[: first + 1] == bias_b[: first + 1]


def test_extend_scenario_prepends_the_first_sample(short_scenarios) -> None:
    s = short_scenarios[2]
    e = extend_scenario(s, 600.0)
    n = round(600.0 / s.sample_time_s)
    assert e.n_samples == s.n_samples + n
    assert np.array_equal(e.disturbances[:, :n], np.repeat(s.disturbances[:, :1], n, axis=1))
    assert np.array_equal(e.disturbances[:, n:], s.disturbances)
    assert extend_scenario(s, 0.0) is s


def test_pre_roll_is_not_counted(setup, short_scenarios) -> None:
    """With a pre-roll, energy counts only the scenario part (nominal truth, constant start)."""
    _, study, nominal = setup
    scenario = short_scenarios[0]
    with_roll = run_pairs(
        study,
        nominal,
        [scenario],
        [0.003],
        [IdealMeasurement()],
        truth=None,
        truth_initial_states=None,
        pre_roll_s=1800.0,
        measurement_interval_s=25.0,
        bias_time_constant_s=1800.0,
        variant="F1",
        draw=None,
        keep_arrays=True,
    )[0]
    assert len(with_roll.extra["energy_kwh_per_interval"]) == round(
        scenario.duration_s / study.control.interval_s
    )
    assert with_roll.energy_kwh == pytest.approx(sum(with_roll.extra["energy_kwh_per_interval"]))


# ---------------------------------------------------------------------------- bisection


def _run_pair_bisection(threshold: float, lower=0.0, upper=0.05, tol=2.5e-4, cap=0.12):
    search = PairBisection(lower=lower, upper=upper, tolerance=tol, max_upper=cap)
    while not search.done:
        search.report(search.next_margin() >= threshold)
    return search


@pytest.mark.parametrize("threshold", [0.0, 1e-5, 0.0123, 0.0499, 0.05, 0.073, 0.119, 0.2])
def test_pair_bisection_reproduces_bisect_margin(threshold: float) -> None:
    ours = _run_pair_bisection(threshold)
    reference = bisect_margin(
        lambda m: m >= threshold, lower=0.0, upper=0.05, tolerance=2.5e-4, max_upper=0.12
    )
    assert ours.evaluations == reference.evaluations
    assert ours.margin == reference.margin
    assert ours.converged == reference.converged


@pytest.mark.parametrize("threshold", [0.0, 0.0031, 0.0444, 0.09])
def test_pair_bisection_converges(threshold: float) -> None:
    search = _run_pair_bisection(threshold)
    assert search.converged
    assert threshold <= search.smallest_holding <= threshold + 2.5e-4
    # the returned margin was evaluated as holding
    assert (search.smallest_holding, True) in search.evaluations


def test_pair_bisection_reports_infeasible_beyond_cap() -> None:
    search = _run_pair_bisection(0.5)
    assert not search.converged
    assert search.smallest_holding == math.inf


def test_criterion_margins_and_next_candidate() -> None:
    per_pair = np.array([0.0, 0.01, 0.02, 0.03, math.inf])
    assert required_count(400, 0.95) == 380
    assert required_count(20, 0.95) == 19
    assert criterion_margin(per_pair, 4) == 0.03
    assert criterion_margin(per_pair, 5) == math.inf
    assert next_candidate(per_pair, 0.01) == 0.02
    assert next_candidate(per_pair, 0.03) == math.inf


# ---------------------------------------------------------------------------- draws, scenarios


def test_parameter_draws_are_deterministic_and_in_the_bands(setup) -> None:
    config, _, nominal = setup
    sensitivity = load_sensitivity_config(config.sensitivity_config)
    a, rejected_a = parameter_draws(config.draws, sensitivity, nominal)
    b, rejected_b = parameter_draws(config.draws, sensitivity, nominal)
    assert [d.values for d in a] == [d.values for d in b] and rejected_a == rejected_b
    assert len(a) == config.draws.n
    specs = {s.name: s for s in sensitivity.inputs["p2"]}
    for draw in a:
        for name, value in draw.values.items():
            assert specs[name].low <= value <= specs[name].high
            assert getattr(draw.process.material, name) == value
        # nothing else differs from the nominal model
        restored = truth_config(
            draw.process, {n: getattr(nominal.material, n) for n in draw.values}
        )
        assert restored == nominal
    assert set(a[0].values) == set(config.draws.parameters)


def test_scenario_subset_is_spread_and_contains_both_inlet_disturbances(setup) -> None:
    config, study, nominal = setup
    subset = scenario_subset(build_scenarios(study, nominal, "observed"), config.scenarios_per_type)
    assert len(subset) == 20
    kinds: dict[str, list[int]] = {}
    for s in subset:
        kinds.setdefault(s.kind, []).append(s.index)
    assert kinds["ambient"] == [0, 8, 16, 24]
    assert kinds["inlet_temperature_fast"] == [0, 6, 13, 19]
    assert kinds["inlet_temperature_slow"] == [0, 6, 13, 19]
    assert set(kinds) == {
        "ambient",
        "throughput",
        "solid_fraction",
        "inlet_temperature_slow",
        "inlet_temperature_fast",
    }


# ---------------------------------------------------------------------------- inline sensor


def test_inline_sensor_without_outliers_is_the_existing_sensor_model(setup) -> None:
    _, study, nominal = setup
    measurement = load_process_config(study.process_config).measurement
    sensor = InlineSensor(
        nominal,
        measurement,
        25.0,
        outlier_rate=0.0,
        degrees_of_freedom=3.0,
        scale_factor=10.0,
        noise_seed=7,
        outlier_seed=8,
    )
    x = 0.14 + 0.005 * np.sin(np.arange(40) / 5.0)
    got = np.concatenate(
        [sensor.values(x[:12], np.zeros((12, 7))), sensor.values(x[12:], np.zeros((28, 7)))]
    )
    config = dataclasses.replace(
        nominal,
        numerics=dataclasses.replace(nominal.numerics, sample_time_s=25.0),
        measurement=measurement,
    )
    outputs = np.zeros((1, len(x), 4))
    outputs[0, :, 2] = x
    expected = GreyBoxFluidBedGranulator(config).remeasure(outputs, seed=7)[0, :, 2]
    assert np.array_equal(got, expected)
    assert sensor.outliers_drawn == 0


def test_inline_sensor_outlier_share_and_scale(setup) -> None:
    _, study, nominal = setup
    measurement = load_process_config(study.process_config).measurement
    sensor = InlineSensor(
        nominal,
        measurement,
        25.0,
        outlier_rate=0.25,
        degrees_of_freedom=3.0,
        scale_factor=10.0,
        noise_seed=1,
        outlier_seed=2,
    )
    x = np.full(4000, 0.14)
    y = sensor.values(x, np.zeros((4000, 7)))
    assert sensor.outliers_drawn / sensor.samples == pytest.approx(0.25, abs=0.025)
    # Gaussian part: std 0.005 x 0.14; outliers: t3 with scale 10 x that
    residual = y - x
    gaussian = 0.005 * 0.14
    assert np.median(np.abs(residual)) < 3 * gaussian
    assert np.max(np.abs(residual)) > 10 * gaussian


# ---------------------------------------------------------------------------- soft sensor


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_soft_sensor_windows_are_those_of_the_pipeline(tmp_path) -> None:
    """``SoftSensor.estimate`` equals the pipeline's h = 1 prediction of the same window."""
    from twinlink_ml.data.splits import SplitName
    from twinlink_ml.data.windows import TrajectoryWindowDataset, future_actions_of
    from twinlink_ml.evaluation import DataSpec, build_datasets, generate_data
    from twinlink_ml.evaluation.protocol import ObservationMode, spec_for_mode
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig
    from twinlink_ml.sensor_value import SoftSensor, SoftSensorFeedback
    from twinlink_ml.utils.checkpoints import load_checkpoint, save_checkpoint

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config("conf/process/greybox_gab_placeholder.yaml")
    spec = DataSpec(
        sample_time_s=25.0,
        run_duration_s=4000.0,
        n_train_max=4,
        n_val=2,
        n_test=2,
        history_length=8,
        horizon=4,
        stride=4,
        target_names=("product_moisture_kg_per_kg",),
    )
    spec = spec_for_mode(spec, ObservationMode.SOFT_SENSOR)
    data = generate_data(config, spec, master_seed=3)
    train, validation, _ = build_datasets(data, 4)
    model = CompositeSequenceModel(TorchSequenceConfig(hidden_size=8, max_epochs=2, patience=0))
    model.fit(train, seed=1, validation=validation)
    path = tmp_path / "s.pt"
    save_checkpoint(
        path,
        model,
        model_label="gru",
        run_uid="t",
        target_names=train.target_names,
        exogenous_names=train.exogenous_names,
        n_exogenous=len(train.exogenous_names),
        input_scaler=train.input_scaler,
        target_scaler=train.target_scaler,
        process_fingerprint="x",
        data_spec_params={**spec.as_parameters(), "data/observed_target_names": []},
        seed_bundle_params={},
        git_commit="x",
    )
    sensor = SoftSensor.load(path)
    loaded, _ = load_checkpoint(path)
    test = TrajectoryWindowDataset.from_trajectory(
        data.trajectory, data.split, SplitName.TEST, spec.window, scalers=train.scalers
    )
    batch = test.batch(range(test.n_windows))
    reference = test.target_scaler.inverse(loaded.predict(batch))[:, 0, 0]
    tr = data.trajectory
    run_rows = [tr.run_id.index(r) for r in data.split.test]
    exo = np.concatenate([tr.actions, tr.disturbances], axis=-1)
    future = np.concatenate([future_actions_of(tr, spec.window), tr.disturbances], axis=-1)
    histories, current = [], []
    for run, start in zip(batch.run_index.astype(int), batch.window_start.astype(int), strict=True):
        row = run_rows[run]
        histories.append(exo[row, start : start + spec.history_length])
        current.append(future[row, start + spec.history_length])
    got = sensor.estimate(np.array(histories), np.array(current))
    assert np.allclose(got, reference, rtol=0, atol=1e-6)
    # the feedback object hands block j the history of blocks j - L .. j - 1
    row = run_rows[0]
    feedback = SoftSensorFeedback(sensor, "t")
    first = feedback.values(np.zeros(spec.history_length), exo[row, : spec.history_length])
    second = feedback.values(np.zeros(3), exo[row, spec.history_length : spec.history_length + 3])
    expected = sensor.estimate(
        np.stack([exo[row, j : j + spec.history_length] for j in range(3)]),
        exo[row, spec.history_length : spec.history_length + 3],
    )
    assert np.allclose(second, expected, atol=1e-12)
    assert first.shape == (spec.history_length,)

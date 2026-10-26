"""Tests of the follow-up sensor-value study (docs/29_sensor_value_2.md).

Required by the request: the extended window pipeline and closed loop behave bit for bit as before
for every existing configuration (digests of the old code); the domain-randomised training draws
are disjoint from the 20 test draws. Further: measured inputs enter history and future inputs from
the sensor model and never a target; exhaust channels and response sensor use the existing sensor
model; the laboratory sample updates the bias only on arrival, with the delay and the gain; without
mismatch and error it is R1 exactly; the soft sensor with responses sees the pipeline's windows.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import warnings

import numpy as np
import pytest

casadi = pytest.importorskip("casadi")

from _sensor_value_2_digests import loop_digest, window_digests  # noqa: E402
from twinlink_ml.data.splits import SplitName  # noqa: E402
from twinlink_ml.data.windows import TrajectoryWindowDataset, WindowConfig  # noqa: E402
from twinlink_ml.evaluation import DataSpec, build_datasets, generate_data  # noqa: E402
from twinlink_ml.evaluation.protocol import ObservationMode, spec_for_mode  # noqa: E402
from twinlink_ml.headroom.config import load_headroom_config, study_process_config  # noqa: E402
from twinlink_ml.headroom.scenarios import build_scenarios  # noqa: E402
from twinlink_ml.headroom.study import Solvers, run_controller  # noqa: E402
from twinlink_ml.sensitivity.config import load_sensitivity_config  # noqa: E402
from twinlink_ml.sensor_value import (  # noqa: E402
    RESPONSE_NAMES,
    LabSample,
    ResponseSensor,
    SampleAndHoldBias,
    parameter_draws,
    randomised_data,
    run_pairs,
    scenario_subset,
    truth_config,
    with_exhaust_channels,
)
from twinlink_ml.sensor_value.config import DrawSettings, load_follow_up_config  # noqa: E402
from twinlink_ml.simulation import (  # noqa: E402
    GreyBoxFluidBedGranulator,
    PlaceholderConfigWarning,
    load_process_config,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None

# computed with tests/_sensor_value_2_digests.py on origin/master 8f5db3e, before the extension
WINDOW_DIGESTS_BEFORE = {
    "legacy_two_targets": "a3aaf8dfa8c9057fc234",
    "new_study_soft_moisture": "7c46385f868a4bf8fcd6",
    "new_study_three_targets": "ccc9cee3fc422b606434",
}
LOOP_DIGESTS_BEFORE = {
    "F0": "367672042347fa9d4721",
    "F1": "cca0e26504cdbd0cd242",
    "F2": "d027ded6f6964280f27a",
}
EXOGENOUS = (
    "inlet_temperature_celsius,gas_velocity_m_per_s,spray_rate_kg_per_h,nozzle_pressure_bar,"
    "ambient_temperature_celsius,ambient_relative_humidity,suspension_solid_fraction"
)


@pytest.fixture(scope="module")
def process():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config("conf/process/greybox_gab_placeholder.yaml")
    return dataclasses.replace(
        config, numerics=dataclasses.replace(config.numerics, enforce_gas_velocity_projection=True)
    )


@pytest.fixture(scope="module")
def small_data(process):
    spec = DataSpec.new_study(
        sample_time_s=25.0,
        run_duration_s=3000.0,
        n_train_max=4,
        n_val=2,
        n_test=2,
        history_length=8,
        horizon=4,
        stride=4,
        target_names=("product_moisture_kg_per_kg",),
    )
    spec = spec_for_mode(spec, ObservationMode.SOFT_SENSOR)
    return generate_data(process, spec, master_seed=7)


# ------------------------------------------------------------------- bit identity with the old code


def test_existing_window_configurations_are_bit_identical() -> None:
    got = window_digests()
    for label, digest in WINDOW_DIGESTS_BEFORE.items():
        assert got[label] == f"{digest}|{EXOGENOUS}", label


def test_existing_feedback_variants_are_bit_identical_in_the_loop() -> None:
    assert loop_digest() == LOOP_DIGESTS_BEFORE


# ------------------------------------------------------------------- measured inputs


def test_measured_inputs_come_from_the_sensor_model_and_extend_the_exogenous(small_data) -> None:
    data = with_exhaust_channels(small_data, noise_seed=3)
    window = dataclasses.replace(data.spec.window, measured_input_names=RESPONSE_NAMES)
    train, _, test = build_datasets(data, 4, window=window)
    plain, _, plain_test = build_datasets(data, 4)
    assert train.exogenous_names == plain.exogenous_names + RESPONSE_NAMES
    b, p = test.full_batch(), plain_test.full_batch()
    # targets, history of the plain channels and their future are untouched
    assert np.array_equal(b.future_targets, p.future_targets)
    assert np.array_equal(b.future_targets_truth, p.future_targets_truth)
    n = len(plain.exogenous_names)
    names = list(data.trajectory.metadata["output_names"])
    cols = [names.index(r) for r in RESPONSE_NAMES]
    measured = data.trajectory.measurements[..., cols]
    scaler = train.input_scaler.select(range(n, n + 3))
    rows = [data.trajectory.run_id.index(r) for r in data.split.test]
    run, start = int(b.run_index[0]), int(b.window_start[0])
    hist_len = window.history_length
    expected_hist = scaler.transform(measured[rows[run], start : start + hist_len])
    expected_future = scaler.transform(
        measured[rows[run], start + hist_len : start + hist_len + window.horizon]
    )
    assert np.allclose(b.history_inputs[0, :, n : n + 3], expected_hist, atol=1e-12)
    assert np.allclose(b.future_exogenous[0, :, n : n + 3], expected_future, atol=1e-12)
    # the measured, not the noise-free values
    assert not np.allclose(measured, data.trajectory.outputs[..., cols])
    # the unobserved target stays masked
    assert np.all(b.history_inputs[..., -1] == 0.0)


def test_measured_inputs_never_contain_a_target() -> None:
    with pytest.raises(ValueError, match="targets"):
        WindowConfig(
            history_length=4,
            horizon=2,
            target_names=("product_moisture_kg_per_kg",),
            measured_input_names=("product_moisture_kg_per_kg",),
        )


def test_exhaust_channels_are_the_model_quantities(small_data) -> None:
    data = with_exhaust_channels(small_data, noise_seed=3)
    tr, old = data.trajectory, small_data.trajectory
    names = list(tr.metadata["output_names"])
    assert np.array_equal(tr.outputs[..., :4], old.outputs)
    assert np.array_equal(tr.measurements[..., :4], old.measurements)
    loss = data.process_config.exhaust.outlet_duct_heat_loss_k
    assert np.array_equal(
        tr.outputs[..., names.index("exhaust_temperature_celsius")],
        old.outputs[..., names.index("bed_temperature_celsius")] - loss,
    )
    assert np.array_equal(
        tr.outputs[..., names.index("exhaust_absolute_humidity_kg_per_kg")],
        old.diagnostics["outlet_humidity_kg_per_kg"],
    )


def test_response_sensor_is_the_sensor_model_of_the_training_data(small_data, process) -> None:
    """Exhaust channels: the loop's sensor gives exactly the training pool's measured values."""
    one = dataclasses.replace(
        small_data,
        trajectory=small_data.trajectory.select(0),
    )
    data = with_exhaust_channels(one, noise_seed=17)
    tr = data.trajectory
    names = list(tr.metadata["output_names"])
    true = np.stack([tr.outputs[0, :, names.index(r)] for r in RESPONSE_NAMES], axis=-1)
    sensor = ResponseSensor(
        process,
        data.process_config.measurement,
        25.0,
        ("exhaust_temperature_celsius", "exhaust_absolute_humidity_kg_per_kg"),
        bed_seed=1,
        exhaust_seed=17,
    )
    got = np.concatenate([sensor.measure(true[:10]), sensor.measure(true[10:])])
    expected = tr.measurements[0, :, [names.index(r) for r in RESPONSE_NAMES[1:]]].T
    assert np.array_equal(got, expected)
    # the bed temperature channel: PT1 + relative + absolute noise of the same sensor model
    bed = ResponseSensor(
        process,
        data.process_config.measurement,
        25.0,
        ("bed_temperature_celsius",),
        bed_seed=5,
        exhaust_seed=6,
    ).measure(true)[:, 0]
    assert 0.1 < np.std(bed - true[:, 0]) < 1.0


# ------------------------------------------------------------------- laboratory sample


def _lab(every_blocks: int, delay_blocks: int, std: float = 0.0) -> LabSample:
    return LabSample(
        sampling_interval_s=25.0 * every_blocks,
        delay_s=25.0 * delay_blocks,
        error_std_kg_per_kg=std,
        gain=0.5,
        interval_s=25.0,
        seed=1,
        label="t",
    )


def test_lab_sample_updates_on_arrival_only_and_holds() -> None:
    lab = _lab(every_blocks=10, delay_blocks=4)
    bias = lab.bias_filter(1800.0, 25.0)
    assert isinstance(bias, SampleAndHoldBias)
    truth, model = np.full(36, 0.12), np.full(36, 0.10)  # constant model error 0.02
    trace = []
    for j in range(36):
        lab.feed(bias, truth[j : j + 1], model[j : j + 1], np.zeros((1, 7)))
        trace.append(bias.value)
    trace = np.array(trace)
    # samples at blocks 1, 11, 21, 31; results at blocks 5, 15, 25, 35
    assert np.all(trace[:4] == 0.0)
    assert trace[4] == pytest.approx(0.01) and np.all(trace[4:14] == trace[4])
    assert trace[14] == pytest.approx(0.015) and np.all(trace[14:24] == trace[14])
    assert trace[24] == pytest.approx(0.0175)
    assert trace[34] == pytest.approx(0.01875)
    assert lab.samples_taken == 4 and lab.results_arrived == 4


def test_lab_sample_innovation_uses_the_bias_of_the_sample_instant() -> None:
    lab = _lab(every_blocks=2, delay_blocks=3)  # delay longer than the interval: overlap
    bias = lab.bias_filter(1800.0, 25.0)
    for _ in range(8):
        lab.feed(bias, np.array([0.12]), np.array([0.10]), np.zeros((1, 7)))
    # samples at blocks 1, 3, 5, 7 (b = 0, 0, 0.01, 0.015); arrivals at 4, 6, 8
    assert lab.innovations == pytest.approx([0.02, 0.02, 0.01])
    assert bias.value == pytest.approx(0.01 + 0.01 + 0.005)


def test_lab_sample_error_and_seed() -> None:
    a, b = _lab(1, 0, std=0.003), _lab(1, 0, std=0.003)
    fa, fb = a.bias_filter(1, 1), b.bias_filter(1, 1)
    ea = a.feed(fa, np.full(4000, 0.1), np.full(4000, 0.1), np.zeros((4000, 7)))
    eb = b.feed(fb, np.full(4000, 0.1), np.full(4000, 0.1), np.zeros((4000, 7)))
    assert np.array_equal(ea, eb)  # same seed, same errors (common random numbers)
    assert np.std(ea) == pytest.approx(0.003, rel=0.05)


def test_lab_sample_without_mismatch_and_error_is_r1_exactly() -> None:
    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    config = load_follow_up_config().base
    study = load_headroom_config(config.headroom_config)
    nominal = study_process_config(study)
    subset = scenario_subset(build_scenarios(study, nominal, "observed"), 4)
    n = round(3600.0 / subset[0].sample_time_s)
    scenario = dataclasses.replace(
        subset[0],
        disturbances=subset[0].disturbances[:, :n],
        moisture_limit=subset[0].moisture_limit[:n],
    )
    lab = LabSample(
        sampling_interval_s=900.0,
        delay_s=300.0,
        error_std_kg_per_kg=0.0,
        gain=0.5,
        interval_s=25.0,
        seed=1,
        label="t",
    )
    ours = run_pairs(
        study,
        nominal,
        [scenario],
        [0.004],
        [lab],
        truth=None,
        truth_initial_states=None,
        pre_roll_s=0.0,
        measurement_interval_s=25.0,
        bias_time_constant_s=1800.0,
        variant="F4",
        draw=None,
    )[0]
    reference = run_controller("R1", 0.004, [scenario], Solvers(study, nominal))[0]
    assert ours.energy_kwh == reference.energy_kwh
    assert ours.max_violation == reference.max_violation
    assert ours.bias_min == ours.bias_max == 0.0
    assert ours.extra["results_arrived"] == 4


# ------------------------------------------------------------------- domain randomisation


def test_randomised_training_draws_are_disjoint_from_the_test_draws(process) -> None:
    follow_up = load_follow_up_config()
    sensitivity = load_sensitivity_config(follow_up.base.sensitivity_config)
    test_draws, _ = parameter_draws(follow_up.base.draws, sensitivity, process)
    settings = DrawSettings(
        n=128,
        seed=follow_up.soft.randomisation_seed,
        sequence_length=follow_up.soft.randomisation_sequence_length,
        parameters=follow_up.base.draws.parameters,
    )
    assert settings.seed != follow_up.base.draws.seed
    draws, _ = parameter_draws(settings, sensitivity, process)
    assert len(draws) == 128
    p = settings.parameters
    test_units = {tuple(d.unit[k] for k in p) for d in test_draws}
    test_values = {tuple(d.values[k] for k in p) for d in test_draws}
    assert not test_units & {tuple(d.unit[k] for k in p) for d in draws}
    assert not test_values & {tuple(d.values[k] for k in p) for d in draws}
    # same bands: every value inside the range spanned by the specs of the Sobol config
    specs = {s.name: s for s in sensitivity.inputs["p2"]}
    for d in draws:
        for k in p:
            lo, hi = specs[k].value(np.array(0.0)), specs[k].value(np.array(1.0))
            assert min(lo, hi) <= d.values[k] <= max(lo, hi)


def test_randomised_data_resimulates_each_run_with_its_own_config(small_data) -> None:
    n = len(small_data.schedule.run_id)
    same = randomised_data(small_data, [small_data.process_config] * n, list(range(n)))
    assert same.split == small_data.split and same.spec == small_data.spec
    assert np.allclose(same.trajectory.outputs, small_data.trajectory.outputs, rtol=1e-12, atol=0)
    assert np.array_equal(same.trajectory.actions, small_data.trajectory.actions)
    drawn = truth_config(small_data.process_config, {"critical_moisture_kg_per_kg": 0.28})
    mixed = randomised_data(
        small_data, [drawn] + [small_data.process_config] * (n - 1), list(range(n))
    )
    moisture = list(small_data.trajectory.metadata["output_names"]).index(
        "product_moisture_kg_per_kg"
    )
    assert not np.allclose(
        mixed.trajectory.outputs[0, :, moisture], small_data.trajectory.outputs[0, :, moisture]
    )
    assert np.allclose(mixed.trajectory.outputs[1:], small_data.trajectory.outputs[1:], rtol=1e-12)


# ------------------------------------------------------------------- soft sensor with responses


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_soft_sensor_with_responses_sees_the_pipeline_windows(small_data, tmp_path) -> None:
    from twinlink_ml.data.windows import future_actions_of
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig
    from twinlink_ml.sensor_value import SoftSensor, SoftSensorFeedback
    from twinlink_ml.utils.checkpoints import load_checkpoint, save_checkpoint

    inputs = ("bed_temperature_celsius", "exhaust_absolute_humidity_kg_per_kg")
    data = with_exhaust_channels(small_data, noise_seed=3)
    window = dataclasses.replace(data.spec.window, measured_input_names=inputs)
    train, validation, _ = build_datasets(data, 4, window=window)
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
        data_spec_params={
            **data.spec.as_parameters(),
            "data/observed_target_names": [],
            "data/measured_input_names": list(inputs),
        },
        seed_bundle_params={},
        git_commit="x",
    )
    sensor = SoftSensor.load(path)
    assert sensor.measured_input_names == inputs
    loaded, _ = load_checkpoint(path)
    test = TrajectoryWindowDataset.from_trajectory(
        data.trajectory, data.split, SplitName.TEST, window, scalers=train.scalers
    )
    batch = test.batch(range(test.n_windows))
    reference = test.target_scaler.inverse(loaded.predict(batch))[:, 0, 0]
    tr = data.trajectory
    names = list(tr.metadata["output_names"])
    measured = tr.measurements[..., [names.index(n) for n in inputs]]
    rows = [tr.run_id.index(r) for r in data.split.test]
    exo = np.concatenate([tr.actions, tr.disturbances, measured], axis=-1)
    future = np.concatenate([future_actions_of(tr, window), tr.disturbances, measured], axis=-1)
    hist_len = window.history_length
    hist, cur = [], []
    for run, start in zip(batch.run_index.astype(int), batch.window_start.astype(int), strict=True):
        hist.append(exo[rows[run], start : start + hist_len])
        cur.append(future[rows[run], start + hist_len])
    assert np.allclose(sensor.estimate(np.array(hist), np.array(cur)), reference, atol=1e-6)

    # the feedback appends the measured responses of its response sensor to the exogenous values
    class _Fixed:
        names = inputs

        def measure(self, true: np.ndarray) -> np.ndarray:
            return true[:, [0, 2]] + 1.0

    feedback = SoftSensorFeedback(sensor, "t", _Fixed())  # type: ignore[arg-type]
    assert feedback.needs_responses and feedback.name == "F5:t"
    row = rows[0]
    plain = np.concatenate([tr.actions, tr.disturbances], axis=-1)[row]
    true = np.stack([tr.outputs[row, :, names.index(r)] for r in RESPONSE_NAMES], axis=-1)
    bias = feedback.bias_filter(1800.0, 25.0)
    feedback.feed(bias, np.zeros(hist_len), np.zeros(hist_len), plain[:hist_len], true[:hist_len])
    errors = feedback.feed(
        bias,
        np.zeros(3),
        np.zeros(3),
        plain[hist_len : hist_len + 3],
        true[hist_len : hist_len + 3],
    )
    full = np.concatenate([plain, true[:, [0, 2]] + 1.0], axis=-1)
    expected = sensor.estimate(
        np.stack([full[j : j + hist_len] for j in range(3)]), full[hist_len : hist_len + 3]
    )
    assert np.allclose(errors, expected, atol=1e-12)
    with pytest.raises(ValueError, match="response sensor"):
        SoftSensorFeedback(sensor, "t")


def test_truth_plant_records_the_exhaust_humidity_of_the_model() -> None:
    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    from twinlink_ml.headroom.truth import TruthPlant

    config = load_follow_up_config().base
    study = load_headroom_config(config.headroom_config)
    nominal = study_process_config(study)
    scenario = scenario_subset(build_scenarios(study, nominal, "observed"), 4)[0]
    n = round(600.0 / scenario.sample_time_s)
    scenario = dataclasses.replace(
        scenario,
        disturbances=scenario.disturbances[:, :n],
        moisture_limit=scenario.moisture_limit[:n],
    )
    solvers = Solvers(study, nominal)
    from twinlink_ml.headroom.study import initial_conditions

    states, commands = initial_conditions(solvers, [scenario])
    plant = TruthPlant(
        nominal,
        [scenario],
        states,
        commands,
        interval_s=study.control.interval_s,
        gas_heater_efficiency=study.control.gas_heater_efficiency,
    )
    plant.advance(commands)
    sim = GreyBoxFluidBedGranulator(nominal)
    state = np.array(states, dtype=np.float64)
    d = scenario.disturbances[:, 0]
    action = np.array(
        [[commands[0][0] + d[4], commands[0][1], d[3], nominal.actions.nozzle_pressure_bar.nominal]]
    )
    step = sim.step(state, action, d[None, 0:3])
    assert plant.outlet_humidity_so_far[0, 0] == step.diagnostics["outlet_humidity_kg_per_kg"][0]
    assert plant.bed_temperature_so_far[0, 0] == np.asarray(step.state)[0, 3]

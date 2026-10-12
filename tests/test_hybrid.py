"""Hybrid model channels and the simulator-only predictor (study A1)."""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import twinlink_ml.models.hybrid as hybrid
from twinlink_ml.evaluation.experiments import DataSpec, build_datasets
from twinlink_ml.evaluation.plant_data import generate_plant_data
from twinlink_ml.models.hybrid import (
    SIM_PREFIX,
    SimulatorOnlyModel,
    simulator_channels,
    stationary_initial_states,
    with_simulator_channels,
)
from twinlink_ml.models.transfer import build_datasets_with_scalers
from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    PlaceholderConfigWarning,
    load_process_config,
)
from twinlink_ml.simulation.plant_variants import PLANT_VARIANTS

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_gab_placeholder.yaml"
)


@pytest.fixture(scope="module")
def config():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return load_process_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def data(config):
    spec = DataSpec.new_study(
        run_duration_s=1_500.0, n_train_max=4, n_val=2, n_test=2, history_length=8, horizon=4
    )
    return generate_plant_data(config, PLANT_VARIANTS["P3"], spec, master_seed=5)


@pytest.fixture(scope="module")
def hybrid_data(data):
    return with_simulator_channels(data)


def test_channels_are_appended_with_names_and_shape(data, hybrid_data) -> None:
    targets = data.spec.target_names
    n_before = data.trajectory.disturbances.shape[-1]
    names = tuple(hybrid_data.trajectory.metadata["disturbance_names"])
    assert names[-len(targets) :] == tuple(f"{SIM_PREFIX}{t}" for t in targets)
    assert hybrid_data.trajectory.disturbances.shape[-1] == n_before + len(targets)
    assert np.array_equal(
        hybrid_data.trajectory.disturbances[..., :n_before], data.trajectory.disturbances
    )
    assert np.array_equal(hybrid_data.trajectory.measurements, data.trajectory.measurements)
    info = hybrid_data.trajectory.metadata["simulator_channels"]
    assert info["n_runs"] == data.trajectory.n_runs
    assert info["n_stationary"] + info["n_nominal_fallback"] == info["n_runs"]


def test_applying_twice_is_an_error(hybrid_data) -> None:
    with pytest.raises(ValueError, match="already has simulator channels"):
        with_simulator_channels(hybrid_data)


def test_channels_depend_on_the_inputs_only_not_on_any_plant_measurement(data) -> None:
    trajectory = data.trajectory
    rng = np.random.default_rng(0)
    tampered = dataclasses.replace(
        trajectory,
        measurements=rng.normal(size=trajectory.measurements.shape),
        outputs=rng.normal(size=trajectory.outputs.shape),
        states=rng.normal(size=trajectory.states.shape),
    )
    a, _ = simulator_channels(data.base_config, trajectory, data.spec.target_names)
    b, _ = simulator_channels(data.base_config, tampered, data.spec.target_names)
    assert np.array_equal(a, b)


def test_stationary_initial_state_is_stationary(data) -> None:
    trajectory = data.trajectory
    actions = trajectory.actions[:, 0, :]
    disturbances = trajectory.disturbances[:, 0, :]
    states, solved, _ = stationary_initial_states(data.base_config, actions, disturbances)
    # some excited start points have no steady state (low gas velocity, high spray: the bed
    # floods) -- those runs take the nominal fallback, only the solved ones are checked
    assert solved.any()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        simulator = GreyBoxFluidBedGranulator(data.base_config)
    backend = simulator.backend
    for run in np.flatnonzero(solved):
        derivative, _ = simulator.evaluate_derivative(
            backend.asarray(states[run : run + 1]),
            backend.asarray(actions[run : run + 1]),
            backend.asarray(disturbances[run : run + 1]),
        )
        scale = np.maximum(np.abs(states[run]), 1.0)
        assert np.all(np.abs(backend.to_numpy(derivative)[0]) / scale < 1e-4)


def test_failed_solves_fall_back_to_the_nominal_state_and_are_counted(monkeypatch, data) -> None:
    trajectory = data.trajectory
    actions, disturbances = trajectory.actions[:, 0, :], trajectory.disturbances[:, 0, :]
    _, genuine, _ = stationary_initial_states(data.base_config, actions, disturbances)
    expected = genuine.copy()
    expected[1] = False  # the injected failure (run 1 may already fail genuinely)

    real = hybrid.solve_steady_state
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("no sign change")
        return real(*args, **kwargs)

    monkeypatch.setattr(hybrid, "solve_steady_state", flaky)
    states, solved, _ = stationary_initial_states(data.base_config, actions, disturbances)
    assert solved.tolist() == expected.tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        nominal = GreyBoxFluidBedGranulator(data.base_config)
    nominal_state = nominal.backend.to_numpy(nominal.initial_state(1))[0]
    for run in np.flatnonzero(~solved):
        assert np.array_equal(states[run], nominal_state)
    calls["n"] = 0
    _, info = simulator_channels(data.base_config, trajectory, data.spec.target_names)
    assert info.n_nominal_fallback == int((~expected).sum())
    assert info.n_stationary == int(expected.sum())


def test_window_dataset_carries_the_channels_in_history_and_future(data, hybrid_data) -> None:
    plain, _, _ = build_datasets(data, 2)
    train, _, _ = build_datasets(hybrid_data, 2)
    n_targets = len(data.spec.target_names)
    assert len(train.exogenous_names) == len(plain.exogenous_names) + n_targets
    batch = train.batch([0])
    assert batch.history_inputs.shape[-1] == plain.batch([0]).history_inputs.shape[-1] + n_targets
    assert batch.future_exogenous.shape[-1] == len(train.exogenous_names)


def test_simulator_only_predicts_the_simulator_channel(data, hybrid_data) -> None:
    train, _, test = build_datasets_with_scalers(
        hybrid_data, 2, build_datasets(hybrid_data, 4)[0].scalers
    )
    model = SimulatorOnlyModel()
    model.fit(train, seed=0)
    batch = test.batch([0, 3])
    prediction = test.target_scaler.inverse(model.predict(batch))
    channels, _ = simulator_channels(
        hybrid_data.base_config, data.trajectory, data.spec.target_names
    )
    run_positions = [data.trajectory.run_id.index(r) for r in test.run_ids]
    for row in range(2):
        run = run_positions[int(batch.run_index[row])]
        start = int(batch.window_start[row]) + data.spec.history_length
        expected = channels[run, start : start + data.spec.horizon, :]
        assert np.allclose(prediction[row], expected, rtol=1e-9, atol=1e-12)


def test_simulator_only_needs_the_channels(data) -> None:
    train, _, _ = build_datasets(data, 2)
    with pytest.raises(ValueError, match="no simulator channels"):
        SimulatorOnlyModel().fit(train, seed=0)

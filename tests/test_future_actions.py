r"""Review D1 (2026-10-09): ``future_exogenous`` carries the planned set points.

With the gas-velocity projection on, the realised action is clipped to
``[u_mf(d50), u_t(d50)]`` computed from the TRUE grain state. If that clipped series enters
the "known" future inputs of a window, the model is handed information about the state.
``WindowConfig.future_actions="planned"`` (``DataSpec.future_actions``) takes the requested
set points instead. The option defaults to ``"realised"`` (behaviour up to Befund 81) so that
existing scripts and records reproduce; the regression tests below pin that.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.data.point_split import PointSplitter, SplitRatios, windowed_point_split
from twinlink_ml.data.windows import TrajectoryWindowDataset, WindowConfig
from twinlink_ml.evaluation import DataSpec, build_datasets, generate_data
from twinlink_ml.simulation import PlaceholderConfigWarning, ProcessConfig, load_process_config

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_gab_placeholder.yaml"
)
SPEC = DataSpec(
    sample_time_s=25.0,
    run_duration_s=6000.0,
    n_train_max=4,
    n_val=2,
    n_test=2,
    history_length=8,
    horizon=6,
    stride=4,
)

# Values of the data fingerprints of ``SPEC`` with the projection on, recorded at c40d452
# (before the option existed). They must not move for the default ``"realised"``.
LEGACY_FINGERPRINTS = {
    "process/fingerprint": "bddca37f716f5cfc",
    "excitation/fingerprint": "38f9542b",
    "split/digest": "a5a165f3abeb330e",
    "measurement/digest": "c54e5b6864aa",
}


def _config(*, projection: bool) -> ProcessConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    return dataclasses.replace(
        config,
        numerics=dataclasses.replace(config.numerics, enforce_gas_velocity_projection=projection),
    )


def _expected_future(dataset: TrajectoryWindowDataset, source, run_ids) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Action values over the horizon of every window, taken directly from ``source``."""
    history, horizon = SPEC.history_length, SPEC.horizon
    windows = []
    for run, start in dataset._index:
        row = run_ids.index(dataset.run_ids[run])
        windows.append(source[row, start + history : start + history + horizon])
    return np.stack(windows)


@pytest.fixture(scope="module")
def projected_data():  # type: ignore[no-untyped-def]
    return generate_data(_config(projection=True), SPEC, master_seed=7)


def _future_actions_physical(dataset: TrajectoryWindowDataset, n_actions: int) -> np.ndarray:
    """Un-standardised action columns of ``future_exogenous`` over all windows."""
    batch = dataset.batch(np.arange(dataset.n_windows))
    scaler = dataset.input_scaler.select(range(batch.future_exogenous.shape[-1]))
    return scaler.inverse(batch.future_exogenous)[..., :n_actions]


def test_projection_is_active_in_the_fixture(projected_data) -> None:  # type: ignore[no-untyped-def]
    trajectory = projected_data.trajectory
    assert trajectory.requested_actions is not None
    assert trajectory.n_projected_steps is not None
    assert int(trajectory.n_projected_steps.sum()) > 0, "no run is projected, test is void"


def test_planned_future_equals_requested_and_differs_from_realised(projected_data) -> None:  # type: ignore[no-untyped-def]
    data = dataclasses.replace(
        projected_data, spec=dataclasses.replace(SPEC, future_actions="planned")
    )
    trajectory = projected_data.trajectory
    assert trajectory.requested_actions is not None
    n_actions = trajectory.actions.shape[-1]

    # train split: contains run 1, the run in which the projection acts
    planned = build_datasets(data, 4)[0]
    realised = build_datasets(projected_data, 4)[0]
    planned_actions = _future_actions_physical(planned, n_actions)
    realised_actions = _future_actions_physical(realised, n_actions)

    run_ids = trajectory.run_id
    np.testing.assert_allclose(
        planned_actions,
        _expected_future(planned, trajectory.requested_actions, run_ids),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        realised_actions, _expected_future(realised, trajectory.actions, run_ids), atol=1e-9
    )
    assert np.abs(planned_actions - realised_actions).max() > 0.0, "no window is affected"


def test_planned_future_is_independent_of_the_grain_state(projected_data) -> None:  # type: ignore[no-untyped-def]
    """The planned series is the excitation schedule itself -- no state enters."""
    data = dataclasses.replace(
        projected_data, spec=dataclasses.replace(SPEC, future_actions="planned")
    )
    n_actions = projected_data.trajectory.actions.shape[-1]
    planned = build_datasets(data, 4)[0]
    schedule = projected_data.schedule
    np.testing.assert_allclose(
        _future_actions_physical(planned, n_actions),
        _expected_future(planned, schedule.actions, schedule.run_id),
        atol=1e-9,
    )


def test_history_inputs_are_unchanged_by_the_option(projected_data) -> None:  # type: ignore[no-untyped-def]
    planned_data = dataclasses.replace(
        projected_data, spec=dataclasses.replace(SPEC, future_actions="planned")
    )
    planned = build_datasets(planned_data, 4)[2]
    realised = build_datasets(projected_data, 4)[2]
    indices = np.arange(realised.n_windows)
    np.testing.assert_array_equal(
        planned.batch(indices).history_inputs, realised.batch(indices).history_inputs
    )
    np.testing.assert_array_equal(
        planned.batch(indices).future_targets, realised.batch(indices).future_targets
    )


def test_without_projection_the_option_is_a_no_op() -> None:
    data = generate_data(_config(projection=False), SPEC, master_seed=7)
    assert data.trajectory.requested_actions is None
    planned_data = dataclasses.replace(
        data, spec=dataclasses.replace(SPEC, future_actions="planned")
    )
    a = build_datasets(data, 4)[2]
    b = build_datasets(planned_data, 4)[2]
    indices = np.arange(a.n_windows)
    np.testing.assert_array_equal(
        a.batch(indices).future_exogenous, b.batch(indices).future_exogenous
    )


def test_point_split_follows_the_option(projected_data) -> None:  # type: ignore[no-untyped-def]
    trajectory = projected_data.trajectory
    run_ids = trajectory.run_id
    splitter = PointSplitter(seed=1, ratios=SplitRatios())
    base = WindowConfig(history_length=8, horizon=1, stride=1, target_names=("energy_target",))
    planned_cfg = dataclasses.replace(base, future_actions="planned")
    _, _, realised_test = windowed_point_split(trajectory, run_ids, base, splitter)
    _, _, planned_test = windowed_point_split(trajectory, run_ids, planned_cfg, splitter)
    indices = np.arange(realised_test.n_windows)
    assert not np.array_equal(
        realised_test.batch(indices).future_exogenous,
        planned_test.batch(indices).future_exogenous,
    )
    np.testing.assert_array_equal(
        realised_test.batch(indices).history_inputs, planned_test.batch(indices).history_inputs
    )


# ------------------------------------------------ fingerprints of old configurations


def test_default_fingerprints_are_the_legacy_values(projected_data) -> None:  # type: ignore[no-untyped-def]
    assert SPEC.future_actions == "realised"
    assert projected_data.fingerprints == LEGACY_FINGERPRINTS
    assert "window/future_actions" not in projected_data.fingerprints


def test_planned_enters_the_fingerprints_only_when_set(projected_data) -> None:  # type: ignore[no-untyped-def]
    planned = dataclasses.replace(
        projected_data, spec=dataclasses.replace(SPEC, future_actions="planned")
    )
    fingerprints = planned.fingerprints
    assert fingerprints["window/future_actions"] == "planned"
    assert {k: v for k, v in fingerprints.items() if k != "window/future_actions"} == (
        LEGACY_FINGERPRINTS
    )


def test_window_config_default_is_realised() -> None:
    assert WindowConfig(history_length=2, horizon=1, target_names=("a",)).future_actions == (
        "realised"
    )

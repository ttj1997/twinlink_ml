"""Shared data for the tests of :mod:`twinlink_ml.api`: one small pool, built once per session.

The pool follows the new-study preset (planned future set points), so the requests built here take
the planned inputs from the same series as ``TrajectoryWindowDataset`` does.
"""

from __future__ import annotations

import functools
import warnings
from pathlib import Path

import numpy as np

from twinlink_ml.api import ForecastRequest
from twinlink_ml.api.schemas import EXOGENOUS_FIELDS, TARGET_FIELD_OF_MODEL_NAME
from twinlink_ml.data.windows import TrajectoryWindowDataset, WindowBatch, future_actions_of
from twinlink_ml.evaluation import DataSpec, build_datasets, generate_data
from twinlink_ml.evaluation.experiments import GeneratedData
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_gab_placeholder.yaml"
)
TARGETS = ("energy_target", "bed_temperature_celsius", "product_moisture_kg_per_kg")
N_TRAIN = 8
SAMPLE_TIME_S = 25.0


def spec(observed: tuple[str, ...] | None = None) -> DataSpec:
    """Small pool: 16 runs of 4000 s, windows of 16 + 8 steps."""
    return DataSpec.new_study(
        sample_time_s=SAMPLE_TIME_S,
        run_duration_s=4000.0,
        n_train_max=N_TRAIN,
        n_val=4,
        n_test=4,
        history_length=16,
        horizon=8,
        stride=8,
        target_names=TARGETS,
        observed_target_names=observed,
    )


@functools.cache
def generated(observed: tuple[str, ...] | None = None) -> GeneratedData:
    """The pool; ``observed`` selects the measured targets (``None``: all, the inline case)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    return generate_data(config, spec(observed), master_seed=1)


@functools.cache
def datasets(
    observed: tuple[str, ...] | None = None,
) -> tuple[TrajectoryWindowDataset, TrajectoryWindowDataset | None, TrajectoryWindowDataset]:
    """Training, validation and test data set of the full training budget."""
    return build_datasets(generated(observed), N_TRAIN)


def window_request(
    data: GeneratedData,
    dataset: TrajectoryWindowDataset,
    index: int,
    *,
    extra_history: int = 0,
    measured: tuple[str, ...] | None = None,
) -> tuple[ForecastRequest, WindowBatch]:
    """The request that describes window ``index`` of ``dataset`` in engineering units.

    Built from the trajectory, not from the standardised arrays of the data set, so that a
    comparison with ``model.predict(batch)`` checks the standardisation of the service as well.
    ``extra_history`` prepends earlier samples; ``measured`` overrides which target histories are
    sent (default: the observed targets of the data set).
    """
    batch = dataset.batch([index])
    run, start = int(batch.run_index[0]), int(batch.window_start[0])
    trajectory = data.trajectory
    row = trajectory.run_id.index(dataset.run_ids[run])
    window = dataset.config
    first, now = start - extra_history, start + window.history_length
    end = now + window.horizon
    if first < 0:
        raise ValueError("extra_history reaches before the start of the run")
    history = np.concatenate([trajectory.actions[row], trajectory.disturbances[row]], axis=-1)
    planned = np.concatenate(
        [future_actions_of(trajectory, window)[row], trajectory.disturbances[row]], axis=-1
    )
    names = (*trajectory.metadata["action_names"], *trajectory.metadata["disturbance_names"])
    assert tuple(names) == EXOGENOUS_FIELDS
    output_names = tuple(trajectory.metadata["output_names"])
    sent = dataset.observed_target_names if measured is None else measured
    measured_history = {
        TARGET_FIELD_OF_MODEL_NAME[name]: trajectory.measurements[row][
            first:now, output_names.index(name)
        ].tolist()
        for name in sent
    }
    request = ForecastRequest(
        sample_time_s=data.spec.sample_time_s,
        observation_mode="inline" if dataset.is_fully_observed else "soft_sensor",
        history={name: history[first:now, i].tolist() for i, name in enumerate(names)},
        measured_history=measured_history,
        planned={name: planned[now:end, i].tolist() for i, name in enumerate(names)},
    )
    return request, batch


def response_matrix(response: object, targets: tuple[str, ...] = TARGETS) -> np.ndarray:
    """``(horizon, n_targets)`` from the fields of a forecast response."""
    return np.stack(
        [np.asarray(getattr(response, TARGET_FIELD_OF_MODEL_NAME[name])) for name in targets],
        axis=-1,
    )

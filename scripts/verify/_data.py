"""Data-generation entry points for the audit.

The audit is allowed to use the simulator and the data generation (``generate_data``); it
checks them with its own code. The three data sets and their specifications are written down
here from the *documented protocol* (Befund 51 text, Befund 55 text) and are cross-checked
against the record parameters in ``pr1a_determinism.py`` -- they are not imported from the
worker scripts, so a silent change in a worker's ``_excited_spec`` would show up as a
mismatch.
"""

from __future__ import annotations

import dataclasses
import hashlib
import warnings
from typing import TYPE_CHECKING

import numpy as np
from _common import CONFIG_PATH, TARGETS

if TYPE_CHECKING:
    from twinlink_ml.evaluation.experiments import GeneratedData
    from twinlink_ml.simulation.config import ProcessConfig

HOLDOUT_SEED = 999  # Befund 51: fixed constant-run holdout, independent of master_seed

# Befund 51 "Konfiguration": 28800 s / 25 s = 1152 steps, 128 runs = 64 + 32 + 32, window
# geometry 32 history + 36 horizon, stride 8, three target channels.
EXCITED = {
    "sample_time_s": 25.0,
    "run_duration_s": 28_800.0,
    "n_train_max": 64,
    "n_val": 32,
    "n_test": 32,
    "history_length": 32,
    "horizon": 36,
    "stride": 8,
    "constant_run_fraction": 0.10,
}
CONSTANT_TRAIN = {**EXCITED, "constant_run_fraction": 1.0}  # Befund 55
HOLDOUT = {**EXCITED, "n_train_max": 2, "n_val": 2, "constant_run_fraction": 1.0}  # Befund 51


def load_config(*, projection: bool = True) -> ProcessConfig:
    """Process config as used for Befund 51 (gas-velocity projection ON)."""
    from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    return dataclasses.replace(
        config,
        numerics=dataclasses.replace(config.numerics, enforce_gas_velocity_projection=projection),
    )


def make_data(kind: str, master_seed: int, *, config: ProcessConfig | None = None) -> GeneratedData:
    """Generate one data set: ``excited`` / ``constant_train`` (master seed) or ``holdout``."""
    from twinlink_ml.evaluation.experiments import DataSpec, generate_data

    parameters = {"excited": EXCITED, "constant_train": CONSTANT_TRAIN, "holdout": HOLDOUT}[kind]
    seed = HOLDOUT_SEED if kind == "holdout" else master_seed
    spec = DataSpec(target_names=TARGETS, **parameters)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return generate_data(
            config if config is not None else load_config(), spec, master_seed=seed
        )


def digest_arrays(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    """SHA-256 of the raw float64/uint8 bytes of each array."""
    return {
        name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for name, value in arrays.items()
    }


def trajectory_arrays(data: GeneratedData) -> dict[str, np.ndarray]:
    """All numeric content of a generated data set that a model could ever see."""
    t = data.trajectory
    arrays: dict[str, np.ndarray] = {
        "actions_realised": t.actions,
        "disturbances": t.disturbances,
        "states": t.states,
        "outputs": t.outputs,
        "measurements": t.measurements,
        "schedule_actions_requested": data.schedule.actions,
        "schedule_initial_states": data.schedule.initial_states,
    }
    if t.requested_actions is not None:
        arrays["requested_actions"] = t.requested_actions
    if t.n_projected_steps is not None:
        arrays["n_projected_steps"] = t.n_projected_steps
    for name in sorted(t.diagnostics):
        arrays[f"diag/{name}"] = t.diagnostics[name]
    return arrays


def run_index(data: GeneratedData, ids: tuple[str, ...]) -> np.ndarray:
    """Row positions of the given run IDs in the trajectory arrays."""
    position = {rid: i for i, rid in enumerate(data.trajectory.run_id)}
    return np.asarray([position[rid] for rid in ids], dtype=np.intp)

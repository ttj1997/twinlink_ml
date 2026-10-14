r"""Ensemble-disagreement acquisition for active learning (arm B5 of study B1).

Question: after each round an ensemble is trained on the runs chosen so far, and the next runs are
the candidates on which the ensemble members disagree most. The disagreement uses **only the
inputs of a candidate**, never its truth:

* a candidate is an ``ExcitationSchedule`` (planned actions, disturbances, physical initial
  state); :class:`~twinlink_ml.evaluation.growing_pool.GrowingPool` offers nothing else before a
  run is added, and every function here takes such a schedule;
* a window needs a history of measured targets that a candidate does not have. It is built from
  the **output map at the plan's physical initial state** (decision of the project lead,
  2026-10-12): the targets at ``t = 0`` computed from the initial state and the first action and
  disturbance, held constant over the first history. After the first window the history is the
  member's **own** prediction (free run): each member is run autoregressively over the whole
  candidate, in non-overlapping windows of the horizon.

Disagreement of a candidate = the standard deviation across the members of the predicted targets
(standardised with the scalers of the current training set), averaged over windows, horizon steps
and targets. Selection is greedy top-``k`` within a round; ties are broken by the lower index.
There is no diversity term in this version (one of the choices the draft names).
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable, Sequence

import numpy as np

from ..data.windows import FloatArray, SequenceDataset, WindowBatch
from ..models.base import SequenceModel
from ..simulation.config import PlaceholderConfigWarning, ProcessConfig
from ..simulation.excitation import ExcitationSchedule
from ..simulation.greybox import IDX_BED_TEMPERATURE, GreyBoxFluidBedGranulator

__all__ = [
    "disagreement",
    "ensemble_scores",
    "free_run",
    "initial_outputs",
    "member_seed",
    "select_top_k",
    "train_ensemble",
]


def member_seed(base_seed: int, member: int) -> int:
    """Seed of ensemble member ``member``; member 0 uses the base seed itself."""
    if member == 0:
        return base_seed
    digest = hashlib.sha256(f"ensemble:{base_seed}:{member}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def train_ensemble(
    factory: Callable[[], SequenceModel],
    train: SequenceDataset,
    *,
    base_seed: int,
    n_members: int,
) -> list[SequenceModel]:
    """``n_members`` models of the same kind, differing in seed (initialisation, batch order)."""
    members = []
    for index in range(n_members):
        model = factory()
        model.fit(train, seed=member_seed(base_seed, index), validation=None)
        members.append(model)
    return members


def initial_outputs(
    config: ProcessConfig, inputs: ExcitationSchedule, target_names: Sequence[str]
) -> FloatArray:
    """Targets at ``t = 0`` from the physical initial state and the first inputs, ``(C, n_y)``.

    Uses the simulator's output map (``evaluate_derivative``) at the plan's initial state; no
    integration, nothing of the candidate's trajectory.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        simulator = GreyBoxFluidBedGranulator(config)
    backend = simulator.backend
    n = inputs.n_runs
    if inputs.initial_states is None:
        state = simulator.initial_state(n)
    else:
        columns = inputs.initial_states
        state = simulator.state_from_physical(
            columns[:, 0], columns[:, 1], columns[:, 2], columns[:, 3]
        )
    _, diagnostics = simulator.evaluate_derivative(
        state,
        backend.asarray(inputs.actions[:, 0, :]),
        backend.asarray(inputs.disturbances[:, 0, :]),
    )
    state_np = backend.to_numpy(state)
    available = {
        "energy_target": backend.to_numpy(diagnostics["energy_target"]),
        "product_moisture_kg_per_kg": backend.to_numpy(diagnostics["particle_moisture_kg_per_kg"]),
        "bed_temperature_celsius": state_np[:, IDX_BED_TEMPERATURE],
    }
    missing = [name for name in target_names if name not in available]
    if missing:
        raise ValueError(f"no output map for targets {missing}")
    return np.stack(
        [np.asarray(available[name], dtype=np.float64) for name in target_names], axis=1
    )


def free_run(
    model: SequenceModel,
    dataset: SequenceDataset,
    inputs: ExcitationSchedule,
    y0: FloatArray,
) -> FloatArray:
    """Autoregressive free run of one model over all candidates.

    Returns the standardised predictions ``(C, windows, horizon, n_y)``. Windows do not
    overlap in their prediction span; the history of a window is the held initial output for
    the first ``history_length`` steps and the member's own earlier predictions afterwards.
    """
    config = dataset.config
    history, horizon = config.history_length, config.horizon
    input_scaler, target_scaler = dataset.input_scaler, dataset.target_scaler
    n_exogenous = len(dataset.exogenous_names)
    n_targets = len(dataset.target_names)
    exogenous = np.concatenate([inputs.actions, inputs.disturbances], axis=-1)
    exogenous_std = input_scaler.select(range(n_exogenous)).transform(exogenous)
    target_channels = input_scaler.select(dataset.autoregressive_index)
    n_candidates, n_steps, _ = exogenous.shape
    if n_steps < history + horizon:
        raise ValueError("candidates are shorter than one window")
    series = np.empty((n_candidates, n_steps, n_targets), dtype=np.float64)
    series[:, :history] = y0[:, None, :]
    predictions: list[FloatArray] = []
    for start in range(0, n_steps - history - horizon + 1, horizon):
        hist_targets = target_channels.transform(series[:, start : start + history])
        batch = WindowBatch(
            history_inputs=np.concatenate(
                [exogenous_std[:, start : start + history], hist_targets], axis=-1
            ),
            future_exogenous=exogenous_std[:, start + history : start + history + horizon],
            future_targets=np.zeros((n_candidates, horizon, n_targets)),
            run_index=np.arange(n_candidates, dtype=np.float64),
            window_start=np.full(n_candidates, float(start)),
        )
        standardised = model.predict(batch)
        series[:, start + history : start + history + horizon] = target_scaler.inverse(standardised)
        predictions.append(standardised)
    return np.stack(predictions, axis=1)


def disagreement(member_predictions: Sequence[FloatArray]) -> FloatArray:
    """Mean standard deviation across members, per candidate ``(C,)``."""
    if len(member_predictions) < 2:
        raise ValueError("disagreement needs at least two members")
    stack = np.stack(member_predictions)
    return np.asarray(stack.std(axis=0, ddof=1).mean(axis=(1, 2, 3)), dtype=np.float64)


def ensemble_scores(
    members: Sequence[SequenceModel],
    dataset: SequenceDataset,
    config: ProcessConfig,
    inputs: ExcitationSchedule,
) -> FloatArray:
    """Disagreement of the ensemble for every candidate of ``inputs`` (inputs only)."""
    y0 = initial_outputs(config, inputs, dataset.target_names)
    return disagreement([free_run(member, dataset, inputs, y0) for member in members])


def select_top_k(scores: FloatArray, candidate_indices: Sequence[int], k: int) -> list[int]:
    """The ``k`` candidates with the largest score; ties go to the lower index."""
    if k > len(candidate_indices):
        raise ValueError("fewer candidates than requested")
    order = sorted(range(len(candidate_indices)), key=lambda i: (-float(scores[i]), i))
    return [candidate_indices[i] for i in order[:k]]

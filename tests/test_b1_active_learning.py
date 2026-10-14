"""B1.3: ensemble-disagreement acquisition (B5): inputs only, free run, selection."""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.data.splits import SplitName
from twinlink_ml.evaluation import growing_pool
from twinlink_ml.evaluation.active_learning import (
    disagreement,
    ensemble_scores,
    free_run,
    initial_outputs,
    member_seed,
    select_top_k,
    train_ensemble,
)
from twinlink_ml.evaluation.experiments import DataSpec, generate_data
from twinlink_ml.evaluation.growing_pool import (
    GrowingPool,
    guarded_config,
    prepare_process_config,
    window_dataset,
)
from twinlink_ml.models.base import SequenceModel
from twinlink_ml.models.narx import RandomFeatureNARX
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

TARGETS = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
SPEC = DataSpec.new_study(n_train_max=4, n_val=0, n_test=0, target_names=TARGETS)


@pytest.fixture(scope="module")
def setup():
    path = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_gab_placeholder.yaml"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = guarded_config(load_process_config(path))
    spec = dataclasses.replace(SPEC, run_duration_s=6000.0)
    data = generate_data(
        config, dataclasses.replace(spec, n_train_max=4, n_val=2, n_test=2), master_seed=2
    )
    pool = GrowingPool(config, spec, data.schedule, noise_seed=1)
    pool.add([0, 1, 2])
    train = window_dataset(pool.trajectory(), spec.window, role=SplitName.TRAIN)
    return config, spec, data, pool, train


def test_member_seed_zero_is_the_base_seed_and_others_differ():
    assert member_seed(123, 0) == 123
    seeds = {member_seed(123, i) for i in range(1, 6)}
    assert len(seeds) == 5 and 123 not in seeds
    assert member_seed(123, 3) == member_seed(123, 3)


def test_initial_outputs_are_the_output_map_at_the_physical_initial_state(setup):
    config, spec, data, pool, _ = setup
    prepared = prepare_process_config(config, spec)
    inputs = pool.candidate_inputs([3, 4, 5])
    y0 = initial_outputs(prepared, inputs, TARGETS)
    columns = inputs.initial_states
    assert y0.shape == (3, 3)
    assert np.allclose(y0[:, 1], columns[:, 1], rtol=1e-9)  # product moisture = initial moisture
    assert np.allclose(y0[:, 2], columns[:, 3])  # bed temperature = initial bed temperature
    with pytest.raises(ValueError):
        initial_outputs(prepared, inputs, ("d50_um",))


def test_disagreement_zero_for_identical_members_and_orders_by_spread():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(5, 3, 24, 3))
    assert np.allclose(disagreement([base, base, base]), 0.0)
    wide = np.stack([base, base + 1.0 * np.arange(5)[:, None, None, None]])
    narrow = np.stack([base, base + 0.1 * np.arange(5)[:, None, None, None]])
    scores_wide = disagreement(list(wide))
    scores_narrow = disagreement(list(narrow))
    assert np.all(np.diff(scores_wide) > 0)  # candidate 0 has no disagreement, 4 the most
    assert np.all(scores_wide[1:] > scores_narrow[1:])
    with pytest.raises(ValueError):
        disagreement([base])


def test_select_top_k_breaks_ties_by_the_lower_index():
    scores = np.array([0.5, 0.9, 0.9, 0.1])
    assert select_top_k(scores, [10, 11, 12, 13], 2) == [11, 12]
    assert select_top_k(np.zeros(3), [7, 8, 9], 2) == [7, 8]
    with pytest.raises(ValueError):
        select_top_k(scores, [1, 2], 3)


class _RecordingModel(SequenceModel):
    """Predicts a constant (in standardised units) and records every batch it sees."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.batches: list = []

    @property
    def name(self) -> str:
        return "recording"

    @property
    def n_parameters(self) -> int:
        return 0

    def fit(self, train, *, seed, validation=None) -> None:  # noqa: ARG002
        return None

    def predict(self, batch):
        self.batches.append(batch)
        n, horizon = batch.future_exogenous.shape[:2]
        return np.full((n, horizon, 3), self.value)


def test_free_run_history_is_the_held_initial_output_then_the_members_own_predictions(setup):
    config, spec, _, pool, train = setup
    prepared = prepare_process_config(config, spec)
    inputs = pool.candidate_inputs([3, 4])
    y0 = initial_outputs(prepared, inputs, TARGETS)
    model = _RecordingModel(value=0.7)
    predictions = free_run(model, train, inputs, y0)
    history, horizon = spec.history_length, spec.horizon
    n_windows = (inputs.actions.shape[1] - history - horizon) // horizon + 1
    assert predictions.shape == (2, n_windows, horizon, 3)
    n_exogenous = len(train.exogenous_names)
    channels = train.input_scaler.select(train.autoregressive_index)
    first, second = model.batches[0], model.batches[1]
    held = channels.transform(np.repeat(y0[:, None, :], history, axis=1))
    assert np.allclose(first.history_inputs[:, :, n_exogenous:], held)
    # second window starts one horizon later: the first (history - horizon) steps are still the
    # held initial output, the last horizon steps are the first window's predictions
    predicted_physical = train.target_scaler.inverse(np.full((2, horizon, 3), 0.7))
    expected_tail = channels.transform(predicted_physical)
    assert np.allclose(second.history_inputs[:, history - horizon :, n_exogenous:], expected_tail)
    assert np.allclose(
        second.history_inputs[:, : history - horizon, n_exogenous:], held[:, : history - horizon]
    )
    # the batches carry no targets of the candidate
    assert np.all(first.future_targets == 0.0)


def test_scoring_needs_no_simulation_and_only_inputs(setup, monkeypatch):
    config, spec, _, pool, train = setup
    prepared = prepare_process_config(config, spec)
    members = train_ensemble(lambda: RandomFeatureNARX(), train, base_seed=5, n_members=3)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("scoring must not simulate")

    monkeypatch.setattr(growing_pool, "simulate_schedule", forbidden)
    inputs = pool.candidate_inputs(list(pool.remaining))
    scores = ensemble_scores(members, train, prepared, inputs)
    assert scores.shape == (len(pool.remaining),)
    assert np.all(np.isfinite(scores)) and np.all(scores >= 0.0)
    assert scores.max() > 0.0  # random-feature members with different seeds do disagree
    again = ensemble_scores(members, train, prepared, inputs)
    assert np.array_equal(scores, again)

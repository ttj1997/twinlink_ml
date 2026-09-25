"""Tests für den punktweisen Split (Kräuter-Vergleichsvariante, ``data/point_split.py``).

Schwerpunkt: die beiden Invarianten, auf die der Splitvergleich mit dem laufweisen
Pfad angewiesen ist -- Fenster verschiedener Splits dürfen aus demselben Lauf stammen
(das ist der ganze Punkt), aber der Skalierer darf ausschließlich aus den train-Fenstern
stammen, nie aus val/test.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.data import (
    PointSplitter,
    SplitDegenerateError,
    SplitRatios,
    TrajectoryWindowDataset,
    WindowConfig,
    windowed_point_split,
)
from twinlink_ml.simulation import (
    ExcitationConfig,
    GreyBoxFluidBedGranulator,
    MixedExcitationDesign,
    PlaceholderConfigWarning,
    load_process_config,
)
from twinlink_ml.simulation.base import Trajectory
from twinlink_ml.utils.seeding import SeedBundle

CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_placeholder.yaml"

WINDOW = WindowConfig(
    history_length=3,
    horizon=1,
    stride=1,
    target_names=("energy_target", "product_moisture_kg_per_kg"),
)


@pytest.fixture(scope="module")
def trajectory_and_ids() -> tuple[object, tuple[str, ...]]:
    """Kleiner, aber echter Datensatz -- derselbe Aufbau wie ``test_pipeline.pipeline``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base = load_process_config(CONFIG_PATH)
        config = dataclasses.replace(
            base, numerics=dataclasses.replace(base.numerics, sample_time_s=25.0)
        )
        simulator = GreyBoxFluidBedGranulator(config)
    design = MixedExcitationDesign(
        ExcitationConfig(run_duration_s=6000.0),
        config.actions,
        config.disturbances,
        sample_time_s=config.numerics.sample_time_s,
        geometry=config.geometry,
        material=config.material,
        operation=config.operation,
    )
    seeds = SeedBundle.from_master(4242)
    schedule = design.build(16, seed=seeds.data_seed)
    columns = schedule.initial_states
    assert columns is not None
    initial = simulator.state_from_physical(
        columns[:, 0], columns[:, 1], columns[:, 2], columns[:, 3]
    )
    trajectory = simulator.rollout(
        schedule.actions,
        schedule.disturbances,
        run_id=schedule.run_id,
        seed=seeds.noise_seed,
        initial_state=initial,
    )
    return trajectory, schedule.run_id


def _split(trajectory_and_ids, ratios=None, seed=1):
    trajectory, run_ids = trajectory_and_ids
    splitter = PointSplitter(seed=seed, ratios=ratios if ratios is not None else SplitRatios())
    return windowed_point_split(trajectory, run_ids, WINDOW, splitter)


def test_point_split_windows_of_same_run_can_land_on_both_sides(trajectory_and_ids) -> None:
    """Der ganze Punkt des punktweisen Splits -- Gegenteil des laufweisen Verhaltens."""
    train, _, test = _split(trajectory_and_ids)
    assert set(train.run_ids) == set(test.run_ids)
    train_runs = {train.run_ids[i] for i in train._index[:, 0]}
    test_runs = {test.run_ids[i] for i in test._index[:, 0]}
    assert train_runs & test_runs, "erwartete Überschneidung der Lauf-Mengen fehlt"


def test_point_split_windows_are_disjoint(trajectory_and_ids) -> None:
    train, val, test = _split(trajectory_and_ids)
    assert val is not None

    def keys(dataset: TrajectoryWindowDataset) -> set[tuple[int, int]]:
        return {(int(run), int(start)) for run, start in dataset._index}

    train_keys, val_keys, test_keys = keys(train), keys(val), keys(test)
    assert not (train_keys & val_keys)
    assert not (train_keys & test_keys)
    assert not (val_keys & test_keys)


def test_point_split_scaler_comes_from_training_windows_only(trajectory_and_ids) -> None:
    """Analog zu ``test_pipeline.test_test_scaler_comes_from_training_split``.

    Ein Skalierer, der stattdessen aus dem Testanteil geschätzt wird, muss andere
    Statistiken liefern -- sonst prüft der Test nichts.
    """
    train, _, test = _split(trajectory_and_ids)
    assert np.allclose(train.target_scaler.mean, test.target_scaler.mean)
    assert np.allclose(train.input_scaler.mean, test.input_scaler.mean)

    trajectory, run_ids = trajectory_and_ids
    swapped = PointSplitter(seed=1, ratios=SplitRatios(0.15, 0.15, 0.70))
    swapped_train, _, _ = windowed_point_split(trajectory, run_ids, WINDOW, swapped)
    assert not np.allclose(train.target_scaler.mean, swapped_train.target_scaler.mean)


def test_point_split_ratios_converge(trajectory_and_ids) -> None:
    train, val, test = _split(
        trajectory_and_ids, ratios=SplitRatios(0.70, 0.15, 0.15), seed=7
    )
    total = train.n_windows + val.n_windows + test.n_windows
    assert train.n_windows / total == pytest.approx(0.70, abs=0.03)
    assert test.n_windows / total == pytest.approx(0.15, abs=0.03)


def test_point_split_val_optional_when_ratio_zero(trajectory_and_ids) -> None:
    train, val, test = _split(
        trajectory_and_ids, ratios=SplitRatios(0.85, 0.0, 0.15), seed=3
    )
    assert val is None
    assert train.n_windows > 0 and test.n_windows > 0


def test_point_split_rejects_degenerate_pool() -> None:
    """Zu wenige Fenster im Testanteil -- ``min_per_split`` greift wie beim laufweisen Pfad.

    Synthetische, minimale Trajektorie statt eines echten Rollouts: nur die Fensterzahl
    zählt hier, nicht die Physik. ``seed=1``/``ratios`` sind vorab so gewählt (siehe
    Kommentar), dass bei diesem winzigen Pool *jedes* Fenster nach ``train`` fällt --
    ``test`` bleibt leer und muss den Fehler auslösen.
    """
    n_steps = 20
    rng = np.random.default_rng(0)
    trajectory = Trajectory(
        run_id=("only-run",),
        time_s=np.arange(n_steps, dtype=np.float64),
        actions=rng.normal(size=(1, n_steps, 1)),
        disturbances=rng.normal(size=(1, n_steps, 1)),
        states=rng.normal(size=(1, n_steps, 1)),
        outputs=rng.normal(size=(1, n_steps, 2)),
        measurements=rng.normal(size=(1, n_steps, 2)),
        metadata={
            "output_names": ("energy_target", "product_moisture_kg_per_kg"),
            "action_names": ("u1",),
            "disturbance_names": ("z1",),
        },
    )
    # Verifiziert (siehe Docstring): mit diesem Seed/diesen Anteilen landet bei einem
    # einzigen Lauf mit 20 Schritten jedes der 17 möglichen Fenster (history=3, horizon=1)
    # in "train" -- "test" ist damit garantiert leer, nicht nur unwahrscheinlich klein.
    splitter = PointSplitter(seed=1, ratios=SplitRatios(0.99, 0.0, 0.01))
    with pytest.raises(SplitDegenerateError):
        windowed_point_split(trajectory, trajectory.run_id, WINDOW, splitter)


def test_point_split_matches_run_level_dataset_shapes(trajectory_and_ids) -> None:
    """Fensterform ist identisch zum laufweisen Pfad -- nur die Zuordnung unterscheidet sich."""
    train, _, _ = _split(trajectory_and_ids)
    batch = train.batch(range(min(16, train.n_windows)))
    n_exogenous = len(train.exogenous_names)
    n_targets = len(train.target_names)
    assert batch.history_inputs.shape[1:] == (3, n_exogenous + n_targets)
    assert batch.future_targets.shape[1:] == (1, n_targets)

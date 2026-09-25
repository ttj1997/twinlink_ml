"""Tests für :class:`~twinlink_ml.models.extra_trees.ExtraTreesModel`.

Läuft nur mit installiertem scikit-learn (optionales Extra, siehe ``pyproject.toml``);
ohne sklearn werden die Tests übersprungen, analog zu ``HAS_TORCH`` in
``test_composite.py``.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.data import (
    IndexBlockSplitter,
    SplitName,
    TrajectoryWindowDataset,
    WindowConfig,
    split_schedule,
)
from twinlink_ml.models import ExtraTreesModel
from twinlink_ml.simulation import (
    ExcitationConfig,
    GreyBoxFluidBedGranulator,
    MixedExcitationDesign,
    PlaceholderConfigWarning,
    load_process_config,
)
from twinlink_ml.utils.seeding import SeedBundle

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
pytestmark = pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn nicht installiert")

CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_placeholder.yaml"

WINDOW = WindowConfig(
    history_length=3,
    horizon=1,
    stride=1,
    target_names=("energy_target", "product_moisture_kg_per_kg"),
)


@pytest.fixture(scope="module")
def pipeline():
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
    seeds = SeedBundle.from_master(99)
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
    split = split_schedule(schedule, IndexBlockSplitter(n_train=8, n_val=4, n_test=4))
    return trajectory, split


def _datasets(pipeline):
    trajectory, split = pipeline
    train = TrajectoryWindowDataset.from_trajectory(trajectory, split, SplitName.TRAIN, WINDOW)
    test = TrajectoryWindowDataset.from_trajectory(
        trajectory, split, SplitName.TEST, WINDOW, scalers=train.scalers
    )
    return train, test


def test_extra_trees_rejects_horizon_other_than_one(pipeline) -> None:
    trajectory, split = pipeline
    wide = dataclasses.replace(WINDOW, horizon=4)
    train = TrajectoryWindowDataset.from_trajectory(trajectory, split, SplitName.TRAIN, wide)
    with pytest.raises(ValueError, match="Ein-Schritt"):
        ExtraTreesModel(n_estimators=5).fit(train, seed=0)


def test_extra_trees_rejects_short_history(pipeline) -> None:
    trajectory, split = pipeline
    short = dataclasses.replace(WINDOW, history_length=2)
    train = TrajectoryWindowDataset.from_trajectory(trajectory, split, SplitName.TRAIN, short)
    with pytest.raises(ValueError, match="history_length"):
        ExtraTreesModel(n_estimators=5).fit(train, seed=0)


def test_extra_trees_feature_channels_exclude_unobserved_targets(pipeline) -> None:
    """Nicht inline gemessene Zielgrößen dürfen keine Merkmale liefern.

    Ihr Historienkanal ist konstant der Trainingsmittelwert -- eine Ableitung davon ist
    eine Ableitung von Rauschen um eine Konstante, siehe Moduldocstring.
    """
    trajectory, split = pipeline
    window = dataclasses.replace(WINDOW, observed_target_names=("energy_target",))
    train = TrajectoryWindowDataset.from_trajectory(trajectory, split, SplitName.TRAIN, window)
    model = ExtraTreesModel(n_estimators=5)
    model.fit(train, seed=0)
    n_exogenous = len(train.exogenous_names)
    # energy_target (Index 0 der Zielgrößen) ist beobachtet, product_moisture (Index 1) nicht.
    assert model._feature_channels == (*range(n_exogenous), n_exogenous)


def test_extra_trees_fits_and_predicts_one_step(pipeline) -> None:
    train, test = _datasets(pipeline)
    model = ExtraTreesModel(n_estimators=10)
    model.fit(train, seed=0)
    batch = test.batch(range(min(32, test.n_windows)))
    prediction = model.predict(batch)
    assert prediction.shape == (batch.n_windows, 1, len(train.target_names))
    assert np.all(np.isfinite(prediction))


def test_extra_trees_predict_rejects_multi_step_batch(pipeline) -> None:
    trajectory, split = pipeline
    wide = dataclasses.replace(WINDOW, horizon=1)
    train = TrajectoryWindowDataset.from_trajectory(trajectory, split, SplitName.TRAIN, wide)
    model = ExtraTreesModel(n_estimators=5)
    model.fit(train, seed=0)
    batch = train.full_batch()
    tampered = dataclasses.replace(
        batch,
        future_exogenous=np.repeat(batch.future_exogenous, 2, axis=1),
    )
    with pytest.raises(ValueError, match="h=1"):
        model.predict(tampered)


def test_extra_trees_is_deterministic_given_seed(pipeline) -> None:
    train, test = _datasets(pipeline)
    batch = test.batch(range(min(16, test.n_windows)))
    first = ExtraTreesModel(n_estimators=10)
    second = ExtraTreesModel(n_estimators=10)
    first.fit(train, seed=3)
    second.fit(train, seed=3)
    assert np.allclose(first.predict(batch), second.predict(batch))


def test_extra_trees_beats_climatology_on_energy(pipeline) -> None:
    """Inhaltliche Mindestzusicherung, analog ``test_arx_beats_static_regression_on_energy``."""
    from twinlink_ml.evaluation import RolloutEvaluator
    from twinlink_ml.models import ClimatologyModel

    train, test = _datasets(pipeline)
    evaluator = RolloutEvaluator(horizons=(1,))
    scores = {}
    for model in (ClimatologyModel(), ExtraTreesModel(n_estimators=50)):
        model.fit(train, seed=0)
        result = evaluator.evaluate(model, test)
        scores[model.name] = result.nrmse_run_spread["energy_target"][1][0]
    assert scores["extra_trees50"] < scores["climatology"]

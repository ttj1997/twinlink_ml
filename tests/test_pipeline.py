"""Tests für Splits, Fensterung, Baselines und Evaluator (Arbeitsplan Schritte 3 und 4).

Schwerpunkt sind die Fehler, die im Betrieb stumm bleiben: prozessabhängige Hashes,
Skalierer aus dem falschen Split, Fenster über Laufgrenzen hinweg, und ein Rollout, der
heimlich die Wahrheit statt der eigenen Vorhersage fortschreibt.
"""

from __future__ import annotations

import dataclasses
import itertools
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.data import (
    DEFAULT_LEARNING_CURVE_SIZES,
    HashRunSplitter,
    IndexBlockSplitter,
    PredefinedRunSplitter,
    RunSplit,
    SplitConfigError,
    SplitDegenerateError,
    SplitError,
    SplitName,
    SplitRatios,
    Standardizer,
    TrajectoryWindowDataset,
    WindowConfig,
    design_key,
    learning_curve,
    realized_ratios,
    signal_kind_strata,
    split_report,
    split_schedule,
    stable_unit_hash,
    stratum_coverage,
)
from twinlink_ml.evaluation import RolloutEvaluator, nrmse
from twinlink_ml.evaluation.evaluator import per_run_nrmse, pooled_nrmse
from twinlink_ml.models import LinearARXModel, PersistenceModel, StaticRegressionModel
from twinlink_ml.simulation import (
    ExcitationConfig,
    GreyBoxFluidBedGranulator,
    MixedExcitationDesign,
    PlaceholderConfigWarning,
    load_process_config,
)
from twinlink_ml.simulation.excitation import _constant_run_flags
from twinlink_ml.utils.seeding import SeedBundle

CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_placeholder.yaml"


def make_ids(n: int, prefix: str = "mixed-deadbeef") -> tuple[str, ...]:
    """Lauf-IDs im Schema von ``MixedExcitationDesign.build``."""
    return tuple(f"{prefix}-{i:05d}" for i in range(n))


# ------------------------------------------------------------------------- Hashing


def test_stable_unit_hash_in_unit_interval() -> None:
    values = [stable_unit_hash("run", i) for i in range(500)]
    assert all(0.0 <= value < 1.0 for value in values)


def test_stable_unit_hash_avoids_concatenation_collision() -> None:
    assert stable_unit_hash("run", "12") != stable_unit_hash("run1", "2")


def test_stable_unit_hash_independent_of_python_hash_seed() -> None:
    """Regression gegen die Verwendung von ``hash()``.

    ``seed_everything`` setzt ``PYTHONHASHSEED`` erst zur Laufzeit; der Interpreterstart
    ist damit nicht erreicht. Nur dieser Subprozess-Test deckt den Fehler auf.
    """
    source = Path(__file__).resolve().parents[1] / "src"
    snippet = (
        "from twinlink_ml.data import HashRunSplitter as H;"
        "s = H(seed=17);"
        "print(','.join(s.assign(f'mixed-abc-{i:05d}').value for i in range(40)))"
    )
    outputs = []
    for hash_seed in ("0", "1", "random"):
        environment = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(source)}
        completed = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
        )
        outputs.append(completed.stdout.strip())
    assert len(set(outputs)) == 1


# -------------------------------------------------------------------------- Splits


def test_ratios_must_sum_to_one() -> None:
    with pytest.raises(SplitConfigError):
        SplitRatios(train=0.7, val=0.2, test=0.2)


def test_design_key_extracts_fingerprint() -> None:
    assert design_key("mixed-1989bc98-00007") == "mixed-1989bc98"
    assert design_key("dyssol_run_alpha") is None


def test_splitter_rejects_mixed_designs() -> None:
    """Zwei separat gebaute Designs sind nicht verschachtelt -- Fehler, keine Warnung."""
    pool = make_ids(8, "mixed-aaaaaaaa") + make_ids(8, "mixed-bbbbbbbb")
    with pytest.raises(SplitConfigError):
        IndexBlockSplitter(n_train=8, n_val=4, n_test=4).split(pool)


def test_hash_split_stable_under_pool_growth() -> None:
    splitter = HashRunSplitter(seed=11)
    small = splitter.split(make_ids(64)).membership()
    large = splitter.split(make_ids(256)).membership()
    assert all(large[key] == value for key, value in small.items())


def test_hash_split_ratios_converge() -> None:
    split = HashRunSplitter(seed=3, ratios=SplitRatios(0.70, 0.15, 0.15)).split(make_ids(4096))
    realized = realized_ratios(split)
    assert realized[SplitName.TRAIN] == pytest.approx(0.70, abs=0.02)
    assert realized[SplitName.TEST] == pytest.approx(0.15, abs=0.02)


def test_hash_split_rejects_degenerate_small_pool() -> None:
    with pytest.raises(SplitDegenerateError):
        HashRunSplitter(seed=1).split(make_ids(4))


def test_index_block_preserves_design_prefix() -> None:
    ids = make_ids(128)
    split = IndexBlockSplitter(n_train=64, n_val=32, n_test=32).split(ids)
    assert split.train == ids[:64]
    assert split.test == ids[96:128]


def test_index_block_strict_pool_size() -> None:
    with pytest.raises(SplitConfigError):
        IndexBlockSplitter(n_train=64, n_val=32, n_test=32).split(make_ids(160))


def test_subset_is_nested_and_keeps_val_test_constant() -> None:
    split = IndexBlockSplitter(n_train=128, n_val=32, n_test=32).split(make_ids(192))
    points = learning_curve(split)
    assert [point.n_train for point in points] == list(DEFAULT_LEARNING_CURVE_SIZES)
    for small, large in itertools.pairwise(points):
        assert large.split.train[: small.n_train] == small.split.train
    assert len({point.split.val for point in points}) == 1
    assert len({point.split.test for point in points}) == 1


def test_learning_curve_raises_when_pool_too_small_and_strict() -> None:
    split = IndexBlockSplitter(n_train=16, n_val=8, n_test=8).split(make_ids(32))
    assert [point.n_train for point in learning_curve(split)] == [8, 16]
    with pytest.raises(SplitConfigError):
        learning_curve(split, skip_oversized=False)


def test_runsplit_rejects_overlap_and_duplicates() -> None:
    with pytest.raises(SplitError):
        RunSplit(train=("a", "b"), val=("c",), test=("b", "d"), splitter_digest="x")
    with pytest.raises(SplitError):
        RunSplit(train=("a", "a"), val=(), test=("b",), splitter_digest="x")


def test_digest_is_order_sensitive_for_train() -> None:
    first = RunSplit(train=("a", "b"), val=(), test=("c", "d"), splitter_digest="x")
    second = RunSplit(train=("b", "a"), val=(), test=("c", "d"), splitter_digest="x")
    assert first.digest != second.digest


def test_predefined_split_rejects_unassigned_runs() -> None:
    ids = make_ids(10)
    splitter = PredefinedRunSplitter(
        label="partial", train_ids=ids[:5], val_ids=(), test_ids=ids[5:8]
    )
    with pytest.raises(SplitConfigError):
        splitter.split(ids)


# ------------------------------------------------------- Konstantläufe im Design


def test_constant_runs_are_spread_over_the_design() -> None:
    """Jedes Präfix muss ungefähr den Sollanteil an Konstantläufen haben.

    Regression gegen die alte Regel ``run < n_constant``: die legte alle Konstantläufe
    als Block an den Anfang, wodurch die kleinste Lernkurven-Stützstelle 37 % statt 10 %
    anregungsfreie Läufe enthielt und ein Präfix-Block-Split sie vollständig in ``train``
    schob.
    """
    flags = _constant_run_flags(128, 0.10)
    assert 12 <= int(flags.sum()) <= 14
    for prefix in (8, 16, 32, 64, 128):
        share = flags[:prefix].sum() / prefix
        assert 0.0 < share <= 0.2, f"Anteil {share:.3f} im Präfix {prefix}"
    assert not _constant_run_flags(16, 0.0).any()


def test_constant_run_flags_reject_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="constant_run_fraction"):
        _constant_run_flags(8, 1.5)


# ------------------------------------------------------------------ Fensterung


@pytest.fixture(scope="module")
def pipeline() -> tuple[object, object, object]:
    """Kleiner, aber echter Datensatz: Design, Rollout, Split."""
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
    seeds = SeedBundle.from_master(4711)
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
    return trajectory, split, schedule


WINDOW = WindowConfig(
    history_length=16,
    horizon=8,
    stride=4,
    target_names=("energy_target", "product_moisture_kg_per_kg"),
)


def _datasets(pipeline):
    trajectory, split, _ = pipeline
    train = TrajectoryWindowDataset.from_trajectory(trajectory, split, SplitName.TRAIN, WINDOW)
    test = TrajectoryWindowDataset.from_trajectory(
        trajectory, split, SplitName.TEST, WINDOW, scalers=train.scalers
    )
    return train, test


def test_dataset_shapes_match_window_config(pipeline) -> None:
    train, _ = _datasets(pipeline)
    batch = train.batch(range(min(32, train.n_windows)))
    n_exogenous = len(train.exogenous_names)
    n_targets = len(train.target_names)
    assert batch.history_inputs.shape[1:] == (16, n_exogenous + n_targets)
    assert batch.future_exogenous.shape[1:] == (8, n_exogenous)
    assert batch.future_targets.shape[1:] == (8, n_targets)


def test_dataset_only_contains_runs_of_its_split(pipeline) -> None:
    _, split, _ = pipeline
    train, test = _datasets(pipeline)
    assert set(train.run_ids) == set(split.train)
    assert set(test.run_ids) == set(split.test)
    assert not set(train.run_ids) & set(test.run_ids)


def test_dataset_preserves_split_order(pipeline) -> None:
    """Sonst stimmt ``subset(n)`` nicht mehr mit der Design-Reihenfolge überein."""
    _, split, _ = pipeline
    train, _ = _datasets(pipeline)
    assert train.run_ids == split.train


def test_windows_never_cross_run_boundaries(pipeline) -> None:
    train, _ = _datasets(pipeline)
    trajectory, _, _ = pipeline
    batch = train.full_batch()
    span = WINDOW.history_length + WINDOW.horizon
    assert np.all(batch.window_start + span <= trajectory.n_steps)


def test_test_scaler_comes_from_training_split(pipeline) -> None:
    """Der häufigste stille Leak nach dem Split selbst."""
    train, test = _datasets(pipeline)
    assert np.allclose(train.target_scaler.mean, test.target_scaler.mean)
    trajectory, split, _ = pipeline
    own = TrajectoryWindowDataset.from_trajectory(trajectory, split, SplitName.TEST, WINDOW)
    assert not np.allclose(train.target_scaler.mean, own.target_scaler.mean)


def test_standardizer_roundtrip() -> None:
    values = np.random.default_rng(0).normal(size=(50, 3)) * [2.0, 100.0, 0.01]
    scaler = Standardizer.fit(values)
    assert np.allclose(scaler.inverse(scaler.transform(values)), values)


def test_standardizer_tolerates_constant_channel() -> None:
    values = np.concatenate([np.ones((20, 1)), np.arange(20.0)[:, None]], axis=1)
    scaler = Standardizer.fit(values)
    assert np.all(np.isfinite(scaler.transform(values)))


def test_window_config_rejects_horizon_longer_than_run(pipeline) -> None:
    trajectory, split, _ = pipeline
    with pytest.raises(ValueError, match="Fenster"):
        TrajectoryWindowDataset.from_trajectory(
            trajectory,
            split,
            SplitName.TRAIN,
            dataclasses.replace(WINDOW, horizon=trajectory.n_steps),
        )


# ------------------------------------------------------------------- Baselines


def test_persistence_holds_last_observation(pipeline) -> None:
    train, _ = _datasets(pipeline)
    model = PersistenceModel()
    model.fit(train, seed=0)
    batch = train.batch(range(min(16, train.n_windows)))
    prediction = model.predict(batch)
    last = batch.history_inputs[:, -1, list(train.autoregressive_index)]
    assert np.allclose(prediction, last[:, None, :])
    assert np.allclose(prediction[:, 0, :], prediction[:, -1, :])


def test_static_regression_is_horizon_independent(pipeline) -> None:
    """Kein Zustandsgedächtnis, also darf der Fehler nicht mit dem Horizont wachsen."""
    train, test = _datasets(pipeline)
    model = StaticRegressionModel()
    model.fit(train, seed=0)
    result = RolloutEvaluator(horizons=(1, 8)).evaluate(model, test)
    for target in result.target_names:
        first = result.nrmse_by_horizon[target][1]
        last = result.nrmse_by_horizon[target][8]
        assert abs(last - first) < 0.3 * max(first, 1e-9)


def test_arx_rollout_uses_own_predictions(pipeline) -> None:
    """Ein Rollout, der heimlich die Wahrheit fortschreibt, wäre nicht erkennbar klein.

    Prüfung: werden die zukünftigen Sollwerte verändert, darf sich die Vorhersage nicht
    ändern -- das Modell darf sie nicht gesehen haben.
    """
    train, test = _datasets(pipeline)
    model = LinearARXModel()
    model.fit(train, seed=0)
    batch = test.batch(range(min(64, test.n_windows)))
    reference = model.predict(batch)
    tampered = dataclasses.replace(batch, future_targets=batch.future_targets * 3.0 + 7.0)
    assert np.allclose(reference, model.predict(tampered))


def test_arx_needs_sufficient_history(pipeline) -> None:
    trajectory, split, _ = pipeline
    short = TrajectoryWindowDataset.from_trajectory(
        trajectory, split, SplitName.TRAIN, dataclasses.replace(WINDOW, history_length=2)
    )
    with pytest.raises(ValueError, match="ARX-Ordnung"):
        LinearARXModel(n_lags_target=4, n_lags_exogenous=6).fit(short, seed=0)


def test_arx_beats_static_regression_on_energy(pipeline) -> None:
    """Inhaltliche Zusicherung: es gibt einen dynamischen Anteil in ``energy_target``.

    Schlägt der ARX die statische Regression **nicht**, ist die Zielgröße vollständig
    statisch und der ganze Architekturvergleich auf ``dE`` gegenstandslos. Der Test ist
    damit ein Wächter über die Interpretierbarkeit von AP 1, nicht über den Code.
    """
    train, test = _datasets(pipeline)
    evaluator = RolloutEvaluator(horizons=(1,))
    scores = {}
    for model in (StaticRegressionModel(), LinearARXModel()):
        model.fit(train, seed=0)
        result = evaluator.evaluate(model, test)
        scores[model.name] = result.nrmse_run_spread["energy_target"][1][0]
    assert scores["arx_p4_q2"] < scores["static_quadratic"]


def test_models_are_deterministic(pipeline) -> None:
    train, test = _datasets(pipeline)
    batch = test.batch(range(min(32, test.n_windows)))
    for factory in (PersistenceModel, StaticRegressionModel, LinearARXModel):
        first, second = factory(), factory()
        first.fit(train, seed=1)
        second.fit(train, seed=1)
        assert np.allclose(first.predict(batch), second.predict(batch))


# ------------------------------------------------------------------ Evaluator


def test_nrmse_is_zero_for_perfect_prediction() -> None:
    reference = np.random.default_rng(1).normal(size=(20, 4, 2))
    assert np.allclose(nrmse(reference, reference, axis=(0, 1)), 0.0)


def test_nrmse_is_one_for_mean_prediction() -> None:
    reference = np.random.default_rng(2).normal(size=(400, 1, 1))
    prediction = np.full_like(reference, reference.mean())
    assert nrmse(prediction, reference, axis=(0, 1)) == pytest.approx(1.0, abs=1e-3)


def test_evaluator_rejects_horizon_outside_window(pipeline) -> None:
    _, test = _datasets(pipeline)
    model = PersistenceModel()
    model.fit(test, seed=0)
    with pytest.raises(ValueError, match="außerhalb"):
        RolloutEvaluator(horizons=(99,)).evaluate(model, test)


def test_evaluation_result_is_flattenable(pipeline) -> None:
    train, test = _datasets(pipeline)
    model = PersistenceModel()
    model.fit(train, seed=0)
    flat = RolloutEvaluator(horizons=(1, 8)).evaluate(model, test).as_flat_metrics()
    assert "nrmse/energy_target/h8" in flat
    assert all(isinstance(value, float) for value in flat.values())


def test_rmse_run_median_physical_is_present_and_positive(pipeline) -> None:
    train, test = _datasets(pipeline)
    model = LinearARXModel()
    model.fit(train, seed=0)
    result = RolloutEvaluator(horizons=(1, 8)).evaluate(model, test)
    for target in result.target_names:
        assert result.rmse_run_median_physical[target][1] > 0.0
        assert result.rmse_run_median_physical[target][8] > 0.0
    flat = result.as_flat_metrics()
    assert "rmse_run_median_physical/energy_target/h8" in flat


def test_existing_nrmse_keys_unchanged_by_rmse_addition(pipeline) -> None:
    """Regressionstest (G3).

    ``rmse_run_median_physical`` darf die bestehenden NRMSE-Werte nicht verändern -- weder
    unter ``physical_units=True`` noch ``=False``. Referenz ist dieselbe Rechnung,
    unabhängig über die öffentliche API nachvollzogen, ohne den Umweg über die neuen
    ``*_physical``-Zwischengrößen in ``RolloutEvaluator.evaluate``.
    """
    train, test = _datasets(pipeline)
    model = LinearARXModel()
    model.fit(train, seed=0)
    for physical_units in (True, False):
        evaluator = RolloutEvaluator(horizons=(1, 8), physical_units=physical_units)
        result = evaluator.evaluate(model, test)

        predictions = np.concatenate(
            [model.predict(batch) for batch in test.iter_batches(evaluator.batch_size)], axis=0
        )
        references = np.concatenate(
            [batch.future_targets for batch in test.iter_batches(evaluator.batch_size)], axis=0
        )
        runs = np.concatenate(
            [batch.run_index for batch in test.iter_batches(evaluator.batch_size)], axis=0
        )
        if physical_units:
            predictions = test.target_scaler.inverse(predictions)
            references = test.target_scaler.inverse(references)
        expected_pooled = pooled_nrmse(predictions, references, (1, 8), result.target_names)
        expected_spread = per_run_nrmse(predictions, references, runs, (1, 8), result.target_names)
        for target in result.target_names:
            for h in (1, 8):
                pooled = result.nrmse_by_horizon[target][h]
                spread = result.nrmse_run_spread[target][h]
                assert pooled == pytest.approx(expected_pooled[target][h])
                assert spread == pytest.approx(expected_spread[target][h])


def test_pooled_and_per_run_normalisation_differ_for_quality_targets(pipeline) -> None:
    """Der Befund, der die Primärmetrik festlegt.

    Der gepoolte Nenner enthält die Niveauunterschiede zwischen den Läufen, die durch die
    gestreuten Anfangszustände entstehen und von keinem Modell vorherzusagen sind. Bei den
    Qualitätsgrößen ist er dadurch deutlich zu groß.
    """
    train, test = _datasets(pipeline)
    model = LinearARXModel()
    model.fit(train, seed=0)
    result = RolloutEvaluator(horizons=(8,)).evaluate(model, test)
    pooled = result.nrmse_by_horizon["product_moisture_kg_per_kg"][8]
    per_run = result.nrmse_run_spread["product_moisture_kg_per_kg"][8][0]
    assert per_run > pooled
    energy_ratio = (
        result.nrmse_run_spread["energy_target"][8][0] / result.nrmse_by_horizon["energy_target"][8]
    )
    assert per_run / pooled > energy_ratio


def test_split_report_and_strata(pipeline) -> None:
    _, split, schedule = pipeline
    strata = signal_kind_strata(schedule, channel=0)
    coverage = stratum_coverage(split, strata)
    assert sum(coverage[SplitName.TRAIN].values()) == len(split.train)
    assert split.digest in split_report(split, strata)

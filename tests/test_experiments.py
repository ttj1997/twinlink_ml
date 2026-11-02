"""Tests for the experiment core, tracking and GRU (work plan steps 3 to 5).

The GRU is checkable only with PyTorch; the corresponding tests skip themselves without
Torch. Without Torch it is nevertheless checked that the package stays importable and that
the error message is usable -- both are a guarantee to steps 1 to 4, which do not need
PyTorch. The skipped tests must run green once on a machine with Torch
**before** a GRU result is reported.
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.evaluation import (
    DataSpec,
    RolloutEvaluator,
    ValidationPolicy,
    build_datasets,
    generate_data,
    run_learning_curve,
    run_sample_time_study,
    summarise_curve,
)
from twinlink_ml.evaluation.experiments import evaluate_point
from twinlink_ml.models import LinearARXModel, PersistenceModel, StaticRegressionModel
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils import (
    JsonlExperimentLogger,
    MultiExperimentLogger,
    RunRecord,
    environment_fingerprint,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_placeholder.yaml"
HAS_TORCH = importlib.util.find_spec("torch") is not None

SMALL = DataSpec(
    sample_time_s=25.0,
    run_duration_s=4000.0,
    n_train_max=8,
    n_val=4,
    n_test=4,
    history_length=16,
    horizon=8,
    stride=8,
    target_names=("energy_target", "product_moisture_kg_per_kg"),
)


@pytest.fixture(scope="module")
def process_config():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return load_process_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def data(process_config):
    return generate_data(process_config, SMALL, master_seed=1)


# ------------------------------------------------------------------------- Data set


def test_pool_size_matches_specification(data) -> None:
    assert data.trajectory.n_runs == SMALL.pool_size
    assert len(data.split.train) == SMALL.n_train_max


def test_generation_is_reproducible(process_config, data) -> None:
    again = generate_data(process_config, SMALL, master_seed=1)
    assert again.fingerprints == data.fingerprints
    assert np.allclose(again.trajectory.outputs, data.trajectory.outputs)


def test_generation_depends_on_master_seed(process_config, data) -> None:
    other = generate_data(process_config, SMALL, master_seed=2)
    assert other.fingerprints["split/digest"] != data.fingerprints["split/digest"]


def test_fingerprints_are_complete(data) -> None:
    """Without these four a learning-curve point cannot be reconstructed.

    ``measurement/digest`` is included from step 5 on: with the noise level as an
    experimental factor, two data sets differ despite the same process fingerprint.
    """
    assert set(data.fingerprints) == {
        "process/fingerprint",
        "excitation/fingerprint",
        "split/digest",
        "measurement/digest",
    }
    assert all(value not in ("", "unknown") for value in data.fingerprints.values())


# ------------------------------------------------------------------ Support points


def test_datasets_share_training_scalers(data) -> None:
    train, validation, test = build_datasets(data, 8)
    assert validation is not None
    for other in (validation, test):
        assert np.allclose(train.target_scaler.mean, other.target_scaler.mean)
        assert np.allclose(train.input_scaler.scale, other.input_scaler.scale)


def test_support_points_are_nested_and_share_test_set(data) -> None:
    small_train, _, small_test = build_datasets(data, 4)
    large_train, _, large_test = build_datasets(data, 8)
    assert large_train.run_ids[:4] == small_train.run_ids
    assert small_test.run_ids == large_test.run_ids


def test_scalers_are_refitted_per_support_point(data) -> None:
    """Estimating statistics once over the pool lifts exactly the left end of the curve."""
    small, _, _ = build_datasets(data, 4)
    large, _, _ = build_datasets(data, 8)
    assert not np.allclose(small.target_scaler.mean, large.target_scaler.mean)


def test_validation_policy_none_omits_validation_set(data) -> None:
    _, absent, _ = build_datasets(data, 8, policy=ValidationPolicy.NONE)
    assert absent is None
    _, present, _ = build_datasets(data, 8, policy=ValidationPolicy.FIXED_POOL)
    assert present is not None
    assert set(present.run_ids) == set(data.split.val)


def test_splits_stay_disjoint_over_all_support_points(data) -> None:
    for n_train in (4, 8):
        train, validation, test = build_datasets(data, n_train)
        assert validation is not None
        sets = [set(train.run_ids), set(validation.run_ids), set(test.run_ids)]
        assert len(sets[0] | sets[1] | sets[2]) == sum(len(item) for item in sets)


# ----------------------------------------------------------------- Learning curve


def test_learning_curve_record_count(process_config) -> None:
    records = run_learning_curve(
        process_config,
        {"persistence": PersistenceModel, "arx": LinearARXModel},
        spec=SMALL,
        sizes=(4, 8),
        master_seeds=(1, 2),
        evaluator=RolloutEvaluator(horizons=(1, 8)),
    )
    assert len(records) == 2 * 2 * 2


def test_persistence_is_invariant_to_training_size(process_config) -> None:
    """Guard over the evaluation: a parameter-free model must not learn.

    If the persistence error changes with ``n_train``, the test evaluation depends on the
    training set -- i.e. a leak or an error in passing on the scalers.
    """
    records = run_learning_curve(
        process_config,
        {"persistence": PersistenceModel},
        spec=SMALL,
        sizes=(4, 8),
        master_seeds=(1,),
        evaluator=RolloutEvaluator(horizons=(1,)),
    )
    values = {
        int(record.params["n_train"]): record.metrics["nrmse/energy_target/h1"]
        for record in records
    }
    assert values[4] == pytest.approx(values[8], abs=1e-9)


def test_budget_is_logged(process_config) -> None:
    records = run_learning_curve(
        process_config,
        {"persistence": PersistenceModel},
        spec=SMALL,
        sizes=(4,),
        master_seeds=(1,),
        evaluator=RolloutEvaluator(horizons=(1,)),
    )
    record = records[0]
    assert record.params["n_budget"] == record.params["n_train"] + record.params["n_val"]
    assert record.params["validation_policy"] == "fixed_pool"


def test_summarise_curve_reports_median_and_range(process_config) -> None:
    records = run_learning_curve(
        process_config,
        {"arx": LinearARXModel},
        spec=SMALL,
        sizes=(4, 8),
        master_seeds=(1, 2, 3),
        evaluator=RolloutEvaluator(horizons=(1,)),
    )
    curve = summarise_curve(records, target="energy_target", horizon=1)
    assert set(curve) == {"arx"}
    for n_train, (median, low, high) in curve["arx"].items():
        assert n_train in (4, 8)
        assert low <= median <= high


def test_learning_curve_writes_one_log_line_per_record(process_config, tmp_path: Path) -> None:
    path = tmp_path / "curve.jsonl"
    records = run_learning_curve(
        process_config,
        {"arx": LinearARXModel},
        spec=SMALL,
        sizes=(4, 8),
        master_seeds=(1,),
        evaluator=RolloutEvaluator(horizons=(1,)),
        logger=JsonlExperimentLogger(path),
    )
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(records)
    payload = json.loads(lines[0])
    assert payload["params"]["split/digest"]
    assert payload["params"]["git/commit"]


def test_record_schema_includes_training_elapsed_s(data) -> None:
    """G4c: cost questions (A1) answerable -- every record carries training/elapsed_s."""
    records = evaluate_point(data, 4, {"persistence": PersistenceModel, "arx": LinearARXModel})
    assert len(records) == 2
    for record in records:
        elapsed = record.params["training/elapsed_s"]
        assert isinstance(elapsed, float)
        assert elapsed >= 0.0


# ------------------------------------------------------------------ Δt pre-study


def test_sample_time_study_keeps_physical_window_constant(process_config) -> None:
    """Otherwise the comparison mixes sample time and model horizon."""
    records = run_sample_time_study(
        process_config,
        {"persistence": PersistenceModel},
        sample_times_s=(25.0, 50.0),
        base_spec=SMALL,
        master_seed=1,
    )
    spans = {
        float(record.params["data/sample_time_s"]): (
            int(record.params["data/history_length"]) * float(record.params["data/sample_time_s"]),
            int(record.params["data/horizon"]) * float(record.params["data/sample_time_s"]),
        )
        for record in records
    }
    assert spans[25.0] == spans[50.0]
    assert spans[25.0] == (SMALL.history_length * 25.0, SMALL.horizon * 25.0)


# -------------------------------------------------------------------- Tracking


def test_record_serialises_non_json_values() -> None:
    record = RunRecord(
        group="test",
        params={"a": 1, "b": (1, 2), "c": np.float64(3.0), "d": Path("/tmp")},
        metrics={"nrmse": np.float64(0.5)},
        tags={"model": "x"},
    )
    assert "nrmse" in json.dumps(record.to_json())


def test_jsonl_logger_appends_and_is_parsable(tmp_path: Path) -> None:
    """Appending, so that an abort after 40 of 45 points keeps the 40.

    ``allow_missing_git=True``: this test checks the append mechanics, not the
    git provenance (G4a) -- the records deliberately carry no ``git/commit_available``.
    """
    path = tmp_path / "nested" / "log.jsonl"
    logger = JsonlExperimentLogger(path, allow_missing_git=True)
    with pytest.warns(UserWarning, match="ohne Git-Commit"):
        for index in range(3):
            logger.log(RunRecord(group="g", params={"i": index}, metrics={"m": float(index)}))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line)["params"]["i"] for line in lines] == [0, 1, 2]


def test_multi_logger_forwards(tmp_path: Path) -> None:
    first = JsonlExperimentLogger(tmp_path / "a.jsonl", allow_missing_git=True)
    second = JsonlExperimentLogger(tmp_path / "b.jsonl", allow_missing_git=True)
    with pytest.warns(UserWarning, match="ohne Git-Commit"):
        MultiExperimentLogger((first, second)).log(
            RunRecord(group="g", params={}, metrics={"m": 1.0})
        )
    assert (tmp_path / "a.jsonl").exists() and (tmp_path / "b.jsonl").exists()


def test_environment_fingerprint_has_versions() -> None:
    fingerprint = environment_fingerprint()
    assert "git/commit" in fingerprint
    assert "version/numpy" in fingerprint


# ------------------------------------------------------------------------- GRU


def test_package_imports_without_torch() -> None:
    """Steps 1 to 4 must not need PyTorch."""
    import twinlink_ml.data
    import twinlink_ml.evaluation
    import twinlink_ml.models
    import twinlink_ml.simulation

    assert "TorchSequenceModel" not in twinlink_ml.models.__all__


@pytest.mark.skipif(HAS_TORCH, reason="prüft nur den Fehlerpfad ohne Torch")
def test_gru_raises_actionable_error_without_torch() -> None:
    from twinlink_ml.models.torch_sequence import TorchSequenceModel

    with pytest.raises(ImportError, match="twinlink-ml\\[torch\\]"):
        TorchSequenceModel()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_gru_matches_persistence_when_untrained(data) -> None:
    """Residual parametrisation with a zero head must be exactly persistence.

    This justifies the choice of parametrisation: an untrained network starts on the
    baseline instead of on a random mapping and, at the small support points, spends
    no data on first finding the operating point.
    """
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, _, test = build_datasets(data, 8)
    gru = TorchSequenceModel(TorchSequenceConfig(max_epochs=0, patience=0, hidden_size=8))
    gru.fit(train, seed=0)
    baseline = PersistenceModel()
    baseline.fit(train, seed=0)
    batch = test.batch(range(min(32, test.n_windows)))
    assert np.allclose(gru.predict(batch), baseline.predict(batch), atol=1e-5)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_gru_trains_and_moves_away_from_the_zero_init(data) -> None:
    """The trainer works -- **not** that the GRU beats persistence.

    Deliberately no generalisation claim. With ``n_train = 8`` and a single seed
    every comparison lies far below the resolution threshold from step 5 (13.2 % with three
    seeds; with one seed even less is detectable) -- an assert of the form
    "GRU beats persistence" would here disguise an empirical claim as a sanity check,
    one that the project itself has classified as not reliable. Exactly for that,
    ``study_architectures.py`` exists instead, with several seeds and
    ``architecture_verdicts``, which supplies the threshold.

    What is checked here instead -- deterministically, without training luck:

    1. The training loss at the end is lower than at the start (the optimiser actually
       moves towards its own objective).
    2. The weights deviate from the zero init after training (the network has moved,
       not just confirmed the same point `max_epochs` times).
    3. The rollout prediction stays finite and is no longer the persistence baseline
       (the training has changed something, without claiming a direction).

    Whether a GRU beats persistence on `energy_target` with a realistic seed budget
    is a question for the learning curve, not for this test.
    """
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, validation, test = build_datasets(data, 8)
    evaluator = RolloutEvaluator(horizons=(8,))
    gru = TorchSequenceModel(TorchSequenceConfig(hidden_size=16, max_epochs=40, patience=15))
    gru.fit(train, seed=0, validation=validation)
    baseline = PersistenceModel()
    baseline.fit(train, seed=0)

    assert gru.n_parameters > 0
    assert len(gru.training_history) > 1

    losses = [record["train_loss"] for record in gru.training_history]
    assert losses[-1] < losses[0], "AdamW hat den Trainingsverlust nicht reduziert"

    head_weight = gru._network.head.weight.detach().cpu().numpy()
    assert not np.allclose(head_weight, 0.0), "Kopfgewichte sind nach dem Training noch Null"

    trained = evaluator.evaluate(gru, test).nrmse_run_spread["energy_target"][8][0]
    naive = evaluator.evaluate(baseline, test).nrmse_run_spread["energy_target"][8][0]
    assert np.isfinite(trained)
    assert trained != pytest.approx(naive), "Training hat die Vorhersage nicht verändert"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_gru_rollout_does_not_see_future_targets(data) -> None:
    """The same guarantee as for the ARX: no covert carrying forward of the truth."""
    import dataclasses

    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, _, test = build_datasets(data, 8)
    gru = TorchSequenceModel(TorchSequenceConfig(hidden_size=8, max_epochs=3, patience=0))
    gru.fit(train, seed=0)
    batch = test.batch(range(min(32, test.n_windows)))
    tampered = dataclasses.replace(batch, future_targets=batch.future_targets * 3.0 + 7.0)
    assert np.allclose(gru.predict(batch), gru.predict(tampered))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_gru_prediction_is_deterministic(data) -> None:
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, _, test = build_datasets(data, 8)
    batch = test.batch(range(min(16, test.n_windows)))
    config = TorchSequenceConfig(hidden_size=8, max_epochs=3, patience=0)
    first, second = TorchSequenceModel(config), TorchSequenceModel(config)
    first.fit(train, seed=7)
    second.fit(train, seed=7)
    assert np.allclose(first.predict(batch), second.predict(batch), atol=1e-6)


def test_static_regression_stays_competitive_on_energy(data) -> None:
    """Content guard: ``energy_target`` is algebraic in the current inputs.

    If the static regression deteriorates drastically, something is wrong with the
    target definition or the excitation design.
    """
    train, _, test = build_datasets(data, 8)
    model = StaticRegressionModel()
    model.fit(train, seed=0)
    result = RolloutEvaluator(horizons=(1,)).evaluate(model, test)
    assert result.nrmse_run_spread["energy_target"][1][0] < 0.5


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_lstm_core_shares_the_trainer(data) -> None:
    """The core is interchangeable: the same training, the same rollout semantics.

    If the test were red, the open-closed promise of step 6 would be violated -- a new
    architecture would then need a change to the training code after all. ``xlstm`` included
    since step 6, part 2 (Befund 29, ``docs/04_stand_ap1.md``) -- the same trainer,
    without touching it (module docstring ``torch_sequence.py``).
    """
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, _, test = build_datasets(data, 8)
    evaluator = RolloutEvaluator(horizons=(1, SMALL.horizon))
    for core in ("gru", "lstm", "xlstm"):
        model = TorchSequenceModel(
            TorchSequenceConfig(core=core, hidden_size=8, max_epochs=2, patience=0)
        )
        model.fit(train, seed=1, validation=None)
        assert model.n_parameters > 0
        result = evaluator.evaluate(model, test)
        assert np.isfinite(result.nrmse_run_spread["energy_target"][SMALL.horizon][0])


# --------------------------------------------------------------------------- xLSTM
#
# The same sanity checks as above for GRU (preceding lines), here on
# ``TorchSequenceModel`` (single-network level, not composite) for ``core="xlstm"``. The
# composite-specific guarantees (encapsulation of the three sub-networks, long-term
# rollout stability) are in ``tests/test_composite.py`` -- where ``xlstm`` is actually
# used via ``registry.py``.


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_xlstm_matches_persistence_when_untrained(data) -> None:
    """The same residual invariant as for GRU, for the mLSTM core."""
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, _, test = build_datasets(data, 8)
    xlstm = TorchSequenceModel(
        TorchSequenceConfig(core="xlstm", max_epochs=0, patience=0, hidden_size=8)
    )
    xlstm.fit(train, seed=0)
    baseline = PersistenceModel()
    baseline.fit(train, seed=0)
    batch = test.batch(range(min(32, test.n_windows)))
    assert np.allclose(xlstm.predict(batch), baseline.predict(batch), atol=1e-5)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_xlstm_rollout_does_not_see_future_targets(data) -> None:
    """The same guarantee as for GRU: no covert carrying forward of the truth."""
    import dataclasses

    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, _, test = build_datasets(data, 8)
    xlstm = TorchSequenceModel(
        TorchSequenceConfig(core="xlstm", hidden_size=8, max_epochs=3, patience=0)
    )
    xlstm.fit(train, seed=0)
    batch = test.batch(range(min(32, test.n_windows)))
    tampered = dataclasses.replace(batch, future_targets=batch.future_targets * 3.0 + 7.0)
    assert np.allclose(xlstm.predict(batch), xlstm.predict(tampered))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_xlstm_prediction_is_deterministic_and_seed_sensitive(data) -> None:
    """Same seed -> bit-identical; different seed -> different, after training.

    Like ``test_composite_xlstm_prediction_is_reproducible_and_seed_sensitive`` in
    ``test_composite.py``, here at single-network level: with a zero-initialised head
    (``max_epochs=0``) every prediction is exactly persistence, independent of the seed --
    seed sensitivity can only be checked after training.
    """
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, _, test = build_datasets(data, 8)
    batch = test.batch(range(min(16, test.n_windows)))
    config = TorchSequenceConfig(core="xlstm", hidden_size=8, max_epochs=5, patience=0)

    first, second = TorchSequenceModel(config), TorchSequenceModel(config)
    first.fit(train, seed=7)
    second.fit(train, seed=7)
    assert np.allclose(first.predict(batch), second.predict(batch), atol=1e-6)

    third = TorchSequenceModel(config)
    third.fit(train, seed=8)
    assert not np.allclose(first.predict(batch), third.predict(batch), atol=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_untrained_network_equals_persistence(data) -> None:
    """Zero-initialised head = persistence, exactly.

    The requirement of the work plan for
    all architectures; without Torch it is checked only on the NARX (tests/test_step6.py).
    """
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, _, test = build_datasets(data, 8)
    model = TorchSequenceModel(TorchSequenceConfig(hidden_size=8, max_epochs=0, patience=0))
    model.fit(train, seed=1, validation=None)
    reference = PersistenceModel()
    reference.fit(train, seed=1)
    batch = test.full_batch()
    assert np.allclose(model.predict(batch), reference.predict(batch), atol=1e-5)

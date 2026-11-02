"""Tests for step 6: observability, nonlinear NARX, slots, verdicts.

The most important guarantee of this module is a leak check: in soft-sensor operation
the unmeasured target must **nowhere** appear in the window history. A leak there
looks like an excellent soft sensor in all metrics and would not be reproducible at the
plant -- exactly the error that makes a comparison unusable without making it
suspicious.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.data.windows import WindowConfig
from twinlink_ml.evaluation import (
    DataSpec,
    ObservationMode,
    RolloutEvaluator,
    build_datasets,
    generate_data,
    spec_for_mode,
)
from twinlink_ml.evaluation.resolution import architecture_verdicts
from twinlink_ml.models import (
    ClimatologyModel,
    LinearARXModel,
    PersistenceModel,
    RandomFeatureNARX,
    StaticRegressionModel,
)
from twinlink_ml.models.registry import (
    MODEL_SLOTS,
    ModelKind,
    ModelSlot,
    available_labels,
    baseline_labels,
    build_factories,
    describe_slots,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.seeding import SeedBundle, VarianceSource
from twinlink_ml.utils.tracking import RunRecord

CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_placeholder.yaml"

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
SOFT = spec_for_mode(SMALL, ObservationMode.SOFT_SENSOR)


@pytest.fixture(scope="module")
def process_config():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return load_process_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def inline_data(process_config):
    return generate_data(process_config, SMALL, seeds=SeedBundle.from_master(1))


@pytest.fixture(scope="module")
def soft_data(process_config):
    return generate_data(process_config, SOFT, seeds=SeedBundle.from_master(1))


# ------------------------------------------------------------------ Configuration


def test_observed_target_selection() -> None:
    config = WindowConfig(
        history_length=4,
        horizon=2,
        target_names=("a", "b", "c"),
        observed_target_names=("c", "a"),
    )
    # Order follows target_names, not the specification -- otherwise column indices
    # would depend on how the call is written.
    assert config.observed_targets == ("a", "c")
    assert config.unobserved_targets == ("b",)
    assert not config.is_fully_observed


def test_unknown_observed_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        WindowConfig(
            history_length=4,
            horizon=2,
            target_names=("a",),
            observed_target_names=("b",),
        )


def test_spec_for_mode_keeps_energy_observed() -> None:
    assert spec_for_mode(SMALL, ObservationMode.INLINE).observed_target_names is None
    assert spec_for_mode(SMALL, ObservationMode.SOFT_SENSOR).observed_target_names == (
        "energy_target",
    )


# ------------------------------------------------------------------- Leak check


def test_unobserved_channel_carries_no_information(soft_data) -> None:
    """The core guarantee: the unmeasured channel is constantly zero in the history.

    It matches the measured value at no point.
    """
    train, _, _ = build_datasets(soft_data, 8)
    batch = train.full_batch()
    offset = len(train.exogenous_names)
    column = offset + train.target_names.index("product_moisture_kg_per_kg")
    assert np.all(batch.history_inputs[..., column] == 0.0)
    # The observed channel, by contrast, is not constant.
    energy_column = offset + train.target_names.index("energy_target")
    assert np.std(batch.history_inputs[..., energy_column]) > 0.0


def test_target_itself_is_untouched_in_soft_sensor_mode(soft_data) -> None:
    """The history is masked, not the target -- otherwise there would be nothing to learn."""
    train, _, _ = build_datasets(soft_data, 8)
    batch = train.full_batch()
    column = train.target_names.index("product_moisture_kg_per_kg")
    assert np.std(batch.future_targets[..., column]) > 0.0


def test_scalers_are_identical_across_modes(inline_data, soft_data) -> None:
    """The masking engages after the standardisation.

    If the scalers were different,
    the two operating modes would additionally differ in the normalisation of the inputs
    and the curves would not be comparable.
    """
    inline_train, _, _ = build_datasets(inline_data, 8)
    soft_train, _, _ = build_datasets(soft_data, 8)
    assert inline_train.target_scaler.digest == soft_train.target_scaler.digest
    assert np.allclose(inline_train.input_scaler.mean, soft_train.input_scaler.mean)


# ----------------------------------------------------------------- Baselines


def test_persistence_refuses_soft_sensor_mode(soft_data) -> None:
    train, _, _ = build_datasets(soft_data, 8)
    with pytest.raises(ValueError, match="Climatology"):
        PersistenceModel().fit(train, seed=0)


def test_climatology_is_the_parameter_free_reference(soft_data) -> None:
    train, _, test = build_datasets(soft_data, 8)
    model = ClimatologyModel()
    model.fit(train, seed=0)
    prediction = model.predict(test.full_batch())
    # Constant prediction: identical over windows and horizon.
    assert np.allclose(prediction, prediction[0:1, 0:1, :])
    # And independent of the model seed -- the precondition for its scatter
    # being pure data-set realisation.
    other = ClimatologyModel()
    other.fit(train, seed=999)
    assert np.allclose(other.predict(test.full_batch()), prediction)


def test_arx_and_static_still_work_without_the_measurement(soft_data) -> None:
    train, _, test = build_datasets(soft_data, 8)
    evaluator = RolloutEvaluator(horizons=(1, SOFT.horizon))
    for model in (StaticRegressionModel(), LinearARXModel()):
        model.fit(train, seed=0)
        result = evaluator.evaluate(model, test)
        value = result.nrmse_run_spread["product_moisture_kg_per_kg"][SOFT.horizon][0]
        assert np.isfinite(value)


# ---------------------------------------------------------------------- NARX


def test_narx_with_zero_head_is_exactly_persistence(inline_data) -> None:
    """Residual parametrisation: an output layer of zeros **is** persistence.

    The same guarantee that the work plan demands for every architecture in step 6 --
    here checkable without PyTorch, and thus the safeguard of the principle itself and not
    just of an implementation.
    """
    train, _, test = build_datasets(inline_data, 8)
    narx = RandomFeatureNARX(n_features=32)
    narx.fit(train, seed=7)
    narx._coefficients = np.zeros_like(narx._coefficients)
    persistence = PersistenceModel()
    persistence.fit(train, seed=7)
    batch = test.full_batch()
    assert np.array_equal(narx.predict(batch), persistence.predict(batch))


def test_narx_at_large_ridge_is_persistence_plus_a_constant_drift(inline_data) -> None:
    """With overwhelming regularisation, what remains is **not** persistence, but.

    persistence plus a constant.

    The reason lies in :func:`~twinlink_ml.models.baselines._ridge_solve`: the
    intercept is deliberately not penalised. The limiting case is thus the mean
    one-step increment of the training split -- a drift-corrected persistence. That is
    intended, but it is **not** the same as the persistence baseline, and an
    architecture comparison that confuses the two would have a slightly different lower bound.
    """
    train, _, test = build_datasets(inline_data, 8)
    narx = RandomFeatureNARX(n_features=32, ridge=1e12)
    narx.fit(train, seed=7)
    persistence = PersistenceModel()
    persistence.fit(train, seed=7)
    batch = test.full_batch()
    difference = narx.predict(batch)[:, 0, :] - persistence.predict(batch)[:, 0, :]
    assert np.allclose(difference, difference[0:1, :], atol=1e-8)
    assert np.abs(difference).max() > 0.0


def test_narx_depends_on_the_model_seed(inline_data) -> None:
    """The first model with a real model scatter.

    Without it, the axis ``model``
    of the variance decomposition was not checked in any case.
    """
    train, _, test = build_datasets(inline_data, 8)
    batch = test.full_batch()
    predictions = []
    for seed in (1, 2):
        model = RandomFeatureNARX(n_features=64)
        model.fit(train, seed=seed)
        predictions.append(model.predict(batch))
    assert not np.allclose(predictions[0], predictions[1])


def test_narx_is_reproducible_for_a_given_seed(inline_data) -> None:
    train, _, test = build_datasets(inline_data, 8)
    batch = test.full_batch()
    first, second = (RandomFeatureNARX(n_features=64) for _ in range(2))
    first.fit(train, seed=5)
    second.fit(train, seed=5)
    assert np.allclose(first.predict(batch), second.predict(batch))


def test_narx_does_not_peek_at_future_targets(inline_data) -> None:
    """The same check as for the baselines: no look at ``future_targets``.

    Manipulating ``future_targets`` must not change the prediction, otherwise
    the model carries forward the truth in the rollout.
    """
    train, _, test = build_datasets(inline_data, 8)
    model = RandomFeatureNARX(n_features=64)
    model.fit(train, seed=3)
    batch = test.full_batch()
    reference = model.predict(batch)
    tampered = type(batch)(
        history_inputs=batch.history_inputs,
        future_exogenous=batch.future_exogenous,
        future_targets=batch.future_targets * 3.0 + 1.0,
        run_index=batch.run_index,
        window_start=batch.window_start,
        future_targets_truth=batch.future_targets_truth,
    )
    assert np.allclose(model.predict(tampered), reference)


def test_narx_rejects_degenerate_configuration() -> None:
    with pytest.raises(ValueError):
        RandomFeatureNARX(n_features=0)
    with pytest.raises(ValueError):
        RandomFeatureNARX(length_scale=0.0)


# ------------------------------------------------------------------- Registration


def test_registry_covers_the_plan() -> None:
    labels = {slot.label for slot in MODEL_SLOTS}
    assert {"gru", "lstm", "xlstm", "mamba", "transformer"} <= labels
    assert {"persistence", "climatology", "static", "arx", "narx"} <= labels


def test_registry_hides_persistence_in_soft_sensor_mode() -> None:
    assert "persistence" in available_labels()
    assert "persistence" not in available_labels(fully_observed=False)
    assert "climatology" in available_labels(fully_observed=False)


def test_registry_rejects_unknown_label() -> None:
    with pytest.raises(KeyError):
        build_factories(("gibtsnicht",))


def test_registry_availability_tracks_dependencies_not_the_lazy_factory() -> None:
    """Regression test: availability must check ``implemented``, not ``factory is not None``.

    gru/lstm are deliberately registered with ``factory=None``, so that the Torch factory
    is only resolved when needed. Without this test, this would only have been noticed again in
    `study_architectures.py --mode both` (without an explicit `--models` list),
    not in the test suite.

    All four architecture slots (gru/lstm/xlstm/transformer/mamba) are by now
    implemented -- no real slot is available any more as a stand-in for "declared, but
    not implemented". A synthetic slot with ``implemented=False``
    takes on that role here, so that this regression test does not become obsolete with the next
    completed architecture.
    """
    from unittest.mock import patch

    from twinlink_ml.models import registry as registry_module

    by_label = {slot.label: slot for slot in MODEL_SLOTS}
    gru = by_label["gru"]
    fake_unimplemented = ModelSlot(
        label="_test_declared_not_implemented",
        factory=None,
        kind=ModelKind.ARCHITECTURE,
        requires=(),
        implemented=False,
    )

    with patch.object(registry_module, "find_spec", return_value=object()):
        assert gru.is_available, "gru muss verfügbar sein, sobald torch importierbar ist"
        assert not fake_unimplemented.is_available, (
            "implemented=False darf auch mit erfüllten Abhängigkeiten nie verfügbar sein"
        )
        with patch.object(registry_module, "MODEL_SLOTS", (*MODEL_SLOTS, fake_unimplemented)):
            assert "gru" in available_labels()
            assert "_test_declared_not_implemented" not in available_labels()


def test_registry_declared_but_unimplemented_slots_fail_fast() -> None:
    """As above: no real slot is ``implemented=False`` any more, a fake stands in."""
    from unittest.mock import patch

    from twinlink_ml.models import registry as registry_module

    fake_unimplemented = ModelSlot(
        label="_test_declared_not_implemented",
        factory=None,
        kind=ModelKind.ARCHITECTURE,
        requires=(),
        implemented=False,
    )
    with (
        patch.object(registry_module, "MODEL_SLOTS", (*MODEL_SLOTS, fake_unimplemented)),
        pytest.raises(NotImplementedError),
    ):
        build_factories(("_test_declared_not_implemented",))


def test_registry_factories_are_fresh_instances() -> None:
    """A new instance per seed and support point -- otherwise state survives across.

    learning-curve points and the left end of the curve is lifted.
    """
    factories = build_factories(("narx",))
    first, second = factories["narx"](), factories["narx"]()
    assert first is not second


def test_baseline_labels_exclude_architectures() -> None:
    architectures = {slot.label for slot in MODEL_SLOTS if slot.kind is ModelKind.ARCHITECTURE}
    assert not architectures & set(baseline_labels())
    assert describe_slots()


# ---------------------------------------------------------------------- Verdict


def _record(label: str, value: float, replicate: int, n_train: int = 8) -> RunRecord:
    return RunRecord(
        group="architecture_comparison",
        params={
            "model/label": label,
            "n_train": n_train,
            "variance/source": VarianceSource.JOINT.value,
            "variance/replicate": replicate,
        },
        metrics={"nrmse_run_median/product_moisture_kg_per_kg/h24": value},
        tags={"model": label},
    )


def test_verdict_refuses_to_claim_a_difference_below_the_threshold() -> None:
    records = [_record("persistence", value, k) for k, value in enumerate((0.79, 0.87, 0.80))]
    records += [_record("narx", value, k) for k, value in enumerate((0.78, 0.79, 0.77))]
    records += [_record("arx", value, k) for k, value in enumerate((1.30, 1.55, 1.20))]

    verdicts = architecture_verdicts(
        records,
        target="product_moisture_kg_per_kg",
        horizon=24,
        reference_label="persistence",
        resolution_label="persistence",
    )
    by_label = {verdict.candidate: verdict for verdict in verdicts}
    assert not by_label["narx"].is_resolved
    assert "kein nachweisbarer Unterschied" in by_label["narx"].statement
    assert by_label["arx"].is_resolved
    assert "schlechter" in by_label["arx"].statement


def test_verdict_uses_climatology_as_resolution_reference() -> None:
    records = [_record("climatology", value, k) for k, value in enumerate((1.02, 1.05, 1.01))]
    records += [_record("narx", value, k) for k, value in enumerate((0.60, 0.62, 0.59))]
    verdicts = architecture_verdicts(
        records,
        target="product_moisture_kg_per_kg",
        horizon=24,
        reference_label="climatology",
        resolution_label="climatology",
    )
    assert len(verdicts) == 1
    assert verdicts[0].is_resolved
    assert verdicts[0].improvement > 0.0

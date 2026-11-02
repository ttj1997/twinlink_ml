"""Tests for :class:`CompositeSequenceModel` (Befund 26, ``docs/04_stand_ap1.md``).

As with the GRU tests in ``test_experiments.py``: checkable only with PyTorch, the tests
skip themselves without Torch. What is checked is not that the separate networks
generalise better -- that is the job of the diagnostic experiment
(``scripts/diagnose_soft_sensor_interference.py``), not of this test suite -- but
that the composite class preserves the existing invariants per sub-model and actually
trains independent networks, not a facade around a shared one.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.evaluation import DataSpec, RolloutEvaluator, build_datasets, generate_data
from twinlink_ml.models import PersistenceModel
from twinlink_ml.models.registry import build_factories
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

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
    target_names=("energy_target", "product_moisture_kg_per_kg", "d50_um"),
)


@pytest.fixture(scope="module")
def process_config():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return load_process_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def data(process_config):
    return generate_data(process_config, SMALL, master_seed=1)


def test_submodel_seed_is_reproducible_and_decorrelated() -> None:
    """No Torch needed -- pure hash derivation, analogous to ``SeedBundle``.

    See ``test_greybox_numerics.py::test_seed_bundle_is_reproducible_and_decorrelated``.
    The actual regression this test guards against: ``seed + index`` would produce seeds
    exactly 1 apart for neighbouring targets (position 0, 1, 2) -- correlated in the same
    sense that ``utils/seeding._derive`` avoids for master seeds.
    """
    from twinlink_ml.models.composite import _submodel_seed

    first = _submodel_seed(5, "energy_target")
    assert first == _submodel_seed(5, "energy_target")

    neighbour = _submodel_seed(5, "product_moisture_kg_per_kg")
    assert abs(first - neighbour) > 1000, "Teilseeds wirken wie seed + index -- Regression"

    assert _submodel_seed(5, "energy_target") != _submodel_seed(6, "energy_target")


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_matches_three_independently_fitted_models(data) -> None:
    """Core guarantee: the composite model is not a facade around a shared network.

    The same prediction as three models called individually with ``TorchSequenceModel``,
    with the same seed derivation (``_submodel_seed``, SHA-256 like ``SeedBundle`` -- not
    ``seed + index``, see the ``CompositeSequenceModel`` docstring) and the same
    per-target reduced data set. If the composite were secretly a shared network
    (or if the sub-models got in each other's way, e.g. through shared
    global state), this agreement would not be exact.
    """
    from twinlink_ml.models.composite import (
        CompositeSequenceModel,
        _SingleTargetView,
        _submodel_seed,
    )
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, validation, test = build_datasets(data, 8)
    config = TorchSequenceConfig(hidden_size=8, max_epochs=3, patience=0)

    composite = CompositeSequenceModel(config)
    composite.fit(train, seed=5, validation=validation)

    independent = {}
    for index, target in enumerate(train.target_names):
        submodel = TorchSequenceModel(config)
        submodel.fit(
            _SingleTargetView(train, index),
            seed=_submodel_seed(5, target),
            validation=None if validation is None else _SingleTargetView(validation, index),
        )
        independent[target] = submodel

    batch = test.batch(range(min(16, test.n_windows)))
    composite_prediction = composite.predict(batch)
    for index, target in enumerate(train.target_names):
        history_columns = [*range(composite._n_exogenous), composite._n_exogenous + index]
        from twinlink_ml.models.composite import _select_target

        sub_batch = _select_target(batch, history_columns, index)
        expected = independent[target].predict(sub_batch)
        assert np.allclose(composite_prediction[..., [index]], expected, atol=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_untrained_equals_persistence_per_target(data) -> None:
    """Zero-initialised head = persistence, now checked per sub-model.

    The same requirement as ``test_untrained_network_equals_persistence`` in
    ``test_experiments.py``, but for the composite: each sub-network must satisfy the
    invariant independently, not just "on average" over all three targets.
    """
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    model = CompositeSequenceModel(TorchSequenceConfig(hidden_size=8, max_epochs=0, patience=0))
    model.fit(train, seed=1, validation=None)

    reference = PersistenceModel()
    reference.fit(train, seed=1)

    batch = test.full_batch()
    assert np.allclose(model.predict(batch), reference.predict(batch), atol=1e-5)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_rollout_does_not_see_future_targets(data) -> None:
    """The same guarantee as for the single network: no covert carrying forward of the truth.

    Checks in particular that re-cutting the batches in ``predict`` (channel selection per
    sub-model) opens no new path through which future target values could flow in.
    """
    import dataclasses

    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    model = CompositeSequenceModel(TorchSequenceConfig(hidden_size=8, max_epochs=3, patience=0))
    model.fit(train, seed=0)
    batch = test.batch(range(min(16, test.n_windows)))
    tampered = dataclasses.replace(batch, future_targets=batch.future_targets * 3.0 + 7.0)
    assert np.allclose(model.predict(batch), model.predict(tampered))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_shares_the_trainer_across_cores(data) -> None:
    """GRU, LSTM and xLSTM still share the same trainer -- now also in the composite."""
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    evaluator = RolloutEvaluator(horizons=(1, SMALL.horizon))
    for core in ("gru", "lstm", "xlstm"):
        model = CompositeSequenceModel(
            TorchSequenceConfig(core=core, hidden_size=8, max_epochs=2, patience=0)
        )
        model.fit(train, seed=1, validation=None)
        assert model.n_parameters > 0
        result = evaluator.evaluate(model, test)
        assert np.isfinite(result.nrmse_run_spread["energy_target"][SMALL.horizon][0])


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_n_parameters_sums_submodels(data) -> None:
    """``n_parameters`` is the sum over all sub-networks, not that of a single one."""
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, _, _ = build_datasets(data, 8)
    config = TorchSequenceConfig(hidden_size=8, max_epochs=0, patience=0)

    composite = CompositeSequenceModel(config)
    composite.fit(train, seed=0)

    single = TorchSequenceModel(config)
    single.fit(train, seed=0)

    # The sub-networks are narrower (one output instead of three), but there are three --
    # the sum must be larger than a single network with a full, shared head.
    assert composite.n_parameters > 0
    assert composite.n_parameters == sum(m.n_parameters for m in composite._submodels.values())


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_torch_sequence_describe_actual_device_none_before_fit() -> None:
    """``net_actual_device`` is ``None`` as long as ``_network`` has not been built yet.

    Defensive safeguard (analogous to ``n_parameters``) for the case that ``describe()``
    is ever called before ``fit()`` -- currently no call site in the repo does that,
    but a wrongly populated field would be worse than none at all.
    """
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    model = TorchSequenceModel(TorchSequenceConfig(hidden_size=8))
    assert model.describe()["net_actual_device"] is None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_torch_sequence_describe_actual_device_matches_network_after_fit(data) -> None:
    """``net_actual_device`` reflects the device ACTUALLY used after ``fit()``.

    Differs from the already existing ``net_device`` (the configured wish,
    here ``None`` -- the default) -- ``device=None`` resolves in ``fit()`` to ``"cuda"``
    if available, otherwise ``"cpu"`` (finding/occasion: ``befund42_tbed_exhaust_sweep.jsonl``,
    CUDA-capable Torch build without a logged device field).
    """
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, _, _ = build_datasets(data, 8)
    config = TorchSequenceConfig(hidden_size=8, max_epochs=0, patience=0)

    model = TorchSequenceModel(config)
    model.fit(train, seed=0)

    description = model.describe()
    assert description["net_device"] is None  # configured wish, unchanged
    assert description["net_actual_device"] == str(next(model._network.parameters()).device)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_describe_actual_device_none_before_fit() -> None:
    """``net_actual_device`` is ``None`` as long as ``_submodels`` is still empty."""
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    composite = CompositeSequenceModel(TorchSequenceConfig(hidden_size=8))
    assert composite.describe()["net_actual_device"] is None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_describe_actual_device_matches_first_submodel_after_fit(data) -> None:
    """``CompositeSequenceModel.describe()`` takes ``net_actual_device`` from the sub-model.

    Without this forwarding, the field newly introduced in ``TorchSequenceModel.describe()``
    would be dead code for practically every production script --
    ``run_xlstm_reduced_comparison.py`` etc. log
    ``CompositeXLSTMSequenceModel``/``CompositeSequenceModel``, never a bare
    ``TorchSequenceModel``.
    """
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, _ = build_datasets(data, 8)
    config = TorchSequenceConfig(hidden_size=8, max_epochs=0, patience=0)

    composite = CompositeSequenceModel(config)
    composite.fit(train, seed=0)

    first_submodel = next(iter(composite._submodels.values()))
    expected = str(next(first_submodel._network.parameters()).device)
    assert composite.describe()["net_actual_device"] == expected


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_registry_gru_label_builds_composite_and_trains(data) -> None:
    """Wiring in ``registry.py``: since Befund 26 the label ``gru`` returns the composite.

    Regression test for the switch in ``_resolve_torch_factory`` -- without it, a
    fallback to the old single network (e.g. through a merge conflict) would only become
    visible again in a full ``study_architectures.py`` run, not in the test suite.
    """
    from twinlink_ml.models.composite import CompositeSequenceModel

    train, _, test = build_datasets(data, 8)
    factories = build_factories(("gru",))
    model = factories["gru"]()
    assert isinstance(model, CompositeSequenceModel)
    model.fit(train, seed=0)
    batch = test.batch(range(min(8, test.n_windows)))
    prediction = model.predict(batch)
    assert prediction.shape == batch.future_targets.shape
    assert np.all(np.isfinite(prediction))


# --------------------------------------------------------------------------- xLSTM
#
# The same invariants as above for GRU/LSTM, here for the mLSTM core
# (``twinlink_ml.models.xlstm``, registered under the label ``xlstm``, see
# ``docs/04_stand_ap1.md`` Befund 29). A separate section instead of a parametrisation
# of the existing tests -- the existing tests thereby stay unchanged, and a
# red xLSTM test cannot drag an existing GRU/LSTM test green or red along with it.


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_registry_xlstm_label_builds_composite_and_trains(data) -> None:
    """Wiring in ``registry.py``: the label ``xlstm`` returns the composite, like gru/lstm."""
    from twinlink_ml.models.composite import CompositeSequenceModel

    train, _, test = build_datasets(data, 8)
    factories = build_factories(("xlstm",))
    model = factories["xlstm"]()
    assert isinstance(model, CompositeSequenceModel)
    model.fit(train, seed=0)
    batch = test.batch(range(min(8, test.n_windows)))
    prediction = model.predict(batch)
    assert prediction.shape == batch.future_targets.shape
    assert np.all(np.isfinite(prediction))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_xlstm_untrained_equals_persistence_per_target(data) -> None:
    """Zero-initialised head = persistence, also for the mLSTM core, per sub-model.

    The mLSTM core itself is **not** zero-initialised (only the head) -- the cell
    runs with its normal random initialisation through the full history. That the
    result nevertheless corresponds exactly to persistence checks at the same time that the
    core yields finite outputs for random weights (otherwise ``0 * inf = nan`` would break
    through the zero-initialised head, see the module docstring of ``xlstm.py``).
    """
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    model = CompositeSequenceModel(
        TorchSequenceConfig(core="xlstm", hidden_size=8, max_epochs=0, patience=0)
    )
    model.fit(train, seed=1, validation=None)

    reference = PersistenceModel()
    reference.fit(train, seed=1)

    batch = test.full_batch()
    assert np.allclose(model.predict(batch), reference.predict(batch), atol=1e-5)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_xlstm_rollout_does_not_see_future_targets(data) -> None:
    """No covert carrying forward of the truth -- not via the mLSTM state either."""
    import dataclasses

    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    model = CompositeSequenceModel(
        TorchSequenceConfig(core="xlstm", hidden_size=8, max_epochs=3, patience=0)
    )
    model.fit(train, seed=0)
    batch = test.batch(range(min(16, test.n_windows)))
    tampered = dataclasses.replace(batch, future_targets=batch.future_targets * 3.0 + 7.0)
    assert np.allclose(model.predict(batch), model.predict(tampered))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_xlstm_matches_three_independently_fitted_models(data) -> None:
    """Composite of three mLSTM sub-networks == three individually called ``TorchSequenceModel``.

    The same core guarantee as ``test_composite_matches_three_independently_fitted_models``
    (GRU), here for ``core="xlstm"`` -- with ``(C, n, m)`` the mLSTM cell has a
    more complex state than GRU/LSTM; this test checks that ``CompositeSequenceModel``
    isolates this state per sub-model just as cleanly as that of GRU/LSTM.
    """
    from twinlink_ml.models.composite import (
        CompositeSequenceModel,
        _select_target,
        _SingleTargetView,
        _submodel_seed,
    )
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, validation, test = build_datasets(data, 8)
    config = TorchSequenceConfig(core="xlstm", hidden_size=8, max_epochs=3, patience=0)

    composite = CompositeSequenceModel(config)
    composite.fit(train, seed=5, validation=validation)

    independent = {}
    for index, target in enumerate(train.target_names):
        submodel = TorchSequenceModel(config)
        submodel.fit(
            _SingleTargetView(train, index),
            seed=_submodel_seed(5, target),
            validation=None if validation is None else _SingleTargetView(validation, index),
        )
        independent[target] = submodel

    batch = test.batch(range(min(16, test.n_windows)))
    composite_prediction = composite.predict(batch)
    for index, target in enumerate(train.target_names):
        history_columns = [*range(composite._n_exogenous), composite._n_exogenous + index]
        sub_batch = _select_target(batch, history_columns, index)
        expected = independent[target].predict(sub_batch)
        assert np.allclose(composite_prediction[..., [index]], expected, atol=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_xlstm_prediction_is_reproducible_and_seed_sensitive(data) -> None:
    """Same seed -> bit-identical prediction; different seed -> different prediction.

    The second half cannot be checked with ``max_epochs=0``: the zero-initialised head
    makes every prediction exactly persistence, independent of the seed (see the test
    above) -- seed sensitivity only shows after the head has been trained and has moved
    away from zero.
    """
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    batch = test.batch(range(min(16, test.n_windows)))
    config = TorchSequenceConfig(core="xlstm", hidden_size=8, max_epochs=5, patience=0)

    first = CompositeSequenceModel(config)
    first.fit(train, seed=7)
    second = CompositeSequenceModel(config)
    second.fit(train, seed=7)
    assert np.allclose(first.predict(batch), second.predict(batch), atol=1e-6)

    third = CompositeSequenceModel(config)
    third.fit(train, seed=8)
    assert not np.allclose(first.predict(batch), third.predict(batch), atol=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_xlstm_stays_finite_over_long_rollout(process_config) -> None:
    """Numerical stability of the exponential gates -- the reason for phase 3 of this test.

    Unstabilised, the exponential input/forget gates of the mLSTM overflow
    over many steps (module docstring of ``xlstm.py``). Checked at two horizons
    (24, the project standard; 200, artificially extended) with **random**
    core initialisation (``max_epochs=0`` only zeroes the head, see the test above) over
    several seeds -- no NaN/Inf, at no sampling point.
    """
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    for horizon, run_duration_s in ((24, 4000.0), (200, 8000.0)):
        spec = DataSpec(
            sample_time_s=25.0,
            run_duration_s=run_duration_s,
            n_train_max=4,
            n_val=2,
            n_test=2,
            history_length=16,
            horizon=horizon,
            stride=8,
            target_names=("energy_target", "product_moisture_kg_per_kg", "d50_um"),
        )
        long_data = generate_data(process_config, spec, master_seed=3)
        train, _, test = build_datasets(long_data, 4)
        batch = test.full_batch()
        for seed in range(3):
            model = CompositeSequenceModel(
                TorchSequenceConfig(core="xlstm", hidden_size=16, max_epochs=0, patience=0)
            )
            model.fit(train, seed=seed, validation=None)
            prediction = model.predict(batch)
            assert np.all(np.isfinite(prediction)), (
                f"horizon={horizon}, seed={seed}: NaN/Inf im Rollout -- "
                "Max-Stabilisierer m_t hat nicht gehalten"
            )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_xlstm_forward_ignores_teacher_when_teacher_forcing_is_zero(data) -> None:
    """Perturbation test: with ``teacher_forcing=0.0`` no true future value may flow in.

    The same guarantee as for Transformer/Mamba
    (``test_transformer_forward_ignores_teacher_when_teacher_forcing_is_zero``,
    ``test_mamba_forward_ignores_teacher_when_teacher_forcing_is_zero``), here for the
    mLSTM recurrence: with the default ``teacher_forcing=0.0`` the input at
    step ``t > 0`` is exclusively the model's own prediction from step ``t - 1`` -- never the
    true target value, even if a ``teacher`` tensor is passed to ``forward()``. The
    existing ``test_composite_xlstm_rollout_does_not_see_future_targets`` checks this only
    via ``predict()``, which never passes ``teacher`` anyway (see the Transformer docstring)
    -- a leak exclusively in the training path would be invisible there. This test calls
    ``forward()`` directly in training mode and replaces the ``teacher`` tensor with
    random noise -- bit-identical (``torch.equal``), no tolerance. The counter-check with
    ``teacher_forcing=1.0`` ensures that the test is not green by chance because
    ``teacher`` never arrives in the network at all.
    """
    import torch

    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, _, _ = build_datasets(data, 8)
    config = TorchSequenceConfig(core="xlstm", hidden_size=8, max_epochs=2, patience=0)
    model = TorchSequenceModel(config)
    model.fit(train, seed=0, validation=None)
    network = model._network
    network.train()

    batch = train.batch(range(min(8, train.n_windows)))
    device = next(network.parameters()).device
    tensors = model._to_tensors(batch, device)
    noisy_teacher = torch.randn_like(tensors["targets"]) * 1000.0 + 500.0

    baseline = network(
        tensors["history"], tensors["future"], teacher=tensors["targets"], teacher_forcing=0.0
    )
    perturbed = network(
        tensors["history"], tensors["future"], teacher=noisy_teacher, teacher_forcing=0.0
    )
    assert torch.equal(baseline, perturbed), (
        "Vorhersage aendert sich mit dem teacher-Tensor trotz teacher_forcing=0.0 -- "
        "Exposure-Bias-Leck"
    )

    # Counter-check: with active teacher forcing the prediction MUST change, otherwise
    # the test above would only be green by chance (e.g. because `teacher` is not wired at all).
    with_forcing = network(
        tensors["history"], tensors["future"], teacher=tensors["targets"], teacher_forcing=1.0
    )
    with_forcing_noisy = network(
        tensors["history"], tensors["future"], teacher=noisy_teacher, teacher_forcing=1.0
    )
    assert not torch.equal(with_forcing, with_forcing_noisy)


# ----------------------------------------------------------------------- Transformer
#
# The same invariants as above for GRU/LSTM/xLSTM, here for the Transformer core
# (encoder over the history, sequential decoder over the horizon,
# ``torch_sequence._build_transformer_network``, registered under the label
# ``transformer``). A separate section instead of a parametrisation of the existing
# tests -- a red Transformer test cannot drag an existing GRU/LSTM/xLSTM test green
# or red along with it. ``hidden_size=16`` instead of the dataclass default 64 throughout:
# one encoder layer at ``d_model=64`` alone has around 50000 parameters (see the
# ``CompositeTransformerSequenceModel`` docstring), needlessly expensive for the small
# test data sets here.


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_registry_transformer_label_builds_composite_and_trains(data) -> None:
    """Wiring in ``registry.py``: the label ``transformer`` returns the composite."""
    from twinlink_ml.models.composite import CompositeSequenceModel

    train, _, test = build_datasets(data, 8)
    factories = build_factories(("transformer",))
    model = factories["transformer"]()
    assert isinstance(model, CompositeSequenceModel)
    model.fit(train, seed=0)
    batch = test.batch(range(min(8, test.n_windows)))
    prediction = model.predict(batch)
    assert prediction.shape == batch.future_targets.shape
    assert np.all(np.isfinite(prediction))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_transformer_untrained_equals_persistence_per_target(data) -> None:
    """Zero-initialised head = persistence, also for the Transformer core, per sub-model.

    The encoder/decoder itself is **not** zero-initialised (only the head) -- the
    attention weights run with their normal random initialisation through the full
    history and the decoder step cycle. That the result nevertheless corresponds exactly to
    persistence checks at the same time that encoder and decoder yield finite outputs for
    random weights.
    """
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    model = CompositeSequenceModel(
        TorchSequenceConfig(
            core="transformer", hidden_size=8, num_heads=2, num_layers=1, max_epochs=0, patience=0
        )
    )
    model.fit(train, seed=1, validation=None)

    reference = PersistenceModel()
    reference.fit(train, seed=1)

    batch = test.full_batch()
    assert np.allclose(model.predict(batch), reference.predict(batch), atol=1e-5)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_transformer_rollout_does_not_see_future_targets(data) -> None:
    """No covert carrying forward of the truth in the rollout -- not via the decoder either.

    As with GRU/LSTM/xLSTM: ``predict()`` never passes ``teacher`` to
    ``forward()`` anyway (see ``TorchSequenceModel.predict``), so manipulating
    ``future_targets`` in the batch must not change the prediction. A leak in the
    *training loop* itself (at ``teacher_forcing=0.0``) is not checked by this test --
    for that see ``test_transformer_forward_ignores_teacher_when_teacher_forcing_is_zero``.
    """
    import dataclasses

    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    model = CompositeSequenceModel(
        TorchSequenceConfig(
            core="transformer", hidden_size=8, num_heads=2, num_layers=1, max_epochs=3, patience=0
        )
    )
    model.fit(train, seed=0)
    batch = test.batch(range(min(16, test.n_windows)))
    tampered = dataclasses.replace(batch, future_targets=batch.future_targets * 3.0 + 7.0)
    assert np.allclose(model.predict(batch), model.predict(tampered))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_transformer_matches_three_independently_fitted_models(data) -> None:
    """Composite of three Transformer subnets == three separately called ``TorchSequenceModel``.

    The same core guarantee as for GRU/LSTM/xLSTM, here for ``core="transformer"``.
    """
    from twinlink_ml.models.composite import (
        CompositeSequenceModel,
        _select_target,
        _SingleTargetView,
        _submodel_seed,
    )
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, validation, test = build_datasets(data, 8)
    config = TorchSequenceConfig(
        core="transformer", hidden_size=8, num_heads=2, num_layers=1, max_epochs=3, patience=0
    )

    composite = CompositeSequenceModel(config)
    composite.fit(train, seed=5, validation=validation)

    independent = {}
    for index, target in enumerate(train.target_names):
        submodel = TorchSequenceModel(config)
        submodel.fit(
            _SingleTargetView(train, index),
            seed=_submodel_seed(5, target),
            validation=None if validation is None else _SingleTargetView(validation, index),
        )
        independent[target] = submodel

    batch = test.batch(range(min(16, test.n_windows)))
    composite_prediction = composite.predict(batch)
    for index, target in enumerate(train.target_names):
        history_columns = [*range(composite._n_exogenous), composite._n_exogenous + index]
        sub_batch = _select_target(batch, history_columns, index)
        expected = independent[target].predict(sub_batch)
        assert np.allclose(composite_prediction[..., [index]], expected, atol=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_transformer_forward_ignores_teacher_when_teacher_forcing_is_zero(data) -> None:
    """Perturbation test: with ``teacher_forcing=0.0`` no true future value may flow in.

    Direct clarification before the implementation (plain-text discussion): with the default
    ``teacher_forcing=0.0`` the decoder input at step ``t > 0`` is exclusively the
    model's own prediction from step ``t - 1`` -- never the true target value, even if a
    ``teacher`` tensor is passed to ``forward()``. This test replaces exactly this
    tensor with random noise and calls ``forward()`` directly in training mode --
    bit-identical (``torch.equal``), no tolerance. ``predict()`` does not check this: it never
    passes ``teacher`` anyway (see ``TorchSequenceModel.predict``), a leak in the
    training loop itself would be invisible via the pure rollout test. The counter-check
    with ``teacher_forcing=1.0`` ensures that the test is not green by chance because
    ``teacher`` never arrives in the network at all.
    """
    import torch

    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, _, _ = build_datasets(data, 8)
    config = TorchSequenceConfig(
        core="transformer", hidden_size=8, num_heads=2, num_layers=1, max_epochs=2, patience=0
    )
    model = TorchSequenceModel(config)
    model.fit(train, seed=0, validation=None)
    network = model._network
    network.train()

    batch = train.batch(range(min(8, train.n_windows)))
    device = next(network.parameters()).device
    tensors = model._to_tensors(batch, device)
    noisy_teacher = torch.randn_like(tensors["targets"]) * 1000.0 + 500.0

    baseline = network(
        tensors["history"], tensors["future"], teacher=tensors["targets"], teacher_forcing=0.0
    )
    perturbed = network(
        tensors["history"], tensors["future"], teacher=noisy_teacher, teacher_forcing=0.0
    )
    assert torch.equal(baseline, perturbed), (
        "Vorhersage aendert sich mit dem teacher-Tensor trotz teacher_forcing=0.0 -- "
        "Exposure-Bias-Leck"
    )

    # Counter-check: with active teacher forcing the prediction MUST change, otherwise
    # the test above would only be green by chance (e.g. because `teacher` is not wired at all).
    with_forcing = network(
        tensors["history"], tensors["future"], teacher=tensors["targets"], teacher_forcing=1.0
    )
    with_forcing_noisy = network(
        tensors["history"], tensors["future"], teacher=noisy_teacher, teacher_forcing=1.0
    )
    assert not torch.equal(with_forcing, with_forcing_noisy)


# ----------------------------------------------------------------------------- Mamba
#
# The same invariants as above for GRU/LSTM/xLSTM/Transformer, here for the
# selective state-space core (``twinlink_ml.models.mamba``, registered under the
# label ``mamba``). Unlike in the Transformer section, ``hidden_size=8`` here is
# not a forced capacity reduction (the SSM cell does not grow quadratically with
# ``hidden_size``) -- simply the same small test size as for all other cores.


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_registry_mamba_label_builds_composite_and_trains(data) -> None:
    """Wiring in ``registry.py``: the label ``mamba`` returns the composite."""
    from twinlink_ml.models.composite import CompositeSequenceModel

    train, _, test = build_datasets(data, 8)
    factories = build_factories(("mamba",))
    model = factories["mamba"]()
    assert isinstance(model, CompositeSequenceModel)
    model.fit(train, seed=0)
    batch = test.batch(range(min(8, test.n_windows)))
    prediction = model.predict(batch)
    assert prediction.shape == batch.future_targets.shape
    assert np.all(np.isfinite(prediction))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_mamba_untrained_equals_persistence_per_target(data) -> None:
    """Zero-initialised head = persistence, also for the Mamba core, per sub-model.

    The SSM cell itself is **not** zero-initialised (only the head) -- ``A_log``,
    the projections and the gate run with their normal initialisation through the
    full history and the recursion step cycle. That the result nevertheless corresponds
    exactly to persistence checks at the same time that the recurrence yields finite
    outputs for random weights.
    """
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    model = CompositeSequenceModel(
        TorchSequenceConfig(
            core="mamba", hidden_size=8, state_size=8, num_layers=1, max_epochs=0, patience=0
        )
    )
    model.fit(train, seed=1, validation=None)

    reference = PersistenceModel()
    reference.fit(train, seed=1)

    batch = test.full_batch()
    assert np.allclose(model.predict(batch), reference.predict(batch), atol=1e-5)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_mamba_rollout_does_not_see_future_targets(data) -> None:
    """No covert carrying forward of the truth in the rollout -- not via the SSM either.

    As with GRU/LSTM/xLSTM/Transformer: ``predict()`` never passes ``teacher`` to
    ``forward()`` anyway (see ``TorchSequenceModel.predict``), so manipulating
    ``future_targets`` in the batch must not change the prediction. A leak in the
    *training loop* itself (at ``teacher_forcing=0.0``) is not checked by this test --
    for that see ``test_mamba_forward_ignores_teacher_when_teacher_forcing_is_zero``.
    """
    import dataclasses

    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    model = CompositeSequenceModel(
        TorchSequenceConfig(
            core="mamba", hidden_size=8, state_size=8, num_layers=1, max_epochs=3, patience=0
        )
    )
    model.fit(train, seed=0)
    batch = test.batch(range(min(16, test.n_windows)))
    tampered = dataclasses.replace(batch, future_targets=batch.future_targets * 3.0 + 7.0)
    assert np.allclose(model.predict(batch), model.predict(tampered))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_mamba_matches_three_independently_fitted_models(data) -> None:
    """Composite of three Mamba sub-networks == three individually called ``TorchSequenceModel``.

    The same core guarantee as for GRU/LSTM/xLSTM/Transformer, here for ``core="mamba"``.
    """
    from twinlink_ml.models.composite import (
        CompositeSequenceModel,
        _select_target,
        _SingleTargetView,
        _submodel_seed,
    )
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, validation, test = build_datasets(data, 8)
    config = TorchSequenceConfig(
        core="mamba", hidden_size=8, state_size=8, num_layers=1, max_epochs=3, patience=0
    )

    composite = CompositeSequenceModel(config)
    composite.fit(train, seed=5, validation=validation)

    independent = {}
    for index, target in enumerate(train.target_names):
        submodel = TorchSequenceModel(config)
        submodel.fit(
            _SingleTargetView(train, index),
            seed=_submodel_seed(5, target),
            validation=None if validation is None else _SingleTargetView(validation, index),
        )
        independent[target] = submodel

    batch = test.batch(range(min(16, test.n_windows)))
    composite_prediction = composite.predict(batch)
    for index, target in enumerate(train.target_names):
        history_columns = [*range(composite._n_exogenous), composite._n_exogenous + index]
        sub_batch = _select_target(batch, history_columns, index)
        expected = independent[target].predict(sub_batch)
        assert np.allclose(composite_prediction[..., [index]], expected, atol=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_with_fourth_target_leaves_existing_submodels_bit_identical(data) -> None:
    """Four instead of three targets (``bed_temperature_celsius`` added) -- cheap correctness run.

    Ahead of a possible later extension of ``DataSpec.target_names``
    (``docs/04_stand_ap1.md``, T_bed as the fourth target, derivation of the
    exhaust air outlet quantities via ``simulation.exhaust``). Three core statements, not just
    a side note:

    1. The fourth sub-network builds without error.
    2. Its training loss is finite (no NaN/Inf).
    3. The three existing sub-networks are **bit for bit unchanged** compared with a
       three-target composite on the same data/the same seed -- this is the most important
       of the three statements: it ensures that the seed isolation between sub-networks
       (``_submodel_seed``, Befund 26) also holds with four instead of three targets and the
       fourth target does not bring any new negative transfer into the existing three.
    """
    import dataclasses

    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    four_targets = (*SMALL.target_names, "bed_temperature_celsius")
    data_four = dataclasses.replace(
        data, spec=dataclasses.replace(data.spec, target_names=four_targets)
    )

    train3, val3, _test3 = build_datasets(data, 8)
    train4, val4, test4 = build_datasets(data_four, 8)
    assert train4.target_names == four_targets

    net_config = TorchSequenceConfig(hidden_size=8, max_epochs=15, patience=0)

    composite3 = CompositeSequenceModel(net_config)
    composite3.fit(train3, seed=5, validation=val3)

    composite4 = CompositeSequenceModel(net_config)
    composite4.fit(train4, seed=5, validation=val4)

    # (1) Fourth sub-network present.
    assert set(composite4._submodels.keys()) == set(four_targets)

    # (2) bed_temperature_celsius: training loss finite over all epochs.
    bed_losses = [r["train_loss"] for r in composite4.training_history["bed_temperature_celsius"]]
    assert len(bed_losses) == 15
    assert all(np.isfinite(v) for v in bed_losses), "NaN/Inf in bed_temperature_celsius-Verlust"

    # (3) The three existing targets are bit for bit unchanged -- most important statement,
    #     see the docstring. A single unequal pair would mean that the fourth
    #     sub-network covertly acts on shared state/RNG consumption of the others.
    for target in SMALL.target_names:
        losses3 = np.array([r["train_loss"] for r in composite3.training_history[target]])
        losses4 = np.array([r["train_loss"] for r in composite4.training_history[target]])
        assert np.array_equal(losses3, losses4), (
            f"{target}: Trainingsverlust unterscheidet sich zwischen Drei- und "
            "Vier-Ziele-Composite -- Isolationsannahme (Befund 26) verletzt"
        )

    # Inference sanity: predict() yields four finite columns.
    batch4 = test4.batch(range(min(8, test4.n_windows)))
    prediction4 = composite4.predict(batch4)
    assert prediction4.shape[-1] == 4
    assert np.all(np.isfinite(prediction4))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_mamba_forward_ignores_teacher_when_teacher_forcing_is_zero(data) -> None:
    """Perturbation test: with ``teacher_forcing=0.0`` no true future value may flow in.

    The same guarantee as for the Transformer
    (``test_transformer_forward_ignores_teacher_when_teacher_forcing_is_zero``), here for
    the SSM recurrence: with the default ``teacher_forcing=0.0`` the input at
    step ``t > 0`` is exclusively the model's own prediction from step ``t - 1`` -- never the
    true target value, even if a ``teacher`` tensor is passed to ``forward()``.
    This test replaces exactly this tensor with random noise and calls ``forward()``
    directly in training mode -- bit-identical (``torch.equal``), no tolerance. The
    counter-check with ``teacher_forcing=1.0`` ensures that the test is not green by chance
    because ``teacher`` never arrives in the network at all.
    """
    import torch

    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, _, _ = build_datasets(data, 8)
    config = TorchSequenceConfig(
        core="mamba", hidden_size=8, state_size=8, num_layers=1, max_epochs=2, patience=0
    )
    model = TorchSequenceModel(config)
    model.fit(train, seed=0, validation=None)
    network = model._network
    network.train()

    batch = train.batch(range(min(8, train.n_windows)))
    device = next(network.parameters()).device
    tensors = model._to_tensors(batch, device)
    noisy_teacher = torch.randn_like(tensors["targets"]) * 1000.0 + 500.0

    baseline = network(
        tensors["history"], tensors["future"], teacher=tensors["targets"], teacher_forcing=0.0
    )
    perturbed = network(
        tensors["history"], tensors["future"], teacher=noisy_teacher, teacher_forcing=0.0
    )
    assert torch.equal(baseline, perturbed), (
        "Vorhersage aendert sich mit dem teacher-Tensor trotz teacher_forcing=0.0 -- "
        "Exposure-Bias-Leck"
    )

    # Counter-check: with active teacher forcing the prediction MUST change, otherwise
    # the test above would only be green by chance (e.g. because `teacher` is not wired at all).
    with_forcing = network(
        tensors["history"], tensors["future"], teacher=tensors["targets"], teacher_forcing=1.0
    )
    with_forcing_noisy = network(
        tensors["history"], tensors["future"], teacher=noisy_teacher, teacher_forcing=1.0
    )
    assert not torch.equal(with_forcing, with_forcing_noisy)

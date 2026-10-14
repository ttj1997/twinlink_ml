"""Fine-tuning (study A1): ``TransferSequenceModel.fine_tune``, freezing, scaler preservation."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from twinlink_ml.evaluation.experiments import DataSpec, build_datasets
from twinlink_ml.evaluation.plant_data import generate_plant_data
from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel
from twinlink_ml.models.transfer import (
    FreezeSpec,
    TransferSequenceModel,
    build_datasets_with_scalers,
    load_pretrained,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.plant_variants import PLANT_VARIANTS
from twinlink_ml.utils.checkpoints import save_checkpoint

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="PyTorch not installed"
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_gab_placeholder.yaml"
)


@pytest.fixture(scope="module")
def data():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    spec = DataSpec.new_study(
        run_duration_s=1_500.0, n_train_max=4, n_val=2, n_test=2, history_length=8, horizon=4
    )
    return generate_plant_data(config, PLANT_VARIANTS["P0"], spec, master_seed=5)


def _config(core: str = "lstm", **overrides: Any) -> TorchSequenceConfig:
    values = {"core": core, "hidden_size": 8, "max_epochs": 3, "patience": 0, "batch_size": 32}
    values.update(overrides)
    return TorchSequenceConfig(**values)


def _equal_states(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[k].numpy(), b[k].numpy()) for k in a)


@pytest.mark.parametrize("core", ["gru", "lstm"])
def test_fine_tune_from_a_fresh_network_is_bit_identical_to_fit(data, core: str) -> None:
    """The newly written loop must not deviate from ``fit`` (k epochs, no early stopping)."""
    train, _, _ = build_datasets(data, 2)
    config = _config(core)
    reference = TorchSequenceModel(config)
    reference.fit(train, seed=7)

    tuned = TransferSequenceModel(config)
    tuned.initialise(train, seed=7)
    tuned.fine_tune(train, seed=7, learning_rate=config.learning_rate, max_epochs=config.max_epochs)
    assert _equal_states(reference.state_dict(), tuned.state_dict())
    assert [r["train_loss"] for r in reference.training_history] == [
        r["train_loss"] for r in tuned.training_history
    ]


def test_fine_tune_is_reproducible(data) -> None:
    train, _, _ = build_datasets(data, 2)
    states = []
    for _ in range(2):
        model = TransferSequenceModel(_config())
        model.initialise(train, seed=3)
        model.fine_tune(train, seed=11, learning_rate=1e-2, max_epochs=2)
        states.append(model.state_dict())
    assert _equal_states(*states)


def test_fine_tune_needs_a_network(data) -> None:
    train, _, _ = build_datasets(data, 2)
    with pytest.raises(RuntimeError, match="no network"):
        TransferSequenceModel(_config()).fine_tune(train, seed=0, learning_rate=1e-3, max_epochs=1)


def test_fine_tune_starts_from_the_present_weights(data) -> None:
    train, _, _ = build_datasets(data, 2)
    model = TransferSequenceModel(_config(max_epochs=4))
    model.fit(train, seed=1)
    before = model.state_dict()
    model.fine_tune(train, seed=2, learning_rate=1e-9, max_epochs=1)
    after = model.state_dict()
    for key in before:
        assert np.allclose(before[key].numpy(), after[key].numpy(), atol=1e-6), key
    # the starting point is the trained network, not a fresh (zero-head) one
    assert any(np.abs(before[k].numpy()).max() > 0 for k in before if k.startswith("head"))


def test_frozen_core_and_head_do_not_move(data) -> None:
    train, _, _ = build_datasets(data, 2)
    for freeze, frozen_prefix, moving_prefix in (
        (FreezeSpec(core=True), "core", "head"),
        (FreezeSpec(head=True), "head", "core"),
    ):
        model = TransferSequenceModel(_config())
        model.fit(train, seed=1)
        before = model.state_dict()
        model.fine_tune(train, seed=2, learning_rate=1e-2, max_epochs=2, freeze=freeze)
        after = model.state_dict()
        assert all(
            np.array_equal(before[k].numpy(), after[k].numpy())
            for k in before
            if k.startswith(frozen_prefix)
        )
        assert any(
            not np.array_equal(before[k].numpy(), after[k].numpy())
            for k in before
            if k.startswith(moving_prefix)
        )
        assert model.frozen_parameters


def test_freezing_single_layers(data) -> None:
    train, _, _ = build_datasets(data, 2)
    model = TransferSequenceModel(_config(num_layers=2))
    model.fit(train, seed=1)
    before = model.state_dict()
    model.fine_tune(train, seed=2, learning_rate=1e-2, max_epochs=2, freeze=FreezeSpec(layers=(0,)))
    after = model.state_dict()
    for key in before:
        if key.endswith("_l0"):
            assert np.array_equal(before[key].numpy(), after[key].numpy()), key
    assert any(
        not np.array_equal(before[k].numpy(), after[k].numpy()) for k in before if k.endswith("_l1")
    )
    with pytest.raises(ValueError, match="no parameters of layer 5"):
        model.fine_tune(
            train, seed=2, learning_rate=1e-2, max_epochs=1, freeze=FreezeSpec(layers=(5,))
        )


def test_freezing_everything_is_an_error(data) -> None:
    train, _, _ = build_datasets(data, 2)
    model = TransferSequenceModel(_config())
    model.fit(train, seed=1)
    with pytest.raises(ValueError, match="every parameter is frozen"):
        model.fine_tune(
            train, seed=1, learning_rate=1e-2, max_epochs=1, freeze=FreezeSpec(core=True, head=True)
        )


def test_build_datasets_with_scalers_keeps_the_given_scalers(data) -> None:
    big_train, _, _ = build_datasets(data, 4)
    small_train, _, _ = build_datasets(data, 2)
    train, val, test = build_datasets_with_scalers(data, 2, big_train.scalers)
    assert not np.array_equal(small_train.input_scaler.mean, big_train.input_scaler.mean)
    for dataset in (train, val, test):
        assert dataset is not None
        assert np.array_equal(dataset.input_scaler.mean, big_train.input_scaler.mean)
        assert np.array_equal(dataset.target_scaler.scale, big_train.target_scaler.scale)
    assert train.n_windows == small_train.n_windows  # the same runs, only other statistics


def test_pretrained_scalers_are_enforced_and_the_checkpoint_round_trips(data, tmp_path) -> None:
    pre_train, _, pre_test = build_datasets(data, 4)
    model = TransferSequenceModel(_config())
    model.fit(pre_train, seed=1)
    path = tmp_path / "pre.pt"
    save_checkpoint(
        path,
        model,
        model_label="t",
        run_uid="t",
        target_names=pre_train.target_names,
        exogenous_names=pre_train.exogenous_names,
        n_exogenous=len(pre_train.exogenous_names),
        input_scaler=pre_train.input_scaler,
        target_scaler=pre_train.target_scaler,
        process_fingerprint="x",
        data_spec_params={},
        seed_bundle_params={},
        git_commit="x",
    )
    loaded = load_pretrained(path)
    batch = pre_test.full_batch()
    assert np.array_equal(model.predict(batch), loaded.predict(batch))
    assert loaded.pretrained_scalers is not None

    own_scalers_train, _, _ = build_datasets(data, 2)  # re-estimated statistics: refused
    with pytest.raises(ValueError, match="not the pretraining's"):
        loaded.fine_tune(own_scalers_train, seed=1, learning_rate=1e-3, max_epochs=1)
    kept_train, _, _ = build_datasets_with_scalers(data, 2, loaded.pretrained_scalers)
    loaded.fine_tune(kept_train, seed=1, learning_rate=1e-3, max_epochs=1)


def test_channel_mismatch_is_refused(data) -> None:
    train, _, _ = build_datasets(data, 2)
    model = TransferSequenceModel(_config())
    model.fit(train, seed=1)
    model._n_exogenous += 1
    with pytest.raises(ValueError, match="exogenous channels"):
        model.fine_tune(train, seed=1, learning_rate=1e-3, max_epochs=1)

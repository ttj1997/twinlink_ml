"""Tests for the stored loss history (VH1-VH3 of the history infrastructure task, Befund 67)."""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import pytest

from twinlink_ml.evaluation import DataSpec, build_datasets, generate_data
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.tracking import RunRecord, training_curves

HAS_TORCH = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "conf" / "process" / "greybox_placeholder.yaml"
HEADS = ("energy_target", "product_moisture_kg_per_kg", "d50_um")


@pytest.fixture(scope="module")
def data():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    spec = DataSpec(
        sample_time_s=25.0,
        run_duration_s=4000.0,
        n_train_max=8,
        n_val=4,
        n_test=4,
        history_length=16,
        horizon=8,
        stride=8,
        target_names=HEADS,
    )
    return generate_data(config, spec, master_seed=1)


def _fit(data, *, patience: int, max_epochs: int, validation: bool = True):
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, val, _ = build_datasets(data, 8)
    model = CompositeSequenceModel(
        TorchSequenceConfig(
            core="gru", hidden_size=8, max_epochs=max_epochs, patience=patience, learning_rate=0.05
        )
    )
    model.fit(train, seed=1, validation=val if validation else None)
    return model


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_history_arrives_and_best_epoch_matches_early_stopping(data) -> None:
    patience = 3
    model = _fit(data, patience=patience, max_epochs=80)
    diagnostics = model.training_diagnostics()
    stopped_early = 0
    for head in HEADS:
        epochs = diagnostics[f"epochs_run/{head}"]
        train = diagnostics[f"loss_history_train/{head}"]
        val = diagnostics[f"loss_history_val/{head}"]
        best = diagnostics[f"best_epoch/{head}"]
        assert len(train) == len(val) == epochs
        assert 0 <= best < epochs
        # the kept weights are those of the last epoch that improved the validation loss
        assert val[best] <= min(val) + 1e-5
        if epochs < 80:
            stopped_early += 1
            assert epochs == best + 1 + patience
    assert stopped_early >= 1, "no head stopped early; the test would not check the stop time"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_without_validation_the_last_epoch_is_kept(data) -> None:
    model = _fit(data, patience=0, max_epochs=3, validation=False)
    diagnostics = model.training_diagnostics()
    for head in HEADS:
        assert len(diagnostics[f"loss_history_train/{head}"]) == 3
        assert diagnostics[f"loss_history_val/{head}"] == []
        assert diagnostics[f"best_epoch/{head}"] == 2


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_new_record_roundtrips_through_json_and_the_reader(data) -> None:
    model = _fit(data, patience=3, max_epochs=10)
    params = {f"training/{k}": v for k, v in model.training_diagnostics().items()}
    record = RunRecord(group="t", params=params, metrics={"x": 1.0})
    restored = json.loads(json.dumps(record.to_json()))["params"]
    curves = training_curves(restored, HEADS[0])
    assert curves is not None
    assert len(curves.train) == len(curves.validation) == params[f"training/epochs_run/{HEADS[0]}"]
    assert curves.best_epoch == params[f"training/best_epoch/{HEADS[0]}"]


def test_old_records_without_history_are_still_read() -> None:
    """Committed records written before VH1 carry no history: reader returns None, no error."""
    path = ROOT / "artifacts" / "runs" / "arch_learning_curve_projected.jsonl"
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    assert records
    trained = [r for r in records if "training/epochs_run/energy_target" in r["params"]]
    assert trained
    for r in trained:
        assert training_curves(r["params"], "energy_target") is None
        assert not any("loss_history" in key for key in r["params"])
    # the existing flat-key readers of the analysis scripts are untouched
    assert all(isinstance(r["params"]["training/epochs_run/energy_target"], int) for r in trained)

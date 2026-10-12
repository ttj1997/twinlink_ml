"""Named preset for new studies (review D1/E3): ``DataSpec.new_study``."""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import pytest

from twinlink_ml.evaluation import NEW_STUDY_OPTIONS, DataSpec, generate_data
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

CONFIG = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_gab_placeholder.yaml"


def test_preset_sets_both_options_and_the_default_stays_legacy() -> None:
    spec = DataSpec.new_study()
    assert spec.future_actions == "planned"
    assert spec.solid_fraction_process == "independent"
    assert spec.is_new_study_preset
    legacy = DataSpec()
    assert legacy.future_actions == "realised"
    assert legacy.solid_fraction_process == "diurnal"
    assert not legacy.is_new_study_preset
    assert NEW_STUDY_OPTIONS == {
        "future_actions": "planned",
        "solid_fraction_process": "independent",
    }


def test_other_fields_pass_through_and_the_options_cannot_be_overridden() -> None:
    spec = DataSpec.new_study(n_train_max=8, horizon=12)
    assert (spec.n_train_max, spec.horizon) == (8, 12)
    assert spec.is_new_study_preset
    with pytest.raises(ValueError, match="fixed by the new-study preset"):
        DataSpec.new_study(future_actions="realised")
    with pytest.raises(ValueError, match="fixed by the new-study preset"):
        DataSpec.new_study(solid_fraction_process="diurnal")


def test_preset_data_carries_both_options_in_its_fingerprints() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG)
    small = {
        "sample_time_s": 25.0,
        "run_duration_s": 6000.0,
        "n_train_max": 4,
        "n_val": 2,
        "n_test": 2,
        "history_length": 8,
        "horizon": 6,
        "stride": 4,
    }
    legacy = generate_data(config, DataSpec(**small), master_seed=7)
    new = generate_data(config, DataSpec.new_study(**small), master_seed=7)
    assert new.fingerprints["window/future_actions"] == "planned"
    assert "window/future_actions" not in legacy.fingerprints
    assert (
        new.fingerprints["excitation/fingerprint"] != legacy.fingerprints["excitation/fingerprint"]
    )
    assert dataclasses.asdict(new.spec)["solid_fraction_process"] == "independent"

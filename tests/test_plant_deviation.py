"""Deviation measure of a plant from the simulator (study A1)."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.evaluation.experiments import DataSpec
from twinlink_ml.evaluation.plant_deviation import REFERENCE_SEED, plant_deviation
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.plant_variants import PLANT_VARIANTS
from twinlink_ml.utils.seeding import SeedBundle

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_gab_placeholder.yaml"
)


@pytest.fixture(scope="module")
def config():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return load_process_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def spec():
    return DataSpec.new_study(run_duration_s=3_000.0)


@pytest.fixture(scope="module")
def deviations(config, spec):
    return {
        name: plant_deviation(config, PLANT_VARIANTS[name], spec)
        for name in ("P0", "P1", "P2a", "P2b", "P3", "PP")
    }


def test_p0_has_no_deviation(deviations) -> None:
    p0 = deviations["P0"]
    assert all(v == 0.0 for v in p0.rmse_outputs.values())
    assert all(v == 0.0 for v in p0.rmse_measured.values())
    assert p0.summary_outputs == 0.0 and p0.summary_measured == 0.0


def test_p2b_deviates_in_the_measurement_only(deviations) -> None:
    p2b = deviations["P2b"]
    assert all(v == 0.0 for v in p2b.rmse_outputs.values())
    assert p2b.rmse_measured["bed_temperature_celsius"] > 0.0
    assert p2b.rmse_measured["product_moisture_kg_per_kg"] > 0.0
    assert p2b.rmse_measured["energy_target"] == 0.0
    assert p2b.summary_outputs == 0.0 and p2b.summary_measured > 0.0


def test_d50_is_unaffected_by_the_kinetics_and_the_parameters(deviations) -> None:
    for name in ("P1", "P2a", "PP"):
        assert deviations[name].rmse_outputs["d50_um"] == 0.0
        assert deviations[name].rmse_outputs["product_moisture_kg_per_kg"] > 0.0


def test_summary_is_the_mean_of_the_normalised_target_channels(deviations, spec) -> None:
    for name in ("P1", "P3", "PP"):
        d = deviations[name]
        assert d.summary_channels == spec.target_names
        assert "d50_um" not in d.summary_channels
        ratios = [d.rmse_outputs[c] / d.reference_std_outputs[c] for c in d.summary_channels]
        assert d.summary_outputs == pytest.approx(float(np.mean(ratios)), rel=1e-12)
        assert all(s > 0.0 for s in d.reference_std_outputs.values())


def test_units_and_record(deviations) -> None:
    d = deviations["P1"]
    assert d.units["bed_temperature_celsius"] == "degC"
    assert d.units["product_moisture_kg_per_kg"] == "kg/kg"
    assert d.n_runs == 8 and d.reference_seed == REFERENCE_SEED
    assert set(d.as_dict()) >= {"rmse_outputs", "rmse_measured", "summary_measured"}


def test_reference_seed_is_not_a_study_seed() -> None:
    derived = set()
    for master in range(1, 201):
        bundle = SeedBundle.from_master(master)
        derived |= {bundle.data_seed, bundle.noise_seed, bundle.model_seed}
    assert REFERENCE_SEED not in derived
    assert REFERENCE_SEED not in range(1, 201)

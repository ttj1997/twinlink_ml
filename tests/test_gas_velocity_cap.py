"""Per-run cap of the requested gas velocity (Befund 86, preset version 2).

Every run requests at most ``u_t`` at its INITIAL diameter; the projection stays on as a guard.
The decisive test is the re-simulation: a pool built with the cap has zero limited steps where the
legacy pool has many. The fingerprints of every existing configuration stay unchanged (pinned in
``tests/test_future_actions.py`` and ``tests/test_solid_fraction_process.py``); here the new option
is checked to enter only when set.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.evaluation import NEW_STUDY_OPTIONS, DataSpec, generate_data
from twinlink_ml.simulation import PlaceholderConfigWarning, ProcessConfig, load_process_config
from twinlink_ml.simulation.backend import NumpyBackend
from twinlink_ml.simulation.excitation import (
    MixedExcitationDesign,
    gas_velocity_cap_from_initial_diameter,
    load_excitation_config,
)
from twinlink_ml.simulation.fluidization import terminal_velocity_m_per_s
from twinlink_ml.utils.tracking import environment_fingerprint

ROOT = Path(__file__).resolve().parents[1]
GAS_INDEX = 1  # gas_velocity_m_per_s


@pytest.fixture(scope="module")
def config() -> ProcessConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base = load_process_config(ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml")
    return dataclasses.replace(
        base,
        numerics=dataclasses.replace(base.numerics, enforce_gas_velocity_projection=True),
    )


def _design(config: ProcessConfig, cap: str) -> MixedExcitationDesign:
    return MixedExcitationDesign(
        load_excitation_config(ROOT / "conf" / "excitation" / "broadband_derived.yaml"),
        config.actions,
        config.disturbances,
        sample_time_s=25.0,
        geometry=config.geometry,
        material=config.material,
        operation=config.operation,
        gas=config.gas,
        gas_velocity_cap=cap,  # type: ignore[arg-type]
    )


def test_caps_are_terminal_velocities_with_worst_case_conditions(config: ProcessConfig) -> None:
    diameters = np.array([250e-6, 400e-6, 800e-6])
    caps = gas_velocity_cap_from_initial_diameter(
        diameters,
        gas=config.gas,
        material=config.material,
        actions=config.actions,
        disturbances=config.disturbances,
    )
    assert np.all(np.diff(caps) > 0.0), "u_t grows with the diameter"
    assert caps[0] >= config.actions.gas_velocity_m_per_s.minimum
    # never above u_t at any single condition of the a-priori range
    backend = NumpyBackend()
    nominal = terminal_velocity_m_per_s(
        backend, config.gas, config.material, diameters, np.full(3, 62.0), np.full(3, 0.01)
    )
    assert np.all(caps <= nominal + 1e-12)
    # smallest initial diameter of the design: about 1.2 m/s (Befund 85: 1.23)
    assert 1.0 < caps[0] < 1.3


def test_schedule_respects_the_cap_and_changes_nothing_else(config: ProcessConfig) -> None:
    capped = _design(config, "initial_diameter").build(32, seed=5)
    plain = _design(config, "none").build(32, seed=5)
    caps = np.array(capped.metadata["gas_velocity_cap_m_per_s"])
    assert caps.shape == (32,)
    assert np.all(capped.actions[:, :, GAS_INDEX] <= caps[:, None] + 1e-12)
    assert np.any(plain.actions[:, :, GAS_INDEX] > caps[:, None]), "the test must be able to fail"
    others = [i for i in range(capped.actions.shape[-1]) if i != GAS_INDEX]
    np.testing.assert_array_equal(capped.actions[..., others], plain.actions[..., others])
    np.testing.assert_array_equal(capped.disturbances, plain.disturbances)
    np.testing.assert_array_equal(capped.initial_states, plain.initial_states)
    # runs that never exceed their cap are untouched
    untouched = np.all(plain.actions[:, :, GAS_INDEX] <= caps[:, None], axis=1)
    np.testing.assert_array_equal(
        capped.actions[untouched, :, GAS_INDEX], plain.actions[untouched, :, GAS_INDEX]
    )
    # the rate limit of the signal survives the minimum with a constant
    rate = config.actions.gas_velocity_m_per_s.max_rate_per_s * 25.0
    assert np.abs(np.diff(capped.actions[:, :, GAS_INDEX], axis=1)).max() <= rate + 1e-9


def test_fingerprint_changes_only_when_the_cap_is_set(config: ProcessConfig) -> None:
    plain = _design(config, "none").build(8, seed=5).metadata["fingerprint"]
    capped = _design(config, "initial_diameter").build(8, seed=5).metadata["fingerprint"]
    assert plain != capped
    # same as a design built without the new arguments at all (every existing schedule)
    legacy = MixedExcitationDesign(
        load_excitation_config(ROOT / "conf" / "excitation" / "broadband_derived.yaml"),
        config.actions,
        config.disturbances,
        sample_time_s=25.0,
        geometry=config.geometry,
        material=config.material,
        operation=config.operation,
    )
    assert legacy.build(8, seed=5).metadata["fingerprint"] == plain


def test_cap_needs_the_gas_config(config: ProcessConfig) -> None:
    with pytest.raises(ValueError, match="GasConfig"):
        MixedExcitationDesign(
            load_excitation_config(ROOT / "conf" / "excitation" / "broadband_derived.yaml"),
            config.actions,
            config.disturbances,
            sample_time_s=25.0,
            geometry=config.geometry,
            material=config.material,
            operation=config.operation,
            gas_velocity_cap="initial_diameter",
        )


SMALL = {
    "sample_time_s": 25.0,
    "run_duration_s": 28_800.0,
    "n_train_max": 8,
    "n_val": 4,
    "n_test": 4,
    "history_length": 8,
    "horizon": 6,
    "stride": 8,
}


def test_resimulated_pool_with_the_cap_has_zero_limited_steps(config: ProcessConfig) -> None:
    legacy = generate_data(config, DataSpec(**SMALL), master_seed=1)
    preset = generate_data(config, DataSpec.new_study(**SMALL), master_seed=1)
    assert legacy.projection_guard["projection_active"] == 1
    assert legacy.projection_guard["projected_steps"] > 0, "the legacy pool must be limited"
    guard = preset.projection_guard
    assert guard["projection_active"] == 1, "the projection stays on as a guard"
    assert guard["projected_steps"] == 0 and guard["runs_with_projection"] == 0
    assert guard["total_steps"] == 16 * 1152


def test_constant_run_holdout_with_the_cap_has_zero_limited_steps(config: ProcessConfig) -> None:
    spec = dataclasses.replace(
        DataSpec.new_study(**SMALL), n_train_max=2, n_val=2, constant_run_fraction=1.0
    )
    legacy_spec = dataclasses.replace(
        DataSpec(**SMALL), n_train_max=2, n_val=2, constant_run_fraction=1.0
    )
    holdout = generate_data(config, spec, master_seed=999)
    legacy = generate_data(config, legacy_spec, master_seed=999)
    assert legacy.projection_guard["projected_steps"] > 0
    assert holdout.projection_guard["projected_steps"] == 0


def test_fingerprints_carry_the_preset_version_only_for_the_preset(config: ProcessConfig) -> None:
    legacy = generate_data(config, DataSpec(**SMALL), master_seed=1)
    preset = generate_data(config, DataSpec.new_study(**SMALL), master_seed=1)
    assert "preset/new_study_version" not in legacy.fingerprints
    assert preset.fingerprints["preset/new_study_version"] == "2"
    assert (
        preset.fingerprints["excitation/fingerprint"]
        != legacy.fingerprints["excitation/fingerprint"]
    )
    assert (
        preset.spec.as_parameters()["data/new_study_version"]
        == NEW_STUDY_OPTIONS["new_study_version"]
    )
    assert legacy.spec.as_parameters()["data/new_study_version"] is None
    assert legacy.spec.as_parameters()["data/gas_velocity_cap"] == "none"


def test_every_provenance_block_names_the_preset_version() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert environment_fingerprint()["preset/new_study_version"] == 2

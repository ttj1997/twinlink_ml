r"""Review E3 (2026-10-09): own disturbance process for ``suspension_solid_fraction``.

Until Befund 81 every disturbance channel without "humidity" in its name got the diurnal
cycle in phase with ``T_amb`` -- including the solid fraction of the suspension. Inside a run
that makes heating duty and water load nearly collinear (review: median correlation 0.71).
``DiurnalDisturbanceDesign(solid_fraction_process="independent")`` (``DataSpec.
solid_fraction_process``) gives the channel its own Ornstein-Uhlenbeck process instead. The
default stays ``"diurnal"`` so that every existing schedule keeps its fingerprint.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.evaluation import DataSpec, generate_data
from twinlink_ml.simulation import PlaceholderConfigWarning, ProcessConfig, load_process_config
from twinlink_ml.simulation.excitation import (
    DiurnalDisturbanceDesign,
    MixedExcitationDesign,
    PiecewiseConstantDisturbanceDesign,
    load_excitation_config,
)

ROOT = Path(__file__).resolve().parents[1]
N_RUNS = 64
T_AMB, RELATIVE_HUMIDITY, SOLID = 0, 1, 2


@pytest.fixture(scope="module")
def config() -> ProcessConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return load_process_config(ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml")


def _design(config: ProcessConfig, **options: object) -> MixedExcitationDesign:
    return MixedExcitationDesign(
        load_excitation_config(ROOT / "conf" / "excitation" / "broadband_derived.yaml"),
        config.actions,
        config.disturbances,
        sample_time_s=25.0,
        geometry=config.geometry,
        material=config.material,
        operation=config.operation,
        disturbance_design=DiurnalDisturbanceDesign(**options),  # type: ignore[arg-type]
    )


@pytest.fixture(scope="module")
def diurnal(config: ProcessConfig):  # type: ignore[no-untyped-def]
    return _design(config).build(N_RUNS, seed=11)


@pytest.fixture(scope="module")
def independent(config: ProcessConfig):  # type: ignore[no-untyped-def]
    return _design(config, solid_fraction_process="independent").build(N_RUNS, seed=11)


def _within_run_correlation(schedule, channel: int) -> np.ndarray:  # type: ignore[no-untyped-def]
    d = schedule.disturbances
    return np.array([np.corrcoef(d[r, :, T_AMB], d[r, :, channel])[0, 1] for r in range(N_RUNS)])


def test_channel_order_is_as_assumed(config: ProcessConfig) -> None:
    assert config.disturbances.names[SOLID] == "suspension_solid_fraction"
    assert config.disturbances.names[T_AMB] == "ambient_temperature_celsius"
    assert config.disturbances.names[RELATIVE_HUMIDITY] == "ambient_relative_humidity"


def test_legacy_mode_has_the_diurnal_coupling(diurnal) -> None:  # type: ignore[no-untyped-def]
    """Documents the confound that the option removes."""
    assert np.median(_within_run_correlation(diurnal, SOLID)) > 0.4


def test_independent_removes_the_coupling_to_ambient_temperature(independent) -> None:  # type: ignore[no-untyped-def]
    correlation = _within_run_correlation(independent, SOLID)
    assert abs(float(np.median(correlation))) < 0.15
    assert abs(float(np.mean(correlation))) < 0.15
    assert float(np.median(np.abs(correlation))) < 0.4  # slow independent processes still
    # correlate by chance over 8 h, but not systematically


def test_temperature_and_humidity_are_bitwise_unchanged(diurnal, independent) -> None:  # type: ignore[no-untyped-def]
    np.testing.assert_array_equal(
        diurnal.disturbances[..., [T_AMB, RELATIVE_HUMIDITY]],
        independent.disturbances[..., [T_AMB, RELATIVE_HUMIDITY]],
    )
    np.testing.assert_array_equal(diurnal.actions, independent.actions)
    np.testing.assert_array_equal(diurnal.initial_states, independent.initial_states)


def test_within_run_variability_matches_the_legacy_channel(diurnal, independent) -> None:  # type: ignore[no-untyped-def]
    legacy = diurnal.disturbances[..., SOLID].std(axis=1).mean()
    own = independent.disturbances[..., SOLID].std(axis=1).mean()
    assert 0.8 < own / legacy < 1.25


def test_solid_fraction_stays_inside_its_range(config: ProcessConfig, independent) -> None:  # type: ignore[no-untyped-def]
    bounds = config.disturbances.ranges[SOLID]
    values = independent.disturbances[..., SOLID]
    assert values.min() >= bounds.minimum - 1e-12
    assert values.max() <= bounds.maximum + 1e-12
    assert np.ptp(values, axis=1).min() > 0.0, "the channel must vary inside every run"


# ------------------------------------------------- fingerprints of old configurations


def test_default_design_adds_nothing_to_the_fingerprint(config: ProcessConfig, diurnal) -> None:  # type: ignore[no-untyped-def]
    assert DiurnalDisturbanceDesign().fingerprint_options() == {}
    assert DiurnalDisturbanceDesign(solid_fraction_process="diurnal").fingerprint_options() == {}
    assert PiecewiseConstantDisturbanceDesign().fingerprint_options() == {}
    # explicit "diurnal" and the implicit default give the identical schedule fingerprint
    default_design = MixedExcitationDesign(
        load_excitation_config(ROOT / "conf" / "excitation" / "broadband_derived.yaml"),
        config.actions,
        config.disturbances,
        sample_time_s=25.0,
        geometry=config.geometry,
        material=config.material,
        operation=config.operation,
    )
    default = default_design.build(N_RUNS, seed=11)
    assert default.metadata["fingerprint"] == diurnal.metadata["fingerprint"]
    np.testing.assert_array_equal(default.disturbances, diurnal.disturbances)


def test_independent_changes_the_schedule_fingerprint(diurnal, independent) -> None:  # type: ignore[no-untyped-def]
    assert independent.metadata["fingerprint"] != diurnal.metadata["fingerprint"]


def test_std_fraction_enters_the_fingerprint_when_independent(config: ProcessConfig) -> None:
    a = _design(config, solid_fraction_process="independent").build(4, seed=3)
    b = _design(
        config, solid_fraction_process="independent", solid_fraction_std_fraction=0.2
    ).build(4, seed=3)
    assert a.metadata["fingerprint"] != b.metadata["fingerprint"]
    # irrelevant while the channel is diurnal: the value must not move the fingerprint
    c = _design(config, solid_fraction_std_fraction=0.2).build(4, seed=3)
    d = _design(config).build(4, seed=3)
    assert c.metadata["fingerprint"] == d.metadata["fingerprint"]


def test_invalid_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="solid_fraction_process"):
        DiurnalDisturbanceDesign(solid_fraction_process="weekly")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="solid_fraction_std_fraction"):
        DiurnalDisturbanceDesign(solid_fraction_std_fraction=-0.1)


# ------------------------------------------------------------------- DataSpec path

SPEC = DataSpec(
    sample_time_s=25.0,
    run_duration_s=6000.0,
    n_train_max=4,
    n_val=2,
    n_test=2,
    history_length=8,
    horizon=6,
    stride=4,
)


def test_dataspec_default_is_diurnal_and_keeps_the_legacy_fingerprints(
    config: ProcessConfig,
) -> None:
    assert SPEC.solid_fraction_process == "diurnal"
    data = generate_data(config, SPEC, master_seed=7)
    # recorded at cd7b562 (before the option existed), same value as in test_future_actions
    assert data.fingerprints["excitation/fingerprint"] == "38f9542b"


def test_dataspec_independent_changes_the_excitation_fingerprint_not_the_process_one(
    config: ProcessConfig,
) -> None:
    legacy = generate_data(config, SPEC, master_seed=7)
    new = generate_data(
        config, dataclasses.replace(SPEC, solid_fraction_process="independent"), master_seed=7
    )
    assert (
        new.fingerprints["excitation/fingerprint"] != legacy.fingerprints["excitation/fingerprint"]
    )
    # split/digest moves with it: the run ids embed the excitation fingerprint
    for key in ("process/fingerprint", "measurement/digest"):
        assert new.fingerprints[key] == legacy.fingerprints[key]
    np.testing.assert_array_equal(
        new.trajectory.disturbances[..., [T_AMB, RELATIVE_HUMIDITY]],
        legacy.trajectory.disturbances[..., [T_AMB, RELATIVE_HUMIDITY]],
    )
    assert not np.array_equal(
        new.trajectory.disturbances[..., SOLID], legacy.trajectory.disturbances[..., SOLID]
    )

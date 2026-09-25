"""Tests des Anregungsdesigns.

Zwei Tests tragen hier die eigentliche Aussage: ``test_sobol_design_is_nested`` und
``test_latin_hypercube_is_not_nested``. Sie belegen den Grund für die Abweichung vom
Arbeitsplan (Sobol statt LHS) und würden sofort auffallen, wenn jemand den Sampler
zurückstellt.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from conftest import build_design
from twinlink_ml.simulation import (
    ConstantGenerator,
    DiurnalDisturbanceDesign,
    ExcitationConfig,
    GreyBoxFluidBedGranulator,
    HoldTimeRange,
    InitialConditionDesign,
    MixedExcitationDesign,
    PiecewiseConstantDisturbanceDesign,
    PrbsGenerator,
    ProcessConfig,
    RampGenerator,
    StepSequenceGenerator,
    apply_rate_limit,
    constraint_violation_summary,
    excitation_report,
    quasi_random_points,
)
from twinlink_ml.simulation.excitation import _LFSR_TAPS


@pytest.mark.parametrize("kind", ["step", "prbs", "ramp", "constant"])
def test_generators_stay_in_unit_interval(kind: str) -> None:
    generators = {
        "step": StepSequenceGenerator(HoldTimeRange(30.0, 300.0)),
        "prbs": PrbsGenerator(8, 60.0),
        "ramp": RampGenerator(HoldTimeRange(200.0, 900.0)),
        "constant": ConstantGenerator(),
    }
    rng = np.random.default_rng(0)
    for amplitude in (0.15, 0.5, 1.0):
        signal = generators[kind].generate(600, 5.0, rng, amplitude)
        assert signal.shape == (600,)
        assert np.all(signal >= -1e-12)
        assert np.all(signal <= 1.0 + 1e-12)


@pytest.mark.parametrize("kind", ["step", "prbs", "ramp"])
def test_generators_respect_amplitude(kind: str) -> None:
    """Kleine Amplitudenstufen müssen tatsächlich klein sein.

    Das ist die Voraussetzung dafür, dass schnelle Anregung trotz Ratenbegrenzung möglich
    ist -- siehe ``feasible_switch_time_over_tau_fast`` im Report.
    """
    generators = {
        "step": StepSequenceGenerator(HoldTimeRange(30.0, 120.0)),
        "prbs": PrbsGenerator(7, 30.0),
        "ramp": RampGenerator(HoldTimeRange(100.0, 300.0)),
    }
    rng = np.random.default_rng(1)
    spans = []
    for amplitude in (0.15, 1.0):
        widths = [
            float(np.ptp(generators[kind].generate(2000, 5.0, rng, amplitude))) for _ in range(20)
        ]
        spans.append(float(np.mean(widths)))
    assert spans[0] < 0.35
    assert spans[1] > 2.0 * spans[0]


def test_step_sequence_hold_times_within_range() -> None:
    hold = HoldTimeRange(60.0, 120.0)
    generator = StepSequenceGenerator(hold)
    rng = np.random.default_rng(2)
    signal = generator.generate(4000, 5.0, rng, 1.0)
    changes = np.flatnonzero(np.abs(np.diff(signal)) > 1e-12) + 1
    intervals = np.diff(changes) * 5.0
    assert intervals.size > 5
    assert intervals.min() >= hold.minimum_s - 1e-9
    assert intervals.max() <= hold.maximum_s + 1e-9


def test_prbs_is_balanced_and_has_full_period() -> None:
    """m-Sequenz: gleich viele Nullen und Einsen (bis auf eine) und Periode 2^n - 1."""
    generator = PrbsGenerator(7, 5.0)
    rng = np.random.default_rng(3)
    period = 2**7 - 1
    signal = generator.generate(period, 5.0, rng, 1.0)
    levels = np.unique(signal)
    assert levels.size == 2
    count_high = int(np.sum(signal == levels[1]))
    assert abs(count_high - period // 2) <= 1

    # Autokorrelation einer m-Sequenz: scharfe Spitze bei Verschiebung 0, sonst flach
    binary = np.where(signal == levels[1], 1.0, -1.0)
    autocorrelation = np.array(
        [float(np.mean(binary * np.roll(binary, shift))) for shift in range(1, period)]
    )
    assert abs(autocorrelation).max() < 0.2


@pytest.mark.parametrize("register_length", sorted(_LFSR_TAPS))
def test_prbs_registers_produce_nondegenerate_sequences(register_length: int) -> None:
    generator = PrbsGenerator(register_length, 5.0)
    rng = np.random.default_rng(register_length)
    signal = generator.generate(2 ** (register_length + 1), 5.0, rng, 1.0)
    assert np.unique(signal).size == 2


def test_rate_limit_is_respected_and_idempotent() -> None:
    rng = np.random.default_rng(4)
    signal = rng.uniform(60.0, 120.0, size=(6, 500))
    limited = apply_rate_limit(signal, max_rate_per_s=0.5, sample_time_s=5.0)
    maximum_step = 0.5 * 5.0
    assert np.all(np.abs(np.diff(limited, axis=1)) <= maximum_step + 1e-9)
    again = apply_rate_limit(limited, max_rate_per_s=0.5, sample_time_s=5.0)
    np.testing.assert_allclose(again, limited, rtol=0.0, atol=1e-12)


def test_rate_limit_preserves_bounds() -> None:
    """Die Begrenzung darf den Stellgrößenbereich nicht verlassen (konvexe Operation)."""
    rng = np.random.default_rng(5)
    signal = rng.uniform(0.8, 2.5, size=(4, 300))
    limited = apply_rate_limit(signal, max_rate_per_s=0.05, sample_time_s=5.0)
    assert limited.min() >= 0.8 - 1e-12
    assert limited.max() <= 2.5 + 1e-12


def test_sobol_design_is_nested() -> None:
    """Der Grund für Sobol als Standard: Teilmengen bleiben Folgen niedriger Diskrepanz."""
    full = quasi_random_points(64, 5, sampler="sobol", seed=7)
    small = quasi_random_points(8, 5, sampler="sobol", seed=7)
    np.testing.assert_allclose(full[:8], small, rtol=0.0, atol=0.0)


def test_latin_hypercube_is_not_nested() -> None:
    """Gegenprobe: bei LHS ändert sich die Abdeckung mit der Stützstelle der Lernkurve."""
    full = quasi_random_points(64, 5, sampler="lhs", seed=7)
    small = quasi_random_points(8, 5, sampler="lhs", seed=7)
    assert not np.allclose(full[:8], small)


def test_schedule_subset_is_prefix(config: ProcessConfig) -> None:
    design = build_design(config, n_runs_hint=16)
    schedule = design.build(16, seed=0)
    subset = schedule.subset(4)
    assert subset.n_runs == 4
    assert subset.run_id == schedule.run_id[:4]
    np.testing.assert_array_equal(subset.actions, schedule.actions[:4])
    assert subset.metadata["subset_of"] == 16


def test_design_is_deterministic(config: ProcessConfig) -> None:
    design = build_design(config)
    first = design.build(8, seed=11)
    second = design.build(8, seed=11)
    np.testing.assert_array_equal(first.actions, second.actions)
    np.testing.assert_array_equal(first.disturbances, second.disturbances)
    assert first.run_id == second.run_id

    third = design.build(8, seed=12)
    assert not np.array_equal(first.actions, third.actions)
    assert third.run_id != first.run_id, "Lauf-IDs müssen den Seed unterscheiden"


def test_actions_stay_within_bounds_and_rate_limits(config: ProcessConfig) -> None:
    design = build_design(config)
    schedule = design.build(8, seed=0)
    for index, bound in enumerate(config.actions.bounds):
        values = schedule.actions[:, :, index]
        assert values.min() >= bound.minimum - 1e-9
        assert values.max() <= bound.maximum + 1e-9
        rate = np.abs(np.diff(values, axis=1)) / config.numerics.sample_time_s
        assert rate.max() <= bound.max_rate_per_s + 1e-9


def test_rate_limits_can_be_disabled(config: ProcessConfig) -> None:
    """Ohne Begrenzung müssen die Sprünge tatsächlich größer sein -- sonst wirkt nichts."""
    unlimited = dataclasses.replace(_excitation_config(), apply_rate_limits=False)
    design = build_design(config, excitation=unlimited)
    schedule = design.build(8, seed=0)
    bound = config.actions.bounds[0]
    rate = np.abs(np.diff(schedule.actions[:, :, 0], axis=1)) / config.numerics.sample_time_s
    assert rate.max() > 2.0 * bound.max_rate_per_s


def test_disturbances_stay_within_ranges(config: ProcessConfig) -> None:
    design = build_design(config)
    schedule = design.build(8, seed=0)
    for index, span in enumerate(config.disturbances.ranges):
        values = schedule.disturbances[:, :, index]
        assert values.min() >= span.minimum - 1e-9
        assert values.max() <= span.maximum + 1e-9


def test_piecewise_constant_disturbances_are_constant_within_run(config: ProcessConfig) -> None:
    design = build_design(config, disturbance_design=PiecewiseConstantDisturbanceDesign())
    schedule = design.build(8, seed=0)
    assert np.all(np.abs(np.diff(schedule.disturbances, axis=1)) < 1e-12)
    assert np.ptp(schedule.disturbances[:, 0, 0]) > 1.0, "zwischen den Läufen muss gestreut werden"


def test_diurnal_design_produces_partial_decorrelation(config: ProcessConfig) -> None:
    """Tagesgang koppelt Temperatur und Feuchte -- der OU-Anteil muss die Kopplung brechen."""
    design = build_design(
        config,
        disturbance_design=DiurnalDisturbanceDesign(drift_amplitude_fraction=0.10),
    )
    schedule = design.build(16, seed=0)
    temperature = schedule.disturbances[:, :, 0].ravel()
    humidity = schedule.disturbances[:, :, 1].ravel()
    correlation = float(np.corrcoef(temperature, humidity)[0, 1])
    assert -0.95 < correlation < 0.2, f"Korrelation {correlation:.2f} unbrauchbar"


def test_initial_conditions_are_spread(config: ProcessConfig) -> None:
    design = build_design(config)
    schedule = design.build(16, seed=0)
    assert schedule.initial_states is not None
    diameters = schedule.initial_states[:, 2]
    assert diameters.max() / diameters.min() > 2.0
    temperatures = schedule.initial_states[:, 3]
    assert np.ptp(temperatures) > 20.0


def test_initial_conditions_can_be_disabled(config: ProcessConfig) -> None:
    excitation = dataclasses.replace(_excitation_config(), randomise_initial_conditions=False)
    design = build_design(config, excitation=excitation, initial_condition_design=False)
    schedule = design.build(4, seed=0)
    assert schedule.initial_states is None


def test_constant_runs_are_present(config: ProcessConfig) -> None:
    excitation = dataclasses.replace(_excitation_config(), constant_run_fraction=0.25)
    design = build_design(config, excitation=excitation)
    schedule = design.build(8, seed=0)
    constant_runs = [
        index for index, kinds in enumerate(schedule.signal_kinds) if set(kinds) == {"constant"}
    ]
    assert len(constant_runs) == 2
    for index in constant_runs:
        assert np.all(np.abs(np.diff(schedule.actions[index], axis=0)) < 1e-12)


def test_config_from_time_constants_respects_fast_mode() -> None:
    config = ExcitationConfig.from_time_constants(
        tau_fast_s=9.5, tau_slow_s=16_000.0, sample_time_s=5.0
    )
    assert config.hold_time.minimum_s >= 2.0 * 9.5
    assert config.hold_time.minimum_s >= 4.0 * 5.0
    assert config.hold_time.maximum_s > config.hold_time.minimum_s
    # Periode muss die Laufdauer überschreiten -- Wiederholung innerhalb eines Laufs
    # verschwendet Abtastpunkte. Die langsame Mode ist nicht stellgrößenanregbar.
    assert config.prbs_period_s >= config.run_duration_s


def test_invalid_excitation_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="prbs_register_length"):
        ExcitationConfig(prbs_register_length=3)
    with pytest.raises(ValueError, match="amplitude_levels"):
        ExcitationConfig(amplitude_levels=(0.0, 1.0))
    with pytest.raises(ValueError, match="Verweilzeitbereich"):
        HoldTimeRange(100.0, 50.0)


def test_constant_run_fraction_upper_bound_is_inclusive() -> None:
    """1.0 ist der reine Konstantlauf-Auswertungsdatensatz und muss zulässig sein."""
    ExcitationConfig(constant_run_fraction=1.0)
    with pytest.raises(ValueError, match=r"constant_run_fraction muss in \[0, 1\] liegen"):
        ExcitationConfig(constant_run_fraction=1.0001)


def test_constant_run_fraction_one_yields_all_constant_runs(config: ProcessConfig) -> None:
    excitation = dataclasses.replace(_excitation_config(), constant_run_fraction=1.0)
    design = build_design(config, excitation=excitation)
    schedule = design.build(8, seed=0)
    assert all(set(kinds) == {"constant"} for kinds in schedule.signal_kinds)
    for run in range(schedule.n_runs):
        assert np.all(np.abs(np.diff(schedule.actions[run], axis=0)) < 1e-12)


def test_constant_run_fraction_mask_unchanged_at_ten_percent() -> None:
    """Regression: die Maske bei fraction=0.10 darf sich nicht ändern.

    Die geöffnete Obergrenze betrifft nur ``ExcitationConfig.__post_init__``;
    ``_constant_run_flags`` selbst ist unverändert.
    """
    from twinlink_ml.simulation.excitation import _constant_run_flags

    flags = _constant_run_flags(32, 0.10)
    expected = np.mod(np.arange(32, dtype=np.float64) * 0.10, 1.0) < 0.10 - 1e-12
    np.testing.assert_array_equal(flags, expected)
    assert int(np.sum(flags)) == 4


def test_schedule_drives_simulator_without_producing_invalid_states(
    config: ProcessConfig,
) -> None:
    """Ende-zu-Ende: Plan -> Rollout, endliche Zustände, Constraints eingehalten."""
    short = dataclasses.replace(_excitation_config(), run_duration_s=3000.0)
    design = build_design(config, excitation=short)
    schedule = design.build(8, seed=0)
    simulator = GreyBoxFluidBedGranulator(config)
    assert schedule.initial_states is not None
    initial = simulator.state_from_physical(
        schedule.initial_states[:, 0],
        schedule.initial_states[:, 1],
        schedule.initial_states[:, 2],
        schedule.initial_states[:, 3],
    )
    trajectory = simulator.rollout(
        schedule.actions,
        schedule.disturbances,
        run_id=schedule.run_id,
        seed=0,
        initial_state=initial,
    )
    assert np.all(np.isfinite(trajectory.states))
    assert np.all(np.isfinite(trajectory.outputs))
    assert np.all(
        trajectory.diagnostics["outlet_humidity_kg_per_kg"]
        <= trajectory.diagnostics["saturation_humidity_kg_per_kg"] + 1e-9
    )
    assert np.all(
        trajectory.diagnostics["particle_moisture_kg_per_kg"]
        >= config.material.equilibrium_moisture_kg_per_kg - 1e-9
    )
    assert trajectory.run_id == schedule.run_id


def test_initial_state_round_trip(config: ProcessConfig) -> None:
    """state_from_physical und die Diagnose müssen denselben Durchmesser ergeben."""
    simulator = GreyBoxFluidBedGranulator(config)
    diameters = np.array([300e-6, 600e-6, 900e-6])
    state = simulator.state_from_physical(
        np.full(3, 2.0), np.full(3, 0.006), diameters, np.full(3, 55.0)
    )
    actions = np.repeat(simulator.nominal_action(3)[:, None, :], 2, axis=1)
    disturbances = np.repeat(simulator.nominal_disturbance(3)[:, None, :], 2, axis=1)
    trajectory = simulator.rollout(actions, disturbances, seed=0, initial_state=state)
    np.testing.assert_allclose(trajectory.diagnostics["d50_m"][:, 0], diameters, rtol=2e-3)


def _excitation_config() -> ExcitationConfig:
    return ExcitationConfig(
        run_duration_s=6000.0,
        hold_time=HoldTimeRange(60.0, 900.0),
        prbs_clock_s=60.0,
        prbs_register_length=8,
        ramp_segment_time=HoldTimeRange(300.0, 1800.0),
        constant_run_fraction=0.0,
    )


def test_design_reports_metadata(config: ProcessConfig) -> None:
    design = build_design(config)
    schedule = design.build(4, seed=0)
    for key in ("design", "fingerprint", "sampler", "prbs_period_s", "rate_limited"):
        assert key in schedule.metadata
    assert isinstance(design, MixedExcitationDesign)
    assert isinstance(InitialConditionDesign().n_dimensions, int)


# ------------------------------------------------------ Anregungsdiagnose


def test_report_detects_rate_limit_violation(config: ProcessConfig) -> None:
    """Ohne Begrenzung muss der Report die Verletzung melden, nicht verschweigen."""
    unlimited = dataclasses.replace(_excitation_config(), apply_rate_limits=False)
    design = build_design(config, excitation=unlimited)
    schedule = design.build(8, seed=0)
    report = excitation_report(
        schedule,
        config.actions,
        config.disturbances,
        sample_time_s=config.numerics.sample_time_s,
        tau_fast_s=9.5,
        tau_slow_s=16_000.0,
        excitation_config=unlimited,
    )
    assert report.channels[0].rate_limit_utilisation > 1.0
    assert any("Ratenbegrenzung verletzt" in message for message in report.warnings)


def test_report_detects_collinear_channels(config: ProcessConfig) -> None:
    """Kollinearität muss auffallen -- sonst sind die Monotonie-Constraints nicht prüfbar."""
    design = build_design(config)
    schedule = design.build(8, seed=0)
    collinear = schedule.actions.copy()
    bounds = config.actions.bounds
    normalised = (collinear[:, :, 0] - bounds[0].minimum) / bounds[0].span
    collinear[:, :, 1] = bounds[1].minimum + normalised * bounds[1].span
    duplicated = dataclasses.replace(schedule, actions=collinear)
    report = excitation_report(
        duplicated,
        config.actions,
        config.disturbances,
        sample_time_s=config.numerics.sample_time_s,
        tau_fast_s=9.5,
        tau_slow_s=16_000.0,
    )
    assert report.max_absolute_action_correlation > 0.99
    assert any("korrelieren" in message for message in report.warnings)


def test_report_shows_rate_limited_bandwidth_loss(config: ProcessConfig) -> None:
    """Die realisierte Bandbreite muss unter der Eckfrequenz der schnellen Mode liegen.

    Das ist die quantitative Fassung des Befunds: mit den Platzhalter-Ratengrenzen ist die
    schnellste Prozessmode nicht anregbar.
    """
    excitation = _excitation_config()
    design = build_design(config, excitation=excitation)
    schedule = design.build(8, seed=0)
    report = excitation_report(
        schedule,
        config.actions,
        config.disturbances,
        sample_time_s=config.numerics.sample_time_s,
        tau_fast_s=9.5,
        tau_slow_s=16_000.0,
        excitation_config=excitation,
    )
    inlet = report.channels[0]
    assert inlet.bandwidth_over_fast_corner < 1.0
    assert inlet.full_amplitude_switch_time_over_tau_fast > 10.0
    assert 0.0 < inlet.spectral_flatness < 1.0


def test_report_coverage_is_high_for_full_amplitude_design(config: ProcessConfig) -> None:
    design = build_design(config)
    schedule = design.build(16, seed=0)
    report = excitation_report(
        schedule,
        config.actions,
        config.disturbances,
        sample_time_s=config.numerics.sample_time_s,
        tau_fast_s=9.5,
        tau_slow_s=16_000.0,
    )
    for channel in report.channels:
        assert channel.range_utilisation > 0.8
    for coverage in report.disturbance_coverage.values():
        assert coverage > 0.8


def test_constraint_violation_summary_returns_fractions(config: ProcessConfig) -> None:
    short = dataclasses.replace(_excitation_config(), run_duration_s=3000.0)
    design = build_design(config, excitation=short)
    schedule = design.build(8, seed=0)
    simulator = GreyBoxFluidBedGranulator(config)
    assert schedule.initial_states is not None
    initial = simulator.state_from_physical(
        schedule.initial_states[:, 0],
        schedule.initial_states[:, 1],
        schedule.initial_states[:, 2],
        schedule.initial_states[:, 3],
    )
    trajectory = simulator.rollout(
        schedule.actions, schedule.disturbances, seed=0, initial_state=initial
    )
    summary = constraint_violation_summary(trajectory)
    assert set(summary) >= {"below_minimum_fluidization", "gas_velocity_above_terminal"}
    assert all(0.0 <= value <= 1.0 for value in summary.values())


def test_excitation_config_roundtrip(tmp_path) -> None:
    """Der aufgelöste Anregungsstand muss serialisierbar sein."""
    from twinlink_ml.simulation import dump_excitation_config, load_excitation_config

    original = ExcitationConfig.from_time_constants(
        tau_fast_s=9.5, tau_slow_s=16_000.0, sample_time_s=5.0, run_duration_s=28_800.0
    )
    path = tmp_path / "excitation.yaml"
    dump_excitation_config(original, path)
    assert load_excitation_config(path) == original


def test_shipped_excitation_config_loads() -> None:
    from pathlib import Path as _Path

    from twinlink_ml.simulation import load_excitation_config

    path = _Path(__file__).resolve().parents[1] / "conf" / "excitation" / "broadband_derived.yaml"
    config = load_excitation_config(path)
    assert config.prbs_period_s >= config.run_duration_s

"""Tests of the dynamic headroom study (``twinlink_ml.headroom``, docs/24_dynamic_headroom.md).

Required by the study request: the path condition is checked (at every truth sample, not only at
the end or at optimiser nodes), R2 equals R1 on a scenario without disturbance, the margin
bisection converges. Further: the symbolic model is the truth model's right-hand side, the
optimiser's interval model tracks the truth, the scenarios are deterministic and inside their
bands.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

casadi = pytest.importorskip("casadi")

from twinlink_ml.headroom.config import (  # noqa: E402
    HeadroomStudyConfig,
    load_headroom_config,
    study_process_config,
)
from twinlink_ml.headroom.controllers import (  # noqa: E402
    DynamicOptimizer,
    SteadyStateOptimizer,
    nominal_disturbance,
)
from twinlink_ml.headroom.margin import (  # noqa: E402
    bisect_margin,
    bisect_margin_k_of_n,
    bisect_margin_over_scenarios,
)
from twinlink_ml.headroom.scenarios import Scenario, build_scenarios  # noqa: E402
from twinlink_ml.headroom.study import Solvers, run_controller  # noqa: E402
from twinlink_ml.headroom.symbolic import (  # noqa: E402
    interval_function,
    ramp_mean,
    steady_particle_count,
    step_means,
    symbolic_derivative,
)
from twinlink_ml.headroom.truth import TruthPlant  # noqa: E402
from twinlink_ml.simulation import GreyBoxFluidBedGranulator, ProcessConfig  # noqa: E402


@pytest.fixture(scope="module")
def study() -> HeadroomStudyConfig:
    return load_headroom_config()


@pytest.fixture(scope="module")
def process(study: HeadroomStudyConfig) -> ProcessConfig:
    return study_process_config(study)


def _constant_scenario(
    study: HeadroomStudyConfig, process: ProcessConfig, hours: float, *, offset: float = 0.0
) -> Scenario:
    n = round(hours * 3600.0 / study.scenarios.sample_time_s)
    d = nominal_disturbance(study, process).copy()
    d[4] = offset
    return Scenario(
        name="test/constant/00",
        family="test",
        kind="constant",
        index=0,
        sample_time_s=study.scenarios.sample_time_s,
        disturbances=np.repeat(d[:, None], n, axis=1),
        moisture_limit=np.full(n, study.line.product_moisture_limit_kg_per_kg),
    )


# ------------------------------------------------------------------------------ path condition


def test_path_condition_is_checked_at_every_sample(
    study: HeadroomStudyConfig, process: ProcessConfig
) -> None:
    """A violation that lasts only inside one control interval is found by the truth record."""
    scenario = _constant_scenario(study, process, 2.0)
    steady = SteadyStateOptimizer(study, process)
    limit = study.line.product_moisture_limit_kg_per_kg
    point = steady.solve(scenario.disturbances[:, 0], limit - 0.002)
    state = np.array([[point.state[0], point.state[1], 3e6, point.state[2]]])
    plant = TruthPlant(
        process,
        [scenario],
        state,
        point.command[None],
        interval_s=300.0,
        gas_heater_efficiency=0.8,
    )
    commands = np.tile(point.command, (plant.n_intervals, 1))
    commands[5, 0] -= 25.0  # one interval much colder: the bed gets wetter, then recovers
    for command in commands:
        plant.advance(command[None])
    result = plant.result()
    moisture = result.moisture[0]
    assert result.max_violation[0] == pytest.approx(max(moisture.max() - limit, 0.0))
    assert result.max_violation[0] > 1e-3
    # the record holds every truth sample, not only interval ends
    assert moisture.shape == (plant.n_intervals * 60,)
    interval_ends = moisture[59::60]
    assert moisture.max() >= interval_ends.max()
    # back below the limit at the end: an end-only check would miss the violation
    assert moisture[-1] < limit


def test_path_condition_in_the_optimiser_covers_every_rk4_step(
    study: HeadroomStudyConfig, process: ProcessConfig
) -> None:
    """The optimiser constrains the moisture at the end of every RK4 step of every interval."""
    optimizer = DynamicOptimizer(study, process, 4, expression="MX", warm_start=False)
    assert optimizer.n_steps == study.control.steps_per_interval
    assert optimizer._g_blocks[2] == 4 * optimizer.n_steps  # moisture rows


# ------------------------------------------------------------------------------ R2 == R1


def test_r2_equals_r1_without_disturbance(
    study: HeadroomStudyConfig, process: ProcessConfig
) -> None:
    """Constant disturbance, start at R1's own steady state: the preview optimum is R1.

    Up to the finite-horizon end effect (checked: all but the last four intervals): in the last
    two intervals the preview optimum dries a
    little more and then lets the moisture rise towards the bound (water stored in the bed at the
    end is free); about 0.006 kWh, i.e. 0.3 % of this 2 h scenario and 0.02 % of a 24 h one.
    """
    margin = 0.005
    trivial = dataclasses.replace(
        study, line=dataclasses.replace(study.line, start_margin_kg_per_kg=margin)
    )
    solvers = Solvers(trivial, process)
    scenario = _constant_scenario(trivial, process, 2.0)
    r1 = run_controller("R1", margin, [scenario], solvers)[0]
    r2 = run_controller("R2", margin, [scenario], solvers)[0]
    assert r2.solver["failures"] == 0
    assert not r2.solver["fallback_to_guess"]
    commands_r1 = np.array(r1.commands)
    commands_r2 = np.array(r2.commands)
    body = slice(0, len(commands_r1) - 4)
    np.testing.assert_allclose(commands_r2[body], commands_r1[body], atol=0.05)
    energy_r1 = np.array(r1.energy_kwh_per_interval)
    energy_r2 = np.array(r2.energy_kwh_per_interval)
    np.testing.assert_allclose(energy_r2[body], energy_r1[body], rtol=5e-4)
    assert r2.energy_kwh <= r1.energy_kwh
    assert r2.energy_kwh == pytest.approx(r1.energy_kwh, rel=5e-3)
    assert r1.max_violation == 0.0
    assert r2.max_violation <= study.control.violation_tolerance_kg_per_kg


# ------------------------------------------------------------------------------ bisection


def test_bisection_converges_to_threshold() -> None:
    threshold = 0.0123
    search = bisect_margin(lambda m: m >= threshold, upper=0.05, tolerance=1e-5)
    assert search.converged
    assert threshold <= search.margin <= threshold + 1e-5
    assert search.lower < threshold
    assert all(ok == (m >= threshold) for m, ok in search.evaluations)


def test_bisection_expands_the_bracket_and_reports_failure() -> None:
    search = bisect_margin(lambda m: m >= 0.07, upper=0.02, tolerance=1e-4, max_upper=0.12)
    assert search.converged
    assert 0.07 <= search.margin <= 0.07 + 1e-4
    hopeless = bisect_margin(lambda _m: False, upper=0.02, tolerance=1e-4, max_upper=0.05)
    assert not hopeless.converged


def test_bisection_feasible_at_lower_returns_lower() -> None:
    search = bisect_margin(lambda _m: True, upper=0.05, tolerance=1e-4)
    assert search.margin == 0.0


def test_bisection_over_scenarios_verifies_on_all() -> None:
    """The critical subset misses the hardest scenario; the verification adds it."""
    thresholds = np.array([0.001, 0.004, 0.009, 0.002])
    calls: list[tuple[float, tuple[int, ...]]] = []

    def violations(m: float, ids: list[int]) -> np.ndarray:
        calls.append((m, tuple(ids)))
        return np.maximum(thresholds[list(ids)] - m, 0.0)

    search, final = bisect_margin_over_scenarios(
        violations,
        4,
        tolerance_violation=0.0,
        upper=0.05,
        tolerance=1e-5,
        initial_critical=[1],
    )
    assert search.converged
    assert 0.009 <= search.margin <= 0.009 + 1e-5
    assert np.all(final == 0.0)
    assert any(len(ids) == 4 for _, ids in calls)  # verified on all scenarios
    assert any(ids == (1, 2) for _, ids in calls)  # scenario 2 joined the critical set


def test_bisection_over_scenarios_finds_feasible_lower_end() -> None:
    search, final = bisect_margin_over_scenarios(
        lambda _m, ids: np.zeros(len(ids)),
        3,
        tolerance_violation=0.0,
        upper=0.05,
        tolerance=1e-4,
        initial_critical=[0],
    )
    assert search.margin == 0.0
    assert np.all(final == 0.0)


def test_k_of_n_bisection_allows_the_permitted_failures() -> None:
    """19 of 20 draws must hold: the margin is the second-largest per-draw threshold."""
    rng = np.random.default_rng(3)
    thresholds = rng.uniform(0.0, 0.02, 20)
    ranking = list(np.argsort(-thresholds))
    ranking[0], ranking[5] = ranking[5], ranking[0]  # imperfect ranking: verification must fix it

    def holds(m: float, ids: list[int]) -> np.ndarray:
        return thresholds[list(ids)] <= m

    search, final = bisect_margin_k_of_n(
        holds, 20, 19, ranking, lower=0.0, upper=0.05, tolerance=1e-6
    )
    second_largest = np.sort(thresholds)[-2]
    assert second_largest <= search.margin <= second_largest + 1e-6
    assert int(final.sum()) >= 19


# ------------------------------------------------------------------------------ model identity


def test_symbolic_derivative_is_the_truth_rhs(process: ProcessConfig) -> None:
    simulator = GreyBoxFluidBedGranulator(process)
    x = casadi.SX.sym("x", 3)
    u = casadi.SX.sym("u", 4)
    d = casadi.SX.sym("d", 3)
    function = casadi.Function("f", [x, u, d], list(symbolic_derivative(process, x, u, d)))
    rng = np.random.default_rng(0)
    for _ in range(50):
        state = np.array([rng.uniform(1.8, 2.2), rng.uniform(0.1, 0.4), 3e6, rng.uniform(30, 60)])
        action = np.array([rng.uniform(60, 120), rng.uniform(0.8, 2.5), rng.uniform(1, 2), 1.0])
        disturbance = np.array([rng.uniform(5, 35), rng.uniform(0.25, 0.85), rng.uniform(0.2, 0.4)])
        derivative, diagnostics = simulator.evaluate_derivative(
            state[None], action[None], disturbance[None]
        )
        symbolic, power = function(state[[0, 1, 3]], action, disturbance)
        expected = derivative[0, [0, 1, 3]]
        np.testing.assert_allclose(np.asarray(symbolic).ravel(), expected, rtol=1e-10, atol=1e-14)
        assert float(power) == pytest.approx(float(diagnostics["power_total_kw"][0]), rel=1e-10)


def test_ramp_mean_matches_numerical_mean_and_is_smooth_at_zero() -> None:
    delta = casadi.SX.sym("delta")
    for start, end in ((2.5, 32.5), (62.5, 92.5)):
        function = casadi.Function("r", [delta], [ramp_mean(delta, 0.5, start, end)])
        gradient = casadi.Function(
            "g", [delta], [casadi.gradient(ramp_mean(delta, 0.5, start, end), delta)]
        )
        times = np.linspace(start, end, 20001)
        for value in (-40.0, -12.0, -3.0, 0.0, 1.0, 5.0, 20.0, 60.0):
            numeric = np.mean(np.clip(value, -0.5 * times, 0.5 * times))
            assert float(function(value)) == pytest.approx(numeric, abs=1e-3)
        assert float(gradient(0.0)) == pytest.approx(1.0)
        assert float(gradient(1e-9)) == pytest.approx(1.0, abs=1e-6)


def test_interval_model_tracks_the_truth(
    study: HeadroomStudyConfig, process: ProcessConfig
) -> None:
    """Random 5-min commands (ramped in the truth): optimiser moisture within 5e-4 of the truth."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scenarios = build_scenarios(study, process, "observed")
    scenario = next(s for s in scenarios if s.kind == "inlet_temperature_fast")
    k_intervals = 24
    steady = SteadyStateOptimizer(study, process)
    point = steady.solve(scenario.disturbances[:, 0], 0.14)
    state = np.array(
        [point.state[0], point.state[1], steady_particle_count(process, 0.3), point.state[2]]
    )
    rng = np.random.default_rng(1)
    commands = np.stack([rng.uniform(70, 110, k_intervals), rng.uniform(0.8, 1.2, k_intervals)], 1)
    plant = TruthPlant(
        process,
        [scenario],
        state[None],
        point.command[None],
        interval_s=300.0,
        gas_heater_efficiency=0.8,
    )
    for command in commands:
        plant.advance(command[None])
    truth = plant.moisture_so_far[0]
    step = study.control.optimizer_step_s
    per_step = round(step / study.scenarios.sample_time_s)
    function = interval_function(
        process,
        interval_s=300.0,
        step_s=step,
        nozzle_pressure_bar=study.line.nozzle_pressure_bar,
        sample_time_s=study.scenarios.sample_time_s,
    )
    dist = step_means(scenario.disturbances[:, : k_intervals * 60], per_step)
    x = state[[0, 1, 3]]
    before = point.command
    predicted = []
    n_steps = round(300.0 / step)
    for k in range(k_intervals):
        out = function(x, commands[k], before, dist[:, k * n_steps : (k + 1) * n_steps])
        x = np.asarray(out[0]).ravel()
        predicted.append(np.asarray(out[1]).ravel())
        before = commands[k]
    nodes = truth[per_step - 1 :: per_step]
    assert np.max(np.abs(np.concatenate(predicted) - nodes)) < 5e-4


# ------------------------------------------------------------------------------ scenarios


def test_scenarios_are_deterministic_and_in_their_bands(
    study: HeadroomStudyConfig, process: ProcessConfig
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = build_scenarios(study, process, "observed")
        second = build_scenarios(study, process, "observed")
        stress = build_scenarios(study, process, "stress")
        change = build_scenarios(study, process, "product_change")
    for a, b in zip(first, second, strict=True):
        assert a.name == b.name
        np.testing.assert_array_equal(a.disturbances, b.disturbances)
    kinds = {s.kind for s in first}
    for kind in kinds:
        assert sum(s.kind == kind for s in first) >= study.scenarios.per_type
    obs = study.scenarios.observed
    nominal_solid = study.line.nominal_solid_fraction
    spray = study.line.nominal_spray_rate_kg_per_h
    for s in first:
        assert s.duration_s == study.scenarios.duration_s
        if s.kind == "solid_fraction":
            half = obs.solid_fraction_relative_half_band * nominal_solid
            assert np.all(np.abs(s.disturbances[2] - nominal_solid) <= half + 1e-12)
        if s.kind == "throughput":
            relative = s.disturbances[3] / spray - 1.0
            assert np.all(np.abs(relative) <= obs.throughput_relative_amplitude + 1e-12)
        if s.kind != "ambient":
            assert np.ptp(s.disturbances[0]) == 0.0
    for s in stress:
        if s.kind == "solid_fraction_steps":
            low, high = study.scenarios.stress.solid_fraction_band
            assert np.all((s.disturbances[2] >= low) & (s.disturbances[2] <= high))
    assert len(change) == 4 * study.scenarios.product_change.per_direction
    for s in change:
        assert s.change_time_s == study.scenarios.product_change.change_time_s

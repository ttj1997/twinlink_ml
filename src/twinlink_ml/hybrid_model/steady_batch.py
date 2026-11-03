r"""Steady state of the grey-box model at many operating points at once.

The study of docs/NN_hybrid_model.md evaluates the steady grey-box model (truth, prior knowledge,
refitted model) at several hundred thousand operating points. The scalar solver
:func:`twinlink_ml.simulation.steady_state.solve_steady_state` costs about 0.17 s per point, almost
all of it in 756 single-row calls of the right-hand side. This module runs the SAME algorithm --
nested bisection, outer over the bed temperature, inner over the particle moisture, the same
brackets, the same fallback interval of Befund 78, the same termination rule per point -- on a
batch of points, with a per-point mask. Every point follows exactly the path the scalar solver
would take; the arithmetic is the same element-wise arithmetic on longer arrays. The test
``tests/test_hybrid_model.py`` compares it with the bandit evaluation of Befund 52/96
(``OperatingPointBanditEnv``) point by point.

Only what the study needs is derived: the state, the diagnostics of the right-hand side at the
solution, the feasibility flags of :class:`~twinlink_ml.simulation.steady_state.SteadyStateSolution`
and the constraint check of :func:`~twinlink_ml.simulation.operating_point.check_constraints`
(dew point and exhaust enthalpies are not computed). A point whose bisection meets a bracket
without a sign change is reported as ``solve_failed`` (the scalar solver raises there).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from twinlink_ml.simulation import GreyBoxFluidBedGranulator, ProcessConfig
from twinlink_ml.simulation.greybox import (
    IDX_BED_TEMPERATURE,
    IDX_WATER_MASS,
)
from twinlink_ml.simulation.operating_point import (
    BED_TEMPERATURE_LIMIT_C,
    DEFAULT_SEARCH_MAX_ITERATIONS,
    DEFAULT_SEARCH_TOLERANCE,
    TOLERANCE_MOISTURE_KG_PER_KG,
    TOLERANCE_TEMPERATURE_K,
    TOLERANCE_VELOCITY_M_PER_S,
)
from twinlink_ml.simulation.sorption import isotherm_from_material, sorption_safety_grid
from twinlink_ml.simulation.steady_state import (
    BED_TEMPERATURE_FALLBACK_MARGIN_K,
    BED_TEMPERATURE_SEARCH_MARGIN_K,
    LOWER_SEARCH_BOUND_TOLERANCE_K,
)
from twinlink_ml.simulation.thermo import (
    relative_humidity_from_absolute_humidity,
    saturation_humidity_kg_per_kg,
)

__all__ = ["BatchSteadyResult", "solve_steady_states"]

Payload = dict[str, np.ndarray]
BatchFunction = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, Payload, np.ndarray]]
"""``(x (m,), index (m,)) -> (f (m,), payload, ok (m,))`` for the points ``index`` of the batch."""


@dataclass(frozen=True, slots=True)
class BatchSteadyResult:
    """Steady state at ``n`` points; NaN where ``solve_failed``."""

    solve_failed: np.ndarray
    moisture_kg_per_kg: np.ndarray
    bed_temperature_celsius: np.ndarray
    energy_kw: np.ndarray
    solution_feasible: np.ndarray
    """Feasibility flags of ``SteadyStateSolution`` (fluidisation window, drying regime, ...)."""
    feasible: np.ndarray
    """``ConstraintCheck.feasible`` against the quality limit passed to the solver."""
    normalized_violation: np.ndarray
    diagnostics: Payload


def _take(payload: Payload, rows: np.ndarray) -> Payload:
    return {key: value[rows] for key, value in payload.items()}


def _put(target: Payload, rows: np.ndarray, source: Payload, n: int) -> None:
    for key, value in source.items():
        if key not in target:
            target[key] = np.full((n, *value.shape[1:]), np.nan)
        target[key][rows] = value


def _bisect_batch(
    fn: BatchFunction,
    low: np.ndarray,
    high: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
    low_value: tuple[np.ndarray, Payload] | None = None,
    high_value: tuple[np.ndarray, Payload] | None = None,
) -> tuple[np.ndarray, Payload, np.ndarray]:
    """``_bisect_with_payload`` of ``steady_state.py`` per point; returns ``(x, payload, ok)``."""
    n = low.shape[0]
    every = np.arange(n)
    ok = np.ones(n, dtype=bool)
    if low_value is None:
        f_low, p_low, ok_low = fn(low, every)
        ok &= ok_low
    else:
        f_low, p_low = low_value
    if high_value is None:
        f_high, p_high, ok_high = fn(high, every)
        ok &= ok_high
    else:
        f_high, p_high = high_value
    x = np.full(n, np.nan)
    payload: Payload = {}
    done = ~ok
    at_low = ~done & (f_low == 0.0)
    x[at_low] = low[at_low]
    _put(payload, np.flatnonzero(at_low), _take(p_low, at_low), n)
    done |= at_low
    at_high = ~done & (f_high == 0.0)
    x[at_high] = high[at_high]
    _put(payload, np.flatnonzero(at_high), _take(p_high, at_high), n)
    done |= at_high
    no_sign_change = ~done & (f_low * f_high > 0.0)
    ok &= ~no_sign_change
    done |= no_sign_change

    active = ~done
    a, b, fa = low.copy(), high.copy(), f_low.copy()
    rows = np.flatnonzero(active)
    x[rows] = low[rows]
    _put(payload, rows, _take(p_low, rows), n)
    for _ in range(max_iterations):
        rows = np.flatnonzero(active)
        if rows.size == 0:
            break
        mid = 0.5 * (a[rows] + b[rows])
        f_mid, p_mid, ok_mid = fn(mid, rows)
        x[rows] = mid
        _put(payload, rows, p_mid, n)
        failed = rows[~ok_mid]
        ok[failed] = False
        width = b[rows] - a[rows]
        scale = np.maximum(np.maximum(np.abs(a[rows]), np.abs(b[rows])), 1.0)
        stop = ~ok_mid | (f_mid == 0.0) | (width < tolerance * scale)
        same = (fa[rows] < 0.0) == (f_mid < 0.0)
        move_a = ~stop & same
        move_b = ~stop & ~same
        a[rows[move_a]] = mid[move_a]
        fa[rows[move_a]] = f_mid[move_a]
        b[rows[move_b]] = mid[move_b]
        active[rows[stop]] = False
    x[~ok] = np.nan
    return x, payload, ok


def solve_steady_states(
    config: ProcessConfig,
    inlet_temperature_celsius: np.ndarray,
    gas_velocity_m_per_s: np.ndarray,
    ambient_temperature_celsius: np.ndarray,
    ambient_relative_humidity: np.ndarray,
    *,
    nozzle_pressure_bar: float,
    quality_limit_kg_per_kg: float,
    max_iterations: int = DEFAULT_SEARCH_MAX_ITERATIONS,
    tolerance: float = DEFAULT_SEARCH_TOLERANCE,
) -> BatchSteadyResult:
    """Steady states of ``config`` at the given points (spray rate and solids at their nominal).

    Defaults: the search precision of the bandit (30 steps, 1e-7), as Befund 52/96.
    """
    t_in = np.atleast_1d(np.asarray(inlet_temperature_celsius, dtype=np.float64))
    u_gas = np.atleast_1d(np.asarray(gas_velocity_m_per_s, dtype=np.float64))
    t_amb = np.atleast_1d(np.asarray(ambient_temperature_celsius, dtype=np.float64))
    phi = np.atleast_1d(np.asarray(ambient_relative_humidity, dtype=np.float64))
    n = t_in.shape[0]
    if not (u_gas.shape == t_amb.shape == phi.shape == (n,)):
        raise ValueError("all point arrays need the same length")
    if config.operation.mode == "batch":
        raise ValueError("no steady state in batch operation")
    simulator = GreyBoxFluidBedGranulator(config)
    backend = simulator.backend
    material, geometry, operation, gas = (
        config.material,
        config.geometry,
        config.operation,
        config.gas,
    )

    nominal_action = simulator.nominal_action(1)[0]
    nominal_disturbance = simulator.nominal_disturbance(1)[0]
    action = np.tile(nominal_action, (n, 1))
    action[:, 0] = t_in
    action[:, 1] = u_gas
    action[:, 3] = nozzle_pressure_bar
    disturbance = np.tile(nominal_disturbance, (n, 1))
    disturbance[:, 0] = t_amb
    disturbance[:, 1] = phi

    spray_rate = action[:, 2] / 3600.0
    solid_fraction = disturbance[:, 2]
    solid_mass = geometry.bed_mass_solid_setpoint_kg
    spray_solid = spray_rate * solid_fraction
    seed_flow = operation.seed_mass_flow_kg_per_h / 3600.0
    solid_inflow = spray_solid + seed_flow
    nucleation_denominator = material.nucleation_solid_fraction * spray_solid + seed_flow
    diameter = material.nucleus_diameter_m * (solid_inflow / nucleation_denominator) ** (1.0 / 3.0)
    particle_mass = material.particle_density_kg_per_m3 * math.pi * diameter**3 / 6.0
    particle_count = solid_mass / particle_mass

    isotherm = isotherm_from_material(material)
    phi_min, _phi_max, _t_min, t_max = sorption_safety_grid(
        backend, gas, config.actions, config.disturbances
    )
    moisture_low = float(
        backend.to_numpy(
            isotherm.equilibrium_moisture(backend, backend.asarray(phi_min), backend.asarray(t_max))
        )
    )
    moisture_high = max(10.0 * material.critical_moisture_kg_per_kg, 1.0)

    def evaluate(moisture: np.ndarray, temperature: np.ndarray, rows: np.ndarray) -> Payload:
        state = np.column_stack(
            [
                np.full(rows.size, solid_mass),
                moisture * solid_mass,
                particle_count[rows],
                temperature,
            ]
        )
        derivative, diagnostics = simulator.evaluate_derivative(
            backend.asarray(state),
            backend.asarray(action[rows]),
            backend.asarray(disturbance[rows]),
        )
        out = {
            key: np.asarray(backend.to_numpy(v), dtype=np.float64) for key, v in diagnostics.items()
        }
        out["_derivative"] = np.asarray(backend.to_numpy(derivative), dtype=np.float64)
        return out

    def outer(temperature: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, Payload, np.ndarray]:
        def inner(moisture: np.ndarray, sub: np.ndarray) -> tuple[np.ndarray, Payload, np.ndarray]:
            payload = evaluate(moisture, temperature[sub], rows[sub])
            return payload["_derivative"][:, IDX_WATER_MASS], payload, np.ones(sub.size, bool)

        m = rows.size
        moisture, payload, ok = _bisect_batch(
            inner,
            np.full(m, moisture_low),
            np.full(m, moisture_high),
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        payload["_moisture"] = moisture
        f = np.where(ok, payload["_derivative"][:, IDX_BED_TEMPERATURE], np.nan)
        return f, payload, ok

    every = np.arange(n)
    temperature_low = t_amb + BED_TEMPERATURE_SEARCH_MARGIN_K
    temperature_high = np.maximum(t_in, operation.suspension_temperature_celsius) + 5.0
    lower_search_bound = t_amb + BED_TEMPERATURE_FALLBACK_MARGIN_K
    f_low, p_low, ok_low = outer(temperature_low, every)
    f_high, p_high, ok_high = outer(temperature_high, every)
    fallback = ok_low & ok_high & (f_low < 0.0) & (f_high < 0.0)
    fb = np.flatnonzero(fallback)
    if fb.size:
        temperature_high = temperature_high.copy()
        temperature_high[fb] = temperature_low[fb]
        f_high = f_high.copy()
        f_high[fb] = f_low[fb]
        for key in p_high:
            p_high[key] = p_high[key].copy()
            p_high[key][fb] = p_low[key][fb]
        temperature_low = temperature_low.copy()
        temperature_low[fb] = lower_search_bound[fb]
        f_fb, p_fb, ok_fb = outer(temperature_low[fb], fb)
        f_low = f_low.copy()
        f_low[fb] = f_fb
        for key in p_low:
            p_low[key] = p_low[key].copy()
            p_low[key][fb] = p_fb[key]
        ok_low = ok_low.copy()
        ok_low[fb] = ok_fb
    pre_ok = ok_low & ok_high
    f_low = np.where(pre_ok, f_low, 1.0)
    f_high = np.where(pre_ok, f_high, 1.0)  # same sign -> marked as failed by the bisection
    temperature, payload, ok = _bisect_batch(
        outer,
        temperature_low,
        temperature_high,
        max_iterations=max_iterations,
        tolerance=tolerance,
        low_value=(f_low, p_low),
        high_value=(f_high, p_high),
    )
    ok &= pre_ok
    moisture = np.where(ok, payload["_moisture"], np.nan)
    temperature = np.where(ok, temperature, np.nan)

    inlet_humidity = payload["inlet_humidity_kg_per_kg"]
    outlet_humidity = payload["outlet_humidity_kg_per_kg"]
    u_mf = payload["minimum_fluidization_velocity_m_per_s"]
    u_terminal = payload["terminal_velocity_m_per_s"]
    saturation_degree = payload["saturation_degree"]
    with np.errstate(invalid="ignore"):
        safe_t = np.where(ok, temperature, 50.0)
        safe_y = np.where(ok, outlet_humidity, 0.01)
        y_sat = np.asarray(
            backend.to_numpy(saturation_humidity_kg_per_kg(backend, gas, backend.asarray(safe_t)))
        )
        rh = relative_humidity_from_absolute_humidity(
            backend, gas, backend.asarray(safe_y), backend.asarray(safe_t)
        )
        x_eq = np.asarray(
            backend.to_numpy(isotherm.equilibrium_moisture(backend, rh, backend.asarray(safe_t)))
        )
        solution_feasible = (
            (u_gas > u_mf)
            & (u_gas < u_terminal)
            & (inlet_humidity < outlet_humidity)
            & (outlet_humidity < y_sat)
            & ~(temperature - lower_search_bound <= LOWER_SEARCH_BOUND_TOLERANCE_K)
            & (saturation_degree <= 1.0 + 1e-9)
            & (x_eq - 1e-9 <= moisture)
            & (moisture <= material.critical_moisture_kg_per_kg + 1e-9)
        )
    solution_feasible &= ok

    moisture_violation = np.maximum(0.0, moisture - quality_limit_kg_per_kg)
    gas_lower = np.maximum(0.0, u_mf - u_gas)
    gas_upper = np.maximum(0.0, u_gas - u_terminal)
    bed_violation = np.maximum(0.0, temperature - BED_TEMPERATURE_LIMIT_C)
    feasible = (
        ok
        & (moisture_violation <= TOLERANCE_MOISTURE_KG_PER_KG)
        & (gas_lower <= TOLERANCE_VELOCITY_M_PER_S)
        & (gas_upper <= TOLERANCE_VELOCITY_M_PER_S)
        & (bed_violation <= TOLERANCE_TEMPERATURE_K)
        & solution_feasible
    )
    window = np.maximum(u_terminal - u_mf, 1e-3)
    normalized_violation = (
        moisture_violation / max(quality_limit_kg_per_kg, 1e-6)
        + gas_lower / window
        + gas_upper / window
        + bed_violation / 10.0
        + np.where(solution_feasible, 0.0, 1.0)
    )
    diagnostics = {k: np.where(ok, v, np.nan) for k, v in payload.items() if v.ndim == 1}
    return BatchSteadyResult(
        solve_failed=~ok,
        moisture_kg_per_kg=moisture,
        bed_temperature_celsius=temperature,
        energy_kw=np.where(ok, payload["energy_target"], np.nan),
        solution_feasible=solution_feasible,
        feasible=feasible,
        normalized_violation=np.where(ok, normalized_violation, np.nan),
        diagnostics=diagnostics,
    )

r"""Per-instance oracle and fixed operating point with a carrier-separated objective (M1).

The mechanics are those of Befund 52 (``simulation.oracle_search``: DE + Nelder-Mead, same budget
and seed handling, search precision 30 / 1e-7), with two changes that are the point of M1:

* the quality threshold is an explicit argument, decoupled from the kinetics (decoupled mechanics
  of Befund 64: the kinetics keep ``critical_moisture_kg_per_kg`` of the config);
* the objective is selectable: ``legacy_power_kw`` reproduces the Befund 52 objective
  (``energy_target``, electric booking) exactly and is the anchor check; ``co2_gas_heater`` (main
  case, D5) and ``co2_electric_heater`` (sensitivity) book per carrier
  (:mod:`twinlink_ml.evaluation.carrier_energy`).

Penalty and failure value for the CO2 objectives are the Befund 52 constants converted from kW to
kg CO2e/h with the gas factor (``PENALTY_WEIGHT_KW * 0.201`` and ``SOLVE_FAILURE_PENALTY_KW *
0.201``): the penalty keeps its ratio to a gas kW, which is the order of magnitude of the
objective (a setting of this study, not a tuned value). The failure value is only safe for an
ordinal optimiser (``differential_evolution``); nothing here feeds it into a learned value
function or an interpolation.

Process-pool state is a module global set by :func:`init_worker`, the same pattern as the script of
Befund 52 (picklable under the Windows ``spawn`` start method).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, Literal

import numpy as np

from twinlink_ml.simulation.config import ProcessConfig
from twinlink_ml.simulation.operating_point import (
    PENALTY_WEIGHT_KW,
    SOLVE_FAILURE_PENALTY_KW,
    check_constraints,
)
from twinlink_ml.simulation.oracle_search import OptimizationBudget, evaluate_point, run_de_nm
from twinlink_ml.simulation.steady_state import SteadyStateSolution

from .carrier_energy import CarrierEnergy, carrier_energy_from_diagnostics
from .net_energy import GAS_CO2_KG_PER_KWH

__all__ = [
    "ObjectiveKind",
    "OracleSpec",
    "find_fixed_point",
    "init_worker",
    "objective_value",
    "optimize_point",
    "penalty_weight",
    "run_oracle_grid",
    "solve_failure_value",
]

ObjectiveKind = Literal["legacy_power_kw", "co2_gas_heater", "co2_electric_heater"]

_BASE_CONFIG: ProcessConfig | None = None


def init_worker(base_config: ProcessConfig) -> None:
    """Pool initializer: the config of this run (identical under ``fork`` and ``spawn``)."""
    global _BASE_CONFIG
    _BASE_CONFIG = base_config


@dataclass(frozen=True, slots=True)
class OracleSpec:
    """Objective and constraint settings of one oracle run."""

    quality_threshold_kg_per_kg: float
    objective: ObjectiveKind
    gas_heater_efficiency: float = 0.80
    search_max_iterations: int = 30
    search_tolerance: float = 1.0e-7


def penalty_weight(spec: OracleSpec) -> float:
    """Weight of the normalised violation, in the unit of the objective."""
    if spec.objective == "legacy_power_kw":
        return PENALTY_WEIGHT_KW
    return PENALTY_WEIGHT_KW * GAS_CO2_KG_PER_KWH


def solve_failure_value(spec: OracleSpec) -> float:
    """Objective value returned for a solver failure (ordinal optimiser only)."""
    if spec.objective == "legacy_power_kw":
        return SOLVE_FAILURE_PENALTY_KW
    return SOLVE_FAILURE_PENALTY_KW * GAS_CO2_KG_PER_KWH


def _carrier(
    solution: SteadyStateSolution, config: ProcessConfig, spec: OracleSpec
) -> CarrierEnergy:
    heater = "electric" if spec.objective == "co2_electric_heater" else "gas"
    return carrier_energy_from_diagnostics(
        solution.diagnostics,
        heater=heater,
        process_heater_efficiency=config.energy.heater_efficiency,
        gas_heater_efficiency=spec.gas_heater_efficiency,
    )


def objective_value(
    solution: SteadyStateSolution, config: ProcessConfig, spec: OracleSpec
) -> float:
    """Objective of ``spec`` for one steady state (kW for legacy, kg CO2e/h otherwise)."""
    if spec.objective == "legacy_power_kw":
        return float(solution.diagnostics["energy_target"])
    return _carrier(solution, config, spec).co2_kg_per_h


def penalized_objective(x: np.ndarray, t_amb: float, phi_amb: float, spec: OracleSpec) -> float:
    """Search objective of one instance (module level for picklability)."""
    assert _BASE_CONFIG is not None
    t_in, u_gas, p_nozzle = x
    try:
        solution = evaluate_point(
            t_in,
            u_gas,
            p_nozzle,
            t_amb,
            phi_amb,
            max_iterations=spec.search_max_iterations,
            tolerance=spec.search_tolerance,
            base_config=_BASE_CONFIG,
        )
    except (RuntimeError, ValueError):
        return solve_failure_value(spec)
    check = check_constraints(
        solution,
        critical_moisture_kg_per_kg=spec.quality_threshold_kg_per_kg,
        gas_velocity_m_per_s=u_gas,
    )
    return float(
        objective_value(solution, _BASE_CONFIG, spec)
        + penalty_weight(spec) * check.normalized_violation
    )


def _report(
    solution: SteadyStateSolution,
    config: ProcessConfig,
    spec: OracleSpec,
    u_gas: float,
) -> dict[str, Any]:
    """Result fields of one evaluated operating point (all carriers, both weightings)."""
    check = check_constraints(
        solution,
        critical_moisture_kg_per_kg=spec.quality_threshold_kg_per_kg,
        gas_velocity_m_per_s=u_gas,
    )
    gas = carrier_energy_from_diagnostics(
        solution.diagnostics,
        heater="gas",
        process_heater_efficiency=config.energy.heater_efficiency,
        gas_heater_efficiency=spec.gas_heater_efficiency,
    )
    electric = carrier_energy_from_diagnostics(
        solution.diagnostics,
        heater="electric",
        process_heater_efficiency=config.energy.heater_efficiency,
        gas_heater_efficiency=spec.gas_heater_efficiency,
    )
    return {
        "objective": objective_value(solution, config, spec),
        "legacy_energy_target_kw": float(solution.diagnostics["energy_target"]),
        "gas_heater": {
            "electric_kw": gas.electric_kw,
            "gas_kw": gas.gas_kw,
            "kwh_sum_kw": gas.kwh_sum_kw,
            "co2_kg_per_h": gas.co2_kg_per_h,
        },
        "electric_heater": {
            "electric_kw": electric.electric_kw,
            "gas_kw": electric.gas_kw,
            "kwh_sum_kw": electric.kwh_sum_kw,
            "co2_kg_per_h": electric.co2_kg_per_h,
        },
        "bed_temperature_celsius": solution.bed_temperature_celsius,
        "outlet_temperature_celsius": solution.bed_temperature_celsius
        - config.exhaust.outlet_duct_heat_loss_k,
        "outlet_humidity_kg_per_kg": solution.outlet_humidity_kg_per_kg,
        "dew_point_celsius": solution.dew_point_celsius,
        "dry_air_mass_flow_kg_per_s": solution.dry_air_mass_flow_kg_per_s,
        "particle_moisture_kg_per_kg": solution.particle_moisture_kg_per_kg,
        "distance_to_validity_boundary_kg_per_kg": config.material.critical_moisture_kg_per_kg
        - solution.particle_moisture_kg_per_kg,
        "residual_inf_norm": solution.residual_inf_norm,
        "feasible": check.feasible,
        "infeasibility_reasons": list(solution.infeasibility_reasons),
        "constraint_check": check.as_dict(),
    }


def optimize_point(
    t_amb: float,
    phi_amb: float,
    bounds: list[tuple[float, float]],
    budget: OptimizationBudget,
    spec: OracleSpec,
) -> dict[str, Any]:
    """One instance, single-threaded (the pool distributes instances, as in Befund 52)."""
    assert _BASE_CONFIG is not None
    de_result, refine_result, de_elapsed, refine_elapsed = run_de_nm(
        penalized_objective, bounds, (t_amb, phi_amb, spec), budget, de_workers=1
    )
    best_x = refine_result.x if refine_result.fun <= de_result.fun else de_result.x
    t_in, u_gas, p_nozzle = (float(v) for v in best_x)
    result: dict[str, Any] = {
        "ambient_temperature_celsius": t_amb,
        "ambient_relative_humidity": phi_amb,
        "inlet_temperature_celsius": t_in,
        "gas_velocity_m_per_s": u_gas,
        "nozzle_pressure_bar": p_nozzle,
        "de_converged": bool(de_result.success),
        "de_nfev": int(de_result.nfev),
        "de_elapsed_s": de_elapsed,
        "refine_converged": bool(refine_result.success),
        "refine_nit": int(refine_result.nit),
        "refine_elapsed_s": refine_elapsed,
        "converged": bool(refine_result.success),
        "solve_failed": False,
    }
    try:
        solution = evaluate_point(
            t_in,
            u_gas,
            p_nozzle,
            t_amb,
            phi_amb,
            max_iterations=spec.search_max_iterations,
            tolerance=spec.search_tolerance,
            base_config=_BASE_CONFIG,
        )
    except (RuntimeError, ValueError) as exc:
        result.update({"solve_failed": True, "solve_failure_reason": str(exc), "feasible": False})
        return result
    result.update(_report(solution, _BASE_CONFIG, spec, u_gas))
    return result


def _grid_task(
    task: tuple[int, float, float, list[tuple[float, float]], OptimizationBudget, OracleSpec],
) -> dict[str, Any]:
    index, t_amb, phi_amb, bounds, budget, spec = task
    start = time.perf_counter()
    result = optimize_point(t_amb, phi_amb, bounds, budget, spec)
    result["_grid_index"] = index
    result["_wall_elapsed_s"] = time.perf_counter() - start
    return result


def run_oracle_grid(
    base_config: ProcessConfig,
    grid_points: list[tuple[float, float]],
    bounds: list[tuple[float, float]],
    budget: OptimizationBudget,
    spec: OracleSpec,
    *,
    workers: int,
    label: str,
) -> list[dict[str, Any]]:
    """Oracle at every grid point, one instance per pool task; results ordered by grid index."""
    tasks = [(i, t_amb, phi, bounds, budget, spec) for i, (t_amb, phi) in enumerate(grid_points)]
    print(f"\n=== {label}: {len(tasks)} points, {spec} ===", flush=True)
    results: list[dict[str, Any]] = []
    start = time.perf_counter()
    with Pool(workers, initializer=init_worker, initargs=(base_config,)) as pool:
        for done, result in enumerate(pool.imap_unordered(_grid_task, tasks), start=1):
            results.append(result)
            print(
                f"  [{done}/{len(tasks)}] point {result['_grid_index']} "
                f"({time.perf_counter() - start:.0f} s)",
                flush=True,
            )
    return sorted(results, key=lambda r: r["_grid_index"])


def _fixed_worker(
    x: tuple[float, float, float], t_amb: float, phi_amb: float, spec: OracleSpec
) -> tuple[float, float] | None:
    """One grid point for one fixed candidate: (objective, normalised violation) or None."""
    assert _BASE_CONFIG is not None
    t_in, u_gas, p_nozzle = x
    try:
        solution = evaluate_point(
            t_in,
            u_gas,
            p_nozzle,
            t_amb,
            phi_amb,
            max_iterations=spec.search_max_iterations,
            tolerance=spec.search_tolerance,
            base_config=_BASE_CONFIG,
        )
    except (RuntimeError, ValueError):
        return None
    check = check_constraints(
        solution,
        critical_moisture_kg_per_kg=spec.quality_threshold_kg_per_kg,
        gas_velocity_m_per_s=u_gas,
    )
    return float(objective_value(solution, _BASE_CONFIG, spec)), check.normalized_violation


def _fixed_objective(
    x: np.ndarray, grid_points: list[tuple[float, float]], pool: Any, spec: OracleSpec
) -> float:
    tasks = [(tuple(float(v) for v in x), t_amb, phi, spec) for t_amb, phi in grid_points]
    values: list[float] = []
    violation_sum = 0.0
    for outcome in pool.starmap(_fixed_worker, tasks):
        if outcome is None:
            values.append(solve_failure_value(spec))
            violation_sum += 5.0
        else:
            values.append(outcome[0])
            violation_sum += outcome[1]
    return float(np.mean(values)) + penalty_weight(spec) * violation_sum


def find_fixed_point(
    base_config: ProcessConfig,
    grid_points: list[tuple[float, float]],
    bounds: list[tuple[float, float]],
    budget: OptimizationBudget,
    spec: OracleSpec,
) -> dict[str, Any]:
    """One set point, feasible at every grid point, minimal mean objective (BB3(a) of Befund 52)."""
    start = time.perf_counter()
    with Pool(budget.workers, initializer=init_worker, initargs=(base_config,)) as pool:
        de_result, refine_result, de_elapsed, refine_elapsed = run_de_nm(
            _fixed_objective, bounds, (grid_points, pool, spec), budget, de_workers=1
        )
    best_x = refine_result.x if refine_result.fun <= de_result.fun else de_result.x
    t_in, u_gas, p_nozzle = (float(v) for v in best_x)

    per_point: list[dict[str, Any]] = []
    all_feasible = True
    for t_amb, phi in grid_points:
        entry: dict[str, Any] = {
            "ambient_temperature_celsius": t_amb,
            "ambient_relative_humidity": phi,
        }
        try:
            solution = evaluate_point(
                t_in,
                u_gas,
                p_nozzle,
                t_amb,
                phi,
                max_iterations=spec.search_max_iterations,
                tolerance=spec.search_tolerance,
                base_config=base_config,
            )
        except (RuntimeError, ValueError) as exc:
            entry.update(
                {"solve_failed": True, "solve_failure_reason": str(exc), "feasible": False}
            )
            all_feasible = False
            per_point.append(entry)
            continue
        report = _report(solution, base_config, spec, u_gas)
        all_feasible = all_feasible and report["feasible"]
        entry.update({"solve_failed": False, **report})
        per_point.append(entry)
    return {
        "inlet_temperature_celsius": t_in,
        "gas_velocity_m_per_s": u_gas,
        "nozzle_pressure_bar": p_nozzle,
        "feasible_everywhere": all_feasible,
        "de_converged": bool(de_result.success),
        "de_elapsed_s": de_elapsed,
        "refine_converged": bool(refine_result.success),
        "refine_elapsed_s": refine_elapsed,
        "wall_elapsed_s": time.perf_counter() - start,
        "per_grid_point": per_point,
    }

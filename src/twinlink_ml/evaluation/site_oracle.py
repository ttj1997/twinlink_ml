r"""Joint optimum of one line against the site CO2 (M2, arm M) and evaluation of fixed set points.

Same mechanics as :mod:`twinlink_ml.evaluation.carrier_oracle` (Befund 52 budget and seed
handling, DE + Nelder-Mead, search precision 30 / 1e-7, three variables: inlet temperature, gas
velocity, nozzle pressure; spray rate fixed), with the objective of
:mod:`twinlink_ml.evaluation.site_network`: the CO2 of the whole site (the line, the fixed external
source Q1 and sink S1, the cascade) for the BEST of the enumerated allocations, evaluated inside
every objective call. The quality threshold is an explicit argument decoupled from the kinetics
(decoupled mechanics of Befund 64).

Penalty and failure value are those of ``carrier_oracle`` for the CO2 objectives (the Befund 52
constants converted with the gas factor); they are only used by the ordinal DE and never enter a
table or a learned function. A solver failure of a final report is flagged and counted, not
replaced.

Process-pool state is a module global set by :func:`init_worker` (Windows ``spawn``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any

from twinlink_ml.simulation.config import ProcessConfig
from twinlink_ml.simulation.operating_point import (
    PENALTY_WEIGHT_KW,
    SOLVE_FAILURE_PENALTY_KW,
    check_constraints,
)
from twinlink_ml.simulation.oracle_search import OptimizationBudget, evaluate_point, run_de_nm
from twinlink_ml.simulation.steady_state import SteadyStateSolution

from .net_energy import GAS_CO2_KG_PER_KWH
from .site_network import (
    SiteResult,
    SiteSpec,
    all_allocations,
    all_to_all_allocation,
    evaluate_site,
    fixed_allocation,
    line_state,
)

__all__ = [
    "SiteOracleSpec",
    "evaluate_site_point",
    "init_worker",
    "optimize_point",
    "run_site_grid",
]

_BASE_CONFIG: ProcessConfig | None = None
_ALLOCATIONS = all_allocations(1)
_FIXED = fixed_allocation(1)
_ALL_TO_ALL = all_to_all_allocation(1)
_PENALTY_WEIGHT = PENALTY_WEIGHT_KW * GAS_CO2_KG_PER_KWH
_FAILURE_VALUE = SOLVE_FAILURE_PENALTY_KW * GAS_CO2_KG_PER_KWH


def init_worker(base_config: ProcessConfig) -> None:
    """Pool initializer: the config of this run."""
    global _BASE_CONFIG
    _BASE_CONFIG = base_config


@dataclass(frozen=True, slots=True)
class SiteOracleSpec:
    """Constraint and network settings of one site oracle run."""

    quality_threshold_kg_per_kg: float
    site: SiteSpec
    search_max_iterations: int = 30
    search_tolerance: float = 1.0e-7


def _site_results(
    solution: SteadyStateSolution,
    config: ProcessConfig,
    spec: SiteOracleSpec,
    t_amb: float,
    phi_amb: float,
    t_in: float,
) -> tuple[list[SiteResult], int]:
    state = line_state("A", solution, config, t_amb=t_amb, phi_amb=phi_amb, t_in=t_in)
    return evaluate_site((state,), spec.site, _ALLOCATIONS)


def _best(results: list[SiteResult]) -> SiteResult:
    return min(results, key=lambda r: r.co2_kg_per_h)


def penalized_objective(x: Any, t_amb: float, phi_amb: float, spec: SiteOracleSpec) -> float:
    """Search objective of one instance: best-allocation site CO2 plus the constraint penalty."""
    assert _BASE_CONFIG is not None
    t_in, u_gas, p_nozzle = (float(v) for v in x)
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
        return _FAILURE_VALUE
    results, _ = _site_results(solution, _BASE_CONFIG, spec, t_amb, phi_amb, t_in)
    if not results:
        return _FAILURE_VALUE
    check = check_constraints(
        solution,
        critical_moisture_kg_per_kg=spec.quality_threshold_kg_per_kg,
        gas_velocity_m_per_s=u_gas,
    )
    return float(_best(results).co2_kg_per_h + _PENALTY_WEIGHT * check.normalized_violation)


def evaluate_site_point(
    config: ProcessConfig,
    spec: SiteOracleSpec,
    *,
    t_in: float,
    u_gas: float,
    p_nozzle: float,
    t_amb: float,
    phi_amb: float,
) -> dict[str, Any]:
    """One operating point under the fixed, the all-to-all and the best allocation.

    A solver failure returns ``solve_failed`` and nothing else (masked by the caller, counted).
    """
    out: dict[str, Any] = {
        "ambient_temperature_celsius": t_amb,
        "ambient_relative_humidity": phi_amb,
        "inlet_temperature_celsius": t_in,
        "gas_velocity_m_per_s": u_gas,
        "nozzle_pressure_bar": p_nozzle,
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
            base_config=config,
        )
    except (RuntimeError, ValueError) as exc:
        out.update({"solve_failed": True, "solve_failure_reason": str(exc), "feasible": False})
        return out
    check = check_constraints(
        solution,
        critical_moisture_kg_per_kg=spec.quality_threshold_kg_per_kg,
        gas_velocity_m_per_s=u_gas,
    )
    results, rejected = _site_results(solution, config, spec, t_amb, phi_amb, t_in)
    by_connections = {r.connections: r for r in results}
    out.update(
        {
            "feasible": check.feasible,
            "constraint_check": check.as_dict(),
            "infeasibility_reasons": list(solution.infeasibility_reasons),
            "particle_moisture_kg_per_kg": solution.particle_moisture_kg_per_kg,
            "distance_to_validity_boundary_kg_per_kg": config.material.critical_moisture_kg_per_kg
            - solution.particle_moisture_kg_per_kg,
            "outlet_temperature_celsius": solution.bed_temperature_celsius
            - config.exhaust.outlet_duct_heat_loss_k,
            "outlet_humidity_kg_per_kg": solution.outlet_humidity_kg_per_kg,
            "dew_point_celsius": solution.dew_point_celsius,
            "n_allocations_rejected": rejected,
            "n_allocations_evaluated": len(results),
            "routing_fixed": by_connections[_FIXED].as_dict() if _FIXED in by_connections else None,
            "routing_all_to_all": by_connections[_ALL_TO_ALL].as_dict()
            if _ALL_TO_ALL in by_connections
            else None,
            "routing_best": _best(results).as_dict() if results else None,
        }
    )
    return out


def optimize_point(
    t_amb: float,
    phi_amb: float,
    bounds: list[tuple[float, float]],
    budget: OptimizationBudget,
    spec: SiteOracleSpec,
) -> dict[str, Any]:
    """One instance, single-threaded (the pool distributes instances, as in Befund 52)."""
    assert _BASE_CONFIG is not None
    de_result, refine_result, de_elapsed, refine_elapsed = run_de_nm(
        penalized_objective, bounds, (t_amb, phi_amb, spec), budget, de_workers=1
    )
    best_x = refine_result.x if refine_result.fun <= de_result.fun else de_result.x
    t_in, u_gas, p_nozzle = (float(v) for v in best_x)
    result = evaluate_site_point(
        _BASE_CONFIG,
        spec,
        t_in=t_in,
        u_gas=u_gas,
        p_nozzle=p_nozzle,
        t_amb=t_amb,
        phi_amb=phi_amb,
    )
    result.update(
        {
            "de_converged": bool(de_result.success),
            "de_nfev": int(de_result.nfev),
            "de_elapsed_s": de_elapsed,
            "refine_converged": bool(refine_result.success),
            "refine_nit": int(refine_result.nit),
            "refine_elapsed_s": refine_elapsed,
            "converged": bool(refine_result.success),
            "search_objective_value": float(min(refine_result.fun, de_result.fun)),
        }
    )
    return result


def _grid_task(
    task: tuple[int, float, float, list[tuple[float, float]], OptimizationBudget, SiteOracleSpec],
) -> dict[str, Any]:
    index, t_amb, phi_amb, bounds, budget, spec = task
    start = time.perf_counter()
    result = optimize_point(t_amb, phi_amb, bounds, budget, spec)
    result["_grid_index"] = index
    result["_wall_elapsed_s"] = time.perf_counter() - start
    return result


def run_site_grid(
    base_config: ProcessConfig,
    grid_points: list[tuple[float, float]],
    bounds: list[tuple[float, float]],
    budget: OptimizationBudget,
    spec: SiteOracleSpec,
    *,
    workers: int,
    label: str,
) -> list[dict[str, Any]]:
    """Site oracle at every grid point, one instance per pool task; ordered by grid index."""
    tasks = [(i, t_amb, phi, bounds, budget, spec) for i, (t_amb, phi) in enumerate(grid_points)]
    print(f"\n=== {label}: {len(tasks)} points ===", flush=True)
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

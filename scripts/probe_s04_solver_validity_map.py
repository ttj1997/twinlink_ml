r"""S0.4: which steady-state solutions the solver marks infeasible, and where that boundary lies.

Read-only diagnosis, no training, no change of any setting.

Part A: the grid points of the S0.3 diagnosis (13 x 9 reference grid, nominal disturbance) with
``g_tilde > 0``: the exact ``infeasibility_reasons`` of ``solve_steady_state``, the product
moisture, the critical moisture of the config, the saturation degree and the residual of the
solution (a converged solution is not a numerical failure).

Part B: map over the 25 Befund 52 test contexts x the 13 x 9 action grid: per grid point the
product moisture, the solver feasibility and the CATEGORY of each reason (moisture outside the
plausible range, saturation degree above 1, fluidisation window, solver failure). Per context:
the largest moisture among solver-feasible points, the smallest moisture among points that are
infeasible by the moisture reason, and whether any point is infeasible for a reason other than
moisture while its moisture lies below the critical moisture (a boundary somewhere else).

The source line that sets the moisture reason is read from ``steady_state.py`` and stored
verbatim in the artifact.

    python scripts/probe_s04_solver_validity_map.py
"""

from __future__ import annotations

import inspect
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import provenance, require_clean_tree, write_artifact

from twinlink_ml.rl import load_test_points
from twinlink_ml.rl.env import CONFIG_PATH, FIXED_NOZZLE_PRESSURE_BAR
from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    PlaceholderConfigWarning,
    load_process_config,
)
from twinlink_ml.simulation import steady_state as steady_state_module
from twinlink_ml.simulation.operating_point import (
    DEFAULT_SEARCH_MAX_ITERATIONS,
    DEFAULT_SEARCH_TOLERANCE,
    build_variant_config,
)
from twinlink_ml.simulation.steady_state import solve_steady_state

GRID_T_IN = np.linspace(60.0, 120.0, 13)
GRID_U_GAS = np.linspace(0.8, 2.5, 9)
NOMINAL_T_AMB = 20.0
NOMINAL_PHI = 0.5
WORKERS = 8
MARGIN_KG_PER_KG = 0.005961819615175404


def _category(reason: str) -> str:
    if reason.startswith("Partikelfeuchte"):
        return "moisture_outside_plausible_range"
    if reason.startswith("Sättigungsgrad"):
        return "saturation_degree_above_1"
    if reason.startswith("u_gas"):
        return "fluidisation_window"
    return "other"


def _solve(t_in: float, u_gas: float, t_amb: float, phi: float) -> dict[str, Any]:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    base = load_process_config(CONFIG_PATH)
    config = build_variant_config(
        base, t_in=t_in, u_gas=u_gas, p_nozzle=FIXED_NOZZLE_PRESSURE_BAR, t_amb=t_amb, phi_amb=phi
    )
    try:
        solution = solve_steady_state(
            GreyBoxFluidBedGranulator(config),
            max_iterations=DEFAULT_SEARCH_MAX_ITERATIONS,
            tolerance=DEFAULT_SEARCH_TOLERANCE,
        )
    except (RuntimeError, ValueError) as exc:
        return {"t_in": t_in, "u_gas": u_gas, "solve_failed": True, "error": str(exc)[:120]}
    return {
        "t_in": t_in,
        "u_gas": u_gas,
        "solve_failed": False,
        "moisture_kg_per_kg": float(solution.particle_moisture_kg_per_kg),
        "critical_moisture_kg_per_kg": base.material.critical_moisture_kg_per_kg,
        "saturation_degree": float(solution.diagnostics["saturation_degree"]),
        "residual_inf_norm": float(solution.residual_inf_norm),
        "solver_feasible": bool(solution.feasible),
        "reasons": list(solution.infeasibility_reasons),
        "categories": sorted({_category(r) for r in solution.infeasibility_reasons}),
    }


def _context_task(context: tuple[float, float]) -> dict[str, Any]:
    t_amb, phi = context
    points = [_solve(float(t), float(u), t_amb, phi) for t in GRID_T_IN for u in GRID_U_GAS]
    return {"t_amb": t_amb, "phi": phi, "points": points}


def _summarise(context: dict[str, Any], critical: float) -> dict[str, Any]:
    points = context["points"]
    solved = [p for p in points if not p["solve_failed"]]
    feasible = [p for p in solved if p["solver_feasible"]]
    moisture_infeasible = [
        p for p in solved if "moisture_outside_plausible_range" in p["categories"]
    ]
    other = [
        p
        for p in solved
        if not p["solver_feasible"] and "moisture_outside_plausible_range" not in p["categories"]
    ]
    return {
        "t_amb": context["t_amb"],
        "phi": context["phi"],
        "n_points": len(points),
        "n_solve_failed": len(points) - len(solved),
        "n_solver_feasible": len(feasible),
        "n_infeasible_moisture_reason": len(moisture_infeasible),
        "n_infeasible_other_reason_only": len(other),
        "max_moisture_among_feasible": max(
            (p["moisture_kg_per_kg"] for p in feasible), default=None
        ),
        "min_moisture_among_moisture_infeasible": min(
            (p["moisture_kg_per_kg"] for p in moisture_infeasible), default=None
        ),
        "infeasible_other_reason_points_with_moisture_below_critical": [
            {k: p[k] for k in ("t_in", "u_gas", "moisture_kg_per_kg", "reasons")}
            for p in other
            if p["moisture_kg_per_kg"] < critical
        ],
        "all_moisture_infeasible_points_above_critical": all(
            p["moisture_kg_per_kg"] > critical for p in moisture_infeasible
        ),
        "max_residual_inf_norm": max((p["residual_inf_norm"] for p in solved), default=None),
    }


def main() -> int:
    require_clean_tree()
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    critical = load_process_config(CONFIG_PATH).material.critical_moisture_kg_per_kg

    source = inspect.getsource(steady_state_module).splitlines()
    marker = [i for i, line in enumerate(source) if "local_equilibrium_moisture - 1e-9" in line]
    code_lines = source[marker[0] - 2 : marker[0] + 9] if marker else []

    nominal = _context_task((NOMINAL_T_AMB, NOMINAL_PHI))
    part_a = [
        p
        for p in nominal["points"]
        if not p["solve_failed"] and p["moisture_kg_per_kg"] > critical - MARGIN_KG_PER_KG
    ]

    contexts = [
        (p.ambient_temperature_celsius, p.ambient_relative_humidity) for p in load_test_points()
    ]
    with Pool(WORKERS) as pool:
        mapped = pool.map(_context_task, contexts)
    summaries = [_summarise(c, critical) for c in mapped]

    payload = {
        "_provenance": provenance(),
        "critical_moisture_kg_per_kg": critical,
        "source_of_moisture_reason": {
            "file": "src/twinlink_ml/simulation/steady_state.py",
            "lines_verbatim": code_lines,
        },
        "part_a_nominal_points_with_g_tilde_positive": part_a,
        "part_b_context_summaries": summaries,
        "part_b_all_moisture_infeasible_above_critical": all(
            s["all_moisture_infeasible_points_above_critical"] for s in summaries
        ),
        "part_b_contexts_with_other_reason_below_critical": [
            (s["t_amb"], s["phi"])
            for s in summaries
            if s["infeasible_other_reason_points_with_moisture_below_critical"]
        ],
        "part_b_points": [
            {"t_amb": c["t_amb"], "phi": c["phi"], "points": c["points"]} for c in mapped
        ],
    }
    write_artifact("s04_solver_validity_map", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

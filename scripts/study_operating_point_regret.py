r"""BB1-BB3: does the energy-optimal operating point shift across the disturbance range.

Preliminary study on RL suitability -- NO reinforcement learning, NO network training, NO
dataset sweep. Only ``solve_steady_state`` (``simulation/steady_state.py``) plus classical
optimisation per grid point. The question: does the benefit of a learned policy (mapping
disturbances -> set points) depend on anything measurable at all, or does a fixed operating
point do the job?

BB1 -- Problem definition
-------------------------
Optimisation variables (the later RL actions), bounds from ``ActionSpaceConfig``
(``conf/process/greybox_gab_placeholder.yaml``, section ``actions``), NOT set anew:

    inlet_temperature_celsius, gas_velocity_m_per_s, nozzle_pressure_bar

FIXED, not optimised: ``spray_rate_kg_per_h`` at its nominal value. Reason: at a constant
suspension solid fraction the spray rate determines the throughput. Without a fixed
throughput, "minimise energy" is degenerate -- the optimum would be spray rate zero, i.e. no
production.

Disturbance grid: ``ambient_temperature_celsius`` x ``ambient_relative_humidity``, ranges
from ``DisturbanceSpaceConfig``, equidistant. ``suspension_solid_fraction`` fixed at its
nominal value (third disturbance, not part of the grid here -- a 3D disturbance map would be
over-engineering for a preliminary estimate). Default grid 5x5 (run time measured: see
below); 7x7 reachable via ``--grid-size``, according to the run-time measurement clearly more
expensive (factor ~2, see the ``--grid-size`` help text) -- "acceptable run time" is thus an
explicit decision at invocation, not one pre-decided in the script.

Objective: ``diagnostics["energy_target"]`` in steady state -- respects the formulation still
open in the project (``energy.target: power_kw | specific_kwh_per_kg`` in the config, see
``simulation/energy.py``) without deciding it anew here. In addition, the specific
consumption ``power_total_kw / (3600 * product_mass_flow_kg_per_s)`` is ALWAYS reported,
regardless of what ``energy.target`` currently is -- which of the two formulations ends up
as the target is open in the project (see the ``energy.py`` docstring); this study only
answers how the operating point feels depending on the choice.

Hard constraints (a point that violates one of them is INFEASIBLE -- no penalty function
blurs that in the reporting; the internal optimiser uses a penalty function only as a search
heuristic, see ``_normalized_violation`` below, but every reported feasibility comes from a
separate, unweighted check at the end point):

1. ``product_moisture_kg_per_kg <= material.critical_moisture_kg_per_kg`` (existing field).
2. ``u_mf(d50) <= gas_velocity <= u_terminal(d50)``, evaluated at steady state with
   ``simulation/fluidization.py`` (identical calculation to the diagnostic quantities from
   ``solve_steady_state``).
3. ``bed_temperature_celsius <= 80`` -- a freely set placeholder (Befund 46, see
   ``docs/04_stand_ap1.md``), reported separately WHETHER it binds: if it binds, that is an
   artifact of the (unsupported) bound and not a statement about the real plant.

In addition, ``SteadyStateSolution.feasible`` (the solver's own validity check: degree of
saturation <= 1, particle moisture in the range plausible from the sorption isotherm) must
hold -- this is not a fourth process constraint but the condition that the "steady state"
found is a valid one at all (see the ``steady_state.py`` module docstring, section
"Feasibility is a flag, not an exception"). Without it, a numerically degenerate point
could "win" an optimisation.

BB2 -- Optimisation per grid point
-----------------------------------
Three variables, hence ``scipy.optimize.differential_evolution`` (robust against non-convex
feasible regions -- the fluidization window and the moisture bound cut the box down to a
subset that is in general not convex), followed by a local refinement (Nelder-Mead, not
gradient-based: ``evaluate_derivative`` contains ``clip``/``max`` terms at which a gradient
method should, for good reason, be avoided, as in ``steady_state.py``). Seed fixed
(``--seed``, default 0).

Run-time compromise (measured on this machine, 16 cores): a single ``solve_steady_state``
call with the solver defaults (100 bisection steps, tolerance 1e-13) costs ~0.46 s.
``differential_evolution`` with default settings (popsize=15) would need several thousand
calls per grid point for this -- impractical. Two adjustments, both ONLY for the search, not
for the reported result:

* During the search the bisection runs with a relaxed tolerance
  (``--search-max-iterations``/``--search-tolerance``, default 30 / 1e-7 instead of 100 / 1e-13
  -- residual at the nominal point thus ~2e-9 instead of ~5e-15, irrelevant for an
  optimisation search). The FINALLY reported point is always solved again with the solver
  defaults.
* Parallelisation is ACROSS THE GRID POINTS, not within the DE population of a single point
  (``differential_evolution(workers=...)``). The first draft did it the other way round --
  population parallelism across all cores, then a (fundamentally serial) Nelder-Mead
  refinement. Measured: the refinement alone cost as much wall time as the parallel DE phase
  before it, while 15 of 16 cores did nothing in the meantime -- per grid point exactly the
  pattern "N candidates in parallel, then one alone that takes as long as the parallel
  phase". Since the 25-49 grid points are truly independent, ``main()`` instead distributes
  them over a process pool (``pool.imap_unordered``, ``optimize_grid_point`` runs
  single-threaded per point), which keeps ALL cores busy over the ENTIRE computing time of
  each point (DE **and** refinement) and additionally removes the synchronisation barrier
  that ``differential_evolution`` otherwise inserts at the end of every generation (there a
  slow population delays the entire generation, here only its own grid point). Measured cost
  of ONE grid point, single-threaded, with popsize=10/maxiter=50: DE ~90 s, refinement
  (Nelder-Mead, ``maxiter=200``) an additional up to ~30-60 s -- with 16 parallel grid points
  (``ceil(25/16)=2`` rounds) this makes many times less wall time for the entire grid than
  the initial population parallelism per point.

Non-converged grid points (``differential_evolution`` or Nelder-Mead result with
``success=False``) are REPORTED (field ``de_converged``/``refine_converged`` per grid point),
not skipped -- the best point found is still used and flagged.

BB3 -- Regret of a fixed operating point
--------------------------------------------
(a) Fixed operating point = the set-point set that minimises the MEAN energy over the grid,
    SUBJECT TO THE CONSTRAINT of being feasible at EVERY grid point (otherwise it is
    unsuitable as a comparison, see the task). Same optimisation machinery as BB2, only with
    the objective averaged over the grid and the violation summed over all grid points; the
    grid evaluation per candidate is parallelised over a separate, reused multiprocessing
    pool (``differential_evolution`` itself runs with ``workers=1`` for this -- nested pools
    are not intended).

    If no fixed point feasible everywhere exists, this is reported IMMEDIATELY and BEFORE all
    other numbers (criterion: the best candidate solution found in the main run is infeasible
    at at least one grid point; for corroboration a second search then runs in addition,
    which minimises only the summed violation -- if its minimum also stays > 0, that is an
    empirical finding of this search, not a proof, but the best possible statement without
    an exhaustive search over a continuous box).

(b) Regret per grid point = energy(fixed point) - energy(instance-wise optimum from BB2),
    median/95th percentile/maximum, absolute (kW or kWh/kg) and relative (%).

(c) Range of the optimal set points over the grid per variable, absolute and as a fraction
    of the respective action span (``ActionBound.span``).

Classification (no recommendation): the regret is the UPPER BOUND of what an adaptive
control can save compared with a fixed operating point. In the per-mille range a fixed
operating point suffices; in the percent range a policy is justified.

Invocation::

    python scripts/study_operating_point_regret.py --output artifacts/runs/bb1_bb3.json

BB4 (sensitivity to the 80 °C bed temperature bound) is NOT part of this script -- according
to the task only relevant if BB3 shows that ``bed_temperature_celsius <= 80`` binds at
relevant grid points. For this, the script reports the necessary precondition (field
``binding_bed_temperature`` per grid point); the concrete sensitivity analysis follows
separately.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import warnings
from collections.abc import Callable
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np

from twinlink_ml.simulation import (
    PlaceholderConfigWarning,
    ProcessConfig,
    load_process_config,
)
from twinlink_ml.simulation.operating_point import (  # noqa: F401  (re-exported, S0.1)
    BED_TEMPERATURE_LIMIT_C,
    DEFAULT_SEARCH_MAX_ITERATIONS,
    DEFAULT_SEARCH_TOLERANCE,
    ConstraintCheck,
    build_variant_config,
    check_constraints,
    specific_energy_kwh_per_kg,
)
from twinlink_ml.simulation.operating_point import (
    PENALTY_WEIGHT_KW as _PENALTY_WEIGHT_KW,
)
from twinlink_ml.simulation.operating_point import (
    SOLVE_FAILURE_PENALTY_KW as _SOLVE_FAILURE_PENALTY_KW,
)
from twinlink_ml.simulation.oracle_search import (
    OptimizationBudget,
    action_bounds,
    build_grid,
    compute_regret,
    evaluate_point,
)
from twinlink_ml.simulation.oracle_search import run_de_nm as _optimize
from twinlink_ml.simulation.steady_state import SteadyStateSolution
from twinlink_ml.utils.tracking import environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "study_operating_point_regret.json"

DEFAULT_POPSIZE = 10
DEFAULT_MAXITER = 50
DEFAULT_DE_TOL = 1.0e-4

_BASE_CONFIG: ProcessConfig | None = None
"""Modul-globaler Zustand -- vermeidet, die Config bei jedem Zielfunktionsaufruf neu zu
bepicken. Im Hauptprozess EINMAL in ``main()`` gesetzt; in den Worker-Prozessen über
``Pool(..., initializer=_set_base_config, initargs=(base_config,))`` (s. ``main()``), NICHT
per ``fork``-Vererbung -- das ursprüngliche Verfahren (Vererbung des im Hauptprozess
gesetzten Globals) setzte ``fork`` als Worker-Startmethode voraus und lief deshalb nie unter
Windows (``spawn``-Default dort: jeder Worker importiert das Modul frisch, ``_BASE_CONFIG``
bleibt ``None``). Der ``initializer`` funktioniert unter beiden Startmethoden identisch.
Siehe ``_set_base_config``."""


def _set_base_config(config: ProcessConfig) -> None:
    global _BASE_CONFIG
    _BASE_CONFIG = config


_OBJECTIVE_VALUE_FN: Callable[[SteadyStateSolution, ProcessConfig], dict[str, float]] | None = None
"""Optionaler Override der Zielgröße (MA4, ``study_operating_point_regret_net.py``) --
``None`` (Default) reproduziert die bisherige reine Bruttoleistung bitgleich, siehe
:func:`_objective_components`. Gleiches Modul-global-plus-Pool-Initializer-Muster wie
``_BASE_CONFIG`` (siehe dort): ein Funktionsobjekt ist unter Windows-``spawn`` nur
picklebar, wenn es ein Modul-Top-Level-Objekt ist (nicht in ``main()`` definiert) --
``study_operating_point_regret_net.py`` erfüllt das, seine Override-Funktion lebt auf
Modulebene dieses importierenden Skripts."""


def _set_objective_value_fn(
    fn: Callable[[SteadyStateSolution, ProcessConfig], dict[str, float]] | None,
) -> None:
    global _OBJECTIVE_VALUE_FN
    _OBJECTIVE_VALUE_FN = fn


def _objective_components(
    solution: SteadyStateSolution, base_config: ProcessConfig
) -> dict[str, float]:
    """Target plus diagnostic fields for a steady state.

    Without override (``_OBJECTIVE_VALUE_FN is None``, the state before MA4): pure
    gross power, bit-identical to ``solution.diagnostics["energy_target"]`` as previously
    read directly inline -- ``objective_value == gross_energy_target_kw``,
    ``q_rec_kw``/``recovery_credit_kw`` both 0.0. With override (MA4 net variants)
    the injected function returns the same four keys, with
    ``objective_value != gross_energy_target_kw``.
    """
    if _OBJECTIVE_VALUE_FN is not None:
        return _OBJECTIVE_VALUE_FN(solution, base_config)
    gross_kw = float(solution.diagnostics["energy_target"])
    return {
        "objective_value": gross_kw,
        "gross_energy_target_kw": gross_kw,
        "q_rec_kw": 0.0,
        "recovery_credit_kw": 0.0,
    }


def _suppress_placeholder_warning() -> None:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)


def _penalized_energy(
    x: np.ndarray, t_amb: float, phi_amb: float, max_iterations: int, tolerance: float
) -> float:
    """Objective function for the search (BB2, per grid point). Module level for picklability."""
    assert _BASE_CONFIG is not None
    t_in, u_gas, p_nozzle = x
    try:
        solution = evaluate_point(
            t_in,
            u_gas,
            p_nozzle,
            t_amb,
            phi_amb,
            max_iterations=max_iterations,
            tolerance=tolerance,
            base_config=_BASE_CONFIG,
        )
    except (RuntimeError, ValueError):
        return _SOLVE_FAILURE_PENALTY_KW
    check = check_constraints(
        solution,
        critical_moisture_kg_per_kg=_BASE_CONFIG.material.critical_moisture_kg_per_kg,
        gas_velocity_m_per_s=u_gas,
    )
    objective = _objective_components(solution, _BASE_CONFIG)["objective_value"]
    return float(objective + _PENALTY_WEIGHT_KW * check.normalized_violation)


def optimize_grid_point(
    t_amb: float, phi_amb: float, bounds: list[tuple[float, float]], budget: OptimizationBudget
) -> dict[str, Any]:
    """One grid point (BB2) -- RUNS SINGLE-THREADED (``de_workers=1``), deliberately.

    Parallelisation happens NOT here (within the DE population of a single
    grid point), but in ``main()`` ACROSS the grid points (``pool.
    imap_unordered`` on this function) -- see module docstring, section
    "Laufzeit-Kompromiss", for the measurement that justifies this decision: population
    parallelism only speeds up the DE phase and leaves all cores but one idle during
    the (inherently serial) Nelder-Mead refinement -- per grid point
    exactly the pattern "N candidates in parallel, then one alone, taking as long as
    the parallel phase before". 25-49 grid points, by contrast, are genuinely independent
    of each other: distributing them over the pool keeps all cores busy during the ENTIRE
    run (DE **and** refinement of every single point) and additionally avoids
    the synchronisation barrier that ``differential_evolution`` otherwise inserts at the end
    of each generation (a slow population delays the entire
    generation there, here only its own grid point).
    """
    assert _BASE_CONFIG is not None
    args = (t_amb, phi_amb, budget.search_max_iterations, budget.search_tolerance)
    de_result, refine_result, de_elapsed, refine_elapsed = _optimize(
        _penalized_energy, bounds, args, budget, de_workers=1
    )
    best_x = refine_result.x if refine_result.fun <= de_result.fun else de_result.x

    t_in, u_gas, p_nozzle = (float(v) for v in best_x)
    # Same precision as during the search (see ``evaluate_point`` docstring) --
    # a higher precision ONLY for the report call was the cause of the
    # bracket failures in the first full grid run, not their remedy.
    solve_failed = False
    try:
        solution = evaluate_point(
            t_in,
            u_gas,
            p_nozzle,
            t_amb,
            phi_amb,
            max_iterations=budget.search_max_iterations,
            tolerance=budget.search_tolerance,
            base_config=_BASE_CONFIG,
        )
    except (RuntimeError, ValueError) as exc:
        solution = None
        solve_failed = True
        solve_failure_reason = str(exc)

    result: dict[str, Any] = {
        "ambient_temperature_celsius": t_amb,
        "ambient_relative_humidity": phi_amb,
        "inlet_temperature_celsius": t_in,
        "gas_velocity_m_per_s": u_gas,
        "nozzle_pressure_bar": p_nozzle,
        # At this budget DE exhausts ``maxiter`` as planned (see module docstring,
        # "Laufzeit-Kompromiss") -- this is NOT a failure as long as the subsequent
        # refinement converges. ``converged`` therefore only assesses the refinement
        # (the state that is actually reported); ``de_converged`` is kept as a
        # diagnostic field, but on its own is no warning sign.
        "de_converged": bool(de_result.success),
        "de_message": str(de_result.message),
        "de_nit": int(de_result.nit),
        "de_nfev": int(de_result.nfev),
        "de_elapsed_s": de_elapsed,
        "refine_converged": bool(refine_result.success),
        "refine_message": str(refine_result.message),
        "refine_nit": int(refine_result.nit),
        "refine_elapsed_s": refine_elapsed,
        "converged": bool(refine_result.success),
        "solve_failed": solve_failed,
    }
    if solve_failed:
        result["solve_failure_reason"] = solve_failure_reason
        result["feasible"] = False
        return result

    check = check_constraints(
        solution,
        critical_moisture_kg_per_kg=_BASE_CONFIG.material.critical_moisture_kg_per_kg,
        gas_velocity_m_per_s=u_gas,
    )
    components = _objective_components(solution, _BASE_CONFIG)
    result.update(
        {
            "energy_target": components["objective_value"],
            "energy_target_unit": _BASE_CONFIG.energy.target,
            "gross_energy_target_kw": components["gross_energy_target_kw"],
            "q_rec_kw": components["q_rec_kw"],
            "recovery_credit_kw": components["recovery_credit_kw"],
            "specific_energy_kwh_per_kg": specific_energy_kwh_per_kg(solution),
            "bed_temperature_celsius": solution.bed_temperature_celsius,
            "particle_moisture_kg_per_kg": solution.particle_moisture_kg_per_kg,
            "diameter_um": solution.diameter_m * 1e6,
            "residual_inf_norm": solution.residual_inf_norm,
            "feasible": check.feasible,
            "infeasibility_reasons": list(solution.infeasibility_reasons),
            "constraint_check": check.as_dict(),
        }
    )
    return result


def _grid_point_task(
    task: tuple[int, float, float, list[tuple[float, float]], OptimizationBudget],
) -> dict[str, Any]:
    """One element for ``pool.imap_unordered`` in ``main()`` -- one grid point complete
    (DE + refinement + final solver call), single-threaded in ONE worker process.
    The index is passed along because ``imap_unordered`` does NOT guarantee the order
    of arrival -- ``compute_regret``/``compute_setpoint_spread`` however expect the same
    order as ``grid_points``.
    """
    index, t_amb, phi_amb, bounds, budget = task
    start = time.perf_counter()
    result = optimize_grid_point(t_amb, phi_amb, bounds, budget)
    result["_grid_index"] = index
    result["_wall_elapsed_s"] = time.perf_counter() - start
    return result


def _fixed_point_worker(
    x: tuple[float, float, float],
    t_amb: float,
    phi_amb: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[float, float] | None:
    """One grid point for a fixed candidate -- target of the pool in BB3(a)."""
    assert _BASE_CONFIG is not None
    t_in, u_gas, p_nozzle = x
    try:
        solution = evaluate_point(
            t_in,
            u_gas,
            p_nozzle,
            t_amb,
            phi_amb,
            max_iterations=max_iterations,
            tolerance=tolerance,
            base_config=_BASE_CONFIG,
        )
    except (RuntimeError, ValueError):
        return None
    check = check_constraints(
        solution,
        critical_moisture_kg_per_kg=_BASE_CONFIG.material.critical_moisture_kg_per_kg,
        gas_velocity_m_per_s=u_gas,
    )
    objective = _objective_components(solution, _BASE_CONFIG)["objective_value"]
    return float(objective), check.normalized_violation


def _fixed_point_objective(
    x: np.ndarray,
    grid_points: list[tuple[float, float]],
    pool: Pool,
    max_iterations: int,
    tolerance: float,
    energy_weight: float,
) -> float:
    """``energy_weight`` is deliberately NOT keyword-only: ``differential_evolution``
    passes ``args`` purely positionally to the objective function (``func(x, *args)``,
    see ``scipy._lib._util._FunctionWrapper``) -- a keyword-only parameter here
    led to ``TypeError: takes 5 positional arguments but 6 were given``, because the
    fifth ``args`` element (the weighting factor) finds no keyword slot."""
    tasks = [
        (tuple(float(v) for v in x), t_amb, phi_amb, max_iterations, tolerance)
        for t_amb, phi_amb in grid_points
    ]
    results = pool.starmap(_fixed_point_worker, tasks)
    energies: list[float] = []
    violation_sum = 0.0
    for outcome in results:
        if outcome is None:
            energies.append(_SOLVE_FAILURE_PENALTY_KW)
            violation_sum += 5.0
        else:
            energy, violation = outcome
            energies.append(energy)
            violation_sum += violation
    return energy_weight * float(np.mean(energies)) + _PENALTY_WEIGHT_KW * violation_sum


def find_fixed_operating_point(
    grid_points: list[tuple[float, float]],
    bounds: list[tuple[float, float]],
    budget: OptimizationBudget,
    pool: Pool,
) -> dict[str, Any]:
    """BB3(a): fixed operating point, feasible at every grid point, minimum mean energy."""
    args = (grid_points, pool, budget.search_max_iterations, budget.search_tolerance, 1.0)
    de_result, refine_result, de_elapsed, refine_elapsed = _optimize(
        _fixed_point_objective, bounds, args, budget, de_workers=1
    )
    best_x = refine_result.x if refine_result.fun <= de_result.fun else de_result.x
    t_in, u_gas, p_nozzle = (float(v) for v in best_x)

    per_point: list[dict[str, Any]] = []
    all_feasible = True
    for t_amb, phi_amb in grid_points:
        try:
            solution = evaluate_point(
                t_in,
                u_gas,
                p_nozzle,
                t_amb,
                phi_amb,
                max_iterations=budget.search_max_iterations,
                tolerance=budget.search_tolerance,
                base_config=_BASE_CONFIG,
            )
        except (RuntimeError, ValueError) as exc:
            per_point.append(
                {
                    "ambient_temperature_celsius": t_amb,
                    "ambient_relative_humidity": phi_amb,
                    "feasible": False,
                    "solve_failed": True,
                    "solve_failure_reason": str(exc),
                }
            )
            all_feasible = False
            continue
        check = check_constraints(
            solution,
            critical_moisture_kg_per_kg=_BASE_CONFIG.material.critical_moisture_kg_per_kg,
            gas_velocity_m_per_s=u_gas,
        )
        all_feasible = all_feasible and check.feasible
        components = _objective_components(solution, _BASE_CONFIG)
        per_point.append(
            {
                "ambient_temperature_celsius": t_amb,
                "ambient_relative_humidity": phi_amb,
                "feasible": check.feasible,
                "solve_failed": False,
                "energy_target": components["objective_value"],
                "gross_energy_target_kw": components["gross_energy_target_kw"],
                "q_rec_kw": components["q_rec_kw"],
                "recovery_credit_kw": components["recovery_credit_kw"],
                "specific_energy_kwh_per_kg": specific_energy_kwh_per_kg(solution),
                "infeasibility_reasons": list(solution.infeasibility_reasons),
                "constraint_check": check.as_dict(),
            }
        )

    return {
        "inlet_temperature_celsius": t_in,
        "gas_velocity_m_per_s": u_gas,
        "nozzle_pressure_bar": p_nozzle,
        "feasible_everywhere": all_feasible,
        "de_converged": bool(de_result.success),
        "de_message": str(de_result.message),
        "de_elapsed_s": de_elapsed,
        "refine_converged": bool(refine_result.success),
        "refine_elapsed_s": refine_elapsed,
        "per_grid_point": per_point,
    }


def verify_no_feasible_fixed_point_exists(
    grid_points: list[tuple[float, float]],
    bounds: list[tuple[float, float]],
    budget: OptimizationBudget,
    pool: Pool,
) -> dict[str, Any]:
    """Second search, ONLY against the summed violation, to corroborate BB3(a).

    No proof (continuous box, no exhaustive search) -- but the best possible
    statement without one: if the minimum of the violation stays > 0, even a global search aimed
    exclusively at feasibility has found no point that is feasible everywhere.
    """
    args = (grid_points, pool, budget.search_max_iterations, budget.search_tolerance, 0.0)
    de_result, refine_result, de_elapsed, refine_elapsed = _optimize(
        _fixed_point_objective, bounds, args, budget, de_workers=1
    )
    best_fun = min(de_result.fun, refine_result.fun)
    return {
        "minimum_violation_found": best_fun / _PENALTY_WEIGHT_KW,
        "de_elapsed_s": de_elapsed,
        "refine_elapsed_s": refine_elapsed,
    }


def compute_setpoint_spread(
    grid_results: list[dict[str, Any]], base: ProcessConfig
) -> dict[str, Any]:
    feasible = [r for r in grid_results if not r.get("solve_failed") and r.get("feasible", False)]
    variables = {
        "inlet_temperature_celsius": base.actions.inlet_temperature_celsius,
        "gas_velocity_m_per_s": base.actions.gas_velocity_m_per_s,
        "nozzle_pressure_bar": base.actions.nozzle_pressure_bar,
    }
    spread: dict[str, Any] = {"n_feasible_points": len(feasible)}
    if not feasible:
        return spread
    for name, bound in variables.items():
        values = np.asarray([r[name] for r in feasible])
        span = float(np.max(values) - np.min(values))
        spread[name] = {
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "span_absolute": span,
            "span_fraction_of_bound": span / bound.span,
        }
    return spread


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=5,
        choices=[5, 7],
        help="5x5 (~1h Laufzeit, Default) oder 7x7 (~2-2.5h, gemessen mit --workers=-1)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--popsize", type=int, default=DEFAULT_POPSIZE)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--de-tol", type=float, default=DEFAULT_DE_TOL)
    parser.add_argument("--search-max-iterations", type=int, default=DEFAULT_SEARCH_MAX_ITERATIONS)
    parser.add_argument("--search-tolerance", type=float, default=DEFAULT_SEARCH_TOLERANCE)
    parser.add_argument("--workers", type=int, default=-1, help="-1 = alle Kerne (os.cpu_count())")
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--skip-fixed-point",
        action="store_true",
        help="nur BB1/BB2 (je-Gitterpunkt-Optima), kein Regret -- fuer schnelle Testlaeufe",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    _suppress_placeholder_warning()

    env = environment_fingerprint()
    if env["git/dirty"]:
        print(
            "WARNUNG: Arbeitsbaum ist dirty -- git/dirty wird in jeden Record geschrieben, "
            "der Lauf ist trotzdem nicht vollständig reproduzierbar.",
            flush=True,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base_config = load_process_config(arguments.config)
    _set_base_config(base_config)

    workers = arguments.workers if arguments.workers > 0 else cpu_count()
    budget = OptimizationBudget(
        seed=arguments.seed,
        popsize=arguments.popsize,
        maxiter=arguments.maxiter,
        de_tol=arguments.de_tol,
        search_max_iterations=arguments.search_max_iterations,
        search_tolerance=arguments.search_tolerance,
        workers=workers,
    )
    bounds = action_bounds(base_config)
    grid_points = build_grid(base_config, arguments.grid_size)

    print("=" * 78)
    print(f"Config              {arguments.config}")
    print(f"Fingerprint         {base_config.fingerprint}")
    print(
        f"Gitter              {arguments.grid_size}x{arguments.grid_size} = {len(grid_points)} Punkte"
    )
    print(
        f"Fest (Nominalwert)  spray_rate={base_config.actions.spray_rate_kg_per_h.nominal} kg/h, "
        f"suspension_solid_fraction={base_config.disturbances.suspension_solid_fraction.nominal}"
    )
    print(f"Zielgröße           energy_target = {base_config.energy.target}")
    print(f"Worker              {workers}")
    print(f"git/commit={env['git/commit']}  git/dirty={env['git/dirty']}")
    print("=" * 78)

    # ONE pool for the entire run (BB2 distribution + BB3), never created anew per call --
    # see the docstring of ``optimize_grid_point``: pools created repeatedly across many
    # sequential ``differential_evolution`` calls caused a deadlock in testing
    # (fork() with thread leftovers of the previous pool).
    #
    # initializer/initargs instead of inheritance via fork(): sets ``_BASE_CONFIG`` in every
    # worker explicitly, independently of the start method (fork under Linux, spawn under
    # Windows) -- see the docstring of ``_BASE_CONFIG``.
    with Pool(workers, initializer=_set_base_config, initargs=(base_config,)) as pool:
        print(
            f"\n--- BB2: Optimierung je Gitterpunkt ({len(grid_points)} Punkte, über {workers} "
            f"Worker verteilt) " + "-" * 10
        )
        print(
            "Hinweis: ``differential_evolution`` erschöpft bei diesem Suchbudget planmäßig "
            "sein Iterationslimit (kein Fehlschlag, siehe Modul-Docstring) -- konvergiert "
            "werten muss nur die anschließende Nelder-Mead-Verfeinerung, die den tatsächlich "
            "berichteten Punkt liefert. Nur DIESE Konvergenz wird unten als "
            "'nicht konvergiert' geflaggt; ``de_converged`` bleibt separat im Artefakt."
        )
        tasks = [
            (index, t_amb, phi_amb, bounds, budget)
            for index, (t_amb, phi_amb) in enumerate(grid_points)
        ]
        grid_results_by_index: dict[int, dict[str, Any]] = {}
        for n_done, result in enumerate(pool.imap_unordered(_grid_point_task, tasks), start=1):
            index = result["_grid_index"]
            t_amb, phi_amb = grid_points[index]
            status = (
                "OK"
                if not result.get("solve_failed") and result.get("feasible")
                else "UNZULAESSIG/FEHLER"
            )
            conv = "" if result["converged"] else "  [VERFEINERUNG NICHT KONVERGIERT]"
            print(
                f"[{n_done:2d}/{len(grid_points)}] (Punkt {index + 1}) T_amb={t_amb:5.1f} °C "
                f"phi={phi_amb:.2f}  -> {status:20s} ({result['_wall_elapsed_s']:5.1f}s){conv}"
            )
            grid_results_by_index[index] = result
        grid_results = [grid_results_by_index[i] for i in range(len(grid_points))]

        n_infeasible = sum(
            1 for r in grid_results if r.get("solve_failed") or not r.get("feasible", False)
        )
        n_not_converged = sum(1 for r in grid_results if not r["converged"])
        n_de_budget_exhausted = sum(1 for r in grid_results if not r["de_converged"])
        print(
            f"\nGitterpunkte gesamt: {len(grid_results)}, unzulässig/Fehler: {n_infeasible}, "
            f"Verfeinerung nicht konvergiert: {n_not_converged} "
            f"(DE-Suchbudget erschöpft, planmäßig: {n_de_budget_exhausted}/{len(grid_results)})"
        )

        output: dict[str, Any] = {
            "environment": env,
            "config_path": str(arguments.config),
            "config_fingerprint": base_config.fingerprint,
            "grid_size": arguments.grid_size,
            "budget": dataclasses.asdict(budget),
            "grid_results": grid_results,
        }

        if not arguments.skip_fixed_point:
            print(
                f"\n--- BB3(a): fester Betriebspunkt (über {len(grid_points)} Gitterpunkte gemittelt) "
                + "-" * 5
            )
            fixed_point = find_fixed_operating_point(grid_points, bounds, budget, pool)
            if not fixed_point["feasible_everywhere"]:
                print(
                    "\n*** BEFUND: kein an JEDEM Gitterpunkt zulässiger fester Betriebspunkt "
                    "gefunden. ***"
                )
                verification = verify_no_feasible_fixed_point_exists(
                    grid_points, bounds, budget, pool
                )
                print(
                    f"Verifikations-Suche (nur Verletzung minimiert): "
                    f"minimale gefundene Verletzung = {verification['minimum_violation_found']:.4f} "
                    "(0.0 würde Zulässigkeit bedeuten)."
                )
                print(
                    "Kein Beweis (kontinuierliche Box, keine erschöpfende Suche), aber auch eine "
                    "zweite, ausschließlich auf Zulässigkeit zielende globale Suche findet keinen "
                    "Punkt -- ein fester Betriebspunkt ist als Vergleich für BB3(b)/(c) damit "
                    "nicht zu bilden; diese Zahlen entfallen."
                )
                output["fixed_point"] = fixed_point
                output["fixed_point_verification"] = verification
            else:
                print(
                    f"Fester Punkt: T_in={fixed_point['inlet_temperature_celsius']:.2f} °C, "
                    f"u_gas={fixed_point['gas_velocity_m_per_s']:.3f} m/s, "
                    f"p_nozzle={fixed_point['nozzle_pressure_bar']:.3f} bar "
                    f"-- an allen {len(grid_points)} Gitterpunkten zulässig."
                )
                regret = compute_regret(grid_results, fixed_point)
                spread = compute_setpoint_spread(grid_results, base_config)
                output["fixed_point"] = fixed_point
                output["regret"] = regret
                output["setpoint_spread"] = spread

                print("\n--- BB3(b): Regret ------------------------------------------------")
                if regret["n_points"] == 0:
                    print("Keine gemeinsam zulässigen Punkte für einen Regret-Vergleich.")
                else:
                    print(f"n = {regret['n_points']} Gitterpunkte")
                    print(
                        f"absolut [kW]   Median={regret['absolute_kw']['median']:.4f}  "
                        f"P95={regret['absolute_kw']['p95']:.4f}  Max={regret['absolute_kw']['max']:.4f}"
                    )
                    print(
                        f"relativ [%]    Median={regret['relative_pct']['median']:.3f}  "
                        f"P95={regret['relative_pct']['p95']:.3f}  Max={regret['relative_pct']['max']:.3f}"
                    )

                print("\n--- BB3(c): Spannweite optimaler Sollwerte über das Gitter ---------")
                for name in (
                    "inlet_temperature_celsius",
                    "gas_velocity_m_per_s",
                    "nozzle_pressure_bar",
                ):
                    entry = spread.get(name)
                    if entry is None:
                        continue
                    print(
                        f"{name:28s} [{entry['minimum']:7.3f}, {entry['maximum']:7.3f}]  "
                        f"Spanne={entry['span_absolute']:7.3f}  "
                        f"({100 * entry['span_fraction_of_bound']:5.1f} % der Stellgrößenspanne)"
                    )

                print(
                    "\nEinordnung: Regret ist die OBERGRENZE dessen, was eine adaptive Regelung "
                    "gegenüber diesem festen Betriebspunkt einsparen kann. Promillebereich -> "
                    "fester Betriebspunkt genügt; Prozentbereich -> Policy begründet. "
                    "(Keine Empfehlung -- reine Einordnung.)"
                )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nErgebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

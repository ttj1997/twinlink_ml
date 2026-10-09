r"""DE + Nelder-Mead oracle mechanics of Befund 52, shared by the studies (M0).

Moved verbatim from ``scripts/study_operating_point_regret.py`` so that code in ``src/`` and
new studies can use it without loading a script (same step as S0.1 for the constraint
check). The script re-imports every name, so the existing probe scripts that read them off
the loaded script module keep working. The function bodies are unchanged; only
``_optimize`` is public here as ``run_de_nm`` (the script keeps ``_optimize`` as an alias).
Nothing here holds module state: the objective, its arguments and the bounds are passed in.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .config import ProcessConfig
from .greybox import GreyBoxFluidBedGranulator
from .operating_point import build_variant_config
from .steady_state import SteadyStateSolution, solve_steady_state

__all__ = [
    "OptimizationBudget",
    "action_bounds",
    "build_grid",
    "compute_regret",
    "evaluate_point",
    "run_de_nm",
    "with_gas_velocity_minimum",
]


def evaluate_point(
    t_in: float,
    u_gas: float,
    p_nozzle: float,
    t_amb: float,
    phi_amb: float,
    *,
    max_iterations: int,
    tolerance: float,
    base_config: ProcessConfig,
) -> SteadyStateSolution:
    """Steady state of one operating point (set points + ambient state).

    Bewusst OHNE eigenes ``temperature_bracket_celsius`` -- ein erster Versuch, das
    Modul-Default (``ambient+10`` / ``max(inlet, susp)+5``, siehe ``steady_state.py``)
    durch eine breitere, "physikalisch begründete" Marge zu ersetzen, hat sich als
    Fehlschlag erwiesen: ``ambient+10`` ist keine willkürliche Wahl, sondern hält die
    äußere Bisektion aktiv im Trocknungsregime (siehe dortiger Docstring, Abschnitt zur
    unteren Bracket-Grenze) -- eine niedrigere Grenze reicht in die Kondensationsbranche
    hinein, in der die INNERE Feuchte-Bisektion strukturell keinen Vorzeichenwechsel
    findet. Das brach nicht nur die wenigen Randfälle, die es beheben sollte, sondern
    praktisch jede Kandidatenauswertung (siehe Modul-Docstring, Abschnitt
    "Laufzeit-Kompromiss"). Der eigentliche Befund aus dem ersten vollen Gitterlauf --
    Such- und Report-Präzision fanden am Bracket-Rand unterschiedliches Vorzeichen, weil
    sie MIT UNTERSCHIEDLICHER Bisektionspräzision liefen -- wird stattdessen dadurch
    behoben, dass ``optimize_grid_point``/``find_fixed_operating_point`` für den
    ENDGÜLTIGEN Report dieselbe (Such-)Präzision verwenden wie während der Suche, siehe
    dort. Ein danach verbleibender Bracket-Fehlschlag ist damit ein echter Befund
    (der angenommene Beharrungszustand liegt außerhalb des dokumentierten Gültigkeits-
    bereichs des Lösers), keine Artefakt-Diagnose mehr.

    **Nachtrag (Befund 78/79, Prüfbefund K1).** Die Begründung oben gilt nur für Grenzen
    UNTERHALB der Umgebungstemperatur: oberhalb von ``T_amb`` ist ``Y_in < Y_s(T_bed)``, die
    Kondensationsbranche existiert dort nicht. Die feste Grenze ``ambient+10`` schnitt bei
    warmer Umgebung gültige Trocknungszustände ab, und mehrere berichtete Optima lagen genau
    auf ihr. ``solve_steady_state`` durchsucht deshalb, nur wenn an ``ambient+10`` kein
    Vorzeichenwechsel besteht, das Rückfallintervall ``[ambient+0.5, ambient+10]`` (überall
    sonst bitgleich zum Stand davor). Diese Funktion übergibt weiterhin kein eigenes Bracket.
    """
    config = build_variant_config(
        base_config, t_in=t_in, u_gas=u_gas, p_nozzle=p_nozzle, t_amb=t_amb, phi_amb=phi_amb
    )
    simulator = GreyBoxFluidBedGranulator(config)
    return solve_steady_state(simulator, max_iterations=max_iterations, tolerance=tolerance)


@dataclass(frozen=True, slots=True)
class OptimizationBudget:
    """Suchbudget für ``differential_evolution`` + Nelder-Mead, geteilt von BB2 und BB3(a)."""

    seed: int
    popsize: int
    maxiter: int
    de_tol: float
    search_max_iterations: int
    search_tolerance: float
    workers: int


def run_de_nm(
    objective: Callable[..., float],
    bounds: list[tuple[float, float]],
    args: tuple,
    budget: OptimizationBudget,
    *,
    de_workers: int | Callable[[Callable, list], list],
) -> tuple[Any, Any, float, float]:
    """DE + Nelder-Mead-Verfeinerung, gemeinsam für BB2 und BB3(a) verwendet."""
    de_start = time.perf_counter()
    de_result = differential_evolution(
        objective,
        bounds,
        args=args,
        seed=budget.seed,
        polish=False,
        workers=de_workers,
        updating="deferred" if de_workers != 1 else "immediate",
        popsize=budget.popsize,
        maxiter=budget.maxiter,
        tol=budget.de_tol,
    )
    de_elapsed = time.perf_counter() - de_start

    refine_start = time.perf_counter()
    refine_result = minimize(
        objective,
        de_result.x,
        args=args,
        method="Nelder-Mead",
        bounds=bounds,
        options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 200, "adaptive": True},
    )
    refine_elapsed = time.perf_counter() - refine_start

    return de_result, refine_result, de_elapsed, refine_elapsed


def build_grid(base: ProcessConfig, grid_size: int) -> list[tuple[float, float]]:
    """Equidistant ambient-temperature x relative-humidity grid of the disturbance ranges."""
    disturbances = base.disturbances
    t_amb_values = np.linspace(
        disturbances.ambient_temperature_celsius.minimum,
        disturbances.ambient_temperature_celsius.maximum,
        grid_size,
    )
    phi_values = np.linspace(
        disturbances.ambient_relative_humidity.minimum,
        disturbances.ambient_relative_humidity.maximum,
        grid_size,
    )
    return [(float(t), float(phi)) for t in t_amb_values for phi in phi_values]


def action_bounds(base: ProcessConfig) -> list[tuple[float, float]]:
    """Bounds of the three optimised actions (inlet temperature, gas velocity, nozzle pressure)."""
    actions = base.actions
    return [
        (actions.inlet_temperature_celsius.minimum, actions.inlet_temperature_celsius.maximum),
        (actions.gas_velocity_m_per_s.minimum, actions.gas_velocity_m_per_s.maximum),
        (actions.nozzle_pressure_bar.minimum, actions.nozzle_pressure_bar.maximum),
    ]


def with_gas_velocity_minimum(
    bounds: list[tuple[float, float]], minimum_m_per_s: float
) -> list[tuple[float, float]]:
    """Bounds of :func:`action_bounds` with the gas-velocity lower bound replaced (a run parameter).

    The config is not touched: the default stays what it is, only the search box of one run
    changes. The new minimum must lie inside the box (below the upper bound).
    """
    low, high = bounds[1]
    if not minimum_m_per_s < high:
        raise ValueError(f"gas-velocity minimum {minimum_m_per_s} must be below the bound {high}")
    out = list(bounds)
    out[1] = (float(minimum_m_per_s), high)
    return out


def compute_regret(
    grid_results: list[dict[str, Any]], fixed_point: dict[str, Any]
) -> dict[str, Any]:
    """Regret of the fixed point against the per-instance optimum (absolute and relative)."""
    absolute_kw: list[float] = []
    relative_pct: list[float] = []
    for grid_point, fixed in zip(grid_results, fixed_point["per_grid_point"], strict=True):
        if grid_point.get("solve_failed") or not grid_point.get("feasible", False):
            continue
        if fixed.get("solve_failed") or not fixed.get("feasible", False):
            continue
        instance_optimum = grid_point["energy_target"]
        fixed_energy = fixed["energy_target"]
        regret = fixed_energy - instance_optimum
        absolute_kw.append(regret)
        denom = abs(instance_optimum) if abs(instance_optimum) > 1e-9 else 1e-9
        relative_pct.append(100.0 * regret / denom)

    if not absolute_kw:
        return {"n_points": 0}

    absolute_arr = np.asarray(absolute_kw)
    relative_arr = np.asarray(relative_pct)
    return {
        "n_points": len(absolute_kw),
        "absolute_kw": {
            "median": float(np.median(absolute_arr)),
            "p95": float(np.percentile(absolute_arr, 95)),
            "max": float(np.max(absolute_arr)),
        },
        "relative_pct": {
            "median": float(np.median(relative_arr)),
            "p95": float(np.percentile(relative_arr, 95)),
            "max": float(np.max(relative_arr)),
        },
    }

r"""Optimiser C2 of docs/30: multistart Nelder-Mead on the penalised steady objective.

The same optimiser runs on every environment (grey-box U0 and learned U1-U3), so that a difference
between environments is not a difference of optimisers. Objective, as the oracle of Befund 52/96:

.. math:: J(a) = E(a) + P \, v(a), \qquad P = 50 \text{ kW},

with ``v`` the normalised violation against the TIGHTENED limit ``0.15 - k m`` (moisture divided by
the limit, bed temperature above 80 degC divided by 10 K); a point without a steady state (grey-box
solver failure, or no valid fixed point of the learned model) gets the failure value of the oracle
(``carrier_oracle.solve_failure_value``), i.e. it is not admissible.

Starts: the fixed operating point of the limit 0.15 (Befund 93) and the first ``n_sobol`` points of
a scrambled Sobol sequence (seed 0) over the action box [60, 120] degC x [0.8, 2.5] m/s; Nelder-Mead
with the options of the oracle refinement (``oracle_search.run_de_nm``: ``xatol`` 1e-4,
``fatol`` 1e-7, ``maxiter`` 200, adaptive) and the box as bounds. Result: the start with the lowest
``J``; every start is reported.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

__all__ = ["MultistartConfig", "PointEvaluation", "multistart_minimize", "start_points"]


@dataclass(frozen=True, slots=True)
class MultistartConfig:
    """Settings of C2 (fixed before the study runs)."""

    n_sobol: int = 7
    sobol_seed: int = 0
    xatol: float = 1e-4
    fatol: float = 1e-7
    maxiter: int = 200


@dataclass(frozen=True, slots=True)
class PointEvaluation:
    """Penalised objective and its parts at one operating point."""

    objective: float
    energy_kw: float
    moisture_kg_per_kg: float
    bed_temperature_celsius: float
    valid: bool


Evaluator = Callable[[float, float], PointEvaluation]


def start_points(
    bounds: list[tuple[float, float]], fixed_point: tuple[float, float], config: MultistartConfig
) -> np.ndarray:
    """The fixed point followed by ``n_sobol`` scrambled Sobol points over the box."""
    sobol = qmc.Sobol(d=2, scramble=True, seed=config.sobol_seed).random(8)[: config.n_sobol]
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    return np.vstack([np.array(fixed_point)[None, :], lower + sobol * (upper - lower)])


def multistart_minimize(
    evaluate: Evaluator,
    bounds: list[tuple[float, float]],
    fixed_point: tuple[float, float],
    config: MultistartConfig | None = None,
) -> dict[str, Any]:
    """Best of all starts plus the record of every start."""
    cfg = config if config is not None else MultistartConfig()
    n_evaluations = 0

    def objective(x: np.ndarray) -> float:
        nonlocal n_evaluations
        n_evaluations += 1
        return evaluate(float(x[0]), float(x[1])).objective

    starts = start_points(bounds, fixed_point, cfg)
    runs: list[dict[str, Any]] = []
    for start in starts:
        result = minimize(
            objective,
            start,
            method="Nelder-Mead",
            bounds=bounds,
            options={
                "xatol": cfg.xatol,
                "fatol": cfg.fatol,
                "maxiter": cfg.maxiter,
                "adaptive": True,
            },
        )
        runs.append(
            {
                "start": start.tolist(),
                "x": [float(v) for v in result.x],
                "objective": float(result.fun),
                "nit": int(result.nit),
                "success": bool(result.success),
                "message": str(result.message),
            }
        )
    best = min(runs, key=lambda run: run["objective"])
    final = evaluate(best["x"][0], best["x"][1])
    return {
        "inlet_temperature_celsius": best["x"][0],
        "gas_velocity_m_per_s": best["x"][1],
        "objective": final.objective,
        "energy_kw": final.energy_kw,
        "moisture_kg_per_kg": final.moisture_kg_per_kg,
        "bed_temperature_celsius": final.bed_temperature_celsius,
        "valid": final.valid,
        "n_evaluations": n_evaluations,
        "starts": runs,
    }

"""Box-constrained minimisation with constraints on a vectorised model: candidates, then SLSQP.

Used for every optimisation of the study except the R1 NLP of P2's true model: on a surrogate
(cheap, vectorised predictions) and on P1's true model and the fitted grey-box models (one call per
point; :func:`batch_from_problem`).

1. ``candidates`` scrambled Sobol points of the decision box plus its centre are evaluated in one
   batch.
2. The ``polish`` best of them (feasible ones by energy, then the least violating) start SLSQP in
   unit coordinates. Gradients are forward differences, all ``du + 1`` points of a stencil in one
   batch call (:class:`_Stencil`).
3. The result is the best feasible point found (model feasibility, ``max g / scale <= tol``); if
   there is none, the least violating one, flagged ``feasible=False``.

The record keeps every polished end point, so the multistart can be checked (``starts_at_best``,
``spread``) as for the R1 NLP.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from .problem import Box, SteadyProblem

__all__ = ["BatchModel", "OptResult", "batch_from_problem", "minimize_box"]

BatchModel = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
"""``U (n, du)`` -> energy ``(n,)`` and normalised constraints ``(n, m)`` (``<= 0`` feasible)."""


@dataclass(frozen=True)
class OptResult:
    """Result of :func:`minimize_box`."""

    u: np.ndarray
    energy: float
    constraints: np.ndarray
    """Normalised model constraints at ``u``."""
    feasible: bool
    evaluations: int
    starts: int
    starts_at_best: int
    spread: float
    """Range of the energy over the feasible polished end points."""


class _Stencil:
    """Objective, constraints and their forward-difference Jacobians from one batch per point."""

    def __init__(self, model: BatchModel, box: Box, step: float) -> None:
        self.model, self.box, self.step = model, box, step
        self.key: bytes | None = None
        self.calls = 0
        self.e0 = 0.0
        self.g0: np.ndarray = np.zeros(0)
        self.de: np.ndarray = np.zeros(0)
        self.dg: np.ndarray = np.zeros(0)

    def _at(self, z: np.ndarray) -> None:
        key = z.tobytes()
        if key == self.key:
            return
        d = z.size
        pts = np.repeat(z[None, :], d + 1, axis=0)
        h = np.where(z + self.step <= 1.0, self.step, -self.step)
        pts[np.arange(1, d + 1), np.arange(d)] += h
        e, g = self.model(self.box.from_unit(pts))
        self.calls += d + 1
        self.e0, self.g0 = float(e[0]), g[0]
        self.de = (e[1:] - e[0]) / h
        self.dg = ((g[1:] - g[0]) / h[:, None]).T
        self.key = key

    def f(self, z: np.ndarray) -> float:
        self._at(z)
        return self.e0

    def df(self, z: np.ndarray) -> np.ndarray:
        self._at(z)
        return self.de

    def c(self, z: np.ndarray) -> np.ndarray:
        self._at(z)
        out: np.ndarray = -self.g0
        return out

    def dc(self, z: np.ndarray) -> np.ndarray:
        self._at(z)
        out: np.ndarray = -self.dg
        return out


def minimize_box(
    model: BatchModel,
    box: Box,
    *,
    candidates: int = 512,
    polish: int = 4,
    seed: int = 0,
    step: float = 1e-6,
    maxiter: int = 100,
    rtol: float = 1e-4,
    feasibility_tolerance: float = 1e-6,
    extra: np.ndarray | None = None,
) -> OptResult:
    """See the module doc; ``seed`` scrambles the candidate set.

    ``extra``: further candidate points in box coordinates. A point is model-feasible when
    ``max g / scale <= feasibility_tolerance`` (SLSQP ends on active constraints to round-off).
    """
    d = box.dim
    z = qmc.Sobol(d, scramble=True, seed=seed).random(candidates)
    z = np.vstack((z, np.full((1, d), 0.5)))
    if extra is not None:  # given points (box coordinates) join the candidates
        z = np.vstack((z, np.clip(box.to_unit(np.atleast_2d(extra)), 0.0, 1.0)))
    e, g = model(box.from_unit(z))
    evaluations = len(z)
    viol = np.max(g, axis=1)
    feasible = viol <= feasibility_tolerance
    order = np.lexsort((np.where(feasible, e, np.inf), np.where(feasible, 0.0, viol)))
    stencil = _Stencil(model, box, step)
    starts: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    ends: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    for idx in order[:polish]:
        # the start itself stays a candidate (SLSQP may fail on a flat or kinked model)
        starts.append((float(e[idx]), float(viol[idx]), z[idx], g[idx]))
        res = minimize(
            stencil.f,
            z[idx],
            jac=stencil.df,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * d,
            constraints=[{"type": "ineq", "fun": stencil.c, "jac": stencil.dc}],
            options={"maxiter": maxiter, "ftol": 1e-10},
        )
        zz = np.clip(res.x, 0.0, 1.0)
        ee, gg = model(box.from_unit(zz[None, :]))
        evaluations += 1
        ends.append((float(ee[0]), float(np.max(gg[0])), zz, gg[0]))
    evaluations += stencil.calls
    pool = ends + starts
    feas = [t for t in pool if t[1] <= feasibility_tolerance]
    best = min(feas, key=lambda t: t[0]) if feas else min(pool, key=lambda t: t[1])
    polished = np.array([t[0] for t in ends if t[1] <= feasibility_tolerance])
    if feas and polished.size:
        at_best = int(np.sum(polished <= best[0] + rtol * abs(best[0])))
        spread = float(polished.max() - polished.min())
    else:
        at_best, spread = 0, float("nan")
    return OptResult(
        box.from_unit(best[2]),
        best[0],
        best[3],
        bool(best[1] <= feasibility_tolerance),
        evaluations,
        len(ends),
        at_best,
        spread,
    )


def batch_from_problem(problem: SteadyProblem, w: np.ndarray) -> BatchModel:
    """A :data:`BatchModel` of a problem at fixed ``w`` (one :meth:`evaluate` per point)."""
    scales = problem.scales

    def model(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = problem.outputs(u, w)
        return y[:, 0], y[:, 1:] / scales

    return model

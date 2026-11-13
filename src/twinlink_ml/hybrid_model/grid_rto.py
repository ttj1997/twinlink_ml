r"""Steady RTO by grid and zoom, and the per-case search of the margin (docs/39_hybrid_model.md).

Optimiser (decision of the project lead, 2026-11-03: grid + zoom instead of the 8-start
Nelder-Mead of Befund 113, which does not fit the budget with a margin search): the same for
every model and for the oracle.

1. **Multistart = coarse grid.** ``coarse x coarse`` points over the action box (inlet temperature
   x gas velocity), evaluated once per case and shared by every margin; the zoom runs from each of
   the best ``starts`` coarse points, the best end point wins.
2. **Zoom.** ``zoom_levels`` times a ``zoom x zoom`` grid over ``[-h, h]`` around the best point so
   far (``h`` = the previous spacing; the new spacing is ``h / ((zoom - 1) / 2)``), clipped to the
   box. On every level the zoom grid is recentred on its best point until that point no longer
   moves (at most ``max_moves`` times), so the search can follow the moisture constraint to the
   edge of the box (added after the pilot's discretisation check, before the main run: without
   recentring the grid stayed up to 2 % energy above the 8-start Nelder-Mead). With 21 / 9 / 3 the
   final spacing is 1/64 of the coarse one (0.047 K, 0.0013 m/s).
3. **Objective** (the penalised objective of the oracle of Befund 52/96 and of C2 in Befund 113):
   ``J = E + P (max(0, X_c - limit) / limit + max(0, T_bed - 80) / 10)``, ``P`` = 50 kW,
   ``X_c = X + k * std`` the moisture of the constraint (``k = 0`` except for the ensemble M4);
   an invalid point (no steady state, no valid fixed point) gets the solver-failure value 1e6 kW.
   Ties: the first grid index.

Margin search per case (as the per-pair bisection of Befund 111): the constraint parameter ``p``
(constant margin ``m`` with ``limit = 0.15 - m``, or ``k`` of M4 with ``limit = 0.15``) is searched
on ``[0, upper]``. A case *holds* at ``p`` iff the truth at the optimum of the model has a steady
state and moisture ``<= 0.15 + 1e-4`` (tolerance of the bandit's constraint check). ``needed = 0``
if it holds at 0; ``inf`` (reported, not an error) if it does not hold at ``upper``; else bisection
until the bracket is at most ``tolerance`` wide, ``needed`` = its holding end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from twinlink_ml.simulation.operating_point import (
    BED_TEMPERATURE_LIMIT_C,
    PENALTY_WEIGHT_KW,
    SOLVE_FAILURE_PENALTY_KW,
    TOLERANCE_MOISTURE_KG_PER_KG,
)

from .maps import SteadyMap

__all__ = [
    "BisectionSettings",
    "CaseEvaluator",
    "GridSettings",
    "SearchResult",
    "grid_zoom",
    "objective",
    "search_parameter",
    "truth_holds",
]

FIELDS = ("energy_kw", "moisture_kg_per_kg", "moisture_std_kg_per_kg", "bed_temperature_celsius")


@dataclass(frozen=True, slots=True)
class GridSettings:
    """Coarse grid size, zoom grid size and number of zoom levels."""

    coarse: int = 21
    zoom: int = 9
    zoom_levels: int = 2
    max_moves: int = 50
    starts: int = 1

    def __post_init__(self) -> None:
        if self.coarse < 2 or self.zoom < 3 or self.zoom % 2 == 0 or self.zoom_levels < 0:
            raise ValueError("coarse >= 2, zoom odd >= 3, zoom_levels >= 0")
        if self.max_moves < 1 or self.starts < 1:
            raise ValueError("max_moves >= 1, starts >= 1")


@dataclass(frozen=True, slots=True)
class BisectionSettings:
    """Search interval ``[0, upper]`` and width of the final bracket."""

    upper: float
    tolerance: float


class CaseEvaluator:
    """Cached predictions of one map at the cases of one draw.

    ``contexts (c, 2)`` and ``wide (c,)`` per case. Points are cached per case by their exact
    coordinates, so a grid point is predicted once per case however often the search visits it.
    ``max_batch``: at most that many points per call of the map (``None``: one call per request).
    It bounds the memory of the GRU fixed point (the coarse grid of five cases, 2205 points, needs
    about 9.6 GB peak commit in one call, 2.2 GB in calls of 256, with bit-identical predictions;
    check of 2026-11-05 before the rerun of the margins).
    """

    def __init__(
        self,
        smap: SteadyMap,
        contexts: np.ndarray,
        wide: np.ndarray,
        max_batch: int | None = None,
    ) -> None:
        if max_batch is not None and max_batch < 1:
            raise ValueError("max_batch >= 1")
        self.smap = smap
        self.max_batch = max_batch
        self.contexts = np.asarray(contexts, dtype=np.float64)
        self.wide = np.asarray(wide, dtype=bool)
        self._cache: list[dict[tuple[float, float], tuple[float, ...]]] = [
            {} for _ in range(len(self.contexts))
        ]
        self.n_predicted = 0

    def values(self, cases: np.ndarray, actions: np.ndarray) -> dict[str, np.ndarray]:
        """Prediction at ``actions (p, 2)`` for ``cases (p,)``; keys ``FIELDS`` and ``valid``."""
        cases = np.asarray(cases, dtype=int)
        keys = [(float(a), float(b)) for a, b in actions]
        missing = [
            i for i, (c, k) in enumerate(zip(cases, keys, strict=True)) if k not in self._cache[c]
        ]
        for wide in (False, True):
            rows = [i for i in missing if self.wide[cases[i]] == wide]
            # one request per point even if two cases share coordinates
            if not rows:
                continue
            step = self.max_batch or len(rows)
            for start in range(0, len(rows), step):
                chunk = rows[start : start + step]
                idx = np.array(chunk)
                prediction = self.smap.predict(actions[idx], self.contexts[cases[idx]], wide=wide)
                self.n_predicted += len(chunk)
                for j, i in enumerate(chunk):
                    self._cache[cases[i]][keys[i]] = (
                        *(float(getattr(prediction, f)[j]) for f in FIELDS),
                        float(prediction.valid[j]),
                    )
        out = np.array([self._cache[c][k] for c, k in zip(cases, keys, strict=True)])
        if not keys:  # empty request (e.g. a draw without inside cases): empty columns
            out = np.empty((0, len(FIELDS) + 1))
        result = {f: out[:, i] for i, f in enumerate(FIELDS)}
        result["valid"] = out[:, len(FIELDS)] > 0.5
        return result


def objective(values: dict[str, np.ndarray], limit: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Penalised objective (module docstring); ``limit`` and ``k`` per point."""
    moisture = values["moisture_kg_per_kg"] + k * values["moisture_std_kg_per_kg"]
    violation = (
        np.maximum(0.0, moisture - limit) / np.maximum(limit, 1e-6)
        + np.maximum(0.0, values["bed_temperature_celsius"] - BED_TEMPERATURE_LIMIT_C) / 10.0
    )
    j = values["energy_kw"] + PENALTY_WEIGHT_KW * violation
    return np.where(values["valid"] & np.isfinite(j), j, SOLVE_FAILURE_PENALTY_KW)


def _box(bounds: tuple[tuple[float, float], tuple[float, float]], n: int) -> np.ndarray:
    t = np.linspace(bounds[0][0], bounds[0][1], n)
    u = np.linspace(bounds[1][0], bounds[1][1], n)
    return np.array([(a, b) for a in t for b in u])


def grid_zoom(
    evaluator: CaseEvaluator,
    cases: np.ndarray,
    limits: np.ndarray,
    ks: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    settings: GridSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Optimum ``(actions (q, 2), J (q,))`` of every case for its ``limit`` and ``k``.

    With ``settings.starts`` > 1 the zoom runs from each of the best ``starts`` coarse points of a
    case (ties by grid index) and the best end point wins.
    """
    return _multistart(evaluator, np.asarray(cases, dtype=int), limits, ks, bounds, settings)


def _zoom_from(
    evaluator: CaseEvaluator,
    cases: np.ndarray,
    limits: np.ndarray,
    ks: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    settings: GridSettings,
    start: np.ndarray,
    start_j: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q = cases.size
    lower = np.array([bounds[0][0], bounds[1][0]])
    upper = np.array([bounds[0][1], bounds[1][1]])
    best = np.array(start, dtype=np.float64).copy()
    best_j = np.array(start_j, dtype=np.float64).copy()
    h = (upper - lower) / (settings.coarse - 1)
    half = (settings.zoom - 1) // 2
    offsets = np.linspace(-1.0, 1.0, settings.zoom)
    unit = np.array([(a, b) for a in offsets for b in offsets])
    z = unit.shape[0]
    for _ in range(settings.zoom_levels):
        # recentre the zoom grid on its best point until that point stays (pattern search on the
        # zoom lattice, at most ``max_moves`` moves), then refine
        active = np.arange(q)
        for _move in range(settings.max_moves):
            if active.size == 0:
                break
            points = np.clip(
                best[active, None, :] + unit[None, :, :] * h[None, None, :], lower, upper
            )
            values = evaluator.values(np.repeat(cases[active], z), points.reshape(-1, 2))
            jz = objective(values, np.repeat(limits[active], z), np.repeat(ks[active], z)).reshape(
                active.size, z
            )
            index = np.argmin(jz, axis=1)
            candidate = points[np.arange(active.size), index]
            candidate_j = jz[np.arange(active.size), index]
            better = candidate_j < best_j[active]
            best[active[better]] = candidate[better]
            best_j[active[better]] = candidate_j[better]
            active = active[better]
        h = h / half
    return best, best_j


def _multistart(
    evaluator: CaseEvaluator,
    cases: np.ndarray,
    limits: np.ndarray,
    ks: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    settings: GridSettings,
) -> tuple[np.ndarray, np.ndarray]:
    q = cases.size
    coarse = _box(bounds, settings.coarse)
    g = coarse.shape[0]
    values = evaluator.values(np.repeat(cases, g), np.tile(coarse, (q, 1)))
    j = objective(values, np.repeat(limits, g), np.repeat(ks, g)).reshape(q, g)
    order = np.argsort(j, axis=1, kind="stable")[:, : settings.starts]
    best = np.zeros((q, 2))
    best_j = np.full(q, np.inf)
    for r in range(order.shape[1]):
        start_j = j[np.arange(q), order[:, r]]
        a, jr = _zoom_from(
            evaluator, cases, limits, ks, bounds, settings, coarse[order[:, r]], start_j
        )
        better = jr < best_j
        best[better] = a[better]
        best_j[better] = jr[better]
    return best, best_j


class HoldFunction(Protocol):
    """``(case rows (r,), parameter (r,)) -> holds (r,)``: solve the RTO, evaluate the truth."""

    def __call__(self, rows: np.ndarray, parameter: np.ndarray) -> np.ndarray: ...


@dataclass
class SearchResult:
    """Per case: needed parameter (``inf`` if not reached) and the number of RTO solves."""

    needed: np.ndarray
    n_solves: np.ndarray
    trace: list[list[tuple[float, bool]]] = field(default_factory=list)


def search_parameter(
    hold_at: HoldFunction,
    n_cases: int,
    settings: BisectionSettings,
) -> SearchResult:
    """Lockstep bisection of the constraint parameter for ``n_cases`` cases (module docstring)."""
    lo = np.zeros(n_cases)
    hi = np.full(n_cases, settings.upper)
    needed = np.full(n_cases, np.nan)
    solves = np.zeros(n_cases, dtype=int)
    trace: list[list[tuple[float, bool]]] = [[] for _ in range(n_cases)]
    every = np.arange(n_cases)

    def run(rows: np.ndarray, p: np.ndarray) -> np.ndarray:
        holds = hold_at(rows, p)
        solves[rows] += 1
        for r, value, h in zip(rows, p, holds, strict=True):
            trace[r].append((float(value), bool(h)))
        return holds

    holds = run(every, lo.copy())
    needed[holds] = 0.0
    rest = every[~holds]
    if rest.size:
        holds_up = run(rest, hi[rest])
        needed[rest[~holds_up]] = np.inf
        active = rest[holds_up]
        while active.size:
            narrow = hi[active] - lo[active] <= settings.tolerance
            needed[active[narrow]] = hi[active[narrow]]
            active = active[~narrow]
            if not active.size:
                break
            mid = 0.5 * (lo[active] + hi[active])
            h = run(active, mid)
            hi[active[h]] = mid[h]
            lo[active[~h]] = mid[~h]
    return SearchResult(needed=needed, n_solves=solves, trace=trace)


def truth_holds(truth_values: dict[str, np.ndarray], limit: float) -> np.ndarray:
    """Truth has a steady state and the moisture limit holds within the bandit's tolerance."""
    return truth_values["valid"] & (
        truth_values["moisture_kg_per_kg"] <= limit + TOLERANCE_MOISTURE_KG_PER_KG
    )

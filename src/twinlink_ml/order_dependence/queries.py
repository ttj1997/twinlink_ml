r"""Deviations per query: the same steady-state queries in several orders, cold and multi-start.

A *query* is one call of :meth:`~twinlink_ml.headroom.controllers.SteadyStateOptimizer.solve`:
a disturbance vector (5) and a moisture bound. :class:`RecordingOptimizer` collects the queries a
reference controller sends to the optimiser during one run (the caches in front of it pass every
distinct query once), in the order in which they arrive (*natural order*).

:func:`measure` then solves the same queries

* **warm**, the existing behaviour (every solve starts from the previous solution), with a fresh
  optimiser per pass, in the natural order, the reversed order and seeded random orders;
* **cold**, one start from the default guess per query (``cold_starts = 1``);
* **multi-start**, the proposed switch (``cold_starts = n``, :class:`OrderDependenceSettings`);
* **reference**, every start of a dense start set on its own; the reference power is the smallest
  power among the strictly converged starts, and the starts are clustered into local optima by
  their set points.

All relative differences are taken against the multi-start power :math:`P_m` of the query:
:math:`(P - P_m) / P_m`; positive means *worse than the multi-start* (more power at the same
bound).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from twinlink_ml.headroom.controllers import SteadyPoint, SteadyStateOptimizer

from .settings import OrderDependenceSettings

__all__ = [
    "Query",
    "RecordingOptimizer",
    "measure",
    "order_indices",
    "summarise",
    "warm_pass",
]

_STRICT_STATUS = "Solve_Succeeded"


@dataclass(frozen=True, slots=True)
class Query:
    """One call of the steady-state optimiser."""

    disturbance: np.ndarray
    bound: float


class RecordingOptimizer(SteadyStateOptimizer):
    """The unchanged optimiser that also remembers every query it is asked, in order."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.queries: list[Query] = []
        self.results: list[SteadyPoint | None] = []

    def solve(self, disturbance: np.ndarray, moisture_bound: float) -> SteadyPoint:
        """Record the query, then solve it as the parent does."""
        self.queries.append(
            Query(np.array(disturbance, dtype=np.float64, copy=True), float(moisture_bound))
        )
        try:
            point = super().solve(disturbance, moisture_bound)
        except RuntimeError:
            self.results.append(None)
            raise
        self.results.append(point)
        return point


def order_indices(n: int, settings: OrderDependenceSettings) -> dict[str, np.ndarray]:
    """Index sequences of the warm passes: natural, reversed and the seeded permutations."""
    out = {"natural": np.arange(n), "reversed": np.arange(n)[::-1]}
    for seed in settings.permutation_seeds:
        out[f"permutation_{seed}"] = np.random.default_rng(seed).permutation(n)
    return out


def warm_pass(
    optimizer: SteadyStateOptimizer, queries: Sequence[Query], order: Iterable[int]
) -> list[SteadyPoint | None]:
    """Solve ``queries`` in ``order``; results by query index (``None`` where the solve failed)."""
    out: list[SteadyPoint | None] = [None] * len(queries)
    for i in order:
        query = queries[int(i)]
        try:
            out[int(i)] = optimizer.solve(query.disturbance, query.bound)
        except RuntimeError:
            out[int(i)] = None
    return out


def _columns(points: Sequence[SteadyPoint | None]) -> np.ndarray:
    """``(n, 3)`` power, inlet temperature, gas velocity; ``nan`` where the solve failed."""
    out = np.full((len(points), 3), np.nan)
    for i, point in enumerate(points):
        if point is not None:
            out[i] = (point.power_kw, point.command[0], point.command[1])
    return out


def _reference(
    optimizer: SteadyStateOptimizer, query: Query, settings: OrderDependenceSettings
) -> tuple[float, float, float, int, int, float]:
    """Best strictly converged start of the dense start set and the clusters of local optima.

    Returns power, inlet temperature and gas velocity of the best start, the number of local
    optima (clusters of the set points), the number of failed starts and the relative power
    spread between the best and the worst cluster.
    """
    strict: list[tuple[float, float, float]] = []
    loose: list[tuple[float, float, float]] = []
    failed = 0
    for guess in optimizer.start_points:
        point, status = optimizer.solve_from(query.disturbance, query.bound, guess)
        if point is None:
            failed += 1
            continue
        row = (point.power_kw, float(point.command[0]), float(point.command[1]))
        (strict if status == _STRICT_STATUS else loose).append(row)
    rows = strict or loose
    if not rows:
        return float("nan"), float("nan"), float("nan"), 0, failed, float("nan")
    best = min(rows)
    clusters: dict[tuple[int, int], float] = {}
    for power, inlet, gas in rows:
        key = (
            round(inlet / settings.same_optimum_inlet_k),
            round(gas / settings.same_optimum_gas_m_per_s),
        )
        clusters[key] = min(power, clusters.get(key, np.inf))
    powers = np.array(list(clusters.values()))
    spread = float((powers.max() - powers.min()) / abs(powers.min()))
    return best[0], best[1], best[2], len(clusters), failed, spread


def measure(
    queries: Sequence[Query],
    factory: Callable[[int, int], SteadyStateOptimizer],
    settings: OrderDependenceSettings,
) -> dict[str, np.ndarray]:
    """All passes over ``queries``; ``factory(cold_starts, seed)`` builds an optimiser.

    Returns arrays by query index: ``warm`` ``(orders, n, 3)`` (power, inlet temperature, gas
    velocity per warm pass), ``cold`` ``(n, 3)``, ``multi`` ``(n, 3)``, ``reference`` ``(n, 3)``,
    ``reference_optima`` ``(n,)``, ``reference_failed`` ``(n,)``, ``reference_spread`` ``(n,)``.
    """
    n = len(queries)
    orders = order_indices(n, settings)
    passes = []
    for name in settings.orders:
        fresh = factory(0, settings.start_seed)
        passes.append(_columns(warm_pass(fresh, queries, orders[name])))
    warm = np.stack(passes)
    natural = np.arange(n)
    cold = _columns(warm_pass(factory(1, settings.start_seed), queries, natural))
    multi = _columns(
        warm_pass(factory(settings.cold_starts, settings.start_seed), queries, natural)
    )
    dense = factory(settings.reference_starts, settings.reference_seed)
    reference = np.full((n, 3), np.nan)
    optima = np.zeros(n, dtype=np.int64)
    failed = np.zeros(n, dtype=np.int64)
    spread = np.full(n, np.nan)
    for i, query in enumerate(queries):
        power, inlet, gas, optima[i], failed[i], spread[i] = _reference(dense, query, settings)
        reference[i] = (power, inlet, gas)
    return {
        "warm": warm,
        "cold": cold,
        "multi": multi,
        "reference": reference,
        "reference_optima": optima,
        "reference_failed": failed,
        "reference_spread": spread,
    }


def _digits(values: np.ndarray, digits: int) -> list[float | None]:
    """Rounded to significant digits for the record; ``None`` for ``nan``."""
    return [None if not np.isfinite(v) else float(f"{v:.{digits}g}") for v in values]


def summarise(
    measured: dict[str, np.ndarray],
    natural_recorded: Sequence[SteadyPoint | None],
    settings: OrderDependenceSettings,
) -> dict[str, Any]:
    """Per-query columns of one unit for the artifact (compact, rounded) and the control.

    ``natural_recorded``: the results the recording run itself obtained; the warm pass in the
    natural order on a fresh optimiser must reproduce them bit for bit (``natural_reproduced``).

    Columns (lists over the queries of the unit, ``None`` where a solve failed):
    ``power_multi_kw``; ``rel_warm`` (one list per order), ``rel_cold``, ``rel_reference``:
    relative power difference to the multi-start; ``d_inlet_k`` and ``d_gas_m_per_s``: set points
    of the natural warm pass minus those of the multi-start; ``d_inlet_orders_k`` and
    ``d_gas_orders_m_per_s``: largest difference of a set point between two warm passes;
    ``distinct_bits``: number of bit-distinct results among the warm passes (1 = all equal);
    ``reference_optima``, ``reference_failed``, ``reference_spread``.
    """
    warm, multi = measured["warm"], measured["multi"]
    base = multi[:, 0]
    recorded = _columns(natural_recorded)
    reproduced = bool(np.array_equal(recorded, warm[0], equal_nan=True))
    distinct = [len({row.tobytes() for row in warm[:, i, :]}) for i in range(warm.shape[1])]
    with np.errstate(invalid="ignore"):
        inlet_range = np.nanmax(warm[:, :, 1], axis=0) - np.nanmin(warm[:, :, 1], axis=0)
        gas_range = np.nanmax(warm[:, :, 2], axis=0) - np.nanmin(warm[:, :, 2], axis=0)
    return {
        "n_queries": int(warm.shape[1]),
        "natural_reproduced": reproduced,
        "orders": list(settings.orders),
        "power_multi_kw": _digits(base, 9),
        "rel_warm": [_digits((warm[k, :, 0] - base) / base, 4) for k in range(warm.shape[0])],
        "rel_cold": _digits((measured["cold"][:, 0] - base) / base, 4),
        "rel_reference": _digits((measured["reference"][:, 0] - base) / base, 4),
        "d_inlet_k": _digits(warm[0, :, 1] - multi[:, 1], 4),
        "d_gas_m_per_s": _digits(warm[0, :, 2] - multi[:, 2], 4),
        "d_inlet_orders_k": _digits(inlet_range, 4),
        "d_gas_orders_m_per_s": _digits(gas_range, 4),
        "distinct_bits": distinct,
        "reference_optima": [int(v) for v in measured["reference_optima"]],
        "reference_failed": [int(v) for v in measured["reference_failed"]],
        "reference_spread": _digits(measured["reference_spread"], 4),
    }

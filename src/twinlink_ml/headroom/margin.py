"""Smallest moisture margin that keeps the path condition, by bisection.

The predicate is "the path condition holds", judged on the truth model, for a margin ``m``. It
is assumed monotone in ``m`` (a larger margin plans drier); the result is the upper end of the
final bracket, i.e. a margin that was EVALUATED as feasible, never an interpolated one.

:func:`bisect_margin` works on any monotone predicate. :func:`bisect_margin_over_scenarios` is the
version for expensive controllers: it bisects on a critical subset of scenarios (those that
violated at the last evaluation) and verifies the result on all scenarios; a scenario that fails
the verification joins the critical set and the bisection continues above the failed value. The
verification run is part of the result, so the final margin always holds on every scenario.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "MarginSearch",
    "bisect_margin",
    "bisect_margin_k_of_n",
    "bisect_margin_over_scenarios",
]


@dataclass
class MarginSearch:
    """Result of a margin search."""

    margin: float
    lower: float
    evaluations: list[tuple[float, bool]] = field(default_factory=list)
    converged: bool = True

    @property
    def bracket(self) -> tuple[float, float]:
        """Last infeasible and first feasible margin."""
        return self.lower, self.margin


def bisect_margin(
    feasible: Callable[[float], bool],
    *,
    lower: float = 0.0,
    upper: float,
    tolerance: float,
    max_upper: float | None = None,
    lower_infeasible: bool = False,
) -> MarginSearch:
    """Smallest feasible margin in ``[lower, upper]`` to within ``tolerance``.

    If ``lower`` itself is feasible it is returned (not checked when ``lower_infeasible``). If
    ``upper`` is infeasible the bracket is doubled up to ``max_upper``; beyond that
    ``converged=False`` is returned with the largest margin tried.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    evaluations: list[tuple[float, bool]] = []

    def check(m: float) -> bool:
        ok = bool(feasible(m))
        evaluations.append((m, ok))
        return ok

    if not lower_infeasible and check(lower):
        return MarginSearch(lower, lower, evaluations)
    cap = upper if max_upper is None else max_upper
    while not check(upper):
        if upper >= cap:
            return MarginSearch(upper, upper, evaluations, converged=False)
        lower, upper = upper, min(2.0 * upper, cap)
    while upper - lower > tolerance:
        middle = 0.5 * (lower + upper)
        if check(middle):
            upper = middle
        else:
            lower = middle
    return MarginSearch(upper, lower, evaluations)


def bisect_margin_over_scenarios(
    violations: Callable[[float, Sequence[int]], np.ndarray],
    n_scenarios: int,
    *,
    tolerance_violation: float,
    lower: float = 0.0,
    upper: float,
    tolerance: float,
    max_upper: float | None = None,
    initial_critical: Sequence[int] | None = None,
) -> tuple[MarginSearch, np.ndarray]:
    """Margin search with a critical subset, see the module docstring.

    ``violations(m, ids)`` returns the largest violation (kg/kg) of each scenario in ``ids`` at
    margin ``m``; a scenario holds if its violation is ``<= tolerance_violation``. Returns the
    search and the violations of ALL scenarios at the returned margin.
    """
    all_ids = list(range(n_scenarios))
    if initial_critical is None:
        first = violations(lower, all_ids)
        if np.all(first <= tolerance_violation):
            return MarginSearch(lower, lower, [(lower, True)]), first
        critical = sorted(int(i) for i in np.flatnonzero(first > tolerance_violation))
    else:
        critical = sorted(int(i) for i in initial_critical)
    history: list[tuple[float, bool]] = []
    start = lower
    # with seeded critical scenarios the lower end is untested and may already be feasible
    lower_known_infeasible = initial_critical is None
    while True:

        def subset_feasible(m: float, ids: list[int] = critical) -> bool:
            return bool(np.all(violations(m, ids) <= tolerance_violation))

        search = bisect_margin(
            subset_feasible,
            lower=start,
            upper=max(upper, start + tolerance),
            tolerance=tolerance,
            max_upper=max_upper,
            lower_infeasible=lower_known_infeasible,
        )
        history.extend(search.evaluations)
        final = violations(search.margin, all_ids)
        failed = [int(i) for i in np.flatnonzero(final > tolerance_violation)]
        history.append((search.margin, not failed))
        if not failed or not search.converged:
            return (
                MarginSearch(search.margin, search.lower, history, converged=search.converged),
                final,
            )
        critical = sorted(set(critical) | set(failed))
        start = search.margin
        lower_known_infeasible = True


def bisect_margin_k_of_n(
    holds: Callable[[float, Sequence[int]], np.ndarray],
    n_draws: int,
    k_required: int,
    ranking: Sequence[int],
    *,
    lower: float,
    upper: float,
    tolerance: float,
    max_upper: float | None = None,
) -> tuple[MarginSearch, np.ndarray]:
    """Smallest margin at which at least ``k_required`` of ``n_draws`` hold.

    ``holds(m, ids)`` returns, per draw in ``ids``, whether the path condition holds at margin
    ``m``. The bisection runs on a critical set -- the ``n - k + 1`` draws ranked most demanding
    by ``ranking`` (e.g. by a cheaper controller's per-draw margin) -- where "at most ``n - k``
    failures" is necessary for the criterion on all draws; the result is verified on all draws,
    and draws failing the verification join the critical set. Returns the search and the holds of
    all draws at the returned margin.
    """
    allowed = n_draws - k_required
    if allowed < 0:
        raise ValueError("k_required exceeds n_draws")
    critical = sorted(int(i) for i in ranking[: allowed + 1])
    history: list[tuple[float, bool]] = []
    start = lower
    lower_known_infeasible = False
    while True:

        def subset_feasible(m: float, ids: list[int] = critical) -> bool:
            return int(np.sum(~np.asarray(holds(m, ids), dtype=bool))) <= allowed

        search = bisect_margin(
            subset_feasible,
            lower=start,
            upper=max(upper, start + tolerance),
            tolerance=tolerance,
            max_upper=max_upper,
            lower_infeasible=lower_known_infeasible,
        )
        history.extend(search.evaluations)
        final = np.asarray(holds(search.margin, list(range(n_draws))), dtype=bool)
        ok = int(np.sum(~final)) <= allowed
        history.append((search.margin, ok))
        if ok or not search.converged:
            return (
                MarginSearch(search.margin, search.lower, history, converged=search.converged),
                final,
            )
        critical = sorted(set(critical) | {int(i) for i in np.flatnonzero(~final)})
        start = search.margin
        lower_known_infeasible = True

r"""Per-pair margin bisection in synchronised rounds, and the margins of the two criteria.

:class:`PairBisection` is the step-by-step form of :func:`twinlink_ml.headroom.margin.bisect_margin`
(same order of evaluations, same result; tested): ``next_margin`` names the margin to evaluate,
``report`` feeds back whether the path condition held there. All pairs of a variant advance one
evaluation per round, so a round is one batch of closed-loop runs.

From the per-pair smallest margins :math:`m_p` (``inf`` where no margin up to the cap holds), the
variant margin of criterion (i) "all pairs hold" is :math:`\max_p m_p` and that of criterion (ii)
"at least a share ``q`` of the pairs hold" is the :math:`\lceil q N \rceil`-th smallest
:math:`m_p`. Both assume that holding is monotone in the margin; the driver therefore verifies
each on all pairs (:func:`next_candidate` raises a failed candidate to the next order statistic).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["PairBisection", "criterion_margin", "next_candidate", "required_count"]


@dataclass
class PairBisection:
    """Bisection of one pair; mirrors ``bisect_margin`` (lower checked first, bracket doubled)."""

    lower: float
    upper: float
    tolerance: float
    max_upper: float
    phase: str = "lower"
    margin: float | None = None
    converged: bool = True
    evaluations: list[tuple[float, bool]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")

    @property
    def done(self) -> bool:
        return self.phase == "done"

    def next_margin(self) -> float:
        """Margin of the next evaluation."""
        if self.phase == "lower":
            return self.lower
        if self.phase == "expand":
            return self.upper
        if self.phase == "bisect":
            return 0.5 * (self.lower + self.upper)
        raise RuntimeError("search finished")

    def report(self, holds: bool) -> None:
        """Result of the evaluation at :meth:`next_margin`."""
        m = self.next_margin()
        self.evaluations.append((m, bool(holds)))
        if self.phase == "lower":
            if holds:
                self._finish(self.lower)
            else:
                self.phase = "expand"
        elif self.phase == "expand":
            if holds:
                self.phase = "bisect"
                self._maybe_finish()
            elif self.upper >= self.max_upper:
                self._finish(self.upper, converged=False)
            else:
                self.lower, self.upper = self.upper, min(2.0 * self.upper, self.max_upper)
        else:
            if holds:
                self.upper = m
            else:
                self.lower = m
            self._maybe_finish()

    def _maybe_finish(self) -> None:
        if self.upper - self.lower <= self.tolerance:
            self._finish(self.upper)

    def _finish(self, margin: float, *, converged: bool = True) -> None:
        self.margin = margin
        self.converged = converged
        self.phase = "done"

    @property
    def smallest_holding(self) -> float:
        """The pair's smallest margin (``inf`` if none up to the cap holds)."""
        if not self.done:
            raise RuntimeError("search not finished")
        assert self.margin is not None
        return self.margin if self.converged else math.inf


def required_count(n_pairs: int, share: float) -> int:
    """Pairs that must hold for criterion (ii)."""
    return math.ceil(share * n_pairs - 1e-12)


def criterion_margin(per_pair: np.ndarray, k_required: int) -> float:
    """Smallest variant margin at which (under monotonicity) ``k_required`` pairs hold."""
    ordered = np.sort(np.asarray(per_pair, dtype=np.float64))
    if not 1 <= k_required <= ordered.size:
        raise ValueError("k_required out of range")
    return float(ordered[k_required - 1])


def next_candidate(per_pair: np.ndarray, failed: float) -> float:
    """The smallest per-pair margin above a candidate that failed its verification."""
    values = np.unique(np.asarray(per_pair, dtype=np.float64))
    above = values[values > failed]
    return float(above[0]) if above.size else math.inf

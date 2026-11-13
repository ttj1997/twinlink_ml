"""Best case, ties, regret and ranks over many points (array functions).

``fuel`` is always an array of shape ``(n, 7)``: column ``k`` = case ``k`` (column 0 = case 0, no
recovery), kg/h. Ties (pre-registered, decision of the project lead F4):

* exact: two fuels differ by less than ``exact`` (1e-9 kg/h, docs/11 section 13),
* practical: the gap is smaller than ``fraction`` of the saving of the best case against case 0
  (never below ``exact``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta

__all__ = [
    "ShareEstimate",
    "best_fuel",
    "case_one_flags",
    "clopper_pearson",
    "competition_ranks",
    "practical_threshold",
    "regret",
    "share",
    "top_two_gap",
    "winner_labels",
]


def _cases(fuel: np.ndarray) -> np.ndarray:
    fuel = np.asarray(fuel, dtype=np.float64)
    if fuel.ndim != 2 or fuel.shape[1] != 7:
        raise ValueError("fuel must have shape (n, 7): case 0..6")
    return fuel[:, 1:]


def best_fuel(fuel: np.ndarray) -> np.ndarray:
    """Lowest fuel of cases 1-6 per point."""
    return _cases(fuel).min(axis=1)


def practical_threshold(fuel: np.ndarray, fraction: float, exact: float) -> np.ndarray:
    """``max(fraction * (F0 - F_best), exact)`` per point, kg/h."""
    saving = np.asarray(fuel)[:, 0] - best_fuel(fuel)
    return np.maximum(fraction * saving, exact)


def case_one_flags(fuel: np.ndarray, exact: float, fraction: float) -> dict[str, np.ndarray]:
    """Per point: case 1 strictly best / best or exactly tied / best or practically equal."""
    cases = _cases(fuel)
    f1 = cases[:, 0]
    others = cases[:, 1:].min(axis=1)
    gap = f1 - cases.min(axis=1)
    return {
        "strict": others - f1 >= exact,
        "co_best": gap < exact,
        "practical": gap < practical_threshold(fuel, fraction, exact),
    }


def regret(fuel: np.ndarray) -> dict[str, np.ndarray]:
    """Extra fuel of always choosing case 1: absolute, against the best fuel, against the saving."""
    f1 = np.asarray(fuel)[:, 1]
    best = best_fuel(fuel)
    saving = np.asarray(fuel)[:, 0] - best
    r = f1 - best
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_saving = np.where(saving > 0.0, r / saving, np.nan)
    return {"kg_per_h": r, "relative_to_best": r / best, "relative_to_best_saving": rel_saving}


def competition_ranks(fuel: np.ndarray, exact: float) -> np.ndarray:
    """Rank of cases 1-6 per point (1 = best; exact ties share the better rank), shape (n, 6)."""
    cases = _cases(fuel)
    diff = cases[:, None, :] - cases[:, :, None]  # [i, k, j] = F_j - F_k
    better = (diff <= -exact).sum(axis=2)  # number of cases j with F_j <= F_k - exact
    return 1 + better


def winner_labels(fuel: np.ndarray, exact: float) -> np.ndarray:
    """``"1"``, ``"1=2"``, ... -- the set of cases within ``exact`` of the best, per point."""
    cases = _cases(fuel)
    best = cases.min(axis=1, keepdims=True)
    tied = cases - best < exact
    return np.array(["=".join(str(k + 1) for k in np.flatnonzero(row)) for row in tied])


def top_two_gap(fuel: np.ndarray) -> np.ndarray:
    """Second-lowest minus lowest fuel of cases 1-6 (kg/h)."""
    s = np.sort(_cases(fuel), axis=1)
    return s[:, 1] - s[:, 0]


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact binomial confidence interval of a share ``k / n``."""
    if n == 0:
        return (float("nan"), float("nan"))
    a = (1.0 - level) / 2.0
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - a, k + 1, n - k))
    return lo, hi


@dataclass(frozen=True)
class ShareEstimate:
    """A share ``k / n`` with its exact 95 % interval."""

    k: int
    n: int
    share: float
    ci95: tuple[float, float]

    def as_dict(self) -> dict[str, object]:
        return {"k": self.k, "n": self.n, "share": self.share, "ci95": list(self.ci95)}


def share(flags: np.ndarray) -> ShareEstimate:
    """Share of ``True`` in ``flags`` with the Clopper-Pearson interval."""
    flags = np.asarray(flags, dtype=bool)
    k, n = int(flags.sum()), int(flags.size)
    return ShareEstimate(k, n, k / n if n else float("nan"), clopper_pearson(k, n))

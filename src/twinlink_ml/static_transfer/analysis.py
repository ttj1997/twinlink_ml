"""Paired verdicts of the static transfer study (reading: ``scripts/analyze_static_transfer.py``).

Metric: RMSE in physical units against the noise-free plant truth, one value per seed. A
comparison of a candidate with a reference is the paired t-test of
:func:`twinlink_ml.evaluation.paired_resolution.paired_comparison` (two-sided, ``n - 1`` degrees
of freedom); the family-wise correction is Holm
(:func:`twinlink_ml.evaluation.b1_analysis.holm`) over all tests of ONE hypothesis family.

"Resolved better" = Holm-rejected AND the candidate better in at least ``min_sign`` of the seeds
(same sign); "resolved worse" likewise in the other direction. The uncorrected verdict (p below
alpha and the same sign rule) is reported beside it. Effect size = median over seeds of the paired
difference ``reference - candidate`` (positive: candidate better), in the unit of the target.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from twinlink_ml.evaluation.b1_analysis import holm
from twinlink_ml.evaluation.paired_resolution import paired_comparison

from .config import TARGET_NAMES

__all__ = [
    "Comparison",
    "Family",
    "MetricTable",
    "compare",
    "smallest_n_from_which",
]


class MetricTable:
    """RMSE per (plant, condition, method, N, test set, target) -> {seed: value}.

    Methods without plant data (L0, L0_v412, L3_0) are stored once and returned for every N.
    """

    def __init__(self, records: Iterable[Mapping[str, Any]], metric: str = "rmse") -> None:
        self._fixed: dict[tuple[str, str, str, str, str], dict[int, float]] = {}
        self._fitted: dict[tuple[str, str, str, int, str, str], dict[int, float]] = {}
        for rec in records:
            plant, cond, seed = rec["plant"], rec["condition"], int(rec["seed"])
            for method, sets in rec["fixed"].items():
                for test_set, err in sets.items():
                    for j, target in enumerate(TARGET_NAMES):
                        key = (plant, cond, method, test_set, target)
                        self._fixed.setdefault(key, {})[seed] = float(err[metric][j])
            for n, methods in rec["fitted"].items():
                for method, payload in methods.items():
                    for test_set, err in payload["test"].items():
                        for j, target in enumerate(TARGET_NAMES):
                            key2 = (plant, cond, method, int(n), test_set, target)
                            self._fitted.setdefault(key2, {})[seed] = float(err[metric][j])

    def values(
        self, plant: str, condition: str, method: str, n: int, test_set: str, target: str
    ) -> dict[int, float]:
        fixed = self._fixed.get((plant, condition, method, test_set, target))
        if fixed is not None:
            return dict(fixed)
        return dict(self._fitted[(plant, condition, method, n, test_set, target)])

    def summary(
        self, plant: str, condition: str, method: str, n: int, test_set: str, target: str
    ) -> dict[str, float]:
        v = np.asarray(list(self.values(plant, condition, method, n, test_set, target).values()))
        return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


@dataclass
class Comparison:
    """One paired comparison (candidate against reference) at one cell."""

    candidate: str
    reference: str
    plant: str
    condition: str
    n: int
    test_set: str
    target: str
    p_value: float
    mean_difference: float
    median_difference: float
    seeds_better: int
    seeds_worse: int
    n_seeds: int
    holm_rejected: bool = False
    verdict: str = "unresolved"
    verdict_uncorrected: str = "unresolved"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare(
    table: MetricTable,
    candidate: str,
    reference: str,
    plant: str,
    condition: str,
    n: int,
    test_set: str,
    target: str,
) -> Comparison:
    """Paired comparison of two methods at one cell (verdict set by the :class:`Family`)."""
    cand = table.values(plant, condition, candidate, n, test_set, target)
    ref = table.values(plant, condition, reference, n, test_set, target)
    result = paired_comparison(cand, ref, candidate_label=candidate, reference_label=reference)
    diffs = np.asarray([ref[s] - cand[s] for s in sorted(cand)])
    return Comparison(
        candidate=candidate,
        reference=reference,
        plant=plant,
        condition=condition,
        n=n,
        test_set=test_set,
        target=target,
        p_value=float(result.p_value),
        mean_difference=float(result.mean_difference),
        median_difference=float(np.median(diffs)),
        seeds_better=int(np.sum(diffs > 0.0)),
        seeds_worse=int(np.sum(diffs < 0.0)),
        n_seeds=len(diffs),
    )


def _verdict(c: Comparison, significant: bool, min_sign: int) -> str:
    if not significant:
        return "unresolved"
    if c.mean_difference > 0.0 and c.seeds_better >= min_sign:
        return "better"
    if c.mean_difference < 0.0 and c.seeds_worse >= min_sign:
        return "worse"
    return "unresolved"


@dataclass
class Family:
    """A hypothesis family: its comparisons get one Holm correction."""

    name: str
    comparisons: list[Comparison]
    alpha: float
    min_sign: int

    def __post_init__(self) -> None:
        rejected = holm([c.p_value for c in self.comparisons], self.alpha)
        for c, rej in zip(self.comparisons, rejected, strict=True):
            c.holm_rejected = bool(rej)
            c.verdict = _verdict(c, bool(rej), self.min_sign)
            uncorrected = not np.isnan(c.p_value) and c.p_value < self.alpha
            c.verdict_uncorrected = _verdict(c, uncorrected, self.min_sign)

    @property
    def size(self) -> int:
        return len(self.comparisons)

    def get(self, **where: Any) -> list[Comparison]:
        return [c for c in self.comparisons if all(getattr(c, k) == v for k, v in where.items())]

    def all_better(self, **where: Any) -> bool:
        cells = self.get(**where)
        return bool(cells) and all(c.verdict == "better" for c in cells)

    def any_verdict(self, verdict: str, **where: Any) -> bool:
        return any(c.verdict == verdict for c in self.get(**where))

    def counts(self) -> dict[str, int]:
        out = {"better": 0, "worse": 0, "unresolved": 0}
        for c in self.comparisons:
            out[c.verdict] += 1
        out_unc = {"better": 0, "worse": 0, "unresolved": 0}
        for c in self.comparisons:
            out_unc[c.verdict_uncorrected] += 1
        return {**out, **{f"uncorrected_{k}": v for k, v in out_unc.items()}}

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "alpha": self.alpha,
            "min_sign_agreement": self.min_sign,
            "counts": self.counts(),
            "comparisons": [c.as_dict() for c in self.comparisons],
        }


def smallest_n_from_which(budgets: Sequence[int], holds: Mapping[int, bool]) -> int | None:
    """Smallest N from which ``holds`` is true at that N and every larger one (else None)."""
    result: int | None = None
    for n in sorted(budgets, reverse=True):
        if holds.get(n, False):
            result = n
        else:
            break
    return result


def affected(targets: Sequence[str]) -> list[str]:
    """The given targets in the column order of :data:`TARGET_NAMES`."""
    return [t for t in TARGET_NAMES if t in targets]

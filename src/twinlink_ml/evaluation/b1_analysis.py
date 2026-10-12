r"""Analysis of the B1 main run: the criteria of HB1-HB3 as fixed in ``docs/`` before the run.

All comparisons over seeds use :func:`twinlink_ml.evaluation.paired_resolution.paired_comparison`
(paired t over master seeds, ``n - 1`` degrees of freedom, alpha = 0.05, two-sided; a candidate is
better when its metric is lower). The metric is the run-median RMSE in physical units, the primary
target is the product moisture at horizon 24; every table also reports the value relative to
persistence **with the test set named** (CLAUDE.md).

Criteria (identical to the draft, nothing is tuned here)
-------------------------------------------------------
HB1  LSTM, moisture, h = 24, T3. Each of B2, B3, B4 against B1 at N = 4, Holm over the three tests.
     Supported for X: Holm-resolved better at N = 4 AND mean difference better at N = 2 and N = 8.
     Refuted: for every X the mean difference is not better at any of N = 2, 4, 8.
     Otherwise not established.
HB2  LSTM, moisture, h = 24, on T3 and on T1 separately, B5 against B5r.
     Supported: at N = 8 or 16 B5 is resolved better AND the runs B5r needs to reach the median
     error of B5 at that N (log-log interpolation of the median curve of B5r) are at least 1.5
     times N.
     Refuted: no resolved difference in favour of B5 at N = 4, 8, 16, 32 AND the median of B5 is not
     below the median of B5r at N = 8 and 16. Otherwise not established.
HB3  B0-B4, LSTM, moisture, h = 24, at N = 4, 8, 16, pairwise on T1, T2, T3.
     Supported: a resolved reversal (X resolved better than Y on one test set, Y resolved better
     than X on another, at the same N). Refuted: no reversal at any of the three N and the rank
     orders of the five arms by the median differ by at most one position per arm between every
     two test sets.
     Otherwise not established.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from .paired_resolution import PairedComparison, paired_comparison

__all__ = [
    "PRIMARY_METRIC",
    "RecordTable",
    "hb1",
    "hb2",
    "hb3",
    "holm",
    "load_records",
    "relative_to_persistence",
    "runs_needed",
]

PRIMARY_METRIC = "rmse_run_median_physical"
PRIMARY_TARGET = "product_moisture_kg_per_kg"
PRIMARY_HORIZON = 24
PLANNED = ("b2", "b3", "b4")


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Records of the jobs that have a ``job_done`` marker (partial jobs are ignored)."""
    lines = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    done = {(r["arm"], r["master_seed"]) for r in lines if r.get("type") == "job_done"}
    return [
        r
        for r in lines
        if r.get("type") in ("model", "information", "selection")
        and (r["arm"], r["master_seed"]) in done
    ]


class RecordTable:
    """Metric values indexed by arm, N, model, test set, target and horizon."""

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self._values: dict[tuple, dict[int, float]] = defaultdict(dict)
        for record in records:
            if record.get("type") != "model":
                continue
            for test_set, metrics in record["metrics"].items():
                for key, value in metrics.items():
                    metric, target, horizon = key.split("/")
                    self._values[
                        (
                            record["arm"],
                            record["n_train"],
                            record["model"],
                            test_set,
                            metric,
                            target,
                            int(horizon[1:]),
                        )
                    ][record["master_seed"]] = float(value)

    def values(
        self,
        arm: str,
        n_train: int,
        model: str,
        test_set: str,
        *,
        target: str = PRIMARY_TARGET,
        horizon: int = PRIMARY_HORIZON,
        metric: str = PRIMARY_METRIC,
    ) -> dict[int, float]:
        """Metric per seed; empty if there is none."""
        return dict(self._values.get((arm, n_train, model, test_set, metric, target, horizon), {}))

    def median(self, *args: Any, **kwargs: Any) -> float:
        values = self.values(*args, **kwargs)
        return float(np.median(list(values.values()))) if values else math.nan

    def arms(self) -> list[str]:
        return sorted({key[0] for key in self._values})

    def sizes(self, arm: str) -> list[int]:
        return sorted({key[1] for key in self._values if key[0] == arm})


def holm(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down: which hypotheses are rejected (NaN never is)."""
    order = sorted(range(len(p_values)), key=lambda i: (math.isnan(p_values[i]), p_values[i]))
    rejected = [False] * len(p_values)
    for rank, index in enumerate(order):
        p = p_values[index]
        if math.isnan(p) or p > alpha / (len(p_values) - rank):
            break
        rejected[index] = True
    return rejected


def runs_needed(curve: Mapping[int, float], error: float) -> float:
    """Runs a learning curve needs to reach ``error`` (log-log interpolation).

    Returns ``inf`` when the curve never gets that low within its range and ``nan`` for fewer than
    two points. Below the first point the first N is returned.
    """
    points = sorted((n, v) for n, v in curve.items() if v > 0 and not math.isnan(v))
    if len(points) < 2:
        return math.nan
    if error >= points[0][1]:
        return float(points[0][0])
    for (n0, v0), (n1, v1) in itertools.pairwise(points):
        if v1 <= error <= v0:
            if v0 == v1:
                return float(n0)
            fraction = (math.log(v0) - math.log(error)) / (math.log(v0) - math.log(v1))
            return float(math.exp(math.log(n0) + fraction * (math.log(n1) - math.log(n0))))
    return math.inf


def relative_to_persistence(
    table: RecordTable,
    arm: str,
    n_train: int,
    model: str,
    test_set: str,
    *,
    target: str = PRIMARY_TARGET,
    horizon: int = PRIMARY_HORIZON,
) -> float:
    """Median metric of ``model`` divided by the median of persistence on the same test set.

    Below 1 the model beats persistence on that test set. Persistence is taken from the same arm and
    N (it does not depend on the training data, so any arm gives the same value).
    """
    reference = table.median(arm, n_train, "persistence", test_set, target=target, horizon=horizon)
    value = table.median(arm, n_train, model, test_set, target=target, horizon=horizon)
    return value / reference if reference else math.nan


def _compare(
    table: RecordTable,
    candidate: str,
    reference: str,
    n_train: int,
    test_set: str,
    model: str,
    **kwargs: Any,
) -> PairedComparison | None:
    cand = table.values(candidate, n_train, model, test_set, **kwargs)
    ref = table.values(reference, n_train, model, test_set, **kwargs)
    if not cand or set(cand) != set(ref):
        return None
    return paired_comparison(cand, ref, candidate_label=candidate, reference_label=reference)


def _summary(comparison: PairedComparison | None) -> dict[str, Any] | None:
    if comparison is None:
        return None
    return {
        "statement": comparison.statement,
        "mean_difference": comparison.mean_difference,
        "threshold": comparison.threshold,
        "p_value": comparison.p_value,
        "resolved": comparison.is_resolved,
        "n_pairs": comparison.n_pairs,
    }


def _better(comparison: PairedComparison | None) -> bool:
    return comparison is not None and comparison.mean_difference > 0.0


def hb1(table: RecordTable, *, model: str = "lstm", reference: str = "b1") -> dict[str, Any]:
    """HB1, see module docstring."""
    comparisons = {
        x: {n: _compare(table, x, reference, n, "T3", model) for n in (2, 4, 8)} for x in PLANNED
    }
    p_values = [
        comparisons[x][4].p_value if comparisons[x][4] is not None else math.nan for x in PLANNED
    ]
    rejected = holm(p_values)
    per_design = {}
    for x, reject in zip(PLANNED, rejected, strict=True):
        primary = comparisons[x][4]
        supported = bool(
            reject
            and _better(primary)
            and _better(comparisons[x][2])
            and _better(comparisons[x][8])
        )
        per_design[x] = {
            "holm_resolved_better_at_4": bool(reject and _better(primary)),
            "better_in_mean": {n: _better(comparisons[x][n]) for n in (2, 4, 8)},
            "supported": supported,
            "comparisons": {n: _summary(comparisons[x][n]) for n in (2, 4, 8)},
        }
    any_supported = any(d["supported"] for d in per_design.values())
    refuted = not any(any(d["better_in_mean"].values()) for d in per_design.values())
    return {
        "hypothesis": "HB1",
        "verdict": "supported" if any_supported else "refuted" if refuted else "not established",
        "model": model,
        "n_tests": 9,
        "per_design": per_design,
    }


def hb2(table: RecordTable, *, model: str = "lstm") -> dict[str, Any]:
    """HB2, see module docstring (per test set)."""
    result: dict[str, Any] = {"hypothesis": "HB2", "model": model, "per_test_set": {}}
    verdicts = []
    for test_set in ("T3", "T1"):
        curve_random = {n: table.median("b5r", n, model, test_set) for n in table.sizes("b5r")}
        rows = {}
        supported = False
        resolved_any = False
        for n in (4, 8, 16, 32):
            comparison = _compare(table, "b5", "b5r", n, test_set, model)
            median_b5 = table.median("b5", n, model, test_set)
            needed = runs_needed(curve_random, median_b5)
            ratio = needed / n if not math.isnan(needed) else math.nan
            resolved = bool(
                comparison is not None and comparison.is_resolved and _better(comparison)
            )
            resolved_any |= resolved
            if n in (8, 16) and resolved and ratio >= 1.5:
                supported = True
            rows[n] = {
                "comparison": _summary(comparison),
                "median_b5": median_b5,
                "median_b5r": table.median("b5r", n, model, test_set),
                "runs_b5r_needs": needed,
                "ratio": ratio,
                "resolved_better": resolved,
            }
        not_below = all(not (rows[n]["median_b5"] < rows[n]["median_b5r"]) for n in (8, 16))
        verdict = (
            "supported"
            if supported
            else "refuted"
            if (not resolved_any and not_below)
            else "not established"
        )
        verdicts.append(verdict)
        result["per_test_set"][test_set] = {"verdict": verdict, "rows": rows}
    result["verdict"] = verdicts[0] if len(set(verdicts)) == 1 else "differs by test set"
    result["n_tests"] = 8
    return result


def _rank_positions(medians: Mapping[str, float]) -> dict[str, int]:
    order = sorted(medians, key=lambda a: medians[a])
    return {arm: position for position, arm in enumerate(order)}


def hb3(
    table: RecordTable,
    *,
    model: str = "lstm",
    arms: Sequence[str] = ("b0", "b1", "b2", "b3", "b4"),
    sizes: Sequence[int] = (4, 8, 16),
    test_sets: Sequence[str] = ("T1", "T2", "T3"),
) -> dict[str, Any]:
    """HB3, see module docstring."""
    reversals: list[dict[str, Any]] = []
    tau: dict[str, dict[str, float]] = {}
    max_rank_shift = 0
    n_tests = 0
    for n in sizes:
        outcome: dict[tuple[str, str, str], int] = {}
        for x, y in itertools.combinations(arms, 2):
            for test_set in test_sets:
                comparison = _compare(table, x, y, n, test_set, model)
                n_tests += 1
                if comparison is not None and comparison.is_resolved:
                    outcome[(x, y, test_set)] = 1 if comparison.mean_difference > 0 else -1
        for x, y in itertools.combinations(arms, 2):
            for a, b in itertools.combinations(test_sets, 2):
                if outcome.get((x, y, a), 0) * outcome.get((x, y, b), 0) < 0:
                    reversals.append({"n_train": n, "pair": [x, y], "test_sets": [a, b]})
        medians = {t: {arm: table.median(arm, n, model, t) for arm in arms} for t in test_sets}
        for a, b in itertools.combinations(test_sets, 2):
            ranks_a, ranks_b = _rank_positions(medians[a]), _rank_positions(medians[b])
            max_rank_shift = max(max_rank_shift, max(abs(ranks_a[x] - ranks_b[x]) for x in arms))
            statistic = stats.kendalltau(
                [medians[a][x] for x in arms], [medians[b][x] for x in arms]
            ).statistic
            tau[f"N{n}:{a}-{b}"] = float(statistic)
    supported = bool(reversals)
    refuted = not reversals and max_rank_shift <= 1
    return {
        "hypothesis": "HB3",
        "verdict": "supported" if supported else "refuted" if refuted else "not established",
        "model": model,
        "n_tests": n_tests,
        "reversals": reversals,
        "max_rank_shift": max_rank_shift,
        "kendall_tau": tau,
    }

r"""Descriptive analysis of the B1 main run (no new hypotheses): what the data show and exclude.

Four questions (project lead, 2026-10-13), all on the records of ``scripts/run_b1_main.py``:

1. **Information measures against test errors.** Do the Fisher information (FIR and ARX) and the
   spectral coverage of the pool explain the order of the excitations? Spearman and Kendall rank
   correlation per N, test set, model and target, at two levels: *arm level* (mean measure over the
   seeds against the median error over the seeds, one point per arm) and *seed level* (one point per
   arm and seed, pooled within N). The expected sign is negative for information measures (more
   information, lower error) and positive for ``fir_condition``, ``arx_condition`` (logarithm) and
   ``max_abs_cross_correlation`` (more confounding, higher error).
2. **What the data exclude.** The paired 95 % confidence interval of the difference of the metric
   (positive = the candidate is better), absolute and relative to the mean error of the reference.
   The
   wording is "differences larger than X % are excluded at N = ...": X is the larger absolute
   bound of the
   interval in percent of the reference error. Per pair and N, no multiplicity correction; the
   intervals
   are those of the paired t over the master seeds (``n - 1`` degrees of freedom) on the mean
   difference.
3. **Run equivalent.** For an arm X and an N, the number of runs the reference (B1) needs for the
   median
   error that X has at N (log-log interpolation along the median learning curve of the reference,
   the same
   function as for B5r in the HB2 criterion), reported as "X at N is worth about k additional runs
   of B1",
   with a bootstrap interval over the master seeds (the same resampled seeds for X and the
   reference).
   Beyond the range of the reference curve the value is bounded, not estimated: below its error at
   the
   largest N the result is ``> N_max`` (reported as infinity), above its error at the smallest N
   the first N.
4. The same tables for the secondary targets (energy target, bed temperature) and for horizon 1.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats

from .b1_analysis import PRIMARY_HORIZON, PRIMARY_METRIC, PRIMARY_TARGET, RecordTable, runs_needed
from .paired_resolution import paired_comparison

__all__ = [
    "INFO_EXPECTED_SIGN",
    "bootstrap_run_equivalent",
    "information_by_arm",
    "pair_interval",
    "rank_correlations",
    "run_equivalent",
]

INFO_EXPECTED_SIGN: dict[str, int] = {
    "fir_logdet_per_parameter": -1,
    "fir_lambda_min": -1,
    "arx_logdet_per_parameter": -1,
    "arx_lambda_min": -1,
    "spectral_coverage_1pct": -1,
    "spectral_coverage_10pct": -1,
    "spectral_coverage_25pct": -1,
    "spectral_flatness": -1,
    "fir_condition": 1,
    "arx_condition": 1,
    "max_abs_cross_correlation": 1,
}
"""Measures and the sign of the correlation with the error that "more information" predicts."""

_LOG_KEYS = ("fir_condition", "arx_condition")


def information_by_arm(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], dict[int, dict[str, float]]]:
    """``{(arm, N): {seed: {measure: value}}}`` from the ``information`` records."""
    out: dict[tuple[str, int], dict[int, dict[str, float]]] = {}
    for record in records:
        if record.get("type") != "information":
            continue
        values = {}
        for key, value in record["information"].items():
            value = float(value)
            values[key] = math.log10(value) if key in _LOG_KEYS and value > 0 else value
        out.setdefault((record["arm"], record["n_train"]), {})[record["master_seed"]] = values
    return out


def _rank_stat(function: Any, x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return math.nan
    return float(function(x, y).statistic)


def rank_correlations(
    table: RecordTable,
    information: Mapping[tuple[str, int], Mapping[int, Mapping[str, float]]],
    *,
    arms: Sequence[str],
    n_train: int,
    model: str,
    test_set: str,
    target: str = PRIMARY_TARGET,
    horizon: int = PRIMARY_HORIZON,
) -> dict[str, dict[str, float]]:
    """Rank correlation of every information measure with the error, arm level and seed level."""
    result: dict[str, dict[str, float]] = {}
    for key in INFO_EXPECTED_SIGN:
        arm_info, arm_error = [], []
        seed_info, seed_error = [], []
        for arm in arms:
            per_seed_info = information.get((arm, n_train), {})
            errors = table.values(arm, n_train, model, test_set, target=target, horizon=horizon)
            common = sorted(s for s in set(per_seed_info) & set(errors) if key in per_seed_info[s])
            if not common:
                continue
            arm_info.append(float(np.mean([per_seed_info[s][key] for s in common])))
            arm_error.append(float(np.median([errors[s] for s in common])))
            seed_info.extend(per_seed_info[s][key] for s in common)
            seed_error.extend(errors[s] for s in common)
        result[key] = {
            "arm_spearman": _rank_stat(stats.spearmanr, arm_info, arm_error),
            "arm_kendall": _rank_stat(stats.kendalltau, arm_info, arm_error),
            "seed_spearman": _rank_stat(stats.spearmanr, seed_info, seed_error),
            "n_arms": float(len(arm_info)),
            "n_points": float(len(seed_info)),
        }
    return result


def pair_interval(
    table: RecordTable,
    candidate: str,
    reference: str,
    n_train: int,
    model: str,
    test_set: str,
    *,
    target: str = PRIMARY_TARGET,
    horizon: int = PRIMARY_HORIZON,
) -> dict[str, Any] | None:
    """Paired 95 % interval of ``reference - candidate`` (positive = candidate better)."""
    kwargs = {"target": target, "horizon": horizon}
    cand = table.values(candidate, n_train, model, test_set, **kwargs)
    ref = table.values(reference, n_train, model, test_set, **kwargs)
    if not cand or set(cand) != set(ref):
        return None
    comparison = paired_comparison(cand, ref, candidate_label=candidate, reference_label=reference)
    if comparison.degenerate:
        return {"degenerate": True, "mean_difference": comparison.mean_difference}
    mean_reference = comparison.mean_reference
    low = comparison.mean_difference - comparison.threshold
    high = comparison.mean_difference + comparison.threshold
    rel = (100.0 * low / mean_reference, 100.0 * high / mean_reference)
    bound = max(abs(rel[0]), abs(rel[1]))
    if low > 0.0 or high < 0.0:
        wording = (
            f"{candidate} is {'better' if low > 0 else 'worse'} than {reference} at N = {n_train}, "
            f"by {min(abs(rel[0]), abs(rel[1])):.0f} to {bound:.0f} % of the reference error"
        )
    else:
        wording = (
            f"differences larger than {bound:.0f} % of the reference error are excluded "
            f"between {candidate} and {reference} at N = {n_train}"
        )
    return {
        "degenerate": False,
        "n_pairs": comparison.n_pairs,
        "mean_difference": comparison.mean_difference,
        "ci_low": low,
        "ci_high": high,
        "mean_reference": mean_reference,
        "relative_low_percent": rel[0],
        "relative_high_percent": rel[1],
        "excluded_above_percent": bound,
        "resolved": comparison.is_resolved,
        "statement": wording,
    }


def _curve(
    table: RecordTable,
    arm: str,
    sizes: Sequence[int],
    model: str,
    test_set: str,
    seeds: Sequence[int] | None,
    target: str,
    horizon: int,
) -> dict[int, float]:
    curve = {}
    for n in sizes:
        values = table.values(arm, n, model, test_set, target=target, horizon=horizon)
        picked = [values[s] for s in (seeds if seeds is not None else values) if s in values]
        if picked:
            curve[n] = float(np.median(picked))
    return curve


def run_equivalent(
    table: RecordTable,
    arm: str,
    reference: str,
    n_train: int,
    model: str,
    test_set: str,
    *,
    sizes: Sequence[int] = (2, 4, 8, 16, 32),
    target: str = PRIMARY_TARGET,
    horizon: int = PRIMARY_HORIZON,
    seeds: Sequence[int] | None = None,
) -> dict[str, float]:
    """Runs the reference needs for the median error of ``arm`` at ``n_train``."""
    own = _curve(table, arm, sizes, model, test_set, seeds, target, horizon)
    ref = _curve(table, reference, sizes, model, test_set, seeds, target, horizon)
    if n_train not in own or len(ref) < 2:
        return {"error": math.nan, "reference_runs": math.nan, "extra_runs": math.nan}
    needed = runs_needed(ref, own[n_train])
    return {
        "error": own[n_train],
        "reference_runs": needed,
        "extra_runs": needed - n_train if math.isfinite(needed) else math.inf,
    }


def bootstrap_run_equivalent(
    table: RecordTable,
    arm: str,
    reference: str,
    n_train: int,
    model: str,
    test_set: str,
    *,
    sizes: Sequence[int] = (2, 4, 8, 16, 32),
    target: str = PRIMARY_TARGET,
    horizon: int = PRIMARY_HORIZON,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """95 % bootstrap interval (percentiles) of the run equivalent, resampling the master seeds.

    The same resampled seeds are used for the arm and the reference (paired design). A resample in
    which the reference never reaches the error counts as infinity (``> N_max``).
    """
    values = table.values(arm, n_train, model, test_set, target=target, horizon=horizon)
    seeds = np.array(sorted(values))
    if len(seeds) < 3:
        return {"extra_low": math.nan, "extra_high": math.nan, "ratio_low": math.nan}
    rng = np.random.default_rng(seed)
    extra, ratio = [], []
    for _ in range(n_boot):
        picked = rng.choice(seeds, size=len(seeds), replace=True)
        result = _resampled(
            table, arm, reference, n_train, model, test_set, sizes, picked, target, horizon
        )
        extra.append(result[0])
        ratio.append(result[1])
    return {
        "extra_low": float(np.percentile(extra, 2.5)),
        "extra_high": float(np.percentile(extra, 97.5)),
        "ratio_low": float(np.percentile(ratio, 2.5)),
        "ratio_high": float(np.percentile(ratio, 97.5)),
        "share_beyond_range": float(np.mean(np.isinf(extra))),
    }


def _resampled(
    table: RecordTable,
    arm: str,
    reference: str,
    n_train: int,
    model: str,
    test_set: str,
    sizes: Sequence[int],
    picked: np.ndarray,
    target: str,
    horizon: int,
) -> tuple[float, float]:
    def median_at(a: str, n: int) -> float:
        values = table.values(a, n, model, test_set, target=target, horizon=horizon)
        return float(np.median([values[int(s)] for s in picked]))

    curve = {n: median_at(reference, n) for n in sizes}
    needed = runs_needed(curve, median_at(arm, n_train))
    if math.isnan(needed):
        return math.nan, math.nan
    return (needed - n_train if math.isfinite(needed) else math.inf), needed / n_train


DEFAULT_COMBOS: tuple[tuple[str, int], ...] = (
    (PRIMARY_TARGET, PRIMARY_HORIZON),
    (PRIMARY_TARGET, 1),
    ("energy_target", PRIMARY_HORIZON),
    ("energy_target", 1),
    ("bed_temperature_celsius", PRIMARY_HORIZON),
    ("bed_temperature_celsius", 1),
)
__all__ += ["DEFAULT_COMBOS", "PRIMARY_METRIC"]

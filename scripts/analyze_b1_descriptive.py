r"""Descriptive evaluation of the B1 main run (no new hypotheses). Committed BEFORE the run.

Reads ``artifacts/runs/b1_main.jsonl`` (ten seeds, arms B0-B5 and B5r, N = 2-32, test sets T1-T3, models
persistence, ARX, LSTM) through ``evaluation/b1_analysis``. Four parts, fixed here before any number is
computed; nothing is tuned and no verdict of HB1-HB3 is changed.

1. INFORMATION MASS AGAINST TEST ERROR. Do the model-free measures of the training runs (records of type
   ``information``: spectral coverage at 1, 10 and 25 % threshold, Fisher information of the FIR regression
   -- usable before a run -- and of the ARX regression, as ``logdet per parameter``) explain the ranking of
   the excitations by test error? Per N, test set and model: Spearman rank correlation between the
   per-arm medians over the seeds of the measure and of the error (the seven arms B0-B5, B5r; a negative
   correlation means more information, lower error), once with all seven arms and once without B0 (the
   constant plan, which dominates every ranking); and the mean over the seeds of the Spearman correlation
   across the seven arms within a seed (paired), with the number of seeds with a negative sign. With seven
   arms |rho| >= 0.79 is needed for p < 0.05 (two-sided); the p values are reported uncorrected, the number
   of correlations is stated.

2. WHAT THE DATA EXCLUDE. For each of B2, B3, B4, B5 against B1 and B5 against B5r, per N, test set and
   model: the paired 95 % confidence interval of the difference ``reference - candidate`` (positive: the
   candidate has the lower error; paired t, ``paired_resolution``), absolute and relative to the mean error
   of the reference, and the same on the log scale (mean of ``log(candidate / reference)``, ratio bounds).
   Wording: "improvements larger than X % are excluded" (X = upper end of the interval relative to the
   reference mean) and "deteriorations larger than Y % are excluded" (Y = minus the lower end). Individual
   95 % intervals, no multiplicity correction; the number of intervals is stated.

3. RUNS EQUIVALENT. For each arm X (B0, B2, B3, B4, B5, B5r) and each N: the number of runs B1 needs to
   reach the median error of X at N (log-log interpolation along the median curve of B1,
   ``runs_needed``; the same method as for B5r in the analysis of HB2), and the difference to N
   ("worth k additional runs"; negative: worse than B1). B5 additionally against the B5r curve. A paired
   bootstrap over the seeds (2000 resamples of the ten seeds, the same resample for all arms, fixed seed)
   gives a 95 % interval; resamples in which the curve of B1 does not get that low count as "beyond the
   range" (reported as a share) and enter the interval as infinity.

4. SECONDARY QUANTITIES AND HORIZON. The same four parts for the energy target and the bed temperature and
   for horizon 1 next to horizon 24 (primary: product moisture, horizon 24, run-median RMSE in physical
   units). In the printed summary only the medians relative to persistence, the exclusion bounds and the
   runs equivalent at N = 4, 8, 32 are shown; the artifact has everything.

Caveat carried into the lab-book entry: the gas velocity sits on its per-run limit in 20-24 % of the steps of
every plan (preset version 2, Befund 86), which may have made the plans more alike in that channel and
shrunk differences between them.

    tools/run_here.sh scripts/analyze_b1_descriptive.py --log artifacts/runs/b1_main.jsonl --name b1_descriptive_eval
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact

from twinlink_ml.evaluation.b1_analysis import RecordTable, load_records, runs_needed
from twinlink_ml.evaluation.paired_resolution import paired_comparison

ARMS = ("b0", "b1", "b2", "b3", "b4", "b5", "b5r")
SIZES = (2, 4, 8, 16, 32)
TEST_SETS = ("T1", "T2", "T3")
MODELS = ("lstm", "arx")
QUANTITIES = (
    ("product_moisture_kg_per_kg", 24),  # primary
    ("product_moisture_kg_per_kg", 1),
    ("energy_target", 24),
    ("energy_target", 1),
    ("bed_temperature_celsius", 24),
    ("bed_temperature_celsius", 1),
)
MEASURES = (
    "spectral_coverage_1pct",
    "spectral_coverage_10pct",
    "spectral_coverage_25pct",
    "fir_logdet_per_parameter",
    "arx_logdet_per_parameter",
)
PAIRS = (("b2", "b1"), ("b3", "b1"), ("b4", "b1"), ("b5", "b1"), ("b5", "b5r"))
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261013
PRINT_SIZES = (4, 8, 32)


def information_table(
    records: list[dict[str, Any]],
) -> dict[tuple[str, int, str], dict[int, float]]:
    out: dict[tuple[str, int, str], dict[int, float]] = {}
    for record in records:
        if record.get("type") != "information":
            continue
        for measure in MEASURES:
            value = record["information"].get(measure)
            if value is not None:
                out.setdefault((record["arm"], record["n_train"], measure), {})[
                    record["master_seed"]
                ] = float(value)
    return out


def _spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    if len(set(x)) < 2 or len(set(y)) < 2:
        return math.nan, math.nan
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def part1(table: RecordTable, info: dict, target: str, horizon: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    seeds = sorted(table.values("b1", 4, "lstm", "T3", target=target, horizon=horizon))
    for n, test_set, model in itertools.product(SIZES, TEST_SETS, MODELS):
        errors = {
            arm: table.values(arm, n, model, test_set, target=target, horizon=horizon)
            for arm in ARMS
        }
        cell: dict[str, Any] = {}
        for measure in MEASURES:
            values = {arm: info.get((arm, n, measure), {}) for arm in ARMS}
            med_x = {arm: float(np.median(list(v.values()))) for arm, v in values.items() if v}
            med_y = {arm: float(np.median(list(v.values()))) for arm, v in errors.items() if v}
            arms_all = [a for a in ARMS if a in med_x and a in med_y]
            rho, p = _spearman([med_x[a] for a in arms_all], [med_y[a] for a in arms_all])
            arms_wo = [a for a in arms_all if a != "b0"]
            rho_wo, p_wo = _spearman([med_x[a] for a in arms_wo], [med_y[a] for a in arms_wo])
            within = []
            for seed in seeds:
                xs = [values[a].get(seed, math.nan) for a in arms_all]
                ys = [errors[a].get(seed, math.nan) for a in arms_all]
                r, _ = _spearman(xs, ys)
                if not math.isnan(r):
                    within.append(r)
            cell[measure] = {
                "arms": arms_all,
                "rho_medians_all_arms": rho,
                "p_all_arms": p,
                "rho_medians_without_b0": rho_wo,
                "p_without_b0": p_wo,
                "mean_within_seed_rho": float(np.mean(within)) if within else math.nan,
                "seeds_with_negative_rho": int(sum(r < 0 for r in within)),
                "seeds_used": len(within),
            }
        out[f"N{n}/{test_set}/{model}"] = cell
    return out


def part2(table: RecordTable, target: str, horizon: int) -> list[dict[str, Any]]:
    rows = []
    kw = {"target": target, "horizon": horizon}
    for (cand, ref), n, test_set, model in itertools.product(PAIRS, SIZES, TEST_SETS, MODELS):
        c = table.values(cand, n, model, test_set, **kw)
        r = table.values(ref, n, model, test_set, **kw)
        if not c or set(c) != set(r):
            continue
        result = paired_comparison(c, r, candidate_label=cand, reference_label=ref)
        mean_ref = result.mean_reference
        d, thr = result.mean_difference, result.threshold
        lo, hi = d - thr, d + thr
        logs = paired_comparison(
            {s: math.log(v) for s, v in c.items()},
            {s: math.log(v) for s, v in r.items()},
        )
        # mean log(candidate / reference) = -(mean of log(ref) - log(cand)); positive difference = better
        log_mean = -logs.mean_difference
        if logs.degenerate:
            ratio_ci = None
            log_improvement = log_deterioration = None
        else:
            log_lo = -(logs.mean_difference + logs.threshold)
            log_hi = -(logs.mean_difference - logs.threshold)
            ratio_ci = [math.exp(log_lo), math.exp(log_hi)]
            log_improvement = 100.0 * (1.0 - math.exp(log_lo))
            log_deterioration = 100.0 * (math.exp(log_hi) - 1.0)
        rows.append(
            {
                "candidate": cand,
                "reference": ref,
                "n_train": n,
                "test_set": test_set,
                "model": model,
                "target": target,
                "horizon": horizon,
                "n_pairs": result.n_pairs,
                "mean_difference": d,
                "ci95_low": lo,
                "ci95_high": hi,
                "mean_reference": mean_ref,
                "relative_mean_difference": d / mean_ref,
                "relative_ci95_low": lo / mean_ref,
                "relative_ci95_high": hi / mean_ref,
                "improvement_excluded_above_pct": 100.0 * max(hi, 0.0) / mean_ref,
                "deterioration_excluded_above_pct": 100.0 * max(-lo, 0.0) / mean_ref,
                "p_value": result.p_value,
                "resolved": bool(result.is_resolved),
                "log_ratio_mean": log_mean,
                "ratio_ci95": ratio_ci,
                "improvement_excluded_above_pct_log": log_improvement,
                "deterioration_excluded_above_pct_log": log_deterioration,
            }
        )
    return rows


def _curve(values_by_n: dict[int, dict[int, float]], seeds: list[int]) -> dict[int, float]:
    return {n: float(np.median([v[s] for s in seeds])) for n, v in values_by_n.items()}


def part3(table: RecordTable, target: str, horizon: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    kw = {"target": target, "horizon": horizon}
    for model, test_set in itertools.product(MODELS, TEST_SETS):
        seeds = sorted(table.values("b1", 4, model, test_set, **kw))
        # arrays (seeds, N) per arm; the same resample of the seeds is used for every arm
        matrix = {
            arm: np.array(
                [[table.values(arm, n, model, test_set, **kw)[s] for n in SIZES] for s in seeds]
            )
            for arm in ARMS
        }
        draws = rng.integers(0, len(seeds), size=(BOOTSTRAP, len(seeds)))
        for arm, reference in [(a, "b1") for a in ("b0", "b2", "b3", "b4", "b5", "b5r")] + [
            ("b5", "b5r")
        ]:
            errors = np.median(matrix[arm], axis=0)
            ref_curve = dict(zip(SIZES, np.median(matrix[reference], axis=0), strict=True))
            boot_errors = np.median(matrix[arm][draws], axis=1)  # (B, N)
            boot_curves = np.median(matrix[reference][draws], axis=1)  # (B, N)
            for j, n in enumerate(SIZES):
                needed = runs_needed(ref_curve, float(errors[j]))
                boot_arr = np.array(
                    [
                        runs_needed(
                            dict(zip(SIZES, boot_curves[i], strict=True)), boot_errors[i, j]
                        )
                        for i in range(BOOTSTRAP)
                    ],
                    dtype=float,
                )
                finite = np.isfinite(boot_arr)
                rows.append(
                    {
                        "arm": arm,
                        "reference_curve": reference,
                        "n_train": n,
                        "test_set": test_set,
                        "model": model,
                        "target": target,
                        "horizon": horizon,
                        "median_error_arm": float(errors[j]),
                        "runs_reference_needs": None if not math.isfinite(needed) else needed,
                        "beyond_range": bool(math.isinf(needed)),
                        "worth_additional_runs": None if not math.isfinite(needed) else needed - n,
                        "ratio_needed_over_n": None if not math.isfinite(needed) else needed / n,
                        "bootstrap_share_beyond_range": float(1.0 - finite.mean()),
                        "bootstrap_needed_ci95": [
                            float(np.percentile(boot_arr, 2.5, method="nearest")),
                            float(np.percentile(boot_arr, 97.5, method="nearest")),
                        ],
                    }
                )
    return rows


def medians_relative(table: RecordTable, target: str, horizon: int) -> dict[str, Any]:
    out = {}
    for model, test_set, arm, n in itertools.product(MODELS, TEST_SETS, ARMS, SIZES):
        kw = {"target": target, "horizon": horizon}
        value = table.median(arm, n, model, test_set, **kw)
        persistence = table.median(arm, n, "persistence", test_set, **kw)
        out[f"{model}/{test_set}/{arm}/N{n}"] = {
            "median": value,
            "relative_to_persistence": value / persistence if persistence else math.nan,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--name", required=True)
    arguments = parser.parse_args()
    require_clean_tree()
    if (RUNS_DIR / f"{arguments.name}.json").exists():
        raise SystemExit("artifact exists; use a new --name")
    records = load_records(arguments.log)
    table = RecordTable(records)
    info = information_table(records)
    quantities = {}
    for target, horizon in QUANTITIES:
        quantities[f"{target}/h{horizon}"] = {
            "medians": medians_relative(table, target, horizon),
            "part1_information_vs_error": part1(table, info, target, horizon),
            "part2_confidence_intervals": part2(table, target, horizon),
            "part3_runs_equivalent": part3(table, target, horizon),
        }
        print(f"done {target} h{horizon}", flush=True)
    n_corr = len(SIZES) * len(TEST_SETS) * len(MODELS) * len(MEASURES)
    write_artifact(
        arguments.name,
        {
            "_provenance": provenance(),
            "log": arguments.log.name,
            "n_records": len(records),
            "seeds": sorted({r["master_seed"] for r in records}),
            "primary": {"target": "product_moisture_kg_per_kg", "horizon": 24},
            "counts": {
                "correlations_per_quantity": n_corr,
                "intervals_per_quantity": len(PAIRS) * len(SIZES) * len(TEST_SETS) * len(MODELS),
                "bootstrap_resamples": BOOTSTRAP,
                "bootstrap_seed": BOOTSTRAP_SEED,
            },
            "quantities": quantities,
        },
    )
    print(json.dumps({"written": arguments.name}))


if __name__ == "__main__":
    main()

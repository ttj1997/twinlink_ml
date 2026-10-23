r"""Outlier robustness of the architecture ranking (docs/27_outlier_robustness.md): gate, verdicts, readings.

PRE-REGISTRATION. Committed together with ``run_outlier_robustness.py`` and
``run_outlier_robustness_worker.py`` BEFORE any record of the run exists; no rule below is changed
after the records are seen.

Question. The ranking of the architectures is robust against the LEVEL of Gaussian sensor noise
(step 5) but was never tested against outliers; the lab data of the predecessor project lost 17-34 %
of the raw points per batch as outliers. Does the ranking on product moisture survive a realistic
outlier share (p = 0.25), with and without a causal Hampel filter?

Run (driver ``run_outlier_robustness.py``, worker ``run_outlier_robustness_worker.py``): protocol of
arm N of Befund 94 (``DataSpec.new_study()``, preset version 2; seeds 1-5; run-level splits;
``n_train`` 16 and 64; constant-run holdout seed 999; ``max_epochs`` 200, patience 20; primary
evaluation variant); outliers on the moisture channel only, INLINE operating mode, on top of the
unchanged sensor model, via ``remeasure`` (no new simulation); outliers in the input history AND
in the training target. Conditions ``p<rate>_<type>_<filter>``: p in {0, 0.05, 0.25}; type
independent or burst for p > 0; filter none or causal Hampel, window 8, threshold 3 x scaled MAD,
identical for every model (persistence uses the filtered last value). Settings:
``conf/sensor/outliers_placeholder.yaml`` (placeholders). Models persistence, ARX, GRU, LSTM,
Transformer. Target ``product_moisture_kg_per_kg``, h = 1 and 36.

Metric. MAIN: ``rmse_run_median_physical_truth`` (per-run RMSE in kg/kg against the noise-free
simulator output, median over the runs). Reported in addition, not used for a reading:
``..._disturbed`` (against the measurement with outliers, before any filter) and
``..._processed`` (against the signal the models see).

Cells per condition: ``n_train`` {16, 64} x test set {excited, constant-run holdout} (always named)
x h {1, 36} = 8. Pairs (candidate against reference), 9 per cell, 72 rank cells per condition:

* NET ORDER: GRU against LSTM, Transformer against LSTM, Transformer against GRU;
* NETS AGAINST BASELINES: ARX and persistence (candidate) each against GRU, LSTM and Transformer
  (reference); "candidate worse" = the network is ahead.

Statistics as Befund 94: paired t test over the five seeds (``compare`` of
``analyze_paired_reevaluation.py``, i.e. ``evaluation/paired_resolution.py``; 4 degrees of
freedom), verdict "candidate better" / "candidate worse" / "unresolved"; Holm 5 % as the MAIN form,
family = the 72 rank cells of ONE condition (the analogue of "all rank cells of one arm" of
Befund 94/98); a cell is resolved after Holm if its uncorrected verdict is resolved and its
adjusted p value is below 0.05. Next to every verdict: ratio of the medians over the seeds
(candidate / reference, > 1 = candidate worse), sign consistency (seeds in which the candidate is
worse), median and range of both sides, p value, Holm-adjusted p value.

Gate (checked first; if it fails no reading is assigned and the script exits with 1):

* G1 every record: ``git/dirty`` False, one ``git/commit``, preset version 2, process fingerprint
  ``bddca37f716f5cfc``, zero projected steps in pool and holdout pool, one outlier config digest;
* G2 completeness: every planned (condition, model, ``n_train``, seed, test set) exactly once (plan
  from the timing file of the driver);
* G3 reproduction: in ``p0_none_none`` the ``processed`` (= ``disturbed``) metric equals the
  ``rmse_run_median_physical`` of moisture of the Befund 94 record (``rerun_full_N.jsonl``, primary
  variant) for all five models, budgets, seeds, test sets and both horizons (relative tolerance
  1e-6); for the networks the moisture-head epochs equal Befund 94; process, excitation, split and
  measurement fingerprints equal.

Reference condition C0 = ``p0_none_none`` (the pipeline of Befund 94). For a condition C and each of
the 72 rank cells, compare the Holm verdict in C with the Holm verdict in C0: REVERSAL (both
resolved, opposite), NEW (C0 unresolved, C resolved), LOST (C0 resolved, C unresolved).
LEAD(C) = number of the 48 nets-against-baselines cells with Holm verdict "candidate worse"
(network ahead).

RANKING UNCHANGED in C (against C0) if all of:

* (a) no REVERSAL in any of the 72 cells;
* (b) no NEW nets-against-baselines cell with the baseline ahead ("candidate better");
* (c) NET ORDER: no NEW cell, and at most half of the C0-resolved net-order cells LOST;
* (d) LEAD(C) >= LEAD(C0) / 2.

LEAD VANISHED in C if LEAD(C) = 0.

Readings, separately for each outlier type T in {independent, burst}; Cn = ``p0.25_T_none``,
Cf = ``p0.25_T_hampel8``:

* R3 (the comparison does not carry over to real sensor quality; limit for docs/08):
  LEAD VANISHED in Cf.
* R1 (comparison robust against outliers): not R3, RANKING UNCHANGED in Cn and in Cf.
* R2 (filter becomes a mandatory part of the pipeline; recommendation for AP 2): not R3, ranking
  changed in Cn, RANKING UNCHANGED in Cf.
* NO READING otherwise (e.g. changed with the filter too, but lead not vanished; or unchanged
  without and changed with the filter); the deviating cells are listed and the consequence is left
  to the project lead.
* If LEAD(C0) = 0 no reading is assigned at all (the reference has no lead to lose).

Control (reported as a caveat of the readings, not a reading): RANKING UNCHANGED in
``p0_none_hampel8``, i.e. whether the filter alone changes the ranking on clean data.

Descriptive, no reading: the p = 0.05 conditions (same quantities); the readings repeated with the
uncorrected verdicts and with the ``disturbed`` metric; per cell the order of the five models by
the median of the main metric; per model the ratio of its median in C to C0; the realised outlier
rates and the share of samples the filter replaced; the side check (GRU, p = 0.25, Hampel windows
4 / 8 / 16: medians and paired comparison against window 8, outside every family).

Not claimed: any statement for xLSTM, Mamba or static regression, for energy or bed temperature,
for ``n_train`` 8 and 32, or for the SOFT_SENSOR mode; any calibrated outlier model (all shape
parameters are placeholders). Five seeds, four degrees of freedom; the cells of a condition are not
independent; Holm is conservative for correlated cells.

    tools/run_here.sh scripts/analyze_outlier_robustness.py --run-name main
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import RUNS_DIR, provenance, require_clean_tree
from analyze_83_addendum import holm
from analyze_paired_reevaluation import compare

ALPHA = 0.05
REL_TOL = 1e-6
SEEDS = (1, 2, 3, 4, 5)
SIZES = (16, 64)
TEST_SETS = ("excited", "constant_holdout")
HORIZONS = (1, 36)
CHANNEL = "product_moisture_kg_per_kg"
MODELS = ("persistence", "arx", "gru", "lstm", "transformer")
NETWORKS = ("gru", "lstm", "transformer")
BASELINES = ("arx", "persistence")
NET_ORDER = (("gru", "lstm"), ("transformer", "lstm"), ("transformer", "gru"))
NETS_VS_BASELINES = tuple((b, n) for b in BASELINES for n in NETWORKS)
PAIRS = NET_ORDER + NETS_VS_BASELINES
REFERENCE = "p0_none_none"
FILTER_CONTROL = "p0_none_hampel8"
MAIN_METRIC = "rmse_run_median_physical_truth"
SECONDARY_METRICS = ("rmse_run_median_physical_disturbed", "rmse_run_median_physical_processed")
B94_METRIC = "rmse_run_median_physical"
PROCESS_FINGERPRINT = "bddca37f716f5cfc"
FINGERPRINT_KEYS = (
    "process/fingerprint",
    "excitation/fingerprint",
    "split/digest",
    "measurement/digest",
)
BETTER, WORSE, UNRESOLVED = "candidate better", "candidate worse", "unresolved"
SIDE_WINDOWS = (4, 16)
MAIN_WINDOW = 8
GZIP_ABOVE_BYTES = 5 * 1024 * 1024

Records = list[dict[str, Any]]
Key = tuple[str, str, int, int, str]  # condition, model, n_train, seed, test set


def load_jsonl(path: Path) -> Records:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def index(records: Records) -> dict[Key, dict[str, Any]]:
    out: dict[Key, dict[str, Any]] = {}
    for r in records:
        p = r["params"]
        key = (
            p["condition/id"],
            p["model/label"],
            int(p["n_train"]),
            int(p["master_seed"]),
            p["test_set"],
        )
        if key in out:
            raise SystemExit(f"duplicate record {key}")
        out[key] = r
    return out


def cells() -> list[tuple[int, str, int]]:
    return [(n, t, h) for n in SIZES for t in TEST_SETS for h in HORIZONS]


def metric_key(metric: str, h: int) -> str:
    return f"{metric}/{CHANNEL}/h{h}"


def series(idx: dict[Key, dict[str, Any]], condition: str, model: str, n: int, test_set: str, h: int,
           metric: str = MAIN_METRIC) -> dict[int, float]:  # fmt: skip
    return {
        s: float(idx[(condition, model, n, s, test_set)]["metrics"][metric_key(metric, h)])
        for s in SEEDS
    }


# ----------------------------------------------------------------------------------------- gate


def gate(records: Records, idx: dict[Key, dict[str, Any]], old: Records,
         plan_conditions: list[str], side_conditions: list[str]) -> dict[str, Any]:  # fmt: skip
    failures: list[str] = []
    commits = {r["params"].get("git/commit") for r in records}
    digests = {r["params"].get("outlier_config/digest") for r in records}
    if len(commits) != 1:
        failures.append(f"G1 git/commit not unique: {sorted(map(str, commits))}")
    if len(digests) != 1:
        failures.append(f"G1 outlier config digest not unique: {sorted(map(str, digests))}")
    for r in records:
        p = r["params"]
        tag = f"{p['condition/id']}/{p['model/label']}/n{p['n_train']}/s{p['master_seed']}/{p['test_set']}"
        if p.get("git/dirty") is not False:
            failures.append(f"G1 {tag}: git/dirty")
        if str(p.get("preset/new_study_version")) != "2":
            failures.append(f"G1 {tag}: preset version {p.get('preset/new_study_version')}")
        if p.get("process/fingerprint") != PROCESS_FINGERPRINT:
            failures.append(f"G1 {tag}: process fingerprint {p.get('process/fingerprint')}")
        if p.get("pool/projected_steps") != 0 or p.get("holdout_pool/projected_steps") != 0:
            failures.append(f"G1 {tag}: projected steps")

    expected: set[Key] = set()
    for condition in plan_conditions:
        for model in MODELS:
            expected |= {
                (condition, model, n, s, t) for n in SIZES for s in SEEDS for t in TEST_SETS
            }
    for condition in side_conditions:
        expected |= {(condition, "gru", n, s, t) for n in SIZES for s in SEEDS for t in TEST_SETS}
    missing, extra = expected - set(idx), set(idx) - expected
    if missing:
        failures.append(f"G2 {len(missing)} planned records missing, e.g. {sorted(missing)[:3]}")
    if extra:
        failures.append(f"G2 {len(extra)} unplanned records, e.g. {sorted(extra)[:3]}")

    old_idx: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for r in old:
        p = r["params"]
        if p.get("eval/primary") and p["model/label"] in MODELS:
            old_idx[(p["model/label"], int(p["n_train"]), int(p["master_seed"]), p["test_set"])] = r
    max_dev = 0.0
    n_checked = 0
    for model in MODELS:
        for n in SIZES:
            for s in SEEDS:
                for t in TEST_SETS:
                    new = idx.get((REFERENCE, model, n, s, t))
                    ref = old_idx.get((model, n, s, t))
                    if new is None or ref is None:
                        failures.append(f"G3 {model}/n{n}/s{s}/{t}: record missing")
                        continue
                    for h in HORIZONS:
                        b94 = float(ref["metrics"][metric_key(B94_METRIC, h)])
                        for metric in SECONDARY_METRICS:
                            value = float(new["metrics"][metric_key(metric, h)])
                            dev = abs(value - b94) / abs(b94)
                            max_dev = max(max_dev, dev)
                            n_checked += 1
                            if dev > REL_TOL:
                                failures.append(
                                    f"G3 {model}/n{n}/s{s}/{t}/h{h} {metric}: {dev:.2e}"
                                )
                    for key in FINGERPRINT_KEYS:
                        if new["params"].get(key) != ref["params"].get(key):
                            failures.append(f"G3 {model}/n{n}/s{s}/{t}: {key} differs")
                    if model in NETWORKS:
                        epochs = f"training/epochs_run/{CHANNEL}"
                        if new["params"].get(epochs) != ref["params"].get(epochs):
                            failures.append(f"G3 {model}/n{n}/s{s}/{t}: moisture epochs differ")
    return {
        "passed": not failures,
        "n_failures": len(failures),
        "failures": failures[:200],
        "reproduction": {"values_checked": n_checked, "max_relative_deviation": max_dev},
        "git_commit": sorted(map(str, commits)),
    }


# ------------------------------------------------------------------------------- rank cells


def rank_cells(
    idx: dict[Key, dict[str, Any]], condition: str, metric: str = MAIN_METRIC
) -> list[dict[str, Any]]:
    rows = []
    for cand, ref in PAIRS:
        for n, t, h in cells():
            c = series(idx, condition, cand, n, t, h, metric)
            r = series(idx, condition, ref, n, t, h, metric)
            row = compare(c, r, cand, ref)
            row.update(
                {
                    "condition": condition,
                    "block": "net_order" if (cand, ref) in NET_ORDER else "nets_vs_baselines",
                    "n_train": n,
                    "test_set": t,
                    "horizon": h,
                    "ratio_of_medians": row["median_candidate"] / row["median_reference"],
                    "range_candidate": [min(c.values()), max(c.values())],
                    "range_reference": [min(r.values()), max(r.values())],
                }
            )
            rows.append(row)
    p = np.array([1.0 if math.isnan(r["p_value"]) else r["p_value"] for r in rows])
    adjusted = holm(p)
    for row, q in zip(rows, adjusted, strict=True):
        resolved = row["verdict"] in (BETTER, WORSE)
        row["holm_p_value"] = float(q)
        row["verdict_holm"] = row["verdict"] if resolved and q < ALPHA else UNRESOLVED
        row["verdict_uncorrected"] = row["verdict"] if resolved else UNRESOLVED
    return rows


def cell_key(row: dict[str, Any]) -> tuple:
    return (row["candidate"], row["reference"], row["n_train"], row["test_set"], row["horizon"])


def compare_to_reference(
    rows: list[dict[str, Any]], ref_rows: list[dict[str, Any]], form: str
) -> dict[str, Any]:
    """Change categories and the rule RANKING UNCHANGED; ``form`` = verdict_holm or verdict_uncorrected."""
    ref = {cell_key(r): r for r in ref_rows}
    reversal, new, lost, new_baseline_ahead = [], [], [], []
    order_ref_resolved = sum(r[form] != UNRESOLVED for r in ref_rows if r["block"] == "net_order")
    order_new, order_lost = [], []
    for row in rows:
        base = ref[cell_key(row)]
        v, v0 = row[form], base[form]
        entry = {"cell": list(cell_key(row)), "reference": v0, "condition": v}
        if v != UNRESOLVED and v0 != UNRESOLVED and v != v0:
            reversal.append(entry)
        elif v0 == UNRESOLVED and v != UNRESOLVED:
            new.append(entry)
            if row["block"] == "net_order":
                order_new.append(entry)
            elif v == BETTER:
                new_baseline_ahead.append(entry)
        elif v0 != UNRESOLVED and v == UNRESOLVED:
            lost.append(entry)
            if row["block"] == "net_order":
                order_lost.append(entry)
    lead = sum(r[form] == WORSE for r in rows if r["block"] == "nets_vs_baselines")
    lead_ref = sum(r[form] == WORSE for r in ref_rows if r["block"] == "nets_vs_baselines")
    criteria = {
        "a_no_reversal": not reversal,
        "b_no_new_baseline_ahead": not new_baseline_ahead,
        "c_net_order_no_new_and_at_most_half_lost": not order_new
        and len(order_lost) <= order_ref_resolved / 2,
        "d_lead_not_halved": lead >= lead_ref / 2,
    }
    return {
        "unchanged": all(criteria.values()),
        "criteria": criteria,
        "lead": lead,
        "lead_reference": lead_ref,
        "lead_vanished": lead == 0,
        "baseline_ahead": sum(r[form] == BETTER for r in rows if r["block"] == "nets_vs_baselines"),
        "net_order_resolved_reference": order_ref_resolved,
        "reversals": reversal,
        "new": new,
        "lost": lost,
        "net_order_new": order_new,
        "net_order_lost": order_lost,
        "new_baseline_ahead": new_baseline_ahead,
    }


def readings(comparisons: dict[str, dict[str, Any]], lead_reference: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in ("independent", "burst"):
        cn, cf = f"p0.25_{kind}_none", f"p0.25_{kind}_hampel8"
        if cn not in comparisons or cf not in comparisons:
            out[kind] = {"reading": None, "why": "condition missing from the run"}
            continue
        n, f = comparisons[cn], comparisons[cf]
        if lead_reference == 0:
            label, why = None, "LEAD(C0) = 0: no reading"
        elif f["lead_vanished"]:
            label, why = "R3", "lead vanished with the filter"
        elif n["unchanged"] and f["unchanged"]:
            label, why = "R1", "ranking unchanged without and with the filter"
        elif not n["unchanged"] and f["unchanged"]:
            label, why = "R2", "ranking changed without, unchanged with the filter"
        else:
            failed_n = [k for k, v in n["criteria"].items() if not v]
            failed_f = [k for k, v in f["criteria"].items() if not v]
            label = "none"
            why = f"no reading: unchanged without filter {n['unchanged']} (failed {failed_n}), with filter {f['unchanged']} (failed {failed_f})"
        out[kind] = {
            "reading": label,
            "why": why,
            "without_filter": {k: n[k] for k in ("unchanged", "criteria", "lead", "lead_vanished")},
            "with_filter": {k: f[k] for k in ("unchanged", "criteria", "lead", "lead_vanished")},
        }
    return out


# ------------------------------------------------------------------------------ descriptive


def medians(
    idx: dict[Key, dict[str, Any]], condition: str, metric: str = MAIN_METRIC
) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for n, t, h in cells():
        per_model = {}
        for model in MODELS:
            values = list(series(idx, condition, model, n, t, h, metric).values())
            per_model[model] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        order = sorted(MODELS, key=lambda m: per_model[m]["median"])
        table[f"n{n}/{t}/h{h}"] = {"models": per_model, "order_by_median": order}
    return table


def mean_rank(table: dict[str, Any]) -> dict[str, float]:
    ranks: dict[str, list[int]] = {m: [] for m in MODELS}
    for cell in table.values():
        for position, model in enumerate(cell["order_by_median"], start=1):
            ranks[model].append(position)
    return {m: float(np.mean(v)) for m, v in ranks.items()}


def degradation(tables: dict[str, Any]) -> dict[str, Any]:
    """Ratio of the median of each model in C to its median in C0, per cell."""
    ref = tables[REFERENCE]
    out: dict[str, Any] = {}
    for condition, table in tables.items():
        if condition == REFERENCE:
            continue
        out[condition] = {
            cell: {
                m: table[cell]["models"][m]["median"] / ref[cell]["models"][m]["median"]
                for m in MODELS
            }
            for cell in table
        }
    return out


def contamination_diagnostics(records: Records) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r in records:
        p = r["params"]
        entry = out.setdefault(
            p["condition/id"],
            {"pool_rate": [], "holdout_rate": [], "pool_replaced": [], "holdout_replaced": []},
        )
        entry["pool_rate"].append(p["pool/outlier/realised_rate"])
        entry["holdout_rate"].append(p["holdout_pool/outlier/realised_rate"])
        entry["pool_replaced"].append(p["pool/preprocess/replaced_fraction"])
        entry["holdout_replaced"].append(p["holdout_pool/preprocess/replaced_fraction"])
    return {
        c: {k: {"min": min(v), "max": max(v), "median": statistics.median(v)} for k, v in e.items()}
        for c, e in out.items()
    }


def side_check(idx: dict[Key, dict[str, Any]], side_conditions: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in ("independent", "burst"):
        main = f"p0.25_{kind}_hampel{MAIN_WINDOW}"
        for window in SIDE_WINDOWS:
            condition = f"p0.25_{kind}_hampel{window}"
            if condition not in side_conditions:
                continue
            rows = []
            for n, t, h in cells():
                row = compare(
                    series(idx, condition, "gru", n, t, h),
                    series(idx, main, "gru", n, t, h),
                    f"gru_w{window}",
                    f"gru_w{MAIN_WINDOW}",
                )
                row.update({"n_train": n, "test_set": t, "horizon": h})
                row["ratio_of_medians"] = row["median_candidate"] / row["median_reference"]
                rows.append(row)
            out[condition] = rows
    return out


def evaluate(
    records: Records, old: Records, plan_conditions: list[str], side_conditions: list[str]
) -> dict[str, Any]:
    idx = index(records)
    result: dict[str, Any] = {"gate": gate(records, idx, old, plan_conditions, side_conditions)}
    result["contamination"] = contamination_diagnostics(records)
    if not result["gate"]["passed"]:
        result["readings"] = None
        return result
    rows = {c: rank_cells(idx, c) for c in plan_conditions}
    comparisons = {
        form: {
            c: compare_to_reference(rows[c], rows[REFERENCE], form)
            for c in plan_conditions
            if c != REFERENCE
        }
        for form in ("verdict_holm", "verdict_uncorrected")
    }
    lead_ref = {
        form: sum(r[form] == WORSE for r in rows[REFERENCE] if r["block"] == "nets_vs_baselines")
        for form in ("verdict_holm", "verdict_uncorrected")
    }
    disturbed_rows = {c: rank_cells(idx, c, SECONDARY_METRICS[0]) for c in plan_conditions}
    disturbed_cmp = {
        c: compare_to_reference(disturbed_rows[c], disturbed_rows[REFERENCE], "verdict_holm")
        for c in plan_conditions
        if c != REFERENCE
    }
    disturbed_lead_ref = sum(
        r["verdict_holm"] == WORSE
        for r in disturbed_rows[REFERENCE]
        if r["block"] == "nets_vs_baselines"
    )
    tables = {c: medians(idx, c) for c in plan_conditions}
    result.update(
        {
            "rank_cells": rows,
            "verdict_counts": {
                c: {
                    form: {
                        block: {
                            v: sum(r[form] == v for r in rows[c] if r["block"] == block)
                            for v in (BETTER, WORSE, UNRESOLVED)
                        }
                        for block in ("net_order", "nets_vs_baselines")
                    }
                    for form in ("verdict_holm", "verdict_uncorrected")
                }
                for c in plan_conditions
            },
            "comparison_to_reference": comparisons,
            "readings": readings(comparisons["verdict_holm"], lead_ref["verdict_holm"]),
            "readings_uncorrected": readings(
                comparisons["verdict_uncorrected"], lead_ref["verdict_uncorrected"]
            ),
            "readings_disturbed_metric": readings(disturbed_cmp, disturbed_lead_ref),
            "filter_control_unchanged": comparisons["verdict_holm"]
            .get(FILTER_CONTROL, {})
            .get("unchanged"),
            "medians": tables,
            "mean_rank_by_median": {c: mean_rank(t) for c, t in tables.items()},
            "degradation_against_reference": degradation(tables),
            "side_check": side_check(idx, side_conditions),
        }
    )
    return result


def write(name: str, payload: dict[str, Any]) -> Path:
    from _common import _json_default

    text = json.dumps(payload, indent=1, default=_json_default)
    plain, packed = RUNS_DIR / f"{name}.json", RUNS_DIR / f"{name}.json.gz"
    if plain.exists() or packed.exists():
        raise SystemExit(f"{name} exists -- artifacts are never overwritten; use a new name")
    if len(text.encode()) > GZIP_ABOVE_BYTES:
        with gzip.open(packed, "wt", encoding="utf-8") as handle:
            handle.write(text)
        print(f"wrote {packed}")
        return packed
    plain.write_text(text, encoding="utf-8")
    print(f"wrote {plain}")
    return plain


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--old", default="rerun_full_N.jsonl")
    parser.add_argument("--artifact-suffix", default="")
    arguments = parser.parse_args()
    require_clean_tree()
    prefix = f"outlier_robustness_{arguments.run_name}"
    records = load_jsonl(RUNS_DIR / f"{prefix}.jsonl.gz")
    timing = json.loads((RUNS_DIR / f"{prefix}_timing.json").read_text(encoding="utf-8"))
    conditions = timing["plan"]["conditions"]
    side = [
        c
        for c in conditions
        if c.startswith("p0.25_") and c.endswith(tuple(f"hampel{w}" for w in SIDE_WINDOWS))
    ]
    main_conditions = [c for c in conditions if c not in side]
    old = load_jsonl(RUNS_DIR / arguments.old)
    result = evaluate(records, old, main_conditions, side)
    result["timing"] = {
        "plan": timing["plan"],
        "sessions": timing["sessions"],
        "job_wall_s": {k: v["wall_s"] for k, v in timing["jobs"].items()},
    }
    write(
        f"outlier_robustness_analysis_{arguments.run_name}{arguments.artifact_suffix}",
        {"provenance": provenance(), "prefix": prefix, **result},
    )
    print(f"gate passed: {result['gate']['passed']} ({result['gate']['n_failures']} failures)")
    if result["readings"] is None:
        for failure in result["gate"]["failures"][:20]:
            print("  ", failure)
        return 1
    for kind, reading in result["readings"].items():
        print(f"{kind}: {reading['reading']} -- {reading['why']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

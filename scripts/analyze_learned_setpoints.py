r"""Analysis of the learned-set-points study (``docs/NN_learned_setpoints.md``).

Reads ``artifacts/runs/<name>.json.gz`` (main run of ``scripts/run_learned_setpoints.py``) and,
if present, ``<name>_stage0.json.gz``; writes ``artifacts/runs/<name>_analysis.json`` and prints
the tables. No run is started here.

PRE-REGISTERED READING (section 7 of the study document, copied verbatim; committed before the
first run; thresholds: decision 6)
--------------------------------------------------------------------------------------------------

**Step 1, admissibility.** Per family the reading starts with the share of admissible cells per
arm and budget. An arm with fewer than 4 of 5 admissible seeds at the largest budget (`n_train` 64
or `K` 40) is **"not admissible"** for the reading and enters no class. A family in which no arm
is admissible has the reading "not admissible".

**Step 2, classes.** On the admissible arms only, at the largest budget, median over the admissible
seeds, first class that applies:

| class | condition | meaning |
|---|---|---|
| **U0** | no admissible arm of the family is cheaper than Cfix | the margin of the learned model eats the whole benefit; learned set points are not usable at this budget |
| **U1** | the baseline arm (M-quad, P-lin) is within 2 % of its twin | no network is needed for the set points |
| **U2** | the baseline arm is not, another arm of the family (M-gru; P-mlp or P-gru, named) is within 2 % | the network earns its place in control |
| **U3** | otherwise | better than a fixed point, but the gap to the known model remains; the decomposition says whether margin or set points carry it |

Family M is read once without feedback and once with the level *fb*; the level *lab* has no
reading. Family D has one arm and no baseline: U1 cannot occur there; its classes are U0, U2
(D-gru within 2 % of its twin) and U3. **Data need:** the smallest budget at which the arm named by
the reading is admissible (at least 4 of 5 seeds), within 2 % in the median and within 5 % in
every admissible seed; "not determined" if this is not reached at the largest budget. A margin
search that does not reach a margin is not an error but a not admissible cell (step 1); the reading
of a family is invalid, and reported as such, only if a unit of the family ended with an error.

--------------------------------------------------------------------------------------------------

How the text is applied (fixed with it; points settled in section 12 of the study document)

* **Cell** = arm x feedback level x budget x seed. **Admissible** iff its margin search reached
  a margin within the upper end (0.12 kg/kg) and the path condition holds at that margin on all
  test days, tail included (as ``docs/42_hybrid_structural_2.md``). A cell that is not admissible
  has no cost and takes part in no comparison.
* Cost of a cell: ``J`` = sum of the daily costs (variant E) over the test days divided by the
  sum of the dry product, at the margin of the cell. ``gap = (J - J_twin) / J_twin`` with the
  twin (C2 for the reading T1 of the load-following study) at its own margin. "Within 2 %" means
  ``gap <= 0.02``; a cell cheaper than its twin is within.
* U0 compares the median ``J`` of every admissible arm with ``J`` of Cfix at its own margin. If
  Cfix itself is not admissible, U0 cannot apply and this is stated.
* With the cut "seeds 5 -> 3 for D-gru" an arm is admissible for the reading only if all three
  seeds are admissible.
* Family D: U0, U2 (D-gru within 2 % of the twin C2), U3. C3-15 (the same law on the true model)
  is reported next to it and takes no part in the class.
* Rank statements between two learned arms: paired over the seeds at which both cells are
  admissible (``twinlink_ml.evaluation.paired_resolution``), Holm per family (M, D and P
  separately), effect size and sign consistency next to the verdict. Families of tests, fixed
  here: M -- M-gru against M-quad and level fb against level none per arm, at every budget;
  D -- D-gru against M-gru (levels none and fb) at every budget of D; P -- every pair of arms at
  every budget.
* Decomposition (``docs/24``, section 6): margin share = ``(J_twin(m_cell) - J_twin(m_twin)) /
  J_twin``, set-point share = gap minus margin share. ``J_twin(m_cell)`` is run for the cells of
  the largest budget and interpolated linearly on the twin's margin grid for the others (marked).
* Deployable margin (decision 7): reported for the arms the reading names.
* Everything else in the output is descriptive.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import RUNS_DIR, provenance  # noqa: E402

from twinlink_ml.evaluation.paired_resolution import paired_comparison  # noqa: E402

BASELINE = {"M": "M-quad", "D": None, "P": "P-lin"}
ARMS = {"M": ("M-quad", "M-gru"), "D": ("D-gru",), "P": ("P-lin", "P-mlp", "P-gru")}


def cost_per_kg(records: Sequence[dict]) -> float:
    """Cost per kg of dry product over the days of ``records``."""
    return float(sum(r["cost_e"] for r in records) / sum(r["product_dry_kg"] for r in records))


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values (``nan`` entries stay ``nan`` and do not count)."""
    out = [math.nan] * len(p_values)
    order = sorted((p, i) for i, p in enumerate(p_values) if not math.isnan(p))
    running = 0.0
    for rank, (p, index) in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p))
        out[index] = running
    return out


def cell_summary(block: dict[str, Any], j_twin: float, j_fix: float | None, j_c0: float) -> dict:
    """Numbers of one cell from its block of the main artifact."""
    out: dict[str, Any] = {
        "admissible": bool(block.get("admissible")),
        "margin": block.get("margin"),
        "largest_margin_tried": block.get("largest_margin_tried"),
        "errored_units": int(block.get("errored_units", 0)) + int(bool(block.get("job_error"))),
        "evaluations": len(block.get("search", {}).get("evaluations", [])),
    }
    if not out["admissible"]:
        return out
    records = block["records"]
    j = cost_per_kg(records)
    solver = [r["solver"] for r in records]

    def total(key: str) -> int:
        return int(sum(s.get(key, 0) or 0 for s in solver))

    decisions = total("decisions")
    out.update(
        {
            "J": j,
            "gap": (j - j_twin) / j_twin,
            "saving_vs_cfix": None if j_fix is None else (j_fix - j) / j_fix,
            "saving_vs_c0": (j_c0 - j) / j_c0,
            "largest_violation": float(max(r["max_violation"] for r in records)),
            "target_deviation_kg_max": float(
                max(abs(r["sprayed_day_kg"] - r["target_day_kg"]) for r in records)
            ),
            "guard": {
                k: total(k) for k in ("guard_box", "guard_band", "guard_target", "guard_last_hour")
            },
            "map_failures": total("map_failures") + total("failures"),
            "map_clipped": total("map_clipped"),
            "outside_label_range_share": (
                total("outside_label_range") / decisions if decisions else None
            ),
            "decide_s_mean": float(
                np.mean([r["decide_s"]["mean"] for r in records if r.get("decide_s")])
            ),
            "moisture_mean": float(np.mean([r["moisture_mean"] for r in records])),
        }
    )
    return out


def median_range(values: Sequence[float]) -> dict[str, float] | None:
    """Median and range, ``None`` for no values."""
    if not values:
        return None
    return {
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": len(values),
    }


def arm_table(
    cells: dict[str, dict], arm: str, level: str, budgets: Sequence[int], seeds: Sequence[int]
) -> dict[str, Any]:
    """Per budget: admissible seeds, median and range of gap, margin and cost."""
    out = {}
    for budget in budgets:
        present = [s for s in seeds if f"{arm}|{level}|{budget}|{s}" in cells]
        rows = {s: cells[f"{arm}|{level}|{budget}|{s}"] for s in present}
        good = {s: r for s, r in rows.items() if r["admissible"]}
        out[str(budget)] = {
            "seeds": present,
            "admissible_seeds": sorted(good),
            "gap": median_range([r["gap"] for r in good.values()]),
            "margin": median_range([r["margin"] for r in good.values()]),
            "J": median_range([r["J"] for r in good.values()]),
            "gap_per_seed": {str(s): r["gap"] for s, r in good.items()},
            "J_per_seed": {str(s): r["J"] for s, r in good.items()},
            "errored_units": int(sum(r["errored_units"] for r in rows.values())),
        }
    return out


def arm_admissible(row: dict[str, Any], minimum: int, all_seeds: int) -> bool:
    """Admissible for the reading: at least ``minimum`` of 5 seeds; all of them with fewer seeds."""
    present = len(row["seeds"])
    if present == 0:
        return False
    needed = minimum if present >= all_seeds else present
    return len(row["admissible_seeds"]) >= needed


def read_family(
    family: str,
    tables: dict[str, dict[str, Any]],
    budgets: Sequence[int],
    j_twin: float,
    j_fix: float | None,
    thresholds: dict[str, Any],
    all_seeds: int,
) -> dict[str, Any]:
    """Steps 1 and 2 of the reading for one family at one feedback level."""
    largest = str(max(budgets))
    minimum = thresholds["admissible_seeds_minimum"]
    within, every = thresholds["within_twin"], thresholds["within_twin_every_seed"]
    step1 = {
        arm: {
            b: f"{len(table[b]['admissible_seeds'])} of {len(table[b]['seeds'])}"
            for b in map(str, budgets)
        }
        for arm, table in tables.items()
    }
    admissible = [a for a, t in tables.items() if arm_admissible(t[largest], minimum, all_seeds)]
    errored = int(sum(t[b]["errored_units"] for t in tables.values() for b in map(str, budgets)))
    out: dict[str, Any] = {
        "step1_admissible_seeds": step1,
        "arms_admissible_for_reading": admissible,
        "valid": errored == 0,
        "errored_units": errored,
    }
    if not admissible:
        out.update({"reading": "not admissible", "named_arm": None, "data_need": None})
        return out
    gap = {a: tables[a][largest]["gap"]["median"] for a in admissible}
    cost = {a: tables[a][largest]["J"]["median"] for a in admissible}
    out["median_gap_at_largest_budget"] = gap
    baseline = BASELINE[family]
    named: str | None = None
    if j_fix is None:
        out["U0_note"] = "Cfix is not admissible: U0 cannot apply"
    if j_fix is not None and all(cost[a] >= j_fix for a in admissible):
        reading = "U0"
    elif baseline is not None and baseline in gap and gap[baseline] <= within:
        reading, named = "U1", baseline
    else:
        others = [a for a in admissible if a != baseline and gap[a] <= within]
        if others:
            named = min(others, key=lambda a: gap[a])
            reading = "U2"
            out["arms_within"] = others
        else:
            reading = "U3"
    out.update({"reading": reading, "named_arm": named, "J_twin": j_twin, "J_cfix": j_fix})
    need: Any = None
    if named is not None:
        need = "not determined"
        for budget in map(str, sorted(budgets)):
            row = tables[named][budget]
            if not arm_admissible(row, minimum, all_seeds):
                continue
            if row["gap"]["median"] <= within and row["gap"]["max"] <= every:
                need = int(budget)
                break
    out["data_need"] = need
    return out


def rank_test(a: dict[str, float], b: dict[str, float], label: str, j_twin: float) -> dict:
    """Paired comparison of two arms over their common admissible seeds (lower cost is better)."""
    seeds = sorted(set(a) & set(b))
    out: dict[str, Any] = {"comparison": label, "seeds": seeds, "p_value": math.nan}
    if len(seeds) < 2:
        out["note"] = "fewer than two common admissible seeds"
        return out
    cmp = paired_comparison(
        {int(s): a[s] for s in seeds}, {int(s): b[s] for s in seeds}, candidate_label=label
    )
    diffs = np.array([b[s] - a[s] for s in seeds])
    sign = np.sign(diffs.mean())
    out.update(
        {
            "mean_difference": cmp.mean_difference,
            "effect_relative_to_twin": cmp.mean_difference / j_twin,
            "threshold": cmp.threshold,
            "p_value": math.nan if cmp.degenerate else cmp.p_value,
            "sign_consistency": float(np.mean(np.sign(diffs) == sign)) if sign != 0 else 0.0,
            "degenerate": cmp.degenerate,
        }
    )
    return out


def with_holm(tests: list[dict]) -> list[dict]:
    """Add Holm-adjusted p-values and the verdict at 5 % to the tests of one family."""
    adjusted = holm([t["p_value"] for t in tests])
    for test, p in zip(tests, adjusted, strict=True):
        test["p_holm"] = p
        test["resolved"] = bool(not math.isnan(p) and p < 0.05)
        if test["resolved"]:
            test["verdict"] = "first cheaper" if test["mean_difference"] > 0 else "second cheaper"
    return tests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    parser.add_argument("--output-suffix", default="_analysis")
    args = parser.parse_args()
    runs = Path(args.runs_dir)
    with gzip.open(runs / f"{args.name}.json.gz", "rt", encoding="utf-8") as handle:
        main_run = json.load(handle)
    output = runs / f"{args.name}{args.output_suffix}.json"
    if output.exists():
        raise SystemExit(f"{output} exists -- artifacts are never overwritten")
    raw = main_run["study_config"]
    thresholds = raw["reading"]
    task = main_run["task"]
    cuts = main_run.get("cuts", [])
    reference = main_run["reference"]
    j_twin = cost_per_kg(reference[task["twin"]]["records"])
    j_c0 = cost_per_kg(reference["C0"]["records"])
    fix_block = reference["Cfix|none|0|0"]
    j_fix = cost_per_kg(fix_block["records"]) if fix_block.get("admissible") else None

    cells: dict[str, dict] = {}
    for part in ("reference", "M", "D", "P"):
        for key, block in main_run.get(part, {}).items():
            if "|" in key:
                cells[key] = cell_summary(block, j_twin, j_fix, j_c0)

    n_budgets = [n for n in raw["models"]["n_train"] if not ("n_train_32" in cuts and n == 32)]
    d_budgets = [n for n in raw["gru_mpc"]["n_train"] if n in n_budgets]
    k_budgets = raw["policy"]["budgets_days"]
    seeds = raw["models"]["seeds"]
    all_seeds = len(seeds)
    levels = ["none", "fb"]

    tables: dict[str, Any] = {}
    for arm in ARMS["M"]:
        for level in [*levels, "lab"]:
            if any(k.startswith(f"{arm}|{level}|") for k in cells):
                tables[f"{arm}|{level}"] = arm_table(cells, arm, level, n_budgets, seeds)
    tables["D-gru|none"] = arm_table(cells, "D-gru", "none", d_budgets, seeds)
    for arm in ARMS["P"]:
        if any(k.startswith(f"{arm}|none|") for k in cells):
            tables[f"{arm}|none"] = arm_table(cells, arm, "none", k_budgets, raw["policy"]["seeds"])

    readings: dict[str, Any] = {}
    for level in levels:
        fam = {a: tables[f"{a}|{level}"] for a in ARMS["M"] if f"{a}|{level}" in tables}
        readings[f"M/{level}"] = read_family(
            "M", fam, n_budgets, j_twin, j_fix, thresholds, all_seeds
        )
    readings["D"] = read_family(
        "D", {"D-gru": tables["D-gru|none"]}, d_budgets, j_twin, j_fix, thresholds, all_seeds
    )
    fam_p = {a: tables[f"{a}|none"] for a in ARMS["P"] if f"{a}|none" in tables}
    readings["P"] = read_family("P", fam_p, k_budgets, j_twin, j_fix, thresholds, all_seeds)
    job_errors = [e for e in main_run.get("errors", []) if "job" in e]
    if job_errors:
        for reading in readings.values():
            reading["valid"] = False
            reading["job_errors"] = len(job_errors)

    def per_seed(arm: str, level: str, budget: int) -> dict[str, float]:
        return tables.get(f"{arm}|{level}", {}).get(str(budget), {}).get("J_per_seed", {})

    tests_m = []
    for budget in n_budgets:
        for level in levels:
            tests_m.append(
                rank_test(
                    per_seed("M-gru", level, budget),
                    per_seed("M-quad", level, budget),
                    f"M-gru vs M-quad, level {level}, n_train {budget}",
                    j_twin,
                )
            )
        for arm in ARMS["M"]:
            tests_m.append(
                rank_test(
                    per_seed(arm, "fb", budget),
                    per_seed(arm, "none", budget),
                    f"{arm}: fb vs none, n_train {budget}",
                    j_twin,
                )
            )
    tests_d = [
        rank_test(
            per_seed("D-gru", "none", budget),
            per_seed("M-gru", level, budget),
            f"D-gru vs M-gru level {level}, n_train {budget}",
            j_twin,
        )
        for budget in d_budgets
        for level in levels
    ]
    present_p = [a for a in ARMS["P"] if f"{a}|none" in tables]
    tests_p = [
        rank_test(
            per_seed(a, "none", budget),
            per_seed(b, "none", budget),
            f"{a} vs {b}, K {budget}",
            j_twin,
        )
        for budget in k_budgets
        for i, a in enumerate(present_p)
        for b in present_p[i + 1 :]
    ]
    rank = {"M": with_holm(tests_m), "D": with_holm(tests_d), "P": with_holm(tests_p)}

    # decomposition
    curve = main_run.get("twin_curve")
    curve_j = None
    if curve:
        curve_j = [cost_per_kg(recs) if recs else math.nan for recs in curve["records"]]
    exact = {
        key: cost_per_kg(block["records"])
        for key, block in main_run.get("decomposition", {}).items()
        if block["records"]
    }
    decomposition = {}
    for key, cell in cells.items():
        if not cell["admissible"] or key.startswith(("C0", "C2")):
            continue
        if key in exact:
            j_at, how = exact[key], "run"
        elif curve_j is not None:
            j_at = float(np.interp(cell["margin"], curve["margins"], curve_j))
            how = "interpolated"
        else:
            continue
        margin_share = (j_at - j_twin) / j_twin
        decomposition[key] = {
            "margin_share": margin_share,
            "set_point_share": cell["gap"] - margin_share,
            "twin_at_margin": how,
        }

    # deployable margin
    deployable = {}
    tol = main_run["violation_tolerance"]
    for key, block in main_run.get("deployable", {}).items():
        if block is None:
            continue
        entry: dict[str, Any] = {
            "admissible_on_tuning_days": bool(block.get("admissible")),
            "margin": block.get("margin"),
            "margin_on_test_days": cells.get(key, {}).get("margin"),
        }
        if block.get("test_records"):
            records = block["test_records"]
            violations = [r["max_violation"] for r in records if r["max_violation"] is not None]
            entry.update(
                {
                    "J_test": cost_per_kg(records),
                    "gap_test": (cost_per_kg(records) - j_twin) / j_twin,
                    "test_days_violated": int(sum(v > tol for v in violations)),
                    "largest_violation_test": float(max(violations)),
                }
            )
        deployable[key] = entry
    named = {
        name: reading.get("named_arm")
        for name, reading in readings.items()
        if reading.get("named_arm")
    }

    result: dict[str, Any] = {
        "provenance": provenance("not used"),
        "source": f"{args.name}.json.gz",
        "source_commit": main_run["provenance"]["git/commit"],
        "smoke": main_run["smoke"],
        "thresholds": thresholds,
        "cuts": cuts,
        "budget_rule": main_run.get("budget_rule"),
        "load_following_reading": task["load_following_reading"],
        "twin": task["twin"],
        "J": {
            "twin": j_twin,
            "C0": j_c0,
            "Cfix": j_fix,
            "C3_C4_load_following": {k: task["cost_per_kg"].get(k) for k in ("C3", "C4")},
        },
        "controls": {
            "c2_against_load_following": reference.get("control_c2_against_load_following"),
            "M-tab": cells.get("M-tab|none|0|0"),
            "C3-15": cells.get("C3-15|none|0|0"),
            "Cfix": cells.get("Cfix|none|0|0"),
        },
        "readings": readings,
        "named_arms": named,
        "tables": tables,
        "rank_statements": rank,
        "decomposition": decomposition,
        "deployable_margin": deployable,
        "cells": cells,
        "errors_total": len(main_run.get("errors", [])),
        "units_run": main_run.get("units_run"),
        "runtime_h": main_run["runtime_s"] / 3600.0,
        "part_runtime_h": {k: v / 3600.0 for k, v in main_run.get("part_runtime_s", {}).items()},
        "peak_memory_mb_per_worker": main_run.get("peak_memory_mb_per_worker"),
    }
    stage0_path = runs / f"{args.name}_stage0.json.gz"
    if stage0_path.exists():
        with gzip.open(stage0_path, "rt", encoding="utf-8") as handle:
            stage0 = json.load(handle)["stage0"]
        models = {}
        for name, unit in stage0["units"].items():
            if not name.startswith("model_"):
                continue
            models[name] = {
                kind: {
                    "reproduction_befund94": unit[kind].get("reproduction_befund94"),
                    "table_valid_share": unit[kind]["table_valid_share"],
                    "map_error_at_limit": unit[kind]["map_error_at_limit"],
                }
                for kind in ("gru", "quad")
            }
        result["stage0"] = {
            "models": models,
            "truth_table": stage0["units"].get("table_truth"),
            "failures": len(stage0["failures"]),
        }
    output.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"twin {task['twin']}: J = {j_twin:.5f}; C0 {j_c0:.5f}; Cfix {j_fix}")
    for name, reading in readings.items():
        print(
            f"family {name}: reading {reading['reading']}"
            f"{'' if reading['valid'] else ' (INVALID: errored units)'}, named arm "
            f"{reading.get('named_arm')}, data need {reading.get('data_need')}"
        )
        for arm, row in reading["step1_admissible_seeds"].items():
            print(f"   admissible seeds {arm}: {row}")
    print("| arm | level | budget | admissible | gap median (min .. max) % | margin median |")
    print("|---|---|---|---|---|---|")
    for key, table in tables.items():
        arm, level = key.split("|")
        for budget, row in table.items():
            gap, margin = row["gap"], row["margin"]
            text = (
                "--"
                if gap is None
                else (
                    f"{100 * gap['median']:+.2f} ({100 * gap['min']:+.2f} .. {100 * gap['max']:+.2f})"
                )
            )
            m_text = "--" if margin is None else f"{margin['median']:.4f}"
            share = f"{len(row['admissible_seeds'])} of {len(row['seeds'])}"
            print(f"| {arm} | {level} | {budget} | {share} | {text} | {m_text} |")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

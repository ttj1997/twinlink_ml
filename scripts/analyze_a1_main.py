r"""Analysis of the A1 main run: paired resolution, learning curves, HA1-HA3, crossover N*.

Reads the committed records ``artifacts/runs/a1_main_records.json`` (no training; the only new computation is
the steady-state solves of item 2 below). Written before the analysis is applied (this file is committed before
it is executed; clean tree, ``git/dirty`` is logged). The rules are those of the design draft
``docs/17_a1_transfer_main_run_design.md``; they are repeated here and fixed.

Metric and test set
-------------------
The tested quantity is the RMSE in physical units at h = 36, median over the test runs of a record (the
``rmse_run_median_physical`` of the evaluator), per channel (energy target, product moisture, bed temperature).
PRIMARY test set: ``excited`` (32 excited test runs of the plant's own pool). SECONDARY: ``constant_holdout`` (16
constant runs), reported separately and never pooled. The h = 1 values are in the learning-curve tables.
Persistence and the simulator alone do not depend on N (``n = 0`` records); they are paired per seed like every
other method.

Paired resolution
-----------------
``evaluation.paired_resolution.paired_comparison`` over the five seeds (paired t on the per-seed differences
``reference - candidate``, two-sided 95 %, 4 degrees of freedom); "candidate better" / "worse" = a resolved difference,
else "unresolved". No multiplicity correction; the total number of tests is reported. The tested quantity is the
MEAN of the differences; the learning curves report median and range over the seeds.

Hypotheses (criteria exactly as in the design draft; "k of 3 channels" = energy target, bed temperature, moisture)
-------------------------------------------------------------------------------------------------------------
* HA1: on P0 and on P1 at N = 2 AND N = 4 pretraining + fine-tuning is better than from scratch in at least 2 of 3
  channels, AND on both plants the ratio RMSE(from scratch) / RMSE(pretraining + fine-tuning) (median over seeds per
  channel, mean over the 3 channels) at N = 64 is smaller than at N = 2. Not supported if the first part fails on
  either plant. The crossover N* (the smallest N from which pretraining is not resolved better in at least 2 of 3
  channels at that N and at every larger N) is reported as a bracket between grid points per plant, or "not reached".
* HA2: on P3 or P2a there is an N at which pretraining + fine-tuning is WORSE than from scratch in at least 2 of 3
  channels; the smallest such N is reported.
* HA3: on at least 2 of the 3 structural plants (P2a, P2b, P3) the hybrid is better than each of the other three
  methods (from scratch, simulator alone, pretraining + fine-tuning) in at least 2 of 3 channels at at least 3 of the
  4 small N (2, 4, 8, 16). The hybrid is the main one (16000 steps). Reported whether or not supported: which method
  wins under wrong physics (P2a) and under wrong measurement (P2b).
* Baseline rule: a method is called useful at (plant, N) only where it beats persistence AND ARX with a resolved
  difference in at least 2 of 3 channels; the table is in the artifact.

Descriptive additions (no hypothesis of their own)
--------------------------------------------------
1. Hybrid budget: at N = 2 and 4 the hybrid with 8000 steps (``hybrid_steps8000``) against the one with 16000 steps,
   paired per seed; every verdict cell of the hybrid against the other methods at N = 2 and 4 is recomputed with the
   8000-step hybrid and the cells whose verdict differs are listed, as is HA3 recomputed with 8000 steps at N = 2, 4.
2. Fall-back to the nominal state: the runs for which the steady-state solver failed (the channel did not start in
   the steady state) are identified by re-solving the start of every run of the pools (the set points and
   disturbances of a seed are the same for every plant, solved once per seed on P0 and checked against the counts of
   the pool records). Per (seed, N) the number of such runs in the training prefix and in the 32 test runs is
   recorded. The hybrid is then compared with the other methods (a) on all seeds as before and (b) on the "clean"
   seeds only (no fall-back run in the training prefix of that N and none in the test runs), with the number of
   seeds; where fewer than 3 clean seeds exist the cell is marked low-power. A retraining without these runs was NOT
   done (the trained models were not kept): the test-side effect cannot be separated and is bounded by the share of
   affected test runs.

    tools/run_here.sh scripts/analyze_a1_main.py
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact
from run_a1_main import HOLDOUT_SEED, SEEDS, SIZES, make_plan
from run_a1_pilot_pp import generate, load_config

from twinlink_ml.data.splits import SplitName
from twinlink_ml.evaluation.paired_resolution import paired_comparison
from twinlink_ml.models import hybrid

OUTPUT_NAME = "a1_main_analysis"
RECORDS = "a1_main_records"
CHANNELS = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
PLANTS = ("P0", "P1", "P2a", "P2b", "P3")
STRUCTURAL = ("P2a", "P2b", "P3")
SMALL_N = (2, 4, 8, 16)
TEST_SETS = ("excited", "constant_holdout")
HORIZONS = ("h1", "h36")
SCRATCH, FINETUNE, HYBRID = "from_scratch", "pretrain_finetune", "hybrid"
HYBRID_8000, SIMULATOR, PERSISTENCE, ARX = (
    "hybrid_steps8000",
    "simulator_only",
    "persistence",
    "arx",
)
PARAMETER_FREE = (SIMULATOR, PERSISTENCE)


class Counter:
    """Counts the paired tests."""

    n = 0


def load_values() -> dict[tuple[Any, ...], float]:
    """(plant, seed, method, n, test_set, channel, horizon) -> RMSE (physical units, median over runs)."""
    payload = json.loads((RUNS_DIR / f"{RECORDS}.json").read_text(encoding="utf-8"))
    values: dict[tuple[Any, ...], float] = {}
    for record in payload["records"]:
        if record["kind"] != "result":
            continue
        for channel in CHANNELS:
            for horizon in HORIZONS:
                key = (
                    record["plant"],
                    record["seed"],
                    record["method"],
                    record["n"],
                    record["test_set"],
                    channel,
                    horizon,
                )
                values[key] = float(record["metrics"]["rmse_run_median_physical"][channel][horizon])
    return values


def series(
    values: dict[tuple[Any, ...], float],
    plant: str,
    method: str,
    n: int,
    test_set: str,
    channel: str,
    horizon: str = "h36",
    seeds: tuple[int, ...] = SEEDS,
) -> dict[int, float]:
    n_key = 0 if method in PARAMETER_FREE else n
    return {s: values[(plant, s, method, n_key, test_set, channel, horizon)] for s in seeds}


def compare(
    values: dict[tuple[Any, ...], float],
    plant: str,
    candidate: str,
    reference: str,
    n: int,
    test_set: str,
    channel: str,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    cand = series(values, plant, candidate, n, test_set, channel, seeds=seeds)
    ref = series(values, plant, reference, n, test_set, channel, seeds=seeds)
    result = paired_comparison(cand, ref, candidate_label=candidate, reference_label=reference)
    Counter.n += 1
    if result.degenerate or not result.is_resolved:
        verdict = "unresolved"
    else:
        verdict = "better" if result.mean_difference > 0 else "worse"
    return {
        "plant": plant,
        "test_set": test_set,
        "n": n,
        "channel": channel,
        "candidate": candidate,
        "reference": reference,
        "n_seeds": len(seeds),
        "mean_candidate": result.mean_candidate,
        "mean_reference": result.mean_reference,
        "mean_difference": result.mean_difference,
        "threshold": result.threshold,
        "p_value": result.p_value,
        "verdict": verdict,
    }


def count_channels(cells: list[dict[str, Any]], verdict: str) -> int:
    return sum(1 for c in cells if c["verdict"] == verdict)


# ---------------------------------------------------------------- learning curves


def learning_curves(values: dict[tuple[Any, ...], float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    methods = (SCRATCH, FINETUNE, HYBRID, ARX, SIMULATOR, PERSISTENCE, HYBRID_8000)
    for plant, test_set, method in itertools.product(PLANTS, TEST_SETS, methods):
        sizes = (
            (0,)
            if method in PARAMETER_FREE
            else (tuple(n for n in SIZES if n in (2, 4)) if method == HYBRID_8000 else SIZES)
        )
        for n in sizes:
            for channel, horizon in itertools.product(CHANNELS, HORIZONS):
                data = [values[(plant, s, method, n, test_set, channel, horizon)] for s in SEEDS]
                out.setdefault(plant, {}).setdefault(test_set, {}).setdefault(
                    method, {}
                ).setdefault(str(n), {}).setdefault(channel, {})[horizon] = {
                    "median": float(np.median(data)),
                    "min": float(min(data)),
                    "max": float(max(data)),
                    "by_seed": data,
                }
    return out


# -------------------------------------------------------------------- comparisons


def all_comparisons(values: dict[tuple[Any, ...], float]) -> list[dict[str, Any]]:
    pairs = [
        (FINETUNE, SCRATCH),
        (HYBRID, SCRATCH),
        (HYBRID, FINETUNE),
        (HYBRID, SIMULATOR),
        (FINETUNE, SIMULATOR),
        (SCRATCH, SIMULATOR),
    ]
    cells = []
    for plant, test_set, n, channel in itertools.product(PLANTS, TEST_SETS, SIZES, CHANNELS):
        for candidate, reference in pairs:
            cells.append(compare(values, plant, candidate, reference, n, test_set, channel))
        for method in (SCRATCH, FINETUNE, HYBRID):
            for baseline in (PERSISTENCE, ARX):
                cells.append(compare(values, plant, method, baseline, n, test_set, channel))
        cells.append(compare(values, plant, ARX, PERSISTENCE, n, test_set, channel))
        cells.append(compare(values, plant, SIMULATOR, ARX, n, test_set, channel))
    return cells


def index_cells(cells: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {
        (c["plant"], c["test_set"], c["n"], c["channel"], c["candidate"], c["reference"]): c
        for c in cells
    }


def channel_cells(
    index: dict[tuple[Any, ...], dict[str, Any]],
    plant: str,
    n: int,
    candidate: str,
    reference: str,
    test_set: str = "excited",
) -> list[dict[str, Any]]:
    return [index[(plant, test_set, n, ch, candidate, reference)] for ch in CHANNELS]


# --------------------------------------------------------------------------- HA1-3


def ratio_scratch_over_finetune(values: dict[tuple[Any, ...], float], plant: str, n: int) -> float:
    ratios = []
    for channel in CHANNELS:
        scratch = float(
            np.median(list(series(values, plant, SCRATCH, n, "excited", channel).values()))
        )
        tuned = float(
            np.median(list(series(values, plant, FINETUNE, n, "excited", channel).values()))
        )
        ratios.append(scratch / tuned)
    return float(np.mean(ratios))


def crossover(
    index: dict[tuple[Any, ...], dict[str, Any]], plant: str, candidate: str, reference: str
) -> dict[str, Any]:
    """N* of ``candidate`` against ``reference``: smallest N from which it is not resolved better (>= 2 of 3)."""
    better = {
        n: count_channels(channel_cells(index, plant, n, candidate, reference), "better") >= 2
        for n in SIZES
    }
    worse = {
        n: count_channels(channel_cells(index, plant, n, candidate, reference), "worse") >= 2
        for n in SIZES
    }
    n_star = None
    for n in SIZES:
        if all(not better[m] for m in SIZES if m >= n):
            n_star = n
            break
    last_better = max((n for n in SIZES if better[n]), default=None)
    return {
        "candidate": candidate,
        "reference": reference,
        "resolved_better_at": [n for n in SIZES if better[n]],
        "resolved_worse_at": [n for n in SIZES if worse[n]],
        "n_star": n_star,
        "bracket": None
        if n_star is None
        else {"largest_n_resolved_better_before": last_better, "n_star": n_star},
        "statement": (
            "not reached: resolved better at N = 64"
            if n_star is None
            else f"no longer resolved better from N = {n_star}"
            + ("" if last_better is None else f" (last resolved better N = {last_better})")
        ),
    }


def ha1(
    values: dict[tuple[Any, ...], float], index: dict[tuple[Any, ...], dict[str, Any]]
) -> dict[str, Any]:
    part_a = {}
    for plant in ("P0", "P1"):
        part_a[plant] = {
            n: count_channels(channel_cells(index, plant, n, FINETUNE, SCRATCH), "better")
            for n in (2, 4)
        }
    a_ok = all(part_a[p][n] >= 2 for p in part_a for n in (2, 4))
    ratios = {
        plant: {str(n): ratio_scratch_over_finetune(values, plant, n) for n in (2, 64)}
        for plant in ("P0", "P1")
    }
    b_ok = all(ratios[p]["64"] < ratios[p]["2"] for p in ratios)
    return {
        "part_a_channels_better_at_N2_N4": part_a,
        "part_a_ok": bool(a_ok),
        "part_b_ratio_scratch_over_finetune": ratios,
        "part_b_ok": bool(b_ok),
        "supported": bool(a_ok and b_ok),
        "crossover_finetune_vs_scratch": {
            plant: crossover(index, plant, FINETUNE, SCRATCH) for plant in PLANTS
        },
    }


def ha2(index: dict[tuple[Any, ...], dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"plants": {}}
    smallest: dict[str, int | None] = {}
    for plant in ("P3", "P2a"):
        worse_n = [
            n
            for n in SIZES
            if count_channels(channel_cells(index, plant, n, FINETUNE, SCRATCH), "worse") >= 2
        ]
        out["plants"][plant] = {
            "n_with_worse_in_2_of_3_channels": worse_n,
            "smallest_n": min(worse_n) if worse_n else None,
        }
        smallest[plant] = min(worse_n) if worse_n else None
    out["supported"] = any(v is not None for v in smallest.values())
    return out


def hybrid_beats_all(
    index: dict[tuple[Any, ...], dict[str, Any]],
    plant: str,
    n: int,
    hybrid_method: str,
    test_set: str = "excited",
) -> dict[str, Any]:
    per_reference = {
        ref: count_channels(channel_cells(index, plant, n, hybrid_method, ref, test_set), "better")
        for ref in (SCRATCH, SIMULATOR, FINETUNE)
    }
    return {"channels_better": per_reference, "ok": all(v >= 2 for v in per_reference.values())}


def ha3(
    index: dict[tuple[Any, ...], dict[str, Any]], substitute: dict[int, str] | None = None
) -> dict[str, Any]:
    plants: dict[str, Any] = {}
    for plant in STRUCTURAL:
        cells = {}
        for n in SMALL_N:
            method = (substitute or {}).get(n, HYBRID)
            cells[str(n)] = hybrid_beats_all(index, plant, n, method)
        count = sum(1 for n in SMALL_N if cells[str(n)]["ok"])
        plants[plant] = {"by_n": cells, "n_small_n_ok": count, "plant_ok": count >= 3}
    return {
        "plants": plants,
        "supported": sum(1 for p in plants.values() if p["plant_ok"]) >= 2,
        "substitution": substitute,
    }


def baseline_table(index: dict[tuple[Any, ...], dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for plant, n in itertools.product(PLANTS, SIZES):
        for method in (SCRATCH, FINETUNE, HYBRID):
            vs_p = count_channels(channel_cells(index, plant, n, method, PERSISTENCE), "better")
            vs_a = count_channels(channel_cells(index, plant, n, method, ARX), "better")
            table.setdefault(plant, {}).setdefault(method, {})[str(n)] = {
                "channels_better_than_persistence": vs_p,
                "channels_better_than_arx": vs_a,
                "useful": bool(vs_p >= 2 and vs_a >= 2),
            }
    return table


# ------------------------------------------------------------- descriptive 1: budget


def budget_sensitivity(
    values: dict[tuple[Any, ...], float], index: dict[tuple[Any, ...], dict[str, Any]]
) -> dict[str, Any]:
    direct = []
    extra_cells: list[dict[str, Any]] = []
    for plant, n, channel in itertools.product(PLANTS, (2, 4), CHANNELS):
        direct.append(compare(values, plant, HYBRID_8000, HYBRID, n, "excited", channel))
        for ref in (SCRATCH, SIMULATOR, FINETUNE):
            extra_cells.append(compare(values, plant, HYBRID_8000, ref, n, "excited", channel))
    extra_index = index_cells(extra_cells)
    changed = []
    for plant, n, channel, ref in itertools.product(
        PLANTS, (2, 4), CHANNELS, (SCRATCH, SIMULATOR, FINETUNE)
    ):
        main = index[(plant, "excited", n, channel, HYBRID, ref)]
        alt = extra_index[(plant, "excited", n, channel, HYBRID_8000, ref)]
        if main["verdict"] != alt["verdict"]:
            changed.append(
                {
                    "plant": plant,
                    "n": n,
                    "channel": channel,
                    "against": ref,
                    "verdict_16000": main["verdict"],
                    "verdict_8000": alt["verdict"],
                    "mean_candidate_16000": main["mean_candidate"],
                    "mean_candidate_8000": alt["mean_candidate"],
                }
            )
    merged = dict(index)
    merged.update(extra_index)
    ha3_main = ha3(index)
    ha3_alt = ha3(merged, substitute={2: HYBRID_8000, 4: HYBRID_8000})
    return {
        "hybrid8000_vs_hybrid16000_paired": direct,
        "n_cells_compared": len(extra_cells),
        "cells_with_changed_verdict": changed,
        "ha3_with_16000": ha3_main["supported"],
        "ha3_with_8000_at_n2_n4": ha3_alt["supported"],
        "ha3_plant_counts_16000": {p: v["n_small_n_ok"] for p, v in ha3_main["plants"].items()},
        "ha3_plant_counts_8000_at_n2_n4": {
            p: v["n_small_n_ok"] for p, v in ha3_alt["plants"].items()
        },
        "ha3_detail_with_8000": ha3_alt,
    }


# --------------------------------------------------- descriptive 2: nominal fall-back


def fallback_masks() -> dict[str, Any]:
    """Which runs of each seed's pool (and of the constant-run holdout) started from the nominal state."""
    plan = make_plan(False)
    config = load_config()
    per_seed: dict[int, Any] = {}
    for seed in SEEDS:
        data = generate(config, "P0", plan.pool_spec, seed)
        trajectory = data.trajectory
        set_points = (
            trajectory.requested_actions
            if trajectory.requested_actions is not None
            else trajectory.actions
        )
        _, solved, _ = hybrid.stationary_initial_states(
            data.base_config, set_points[:, 0, :], trajectory.disturbances[:, 0, :]
        )
        position = {run: i for i, run in enumerate(trajectory.run_id)}
        train = [position[r] for r in data.split.as_mapping[SplitName.TRAIN]]
        test = [position[r] for r in data.split.as_mapping[SplitName.TEST]]
        per_seed[seed] = {
            "failed_train_prefix_positions": [i for i, run in enumerate(train) if not solved[run]],
            "n_failed_in_test": int(sum(1 for run in test if not solved[run])),
            "n_failed_in_pool": int((~solved).sum()),
            "n_runs_pool": len(solved),
        }
        print(
            f"seed {seed}: fallback runs in pool {per_seed[seed]['n_failed_in_pool']}", flush=True
        )
    holdout = generate(config, "P0", plan.holdout_spec, HOLDOUT_SEED)
    t = holdout.trajectory
    sp = t.requested_actions if t.requested_actions is not None else t.actions
    _, solved_h, _ = hybrid.stationary_initial_states(
        holdout.base_config, sp[:, 0, :], t.disturbances[:, 0, :]
    )
    position = {run: i for i, run in enumerate(t.run_id)}
    test_h = [position[r] for r in holdout.split.as_mapping[SplitName.TEST]]
    return {
        "per_seed": per_seed,
        "holdout": {
            "n_failed_in_pool": int((~solved_h).sum()),
            "n_runs_pool": len(solved_h),
            "n_failed_in_test": int(sum(1 for run in test_h if not solved_h[run])),
            "n_test_runs": len(test_h),
        },
    }


def fallback_analysis(
    values: dict[tuple[Any, ...], float],
    index: dict[tuple[Any, ...], dict[str, Any]],
    masks: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads((RUNS_DIR / f"{RECORDS}.json").read_text(encoding="utf-8"))
    pool_counts: dict[tuple[str, int, str], int] = {}
    for record in payload["records"]:
        if record["kind"] == "pool":
            pool_counts[(record["plant"], record["seed"], record["pool"])] = record[
                "simulator_in_hybrid_channel"
            ]["n_nominal_fallback"]
    consistent = all(
        pool_counts[(plant, seed, "excited")] == masks["per_seed"][seed]["n_failed_in_pool"]
        for plant in PLANTS
        for seed in SEEDS
    ) and all(
        pool_counts[(plant, seed, "constant_holdout")] == masks["holdout"]["n_failed_in_pool"]
        for plant in PLANTS
        for seed in SEEDS
    )
    clean_seeds: dict[int, tuple[int, ...]] = {}
    seed_table: dict[str, Any] = {}
    for n in SMALL_N:
        clean = []
        for seed in SEEDS:
            prefix = [i for i in masks["per_seed"][seed]["failed_train_prefix_positions"] if i < n]
            test_failed = masks["per_seed"][seed]["n_failed_in_test"]
            seed_table.setdefault(str(n), {})[str(seed)] = {
                "fallback_runs_in_training_prefix": len(prefix),
                "fallback_runs_in_test": test_failed,
            }
            if not prefix and test_failed == 0:
                clean.append(seed)
        clean_seeds[n] = tuple(clean)
    cells = []
    for plant, n, channel in itertools.product(PLANTS, SMALL_N, CHANNELS):
        seeds = clean_seeds[n]
        for ref in (SCRATCH, SIMULATOR, FINETUNE):
            all_seeds = index[(plant, "excited", n, channel, HYBRID, ref)]
            row: dict[str, Any] = {
                "plant": plant,
                "n": n,
                "channel": channel,
                "against": ref,
                "verdict_all_seeds": all_seeds["verdict"],
                "mean_difference_all_seeds": all_seeds["mean_difference"],
                "n_clean_seeds": len(seeds),
                "clean_seeds": list(seeds),
            }
            if len(seeds) >= 2:
                clean_cell = compare(values, plant, HYBRID, ref, n, "excited", channel, seeds=seeds)
                row["verdict_clean_seeds"] = clean_cell["verdict"]
                row["mean_difference_clean_seeds"] = clean_cell["mean_difference"]
                row["low_power"] = len(seeds) < 3
            else:
                row["verdict_clean_seeds"] = None
                row["low_power"] = True
            cells.append(row)
    comparable = [
        c for c in cells if c["verdict_clean_seeds"] is not None and c["n_clean_seeds"] >= 3
    ]
    changed = [c for c in comparable if c["verdict_clean_seeds"] != c["verdict_all_seeds"]]
    direction_flips = [
        c
        for c in comparable
        if c["verdict_clean_seeds"] != c["verdict_all_seeds"]
        and {"better", "worse"} <= {c["verdict_clean_seeds"], c["verdict_all_seeds"]}
    ]
    return {
        "mask_consistent_with_pool_records": bool(consistent),
        "fallback_runs_by_seed": masks["per_seed"],
        "holdout_fallback": masks["holdout"],
        "n_fallback_runs_in_all_pools_of_records": int(sum(pool_counts.values())),
        "seed_table_by_n": seed_table,
        "clean_seeds_by_n": {str(n): list(s) for n, s in clean_seeds.items()},
        "cells": cells,
        "n_cells_with_at_least_3_clean_seeds": len(comparable),
        "cells_with_changed_verdict": changed,
        "cells_with_direction_flip": direction_flips,
        "note": (
            "retraining without these runs was not done; the effect of fall-back runs among the test runs "
            "cannot be separated and is bounded by their share"
        ),
    }


def main() -> None:
    require_clean_tree()
    values = load_values()
    curves = learning_curves(values)
    cells = all_comparisons(values)
    index = index_cells(cells)
    n_main_tests = Counter.n

    h1 = ha1(values, index)
    h2 = ha2(index)
    h3 = ha3(index)
    baselines = baseline_table(index)
    crossover_all = {
        plant: {
            "finetune_vs_scratch": crossover(index, plant, FINETUNE, SCRATCH),
            "hybrid_vs_scratch": crossover(index, plant, HYBRID, SCRATCH),
            "finetune_vs_simulator_only": crossover(index, plant, FINETUNE, SIMULATOR),
            "hybrid_vs_simulator_only": crossover(index, plant, HYBRID, SIMULATOR),
        }
        for plant in PLANTS
    }
    budget = budget_sensitivity(values, index)
    masks = fallback_masks()
    fallback = fallback_analysis(values, index, masks)

    write_artifact(
        OUTPUT_NAME,
        {
            "_provenance": provenance(),
            "design": {
                "records": f"{RECORDS}.json",
                "metric": "rmse_run_median_physical, h = 36, per channel; h = 1 in the curves",
                "primary_test_set": "excited",
                "secondary_test_set": "constant_holdout",
                "seeds": list(SEEDS),
                "sizes": list(SIZES),
                "channels": list(CHANNELS),
                "n_paired_tests_main_comparisons": n_main_tests,
                "n_paired_tests_total": Counter.n,
                "multiplicity_correction": "none",
            },
            "ha1": h1,
            "ha2": h2,
            "ha3": h3,
            "ha3_winners_note": "see comparisons; per plant and N the verdict cells of the hybrid are in ha3.plants",
            "crossover": crossover_all,
            "baseline_rule": baselines,
            "budget_sensitivity": budget,
            "fallback_analysis": fallback,
            "learning_curves": curves,
            "comparisons": cells,
        },
    )
    print("HA1 supported:", h1["supported"], "| HA2:", h2["supported"], "| HA3:", h3["supported"])
    print(
        "budget: HA3 with 8000 at N=2,4:",
        budget["ha3_with_8000_at_n2_n4"],
        "| changed cells:",
        len(budget["cells_with_changed_verdict"]),
        "of",
        budget["n_cells_compared"],
    )
    print(
        "fallback: consistent",
        fallback["mask_consistent_with_pool_records"],
        "| changed:",
        len(fallback["cells_with_changed_verdict"]),
        "of",
        fallback["n_cells_with_at_least_3_clean_seeds"],
        "| flips:",
        len(fallback["cells_with_direction_flip"]),
    )
    print("tests:", Counter.n, math.isfinite(Counter.n))


if __name__ == "__main__":
    main()

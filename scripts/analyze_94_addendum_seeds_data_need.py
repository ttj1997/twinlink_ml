r"""Addendum to Befund 94: seeds needed in arm N, and the data need read from the learning curves.

Pure re-analysis of committed artifacts (no training, no new data). Requested by review 4 of the
synthesis (``docs/reviews/2026-10-19/review4_synthesis_v5.md``, findings 2 and 3). Rules fixed here,
before the run:

PART A -- SEEDS NEEDED, SAME METHOD AS BEFUND 83 ADDENDUM B, PER ARM.
* Cells: the ``rank_cells`` of ``rerun_evaluation_rerun_full.json`` with candidate GRU or Transformer
  (reference LSTM), arms A0, A1, N, every ``n_train``, test set, channel and horizon.
* Per cell ``analyze_83_addendum.seeds_row`` (paired differences of the five seeds, noncentral t,
  two-sided 5 %, power 0.80, delta 12.5 % and 10 % of the LSTM mean; 80 % interval of the standard
  deviation from its chi-square distribution; log version). These are the figures that
  ``analyze_rerun.py`` already stored in ``seeds_needed`` -- without the horizon. The script
  re-attaches the horizon and ASSERTS that every recomputed row equals the stored row (same order).
* A0 at ``n_train`` 64, h = 36 must reproduce ``paired_reevaluation_83_addendum.json`` (Befund 83
  addendum b): asserted for the 12.5 % and 10 % figures.
* Output: the table of Befund 83 addendum b for arm N (``n_train`` 64, h = 36 and h = 1) next to A0,
  and per arm, channel, test set and candidate the range of the 12.5 % figure over all budgets and
  both horizons.

PART B -- DATA NEED FROM THE LEARNING CURVES OF ARM N (records ``rerun_full_N.jsonl`` with
``eval/primary``, metric ``rmse_run_median_physical``, h = 36, the data of Figures 1 and 2 of the
synthesis; eight methods, ``n_train`` 8, 16, 32, 64, both test sets, three channels, five seeds paired by
``master_seed``).
* Per method, channel, test set and ``n_train``: median, minimum and maximum over the seeds.
* Ratio of the median at ``n_train`` to the median at 64 ("error at n relative to the error at 64").
* Does the curve still fall between 32 and 64? Paired t test (``paired_resolution``, as Befund 83) of
  the error at 64 against the error at 32 per method, channel and test set; family = all 48 such tests;
  verdict uncorrected and after Holm (5 %); the number of seeds with a lower error at 64.
* Runs needed to beat the static regression and persistence: per network (GRU, LSTM, Transformer,
  xLSTM, Mamba), channel and test set, the smallest ``n_train`` at which the network's error is below
  the baseline's in all five seeds (paired by ``master_seed``), or ``null`` if not reached at 64.
* Run counts behind every point: ``n_train`` training runs plus the validation and test runs of the
  record (``data/n_val``, ``data/n_test``), each run ``data/run_duration_s`` long.

Not claimed: anything beyond ``n_train`` 64; a statement about a plant (grey-box simulator with
placeholder parameters); the seed figures are planning figures from five seeds, not tests.

Call::

    tools/run_here.sh scripts/analyze_94_addendum_seeds_data_need.py --prefix rerun_full
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact
from analyze_83_addendum import holm, seeds_row
from analyze_paired_reevaluation import compare

ALPHA = 0.05
ARMS = ("A0", "A1", "N")
SEED_CANDIDATES = ("gru", "transformer")
METRIC = "rmse_run_median_physical"
HORIZON = 36
N_TRAIN = (8, 16, 32, 64)
CHANNELS = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
TEST_SETS = ("excited", "constant_holdout")
METHODS = ("persistence", "static", "arx", "gru", "lstm", "transformer", "xlstm", "mamba")
NETWORKS = ("gru", "lstm", "transformer", "xlstm", "mamba")
BASELINES = ("static", "persistence")
SEED_KEYS = ("seeds_0.125", "seeds_0.100")


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0) or (math.isnan(a) and math.isnan(b))
    return bool(a == b)


def seed_rows(
    rank_cells: list[dict[str, Any]], stored: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Recompute ``seeds_row`` per GRU/Transformer cell, attach the horizon, check the stored rows."""
    cells = [c for c in rank_cells if c["candidate"] in SEED_CANDIDATES]
    if len(cells) != len(stored):
        raise SystemExit(f"{len(cells)} candidate cells but {len(stored)} stored seed rows")
    rows = []
    for cell, old in zip(cells, stored, strict=True):
        row = seeds_row(
            {
                "channel": cell["channel"],
                "test_set": cell["test_set"],
                "n_train": cell["n_train"],
                "candidate": cell["candidate"],
                "reference": "lstm",
                "per_seed_candidate": cell["per_seed_candidate"],
                "per_seed_reference": cell["per_seed_reference"],
            }
        )
        if set(row) != set(old) or not all(_same(row[k], old[k]) for k in row):
            raise SystemExit(f"recomputed seed row differs from the stored one: {row} / {old}")
        rows.append({**row, "horizon": cell["horizon"]})
    return rows


def check_a0_against_83b(rows_a0: list[dict[str, Any]], addendum: dict[str, Any]) -> int:
    old = {
        (r["channel"], r["test_set"], r["candidate"]): r
        for r in addendum["seeds_needed"]
        if r["n_train"] == 64
    }
    checked = 0
    for row in rows_a0:
        if row["n_train"] != 64 or row["horizon"] != HORIZON:
            continue
        ref = old[(row["channel"], row["test_set"], row["candidate"])]
        for key in (
            *SEED_KEYS,
            *(f"{k}_sd_low" for k in SEED_KEYS),
            *(f"{k}_sd_high" for k in SEED_KEYS),
        ):
            if row[key] != ref[key]:
                raise SystemExit(f"A0 does not reproduce Befund 83 addendum b: {row} / {ref}")
        checked += 1
    if checked != len(old):
        raise SystemExit(f"A0 check covered {checked} of {len(old)} cells")
    return checked


def seed_tables(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    keep = (
        "channel",
        "test_set",
        "candidate",
        "sd_relative_to_reference_mean",
        "seeds_0.125",
        "seeds_0.125_sd_low",
        "seeds_0.125_sd_high",
        "seeds_0.100",
        "seeds_0.100_sd_low",
        "seeds_0.100_sd_high",
        "seeds_0.125_log",
        "seeds_0.100_log",
    )
    tables: dict[str, Any] = {}
    for arm in ("A0", "N"):
        for horizon in (36, 1):
            tables[f"{arm}_n64_h{horizon}"] = [
                {k: r[k] for k in keep}
                for r in rows[arm]
                if r["n_train"] == 64 and r["horizon"] == horizon
            ]
    ranges: dict[str, Any] = {}
    for arm in ARMS:
        for channel in CHANNELS:
            for test_set in TEST_SETS:
                for candidate in SEED_CANDIDATES:
                    sub = [
                        r["seeds_0.125"]
                        for r in rows[arm]
                        if r["channel"] == channel
                        and r["test_set"] == test_set
                        and r["candidate"] == candidate
                        and r["seeds_0.125"] is not None
                    ]
                    if sub:
                        ranges[f"{arm}/{channel}/{test_set}/{candidate}"] = {
                            "cells": len(sub),
                            "min": min(sub),
                            "median": statistics.median(sub),
                            "max": max(sub),
                        }
    tables["ranges_seeds_0.125_all_budgets_and_horizons"] = ranges
    return tables


def load_primary(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return [r for r in records if r["params"].get("eval/primary")]


def per_seed(
    records: list[dict[str, Any]], method: str, n: int, test_set: str, channel: str
) -> dict[int, float]:
    key = f"{METRIC}/{channel}/h{HORIZON}"
    values = {
        int(r["params"]["master_seed"]): float(r["metrics"][key])
        for r in records
        if r["tags"]["model"] == method
        and int(r["tags"]["n_train"]) == n
        and r["tags"]["test_set"] == test_set
    }
    if len(values) != 5:
        raise SystemExit(f"{method} n={n} {test_set} {channel}: {len(values)} seeds")
    return values


def data_need(records: list[dict[str, Any]]) -> dict[str, Any]:
    curves: dict[str, Any] = {}
    values: dict[tuple[str, str, str, int], dict[int, float]] = {}
    for method in METHODS:
        for channel in CHANNELS:
            for test_set in TEST_SETS:
                points = {}
                for n in N_TRAIN:
                    v = per_seed(records, method, n, test_set, channel)
                    values[(method, channel, test_set, n)] = v
                    points[str(n)] = {
                        "median": statistics.median(v.values()),
                        "min": min(v.values()),
                        "max": max(v.values()),
                    }
                at64 = points["64"]["median"]
                for n in N_TRAIN:
                    points[str(n)]["median_over_median_at_64"] = points[str(n)]["median"] / at64
                curves[f"{method}/{channel}/{test_set}"] = points
    falls = []
    for method in METHODS:
        for channel in CHANNELS:
            for test_set in TEST_SETS:
                cell = compare(
                    values[(method, channel, test_set, 64)],
                    values[(method, channel, test_set, 32)],
                    f"{method}@64",
                    f"{method}@32",
                )
                cell.update(
                    {"method": method, "channel": channel, "test_set": test_set, "horizon": HORIZON}
                )
                cell["seeds_lower_at_64"] = 5 - cell["seeds_candidate_worse"]
                falls.append(cell)
    p = np.array([1.0 if math.isnan(c["p_value"]) else c["p_value"] for c in falls])
    adjusted = holm(p)
    for cell, adj in zip(falls, adjusted, strict=True):
        cell["p_holm"] = float(adj)
        cell["falls_uncorrected"] = cell["verdict"] == "candidate better"
        cell["rises_uncorrected"] = cell["verdict"] == "candidate worse"
        cell["falls_holm"] = bool(cell["falls_uncorrected"] and adj < ALPHA)
    beats: dict[str, Any] = {}
    for network in NETWORKS:
        for baseline in BASELINES:
            for channel in CHANNELS:
                for test_set in TEST_SETS:
                    first = None
                    seeds_better = {}
                    for n in N_TRAIN:
                        net = values[(network, channel, test_set, n)]
                        base = values[(baseline, channel, test_set, n)]
                        better = sum(net[s] < base[s] for s in net)
                        seeds_better[str(n)] = better
                        if first is None and better == len(net):
                            first = n
                    beats[f"{network}/{baseline}/{channel}/{test_set}"] = {
                        "first_n_train_all_seeds_better": first,
                        "seeds_better_per_n_train": seeds_better,
                    }
    params = records[0]["params"]
    runs = {
        "n_val": params["data/n_val"],
        "n_test": params["data/n_test"],
        "run_duration_s": params["data/run_duration_s"],
        "sample_time_s": params["data/sample_time_s"],
        "holdout_runs": params["holdout/n_runs"],
    }
    for r in records:
        for key, name in (
            ("data/n_val", "n_val"),
            ("data/n_test", "n_test"),
            ("data/run_duration_s", "run_duration_s"),
        ):
            if r["params"][key] != runs[name]:
                raise SystemExit(f"{key} differs between records")
    summary = {
        "tests": len(falls),
        "falls_uncorrected": sum(c["falls_uncorrected"] for c in falls),
        "rises_uncorrected": sum(c["rises_uncorrected"] for c in falls),
        "falls_holm": sum(c["falls_holm"] for c in falls),
        "falls_uncorrected_networks": sum(
            c["falls_uncorrected"] for c in falls if c["method"] in NETWORKS
        ),
        "falls_holm_networks": sum(c["falls_holm"] for c in falls if c["method"] in NETWORKS),
        "tests_networks": sum(c["method"] in NETWORKS for c in falls),
    }
    return {
        "curves": curves,
        "falls_32_to_64": falls,
        "falls_32_to_64_summary": summary,
        "beats_baseline": beats,
        "runs_behind_each_point": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    arguments = parser.parse_args()
    require_clean_tree()
    source = RUNS_DIR / f"rerun_evaluation_{arguments.prefix}.json"
    evaluation = json.loads(source.read_text(encoding="utf-8"))
    addendum = json.loads((RUNS_DIR / "paired_reevaluation_83_addendum.json").read_text("utf-8"))
    rows = {
        arm: seed_rows(evaluation["rank_cells"][arm], evaluation["seeds_needed"][arm])
        for arm in ARMS
    }
    checked = check_a0_against_83b(rows["A0"], addendum)
    records = load_primary(RUNS_DIR / f"{arguments.prefix}_N.jsonl")
    payload = {
        "_provenance": provenance(),
        "rule": "see the module docstring of scripts/analyze_94_addendum_seeds_data_need.py",
        "sources": [
            source.name,
            f"{arguments.prefix}_N.jsonl",
            "paired_reevaluation_83_addendum.json",
        ],
        "part_a_seeds": {
            "stored_rows_reproduced": {arm: len(rows[arm]) for arm in ARMS},
            "a0_cells_reproducing_befund_83b": checked,
            "rows": rows,
            **seed_tables(rows),
        },
        "part_b_data_need": data_need(records),
    }
    write_artifact(f"befund94_addendum_seeds_data_need_{arguments.prefix}", payload)
    for row in payload["part_a_seeds"]["N_n64_h36"]:
        print("N n64 h36", row["channel"], row["test_set"], row["candidate"], row["seeds_0.125"])
    print(payload["part_b_data_need"]["falls_32_to_64_summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

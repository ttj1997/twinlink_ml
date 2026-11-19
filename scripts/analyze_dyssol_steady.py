"""Surrogates of stationary Dyssol flowsheets: the pre-registered reading.

Question (decision of the project lead, 2026-11-19): how many Dyssol runs does a surrogate of a
stationary flowsheet need when no rebuild of the flowsheet exists? The balance itself is exact
(the Python rebuild reproduces Dyssol; control below); the study measures the data need of a
surrogate, not whether a learned model beats the balance (Befund 120, docs/37).

PRE-REGISTERED READING (fixed before any Dyssol data set existed; the same for every case)
------------------------------------------------------------------------------------------
Data: per case one training pool and one test set of scrambled Sobol points in the box of the
varied inputs (five per line), every point one Dyssol run. Over-saturated points (the dryer
exhaust holds more water than saturated air; the unit has no saturation limit) are flagged and
counted, NOT removed: they are part of the pool, of the test set and of every number below.

Test set: the in-box Sobol test set, independent of the training pool.

Per cell (model, N, seed): fit on the first N points of the seed's permutation of the pool, score
on the test set. Two numbers per cell:

* ``exhaust_p95_k``: 95th percentile over the test set of the absolute error of the exhaust
  temperature after the heat recovery [K]; with two lines the larger of the two;
* ``fuel_p95_percent``: 95th percentile of the relative error of the TOTAL fuel consumption [%].

Thresholds: 1 K and 1 %. A model meets them at N if the MEDIAN over the five seeds of both
numbers is at or below its threshold. ``N*`` of a model: the smallest N of the grid from which on
it meets them at every larger N of the grid as well.

0. Control (before any fit): Dyssol against the rebuild at every point; above the bounds of the
   YAML the data set is not used.
1. Feasibility first: per model and N the number of seeds (of five) that meet both thresholds.
2. Verdict:
   D1  the MLP has an N* and no baseline (linear, quadratic regression) has one at or below it:
       an MLP is needed; data need = N* of the MLP.
   D2  a baseline has an N* at or below that of the MLP, or the MLP has none: a regression
       suffices; data need = the smallest baseline N* (the simpler model at equal N*).
   D3  no model has an N*: the data need exceeds the largest N of the grid.
3. Rank claims (MLP against each baseline, per N and per number): paired t-test over the seeds
   (same training subsets), Holm per family; a family is one baseline and one number over all N.
   A rank is claimed only where the Holm-adjusted test is resolved. The verdict does not depend
   on these tests.

Reported WITHOUT a reading: the same numbers on the admissible (not over-saturated) test points;
median and maximum error next to the 95th percentile; the other targets (dryer outlet
temperature, preheated-air temperature, recovered heat, fuel per line).

Usage: ``tools/run_here.sh scripts/analyze_dyssol_steady.py --training <name> --name <name>``
(reads ``artifacts/runs/<training>.json.gz``, writes ``artifacts/runs/<name>.json``).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact  # noqa: E402

from twinlink_ml.dyssol_steady import study  # noqa: E402


def median_table(
    cells: list[dict[str, Any]], key: str, config: study.StudyConfig
) -> dict[str, Any]:
    """Median and range over the seeds of every metric of ``key`` per model and N."""
    table: dict[str, Any] = {}
    for model in study.MODELS:
        table[model] = {}
        for n in config.budgets:
            group = [c[key] for c in cells if c["model"] == model and c["n"] == n and key in c]
            if not group:
                continue
            table[model][str(n)] = {
                metric: {
                    "median": float(np.median([g[metric] for g in group])),
                    "min": float(min(g[metric] for g in group)),
                    "max": float(max(g[metric] for g in group)),
                }
                for metric in group[0]
            }
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training", required=True)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_clean_tree()
    config = study.load_config()
    with gzip.open(RUNS_DIR / f"{args.training}.json.gz", "rt", encoding="utf-8") as fh:
        training = json.load(fh)
    if training["header"]["config_fingerprint"] != config.fingerprint:
        raise SystemExit("the training artifact was written with another config")
    cells = training["cells"]
    result = study.reading(cells, config)
    tests = study.rank_tests(cells, config)
    admissible = [
        {**c, "metrics": c["metrics_admissible"]} for c in cells if "metrics_admissible" in c
    ]
    payload = {
        "_provenance": provenance(),
        "training": args.training,
        "case": training["header"]["case"],
        "test_set": training["test_set"],
        "data_summary": training["data"]["summary"],
        "reading": result,
        "rank_tests": tests,
        "without_reading": {
            "reading_on_admissible_test_points": (
                study.reading(admissible, config) if len(admissible) == len(cells) else None
            ),
            "all_metrics": median_table(cells, "metrics", config),
        },
    }
    print(
        f"case {payload['case']}: verdict {result['verdict']}, data need "
        f"{result['data_need_runs']} runs, reached by {result['reached_by']}"
    )
    print("N* per model:", json.dumps(result["n_star"]))
    for model in study.MODELS:
        for n in config.budgets:
            row = result["table"][model][n]
            print(
                f"  {model:9s} N={n:4d}  exhaust P95 {row['exhaust_p95_k']['median']:9.4f} K  "
                f"fuel P95 {row['fuel_p95_percent']['median']:9.4f} %  "
                f"seeds meeting both {row['seeds_meeting_both']}/{len(config.seeds)}  met={row['met']}"
            )
    resolved = [t for t in tests if t["resolved_holm"]]
    print(f"rank tests resolved after Holm: {len(resolved)} of {len(tests)}")
    write_artifact(args.name, payload)


if __name__ == "__main__":
    main()

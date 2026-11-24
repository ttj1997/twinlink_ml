"""Surrogate of the one-line Dyssol flowsheet with the fresh-gas state as an input: the reading.

Design: ``docs/NN_dyssol_freshgas.md``; settings ``conf/dyssol_steady/dyssol_freshgas.yaml``;
data and fits by ``scripts/run_dyssol_steady.py --config dyssol_freshgas.yaml``.

**Occasion (Befund 130, open point 3; docs/45 section 17).** The partner sheet of 2026-11-24 (two
one-line processes over eleven flows of the heat transfer medium) was no valid test of the
surrogates of Befund 128: the fresh-gas state is not one of their inputs and 3 of 22 points lie in
their box. This study generates a data set in which the fresh-gas temperature, the relative
humidity of the fresh gas and the full range of the medium flow are varied inputs (seven inputs,
case ``one_line`` only).

**Placing.** The Python rebuild of the units is exact and available (Befund 128; the control below
checks it again at every point). The study does not ask whether a learned model beats the balance
(Befund 120); it measures the data need of a surrogate for the case without a rebuild and tests
the rule of thumb of Befund 128.

PRE-REGISTERED READING (fixed in this docstring before the first run of this script and before any
Dyssol data set of this study existed)
------------------------------------------------------------------------------------------------
Everything below stands under the reading "the gas mass flow of the sheet is the DRY gas". That
reading is inferred from the agreement in Befund 130 and is not confirmed by the sender.

**Step 0, control (before any fit).** Dyssol against the rebuild at EVERY point of the data set
(pool, Sobol test set, points of the sheet), bounds of the YAML (1e-6 K; 1e-6 % of heat flows and
fuel relative to max(|value|, 1)), as docs/45. The largest deviation is tabulated over the
fresh-gas temperature and given for the points whose heater inlet is colder than the lower bound
of the vapour-enthalpy correlation. If the control exceeds the bounds, the data set is not used,
the deviation is reported and the study stops (question to the project lead); the training stage
refuses to fit.

**Reading D (data need), as docs/45 section 6, unchanged.** Test set: **in-box Sobol test set,
7 inputs** (256 scrambled Sobol points in the box, independent of the pool of 1024). Per cell
(model, N, seed) the 95th percentile over the test set of the absolute error of the exhaust
temperature after the heat recovery [K] and of the relative error of the fuel [%]. Thresholds 1 K
and 1 %; a model meets them at N if the MEDIAN over the five seeds of both numbers is at or below
its threshold; ``N*`` = the smallest N of the grid (8 ... 512) from which on it meets them at
every larger N of the grid as well.

1. Feasibility first: per model and N the number of seeds (of five) that meet both thresholds.
2. Verdict: **D1** the MLP has an N* and no baseline (linear, quadratic regression) has one at or
   below it; **D2** a baseline has an N* at or below that of the MLP, or the MLP has none (data
   need = the smallest baseline N*, the simpler model at equal N*); **D3** no model has an N*.
3. Rank claims (MLP against each baseline, per N and number) only from the paired test over the
   seeds (``evaluation/paired_resolution.py``) with Holm per family (one baseline, one number,
   all N). The verdict does not depend on them.

**Prediction (fixed before the data).** Befund 128: N* of the quadratic regression was 32 and 128
at 21 and 66 coefficients, a data need of about 1.5 to 2 times the number of coefficients. With
seven inputs the quadratic regression has 36 coefficients; predicted N* = 64, the first step of
the grid above 1.5 x 36 = 54. **Confirmed** if N* of the quadratic regression is 64; **not
confirmed** otherwise (the found N* is reported). Whether the data need of reading D is that N* is
reported next to it.

**Reading B (the sheet).** Second test set: **partner sheet 2026-11-24, process 2, 11 points**.
The truth at each point is OUR Dyssol run there. For the model of reading D at its N* (the model
that reaches the data need; with D3 there is none and B is "not applicable"), per seed the LARGEST
error over the eleven points of the exhaust temperature [K] and of the fuel [%]:

* **B0** the median over the seeds of both largest errors is at or below its threshold (1 K, 1 %);
* **B1** otherwise.

**Disclosure.** The values of the sheet and the results of Befund 130 are known (our Dyssol runs
and the rebuild at these eleven points, their agreement with the sheet). The points of the sheet
are NOT a blind test. They enter neither the pool nor the Sobol test set nor the pilot that
selects the MLP setting (the pilot stage does not load the sheet; the data stage aborts if a
design point coincides with a point of the sheet).

**Reported WITHOUT a reading:** reading D on the admissible test points (dryer exhaust not above
saturation, the flag of docs/45) and on the points that are admissible also after the heat
recovery (rule of docs/49); the counts of over-saturated points per set; median and maximum error
next to the 95th percentile, and the other targets; for every model and N the largest errors on
the sheet; the surrogate of reading B against the sheet's own values (how many of the reported
values lie within half a unit / one unit of the last reported decimal, as
``dyssol_steady.sweep_sheet.compare``), next to the same counts for our Dyssol runs; where the
points of the sheet lie in the box (inside; strictly inside in the fresh-gas state; on a face).

Process 1 of the sheet is over-saturated at every point (ratio 2.015, Befund 130) and is not part
of this study: no statement.

Usage: ``tools/run_here.sh scripts/analyze_dyssol_freshgas.py --training <name> --name <name>``
(reads ``artifacts/runs/<training>.json.gz``, writes ``artifacts/runs/<name>.json``; the artifact
holds errors in physical units, counts and digests -- no input value, no value of the sheet).
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
sys.path.insert(0, str(SCRIPTS))

from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact  # noqa: E402
from analyze_dyssol_steady import median_table  # noqa: E402

from twinlink_ml.dyssol_steady import study  # noqa: E402

CONFIG = SCRIPTS.parent / "conf" / "dyssol_steady" / "dyssol_freshgas.yaml"


def with_metrics(cells: list[dict[str, Any]], key: str) -> list[dict[str, Any]] | None:
    """The cells with ``key`` as their metrics, or ``None`` unless every cell has it."""
    chosen = [{**c, "metrics": c[key]} for c in cells if key in c]
    return chosen if len(chosen) == len(cells) else None


def sheet_table(cells: list[dict[str, Any]], config: study.StudyConfig) -> dict[str, Any]:
    """Per model and N: median and range over the seeds of the largest errors on the sheet."""
    table: dict[str, Any] = {}
    for model in study.MODELS:
        table[model] = {}
        for n in config.budgets:
            group = [c["metrics_sheet"] for c in cells if c["model"] == model and c["n"] == n]
            entry: dict[str, Any] = {}
            for key in ("exhaust_max_k", "fuel_max_percent"):
                values = [g[key] for g in group]
                entry[key] = {
                    "median": float(np.median(values)),
                    "min": float(min(values)),
                    "max": float(max(values)),
                }
            entry["seeds_meeting_both"] = sum(
                g["exhaust_max_k"] <= config.threshold_exhaust_k
                and g["fuel_max_percent"] <= config.threshold_fuel_percent
                for g in group
            )
            table[model][str(n)] = entry
    return table


def against_sheet(cells: list[dict[str, Any]], model: str | None, n: int | None) -> Any:
    """Counts of the surrogate of reading B against the sheet's own values, per seed."""
    if model is None or n is None:
        return None
    group = sorted(
        (c for c in cells if c["model"] == model and c["n"] == n), key=lambda c: c["seed"]
    )
    return [
        {
            "seed": c["seed"],
            "total": c["metrics_sheet"]["against_sheet"]["total"],
            "per_quantity": {
                key: {k: entry[k] for k in ("n", "within_half", "within_one", "max_abs")}
                for key, entry in c["metrics_sheet"]["against_sheet"]["per_quantity"].items()
            },
        }
        for c in group
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training", required=True)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_clean_tree()
    config = study.load_config(CONFIG)
    if config.prediction is None or config.sheet_test is None:
        raise SystemExit("the settings name no prediction or no sheet test set")
    with gzip.open(RUNS_DIR / f"{args.training}.json.gz", "rt", encoding="utf-8") as fh:
        training = json.load(fh)
    if training["header"]["config_fingerprint"] != config.fingerprint:
        raise SystemExit("the training artifact was written with another config")
    if training["header"]["dry_run"]:
        raise SystemExit("a dry-run output is never read")
    cells = training["cells"]
    control = training["data"]["summary"]["control"]
    if (
        control["max_temperature_k"] > config.control_temperature_k
        or control["max_relative_percent"] > config.control_relative_percent
    ):
        raise SystemExit("control exceeded: no reading")
    result = study.reading(cells, config)
    tests = study.rank_tests(cells, config)
    prediction = study.prediction_check(result, config.prediction)
    reading_b = study.sheet_reading(cells, config, result["reached_by"], result["data_need_runs"])
    admissible = with_metrics(cells, "metrics_admissible")
    strict = with_metrics(cells, "metrics_admissible_strict")
    payload = {
        "_provenance": provenance(),
        "training": args.training,
        "case": training["header"]["case"],
        "conditional_on": "gas mass flow of the sheet = dry gas (inferred in Befund 130, "
        "not confirmed by the sender)",
        "test_set": training["test_set"],
        "sheet_test": training["sheet_test"],
        "data_summary": training["data"]["summary"],
        "control": {
            "bounds": {
                "temperature_k": config.control_temperature_k,
                "relative_percent": config.control_relative_percent,
            },
            "held": True,
            **control,
        },
        "reading": result,
        "prediction": prediction,
        "rank_tests": tests,
        "reading_sheet": reading_b,
        "without_reading": {
            "reading_on_admissible_test_points": (
                study.reading(admissible, config) if admissible is not None else None
            ),
            "reading_on_test_points_admissible_also_after_recovery": (
                study.reading(strict, config) if strict is not None else None
            ),
            "test_points_admissible_also_after_recovery": training.get(
                "test_set_admissible_also_after_recovery"
            ),
            "all_metrics": median_table(cells, "metrics", config),
            "sheet_largest_errors": sheet_table(cells, config),
            "surrogate_of_reading_b_against_sheet": against_sheet(
                cells, result["reached_by"], result["data_need_runs"]
            ),
        },
    }
    print(
        f"control held: {control['max_temperature_k']:.3g} K, "
        f"{control['max_relative_percent']:.3g} % at {control['points_compared']} points"
    )
    for row in control.get("by_fresh_gas_celsius", []):
        print(
            f"  fresh gas {row['lower']:6.1f} .. {row['upper']:6.1f} degC: {row['points']:4d} points, "
            f"{row['max_temperature_k']:.3g} K, {row['max_relative_percent']:.3g} %"
        )
    print(
        f"reading D: {result['verdict']}, data need {result['data_need_runs']} runs, "
        f"reached by {result['reached_by']}; N* {json.dumps(result['n_star'])}"
    )
    for model in study.MODELS:
        for n in config.budgets:
            row = result["table"][model][n]
            print(
                f"  {model:9s} N={n:4d}  exhaust P95 {row['exhaust_p95_k']['median']:9.4f} K  "
                f"fuel P95 {row['fuel_p95_percent']['median']:9.4f} %  "
                f"seeds meeting both {row['seeds_meeting_both']}/{len(config.seeds)}  met={row['met']}"
            )
    print(
        f"prediction N*({prediction['model']}) = {prediction['predicted_n_star']}: found "
        f"{prediction['found_n_star']} -> {'confirmed' if prediction['confirmed'] else 'not confirmed'}"
    )
    print("reading B:", json.dumps(reading_b))
    resolved = [t for t in tests if t["resolved_holm"]]
    print(f"rank tests resolved after Holm: {len(resolved)} of {len(tests)}")
    write_artifact(args.name, payload)


if __name__ == "__main__":
    main()

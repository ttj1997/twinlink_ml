r"""Addendum to Befund 55: the ratio "constant-trained over excited-trained" recomputed per model and budget.

Pure re-reading of the two committed record files of Befund 55 (no training, no new data). Reason:
review 4 of the synthesis (``docs/reviews/2026-10-19/review4_synthesis_v5.md``, finding 10) recomputed the
GRU range of the moisture ratio as 2.22-2.52, while Befund 55 writes "2,2-3,1 (GRU)". Rules fixed here,
before the run, exactly as Befund 55 defines its numbers:

* Records: ``arch_learning_curve_constant_trained.jsonl`` (trained on constant runs) and
  ``arch_learning_curve_projected.jsonl`` (Befund 51, trained on the excited pool); test set of both:
  ``constant_holdout`` (the same constant-run holdout, ``HOLDOUT_SEED`` 999), metric
  ``nrmse_run_median/<channel>/h36``; models GRU and LSTM; ``n_train`` 8, 16, 32, 64; five seeds.
* Per model, channel and ``n_train``: median over the seeds in each arm, the ratio
  constant / excited of the medians, and the seed range (max - min) in each arm.
* The seed-range check of Befund 55 (substitute yardstick, not a test): difference of the medians
  against the larger of the two within-arm seed ranges.
* Data configuration of the records (process fingerprint, ``data/future_actions``,
  ``data/solid_fraction_process``) so that the addendum can name the data arm.

Call::

    tools/run_here.sh scripts/analyze_55_addendum_ratio_range.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact

MODELS = ("gru", "lstm")
N_TRAIN = ("8", "16", "32", "64")
CHANNELS = ("product_moisture_kg_per_kg", "energy_target", "bed_temperature_celsius")
TEST_SET = "constant_holdout"
FILES = {
    "constant_trained": "arch_learning_curve_constant_trained",
    "excited_trained": "arch_learning_curve_projected",
}


def load(name: str) -> list[dict[str, Any]]:
    with (RUNS_DIR / f"{name}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def values(records: list[dict[str, Any]], model: str, n: str, channel: str) -> list[float]:
    key = f"nrmse_run_median/{channel}/h36"
    out = [
        r["metrics"][key]
        for r in records
        if r["tags"].get("model") == model
        and r["tags"].get("n_train") == n
        and r["tags"].get("test_set") == TEST_SET
        and key in r["metrics"]
    ]
    if len(out) != 5:
        raise SystemExit(f"{model} n={n} {channel}: {len(out)} seeds")
    return out


def main() -> int:
    require_clean_tree()
    records = {arm: load(name) for arm, name in FILES.items()}
    config = {
        arm: {
            key: sorted({json.dumps(r["params"].get(key)) for r in recs})
            for key in (
                "process/fingerprint",
                "data/future_actions",
                "data/solid_fraction_process",
                "data/gas_velocity_cap",
                "data/new_study_version",
                "data/constant_run_fraction",
            )
        }
        for arm, recs in records.items()
    }
    table: dict[str, Any] = {}
    ranges: dict[str, Any] = {}
    for channel in CHANNELS:
        for model in MODELS:
            rows = {}
            for n in N_TRAIN:
                const = values(records["constant_trained"], model, n, channel)
                excit = values(records["excited_trained"], model, n, channel)
                m_c, m_e = statistics.median(const), statistics.median(excit)
                span = max(max(const) - min(const), max(excit) - min(excit))
                rows[n] = {
                    "median_constant_trained": m_c,
                    "median_excited_trained": m_e,
                    "ratio_constant_over_excited": m_c / m_e,
                    "difference_of_medians": m_c - m_e,
                    "seed_range_constant_trained": max(const) - min(const),
                    "seed_range_excited_trained": max(excit) - min(excit),
                    "difference_exceeds_larger_seed_range": (m_c - m_e) > span,
                }
            ratios = [r["ratio_constant_over_excited"] for r in rows.values()]
            table[f"{channel}/{model}"] = rows
            ranges[f"{channel}/{model}"] = {"min": min(ratios), "max": max(ratios)}
    write_artifact(
        "befund55_addendum_ratio_range",
        {
            "_provenance": provenance(),
            "rule": "see the module docstring of scripts/analyze_55_addendum_ratio_range.py",
            "sources": [f"{name}.jsonl" for name in FILES.values()],
            "test_set": TEST_SET,
            "metric": "nrmse_run_median, h36, median over five seeds",
            "data_configuration": config,
            "ratio_ranges": ranges,
            "table": table,
        },
    )
    for key, row in ranges.items():
        print(key, f"{row['min']:.3f}-{row['max']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

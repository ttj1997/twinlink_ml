r"""Hybrid results with and without the runs that started from the nominal state: TRAINING side only (POST HOC).

NOT pre-registered. The pre-registered analysis (``a1_main_analysis.json``, ``fallback_analysis``) called a seed
"clean" if no run with a nominal start lay in the training prefix of that N AND none among the 32 test runs. Every
seed has 2-3 such runs among its test runs, so there was no clean seed and no comparable cell. This script
therefore drops the test condition and looks at the training side only: for every N the seeds without a
nominal-start run in the training prefix. The test-side contamination (2-3 of 32 test runs per seed, 1 of 16 in the
constant-run holdout) is NOT removed and cannot be removed from the records (the metric is the median over the test
runs of a record; per-run values and the trained models were not kept). Everything below is descriptive.

Written before it is run (committed before it is executed; clean tree). Reads ``a1_main_records.json`` and the
fall-back runs in ``a1_main_analysis.json``; no training, no new simulation.

For every plant, small N (2, 4, 8, 16), channel and reference method (from scratch, the simulator alone,
pretraining + fine-tuning): the paired verdict of the hybrid (16000 steps) on all five seeds and on the seeds
without a nominal-start run in the training prefix; cells with fewer than 3 such seeds are marked low-power and not
compared. Also HA3 recomputed on those seeds with the criterion of the design draft (the hybrid better than each of
the other three methods in at least 2 of 3 channels, at at least 3 of the 4 small N, on at least 2 of the 3
structural plants).

    tools/run_here.sh scripts/analyze_a1_fallback_training_side.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

import analyze_a1_main as base
from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact

OUTPUT_NAME = "a1_fallback_training_side"
REFERENCES = (base.SCRATCH, base.SIMULATOR, base.FINETUNE)


def main() -> None:
    require_clean_tree()
    analysis = json.loads((RUNS_DIR / "a1_main_analysis.json").read_text(encoding="utf-8"))
    failed = {
        int(seed): entry["failed_train_prefix_positions"]
        for seed, entry in analysis["fallback_analysis"]["fallback_runs_by_seed"].items()
    }
    values = base.load_values()
    seeds_by_n = {
        n: tuple(s for s in base.SEEDS if not any(p < n for p in failed[s])) for n in base.SMALL_N
    }
    cells: list[dict[str, Any]] = []
    for plant, n, channel, reference in itertools.product(
        base.PLANTS, base.SMALL_N, base.CHANNELS, REFERENCES
    ):
        everything = base.compare(values, plant, base.HYBRID, reference, n, "excited", channel)
        row: dict[str, Any] = {
            "plant": plant,
            "n": n,
            "channel": channel,
            "against": reference,
            "verdict_all_seeds": everything["verdict"],
            "mean_difference_all_seeds": everything["mean_difference"],
            "seeds_without_nominal_start_in_training": list(seeds_by_n[n]),
            "n_seeds": len(seeds_by_n[n]),
            "low_power": len(seeds_by_n[n]) < 3,
        }
        if len(seeds_by_n[n]) >= 3:
            clean = base.compare(
                values,
                plant,
                base.HYBRID,
                reference,
                n,
                "excited",
                channel,
                seeds=seeds_by_n[n],
            )
            row["verdict_subset"] = clean["verdict"]
            row["mean_difference_subset"] = clean["mean_difference"]
        else:
            row["verdict_subset"] = None
        cells.append(row)

    comparable = [c for c in cells if c["verdict_subset"] is not None]
    changed = [c for c in comparable if c["verdict_subset"] != c["verdict_all_seeds"]]
    flips = [
        c for c in changed if {c["verdict_subset"], c["verdict_all_seeds"]} == {"better", "worse"}
    ]

    ha3_plants: dict[str, Any] = {}
    for plant in base.STRUCTURAL:
        by_n = {}
        for n in base.SMALL_N:
            if len(seeds_by_n[n]) < 3:
                by_n[str(n)] = {"ok": None, "low_power": True}
                continue
            better = {
                ref: sum(
                    1
                    for channel in base.CHANNELS
                    if base.compare(
                        values, plant, base.HYBRID, ref, n, "excited", channel, seeds=seeds_by_n[n]
                    )["verdict"]
                    == "better"
                )
                for ref in REFERENCES
            }
            by_n[str(n)] = {
                "channels_better": better,
                "ok": all(v >= 2 for v in better.values()),
                "low_power": False,
            }
        ok_count = sum(1 for v in by_n.values() if v["ok"])
        ha3_plants[plant] = {"by_n": by_n, "n_small_n_ok": ok_count, "plant_ok": ok_count >= 3}
    ha3_supported = sum(1 for v in ha3_plants.values() if v["plant_ok"]) >= 2

    write_artifact(
        OUTPUT_NAME,
        {
            "_provenance": provenance(),
            "design": {
                "status": "POST HOC, descriptive; the pre-registered 'clean' set was empty",
                "source_records": "a1_main_records.json",
                "source_fallback_runs": "a1_main_analysis.json (fallback_analysis)",
                "seeds_without_nominal_start_in_training_prefix_by_n": {
                    str(n): list(s) for n, s in seeds_by_n.items()
                },
                "test_side_contamination": (
                    "2-3 of the 32 excited test runs per seed and 1 of the 16 constant-run test runs started "
                    "from the nominal state; not removed"
                ),
                "n_paired_tests": base.Counter.n,
            },
            "n_cells_compared": len(comparable),
            "n_cells_low_power_not_compared": len(cells) - len(comparable),
            "cells_with_changed_verdict": changed,
            "cells_with_direction_flip": flips,
            "ha3_on_subset": {"plants": ha3_plants, "supported": ha3_supported},
            "ha3_on_all_seeds_for_comparison": analysis["ha3"]["supported"],
            "cells": cells,
        },
    )
    print(
        "compared:",
        len(comparable),
        "| changed:",
        len(changed),
        "| flips:",
        len(flips),
        "| HA3 on subset:",
        ha3_supported,
    )


if __name__ == "__main__":
    main()

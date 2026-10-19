r"""Addendum to Befund 98: effect size and sign consistency next to every Holm verdict of the rank cells.

Pure re-analysis of the committed artifact ``rerun_evaluation_rerun_full.json`` (rank cells of Befund 94,
paired t over five seeds, ``rmse_run_median_physical``); no training, no new data. Requested by review 4 of
the synthesis (``docs/reviews/2026-10-19/review4_synthesis_v5.md``, finding 4): "uncorrected tendency"
understates effects that are large and consistent but not resolved after Holm. Rules fixed here, before
the run:

* Cells and families as Befund 98 (main family = all rank cells of one arm; Holm 5 %; a cell is resolved
  after Holm if its uncorrected verdict is resolved and its adjusted p value is below 0.05). The script
  ASSERTS that its Holm counts per candidate equal ``holm_rank_cells_rerun_full.json``.
* EFFECT SIZE per cell: ratio of the medians over the seeds, candidate / LSTM (> 1 = candidate worse).
* SIGN CONSISTENCY per cell: number of seeds in which the candidate is worse (``seeds_candidate_worse``);
  "all seeds worse" = 5 of 5, "all seeds better" = 0 of 5.
* CLASSES, used for the wording of the synthesis (fixed here):
    - "resolved after Holm";
    - "large and consistent, not resolved after Holm": not resolved after Holm, ratio of the medians at
      least 1.5 (or at most 1 / 1.5) and all five seeds on the side of the median ratio;
    - "small or inconsistent": every other cell.
  Reported per arm, candidate and channel (and over all channels) with the range and median of the ratio.
* xLSTM ON ``energy_target``, h = 36, every ``n_train`` and both test sets: ratio, seeds worse, p
  value, Holm-adjusted p value. Against the STATIC REGRESSION (same cells; the per-seed values of the
  xLSTM and of the static regression from their rank cells, which share the LSTM reference -- asserted):
  ratio of the medians xLSTM / static, seeds in which the xLSTM is worse, paired t test (uncorrected, not
  part of any family; reported as a single descriptive test).

Not claimed: a different family or test (the log-ratio test stored next to every cell is not used here;
whether a smaller family or that test becomes a pre-registered sensitivity is a decision of the project
lead). Five seeds, four degrees of freedom; cells of one channel are not independent.

Call::

    tools/run_here.sh scripts/analyze_98_addendum_effect_sizes.py --prefix rerun_full
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact
from analyze_94_holm_rank_cells import adjusted
from analyze_paired_reevaluation import compare

ALPHA = 0.05
LARGE = 1.5
ARMS = ("A0", "A1", "N")
BETTER, WORSE = "candidate better", "candidate worse"
CHANNELS = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
SEEDS = (1, 2, 3, 4, 5)


def classify(cell: dict[str, Any], holm_p: float) -> str:
    resolved = cell["verdict"] in (BETTER, WORSE) and holm_p < ALPHA
    if resolved:
        return "resolved_holm"
    ratio = cell["median_candidate"] / cell["median_reference"]
    worse = cell["seeds_candidate_worse"]
    if ratio >= LARGE and worse == len(SEEDS):
        return "large_consistent_unresolved"
    if ratio <= 1.0 / LARGE and worse == 0:
        return "large_consistent_unresolved"
    return "small_or_inconsistent"


def summarise(cells: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [c["ratio_of_medians"] for c in cells]
    classes = [c["class"] for c in cells]
    return {
        "cells": len(cells),
        "worse_uncorrected": sum(c["verdict"] == WORSE for c in cells),
        "better_uncorrected": sum(c["verdict"] == BETTER for c in cells),
        "worse_holm": sum(c["verdict"] == WORSE and c["p_holm"] < ALPHA for c in cells),
        "better_holm": sum(c["verdict"] == BETTER and c["p_holm"] < ALPHA for c in cells),
        "all_seeds_worse": sum(c["seeds_candidate_worse"] == len(SEEDS) for c in cells),
        "all_seeds_better": sum(c["seeds_candidate_worse"] == 0 for c in cells),
        "ratio_min": min(ratios),
        "ratio_median": statistics.median(ratios),
        "ratio_max": max(ratios),
        "resolved_holm": classes.count("resolved_holm"),
        "large_consistent_unresolved": classes.count("large_consistent_unresolved"),
        "small_or_inconsistent": classes.count("small_or_inconsistent"),
    }


def analyse_arm(cells: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    holm_p = adjusted(cells)
    rows = []
    for cell, adj in zip(cells, holm_p, strict=True):
        rows.append(
            {
                "candidate": cell["candidate"],
                "channel": cell["channel"],
                "test_set": cell["test_set"],
                "n_train": cell["n_train"],
                "horizon": cell["horizon"],
                "verdict": cell["verdict"],
                "p_value": cell["p_value"],
                "p_holm": float(adj),
                "median_candidate": cell["median_candidate"],
                "median_reference": cell["median_reference"],
                "ratio_of_medians": cell["median_candidate"] / cell["median_reference"],
                "seeds_candidate_worse": cell["seeds_candidate_worse"],
                "class": classify(cell, float(adj)),
            }
        )
    summary: dict[str, Any] = {}
    for candidate in sorted({r["candidate"] for r in rows}):
        per = {}
        for channel in (*CHANNELS, "all"):
            sub = [
                r
                for r in rows
                if r["candidate"] == candidate and (channel == "all" or r["channel"] == channel)
            ]
            per[channel] = summarise(sub)
        summary[candidate] = per
    return rows, summary


def check_against_holm_artifact(arm: str, summary: dict[str, Any], holm: dict[str, Any]) -> None:
    for candidate, per in summary.items():
        for channel, row in per.items():
            ref = holm[arm]["by_candidate"][candidate][channel]
            for key in (
                "cells",
                "worse_uncorrected",
                "better_uncorrected",
                "worse_holm",
                "better_holm",
            ):
                if row[key] != ref[key]:
                    raise SystemExit(f"{arm} {candidate} {channel} {key}: {row[key]} != {ref[key]}")


def xlstm_energy(cells: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if not (
            row["candidate"] == "xlstm"
            and row["channel"] == "energy_target"
            and row["horizon"] == 36
        ):
            continue
        key = (row["channel"], row["test_set"], row["n_train"], row["horizon"])

        def find(candidate: str, key: tuple[Any, ...] = key) -> dict[str, Any]:
            return next(
                c
                for c in cells
                if c["candidate"] == candidate
                and (c["channel"], c["test_set"], c["n_train"], c["horizon"]) == key
            )

        xl, st = find("xlstm"), find("static")
        if xl["per_seed_reference"] != st["per_seed_reference"]:
            raise SystemExit(f"LSTM reference differs between the xLSTM and static cells at {key}")
        versus = compare(
            dict(zip(SEEDS, xl["per_seed_candidate"], strict=True)),
            dict(zip(SEEDS, st["per_seed_candidate"], strict=True)),
            "xlstm",
            "static",
        )
        out.append(
            {
                **{
                    k: row[k]
                    for k in ("test_set", "n_train", "horizon", "verdict", "p_value", "p_holm")
                },
                "ratio_xlstm_over_lstm": row["ratio_of_medians"],
                "seeds_xlstm_worse_than_lstm": row["seeds_candidate_worse"],
                "class": row["class"],
                "vs_static": {
                    "ratio_xlstm_over_static": versus["median_candidate"]
                    / versus["median_reference"],
                    "seeds_xlstm_worse": versus["seeds_candidate_worse"],
                    "verdict_uncorrected": versus["verdict"],
                    "p_value": versus["p_value"],
                },
            }
        )
    return sorted(out, key=lambda r: (r["test_set"], r["n_train"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    arguments = parser.parse_args()
    require_clean_tree()
    source = RUNS_DIR / f"rerun_evaluation_{arguments.prefix}.json"
    evaluation = json.loads(source.read_text(encoding="utf-8"))
    holm = json.loads((RUNS_DIR / f"holm_rank_cells_{arguments.prefix}.json").read_text("utf-8"))
    result: dict[str, Any] = {}
    for arm in ARMS:
        cells = evaluation["rank_cells"][arm]
        rows, summary = analyse_arm(cells)
        check_against_holm_artifact(arm, summary, holm)
        result[arm] = {"summary": summary, "cells": rows}
        if arm == "N":
            result[arm]["xlstm_energy_h36"] = xlstm_energy(cells, rows)
    write_artifact(
        f"befund98_addendum_effect_sizes_{arguments.prefix}",
        {
            "_provenance": provenance(),
            "rule": "see the module docstring of scripts/analyze_98_addendum_effect_sizes.py",
            "source": source.name,
            "large_ratio_threshold": LARGE,
            **result,
        },
    )
    for candidate, per in result["N"]["summary"].items():
        print("N", candidate, per["all"])
    for row in result["N"]["xlstm_energy_h36"]:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

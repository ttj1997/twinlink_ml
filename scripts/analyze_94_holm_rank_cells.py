r"""Holm counts of the rank cells of Befund 94, by arm, channel and candidate (addendum to Befund 94).

Pure re-analysis of the committed artifact ``rerun_evaluation_rerun_full.json`` (written by
``scripts/analyze_rerun.py``); no training, no new data.  Rule fixed here, before the run:

* Cells: the ``rank_cells`` of the artifact (candidate against LSTM; cells over ``n_train``,
  test set, channel, horizon), arms A0, A1, N.  Verdict per cell as in the artifact (paired t
  over five seeds, uncorrected: "candidate better" / "candidate worse" / "unresolved").
* MAIN FAMILY (as in Befund 94): all rank cells of ONE arm, all candidates and channels
  together (A0 336, A1 168, N 336).  A cell counts as resolved after Holm if its uncorrected
  verdict is resolved and its Holm-adjusted p value is below 0.05 (the rule of
  ``analyze_rerun.correction``, so the totals reproduce ``multiplicity['rank_<arm>']``).
* SENSITIVITY FAMILY (added after the main result was seen; marked as such): only the
  GRU-against-LSTM cells of the arm (A0 and N 32, A1 16 cells).
* Output: per arm and channel, for every candidate, the number of cells better / worse /
  unresolved uncorrected and the number resolved better / worse after Holm (main family;
  for GRU against LSTM also the sensitivity family), plus GRU against LSTM split by horizon.

Call::

    tools/run_here.sh scripts/analyze_94_holm_rank_cells.py --prefix rerun_full
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact
from analyze_83_addendum import holm

ALPHA = 0.05
ARMS = ("A0", "A1", "N")
BETTER, WORSE, UNRESOLVED = "candidate better", "candidate worse", "unresolved"


def adjusted(cells: list[dict[str, Any]]) -> np.ndarray:
    p = np.array([1.0 if math.isnan(c["p_value"]) else c["p_value"] for c in cells])
    return holm(p)


def tally(cells: list[dict[str, Any]], holm_p: np.ndarray) -> dict[str, int]:
    count: Counter[str] = Counter()
    for cell, adj in zip(cells, holm_p, strict=True):
        count["cells"] += 1
        key = {BETTER: "better", WORSE: "worse"}.get(cell["verdict"], "unresolved")
        count[f"{key}_uncorrected"] += 1
        if key != "unresolved" and adj < ALPHA:
            count[f"{key}_holm"] += 1
    return {
        name: count[name]
        for name in (
            "cells",
            "better_uncorrected",
            "worse_uncorrected",
            "unresolved_uncorrected",
            "better_holm",
            "worse_holm",
        )
    }


def analyse(rank_cells: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        cells = rank_cells[arm]
        main_p = adjusted(cells)
        gru_index = [i for i, c in enumerate(cells) if c["candidate"] == "gru"]
        gru_cells = [cells[i] for i in gru_index]
        gru_sens_p = adjusted(gru_cells)
        channels = sorted({c["channel"] for c in cells})
        candidates = sorted({c["candidate"] for c in cells})
        entry: dict[str, Any] = {
            "family_main": f"all {len(cells)} rank cells of arm {arm} (as Befund 94)",
            "family_sensitivity": (
                f"only the {len(gru_cells)} GRU-against-LSTM cells of arm {arm} "
                "(post hoc, added after the main result)"
            ),
            "by_candidate": {},
            "gru_by_channel_and_horizon": {},
        }
        for candidate in candidates:
            index = [i for i, c in enumerate(cells) if c["candidate"] == candidate]
            per_channel = {}
            for channel in (*channels, "all"):
                sub = [i for i in index if channel == "all" or cells[i]["channel"] == channel]
                per_channel[channel] = tally([cells[i] for i in sub], main_p[sub])
            entry["by_candidate"][candidate] = per_channel
        gru_channel = {}
        for channel in (*channels, "all"):
            sub = [
                k for k, c in enumerate(gru_cells) if channel == "all" or c["channel"] == channel
            ]
            main_sub = [gru_index[k] for k in sub]
            gru_channel[channel] = {
                "main_family": tally([cells[i] for i in main_sub], main_p[main_sub]),
                "sensitivity_family": tally([gru_cells[k] for k in sub], gru_sens_p[sub]),
            }
        entry["gru_vs_lstm"] = gru_channel
        horizon: Counter[str] = Counter()
        for cell in gru_cells:
            horizon[f"{cell['channel']}/h{cell['horizon']}/{cell['verdict']}"] += 1
        entry["gru_by_channel_and_horizon"] = dict(sorted(horizon.items()))
        result[arm] = entry
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    arguments = parser.parse_args()
    require_clean_tree()
    source = RUNS_DIR / f"rerun_evaluation_{arguments.prefix}.json"
    evaluation = json.loads(source.read_text(encoding="utf-8"))
    result = analyse(evaluation["rank_cells"])
    for arm in ARMS:
        recorded = evaluation["multiplicity"][f"rank_{arm}"]
        total = result[arm]["by_candidate"]
        holm_total = sum(c["all"]["better_holm"] + c["all"]["worse_holm"] for c in total.values())
        if holm_total != recorded["resolved_holm"]:
            raise SystemExit(f"{arm}: Holm total {holm_total} != artifact {recorded}")
    write_artifact(
        f"holm_rank_cells_{arguments.prefix}",
        {"_provenance": provenance(), "source": source.name, **result},
    )
    for arm in ARMS:
        for channel, row in result[arm]["gru_vs_lstm"].items():
            print(arm, channel, row["main_family"], row["sensitivity_family"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

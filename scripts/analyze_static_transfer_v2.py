"""Analysis of the POST-HOC sensitivity v2 (stronger fine-tuning of L3), reading fixed BEFORE its run.

Declared post hoc (decision of the project lead, 2026-11-06). The main reading of the study stays
R3 (v1, ``scripts/analyze_static_transfer.py``); this sensitivity only decides between R2 and R3.
This docstring and the code were committed before pilot 2 and the rerun.

Data: the v1 artifact (``static_transfer_v1.json.gz``) for every method except L3, and the v2 rerun
(``static_transfer_v2.json.gz``) for L3 with the setting selected by pilot 2 on the full-strength
pilot plant PP-full. Same seeds, designs, noise draws, test sets and pretrained nets, so the values
are paired seed by seed (the rerun recomputes the other methods and checks that they equal v1).
Condition ``clean_low`` only.

Definitions: unchanged from v1 -- RMSE in physical units against the noise-free plant truth; paired
t-test over the 5 seeds; Holm per family; "resolved better" = Holm-rejected AND the same sign in at
least 4 of 5 seeds; "L3 beats X in a cell (plant, N)" = resolved better on every affected target.

Reading of the sensitivity (family per test set: L3-v2 against L0, L1, L2, L4, B1 on the five
deviating plants, affected targets, every N = 270 tests, as v1):

* primary: the interpolation set. With the v1 order R1 > R2 > R3 > none:
  - **R3** (no cell with L3-v2 beating L0, L1 and L4 jointly): R3 of v1 is robust against a stronger
    fine-tuning.
  - **R2** (some cell with L3-v2 beating L0, L1 and L4, none beating all five): with a stronger
    fine-tuning the reading would be R2; R3 of v1 depends on the fine-tuning setting.
  - **R1** (some cell beating all five) is reported as such: R3 of v1 depends on the setting, and the
    stronger fine-tuning goes beyond R2.
  - **none**: reported as such (the stronger fine-tuning is weaker than v1's).
* the plant-like set: the same reading, reported beside it.

Descriptive (own Holm families, no reading): L3-v2 against L3-v1 and L3-v2 against L2 per reported
test set (54 tests each); learning curves of L3-v2.

Output: ``artifacts/runs/static_transfer_v2_analysis.json`` (never overwritten).
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))
sys.path.insert(0, str(SCRIPTS))

import analyze_static_transfer as v1analysis  # noqa: E402
from _common import RUNS_DIR, _json_default, provenance, require_clean_tree  # noqa: E402

from twinlink_ml.static_transfer.config import TARGET_NAMES  # noqa: E402

CONDITION = "clean_low"


def merge(v1: dict[str, Any], v2: dict[str, Any]) -> list[dict[str, Any]]:
    """v1 units of ``clean_low`` with L3 of v1 renamed ``L3v1`` and L3 of v2 as ``L3``."""
    v2_units = {(u["plant"], int(u["seed"])): u for u in v2["units"]}
    merged = []
    for unit in v1["units"]:
        if unit["condition"] != CONDITION:
            continue
        other = v2_units[(unit["plant"], int(unit["seed"]))]
        rec = copy.deepcopy(unit)
        for n, methods in rec["fitted"].items():
            methods["L3v1"] = methods.pop("L3")
            methods["L3"] = other["fitted"][n]["L3"]
        merged.append(rec)
    return merged


def verdict_text(reading: str) -> str:
    return {
        "R3": "R3 of v1 is robust against the stronger fine-tuning",
        "R2": "with the stronger fine-tuning the reading would be R2; R3 of v1 depends on the setting",
        "R1": "R1: R3 of v1 depends on the setting; the stronger fine-tuning goes beyond R2",
        "none": "none: the stronger fine-tuning is weaker than v1's",
    }[reading]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--v1", default="static_transfer_v1")
    ap.add_argument("--v2", default="static_transfer_v2")
    args = ap.parse_args()
    out_path = RUNS_DIR / f"{args.v2}_analysis.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists -- artifacts are never overwritten")
    require_clean_tree()
    with gzip.open(RUNS_DIR / f"{args.v1}.json.gz", "rt", encoding="utf-8") as fh:
        v1 = json.load(fh)
    with gzip.open(RUNS_DIR / f"{args.v2}.json.gz", "rt", encoding="utf-8") as fh:
        v2 = json.load(fh)
    if v2["failures"]:
        raise SystemExit(f"{len(v2['failures'])} failed units in the rerun")
    if v2["check_max_abs_difference"] != 0.0:
        print(f"NOTE: other methods differ from v1 by up to {v2['check_max_abs_difference']}")
    run = {"units": merge(v1, v2), "config": v1["config"], "conditions": [CONDITION]}
    study = v1analysis.Study(run)
    sets = v2["reported_test_sets"]
    readings = {ts: study.reading(ts, CONDITION) for ts in sets}
    plants = ("P-loss", "P-air", "P-eta", "P-moist", "P-comb")
    descriptive = {}
    for ts in sets:
        for ref in ("L3v1", "L2"):
            specs = [
                ("L3", ref, p, CONDITION, n, ts, t)
                for p in plants
                for n in study.budgets
                for t in study.affected[p]
            ]
            descriptive[f"L3v2_vs_{ref}/{ts}"] = study.family(
                f"L3v2_vs_{ref}/{ts}", specs
            ).as_dict()
    curves = {
        f"{p}|{ts}|{m}|{t}": {
            str(n): study.table.summary(p, CONDITION, m, n, ts, t) for n in study.budgets
        }
        for p in ("P0", *plants)
        for ts in sets
        for m in ("L3", "L3v1", "L2", "L1")
        for t in TARGET_NAMES
    }
    primary = readings["interpolation"]["reading"]
    result = {
        "provenance": provenance(),
        "post_hoc": True,
        "v1_run": args.v1,
        "v2_run": args.v2,
        "v2_run_commit": v2["provenance"]["git/commit"],
        "l3_setting_v2": v2["l3_setting"],
        "l3_setting_v1": v1["config"]["selected"]["L3"],
        "check_against_v1_max_abs_difference": v2["check_max_abs_difference"],
        "reading_v2": dict(readings),
        "primary_reading_v2": primary,
        "verdict": verdict_text(primary),
        "descriptive": descriptive,
        "learning_curves": curves,
    }
    out_path.write_text(json.dumps(result, indent=1, default=_json_default), encoding="utf-8")
    print(f"wrote {out_path}")
    print(
        json.dumps(
            {
                ts: {
                    "reading": r["reading"],
                    "R4_flag": r["R4_flag"],
                    "cells_L3_beats_L0_L1_L4": r["cells_L3_beats_L0_L1_L4"],
                    "cells_L3_beats_L2": r["cells_L3_beats_L2"],
                    "cells_L3_beats_L4": r["cells_L3_beats_L4"],
                    "counts": r["family"]["counts"],
                }
                for ts, r in readings.items()
            },
            indent=1,
        )
    )
    print(result["verdict"])


if __name__ == "__main__":
    main()

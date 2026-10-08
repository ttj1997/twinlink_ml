r"""Effect of the shortlist cap of five allocations (``m3x_robust_rest_umin126``): per affected scenario how many
allocations were cut and how far the best cut one lay from the best kept one (table value and polished value of
the joint stage of Befund 73, kg CO2e/h; positive = the cut one is worse). Read-only on artifacts.

    python scripts/analyze_m3x_cap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

import run_m3x_confirm as c
from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact

SUFFIX = c.SUFFIX


def main() -> int:
    require_clean_tree()
    art = json.loads((RUNS_DIR / f"m3x_robust_rest_{SUFFIX}.json").read_text("utf-8"))
    control = {
        r["grid_index"]: r
        for r in json.loads((RUNS_DIR / f"m3_joint_control_{SUFFIX}.json").read_text("utf-8"))[
            "scenarios"
        ]
    }
    c.SCENARIOS = tuple(art["scenarios"])
    c.SHORTLIST_CAP = None
    full = c.shortlists()
    out = {}
    for s in art["scenarios"]:
        kept = {c._allocation_key(a) for a in art["shortlists"][str(s)]}
        polished = {
            c._allocation_key(e["allocation"]): e
            for e in control[s]["search"]["polished"].values()
            if e.get("polished_value") is not None
        }
        cut = [a for a in full[s] if c._allocation_key(a) not in kept]
        row = {"n_uncapped": len(full[s]), "n_kept": len(kept), "n_cut": len(cut)}
        if cut:
            kept_e = [polished[k] for k in kept if k in polished]
            cut_e = [
                polished[c._allocation_key(a)] for a in cut if c._allocation_key(a) in polished
            ]
            row["best_cut_minus_best_kept_polished"] = min(
                e["polished_value"] for e in cut_e
            ) - min(e["polished_value"] for e in kept_e)
            row["best_cut_minus_best_kept_table"] = min(e["table_value"] for e in cut_e) - min(
                e["table_value"] for e in kept_e
            )
        out[str(s)] = row
        print(s, row)
    write_artifact(f"m3x_cap_effect_{SUFFIX}", {"_provenance": provenance(), "scenarios": out})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

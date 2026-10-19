r"""Heat-pump study: solid fraction 0.634 (study 4) against the Befund-99 assumption 0.3.

Pure aggregation of two artifact sets (nothing is recomputed):

* Befund-99 assumption: ``gate_heat_pump``, ``gate_heat_pump_tables``,
  ``gate_heat_pump_rule_tuning_posthoc`` (solid fraction 0.3, band 0.2-0.5);
* study 4: the same three names with a suffix (default ``_sf0634``; solid fraction 0.634, band
  0.60-0.70, PLACEHOLDER from branch ``plant-a-operating-point``, ``docs/23`` section 5: the energy
  balance of the two measured days of plant A needs at least 0.575 / 0.625).

Reported side by side with the difference (study 4 minus Befund 99): Stage-A references (annual
saving per central criterion at 250 / 500 / 1000 kW, design, COP, pinch, exhaust share capped at
saturation), the grid table per (site, exhaust-C basis, day) and per quality, the Stage-B gate
statistics, and the post-hoc rule-tuning summary.

Call::

    tools/run_here.sh scripts/analyze_heat_pump_solid_fraction_comparison.py \
        --name gate_heat_pump_solid_fraction_comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
from _common import (  # type: ignore[import-untyped]
    NO_HEAT_NETWORK,
    REPO_ROOT,
    provenance,
    require_clean_tree,
    write_artifact,
)

RUNS = REPO_ROOT / "artifacts" / "runs"
VERDICT_KEYS = (
    "reading",
    "n_cells_realistic",
    "n_cells_gap_at_least_2_points",
    "n_runs_gap_at_least_2_points",
    "n_cells_gain_at_least_5_percent",
    "max_gap",
    "max_gain",
    "max_weekly_over_daily_foresight",
    "n_unrealistic",
)


def load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((RUNS / f"{name}.json").read_text(encoding="utf-8"))
    return data


def pair(a: Any, b: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"befund99": a, "study4": b}
    if isinstance(a, int | float) and isinstance(b, int | float) and not isinstance(a, bool):
        out["difference"] = b - a
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--suffix", default="_sf0634")
    args = parser.parse_args()
    require_clean_tree()
    names = ("gate_heat_pump", "gate_heat_pump_tables", "gate_heat_pump_rule_tuning_posthoc")
    old = {n: load(n) for n in names}
    new = {n: load(n + args.suffix) for n in names}

    refs: dict[str, Any] = {}
    to = old["gate_heat_pump_tables"]["tables"]
    tn = new["gate_heat_pump_tables"]["tables"]
    for rid, ro in to["references"].items():
        rn = tn["references"][rid]
        rows = []
        for a, b in zip(ro["rows"], rn["rows"], strict=True):
            assert a["size_kw"] == b["size_kw"] and a["criterion"] == b["criterion"]
            rows.append(
                {
                    "size_kw": a["size_kw"],
                    "criterion": a["criterion"],
                    "pair": pair(a["pair"], b["pair"]),
                    "cop": pair(a["cop"], b["cop"]),
                    "annual_saving": pair(a["annual_saving"], b["annual_saving"]),
                    "gas_heat_replaced_kwh_per_day": pair(
                        a["gas_heat_replaced_kwh_per_day"], b["gas_heat_replaced_kwh_per_day"]
                    ),
                }
            )
        ref_old = old["gate_heat_pump"]["stage_a"]["references"][rid]["central"]
        ref_new = new["gate_heat_pump"]["stage_a"]["references"][rid]["central"]
        refs[rid] = {
            "label": ro["label"],
            "direct_kwh_per_day": pair(ro["direct_kwh_per_day"], rn["direct_kwh_per_day"]),
            "cold_utility_kwh_per_day": pair(
                ro["pinch"]["cold_utility_kwh_per_day"], rn["pinch"]["cold_utility_kwh_per_day"]
            ),
            "pinch_high_mean_shifted": pair(
                ro["pinch"]["pinch_high_mean_shifted"], rn["pinch"]["pinch_high_mean_shifted"]
            ),
            "exhaust_capped_fraction": pair(
                ref_old["notes"].get("exhaust_capped_fraction"),
                ref_new["notes"].get("exhaust_capped_fraction"),
            ),
            "exhaust_a_mean_kw": pair(
                ref_old["notes"].get("exhaust_a_mean_kw"), ref_new["notes"].get("exhaust_a_mean_kw")
            ),
            "rows": rows,
        }
    grid = {
        g: {
            c: {k: pair(to["grid"][g][c][k], tn["grid"][g][c][k]) for k in ("median", "min", "max")}
            for c in to["grid"][g]
        }
        for g in to["grid"]
    }
    quality = {
        q: {
            c: {
                k: pair(to["quality"][q][c][k], tn["quality"][q][c][k])
                for k in ("median", "min", "max")
            }
            for c in to["quality"][q]
        }
        for q in to["quality"]
    }
    vo = old["gate_heat_pump"]["stage_b"]["gate"]["verdict"]
    vn = new["gate_heat_pump"]["stage_b"]["gate"]["verdict"]
    gate = {k: pair(vo.get(k), vn.get(k)) for k in VERDICT_KEYS}
    gate["max_gap_meta"] = pair(vo.get("max_gap_meta"), vn.get("max_gap_meta"))
    per_group = {
        k: {
            m: pair(v[m], new["gate_heat_pump"]["stage_b"]["gate"]["per_group"][k][m])
            for m in ("reading", "n_cells_gap_at_least_2_points", "max_gap")
        }
        for k, v in old["gate_heat_pump"]["stage_b"]["gate"]["per_group"].items()
        if k in new["gate_heat_pump"]["stage_b"]["gate"]["per_group"]
    }
    so = old["gate_heat_pump_rule_tuning_posthoc"]["summary"]
    sn = new["gate_heat_pump_rule_tuning_posthoc"]["summary"]
    posthoc = {k: pair(so.get(k), sn.get(k)) for k in so if not isinstance(so[k], dict)}
    sens = {
        rid: {
            crit: [
                {"variant": r["variant"], "delta_annual_saving": r["delta_annual_saving"]}
                for r in lst
                if r["variant"].startswith("solid_fraction")
            ]
            for crit, lst in new["gate_heat_pump"]["stage_a"]["sensitivity"][rid]["ranking"].items()
        }
        for rid in new["gate_heat_pump"]["stage_a"]["sensitivity"]
    }
    write_artifact(
        args.name,
        {
            "_provenance": provenance(NO_HEAT_NETWORK),
            "name": args.name,
            "sources": {
                "befund99": {n: old[n]["provenance"]["git/commit"] for n in names},
                "study4": {n + args.suffix: new[n]["provenance"]["git/commit"] for n in names},
            },
            "solid_fraction_setting_study4": new["gate_heat_pump"].get("solid_fraction_setting"),
            "references": refs,
            "grid": grid,
            "quality": quality,
            "gate": gate,
            "gate_per_group": per_group,
            "posthoc_rule_tuning": posthoc,
            "study4_solid_fraction_sensitivity": sens,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

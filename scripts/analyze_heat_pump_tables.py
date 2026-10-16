r"""Heat-pump study: report tables aggregated from ``artifacts/runs/gate_heat_pump.json``.

Pure aggregation of the committed artifact (nothing is recomputed from plant data), so that every
number of ``docs/21_heat_pump.md`` stands in an artifact. Annual figures are day values x 365
(t CO2/a, kEUR/a); a scaling of the measured day, not an annual simulation.

Tables
    ``references``   RA1-RA4 at 250 / 500 / 1000 kW, quality 0.50: pair, COP, heat, gas heat
                     replaced, work, annual saving per central criterion, share of the site CO2
                     (11 500 t/a, partner), simple payback at 400 / 600 / 1000 EUR/kW.
    ``grid``         main grid A at 1000 kW, quality 0.50: median, min, max of the annual saving
                     per (site, exhaust-C basis, product day) and per central criterion.
    ``grid_by_mode`` the same per (site, mode).
    ``quality``      min / median / max over the main grid per quality and central criterion.
    ``across``       smallest share of heat-pump heat delivered in steps where the chosen pair
                     straddles the pinch, over the main grid (all qualities, sizes, criteria).
    ``breakeven``    analytic break-even COP per central criterion (pump heat replacing gas heat
                     1:1) and the break-even electricity CO2 factor for given COPs.

Call::

    tools/run_here.sh scripts/analyze_heat_pump_tables.py --name gate_heat_pump_tables
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
from _common import (  # type: ignore[import-untyped]
    NO_HEAT_NETWORK,
    REPO_ROOT,
    provenance,
    require_clean_tree,
    write_artifact,
)

RUNS = REPO_ROOT / "artifacts" / "runs"
CENTRAL = ("co2_today", "co2_2030", "cost_today", "cost_150", "cost_300")
SITE_CO2_T_PER_A = 11500.0
DAYS = 365.0


def annual(per_day: float) -> float:
    return per_day * DAYS / 1000.0


def stats(values: list[float]) -> dict[str, float]:
    a = np.asarray(values)
    return {
        "n": len(a),
        "median": float(np.median(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", default="gate_heat_pump", help="source artifact name")
    args = parser.parse_args()
    require_clean_tree()
    src = json.loads((RUNS / f"{args.source}.json").read_text(encoding="utf-8"))
    sa = src["stage_a"]

    refs: dict[str, Any] = {}
    for rid, ref in sa["references"].items():
        c = ref["central"]
        rows = []
        for size in (250.0, 500.0, 1000.0):
            i = c["sizes_kw"].index(size)
            for name in CENTRAL:
                j = c["criteria"].index(name)
                f = c["full"][name]
                saving = annual(c["objective_saving_per_day"][i][j])
                row = {
                    "size_kw": size,
                    "criterion": name,
                    "pair": f["pair"][i],
                    "cop": f["cop"][i],
                    "heat_kwh_per_day": f["heat_kwh_per_day"][i],
                    "gas_heat_replaced_kwh_per_day": f["utility_saving_kwh_per_day"][i],
                    "work_kwh_per_day": f["work_kwh_per_day"][i],
                    "annual_saving": saving,
                    "annual_direct": annual(c["objective_direct_per_day"][j]),
                }
                if name.startswith("co2"):
                    row["share_of_site_co2"] = saving / SITE_CO2_T_PER_A
                else:
                    row["simple_payback_a"] = {
                        f"{capex:g}": (size * capex / (saving * 1000.0) if saving > 1e-9 else None)
                        for capex in (400.0, 600.0, 1000.0)
                    }
                rows.append(row)
        refs[rid] = {
            "label": ref["label"],
            "direct_kwh_per_day": c["direct_kwh_per_day"],
            "pinch": c["pinch"],
            "rows": rows,
            "breakeven_sweeps": {
                "co2_factor_kg_per_kwh_at_500_1000_kw": ref["sweeps"]["co2_factor"]["breakeven"],
                "co2_factor_axis_end": ref["sweeps"]["co2_factor"]["axis"][-1],
                "value_ratio_at_500_1000_kw": ref["sweeps"]["value_ratio"]["breakeven"],
                "value_ratio_axis_end": ref["sweeps"]["value_ratio"]["axis"][-1],
            },
        }

    grid = sa["grid"]
    names = next(iter(grid.values()))["qualities"]["0.5"]["criteria"]
    sizes = next(iter(grid.values()))["qualities"]["0.5"]["sizes_kw"]
    i1000 = sizes.index(1000.0)
    by_group: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_mode: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    quality: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    across_min = 1.0
    for rec in grid.values():
        m = rec["meta"]
        g = f"{m['site']}|{m['exhaust_c_basis']}|{m['day']}"
        gm = f"{m['site']}|{m['mode']}"
        for qk, q in rec["qualities"].items():
            for name in CENTRAL:
                j = names.index(name)
                val = annual(q["objective_saving_per_day"][i1000][j])
                quality[qk][name].append(val)
                if qk == "0.5":
                    by_group[g][name].append(val)
                    by_mode[gm][name].append(val)
                f = q["full"][name]
                for i in range(len(sizes)):
                    if f["heat_kwh_per_day"][i] > 1e-6:
                        across_min = min(across_min, f["across_share"][i])
    tables = {
        "references": refs,
        "grid": {g: {n: stats(v) for n, v in d.items()} for g, d in sorted(by_group.items())},
        "grid_by_mode": {
            g: {n: stats(v) for n, v in d.items()} for g, d in sorted(by_mode.items())
        },
        "quality": {q: {n: stats(v) for n, v in d.items()} for q, d in sorted(quality.items())},
        "across_share_min_over_grid": across_min,
        "breakeven": {
            "analytic_cop": {n: src["criteria_values"][n]["breakeven_cop"] for n in CENTRAL},
            "co2_factor_for_cop": {
                f"{cop:g}": cop * src["criteria_values"]["co2_today"]["heat_value_per_kwh"]
                for cop in (2.0, 3.0, 4.0, 5.0)
            },
        },
    }
    write_artifact(
        args.name,
        {
            "_provenance": provenance(NO_HEAT_NETWORK),
            "name": args.name,
            "source_artifact": args.source,
            "solid_fraction_setting": src.get("solid_fraction_setting"),
            "source_provenance": src["provenance"],
            "tables": tables,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

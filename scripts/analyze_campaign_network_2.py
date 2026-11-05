"""Reading of the campaign-network follow-up study (pre-registered: committed before the run).

Input ``artifacts/runs/<name>.json.gz`` of ``scripts/run_campaign_network_2.py`` and, for the
control, ``artifacts/runs/campaign_network_v2.json.gz``; output ``artifacts/runs/<name>_analysis.json``
(never overwritten) and tables on stdout.

Question. Does a better simple rule or a rolling dispatch planning close the gap of the simple rules
to the bound with foresight K3 (Befund 118: RB1, K2b up to 6.30 % below K3)? Basis: the model, the
configuration and the schedule draws of Befund 118 unchanged (paired); main variant 10 K, 80 degC;
process water 0 / 200 kW / seasonal. New: K0b (fair baseline without network, fixed shares of the
source to A, B, C), K1d (dynamic cascade order per hour by the gas each stage could replace now,
storage logic of K2b), P1 (rolling LP, horizon 7 d, standstill plan known exactly), P2 (as P1, plan
shifted by U[-24, +24] h per change, naive correction: a change that happened is taken at its true
time, an overdue one is expected in the next hour -- P2 is therefore a LOWER bound for planning with
deviations). P1 / P2 make a new plan every 24 h and at once when the observed state differs from
the plan (decision of the project lead, 2026-11-04). K1 and K2b for comparison, K3 as bound.

Statistic. Gap ``g = (K3(S) - X(S)) / K3(S)`` of the net CO2 saving (gas CO2 minus pump electricity
CO2), median over the 5 draws per cell (level, source power P, a, mode, storage S). Cells of K1d:
S = 0 / 2 / 8 / 24 h (K1d with S = 0 is the dynamic cascade alone). Cells of P1 / P2: S > 0 (with
S = 0 both equal K3 without storage by structure). At a = 1 the one schedule counts for every draw
and mode, and P2 = P1.

Pre-registered reading:

* **Q0** -- the median gap of K1d is <= 1 % in every cell: a simple, well-chosen rule suffices; no
  planning needed.
* **Q1** -- not Q0, but the median gap of P2 is <= 1 % in every cell with storage: planning with the
  production plan is justified also with deviations from the plan; RL is not needed.
* **Q2** -- otherwise (P2 > 1 % below K3 somewhere): gap and cells are reported.

Reported alongside (declared before the run): the gaps of K1, K2b and P1, the number of cells above
1 % and the largest cells per configuration; P1 - P2 (cost of the plan deviations under the naive
correction); K1d against K2b and K1 (does the dynamic order help). Without a reading: the network
benefit against the fair baseline, ``V_N_fair = (use K1 - use K0b) / source energy`` per P (and
every configuration against K0b); citable as a network benefit is the value at 400 kW (decision of
the project lead on Befund 118, item 3).

Checks: K3 >= every configuration (K0, K0b, K1, K1d, K2b, P1, P2) in heat used and net CO2 in every
draw (tolerance 1e-3; K3 >= P1 >= P2 is NOT assumed); every LP of K3 optimal, no failed planning
LP; control against ``campaign_network_v2``: K0, K1, K2b and K3 of the same cells identical; the
check ``check:replan`` (P1 with a new plan every hour against every 24 h, one draw).

Seen before this reading was committed (declared): on one draw (process water 200 kW, a = 0.57,
independent, draw 0, 800 kW, 24 h) while building the model: heat used K3 6 294 632, P1 6 226 657
(daily planning; hourly 6 239 674, 0.21 % more), P2 6 185 224, K2b 5 914 909, K1d 5 894 629
(0 h: 5 777 713), K1 5 782 781, K0b 3 902 997 kWh/a; P1 with a horizon to the end of a four-week
test year reproduced K3 exactly. The thresholds come from the task.
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

from _common import RUNS_DIR, provenance, require_clean_tree  # noqa: E402

GAP_LIMIT = 0.01
ORDER_TOL = 1e-3
V2_NAME = "campaign_network_v2"
FIGS = ("use_kwh", "co2_net_kg")


def _stats(values: list[float]) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def _rows(data: dict[str, Any], level: str, a: float, mode: str, draws: int) -> list[dict]:
    if a >= 1.0:
        return [data["draws"][f"draw:{level}:1:independent:0"]] * draws
    return [data["draws"][f"draw:{level}:{a:g}:{mode}:{d}"] for d in range(draws)]


def _config(row: dict[str, Any], name: str, s: str, a: float) -> dict[str, Any] | None:
    if name in ("K0", "K0b", "K1"):
        return row[name]
    if name == "P2" and a >= 1.0:
        return row["P1"].get(s)
    return row[name].get(s)


def analyse(data: dict[str, Any]) -> dict[str, Any]:
    draws = int(data["budget"]["draws"])
    sizes = [f"{s:g}" for s in data["budget"]["storage_hours"]]
    names = ("K1", "K1d", "K2b", "P1", "P2")
    gaps: dict[str, dict[str, Any]] = {n: {} for n in names}
    fair: dict[str, Any] = {}
    violations: list[str] = []
    failures: list[str] = []
    for level, cfg in data["variant_configs"].items():
        powers = [f"{p:g}" for p in cfg["source"]["power_kw_levels"]]
        for mode in cfg["campaigns"]["modes"]:
            for p in powers:
                for a in (float(x) for x in cfg["campaigns"]["availability_levels"]):
                    rows = [r["powers"][p] for r in _rows(data, level, a, mode, draws)]
                    base = f"{level}|{mode}|{p}|{a:g}"
                    k0b = [r["K0b"] for r in rows]
                    entry = {"V_N_fair": _stats([r["V_N_fair"] for r in rows])}
                    for name in ("K1", "K1d", "K2b", "K3", "P1", "P2"):
                        for s in sizes:
                            figs = [_config(r, name, s, a) for r in rows]
                            if name == "K1" and s != "0":
                                continue
                            if any(f is None for f in figs):
                                continue
                            entry[f"{name}_{s}_minus_K0b"] = _stats(
                                [
                                    (f["use_kwh"] - b["use_kwh"]) / b["source_kwh"]
                                    for f, b in zip(figs, k0b, strict=True)
                                ]
                            )
                    fair[base] = entry
                    for s in sizes:
                        k3 = [r["K3"][s] for r in rows]
                        for name in names:
                            if name == "K1" and s != "0":
                                continue
                            figs = [_config(r, name, s, a) for r in rows]
                            if any(f is None for f in figs):
                                continue
                            g = [
                                (b["co2_net_kg"] - f["co2_net_kg"]) / b["co2_net_kg"]
                                for f, b in zip(figs, k3, strict=True)
                            ]
                            gu = [
                                (b["use_kwh"] - f["use_kwh"]) / b["use_kwh"]
                                for f, b in zip(figs, k3, strict=True)
                            ]
                            gaps[name][f"{base}|{s}"] = {"co2_net": _stats(g), "use": _stats(gu)}
                    for d, r in enumerate(rows):
                        for s in sizes:
                            k3 = r["K3"][s]
                            if k3.get("lp_status", 0) != 0:
                                failures.append(f"{base}|{d}|{s}: K3 status {k3['lp_status']}")
                            for name in ("K0", "K0b", "K1", "K1d", "K2b", "P1", "P2"):
                                f = _config(r, name, s, a)
                                if f is None:
                                    continue
                                if f.get("lp_failures", 0):
                                    failures.append(f"{base}|{d}|{s}: {name} LP failures")
                                for key in FIGS:
                                    if f[key] > k3[key] + ORDER_TOL:
                                        violations.append(f"{base}|{d}|{s}: {name} > K3 {key}")
    summary: dict[str, Any] = {}
    for name, cells in gaps.items():
        med = {k: v["co2_net"]["median"] for k, v in cells.items()}
        if not med:
            continue
        worst = max(med, key=med.get)
        above = sorted(((v, k) for k, v in med.items() if v > GAP_LIMIT), reverse=True)
        summary[name] = {
            "cells": len(med),
            "cells_above_limit": len(above),
            "largest": {"cell": worst, "co2_net_median": med[worst]},
            "median_of_cell_medians": float(np.median(list(med.values()))),
            "above_limit": [{"cell": k, "gap": v} for v, k in above],
        }
    q0 = summary["K1d"]["largest"]["co2_net_median"] <= GAP_LIMIT
    q1 = summary["P2"]["largest"]["co2_net_median"] <= GAP_LIMIT
    reading = "Q0" if q0 else "Q1" if q1 else "Q2"
    p1_p2 = [
        gaps["P2"][k]["co2_net"]["median"] - gaps["P1"][k]["co2_net"]["median"]
        for k in gaps["P2"]
        if k in gaps["P1"]
    ]
    return {
        "reading": reading,
        "gaps": gaps,
        "summary": summary,
        "p2_minus_p1_gap": _stats(p1_p2) if p1_p2 else None,
        "fair_baseline": fair,
        "checks": {"order_violations": violations, "failures": failures},
    }


def control_against_v2(data: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    worst = 0.0
    compared = 0
    for key, row in data["draws"].items():
        old = v2["draws"].get(key)
        if old is None:
            continue
        for p, r in row["powers"].items():
            o = old["powers"][p]
            pairs = [(r["K0"], o["K0"]), (r["K1"], o["K1"])]
            pairs += [(r["K2b"][s], o["K2b"][s]) for s in r["K2b"]]
            pairs += [(r["K3"][s], o["K3"][s]) for s in r["K3"]]
            for new, ref in pairs:
                for k in FIGS:
                    worst = max(worst, abs(float(new[k]) - float(ref[k])))
                    compared += 1
    return {"max_abs_difference": worst, "values_compared": compared}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--dir", default=str(RUNS_DIR), help="other than artifacts/runs: dry run")
    parser.add_argument("--v2-dir", default=str(RUNS_DIR))
    args = parser.parse_args()
    runs = Path(args.dir).resolve()
    src = runs / f"{args.name}.json.gz"
    out_path = runs / f"{args.name}_analysis.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists -- artifacts are never overwritten")
    if runs == RUNS_DIR.resolve():
        require_clean_tree([out_path])
    with gzip.open(src, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    with gzip.open(Path(args.v2_dir) / f"{V2_NAME}.json.gz", "rt", encoding="utf-8") as handle:
        v2 = json.load(handle)
    res = analyse(data)
    result: dict[str, Any] = {
        "provenance": provenance("not used", ignore_paths=[out_path]),
        "source_artifact": src.name,
        "source_commit": data["provenance"]["git/commit"],
        "threshold": GAP_LIMIT,
        **res,
        "control_v2": control_against_v2(data, v2),
        "replan_check": data["checks"].get("check:replan"),
    }
    print(f"reading {res['reading']}")
    for name, s in res["summary"].items():
        print(
            f"  {name:4s} cells {s['cells']:4d}  >1%: {s['cells_above_limit']:4d}  largest "
            f"{100 * s['largest']['co2_net_median']:.2f} % ({s['largest']['cell']})  median "
            f"{100 * s['median_of_cell_medians']:.2f} %"
        )
    print(
        f"  checks: {len(res['checks']['order_violations'])} order violations, "
        f"{len(res['checks']['failures'])} failures; control v2 {result['control_v2']}"
    )
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

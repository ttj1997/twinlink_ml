"""Reading of the post-hoc extension v2 of the campaign-network study (committed before the run v2).

DECLARED POST HOC: written after the run v1 and its reading (N1 everywhere, R1 with gaps up to
7.30 %), on request of the project lead (2026-11-03). v1 and its pre-registered reading
(``scripts/analyze_campaign_network.py``) stay unchanged and are the main statement.

Input ``artifacts/runs/<name>.json.gz`` of ``scripts/run_campaign_network_v2.py`` and, for the
control, ``artifacts/runs/campaign_network_v1.json.gz``; output
``artifacts/runs/<name>_analysis.json`` (never overwritten) and tables on stdout.

Readings (thresholds and statistics as in v1: median over the 5 draws per cell):

1. **Per process-water level** (0 / 100 / 200 kW constant, seasonal 200 kW Oct-Apr): N0 / N1 /
   between exactly as pre-registered for v1 (N0: median V_N < 2 % at every a >= 0.57; N1: median
   V_N >= 5 % at a = 0.57; per source power P and mode; overall N0 if N0 everywhere, N1 if N1
   somewhere). Reported with the table of every (P, mode), never as a letter alone.
2. **Storage rule K2b**: gap ``g = (K3(S) - K2b(S)) / K3(S)`` of the net CO2 saving, median over the
   draws, in every cell (level, P, a, mode) with storage S = 2 / 8 / 24 h (with S = 0 the rule
   K2b is K1, so the cells without storage do not test a storage rule). **RB0** if every median
   gap <= 1 %: a simple rule suffices; the gap of v1 was due to the storage rule. Otherwise
   **RB1**: the gap and the cells above 1 % are reported. Reported alongside (declared before the
   run): the same for the old rule K2, the gap on the heat used, and the decomposition
   ``g = cascade part + storage part`` with cascade part ``(K3(0) - K1) / K3(S)`` (the order of the
   cascade, no storage) and storage part ``((K3(S) - K3(0)) - (K2b(S) - K1)) / K3(S)`` (the
   storage benefit the rule misses).

Checks: K3 >= K2b >= K1 (and K3 >= K2 >= K1 >= K0) in heat used and net CO2 in every draw
(tolerance 1e-3); every LP status 0; control: the level water_200 has the fingerprint of the v1
main case and reproduces every K0 / K1 / K2 / K3 key figure of v1 (largest absolute difference
reported; expected 0). Process water 0 kW has no identically defined cell in v1 (v1 had 200 kW),
so there is nothing to reproduce for it.

Seen before this reading was committed: nothing of v2; a dry run of the pipeline on a two-week
horizon in the scratch directory (not an artifact) checked the scripts.
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

from _common import RUNS_DIR, provenance, require_clean_tree  # noqa: E402
from analyze_campaign_network import (  # noqa: E402
    N0_LIMIT,
    N1_AVAILABILITY,
    N1_LIMIT,
    ORDER_TOL,
    RULE_GAP_LIMIT,
    _draw_rows,
    _print,
    _stats,
    analyse_variant,
)

V1_NAME = "campaign_network_v1"
KEYS = ("use_kwh", "co2_net_kg", "gas_kwh", "co2_pump_kg", "charge_kwh", "discharge_kwh")


def storage_rule_gaps(data: dict[str, Any], level: str) -> dict[str, Any]:
    """Gaps of K2b and K2 to K3 with storage, their decomposition and the K2b order check."""
    cfg = data["variant_configs"][level]
    powers = [f"{p:g}" for p in cfg["source"]["power_kw_levels"]]
    avail = [float(a) for a in cfg["campaigns"]["availability_levels"]]
    draws = int(data["budget"]["draws"])
    sizes = [f"{s:g}" for s in data["budget"]["storage_hours"] if s > 0]
    cells: dict[str, Any] = {}
    violations: list[str] = []
    for mode in cfg["campaigns"]["modes"]:
        for p in powers:
            for a in avail:
                rows = [r["powers"][p] for r in _draw_rows(data, level, a, mode, draws)]
                for s in sizes:
                    out: dict[str, Any] = {}
                    for rule in ("K2b", "K2"):
                        g, g_use, casc, stor = [], [], [], []
                        for r in rows:
                            k3, k30, k1, k = r["K3"][s], r["K3"]["0"], r["K1"], r[rule][s]
                            den = k3["co2_net_kg"]
                            g.append((den - k["co2_net_kg"]) / den)
                            g_use.append((k3["use_kwh"] - k["use_kwh"]) / k3["use_kwh"])
                            casc.append((k30["co2_net_kg"] - k1["co2_net_kg"]) / den)
                            stor.append(
                                ((den - k30["co2_net_kg"]) - (k["co2_net_kg"] - k1["co2_net_kg"]))
                                / den
                            )
                        out[rule] = {
                            "co2_net": _stats(g),
                            "use": _stats(g_use),
                            "cascade_part": _stats(casc),
                            "storage_part": _stats(stor),
                            "utilisation": _stats([r[rule][s]["utilisation"] for r in rows]),
                        }
                    for d, r in enumerate(rows):
                        for key in ("use_kwh", "co2_net_kg"):
                            if r["K2b"][s][key] < r["K1"][key] - ORDER_TOL:
                                violations.append(f"{mode}/{p}/{a:g}/{d}/{s}: K2b<K1 {key}")
                            if r["K3"][s][key] < r["K2b"][s][key] - ORDER_TOL:
                                violations.append(f"{mode}/{p}/{a:g}/{d}/{s}: K3<K2b {key}")
                    cells[f"{mode}|{p}|{a:g}|{s}"] = out
    return {"cells": cells, "order_violations_k2b": violations}


def control_against_v1(data: dict[str, Any], v1: dict[str, Any]) -> dict[str, Any]:
    """Largest absolute difference of every key figure of water_200 against v1 main."""
    worst = 0.0
    compared = 0
    for key, row in data["draws"].items():
        if not key.startswith("draw:water_200:"):
            continue
        old = v1["draws"][key.replace("draw:water_200:", "draw:main:")]
        for p, r in row["powers"].items():
            o = old["powers"][p]
            pairs = [(r["K0"], o["K0"]), (r["K1"], o["K1"])]
            pairs += [(r["K2"][s], o["K2"][s]) for s in o["K2"]]
            pairs += [(r["K3"][s], o["K3"][s]) for s in o["K3"]]
            for new, ref in pairs:
                for k in KEYS:
                    worst = max(worst, abs(float(new[k]) - float(ref[k])))
                    compared += 1
            worst = max(worst, abs(r["V_N"] - o["V_N"]))
    same_print = data["fingerprints"]["water_200"] == v1["fingerprints"]["main"]
    return {
        "max_abs_difference": worst,
        "values_compared": compared,
        "same_fingerprint": same_print,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--dir", default=str(RUNS_DIR), help="other than artifacts/runs: dry run")
    parser.add_argument("--v1-dir", default=str(RUNS_DIR))
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
    with gzip.open(Path(args.v1_dir) / f"{V1_NAME}.json.gz", "rt", encoding="utf-8") as handle:
        v1 = json.load(handle)
    result: dict[str, Any] = {
        "provenance": provenance("not used", ignore_paths=[out_path]),
        "declared_post_hoc": True,
        "source_artifact": src.name,
        "source_commit": data["provenance"]["git/commit"],
        "thresholds": {
            "N0": N0_LIMIT,
            "N1": N1_LIMIT,
            "N1_availability": N1_AVAILABILITY,
            "rule_gap": RULE_GAP_LIMIT,
        },
        "levels": {},
    }
    all_gaps: list[tuple[float, str, str]] = []
    for level in data["variant_configs"]:
        res = analyse_variant(data, level)
        res["storage_rules"] = storage_rule_gaps(data, level)
        result["levels"][level] = res
        _print(level, res)
        for key, cell in res["storage_rules"]["cells"].items():
            all_gaps.append((cell["K2b"]["co2_net"]["median"], level, key))
        print(f"  K2b order violations: {len(res['storage_rules']['order_violations_k2b'])}")
    above = sorted((g for g in all_gaps if g[0] > RULE_GAP_LIMIT), reverse=True)
    worst = max(all_gaps)
    result["rule_k2b"] = {
        "reading": "RB0" if worst[0] <= RULE_GAP_LIMIT else "RB1",
        "largest": {"co2_net_median": worst[0], "level": worst[1], "cell": worst[2]},
        "cells": len(all_gaps),
        "cells_above_limit": [{"gap": g, "level": lv, "cell": c} for g, lv, c in above],
        "median_of_cell_medians": float(np.median([g[0] for g in all_gaps])),
    }
    result["control_v1"] = control_against_v1(data, v1)
    print(
        f"\nK2b reading {result['rule_k2b']['reading']}: largest {100 * worst[0]:.2f} % "
        f"({worst[1]} {worst[2]}), {len(above)} of {len(all_gaps)} cells above 1 %"
    )
    print(f"control v1: {result['control_v1']}")
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

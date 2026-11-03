"""Reading of the campaign-network study (pre-registered: this docstring is committed before the run).

Input: ``artifacts/runs/<name>.json.gz`` of ``scripts/run_campaign_network.py``. Output:
``artifacts/runs/<name>_analysis.json`` (never overwritten) and the tables on stdout.

Question. Does a heat network pay when the consumers do not run all the time? One constant source
(compressor station, water), process-air preheaters of plants A, B, C with on/off campaigns
(availability ``a``), hall heating and process water; configurations K0 (no network, source fixed
to plant A, surplus to the cooling tower), K1 (network, cascade rule), K2 (K1 + storage rule,
0/2/8/24 h of the source power), K3 (LP with full foresight, same storage; upper bound). Every
value is a placeholder (``conf/campaign_network/campaign_network.yaml``); a scenario band, not a
plant statement.

Statistic. ``V_N = (use K1 - use K0) / source energy`` per draw; per cell (variant, source power
P, mode independent/joint, a) the MEDIAN over the 5 draws, with min and max. At ``a = 1`` the one
schedule counts for every draw and mode.

Pre-registered reading (main variant: approach 10 K, supply 80 degC), per source power P and mode:

* **N0** -- median ``V_N < 2 %`` at every ``a >= 0.57`` (1.0, 0.8, 0.57): the distribution benefit
  of the network vanishes also with campaign operation.
* **N1** -- median ``V_N >= 5 %`` at ``a = 0.57``: distribution pays with consumers that stand
  still.
* otherwise **between**: measurable but small.

Overall: N0 if N0 in every (P, mode); N1 if N1 in at least one (P, mode); otherwise between. The
letter is reported together with the table of every (P, mode); it is never quoted alone. Also
reported, declared before the run: the part due to standstills ``V_N(a) - V_N(1)`` per draw (the
rest of ``V_N`` is the network serving further consumers at full availability), and whether the
letter holds in every draw (min/max instead of the median).

Rule against foresight. Gap ``g = (K3(S) - K(S)) / K3(S)`` of the net CO2 saving (gas CO2 minus
pump electricity CO2), with K = K1 for S = 0 and K = K2 for S > 0; median over the draws per cell
(P, a, mode, S). **R0** "a simple rule suffices, no learning or predictive controller needed" if
the median gap is <= 1 % in every cell of the main variant; otherwise **R1**: the gaps are
reported (largest cells, by S, a, P). The same gap on the heat used is reported alongside. Expected
by structure (stated before the run): with storage the gap can only come from the storage
dispatch and from the cascade order; K3 is an ideal-network bound (any split, any match).

Saturation (reference Befund 100: without further sinks the compressor heat saturated between 800
and 1200 kW). Marginal use between neighbouring source levels ``m_i = (use(P_i) - use(P_{i-1})) /
((P_i - P_{i-1}) x 8760 h)``, median over draws; the consumers "saturate from P_i" at the first
``P_i`` with ``m_i < 0.25``, per configuration (K1, K2 24 h, K3 0 h, K3 24 h), a and mode.
Descriptive: the fine sweep (100-3000 kW, a = 1, K1 and K3 0 h) with the same threshold per
100 kW step, and the annual heat the consumers can take at all.

Checks reported: every draw K3 >= K2 >= K1 >= K0 in use and net CO2 (tolerance 1e-3 kWh / kg),
every LP status 0, realised availability per plant (mean, range).

Sensitivity (variants approach_5, supply_70): the same tables and letters, reported as
sensitivity; they do not change the main reading.

Seen before this reading was committed (declared): the thresholds 2 % / 5 % / 1 % and the
factors come from the task, not from a look at results. While building the model, one probe draw
(a = 0.57, independent, draw 0, main variant, full year) gave the utilisation K0 0.51 / K1 0.97 at
400 kW and K0 0.19 / K1 0.63 at 1200 kW, with K3 0 h / 8 h 0.97 / 0.98 and 0.65 / 0.65; a dry run
of the whole pipeline on a two-week horizon (first two weeks of January, into the scratch
directory, not an artifact) checked the scripts and showed overall N1 and rule gaps up to about
9 % in all three variants. No setting was changed after either.
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

N0_LIMIT = 0.02
N1_LIMIT = 0.05
N1_AVAILABILITY = 0.57
RULE_GAP_LIMIT = 0.01
SATURATION_MARGINAL = 0.25
ORDER_TOL = 1e-3


def _draw_rows(data: dict[str, Any], variant: str, a: float, mode: str, draws: int) -> list[dict]:
    if a >= 1.0:
        row = data["draws"][f"draw:{variant}:1:independent:0"]
        return [row] * draws
    return [data["draws"][f"draw:{variant}:{a:g}:{mode}:{d}"] for d in range(draws)]


def _stats(values: list[float]) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def _letter(vn: dict[float, dict[str, float]], key: str) -> str:
    if all(vn[a][key] < N0_LIMIT for a in vn if a >= N1_AVAILABILITY):
        return "N0"
    if vn[N1_AVAILABILITY][key] >= N1_LIMIT:
        return "N1"
    return "between"


def _configs(row: dict[str, Any], sizes: list[str]) -> dict[str, dict[str, Any]]:
    out = {"K0": row["K0"], "K1": row["K1"]}
    for s in sizes:
        if s != "0":
            out[f"K2_{s}"] = row["K2"][s]
        out[f"K3_{s}"] = row["K3"][s]
    return out


def analyse_variant(data: dict[str, Any], variant: str) -> dict[str, Any]:
    cfg = data["variant_configs"][variant]
    powers = [f"{p:g}" for p in cfg["source"]["power_kw_levels"]]
    avail = [float(a) for a in cfg["campaigns"]["availability_levels"]]
    modes = list(cfg["campaigns"]["modes"])
    draws = int(data["budget"]["draws"])
    sizes = [f"{s:g}" for s in data["budget"]["storage_hours"]]
    out: dict[str, Any] = {"cells": {}, "letters": {}, "gaps": {}, "saturation": {}, "checks": {}}
    violations: list[str] = []
    lp_bad: list[str] = []
    realised: dict[str, list[float]] = {}
    for mode in modes:
        for p in powers:
            vn: dict[float, dict[str, float]] = {}
            for a in avail:
                rows = _draw_rows(data, variant, a, mode, draws)
                vals = [r["powers"][p]["V_N"] for r in rows]
                base = [r["powers"][p]["V_N"] for r in _draw_rows(data, variant, 1.0, mode, draws)]
                cell = {
                    "V_N": _stats(vals),
                    "V_N_minus_full_availability": _stats(
                        [v - b for v, b in zip(vals, base, strict=True)]
                    ),
                    "configs": {},
                }
                for name in _configs(rows[0]["powers"][p], sizes):
                    figs = [_configs(r["powers"][p], sizes)[name] for r in rows]
                    cell["configs"][name] = {
                        "utilisation": _stats([f["utilisation"] for f in figs]),
                        "co2_net_t": _stats([f["co2_net_kg"] / 1000.0 for f in figs]),
                        "gas_mwh": _stats([f["gas_kwh"] / 1000.0 for f in figs]),
                        "pump_mwh_el": _stats([f["pump_electricity_kwh"] / 1000.0 for f in figs]),
                    }
                gaps: dict[str, Any] = {}
                for s in sizes:
                    rule = "K1" if s == "0" else f"K2_{s}"
                    g_co2, g_use = [], []
                    for r in rows:
                        c = _configs(r["powers"][p], sizes)
                        k3, k = c[f"K3_{s}"], c[rule]
                        g_co2.append((k3["co2_net_kg"] - k["co2_net_kg"]) / k3["co2_net_kg"])
                        g_use.append((k3["use_kwh"] - k["use_kwh"]) / k3["use_kwh"])
                    gaps[s] = {"rule": rule, "co2_net": _stats(g_co2), "use": _stats(g_use)}
                cell["gaps"] = gaps
                for d, r in enumerate(rows):
                    c = _configs(r["powers"][p], sizes)
                    for key in ("use_kwh", "co2_net_kg"):
                        if c["K1"][key] < c["K0"][key] - ORDER_TOL:
                            violations.append(f"{mode}/{p}/{a:g}/{d}: K1<K0 {key}")
                        if c["K3_0"][key] < c["K1"][key] - ORDER_TOL:
                            violations.append(f"{mode}/{p}/{a:g}/{d}: K3_0<K1 {key}")
                        for s in sizes:
                            if s == "0":
                                continue
                            if c[f"K2_{s}"][key] < c["K1"][key] - ORDER_TOL:
                                violations.append(f"{mode}/{p}/{a:g}/{d}: K2_{s}<K1 {key}")
                            if c[f"K3_{s}"][key] < c[f"K2_{s}"][key] - ORDER_TOL:
                                violations.append(f"{mode}/{p}/{a:g}/{d}: K3_{s}<K2_{s} {key}")
                    for s in sizes:
                        if c[f"K3_{s}"].get("lp_status", 0) != 0:
                            lp_bad.append(f"{mode}/{p}/{a:g}/{d}/{s}")
                    for plant, v in r["realised_availability"].items():
                        realised.setdefault(f"{a:g}/{mode}/{plant}", []).append(v)
                vn[a] = cell["V_N"]
                out["cells"][f"{mode}|{p}|{a:g}"] = cell
            out["letters"][f"{mode}|{p}"] = {
                "median": _letter(vn, "median"),
                "every_draw_min": _letter(vn, "min"),
                "every_draw_max": _letter(vn, "max"),
            }
    letters = [v["median"] for v in out["letters"].values()]
    out["overall"] = (
        "N0" if all(x == "N0" for x in letters) else "N1" if "N1" in letters else "between"
    )
    worst = max(
        (
            (cell["gaps"][s]["co2_net"]["median"], key, s)
            for key, cell in out["cells"].items()
            for s in cell["gaps"]
        ),
        key=lambda x: x[0],
    )
    out["rule_reading"] = "R0" if worst[0] <= RULE_GAP_LIMIT else "R1"
    out["largest_gap"] = {"co2_net_median": worst[0], "cell": worst[1], "storage_h": worst[2]}
    out["gaps_above_limit"] = sorted(
        (
            {"cell": key, "storage_h": s, "co2_net_median": cell["gaps"][s]["co2_net"]["median"]}
            for key, cell in out["cells"].items()
            for s in cell["gaps"]
            if cell["gaps"][s]["co2_net"]["median"] > RULE_GAP_LIMIT
        ),
        key=lambda x: -x["co2_net_median"],
    )
    # saturation on the grid
    pw = [float(p) for p in powers]
    for mode in modes:
        for a in avail:
            for name in ("K1", f"K2_{sizes[-1]}", "K3_0", f"K3_{sizes[-1]}"):
                if name.startswith("K2_0"):
                    continue
                use = [
                    out["cells"][f"{mode}|{p}|{a:g}"]["configs"][name]["utilisation"]["median"]
                    * float(p)
                    for p in powers
                ]
                marg = [(use[i] - use[i - 1]) / (pw[i] - pw[i - 1]) for i in range(1, len(pw))]
                first = next(
                    (pw[i] for i in range(1, len(pw)) if marg[i - 1] < SATURATION_MARGINAL), None
                )
                out["saturation"][f"{mode}|{a:g}|{name}"] = {
                    "marginal": dict(zip(powers[1:], marg, strict=True)),
                    "saturates_from_kw": first,
                }
    sweep = data["sweeps"][variant]
    rows = sweep["rows"]
    sw: dict[str, Any] = {"demand_kwh": sweep["demand_kwh"]}
    for name in ("K1", "K3"):
        p = [r["power_kw"] for r in rows]
        u = [r[name]["use_kwh"] for r in rows]
        marg = [(u[i] - u[i - 1]) / ((p[i] - p[i - 1]) * 8760.0) for i in range(1, len(p))]
        sw[name] = {
            "use_mwh": [x / 1000.0 for x in u],
            "utilisation": [r[name]["utilisation"] for r in rows],
            "marginal": marg,
            "saturates_from_kw": next(
                (p[i] for i in range(1, len(p)) if marg[i - 1] < SATURATION_MARGINAL), None
            ),
            "ceiling_mwh": u[-1] / 1000.0,
        }
    sw["powers_kw"] = [r["power_kw"] for r in rows]
    out["sweep"] = sw
    out["checks"] = {
        "order_violations": violations,
        "lp_not_optimal": lp_bad,
        "realised_availability": {k: _stats(v) for k, v in sorted(realised.items())},
    }
    return out


def _print(variant: str, res: dict[str, Any]) -> None:
    print(
        f"\n=== {variant}: overall {res['overall']}, rule {res['rule_reading']} "
        f"(largest gap {100 * res['largest_gap']['co2_net_median']:.2f} % at "
        f"{res['largest_gap']['cell']} S={res['largest_gap']['storage_h']} h)"
    )
    print("V_N median [min, max] in %")
    for key, cell in res["cells"].items():
        v = cell["V_N"]
        d = cell["V_N_minus_full_availability"]
        k = cell["configs"]
        print(
            f"  {key:28s} V_N {100 * v['median']:6.2f} [{100 * v['min']:6.2f}, {100 * v['max']:6.2f}]"
            f"  dV {100 * d['median']:6.2f}  use K0 {100 * k['K0']['utilisation']['median']:5.1f}"
            f" K1 {100 * k['K1']['utilisation']['median']:5.1f}"
            f" K3_0 {100 * k['K3_0']['utilisation']['median']:5.1f}"
            f" K3_24 {100 * k.get('K3_24', k['K3_0'])['utilisation']['median']:5.1f}"
        )
    for key, v in res["letters"].items():
        print(f"  letter {key}: {v}")
    print(
        f"  sweep K1 saturates from {res['sweep']['K1']['saturates_from_kw']} kW, "
        f"K3 from {res['sweep']['K3']['saturates_from_kw']} kW"
    )
    print(
        f"  checks: {len(res['checks']['order_violations'])} order violations, "
        f"{len(res['checks']['lp_not_optimal'])} LP not optimal"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--dir", default=str(RUNS_DIR), help="other than artifacts/runs: dry run")
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
    result: dict[str, Any] = {
        "provenance": provenance("not used", ignore_paths=[out_path]),
        "source_artifact": src.name,
        "source_commit": data["provenance"]["git/commit"],
        "thresholds": {
            "N0": N0_LIMIT,
            "N1": N1_LIMIT,
            "N1_availability": N1_AVAILABILITY,
            "rule_gap": RULE_GAP_LIMIT,
            "saturation_marginal": SATURATION_MARGINAL,
        },
        "variants": {},
    }
    for variant in data["variant_configs"]:
        res = analyse_variant(data, variant)
        result["variants"][variant] = res
        _print(variant, res)
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

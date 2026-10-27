"""Analysis of the storage-price study; the reading below is fixed BEFORE the run (pre-registration).

Study document ``docs/NN_storage_price.md``; artifact of ``scripts/run_storage_price.py``.

Pre-registered reading (committed before the main run; nothing below may be changed afterwards,
additions only as declared post-hoc analyses)
-----------------------------------------------------------------------------------------------

Test set: the four test months February, May, August, November 2024 (one per quarter, disjoint
from the eight training/tuning months), main case of ``conf/storage_price/storage_price.yaml``.
Each month is one run from an empty storage; the content left at its end is credited at
``end_value`` (mean electricity price of the training months / COP). ``J_X`` = cost of controller
``X`` summed over the four test months (EUR, end credit included); MPC horizon 36 h (the first of
``mpc_horizons_h``; 24 h is reported, not read).

``gap(X, Y) = (J_X - J_Y) / J_Y x 100 %`` (positive: X is dearer than Y).

RL: per algorithm (SAC, TD3) the FINAL policy of each seed (3 seeds, or fewer after the budget
rule); ``J_RL`` = median over the seeds, range = min..max. The study-level RL value is the algorithm
with the lower median ``J_RL`` (a best-of-two choice on the test set, optimistic for RL; both are
reported).

Conditions:

* C0: ``gap(R1, M1) <= 2 %``;
* C1a: ``gap(R1, M2) > 2 %`` (M2 beats R1 by more than 2 %);
* C1b: ``gap(RL, M2) <= 1 %`` and ``gap(RL, R1) <= 0`` (RL at most 1 % behind M2 and not behind R1);
* C2: ``gap(RL, M2) > 1 %`` or ``gap(RL, R1) > 0`` (RL more than 1 % behind M2, or behind R1).

Readings (first that applies):

* **H0** if C0: a simple price rule suffices; neither MPC nor RL is justified for this case.
* **H1** if C1a and C1b: RL is an equivalent alternative to MPC (advantage only computing time / no
  model in operation).
* **H2** if C2: RL is not justified for this case.
* **none** otherwise (C0 fails, M2 does not beat R1 by more than 2 %, RL within 1 % of M2 and not
  behind R1): reported as not covered by H0-H2, with the numbers.

Reported with the reading, not part of it: every condition on its own (also for TD3 and SAC
separately, and with the RL minimum and maximum instead of the median); the same quantities per
test month; costs and CO2 of every controller relative to R0; the gap of every controller to M1;
the gap of M1 to the hindsight optimum P*; violations of power and charge limits (counted cuts of
the projection; rules and programmes are expected at 0, RL is projected and counted); time per
decision (from the run: under parallel load; from ``scripts/measure_storage_price_timing.py`` if
given: one process); the in-sample training-month costs; the learning curves; the variants (rules
and MPC only: C0 and C1a per variant).

Output: ``artifacts/runs/<name>_analysis.json`` (never overwritten) and the tables on stdout.
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

THRESHOLD_H0 = 2.0
THRESHOLD_MPC = 2.0
THRESHOLD_RL = 1.0
PRIMARY_HORIZON_H = 36


def gap(x: float, y: float) -> float:
    """``(x - y) / y`` in %."""
    return (x - y) / y * 100.0


def _total(rows: list[dict[str, Any]], field: str = "cost_eur") -> float:
    return float(sum(r[field] for r in rows))


def _by_month(rows: list[dict[str, Any]], field: str = "cost_eur") -> dict[str, float]:
    return {str(r["month"]): float(r[field]) for r in rows}


def conditions(j: dict[str, float]) -> dict[str, Any]:
    """C0, C1a, C1b, C2 and the reading from the totals ``j`` (keys R1, M1, M2, RL)."""
    c0 = gap(j["R1"], j["M1"]) <= THRESHOLD_H0
    c1a = gap(j["R1"], j["M2"]) > THRESHOLD_MPC
    rl_m2 = gap(j["RL"], j["M2"])
    rl_r1 = gap(j["RL"], j["R1"])
    c1b = rl_m2 <= THRESHOLD_RL and rl_r1 <= 0.0
    c2 = rl_m2 > THRESHOLD_RL or rl_r1 > 0.0
    if c0:
        reading = "H0"
    elif c1a and c1b:
        reading = "H1"
    elif c2:
        reading = "H2"
    else:
        reading = "none"
    return {
        "gap_R1_M1": gap(j["R1"], j["M1"]),
        "gap_R1_M2": gap(j["R1"], j["M2"]),
        "gap_RL_M2": rl_m2,
        "gap_RL_R1": rl_r1,
        "C0": c0,
        "C1a": c1a,
        "C1b": c1b,
        "C2": c2,
        "reading": reading,
    }


def analyse(art: dict[str, Any], timing: dict[str, Any] | None) -> dict[str, Any]:
    main = art["rules"]["main"]
    test = main["test"]
    m1, m2 = f"M1_{PRIMARY_HORIZON_H}h", f"M2_{PRIMARY_HORIZON_H}h"
    names = list(test)
    totals = {n: _total(test[n]) for n in names}
    co2 = {n: _total(test[n], "co2_kg") for n in names}
    months = [str(r["month"]) for r in test["R0"]]

    # RL: per algorithm median and range of the final policies
    rl_runs = art["rl"]
    algos: dict[str, dict[str, Any]] = {}
    for run in rl_runs.values():
        algos.setdefault(run["algorithm"], {}).setdefault("runs", []).append(run)
    for d in algos.values():
        costs = [_total(r["test"]) for r in d["runs"]]
        d["seeds"] = [r["seed"] for r in d["runs"]]
        d["test_cost_eur"] = costs
        d["median"] = float(np.median(costs))
        d["min"] = float(np.min(costs))
        d["max"] = float(np.max(costs))
        d["co2_median_kg"] = float(np.median([_total(r["test"], "co2_kg") for r in d["runs"]]))
        d["by_month_median"] = {
            m: float(np.median([_by_month(r["test"])[m] for r in d["runs"]])) for m in months
        }
        d["violations_content"] = [
            sum(x["violations_content"] for x in r["test"]) for r in d["runs"]
        ]
        d["violations_power"] = [sum(x["violations_power"] for x in r["test"]) for r in d["runs"]]
        d["decide_ms_median"] = float(
            np.median([np.median([x["decide_s_median"] for x in r["test"]]) for r in d["runs"]])
            * 1e3
        )
        d["learning_curves"] = [
            {
                "seed": r["seed"],
                "steps": [p["steps"] for p in r["learning_curve"]],
                "train_cost_eur": [p["train_cost_eur"] for p in r["learning_curve"]],
                "test_cost_eur": [p["test_cost_eur"] for p in r["learning_curve"]],
            }
            for r in d["runs"]
        ]
        d["train_cost_final_eur"] = [_total(r["train"]) for r in d["runs"]]
        del d["runs"]
    best = min(algos, key=lambda a: algos[a]["median"]) if algos else None

    base = {"R1": totals["R1"], "M1": totals[m1], "M2": totals[m2]}
    out: dict[str, Any] = {"primary_horizon_h": PRIMARY_HORIZON_H, "rl_study_algorithm": best}
    if best is not None:
        out["reading"] = conditions({**base, "RL": algos[best]["median"]})
        out["per_algorithm"] = {
            a: {
                stat: conditions({**base, "RL": algos[a][stat]})
                for stat in ("median", "min", "max")
            }
            for a in algos
        }
        out["per_month"] = {
            m: conditions(
                {
                    "R1": _by_month(test["R1"])[m],
                    "M1": _by_month(test[m1])[m],
                    "M2": _by_month(test[m2])[m],
                    "RL": algos[best]["by_month_median"][m],
                }
            )
            for m in months
        }
    out["controllers"] = {
        n: {
            "test_cost_eur": totals[n],
            "test_co2_kg": co2[n],
            "cost_vs_R0_pct": gap(totals[n], totals["R0"]),
            "co2_vs_R0_pct": gap(co2[n], co2["R0"]),
            "gap_to_M1_pct": gap(totals[n], totals[m1]),
            "by_month_cost_eur": _by_month(test[n]),
            "by_month_gap_to_M1_pct": {
                m: gap(_by_month(test[n])[m], _by_month(test[m1])[m]) for m in months
            },
            "violations_power": int(sum(r["violations_power"] for r in test[n])),
            "violations_content": int(sum(r["violations_content"] for r in test[n])),
            "largest_cut_kw": float(max(r["largest_cut_kw"] for r in test[n])),
            "decide_ms_median_under_load": float(
                np.median([r["decide_s_median"] for r in test[n]]) * 1e3
            ),
            "decide_ms_p95_under_load": float(max(r["decide_s_p95"] for r in test[n]) * 1e3),
            "pump_heat_kwh": _total(test[n], "pump_heat_kwh"),
            "gas_heat_kwh": _total(test[n], "gas_heat_kwh"),
            "charged_kwh": _total(test[n], "charged_kwh"),
            "end_kwh": [r["end_kwh"] for r in test[n]],
        }
        for n in names
    }
    for a, d in algos.items():
        out["controllers"][a] = {
            "test_cost_eur_median": d["median"],
            "test_cost_eur_range": [d["min"], d["max"]],
            "test_cost_eur_seeds": d["test_cost_eur"],
            "cost_vs_R0_pct_median": gap(d["median"], totals["R0"]),
            "co2_vs_R0_pct_median": gap(d["co2_median_kg"], co2["R0"]),
            "gap_to_M1_pct_median": gap(d["median"], totals[m1]),
            "gap_to_M1_pct_range": [gap(d["min"], totals[m1]), gap(d["max"], totals[m1])],
            "by_month_gap_to_M1_pct_median": {
                m: gap(d["by_month_median"][m], _by_month(test[m1])[m]) for m in months
            },
            "violations_content_per_seed": d["violations_content"],
            "violations_power_per_seed": d["violations_power"],
            "decide_ms_median_under_load": d["decide_ms_median"],
            "train_cost_final_eur": d["train_cost_final_eur"],
            "learning_curves": d["learning_curves"],
        }
    out["gap_M1_to_hindsight_pct"] = gap(totals[m1], totals["P*"])
    out["tuning"] = {
        "kappa": main["r0_tuning"]["kappa"],
        "kappa_at_grid_edge": main["r0_tuning"]["at_grid_edge"],
        "quantile": main["r1_tuning"]["quantile"],
        "quantile_at_grid_edge": main["r1_tuning"]["at_grid_edge"],
    }
    out["train_in_sample_cost_eur"] = {n: _total(v) for n, v in main.get("train", {}).items()}
    variants: dict[str, Any] = {}
    for v, rec in art["rules"].items():
        t = {n: _total(rec["test"][n]) for n in rec["test"]}
        variants[v] = {
            "gap_R1_M1": gap(t["R1"], t[m1]),
            "gap_R1_M2": gap(t["R1"], t[m2]),
            "gap_R0_M1": gap(t["R0"], t[m1]),
            "gap_M2_M1": gap(t[m2], t[m1]),
            "C0": gap(t["R1"], t[m1]) <= THRESHOLD_H0,
            "C1a": gap(t["R1"], t[m2]) > THRESHOLD_MPC,
            "kappa": rec["r0_tuning"]["kappa"],
            "quantile": rec["r1_tuning"]["quantile"],
            "cost_eur": t,
        }
    out["variants"] = variants
    if timing is not None:
        out["timing_single_process"] = timing.get("controllers")
        out["timing_single_process_note"] = timing.get("note")
    return out


def _print(res: dict[str, Any]) -> None:
    print(
        "| controller | cost EUR | vs R0 % | CO2 vs R0 % | gap to M1 % | viol. P/C | ms/decision |"
    )
    print("|---|---|---|---|---|---|---|")
    for n, c in res["controllers"].items():
        if "test_cost_eur" in c:
            print(
                f"| {n} | {c['test_cost_eur']:.0f} | {c['cost_vs_R0_pct']:+.2f} | "
                f"{c['co2_vs_R0_pct']:+.2f} | {c['gap_to_M1_pct']:+.2f} | "
                f"{c['violations_power']}/{c['violations_content']} | "
                f"{c['decide_ms_median_under_load']:.3f} |"
            )
        else:
            lo, hi = c["gap_to_M1_pct_range"]
            print(
                f"| {n} (median) | {c['test_cost_eur_median']:.0f} | "
                f"{c['cost_vs_R0_pct_median']:+.2f} | {c['co2_vs_R0_pct_median']:+.2f} | "
                f"{c['gap_to_M1_pct_median']:+.2f} [{lo:+.2f}, {hi:+.2f}] | "
                f"0/{c['violations_content_per_seed']} | {c['decide_ms_median_under_load']:.3f} |"
            )
    if "reading" in res:
        print("reading:", json.dumps(res["reading"]))
        for m, c in res["per_month"].items():
            print(f"month {m}:", json.dumps(c))
    print(f"M1 vs P*: {res['gap_M1_to_hindsight_pct']:+.3f} %")
    for v, c in res["variants"].items():
        print(
            v, json.dumps({k: c[k] for k in ("gap_R1_M1", "gap_R1_M2", "gap_R0_M1", "C0", "C1a")})
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", required=True, help="name of the run artifact")
    parser.add_argument("--timing", default=None, help="name of the timing artifact (optional)")
    parser.add_argument("--name", required=True, help="name of the analysis artifact")
    args = parser.parse_args()
    out_path = RUNS_DIR / f"{args.name}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists -- artifacts are never overwritten")
    require_clean_tree([out_path])
    with gzip.open(RUNS_DIR / f"{args.artifact}.json.gz", "rt", encoding="utf-8") as handle:
        art = json.load(handle)
    if art["provenance"]["git/dirty"]:
        raise SystemExit("the run artifact was written from a dirty tree")
    timing = None
    if args.timing:
        timing = json.loads((RUNS_DIR / f"{args.timing}.json").read_text(encoding="utf-8"))
    res = analyse(art, timing)
    res = {
        "provenance": provenance("not used", ignore_paths=[out_path]),
        "artifact": args.artifact,
        "artifact_commit": art["provenance"]["git/commit"],
        "timing_artifact": args.timing,
        "timing_note_run": art["timing_note"],
        "budget": art["budget"],
        **res,
    }
    out_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    _print(res)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

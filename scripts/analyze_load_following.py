"""Analysis of the load-following study and its PRE-REGISTERED reading.

Study document: ``docs/NN_load_following.md``. Input: ``artifacts/runs/<name>.json.gz`` written
by ``scripts/run_load_following.py --stage main``. Output: ``artifacts/runs/<name>_analysis.json``.

PRE-REGISTERED READING (committed before stage 0, the dry run and the main run; decisions of the
project lead of 2026-11-17)
------------------------------------------------------------------------------------------------
Family: the test days of the main run; variant E (electric heater, the whole power priced with
the hourly tariff); main band of the throughput (+-20 % around the nominal spray rate, cut with
the admissible range of stage 0); true prices. Every controller at its smallest family-wide
moisture margin with which the path condition holds at every truth sample of every day of the
family, tail included (tolerance of the headroom config); the tail of every run with the margin
of C0.

``J_X`` = cost of controller X per kg of dry product over the family: the sum over the days of
the cost of the day and its tail (hourly truth energy times the TRUE tariff of the hour), divided
by the sum of the dry product (solids sprayed in the day and its tail).
``gap(X, Y) = (J_X - J_Y) / J_Y``.

First class that applies:

* **T0** if ``(J_C0 - J_C4) / J_C0 < 2 %``: load following does not pay on this line, not even
  with perfect preview; the item is closed.
* **T1** if ``gap(C1, C3) <= 2 %``: a price rule suffices; neither schedule, MPC nor RL justified.
* **T2** if ``gap(C2, C3) <= 2 %``: a schedule on the steady-state map suffices; the dynamics add
  nothing.
* **T3** if ``gap(C2, C3) > 2 %``: the dynamics pay, MPC is justified; an RL-against-MPC study is
  justified only if the gap exceeds 5 %.

Every condition is also reported on its own, per test month and per tercile of the daily price
spread. ``gap(C3, C4)`` is reported as the value of perfect information. Reduced stage 1 (rule of
the study document, decided by stage 0 before any truth run): if the static gain on the training
days is below 2 % and the admissible range is narrower than +-10 %, only C0, C2 and C4 are run;
the reading is then T0 or "none (reduced)".

The reading is INVALID (reported as such, not replaced) if a margin search of C0-C4 of this family
did not converge or if a unit of this family ended with an error.

No reading is taken from: variant G (C0, C1, C2; reported next to E), the constant-price control,
the box sensitivity (C0, C2, C4), the weekly target (steady-state map only).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import RUNS_DIR, provenance, require_clean_tree  # noqa: E402

from twinlink_ml.evaluation.net_energy import ELECTRICITY_CO2_KG_PER_KWH  # noqa: E402

THRESHOLD = 0.02
THRESHOLD_RL = 0.05
READING_KINDS = ("C0", "C1", "C2", "C3", "C4")


def cost_per_kg(records: list[dict[str, Any]], field: str = "cost_e") -> float:
    """Sum of the cost over the days divided by the sum of the dry product (EUR/kg)."""
    good = [r for r in records if r.get("max_violation") is not None]
    return float(sum(r[field] for r in good) / sum(r["product_dry_kg"] for r in good))


def gap(x: float, y: float) -> float:
    """``(x - y) / y``."""
    return (x - y) / y


def classify(costs: dict[str, float], *, reduced: bool) -> dict[str, Any]:
    """Conditions and class of the pre-registered reading for one set of ``J_X``."""
    out: dict[str, Any] = {"J": costs}
    out["headroom_preview"] = (costs["C0"] - costs["C4"]) / costs["C0"]
    out["T0"] = bool(out["headroom_preview"] < THRESHOLD)
    if reduced or "C3" not in costs:
        out["reading"] = "T0" if out["T0"] else "none (reduced)"
        return out
    out["gap_C1_C3"] = gap(costs["C1"], costs["C3"])
    out["gap_C2_C3"] = gap(costs["C2"], costs["C3"])
    out["gap_C3_C4"] = gap(costs["C3"], costs["C4"])
    for kind in ("C1", "C2", "C3"):
        out[f"gain_{kind}_vs_C0"] = (costs["C0"] - costs[kind]) / costs["C0"]
    out["T1"] = bool(out["gap_C1_C3"] <= THRESHOLD)
    out["T2"] = bool(out["gap_C2_C3"] <= THRESHOLD)
    out["T3"] = bool(out["gap_C2_C3"] > THRESHOLD)
    out["rl_study_justified"] = bool(out["gap_C2_C3"] > THRESHOLD_RL)
    if out["T0"]:
        out["reading"] = "T0"
    elif out["T1"]:
        out["reading"] = "T1"
    elif out["T2"]:
        out["reading"] = "T2"
    else:
        out["reading"] = "T3"
    return out


def describe(records: list[dict[str, Any]], search: dict[str, Any] | None) -> dict[str, Any]:
    """Diagnostics of one controller on the family."""
    good = [r for r in records if r.get("max_violation") is not None]
    energy = np.array([r["energy_kwh"] for r in good])
    heater = np.array([r["heater_kwh"] for r in good])
    deviation = np.array([r["sprayed_day_kg"] / r["target_day_kg"] - 1.0 for r in good])
    solver = [r["solver"] for r in good]
    decide = [r["decide_s"] for r in good if r["decide_s"]]
    out: dict[str, Any] = {
        "days": len(good),
        "errors": len(records) - len(good),
        "margin": good[0]["margin"],
        "tail_margin": good[0]["tail_margin"],
        "cost_e_per_kg": cost_per_kg(records, "cost_e"),
        "cost_g_per_kg": cost_per_kg(records, "cost_g"),
        "energy_kwh_per_day_median": float(np.median(energy)),
        "energy_kwh_total": float(energy.sum()),
        "co2_kg_all_electric": float(ELECTRICITY_CO2_KG_PER_KWH * energy.sum()),
        "heater_share_of_energy": float(heater.sum() / energy.sum()),
        "max_violation": float(max(r["max_violation"] for r in good)),
        "max_violation_tail": float(max(r["max_violation_tail"] for r in good)),
        "max_bed_temperature": float(max(r["max_bed_temperature"] for r in good)),
        "production_deviation_max_abs": float(np.abs(deviation).max()),
        "share_at_low_mean": float(np.mean([r["share_at_low"] for r in good])),
        "share_at_high_mean": float(np.mean([r["share_at_high"] for r in good])),
        "d50_um_range": [
            float(min(r["d50_um_range"][0] for r in good)),
            float(max(r["d50_um_range"][1] for r in good)),
        ],
        "fluidisation_margin_min": [
            float(min(r["fluidisation_margin_min"][0] for r in good)),
            float(min(r["fluidisation_margin_min"][1] for r in good)),
        ],
        "setpoint_failures": int(sum(s.get("failures", 0) for s in solver)),
        "tail_failures": int(sum(s.get("tail_failures", 0) for s in solver)),
        "plan_failures": int(sum(s.get("plan_failures", 0) for s in solver)),
        "fallback_to_guess": int(sum(bool(s.get("fallback_to_guess", False)) for s in solver)),
        "decide_s_per_decision_mean": float(np.mean([d["mean"] for d in decide])),
        "decide_s_per_decision_max": float(np.max([d["max"] for d in decide])),
        "reset_s_mean": float(np.mean([r["reset_s"] for r in good])),
    }
    if search is not None:
        out["search_converged"] = bool(search["converged"])
        out["search_bracket"] = search["bracket"]
        out["search_evaluations"] = len(search["evaluations"])
    return out


def main() -> int:
    """Read the artifact of the main run, apply the reading, write the analysis artifact."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", default=str(RUNS_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out_dir).resolve()
    smoke = out_dir != RUNS_DIR.resolve()
    source = out_dir / f"{args.name}.json.gz"
    output = out_dir / f"{args.name}_analysis.json"
    if output.exists():
        raise SystemExit(f"{output} exists -- artifacts are never overwritten; use a new name")
    if not smoke:
        require_clean_tree([output])
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        run = json.load(handle)
    part = run["main"]
    reduced = bool(run["reduced_stage1"])
    kinds = [k for k in READING_KINDS if k in part]
    spread = {int(d): v for d, v in run["day_price_spread"].items()}
    month = {int(d): v for d, v in run["day_month"].items()}

    def costs(
        selector: Callable[[int], bool] | None = None, field: str = "cost_e"
    ) -> dict[str, float]:
        out = {}
        for kind in kinds:
            records = [
                r for r in part[kind]["records"] if selector is None or selector(r["day_of_year"])
            ]
            out[kind] = cost_per_kg(records, field)
        return out

    reading = classify(costs(), reduced=reduced)
    invalid = [k for k in kinds if not part[k]["search"]["converged"]]
    errors = [e for e in run["errors"] if e["band"] == "main" and e["carrier"] == "E"]
    reading["valid"] = not invalid and not errors
    reading["invalid_reasons"] = {"margin_not_converged": invalid, "errored_units": len(errors)}

    months = sorted(set(month.values()))
    per_month = {
        str(m): classify(costs(lambda d, m=m: month[d] == m), reduced=reduced) for m in months
    }
    tercile: dict[int, int] = {}
    for m in months:
        days = sorted((d for d in month if month[d] == m), key=lambda d: (spread[d], d))
        for index, chunk in enumerate(np.array_split(np.array(days), 3)):
            for d in chunk:
                tercile[int(d)] = index
    per_tercile = {
        name: classify(costs(lambda d, i=i: tercile[d] == i), reduced=reduced)
        for i, name in enumerate(("low", "middle", "high"))
    }

    payload: dict[str, Any] = {
        "provenance": provenance("not used", ignore_paths=[output]),
        "source": source.name,
        "source_commit": run["provenance"]["git/commit"],
        "smoke": smoke,
        "days": run["days"],
        "n_days": len(run["day_ids"]),
        "thresholds": {"reading": THRESHOLD, "rl_study": THRESHOLD_RL},
        "reduced_stage1": reduced,
        "margins": run["margins"],
        "reading": reading,
        "per_month": per_month,
        "per_tercile": per_tercile,
        "controllers": {k: describe(part[k]["records"], part[k]["search"]) for k in kinds},
        "errors_total": len(run["errors"]),
        "units_run": run["units_run"],
        "runtime_h": run["runtime_s"] / 3600.0,
        "part_runtime_h": {k: v / 3600.0 for k, v in run["part_runtime_s"].items()},
        "peak_memory_mb_per_worker": run["peak_memory_mb_per_worker"],
        "seconds_per_day_run": run["seconds_per_day_run"],
    }
    j0 = reading["J"]["C0"]
    if "decomposition" in run:
        payload["decomposition"] = {}
        for kind, records in run["decomposition"].items():
            c0_at = cost_per_kg(records)
            payload["decomposition"][kind] = {
                "margin_share": (j0 - c0_at) / j0,
                "schedule_share": (c0_at - reading["J"][kind]) / j0,
                "C0_at_margin_cost_per_kg": c0_at,
            }
    if "control" in run:
        flat = {k: cost_per_kg(v) for k, v in run["control"].items()}
        payload["control_constant_price"] = {
            "J": flat,
            "gain_vs_C0": {k: (flat["C0"] - v) / flat["C0"] for k, v in flat.items()},
            "max_violation": {
                k: float(max(r["max_violation"] for r in v if r.get("max_violation") is not None))
                for k, v in run["control"].items()
            },
        }
    if "gas" in run:
        g0 = cost_per_kg(part["C0"]["records"], "cost_g")
        gas_costs = {k: cost_per_kg(v["records"], "cost_g") for k, v in run["gas"].items()}
        payload["gas"] = {
            "J": {"C0": g0, **gas_costs},
            "gain_vs_C0": {k: (g0 - v) / g0 for k, v in gas_costs.items()},
            "margins": {k: v["margin"] for k, v in run["gas"].items()},
            "converged": {k: v["search"]["converged"] for k, v in run["gas"].items()},
            "E_controllers_booked_as_gas": costs(field="cost_g"),
        }
    if "box" in run:
        box = {k: cost_per_kg(v["records"]) for k, v in run["box"].items()}
        payload["box"] = {
            "J": {"C0": j0, **box},
            "gain_vs_C0": {k: (j0 - v) / j0 for k, v in box.items()},
            "margins": {k: v["margin"] for k, v in run["box"].items()},
            "converged": {k: v["search"]["converged"] for k, v in run["box"].items()},
            "controllers": {k: describe(v["records"], v["search"]) for k, v in run["box"].items()},
        }
    if "weekly" in run and run["weekly"].get("weeks"):
        payload["weekly"] = {}
        for carrier in ("E", "G"):
            weeks = run["weekly"][carrier]
            constant = sum(w["cost_constant"] for w in weeks)
            daily = sum(w["cost_daily_target"] for w in weeks)
            weekly = sum(w["cost_weekly_target"] for w in weeks)
            payload["weekly"][carrier] = {
                "weeks": len(weeks),
                "gain_daily_target": (constant - daily) / constant,
                "gain_weekly_target": (constant - weekly) / constant,
            }
    output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"reading: {reading['reading']} (valid: {reading['valid']})")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

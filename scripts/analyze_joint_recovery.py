"""Analysis of the joint-recovery study (docs/31_joint_recovery.md).

Question: is the joint optimum of operating point and recovery worth more than "first the dryer,
then the recovery"?

Input: ``artifacts/runs/<name>.json.gz`` of ``scripts/run_joint_recovery.py`` and, for gate G1,
``artifacts/runs/surrogate_rto_v1.json.gz`` (Befund 107). Output:
``artifacts/runs/<name>_analysis.json`` (never overwritten) and a text summary.

PRE-REGISTERED READING (fixed in this docstring and committed before the main run)
===================================================================================

Metric. Per test state (the 50 test states of Befund 107; test set named "50 test states of
P1") the value of the joint optimisation ``V_J = (E(S1) - E(S2)) / E(S1)``, ``E`` the final
energy (fuel + fans + pump, kW) at the true model. Statistic: the median over the 50 states,
reported with the range (min, max).

Reading, per ``dT_min`` in {5, 10, 20} K, applied to each medium variant separately (six
verdicts; unpressurised is the main variant):

* **J0**: median ``V_J`` < 1 % -> optimising one after the other suffices.
* **J1**: 1 % <= median ``V_J`` <= 5 % -> the joint optimisation pays measurably.
* **J2**: median ``V_J`` > 5 % -> operating point and recovery must be optimised jointly.

Additionally (also fixed now):

(a) Recovery unused at S2: a state counts as "used" when the recovered heat at S2 exceeds
    ``thresholds.recovery_used_kw`` (1 W). Reported per combination: number of states with
    recovery used. "From which ``dT_min`` the recovery stays unused at the optimum S2": the
    smallest level of {5, 10, 20} K at which it is unused in all 50 states, and only if it is
    also unused at every larger level; otherwise "at none of the levels". Only these three
    levels are claimed; anything between them is descriptive (b').
(b) Condensation at S2: a state counts when the condensate in the recovery exchanger at S2
    exceeds ``thresholds.condensate_kg_per_h`` (1 mg/h); number of states per combination.
(b') Descriptive, not a reading: per state the geometric bound ``(T_exhaust(S0) - T_amb) / 2``
    above which no heat can pass at S0's operating point (the medium must be at least
    ``dT_min`` above the fresh gas and at least ``dT_min`` below the exhaust).

Gates (checked before any reading is stated):

G1 Reproduction. The reference of the combination 20 K / unpressurised (the truth of Befund
   107) reproduces the P1 reference of ``surrogate_rto_v1``: test states, ``u*`` per state and
   ``u0`` equal within 1e-9 (absolute, decision units), energies at ``u*`` within 1e-9 kW. The
   joint line differs from Befund 107's only by the bypass at a medium flow of 0, which ``u*``
   (medium in its box) never uses. If G1 fails: stop, report, no reading.
G2 Optimality. In every state of every combination ``E(S2) <= E(S1) <= E(S0)`` and
   ``E(S2) <= E(S3)``, each within the relative tolerance ``thresholds.optimality_rtol``
   (1e-6). Additionally reported, not part of G2: the same for the parts with the medium in its
   box (``S2_on <= S1_on``, ``S2_on <= S3_on``). If G2 fails anywhere: the failing states are
   listed, and the reading is stated twice, for S2 as found and for ``S2* = min(S2, S1, S3)``
   (the best known feasible point of the joint problem), both marked.
G3 Feasibility. Every strategy feasible (true constraints within their tolerances of the base
   config) in every state; failures are counted and reported, the reading stands on V_J as
   defined.

Expectation stated before the run (not a reading): the medium limit of 95 degC cannot bind on
this line, because the medium leaves the recovery exchanger at least ``dT_min`` below the dryer
exhaust, and the exhaust (= product) is limited to 80 degC; the two medium variants are
therefore expected to coincide. Checked and reported as the largest energy difference between
the variants per strategy and ``dT_min``, and as the highest medium temperature reached.

Descriptive figures (no reading): energy and CO2 per strategy (median, range); the sequential
value of the recovery ``(E(S0) - E(S1)) / E(S0)``; the price of the robust operating point
``(E(S3) - E(S2)) / E(S2)``; recovered heat, exhaust temperature before and after the recovery
exchanger, condensate, medium flow and bypass share per strategy; active constraints (``g /
scale >= -1e-3``) counted per strategy; ``V_J`` on CO2; share of states per class of ``V_J``.
Run times are wall-clock times measured under parallel load (other studies on the machine).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact  # noqa: E402

STRATEGIES = ("S0", "S1", "S2", "S3")
G1_TOL = 1e-9
J_LOW, J_HIGH = 0.01, 0.05


def reading(median_vj: float) -> str:
    """J0 / J1 / J2 for the median of ``V_J`` (boundaries 1 % and 5 % belong to J1)."""
    if median_vj < J_LOW:
        return "J0"
    if median_vj <= J_HIGH:
        return "J1"
    return "J2"


def _stats(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def _load(name: str) -> dict[str, Any]:
    with gzip.open(RUNS_DIR / f"{name}.json.gz", "rt", encoding="utf-8") as fh:
        out: dict[str, Any] = json.load(fh)
    return out


def gate_g1(run: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    ours = next(r for r in run["records"] if r["task"] == ["reference", 20.0, "unpressurised"])
    theirs = next(r for r in base["records"] if r["task"] == ["reference", "P1"])
    a, b = ours["result"], theirs["result"]
    d_w = float(np.max(np.abs(np.array(a["test_w"]) - np.array(b["test_w"]))))
    d_star = float(np.max(np.abs(np.array(a["u_star"]) - np.array(b["u_star"]))))
    d_u0 = float(np.max(np.abs(np.array(a["u0"]) - np.array(b["u0"]))))
    d_e = float(
        np.max(np.abs(np.array(a["score_star"]["energy"]) - np.array(b["score_star"]["energy"])))
    )
    return {
        "test_w_max_abs": d_w,
        "u_star_max_abs": d_star,
        "u0_max_abs": d_u0,
        "energy_star_max_abs_kw": d_e,
        "tolerance": G1_TOL,
        "passed": max(d_w, d_star, d_u0, d_e) <= G1_TOL,
    }


def combo_summary(combo: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    th = config["thresholds"]
    states = combo["states"]
    rtol = float(th["optimality_rtol"])

    def side(s: str, key: str) -> list[float]:
        return [st[s]["side"][key] for st in states]

    def energy(s: str) -> list[float]:
        return [st[s]["energy"] for st in states]

    out: dict[str, Any] = {"dt_min_k": combo["dt_min_k"], "medium": combo["medium"]}
    out["u0"] = combo["u0"]
    vj = [st["V_J"] for st in states]
    out["V_J"] = _stats(vj)
    out["reading"] = reading(out["V_J"]["median"])
    out["V_J_classes"] = dict(Counter(reading(v) for v in vj))
    e = {s: energy(s) for s in STRATEGIES}
    co2 = {s: side(s, "co2_kg_per_h") for s in STRATEGIES}
    out["energy_kw"] = {s: _stats(e[s]) for s in STRATEGIES}
    out["co2_kg_per_h"] = {s: _stats(co2[s]) for s in STRATEGIES}
    out["V_J_co2"] = _stats([(a - b) / a for a, b in zip(co2["S1"], co2["S2"], strict=True)])
    out["sequential_recovery_value"] = _stats(
        [(a - b) / a for a, b in zip(e["S0"], e["S1"], strict=True)]
    )
    out["robust_point_price"] = _stats([(a - b) / b for a, b in zip(e["S3"], e["S2"], strict=True)])
    per: dict[str, Any] = {}
    for s in STRATEGIES:
        rec = side(s, "recovered_kw")
        cond = side(s, "condensate_kg_per_h")
        hot = [v for v in side(s, "medium_hot_celsius") if np.isfinite(v)]
        per[s] = {
            "recovered_kw": _stats(rec),
            "recovery_used_states": int(sum(v > th["recovery_used_kw"] for v in rec)),
            "exhaust_before_celsius": _stats(side(s, "exhaust_celsius")),
            "exhaust_after_celsius": _stats(side(s, "exhaust_after_recovery_celsius")),
            "condensate_kg_per_h": _stats(cond),
            "condensation_states": int(sum(v > th["condensate_kg_per_h"] for v in cond)),
            "medium_kg_per_h": _stats(side(s, "medium_kg_per_h")),
            "bypass_states": int(sum(side(s, "bypass"))),
            "medium_hot_max_celsius": max(hot) if hot else None,
            "inlet_temperature_celsius": _stats([st[s]["u"][0] for st in states]),
            "air_kg_per_h": _stats([st[s]["u"][1] for st in states]),
            "active": dict(Counter(a for st in states for a in st[s]["active"])),
            "infeasible_states": int(sum(not st[s]["feasible"] for st in states)),
        }
    out["per_strategy"] = per
    fails = {
        k: [i for i, st in enumerate(states) if not st["checks"][k]] for k in states[0]["checks"]
    }
    out["checks_failed"] = {k: v for k, v in fails.items() if v}
    out["check_max_relative_excess"] = {
        "S2_over_S1": max((a - b) / b for a, b in zip(e["S2"], e["S1"], strict=True)),
        "S1_over_S0": max((a - b) / b for a, b in zip(e["S1"], e["S0"], strict=True)),
        "S2_over_S3": max((a - b) / b for a, b in zip(e["S2"], e["S3"], strict=True)),
    }
    g2 = [k for k in ("S1_le_S0", "S2_le_S1", "S2_le_S3") if fails[k]]
    out["G2_passed"] = not g2
    if g2:
        star = [min(a, b, c) for a, b, c in zip(e["S2"], e["S1"], e["S3"], strict=True)]
        vj_star = [(a - b) / a for a, b in zip(e["S1"], star, strict=True)]
        out["V_J_S2star"] = _stats(vj_star)
        out["reading_S2star"] = reading(out["V_J_S2star"]["median"])
    out["G3_infeasible"] = {s: per[s]["infeasible_states"] for s in STRATEGIES}
    out["geometric_bound_k"] = _stats(
        [(st["S0"]["side"]["exhaust_celsius"] - st["w"][0]) / 2.0 for st in states]
    )
    out["recovery_unused_all_states_S2"] = per["S2"]["recovery_used_states"] == 0
    out["optimality_rtol"] = rtol
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="joint_recovery_v1")
    ap.add_argument("--base", default="surrogate_rto_v1")
    ap.add_argument("--out", default=None, help="artifact name (default <name>_analysis)")
    args = ap.parse_args()
    require_clean_tree()
    run, base = _load(args.name), _load(args.base)
    config = run["config"]
    g1 = gate_g1(run, base)
    combos = {k: combo_summary(v, config) for k, v in run["table"].items()}
    dts = sorted({c["dt_min_k"] for c in combos.values()})
    media = list(config["factors"]["media"])
    unused_from: dict[str, Any] = {}
    for m in media:
        unused = [combos[f"{dt:g}K/{m}"]["recovery_unused_all_states_S2"] for dt in dts]
        level = next((dt for k, dt in enumerate(dts) if all(unused[k:])), None)
        unused_from[m] = level

    def energies(key: str, s: str) -> np.ndarray:
        return np.array([st[s]["energy"] for st in run["table"][key]["states"]])

    variant_diff = {
        f"{dt:g}K": {
            s: float(
                np.max(
                    np.abs(energies(f"{dt:g}K/{media[0]}", s) - energies(f"{dt:g}K/{media[1]}", s))
                )
            )
            for s in STRATEGIES
        }
        for dt in dts
    }
    payload = {
        "_provenance": provenance(),
        "input": {
            "run": f"artifacts/runs/{args.name}.json.gz",
            "run_commit": run["provenance"].get("git/commit"),
            "run_dirty": run["provenance"].get("git/dirty"),
            "base": f"artifacts/runs/{args.base}.json.gz",
        },
        "test_set": "50 test states of P1 (Befund 107)",
        "G1": g1,
        "G2_passed": all(c["G2_passed"] for c in combos.values()),
        "G3_infeasible_total": sum(sum(c["G3_infeasible"].values()) for c in combos.values()),
        "readings": {k: c["reading"] for k, c in combos.items()} if g1["passed"] else None,
        "recovery_unused_from_dt_min_k": unused_from,
        "condensation_states_S2": {
            k: c["per_strategy"]["S2"]["condensation_states"] for k, c in combos.items()
        },
        "medium_variant_max_energy_difference_kw": variant_diff,
        "timing": run["timing"],
        "combos": combos,
    }
    print(json.dumps({k: payload[k] for k in ("G1", "G2_passed", "G3_infeasible_total",
                                             "readings", "recovery_unused_from_dt_min_k",
                                             "condensation_states_S2")}, indent=1))  # fmt: skip
    for k, c in combos.items():
        print(
            f"{k:22s} V_J median {c['V_J']['median']:+.3e} [{c['V_J']['min']:+.3e}, "
            f"{c['V_J']['max']:+.3e}] -> {c['reading']}; rec used S1/S2 "
            f"{c['per_strategy']['S1']['recovery_used_states']}/"
            f"{c['per_strategy']['S2']['recovery_used_states']}"
        )
    write_artifact(args.out or f"{args.name}_analysis", payload)


if __name__ == "__main__":
    main()

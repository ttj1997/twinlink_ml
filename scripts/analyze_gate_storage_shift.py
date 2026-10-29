r"""POST-HOC analysis of the S2 gate (``gate_storage_shift.json``). Declared post-hoc: written AFTER the
gate result was seen; it changes no pre-registered number and no verdict of the gate.

Why it exists. The gate result showed (i) the decomposition puts the whole saving on "extra heat
used" and none on exchanger use or timing, (ii) the gap between rule R1 and optimum R* appears on
product 2 only, where R* runs the preheater loop at about half of the flow R1 uses. Three questions,
fixed here before this script is run:

1. R1b, the plainest rule: logged preheater-loop flow, plus request all discarded compressor heat
   (taken as far as it fits below T_S,max). Does the gap R1b - R* stay below the 2 % of the gate on
   the whole grid? (If yes, the gap of R1 is a property of the max-flow rule, not of the problem.)
   The gate rule is re-evaluated with R1b in place of R1 and labelled post-hoc.
2. Duty against flow: Q_out as a function of the preheater-loop flow V_2 at fixed storage
   temperatures (50 .. 90 degC) for the fitted conductances of each day, mean inputs of the day.
3. Energy accounting: saving of R* (kWh steam) divided by the extra heat taken (kWh) on the cases
   with discarded heat, to show how much of the saving is only the added energy.

Call (needs the plant exports like the gate)::

    TWINLINK_PLANT_DATA_P1=... TWINLINK_PLANT_DATA_P2=... \
        tools/run_here.sh scripts/analyze_gate_storage_shift.py --name gate_storage_shift_posthoc
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "verify"))
from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact  # noqa: E402

_spec = importlib.util.spec_from_file_location("gate_storage_shift", HERE / "gate_storage_shift.py")
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

GATE_ARTIFACT = RUNS_DIR / "gate_storage_shift.json"


def logged_flow_policy(problem: Any) -> Any:
    """R1b: logged flow, request all discarded heat (the step curtails what does not fit)."""
    return lambda k, _t: (float(problem.v2_logged[k]), float(problem.q_dump_kw[k]))


def duty_against_flow(day: Any, cfg: Any) -> dict[str, Any]:
    """``Q_out`` for V_2 = 5 .. 40 m3/h at fixed storage temperatures, mean inputs of the day."""
    from twinlink_ml.storage_a1.model import ExtractionChain

    chain = ExtractionChain.from_config(cfg)
    flows = np.array([5.0, 10.0, 15.0, 18.5, 25.0, 30.0, 40.0])
    mean = {r: float(np.mean(day[r])) for r in day.roles}
    out: dict[str, Any] = {"v2_m3_per_h": flows.tolist(), "q_out_kw_by_storage_temperature": {}}
    for t_s in (50.0, 60.0, 70.0, 80.0, 90.0):
        res = chain.solve_given_q_in(
            np.full_like(flows, t_s),
            np.zeros_like(flows),
            np.full_like(flows, mean["storage_loop_flow"]),
            flows,
            np.full_like(flows, mean["process_air_flow"]),
            np.full_like(flows, mean["ambient_temperature"]),
        )
        out["q_out_kw_by_storage_temperature"][f"{t_s:g}"] = [
            round(float(v), 1) for v in res.q_out_kw
        ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    require_clean_tree()
    gate_data = json.loads(GATE_ARTIFACT.read_text(encoding="utf-8"))
    role_map = gate.load_role_map(gate.ROLE_FILE)
    days = {}
    for label, var in gate.PRODUCT_ENV.items():
        day, _ = gate.repair_glitches(gate.load_day(os.environ[var], role_map, label))
        days[label] = day
    base = gate.StorageSystemConfig()
    fitted = {
        k: gate.fit_conductances(gate.with_volume(base, gate.V0_M3), d).config
        for k, d in days.items()
    }
    if any(c.fingerprint != gate_data["config/fingerprint_fitted"][k] for k, c in fitted.items()):
        raise SystemExit("fitted configs differ from the gate artifact")

    by_key = {
        (c["product_day"], c["ambient_shift_k"], c["volume_m3"], c["dump_scenario"]): c
        for c in gate_data["cases"]
    }
    rows: list[dict[str, Any]] = []
    for (label, shift, volume, dump), c in by_key.items():
        problem = gate.make_problem(days[label], fitted[label], volume, dump, shift)
        r1b = gate.simulate(problem, logged_flow_policy(problem))
        ref = c["t_end_celsius"]["R0"]
        e_b = gate.corrected_steam_kwh(r1b, ref)
        e = c["steam_kwh_per_day"]
        extra = c["extra_heat_taken_kwh"]["R*"]
        rows.append(
            {
                "product_day": label,
                "ambient_shift_k": shift,
                "volume_m3": volume,
                "dump_scenario": dump,
                "mean_dump_kw": c["mean_dump_kw"],
                "saving_fraction_R*": c["saving_fraction_R*"],
                "gap_fraction_R1b_minus_R*": (e_b - e["R*"]) / e["R0"],
                "gap_fraction_R1_minus_R*": c["gap_fraction_R1_minus_R*"],
                "steam_kwh_R1b": e_b,
                "feasible_R1b": r1b.feasible,
                "saving_over_extra_heat_R*": ((e["R0"] - e["R*"]) / extra) if extra > 1.0 else None,
            }
        )

    def verdict(subset: list[dict[str, Any]]) -> dict[str, Any]:
        real = [
            r
            for r in subset
            if r["mean_dump_kw"] > 0 and r["volume_m3"] <= gate.REALISTIC_VOLUME_MAX_M3
        ]
        hit = [r for r in real if r["saving_fraction_R*"] >= gate.GATE_G]
        both = [r for r in hit if r["gap_fraction_R1b_minus_R*"] >= gate.GATE_GAP]
        return {
            "reading_with_R1b": "L1" if both else ("L2" if hit else "L3"),
            "n_realistic": len(real),
            "n_saving_at_least_5_percent": len(hit),
            "n_saving_and_gap_R1b": len(both),
            "max_gap_R1b": max((r["gap_fraction_R1b_minus_R*"] for r in real), default=None),
        }

    ratios = [
        r["saving_over_extra_heat_R*"] for r in rows if r["saving_over_extra_heat_R*"] is not None
    ]
    payload: dict[str, Any] = {
        "_provenance": provenance(),
        "status": "POST-HOC analysis of gate_storage_shift.json; changes no pre-registered number",
        "gate_artifact_sha256": hashlib.sha256(GATE_ARTIFACT.read_bytes()).hexdigest(),
        "cases": rows,
        "gate_with_R1b_post_hoc": {
            "all_products": verdict(rows),
            **{label: verdict([r for r in rows if r["product_day"] == label]) for label in days},
        },
        "saving_over_extra_heat_R*": {
            "min": float(np.min(ratios)),
            "median": float(np.median(ratios)),
            "max": float(np.max(ratios)),
            "n": len(ratios),
        },
        "duty_against_flow": {
            label: duty_against_flow(days[label], fitted[label]) for label in days
        },
    }
    write_artifact(args.name, payload)


if __name__ == "__main__":
    main()

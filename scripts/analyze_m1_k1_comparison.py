r"""M1 after the correction K1 (Befund 79) against Befund 69: figures, binding bounds, set points. Pure evaluation.

Inputs (committed artifacts): Befund 69 -- ``m1_margin_cost_l015`` (0.80 m/s), ``m1_margin_cost_l015_umin126``
(1.26 m/s), ``m1_gas_bound_sensitivity``; re-run -- ``m1_margin_cost_l015_umin080_k1``,
``m1_margin_cost_l015_umin126_k1``, ``m1_gas_bound_sensitivity_k1`` (``run_m1_margin_cost.py`` and
``analyze_m1_gas_bound_sensitivity.py`` with the suffix ``_k1``; every setting of Befund 69 unchanged: line A,
gas heater, ``eta_gas`` 0.80, quality limit 0.15, margin m = 0.005961819615175404 kg/kg, 25 grid points of Befund
52, optimiser budget of Befund 52). The only difference is the solver (Befund 79, K1: fallback bracket
``[T_amb + 0.5, T_amb + 10]``); M1 has no heat network, so the correction A1 does not enter.

The two runs are kept apart in the artifact (``befund_69`` / ``rerun_k1``); no figure of one is combined with
a figure of the other. Reported per bound and arm: the stability figures and verdicts of the rule of Befund 69
(applied unchanged by ``analyze_m1_gas_bound_sensitivity.py``), the fixed point, and how many solutions lie on
the old solver bound ``T_amb + 10 K`` (within 1e-3 K) or below it (the region the old solver could not reach),
on the inlet-temperature box minimum 60 degC, on the lower gas-velocity bound, and bind the moisture limit.

    python scripts/analyze_m1_k1_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import (
    NO_HEAT_NETWORK,
    RUNS_DIR,
    provenance,
    require_clean_tree,
    write_artifact,
)

RUNS = {
    "befund_69": {
        "0.80": "m1_margin_cost_l015",
        "1.26": "m1_margin_cost_l015_umin126",
        "sensitivity": "m1_gas_bound_sensitivity",
    },
    "rerun_k1": {
        "0.80": "m1_margin_cost_l015_umin080_k1",
        "1.26": "m1_margin_cost_l015_umin126_k1",
        "sensitivity": "m1_gas_bound_sensitivity_k1",
    },
}
OLD_MARGIN_K = 10.0
ON_TOLERANCE_K = 1.0e-3
BOX_MINIMUM_CELSIUS = 60.0
AT_BOUND_TOLERANCE_M_PER_S = 1.0e-3


def _bounds(rows: list[dict[str, Any]], gas_bound: float) -> dict[str, int]:
    valid = [r for r in rows if not r.get("solve_failed")]
    gap = [
        r["bed_temperature_celsius"] - r["ambient_temperature_celsius"] - OLD_MARGIN_K
        for r in valid
    ]
    return {
        "n": len(valid),
        "n_on_old_solver_bound": sum(abs(g) <= ON_TOLERANCE_K for g in gap),
        "n_below_old_solver_bound": sum(g < -ON_TOLERANCE_K for g in gap),
        "n_binding_moisture": sum(bool(r["constraint_check"]["binding_moisture"]) for r in valid),
        "n_on_inlet_box_minimum": sum(
            abs(r.get("inlet_temperature_celsius", float("nan")) - BOX_MINIMUM_CELSIUS) < 1e-3
            for r in valid
        ),
        "n_on_lower_gas_bound": sum(
            abs(r.get("gas_velocity_m_per_s", float("nan")) - gas_bound)
            < AT_BOUND_TOLERANCE_M_PER_S
            for r in valid
        ),
    }


def main() -> int:
    require_clean_tree()
    out: dict[str, Any] = {}
    for run, names in RUNS.items():
        sensitivity = json.loads((RUNS_DIR / f"{names['sensitivity']}.json").read_text("utf-8"))
        block: dict[str, Any] = {
            "artifacts": names,
            "sensitivity_figures": sensitivity["figures"],
            "statements": {s: v["stable"] for s, v in sensitivity["statements"].items()},
            "statement_figures": {s: v["figures"] for s, v in sensitivity["statements"].items()},
            "overall": sensitivity["overall"],
        }
        for bound in ("0.80", "1.26"):
            artifact = json.loads((RUNS_DIR / f"{names[bound]}.json").read_text("utf-8"))
            gas_bound = float(artifact.get("gas_velocity_lower_bound_m_per_s", 0.8))
            fixed = artifact["arm_F_fixed_point"]
            fixed_rows = [
                {
                    **r,
                    "inlet_temperature_celsius": fixed["inlet_temperature_celsius"],
                    "gas_velocity_m_per_s": fixed["gas_velocity_m_per_s"],
                }
                for r in fixed["per_grid_point"]
            ]
            block[bound] = {
                "fixed_point": {
                    k: fixed[k]
                    for k in (
                        "inlet_temperature_celsius",
                        "gas_velocity_m_per_s",
                        "nozzle_pressure_bar",
                    )
                },
                "fixed_point_feasible_everywhere": artifact["fixed_point_feasible_everywhere"],
                "bounds": {
                    "M0": _bounds(artifact["arm_M0_oracle"], gas_bound),
                    "M": _bounds(artifact["arm_M_oracle"], gas_bound),
                    "F": _bounds(fixed_rows, gas_bound),
                },
                "m0_set_points": [
                    [
                        r["inlet_temperature_celsius"],
                        r["gas_velocity_m_per_s"],
                        r["nozzle_pressure_bar"],
                    ]
                    for r in artifact["arm_M0_oracle"]
                ],
                "m0_co2_kg_per_h": [
                    r["gas_heater"]["co2_kg_per_h"] for r in artifact["arm_M0_oracle"]
                ],
            }
        out[run] = block
    write_artifact(
        "m1_k1_comparison",
        {
            "_provenance": provenance(NO_HEAT_NETWORK),
            "test_set": "25 Befund-52 grid points; line A, gas heater, quality limit 0.15, margin m",
            "note": "the two runs are kept apart; no figure of one is combined with a figure of the other",
            **out,
        },
    )
    for run, block in out.items():
        print(run, block["statements"], block["overall"])
        for bound in ("0.80", "1.26"):
            print("  ", bound, block[bound]["fixed_point"], json.dumps(block[bound]["bounds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

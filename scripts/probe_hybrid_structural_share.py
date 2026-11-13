r"""Fix the freeboard share of the structurally deviating truth (docs/NN_hybrid_structural.md).

Run BEFORE any data, training or margin run of the study; the result is written into
``conf/hybrid_structural/hybrid_structural.yaml`` (``structure.share``) by hand and checked by a
test against this artifact.

Rule (committed in the config before this script ran): among ``structure.share_candidates`` the
share whose bed-to-exhaust gap at the NOMINAL POINT is closest to ``structure.target_gap_k`` (7.5 K,
the middle of the 5-10 K of Befund 117). Nominal point: the nominal commands and disturbances of
the process config (inlet 90 degC, gas velocity 1.6 m/s, spray 1.5 kg/h, ambient 20 degC / 0.50,
solid fraction 0.30), nominal material parameters. The gap is that of the truth itself: steady
state of the bed with the bed feed (``hybrid_structural.truth.steady_config``, exact scalar
solver), exhaust temperature after the evaporation of the freeboard water
(``exhaust_temperature_celsius``, the balance of ``plant_scale.model.exhaust_after_freeboard``).

Reported next to the choice, without any rule: for every candidate the gap, the steady moisture
and bed temperature and their shift against the model without the term, at the nominal parameters
and over the 20 draws of Befund 111 (min / median / max); the share of pump and atomiser power in
the total power (side effect of the emulation by the inputs).

    tools/run_here.sh scripts/probe_hybrid_structural_share.py --name hybrid_structural_share_v1
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

ROOT = SCRIPTS.parent


def steady_point(config: Any, share: float) -> dict[str, Any]:
    """Steady state of the truth with ``share`` at the nominal point of ``config``, and the gap."""
    from twinlink_ml.hybrid_structural.truth import (
        bed_feed,
        exhaust_temperature_celsius,
        steady_config,
    )
    from twinlink_ml.simulation import GreyBoxFluidBedGranulator, PlaceholderConfigWarning
    from twinlink_ml.simulation.steady_state import solve_steady_state

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        solution = solve_steady_state(GreyBoxFluidBedGranulator(steady_config(config, share)))
    diag = solution.diagnostics
    m_air = float(solution.dry_air_mass_flow_kg_per_s)
    _, _, water_kg_h = bed_feed(
        config.actions.spray_rate_kg_per_h.nominal,
        config.disturbances.suspension_solid_fraction.nominal,
        share,
    )
    # humidity of the gas leaving the bed that closes the bed's water balance
    y_bed = (
        float(diag["inlet_humidity_kg_per_kg"]) + float(diag["evaporation_rate_kg_per_s"]) / m_air
    )
    t_bed = float(solution.bed_temperature_celsius)
    t_exh = exhaust_temperature_celsius(
        config,
        bed_temperature_celsius=t_bed,
        dry_air_flow_kg_per_s=m_air,
        outlet_humidity_kg_per_kg=y_bed,
        freeboard_water_kg_per_s=float(water_kg_h) / 3600.0,
    )
    return {
        "share": share,
        "feasible": bool(solution.feasible),
        "infeasibility_reasons": list(solution.infeasibility_reasons),
        "moisture_kg_per_kg": float(solution.particle_moisture_kg_per_kg),
        "bed_temperature_celsius": t_bed,
        "exhaust_temperature_celsius": t_exh,
        "gap_k": t_bed - t_exh,
        "energy_target": float(diag["energy_target"]),
        "power_total_kw": float(diag["power_total_kw"]),
        "power_pump_plus_atomiser_share": (
            float(diag["power_pump_kw"]) + float(diag["power_atomiser_kw"])
        )
        / float(diag["power_total_kw"]),
        "freeboard_water_kg_per_h": float(water_kg_h),
    }


def spread(values: list[float]) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    from _common import provenance, require_clean_tree, write_artifact

    from twinlink_ml.hybrid_model.study import build_setup
    from twinlink_ml.hybrid_structural.study import load_structural_config
    from twinlink_ml.utils.code_location import require_code_location

    require_code_location(ROOT)
    require_clean_tree()
    started = time.time()
    config = load_structural_config()
    structure = config.raw["structure"]
    setup = build_setup(config)  # the set-up of Befund 122: configs and the 20 draws
    candidates = [float(v) for v in structure["share_candidates"]]
    target = float(structure["target_gap_k"])
    low, high = (float(v) for v in structure["accepted_gap_k"])

    nominal = setup.steady_nominal
    reference = steady_point(nominal, 0.0)
    truths = [setup.steady_truth(d) for d in range(len(setup.draws))]  # draws, without the term
    references = [steady_point(c, 0.0) for c in truths]
    rows = []
    for share in candidates:
        point = steady_point(nominal, share)
        draws = []
        for d in range(len(setup.draws)):
            with_term, without = steady_point(truths[d], share), references[d]
            draws.append(
                {
                    "draw": d,
                    "feasible": with_term["feasible"],
                    "gap_k": with_term["gap_k"],
                    "moisture_kg_per_kg": with_term["moisture_kg_per_kg"],
                    "moisture_shift_kg_per_kg": with_term["moisture_kg_per_kg"]
                    - without["moisture_kg_per_kg"],
                    "bed_temperature_shift_k": with_term["bed_temperature_celsius"]
                    - without["bed_temperature_celsius"],
                }
            )
        rows.append(
            {
                **point,
                "moisture_shift_kg_per_kg": point["moisture_kg_per_kg"]
                - reference["moisture_kg_per_kg"],
                "bed_temperature_shift_k": point["bed_temperature_celsius"]
                - reference["bed_temperature_celsius"],
                "energy_target_shift_relative": point["energy_target"] / reference["energy_target"]
                - 1.0,
                "draws": {
                    "n_feasible": sum(r["feasible"] for r in draws),
                    "gap_k": spread([r["gap_k"] for r in draws]),
                    "moisture_kg_per_kg": spread([r["moisture_kg_per_kg"] for r in draws]),
                    "moisture_shift_kg_per_kg": spread(
                        [r["moisture_shift_kg_per_kg"] for r in draws]
                    ),
                    "bed_temperature_shift_k": spread(
                        [r["bed_temperature_shift_k"] for r in draws]
                    ),
                    "per_draw": draws,
                },
            }
        )
        print(
            f"share {share:.2f}: gap {point['gap_k']:.2f} K, X {point['moisture_kg_per_kg']:.4f} "
            f"({rows[-1]['moisture_shift_kg_per_kg']:+.4f}), T_bed "
            f"{point['bed_temperature_celsius']:.2f} degC, feasible {point['feasible']}; draws gap "
            f"{rows[-1]['draws']['gap_k']['min']:.2f}-{rows[-1]['draws']['gap_k']['max']:.2f} K",
            flush=True,
        )
    feasible = [r for r in rows if r["feasible"]]
    selected = min(feasible, key=lambda r: (abs(r["gap_k"] - target), r["share"]))
    within = low <= selected["gap_k"] <= high
    payload = {
        "_provenance": provenance("not used"),
        "study": "hybrid against GRU against refit, structurally deviating truth",
        "rule": {
            "candidates": candidates,
            "target_gap_k": target,
            "accepted_gap_k": [low, high],
            "point": "nominal commands and disturbances, nominal material parameters",
        },
        "selected_share": selected["share"],
        "selected_gap_k": selected["gap_k"],
        "selected_within_accepted_band": bool(within),
        "reference_without_term": reference,
        "candidates": rows,
        "runtime_s": time.time() - started,
    }
    write_artifact(args.name, payload)
    print(
        f"selected share {selected['share']:.2f}: gap {selected['gap_k']:.2f} K at the nominal "
        f"point (within {low}-{high} K: {within})"
    )
    return 0 if within else 1


if __name__ == "__main__":
    raise SystemExit(main())

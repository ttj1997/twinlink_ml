r"""Analysis of the observability study (docs/33_observability.md), reading fixed before the run.

Input: ``artifacts/runs/<name>.json.gz`` of ``scripts/run_observability.py``. Output:
``artifacts/runs/<name>_analysis.json`` (never overwritten) and a summary on stdout.

Hypothesis under test (project lead, Befund 112, decision paragraph (3)): in quasi-stationary
continuous operation exhaust humidity, exhaust temperature and bed temperature carry hardly any
information on the bed moisture X, because they are fixed by the water feed; X acts only through
the water discharged with the product, Delta Y_out ~ Delta X * m_solid / m_air.

Quantities (steady states; per case c in {lab, plant-like 0.5 t/h, plant-like 1.0 t/h, the latter
two "plant-like, placeholder"} and operating point P)
--------------------------------------------------------------------------------------------
Over the 20 draws of Befund 111 at identical commands and disturbances: the moisture X_d and the
measurements m_d. Measurement variants:

* ``Y_out`` -- exhaust humidity, the model's diagnostic (damped Picard iterate; the quantity
  measured in Befund 112), sensor sigma ``relative_noise_std * Y``;
* ``Y_out_balance`` -- exhaust humidity that closes the water balance, ``Y_in + evaporation /
  m_air``; same sigma;
* ``T_out`` -- exhaust temperature (= bed temperature minus the duct loss, 0.0 here), sigma
  ``sqrt((relative_noise_std * T)^2 + absolute_noise_std_temperature_k^2)``, one sensor;
* ``T_bed`` -- bed temperature, same sigma, one sensor;
* ``T_bed_x6`` -- bed temperature with sigma / sqrt(6) (six redundant sensors as at the plant).

Sensitivity dm/dX = slope of the OLS fit of m_d on X_d over the draws with a solution (feasible or
not; the number of infeasible and failed draws is reported, and the fit over feasible draws only
if they differ), with R^2. Sigma at the median of m_d over the draws (the existing sensor model,
``MeasurementConfig`` of the process config). **SNR = |dm/dX| * 0.01 kg/kg / sigma** (one sample).

Analytic check (lab and plant-like, every point): the hypothesis' slope -m_s,in / m_air at the
nominal model's steady state of that point against the simulated slope of ``Y_out_balance``
(main) and of ``Y_out``; ratio simulated / analytic. "Agrees" if |ratio - 1| <= 0.05 (config
``analytic_tolerance``) at every point. Reported with it: the linearised slopes with the density
term of the dry-air flow (Y) and of the energy balance (T), see
:mod:`twinlink_ml.observability.core`.

Pre-registered reading (lab line, steady state; threshold SNR 1, inclusive for O2)
--------------------------------------------------------------------------------
* **O1** if SNR < 1 for EVERY measurement variant (``Y_out``, ``Y_out_balance``, ``T_out``,
  ``T_bed``, ``T_bed_x6``) at EVERY operating point P1-P7: "the hypothesis explains the failure of
  F5 (Befund 112); a soft sensor needs transients, batch operation or a direct moisture
  measurement."
* **O2** otherwise (SNR >= 1 for at least one variant at at least one point): "observability is
  given; the failure of F5 lies in training or architecture; next step would be a model-based
  observer (docs/12 section I, item 19)". Reported with O2: every (variant, point) with SNR >= 1,
  the number of points per variant, and two qualifiers, each stated if it applies:
  "O2 only through the six-sensor redundancy" (only ``T_bed_x6`` reaches 1) and "O2 only through
  the water-balance humidity" (only ``Y_out_balance`` reaches 1).

Decisions taken without asking (fixed here, before the run): T_bed and both humidity variants are
counted among the measurements of the reading (the request lists Y_out, T_out and T_bed per
measured quantity and T_bed additionally with sigma / sqrt(6)); O1 needs SNR < 1 at every point,
so O2 is "at least one point".

Reported without reading
------------------------
* the same SNR table for the plant-like placeholder cases (0.5 and 1.0 t/h dry solid per
  30 000 Nm3/h air; only the solid-to-air ratio scaled, see the driver);
* dynamics (P3, slow/median/fast kinetics, steps of spray rate and inlet temperature +-10 %): per
  step and variant the largest distance ratio over the three pairs of draws and the longest time
  above 2 sigma, absolute and of the deviations from each draw's unstepped reference;
* context (not part of the reading): the steady shifts of the nominal model for the input
  amplitudes of the "observed" scenarios, next to the signal |dm/dX| * 0.01.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import RUNS_DIR  # noqa: E402

from twinlink_ml.headroom.config import load_headroom_config, study_process_config  # noqa: E402
from twinlink_ml.observability import (  # noqa: E402
    bed_temperature_shift,
    linear_fit,
    operating_point_config,
    sensor_sigma,
    snr,
)
from twinlink_ml.observability.core import SteadyMeasurements  # noqa: E402
from twinlink_ml.sensor_value import load_sensor_value_config  # noqa: E402
from twinlink_ml.simulation import MeasurementConfig, PlaceholderConfigWarning  # noqa: E402

THRESHOLD = 1.0
KELVIN = 273.15
VARIANTS = {
    # name: (steady key, temperature channel, redundancy key)
    "Y_out": ("outlet_humidity_kg_per_kg", False, False),
    "Y_out_balance": ("outlet_humidity_balance_kg_per_kg", False, False),
    "T_out": ("exhaust_temperature_celsius", True, False),
    "T_bed": ("bed_temperature_celsius", True, False),
    "T_bed_x6": ("bed_temperature_celsius", True, True),
}


def _steady(d: dict[str, Any]) -> SteadyMeasurements:
    fields = {k: v for k, v in d.items() if k not in ("draw", "state", "infeasibility_reasons")}
    return SteadyMeasurements(
        **fields,
        infeasibility_reasons=tuple(d["infeasibility_reasons"]),
        state=np.asarray(d["state"]),
    )


def snr_block(
    block: dict[str, Any],
    measurement: MeasurementConfig,
    delta_x: float,
    redundancy: int,
    config: Any,
    tolerance: float,
) -> dict[str, Any]:
    """Fits, sigmas, SNRs and the analytic check for one (case, point)."""
    solved = [d for d in block["draws"] if "failure" not in d]
    feasible = [d for d in solved if d["feasible"]]
    x = np.array([d["moisture_kg_per_kg"] for d in solved])
    nominal = _steady(block["nominal"])
    out: dict[str, Any] = {
        "n_draws": len(block["draws"]),
        "n_failed": len(block["draws"]) - len(solved),
        "n_infeasible": len(solved) - len(feasible),
        "infeasibility_reasons": sorted({r for d in solved for r in d["infeasibility_reasons"]}),
        "moisture_range_kg_per_kg": [float(x.min()), float(x.max())],
        "moisture_spread_kg_per_kg": float(np.ptp(x)),
        "nominal_moisture_kg_per_kg": nominal.moisture_kg_per_kg,
        "solid_to_air_ratio": nominal.solid_inflow_kg_per_s / nominal.dry_air_flow_kg_per_s,
        "variants": {},
    }
    for name, (key, temperature, redundant) in VARIANTS.items():
        y = np.array([d[key] for d in solved])
        fit = linear_fit(x, y)
        sigma = float(sensor_sigma(float(np.median(y)), measurement, temperature=temperature))
        n = redundancy if redundant else 1
        entry = {
            "fit": fit.as_dict(),
            "median_value": float(np.median(y)),
            "sigma_one_sensor": sigma,
            "redundancy": n,
            "signal_for_delta_x": abs(fit.slope) * delta_x,
            "snr": snr(fit.slope, delta_x, sigma, n),
        }
        if len(feasible) != len(solved) and len(feasible) >= 2:
            xf = np.array([d["moisture_kg_per_kg"] for d in feasible])
            yf = np.array([d[key] for d in feasible])
            if np.ptp(xf) > 0:
                fit_f = linear_fit(xf, yf)
                entry["fit_feasible_only"] = fit_f.as_dict()
                entry["snr_feasible_only"] = snr(fit_f.slope, delta_x, sigma, n)
        out["variants"][name] = entry
    hypothesis = -nominal.solid_inflow_kg_per_s / nominal.dry_air_flow_kg_per_s
    t_slope = bed_temperature_shift(1.0, nominal, config)
    pickup = nominal.outlet_humidity_balance_kg_per_kg - nominal.inlet_humidity_kg_per_kg
    linearised_y = hypothesis + pickup * t_slope / (nominal.bed_temperature_celsius + KELVIN)
    sim_bal = out["variants"]["Y_out_balance"]["fit"]["slope"]
    sim_diag = out["variants"]["Y_out"]["fit"]["slope"]
    sim_t = out["variants"]["T_bed"]["fit"]["slope"]
    out["analytic"] = {
        "hypothesis_slope_Y": hypothesis,
        "ratio_balance_to_hypothesis": sim_bal / hypothesis,
        "ratio_diagnostic_to_hypothesis": sim_diag / hypothesis,
        "agrees_balance": abs(sim_bal / hypothesis - 1.0) <= tolerance,
        "linearised_slope_Y": linearised_y,
        "ratio_balance_to_linearised": sim_bal / linearised_y,
        "linearised_slope_T_bed": t_slope,
        "ratio_T_bed_to_linearised": sim_t / t_slope,
    }
    return out


def point_config(nominal: Any, point: dict[str, Any], scaling: dict[str, Any] | None) -> Any:
    kwargs = {
        "inlet_temperature_celsius": point["inlet_temperature_celsius"],
        "gas_velocity_m_per_s": point["gas_velocity_m_per_s"],
    }
    if scaling is not None:
        kwargs["spray_rate_kg_per_h"] = scaling["spray_rate_kg_per_h"]
        kwargs["solid_fraction"] = scaling["solid_fraction"]
    return operating_point_config(nominal, **kwargs)


def reading(lab: dict[str, Any]) -> dict[str, Any]:
    hits = [
        {"point": p, "variant": v, "snr": e["snr"]}
        for p, block in lab.items()
        for v, e in block["variants"].items()
        if e["snr"] >= THRESHOLD
    ]
    max_snr = {v: max(block["variants"][v]["snr"] for block in lab.values()) for v in VARIANTS}
    if not hits:
        return {
            "reading": "O1",
            "text": "the hypothesis explains the failure of F5 (Befund 112); a soft sensor needs "
            "transients, batch operation or a direct moisture measurement",
            "hits": [],
            "max_snr_per_variant": max_snr,
            "qualifiers": [],
        }
    variants = {h["variant"] for h in hits}
    qualifiers = []
    if variants == {"T_bed_x6"}:
        qualifiers.append("O2 only through the six-sensor redundancy")
    if variants == {"Y_out_balance"}:
        qualifiers.append("O2 only through the water-balance humidity")
    return {
        "reading": "O2",
        "text": "observability is given; the failure of F5 lies in training or architecture; next "
        "step would be a model-based observer (docs/12 section I, item 19)",
        "hits": hits,
        "points_per_variant": {v: sum(h["variant"] == v for h in hits) for v in sorted(variants)},
        "max_snr_per_variant": max_snr,
        "qualifiers": qualifiers,
    }


def dynamics_summary(dyn: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for step, per_step in dyn["separation"].items():
        out[step] = {}
        for variant, pairs in per_step.items():
            entry = {}
            for kind in ("absolute", "deviation"):
                best = max(pairs.items(), key=lambda kv: kv[1][kind]["max_ratio"])
                longest = max(pairs.items(), key=lambda kv: kv[1][kind]["time_above_s"])
                entry[kind] = {
                    "max_ratio": best[1][kind]["max_ratio"],
                    "max_ratio_pair": best[0],
                    "time_of_max_ratio_h": best[1][kind]["time_of_max_ratio_s"] / 3600.0,
                    "longest_time_above_h": longest[1][kind]["time_above_s"] / 3600.0,
                    "longest_time_above_pair": longest[0],
                }
            out[step][variant] = entry
    return out


def context_block(lab_raw: dict[str, Any], lab: dict[str, Any], delta_x: float) -> dict[str, Any]:
    out = {}
    for p, block in lab_raw.items():
        signals = {
            v: lab[p]["variants"][v]["signal_for_delta_x"]
            for v in ("Y_out", "Y_out_balance", "T_bed")
        }
        out[p] = {
            "signal_for_delta_x": signals,
            "delta_x": delta_x,
            "input_shifts": block["input_context"],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", default="observability_v1")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    source = Path(args.input) if args.input else RUNS_DIR / f"{args.name}.json.gz"
    output = Path(args.output) if args.output else RUNS_DIR / f"{args.name}_analysis.json"
    if output.exists():
        raise SystemExit(f"{output} exists; artifacts are never overwritten")
    with gzip.open(source, "rt", encoding="utf-8") as f:
        record = json.load(f)
    cfg = record["config"]
    delta_x = float(cfg["snr"]["delta_x_kg_per_kg"])
    redundancy = int(cfg["snr"]["redundant_bed_sensors"])
    tolerance = float(cfg["snr"]["analytic_tolerance"])
    measurement = MeasurementConfig(**record["measurement_config"])
    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    sv = load_sensor_value_config(Path(record["sensor_value_config"]).resolve())
    nominal = study_process_config(load_headroom_config(sv.headroom_config))
    points = {p["name"]: p for p in record["operating_points"]}

    lab = {
        name: snr_block(
            block,
            measurement,
            delta_x,
            redundancy,
            point_config(nominal, points[name], None),
            tolerance,
        )
        for name, block in record["lab"].items()
    }
    plant = {
        case: {
            name: {
                "scaling": block["scaling"],
                **snr_block(
                    block,
                    measurement,
                    delta_x,
                    redundancy,
                    point_config(nominal, points[name], block["scaling"]),
                    tolerance,
                ),
            }
            for name, block in per_point.items()
        }
        for case, per_point in record["plant_like"]["points"].items()
    }
    result = {
        "source": str(source),
        "source_commit": record["provenance"]["git/commit"],
        "source_dirty": record["provenance"]["git/dirty"],
        "smoke": record["smoke"],
        "operating_points": record["operating_points"],
        "lab": lab,
        "plant_like": {"label": record["plant_like"]["label"], "cases": plant},
        "analytic_agrees_everywhere_lab": all(
            b["analytic"]["agrees_balance"] for b in lab.values()
        ),
        "reading": reading(lab),
        "dynamics": {
            "selection": record["dynamics"]["selection"],
            "settling": record["dynamics"]["settling"],
            "summary": dynamics_summary(record["dynamics"]),
        },
        "context": context_block(record["lab"], lab, delta_x),
    }
    output.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"reading: {result['reading']['reading']} -- {result['reading']['text']}")
    for q in result["reading"]["qualifiers"]:
        print(f"  qualifier: {q}")
    header = "point  ratio   " + "  ".join(f"{v:>13}" for v in VARIANTS)
    for case, table in [("lab", lab), *[(f"plant-like {c}", t) for c, t in plant.items()]]:
        print(f"\n{case}: SNR (R^2 of the fit)\n{header}")
        for name, b in table.items():
            cells = "  ".join(
                f"{b['variants'][v]['snr']:6.3f} ({b['variants'][v]['fit']['r2']:.3f})"
                for v in VARIANTS
            )
            print(f"{name:5}  {b['solid_to_air_ratio']:.4f}  {cells}")
    print(
        "\nanalytic (lab): ratio Y_balance/hypothesis, Y_diag/hypothesis, Y_bal/linearised, T/linearised"
    )
    for name, b in lab.items():
        a = b["analytic"]
        print(
            f"{name}: {a['ratio_balance_to_hypothesis']:.3f} {a['ratio_diagnostic_to_hypothesis']:.3f}"
            f" {a['ratio_balance_to_linearised']:.3f} {a['ratio_T_bed_to_linearised']:.3f}"
        )
    print("\ndynamics: max ratio absolute / deviation, longest time above 2 sigma (h)")
    for step, per in result["dynamics"]["summary"].items():
        for v, e in per.items():
            print(
                f"{step:12} {v:14} abs {e['absolute']['max_ratio']:7.2f} "
                f"{e['absolute']['longest_time_above_h']:5.2f} h | dev "
                f"{e['deviation']['max_ratio']:6.2f} {e['deviation']['longest_time_above_h']:5.2f} h"
            )
    print(f"\nwrote {output}")
    return 0 if all(math.isfinite(b["variants"]["T_bed"]["snr"]) for b in lab.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

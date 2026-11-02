r"""QA1-QA4: link between process operation and heat integration.

Checks whether and how strongly the site pinch moves with the operating mode of the fluidized-bed
line, and how strongly the minimum fuel demand of the site changes as a result.

Starting point (``run_pinch_benchmark.py``, branch ``lp-benchmark``): the pinch sits in
EVERY tested (T_kalt, dT_min) combination exactly at the inlet of source 2
(fluidized-bed line, 81.42 C) -- exactly the line that this project models as a grey-box
model. This temperature can thus be adjusted via the process set points. Question: does
the pinch move when the fluidized-bed line is operated differently, and how does
the minimum fuel demand change as a result?

QA1 -- scale: plant parameters instead of scaling
----------------------------------------------------
Befund 52 (``artifacts/runs/bb1_bb3_operating_point_regret.json``) is at LAB SCALE
(~0.03 kg/s air) -- the optimal set points from there are NOT adopted here.
Instead: Befund 56/NA5 (``scripts/study_plant_validation.py``) showed that the
grey-box model with PLANT PARAMETERS hits the measured bed temperature of both F-5
operating points (application partner, granulator D 3204) to within 0.4 and 2.1 K,
respectively, with ONE parameter set for both. This script builds on the same parameter set
(imports ``build_plant_config``/``OperatingPoint`` directly from
``study_plant_validation.py`` -- no second parametrisation).

Anchor: the product 1 operating point (better validation, -0.4 K against -2.1 K for
product 2). Values verbatim from Befund 56/NA5 (24h mean,
``scripts/study_plant_validation.py`` against the raw data under
``TWINLINK_PLANT_DATA_PROD1``, see NA1 confidentiality -- these aggregated 24h means are
already committed in Befund 56, e.g. the spray rates 934/2118 kg/h in its NA4 section):

    T_amb=4.9 C, T_zuluft=84.6 C, air flow=33003 Nm3/h, spray rate=934 kg/h,
    T_bett(measured)=67.3 C, bed mass=800 kg (oral statement by the partner -- irrelevant
    for T_bed, see Befund 56, here only sets cross-section area/gas velocity).

Source 1 (spray tower) stays FIXED at the Model_v3 values (PA2, ``lp-benchmark``):
20 kg/s, 125.233 C, Y=0.06482 -- this project only models the fluidized-bed line.

QA2 -- grid over the three actions
---------------------------------------------
``build_plant_config`` already sets the plant ActionConfig limits (inlet air temperature
0-150 C, gas velocity 0 to max(6, 1.5*product 1 nominal value), spray rate 0-3000 kg/h).
Coarse grid: 3 levels per quantity at 25/50/75 % of the span [minimum, maximum] -- NOT
the boundary values themselves (0 kg/h spray rate or 0 m/s gas velocity are degenerate
operating points without drying, numerically not meaningful). 27 points in total.

QA3/QA4 -- pinch and trade-off per point
----------------------------------------------
For each FEASIBLE point (``solution.feasible`` -- fluidization, degree of saturation,
particle moisture plausibility, see ``steady_state.py``): pinch target
(``heatnetwork.pinch``, dT_min=20 K, T_kalt=29.1 C as in the main task) with source 2 from
the steady-state solution (exhaust air temperature = bed temperature, ``outlet_duct_heat_loss_k
=0`` convention, no calibrated alternative available, see Befund 56 NA4).
``energy_target``/``power_heater_kw`` (``diagnostics``, already computed in the solver) per
point set against the pinch fuel demand (QA4) -- the two in different
energy carriers (process side: electricity + possibly gas mixed across four units; site:
natural gas equivalent), NOT directly additive (the same caution as Befund 57i against an
unjustified kW sum across energy carriers).

Usage::

    python scripts/run_process_pinch_sensitivity.py \
        --output artifacts/runs/run_process_pinch_sensitivity.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from study_plant_validation import (  # noqa: E402
    OperatingPoint,
    _solve_steady_state_widening_bracket,
    build_plant_config,
)

from twinlink_ml.heatnetwork.config import HeatSinkConfig, HeatSourceConfig  # noqa: E402
from twinlink_ml.heatnetwork.pinch import StreamSegment, evaluate_pinch_target  # noqa: E402
from twinlink_ml.simulation import GreyBoxFluidBedGranulator  # noqa: E402
from twinlink_ml.simulation.config import GasConfig, ProcessConfig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "run_process_pinch_sensitivity.json"

GAS = GasConfig()

# --- QA1: plant parameters, verbatim from Befund 56/NA5 (product 1, better validation) -------

PROD1_AMBIENT_TEMPERATURE_C = 4.9
PROD1_INLET_TEMPERATURE_C = 84.6
PROD1_BED_TEMPERATURE_MEASURED_C = 67.3
PROD1_PROCESS_AIR_FLOW_NM3_PER_H = 33003.0
PROD1_SPRAY_RATE_KG_PER_H = 934.0
BED_MASS_SOLID_SETPOINT_KG = 800.0
"""Muendliche Partnerangabe (vs. 600 kg Anzeige A-3215) -- Befund 56: fuer T_bed identisch,
wirkt hier nur auf Querschnittsflaeche/Gasgeschwindigkeit."""
BED_PRESSURE_DROP_PA = 1920.0
"""19.2 mbar, PDIRAH-320404 (Screenshot), identisch zu study_plant_validation.py."""
CROSS_SECTION_AREA_M2 = BED_MASS_SOLID_SETPOINT_KG * 9.81 / BED_PRESSURE_DROP_PA

_BASE_OPERATING_POINT = OperatingPoint(
    label="prod1",
    ambient_temperature_c=PROD1_AMBIENT_TEMPERATURE_C,
    inlet_temperature_c=PROD1_INLET_TEMPERATURE_C,
    bed_temperature_measured_c=PROD1_BED_TEMPERATURE_MEASURED_C,
    upper_position_measured_c=0.0,  # not used by build_plant_config
    outlet_temperature_measured_c=0.0,  # not used by build_plant_config
    process_air_flow_nm3_per_h=PROD1_PROCESS_AIR_FLOW_NM3_PER_H,
    spray_rate_kg_per_h=PROD1_SPRAY_RATE_KG_PER_H,
)

_BASE_PLANT_CONFIG = build_plant_config(
    _BASE_OPERATING_POINT,
    cross_section_area_m2=CROSS_SECTION_AREA_M2,
    bed_mass_solid_setpoint_kg=BED_MASS_SOLID_SETPOINT_KG,
)

# --- QA2: grid, 25/50/75 % of the ActionConfig span (see module docstring) --------------

_GRID_FRACTIONS = (0.25, 0.5, 0.75)


def _grid_levels(minimum: float, maximum: float) -> tuple[float, ...]:
    return tuple(minimum + fraction * (maximum - minimum) for fraction in _GRID_FRACTIONS)


INLET_TEMPERATURE_LEVELS_C = _grid_levels(
    _BASE_PLANT_CONFIG.actions.inlet_temperature_celsius.minimum,
    _BASE_PLANT_CONFIG.actions.inlet_temperature_celsius.maximum,
)
GAS_VELOCITY_LEVELS_M_PER_S = _grid_levels(
    _BASE_PLANT_CONFIG.actions.gas_velocity_m_per_s.minimum,
    _BASE_PLANT_CONFIG.actions.gas_velocity_m_per_s.maximum,
)
SPRAY_RATE_LEVELS_KG_PER_H = _grid_levels(
    _BASE_PLANT_CONFIG.actions.spray_rate_kg_per_h.minimum,
    _BASE_PLANT_CONFIG.actions.spray_rate_kg_per_h.maximum,
)

# --- QA3: pinch parameters, PA2 data case (source 1 fixed, sinks unchanged) ------------

SOURCE_1_SPRUEHTURM = HeatSourceConfig("Q1", 20.0, 125.233, 0.06482)
SINK_1 = HeatSinkConfig("S1", 20.0, 0.005, 15.0, 300.0)
SINK_2 = HeatSinkConfig("S2", 8.0, 0.005, 15.0, 120.0)
COLD_TEMPERATURE_CELSIUS = 29.1
MINIMUM_PINCH_K = 20.0
FUEL_LOWER_HEATING_VALUE_J_PER_KG = 45.0e6
HEATER_EFFICIENCY = 0.80


def _fuel_kg_per_h(power_kw: float) -> float:
    return power_kw * 1000.0 / (FUEL_LOWER_HEATING_VALUE_J_PER_KG * HEATER_EFFICIENCY) * 3600.0


def _build_grid_config(
    inlet_temperature_c: float, gas_velocity_m_per_s: float, spray_rate_kg_per_h: float
) -> ProcessConfig:
    actions = dataclasses.replace(
        _BASE_PLANT_CONFIG.actions,
        inlet_temperature_celsius=dataclasses.replace(
            _BASE_PLANT_CONFIG.actions.inlet_temperature_celsius, nominal=inlet_temperature_c
        ),
        gas_velocity_m_per_s=dataclasses.replace(
            _BASE_PLANT_CONFIG.actions.gas_velocity_m_per_s, nominal=gas_velocity_m_per_s
        ),
        spray_rate_kg_per_h=dataclasses.replace(
            _BASE_PLANT_CONFIG.actions.spray_rate_kg_per_h, nominal=spray_rate_kg_per_h
        ),
    )
    return dataclasses.replace(_BASE_PLANT_CONFIG, actions=actions)


def _locate_pinch(
    pinch_temperature_hot_side_celsius: float,
    hot_segments: tuple[StreamSegment, ...],
    cold_segments: tuple[StreamSegment, ...],
    minimum_pinch_k: float,
) -> str:
    half = minimum_pinch_k / 2.0
    pinch_hot_shifted = pinch_temperature_hot_side_celsius - half
    tolerance = 1e-6
    for seg in hot_segments:
        shifted_high, shifted_low = seg.shifted(minimum_pinch_k)
        if abs(shifted_high - pinch_hot_shifted) < tolerance:
            return f"{seg.name} Eintritt bei {seg.high_temperature_celsius:.2f} C (heiss)"
        if abs(shifted_low - pinch_hot_shifted) < tolerance:
            return f"{seg.name} Kondensationskurve, Stuetzstelle {seg.low_temperature_celsius:.2f} C (heiss)"
    for seg in cold_segments:
        shifted_high, shifted_low = seg.shifted(minimum_pinch_k)
        if abs(shifted_high - pinch_hot_shifted) < tolerance:
            return f"{seg.name} Austritt bei {seg.high_temperature_celsius:.2f} C (kalt)"
        if abs(shifted_low - pinch_hot_shifted) < tolerance:
            return f"{seg.name} Eintritt bei {seg.low_temperature_celsius:.2f} C (kalt)"
    return "nicht eindeutig lokalisiert"


def evaluate_grid_point(
    inlet_temperature_c: float, gas_velocity_m_per_s: float, spray_rate_kg_per_h: float
) -> dict[str, Any]:
    config = _build_grid_config(inlet_temperature_c, gas_velocity_m_per_s, spray_rate_kg_per_h)
    simulator = GreyBoxFluidBedGranulator(config)
    grid_point = OperatingPoint(
        label=f"T{inlet_temperature_c:.0f}_v{gas_velocity_m_per_s:.2f}_S{spray_rate_kg_per_h:.0f}",
        ambient_temperature_c=PROD1_AMBIENT_TEMPERATURE_C,
        inlet_temperature_c=inlet_temperature_c,
        bed_temperature_measured_c=PROD1_BED_TEMPERATURE_MEASURED_C,
        upper_position_measured_c=0.0,
        outlet_temperature_measured_c=0.0,
        process_air_flow_nm3_per_h=PROD1_PROCESS_AIR_FLOW_NM3_PER_H,
        spray_rate_kg_per_h=spray_rate_kg_per_h,
    )

    record: dict[str, Any] = {
        "inlet_temperature_c": inlet_temperature_c,
        "gas_velocity_m_per_s": gas_velocity_m_per_s,
        "spray_rate_kg_per_h": spray_rate_kg_per_h,
    }
    try:
        solution, widened_low = _solve_steady_state_widening_bracket(simulator, grid_point)
    except RuntimeError as exc:
        record["solver_failed"] = True
        record["solver_error"] = str(exc)
        record["feasible"] = False
        return record

    record["solver_failed"] = False
    record["widened_bracket_low_c"] = widened_low
    record["outlet_temperature_c"] = solution.bed_temperature_celsius
    record["outlet_humidity_kg_per_kg"] = solution.outlet_humidity_kg_per_kg
    record["dry_air_mass_flow_kg_per_s"] = solution.dry_air_mass_flow_kg_per_s
    record["energy_target_kw"] = solution.diagnostics["energy_target"]
    record["power_heater_kw"] = solution.diagnostics["power_heater_kw"]
    record["power_total_kw"] = solution.diagnostics["power_total_kw"]
    record["product_moisture_kg_per_kg"] = solution.particle_moisture_kg_per_kg
    record["critical_moisture_kg_per_kg"] = config.material.critical_moisture_kg_per_kg
    record["quality_ok"] = (
        solution.particle_moisture_kg_per_kg <= config.material.critical_moisture_kg_per_kg
    )
    record["minimum_fluidization_velocity_m_per_s"] = solution.diagnostics[
        "minimum_fluidization_velocity_m_per_s"
    ]
    record["terminal_velocity_m_per_s"] = solution.diagnostics["terminal_velocity_m_per_s"]
    record["fluidization_margin_lower_m_per_s"] = (
        gas_velocity_m_per_s - solution.diagnostics["minimum_fluidization_velocity_m_per_s"]
    )
    record["fluidization_margin_upper_m_per_s"] = (
        solution.diagnostics["terminal_velocity_m_per_s"] - gas_velocity_m_per_s
    )
    record["solver_feasible"] = solution.feasible
    record["infeasibility_reasons"] = list(solution.infeasibility_reasons)
    record["feasible"] = solution.feasible

    if not solution.feasible:
        return record

    source_2 = HeatSourceConfig(
        "Q2",
        solution.dry_air_mass_flow_kg_per_s,
        solution.bed_temperature_celsius,
        solution.outlet_humidity_kg_per_kg,
    )
    pinch_result, hot_segments, cold_segments = evaluate_pinch_target(
        (SOURCE_1_SPRUEHTURM, source_2),
        (SINK_1, SINK_2),
        GAS,
        cold_temperature_celsius=COLD_TEMPERATURE_CELSIUS,
        minimum_pinch_k=MINIMUM_PINCH_K,
    )
    pinch_hot_c = pinch_result.pinch_temperature_hot_side_celsius
    heat_above_pinch_kw = sum(
        seg.capacity_rate_kw_per_k
        * max(0.0, seg.high_temperature_celsius - max(seg.low_temperature_celsius, pinch_hot_c))
        for seg in hot_segments
    )
    record["pinch_temperature_hot_side_c"] = pinch_result.pinch_temperature_hot_side_celsius
    record["pinch_temperature_cold_side_c"] = pinch_result.pinch_temperature_cold_side_celsius
    record["pinch_location"] = _locate_pinch(
        pinch_result.pinch_temperature_hot_side_celsius,
        hot_segments,
        cold_segments,
        MINIMUM_PINCH_K,
    )
    record["minimum_hot_utility_kw"] = pinch_result.minimum_hot_utility_kw
    record["minimum_cold_utility_kw"] = pinch_result.minimum_cold_utility_kw
    record["heat_above_pinch_kw"] = heat_above_pinch_kw
    record["pinch_fuel_kg_per_h"] = _fuel_kg_per_h(pinch_result.minimum_hot_utility_kw)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    arguments = parser.parse_args()

    print("=" * 96)
    print("QA1 -- Anlagenparameter (Befund 56/NA5, product 1-Betriebspunkt)")
    print("=" * 96)
    print(
        f"T_amb={PROD1_AMBIENT_TEMPERATURE_C} C, T_zuluft={PROD1_INLET_TEMPERATURE_C} C, "
        f"Luftmenge={PROD1_PROCESS_AIR_FLOW_NM3_PER_H:.0f} Nm3/h, "
        f"Spruehrate={PROD1_SPRAY_RATE_KG_PER_H:.0f} kg/h, m_bed={BED_MASS_SOLID_SETPOINT_KG:.0f} kg, "
        f"A={CROSS_SECTION_AREA_M2:.2f} m^2"
    )
    print(
        "Gitter (25/50/75 % der ActionConfig-Spanne): "
        f"T_zuluft={tuple(round(v, 1) for v in INLET_TEMPERATURE_LEVELS_C)} C, "
        f"u_gas={tuple(round(v, 2) for v in GAS_VELOCITY_LEVELS_M_PER_S)} m/s, "
        f"Spruehrate={tuple(round(v, 0) for v in SPRAY_RATE_LEVELS_KG_PER_H)} kg/h"
    )

    print("\n" + "=" * 96)
    print("QA2/QA3/QA4 -- 27 Gitterpunkte")
    print("=" * 96)
    results: list[dict[str, Any]] = []
    for inlet_t, gas_v, spray_r in product(
        INLET_TEMPERATURE_LEVELS_C, GAS_VELOCITY_LEVELS_M_PER_S, SPRAY_RATE_LEVELS_KG_PER_H
    ):
        record = evaluate_grid_point(inlet_t, gas_v, spray_r)
        results.append(record)
        if record.get("solver_failed"):
            print(
                f"T={inlet_t:5.1f} C v={gas_v:.2f} m/s S={spray_r:6.0f} kg/h: "
                f"KEIN Beharrungszustand ({record['solver_error']})"
            )
        elif not record["feasible"]:
            print(
                f"T={inlet_t:5.1f} C v={gas_v:.2f} m/s S={spray_r:6.0f} kg/h: "
                f"UNZULAESSIG [{'; '.join(record['infeasibility_reasons'])}]"
            )
        else:
            print(
                f"T={inlet_t:5.1f} C v={gas_v:.2f} m/s S={spray_r:6.0f} kg/h: "
                f"T_out={record['outlet_temperature_c']:6.2f} C  "
                f"energy_target={record['energy_target_kw']:7.2f} kW  "
                f"Pinch={record['pinch_temperature_hot_side_c']:6.2f} C ({record['pinch_location']})  "
                f"Standort-Brennstoff={record['pinch_fuel_kg_per_h']:7.2f} kg/h"
            )

    feasible_results = [r for r in results if r.get("feasible")]
    infeasible_results = [r for r in results if not r.get("feasible")]

    print("\n" + "=" * 96)
    print("QA3 -- Kernfragen")
    print("=" * 96)
    print(f"Zulaessige Punkte: {len(feasible_results)} von {len(results)}")
    if feasible_results:
        pinch_temperatures = [r["pinch_temperature_hot_side_c"] for r in feasible_results]
        fuels = [r["pinch_fuel_kg_per_h"] for r in feasible_results]
        locations = {r["pinch_location"] for r in feasible_results}
        print(
            f"Pinch-Temperatur (heiss) ueber alle zulaessigen Punkte: "
            f"{min(pinch_temperatures):.2f}-{max(pinch_temperatures):.2f} C "
            f"(Spanne {max(pinch_temperatures) - min(pinch_temperatures):.2f} K)"
        )
        print(f"Beobachtete Pinch-Lagen: {len(locations)} verschiedene")
        for location in sorted(locations):
            print(f"  - {location}")
        print(
            f"Standort-Brennstoffbedarf: {min(fuels):.2f}-{max(fuels):.2f} kg/h "
            f"(absolut {max(fuels) - min(fuels):.2f} kg/h, relativ "
            f"{(max(fuels) - min(fuels)) / min(fuels) * 100.0:.1f} %)"
        )

    print("\n" + "=" * 96)
    print("QA4 -- Zielkonflikt: energy_target (Prozess) vs. Pinch-Brennstoff (Standort)")
    print("=" * 96)
    inversion_found = False
    max_inversion_effect_kw = 0.0
    max_inversion_pair: tuple[dict[str, Any], dict[str, Any]] | None = None
    if len(feasible_results) >= 2:
        by_energy = sorted(feasible_results, key=lambda r: r["energy_target_kw"])
        for i in range(len(by_energy)):
            for j in range(i + 1, len(by_energy)):
                cheaper_process, pricier_process = by_energy[i], by_energy[j]
                if cheaper_process["pinch_fuel_kg_per_h"] > pricier_process["pinch_fuel_kg_per_h"]:
                    inversion_found = True
                    effect_kw = (
                        pricier_process["energy_target_kw"] - cheaper_process["energy_target_kw"]
                    )
                    if effect_kw > max_inversion_effect_kw:
                        max_inversion_effect_kw = effect_kw
                        max_inversion_pair = (cheaper_process, pricier_process)
        if inversion_found and max_inversion_pair is not None:
            cheap, pricy = max_inversion_pair
            fuel_gap = cheap["pinch_fuel_kg_per_h"] - pricy["pinch_fuel_kg_per_h"]
            print(
                "JA -- es gibt Fahrweisen, die prozessseitig teurer, standortseitig aber "
                "guenstiger sind."
            )
            print(
                f"Groesster gefundener Effekt: +{max_inversion_effect_kw:.2f} kW Prozessleistung "
                f"sparen {fuel_gap:.2f} kg/h Standort-Erdgas "
                f"({cheap['pinch_fuel_kg_per_h']:.2f} vs. {pricy['pinch_fuel_kg_per_h']:.2f} kg/h)."
            )
        else:
            print(
                "NEIN -- keine Inversion gefunden: hoehere Prozessleistung geht in diesem Gitter nie mit niedrigerem Standort-Brennstoffbedarf einher."
            )
    else:
        print("Zu wenige zulaessige Punkte fuer einen Vergleich.")

    all_results: dict[str, Any] = {
        "ace_operating_point": {
            "ambient_temperature_c": PROD1_AMBIENT_TEMPERATURE_C,
            "inlet_temperature_c": PROD1_INLET_TEMPERATURE_C,
            "bed_temperature_measured_c": PROD1_BED_TEMPERATURE_MEASURED_C,
            "process_air_flow_nm3_per_h": PROD1_PROCESS_AIR_FLOW_NM3_PER_H,
            "spray_rate_kg_per_h": PROD1_SPRAY_RATE_KG_PER_H,
            "bed_mass_solid_setpoint_kg": BED_MASS_SOLID_SETPOINT_KG,
            "cross_section_area_m2": CROSS_SECTION_AREA_M2,
        },
        "grid_levels": {
            "inlet_temperature_c": INLET_TEMPERATURE_LEVELS_C,
            "gas_velocity_m_per_s": GAS_VELOCITY_LEVELS_M_PER_S,
            "spray_rate_kg_per_h": SPRAY_RATE_LEVELS_KG_PER_H,
        },
        "pinch_parameters": {
            "cold_temperature_celsius": COLD_TEMPERATURE_CELSIUS,
            "minimum_pinch_k": MINIMUM_PINCH_K,
            "fuel_lower_heating_value_j_per_kg": FUEL_LOWER_HEATING_VALUE_J_PER_KG,
            "heater_efficiency": HEATER_EFFICIENCY,
        },
        "results": results,
        "n_feasible": len(feasible_results),
        "n_infeasible": len(infeasible_results),
        "inversion_found": inversion_found,
        "max_inversion_effect_kw": max_inversion_effect_kw,
    }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(all_results, indent=2, sort_keys=True))
    print(f"\nErgebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

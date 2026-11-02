r"""MA5: six interconnection cases of the partner model (Model_v3.xlsx) -- reproduction,
feasibility, feasible optima.

Data basis (reconstructed by the user from Model_v3.xlsx, not documented in the repo --
see the conversation, not ``docs/06_waermenetz.md``): two exhaust air sources (Q1 spray tower,
Q2 fluidized bed), two inlet air sinks (S1, S2), each with its own condensation temperature
T_kalt PER SOURCE (Q1: 10 C, Q2: 15 C) -- ``heatnetwork.config.HeatNetworkConfig.
cold_temperature_celsius`` is network-wide, not per source, therefore
``heatnetwork.balance.evaluate_cascade`` is NOT reused directly here; instead the same
cascade algorithm (EA2/EA1 formulas, see the ``balance.py`` module docstring) is rebuilt
manually with thermo primitives, generalised to one T_kalt per source. Pure
diagnostic/reconciliation computation, NO change to ``src/``.

Formula error in the sheet (identified by the user, only retraced here, not newly
found): ``Balances!E20``/``E27`` combines the enthalpy of evaporation
(``Inputs!I12 = 2501``, labelled in the sheet as kJ/kg) WITHOUT conversion with the
J-based specific heat capacities -- effectively as 2501 J/kg instead of 2.501e6 J/kg,
a factor of 1000 too small. In addition, a sign error in the condensate term (condensate
leaves the balance, the term must be subtracted). MA5a recomputes BOTH corrections and
compares against the uncorrected sheet value -- the CORRECTED quantity is thereafter the
basis for MA5b/MA5c, NOT the sheet value (user instruction).

Six cases (source -> sink, cascaded cases in order)::

    Case 1  Q1->S1, Q2->S2                       (separate)
    Case 2  Q2->S1, Q1->S2                        (separate, swapped)
    Case 3  Q2 then Q1 -> S1                       (cold source first, S2 without recovery)
    Case 4  Q2 then Q1 -> S2                       (S1 without recovery)
    Case 5  Q1 then Q2 -> S1                       (hot source first, S2 without recovery)
    Case 6  only Q1 -> S2                          (Q2 unused, S1 without recovery)

MA5a -- reproduction (without approach-temperature check, ``minimum_pinch_k=0``)
MA5b -- feasibility per heat exchanger stage (air outlet vs. source inlet of the same
        stage -- a pure energy-balance cascade without approach temperature can violate this)
MA5c -- feasible optima with ``minimum_pinch_k=20`` (sensitivity: 10)

Usage::

    python scripts/probe_six_case_reconciliation.py \
        --output artifacts/runs/probe_six_case_reconciliation.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from twinlink_ml.simulation.backend import NumpyBackend
from twinlink_ml.simulation.config import GasConfig
from twinlink_ml.simulation.thermo import (
    dew_point_celsius,
    latent_heat_j_per_kg,
    saturation_humidity_kg_per_kg,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "probe_six_case_reconciliation.json"

BACKEND = NumpyBackend()
GAS = GasConfig()

FUEL_LOWER_HEATING_VALUE_J_PER_KG = 45.0e6
"""``Inputs!I9`` (Model_v3.xlsx) -- laut Nutzer korrekt als ``I9*1e6`` in der
Trocknerbilanz verwendet (Balances!F5/F12), hier fuer die Erhitzer-Brennstoffbilanz mit
demselben Wert uebernommen."""

HEATER_EFFICIENCY = 0.80
"""Model_v3.xlsx, gegeben -- ANDERER Wert als ``heatnetwork.config.
DEFAULT_HEATER_EFFICIENCY`` (0.90, Projekt-Platzhalter ohne Quelle). Hier bewusst der
Blattwert, weil MA5 explizit das Partnermodell nachrechnet."""

AMBIENT_TEMPERATURE_CELSIUS = 15.0
"""Model_v3.xlsx, gegeben -- Bezugstemperatur, Sinken-Ansaugtemperatur."""

SENSITIVITY_PINCH_VALUES_K = (20.0, 10.0)
"""MA5c: Hauptrechnung 20 K (wie ``heatnetwork.config.DEFAULT_MINIMUM_PINCH_K``),
Sensitivitaet 10 K."""


@dataclass(frozen=True, slots=True)
class Source:
    """An exhaust air source with its OWN condensation temperature (Model_v3.xlsx)."""

    name: str
    mass_flow_kg_per_s: float
    outlet_temperature_celsius: float
    outlet_humidity_kg_per_kg: float
    cold_temperature_celsius: float


@dataclass(frozen=True, slots=True)
class Sink:
    """An inlet air sink (Model_v3.xlsx)."""

    name: str
    mass_flow_kg_per_s: float
    humidity_kg_per_kg: float
    intake_temperature_celsius: float
    setpoint_temperature_celsius: float


SOURCES: dict[str, Source] = {
    "Q1": Source("Q1", 20.0, 125.233, 0.06482, 10.0),
    "Q2": Source("Q2", 8.0, 81.416, 0.01998, 15.0),
}
SINKS: dict[str, Sink] = {
    "S1": Sink("S1", 20.0, 0.005, AMBIENT_TEMPERATURE_CELSIUS, 300.0),
    "S2": Sink("S2", 8.0, 0.005, AMBIENT_TEMPERATURE_CELSIUS, 120.0),
}

CASES: dict[str, tuple[tuple[str, str], ...]] = {
    "Fall1": (("Q1", "S1"), ("Q2", "S2")),
    "Fall2": (("Q2", "S1"), ("Q1", "S2")),
    "Fall3": (("Q2", "S1"), ("Q1", "S1")),
    "Fall4": (("Q2", "S2"), ("Q1", "S2")),
    "Fall5": (("Q1", "S1"), ("Q2", "S1")),
    "Fall6": (("Q1", "S2"),),
}

SHEET_FUEL_UNCORRECTED_KG_PER_H: dict[str, float] = {
    "Fall1": 340.73,
    "Fall2": 522.62,
    "Fall3": 340.73,
    "Fall4": 578.78,
    "Fall5": 340.73,
    "Fall6": 578.78,
}


def _saturation_humidity(temperature_celsius: float) -> float:
    return float(
        BACKEND.to_numpy(
            saturation_humidity_kg_per_kg(BACKEND, GAS, BACKEND.asarray(temperature_celsius))
        )
    )


def _latent_heat(temperature_celsius: float) -> float:
    return float(
        BACKEND.to_numpy(latent_heat_j_per_kg(BACKEND, BACKEND.asarray(temperature_celsius)))
    )


def _dew_point(humidity_kg_per_kg: float) -> float:
    return float(
        BACKEND.to_numpy(dew_point_celsius(BACKEND, GAS, BACKEND.asarray(humidity_kg_per_kg)))
    )


def source_available_power_kw(source: Source) -> dict[str, float]:
    """MA1 formula (heatnetwork.balance, recovery_mode='condensing'), directly from thermo.py,
    here with the SOURCE'S OWN T_kalt -- the full, uncapped availability (no pinch,
    no sink capacity)."""
    y_sat_cold = _saturation_humidity(source.cold_temperature_celsius)
    condensate_kg_per_s = max(0.0, source.outlet_humidity_kg_per_kg - y_sat_cold) * (
        source.mass_flow_kg_per_s
    )
    dh_v = _latent_heat(source.cold_temperature_celsius)
    q_latent_w = condensate_kg_per_s * dh_v
    capacity_w_per_k = source.mass_flow_kg_per_s * (
        GAS.cp_dry_air_j_per_kg_k + source.outlet_humidity_kg_per_kg * GAS.cp_vapour_j_per_kg_k
    )
    q_sensible_w = capacity_w_per_k * (
        source.outlet_temperature_celsius - source.cold_temperature_celsius
    )
    dew_point = _dew_point(source.outlet_humidity_kg_per_kg)
    # Additional diagnostic (MA5, "split by temperature level"): how much of the supply
    # lies ABOVE the dew point (high-grade, purely sensible heat, usable in every
    # exchanger design) versus BELOW (sensible remainder + all of the
    # latent heat -- only usable if actually cooled below the dew point,
    # physically lower-grade).
    q_above_dew_point_kw = (
        max(0.0, capacity_w_per_k * (source.outlet_temperature_celsius - dew_point)) / 1000.0
    )
    q_below_dew_point_kw = (q_sensible_w + q_latent_w) / 1000.0 - q_above_dew_point_kw
    return {
        "q_rec_kw": (q_sensible_w + q_latent_w) / 1000.0,
        "q_sensible_kw": q_sensible_w / 1000.0,
        "q_latent_kw": q_latent_w / 1000.0,
        "q_above_dew_point_kw": q_above_dew_point_kw,
        "q_below_dew_point_kw": q_below_dew_point_kw,
        "condensate_kg_per_s": condensate_kg_per_s,
        "capacity_w_per_k": capacity_w_per_k,
        "dew_point_celsius": dew_point,
    }


def evaluate_case(stages: tuple[tuple[str, str], ...], *, minimum_pinch_k: float) -> dict[str, Any]:
    """Multi-stage cascade, EA1/EA2 algorithm (``heatnetwork.balance.evaluate_cascade``)
    rebuilt manually with the SOURCE'S OWN T_kalt instead of a network-wide value -- otherwise
    identical mechanics: sensible cooling reserve first, latent store only afterwards (MA1),
    approach temperature ``minimum_pinch_k`` against the CURRENT (already cooled)
    source temperature, unconsumed source enthalpy remains available for the next stage
    (correction 2).

    :param minimum_pinch_k: ``0.0`` for MA5a/MA5b (no approach-temperature check -- the sink
        may heat up to the CURRENT source temperature itself, limited only by available
        energy); ``20.0``/``10.0`` for MA5c.
    """
    current_source_temperature = {name: s.outlet_temperature_celsius for name, s in SOURCES.items()}
    remaining_latent_kw = dict.fromkeys(SOURCES, 0.0)
    used_sources = {source_name for source_name, _ in stages}
    for name in used_sources:
        remaining_latent_kw[name] = source_available_power_kw(SOURCES[name])["q_latent_kw"]
    floor_temperature = {name: s.cold_temperature_celsius for name, s in SOURCES.items()}
    capacity_source_w_per_k = {
        name: s.mass_flow_kg_per_s
        * (GAS.cp_dry_air_j_per_kg_k + s.outlet_humidity_kg_per_kg * GAS.cp_vapour_j_per_kg_k)
        for name, s in SOURCES.items()
    }
    current_sink_temperature = {name: s.intake_temperature_celsius for name, s in SINKS.items()}
    capacity_sink_w_per_k = {
        name: s.mass_flow_kg_per_s
        * (GAS.cp_dry_air_j_per_kg_k + s.humidity_kg_per_kg * GAS.cp_vapour_j_per_kg_k)
        for name, s in SINKS.items()
    }

    stage_reports: list[dict[str, Any]] = []
    sink_recovered_kw = dict.fromkeys(SINKS, 0.0)

    for source_name, sink_name in stages:
        sink = SINKS[sink_name]
        t_source = current_source_temperature[source_name]
        t_sink = current_sink_temperature[sink_name]
        capacity_source = capacity_source_w_per_k[source_name]
        capacity_sink = capacity_sink_w_per_k[sink_name]

        pinch_cap_temperature = t_source - minimum_pinch_k
        target_sink_temperature = min(sink.setpoint_temperature_celsius, pinch_cap_temperature)
        target_sink_temperature = max(target_sink_temperature, t_sink)

        power_for_target_w = capacity_sink * (target_sink_temperature - t_sink)
        sensible_available_w = max(
            capacity_source * (t_source - floor_temperature[source_name]), 0.0
        )
        power_available_w = sensible_available_w + remaining_latent_kw[source_name] * 1000.0
        power_actual_w = max(0.0, min(power_for_target_w, power_available_w))
        power_sensible_used_w = min(power_actual_w, sensible_available_w)
        power_latent_used_w = power_actual_w - power_sensible_used_w

        new_source_temperature = t_source - power_sensible_used_w / capacity_source
        new_sink_temperature = t_sink + power_actual_w / capacity_sink

        stage_reports.append(
            {
                "source": source_name,
                "sink": sink_name,
                "source_temperature_before_c": t_source,
                "sink_temperature_before_c": t_sink,
                "source_temperature_after_c": new_source_temperature,
                "sink_temperature_after_c": new_sink_temperature,
                "power_kw": power_actual_w / 1000.0,
                # MA5b: physical realisability -- the sink at the OTHER end
                # of an (even ideal) heat exchanger must not become hotter than the
                # source at its INLET into this stage (2nd law). A pure
                # energy balance without this check can violate this (unnoticed).
                "feasibility_violation_k": max(0.0, new_sink_temperature - t_source),
            }
        )

        current_source_temperature[source_name] = new_source_temperature
        remaining_latent_kw[source_name] -= power_latent_used_w / 1000.0
        current_sink_temperature[sink_name] = new_sink_temperature
        sink_recovered_kw[sink_name] += power_actual_w / 1000.0

    fuel_kg_per_h = {}
    for name, sink in SINKS.items():
        demand_kw = (
            sink.mass_flow_kg_per_s
            * (sink.setpoint_temperature_celsius - current_sink_temperature[name])
            * (GAS.cp_dry_air_j_per_kg_k + sink.humidity_kg_per_kg * GAS.cp_vapour_j_per_kg_k)
            / 1000.0
        )
        demand_kw = max(demand_kw, 0.0)
        fuel_kg_per_h[name] = (
            demand_kw * 1000.0 / (FUEL_LOWER_HEATING_VALUE_J_PER_KG * HEATER_EFFICIENCY) * 3600.0
        )

    return {
        "stages": stage_reports,
        "sink_recovered_kw": sink_recovered_kw,
        "sink_outlet_after_recovery_c": current_sink_temperature,
        "fuel_kg_per_h_by_sink": fuel_kg_per_h,
        "total_fuel_kg_per_h": sum(fuel_kg_per_h.values()),
        "max_feasibility_violation_k": max(
            (s["feasibility_violation_k"] for s in stage_reports), default=0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    arguments = parser.parse_args()

    print("=" * 78)
    print("MA5a -- Quellenleistung, korrigierte Formel (MA1) vs. Blatt (unkorrigiert)")
    print("=" * 78)
    source_reports = {name: source_available_power_kw(s) for name, s in SOURCES.items()}
    for name, r in source_reports.items():
        print(
            f"{name}: Q_rec={r['q_rec_kw']:8.1f} kW  (sensibel={r['q_sensible_kw']:7.1f}  "
            f"latent={r['q_latent_kw']:7.1f})  Taupunkt={r['dew_point_celsius']:.2f} C  "
            f"Kondensat={r['condensate_kg_per_s']:.4f} kg/s"
        )
        print(
            f"    oberhalb Taupunkt={r['q_above_dew_point_kw']:7.1f} kW  "
            f"unterhalb Taupunkt={r['q_below_dew_point_kw']:7.1f} kW"
        )
    total_offer_corrected = sum(r["q_rec_kw"] for r in source_reports.values())
    total_offer_sheet = 2671.8 + 561.6
    print(
        f"\nAngebot gesamt: korrigiert={total_offer_corrected:.1f} kW  "
        f"Blatt (unkorrigiert)={total_offer_sheet:.1f} kW"
    )

    print("\n" + "=" * 78)
    print("MA5a/b -- sechs Faelle OHNE Graedigkeitspruefung (minimum_pinch_k=0)")
    print("=" * 78)
    no_pinch_results = {
        name: evaluate_case(stages, minimum_pinch_k=0.0) for name, stages in CASES.items()
    }
    for name, result in no_pinch_results.items():
        sheet_fuel = SHEET_FUEL_UNCORRECTED_KG_PER_H[name]
        print(
            f"{name}: Brennstoff(korrigiert)={result['total_fuel_kg_per_h']:7.2f} kg/h  "
            f"Blatt(unkorrigiert)={sheet_fuel:7.2f} kg/h  "
            f"max. Realisierbarkeits-Verletzung={result['max_feasibility_violation_k']:6.1f} K"
        )
        for stage in result["stages"]:
            if stage["feasibility_violation_k"] > 0.05:
                print(
                    f"    VERLETZUNG {stage['source']}->{stage['sink']}: "
                    f"Senkenaustritt {stage['sink_temperature_after_c']:.2f} C > "
                    f"Quelleneintritt {stage['source_temperature_before_c']:.2f} C "
                    f"({stage['feasibility_violation_k']:.1f} K)"
                )

    all_results: dict[str, Any] = {
        "sources": {
            name: {
                "mass_flow_kg_per_s": s.mass_flow_kg_per_s,
                "outlet_temperature_celsius": s.outlet_temperature_celsius,
                "outlet_humidity_kg_per_kg": s.outlet_humidity_kg_per_kg,
                "cold_temperature_celsius": s.cold_temperature_celsius,
                **source_reports[name],
            }
            for name, s in SOURCES.items()
        },
        "total_offer_corrected_kw": total_offer_corrected,
        "total_offer_sheet_uncorrected_kw": total_offer_sheet,
        "no_pinch": no_pinch_results,
        "pinch_sensitivity": {},
    }

    for pinch_k in SENSITIVITY_PINCH_VALUES_K:
        print("\n" + "=" * 78)
        print(f"MA5c -- sechs Faelle MIT Graedigkeit dT_min={pinch_k:.0f} K")
        print("=" * 78)
        pinch_results = {
            name: evaluate_case(stages, minimum_pinch_k=pinch_k) for name, stages in CASES.items()
        }
        for name, result in pinch_results.items():
            print(f"{name}: Brennstoff={result['total_fuel_kg_per_h']:7.2f} kg/h")
        best_name = min(pinch_results, key=lambda n: pinch_results[n]["total_fuel_kg_per_h"])
        best_fuel = pinch_results[best_name]["total_fuel_kg_per_h"]
        print(f"  -> bester machbarer Fall: {best_name} mit {best_fuel:.2f} kg/h")
        all_results["pinch_sensitivity"][f"{pinch_k:.0f}K"] = {
            "results": pinch_results,
            "best_case": best_name,
            "best_fuel_kg_per_h": best_fuel,
        }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(all_results, indent=2, sort_keys=True))
    print(f"\nErgebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

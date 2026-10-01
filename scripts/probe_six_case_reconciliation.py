r"""MA5: Sechs Verschaltungsfaelle des Partnermodells (Model_v3.xlsx) -- Reproduktion,
Machbarkeit, machbare Optima.

Datengrundlage (vom Nutzer aus Model_v3.xlsx rekonstruiert, nicht im Repo dokumentiert --
siehe Konversation, nicht ``docs/06_waermenetz.md``): zwei Abluftquellen (Q1 Spruehturm,
Q2 Wirbelschicht), zwei Zuluftsenken (S1, S2), je eigene Kondensationstemperatur T_kalt
PRO QUELLE (Q1: 10 C, Q2: 15 C) -- ``heatnetwork.config.HeatNetworkConfig.
cold_temperature_celsius`` ist netzweit, nicht je Quelle, deshalb hier NICHT
``heatnetwork.balance.evaluate_cascade`` direkt wiederverwendet, sondern derselbe
Kaskadenalgorithmus (EA2/EA1-Formeln, siehe ``balance.py``-Moduldocstring) manuell mit
Thermo-Primitiven nachgebaut, verallgemeinert auf ein T_kalt je Quelle. Reine
Diagnose-/Abgleichrechnung, KEINE Aenderung an ``src/``.

Formelfehler im Blatt (vom Nutzer identifiziert, hier nur nachvollzogen, nicht neu
gefunden): ``Balances!E20``/``E27`` verrechnet die Verdampfungsenthalpie
(``Inputs!I12 = 2501``, im Blatt als kJ/kg beschriftet) OHNE Umrechnung mit den
J-basierten spezifischen Waermekapazitaeten -- effektiv wie 2501 J/kg statt 2.501e6 J/kg,
Faktor 1000 zu klein. Zusaetzlich ein Vorzeichenfehler im Kondensatterm (Kondensat verlaesst
die Bilanz, der Term muss subtrahiert werden). MA5a rechnet BEIDE Korrekturen nach und
vergleicht gegen den unkorrigierten Blattwert -- die KORRIGIERTE Groesse ist danach die
Grundlage fuer MA5b/MA5c, NICHT der Blattwert (Nutzeranweisung).

Sechs Faelle (Quelle -> Senke, kaskadierte Faelle in Reihenfolge)::

    Fall 1  Q1->S1, Q2->S2                       (getrennt)
    Fall 2  Q2->S1, Q1->S2                        (getrennt, vertauscht)
    Fall 3  Q2 dann Q1 -> S1                       (kalte Quelle zuerst, S2 ohne Rueckgewinnung)
    Fall 4  Q2 dann Q1 -> S2                       (S1 ohne Rueckgewinnung)
    Fall 5  Q1 dann Q2 -> S1                       (heisse Quelle zuerst, S2 ohne Rueckgewinnung)
    Fall 6  nur Q1 -> S2                           (Q2 ungenutzt, S1 ohne Rueckgewinnung)

MA5a -- Reproduktion (ohne Graedigkeitspruefung, ``minimum_pinch_k=0``)
MA5b -- Machbarkeit je Waermeuebertragerstufe (Luftaustritt vs. Quelleneintritt derselben
        Stufe -- ein reiner Energiebilanz-Kaskade ohne Graedigkeit kann das verletzen)
MA5c -- Machbare Optima mit ``minimum_pinch_k=20`` (Sensitivitaet: 10)

Aufruf::

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
    """Eine Abluftquelle mit ihrer EIGENEN Kondensationstemperatur (Model_v3.xlsx)."""

    name: str
    mass_flow_kg_per_s: float
    outlet_temperature_celsius: float
    outlet_humidity_kg_per_kg: float
    cold_temperature_celsius: float


@dataclass(frozen=True, slots=True)
class Sink:
    """Eine Zuluftsenke (Model_v3.xlsx)."""

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
    """MA1-Formel (heatnetwork.balance, recovery_mode='condensing'), direkt aus thermo.py,
    hier mit dem QUELLENEIGENEN T_kalt -- die volle, ungekappte Verfuegbarkeit (kein Pinch,
    keine Senkenkapazitaet)."""
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
    # Zusatzdiagnose (MA5, "Aufteilung nach Temperaturniveau"): wie viel des Angebots
    # liegt OBERHALB des Taupunkts (hochwertige, rein sensible Waerme, in jedem
    # Uebertrager-Design nutzbar) gegenueber UNTERHALB (sensibler Rest + die gesamte
    # latente Waerme -- nur nutzbar, wenn tatsaechlich bis unter den Taupunkt gekuehlt
    # wird, physikalisch niederwertiger).
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
    """Mehrstufige Kaskade, EA1/EA2-Algorithmus (``heatnetwork.balance.evaluate_cascade``)
    manuell nachgebaut mit QUELLENEIGENEM T_kalt statt einem netzweiten Wert -- sonst
    identische Mechanik: sensible Kuehlreserve zuerst, latenter Vorrat erst danach (MA1),
    Grädigkeit ``minimum_pinch_k`` gegen die AKTUELLE (bereits abgekuehlte)
    Quellentemperatur, nicht verbrauchte Quellenenthalpie bleibt fuer die naechste Stufe
    verfuegbar (Korrektur 2).

    :param minimum_pinch_k: ``0.0`` fuer MA5a/MA5b (keine Graedigkeitspruefung -- Senke
        darf bis zur AKTUELLEN Quellentemperatur selbst aufheizen, nur durch verfuegbare
        Energie begrenzt); ``20.0``/``10.0`` fuer MA5c.
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
                # MA5b: physikalische Realisierbarkeit -- die Senke darf am ANDEREN Ende
                # eines (auch idealen) Waermeuebertragers nicht heisser werden als die
                # Quelle an dessen EINTRITT in diese Stufe (2. Hauptsatz). Eine reine
                # Energiebilanz ohne diese Pruefung kann das (unbemerkt) verletzen.
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

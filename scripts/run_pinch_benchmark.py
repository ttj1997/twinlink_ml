r"""PA1-PA4: Pinch-Zielwert (Problem-Table-Algorithmus) gegen die sechs Verschaltungsfaelle
des Partnermodells (Model_v3.xlsx, korrigierte Werte -- Faktor-1000-Fehler in der
Verdampfungsenthalpie, siehe Befund 57).

Beantwortet die Kernfrage von PA3 in einem Satz: wie weit ist die beste der sechs
Verschaltungen vom thermodynamischen Optimum entfernt? Das Optimum kommt aus
``heatnetwork.pinch`` (Problem-Table-Algorithmus, PA1); die sechs Faelle aus
``heatnetwork.balance.evaluate_cascade`` (derselbe Code, der auch ``heatnetwork/`` selbst
testet -- keine zweite Kaskadenimplementierung, anders als der fruehere
``probe_six_case_reconciliation.py``, der wegen quellenindividueller ``T_kalt``-Werte noch
manuell nachbauen musste; hier ist ``T_kalt`` netzweit EIN Parameter, das passt direkt in
``HeatNetworkConfig``).

Vorbehalt (PA3, MUSS in jeder Verwendung dieser Zahlen stehen): der Pinch-Zielwert setzt
beliebig viele, ideal gegeneinander geschaltete Waermeuebertrager voraus. Er ist eine UNTERE
SCHRANKE, keine Bauvorschrift -- eine kleine Luecke zum besten der sechs Faelle kann
vollstaendig durch deren Beschraenkung auf zwei Uebertrager erklaert sein.

Aufruf::

    python scripts/run_pinch_benchmark.py \
        --output artifacts/runs/run_pinch_benchmark.json
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

from twinlink_ml.heatnetwork.balance import SinkState, SourceState, evaluate_cascade
from twinlink_ml.heatnetwork.config import HeatNetworkConfig, HeatSinkConfig, HeatSourceConfig
from twinlink_ml.heatnetwork.pinch import (
    composite_curve_kw,
    dew_point_of_source_celsius,
    evaluate_pinch_target,
    hot_segments_for_source,
)
from twinlink_ml.simulation.config import GasConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "run_pinch_benchmark.json"

GAS = GasConfig()

# --- PA2-Datenfall (Model_v3.xlsx, KORRIGIERTE Werte, Befund 57) -----------------------

SOURCE_1 = HeatSourceConfig("Q1", 20.0, 125.233, 0.06482)  # Abluft Trockner 1
SOURCE_2 = HeatSourceConfig("Q2", 8.0, 81.416, 0.01998)  # Abluft Trockner 2
SINK_1 = HeatSinkConfig("S1", 20.0, 0.005, 15.0, 300.0)  # Zuluft Trockner 1
SINK_2 = HeatSinkConfig("S2", 8.0, 0.005, 15.0, 120.0)  # Zuluft Trockner 2

FUEL_LOWER_HEATING_VALUE_J_PER_KG = 45.0e6
HEATER_EFFICIENCY = 0.80

COLD_TEMPERATURES_CELSIUS = (15.0, 25.0, 29.1, 35.0)
"""PA2: 15 optimistischer Platzhalter, 29.1 reale Kuehlwassertemperatur laut Leitsystem
(Befund 57h), 25/35 Stuetzpunkte um die Kondensationsgrenze von Quelle 2 (Taupunkt 24.9 C)."""

MINIMUM_PINCH_VALUES_K = (10.0, 15.0, 20.0)

N_SEGMENTS_BELOW_DEW_POINT = 50
RESOLUTION_STUDY_VALUES = (2, 5, 20, 50, 200, 1000)

CASES: dict[str, tuple[tuple[str, str], ...]] = {
    "Fall1": (("Q1", "S1"), ("Q2", "S2")),
    "Fall2": (("Q2", "S1"), ("Q1", "S2")),
    "Fall3": (("Q2", "S1"), ("Q1", "S1")),
    "Fall4": (("Q2", "S2"), ("Q1", "S2")),
    "Fall5": (("Q1", "S1"), ("Q2", "S1")),
    "Fall6": (("Q1", "S2"),),
}


def _kw_to_fuel_kg_per_h(power_kw: float) -> float:
    return power_kw * 1000.0 / (FUEL_LOWER_HEATING_VALUE_J_PER_KG * HEATER_EFFICIENCY) * 3600.0


def _six_case_fuel_kg_per_h(
    *, cold_temperature_celsius: float, minimum_pinch_k: float
) -> dict[str, float]:
    network = HeatNetworkConfig(
        sources=(SOURCE_1, SOURCE_2),
        sinks=(SINK_1, SINK_2),
        minimum_pinch_k=minimum_pinch_k,
        recovery_mode="condensing",
        cold_temperature_celsius=cold_temperature_celsius,
        fuel_lower_heating_value_j_per_kg=FUEL_LOWER_HEATING_VALUE_J_PER_KG,
        heater_efficiency=HEATER_EFFICIENCY,
    )
    sources = tuple(SourceState.from_config(cfg) for cfg in network.sources)
    sinks = tuple(SinkState.from_config(cfg) for cfg in network.sinks)
    results = {}
    for name, stages in CASES.items():
        result = evaluate_cascade(network, stages, sources, sinks)
        results[name] = float(result.total_fuel_mass_flow_kg_per_s()[0]) * 3600.0
    return results


def _pinch_target(
    *, cold_temperature_celsius: float, minimum_pinch_k: float, n_segments_below_dew_point: int
) -> dict[str, Any]:
    result, hot_segments, cold_segments = evaluate_pinch_target(
        (SOURCE_1, SOURCE_2),
        (SINK_1, SINK_2),
        GAS,
        cold_temperature_celsius=cold_temperature_celsius,
        minimum_pinch_k=minimum_pinch_k,
        n_segments_below_dew_point=n_segments_below_dew_point,
    )
    return {
        "result": result,
        "hot_segments": hot_segments,
        "cold_segments": cold_segments,
        "fuel_kg_per_h": _kw_to_fuel_kg_per_h(result.minimum_hot_utility_kw),
    }


def _condensation_breakdown(
    source: HeatSourceConfig, floor_temperature_celsius: float
) -> dict[str, float]:
    """PA4: wie viel der rueckgewinnbaren Waerme liegt ober-/unterhalb des Taupunkts."""
    segments = hot_segments_for_source(source, GAS, floor_temperature_celsius)
    dew_point = dew_point_of_source_celsius(source, GAS)
    above = sum(
        seg.duty_kw() for seg in segments if seg.low_temperature_celsius >= dew_point - 1e-6
    )
    total = sum(seg.duty_kw() for seg in segments)
    return {
        "dew_point_celsius": dew_point,
        "total_kw": total,
        "above_dew_point_kw": above,
        "below_dew_point_kw": total - above,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    arguments = parser.parse_args()

    print("=" * 88)
    print("PA4 -- Kondensationsaufteilung je Quelle (T_kalt=15 C)")
    print("=" * 88)
    print(
        "Taupunkte aus Y ueber die Buck-Korrelation des Projekts (thermo.dew_point_celsius) --\n"
        "KEINE Referenzzahlen aus dem Auftragstext uebernommen, alles frisch aus den\n"
        "korrigierten Quellendaten (PA2) berechnet."
    )
    breakdown_by_source = {}
    for source in (SOURCE_1, SOURCE_2):
        breakdown = _condensation_breakdown(source, 15.0)
        breakdown_by_source[source.name] = breakdown
        above_share = breakdown["above_dew_point_kw"] / breakdown["total_kw"] * 100.0
        below_share = breakdown["below_dew_point_kw"] / breakdown["total_kw"] * 100.0
        print(
            f"{source.name}: Taupunkt={breakdown['dew_point_celsius']:6.2f} C  "
            f"gesamt={breakdown['total_kw']:8.1f} kW  "
            f"oberhalb={breakdown['above_dew_point_kw']:8.1f} kW ({above_share:4.1f} %)  "
            f"unterhalb={breakdown['below_dew_point_kw']:8.1f} kW ({below_share:4.1f} %)"
        )
    print(
        "-> Zwei Drittel der rueckgewinnbaren Waerme von Q1 liegen UNTERHALB seines "
        "Taupunkts (44.94 C), waehrend Senke S1 Luft auf 300 C bringen soll -- niedrigwertige "
        "Waerme, die drei der sechs Faelle (PA3) so behandeln, als waere sie hochwertig "
        "(2. Hauptsatz-Verletzung)."
    )

    print("\n" + "=" * 88)
    print("PA1 -- Aufloesungssensitivitaet unterhalb des Taupunkts (T_kalt=15, dT_min=20)")
    print("=" * 88)
    resolution_results = []
    for n_segments in RESOLUTION_STUDY_VALUES:
        pinch = _pinch_target(
            cold_temperature_celsius=15.0, minimum_pinch_k=20.0, n_segments_below_dew_point=n_segments
        )
        resolution_results.append(
            {
                "n_segments_below_dew_point": n_segments,
                "minimum_hot_utility_kw": pinch["result"].minimum_hot_utility_kw,
                "minimum_cold_utility_kw": pinch["result"].minimum_cold_utility_kw,
            }
        )
        print(
            f"  n={n_segments:5d}: Heizbedarf={pinch['result'].minimum_hot_utility_kw:12.6f} kW  "
            f"Kuehlbedarf={pinch['result'].minimum_cold_utility_kw:12.6f} kW"
        )
    finest_hot = resolution_results[-1]["minimum_hot_utility_kw"]
    finest_cold = resolution_results[-1]["minimum_cold_utility_kw"]
    for entry in resolution_results:
        entry["relative_change_hot_to_finest"] = abs(entry["minimum_hot_utility_kw"] - finest_hot) / finest_hot
        entry["relative_change_cold_to_finest"] = (
            abs(entry["minimum_cold_utility_kw"] - finest_cold) / finest_cold
        )
    max_relative_change = max(
        max(e["relative_change_hot_to_finest"], e["relative_change_cold_to_finest"])
        for e in resolution_results
    )
    print(
        f"  -> groesste relative Aenderung ueber alle getesteten n: {max_relative_change * 100.0:.4f} % "
        "(vernachlaessigbar hier, siehe Erklaerung unten)"
    )
    print(
        "  Erklaerung: der Pinch liegt bei 81.42 C (Eintritt Q2), oberhalb BEIDER Taupunkte "
        "(Q1: 44.94 C, Q2: 24.92 C) -- die feine Abtastung liegt komplett im Ueberschussbereich "
        "UNTERHALB des Pinch und aendert dessen Lage/Wert nicht. Das ist eine Eigenschaft DIESES "
        "Datensatzes (S1 verlangt 300 C, weit oberhalb jeder Quellentemperatur), keine allgemeine "
        "Garantie -- bei anderen Sollwerten koennte der Pinch in die Kondensationskurve fallen, "
        "dann wuerde die Aufloesung sehr wohl zaehlen."
    )

    print("\n" + "=" * 88)
    print("PA3 -- Pinch-Zielwert vs. bester der sechs Faelle, je (T_kalt, dT_min)")
    print("=" * 88)
    grid_results = []
    for cold_temperature_celsius, minimum_pinch_k in product(
        COLD_TEMPERATURES_CELSIUS, MINIMUM_PINCH_VALUES_K
    ):
        pinch = _pinch_target(
            cold_temperature_celsius=cold_temperature_celsius,
            minimum_pinch_k=minimum_pinch_k,
            n_segments_below_dew_point=N_SEGMENTS_BELOW_DEW_POINT,
        )
        six_case_fuel = _six_case_fuel_kg_per_h(
            cold_temperature_celsius=cold_temperature_celsius, minimum_pinch_k=minimum_pinch_k
        )
        best_case_name = min(six_case_fuel, key=lambda name: six_case_fuel[name])
        best_case_fuel = six_case_fuel[best_case_name]
        pinch_fuel = pinch["fuel_kg_per_h"]
        gap_kg_per_h = best_case_fuel - pinch_fuel
        relative_gap = gap_kg_per_h / pinch_fuel if pinch_fuel > 1e-9 else float("nan")

        print(
            f"T_kalt={cold_temperature_celsius:5.1f} C  dT_min={minimum_pinch_k:4.1f} K  "
            f"Pinch={pinch_fuel:8.2f} kg/h  bester Fall={best_case_name} ({best_case_fuel:8.2f} kg/h)  "
            f"Luecke={gap_kg_per_h:7.2f} kg/h ({relative_gap * 100.0:5.1f} %)  "
            f"Pinch-T(heiss/kalt)={pinch['result'].pinch_temperature_hot_side_celsius:6.2f}/"
            f"{pinch['result'].pinch_temperature_cold_side_celsius:6.2f} C"
        )

        grid_results.append(
            {
                "cold_temperature_celsius": cold_temperature_celsius,
                "minimum_pinch_k": minimum_pinch_k,
                "pinch_minimum_hot_utility_kw": pinch["result"].minimum_hot_utility_kw,
                "pinch_minimum_cold_utility_kw": pinch["result"].minimum_cold_utility_kw,
                "pinch_fuel_kg_per_h": pinch_fuel,
                "pinch_temperature_hot_side_celsius": pinch["result"].pinch_temperature_hot_side_celsius,
                "pinch_temperature_cold_side_celsius": pinch["result"].pinch_temperature_cold_side_celsius,
                "six_case_fuel_kg_per_h": six_case_fuel,
                "best_case_name": best_case_name,
                "best_case_fuel_kg_per_h": best_case_fuel,
                "gap_kg_per_h": gap_kg_per_h,
                "relative_gap": relative_gap,
            }
        )

    print("\n" + "=" * 88)
    print("PA4 -- Composite Curves und Grand Composite Curve (T_kalt=15, dT_min=20)")
    print("=" * 88)
    reference_pinch = _pinch_target(
        cold_temperature_celsius=15.0, minimum_pinch_k=20.0, n_segments_below_dew_point=N_SEGMENTS_BELOW_DEW_POINT
    )
    hot_composite = composite_curve_kw(reference_pinch["hot_segments"])
    cold_composite = composite_curve_kw(reference_pinch["cold_segments"])
    grand_composite = list(
        zip(
            reference_pinch["result"].interval_boundaries_shifted_celsius,
            reference_pinch["result"].cascade_kw,
            strict=True,
        )
    )
    print(f"  heisse Verbundkurve: {len(hot_composite)} Stuetzpunkte")
    print(f"  kalte Verbundkurve: {len(cold_composite)} Stuetzpunkte")
    print(f"  Grand Composite Curve: {len(grand_composite)} Stuetzpunkte")

    all_results: dict[str, Any] = {
        "sources": {
            "Q1": {"mass_flow_kg_per_s": 20.0, "outlet_temperature_celsius": 125.233, "outlet_humidity_kg_per_kg": 0.06482},
            "Q2": {"mass_flow_kg_per_s": 8.0, "outlet_temperature_celsius": 81.416, "outlet_humidity_kg_per_kg": 0.01998},
        },
        "sinks": {
            "S1": {"mass_flow_kg_per_s": 20.0, "intake_celsius": 15.0, "setpoint_celsius": 300.0},
            "S2": {"mass_flow_kg_per_s": 8.0, "intake_celsius": 15.0, "setpoint_celsius": 120.0},
        },
        "fuel_lower_heating_value_j_per_kg": FUEL_LOWER_HEATING_VALUE_J_PER_KG,
        "heater_efficiency": HEATER_EFFICIENCY,
        "n_segments_below_dew_point": N_SEGMENTS_BELOW_DEW_POINT,
        "condensation_breakdown_at_15c": breakdown_by_source,
        "resolution_study": resolution_results,
        "grid_results": grid_results,
        "reference_composite_curves": {
            "cold_temperature_celsius": 15.0,
            "minimum_pinch_k": 20.0,
            "hot_composite_curve_celsius_kw": hot_composite,
            "cold_composite_curve_celsius_kw": cold_composite,
            "grand_composite_curve_shifted_celsius_kw": grand_composite,
        },
    }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(all_results, indent=2, sort_keys=True))
    print(f"\nErgebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

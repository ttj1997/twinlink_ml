r"""MA2: Unsicherheit des Feuchtekanals in kW -- Fehlerpropagation an EINEM Betriebspunkt.

Anlass: Der Verbundpartner rechnet jetzt mit einem kondensierenden Wärmeübertrager (MA1,
``heatnetwork``). Damit steigt die Bedeutung der Abluftbeladung ``Y_out`` -- im
kondensierenden Modus liefert Feuchte zusätzliche LATENTE Rückgewinnung, statt nur den
Taupunkt (und damit die sensible Kühlgrenze) zu heben. ``Y_out`` ist eine algebraische
Funktion von Betttemperatur und Produktfeuchte (``simulation.exhaust.
reconstruct_exhaust_state``), beides Größen, deren Vorhersagefehler in den
Lernkurven-Datensätzen als ``rmse_run_median_physical`` bereits gemessen sind -- nicht neu
zu ermitteln, nur zu propagieren.

Vorgehen (kein neues Modell, keine neue Physik -- reine Störrechnung an einem festen
Referenzpunkt mit bereits vorhandenen Bausteinen):

1. Referenzpunkt: Beharrungszustand von ``greybox_gab_placeholder.yaml`` bei NOMINALEN
   Stell-/Störgrößen (``solve_steady_state`` ohne Variante) -- derselbe Nominalpunkt, auf
   den auch ``simulation.exhaust``s Moduldocstring verweist (Befund 31).
2. Betttemperatur und Produktfeuchte an diesem Punkt werden EINZELN um ihren gemessenen
   Fehler gestört (siehe ``BED_TEMPERATURE_RMSE_K``/``PRODUCT_MOISTURE_RMSE_KG_PER_KG``
   unten, mit Quellenangabe), dann BEIDE gleichzeitig in der ungünstigen
   Vorzeichenkombination (die, die \|ΔQ_rec\| maximiert -- durch Ausprobieren aller vier
   Vorzeichenkombinationen, nicht angenommen).
3. Je Störung: ``reconstruct_exhaust_state`` liefert das gestörte ``Y_out``/``T_out`` neu
   (Massenstrom und Zuluftfeuchte unverändert -- das sind Stell-/Störgrößen, keine
   Vorhersagen). ``Q_rec`` wird daraus in BEIDEN Rückgewinnungsmodi neu berechnet
   (sensibel: ``evaluation.recoverable_power.compute_recoverable_power_kw``; kondensierend:
   dieselbe MA1-Formel wie ``heatnetwork.balance.evaluate_cascade``, hier direkt aus
   ``thermo.py`` nachgebaut -- ein Einzelpunkt braucht keine Kaskade).
4. Skalenvorbehalt (wie ``heatnetwork``/``evaluation.recoverable_power`` durchgehend):
   dieselbe ΔT_out/ΔY_out wird bei BEIDEN Massenströmen ausgewertet --
   ``dry_air_mass_flow_kg_per_s`` aus dem Beharrungszustand selbst (Laborsimulator,
   ca. 0.03 kg/s) und ``DISTRICT_MASS_FLOW_KG_PER_S`` (Verbundmodell, 8 kg/s). Jede
   District-Zahl ist eine Skalierungsrechnung, keine Anlagenaussage.

Nicht Teil dieses Auftrags: ein neues Fehlermodell, eine Neubewertung der Lernkurven selbst
(Befund 51/55 bleiben die Quelle), eine Kaskade mit mehreren Quellen/Senken.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

from twinlink_ml.evaluation.recoverable_power import (
    DEW_POINT_MARGIN_K,
    DISTRICT_MASS_FLOW_KG_PER_S,
    compute_recoverable_power_kw,
)
from twinlink_ml.heatnetwork.config import DEFAULT_COLD_TEMPERATURE_CELSIUS
from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    PlaceholderConfigWarning,
    load_process_config,
)
from twinlink_ml.simulation.backend import NumpyBackend
from twinlink_ml.simulation.exhaust import reconstruct_exhaust_state
from twinlink_ml.simulation.sorption import isotherm_from_material
from twinlink_ml.simulation.steady_state import solve_steady_state
from twinlink_ml.simulation.thermo import (
    absolute_humidity_kg_per_kg,
    latent_heat_j_per_kg,
    saturation_humidity_kg_per_kg,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "study_humidity_channel_uncertainty.json"

LAB_MASS_FLOW_LABEL = "lab_simulator"
DISTRICT_MASS_FLOW_LABEL = "district_partner_model"

# --- MA2a: verwendete Fehler je Kanal, mit Quelle ----------------------------------------
#
# Quelle: artifacts/runs/arch_learning_curve_constant_trained.jsonl (Befund 55, Commit
# 1b85ce4), Modell LSTM (bestes Modell je Kanal an diesem Budgetpunkt, siehe unten),
# n_train=64 (groesstes getestetes Budget -- die guenstigste Naeherung an "genug
# Betriebsdaten"), test_set="constant_holdout" (Konstantlauf-Holdout, HOLDOUT_SEED=999 --
# der betriebsrelevante, weil er NICHT auf Anregungsexperimente angewiesen ist, siehe
# Befund 55 Anlass), Horizont h=36, Median ueber 5 Seeds von
# "rmse_run_median_physical/<kanal>/h36". Bewusst NICHT die "own_pool"/"excited"-Testmenge
# (die pruefte Generalisierung auf angeregte Laeufe, hier irrelevant).
BED_TEMPERATURE_RMSE_K = 0.5393090806307859
"""Median (5 Seeds) von rmse_run_median_physical/bed_temperature_celsius/h36, LSTM,
n_train=64, test_set=constant_holdout, arch_learning_curve_constant_trained.jsonl."""

PRODUCT_MOISTURE_RMSE_KG_PER_KG = 0.0014883589441298013
"""Median (5 Seeds) von rmse_run_median_physical/product_moisture_kg_per_kg/h36, LSTM,
n_train=64, test_set=constant_holdout, arch_learning_curve_constant_trained.jsonl."""


@dataclass(frozen=True, slots=True)
class ReferencePoint:
    """Referenzbetriebspunkt (Nominalpunkt) samt rekonstruiertem Abluftzustand."""

    bed_temperature_celsius: float
    particle_moisture_kg_per_kg: float
    dry_air_mass_flow_kg_per_s: float
    inlet_absolute_humidity_kg_per_kg: float
    outlet_temperature_celsius: float
    outlet_humidity_kg_per_kg: float


def _build_reference_point() -> ReferencePoint:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    simulator = GreyBoxFluidBedGranulator(config)
    solution = solve_steady_state(simulator)
    if not solution.feasible:
        raise RuntimeError(
            f"Nominalpunkt nicht zulaessig: {solution.infeasibility_reasons} -- "
            "MA2 braucht einen gueltigen Referenzpunkt."
        )

    backend = NumpyBackend()
    inlet_absolute_humidity = float(
        backend.to_numpy(
            absolute_humidity_kg_per_kg(
                backend,
                config.gas,
                backend.asarray(config.disturbances.ambient_temperature_celsius.nominal),
                backend.asarray(config.disturbances.ambient_relative_humidity.nominal),
            )
        )
    )

    isotherm = isotherm_from_material(config.material)
    t_gas_out, y_gas_out, outlet_temperature = reconstruct_exhaust_state(
        backend,
        config.gas,
        config.material,
        isotherm,
        config.exhaust,
        bed_temperature_celsius=backend.asarray(solution.bed_temperature_celsius),
        particle_moisture_kg_per_kg=backend.asarray(solution.particle_moisture_kg_per_kg),
        dry_air_mass_flow_kg_per_s=backend.asarray(solution.dry_air_mass_flow_kg_per_s),
        inlet_absolute_humidity_kg_per_kg=backend.asarray(inlet_absolute_humidity),
    )
    del t_gas_out  # == bed_temperature_celsius per Modellannahme, hier nicht gebraucht

    # Sanity-Check: reconstruct_exhaust_state() muss den Loeser bitgleich reproduzieren
    # (siehe exhaust.py-Moduldocstring, "Gegen solve_steady_state() ... bitgleiche
    # Uebereinstimmung").
    reconstructed_y = float(backend.to_numpy(y_gas_out))
    if abs(reconstructed_y - solution.outlet_humidity_kg_per_kg) > 1e-9:
        raise RuntimeError(
            f"reconstruct_exhaust_state()={reconstructed_y!r} weicht vom Loeser "
            f"({solution.outlet_humidity_kg_per_kg!r}) ab -- Referenzpunkt inkonsistent."
        )

    return ReferencePoint(
        bed_temperature_celsius=solution.bed_temperature_celsius,
        particle_moisture_kg_per_kg=solution.particle_moisture_kg_per_kg,
        dry_air_mass_flow_kg_per_s=solution.dry_air_mass_flow_kg_per_s,
        inlet_absolute_humidity_kg_per_kg=inlet_absolute_humidity,
        outlet_temperature_celsius=float(backend.to_numpy(outlet_temperature)),
        outlet_humidity_kg_per_kg=reconstructed_y,
    )


def _reconstruct(
    reference: ReferencePoint,
    config_gas: object,
    config_material: object,
    config_exhaust: object,
    isotherm: object,
    *,
    bed_temperature_celsius: float,
    particle_moisture_kg_per_kg: float,
) -> tuple[float, float]:
    """Abluft-Zustand bei gestoerter Betttemperatur/Produktfeuchte, sonst unveraendert."""
    backend = NumpyBackend()
    _t_gas_out, y_out, t_out = reconstruct_exhaust_state(
        backend,
        config_gas,  # type: ignore[arg-type]
        config_material,  # type: ignore[arg-type]
        isotherm,  # type: ignore[arg-type]
        config_exhaust,  # type: ignore[arg-type]
        bed_temperature_celsius=backend.asarray(bed_temperature_celsius),
        particle_moisture_kg_per_kg=backend.asarray(particle_moisture_kg_per_kg),
        dry_air_mass_flow_kg_per_s=backend.asarray(reference.dry_air_mass_flow_kg_per_s),
        inlet_absolute_humidity_kg_per_kg=backend.asarray(
            reference.inlet_absolute_humidity_kg_per_kg
        ),
    )
    return float(backend.to_numpy(t_out)), float(backend.to_numpy(y_out))


def _q_rec_condensing_kw(
    gas: object,
    outlet_temperature_celsius: float,
    outlet_humidity_kg_per_kg: float,
    *,
    mass_flow_kg_per_s: float,
    cold_temperature_celsius: float,
) -> float:
    """MA1-Formel (``heatnetwork.balance``) direkt aus ``thermo.py`` -- ein Einzelpunkt
    braucht keine Kaskade, siehe Moduldocstring."""
    backend = NumpyBackend()
    y_sat_cold = float(
        backend.to_numpy(
            saturation_humidity_kg_per_kg(
                backend,
                gas,  # type: ignore[arg-type]
                backend.asarray(cold_temperature_celsius),
            )
        )
    )
    condensate_kg_per_s = max(0.0, outlet_humidity_kg_per_kg - y_sat_cold) * mass_flow_kg_per_s
    dh_v = float(
        backend.to_numpy(latent_heat_j_per_kg(backend, backend.asarray(cold_temperature_celsius)))
    )
    q_latent_w = condensate_kg_per_s * dh_v
    q_sensible_w = (
        mass_flow_kg_per_s
        * (outlet_temperature_celsius - cold_temperature_celsius)
        * (gas.cp_dry_air_j_per_kg_k + outlet_humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k)  # type: ignore[attr-defined]
    )
    return (q_sensible_w + q_latent_w) / 1000.0


def main() -> None:
    reference = _build_reference_point()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    isotherm = isotherm_from_material(config.material)

    print(
        f"Referenzpunkt: T_bed={reference.bed_temperature_celsius:.3f} C, "
        f"X_p={reference.particle_moisture_kg_per_kg:.5f} kg/kg, "
        f"m_L={reference.dry_air_mass_flow_kg_per_s:.5f} kg/s (Labor), "
        f"Y_in={reference.inlet_absolute_humidity_kg_per_kg:.5f} kg/kg\n"
        f"  -> T_out={reference.outlet_temperature_celsius:.3f} C, "
        f"Y_out={reference.outlet_humidity_kg_per_kg:.5f} kg/kg"
    )

    def q_rec_both_modes(
        t_out: float, y_out: float, mass_flow_kg_per_s: float
    ) -> tuple[float, float]:
        q_sensible = compute_recoverable_power_kw(
            NumpyBackend(),
            config.gas,
            t_out,
            y_out,
            mass_flow_kg_per_s=mass_flow_kg_per_s,
            dew_point_margin_k=DEW_POINT_MARGIN_K,
        )
        q_condensing = _q_rec_condensing_kw(
            config.gas,
            t_out,
            y_out,
            mass_flow_kg_per_s=mass_flow_kg_per_s,
            cold_temperature_celsius=DEFAULT_COLD_TEMPERATURE_CELSIUS,
        )
        return float(q_sensible), float(q_condensing)

    mass_flow_scales = (
        (LAB_MASS_FLOW_LABEL, reference.dry_air_mass_flow_kg_per_s),
        (DISTRICT_MASS_FLOW_LABEL, DISTRICT_MASS_FLOW_KG_PER_S),
    )

    baseline_by_scale = {
        label: q_rec_both_modes(
            reference.outlet_temperature_celsius, reference.outlet_humidity_kg_per_kg, flow
        )
        for label, flow in mass_flow_scales
    }

    scenarios: dict[str, tuple[float, float]] = {
        "temperature_plus": (BED_TEMPERATURE_RMSE_K, 0.0),
        "temperature_minus": (-BED_TEMPERATURE_RMSE_K, 0.0),
        "moisture_plus": (0.0, PRODUCT_MOISTURE_RMSE_KG_PER_KG),
        "moisture_minus": (0.0, -PRODUCT_MOISTURE_RMSE_KG_PER_KG),
        "both_pp": (BED_TEMPERATURE_RMSE_K, PRODUCT_MOISTURE_RMSE_KG_PER_KG),
        "both_pm": (BED_TEMPERATURE_RMSE_K, -PRODUCT_MOISTURE_RMSE_KG_PER_KG),
        "both_mp": (-BED_TEMPERATURE_RMSE_K, PRODUCT_MOISTURE_RMSE_KG_PER_KG),
        "both_mm": (-BED_TEMPERATURE_RMSE_K, -PRODUCT_MOISTURE_RMSE_KG_PER_KG),
    }

    results: dict[str, object] = {
        "reference": {
            "bed_temperature_celsius": reference.bed_temperature_celsius,
            "particle_moisture_kg_per_kg": reference.particle_moisture_kg_per_kg,
            "dry_air_mass_flow_kg_per_s_lab": reference.dry_air_mass_flow_kg_per_s,
            "inlet_absolute_humidity_kg_per_kg": reference.inlet_absolute_humidity_kg_per_kg,
            "outlet_temperature_celsius": reference.outlet_temperature_celsius,
            "outlet_humidity_kg_per_kg": reference.outlet_humidity_kg_per_kg,
        },
        "errors": {
            "bed_temperature_rmse_k": BED_TEMPERATURE_RMSE_K,
            "product_moisture_rmse_kg_per_kg": PRODUCT_MOISTURE_RMSE_KG_PER_KG,
            "source": "artifacts/runs/arch_learning_curve_constant_trained.jsonl, LSTM, "
            "n_train=64, test_set=constant_holdout, h=36, Median 5 Seeds",
        },
        "baseline_kw": {
            label: {"sensible": s, "condensing": c} for label, (s, c) in baseline_by_scale.items()
        },
        "scenarios": {},
    }

    print(
        f"\n{'Szenario':<16}{'dT [K]':>9}{'dX_p [kg/kg]':>14}{'Y_out':>10}"
        f"{'dQ_sens[kW]':>13}{'dQ_sens[%]':>11}{'dQ_cond[kW]':>13}{'dQ_cond[%]':>11}  Skala"
    )
    for name, (dt, dx) in scenarios.items():
        t_out, y_out = _reconstruct(
            reference,
            config.gas,
            config.material,
            config.exhaust,
            isotherm,
            bed_temperature_celsius=reference.bed_temperature_celsius + dt,
            particle_moisture_kg_per_kg=reference.particle_moisture_kg_per_kg + dx,
        )
        results["scenarios"][name] = {
            "delta_bed_temperature_k": dt,
            "delta_particle_moisture_kg_per_kg": dx,
            "outlet_temperature_celsius": t_out,
            "outlet_humidity_kg_per_kg": y_out,
            "by_scale": {},
        }
        for label, flow in mass_flow_scales:
            q_sensible, q_condensing = q_rec_both_modes(t_out, y_out, flow)
            base_sensible, base_condensing = baseline_by_scale[label]
            d_sensible_kw = q_sensible - base_sensible
            d_condensing_kw = q_condensing - base_condensing
            d_sensible_pct = 100.0 * d_sensible_kw / base_sensible if base_sensible else float("nan")
            d_condensing_pct = (
                100.0 * d_condensing_kw / base_condensing if base_condensing else float("nan")
            )
            results["scenarios"][name]["by_scale"][label] = {
                "q_sensible_kw": q_sensible,
                "q_condensing_kw": q_condensing,
                "delta_sensible_kw": d_sensible_kw,
                "delta_sensible_pct": d_sensible_pct,
                "delta_condensing_kw": d_condensing_kw,
                "delta_condensing_pct": d_condensing_pct,
            }
            print(
                f"{name:<16}{dt:>9.3f}{dx:>14.6f}{y_out:>10.5f}"
                f"{d_sensible_kw:>13.4f}{d_sensible_pct:>10.2f}%"
                f"{d_condensing_kw:>13.4f}{d_condensing_pct:>10.2f}%  {label}"
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGeschrieben: {OUT_PATH}")


if __name__ == "__main__":
    main()

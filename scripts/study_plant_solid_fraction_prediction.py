r"""R2/R3 (Befund 56 Korrektur): NA5 mit korrigiertem Luftstrom neu gerechnet, plus
Feststoffanteil als VORHERSAGE statt als Anpassung.

Baut auf der R1-Korrektur in ``study_plant_validation.py`` auf (relative statt absolute
Feuchte in der Nm3/h-Umrechnung, Faktor ~2.2 im Luftmassenstrom -- siehe dortiger
Moduldocstring von ``_gas_velocity_m_per_s``). Zwei Teile:

R2
    NA5 mit dem jetzt korrigierten Luftstrom und UNVERAENDERT 30 % Feststoff neu
    gerechnet: Betttemperatur, Abweichung, Fluidisierungsmarge.

R3
    Bei gemessenem Luftstrom und gemessener Zulufttemperatur legt die Betttemperatur im
    Beharrungszustand im Wesentlichen den Wassergehalt der Suspension fest (die einzige
    andere freie Groesse ist die Bettmasse, die nur die Flaeche/Fluidisierung beeinflusst,
    nicht die Energiebilanz, siehe Moduldocstring von ``study_plant_validation``). Diese
    Rechnung LOEST deshalb nicht nach der Betttemperatur (die ist Messwert), sondern NACH
    dem Feststoffanteil, bei dem das Modell die gemessene Betttemperatur trifft -- eine
    VORHERSAGE, keine Anpassung: der Feststoffanteil war vorher unbekannt, die
    Betttemperatur ist der unabhaengige Messwert.

    Fuer P2 zusaetzlich: aus dem vorhergesagten Feststoffanteil und der im Leitsystem
    genannten Suspensionsdichte (1339 kg/m3, P2) die implizierte
    Feststoffdichte ableiten (ideale Volumenadditivitaet, Wasserdichte bei
    Suspensionstemperatur) -- ein erster, unabhaengiger Konsistenzhinweis, kein Beweis.

F1 (Befund 56 NB2-Erweiterung, extern angestossen)
    R3 schliesst die Energiebilanz ueber das BETT (Luft bis T_bett) -- Befund 56 NA4 zeigt
    aber, dass der Bett-Abluft-Versatz vermutlich Verdampfung OBERHALB des Betts ist, nicht
    ein reiner Waermeverlust. Dann ist die vollstaendige Bilanz eine ueber den GANZEN
    APPARAT (Luft bis zur gemessenen Ablufttemperatur `T_abluft`), nicht nur das Bett.
    Diese Bilanz braucht keinen Beharrungszustands-Loeser (keine Bettmasse, keine
    Fluidisierung) -- eine direkte algebraische Aufloesung nach dem Feststoffanteil, mit
    denselben Stoffwerten wie im Grey-Box-Modell (`ProcessConfig().gas`/`.material`), bei
    zwei Wandverlust-Annahmen: dem Modell-Platzhalter (`geometry.heat_loss_ua_w_per_k`,
    aktuell 2.0 W/K, ergibt < 0.2 kW) und einer testweisen 30-kW-Annahme.

Rohdaten bleiben ausserhalb des Repos (NA1) -- Pfade ausschliesslich aus
``TWINLINK_PLANT_DATA_P1`` / ``_P2``, wie in ``study_plant_validation.py``.

Aufruf::

    python scripts/study_plant_solid_fraction_prediction.py \
        --output artifacts/runs/study_plant_solid_fraction_prediction.json
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
import study_plant_validation as spv  # noqa: E402

from twinlink_ml.simulation import GreyBoxFluidBedGranulator  # noqa: E402
from twinlink_ml.simulation.backend import NumpyBackend  # noqa: E402
from twinlink_ml.simulation.config import DisturbanceRange, ProcessConfig  # noqa: E402
from twinlink_ml.simulation.thermo import (  # noqa: E402
    absolute_humidity_kg_per_kg,
    latent_heat_j_per_kg,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "runs" / "study_plant_solid_fraction_prediction.json"

_SOLID_FRACTION_GRID = tuple(round(0.05 + 0.025 * i, 4) for i in range(37))  # 0.05 .. 0.95
_WATER_DENSITY_KG_PER_M3 = 998.0
"""Bei ~20 degC (Suspensionstemperatur-Annahme), fuer die Feststoffdichte-Rueckrechnung."""
_MEASURED_SUSPENSION_DENSITY_P2_KG_PER_M3 = 1339.0
"""Leitsystem-Anzeige, P2 -- Quelle: Screenshot, nicht Teil der Exporte."""
_WALL_LOSS_SENSITIVITY_KW = 30.0
"""F1: testweise Wandverlust-Annahme, deutlich ueber dem Modell-Platzhalter (< 0.2 kW)."""

_BACKEND = NumpyBackend()
_BASE_CONFIG = ProcessConfig()


def _whole_apparatus_solid_fraction(
    point: spv.OperatingPoint, *, wall_loss_kw: float
) -> dict[str, object]:
    """F1: Feststoffanteil aus einer direkten Energiebilanz ueber den GANZEN APPARAT
    (Luft von `T_zuluft` bis zur gemessenen `T_abluft`), statt nur ueber das Bett wie in
    :func:`_predict_solid_fraction`. Kein Beharrungszustands-Loeser -- die Bilanz ist
    algebraisch nach dem Feststoffanteil aufloesbar, sobald `T_abluft` (statt `T_bett`)
    die Referenztemperatur ist. Nimmt nahezu vollstaendige Verdampfung an (das im
    Bett-Fall vorhergesagte Restfeuchte-Niveau ist < 1.5 % des Spruehwassers, Befund 56
    NB2/R3) -- diese Vereinfachung wird hier nicht erneut geprueft.
    """
    absolute_humidity = float(
        _BACKEND.to_numpy(
            absolute_humidity_kg_per_kg(
                _BACKEND,
                _BASE_CONFIG.gas,
                _BACKEND.asarray(np.array([point.ambient_temperature_c])),
                _BACKEND.asarray(np.array([spv._ASSUMED_AMBIENT_RELATIVE_HUMIDITY])),
            )
        )[0]
    )
    normal_density = _BASE_CONFIG.gas.pressure_pa / (
        _BASE_CONFIG.gas.r_specific_dry_air_j_per_kg_k * 273.15
    )
    dry_air_flow_kg_per_s = point.process_air_flow_nm3_per_h / 3600.0 * normal_density
    cp_moist = (
        _BASE_CONFIG.gas.cp_dry_air_j_per_kg_k
        + absolute_humidity * _BASE_CONFIG.gas.cp_vapour_j_per_kg_k
    )
    spray_rate_kg_per_s = point.spray_rate_kg_per_h / 3600.0
    cp_susp = _BASE_CONFIG.material.cp_suspension_j_per_kg_k
    t_susp = spv._ASSUMED_SUSPENSION_TEMPERATURE_C
    t_out = point.outlet_temperature_measured_c
    t_in = point.inlet_temperature_c

    latent = float(
        _BACKEND.to_numpy(latent_heat_j_per_kg(_BACKEND, _BACKEND.asarray(np.array([t_out]))))[0]
    )
    heat_from_air_w = dry_air_flow_kg_per_s * cp_moist * (t_in - t_out)
    heat_from_suspension_w = spray_rate_kg_per_s * cp_susp * (t_susp - t_out)
    wall_loss_w = wall_loss_kw * 1000.0

    evaporation_kg_per_s = (heat_from_air_w + heat_from_suspension_w - wall_loss_w) / latent
    solid_fraction = 1.0 - evaporation_kg_per_s / spray_rate_kg_per_s

    return {
        "point": point.label,
        "t_in_celsius": t_in,
        "t_abluft_measured_celsius": t_out,
        "wall_loss_kw": wall_loss_kw,
        "heat_from_air_kw": heat_from_air_w / 1000.0,
        "heat_from_suspension_kw": heat_from_suspension_w / 1000.0,
        "latent_heat_at_t_abluft_j_per_kg": latent,
        "evaporation_kg_per_h": evaporation_kg_per_s * 3600.0,
        "solid_fraction": solid_fraction,
    }


def _solve_at_solid_fraction(
    point: spv.OperatingPoint, *, bed_mass_kg: float, solid_fraction: float
) -> tuple[float | None, dict[str, float] | None, str | None]:
    """Beharrungszustand bei gegebenem Feststoffanteil -- korrigierter Luftstrom (R1),
    sonst unveraendertes NA5-Protokoll. Gibt (T_bed, Diagnostics, Fehlermeldung) zurueck,
    genau eines der ersten beiden ist ``None``."""
    area = bed_mass_kg * 9.81 / spv._BED_PRESSURE_DROP_PA
    base_config = spv.build_plant_config(
        point, cross_section_area_m2=area, bed_mass_solid_setpoint_kg=bed_mass_kg
    )
    widened_disturbances = dataclasses.replace(
        base_config.disturbances,
        suspension_solid_fraction=DisturbanceRange(0.02, 0.98, solid_fraction),
    )
    config = dataclasses.replace(base_config, disturbances=widened_disturbances)
    simulator = GreyBoxFluidBedGranulator(config)
    try:
        solution, _widened_low = spv._solve_steady_state_widening_bracket(simulator, point)
    except RuntimeError as exc:
        return None, None, str(exc)
    return solution.bed_temperature_celsius, dict(solution.diagnostics), None


def _predict_solid_fraction(point: spv.OperatingPoint, *, bed_mass_kg: float) -> dict[str, object]:
    """R3: Feststoffanteil-Vorhersage per Gitter-Scan + lineare Interpolation zwischen den
    beiden Gitterpunkten, die die gemessene Betttemperatur einschliessen. Kein Newton/
    Bisektion ueber eine moeglicherweise unstetige Erreichbarkeit (manche Feststoffanteile
    liefern gar keinen Beharrungszustand, siehe R2-Ergebnis bei P2) -- ein Gitter macht
    sichtbar, WO die Kurve abbricht, statt einen Loeser dort blind suchen zu lassen."""
    scan: list[dict[str, object]] = []
    for solid_fraction in _SOLID_FRACTION_GRID:
        t_bed, diagnostics, error = _solve_at_solid_fraction(
            point, bed_mass_kg=bed_mass_kg, solid_fraction=solid_fraction
        )
        scan.append(
            {
                "solid_fraction": solid_fraction,
                "bed_temperature_celsius": t_bed,
                "particle_moisture_kg_per_kg": diagnostics["particle_moisture_kg_per_kg"]
                if diagnostics
                else None,
                "error": error,
            }
        )

    measured = point.bed_temperature_measured_c
    solved = [row for row in scan if row["bed_temperature_celsius"] is not None]
    bracket: tuple[dict[str, object], dict[str, object]] | None = None
    for lower, upper in itertools.pairwise(solved):
        t_lo, t_hi = lower["bed_temperature_celsius"], upper["bed_temperature_celsius"]
        if t_lo <= measured <= t_hi:
            bracket = (lower, upper)
            break

    if bracket is None:
        return {
            "point": point.label,
            "bed_mass_kg": bed_mass_kg,
            "scan": scan,
            "predicted_solid_fraction": None,
            "note": (
                "Gemessene Betttemperatur liegt ausserhalb des im Gitter erreichbaren "
                "Bereichs (Beharrungszustand existiert nicht bei jedem Feststoffanteil) "
                "-- keine Vorhersage moeglich, nur die Kurve selbst ist das Ergebnis."
            ),
        }

    lower, upper = bracket
    t_lo, t_hi = lower["bed_temperature_celsius"], upper["bed_temperature_celsius"]
    weight = (measured - t_lo) / (t_hi - t_lo) if t_hi != t_lo else 0.0
    predicted_solid_fraction = lower["solid_fraction"] + weight * (
        upper["solid_fraction"] - lower["solid_fraction"]
    )
    predicted_moisture = lower["particle_moisture_kg_per_kg"] + weight * (
        upper["particle_moisture_kg_per_kg"] - lower["particle_moisture_kg_per_kg"]
    )
    return {
        "point": point.label,
        "bed_mass_kg": bed_mass_kg,
        "scan": scan,
        "bracket_solid_fraction": [lower["solid_fraction"], upper["solid_fraction"]],
        "bracket_bed_temperature_celsius": [t_lo, t_hi],
        "predicted_solid_fraction": predicted_solid_fraction,
        "predicted_particle_moisture_kg_per_kg": predicted_moisture,
        "note": "Lineare Interpolation zwischen den beiden einschliessenden Gitterpunkten.",
    }


def _implied_solid_density(
    solid_fraction: float,
    *,
    suspension_density_kg_per_m3: float = _MEASURED_SUSPENSION_DENSITY_P2_KG_PER_M3,
) -> float:
    r"""Ideale Volumenadditivitaet: 1/rho_susp = w/rho_solid + (1-w)/rho_water,
    aufgeloest nach rho_solid."""
    inverse = (1.0 / suspension_density_kg_per_m3) - (
        (1.0 - solid_fraction) / _WATER_DENSITY_KG_PER_M3
    )
    if inverse <= 0.0:
        return float("inf")
    return solid_fraction / inverse


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    paths = {
        "P1": os.environ.get("TWINLINK_PLANT_DATA_P1"),
        "P2": os.environ.get("TWINLINK_PLANT_DATA_P2"),
    }
    missing = [label for label, path in paths.items() if not path]
    if missing:
        print(
            "Fehlende Umgebungsvariable(n) fuer: "
            + ", ".join(f"TWINLINK_PLANT_DATA_{label}" for label in missing),
            file=sys.stderr,
        )
        return 1

    results: dict[str, object] = {
        "r2_corrected_at_30_percent": {},
        "r3_predicted_solid_fraction": {},
        "f1_whole_apparatus_balance": {},
    }

    for label, path in paths.items():
        point = spv.load_operating_point(label, path)

        print(
            f"\n{'=' * 88}\n{label} -- R2: korrigierter Luftstrom, unveraendert 30% Feststoff\n{'=' * 88}"
        )
        r2_by_mass: dict[str, object] = {}
        for bed_mass in spv._BED_MASS_CANDIDATES_KG:
            t_bed, diagnostics, error = _solve_at_solid_fraction(
                point, bed_mass_kg=bed_mass, solid_fraction=spv._ASSUMED_SUSPENSION_SOLID_FRACTION
            )
            if error is not None:
                print(f"  m_bed={bed_mass:.0f} kg: KEIN Beharrungszustand ({error})")
                r2_by_mass[str(bed_mass)] = {"error": error}
                continue
            deviation = t_bed - point.bed_temperature_measured_c
            u_gas = None
            margin_lower = margin_upper = None
            area = bed_mass * 9.81 / spv._BED_PRESSURE_DROP_PA
            config = spv.build_plant_config(
                point, cross_section_area_m2=area, bed_mass_solid_setpoint_kg=bed_mass
            )
            u_gas = config.actions.gas_velocity_m_per_s.nominal
            margin_lower = diagnostics["fluidization_margin_lower_m_per_s"]
            margin_upper = diagnostics["fluidization_margin_upper_m_per_s"]
            feasible = margin_lower >= 0.0 and margin_upper >= 0.0
            print(
                f"  m_bed={bed_mass:.0f} kg: T_bed_modell={t_bed:.2f} degC "
                f"(Messung {point.bed_temperature_measured_c:.2f}, Abw {deviation:+.2f} K), "
                f"u_gas={u_gas:.3f} m/s, Marge unten={margin_lower:+.3f} m/s, "
                f"Marge oben={margin_upper:+.3f} m/s, fluidisierungs-zulaessig={feasible}"
            )
            r2_by_mass[str(bed_mass)] = {
                "bed_temperature_celsius": t_bed,
                "deviation_k": deviation,
                "gas_velocity_m_per_s": u_gas,
                "fluidization_margin_lower_m_per_s": margin_lower,
                "fluidization_margin_upper_m_per_s": margin_upper,
                "fluidization_feasible": feasible,
                "particle_moisture_kg_per_kg": diagnostics["particle_moisture_kg_per_kg"],
            }
        results["r2_corrected_at_30_percent"][label] = r2_by_mass

        print(
            f"\n{'-' * 88}\n{label} -- R3: Feststoffanteil als Vorhersage (m_bed=800 kg)\n{'-' * 88}"
        )
        prediction = _predict_solid_fraction(point, bed_mass_kg=800.0)
        if prediction["predicted_solid_fraction"] is not None:
            print(
                f"  Vorhergesagter Feststoffanteil: {prediction['predicted_solid_fraction']:.3f} "
                f"(Gitterklammer {prediction['bracket_solid_fraction']}), "
                f"vorhergesagte Partikelfeuchte: "
                f"{prediction['predicted_particle_moisture_kg_per_kg']:.4f} kg/kg"
            )
            if label == "P2":
                rho_solid = _implied_solid_density(prediction["predicted_solid_fraction"])
                plausible = 1300.0 <= rho_solid <= 1800.0
                print(
                    f"  Implizierte Feststoffdichte (aus Suspensionsdichte "
                    f"{_MEASURED_SUSPENSION_DENSITY_P2_KG_PER_M3:.0f} kg/m3): "
                    f"{rho_solid:.0f} kg/m3 -- {'PLAUSIBEL' if plausible else 'NICHT plausibel'} "
                    "fuer eine organische Formulierung (1300-1800 kg/m3)"
                )
                prediction["implied_solid_density_kg_per_m3"] = rho_solid
                prediction["implied_solid_density_plausible"] = plausible
        else:
            print(f"  {prediction['note']}")
        results["r3_predicted_solid_fraction"][label] = prediction

        print(
            f"\n{'-' * 88}\n{label} -- F1: Feststoffanteil ueber den GANZEN APPARAT "
            f"(Luft bis T_abluft)\n{'-' * 88}"
        )
        f1_by_wall_loss: dict[str, object] = {}
        model_ua_w_per_k = _BASE_CONFIG.geometry.heat_loss_ua_w_per_k
        baseline_wall_loss_kw = (
            model_ua_w_per_k
            * (point.outlet_temperature_measured_c - point.ambient_temperature_c)
            / 1000.0
        )
        for wall_loss_kw in (baseline_wall_loss_kw, _WALL_LOSS_SENSITIVITY_KW):
            whole = _whole_apparatus_solid_fraction(point, wall_loss_kw=wall_loss_kw)
            print(
                f"  Wandverlust={wall_loss_kw:6.3f} kW: T_abluft={whole['t_abluft_measured_celsius']:.2f} degC, "
                f"Verdampfung={whole['evaporation_kg_per_h']:.1f} kg/h, "
                f"Feststoffanteil={whole['solid_fraction']:.4f} ({whole['solid_fraction'] * 100:.1f} %)"
            )
            if label == "P2":
                rho_solid = _implied_solid_density(whole["solid_fraction"])
                whole["implied_solid_density_kg_per_m3"] = rho_solid
                whole["implied_solid_density_plausible"] = 1300.0 <= rho_solid <= 1800.0
                print(f"    implizierte Feststoffdichte: {rho_solid:.0f} kg/m3")
            f1_by_wall_loss[f"{wall_loss_kw:.3f}"] = whole
        results["f1_whole_apparatus_balance"][label] = f1_by_wall_loss

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(results, indent=2, sort_keys=True, default=str))
    print(f"\nErgebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

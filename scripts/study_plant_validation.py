r"""NA3-NA5: first validation of the grey-box model against REAL plant data.

No network training, no RL, no sweep -- two 24h exports from the application partner
(one-minute interval, 1441 rows each, products "prod1"/"prod2", see ``twinlink_ml.plant_data``)
against the exact steady-state solver (``simulation.steady_state``).

Raw data stay outside the repo (NA1, see ``docs/04_stand_ap1.md``): the
two paths come exclusively from the environment variables
``TWINLINK_PLANT_DATA_PROD1`` / ``TWINLINK_PLANT_DATA_PROD2``, nowhere as a literal.

Three parts
----------
1. NA3: upper temperature position (TRZSAH-320417/418/419) vs. bed temperature
   (TIRSAH-320414/415/416) -- time-series statistics, no model run.
2. NA4: ``outlet_duct_heat_loss_k`` -- bed-to-exhaust offset as a time series, correlation
   with air flow/spray rate/ambient temperature, no model run.
3. NA5: steady state of the grey-box model with plant parameters per product,
   comparison of the computed with the measured bed temperature. Parameters are NOT
   adjusted to hit the measurement -- all assumptions are in
   :data:`ASSUMPTIONS` with justification, unchanged between product 1 and product 2.

Geometry assumption (cross-section area)
----------------------------------------
Not measured in the exports. Estimated from the control-system screenshot (third
point in time, 05.-08.09.2025, NOT part of the validated exports) via
``dp_bed = m_bed * g / A`` -- the same formula as
``simulation.fluidization.bed_pressure_drop_pa``. PDIRAH-320404 shows 19.2 mbar there,
consistent with the control-system display "Betthoehe: 20 mbar". Bed mass according to the
screenshot (A-3215) 600 kg; the application partner stated 800 kg orally -- both values are
computed (sensitivity), neither is "the right one".

Important: the area does NOT enter the ENERGY BALANCE (``gas_velocity_m_per_s *
cross_section_area_m2`` cancels exactly against the conversion from Nm3/h, see
:func:`_gas_velocity_m_per_s`) -- it acts exclusively on the
fluidization feasibility check (u_mf < u_gas < u_terminal) and the reported
bed pressure drop. The computed bed temperature is INDEPENDENT of this assumption.
"""

from __future__ import annotations

import dataclasses
import math
import os
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from twinlink_ml.plant_data import (
    ambient_temperature_tag,
    bed_temperature_tags,
    inlet_temperature_tags,
    load_plant_export,
    outlet_temperature_tag,
    process_air_flow_tag,
    spray_rate_tag,
    upper_position_temperature_tags,
)
from twinlink_ml.simulation import GreyBoxFluidBedGranulator, NumpyBackend, ProcessConfig
from twinlink_ml.simulation.steady_state import SteadyStateSolution, solve_steady_state
from twinlink_ml.simulation.thermo import absolute_humidity_kg_per_kg, gas_density_kg_per_m3

_BACKEND = NumpyBackend()

# -- NA5: geometric/material assumptions, see module docstring and ASSUMPTIONS -----
_BED_PRESSURE_DROP_PA = 1920.0
"""19.2 mbar, PDIRAH-320404 im Screenshot (dritter Zeitpunkt, nicht Teil der Exporte)."""
_BED_MASS_CANDIDATES_KG = (600.0, 800.0)
"""A-3215-Anzeige (600 kg) vs. muendliche Angabe des Partners (800 kg) -- beide gerechnet."""
_ASSUMED_AMBIENT_RELATIVE_HUMIDITY = 0.75
"""Nicht gemessen (nur Temperatur, TIR-322412). Herbst/Fruehjahr-Platzhalter, siehe
ASSUMPTIONS -- KEINE Quelle, explizit als Annahme markiert."""
_ASSUMED_SUSPENSION_SOLID_FRACTION = 0.30
"""= Nominalwert von ``DisturbanceSpaceConfig.suspension_solid_fraction`` (unveraendert
uebernommen, keine Anlagenmessung vorhanden -- DIR-325160 ist nicht exportiert)."""
_ASSUMED_SUSPENSION_TEMPERATURE_C = 20.0
"""= Default von ``OperationConfig.suspension_temperature_celsius``. Nicht gemessen;
Innenraum-/Ansatztemperatur plausibler als Aussenlufttemperatur (TIR-322412)."""


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """An operating point, averaged from the export columns (24h)."""

    label: str
    ambient_temperature_c: float
    inlet_temperature_c: float
    bed_temperature_measured_c: float
    upper_position_measured_c: float
    outlet_temperature_measured_c: float
    process_air_flow_nm3_per_h: float
    spray_rate_kg_per_h: float


def _column_mean(columns: dict[str, tuple[float, ...]], tags: tuple[str, ...]) -> float:
    return float(np.mean([columns[tag] for tag in tags]))


def load_operating_point(label: str, path: str) -> OperatingPoint:
    series = load_plant_export(path)
    return OperatingPoint(
        label=label,
        ambient_temperature_c=_column_mean(series.columns, (ambient_temperature_tag(),)),
        inlet_temperature_c=_column_mean(series.columns, inlet_temperature_tags()),
        bed_temperature_measured_c=_column_mean(series.columns, bed_temperature_tags()),
        upper_position_measured_c=_column_mean(series.columns, upper_position_temperature_tags()),
        outlet_temperature_measured_c=_column_mean(series.columns, (outlet_temperature_tag(),)),
        process_air_flow_nm3_per_h=_column_mean(series.columns, (process_air_flow_tag(),)),
        spray_rate_kg_per_h=_column_mean(series.columns, (spray_rate_tag(),)),
    )


# ---------------------------------------------------------------------------------
# NA3: upper position vs. bed temperature -- pure time-series statistics
# ---------------------------------------------------------------------------------


def report_na3_upper_position(label: str, columns: dict[str, tuple[float, ...]]) -> None:
    bed = [
        st.mean(values)
        for values in zip(*(columns[tag] for tag in bed_temperature_tags()), strict=False)
    ]
    upper = [
        st.mean(values)
        for values in zip(
            *(columns[tag] for tag in upper_position_temperature_tags()), strict=False
        )
    ]
    diff = [u - b for u, b in zip(upper, bed, strict=True)]
    upper_spread = [
        max(values) - min(values)
        for values in zip(
            *(columns[tag] for tag in upper_position_temperature_tags()), strict=False
        )
    ]
    bed_spread = [
        max(values) - min(values)
        for values in zip(*(columns[tag] for tag in bed_temperature_tags()), strict=False)
    ]
    print(
        f"[NA3 {label}] obere Position - Bett: mean={st.mean(diff):+.3f} K sd={st.pstdev(diff):.3f} K"
    )
    print(
        f"[NA3 {label}] Streuung ZWISCHEN den drei Sensoren je Trias (Median ueber die Messreihe): "
        f"obere Position {st.median(upper_spread):.3f} K, Bett {st.median(bed_spread):.3f} K"
    )


# ---------------------------------------------------------------------------------
# NA4: outlet_duct_heat_loss_k -- bed-to-exhaust offset
# ---------------------------------------------------------------------------------


def _pearson(x: list[float], y: list[float]) -> float:
    mx, my = st.mean(x), st.mean(y)
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=True))
    return sxy / (sx * sy)


def report_na4_outlet_duct_offset(label: str, columns: dict[str, tuple[float, ...]]) -> None:
    bed = [
        st.mean(values)
        for values in zip(*(columns[tag] for tag in bed_temperature_tags()), strict=False)
    ]
    outlet = list(columns[outlet_temperature_tag()])
    diff = [b - o for b, o in zip(bed, outlet, strict=True)]
    airflow = list(columns[process_air_flow_tag()])
    spray = list(columns[spray_rate_tag()])
    ambient = list(columns[ambient_temperature_tag()])
    print(
        f"[NA4 {label}] Bett-Abluft-Versatz: mean={st.mean(diff):.3f} K sd={st.pstdev(diff):.3f} K "
        f"min={min(diff):.3f} K max={max(diff):.3f} K"
    )
    print(
        f"[NA4 {label}] Korrelation Versatz~Luftmenge={_pearson(diff, airflow):+.3f}  "
        f"Versatz~Spruehrate={_pearson(diff, spray):+.3f}  "
        f"Versatz~Umgebungstemperatur={_pearson(diff, ambient):+.3f}"
    )


def report_na4_cv(label: str, columns: dict[str, tuple[float, ...]]) -> None:
    for name, tag in (
        ("Luftmenge", process_air_flow_tag()),
        ("Zulufttemperatur", inlet_temperature_tags()[0]),
        ("Spruehrate", spray_rate_tag()),
    ):
        values = columns[tag]
        cv_percent = st.pstdev(values) / st.mean(values) * 100.0
        print(f"[NA4/Befund 56 {label}] Variationskoeffizient {name}: {cv_percent:.3f} %")


# ---------------------------------------------------------------------------------
# NA5: steady state with plant parameters
# ---------------------------------------------------------------------------------

ASSUMPTIONS: tuple[str, ...] = (
    "absolute Feuchte Ansaugluft: nicht gemessen (nur TIR-322412, Temperatur) -- "
    f"angenommen relative Feuchte {_ASSUMED_AMBIENT_RELATIVE_HUMIDITY:.0%}, "
    "unbestaetigter Jahreszeit-Platzhalter, keine Quelle.",
    "Feststoffanteil der Suspension: DIR-325160 (Dichte) nicht exportiert, "
    f"Feststoffdichte ohnehin unbekannt -- Nominalwert der Config uebernommen "
    f"({_ASSUMED_SUSPENSION_SOLID_FRACTION:.0%}), nicht anlagenspezifisch.",
    "Produktmassenstrom: keine Messung -- im Beharrungszustand ohnehin ein "
    "Loesungsergebnis (= Feststoffeintrag bei m_s = bed_mass_solid_setpoint_kg), "
    "keine unabhaengige Eingangsgroesse. Ohne Messung bleibt die Feststoff-"
    "Verweilzeit/Partikelwachstums-Vorhersage unvalidiert.",
    "Geometrie (Querschnittsflaeche): nicht gemessen -- abgeschaetzt aus "
    "Bettdruckverlust (Screenshot, dritter Zeitpunkt) und Bettmasse, siehe "
    "Moduldocstring. Geht NICHT in die Energiebilanz/Betttemperatur ein.",
    "Partikeldichte, Keimdurchmesser, Nukleationsanteil: unveraendert Lab-Platzhalter "
    "aus MaterialConfig -- keine Anlagenmessung vorhanden, NICHT angepasst.",
    "Sorptionsisotherme, kritische Gutsfeuchte: unveraendert Lab-Platzhalter -- keine "
    "Anlagenmessung vorhanden, NICHT angepasst.",
    f"Suspensionstemperatur: nicht gemessen -- Config-Default "
    f"{_ASSUMED_SUSPENSION_TEMPERATURE_C:.0f} degC (Ansatztemperatur, nicht "
    "Aussenlufttemperatur) uebernommen.",
    "Keimzufuhr (seed_mass_flow_kg_per_h): auf 0 angenommen (Config-Default) -- kein "
    "Hinweis auf externe Keimzufuhr in den Daten/Screenshots.",
    "geometry.heat_loss_ua_w_per_k: Lab-Platzhalter (2.0 W/K) unveraendert -- bei "
    "typischen Betttemperatur-Umgebungstemperatur-Differenzen hier (30-80 K) ein "
    "Waermeverlust von <200 W, vernachlaessigbar gegen die Luftheizleistung "
    "(Groessenordnung 10-15 kW/K * ~60-80 K, siehe Ausgabe) -- nicht kalibriert, aber "
    "auch bei grober Fehleinschaetzung ohne spuerbaren Einfluss auf T_bed.",
)


def _gas_velocity_m_per_s(
    config: ProcessConfig,
    *,
    normal_flow_nm3_per_h: float,
    area_m2: float,
    bed_temperature_guess_c: float,
    ambient_temperature_c: float,
    ambient_relative_humidity: float,
) -> float:
    """Gas velocity (at bed conditions, model convention) from Nm3/h.

    Standard density of dry air from ``GasConfig`` (0 degC, ``gas.pressure_pa``), no
    second constant newly set here. ``bed_temperature_guess_c`` is the MEASURED
    bed temperature -- a calibration of the INPUT representation (known
    measured value air flow -> model convention "velocity at bed conditions"),
    NO fit to the result: the solution for T_bed stays free, see
    module docstring.

    **Correction (R1):** deliberately takes the RELATIVE humidity (``ambient_relative_humidity``,
    dimensionless, 0-1) plus the ambient temperature and converts internally via
    :func:`~twinlink_ml.simulation.thermo.absolute_humidity_kg_per_kg` into absolute
    humidity -- no second conversion in the caller. The first version expected an
    absolute humidity in kg/kg directly here, but was called with the relative humidity (0.75);
    this drove the derived dry-air mass flow by a factor of ~2.2 above the value that
    follows from standard volume flow times standard density (see
    ``test_gas_velocity_from_normal_flow_matches_mass_conservation`` in
    ``tests/test_study_plant_validation.py`` and Befund 56, correction).
    """
    absolute_humidity = float(
        _BACKEND.to_numpy(
            absolute_humidity_kg_per_kg(
                _BACKEND,
                config.gas,
                _BACKEND.asarray(np.array([ambient_temperature_c])),
                _BACKEND.asarray(np.array([ambient_relative_humidity])),
            )
        )[0]
    )
    normal_density = config.gas.pressure_pa / (config.gas.r_specific_dry_air_j_per_kg_k * 273.15)
    dry_air_flow_kg_per_s = normal_flow_nm3_per_h / 3600.0 * normal_density
    moist_air_flow_kg_per_s = dry_air_flow_kg_per_s * (1.0 + absolute_humidity)
    bed_density = float(
        _BACKEND.to_numpy(
            gas_density_kg_per_m3(
                _BACKEND,
                config.gas,
                _BACKEND.asarray(np.array([bed_temperature_guess_c])),
                _BACKEND.asarray(np.array([absolute_humidity])),
            )
        )[0]
    )
    return moist_air_flow_kg_per_s / (bed_density * area_m2)


def build_plant_config(
    point: OperatingPoint, *, cross_section_area_m2: float, bed_mass_solid_setpoint_kg: float
) -> ProcessConfig:
    base = ProcessConfig()
    geometry = dataclasses.replace(
        base.geometry,
        cross_section_area_m2=cross_section_area_m2,
        bed_mass_solid_setpoint_kg=bed_mass_solid_setpoint_kg,
    )
    operation = dataclasses.replace(
        base.operation,
        seed_mass_flow_kg_per_h=0.0,
        suspension_temperature_celsius=_ASSUMED_SUSPENSION_TEMPERATURE_C,
    )
    gas_velocity = _gas_velocity_m_per_s(
        base,
        normal_flow_nm3_per_h=point.process_air_flow_nm3_per_h,
        area_m2=cross_section_area_m2,
        bed_temperature_guess_c=point.bed_temperature_measured_c,
        ambient_temperature_c=point.ambient_temperature_c,
        ambient_relative_humidity=_ASSUMED_AMBIENT_RELATIVE_HUMIDITY,
    )
    actions = dataclasses.replace(
        base.actions,
        inlet_temperature_celsius=dataclasses.replace(
            base.actions.inlet_temperature_celsius,
            minimum=0.0,
            maximum=150.0,
            nominal=point.inlet_temperature_c,
            max_rate_per_s=1.0,
        ),
        gas_velocity_m_per_s=dataclasses.replace(
            base.actions.gas_velocity_m_per_s,
            minimum=0.0,
            maximum=max(6.0, gas_velocity * 1.5),
            nominal=gas_velocity,
            max_rate_per_s=1.0,
        ),
        spray_rate_kg_per_h=dataclasses.replace(
            base.actions.spray_rate_kg_per_h,
            minimum=0.0,
            maximum=3000.0,
            nominal=point.spray_rate_kg_per_h,
            max_rate_per_s=50.0,
        ),
    )
    disturbances = dataclasses.replace(
        base.disturbances,
        ambient_temperature_celsius=dataclasses.replace(
            base.disturbances.ambient_temperature_celsius,
            minimum=-15.0,
            maximum=40.0,
            nominal=point.ambient_temperature_c,
        ),
        ambient_relative_humidity=dataclasses.replace(
            base.disturbances.ambient_relative_humidity,
            nominal=_ASSUMED_AMBIENT_RELATIVE_HUMIDITY,
        ),
        suspension_solid_fraction=dataclasses.replace(
            base.disturbances.suspension_solid_fraction,
            nominal=_ASSUMED_SUSPENSION_SOLID_FRACTION,
        ),
    )
    return dataclasses.replace(
        base, geometry=geometry, operation=operation, actions=actions, disturbances=disturbances
    )


_BRACKET_LOW_STEP_K = 5.0
_BRACKET_LOW_MAX_OFFSET_K = 60.0


def _solve_steady_state_widening_bracket(
    simulator: GreyBoxFluidBedGranulator, point: OperatingPoint
) -> tuple[SteadyStateSolution, float | None]:
    """Solves the steady state, widening the LOWER bracket limit if needed.

    The solver default (``T_amb + 10``, see ``steady_state.py``) keeps the outer
    bisection in the drying regime -- calibrated at lab scale. At a very high
    spray rate (product 2: 2118 kg/h, >2x product 1) this distance is not enough: below a
    higher threshold the bed gets wetter on net at EVERY assumed particle moisture
    (no solution of the inner bisection). This is a NUMERICAL adjustment of the
    search range, NO change of the balance equations or of a physical
    parameter: it only extends where the root is searched for. If an
    extension becomes necessary, this function returns the lower limit actually used
    (otherwise ``None``) -- that belongs in the report, not hidden.
    """
    default_low = point.ambient_temperature_c + 10.0
    default_high = max(point.inlet_temperature_c, _ASSUMED_SUSPENSION_TEMPERATURE_C) + 5.0
    offset = 0.0
    while offset <= _BRACKET_LOW_MAX_OFFSET_K:
        low = default_low + offset
        try:
            solution = solve_steady_state(
                simulator, temperature_bracket_celsius=(low, default_high)
            )
        except RuntimeError:
            offset += _BRACKET_LOW_STEP_K
            continue
        return solution, (low if offset > 0.0 else None)
    raise RuntimeError(
        f"Kein Beharrungszustand bis Bracket-Untergrenze {default_low + offset:.1f} degC gefunden"
    )


_AIR_HEAT_PLAUSIBILITY_TOLERANCE = 0.10
"""R1: Schwelle fuer die Plausibilitaetspruefung unten -- genau die Pruefung, die den
Faktor-~2.2-Fehler in der Luftstromumrechnung per Hand aufgedeckt hat, jetzt automatisiert."""


def _check_air_heat_plausibility(
    point: OperatingPoint, config: ProcessConfig, solution: SteadyStateSolution
) -> None:
    """Compares the air heat following from the MEASUREMENT (m_L * cp * dT, with dT against
    the MEASURED bed temperature, m_L correctly derived from Nm3/h) with the air heat
    actually used by the model (m_L_modell * cp * dT, dT against the bed temperature
    FOUND BY THE MODEL). If the ratio deviates from 1 by more than
    :data:`_AIR_HEAT_PLAUSIBILITY_TOLERANCE`, a warning is issued -- not automatically
    corrected. This check would have reported the R1 error (factor ~2.2 in the air mass flow due to
    relative instead of absolute humidity) on the FIRST run.
    """
    absolute_humidity = float(
        _BACKEND.to_numpy(
            absolute_humidity_kg_per_kg(
                _BACKEND,
                config.gas,
                _BACKEND.asarray(np.array([point.ambient_temperature_c])),
                _BACKEND.asarray(np.array([_ASSUMED_AMBIENT_RELATIVE_HUMIDITY])),
            )
        )[0]
    )
    normal_density = config.gas.pressure_pa / (config.gas.r_specific_dry_air_j_per_kg_k * 273.15)
    measured_dry_air_flow_kg_per_s = point.process_air_flow_nm3_per_h / 3600.0 * normal_density
    cp_moist = (
        config.gas.cp_dry_air_j_per_kg_k + absolute_humidity * config.gas.cp_vapour_j_per_kg_k
    )

    measured_air_heat_kw = (
        measured_dry_air_flow_kg_per_s
        * cp_moist
        * (point.inlet_temperature_c - point.bed_temperature_measured_c)
        / 1000.0
    )
    model_air_heat_kw = (
        solution.dry_air_mass_flow_kg_per_s
        * cp_moist
        * (point.inlet_temperature_c - solution.bed_temperature_celsius)
        / 1000.0
    )
    if measured_air_heat_kw == 0.0:
        return
    relative_deviation = abs(model_air_heat_kw - measured_air_heat_kw) / abs(measured_air_heat_kw)
    if relative_deviation > _AIR_HEAT_PLAUSIBILITY_TOLERANCE:
        print(
            f"[NA5 {point.label}] WARNUNG (Plausibilitaet Luftwaerme): Modell "
            f"{model_air_heat_kw:.1f} kW gegen Messung {measured_air_heat_kw:.1f} kW "
            f"(Abweichung {relative_deviation:.0%}, Schwelle "
            f"{_AIR_HEAT_PLAUSIBILITY_TOLERANCE:.0%}) -- Luftmassenstrom/Temperaturbasis "
            "pruefen, bevor das Ergebnis als Treffer gelesen wird."
        )


def report_na5_steady_state(point: OperatingPoint) -> None:
    print(
        f"\n[NA5 {point.label}] gemessen: T_zuluft={point.inlet_temperature_c:.1f} degC, "
        f"T_bett={point.bed_temperature_measured_c:.1f} degC, "
        f"T_amb={point.ambient_temperature_c:.1f} degC, "
        f"Luftmenge={point.process_air_flow_nm3_per_h:.0f} Nm3/h, "
        f"Spruehrate={point.spray_rate_kg_per_h:.0f} kg/h"
    )
    for bed_mass in _BED_MASS_CANDIDATES_KG:
        area = bed_mass * 9.81 / _BED_PRESSURE_DROP_PA
        config = build_plant_config(
            point, cross_section_area_m2=area, bed_mass_solid_setpoint_kg=bed_mass
        )
        simulator = GreyBoxFluidBedGranulator(config)
        try:
            solution, widened_low = _solve_steady_state_widening_bracket(simulator, point)
        except RuntimeError as exc:
            print(f"[NA5 {point.label}] m_bed={bed_mass:.0f} kg: KEIN Beharrungszustand ({exc})")
            continue
        if widened_low is not None:
            print(
                f"[NA5 {point.label}] m_bed={bed_mass:.0f} kg: Standard-Bracket-Untergrenze "
                f"(T_amb+10={point.ambient_temperature_c + 10.0:.1f} degC) ohne Loesung -- "
                f"auf {widened_low:.1f} degC angehoben (siehe "
                "_solve_steady_state_widening_bracket-Docstring)."
            )
        _check_air_heat_plausibility(point, config, solution)
        deviation = solution.bed_temperature_celsius - point.bed_temperature_measured_c
        heat_from_air_kw_per_k = (
            solution.dry_air_mass_flow_kg_per_s
            * (
                config.gas.cp_dry_air_j_per_kg_k
                + solution.diagnostics["inlet_humidity_kg_per_kg"] * config.gas.cp_vapour_j_per_kg_k
            )
            / 1000.0
        )
        print(
            f"[NA5 {point.label}] m_bed={bed_mass:.0f} kg, A={area:.2f} m^2: "
            f"T_bett_modell={solution.bed_temperature_celsius:.1f} degC "
            f"(Messung {point.bed_temperature_measured_c:.1f} degC, "
            f"Abweichung {deviation:+.1f} K), feasible={solution.feasible}"
            + ("" if solution.feasible else f" [{'; '.join(solution.infeasibility_reasons)}]")
        )
        print(
            f"[NA5 {point.label}] m_bed={bed_mass:.0f} kg: Luftwaermestrom "
            f"{heat_from_air_kw_per_k:.2f} kW/K, d50={solution.diameter_m * 1e6:.0f} um, "
            f"Partikelfeuchte={solution.particle_moisture_kg_per_kg:.4f} kg/kg"
        )


def main() -> int:
    paths = {
        "prod1": os.environ.get("TWINLINK_PLANT_DATA_PROD1"),
        "prod2": os.environ.get("TWINLINK_PLANT_DATA_PROD2"),
    }
    missing = [label for label, path in paths.items() if not path]
    if missing:
        print(
            "Fehlende Umgebungsvariable(n) fuer: "
            + ", ".join(f"TWINLINK_PLANT_DATA_{label.upper()}" for label in missing)
            + " -- Rohdaten liegen ausserhalb des Repos (NA1), Pfad ist Konfiguration.",
            file=sys.stderr,
        )
        return 1

    print("=== Annahmen (NA5, vollstaendig) ===")
    for assumption in ASSUMPTIONS:
        print(f"- {assumption}")

    for label, path in paths.items():
        assert path is not None
        series = load_plant_export(Path(path))
        print(f"\n=== {label} ({len(series.timestamps)} Zeilen) ===")
        report_na3_upper_position(label, series.columns)
        report_na4_outlet_duct_offset(label, series.columns)
        report_na4_cv(label, series.columns)
        point = load_operating_point(label, path)
        report_na5_steady_state(point)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

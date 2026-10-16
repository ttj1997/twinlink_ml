r"""Leistungsaufnahme der Aggregate -- Erzeugung der Zielgröße ``dE``.

Die Enthalpiebilanz aus ``02_prozess_setup.md`` verknüpft ``dE`` mit ``T_in``, ``u_gas``
und ``m_spray``. Hier wird die *zugeführte* Seite aufgeschlüsselt, denn nur die ist
Zielgröße einer energieoptimalen Regelung:

.. math::
    \dot E = P_{\mathrm{Heizung}} + P_{\mathrm{Ventilator}}
             + P_{\mathrm{Pumpe}} + P_{\mathrm{Zerstäuberluft}}

Monotonie
---------
Die im Projekt geforderten Vorzeichen
:math:`\partial \dot E/\partial T_{in} > 0` und
:math:`\partial \dot E/\partial u_{gas} > 0` gelten hier **strukturell**: alle vier
Terme sind nichtnegativ und in ihren jeweiligen Stellgrößen monoton steigend. Damit ist
das Grey-Box-Modell eine belastbare Referenz, um zu prüfen, ob ein monotones Netz / ICNN
die Constraints tatsächlich erfüllt -- und nicht nur die Trainingsdaten reproduziert.

Wichtiger Vorbehalt für den spezifischen Verbrauch: bei
``target = "specific_kwh_per_kg"`` ist die Monotonie in :math:`T_{in}` nur solange
gegeben, wie der Produktmassenstrom nicht von :math:`T_{in}` abhängt. Im kontinuierlichen
Betrieb mit Massenregelung gilt das (der Feststoffeintrag hängt an ``m_spray`` und
``w_solid``). Sobald eine Qualitätsgrenze (Produktfeuchte) aktiv wird und die
Sprührate begrenzt, gilt es nicht mehr. Der Constraint ist also an die
Zielgrößendefinition gebunden, nicht universell.
"""

from __future__ import annotations

from dataclasses import dataclass

from .backend import Array, ArrayBackend
from .config import EnergyConfig, GasConfig, MaterialConfig
from .fluidization import bed_pressure_drop_pa
from .thermo import kelvin


@dataclass(frozen=True, slots=True)
class PowerBreakdown:
    """Aufschlüsselung der Leistungsaufnahme in kW. Für Diagnose und AP 3."""

    heater_kw: Array
    fan_kw: Array
    pump_kw: Array
    atomiser_kw: Array
    total_kw: Array
    recirculation_fan_kw: Array | None = None
    """Extra fan power of the exhaust recirculation duct, already contained in ``fan_kw``;
    ``None`` without recirculation."""


def power_breakdown_kw(
    backend: ArrayBackend,
    gas: GasConfig,
    material: MaterialConfig,
    energy: EnergyConfig,
    cross_section_area_m2: float,
    dry_air_mass_flow_kg_per_s: Array,
    inlet_humidity_kg_per_kg: Array,
    gas_density_kg_per_m3_value: Array,
    inlet_temperature_celsius: Array,
    gas_velocity_m_per_s: Array,
    spray_rate_kg_per_s: Array,
    nozzle_pressure_bar: Array,
    ambient_temperature_celsius: Array,
    bed_mass_total_kg: Array,
    *,
    heater_intake_temperature_celsius: Array | None = None,
    recirculation_fraction: float | None = None,
) -> PowerBreakdown:
    r"""Leistungsaufnahme aller Aggregate.

    Lufterhitzer
        .. math:: P_{H} = \frac{\dot m_L (c_{p,L} + Y_{in} c_{p,D})
                  (T_{in} - T_{amb})}{\eta_H}

    Ventilator
        .. math:: P_{V} = \frac{\dot V \left(\Delta p_{bed}
                  + \zeta \frac{\rho_g u^2}{2}\right)}{\eta_V},
                  \qquad \dot V = u\,A

    Suspensionspumpe
        .. math:: P_{P} = \frac{\dot m_{spray}\,\Delta p_P}{\rho_{susp}\,\eta_P}

    Zerstäuberluft (isotherme Verdichtung)
        .. math:: P_{Z} = \frac{\dot m_Z R_L T_{amb}
                  \ln(p_{nozzle}/p_{amb})}{\eta_K},
                  \qquad \dot m_Z = c_Z\, p_{nozzle}

    Fallstrick beim Zerstäuberdruck: er ist im Modell fast entkoppelt von der Physik des
    Betts (nur Leistungsterm), weil das reduzierte Modell die Tropfengrößenverteilung
    nicht auflöst. Für den Architekturvergleich ist ``p_nozzle`` damit eine Stellgröße
    mit reiner Kostenwirkung und ohne Nutzen -- ein optimierender Agent fährt sie sofort
    ans Minimum. Das ist ein Modellartefakt, kein Prozessverhalten. Solange die
    Kopplung Tropfengröße -> Benetzung -> Wachstum/Agglomeration fehlt, sollte
    ``p_nozzle`` im RL-Teil entweder fixiert oder mit einer Qualitätsstrafe versehen
    werden.

    Exhaust recirculation (only when ``recirculation_fraction`` is given): the heater takes the
    mixture at ``heater_intake_temperature_celsius`` (:math:`T_{mix}`, see
    :func:`~.thermo.mixing_temperature_celsius`) instead of :math:`T_{amb}`, with the mixed
    humidity ``inlet_humidity_kg_per_kg``; the recirculation duct adds the fan power
    :math:`r u A \zeta_{rec} \rho_g (r u)^2 / (2 \eta_V)` (``EnergyConfig.
    recirculation_duct_loss_coefficient``, placeholder; ``None`` adds nothing). Both arguments
    unset: every term exactly as before.
    """
    heater_intake = (
        ambient_temperature_celsius
        if heater_intake_temperature_celsius is None
        else heater_intake_temperature_celsius
    )
    heater = (
        dry_air_mass_flow_kg_per_s
        * (gas.cp_dry_air_j_per_kg_k + inlet_humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k)
        * backend.maximum(inlet_temperature_celsius - heater_intake, 0.0)
        / energy.heater_efficiency
    )

    volumetric_flow = gas_velocity_m_per_s * cross_section_area_m2
    delta_p = bed_pressure_drop_pa(bed_mass_total_kg, cross_section_area_m2) + (
        energy.distributor_loss_coefficient
        * gas_density_kg_per_m3_value
        * gas_velocity_m_per_s**2
        / 2.0
    )
    fan = volumetric_flow * delta_p / energy.fan_efficiency
    recirculation_fan = None
    if recirculation_fraction is not None:
        coefficient = energy.recirculation_duct_loss_coefficient or 0.0
        recirculated_velocity = recirculation_fraction * gas_velocity_m_per_s
        recirculation_fan = (
            recirculated_velocity
            * cross_section_area_m2
            * coefficient
            * gas_density_kg_per_m3_value
            * recirculated_velocity**2
            / 2.0
            / energy.fan_efficiency
        )
        fan = fan + recirculation_fan

    pump = (
        spray_rate_kg_per_s
        * energy.pump_pressure_rise_pa
        / (material.suspension_density_kg_per_m3 * energy.pump_efficiency)
    )

    pressure_ratio = backend.maximum(nozzle_pressure_bar * 1.0e5 / gas.pressure_pa, 1.0 + 1e-12)
    atomiser_air = energy.atomiser_air_kg_per_s_per_bar * nozzle_pressure_bar
    atomiser = (
        atomiser_air
        * gas.r_specific_dry_air_j_per_kg_k
        * kelvin(ambient_temperature_celsius)
        * backend.log(pressure_ratio)
        / energy.compressor_efficiency
    )

    total = heater + fan + pump + atomiser
    return PowerBreakdown(
        heater_kw=heater / 1000.0,
        fan_kw=fan / 1000.0,
        pump_kw=pump / 1000.0,
        atomiser_kw=atomiser / 1000.0,
        total_kw=total / 1000.0,
        recirculation_fan_kw=None if recirculation_fan is None else recirculation_fan / 1000.0,
    )


def energy_target(
    backend: ArrayBackend,
    energy: EnergyConfig,
    total_power_kw: Array,
    reference_solid_mass_flow_kg_per_s: Array,
) -> Array:
    r"""Zielgröße ``dE`` gemäß Config.

    ``power_kw``
        Momentanleistung in kW.
    ``specific_kwh_per_kg``
        .. math:: e = \frac{\dot E}{3600\,\dot m_{Produkt}} \quad [\mathrm{kWh/kg}]

    Ist ``reference_value`` gesetzt, wird die Differenz zu diesem Referenzbetriebspunkt
    gebildet -- die Lesart "ΔEnergieverbrauch" aus ``02_prozess_setup.md``. Ohne
    Referenz ist ``dE`` der Absolutwert.

    Hinweis zur Skalierung: der spezifische Verbrauch divergiert, wenn der
    Produktmassenstrom gegen Null geht (An-/Abfahren, Batch-Betrieb). Als Regressionsziel
    ist er dadurch schwerer zu lernen als die Momentanleistung. Empfehlung: das
    Sequenzmodell auf ``power_kw`` trainieren und den spezifischen Verbrauch erst in der
    RL-Belohnung bilden, wo der Produktmassenstrom bekannt ist.
    """
    if energy.target == "power_kw":
        value = total_power_kw
    else:
        safe_flow = backend.maximum(reference_solid_mass_flow_kg_per_s, 1e-9)
        value = total_power_kw / (3600.0 * safe_flow)
    if energy.reference_value is not None:
        value = value - energy.reference_value
    return value

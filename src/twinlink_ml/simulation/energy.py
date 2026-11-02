r"""Power consumption of the units -- generation of the target ``dE``.

The enthalpy balance from ``02_prozess_setup.md`` links ``dE`` with ``T_in``, ``u_gas``
and ``m_spray``. Here the *supplied* side is broken down, because only that is the
target of an energy-optimal control:

.. math::
    \dot E = P_{\mathrm{Heizung}} + P_{\mathrm{Ventilator}}
             + P_{\mathrm{Pumpe}} + P_{\mathrm{Zerstäuberluft}}

Monotonicity
------------
The signs required in the project
:math:`\partial \dot E/\partial T_{in} > 0` and
:math:`\partial \dot E/\partial u_{gas} > 0` hold here **structurally**: all four
terms are non-negative and monotonically increasing in their respective actions. The
grey-box model is thus a reliable reference for checking whether a monotonic network / ICNN
actually satisfies the constraints -- and does not merely reproduce the training data.

Important caveat for the specific consumption: with
``target = "specific_kwh_per_kg"`` the monotonicity in :math:`T_{in}` holds only as long
as the product mass flow does not depend on :math:`T_{in}`. In continuous
operation with mass control this holds (the solid input depends on ``m_spray`` and
``w_solid``). As soon as a quality limit (product moisture) becomes active and limits the
spray rate, it no longer holds. The constraint is thus tied to the
target definition, not universal.
"""

from __future__ import annotations

from dataclasses import dataclass

from .backend import Array, ArrayBackend
from .config import EnergyConfig, GasConfig, MaterialConfig
from .fluidization import bed_pressure_drop_pa
from .thermo import kelvin


@dataclass(frozen=True, slots=True)
class PowerBreakdown:
    """Breakdown of the power consumption in kW. For diagnostics and AP 3."""

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
    r"""Power consumption of all units.

    Air heater
        .. math:: P_{H} = \frac{\dot m_L (c_{p,L} + Y_{in} c_{p,D})
                  (T_{in} - T_{amb})}{\eta_H}

    Fan
        .. math:: P_{V} = \frac{\dot V \left(\Delta p_{bed}
                  + \zeta \frac{\rho_g u^2}{2}\right)}{\eta_V},
                  \qquad \dot V = u\,A

    Suspension pump
        .. math:: P_{P} = \frac{\dot m_{spray}\,\Delta p_P}{\rho_{susp}\,\eta_P}

    Atomiser air (isothermal compression)
        .. math:: P_{Z} = \frac{\dot m_Z R_L T_{amb}
                  \ln(p_{nozzle}/p_{amb})}{\eta_K},
                  \qquad \dot m_Z = c_Z\, p_{nozzle}

    Pitfall with the atomiser pressure: in the model it is almost decoupled from the physics of
    the bed (power term only), because the reduced model does not resolve the droplet size
    distribution. For the architecture comparison ``p_nozzle`` is thus an action
    with a pure cost effect and no benefit -- an optimising agent drives it immediately
    to the minimum. That is a model artifact, not process behaviour. As long as the
    coupling droplet size -> wetting -> growth/agglomeration is missing,
    ``p_nozzle`` should in the RL part either be fixed or be given a quality
    penalty.

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
    r"""Target ``dE`` according to the config.

    ``power_kw``
        Instantaneous power in kW.
    ``specific_kwh_per_kg``
        .. math:: e = \frac{\dot E}{3600\,\dot m_{Produkt}} \quad [\mathrm{kWh/kg}]

    If ``reference_value`` is set, the difference to this reference operating point
    is formed -- the reading "Δenergy consumption" from ``02_prozess_setup.md``. Without
    reference ``dE`` is the absolute value.

    Note on scaling: the specific consumption diverges when the
    product mass flow goes to zero (start-up/shutdown, batch operation). As a regression target
    it is thus harder to learn than the instantaneous power. Recommendation: train the
    sequence model on ``power_kw`` and form the specific consumption only in the
    RL reward, where the product mass flow is known.
    """
    if energy.target == "power_kw":
        value = total_power_kw
    else:
        safe_flow = backend.maximum(reference_solid_mass_flow_kg_per_s, 1e-9)
        value = total_power_kw / (3600.0 * safe_flow)
    if energy.reference_value is not None:
        value = value - energy.reference_value
    return value

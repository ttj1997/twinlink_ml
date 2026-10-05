r"""Energy demand booked per energy carrier, CO2-weighted (decision D5, M1).

The net objective of Befund 57 (:mod:`twinlink_ml.evaluation.net_energy`) books the whole
gross power of the process, including the air heater, as electricity and credits recovered heat
as gas. For a *gas-fired* air heater that mixes the carriers. This module books every kW on the
carrier that supplies it and keeps the old function untouched (Befund 57-68 rest on it).

Booking
-------
The process model (:mod:`twinlink_ml.simulation.energy`) gives the heater as electric input power
``P_H = duty / eta_process`` (``EnergyConfig.heater_efficiency``, 0.98), so the thermal duty is
``duty = P_H * eta_process``. Main case, **gas-fired heater**::

    E_el  = P_fan + P_pump + P_atomiser
    E_gas = max(duty - Q_recovered, 0) / eta_gas       (eta_gas: heater efficiency of the network)

Sensitivity, **electric heater** (the booking of Befund 52-68, plus recovery if given)::

    E_el  = P_fan + P_pump + P_atomiser + max(duty - Q_recovered, 0) / eta_process
    E_gas = 0

``CO2 = 0.366 * E_el + 0.201 * E_gas`` [kg CO2e/h] with the factors of
:mod:`twinlink_ml.evaluation.net_energy` (full project description, UBA); ``kWh sum = E_el + E_gas``
[kW]. Both are derived from the same two numbers, so the CO2-weighted and the unweighted result
come from the same solution.

The efficiency ``eta_gas`` is a placeholder of the partner heat model (0.80 in the partner data
case, Befund 60). ``Q_recovered`` is zero for a single line without a network (M1) and the
recovered power of the cascade (scaled) in the network studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .net_energy import ELECTRICITY_CO2_KG_PER_KWH, GAS_CO2_KG_PER_KWH

__all__ = ["CarrierEnergy", "HeaterCarrier", "carrier_energy", "carrier_energy_from_diagnostics"]

HeaterCarrier = Literal["gas", "electric"]


@dataclass(frozen=True, slots=True)
class CarrierEnergy:
    """Electric and gas power of one operating point, in kW."""

    electric_kw: float
    gas_kw: float

    @property
    def kwh_sum_kw(self) -> float:
        """Unweighted sum of both carriers, kW."""
        return self.electric_kw + self.gas_kw

    @property
    def co2_kg_per_h(self) -> float:
        """CO2-weighted emission rate, kg CO2e/h."""
        return ELECTRICITY_CO2_KG_PER_KWH * self.electric_kw + GAS_CO2_KG_PER_KWH * self.gas_kw


def carrier_energy(
    *,
    power_heater_kw: float,
    power_fan_kw: float,
    power_pump_kw: float,
    power_atomiser_kw: float,
    heater: HeaterCarrier,
    process_heater_efficiency: float,
    gas_heater_efficiency: float,
    recovered_heat_kw: float = 0.0,
) -> CarrierEnergy:
    """Book the four power terms of the process on their carriers (see the module docstring)."""
    for name, value in (
        ("process_heater_efficiency", process_heater_efficiency),
        ("gas_heater_efficiency", gas_heater_efficiency),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} muss in (0, 1] liegen, ist {value}")
    if recovered_heat_kw < 0.0:
        raise ValueError(f"recovered_heat_kw darf nicht negativ sein, ist {recovered_heat_kw}")
    other_electric = power_fan_kw + power_pump_kw + power_atomiser_kw
    duty_kw = power_heater_kw * process_heater_efficiency
    remaining_duty_kw = max(duty_kw - recovered_heat_kw, 0.0)
    if heater == "gas":
        return CarrierEnergy(
            electric_kw=other_electric, gas_kw=remaining_duty_kw / gas_heater_efficiency
        )
    if heater == "electric":
        return CarrierEnergy(
            electric_kw=other_electric + remaining_duty_kw / process_heater_efficiency,
            gas_kw=0.0,
        )
    raise ValueError(f"heater muss 'gas' oder 'electric' sein, ist {heater!r}")


def carrier_energy_from_diagnostics(
    diagnostics: dict[str, float],
    *,
    heater: HeaterCarrier,
    process_heater_efficiency: float,
    gas_heater_efficiency: float,
    recovered_heat_kw: float = 0.0,
) -> CarrierEnergy:
    """:func:`carrier_energy` on the ``power_*_kw`` entries of the solution diagnostics."""
    return carrier_energy(
        power_heater_kw=float(diagnostics["power_heater_kw"]),
        power_fan_kw=float(diagnostics["power_fan_kw"]),
        power_pump_kw=float(diagnostics["power_pump_kw"]),
        power_atomiser_kw=float(diagnostics["power_atomiser_kw"]),
        heater=heater,
        process_heater_efficiency=process_heater_efficiency,
        gas_heater_efficiency=gas_heater_efficiency,
        recovered_heat_kw=recovered_heat_kw,
    )

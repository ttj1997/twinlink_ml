"""Fan and pump power as an output (not a switch), so that electricity shows in the CO2.

The partner model has no electricity at all; recovering heat costs pressure drop in the exhaust
and gas ducts and a pump for the medium. These are **placeholders without a source**: the
efficiencies are the defaults of the laboratory process (:class:`~twinlink_ml.simulation.config.
EnergyConfig`, the factors of :mod:`twinlink_ml.evaluation.carrier_energy`), the pressure drops
are rounded orders of magnitude. They are outputs of a scenario, not inputs of the heat
calculation, and they enter the CO2 only:

    ``CO2 = 0.366 * E_el + 0.201 * E_gas`` [kg CO2e/h]

with the electricity factor 0.366 and the gas factor 0.201 kg/kWh of
:mod:`twinlink_ml.evaluation.net_energy` (via :class:`~twinlink_ml.evaluation.carrier_energy.
CarrierEnergy`). ``E_gas`` is the chemical power of the **corrected** fuel flow (``m H_u``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from twinlink_ml.evaluation.carrier_energy import CarrierEnergy
from twinlink_ml.partner_v4 import corrected as c
from twinlink_ml.simulation.config import EnergyConfig

__all__ = [
    "AuxiliaryConfig",
    "AuxiliaryPower",
    "fan_power_w",
    "pump_power_w",
    "volume_flow_m3_per_s",
]

R_DRY_AIR = 287.05
"""J/(kg K)."""
R_VAPOUR = 461.5
"""J/(kg K)."""
KELVIN = 273.15


def volume_flow_m3_per_s(stream: c.Stream, temperature_celsius: float, pressure_pa: float) -> float:
    """Volume flow of moist air (ideal gas) at ``T`` and total pressure."""
    r_mix = R_DRY_AIR + stream.humidity_kg_per_kg * R_VAPOUR
    return stream.dry_air_kg_per_s * r_mix * (temperature_celsius + KELVIN) / pressure_pa


def fan_power_w(volume_flow: float, pressure_drop_pa: float, efficiency: float) -> float:
    """Shaft power of a fan, ``V dp / eta`` [W]."""
    return volume_flow * pressure_drop_pa / efficiency


def pump_power_w(
    mass_flow_kg_per_s: float, pressure_rise_pa: float, density_kg_per_m3: float, efficiency: float
) -> float:
    """Shaft power of a liquid pump, ``m dp / (rho eta)`` [W]."""
    return mass_flow_kg_per_s * pressure_rise_pa / (density_kg_per_m3 * efficiency)


@dataclass(frozen=True)
class AuxiliaryConfig:
    """Placeholders (no source); the efficiencies are the laboratory defaults."""

    fan_efficiency: float = field(default_factory=lambda: EnergyConfig().fan_efficiency)
    pump_efficiency: float = field(default_factory=lambda: EnergyConfig().pump_efficiency)
    dryer_pressure_drop_pa: float = 4000.0
    """Distributor and bed of one dryer, on the process-gas fan."""
    exchanger_pressure_drop_pa: float = 500.0
    """Per exchanger and side (gas side on the process fan, exhaust side on the exhaust fan)."""
    loop_pressure_drop_pa: float = 1.0e5
    """Both exchangers and the piping of one medium circuit."""
    htm_density_kg_per_m3: float = 1000.0
    """The medium is taken as water (``cp`` 4.2 kJ/(kg K) in the workbook)."""


@dataclass(frozen=True)
class AuxiliaryPower:
    """Electric power of the fans and pumps of one scenario, W."""

    process_fans_w: float
    exhaust_fans_w: float
    pumps_w: float

    @property
    def total_w(self) -> float:
        return self.process_fans_w + self.exhaust_fans_w + self.pumps_w

    def carrier_energy(
        self, fuel_corrected_kg_per_s: float, heating_value_j_per_kg: float
    ) -> CarrierEnergy:
        """Electricity and the chemical power of the corrected fuel, in kW."""
        return CarrierEnergy(
            electric_kw=self.total_w / 1e3,
            gas_kw=fuel_corrected_kg_per_s * heating_value_j_per_kg / 1e3,
        )

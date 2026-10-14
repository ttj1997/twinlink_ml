r"""Steam share and daily balance of the process-air heating (reconstruction from the air balance).

Heat that has to be given to the process air (the fan sits behind the steam heater, so its heat is
part of the measured supply-air temperature):

.. math:: Q_{air} = C_{air}\,(T_{supply} - T_{amb}) - f_{fan}\,P_{fan}
          = Q_{out} + Q_{steam},

so the steam heat is RECONSTRUCTED as ``Q_steam = Q_air - Q_out``. It is not measured (no steam
flow in the exports; the valve position is only a proxy). ``f_fan`` and the humidity of the intake
air are placeholders (:mod:`twinlink_ml.storage_a1.config`); the study reports bands over them.
Gas for the steam: ``Q_steam / eta_boiler`` with the project-wide CO2 factor of
:mod:`twinlink_ml.evaluation.net_energy`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..evaluation.net_energy import GAS_CO2_KG_PER_KWH
from .config import StorageSystemConfig
from .model import air_capacity_rate_kw_per_k
from .roles import PlantDay

__all__ = ["DailyBalance", "air_heating_kw", "daily_balance", "steam_share"]

FloatArray = NDArray[np.float64]


def air_heating_kw(day: PlantDay, cfg: StorageSystemConfig) -> FloatArray:
    """Heat to be given to the process air (without the fan's own heat), kW."""
    c_air = air_capacity_rate_kw_per_k(day["process_air_flow"], day["ambient_temperature"], cfg.air)
    fan = cfg.steam.fan_heat_fraction * day["process_fan_power"]
    return np.asarray(c_air * (day["supply_air_temperature"] - day["ambient_temperature"]) - fan)


def steam_share(q_air_kw: FloatArray, q_out_kw: FloatArray) -> FloatArray:
    """Share of the air heating supplied by steam, ``(Q_air - Q_out) / Q_air``."""
    return np.asarray((q_air_kw - q_out_kw) / q_air_kw, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class DailyBalance:
    """One day: heating demand, waste-heat share, steam share, CO2 (all reconstructed)."""

    heating_demand_kwh: float
    waste_heat_kwh: float
    steam_kwh: float
    waste_heat_share: float
    steam_share: float
    steam_negative_minutes: int
    gas_kwh: float
    co2_kg: float
    co2_without_recovery_kg: float

    def to_dict(self) -> dict[str, float | int]:
        return {name: getattr(self, name) for name in self.__slots__}


def daily_balance(day: PlantDay, cfg: StorageSystemConfig, q_out_kw: FloatArray) -> DailyBalance:
    """Integrate the balance over the day (rectangle rule on the export grid)."""
    hours = day.step_s / 3600.0
    q_air = air_heating_kw(day, cfg)
    q_steam = q_air - q_out_kw
    demand = float(np.sum(q_air) * hours)
    waste = float(np.sum(q_out_kw) * hours)
    steam = float(np.sum(q_steam) * hours)
    eta = cfg.steam.boiler_efficiency
    gas = steam / eta
    return DailyBalance(
        heating_demand_kwh=demand,
        waste_heat_kwh=waste,
        steam_kwh=steam,
        waste_heat_share=waste / demand,
        steam_share=steam / demand,
        steam_negative_minutes=int(np.count_nonzero(q_steam < 0.0)),
        gas_kwh=gas,
        co2_kg=gas * GAS_CO2_KG_PER_KWH,
        co2_without_recovery_kg=demand / eta * GAS_CO2_KG_PER_KWH,
    )

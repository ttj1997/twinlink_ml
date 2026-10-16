r"""Operating point of plant A -> outlet state, steam, fan power and CO2 (stationary).

Air-side heat (the fan sits behind the steam heater, so its heat is part of the supply-air
temperature; booking as in ``storage_a1.balance``):

.. math:: Q_{air} = C_{air}\,(T_{in} - T_{amb}),\qquad
          Q_{steam} = Q_{air} - f_{fan} P_{process} - Q_{waste},

``C_air`` the capacity rate of the process air (humid heat at the intake humidity), ``Q_waste``
the preheater duty (:mod:`twinlink_ml.plant_a_op.waste_heat`), capped at what the air still needs
(``Q_air - f_fan P_process``; the excess counts as curtailed waste heat). Steam is supplied by a gas
boiler: gas ``= Q_steam / eta``. CO2 rate ``= gas * f_gas + (P_process + P_exhaust) * f_el``.
Only the process-fan heat enters the supply air; the exhaust fan sits behind the dryer.

The spray rate (the production) is held at the measured day mean in every operating point; the
outlet temperature and humidity follow from the dryer balance (:mod:`twinlink_ml.plant_a_op.dryer`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import EnergyPlaceholders
from .data import DayState
from .dryer import DryerModel, intake_humidity
from .fans import FanModel
from .waste_heat import WasteHeatModel

__all__ = ["OperatingPoints", "PlantAModel"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OperatingPoints:
    """Everything evaluated at a set of operating points (arrays of one common shape)."""

    air_factor: FloatArray
    supply_celsius: FloatArray
    outlet_celsius: FloatArray
    outlet_relative_humidity: FloatArray
    air_heating_kw: FloatArray
    fan_heat_kw: FloatArray
    waste_heat_kw: FloatArray
    waste_heat_curtailed_kw: FloatArray
    steam_kw: FloatArray
    process_fan_kw: FloatArray
    exhaust_fan_kw: FloatArray
    co2_steam_kg_h: FloatArray
    co2_fans_kg_h: FloatArray

    @property
    def co2_kg_h(self) -> FloatArray:
        return np.asarray(self.co2_steam_kg_h + self.co2_fans_kg_h, dtype=np.float64)

    def at(self, index: tuple[int, ...] | int) -> dict[str, float]:
        """Scalar values at one index (for records)."""
        out = {name: float(getattr(self, name)[index]) for name in self.__slots__}
        out["co2_kg_h"] = out["co2_steam_kg_h"] + out["co2_fans_kg_h"]
        return out


@dataclass(frozen=True, slots=True)
class PlantAModel:
    """Dryer balance, two fans, waste heat and the CO2 conversion for one day state."""

    state: DayState
    dryer: DryerModel
    process_fan: FanModel
    exhaust_fan: FanModel
    waste_heat: WasteHeatModel
    energy: EnergyPlaceholders

    def air_capacity_rate_kw_per_k(self, air_factor: ArrayLike) -> FloatArray:
        ph = self.dryer.placeholders
        y_in = intake_humidity(self.state.ambient_celsius, ph)
        c_h = (ph.gas.cp_dry_air_j_per_kg_k + y_in * ph.gas.cp_vapour_j_per_kg_k) / 1000.0
        m = np.asarray(air_factor, dtype=np.float64) * self.state.process_air_nm3h
        return np.asarray(m * ph.normal_density_kg_per_nm3 / 3600.0 * c_h, dtype=np.float64)

    def evaluate(self, air_factor: ArrayLike, supply_celsius: ArrayLike) -> OperatingPoints:
        s = self.state
        f, t_in = np.broadcast_arrays(
            np.asarray(air_factor, dtype=np.float64), np.asarray(supply_celsius, dtype=np.float64)
        )
        flow = f * s.process_air_nm3h
        kwargs = {
            "atomising_air_nm3h": s.atomising_air_nm3h,
            "spray_kg_h": s.spray_kg_h,
            "ambient_celsius": s.ambient_celsius,
        }
        t_out = self.dryer.outlet_celsius(flow, t_in, **kwargs)
        rh = self.dryer.outlet_relative_humidity(flow, t_out, **kwargs)
        c_air = self.air_capacity_rate_kw_per_k(f)
        q_air = c_air * (t_in - s.ambient_celsius)
        p_proc = self.process_fan.power_kw(f)
        p_exh = self.exhaust_fan.power_kw(f)
        fan_heat = self.energy.fan_heat_fraction * p_proc
        need = np.maximum(q_air - fan_heat, 0.0)
        available = self.waste_heat.duty_kw(c_air, s.ambient_celsius)
        waste = np.minimum(available, need)
        steam = need - waste
        e = self.energy
        return OperatingPoints(
            air_factor=np.asarray(f, dtype=np.float64),
            supply_celsius=np.asarray(t_in, dtype=np.float64),
            outlet_celsius=t_out,
            outlet_relative_humidity=rh,
            air_heating_kw=np.asarray(q_air, dtype=np.float64),
            fan_heat_kw=np.asarray(fan_heat, dtype=np.float64),
            waste_heat_kw=np.asarray(waste, dtype=np.float64),
            waste_heat_curtailed_kw=np.asarray(available - waste, dtype=np.float64),
            steam_kw=np.asarray(steam, dtype=np.float64),
            process_fan_kw=p_proc,
            exhaust_fan_kw=p_exh,
            co2_steam_kg_h=np.asarray(
                steam / e.boiler_efficiency * e.gas_co2_kg_per_kwh, dtype=np.float64
            ),
            co2_fans_kg_h=np.asarray(
                (p_proc + p_exh) * e.electricity_co2_kg_per_kwh, dtype=np.float64
            ),
        )

    def baseline(self) -> OperatingPoints:
        """Today's operation: measured air flow and supply temperature (factor 1)."""
        return self.evaluate(np.array([1.0]), np.array([self.state.supply_celsius]))

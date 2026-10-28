"""Prices and values every controller sees: plant tariff, price information, heat and credit values.

* ``electricity`` = day-ahead price + adder (EUR/kWh electric), per hour;
* ``known`` = :func:`~twinlink_ml.storage_price.prices.known_price_forecast` of ``electricity``
  (the adder is constant, so forecasting the spot price or the tariff is the same);
* ``heat_value`` = EUR per kWh of gas heat (``heatpump.config.Economics.gas_cost_per_kwh_heat``,
  CO2 price included);
* ``end_value`` = value of a kWh left in the storage: mean electricity price of the TRAINING
  months / COP (end credit of every evaluated month and terminal value of the MPC).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import StoragePriceConfig
from .prices import PriceYear, known_price_forecast

__all__ = ["Market", "build_market"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Market:
    """Hourly tariff and price information (see module docstring)."""

    electricity: FloatArray
    known: FloatArray
    heat_value: float
    heat_co2: float
    electricity_co2: float
    heater_efficiency: float
    gas_co2_per_kwh_fuel: float
    end_value: float

    @property
    def horizon_h(self) -> int:
        return int(self.known.shape[1])


def build_market(
    config: StoragePriceConfig, prices: PriceYear, *, cop: float, horizon_h: int
) -> Market:
    """Tariff, price information over ``horizon_h`` and the values of the case."""
    el = prices.spot_eur_per_kwh + config.prices.adder_eur_per_kwh
    train = np.isin(prices.clock.month, config.training_months)
    eco = config.economics
    return Market(
        electricity=np.asarray(el, dtype=np.float64),
        known=known_price_forecast(el, prices.clock, horizon_h),
        heat_value=eco.gas_cost_per_kwh_heat,
        heat_co2=eco.gas_co2_per_kwh_heat,
        electricity_co2=eco.electricity_co2_kg_per_kwh,
        heater_efficiency=eco.heater_efficiency,
        gas_co2_per_kwh_fuel=eco.gas_co2_kg_per_kwh,
        end_value=float(np.mean(el[train]) / cop),
    )

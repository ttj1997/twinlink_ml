r"""Plant of the study: waste heat -> (heat pump) -> storage -> preheater, residual demand by gas.

Two streams per hour in the problem table of storage test 1
(:func:`twinlink_ml.storage_timeslice.table.cascade_profile`):

* hot: waste heat from ``T_in`` to ``T_out`` with ``waste_heat_kw x f_t`` (capacity rate
  ``waste_heat_kw / (T_in - T_out)``);
* cold: process air from the intake temperature ``T_a,t`` to the preheater outlet with the capacity
  rate ``air_cp x f_t``;

``f_t`` = 1 on weekdays, ``weekend_factor`` on Saturday and Sunday (production plan). The heat pump
(evaporator and condenser pseudo-streams, COP of Befund 101 via
:func:`twinlink_ml.heatpump.cop.design_pairs`) and the storage level (charging and discharging
streams of Befund 99) enter as extra boundaries, and
:func:`twinlink_ml.heatpump.storage.coupled_pieces` gives the hot utility of every hour as

.. math:: U_t = \max\bigl(0,\ \max_g (M_{t,g} + \alpha_g Q_t + \beta_g q_{c,t}
          - \gamma_g q_{d,t})\bigr),

the gas heat (useful heat; fuel = ``U / eta``). Both packages are only read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..heatpump.config import HeatPumpSpec
from ..heatpump.cop import DesignPair, design_pairs
from ..heatpump.storage import CoupledPieces, coupled_pieces
from ..storage_timeslice.table import (
    Cascade,
    cascade_profile,
    direct_hot_utility_kw,
    storage_stream_temperatures,
)
from .config import StoragePriceConfig
from .prices import Clock

__all__ = ["Plant", "build_plant", "intake_temperature", "production_factor"]

FloatArray = NDArray[np.float64]

_MID_MONTH_DAY = np.array(
    [15.5, 45.0, 74.5, 105.0, 135.5, 166.0, 196.5, 227.5, 258.0, 288.5, 319.0, 349.5]
)
"""Day of the year (0-based, fractional) of the middle of each month (anchor of the means)."""


def intake_temperature(config: StoragePriceConfig, clock: Clock) -> FloatArray:
    """Intake temperature per hour: monthly means interpolated cyclically plus a daily cosine."""
    cl = config.climate
    means = np.asarray(cl.monthly_mean_celsius, dtype=np.float64)
    days_in_year = float(clock.day[-1] + 1)
    x = clock.day + clock.hour / 24.0
    mid = _MID_MONTH_DAY
    xp = np.concatenate([mid - days_in_year, mid, mid + days_in_year])
    fp = np.concatenate([means, means, means])
    base = np.interp(x, xp, fp)
    daily = cl.daily_amplitude_k * np.cos(2.0 * np.pi * (clock.hour - cl.hour_of_maximum) / 24.0)
    return np.asarray(base + daily, dtype=np.float64)


def production_factor(config: StoragePriceConfig, clock: Clock) -> FloatArray:
    """1 on weekdays, ``weekend_factor`` on Saturday and Sunday."""
    return np.where(clock.weekend, config.plant.weekend_factor, 1.0).astype(np.float64)


@dataclass(frozen=True, slots=True)
class Plant:
    """Everything a controller needs about the plant, for all hours of the year.

    :ivar pieces: pieces of ``U`` (heat pump and storage level), one row per hour.
    :ivar direct_kw: ``U`` without pump and storage (pinch target with direct recovery).
    :ivar waste_heat_kw: waste heat offered per hour.
    :ivar intake_celsius: intake temperature per hour.
    :ivar pair: design pair of the heat pump (COP).
    """

    pieces: CoupledPieces
    cascade: Cascade
    direct_kw: FloatArray
    waste_heat_kw: FloatArray
    intake_celsius: FloatArray
    pair: DesignPair
    size_kw: float
    capacity_kwh: float
    charge_max_kw: float
    discharge_max_kw: float
    loss_per_h: float
    dt_h: float = 1.0

    @property
    def cop(self) -> float:
        return self.pair.cop

    @property
    def keep(self) -> float:
        """Share of the content kept over one step."""
        return 1.0 - self.loss_per_h * self.dt_h

    def utility_kw(self, t: int, q: float, q_c: float, q_d: float) -> float:
        """Gas heat ``U`` of hour ``t`` for pump heat ``q`` and storage duties."""
        value = (
            self.pieces.offset_kw[t]
            + q * self.pieces.alpha
            + q_c * self.pieces.beta
            - q_d * self.pieces.gamma
        )
        return float(max(0.0, np.max(value)))


def _design_pair(config: StoragePriceConfig) -> DesignPair:
    hp = config.heat_pump
    pairs = design_pairs(
        HeatPumpSpec(),
        minimum_pinch_k=config.plant.minimum_pinch_k,
        evaporating_celsius=(hp.evaporating_celsius,),
        condensing_celsius=(hp.condensing_celsius,),
    )
    if len(pairs) != 1:
        raise ValueError(
            f"design pair {hp.evaporating_celsius} -> {hp.condensing_celsius} is not admissible "
            "under the Befund-101 heat-pump model"
        )
    return pairs[0]


def build_plant(config: StoragePriceConfig, clock: Clock) -> Plant:
    """Cascade and pieces of every hour of the year."""
    pl, st = config.plant, config.storage
    pair = _design_pair(config)
    t_c, t_d = storage_stream_temperatures(
        np.array([st.level_celsius]),
        minimum_pinch_k=pl.minimum_pinch_k,
        approach_k=pl.minimum_pinch_k,  # approach dT_min on both sides (Befund 99 specification)
    )
    if t_c[0] > pair.condenser_shifted + 1e-9:
        raise ValueError("the condenser cannot charge the storage at this level")
    f = production_factor(config, clock)
    intake = intake_temperature(config, clock)
    if np.any(intake >= pl.preheater_outlet_celsius):
        raise ValueError("intake temperature reaches the preheater outlet")
    n = clock.n_hours
    w_cp = pl.waste_heat_kw / (pl.waste_heat_in_celsius - pl.waste_heat_out_celsius)
    hot = np.zeros((n, 1, 3))
    hot[:, 0, 0] = pl.waste_heat_in_celsius
    hot[:, 0, 1] = pl.waste_heat_out_celsius
    hot[:, 0, 2] = w_cp * f
    cold = np.zeros((n, 1, 3))
    cold[:, 0, 0] = pl.preheater_outlet_celsius
    cold[:, 0, 1] = intake
    cold[:, 0, 2] = pl.air_cp_kw_per_k * f
    extra = np.array([pair.evaporator_shifted, pair.condenser_shifted, t_c[0], t_d[0]])
    cascade = cascade_profile(hot, cold, minimum_pinch_k=pl.minimum_pinch_k, extra_points=extra)
    pieces = coupled_pieces(
        cascade,
        evaporator_shifted=pair.evaporator_shifted,
        condenser_shifted=pair.condenser_shifted,
        cop=pair.cop,
        charge_shifted=float(t_c[0]),
        discharge_shifted=float(t_d[0]),
    )
    return Plant(
        pieces=pieces,
        cascade=cascade,
        direct_kw=direct_hot_utility_kw(cascade),
        waste_heat_kw=pl.waste_heat_kw * f,
        intake_celsius=intake,
        pair=pair,
        size_kw=config.heat_pump.size_kw,
        capacity_kwh=st.capacity_kwh,
        charge_max_kw=st.charge_max_kw,
        discharge_max_kw=st.discharge_max_kw,
        loss_per_h=st.loss_per_h,
    )

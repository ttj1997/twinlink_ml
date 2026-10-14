r"""Storage A1 and the heat-extraction chain (storage, exchanger A1, preheater loop, preheater A1).

Components (ABCs with one implementation each; stratification would be a second
:class:`StorageModel`, not a change of the callers):

* :class:`MixedNodeStorage` -- one mixed node,
  :math:`(\rho V c_p)\,dT_S/dt = Q_{in} - Q_{out} - UA_S (T_S - T_{room})`.
  The balance is the same for both loop topologies (``Q_in`` = compressor heat taken up by the
  storage loop, ``Q_out`` = heat given to the preheater loop over exchanger A1).
* :class:`CounterflowExchanger` -- effectiveness-NTU, counter-current.

Why effectiveness-NTU and not a fixed ``dT_min``: the flows change over the day (the preheater loop
flow dips by a factor of two on one of the two days) and a fixed approach does not follow the part
load, whereas one conductance ``UA`` does. ``dT_min`` is kept as a CHECK, not as the model: the
minimum approach along the exchanger (review finding A1 of the heat-network module,
:mod:`twinlink_ml.heatnetwork.approach`). For constant capacity rates both temperature profiles
are straight lines, so the minimum over the exchanger is at one of its two ends -- there the
dense-profile maximum search of the heat-network module reduces to :func:`min_approach_k`
(test: ``tests/test_storage_a1_model.py``). A violating state is flagged ``feasible=False``, never
silently capped.

Closed form of the chain. With ``a = eps_1 C_min1`` (exchanger A1), ``b = eps_2 C_min2``
(preheater), ``C_2`` the capacity rate of the preheater loop, ``T_h`` the hot inlet of exchanger A1
and ``T_a`` the outside air: both exchangers carry the same duty ``Q``, and
:math:`T_h - Q/a = T_{ret}`, :math:`T_a + Q/b = T_{sup}`, :math:`T_{sup} - T_{ret} = Q/C_2` give

.. math:: Q = \frac{T_h - T_a}{1/a + 1/b - 1/C_2} .

The effectivenesses depend on the flows only (constant fluid properties), so no iteration is
needed. Topology B: ``T_h = T_S``. Topology A: the compressors heat the storage outflow by
``Q_in / C_1``, so ``T_h = T_S + Q_in / C_1``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..simulation.backend import NumpyBackend
from ..simulation.thermo import absolute_humidity_kg_per_kg
from .config import (
    AirConfig,
    ExchangerConfig,
    StorageConfig,
    StorageSystemConfig,
    WaterConfig,
)

__all__ = [
    "CounterflowExchanger",
    "ExtractionChain",
    "ExtractionResult",
    "HeatExchanger",
    "MixedNodeStorage",
    "StorageModel",
    "air_capacity_rate_kw_per_k",
    "effectiveness_counterflow",
    "min_approach_k",
    "water_capacity_rate_kw_per_k",
]

FloatArray = NDArray[np.float64]


def water_capacity_rate_kw_per_k(
    flow_m3_per_h: FloatArray | float, water: WaterConfig
) -> FloatArray:
    """Capacity rate ``rho c_p V`` of a water flow in kW/K (flow in m3/h)."""
    return np.asarray(flow_m3_per_h, dtype=np.float64) * water.rho_cp_kj_per_m3_k / 3600.0


def air_capacity_rate_kw_per_k(
    normal_flow_nm3_per_h: FloatArray | float,
    ambient_celsius: FloatArray | float,
    air: AirConfig,
) -> FloatArray:
    """Capacity rate of the (moist) process air in kW/K from the normal volume flow.

    Dry mass flow = normal flow x normal density; humidity from the assumed relative humidity at
    the ambient temperature (placeholder, not measured).
    """
    flow = np.asarray(normal_flow_nm3_per_h, dtype=np.float64)
    t_amb = np.asarray(ambient_celsius, dtype=np.float64)
    m_dry = flow * air.normal_density_kg_per_nm3 / 3600.0
    rel = np.full_like(t_amb, air.relative_humidity)
    y = np.asarray(absolute_humidity_kg_per_kg(NumpyBackend(), air.gas, t_amb, rel))
    cp = air.gas.cp_dry_air_j_per_kg_k + y * air.gas.cp_vapour_j_per_kg_k
    return np.asarray(m_dry * cp / 1000.0, dtype=np.float64)


def effectiveness_counterflow(ntu: FloatArray, c_ratio: FloatArray) -> FloatArray:
    """Effectiveness of a counter-current exchanger, ``c_ratio = C_min / C_max`` in [0, 1]."""
    ntu = np.asarray(ntu, dtype=np.float64)
    r = np.asarray(c_ratio, dtype=np.float64)
    near_one = np.abs(1.0 - r) < 1e-9
    e = np.exp(-ntu * (1.0 - np.where(near_one, 0.0, r)))
    general = (1.0 - e) / (1.0 - np.where(near_one, 0.0, r) * e)
    balanced = ntu / (1.0 + ntu)
    return np.asarray(np.where(near_one, balanced, general), dtype=np.float64)


def min_approach_k(
    t_hot_in: FloatArray,
    t_hot_out: FloatArray,
    t_cold_in: FloatArray,
    t_cold_out: FloatArray,
) -> FloatArray:
    """Smallest hot-minus-cold temperature difference along a counter-current exchanger.

    Constant capacity rates -> linear profiles -> the minimum is at one end: warm end
    ``T_hot_in - T_cold_out``, cold end ``T_hot_out - T_cold_in``.
    """
    return np.minimum(
        np.asarray(t_hot_in) - np.asarray(t_cold_out),
        np.asarray(t_hot_out) - np.asarray(t_cold_in),
    )


class HeatExchanger(ABC):
    """Heat exchanger between a hot and a cold stream."""

    @abstractmethod
    def effectiveness(self, c_hot: FloatArray, c_cold: FloatArray) -> FloatArray:
        """Effectiveness for the two capacity rates (kW/K)."""

    def conductance_kw_per_k(self, c_hot: FloatArray, c_cold: FloatArray) -> FloatArray:
        """``eps * C_min``: duty per kelvin of (hot inlet - cold inlet)."""
        return self.effectiveness(c_hot, c_cold) * np.minimum(c_hot, c_cold)

    def duty_kw(
        self, t_hot_in: FloatArray, c_hot: FloatArray, t_cold_in: FloatArray, c_cold: FloatArray
    ) -> FloatArray:
        return self.conductance_kw_per_k(c_hot, c_cold) * (t_hot_in - t_cold_in)


class CounterflowExchanger(HeatExchanger):
    """Counter-current effectiveness-NTU exchanger with a single conductance ``UA``."""

    def __init__(self, config: ExchangerConfig) -> None:
        self.config = config

    def effectiveness(self, c_hot: FloatArray, c_cold: FloatArray) -> FloatArray:
        c_hot = np.asarray(c_hot, dtype=np.float64)
        c_cold = np.asarray(c_cold, dtype=np.float64)
        c_min = np.minimum(c_hot, c_cold)
        c_max = np.maximum(c_hot, c_cold)
        return effectiveness_counterflow(self.config.ua_kw_per_k / c_min, c_min / c_max)


class StorageModel(ABC):
    """Thermal storage with one temperature state."""

    @abstractmethod
    def heat_capacity_kj_per_k(self) -> float:
        """Heat capacity of the storage content."""

    @abstractmethod
    def loss_kw(self, temperature_celsius: FloatArray) -> FloatArray:
        """Heat loss to the surroundings."""

    def derivative_k_per_s(
        self, temperature_celsius: FloatArray, q_in_kw: FloatArray, q_out_kw: FloatArray
    ) -> FloatArray:
        """``dT/dt`` in K/s from the energy balance (kW = kJ/s)."""
        net = q_in_kw - q_out_kw - self.loss_kw(temperature_celsius)
        return np.asarray(net / self.heat_capacity_kj_per_k(), dtype=np.float64)


class MixedNodeStorage(StorageModel):
    """One perfectly mixed node (the three height sensors are assumed to agree)."""

    def __init__(self, config: StorageConfig, water: WaterConfig) -> None:
        self.config = config
        self.water = water

    def heat_capacity_kj_per_k(self) -> float:
        return self.config.volume_m3 * self.water.rho_cp_kj_per_m3_k

    def loss_kw(self, temperature_celsius: FloatArray) -> FloatArray:
        return np.asarray(
            self.config.ua_kw_per_k
            * (np.asarray(temperature_celsius) - self.config.room_temperature_celsius),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """States of the extraction chain for given inputs (all arrays of equal shape)."""

    q_out_kw: FloatArray
    q_in_kw: FloatArray
    t_hot_in_celsius: FloatArray
    t_hot_out_celsius: FloatArray
    t_supply_celsius: FloatArray
    t_return_celsius: FloatArray
    t_air_after_preheater_celsius: FloatArray
    approach_exchanger_k: FloatArray
    approach_preheater_k: FloatArray
    feasible: NDArray[np.bool_]


class ExtractionChain:
    """Exchanger A1 + preheater loop + preheater A1 in closed form (see module docstring)."""

    def __init__(
        self,
        config: StorageSystemConfig,
        storage: StorageModel,
        storage_exchanger: HeatExchanger,
        preheater: HeatExchanger,
    ) -> None:
        self.config = config
        self.storage = storage
        self.storage_exchanger = storage_exchanger
        self.preheater = preheater

    @classmethod
    def from_config(cls, config: StorageSystemConfig) -> ExtractionChain:
        return cls(
            config,
            MixedNodeStorage(config.storage, config.water),
            CounterflowExchanger(config.storage_exchanger),
            CounterflowExchanger(config.preheater),
        )

    # -- pieces ---------------------------------------------------------------------------
    def _rates(
        self,
        v_storage: FloatArray,
        v_preheater: FloatArray,
        air_flow: FloatArray,
        t_amb: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        w = self.config.water
        return (
            water_capacity_rate_kw_per_k(v_storage, w),
            water_capacity_rate_kw_per_k(v_preheater, w),
            air_capacity_rate_kw_per_k(air_flow, t_amb, self.config.air),
        )

    def _denominator(self, c1: FloatArray, c2: FloatArray, c_air: FloatArray) -> FloatArray:
        a = self.storage_exchanger.conductance_kw_per_k(c1, c2)
        b = self.preheater.conductance_kw_per_k(c2, c_air)
        return np.asarray(1.0 / a + 1.0 / b - 1.0 / c2, dtype=np.float64)

    def _finish(
        self,
        q: FloatArray,
        q_in: FloatArray,
        t_h: FloatArray,
        c1: FloatArray,
        c2: FloatArray,
        c_air: FloatArray,
        t_amb: FloatArray,
    ) -> ExtractionResult:
        a = self.storage_exchanger.conductance_kw_per_k(c1, c2)
        t_ret = t_h - q / a
        t_sup = t_ret + q / c2
        t_hot_out = t_h - q / c1
        t_air_out = t_amb + q / c_air
        app1 = min_approach_k(t_h, t_hot_out, t_ret, t_sup)
        app2 = min_approach_k(t_sup, t_ret, t_amb, t_air_out)
        floor1 = self.config.storage_exchanger.minimum_approach_k
        floor2 = self.config.preheater.minimum_approach_k
        return ExtractionResult(
            q_out_kw=q,
            q_in_kw=q_in,
            t_hot_in_celsius=t_h,
            t_hot_out_celsius=t_hot_out,
            t_supply_celsius=t_sup,
            t_return_celsius=t_ret,
            t_air_after_preheater_celsius=t_air_out,
            approach_exchanger_k=app1,
            approach_preheater_k=app2,
            feasible=(app1 >= floor1) & (app2 >= floor2),
        )

    # -- the two ways to drive the chain -------------------------------------------------------
    def solve_given_q_in(
        self,
        t_storage: FloatArray,
        q_in_kw: FloatArray,
        v_storage_loop: FloatArray,
        v_preheater_loop: FloatArray,
        air_flow_nm3_per_h: FloatArray,
        t_ambient: FloatArray,
    ) -> ExtractionResult:
        """Predictive use: storage temperature and compressor heat ``Q_in`` are given."""
        t_s, q_in, v1, v2, af, t_a = (
            np.asarray(x, dtype=np.float64)
            for x in (
                t_storage,
                q_in_kw,
                v_storage_loop,
                v_preheater_loop,
                air_flow_nm3_per_h,
                t_ambient,
            )
        )
        c1, c2, c_air = self._rates(v1, v2, af, t_a)
        t_h = t_s + (q_in / c1 if self.config.topology == "exchanger_then_storage" else 0.0)
        q = (t_h - t_a) / self._denominator(c1, c2, c_air)
        return self._finish(q, q_in, t_h, c1, c2, c_air, t_a)

    def solve_given_storage_rate(
        self,
        t_storage: FloatArray,
        dt_storage_dt_k_per_s: FloatArray,
        v_storage_loop: FloatArray,
        v_preheater_loop: FloatArray,
        air_flow_nm3_per_h: FloatArray,
        t_ambient: FloatArray,
    ) -> ExtractionResult:
        """Diagnostic use: the measured ``dT_S/dt`` fixes ``Q_in`` through the storage balance.

        Topology B: ``Q`` follows from ``T_S`` alone and ``Q_in = C_S dT_S/dt + Q + loss``.
        Topology A: ``Q_in`` enters ``T_h``; the two relations are solved together (linear).
        """
        t_s, ds, v1, v2, af, t_a = (
            np.asarray(x, dtype=np.float64)
            for x in (
                t_storage,
                dt_storage_dt_k_per_s,
                v_storage_loop,
                v_preheater_loop,
                air_flow_nm3_per_h,
                t_ambient,
            )
        )
        c1, c2, c_air = self._rates(v1, v2, af, t_a)
        den = self._denominator(c1, c2, c_air)
        stored = self.storage.heat_capacity_kj_per_k() * ds + self.storage.loss_kw(t_s)
        if self.config.topology == "storage_then_exchanger":
            t_h = t_s
            q = (t_h - t_a) / den
        else:
            # Q (den - 1/c1) = T_S - T_a + stored/c1,  T_h = T_S + (stored + Q)/c1
            q = (t_s - t_a + stored / c1) / (den - 1.0 / c1)
            t_h = t_s + (stored + q) / c1
        return self._finish(q, stored + q, t_h, c1, c2, c_air, t_a)

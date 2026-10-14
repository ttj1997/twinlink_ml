r"""What the two plant days can and cannot determine -- checked BEFORE anything is calibrated.

Unmeasured: the heat ``Q_in`` that the compressors put into the storage loop. Measured: the storage
temperature ``T_S`` (one sensor), both loop flows, supply and return temperature of the preheater
loop, hence the duty given to the preheater loop,

.. math:: Q_{out}(t) = \rho c_p\,\dot V_2\,(T_{sup} - T_{ret}) \approx Q_{exchanger} .

The storage balance (the same for both loop topologies, because the loop is closed)

.. math:: Q_{in}(t) = \rho V c_p\,\frac{dT_S}{dt} + Q_{out}(t) + UA_S\,(T_S - T_{room})

therefore determines ``Q_in(t)`` only up to the storage volume ``V``:
``Q_in(t; V) = Q0(t) + V g(t)``
with ``g = rho c_p dT_S/dt`` (kW per m3) and ``Q0 = Q_out + loss`` known. ``V`` can only be
bracketed, by requiring ``0 <= Q_in(t; V) <= Q_max`` (partner statement 1200 kW) for all ``t``:

* ``g < 0`` (cooling): ``Q_in >= 0`` gives ``V <= Q0 / |g|`` (upper bound),
* ``g > 0`` (heating): ``Q_in <= Q_max`` gives ``V <= (Q_max - Q0) / g`` (upper bound),
* the opposite inequalities give lower bounds that are negative (trivial): ``V > 0`` is the only
  lower bound.

``UA_S`` enters only through ``Q0`` (it is not identifiable: ``Q_in`` is free at every instant).
The exchanger and preheater conductances ARE identifiable for topology B (two measured outputs,
two parameters); for topology A the hot inlet of the exchanger is ``T_S + Q_in/C_1`` and the
implied exchanger conductance depends on ``V`` -- :func:`implied_conductances` makes that visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import StorageSystemConfig, Topology
from .model import air_capacity_rate_kw_per_k, water_capacity_rate_kw_per_k
from .roles import PlantDay

__all__ = [
    "VolumeBounds",
    "heat_out_kw",
    "implied_conductances",
    "q_in_curve",
    "smoothed_derivative_k_per_s",
    "volume_bounds",
]

FloatArray = NDArray[np.float64]


def heat_out_kw(day: PlantDay, cfg: StorageSystemConfig) -> FloatArray:
    """``Q_out`` from the water side of the preheater loop (measured quantities only)."""
    c2 = water_capacity_rate_kw_per_k(day["preheater_loop_flow"], cfg.water)
    return np.asarray(
        c2 * (day["preheater_supply_temperature"] - day["preheater_return_temperature"]),
        dtype=np.float64,
    )


def smoothed_derivative_k_per_s(
    values: FloatArray, step_s: float, window: int
) -> tuple[slice, FloatArray]:
    """Derivative of a moving average (``window`` samples, valid part only).

    Returns the slice of the original samples it refers to and the derivative in unit/s.
    ``window = 1`` is the plain central difference.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    kernel = np.ones(window) / window
    smooth = np.convolve(values, kernel, mode="valid")
    deriv = np.gradient(smooth, step_s)
    start = window // 2
    return slice(start, start + len(smooth)), deriv


def _balance_terms(
    day: PlantDay, cfg: StorageSystemConfig, window: int
) -> tuple[slice, FloatArray, FloatArray]:
    """``g`` (kW per m3) and ``Q0 = Q_out + loss`` on the valid part of the day."""
    index, d_t = smoothed_derivative_k_per_s(day["storage_temperature"], day.step_s, window)
    g = cfg.water.rho_cp_kj_per_m3_k * d_t
    t_s = day["storage_temperature"][index]
    loss = cfg.storage.ua_kw_per_k * (t_s - cfg.storage.room_temperature_celsius)
    q0 = heat_out_kw(day, cfg)[index] + loss
    return index, g, q0


def q_in_curve(
    day: PlantDay, cfg: StorageSystemConfig, volume_m3: float, window: int
) -> tuple[slice, FloatArray]:
    """Reconstructed ``Q_in(t)`` for an ASSUMED volume (only meaningful as a family in ``V``)."""
    index, g, q0 = _balance_terms(day, cfg, window)
    return index, np.asarray(q0 + volume_m3 * g, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class VolumeBounds:
    """Bracket of the storage volume that keeps ``0 <= Q_in(t) <= q_max_kw``.

    ``upper_*`` are the bounds after allowing the fraction ``tolerance`` of samples to violate
    (``tolerance = 0``: strict). ``lower_m3`` is 0 (all other lower bounds are negative).
    ``infeasible_samples``: samples with ``Q0 < 0`` while cooling, which no ``V >= 0`` can satisfy.
    """

    window: int
    tolerance: float
    q_max_kw: float
    lower_m3: float
    upper_from_nonnegative_m3: float
    upper_from_q_max_m3: float
    infeasible_samples: int

    @property
    def upper_m3(self) -> float:
        return min(self.upper_from_nonnegative_m3, self.upper_from_q_max_m3)


def volume_bounds(
    day: PlantDay,
    cfg: StorageSystemConfig,
    window: int,
    tolerance: float = 0.0,
    q_max_kw: float | None = None,
) -> VolumeBounds:
    """Bracket ``V_S`` from ``0 <= Q_in(t; V) <= Q_max`` (see module docstring)."""
    q_max = cfg.compressor.total_waste_heat_max_kw if q_max_kw is None else q_max_kw
    _, g, q0 = _balance_terms(day, cfg, window)
    cooling = g < 0.0
    heating = g > 0.0
    infeasible = int(np.count_nonzero(cooling & (q0 < 0.0)))
    with np.errstate(divide="ignore", invalid="ignore"):
        up_nonneg = q0[cooling] / -g[cooling]
        up_max = (q_max - q0[heating]) / g[heating]
    up_nonneg = up_nonneg[q0[cooling] >= 0.0]

    def _bound(per_sample: FloatArray) -> float:
        if per_sample.size == 0:
            return float("inf")
        return float(np.quantile(per_sample, tolerance))

    return VolumeBounds(
        window=window,
        tolerance=tolerance,
        q_max_kw=q_max,
        lower_m3=0.0,
        upper_from_nonnegative_m3=_bound(up_nonneg),
        upper_from_q_max_m3=_bound(up_max),
        infeasible_samples=infeasible,
    )


def _ntu_from_effectiveness(eps: FloatArray, r: FloatArray) -> FloatArray:
    """Invert the counter-current effectiveness relation (``r = C_min/C_max``)."""
    eps = np.asarray(eps, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    near_one = np.abs(1.0 - r) < 1e-9
    safe_r = np.where(near_one, 0.0, r)
    general = np.log((eps - 1.0) / (eps * safe_r - 1.0)) / (safe_r - 1.0)
    balanced = eps / (1.0 - eps)
    return np.asarray(np.where(near_one, balanced, general), dtype=np.float64)


def implied_conductances(
    day: PlantDay,
    cfg: StorageSystemConfig,
    topology: Topology,
    volume_m3: float,
    window: int,
) -> tuple[FloatArray, FloatArray]:
    """UA of exchanger A1 and of the preheater implied by the MEASURED duty, per sample (kW/K).

    Uses only measurements plus, for topology A, the assumed volume (through ``Q_in``). A constant
    value over the day is what a one-parameter exchanger model needs. Samples where the
    effectiveness leaves (0, 1) give NaN.
    """
    index, g, q0 = _balance_terms(day, cfg, window)
    q_out = heat_out_kw(day, cfg)[index]
    v1 = day["storage_loop_flow"][index]
    v2 = day["preheater_loop_flow"][index]
    t_s = day["storage_temperature"][index]
    t_ret = day["preheater_return_temperature"][index]
    t_sup = day["preheater_supply_temperature"][index]
    t_amb = day["ambient_temperature"][index]
    c1 = water_capacity_rate_kw_per_k(v1, cfg.water)
    c2 = water_capacity_rate_kw_per_k(v2, cfg.water)
    c_air = air_capacity_rate_kw_per_k(day["process_air_flow"][index], t_amb, cfg.air)
    if topology == "storage_then_exchanger":
        t_h = t_s
    else:
        q_in = q0 + volume_m3 * g
        t_h = t_s + q_in / c1
    eps1 = q_out / (np.minimum(c1, c2) * (t_h - t_ret))
    eps2 = q_out / (np.minimum(c2, c_air) * (t_sup - t_amb))
    valid1 = (eps1 > 0.0) & (eps1 < 1.0)
    valid2 = (eps2 > 0.0) & (eps2 < 1.0)
    ntu1 = np.where(
        valid1,
        _ntu_from_effectiveness(
            np.where(valid1, eps1, 0.5), np.minimum(c1, c2) / np.maximum(c1, c2)
        ),
        np.nan,
    )
    ntu2 = np.where(
        valid2,
        _ntu_from_effectiveness(
            np.where(valid2, eps2, 0.5), np.minimum(c2, c_air) / np.maximum(c2, c_air)
        ),
        np.nan,
    )
    return ntu1 * np.minimum(c1, c2), ntu2 * np.minimum(c2, c_air)

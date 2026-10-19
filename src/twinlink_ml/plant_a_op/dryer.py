r"""Stationary energy and water balance of the fluidised-bed dryer of plant A.

Streams per unit time (dry-air basis, normal-density convention of ``storage_a1``):
process air ``m_p`` at the supply temperature ``T_in``, atomising air ``m_a`` at ``T_atom``, both
with the intake humidity ``Y_in`` (relative humidity a placeholder); suspension ``spray`` at
``T_feed`` with the solid fraction ``x_s`` (placeholder): water ``m_w = spray (1 - x_s)`` evaporates
completely (product moisture neglected) and leaves as vapour at the outlet temperature, solids
``m_s = spray x_s`` leave at the outlet temperature; heat loss ``UA (T_out - T_room)``.
Enthalpies relative to dry air at 0 degC and liquid water at 0 degC, latent heat ``r_0`` at 0 degC
from :func:`~twinlink_ml.simulation.thermo.latent_heat_j_per_kg`:

.. math::
    m_p c_h (T_{in} - T_{out}) = m_a c_h (T_{out} - T_{atom})
        + m_w (r_0 + c_v T_{out} - c_w T_{feed}) + m_s c_s (T_{out} - T_{feed})
        + UA (T_{out} - T_{room}),

with the humid heat ``c_h = c_{p,a} + Y_{in} c_v``. The balance is LINEAR in ``T_out``, so the
outlet temperature of any operating point is explicit (:meth:`DryerModel.outlet_celsius`) and the
loss conductance of a measured day is explicit too (:func:`loss_conductance_kw_per_k`).
Exhaust humidity ``Y_out = Y_in + m_w / (m_p + m_a)``; relative exhaust humidity from the Buck
saturation pressure of the grey-box model, NOT clipped at 1 (a value above 1 means the operating
point would over-saturate the exhaust, which the constraints reject).

Everything here is a balance, not a drying-kinetics model: the product moisture is not modelled
and not measured, which is why the study uses proxy constraints only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..simulation.backend import NumpyBackend
from ..simulation.thermo import (
    absolute_humidity_kg_per_kg,
    latent_heat_j_per_kg,
    saturation_pressure_pa,
)
from .config import DryerPlaceholders
from .data import DayState, MeasuredDay

__all__ = [
    "DryerModel",
    "consistent_solid_fraction",
    "intake_humidity",
    "loss_conductance_kw_per_k",
    "minimum_admissible_solid_fraction",
    "observed_outlet_extremes",
    "relative_humidity_unclipped",
]

FloatArray = NDArray[np.float64]
_BACKEND = NumpyBackend()
_R0_J_PER_KG = float(latent_heat_j_per_kg(_BACKEND, np.float64(0.0)))


def intake_humidity(ambient_celsius: ArrayLike, ph: DryerPlaceholders) -> FloatArray:
    """Absolute humidity of the intake air at the placeholder relative humidity, kg/kg."""
    t = np.asarray(ambient_celsius, dtype=np.float64)
    rel = np.full_like(t, ph.intake_relative_humidity)
    return np.asarray(absolute_humidity_kg_per_kg(_BACKEND, ph.gas, t, rel), dtype=np.float64)


def relative_humidity_unclipped(
    humidity_kg_per_kg: ArrayLike, temperature_celsius: ArrayLike, ph: DryerPlaceholders
) -> FloatArray:
    """``Y p / ((0.622 + Y) p_s(T))`` without the clip of the thermo module."""
    y = np.asarray(humidity_kg_per_kg, dtype=np.float64)
    t = np.asarray(temperature_celsius, dtype=np.float64)
    p_partial = y * ph.gas.pressure_pa / (0.622 + y)
    return np.asarray(p_partial / saturation_pressure_pa(_BACKEND, t), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class _Flows:
    m_process: FloatArray
    m_atom: FloatArray
    m_water: FloatArray
    m_solid: FloatArray
    c_humid: FloatArray  # kJ/(kg K)
    y_in: FloatArray


def _flows(
    ph: DryerPlaceholders,
    process_air_nm3h: ArrayLike,
    atomising_air_nm3h: ArrayLike,
    spray_kg_h: ArrayLike,
    ambient_celsius: ArrayLike,
) -> _Flows:
    rho = ph.normal_density_kg_per_nm3
    spray = np.asarray(spray_kg_h, dtype=np.float64) / 3600.0
    y_in = intake_humidity(ambient_celsius, ph)
    return _Flows(
        m_process=np.asarray(process_air_nm3h, dtype=np.float64) * rho / 3600.0,
        m_atom=np.asarray(atomising_air_nm3h, dtype=np.float64) * rho / 3600.0,
        m_water=spray * (1.0 - ph.solid_fraction),
        m_solid=spray * ph.solid_fraction,
        c_humid=(ph.gas.cp_dry_air_j_per_kg_k + y_in * ph.gas.cp_vapour_j_per_kg_k) / 1000.0,
        y_in=y_in,
    )


def _residual_kw(
    ph: DryerPlaceholders,
    f: _Flows,
    supply_celsius: ArrayLike,
    outlet_celsius: ArrayLike,
) -> FloatArray:
    """Heat released by the process air minus every sink except the wall loss, kW."""
    t_in = np.asarray(supply_celsius, dtype=np.float64)
    t_out = np.asarray(outlet_celsius, dtype=np.float64)
    c_v = ph.gas.cp_vapour_j_per_kg_k / 1000.0
    c_w = ph.cp_water_liquid_j_per_kg_k / 1000.0
    c_s = ph.cp_solid_j_per_kg_k / 1000.0
    r0 = _R0_J_PER_KG / 1000.0
    released = f.m_process * f.c_humid * (t_in - t_out)
    atom = f.m_atom * f.c_humid * (t_out - ph.atomising_air_temperature_celsius)
    water = f.m_water * (r0 + c_v * t_out - c_w * ph.feed_temperature_celsius)
    solid = f.m_solid * c_s * (t_out - ph.feed_temperature_celsius)
    return np.asarray(released - atom - water - solid, dtype=np.float64)


def loss_conductance_kw_per_k(ph: DryerPlaceholders, state: DayState) -> float:
    """Heat-loss conductance ``UA`` that closes the balance of a measured day (may be < 0).

    A negative value means the placeholders are inadmissible for that day (the evaporation alone
    needs more heat than the process air releases); callers flag it, they never clip it.
    """
    f = _flows(
        ph,
        state.process_air_nm3h,
        state.atomising_air_nm3h,
        state.spray_kg_h,
        state.ambient_celsius,
    )
    q = float(_residual_kw(ph, f, state.supply_celsius, state.outlet_celsius))
    return q / (state.outlet_celsius - ph.room_temperature_celsius)


def _ua_affine_in_solid_fraction(ph: DryerPlaceholders, state: DayState) -> tuple[float, float]:
    """``UA(x_s) = a + b x_s`` (the balance is affine in the solid fraction)."""
    ua0 = loss_conductance_kw_per_k(replace(ph, solid_fraction=0.0), state)
    ua1 = loss_conductance_kw_per_k(replace(ph, solid_fraction=0.5), state)
    return ua0, (ua1 - ua0) / 0.5


def minimum_admissible_solid_fraction(ph: DryerPlaceholders, state: DayState) -> float:
    """Smallest solid fraction with ``UA >= 0`` on this day (other placeholders as given)."""
    a, b = _ua_affine_in_solid_fraction(ph, state)
    return -a / b


def consistent_solid_fraction(
    ph: DryerPlaceholders, first: DayState, second: DayState
) -> tuple[float, float]:
    """Solid fraction (same for both days) at which both days give the same ``UA``; and that UA.

    This is a calibration of ``x_s`` from the two days, not a validation: with one shared ``x_s``
    and one shared ``UA`` the two days are two equations for two unknowns.
    """
    a1, b1 = _ua_affine_in_solid_fraction(ph, first)
    a2, b2 = _ua_affine_in_solid_fraction(ph, second)
    x = (a2 - a1) / (b1 - b2)
    return x, a1 + b1 * x


@dataclass(frozen=True, slots=True)
class DryerModel:
    """The balance above with a fixed loss conductance ``ua_kw_per_k``."""

    placeholders: DryerPlaceholders
    ua_kw_per_k: float

    def outlet_celsius(
        self,
        process_air_nm3h: ArrayLike,
        supply_celsius: ArrayLike,
        *,
        atomising_air_nm3h: ArrayLike,
        spray_kg_h: ArrayLike,
        ambient_celsius: ArrayLike,
    ) -> FloatArray:
        """Explicit outlet temperature of the stationary balance, degC (broadcasts)."""
        ph = self.placeholders
        f = _flows(ph, process_air_nm3h, atomising_air_nm3h, spray_kg_h, ambient_celsius)
        c_v = ph.gas.cp_vapour_j_per_kg_k / 1000.0
        c_w = ph.cp_water_liquid_j_per_kg_k / 1000.0
        c_s = ph.cp_solid_j_per_kg_k / 1000.0
        r0 = _R0_J_PER_KG / 1000.0
        t_in = np.asarray(supply_celsius, dtype=np.float64)
        num = (
            f.m_process * f.c_humid * t_in
            + f.m_atom * f.c_humid * ph.atomising_air_temperature_celsius
            - f.m_water * (r0 - c_w * ph.feed_temperature_celsius)
            + f.m_solid * c_s * ph.feed_temperature_celsius
            + self.ua_kw_per_k * ph.room_temperature_celsius
        )
        den = (
            (f.m_process + f.m_atom) * f.c_humid
            + f.m_water * c_v
            + f.m_solid * c_s
            + self.ua_kw_per_k
        )
        return np.asarray(num / den, dtype=np.float64)

    def outlet_humidity(
        self,
        process_air_nm3h: ArrayLike,
        *,
        atomising_air_nm3h: ArrayLike,
        spray_kg_h: ArrayLike,
        ambient_celsius: ArrayLike,
    ) -> FloatArray:
        """``Y_out = Y_in + m_w / (m_p + m_a)``, kg/kg dry air."""
        f = _flows(
            self.placeholders, process_air_nm3h, atomising_air_nm3h, spray_kg_h, ambient_celsius
        )
        return np.asarray(f.y_in + f.m_water / (f.m_process + f.m_atom), dtype=np.float64)

    def outlet_relative_humidity(
        self,
        process_air_nm3h: ArrayLike,
        outlet_celsius: ArrayLike,
        *,
        atomising_air_nm3h: ArrayLike,
        spray_kg_h: ArrayLike,
        ambient_celsius: ArrayLike,
    ) -> FloatArray:
        """Relative humidity of the exhaust at its temperature (not clipped)."""
        y = self.outlet_humidity(
            process_air_nm3h,
            atomising_air_nm3h=atomising_air_nm3h,
            spray_kg_h=spray_kg_h,
            ambient_celsius=ambient_celsius,
        )
        return relative_humidity_unclipped(y, outlet_celsius, self.placeholders)

    def heat_loss_kw(self, outlet_celsius: ArrayLike) -> FloatArray:
        t = np.asarray(outlet_celsius, dtype=np.float64)
        return np.asarray(
            self.ua_kw_per_k * (t - self.placeholders.room_temperature_celsius), dtype=np.float64
        )


def observed_outlet_extremes(day: MeasuredDay, ph: DryerPlaceholders) -> dict[str, float]:
    """Observed extremes of a measured day on the 15-minute blocks (thresholds of the proxies).

    Outlet temperature: measured. Relative exhaust humidity: RECONSTRUCTED per block from the
    measured outlet temperature, the spray rate and the placeholders (the humidity is not
    measured); it therefore moves with the solid fraction and the intake humidity.
    """
    model = DryerModel(ph, 0.0)  # the loss conductance does not enter the humidity
    rh = model.outlet_relative_humidity(
        day["process_air_flow"],
        day["exhaust_outlet_temperature"],
        atomising_air_nm3h=day["atomising_air_flow"],
        spray_kg_h=day["spray_rate"],
        ambient_celsius=day["ambient_temperature"],
    )
    t = day["exhaust_outlet_temperature"]
    return {
        "outlet_min_celsius": float(np.min(t)),
        "outlet_mean_celsius": float(np.mean(t)),
        "rh_max": float(np.max(rh)),
        "rh_mean": float(np.mean(rh)),
    }

r"""Balance numbers for the gap between bed and exhaust temperature (no model, no fit).

Per measured operating point (:class:`~twinlink_ml.plant_scale.model.PlantPoint`). The gas between
bed and exhaust sensor is the process air plus the atomising air (dry-air basis, normal density of
the config); its humidity is that of complete evaporation of the spray water ``W = s (1 - x_s)``
(product moisture neglected, as docs/23), humid heat ``c_h = c_{p,a} + Y c_{p,v}``. The heat the gas
gives up between the bed sensors and the exhaust sensor:

.. math:: Q_{gap} = \dot m_L c_h (T_{bed} - T_{exh}).

Each candidate explanation is turned into the ONE number it would need to close ``Q_gap`` alone:

* **heat loss** (freeboard, filter, duct) to surroundings at ``T_s``: the conductance of a duct with
  uniform UA, ``UA = m_L c_h ln((T_bed - T_s) / (T_exh - T_s))``; ``T_s`` = ambient (intake) and the
  room placeholder. One physical UA must be the same on both days.
* **atomising air** mixed in after the bed at ``T_atom`` (placeholder): the cooling it CAN cause,
  ``m_atom c_h (T_bed - T_atom) / (m_L c_h)`` -- an upper bound (in the plant it may well pass the
  bed and leave at bed temperature).
* **false air** at ``T_s`` entering between bed and sensor: the leak share of the gas flow,
  ``r = (T_bed - T_exh) / (T_exh - T_s)`` (adiabatic mixing, equal humid heats).
* **evaporation above the bed** (spray zone): the share of the spray water,
  ``f = Q_gap / (W (r_0 + c_v T_exh - c_w T_feed))``.
* **sensor location**: spread of the six bed sensors and the distance of the COLDEST bed sensor
  to the exhaust (if the gas left the bed at the coldest sensor's temperature, this gap
  would remain).

Plus the overall dryer balance of docs/23 (:func:`twinlink_ml.plant_a_op.dryer.
loss_conductance_kw_per_k`, which includes the atomising air) with the outlet temperature taken as
the exhaust and as the bed mean: the wall-loss conductance each needs. A negative conductance means
the balance cannot close with that outlet temperature at that solid fraction.
"""

from __future__ import annotations

import math
from typing import Any

from ..plant_a_op.config import DryerPlaceholders
from ..plant_a_op.data import DayState
from ..plant_a_op.dryer import loss_conductance_kw_per_k
from ..simulation.config import GasConfig
from .config import PlaceholderSettings
from .model import _R0_J_PER_KG, PlantPoint

__all__ = ["gap_balance", "overall_balance"]

_SECONDS_PER_HOUR = 3600.0


def _flows(
    point: PlantPoint, solid_fraction: float, rho_n: float, gas: GasConfig
) -> dict[str, float]:
    m_process = point.process_air_nm3h * rho_n / _SECONDS_PER_HOUR
    m_atom = point.atomising_air_nm3h * rho_n / _SECONDS_PER_HOUR
    water = point.spray_kg_h * (1.0 - solid_fraction) / _SECONDS_PER_HOUR
    m_gas = m_process + m_atom
    y_out = water / m_gas
    c_h = gas.cp_dry_air_j_per_kg_k + y_out * gas.cp_vapour_j_per_kg_k
    return {
        "m_process": m_process,
        "m_atom": m_atom,
        "m_gas": m_gas,
        "water": water,
        "c_h": c_h,
        "y_out": y_out,
    }


def gap_balance(
    point: PlantPoint,
    placeholders: PlaceholderSettings,
    *,
    solid_fraction: float,
    normal_density_kg_per_m3: float,
    gas: GasConfig | None = None,
    cp_water_j_per_kg_k: float = 4180.0,
) -> dict[str, Any]:
    """Every candidate's required number for one operating point (see the module docstring)."""
    gas = gas or GasConfig()
    f = _flows(point, solid_fraction, normal_density_kg_per_m3, gas)
    t_bed, t_exh = point.bed_celsius, point.exhaust_celsius
    gap = t_bed - t_exh
    q_gap_w = f["m_gas"] * f["c_h"] * gap
    surroundings = {"ambient": point.ambient_celsius, "room": placeholders.room_temperature_celsius}
    heat_loss = {}
    false_air = {}
    for name, t_s in surroundings.items():
        if t_exh > t_s:
            ua = f["m_gas"] * f["c_h"] * math.log((t_bed - t_s) / (t_exh - t_s))
            share = gap / (t_exh - t_s)
        else:
            ua = share = math.nan
        heat_loss[name] = {"surroundings_celsius": t_s, "ua_w_per_k": ua}
        false_air[name] = {"surroundings_celsius": t_s, "leak_share_of_gas": share}
    t_atom = placeholders.atomising_air_temperature_celsius
    atom_cooling = f["m_atom"] * (t_bed - t_atom) / f["m_gas"]
    latent = (
        _R0_J_PER_KG
        + gas.cp_vapour_j_per_kg_k * t_exh
        - cp_water_j_per_kg_k * placeholders.feed_temperature_celsius
    )
    evaporation_share = q_gap_w / (f["water"] * latent) if f["water"] > 0.0 else math.nan
    return {
        "point": point.label,
        "solid_fraction": solid_fraction,
        "gap_k": gap,
        "gas_flow_kg_per_s": f["m_gas"],
        "atomising_share_of_gas": f["m_atom"] / f["m_gas"],
        "q_gap_kw": q_gap_w / 1000.0,
        "heat_loss": heat_loss,
        "false_air": false_air,
        "atomising_air_cooling_upper_bound_k": atom_cooling,
        "atomising_air_share_of_gap": atom_cooling / gap if gap else math.nan,
        "evaporation_share_of_water": evaporation_share,
        "bed_sensor_spread_k": point.bed_sensor_max_celsius - point.bed_sensor_min_celsius,
        "bed_group_b_minus_group_a_k": point.bed_group_b_celsius - point.bed_group_a_celsius,
        "coldest_bed_sensor_minus_exhaust_k": point.bed_sensor_min_celsius - t_exh,
        "exhaust_minus_downstream_k": t_exh - point.downstream_celsius,
    }


def overall_balance(
    point: PlantPoint,
    placeholders: PlaceholderSettings,
    *,
    solid_fraction: float,
    normal_density_kg_per_m3: float,
) -> dict[str, float]:
    """Wall-loss conductance (kW/K) of the docs/23 dryer balance with either outlet temperature.

    Intake humidity 0 (dried supply air; the placeholder humidity of the study is negligible here).
    """
    ph = DryerPlaceholders(
        solid_fraction=solid_fraction,
        intake_relative_humidity=0.0,
        feed_temperature_celsius=placeholders.feed_temperature_celsius,
        atomising_air_temperature_celsius=placeholders.atomising_air_temperature_celsius,
        room_temperature_celsius=placeholders.room_temperature_celsius,
        normal_density_kg_per_nm3=normal_density_kg_per_m3,
    )

    def state(outlet: float) -> DayState:
        return DayState(
            label=point.label,
            ambient_celsius=point.ambient_celsius,
            process_air_nm3h=point.process_air_nm3h,
            atomising_air_nm3h=point.atomising_air_nm3h,
            supply_celsius=point.supply_celsius,
            outlet_celsius=outlet,
            spray_kg_h=point.spray_kg_h,
            process_fan_kw=0.0,
            exhaust_fan_kw=0.0,
            preheater_loop_flow_m3h=0.0,
            preheater_water_in_celsius=0.0,
            preheater_duty_kw=0.0,
        )

    return {
        "solid_fraction": solid_fraction,
        "ua_outlet_exhaust_kw_per_k": loss_conductance_kw_per_k(ph, state(point.exhaust_celsius)),
        "ua_outlet_bed_kw_per_k": loss_conductance_kw_per_k(ph, state(point.bed_celsius)),
    }

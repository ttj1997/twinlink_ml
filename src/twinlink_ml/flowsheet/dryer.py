"""The dryer unit: the block of the equation document (1.1-1.5) with exchangeable assumptions.

A thin wrapper of :func:`twinlink_ml.partner_v4_1.dryer_block.dryer_block`; the three switches
that act on a dryer are

``cp_gas``
    the gas and vapour properties of the block (partner 1.2 / 2.0, physical 1.006 / 1.86 kJ/(kg K));
``heat_loss``
    ``UA (T_out - T_amb)`` of the bed (partner: none, physical: the laboratory placeholder);
``product_moisture``
    the moisture ``X_p`` the product leaves with (partner: the equilibrium moisture of the sheet,
    2 g/kg; physical: the value of the fluidised-bed model, **dryer 2 only**, an extrapolation of
    the laboratory model, docs/11, 9.6).

With all three on ``"partner"`` it reproduces ``Balances!F4/F5`` and ``F11/F12`` of the workbook
(tested to rounding, ``partner_v4.corrected.dryer_balance``).

``saturation_limit`` (not a switch of the six-scenario flowsheet; off by default, as in the partner
model) caps the exhaust at saturation: if the block would leave the exhaust above the saturation
humidity at its own temperature, the dryer evaporates only as much as saturates the exhaust, the
rest of the water leaves with the product, and the exhaust temperature follows from the same
energy balance with the smaller evaporation (bisection; the balance falls with the temperature).
Saturation is that of :mod:`.validators` (IAPWS-IF97 at the ambient pressure), so a limited
exhaust passes check (a) by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from twinlink_ml.partner_v4 import corrected as c
from twinlink_ml.partner_v4.inputs import SiAmbient, SiDryer, SiMaterial
from twinlink_ml.partner_v4_1.dryer_block import (
    BlockInputs,
    BlockProperties,
    BlockResult,
    dryer_block,
)

from .base import Assumption, Unit
from .thermo import GasProps
from .validators import saturation_humidity_kg_per_kg

__all__ = ["DryerBlock", "DryerConfig", "DryerInputs", "DryerOut"]


@dataclass(frozen=True)
class DryerConfig:
    """One dryer: the material, the ambient, and the values behind the three switches."""

    material: SiMaterial
    ambient: SiAmbient
    props: GasProps
    """Gas properties in use (follows the ``cp_gas`` switch)."""
    heat_loss_ua_w_per_k: float
    """0 for ``heat_loss="partner"``."""
    product_moisture_kg_per_kg: float
    """``X_p``: the sheet's equilibrium moisture, or the fluidised-bed model's value."""
    assumptions: tuple[tuple[str, Assumption], ...] = ()
    """The switch values behind this configuration (for the record only)."""
    name: str = "dryer"
    saturation_limit: bool = False
    """Cap the exhaust at saturation (see the module doc); off in the partner model."""


@dataclass(frozen=True)
class DryerInputs:
    """The sheet's dryer block (air flow, humidity, inlet temperature, spray)."""

    dryer: SiDryer


@dataclass(frozen=True)
class DryerOut:
    """The exhaust of the dryer as a stream, plus the evaporation and the heater duty it implies."""

    exhaust: c.Stream
    evaporation_kg_per_s: float
    heater_power_w: float
    """Duty of the air heater from ambient to the dryer's inlet temperature (no efficiency)."""
    result: BlockResult
    product_kg_per_s: float = math.nan
    """Solid plus the water it leaves with; the product leaves at the exhaust temperature."""
    product_moisture_kg_per_kg: float = math.nan
    """``X_p`` (dry basis): the configured one, or higher where the saturation limit binds."""
    saturation_limited: bool = False


class DryerBlock(Unit[DryerInputs, DryerOut]):
    """The simplified dryer block (equations 1.1-1.5)."""

    def __init__(self, config: DryerConfig) -> None:
        self.config = config
        self.name = config.name

    def assumptions(self) -> dict[str, Assumption]:
        return dict(self.config.assumptions)

    def properties(self) -> BlockProperties:
        m, p = self.config.material, self.config.props
        return BlockProperties(
            cp_gas_j_per_kg_k=p.cp_air,
            cp_vapour_j_per_kg_k=p.cp_vapour,
            cp_liquid_j_per_kg_k=m.cp_liquid_j_per_kg_k,
            cp_solid_j_per_kg_k=m.cp_solid_j_per_kg_k,
            evaporation_enthalpy_0c_j_per_kg=m.enthalpy_of_evaporation_j_per_kg,
            heat_loss_ua_w_per_k=self.config.heat_loss_ua_w_per_k,
        )

    def solve(self, inputs: DryerInputs) -> DryerOut:
        d = inputs.dryer
        block = BlockInputs(
            dry_air_mass_flow_kg_per_s=d.air_mass_flow_kg_per_s,
            air_humidity_kg_per_kg=d.air_humidity_kg_per_kg,
            air_temperature_celsius=d.air_temperature_celsius,
            spray_mass_flow_kg_per_s=d.spray_mass_flow_kg_per_s,
            spray_solid_fraction_kg_per_kg=d.spray_solid_fraction_kg_per_kg,
            spray_temperature_celsius=d.spray_temperature_celsius,
            product_moisture_kg_per_kg=self.config.product_moisture_kg_per_kg,
            ambient_temperature_celsius=self.config.ambient.temperature_celsius,
        )
        props = self.properties()
        result = dryer_block(block, props)
        limited = False
        if self.config.saturation_limit:
            pressure = self.config.ambient.pressure_pa
            y_sat = saturation_humidity_kg_per_kg(result.outlet_temperature_celsius, pressure)
            if result.outlet_humidity_kg_per_kg > y_sat:
                result = _saturated_exhaust(block, props, result, pressure)
                limited = True
        solid = d.spray_mass_flow_kg_per_s * d.spray_solid_fraction_kg_per_kg
        water_left = d.spray_mass_flow_kg_per_s - solid - result.evaporation_kg_per_s
        return DryerOut(
            exhaust=c.Stream(
                d.air_mass_flow_kg_per_s,
                result.outlet_humidity_kg_per_kg,
                result.outlet_temperature_celsius,
            ),
            evaporation_kg_per_s=result.evaporation_kg_per_s,
            heater_power_w=result.heater_power_w,
            result=result,
            product_kg_per_s=solid + water_left,
            product_moisture_kg_per_kg=water_left / solid if solid > 0.0 else 0.0,
            saturation_limited=limited,
        )


def _saturated_exhaust(
    block: BlockInputs, props: BlockProperties, unlimited: BlockResult, pressure_pa: float
) -> BlockResult:
    """The block with the evaporation that saturates the exhaust (see the module doc).

    ``f(T) = C_air (T_in - T) + C_spray (T_s - T) - E(T) dh_v(T) + UA (T_amb - T)`` with
    ``E(T) = M_g (Y_sat(T) - Y_in)`` falls with ``T``; it is positive at the unlimited outlet
    (less evaporation than there) and negative at the inlet temperature, so the root is between.
    """
    air = block.dry_air_mass_flow_kg_per_s
    y_in = block.air_humidity_kg_per_kg
    c_air = air * (props.cp_gas_j_per_kg_k + y_in * props.cp_vapour_j_per_kg_k)
    c_spray = block.spray_mass_flow_kg_per_s * props.suspension_cp(
        block.spray_solid_fraction_kg_per_kg
    )
    ua = props.heat_loss_ua_w_per_k

    def f(t: float) -> float:
        y_s = saturation_humidity_kg_per_kg(t, pressure_pa)
        if math.isinf(y_s):
            return -math.inf
        return (
            c_air * (block.air_temperature_celsius - t)
            + c_spray * (block.spray_temperature_celsius - t)
            - air * (y_s - y_in) * props.latent_heat_j_per_kg(t)
            + ua * (block.ambient_temperature_celsius - t)
        )

    lo, hi = unlimited.outlet_temperature_celsius, block.air_temperature_celsius
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0.0 else (lo, mid)
        if hi - lo <= 1e-10:
            break
    t_out = 0.5 * (lo + hi)
    y_out = saturation_humidity_kg_per_kg(t_out, pressure_pa)
    return BlockResult(
        outlet_humidity_kg_per_kg=y_out,
        outlet_temperature_celsius=t_out,
        evaporation_kg_per_s=air * (y_out - y_in),
        heater_power_w=unlimited.heater_power_w,
    )

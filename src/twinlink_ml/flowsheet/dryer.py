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
"""

from __future__ import annotations

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
        result = dryer_block(block, self.properties())
        return DryerOut(
            exhaust=c.Stream(
                d.air_mass_flow_kg_per_s,
                result.outlet_humidity_kg_per_kg,
                result.outlet_temperature_celsius,
            ),
            evaporation_kg_per_s=result.evaporation_kg_per_s,
            heater_power_w=result.heater_power_w,
            result=result,
        )

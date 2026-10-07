r"""The simplified dryer block of the equation document (equations 1.1-1.5), generalised.

One block, written once, used for the comparison with the fluidised-bed model
(``evaluation/dryer_block_comparison.py``). Enthalpies are referred to 0 degC with liquid water
at zero enthalpy, as in equation 1.4 of the document: dry gas ``cp_g T``, vapour
``dh_0 + cp_v T``, liquid ``cp_l T``, solid ``cp_s T``.

Equations 1.3 and 1.5 are the special case ``suspension_cp = None`` (mixture rule),
``latent_heat_slope = None`` (``cp_v - cp_l``) and no heat loss. Then, with ``E`` the evaporated
mass flow, the energy balance of the document

.. math:: M_g (c_{p,g} + Y_{in} c_{p,v}) T_{in} - E\,\Delta h_0 + M_s c_{p,susp} T_s
    = \bigl[M_g (c_{p,g} + Y_{out} c_{p,v}) + M_p (c_{p,s} + X_p c_{p,l})\bigr] T_{out}

is the same as the balance written over the evaporation enthalpy at the exhaust temperature,

.. math:: M_g (c_{p,g} + Y_{in} c_{p,v})(T_{in} - T_{out}) + M_s c_{p,susp}(T_s - T_{out})
    - E\,\Delta h_v(T_{out}) - UA\,(T_{out} - T_{amb}) = 0,
    \qquad \Delta h_v(T) = \Delta h_0 + (c_{p,v} - c_{p,l})\,T,

which is the form of the fluidised-bed model (``greybox._rhs``). :func:`dryer_block` solves this
second form, which is linear in ``T_out``; the three optional fields of
:class:`BlockProperties` are the places where the two models can differ (the document's
properties, a suspension heat capacity that is not the mixture rule, a latent-heat correlation
with another slope, a heat loss to the surroundings). With the defaults it reproduces
:func:`twinlink_ml.partner_v4.corrected.dryer_balance` (tested to rounding).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "BlockInputs",
    "BlockProperties",
    "BlockResult",
    "dryer_block",
    "heater_power_w",
]


@dataclass(frozen=True)
class BlockProperties:
    """Material properties and optional extras of the block, SI (J, kg, K, W)."""

    cp_gas_j_per_kg_k: float
    cp_vapour_j_per_kg_k: float
    cp_liquid_j_per_kg_k: float
    cp_solid_j_per_kg_k: float
    evaporation_enthalpy_0c_j_per_kg: float
    """``dh_0``: evaporation enthalpy at the reference temperature 0 degC (sheet: ``h_fg``)."""
    suspension_cp_j_per_kg_k: float | None = None
    """``None``: mixture rule ``(1 - X_s) cp_l + X_s cp_s`` (equation 1.4); else a fixed value."""
    latent_heat_slope_j_per_kg_k: float | None = None
    """``d(dh_v)/dT``; ``None``: ``cp_v - cp_l`` (consistent with the enthalpies of 1.4)."""
    heat_loss_ua_w_per_k: float = 0.0
    """Heat loss ``UA (T_out - T_amb)`` of the bed to the surroundings; 0 in the document."""

    def suspension_cp(self, solid_fraction: float) -> float:
        if self.suspension_cp_j_per_kg_k is not None:
            return self.suspension_cp_j_per_kg_k
        return (
            1.0 - solid_fraction
        ) * self.cp_liquid_j_per_kg_k + solid_fraction * self.cp_solid_j_per_kg_k

    @property
    def latent_heat_slope(self) -> float:
        if self.latent_heat_slope_j_per_kg_k is not None:
            return self.latent_heat_slope_j_per_kg_k
        return self.cp_vapour_j_per_kg_k - self.cp_liquid_j_per_kg_k

    def latent_heat_j_per_kg(self, temperature_celsius: float) -> float:
        """Evaporation enthalpy at ``T`` (linear: ``dh_0 + slope * T``)."""
        return self.evaporation_enthalpy_0c_j_per_kg + self.latent_heat_slope * temperature_celsius


@dataclass(frozen=True)
class BlockInputs:
    """Inputs of the block (nomenclature of the document), SI, temperatures in degC."""

    dry_air_mass_flow_kg_per_s: float
    """``M_g``."""
    air_humidity_kg_per_kg: float
    """``Y_in``."""
    air_temperature_celsius: float
    """``T_g,in``, after the heater."""
    spray_mass_flow_kg_per_s: float
    """``M_s``."""
    spray_solid_fraction_kg_per_kg: float
    """``X_s``."""
    spray_temperature_celsius: float
    """``T_s``."""
    product_moisture_kg_per_kg: float
    """``X_p`` (dry basis): the document sets it to the equilibrium moisture."""
    ambient_temperature_celsius: float
    """Where the air heater starts (and the reference of the heat loss)."""


@dataclass(frozen=True)
class BlockResult:
    """Outlet state of the block; powers in W, flows in kg/s."""

    outlet_humidity_kg_per_kg: float
    """``Y_out`` (equation 1.3)."""
    outlet_temperature_celsius: float
    """``T_g,out = T_p`` (equation 1.5)."""
    evaporation_kg_per_s: float
    heater_power_w: float
    """Thermal duty of the air heater, from ambient to ``T_g,in`` (no efficiency)."""


def heater_power_w(inputs: BlockInputs, props: BlockProperties) -> float:
    """``M_g (cp_g + Y_in cp_v)(T_in - T_amb)``, the duty before any efficiency; not negative."""
    return (
        inputs.dry_air_mass_flow_kg_per_s
        * (props.cp_gas_j_per_kg_k + inputs.air_humidity_kg_per_kg * props.cp_vapour_j_per_kg_k)
        * max(inputs.air_temperature_celsius - inputs.ambient_temperature_celsius, 0.0)
    )


def dryer_block(inputs: BlockInputs, props: BlockProperties) -> BlockResult:
    """Equations 1.1-1.5 (with the optional extras of :class:`BlockProperties`)."""
    air = inputs.dry_air_mass_flow_kg_per_s
    xs = inputs.spray_solid_fraction_kg_per_kg
    spray = inputs.spray_mass_flow_kg_per_s
    # (1.1)/(1.2)/(1.3): water of the spray minus the water that leaves with the product
    evaporation = spray * (1.0 - xs * (1.0 + inputs.product_moisture_kg_per_kg))
    y_out = inputs.air_humidity_kg_per_kg + evaporation / air
    air_heat_rate = air * (
        props.cp_gas_j_per_kg_k + inputs.air_humidity_kg_per_kg * props.cp_vapour_j_per_kg_k
    )
    spray_heat_rate = spray * props.suspension_cp(xs)
    numerator = (
        air_heat_rate * inputs.air_temperature_celsius
        + spray_heat_rate * inputs.spray_temperature_celsius
        - evaporation * props.evaporation_enthalpy_0c_j_per_kg
        + props.heat_loss_ua_w_per_k * inputs.ambient_temperature_celsius
    )
    denominator = (
        air_heat_rate
        + spray_heat_rate
        + evaporation * props.latent_heat_slope
        + props.heat_loss_ua_w_per_k
    )
    return BlockResult(
        outlet_humidity_kg_per_kg=y_out,
        outlet_temperature_celsius=numerator / denominator,
        evaporation_kg_per_s=evaporation,
        heater_power_w=heater_power_w(inputs, props),
    )

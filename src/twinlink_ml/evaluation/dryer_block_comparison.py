r"""Comparison of the partner's simplified dryer block with the fluidised-bed model.

The block is :func:`twinlink_ml.partner_v4_1.dryer_block.dryer_block` (equations 1.3 and 1.5
of the equation document). The bed model is the steady state of
:func:`twinlink_ml.simulation.steady_state.solve_steady_state`. This module holds

* the **mapping** of the bed model's quantities to the block's inputs (:func:`block_inputs`),
* the two **property sets** (:func:`partner_properties`, :func:`own_properties`),
* the **stepwise decomposition** from the block as delivered to the bed model
  (:data:`STEP_ORDER`, :func:`apply_steps`): which property or structural difference is removed
  at which step,
* the **exhaust recovery** of one stream down to ``T_cold`` (:func:`exhaust_recovery`), sensible
  and latent, with the same formula for both models (it is the condensing recovery of
  ``heatnetwork.balance`` for one source).

The decomposition is sequential: a step's contribution depends on the steps before it. Callers
that report contributions should say so and may run a second order (:func:`apply_steps` takes the
order as an argument).

Reference states (the question "which enthalpy is referred to what"): the block refers every
enthalpy to 0 degC with liquid water at zero (equation 1.4). The bed model's balance is written
in differences to the bed temperature (``greybox._rhs``), ``SteadyStateSolution.exhaust_*_kw``
refer to the ambient state, and the recovery here refers to ``T_cold``. The two balances are
identical when the evaporation enthalpy at ``T`` is ``dh_0 + (cp_v - cp_l) T`` and the suspension
heat capacity is the mixture rule (see ``dryer_block``); the remaining differences are properties,
and the decomposition separates them. All powers compared are thermal (no efficiency).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Final

import numpy as np

from ..partner_v4_1.dryer_block import (
    BlockInputs,
    BlockProperties,
    BlockResult,
    dryer_block,
    heater_power_w,
)
from ..simulation.backend import NumpyBackend
from ..simulation.config import GasConfig, ProcessConfig
from ..simulation.sorption import isotherm_from_material
from ..simulation.steady_state import SteadyStateSolution
from ..simulation.thermo import (
    dew_point_celsius,
    latent_heat_j_per_kg,
    relative_humidity_from_absolute_humidity,
    saturation_humidity_kg_per_kg,
)

__all__ = [
    "PARTNER_EQUILIBRIUM_MOISTURE_KG_PER_KG",
    "STEP_ORDER",
    "STEP_ORDER_REVERSED",
    "T_COLD_HEATNETWORK_DEFAULT_CELSIUS",
    "T_COLD_SITE_CELSIUS",
    "Recovery",
    "apply_steps",
    "block_inputs",
    "block_is_normal",
    "evaluate_block",
    "exhaust_recovery",
    "own_equilibrium_moisture",
    "own_properties",
    "own_result",
    "partner_properties",
    "quantities",
]

_BACKEND: Final = NumpyBackend()

PARTNER_EQUILIBRIUM_MOISTURE_KG_PER_KG: Final = 0.002
"""``Inputs`` of ``Model_v4``/``v4.1``: equilibrium moisture 2 g/kg (``docs/11``, section 1)."""

T_COLD_SITE_CELSIUS: Final = 29.1
"""``SiteSpec.t_kalt_celsius`` of the site model (M2/M3): placeholder, no source."""
T_COLD_HEATNETWORK_DEFAULT_CELSIUS: Final = 15.0
"""``DEFAULT_COLD_TEMPERATURE_CELSIUS`` of ``heatnetwork`` (= ambient of the partner model)."""

STEP_ORDER: Final = ("b_gas_properties", "b2_other_properties", "c_product_moisture", "d_heat_loss")
STEP_ORDER_REVERSED: Final = tuple(reversed(STEP_ORDER))


def partner_properties() -> BlockProperties:
    """The material properties of the partner workbook (``Inputs`` G3:I12; ``docs/11`` section 1).

    ``cp_gas`` 1.2, ``cp_vapour`` 2.0, ``cp_liquid`` 4.2, ``cp_solid`` 1.0 kJ/(kg K), ``h_fg``
    2500 kJ/kg; no heat loss; suspension heat capacity by the mixture rule.
    """
    return BlockProperties(
        cp_gas_j_per_kg_k=1200.0,
        cp_vapour_j_per_kg_k=2000.0,
        cp_liquid_j_per_kg_k=4200.0,
        cp_solid_j_per_kg_k=1000.0,
        evaporation_enthalpy_0c_j_per_kg=2.5e6,
    )


def own_properties(config: ProcessConfig) -> BlockProperties:
    """The bed model's properties in the block's form.

    The evaporation enthalpy is read off ``thermo.latent_heat_j_per_kg`` (value at 0 degC and
    slope), not retyped; the suspension heat capacity is the fixed ``cp_suspension`` of the
    material; the heat loss is ``geometry.heat_loss_ua_w_per_k``.
    """
    gas, material = config.gas, config.material
    at_0 = float(latent_heat_j_per_kg(_BACKEND, np.array(0.0)))
    at_100 = float(latent_heat_j_per_kg(_BACKEND, np.array(100.0)))
    return BlockProperties(
        cp_gas_j_per_kg_k=gas.cp_dry_air_j_per_kg_k,
        cp_vapour_j_per_kg_k=gas.cp_vapour_j_per_kg_k,
        cp_liquid_j_per_kg_k=material.cp_water_liquid_j_per_kg_k,
        cp_solid_j_per_kg_k=material.cp_solid_j_per_kg_k,
        evaporation_enthalpy_0c_j_per_kg=at_0,
        suspension_cp_j_per_kg_k=material.cp_suspension_j_per_kg_k,
        latent_heat_slope_j_per_kg_k=(at_100 - at_0) / 100.0,
        heat_loss_ua_w_per_k=config.geometry.heat_loss_ua_w_per_k,
    )


def block_inputs(config: ProcessConfig, solution: SteadyStateSolution) -> BlockInputs:
    """Map the bed model's operating point and steady state to the block's inputs.

    ``config`` is the variant config the steady state was solved with (the solver reads actions
    and disturbances from the ``nominal`` fields). ``M_g`` is the solver's
    ``dry_air_mass_flow_kg_per_s`` (density at the **bed** temperature times gas velocity times
    cross section, divided by ``1 + Y_in``); ``Y_in`` follows from the ambient state; the
    product moisture is the bed's own ``X_p``.
    """
    actions, disturbances = config.actions, config.disturbances
    diagnostics = solution.diagnostics
    return BlockInputs(
        dry_air_mass_flow_kg_per_s=solution.dry_air_mass_flow_kg_per_s,
        air_humidity_kg_per_kg=diagnostics["inlet_humidity_kg_per_kg"],
        air_temperature_celsius=actions.inlet_temperature_celsius.nominal,
        spray_mass_flow_kg_per_s=actions.spray_rate_kg_per_h.nominal / 3600.0,
        spray_solid_fraction_kg_per_kg=disturbances.suspension_solid_fraction.nominal,
        spray_temperature_celsius=config.operation.suspension_temperature_celsius,
        product_moisture_kg_per_kg=solution.particle_moisture_kg_per_kg,
        ambient_temperature_celsius=disturbances.ambient_temperature_celsius.nominal,
    )


def apply_steps(
    steps: tuple[str, ...],
    own_inputs: BlockInputs,
    own_props: BlockProperties,
    *,
    partner_props: BlockProperties | None = None,
) -> tuple[BlockInputs, BlockProperties]:
    """Block inputs and properties after the given steps (all others as delivered).

    Delivered state: partner properties, ``X_p`` = the partner's equilibrium moisture, no heat
    loss, everything else from the bed model's mapping (``own_inputs``). The steps are

    ``b_gas_properties``
        ``cp_gas``, ``cp_vapour``, ``dh_0`` of the bed model;
    ``b2_other_properties``
        ``cp_liquid``, ``cp_solid``, the fixed suspension heat capacity (not in the task's list
        of b; kept apart so that b is exactly what was asked);
    ``c_product_moisture``
        ``X_p`` of the bed model instead of the equilibrium moisture;
    ``d_heat_loss``
        ``UA (T - T_amb)`` of the bed model;
    ``e_latent_slope``
        the slope of the bed model's evaporation-enthalpy correlation instead of
        ``cp_v - cp_l`` (verification, not part of the order: it is what remains of the
        "structure" after b-d).
    """
    props = partner_props if partner_props is not None else partner_properties()
    inputs = dataclasses.replace(
        own_inputs, product_moisture_kg_per_kg=PARTNER_EQUILIBRIUM_MOISTURE_KG_PER_KG
    )
    for step in steps:
        if step == "b_gas_properties":
            props = dataclasses.replace(
                props,
                cp_gas_j_per_kg_k=own_props.cp_gas_j_per_kg_k,
                cp_vapour_j_per_kg_k=own_props.cp_vapour_j_per_kg_k,
                evaporation_enthalpy_0c_j_per_kg=own_props.evaporation_enthalpy_0c_j_per_kg,
            )
        elif step == "b2_other_properties":
            props = dataclasses.replace(
                props,
                cp_liquid_j_per_kg_k=own_props.cp_liquid_j_per_kg_k,
                cp_solid_j_per_kg_k=own_props.cp_solid_j_per_kg_k,
                suspension_cp_j_per_kg_k=own_props.suspension_cp_j_per_kg_k,
            )
        elif step == "c_product_moisture":
            inputs = dataclasses.replace(
                inputs, product_moisture_kg_per_kg=own_inputs.product_moisture_kg_per_kg
            )
        elif step == "d_heat_loss":
            props = dataclasses.replace(props, heat_loss_ua_w_per_k=own_props.heat_loss_ua_w_per_k)
        elif step == "e_latent_slope":
            props = dataclasses.replace(
                props, latent_heat_slope_j_per_kg_k=own_props.latent_heat_slope_j_per_kg_k
            )
        else:
            raise ValueError(f"unknown step {step!r}")
    return inputs, props


@dataclass(frozen=True)
class Recovery:
    """Heat of an exhaust stream cooled to ``T_cold`` (W), see :func:`exhaust_recovery`."""

    sensible_w: float
    latent_w: float
    condensate_kg_per_s: float

    @property
    def total_w(self) -> float:
        return self.sensible_w + self.latent_w


def exhaust_recovery(
    gas: GasConfig,
    props: BlockProperties,
    *,
    dry_air_mass_flow_kg_per_s: float,
    outlet_humidity_kg_per_kg: float,
    outlet_temperature_celsius: float,
    cold_temperature_celsius: float,
) -> Recovery:
    r"""Condensing recovery of one stream down to ``T_cold`` (same formula as ``heatnetwork``).

    .. math:: Q_{sens} = M_g (c_{p,g} + Y_{out} c_{p,v})(T_{out} - T_{cold}),\qquad
        Q_{lat} = M_g \max(0, Y_{out} - Y_s(T_{cold}))\,\Delta h_v(T_{cold})

    The saturation humidity is the project's Buck correlation for both models (the block has no
    saturation formula of its own that is in use: the Antoine table of the workbook is not
    referenced by a formula), the properties and the evaporation enthalpy are the given set's.
    A stream that leaves colder than ``T_cold`` gives zero, not a negative number.
    """
    y_sat_cold = float(
        saturation_humidity_kg_per_kg(_BACKEND, gas, np.array(cold_temperature_celsius))
    )
    sensible = (
        dry_air_mass_flow_kg_per_s
        * (props.cp_gas_j_per_kg_k + outlet_humidity_kg_per_kg * props.cp_vapour_j_per_kg_k)
        * max(outlet_temperature_celsius - cold_temperature_celsius, 0.0)
    )
    condensate = dry_air_mass_flow_kg_per_s * max(0.0, outlet_humidity_kg_per_kg - y_sat_cold)
    latent = condensate * props.latent_heat_j_per_kg(cold_temperature_celsius)
    return Recovery(sensible_w=sensible, latent_w=latent, condensate_kg_per_s=condensate)


def own_equilibrium_moisture(
    config: ProcessConfig, outlet_humidity_kg_per_kg: float, outlet_temperature_celsius: float
) -> float:
    """The bed model's local equilibrium moisture ``X_eq(phi_out, T_out)`` (GAB isotherm)."""
    isotherm = isotherm_from_material(config.material)
    phi = relative_humidity_from_absolute_humidity(
        _BACKEND,
        config.gas,
        np.array(outlet_humidity_kg_per_kg),
        np.array(outlet_temperature_celsius),
    )
    return float(isotherm.equilibrium_moisture(_BACKEND, phi, np.array(outlet_temperature_celsius)))


def quantities(
    gas: GasConfig,
    inputs: BlockInputs,
    props: BlockProperties,
    result: BlockResult,
    cold_temperatures_celsius: tuple[float, ...],
) -> dict[str, float]:
    """The compared quantities of one block (or bed) state; powers in kW, humidity in kg/kg."""
    out: dict[str, float] = {
        "outlet_temperature_celsius": result.outlet_temperature_celsius,
        "outlet_humidity_kg_per_kg": result.outlet_humidity_kg_per_kg,
        "product_moisture_kg_per_kg": inputs.product_moisture_kg_per_kg,
        "heater_power_kw": result.heater_power_w / 1000.0,
        "dew_point_celsius": float(
            dew_point_celsius(_BACKEND, gas, np.array(result.outlet_humidity_kg_per_kg))
        ),
    }
    for t_cold in cold_temperatures_celsius:
        recovery = exhaust_recovery(
            gas,
            props,
            dry_air_mass_flow_kg_per_s=inputs.dry_air_mass_flow_kg_per_s,
            outlet_humidity_kg_per_kg=result.outlet_humidity_kg_per_kg,
            outlet_temperature_celsius=result.outlet_temperature_celsius,
            cold_temperature_celsius=t_cold,
        )
        key = f"{t_cold:g}"
        out[f"recovery_sensible_kw_tcold_{key}"] = recovery.sensible_w / 1000.0
        out[f"recovery_latent_kw_tcold_{key}"] = recovery.latent_w / 1000.0
        out[f"recovery_total_kw_tcold_{key}"] = recovery.total_w / 1000.0
    return out


def own_result(
    inputs: BlockInputs, solution: SteadyStateSolution, props: BlockProperties
) -> BlockResult:
    """The bed model's steady state in the block's result form (same heater formula)."""
    return BlockResult(
        outlet_humidity_kg_per_kg=solution.outlet_humidity_kg_per_kg,
        outlet_temperature_celsius=solution.bed_temperature_celsius,
        evaporation_kg_per_s=solution.diagnostics["evaporation_rate_kg_per_s"],
        heater_power_w=heater_power_w(inputs, props),
    )


def block_is_normal(gas: GasConfig, result: BlockResult) -> dict[str, bool]:
    """Check the three ways in which the block can show an impossible exhaust.

    The block has no moisture limit and no feasibility flag. What it can show is a physically
    impossible exhaust: negative evaporation, supersaturation (``Y_out`` above ``Y_s(T_out)``), or
    a dew point at or above the exhaust temperature.
    """
    y_sat = float(
        saturation_humidity_kg_per_kg(_BACKEND, gas, np.array(result.outlet_temperature_celsius))
    )
    dew = float(dew_point_celsius(_BACKEND, gas, np.array(result.outlet_humidity_kg_per_kg)))
    evaporation_ok = result.evaporation_kg_per_s >= 0.0
    unsaturated = result.outlet_humidity_kg_per_kg < y_sat
    above_dew = result.outlet_temperature_celsius > dew
    return {
        "evaporation_not_negative": evaporation_ok,
        "unsaturated": unsaturated,
        "above_dew_point": above_dew,
        "normal": evaporation_ok and unsaturated and above_dew,
    }


def evaluate_block(
    gas: GasConfig,
    inputs: BlockInputs,
    props: BlockProperties,
    cold_temperatures_celsius: tuple[float, ...],
) -> dict[str, float]:
    """``dryer_block`` plus :func:`quantities` in one call."""
    result = dryer_block(inputs, props)
    return quantities(gas, inputs, props, result, cold_temperatures_celsius)

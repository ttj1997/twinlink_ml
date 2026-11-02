r"""Exhaust air outlet variables from (predicted) bed temperature and particle moisture.

Physical origin: Befund 31 (``docs/04_stand_ap1.md``, "exhaust air as an
interface variable") already computes mass flow, moisture load and temperature of the exhaust air
once -- there as a **one-off diagnostic calculation** directly from the
steady-state solver (``steady_state.py``), with the exact, simulated state as
input. This function moves the same formula to a different input side: bed temperature
and particle moisture as **arguments** instead of as a by-product of a full
ODE integration/bisection -- suitable for values that a sequence model has predicted
(``bed_temperature_celsius``, ``product_moisture_kg_per_kg``), not only for simulator-
internal state.

Two algebraic facts from the model, used here, not derived anew:

1. **Well-mixed bed**: :math:`T_{gas,out} = T_{bed}` exactly (see the
   ``thermo.evaporation_rate_kg_per_s`` docstring: "Assumption: ideally mixed bed,
   exhaust air at bed temperature") -- the same equation that
   ``steady_state.SteadyStateSolution.bed_temperature_celsius`` also uses for
   ``exhaust_sensible_kw``, here only named explicitly instead of implicitly in a
   local variable.
2. **Mass balance of the exhaust air humidity**: :math:`Y_{gas,out} = Y_{in} +
   \dot m_{evap}/\dot m_{L,tr}`, with :math:`\dot m_{evap}` from the same
   damped Picard iteration as in the rollout itself (``thermo.evaporation_rate_kg_per_s``,
   Befund 34).

**No new physics.** This function calls ``evaporation_rate_kg_per_s`` unchanged --
it computes nothing that the simulator does not compute anyway for every time step.
Checked against ``solve_steady_state()`` at the nominal point from Befund 31: bit-identical
agreement (``tests/test_exhaust.py``).

**What is NOT covered.** ``dry_air_mass_flow_kg_per_s`` and
``inlet_absolute_humidity_kg_per_kg`` are arguments here, not return values -- they
depend on actions (gas velocity) and disturbances (ambient temperature/humidity)
that are known to a sequence-model caller anyway (part of the action/disturbance, not
of the prediction), not on variables that would first have to be predicted. Unlike
``bed_temperature_celsius``: at the time of this module (``docs/04_stand_ap1.md``,
Befund 26-40) it is **not** a learned target of the sequence models -- this function does not
presuppose it, but only becomes useful once it is one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .backend import Array, ArrayBackend
from .config import ExhaustConfig, GasConfig, MaterialConfig
from .thermo import evaporation_rate_kg_per_s

if TYPE_CHECKING:
    # The same dependency-injection reasoning as in thermo.py: the isotherm is
    # injected instead of being constructed internally from ``material``; no runtime import needed.
    from .sorption import SorptionIsotherm

__all__ = ["reconstruct_exhaust_state"]


def reconstruct_exhaust_state(
    backend: ArrayBackend,
    gas: GasConfig,
    material: MaterialConfig,
    isotherm: SorptionIsotherm,
    exhaust: ExhaustConfig,
    *,
    bed_temperature_celsius: Array,
    particle_moisture_kg_per_kg: Array,
    dry_air_mass_flow_kg_per_s: Array,
    inlet_absolute_humidity_kg_per_kg: Array,
) -> tuple[Array, Array, Array]:
    r"""Exhaust air temperature and humidity from bed temperature and particle moisture.

    Contract correction (formerly ``reconstruct_exhaust_state`` returned, under the name
    "exhaust air/exhaust state", actually ``bed_temperature_celsius`` -- numerically correct under
    the assumption of an ideally mixed bed, but a contract error at
    the integration interface: a partner wiring up "exhaust air temperature" would get the
    bed temperature). This function now additionally returns ``outlet_temperature_celsius``
    as a separate, explicitly named field -- the bed outlet temperature reduced by the
    duct/filter loss (``exhaust.outlet_duct_heat_loss_k``, placeholder, default 0.0).
    ``T_gas_out_celsius`` (first return value, unchanged) stays
    the gas temperature immediately at the bed outlet, before any duct loss -- by
    model assumption identical to ``bed_temperature_celsius`` (see module docstring point 1).

    :param backend: array backend (NumPy for data generation, Torch for differentiable
        rollouts) -- as everywhere in ``simulation``, via dependency injection.
    :param gas: gas configuration of the process (specific heat capacities, pressure).
    :param material: material configuration (among others for the saturation-degree upper bound in
        ``evaporation_rate_kg_per_s``).
    :param isotherm: sorption isotherm, see ``sorption.isotherm_from_material``.
    :param exhaust: duct/filter loss between bed and heat integration system
        (``ExhaustConfig``, placeholder, see ``config.py``).
    :param bed_temperature_celsius: bed temperature -- simulator state or
        sequence-model prediction; this function does not distinguish.
    :param particle_moisture_kg_per_kg: particle moisture in the bed (``X_p``) -- corresponds to
        ``product_moisture_kg_per_kg`` in ``output_names``/``target_names``.
    :param dry_air_mass_flow_kg_per_s: dry air mass flow -- derivable from the action
        (gas velocity) and disturbance/bed temperature, see
        ``greybox._rhs``; expected here as a known input, not part of the prediction.
    :param inlet_absolute_humidity_kg_per_kg: absolute humidity of the inlet air (``Y_in``) --
        from ambient conditions (disturbance), likewise a known input.
    :returns: ``(T_gas_out_celsius, Y_gas_out_kg_per_kg, outlet_temperature_celsius)``, all
        in the same units as ``bed_temperature_celsius``/``particle_moisture_kg_per_kg``.
    """
    _evaporation, outlet_humidity, _saturation_degree = evaporation_rate_kg_per_s(
        backend,
        gas,
        material,
        isotherm,
        dry_air_mass_flow_kg_per_s,
        bed_temperature_celsius,
        inlet_absolute_humidity_kg_per_kg,
        particle_moisture_kg_per_kg,
    )
    # Well-mixed bed (model assumption, see module docstring point 1): no calculation of its
    # own needed, T_gas_out is by construction the bed temperature passed in.
    outlet_temperature_celsius = bed_temperature_celsius - exhaust.outlet_duct_heat_loss_k
    return bed_temperature_celsius, outlet_humidity, outlet_temperature_celsius

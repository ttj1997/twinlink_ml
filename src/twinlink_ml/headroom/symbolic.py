r"""Symbolic (CasADi) form of the grey-box model for the optimising controllers.

The optimisers R1-R3 must use the SAME model as the truth simulation. This module therefore does
not re-derive any physics: :class:`CasadiBackend` implements the
:class:`~twinlink_ml.simulation.backend.ArrayBackend` protocol on CasADi ``SX`` scalars, and
:func:`symbolic_derivative` calls the unchanged functions of ``simulation.thermo``,
``simulation.sorption`` and ``simulation.energy`` with it. Only the composition of
:meth:`GreyBoxFluidBedGranulator._rhs` (about 40 lines, branch without exhaust recirculation) is
repeated here, and ``tests/test_dynamic_headroom.py`` pins it against ``evaluate_derivative`` at
random states.

Reduced state :math:`[m_s, m_w, T_{bed}]`: the particle count enters neither these three
derivatives nor the power (only ``d50`` diagnostics and its own balance), so the optimisers drop
it; the truth model keeps all four states. ``GreyBoxFluidBedGranulator._floor`` (projection onto
:math:`X \ge X_{eq}`) is a numerical safety net and is not part of the symbolic model; at the
operating points of this study (:math:`X \approx 0.1-0.15` against :math:`X_{eq} \approx 0.03`)
it is inactive.

Interval function
-----------------
:func:`interval_function` integrates one control interval with ``n`` RK4 steps of
``optimizer_step_s``. The commanded set point jumps at the interval start; the truth model rate
limits every change (``ActionBound.max_rate_per_s``, 0.5 K/s and 0.05 m/s per s) sample by sample,
and the ramp matters: a preview optimum swings the inlet temperature by tens of K, whose ramp lasts
up to two minutes of a five-minute interval (without the ramp in the optimisation model the
predicted moisture was off by up to 9e-3 kg/kg). Each RK4 step therefore uses the MEAN of the
ramped command over the step, :func:`ramp_mean` -- a C1 function of the command change (the mean
of a clipped ramp smooths the clip), unlike the clipped ramp itself, which made the
interior-point iterations erratic. The disturbances enter as one value per RK4 step (the mean
of the truth samples of that step, :func:`step_means`). Outputs: end state, moisture and bed
temperature at the end of every RK4 step (path condition), energy of the interval in kWh (power
at the start of each RK4 step times the step).
"""

from __future__ import annotations

import math
from typing import Any

import casadi as cs
import numpy as np

from twinlink_ml.simulation.config import ProcessConfig
from twinlink_ml.simulation.energy import power_breakdown_kw
from twinlink_ml.simulation.sorption import isotherm_from_material
from twinlink_ml.simulation.thermo import (
    absolute_humidity_kg_per_kg,
    evaporation_rate_kg_per_s,
    gas_density_kg_per_m3,
    latent_heat_j_per_kg,
)

__all__ = [
    "DISTURBANCE_CHANNELS",
    "CasadiBackend",
    "interval_function",
    "ramp_mean",
    "step_means",
    "symbolic_derivative",
]

DISTURBANCE_CHANNELS: tuple[str, ...] = (
    "ambient_temperature_celsius",
    "ambient_relative_humidity",
    "suspension_solid_fraction",
    "spray_rate_kg_per_h",
    "inlet_temperature_offset_k",
)
"""Disturbance vector of the optimisers. ``spray_rate`` (throughput) and the additive offset of
the realised inlet temperature are disturbances for the controllers, not decisions."""

_SECONDS_PER_HOUR = 3600.0


class CasadiBackend:
    """:class:`~twinlink_ml.simulation.backend.ArrayBackend` on CasADi ``SX`` expressions."""

    name = "casadi"

    @staticmethod
    def asarray(value: Any) -> Any:
        return value

    @staticmethod
    def exp(x: Any) -> Any:
        return cs.exp(x)

    @staticmethod
    def log(x: Any) -> Any:
        return cs.log(x)

    @staticmethod
    def sqrt(x: Any) -> Any:
        return cs.sqrt(x)

    @staticmethod
    def clip(x: Any, lower: Any, upper: Any) -> Any:
        return cs.fmin(cs.fmax(x, lower), upper)

    @staticmethod
    def maximum(x: Any, y: Any) -> Any:
        return cs.fmax(x, y)

    @staticmethod
    def minimum(x: Any, y: Any) -> Any:
        return cs.fmin(x, y)

    @staticmethod
    def where(condition: Any, x: Any, y: Any) -> Any:
        return cs.if_else(condition, x, y)

    @staticmethod
    def zeros_like(x: Any) -> Any:
        return cs.SX.zeros(cs.SX(x).shape)

    @staticmethod
    def stack(arrays: list[Any], axis: int) -> Any:  # noqa: ARG004 - shared signature
        return cs.vertcat(*arrays)

    @staticmethod
    def to_numpy(x: Any) -> Any:  # noqa: ARG004 - protocol signature
        raise TypeError("symbolic expressions have no numeric value")


def symbolic_derivative(
    config: ProcessConfig, state: Any, action: Any, disturbance: Any
) -> tuple[Any, Any]:
    """Derivative of :math:`[m_s, m_w, T_{bed}]` and total power in kW.

    ``action`` = (realised inlet temperature, gas velocity, spray rate kg/h, nozzle pressure),
    ``disturbance`` = (ambient temperature, ambient relative humidity, solid fraction) -- the
    same channel order as the truth model. Composition copied from
    :meth:`GreyBoxFluidBedGranulator._rhs` (no recirculation), see the module docstring.
    """
    if config.operation.exhaust_recirculation_fraction is not None:
        raise ValueError("the symbolic model covers the line without exhaust recirculation")
    if config.operation.mode != "continuous":
        raise ValueError("the symbolic model covers continuous operation only")
    backend = CasadiBackend()
    material, gas, geometry, operation = (
        config.material,
        config.gas,
        config.geometry,
        config.operation,
    )
    solid_mass = cs.fmax(state[0], config.numerics.state_floor)
    water_mass = cs.fmax(state[1], 0.0)
    bed_temperature = state[2]

    inlet_temperature = action[0]
    gas_velocity = action[1]
    spray_rate = action[2] / _SECONDS_PER_HOUR
    nozzle_pressure = action[3]
    ambient_temperature = disturbance[0]
    ambient_humidity = disturbance[1]
    solid_fraction = disturbance[2]

    moisture = water_mass / solid_mass
    isotherm = isotherm_from_material(material)
    inlet_absolute_humidity = absolute_humidity_kg_per_kg(
        backend, gas, ambient_temperature, ambient_humidity
    )
    bed_gas_density = gas_density_kg_per_m3(backend, gas, bed_temperature, inlet_absolute_humidity)
    moist_air_flow = bed_gas_density * gas_velocity * geometry.cross_section_area_m2
    dry_air_flow = moist_air_flow / (1.0 + inlet_absolute_humidity)
    evaporation, _, _ = evaporation_rate_kg_per_s(
        backend,
        gas,
        material,
        isotherm,
        dry_air_flow,
        bed_temperature,
        inlet_absolute_humidity,
        moisture,
    )

    spray_solid = spray_rate * solid_fraction
    seed_flow = operation.seed_mass_flow_kg_per_h / _SECONDS_PER_HOUR
    solid_inflow = spray_solid + seed_flow
    correction = (
        solid_mass - geometry.bed_mass_solid_setpoint_kg
    ) / operation.mass_control_time_constant_s
    product_flow = cs.fmax(solid_inflow + correction, 0.0)

    d_solid = solid_inflow - product_flow
    d_water = spray_rate * (1.0 - solid_fraction) - evaporation - product_flow * moisture
    heat_capacity = (
        solid_mass * material.cp_solid_j_per_kg_k + water_mass * material.cp_water_liquid_j_per_kg_k
    )
    heat_from_air = (
        dry_air_flow
        * (gas.cp_dry_air_j_per_kg_k + inlet_absolute_humidity * gas.cp_vapour_j_per_kg_k)
        * (inlet_temperature - bed_temperature)
    )
    heat_from_suspension = (
        spray_rate
        * material.cp_suspension_j_per_kg_k
        * (operation.suspension_temperature_celsius - bed_temperature)
    )
    heat_of_evaporation = evaporation * latent_heat_j_per_kg(backend, bed_temperature)
    heat_loss = geometry.heat_loss_ua_w_per_k * (bed_temperature - ambient_temperature)
    d_temperature = (
        heat_from_air + heat_from_suspension - heat_of_evaporation - heat_loss
    ) / heat_capacity

    power = power_breakdown_kw(
        backend,
        gas,
        material,
        config.energy,
        geometry.cross_section_area_m2,
        dry_air_flow,
        inlet_absolute_humidity,
        bed_gas_density,
        inlet_temperature,
        gas_velocity,
        spray_rate,
        nozzle_pressure,
        ambient_temperature,
        solid_mass + water_mass,
    )
    return cs.vertcat(d_solid, d_water, d_temperature), power.total_kw


def ramp_mean(delta: Any, rate_per_s: float, start_s: float, end_s: float) -> Any:
    r"""Mean over :math:`t \in [a, b]` of the ramp :math:`\mathrm{clip}(\Delta, -r t, r t)`.

    For :math:`r a < |\Delta| < r b`:
    :math:`\bigl(\Delta b - \Delta|\Delta|/(2r) - \mathrm{sign}(\Delta)\, r a^2/2\bigr)/(b-a)`;
    :math:`\Delta` below, :math:`\mathrm{sign}(\Delta)\, r (a+b)/2` above. Continuously
    differentiable in :math:`\Delta`, derivative 1 at :math:`\Delta = 0` also for :math:`a = 0`.
    """
    a, b, r = start_s, end_s, rate_per_s
    magnitude = cs.fabs(delta)
    middle = (delta * b - delta * magnitude / (2.0 * r) - cs.sign(delta) * r * a**2 / 2.0) / (b - a)
    full = cs.sign(delta) * r * (a + b) / 2.0
    return cs.if_else(magnitude <= r * a, delta, cs.if_else(magnitude >= r * b, full, middle))


def interval_function(
    config: ProcessConfig,
    *,
    interval_s: float,
    step_s: float,
    nozzle_pressure_bar: float,
    sample_time_s: float,
) -> cs.Function:
    """CasADi function ``(x0[3], u[2], u_prev[2], D[5, n]) -> (x_end, X[n], T[n], E_kwh)``.

    ``u_prev``: the (fully ramped) command of the previous interval. The truth model's command
    during the sample ``[t, t + dt)`` is the ramp value at ``t + dt``; the ramp time of this
    model is shifted by ``sample_time_s / 2`` to match that on average.

    ``u`` = (commanded inlet temperature, commanded gas velocity), ``D`` = one column per RK4
    step in the order of :data:`DISTURBANCE_CHANNELS`. See the module docstring.
    """
    n_steps = round(interval_s / step_s)
    if abs(n_steps * step_s - interval_s) > 1e-9:
        raise ValueError("interval_s must be a multiple of step_s")
    x0 = cs.SX.sym("x0", 3)
    u = cs.SX.sym("u", 2)
    u_prev = cs.SX.sym("u_prev", 2)
    rates = (
        config.actions.inlet_temperature_celsius.max_rate_per_s,
        config.actions.gas_velocity_m_per_s.max_rate_per_s,
    )
    shift = 0.5 * sample_time_s
    disturbances = cs.SX.sym("D", len(DISTURBANCE_CHANNELS), n_steps)

    xs = cs.SX.sym("xs", 3)
    us = cs.SX.sym("us", 4)
    ds = cs.SX.sym("ds", 3)
    derivative, power = symbolic_derivative(config, xs, us, ds)
    rhs = cs.Function("rhs", [xs, us, ds], [derivative, power])

    x = x0
    moisture_nodes = []
    temperature_nodes = []
    energy = 0.0
    for j in range(n_steps):
        column = disturbances[:, j]
        start, end = j * step_s + shift, (j + 1) * step_s + shift
        inlet = u_prev[0] + ramp_mean(u[0] - u_prev[0], rates[0], start, end)
        gas = u_prev[1] + ramp_mean(u[1] - u_prev[1], rates[1], start, end)
        action = cs.vertcat(inlet + column[4], gas, column[3], nozzle_pressure_bar)
        weather = column[0:3]
        k1, p1 = rhs(x, action, weather)
        k2, _ = rhs(x + 0.5 * step_s * k1, action, weather)
        k3, _ = rhs(x + 0.5 * step_s * k2, action, weather)
        k4, _ = rhs(x + step_s * k3, action, weather)
        x = x + step_s / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        energy = energy + p1 * step_s / _SECONDS_PER_HOUR
        moisture_nodes.append(x[1] / x[0])
        temperature_nodes.append(x[2])
    return cs.Function(
        "interval",
        [x0, u, u_prev, disturbances],
        [x, cs.horzcat(*moisture_nodes), cs.horzcat(*temperature_nodes), energy],
        ["x0", "u", "u_prev", "D"],
        ["x_end", "moisture", "bed_temperature", "energy_kwh"],
    )


def step_means(samples: np.ndarray, samples_per_step: int) -> np.ndarray:
    """Mean over consecutive blocks of truth samples, ``(channels, n) -> (channels, n / k)``."""
    channels, n = samples.shape
    if n % samples_per_step:
        raise ValueError("sample count is not a multiple of samples_per_step")
    return samples.reshape(channels, n // samples_per_step, samples_per_step).mean(axis=2)


def steady_particle_count(config: ProcessConfig, solid_fraction: float) -> float:
    """Particle count of the steady state (``steady_state.py`` step 2) at the setpoint mass."""
    material = config.material
    spray_solid_share = solid_fraction  # spray rate cancels in the ratio below
    seed = config.operation.seed_mass_flow_kg_per_h
    if seed != 0.0:
        raise ValueError("steady particle count implemented for zero seed flow only")
    ratio = spray_solid_share / (material.nucleation_solid_fraction * spray_solid_share)
    diameter = material.nucleus_diameter_m * ratio ** (1.0 / 3.0)
    particle_mass = material.particle_density_kg_per_m3 * math.pi * diameter**3 / 6.0
    return config.geometry.bed_mass_solid_setpoint_kg / particle_mass

r"""Sensitivity of the gas-side measurements to the bed moisture, analytic and simulated.

Steady state of the grey-box model (continuous operation, :mod:`twinlink_ml.simulation.greybox`):
the water balance :math:`0 = \dot m_{spray}(1 - w_{solid}) - \dot m_{evap} - \dot m_{s,in} X` and
:math:`\dot m_{evap} = \dot m_L (Y_{out} - Y_{in})` give exactly

.. math:: Y_{out} = Y_{in} + \frac{\dot m_{spray}(1 - w_{solid}) - \dot m_{s,in} X}{\dot m_L}.

Two truths with identical commands and disturbances (same water feed, same inlet humidity) but
different drying kinetics or isotherm therefore differ in the exhaust humidity by

.. math:: \Delta Y_{out} = -\Delta X\,\frac{\dot m_{s,in}}{\dot m_L}

(the expression of the hypothesis, :func:`outlet_humidity_shift`) as long as the dry-air flow
:math:`\dot m_L` is the same. It is not quite: the gas velocity is set at the bed temperature, so
:math:`\dot m_L \propto 1/T_{bed,K}` (ideal gas, :func:`~twinlink_ml.simulation.thermo.
gas_density_kg_per_m3` at constant :math:`Y_{in}`), and a wetter bed is warmer. Linearised
(:func:`outlet_humidity_shift_linearised`):

.. math:: \Delta Y_{out} \approx -\Delta X\,\frac{\dot m_{s,in}}{\dot m_L}
    + (Y_{out} - Y_{in})\,\frac{\Delta T_{bed}}{T_{bed,K}}.

The energy balance at the bed temperature with :math:`\dot m_{evap} = W - \dot m_{s,in} X`
(:math:`W` the water feed), linearised in :math:`X` with every temperature dependence kept
(:func:`bed_temperature_shift`):

.. math:: \Delta T_{bed} \approx \frac{\dot m_{s,in}\,\Delta h_v(T_{bed})\,\Delta X}
    {\dot m_L c_L \bigl(1 + (T_{in} - T_{bed})/T_{bed,K}\bigr) + \dot m_{spray} c_{p,susp} + UA
    + \dot m_{evap}\,\mathrm d\Delta h_v/\mathrm dT}

with :math:`c_L = c_{p,L} + Y_{in} c_{p,D}` and :math:`\mathrm d\Delta h_v/\mathrm dT = -2358.6`
J/(kg K) (:func:`~twinlink_ml.simulation.thermo.latent_heat_j_per_kg`). The exhaust temperature is
``T_bed - exhaust.outlet_duct_heat_loss_k`` (well-mixed bed; the loss is 0.0 in the configs of this
study), so it carries the same sensitivity with its own sensor.

Sensor sigma: the existing sensor model (:meth:`GreyBoxFluidBedGranulator.measure`): Gaussian noise
with standard deviation ``relative_noise_std * |value|`` on every channel, plus
``absolute_noise_std_temperature_k`` on temperature channels (independent), after a PT1 with
``sensor_time_constant_s`` that does not change the noise level. Signal-to-noise ratio of a single
sample, for a moisture difference :math:`\Delta X`:

.. math:: \mathrm{SNR} = |\partial m / \partial X|\,\Delta X / (\sigma / \sqrt{n})

with :math:`n` redundant sensors of independent noise (``n = 1`` unless stated).
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    MeasurementConfig,
    PlaceholderConfigWarning,
    ProcessConfig,
)
from twinlink_ml.simulation.backend import NumpyBackend
from twinlink_ml.simulation.operating_point import build_variant_config
from twinlink_ml.simulation.steady_state import solve_steady_state
from twinlink_ml.simulation.thermo import latent_heat_j_per_kg

from .config import StepSettings

__all__ = [
    "LinearFit",
    "Separation",
    "SteadyMeasurements",
    "bed_temperature_shift",
    "linear_fit",
    "operating_point_config",
    "outlet_humidity_shift",
    "outlet_humidity_shift_linearised",
    "scale_solid_to_air_ratio",
    "sensor_sigma",
    "separation",
    "snr",
    "steady_measurements",
    "step_responses",
]

_SECONDS_PER_HOUR = 3600.0
_KELVIN = 273.15
_LATENT_SLOPE_J_PER_KG_K = -2358.6  # d(Delta h_v)/dT of thermo.latent_heat_j_per_kg


def sensor_sigma(
    value: float | np.ndarray, measurement: MeasurementConfig, *, temperature: bool
) -> np.ndarray:
    """Standard deviation of one sample of the existing sensor model at the given true value."""
    relative = measurement.relative_noise_std * np.abs(np.asarray(value, dtype=np.float64))
    if not temperature:
        return relative
    return np.sqrt(relative**2 + measurement.absolute_noise_std_temperature_k**2)


def snr(sensitivity: float, delta_x: float, sigma: float, redundancy: int = 1) -> float:
    """``|sensitivity| * delta_x / (sigma / sqrt(redundancy))``."""
    if sigma <= 0.0 or redundancy < 1:
        raise ValueError("sigma must be positive and redundancy at least 1")
    return abs(sensitivity) * delta_x / (sigma / math.sqrt(redundancy))


@dataclass(frozen=True, slots=True)
class LinearFit:
    """Ordinary least squares ``y = intercept + slope * x``."""

    slope: float
    intercept: float
    r2: float
    n: int

    def as_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


def linear_fit(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> LinearFit:
    """OLS fit with the coefficient of determination (NaN if ``y`` is constant)."""
    xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if xa.shape != ya.shape or xa.size < 2 or np.ptp(xa) == 0.0:
        raise ValueError("need at least two distinct x values of the same shape as y")
    slope, intercept = np.polyfit(xa, ya, 1)
    residual = ya - (intercept + slope * xa)
    total = float(np.sum((ya - ya.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / total if total > 0.0 else math.nan
    return LinearFit(float(slope), float(intercept), r2, int(xa.size))


def outlet_humidity_shift(delta_x: float, solid_flow: float, dry_air_flow: float) -> float:
    """Analytic ``Delta Y_out = -Delta X * m_solid / m_air`` (flows in the same unit)."""
    return -delta_x * solid_flow / dry_air_flow


def outlet_humidity_shift_linearised(
    delta_x: float, delta_bed_temperature: float, reference: SteadyMeasurements
) -> float:
    """The hypothesis' shift plus the dry-air-flow change of the warmer bed (module docstring)."""
    pickup = reference.outlet_humidity_balance_kg_per_kg - reference.inlet_humidity_kg_per_kg
    kelvin = reference.bed_temperature_celsius + _KELVIN
    return (
        outlet_humidity_shift(
            delta_x, reference.solid_inflow_kg_per_s, reference.dry_air_flow_kg_per_s
        )
        + pickup * delta_bed_temperature / kelvin
    )


def bed_temperature_shift(
    delta_x: float, reference: SteadyMeasurements, config: ProcessConfig
) -> float:
    """Linearised bed-temperature shift at the steady state ``reference`` (module docstring)."""
    t_bed = reference.bed_temperature_celsius
    latent = float(latent_heat_j_per_kg(NumpyBackend(), np.asarray(t_bed)))
    c_air = config.gas.cp_dry_air_j_per_kg_k + (
        reference.inlet_humidity_kg_per_kg * config.gas.cp_vapour_j_per_kg_k
    )
    t_in = config.actions.inlet_temperature_celsius.nominal
    capacity = (
        reference.dry_air_flow_kg_per_s * c_air * (1.0 + (t_in - t_bed) / (t_bed + _KELVIN))
        + reference.spray_flow_kg_per_s * config.material.cp_suspension_j_per_kg_k
        + config.geometry.heat_loss_ua_w_per_k
        + reference.evaporation_kg_per_s * _LATENT_SLOPE_J_PER_KG_K
    )
    return delta_x * reference.solid_inflow_kg_per_s * latent / capacity


def _widened(bound: Any, nominal: float) -> Any:
    return dataclasses.replace(
        bound,
        nominal=nominal,
        minimum=min(bound.minimum, nominal),
        maximum=max(bound.maximum, nominal),
    )


def operating_point_config(
    base: ProcessConfig,
    *,
    inlet_temperature_celsius: float,
    gas_velocity_m_per_s: float,
    spray_rate_kg_per_h: float | None = None,
    solid_fraction: float | None = None,
) -> ProcessConfig:
    """``base`` with the nominal commands of one operating point.

    Nozzle pressure and ambient stay the nominal values of ``base``
    (:func:`~twinlink_ml.simulation.operating_point.build_variant_config`). Spray rate and solid
    fraction, if given, replace their nominal values; the action/disturbance box is widened to
    contain them where needed (sampling bounds only, no physics; the steady-state solver and
    :meth:`step` with ``enforce_action_bounds`` off do not read them).
    """
    config = build_variant_config(
        base,
        t_in=inlet_temperature_celsius,
        u_gas=gas_velocity_m_per_s,
        p_nozzle=base.actions.nozzle_pressure_bar.nominal,
        t_amb=base.disturbances.ambient_temperature_celsius.nominal,
        phi_amb=base.disturbances.ambient_relative_humidity.nominal,
    )
    if spray_rate_kg_per_h is not None:
        config = dataclasses.replace(
            config,
            actions=dataclasses.replace(
                config.actions,
                spray_rate_kg_per_h=_widened(
                    config.actions.spray_rate_kg_per_h, spray_rate_kg_per_h
                ),
            ),
        )
    if solid_fraction is not None:
        config = dataclasses.replace(
            config,
            disturbances=dataclasses.replace(
                config.disturbances,
                suspension_solid_fraction=_widened(
                    config.disturbances.suspension_solid_fraction, solid_fraction
                ),
            ),
        )
    return config


@dataclass(frozen=True)
class SteadyMeasurements:
    """Steady state of one truth at one operating point: moisture and the measured responses.

    Two exhaust humidities: ``outlet_humidity_kg_per_kg`` is the model's diagnostic (the damped
    Picard iterate of :func:`~twinlink_ml.simulation.thermo.evaporation_rate_kg_per_s`, the
    quantity measured in Befund 112); ``outlet_humidity_balance_kg_per_kg`` is
    ``Y_in + evaporation / m_air``, the humidity that closes the water balance (the evaporation of
    the last Picard step is the one the balances use). They differ by the Picard residual.
    """

    moisture_kg_per_kg: float
    outlet_humidity_kg_per_kg: float
    outlet_humidity_balance_kg_per_kg: float
    bed_temperature_celsius: float
    exhaust_temperature_celsius: float
    dry_air_flow_kg_per_s: float
    solid_inflow_kg_per_s: float
    spray_flow_kg_per_s: float
    water_feed_kg_per_s: float
    inlet_humidity_kg_per_kg: float
    evaporation_kg_per_s: float
    feasible: bool
    infeasibility_reasons: tuple[str, ...]
    residual_inf_norm: float
    state: np.ndarray = dataclasses.field(repr=False)

    def as_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in dataclasses.asdict(self).items() if k != "state"}
        out["infeasibility_reasons"] = list(self.infeasibility_reasons)
        out["state"] = [float(s) for s in self.state]
        return out


def steady_measurements(config: ProcessConfig) -> SteadyMeasurements:
    """Steady state at the nominal commands and disturbances of ``config`` (exact solver)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        simulator = GreyBoxFluidBedGranulator(config)
        solution = solve_steady_state(simulator)
    diag = solution.diagnostics
    inlet_humidity = float(diag["inlet_humidity_kg_per_kg"])
    evaporation = float(diag["evaporation_rate_kg_per_s"])
    spray = config.actions.spray_rate_kg_per_h.nominal / _SECONDS_PER_HOUR
    solid_fraction = config.disturbances.suspension_solid_fraction.nominal
    seed = config.operation.seed_mass_flow_kg_per_h / _SECONDS_PER_HOUR
    return SteadyMeasurements(
        moisture_kg_per_kg=solution.particle_moisture_kg_per_kg,
        outlet_humidity_kg_per_kg=solution.outlet_humidity_kg_per_kg,
        outlet_humidity_balance_kg_per_kg=inlet_humidity
        + evaporation / solution.dry_air_mass_flow_kg_per_s,
        bed_temperature_celsius=solution.bed_temperature_celsius,
        exhaust_temperature_celsius=solution.bed_temperature_celsius
        - config.exhaust.outlet_duct_heat_loss_k,
        dry_air_flow_kg_per_s=solution.dry_air_mass_flow_kg_per_s,
        solid_inflow_kg_per_s=spray * solid_fraction + seed,
        spray_flow_kg_per_s=spray,
        water_feed_kg_per_s=spray * (1.0 - solid_fraction),
        inlet_humidity_kg_per_kg=inlet_humidity,
        evaporation_kg_per_s=evaporation,
        feasible=bool(solution.feasible),
        infeasibility_reasons=tuple(solution.infeasibility_reasons),
        residual_inf_norm=float(solution.residual_inf_norm),
        state=np.asarray(solution.state, dtype=np.float64).reshape(-1),
    )


def scale_solid_to_air_ratio(
    config: ProcessConfig, ratio: float, iterations: int
) -> tuple[ProcessConfig, dict[str, float]]:
    """Commands with dry solid feed = ``ratio`` * dry-air flow at unchanged water feed and air.

    Fixed-point iteration on ``config`` (the nominal model): the dry-air flow of the steady state
    (it depends weakly on the bed temperature) sets the solid feed; spray rate and solid fraction
    are raised together so that the water feed ``spray * (1 - w)`` stays that of ``config``. Gas
    velocity, inlet temperature, ambient and every model parameter stay unchanged. Returns the
    new config and the achieved ratio on it.
    """
    spray0 = config.actions.spray_rate_kg_per_h.nominal
    water = spray0 * (1.0 - config.disturbances.suspension_solid_fraction.nominal)  # kg/h
    if config.operation.seed_mass_flow_kg_per_h != 0.0:
        raise ValueError("ratio scaling assumes no seed feed")
    current = config
    for _ in range(iterations):
        air_kg_per_h = steady_measurements(current).dry_air_flow_kg_per_s * _SECONDS_PER_HOUR
        solid = ratio * air_kg_per_h
        spray = water + solid
        current = operating_point_config(
            config,
            inlet_temperature_celsius=config.actions.inlet_temperature_celsius.nominal,
            gas_velocity_m_per_s=config.actions.gas_velocity_m_per_s.nominal,
            spray_rate_kg_per_h=spray,
            solid_fraction=solid / spray,
        )
    final = steady_measurements(current)
    achieved = final.solid_inflow_kg_per_s / final.dry_air_flow_kg_per_s
    return current, {
        "target_ratio": ratio,
        "achieved_ratio": achieved,
        "spray_rate_kg_per_h": current.actions.spray_rate_kg_per_h.nominal,
        "solid_fraction": current.disturbances.suspension_solid_fraction.nominal,
        "water_feed_kg_per_h": final.water_feed_kg_per_s * _SECONDS_PER_HOUR,
        "dry_air_flow_kg_per_h": final.dry_air_flow_kg_per_s * _SECONDS_PER_HOUR,
    }


def step_responses(
    config: ProcessConfig,
    initial_state: np.ndarray,
    steps: Sequence[StepSettings],
    horizon_s: float,
) -> dict[str, np.ndarray]:
    """Responses to steps of the nominal commands of ``config``, from ``initial_state``.

    Batch row 0 is the unstepped reference, row ``i + 1`` the step ``steps[i]`` applied at
    ``t = 0`` (commanded value = factor * nominal, realised through the rate limit of the action
    box, as :mod:`twinlink_ml.headroom.truth`). Disturbances constant at the nominal values.
    Returns arrays ``(1 + len(steps), samples)`` of the true moisture, both exhaust humidities
    (see :class:`SteadyMeasurements`), bed and exhaust temperature at the end of every sample, and
    the time axis.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        simulator = GreyBoxFluidBedGranulator(config)
    dt = config.numerics.sample_time_s
    n = round(horizon_s / dt)
    batch = 1 + len(steps)
    base = simulator.nominal_action(1)[0]
    command = np.tile(base, (batch, 1))
    for i, step in enumerate(steps):
        column = 0 if step.channel == "inlet_temperature" else 2
        command[i + 1, column] = step.factor * base[column]
    rates = np.array([b.max_rate_per_s for b in config.actions.bounds]) * dt
    disturbance = simulator.nominal_disturbance(batch)
    state = np.tile(np.asarray(initial_state, dtype=np.float64), (batch, 1))
    realised = np.tile(base, (batch, 1))
    moisture = np.empty((batch, n))
    humidity = np.empty((batch, n))
    balance = np.empty((batch, n))
    bed = np.empty((batch, n))
    for k in range(n):
        realised = realised + np.clip(command - realised, -rates, rates)
        result = simulator.step(state, realised, disturbance)
        state = np.asarray(result.state)
        moisture[:, k] = result.diagnostics["particle_moisture_kg_per_kg"]
        diag = result.diagnostics
        humidity[:, k] = diag["outlet_humidity_kg_per_kg"]
        balance[:, k] = (
            diag["inlet_humidity_kg_per_kg"]
            + diag["evaporation_rate_kg_per_s"] / diag["dry_air_mass_flow_kg_per_s"]
        )
        bed[:, k] = state[:, 3]
    return {
        "time_s": dt * np.arange(1, n + 1),
        "moisture_kg_per_kg": moisture,
        "outlet_humidity_kg_per_kg": humidity,
        "outlet_humidity_balance_kg_per_kg": balance,
        "bed_temperature_celsius": bed,
        "exhaust_temperature_celsius": bed - config.exhaust.outlet_duct_heat_loss_k,
    }


@dataclass(frozen=True, slots=True)
class Separation:
    """Distance of two trajectories against the sensor sigma."""

    max_distance: float
    max_ratio: float
    time_of_max_ratio_s: float
    time_above_s: float
    first_above_s: float | None
    last_above_s: float | None

    def as_dict(self) -> dict[str, float | None]:
        return dataclasses.asdict(self)


def separation(
    a: np.ndarray,
    b: np.ndarray,
    sigma: np.ndarray,
    threshold: float,
    sample_time_s: float,
    time_s: np.ndarray,
) -> Separation:
    """``|a - b| / sigma`` on a common grid: maximum and total time above ``threshold``."""
    distance = np.abs(np.asarray(a) - np.asarray(b))
    ratio = distance / np.asarray(sigma)
    above = ratio > threshold
    k = int(np.argmax(ratio))
    hits = np.flatnonzero(above)
    return Separation(
        max_distance=float(distance.max()),
        max_ratio=float(ratio[k]),
        time_of_max_ratio_s=float(time_s[k]),
        time_above_s=float(above.sum() * sample_time_s),
        first_above_s=float(time_s[hits[0]]) if hits.size else None,
        last_above_s=float(time_s[hits[-1]]) if hits.size else None,
    )

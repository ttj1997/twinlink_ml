r"""The grey-box model at plant scale: operating points, the bed-to-exhaust switch, calibration.

The grey-box model (:mod:`twinlink_ml.simulation.greybox`) is only PARAMETRISED here, never
changed: :func:`plant_config` sets the nominal commands and disturbances of a measured operating
point on the plant-scale process config, the heat-loss UA of the bed balance and, for the
``offset`` mode, the existing ``exhaust.outlet_duct_heat_loss_k``.

**Gas flow.** The model's gas velocity is defined at the bed temperature (dry-air flow
``rho(T_bed) u A / (1 + Y_in)``). :func:`steady_plant` iterates ``u <- u * m_target / m`` until the
steady dry-air flow equals the measured process-air flow (normal cubic metres of dry air; the supply
air is dried, so its vapour is neglected in the conversion). The atomising air is not part of the
model (lab line: no such stream); its size is quantified in :mod:`.balance`.

**Switch of the bed-to-exhaust term** (:class:`Freeboard`, default ``none``). In the unchanged model
the exhaust leaves at the bed temperature (well-mixed bed). The switch adds ONE of these terms
between the bed and the exhaust sensor, all outside the grey-box code:

* ``none``: ``T_exh = T_bed`` (the model as it is; bit-identical, tested);
* ``offset``: ``T_exh = T_bed - dT`` with the existing ``exhaust.outlet_duct_heat_loss_k = dT``;
* ``heat_loss``: heat loss of the gas to the surroundings at the ambient temperature of the model,
  ``T_exh = T_amb + (T_bed - T_amb) exp(-UA_fb / (m_L c_h))`` (duct with uniform conductance),
  ``c_h = c_{p,a} + Y_{out} c_{p,v}``; the bed balance is unaffected;
* ``evaporation``: a share ``f`` of the spray water evaporates above the bed (spray zone /
  freeboard) instead of in the bed. Emulated by the inputs: the bed receives the spray rate
  ``s - f W`` with the solid fraction ``x_s s / (s - f W)`` (``W = s (1 - x_s)``), so its solid feed
  is unchanged; the gas leaving the bed then evaporates ``f W`` (liquid at the feed temperature,
  vapour at the exhaust temperature, enthalpies of :mod:`twinlink_ml.plant_a_op.dryer`):
  ``m_L c_h (T_bed - T_exh) = f W (r_0 + c_v T_exh - c_w T_feed)``, linear in ``T_exh``. The exhaust
  humidity adds ``f W / m_L``.

**Calibration** (:func:`calibrate`): the heat-loss UA of the bed balance (W/K, >= 0) and, for a
switched-on candidate, its one parameter, fitted by bounded least squares to the bed temperature
(mean of the six sensors) and the exhaust temperature of every stationary segment of one day. The
kinetics are not calibrated (no moisture measurement).
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from ..observability.core import SteadyMeasurements, steady_measurements
from ..simulation import PlaceholderConfigWarning, ProcessConfig
from ..simulation.backend import NumpyBackend
from ..simulation.thermo import gas_density_kg_per_m3, latent_heat_j_per_kg, saturation_pressure_pa
from .config import FREEBOARD_MODES

__all__ = [
    "CalibrationFit",
    "Freeboard",
    "PlantParameters",
    "PlantPoint",
    "PlantSteady",
    "calibrate",
    "dry_air_target_kg_per_s",
    "exhaust_after_freeboard",
    "plant_config",
    "predict",
    "relative_humidity_for",
    "steady_plant",
]

_BACKEND = NumpyBackend()
_SECONDS_PER_HOUR = 3600.0
_R0_J_PER_KG = float(latent_heat_j_per_kg(_BACKEND, np.float64(0.0)))


@dataclass(frozen=True, slots=True)
class PlantPoint:
    """A measured stationary operating point (segment means); temperatures degC."""

    label: str
    ambient_celsius: float
    process_air_nm3h: float
    atomising_air_nm3h: float
    supply_celsius: float
    spray_kg_h: float
    bed_celsius: float
    exhaust_celsius: float
    bed_group_a_celsius: float = math.nan
    bed_group_b_celsius: float = math.nan
    bed_sensor_min_celsius: float = math.nan
    bed_sensor_max_celsius: float = math.nan
    downstream_celsius: float = math.nan

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class Freeboard:
    """Switch of the bed-to-exhaust term; ``value``: K (offset), W/K (heat_loss), share."""

    mode: str = "none"
    value: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in FREEBOARD_MODES:
            raise ValueError(f"unknown freeboard mode {self.mode!r}")
        if self.value < 0.0:
            raise ValueError("freeboard value must not be negative")
        if self.mode == "evaporation" and self.value >= 1.0:
            raise ValueError("evaporation share must be below 1")
        if self.mode == "none" and self.value != 0.0:
            raise ValueError("mode 'none' takes no value")


@dataclass(frozen=True, slots=True)
class PlantParameters:
    """Plant-near parameters of one model evaluation."""

    heat_loss_ua_w_per_k: float
    solid_fraction: float
    freeboard: Freeboard = Freeboard()

    def as_dict(self) -> dict[str, Any]:
        return {
            "heat_loss_ua_w_per_k": self.heat_loss_ua_w_per_k,
            "solid_fraction": self.solid_fraction,
            "freeboard_mode": self.freeboard.mode,
            "freeboard_value": self.freeboard.value,
        }


def dry_air_target_kg_per_s(point: PlantPoint, normal_density_kg_per_m3: float) -> float:
    """Measured process air as dry-air mass flow (normal cubic metres are a mass unit)."""
    return point.process_air_nm3h * normal_density_kg_per_m3 / _SECONDS_PER_HOUR


def relative_humidity_for(
    config: ProcessConfig, temperature_celsius: float, humidity: float
) -> float:
    """Relative humidity at ``temperature_celsius`` that gives the absolute ``humidity``."""
    p_vapour = humidity * config.gas.pressure_pa / (0.622 + humidity)
    p_sat = float(saturation_pressure_pa(_BACKEND, np.float64(temperature_celsius)))
    return p_vapour / p_sat


def _bound(bound: Any, nominal: float) -> Any:
    return dataclasses.replace(
        bound,
        nominal=nominal,
        minimum=min(bound.minimum, nominal),
        maximum=max(bound.maximum, nominal),
    )


def bed_feed(point: PlantPoint, params: PlantParameters) -> tuple[float, float, float]:
    """Spray rate and solid fraction seen by the bed, and the freeboard water (kg/h each)."""
    spray, x_s = point.spray_kg_h, params.solid_fraction
    if params.freeboard.mode != "evaporation":
        return spray, x_s, 0.0
    freeboard_water = params.freeboard.value * spray * (1.0 - x_s)
    bed_spray = spray - freeboard_water
    return bed_spray, x_s * spray / bed_spray, freeboard_water


def plant_config(
    base: ProcessConfig,
    point: PlantPoint,
    params: PlantParameters,
    *,
    gas_velocity_m_per_s: float,
    inlet_humidity_kg_per_kg: float,
    inlet_celsius: float | None = None,
    material_values: Mapping[str, float] | None = None,
) -> ProcessConfig:
    """``base`` with the nominal commands/disturbances of ``point`` and the parameters.

    Box limits are widened to contain the nominal values where needed (sampling bounds only, the
    steady-state solver does not read them). ``inlet_celsius`` overrides the measured supply
    temperature (headroom search); ``material_values`` replaces material parameters (draws).
    """
    spray, solid_fraction, _ = bed_feed(point, params)
    a, d = base.actions, base.disturbances
    t_in = point.supply_celsius if inlet_celsius is None else inlet_celsius
    rh = relative_humidity_for(base, point.ambient_celsius, inlet_humidity_kg_per_kg)
    material = base.material
    if material_values:
        material = dataclasses.replace(material, **dict(material_values))
    return dataclasses.replace(
        base,
        geometry=dataclasses.replace(
            base.geometry, heat_loss_ua_w_per_k=params.heat_loss_ua_w_per_k
        ),
        exhaust=dataclasses.replace(
            base.exhaust,
            outlet_duct_heat_loss_k=params.freeboard.value
            if params.freeboard.mode == "offset"
            else 0.0,
        ),
        material=material,
        actions=dataclasses.replace(
            a,
            inlet_temperature_celsius=_bound(a.inlet_temperature_celsius, t_in),
            gas_velocity_m_per_s=_bound(a.gas_velocity_m_per_s, gas_velocity_m_per_s),
            spray_rate_kg_per_h=_bound(a.spray_rate_kg_per_h, spray),
        ),
        disturbances=dataclasses.replace(
            d,
            ambient_temperature_celsius=_bound(
                d.ambient_temperature_celsius, point.ambient_celsius
            ),
            ambient_relative_humidity=_bound(d.ambient_relative_humidity, rh),
            suspension_solid_fraction=_bound(d.suspension_solid_fraction, solid_fraction),
        ),
    )


def exhaust_after_freeboard(
    config: ProcessConfig,
    steady: SteadyMeasurements,
    freeboard: Freeboard,
    *,
    ambient_celsius: float,
    freeboard_water_kg_per_s: float,
    feed_celsius: float,
) -> tuple[float, float, float]:
    """Exhaust temperature and the two exhaust humidities after the bed-to-exhaust term.

    Returns ``(T_exh, Y_out, Y_out_balance)``; for ``none`` and ``offset`` the humidities are the
    bed's, for ``offset`` ``T_exh`` is the model's own (``T_bed - outlet_duct_heat_loss_k``).
    """
    gas = config.gas
    m = steady.dry_air_flow_kg_per_s
    t_bed = steady.bed_temperature_celsius
    if freeboard.mode in ("none", "offset"):
        return (
            steady.exhaust_temperature_celsius,
            steady.outlet_humidity_kg_per_kg,
            steady.outlet_humidity_balance_kg_per_kg,
        )
    if freeboard.mode == "heat_loss":
        c_h = (
            gas.cp_dry_air_j_per_kg_k
            + steady.outlet_humidity_balance_kg_per_kg * gas.cp_vapour_j_per_kg_k
        )
        decay = math.exp(-freeboard.value / (m * c_h))
        t_exh = ambient_celsius + (t_bed - ambient_celsius) * decay
        return t_exh, steady.outlet_humidity_kg_per_kg, steady.outlet_humidity_balance_kg_per_kg
    # evaporation above the bed
    w = freeboard_water_kg_per_s
    c_h = (
        gas.cp_dry_air_j_per_kg_k
        + steady.outlet_humidity_balance_kg_per_kg * gas.cp_vapour_j_per_kg_k
    )
    c_w = config.material.cp_water_liquid_j_per_kg_k
    t_exh = (m * c_h * t_bed - w * (_R0_J_PER_KG - c_w * feed_celsius)) / (
        m * c_h + w * gas.cp_vapour_j_per_kg_k
    )
    return (
        t_exh,
        steady.outlet_humidity_kg_per_kg + w / m,
        steady.outlet_humidity_balance_kg_per_kg + w / m,
    )


@dataclass(frozen=True)
class PlantSteady:
    """Steady state of the plant-near model at one operating point."""

    point: str
    gas_velocity_m_per_s: float
    velocity_iterations: int
    bed: SteadyMeasurements
    exhaust_temperature_celsius: float
    outlet_humidity_kg_per_kg: float
    outlet_humidity_balance_kg_per_kg: float
    freeboard_water_kg_per_s: float

    @property
    def bed_temperature_celsius(self) -> float:
        return self.bed.bed_temperature_celsius

    @property
    def moisture_kg_per_kg(self) -> float:
        return self.bed.moisture_kg_per_kg

    def as_dict(self) -> dict[str, Any]:
        bed = self.bed.as_dict()
        bed.pop("state", None)
        return {
            "point": self.point,
            "gas_velocity_m_per_s": self.gas_velocity_m_per_s,
            "velocity_iterations": self.velocity_iterations,
            "bed_temperature_celsius": self.bed_temperature_celsius,
            "exhaust_temperature_celsius": self.exhaust_temperature_celsius,
            "moisture_kg_per_kg": self.moisture_kg_per_kg,
            "outlet_humidity_kg_per_kg": self.outlet_humidity_kg_per_kg,
            "outlet_humidity_balance_kg_per_kg": self.outlet_humidity_balance_kg_per_kg,
            "freeboard_water_kg_per_s": self.freeboard_water_kg_per_s,
            "bed": bed,
        }


def initial_velocity(
    base: ProcessConfig, target_kg_per_s: float, bed_celsius: float, humidity: float
) -> float:
    """Gas velocity that gives ``target_kg_per_s`` dry air at ``bed_celsius``."""
    rho = float(
        gas_density_kg_per_m3(_BACKEND, base.gas, np.float64(bed_celsius), np.float64(humidity))
    )
    return target_kg_per_s * (1.0 + humidity) / (rho * base.geometry.cross_section_area_m2)


def steady_plant(
    base: ProcessConfig,
    point: PlantPoint,
    params: PlantParameters,
    *,
    normal_density_kg_per_m3: float,
    inlet_humidity_kg_per_kg: float,
    feed_celsius: float,
    iterations: int,
    relative_tolerance: float,
    velocity_guess: float | None = None,
    material_values: Mapping[str, float] | None = None,
) -> PlantSteady:
    """Steady state at ``point`` with the gas velocity matched to the measured dry-air flow.

    Exact solver at its default precision (:func:`~twinlink_ml.observability.core.
    steady_measurements`, as docs/33). Raises ``RuntimeError`` if the velocity does not converge.
    """
    target = dry_air_target_kg_per_s(point, normal_density_kg_per_m3)
    u = velocity_guess or initial_velocity(
        base, target, point.bed_celsius, inlet_humidity_kg_per_kg
    )
    steady: SteadyMeasurements | None = None
    config: ProcessConfig | None = None
    used = 0
    for _ in range(iterations):
        used += 1
        config = plant_config(
            base,
            point,
            params,
            gas_velocity_m_per_s=u,
            inlet_humidity_kg_per_kg=inlet_humidity_kg_per_kg,
            material_values=material_values,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            steady = steady_measurements(config)
        ratio = target / steady.dry_air_flow_kg_per_s
        if abs(ratio - 1.0) <= relative_tolerance:
            break
        u *= ratio
    else:
        raise RuntimeError(f"gas velocity did not converge at {point.label} (ratio {ratio})")
    assert steady is not None and config is not None
    _, _, freeboard_water = bed_feed(point, params)
    t_exh, y_out, y_bal = exhaust_after_freeboard(
        config,
        steady,
        params.freeboard,
        ambient_celsius=point.ambient_celsius,
        freeboard_water_kg_per_s=freeboard_water / _SECONDS_PER_HOUR,
        feed_celsius=feed_celsius,
    )
    return PlantSteady(
        point=point.label,
        gas_velocity_m_per_s=u,
        velocity_iterations=used,
        bed=steady,
        exhaust_temperature_celsius=t_exh,
        outlet_humidity_kg_per_kg=y_out,
        outlet_humidity_balance_kg_per_kg=y_bal,
        freeboard_water_kg_per_s=freeboard_water / _SECONDS_PER_HOUR,
    )


@dataclass(frozen=True)
class CalibrationFit:
    """Result of :func:`calibrate` on one day."""

    candidate: str
    mode: str
    parameters: PlantParameters
    residuals_k: tuple[float, ...]  # per segment: bed, exhaust
    cost: float
    status: int
    message: str
    evaluations: int
    at_bound: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "mode": self.mode,
            "parameters": self.parameters.as_dict(),
            "residuals_k": list(self.residuals_k),
            "cost": self.cost,
            "status": self.status,
            "message": self.message,
            "evaluations": self.evaluations,
            "at_bound": list(self.at_bound),
        }


def _parameters(x: np.ndarray, mode: str, solid_fraction: float) -> PlantParameters:
    ua = float(x[0]) * 1000.0
    if mode == "none":
        return PlantParameters(ua, solid_fraction)
    value = float(x[1]) * (1000.0 if mode == "heat_loss" else 1.0)
    return PlantParameters(ua, solid_fraction, Freeboard(mode, value))


def calibrate(
    base: ProcessConfig,
    points: Sequence[PlantPoint],
    candidate: str,
    mode: str,
    *,
    solid_fraction: float,
    bounds: Mapping[str, tuple[float, float]],
    start: Mapping[str, float],
    steady_kwargs: Mapping[str, Any],
    material_values: Mapping[str, float] | None = None,
) -> CalibrationFit:
    """Bounded least squares of bed and exhaust temperature over ``points`` (one day).

    Decision vector in kW/K for the UAs (scaled), K for the offset, share for the evaporation.
    A solver failure at a trial point gives a residual of 100 K per temperature (counted in
    ``message`` only through the result; the fit then moves away from it).
    """
    scale = {"heat_loss_ua_w_per_k": 1e-3, "offset": 1.0, "heat_loss": 1e-3, "evaporation": 1.0}
    names = ["heat_loss_ua_w_per_k"] + ([] if mode == "none" else [mode])
    x0 = np.array([start[n] * scale[n] for n in names])
    lo = np.array([bounds[n][0] * scale[n] for n in names])
    hi = np.array([bounds[n][1] * scale[n] for n in names])
    x0 = np.clip(x0, lo, hi)
    guesses: dict[str, float] = {}

    def residuals(x: np.ndarray) -> np.ndarray:
        params = _parameters(x, mode, solid_fraction)
        out: list[float] = []
        for p in points:
            try:
                s = steady_plant(
                    base,
                    p,
                    params,
                    velocity_guess=guesses.get(p.label),
                    material_values=material_values,
                    **steady_kwargs,
                )
            except (RuntimeError, ValueError):
                out.extend([100.0, 100.0])
                continue
            guesses[p.label] = s.gas_velocity_m_per_s
            out.extend(
                [
                    s.bed_temperature_celsius - p.bed_celsius,
                    s.exhaust_temperature_celsius - p.exhaust_celsius,
                ]
            )
        return np.array(out)

    result = least_squares(
        residuals, x0, bounds=(lo, hi), method="trf", diff_step=1e-3, xtol=1e-8, ftol=1e-10
    )
    at_bound = tuple(
        n
        for n, v, a, b in zip(names, result.x, lo, hi, strict=True)
        if v <= a + 1e-9 or v >= b - 1e-9
    )
    final = residuals(result.x)
    return CalibrationFit(
        candidate=candidate,
        mode=mode,
        parameters=_parameters(result.x, mode, solid_fraction),
        residuals_k=tuple(float(r) for r in final),
        cost=float(0.5 * np.sum(final**2)),
        status=int(result.status),
        message=str(result.message),
        evaluations=int(result.nfev),
        at_bound=at_bound,
    )


def predict(
    base: ProcessConfig,
    points: Sequence[PlantPoint],
    params: PlantParameters,
    *,
    steady_kwargs: Mapping[str, Any],
    material_values: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Model against measurement at every point: errors model - measured (K)."""
    rows: list[dict[str, Any]] = []
    for p in points:
        try:
            s = steady_plant(base, p, params, material_values=material_values, **steady_kwargs)
        except (RuntimeError, ValueError) as error:
            rows.append({"point": p.label, "failure": f"{type(error).__name__}: {error}"})
            continue
        rows.append(
            {
                "point": p.label,
                "error_bed_k": s.bed_temperature_celsius - p.bed_celsius,
                "error_exhaust_k": s.exhaust_temperature_celsius - p.exhaust_celsius,
                "model_gap_k": s.bed_temperature_celsius - s.exhaust_temperature_celsius,
                "measured_gap_k": p.bed_celsius - p.exhaust_celsius,
                "moisture_kg_per_kg": s.moisture_kg_per_kg,
                "feasible": s.bed.feasible,
                "infeasibility_reasons": list(s.bed.infeasibility_reasons),
                "gas_velocity_m_per_s": s.gas_velocity_m_per_s,
            }
        )
    return rows

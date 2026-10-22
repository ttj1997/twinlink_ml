"""P1: the line heater -> dryer -> recovery of :mod:`twinlink_ml.flowsheet.line`, physical mode.

The true model is :class:`~twinlink_ml.flowsheet.line.Line` in
:meth:`~twinlink_ml.flowsheet.line.LineMode.physical` (counter-current exchangers, ``dT_min`` per
exchanger, saturation limit on, unpressurised medium with an upper limit), read only. The one
addition is the heat loss of the dryer bed, ``UA (T_out - T_amb)``: :class:`Line` builds its dryer
with ``UA = 0``, :class:`_LineWithHeatLoss` passes the value of the parameter to the same
:class:`~twinlink_ml.flowsheet.dryer.DryerConfig` field (with ``UA = 0`` it is :class:`Line`
unchanged, tested).

Decisions ``u``: inlet temperature of the dryer (the set point of the heater) [degC], dry-air flow
[kg/h], medium flow of the recovery circuit [kg/h]. Disturbances ``w``: ambient temperature
[degC] (= fresh gas), ambient RELATIVE humidity [-] (converted to the humidity ratio of the fresh
gas with the IAPWS saturation pressure of :mod:`~twinlink_ml.flowsheet.validators`), spray rate
[kg/h], solid fraction of the suspension [kg/kg].

Energy ``E`` [kW, final energy]: chemical power of the corrected fuel flow plus the fans and the
pump of :mod:`~twinlink_ml.flowsheet.auxiliary` (same wiring as
:meth:`~twinlink_ml.flowsheet.flowsheet.Flowsheet.evaluate`: process fan on the fresh-gas volume
over dryer and exchanger, exhaust fan on the exhaust volume behind the recovery exchanger, pump on
the medium flow).

Constraints (``g <= 0``; the values are placeholders of ``conf/surrogate_opt/surrogate_rto.yaml``):

``dew_point_margin``
    ``dT_dew,min - (T_out - T_dew(Y_out))`` at the DRYER EXHAUST (ducts and filter between dryer
    and recovery exchanger); condensation inside the recovery exchanger stays allowed and is
    reported (``condensate_kg_per_h``).
``fluidisation``
    ``V_min - V_bed``: volume flow of the moist gas at bed conditions (exhaust temperature and
    humidity, the bed is ideally mixed) at least the fluidisation minimum.
``product_temperature``
    ``T_out - T_prod,max`` (the product leaves at the exhaust temperature, equation 1.5).
``medium_temperature``
    ``T_medium,hot - T_medium,max``; the physical mode caps the medium at the same limit by
    construction, so this constraint cannot be violated by the truth (kept as a target, as
    requested, and reported as such).

The upper limit of the inlet temperature is the upper bound of the decision box (known exactly, no
surrogate target).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from twinlink_ml.evaluation.carrier_energy import CarrierEnergy
from twinlink_ml.flowsheet.auxiliary import (
    AuxiliaryConfig,
    fan_power_w,
    pump_power_w,
    volume_flow_m3_per_s,
)
from twinlink_ml.flowsheet.dryer import DryerBlock
from twinlink_ml.flowsheet.line import Line, LineInputs, LineMode, LineResult
from twinlink_ml.flowsheet.thermo import dew_point
from twinlink_ml.flowsheet.validators import MOLAR_MASS_RATIO, water_saturation_pressure_pa
from twinlink_ml.partner_v4.inputs import ATM_PA, SiDryer, SiMaterial

from .problem import Box, ConstraintSpec, Evaluation, ParameterSpec, ProblemFamily, SteadyProblem

__all__ = [
    "LineProblem",
    "LineProblemFamily",
    "LineSettings",
    "dryer_heat_loss_check",
    "humidity_ratio",
]

_S_PER_H = 3600.0


def humidity_ratio(
    temperature_celsius: float, relative_humidity: float, pressure_pa: float
) -> float:
    """Humidity ratio [kg/kg dry air] of air at ``T`` and relative humidity (IAPWS-IF97)."""
    p_v = relative_humidity * water_saturation_pressure_pa(temperature_celsius)
    return MOLAR_MASS_RATIO * p_v / (pressure_pa - p_v)


@dataclass(frozen=True)
class LineSettings:
    """Fixed settings of P1 (all placeholders, see the YAML)."""

    decisions: Box
    disturbances: Box
    nominal_disturbance: tuple[float, ...]
    material: SiMaterial
    """Synthetic material (heat capacities, heating value); efficiency set by the parameter."""
    spray_temperature_celsius: float
    medium_cp_j_per_kg_k: float
    medium_max_celsius: float
    dew_point_margin_k: float
    fluidisation_min_m3_per_h: float
    product_max_celsius: float
    pressure_pa: float = ATM_PA
    auxiliary: AuxiliaryConfig = field(default_factory=AuxiliaryConfig)
    tolerances: Mapping[str, float] | None = None


class _LineWithHeatLoss(Line):
    """:class:`Line` whose dryer has the bed heat loss ``UA (T_out - T_amb)`` (module doc)."""

    def __init__(self, mode: LineMode, heat_loss_ua_w_per_k: float) -> None:
        super().__init__(mode)
        self.heat_loss_ua_w_per_k = heat_loss_ua_w_per_k

    def dryer_unit(self, inputs: LineInputs) -> DryerBlock:
        base = super().dryer_unit(inputs)
        return DryerBlock(replace(base.config, heat_loss_ua_w_per_k=self.heat_loss_ua_w_per_k))


class LineProblem(SteadyProblem):
    """P1 at fixed parameter values (heater efficiency, bed heat loss, exchanger approach)."""

    def __init__(
        self,
        settings: LineSettings,
        *,
        heater_efficiency: float,
        heat_loss_ua_w_per_k: float,
        dt_min_k: float,
        name: str = "P1_line",
    ) -> None:
        super().__init__()
        self.settings = settings
        self.name = name
        self.decisions = settings.decisions
        self.disturbances = settings.disturbances
        tol = dict(settings.tolerances or {})
        self.constraints = (
            ConstraintSpec("dew_point_margin", "K", 10.0, tol.get("dew_point_margin", 0.05),
                           "minimum margin of the dryer exhaust to its dew point"),
            ConstraintSpec("fluidisation", "m3/h", 50.0, tol.get("fluidisation", 0.05),
                           "fluidisation minimum of the bed volume flow"),
            ConstraintSpec("product_temperature", "K", 10.0, tol.get("product_temperature", 0.05),
                           "product (= exhaust) temperature below its maximum"),
            ConstraintSpec("medium_temperature", "K", 10.0, tol.get("medium_temperature", 0.05),
                           "medium leaves the recovery exchanger below its maximum"),
        )  # fmt: skip
        self.material = replace(settings.material, heater_efficiency=heater_efficiency)
        self.mode = LineMode.physical(
            htm_max_celsius=settings.medium_max_celsius, dt_min_k=dt_min_k
        )
        self.line = _LineWithHeatLoss(self.mode, heat_loss_ua_w_per_k)

    def solve(self, u: np.ndarray, w: np.ndarray) -> LineResult:
        """The flowsheet line at ``(u, w)``."""
        s = self.settings
        t_in, air_kg_per_h, medium_kg_per_h = (float(v) for v in u)
        t_amb, rh, spray_kg_per_h, solid = (float(v) for v in w)
        y_in = humidity_ratio(t_amb, rh, s.pressure_pa)
        dryer = SiDryer(
            air_kg_per_h / _S_PER_H, y_in, t_in, spray_kg_per_h / _S_PER_H, solid,
            s.spray_temperature_celsius,
        )  # fmt: skip
        inputs = LineInputs(
            dryer, t_amb, self.material, medium_kg_per_h / _S_PER_H, s.medium_cp_j_per_kg_k,
            s.pressure_pa,
        )  # fmt: skip
        return self.line.solve(inputs)

    def _evaluate(self, u: np.ndarray, w: np.ndarray) -> Evaluation:
        s, aux = self.settings, self.settings.auxiliary
        r = self.solve(u, w)
        ex = r.dryer.exhaust
        t_out, y_out = ex.temperature_celsius, ex.humidity_kg_per_kg
        fresh = r.inputs.fresh_gas
        fans = fan_power_w(
            volume_flow_m3_per_s(fresh, fresh.temperature_celsius, s.pressure_pa),
            aux.dryer_pressure_drop_pa + aux.exchanger_pressure_drop_pa,
            aux.fan_efficiency,
        ) + fan_power_w(
            volume_flow_m3_per_s(ex, r.stage.exhaust_out_celsius, s.pressure_pa),
            aux.exchanger_pressure_drop_pa,
            aux.fan_efficiency,
        )
        pump = pump_power_w(
            r.inputs.htm_kg_per_s, aux.loop_pressure_drop_pa, aux.htm_density_kg_per_m3,
            aux.pump_efficiency,
        )  # fmt: skip
        carriers = CarrierEnergy(
            electric_kw=(fans + pump) / 1e3,
            gas_kw=r.burner.fuel_corrected_kg_per_s
            * self.material.lower_heating_value_j_per_kg
            / 1e3,
        )
        dew = dew_point(y_out)
        bed_volume_m3_per_h = volume_flow_m3_per_s(ex, t_out, s.pressure_pa) * _S_PER_H
        g = np.array(
            [
                s.dew_point_margin_k - (t_out - dew),
                s.fluidisation_min_m3_per_h - bed_volume_m3_per_h,
                t_out - s.product_max_celsius,
                r.medium_hot_celsius - s.medium_max_celsius,
            ]
        )
        side = {
            "co2_kg_per_h": carriers.co2_kg_per_h,
            "gas_kw": carriers.gas_kw,
            "electric_kw": carriers.electric_kw,
            "condensate_kg_per_h": r.stage.condensate_modelled_kg_per_h,
            "recovered_kw": r.recovered_kw,
            "exhaust_celsius": t_out,
            "exhaust_dew_point_celsius": dew,
            "medium_hot_celsius": r.medium_hot_celsius,
            "product_moisture_kg_per_kg": r.dryer.product_moisture_kg_per_kg,
            "saturation_limited": float(r.dryer.saturation_limited),
        }
        return Evaluation(carriers.kwh_sum_kw, g, side)


class LineProblemFamily(ProblemFamily):
    """P1 for values of ``heater_efficiency``, ``heat_loss_ua_w_per_k`` and ``dt_min_k``."""

    def __init__(self, settings: LineSettings, parameters: tuple[ParameterSpec, ...]) -> None:
        names = {p.name for p in parameters}
        if names != {"heater_efficiency", "heat_loss_ua_w_per_k", "dt_min_k"}:
            raise ValueError(f"P1 needs heater_efficiency, heat_loss_ua_w_per_k, dt_min_k: {names}")
        self.settings = settings
        self.parameters = parameters

    @property
    def nominal_disturbance(self) -> np.ndarray:
        return np.asarray(self.settings.nominal_disturbance, dtype=np.float64)

    def build(self, values: Mapping[str, float]) -> LineProblem:
        v = {**self.nominal_values, **values}
        return LineProblem(
            self.settings,
            heater_efficiency=v["heater_efficiency"],
            heat_loss_ua_w_per_k=v["heat_loss_ua_w_per_k"],
            dt_min_k=v["dt_min_k"],
        )


def dryer_heat_loss_check(settings: LineSettings, u: np.ndarray, w: np.ndarray) -> float:
    """Largest absolute difference of the dryer exhaust between ``UA = 0`` and :class:`Line`.

    For the test that :class:`_LineWithHeatLoss` with ``UA = 0`` is :class:`Line` unchanged.
    """
    problem = LineProblem(settings, heater_efficiency=0.8, heat_loss_ua_w_per_k=0.0, dt_min_k=10.0)
    ours = problem.solve(u, w)
    plain = Line(problem.mode).solve(ours.inputs)
    a, b = ours.dryer.exhaust, plain.dryer.exhaust
    return float(
        max(
            abs(a.temperature_celsius - b.temperature_celsius),
            abs(a.humidity_kg_per_kg - b.humidity_kg_per_kg),
            abs(ours.burner.duty_w - plain.burner.duty_w),
        )
    )

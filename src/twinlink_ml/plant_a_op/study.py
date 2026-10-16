r"""Cases of the operating-point study: model assembly, calibration check and one case's search.

A :class:`Case` fixes every placeholder of one model (product, weather, solid fraction,
calibration day, waste-heat variant, fan law, boiler efficiency, electricity factor, ...); the
constraint variant and the minimum air fraction are evaluated on the same model grid
(:func:`run_case`). :func:`build_model` returns ``None`` when the calibration day gives a negative
loss conductance (the placeholders are inadmissible for that day) -- such cases are recorded as
inadmissible, never clipped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from ..storage_a1.config import WaterConfig
from ..storage_a1.model import water_capacity_rate_kw_per_k
from .config import (
    Calibration,
    ConstraintVariant,
    DryerPlaceholders,
    EnergyPlaceholders,
    FanLawName,
    SearchSpace,
    WasteHeatVariant,
    fingerprint_of,
)
from .data import DayState, MeasuredDay, day_state
from .dryer import (
    DryerModel,
    intake_humidity,
    loss_conductance_kw_per_k,
    observed_outlet_extremes,
)
from .fans import PowerLawFan, fan_law, two_point_exponent
from .optimize import coarse_grid, constraints_for, optimise
from .plant import PlantAModel
from .waste_heat import DutyHeld, FixedWaterInlet, WasteHeatModel, fit_preheater_conductance

__all__ = [
    "Case",
    "StudyData",
    "build_model",
    "calibration_check",
    "preheater_conductance_of",
    "run_case",
    "with_dryer",
    "with_energy",
]


@dataclass(frozen=True, slots=True)
class StudyData:
    """The measured days by product label and the pairing product -> other product."""

    days: Mapping[str, MeasuredDay]
    other: Mapping[str, str]

    @property
    def process_fan_exponent(self) -> float:
        """Two-point exponent of the process fan between the two days (the ``data_fit`` law)."""
        a, b = (day_state(self.days[k]) for k in sorted(self.days))
        return two_point_exponent(
            a.process_fan_kw, a.process_air_nm3h, b.process_fan_kw, b.process_air_nm3h
        )

    @property
    def exhaust_fan_exponent(self) -> float:
        """Two-point exponent of the exhaust fan (reported only: outside the physical range)."""
        a, b = (day_state(self.days[k]) for k in sorted(self.days))
        return two_point_exponent(
            a.exhaust_fan_kw, a.process_air_nm3h, b.exhaust_fan_kw, b.process_air_nm3h
        )


@dataclass(frozen=True, slots=True)
class Case:
    """Every placeholder of one model of the study."""

    product: str
    ambient_shift_k: float = 0.0
    dryer: DryerPlaceholders = field(default_factory=DryerPlaceholders)
    energy: EnergyPlaceholders = field(default_factory=EnergyPlaceholders)
    calibration: Calibration = "own_day"
    waste_heat: WasteHeatVariant = "fixed_water_inlet"
    fan_law: FanLawName = "data_fit"

    @property
    def fingerprint(self) -> str:
        return fingerprint_of(self)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dryer"].pop("gas")
        return {
            "product": self.product,
            "ambient_shift_k": self.ambient_shift_k,
            "calibration": self.calibration,
            "waste_heat": self.waste_heat,
            "fan_law": self.fan_law,
            **{f"dryer.{k}": v for k, v in d["dryer"].items()},
            **{f"energy.{k}": v for k, v in d["energy"].items()},
        }


def _air_capacity(ph: DryerPlaceholders, state: DayState) -> float:
    y = float(intake_humidity(state.ambient_celsius, ph))
    c_h = (ph.gas.cp_dry_air_j_per_kg_k + y * ph.gas.cp_vapour_j_per_kg_k) / 1000.0
    return state.process_air_nm3h * ph.normal_density_kg_per_nm3 / 3600.0 * c_h


def preheater_conductance_of(day: MeasuredDay, ph: DryerPlaceholders) -> float:
    """Preheater conductance fitted on the day means of a measured day (weather as measured)."""
    s = day_state(day)
    c_w = float(water_capacity_rate_kw_per_k(s.preheater_loop_flow_m3h, WaterConfig()))
    return fit_preheater_conductance(
        s.preheater_duty_kw,
        c_w,
        _air_capacity(ph, s),
        s.preheater_water_in_celsius,
        s.ambient_celsius,
    )


def build_model(data: StudyData, case: Case) -> tuple[PlantAModel | None, dict[str, Any]]:
    """Assemble the model of a case; ``None`` if the calibration day's ``UA`` is negative."""
    own = data.days[case.product]
    cal_label = case.product if case.calibration == "own_day" else data.other[case.product]
    ua = loss_conductance_kw_per_k(case.dryer, day_state(data.days[cal_label]))
    info: dict[str, Any] = {"calibration_day": cal_label, "ua_dryer_kw_per_k": ua}
    if ua < 0.0:
        info["inadmissible"] = "negative loss conductance on the calibration day"
        return None, info
    state = day_state(own, case.ambient_shift_k)
    ua_ph = preheater_conductance_of(own, case.dryer)
    info["ua_preheater_kw_per_k"] = ua_ph
    waste: WasteHeatModel
    if case.waste_heat == "fixed_water_inlet":
        waste = FixedWaterInlet(
            ua_ph, state.preheater_loop_flow_m3h, state.preheater_water_in_celsius
        )
    else:
        waste = DutyHeld(
            state.preheater_duty_kw,
            ua_ph,
            state.preheater_loop_flow_m3h,
            case.energy.max_preheater_water_celsius,
        )
    law = fan_law(case.fan_law, data.process_fan_exponent)
    info["fan_exponents"] = [law.exponent_process, law.exponent_exhaust]
    model = PlantAModel(
        state=state,
        dryer=DryerModel(case.dryer, ua),
        process_fan=PowerLawFan(state.process_fan_kw, law.exponent_process),
        exhaust_fan=PowerLawFan(state.exhaust_fan_kw, law.exponent_exhaust),
        waste_heat=waste,
        energy=case.energy,
    )
    return model, info


def run_case(
    data: StudyData,
    case: Case,
    variants: Sequence[tuple[ConstraintVariant, float]],
    space: SearchSpace,
) -> list[dict[str, Any]]:
    """One record per (constraint variant, minimum air fraction) of a case."""
    model, info = build_model(data, case)
    head = {"case": case.to_dict(), "fingerprint": case.fingerprint, **info}
    if model is None:
        return [
            {**head, "constraint": v, "air_factor_min": a, "admissible": False} for v, a in variants
        ]
    extremes = observed_outlet_extremes(data.days[case.product], case.dryer)
    base = model.baseline()
    grid = coarse_grid(model, space)
    out = []
    for variant, air_min in variants:
        c = constraints_for(
            variant,
            observed_outlet_min_celsius=extremes["outlet_min_celsius"],
            observed_rh_max=extremes["rh_max"],
            air_factor_min=air_min,
            baseline=base,
        )
        opt = optimise(model, c, space, grid)
        out.append(
            {
                **head,
                "constraint": variant,
                "air_factor_min": air_min,
                "admissible": True,
                "observed": extremes,
                **opt.to_dict(),
            }
        )
    return out


def calibration_check(
    data: StudyData, ph: DryerPlaceholders, calibration_day: str, check_day: str
) -> dict[str, Any]:
    """Fit ``UA`` on one day, predict the outlet state of the other (day mean and blocks).

    Reference for the block RMSE: the within-day standard deviation of the measured outlet
    temperature (what a perfect day-mean prediction would leave).
    """
    ua = loss_conductance_kw_per_k(ph, day_state(data.days[calibration_day]))
    day = data.days[check_day]
    s = day_state(day)
    model = DryerModel(ph, ua)
    t_mean = float(
        model.outlet_celsius(
            s.process_air_nm3h,
            s.supply_celsius,
            atomising_air_nm3h=s.atomising_air_nm3h,
            spray_kg_h=s.spray_kg_h,
            ambient_celsius=s.ambient_celsius,
        )
    )
    t_blocks = model.outlet_celsius(
        day["process_air_flow"],
        day["supply_air_temperature"],
        atomising_air_nm3h=day["atomising_air_flow"],
        spray_kg_h=day["spray_rate"],
        ambient_celsius=day["ambient_temperature"],
    )
    measured = day["exhaust_outlet_temperature"]
    ua_check = loss_conductance_kw_per_k(ph, s)
    ua_ph_cal = preheater_conductance_of(data.days[calibration_day], ph)
    duty_pred = float(
        FixedWaterInlet(ua_ph_cal, s.preheater_loop_flow_m3h, s.preheater_water_in_celsius).duty_kw(
            np.array([_air_capacity(ph, s)]), np.array([s.ambient_celsius])
        )[0]
    )
    return {
        "solid_fraction": ph.solid_fraction,
        "calibration_day": calibration_day,
        "check_day": check_day,
        "ua_calibration_kw_per_k": ua,
        "ua_check_day_kw_per_k": ua_check,
        "outlet_mean_predicted_celsius": t_mean,
        "outlet_mean_measured_celsius": s.outlet_celsius,
        "outlet_mean_error_k": t_mean - s.outlet_celsius,
        "outlet_block_rmse_k": float(np.sqrt(np.mean((t_blocks - measured) ** 2))),
        "outlet_block_measured_std_k": float(np.std(measured)),
        "loss_predicted_kw": ua * (s.outlet_celsius - ph.room_temperature_celsius),
        "loss_needed_kw": ua_check * (s.outlet_celsius - ph.room_temperature_celsius),
        "preheater_ua_calibration_kw_per_k": ua_ph_cal,
        "preheater_ua_check_day_kw_per_k": preheater_conductance_of(day, ph),
        "preheater_duty_predicted_kw": duty_pred,
        "preheater_duty_measured_kw": s.preheater_duty_kw,
    }


def with_dryer(case: Case, **changes: Any) -> Case:
    """Copy of a case with changed dryer placeholders."""
    return replace(case, dryer=replace(case.dryer, **changes))


def with_energy(case: Case, **changes: Any) -> Case:
    """Copy of a case with changed energy placeholders."""
    return replace(case, energy=replace(case.energy, **changes))

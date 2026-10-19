r"""The two measured days of plant A as stationary day states (day means) plus 15-minute blocks.

Roles, not tags: the assignment role -> channel is read from the git-ignored
``conf/plant/plant_local.yaml`` (key ``operating_point_roles``; template in
``conf/plant/plant_local.yaml.example``). Raw data never enter the repository; the export path is
an argument (the study script takes it from the environment variables of the plant-validation
scripts).

Resolution. Within a day the process-air flow and the supply-air temperature move by less than
0.5 % (the plant runs one operating point per product), so the model is stationary and works on
the DAY MEAN of every input; the 15-minute block means (:func:`~twinlink_ml.storage_timeslice.
plant.block_mean`) are kept for the observed extremes of the proxy constraints and for the
block-level check of the calibration. Waste-heat duty of the preheater: water-side
``rho c_p V (T_supply - T_return)`` per minute, then averaged (same booking as the storage model of
Befund 91).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from ..plant_data.loader import load_plant_export
from ..storage_a1.config import WaterConfig
from ..storage_a1.model import water_capacity_rate_kw_per_k
from ..storage_timeslice.plant import block_mean

__all__ = [
    "REQUIRED_ROLES",
    "DayState",
    "MeasuredDay",
    "day_state",
    "load_measured_day",
    "measured_day_from_columns",
]

FloatArray = NDArray[np.float64]

REQUIRED_ROLES: tuple[str, ...] = (
    "ambient_temperature",
    "process_air_flow",
    "supply_air_temperature",
    "exhaust_outlet_temperature",
    "spray_rate",
    "atomising_air_flow",
    "process_fan_power",
    "exhaust_fan_power",
    "preheater_loop_flow",
    "preheater_supply_temperature",
    "preheater_return_temperature",
)
"""Roles of this study; units: temperatures degC, air flows Nm3/h, spray rate kg/h, fan power kW
electric, loop flow m3/h. Redundant sensors of one role are averaged, except
``atomising_air_flow`` (nozzles are summed)."""

_SUMMED_ROLES = ("atomising_air_flow",)


@dataclass(frozen=True, slots=True)
class MeasuredDay:
    """One measured day as 15-minute block means of every role (96 blocks)."""

    label: str
    blocks: dict[str, FloatArray]
    preheater_duty_kw: FloatArray

    def __post_init__(self) -> None:
        missing = [r for r in REQUIRED_ROLES if r not in self.blocks]
        if missing:
            raise ValueError(f"measured day lacks roles {missing}")
        n = {len(v) for v in self.blocks.values()} | {len(self.preheater_duty_kw)}
        if len(n) != 1:
            raise ValueError("block arrays of different length")

    def __getitem__(self, role: str) -> FloatArray:
        return self.blocks[role]


def measured_day_from_columns(
    label: str, columns: dict[str, FloatArray], water: WaterConfig | None = None
) -> MeasuredDay:
    """Build a :class:`MeasuredDay` from one-minute role series (already averaged/summed)."""
    water = water or WaterConfig()
    duty = water_capacity_rate_kw_per_k(columns["preheater_loop_flow"], water) * (
        columns["preheater_supply_temperature"] - columns["preheater_return_temperature"]
    )
    return MeasuredDay(
        label=label,
        blocks={role: block_mean(np.asarray(columns[role])) for role in REQUIRED_ROLES},
        preheater_duty_kw=block_mean(np.asarray(duty, dtype=np.float64)),
    )


def load_measured_day(export_path: str | Path, role_file: str | Path, label: str) -> MeasuredDay:
    """Read an export and the role file, return the day as 15-minute block means."""
    raw: dict[str, Any] = yaml.safe_load(Path(role_file).read_text(encoding="utf-8"))
    roles = {role: tuple(tags) for role, tags in raw["operating_point_roles"].items()}
    missing = [r for r in REQUIRED_ROLES if not roles.get(r)]
    if missing:
        raise ValueError(f"role file lacks roles {missing}")
    series = load_plant_export(export_path)
    columns: dict[str, FloatArray] = {}
    for role in REQUIRED_ROLES:
        absent = [t for t in roles[role] if t not in series.columns]
        if absent:
            raise KeyError(f"channels of role {role!r} not in export: {absent}")
        stack = np.array([np.asarray(series.columns[t], dtype=np.float64) for t in roles[role]])
        columns[role] = stack.sum(axis=0) if role in _SUMMED_ROLES else stack.mean(axis=0)
    return measured_day_from_columns(label, columns)


@dataclass(frozen=True, slots=True)
class DayState:
    """Stationary state of one day (day means) -- the input of every model evaluation.

    ``ambient_shift_k`` is the weather variant: it is added to the measured intake temperature;
    everything else stays as measured (the preheater water inlet included).
    """

    label: str
    ambient_celsius: float
    process_air_nm3h: float
    atomising_air_nm3h: float
    supply_celsius: float
    outlet_celsius: float
    spray_kg_h: float
    process_fan_kw: float
    exhaust_fan_kw: float
    preheater_loop_flow_m3h: float
    preheater_water_in_celsius: float
    preheater_duty_kw: float
    ambient_shift_k: float = 0.0

    def to_dict(self) -> dict[str, float | str]:
        return {name: getattr(self, name) for name in self.__slots__}


def day_state(day: MeasuredDay, ambient_shift_k: float = 0.0) -> DayState:
    """Day means of a measured day (optionally with shifted weather)."""
    m = {role: float(np.mean(day[role])) for role in REQUIRED_ROLES}
    return DayState(
        label=day.label,
        ambient_celsius=m["ambient_temperature"] + ambient_shift_k,
        process_air_nm3h=m["process_air_flow"],
        atomising_air_nm3h=m["atomising_air_flow"],
        supply_celsius=m["supply_air_temperature"],
        outlet_celsius=m["exhaust_outlet_temperature"],
        spray_kg_h=m["spray_rate"],
        process_fan_kw=m["process_fan_power"],
        exhaust_fan_kw=m["exhaust_fan_power"],
        preheater_loop_flow_m3h=m["preheater_loop_flow"],
        preheater_water_in_celsius=m["preheater_supply_temperature"],
        preheater_duty_kw=float(np.mean(day.preheater_duty_kw)),
        ambient_shift_k=ambient_shift_k,
    )

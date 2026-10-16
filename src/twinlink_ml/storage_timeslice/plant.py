r"""The two measured days of plant A as 15-minute block means, and the reconstructed exhaust state.

Roles, not tags: the assignment role -> channel is read from the git-ignored
``conf/plant/plant_local.yaml`` (key ``timeslice_roles``; template in
``conf/plant/plant_local.yaml.example``). Raw data never enter the repository; the export path is
an argument (the study script takes it from the environment variables of the plant-validation
scripts).

The exhaust HUMIDITY is not measured. It is reconstructed from the water balance of the dryer:
``Y_out = Y_in + m_evap / m_dry`` with the evaporated water ``m_evap = spray rate (1 - solid
fraction)`` (all water of the sprayed suspension evaporates, product moisture neglected; solid
fraction a placeholder) and the dry air ``m_dry = (process air + atomising air) * rho_N``
(normal-density convention of ``storage_a1``). The result is capped at saturation at the measured
outlet temperature (an exhaust cannot be over-saturated); the share of capped steps is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from ..plant_data.loader import load_plant_export
from ..simulation.backend import NumpyBackend
from ..simulation.config import GasConfig
from ..simulation.thermo import absolute_humidity_kg_per_kg, saturation_humidity_kg_per_kg

__all__ = [
    "BLOCK_SAMPLES",
    "REQUIRED_ROLES",
    "STEP_H",
    "MeasuredDay",
    "block_mean",
    "exhaust_humidity",
    "intake_humidity",
    "load_measured_day",
]

FloatArray = NDArray[np.float64]

BLOCK_SAMPLES = 15
STEP_H = 0.25
STEPS_PER_DAY = 96
REQUIRED_ROLES: tuple[str, ...] = (
    "ambient_temperature",
    "process_air_flow",
    "supply_air_temperature",
    "exhaust_outlet_temperature",
    "spray_rate",
    "atomising_air_flow",
)
"""Roles of the time-slice study; units: temperatures degC, air flows Nm3/h, spray rate kg/h.
Redundant sensors of one role are averaged, except ``atomising_air_flow`` (nozzles are summed)."""

_SUMMED_ROLES = ("atomising_air_flow",)
_BACKEND = NumpyBackend()


@dataclass(frozen=True, slots=True)
class MeasuredDay:
    """One measured 24 h day, 96 steps of 15 minutes."""

    label: str
    ambient_c: FloatArray
    process_air_nm3h: FloatArray
    supply_air_c: FloatArray
    exhaust_out_c: FloatArray
    spray_kg_h: FloatArray
    atomising_nm3h: FloatArray

    def __post_init__(self) -> None:
        for name in (
            "ambient_c",
            "process_air_nm3h",
            "supply_air_c",
            "exhaust_out_c",
            "spray_kg_h",
            "atomising_nm3h",
        ):
            if getattr(self, name).shape != (STEPS_PER_DAY,):
                raise ValueError(f"{name} must have {STEPS_PER_DAY} steps")
        if np.any(self.supply_air_c <= self.ambient_c):
            raise ValueError("supply air temperature must exceed the intake temperature")


def block_mean(values: FloatArray, block: int = BLOCK_SAMPLES) -> FloatArray:
    """Block means of a one-minute series (the last sample, 24:00, is dropped)."""
    n = (len(values) // block) * block
    if len(values) - n > 1:
        raise ValueError(f"series length {len(values)} does not fit blocks of {block}")
    return np.asarray(values[:n], dtype=np.float64).reshape(-1, block).mean(axis=1)


def load_measured_day(export_path: str | Path, role_file: str | Path, label: str) -> MeasuredDay:
    """Read an export and the role file, return the day as block means."""
    raw: dict[str, Any] = yaml.safe_load(Path(role_file).read_text(encoding="utf-8"))
    roles = {role: tuple(tags) for role, tags in raw["timeslice_roles"].items()}
    missing = [r for r in REQUIRED_ROLES if not roles.get(r)]
    if missing:
        raise ValueError(f"role file lacks roles {missing}")
    series = load_plant_export(export_path)
    arrays: dict[str, FloatArray] = {}
    for role in REQUIRED_ROLES:
        absent = [t for t in roles[role] if t not in series.columns]
        if absent:
            raise KeyError(f"channels of role {role!r} not in export: {absent}")
        stack = np.array([np.asarray(series.columns[t], dtype=np.float64) for t in roles[role]])
        arrays[role] = block_mean(
            stack.sum(axis=0) if role in _SUMMED_ROLES else stack.mean(axis=0)
        )
    return MeasuredDay(
        label=label,
        ambient_c=arrays["ambient_temperature"],
        process_air_nm3h=arrays["process_air_flow"],
        supply_air_c=arrays["supply_air_temperature"],
        exhaust_out_c=arrays["exhaust_outlet_temperature"],
        spray_kg_h=arrays["spray_rate"],
        atomising_nm3h=arrays["atomising_air_flow"],
    )


def intake_humidity(ambient_c: FloatArray, relative_humidity: float, gas: GasConfig) -> FloatArray:
    """Absolute humidity of the intake air at the given relative humidity (placeholder)."""
    t = np.asarray(ambient_c, dtype=np.float64)
    rel = np.full_like(t, relative_humidity)
    return np.asarray(absolute_humidity_kg_per_kg(_BACKEND, gas, t, rel), dtype=np.float64)


def exhaust_humidity(
    day: MeasuredDay,
    *,
    solid_fraction: float,
    relative_humidity: float,
    normal_density_kg_per_nm3: float,
    gas: GasConfig,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """Reconstructed exhaust humidity: ``(Y_out, dry air kg/s, capped mask)`` per step.

    Dry air of the exhaust: process air plus atomising air; intake humidity at the MEASURED
    intake temperature (not shifted).
    """
    y_in = intake_humidity(day.ambient_c, relative_humidity, gas)
    m_dry = (day.process_air_nm3h + day.atomising_nm3h) * normal_density_kg_per_nm3 / 3600.0
    m_evap = day.spray_kg_h / 3600.0 * (1.0 - solid_fraction)
    y_raw = y_in + m_evap / m_dry
    y_sat = np.asarray(
        saturation_humidity_kg_per_kg(_BACKEND, gas, np.asarray(day.exhaust_out_c)),
        dtype=np.float64,
    )
    y_out = np.minimum(y_raw, y_sat)
    return y_out, m_dry, np.asarray(y_raw > y_sat, dtype=np.bool_)

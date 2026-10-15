r"""Measurement ROLES of the storage model and their assignment to export channels.

Code and documentation use only the roles below. The assignment role -> channel name(s) is read
from a YAML file kept out of the repository (``conf/plant/plant_local.yaml``, git-ignored; the
committed template ``conf/plant/plant_local.yaml.example`` carries placeholder tags). Raw data
never enters the repository: the export path is a function argument (the study script takes it
from the environment variables of the plant-validation scripts).

Several channels under one role (redundant sensors of one quantity) are averaged sample by sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from ..plant_data.loader import PlantTimeSeries, load_plant_export

__all__ = [
    "REQUIRED_ROLES",
    "PlantDay",
    "RoleMap",
    "day_from_series",
    "load_day",
    "load_role_map",
]

REQUIRED_ROLES: tuple[str, ...] = (
    "storage_temperature",
    "storage_loop_flow",
    "preheater_loop_flow",
    "preheater_supply_temperature",
    "preheater_return_temperature",
    "supply_air_temperature",
    "ambient_temperature",
    "process_air_flow",
    "steam_valve_position",
    "process_fan_power",
)
"""Roles the storage model and its study need; units: temperatures degC, loop flows m3/h, process
air flow Nm3/h, valve %, fan power kW."""

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RoleMap:
    """Role -> channel names (``used``) plus roles that are exported but not used."""

    used: dict[str, tuple[str, ...]]
    unused: dict[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        missing = [r for r in REQUIRED_ROLES if not self.used.get(r)]
        if missing:
            raise ValueError(f"role map lacks channels for roles: {missing}")


def load_role_map(path: str | Path) -> RoleMap:
    """Read ``plant_local.yaml`` (or its ``.example``)."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    used = {role: tuple(tags) for role, tags in raw["roles"].items()}
    unused = {role: tuple(tags) for role, tags in (raw.get("unused_roles") or {}).items()}
    return RoleMap(used=used, unused=unused)


@dataclass(frozen=True, slots=True)
class PlantDay:
    """One 24 h export on its own time grid, as role arrays (the unit of every split)."""

    label: str
    time_s: FloatArray
    roles: dict[str, FloatArray]

    def __post_init__(self) -> None:
        n = len(self.time_s)
        bad = [r for r, v in self.roles.items() if len(v) != n]
        if bad:
            raise ValueError(f"role arrays of wrong length: {bad}")

    def __getitem__(self, role: str) -> FloatArray:
        return self.roles[role]

    @property
    def step_s(self) -> float:
        return float(np.median(np.diff(self.time_s)))


def day_from_series(series: PlantTimeSeries, role_map: RoleMap, label: str) -> PlantDay:
    """Assemble role arrays from an export (channel averages for multi-channel roles)."""
    t0 = series.timestamps[0]
    time_s = np.array([(t - t0).total_seconds() for t in series.timestamps], dtype=np.float64)
    roles: dict[str, FloatArray] = {}
    for role, tags in role_map.used.items():
        absent = [t for t in tags if t not in series.columns]
        if absent:
            raise KeyError(f"channels of role {role!r} not in export: {absent}")
        roles[role] = np.mean(
            [np.asarray(series.columns[t], dtype=np.float64) for t in tags], axis=0
        )
    return PlantDay(label=label, time_s=time_s, roles=roles)


def load_day(export_path: str | Path, role_map: RoleMap, label: str) -> PlantDay:
    """Read an export file and return it as a :class:`PlantDay`."""
    return day_from_series(load_plant_export(export_path), role_map, label)

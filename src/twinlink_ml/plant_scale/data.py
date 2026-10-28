r"""Plant exports of plant A as role series, stationary segments and their statistics.

Roles, not tags: the assignment role -> channel is read from the git-ignored
``conf/plant/plant_local.yaml`` (key ``plant_scale_roles``; template with generic placeholder tags
in ``conf/plant/plant_local.yaml.example``). Raw data never enter the repository; the export path
is an argument (the driver takes it from the environment variables of the earlier plant studies).

Roles (units: temperatures degC, air flows Nm3/h, spray rate kg/h). Redundant sensors of one role
are averaged, except ``atomising_air_flow`` (nozzles are summed). The six bed-temperature sensors
are given as two groups of three (``bed_temperature_group_a``, ``bed_temperature_group_b``: two
tag families; their mounting positions are not confirmed); ``bed_temperature`` is the mean of
all six. ``downstream_exhaust_temperature`` is
optional and informative only (sensors further downstream; not used by any model).

Stationary segments (:func:`stationary_segments`): a window of ``window_minutes`` one-minute
samples is stationary if every configured rolling standard deviation is at or below its threshold
(``<series>_k``: absolute, ``<series>_relative``: divided by the window mean). Score of a window =
the largest ratio std / threshold; windows are taken greedily by score, non-overlapping and at
least ``minimum_gap_minutes`` apart, up to ``segments_max``. If fewer than ``segments_min``
windows are admissible, the best remaining windows are added and flagged ``admissible=False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from ..plant_data.loader import load_plant_export
from .config import StationaritySettings

__all__ = [
    "OPTIONAL_ROLES",
    "REQUIRED_ROLES",
    "DaySeries",
    "Segment",
    "block_statistics",
    "day_series_from_columns",
    "load_day_series",
    "rolling_std",
    "segment_statistics",
    "stationary_segments",
    "window_ratios",
]

FloatArray = NDArray[np.float64]

REQUIRED_ROLES: tuple[str, ...] = (
    "ambient_temperature",
    "process_air_flow",
    "supply_air_temperature",
    "bed_temperature_group_a",
    "bed_temperature_group_b",
    "exhaust_outlet_temperature",
    "spray_rate",
    "atomising_air_flow",
)
OPTIONAL_ROLES: tuple[str, ...] = ("downstream_exhaust_temperature",)
_SUMMED_ROLES = ("atomising_air_flow",)
_BED_GROUPS = ("bed_temperature_group_a", "bed_temperature_group_b")


@dataclass(frozen=True)
class DaySeries:
    """One measured day: one-minute role series and the individual bed sensors."""

    label: str
    series: dict[str, FloatArray]
    bed_sensors: FloatArray  # (n_sensors, n_minutes): group a first, then group b
    bed_groups: tuple[int, ...]  # number of sensors per group, in the order of _BED_GROUPS

    def __post_init__(self) -> None:
        missing = [r for r in (*REQUIRED_ROLES, "bed_temperature") if r not in self.series]
        if missing:
            raise ValueError(f"day {self.label} lacks series {missing}")
        lengths = {len(v) for v in self.series.values()} | {self.bed_sensors.shape[1]}
        if len(lengths) != 1:
            raise ValueError("series of different length")
        if sum(self.bed_groups) != self.bed_sensors.shape[0]:
            raise ValueError("bed_groups do not add up to the number of bed sensors")

    @property
    def n_minutes(self) -> int:
        return int(self.bed_sensors.shape[1])

    def __getitem__(self, name: str) -> FloatArray:
        return self.series[name]


def day_series_from_columns(
    label: str, columns: dict[str, FloatArray], bed_sensors: dict[str, FloatArray]
) -> DaySeries:
    """Build a :class:`DaySeries` from role series and the bed sensors per group.

    ``columns``: one-minute series per role (already averaged/summed), WITHOUT the bed groups;
    ``bed_sensors``: per bed group a ``(n_sensors, n_minutes)`` array.
    """
    stacks = [np.atleast_2d(np.asarray(bed_sensors[g], dtype=np.float64)) for g in _BED_GROUPS]
    sensors = np.vstack(stacks)
    series = {k: np.asarray(v, dtype=np.float64) for k, v in columns.items()}
    for group, stack in zip(_BED_GROUPS, stacks, strict=True):
        series[group] = stack.mean(axis=0)
    series["bed_temperature"] = sensors.mean(axis=0)
    return DaySeries(
        label=label,
        series=series,
        bed_sensors=sensors,
        bed_groups=tuple(s.shape[0] for s in stacks),
    )


def load_day_series(
    export_path: str | Path, role_file: str | Path, label: str, role_key: str
) -> DaySeries:
    """Read an export and the role file, return the day as one-minute role series."""
    raw: dict[str, Any] = yaml.safe_load(Path(role_file).read_text(encoding="utf-8"))
    if role_key not in raw:
        raise ValueError(f"role file lacks the key {role_key!r}")
    roles = {role: tuple(tags) for role, tags in raw[role_key].items()}
    missing = [r for r in REQUIRED_ROLES if not roles.get(r)]
    if missing:
        raise ValueError(f"role file lacks roles {missing}")
    export = load_plant_export(export_path)
    columns: dict[str, FloatArray] = {}
    groups: dict[str, FloatArray] = {}
    for role in (*REQUIRED_ROLES, *(r for r in OPTIONAL_ROLES if roles.get(r))):
        absent = [t for t in roles[role] if t not in export.columns]
        if absent:
            raise KeyError(f"channels of role {role!r} not in export: {absent}")
        stack = np.array([np.asarray(export.columns[t], dtype=np.float64) for t in roles[role]])
        if role in _BED_GROUPS:
            groups[role] = stack
        else:
            columns[role] = stack.sum(axis=0) if role in _SUMMED_ROLES else stack.mean(axis=0)
    return day_series_from_columns(label, columns, groups)


def rolling_std(values: FloatArray, window: int) -> tuple[FloatArray, FloatArray]:
    """Population standard deviation and mean of every window ``values[i:i + window]``."""
    x = np.asarray(values, dtype=np.float64)
    if window < 2 or window > len(x):
        raise ValueError(f"window {window} does not fit a series of length {len(x)}")
    c1 = np.concatenate(([0.0], np.cumsum(x)))
    c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s1 = c1[window:] - c1[:-window]
    s2 = c2[window:] - c2[:-window]
    mean = s1 / window
    var = np.maximum(s2 / window - mean * mean, 0.0)
    return np.sqrt(var), mean


@dataclass(frozen=True)
class Segment:
    """A chosen window ``[start, stop)`` in minutes of the day."""

    start: int
    stop: int
    score: float
    admissible: bool
    ratios: dict[str, float] = field(default_factory=dict)  # std / threshold per channel

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_minute": self.start,
            "stop_minute": self.stop,
            "score": self.score,
            "admissible": self.admissible,
            "ratios": dict(self.ratios),
        }


def _threshold_channel(key: str) -> tuple[str, bool]:
    if key.endswith("_relative"):
        return key[: -len("_relative")], True
    if key.endswith("_k"):
        return key[: -len("_k")], False
    raise ValueError(f"threshold key {key!r} must end in '_k' or '_relative'")


def window_ratios(day: DaySeries, settings: StationaritySettings) -> dict[str, FloatArray]:
    """Per threshold channel: rolling std / threshold for every window start."""
    out: dict[str, FloatArray] = {}
    for key, threshold in settings.thresholds.items():
        name, relative = _threshold_channel(key)
        std, mean = rolling_std(day[name], settings.window_minutes)
        if relative:
            std = std / np.abs(mean)
        out[name] = std / threshold
    return out


def stationary_segments(day: DaySeries, settings: StationaritySettings) -> list[Segment]:
    """Greedy choice of stationary windows (rule in the module docstring), in time order."""
    ratios = window_ratios(day, settings)
    names = list(ratios)
    table = np.vstack([ratios[n] for n in names])
    score = table.max(axis=0)
    order = np.argsort(score, kind="stable")
    w = settings.window_minutes
    spacing = w + settings.minimum_gap_minutes
    chosen: list[int] = []

    def free(start: int) -> bool:
        return all(abs(start - c) >= spacing for c in chosen)

    for start in order:  # admissible windows first (score <= 1), best score first
        if score[start] > 1.0 or len(chosen) == settings.segments_max:
            break
        if free(int(start)):
            chosen.append(int(start))
    if len(chosen) < settings.segments_min:
        for start in order:
            if len(chosen) == settings.segments_min:
                break
            if free(int(start)):
                chosen.append(int(start))
    segments = [
        Segment(
            start=s,
            stop=s + w,
            score=float(score[s]),
            admissible=bool(score[s] <= 1.0),
            ratios={n: float(ratios[n][s]) for n in names},
        )
        for s in sorted(chosen)
    ]
    return segments


def segment_statistics(day: DaySeries, segment: Segment) -> dict[str, Any]:
    """Mean and standard deviation of every series and every bed sensor over the segment."""
    sl = slice(segment.start, segment.stop)
    stats: dict[str, Any] = {
        name: {"mean": float(np.mean(v[sl])), "std": float(np.std(v[sl]))}
        for name, v in day.series.items()
    }
    sensors = day.bed_sensors[:, sl]
    stats["bed_sensors_mean"] = [float(v) for v in sensors.mean(axis=1)]
    stats["bed_sensors_std"] = [float(v) for v in sensors.std(axis=1)]
    return stats


def block_statistics(values: FloatArray, block_minutes: int) -> dict[str, float]:
    """Relative standard deviation and range of the block means of a one-minute series.

    The last sample (24:00 of a 1441-sample export) is dropped if it does not fill a block.
    """
    x = np.asarray(values, dtype=np.float64)
    n = (len(x) // block_minutes) * block_minutes
    blocks = x[:n].reshape(-1, block_minutes).mean(axis=1)
    mean = float(blocks.mean())
    return {
        "blocks": len(blocks),
        "mean": mean,
        "std": float(blocks.std()),
        "relative_std": float(blocks.std() / abs(mean)),
        "minimum": float(blocks.min()),
        "maximum": float(blocks.max()),
        "half_range": float((blocks.max() - blocks.min()) / 2.0),
    }

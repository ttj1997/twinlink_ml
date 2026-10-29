"""Settings of the plant-near study, loaded from ``conf/plant_scale/plant_scale.yaml``.

Every value is a study setting or a placeholder (see the YAML header); the dataclasses only give
them names and types.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "FREEBOARD_MODES",
    "CalibrationSettings",
    "DataSettings",
    "HeadroomSettings",
    "ObservabilitySettings",
    "PlaceholderSettings",
    "PlantScaleConfig",
    "StationaritySettings",
    "load_plant_scale_config",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "plant_scale" / "plant_scale.yaml"

FREEBOARD_MODES = ("none", "offset", "heat_loss", "evaporation")
"""Switch of the bed-to-exhaust term (:class:`~twinlink_ml.plant_scale.model.Freeboard`)."""


@dataclass(frozen=True, slots=True)
class DataSettings:
    """Where the exports come from and how flows are converted."""

    role_key: str
    days: dict[str, str]
    normal_density_dry_air_kg_per_m3: float
    block_minutes: int


@dataclass(frozen=True, slots=True)
class StationaritySettings:
    """Criterion of a stationary segment (rolling standard deviations over a window)."""

    window_minutes: int
    thresholds: dict[str, float]
    segments_min: int
    segments_max: int
    minimum_gap_minutes: int

    def __post_init__(self) -> None:
        if not 1 <= self.segments_min <= self.segments_max:
            raise ValueError("need 1 <= segments_min <= segments_max")
        if self.window_minutes < 2 or self.minimum_gap_minutes < 0:
            raise ValueError("window must have at least 2 samples and the gap be non-negative")


@dataclass(frozen=True, slots=True)
class PlaceholderSettings:
    """Unmeasured inputs of the plant-near model and of the balances."""

    solid_fraction_reference: float
    solid_fraction_band: tuple[float, ...]
    inlet_absolute_humidity_kg_per_kg: float
    atomising_air_temperature_celsius: float
    room_temperature_celsius: float
    feed_temperature_celsius: float


@dataclass(frozen=True, slots=True)
class CalibrationSettings:
    """Candidates, bounds and the tolerance of the reading."""

    candidates: dict[str, str]
    bounds: dict[str, tuple[float, float]]
    start: dict[str, float]
    tolerance_k: float
    candidate_order: tuple[str, ...]
    velocity_iterations: int
    velocity_relative_tolerance: float

    def __post_init__(self) -> None:
        unknown = sorted(set(self.candidates.values()) - set(FREEBOARD_MODES))
        if unknown:
            raise ValueError(f"unknown freeboard modes {unknown}")
        if set(self.candidate_order) != set(self.candidates):
            raise ValueError("candidate_order must list every candidate once")


@dataclass(frozen=True, slots=True)
class ObservabilitySettings:
    """Reused settings of docs/33 and the lab record."""

    sensor_value_config: Path
    observability_config: Path
    lab_record: Path
    delta_x_kg_per_kg: float
    redundant_bed_sensors: int


@dataclass(frozen=True, slots=True)
class HeadroomSettings:
    """Steady fixed-against-tracked operating point (see the YAML)."""

    inlet_temperature_celsius: tuple[float, float]
    air_factor: tuple[float, float]
    bed_temperature_limit_celsius: float
    plant_limit: str
    lab_limit_kg_per_kg: float
    lab_inlet_temperature_celsius: tuple[float, float]
    lab_gas_velocity_m_per_s: tuple[float, float]
    spray_sigma_factor: float
    solid_fraction_relative_half_band: float
    velocity_grid_points: int
    inlet_tolerance_k: float
    velocity_relative_tolerance: float
    lab_reference_margins_kg_per_kg: tuple[float, float]
    lab_reference_saving: float

    def __post_init__(self) -> None:
        if self.plant_limit != "today":
            raise ValueError("only plant_limit 'today' is defined")
        if self.velocity_grid_points < 3:
            raise ValueError("velocity_grid_points must be at least 3")
        fixed, tracked = self.lab_reference_margins_kg_per_kg
        if not fixed > tracked >= 0.0:
            raise ValueError("lab_reference_margins_kg_per_kg: (fixed R0, tracked R1), R0 > R1")


@dataclass(frozen=True, slots=True)
class PlantScaleConfig:
    """The whole study configuration."""

    version: int
    process_config: Path
    data: DataSettings
    stationarity: StationaritySettings
    placeholders: PlaceholderSettings
    calibration: CalibrationSettings
    observability: ObservabilitySettings
    headroom: HeadroomSettings
    workers: int
    raw: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the parsed YAML (sorted keys)."""
        payload = json.dumps(self.raw, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _pair(values: Any) -> tuple[float, float]:
    low, high = (float(v) for v in values)
    if not low <= high:
        raise ValueError(f"bounds out of order: {values}")
    return low, high


def load_plant_scale_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PlantScaleConfig:
    """Parse the YAML into :class:`PlantScaleConfig`."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    d, s, p = raw["data"], raw["stationarity"], raw["placeholders"]
    c, o, h = raw["calibration"], raw["observability"], raw["headroom"]
    segments = [int(v) for v in s["segments_per_day"]]
    return PlantScaleConfig(
        version=int(raw["version"]),
        process_config=_path(raw["process_config"]),
        data=DataSettings(
            role_key=str(d["role_key"]),
            days={str(k): str(v) for k, v in d["days"].items()},
            normal_density_dry_air_kg_per_m3=float(d["normal_density_dry_air_kg_per_m3"]),
            block_minutes=int(d["block_minutes"]),
        ),
        stationarity=StationaritySettings(
            window_minutes=int(s["window_minutes"]),
            thresholds={str(k): float(v) for k, v in s["thresholds"].items()},
            segments_min=segments[0],
            segments_max=segments[1],
            minimum_gap_minutes=int(s["minimum_gap_minutes"]),
        ),
        placeholders=PlaceholderSettings(
            solid_fraction_reference=float(p["solid_fraction_reference"]),
            solid_fraction_band=tuple(float(v) for v in p["solid_fraction_band"]),
            inlet_absolute_humidity_kg_per_kg=float(p["inlet_absolute_humidity_kg_per_kg"]),
            atomising_air_temperature_celsius=float(p["atomising_air_temperature_celsius"]),
            room_temperature_celsius=float(p["room_temperature_celsius"]),
            feed_temperature_celsius=float(p["feed_temperature_celsius"]),
        ),
        calibration=CalibrationSettings(
            candidates={str(k): str(v["freeboard"]) for k, v in c["candidates"].items()},
            bounds={str(k): _pair(v) for k, v in c["bounds"].items()},
            start={str(k): float(v) for k, v in c["start"].items()},
            tolerance_k=float(c["tolerance_k"]),
            candidate_order=tuple(str(v) for v in c["candidate_order"]),
            velocity_iterations=int(c["velocity_iterations"]),
            velocity_relative_tolerance=float(c["velocity_relative_tolerance"]),
        ),
        observability=ObservabilitySettings(
            sensor_value_config=_path(o["sensor_value_config"]),
            observability_config=_path(o["observability_config"]),
            lab_record=_path(o["lab_record"]),
            delta_x_kg_per_kg=float(o["delta_x_kg_per_kg"]),
            redundant_bed_sensors=int(o["redundant_bed_sensors"]),
        ),
        headroom=HeadroomSettings(
            inlet_temperature_celsius=_pair(h["inlet_temperature_celsius"]),
            air_factor=_pair(h["air_factor"]),
            bed_temperature_limit_celsius=float(h["bed_temperature_limit_celsius"]),
            plant_limit=str(h["plant_limit"]),
            lab_limit_kg_per_kg=float(h["lab_limit_kg_per_kg"]),
            lab_inlet_temperature_celsius=_pair(h["lab_inlet_temperature_celsius"]),
            lab_gas_velocity_m_per_s=_pair(h["lab_gas_velocity_m_per_s"]),
            spray_sigma_factor=float(h["spray_sigma_factor"]),
            solid_fraction_relative_half_band=float(h["solid_fraction_relative_half_band"]),
            velocity_grid_points=int(h["velocity_grid_points"]),
            inlet_tolerance_k=float(h["inlet_tolerance_k"]),
            velocity_relative_tolerance=float(h["velocity_relative_tolerance"]),
            lab_reference_margins_kg_per_kg=(
                float(h["lab_reference_margins_kg_per_kg"][0]),
                float(h["lab_reference_margins_kg_per_kg"][1]),
            ),
            lab_reference_saving=float(h["lab_reference_saving"]),
        ),
        workers=int(raw["workers"]),
        raw=raw,
    )

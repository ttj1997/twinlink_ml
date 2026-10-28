"""Settings of the observability study, loaded from ``conf/observability/observability.yaml``.

Every value is a study setting or a placeholder (see the YAML header); the dataclasses only give
them names and types.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "DynamicsSettings",
    "ObservabilityConfig",
    "OperatingPointSettings",
    "PlantLikeSettings",
    "SnrSettings",
    "StepSettings",
    "load_observability_config",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "observability" / "observability.yaml"


@dataclass(frozen=True, slots=True)
class OperatingPointSettings:
    """Rule deriving the operating points from the Befund 111 record."""

    source: Path
    variant: str
    criterion: str
    minimum_velocity_tolerance_m_per_s: float
    inlet_percentiles_at_minimum_velocity: tuple[float, ...]
    low_inlet_below_celsius: float
    velocity_percentiles_at_low_inlet: tuple[float, ...]
    round_inlet_celsius: int
    round_velocity_m_per_s: int


@dataclass(frozen=True, slots=True)
class SnrSettings:
    """Moisture difference to resolve, bed-sensor redundancy, report tolerance."""

    delta_x_kg_per_kg: float
    redundant_bed_sensors: int
    analytic_tolerance: float


@dataclass(frozen=True, slots=True)
class PlantLikeSettings:
    """Placeholder solid-to-air ratio of the plant."""

    air_flow_nm3_per_h: float
    product_dry_solid_t_per_h: tuple[float, ...]
    normal_density_dry_air_kg_per_m3: float
    ratio_iterations: int

    def ratios(self) -> tuple[float, ...]:
        """Dry solid per dry air, kg/kg, one per product flow."""
        air_kg_per_h = self.air_flow_nm3_per_h * self.normal_density_dry_air_kg_per_m3
        return tuple(1000.0 * p / air_kg_per_h for p in self.product_dry_solid_t_per_h)


@dataclass(frozen=True, slots=True)
class StepSettings:
    """One step: ``channel`` in {spray_rate, inlet_temperature}, multiplied by ``factor``."""

    name: str
    channel: str
    factor: float

    def __post_init__(self) -> None:
        if self.channel not in ("spray_rate", "inlet_temperature"):
            raise ValueError(f"unknown step channel {self.channel!r}")


@dataclass(frozen=True, slots=True)
class DynamicsSettings:
    """Step responses of the kinetic extremes."""

    operating_point: str
    draws: int
    kinetics_parameters: tuple[str, str]
    steps: tuple[StepSettings, ...]
    horizon_s: float
    record_every_s: float
    threshold_sigma: float


@dataclass(frozen=True)
class ObservabilityConfig:
    """All settings plus the path of the sensor-value config it reads."""

    version: int
    sensor_value_config: Path
    operating_points: OperatingPointSettings
    snr: SnrSettings
    plant_like: PlantLikeSettings
    dynamics: DynamicsSettings
    raw: dict[str, Any] = field(compare=False, repr=False)

    @property
    def fingerprint(self) -> str:
        """Hash of the YAML content (sorted JSON), recorded in every artifact."""
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:16]


def load_observability_config(path: Path | str = DEFAULT_CONFIG_PATH) -> ObservabilityConfig:
    """Read the study YAML."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    op, snr, plant, dyn = raw["operating_points"], raw["snr"], raw["plant_like"], raw["dynamics"]
    kinetics = tuple(str(p) for p in dyn["kinetics_parameters"])
    if len(kinetics) != 2:
        raise ValueError("kinetics_parameters: exactly two (critical moisture, nu_max)")
    return ObservabilityConfig(
        version=int(raw["version"]),
        sensor_value_config=REPO_ROOT / raw["sensor_value_config"],
        operating_points=OperatingPointSettings(
            source=REPO_ROOT / op["source"],
            variant=str(op["variant"]),
            criterion=str(op["criterion"]),
            minimum_velocity_tolerance_m_per_s=float(op["minimum_velocity_tolerance_m_per_s"]),
            inlet_percentiles_at_minimum_velocity=tuple(
                float(p) for p in op["inlet_percentiles_at_minimum_velocity"]
            ),
            low_inlet_below_celsius=float(op["low_inlet_below_celsius"]),
            velocity_percentiles_at_low_inlet=tuple(
                float(p) for p in op["velocity_percentiles_at_low_inlet"]
            ),
            round_inlet_celsius=int(op["round_inlet_celsius"]),
            round_velocity_m_per_s=int(op["round_velocity_m_per_s"]),
        ),
        snr=SnrSettings(
            delta_x_kg_per_kg=float(snr["delta_x_kg_per_kg"]),
            redundant_bed_sensors=int(snr["redundant_bed_sensors"]),
            analytic_tolerance=float(snr["analytic_tolerance"]),
        ),
        plant_like=PlantLikeSettings(
            air_flow_nm3_per_h=float(plant["air_flow_nm3_per_h"]),
            product_dry_solid_t_per_h=tuple(float(p) for p in plant["product_dry_solid_t_per_h"]),
            normal_density_dry_air_kg_per_m3=float(plant["normal_density_dry_air_kg_per_m3"]),
            ratio_iterations=int(plant["ratio_iterations"]),
        ),
        dynamics=DynamicsSettings(
            operating_point=str(dyn["operating_point"]),
            draws=int(dyn["draws"]),
            kinetics_parameters=(kinetics[0], kinetics[1]),
            steps=tuple(
                StepSettings(name=str(k), channel=str(v["channel"]), factor=float(v["factor"]))
                for k, v in dyn["steps"].items()
            ),
            horizon_s=float(dyn["horizon_s"]),
            record_every_s=float(dyn["record_every_s"]),
            threshold_sigma=float(dyn["threshold_sigma"]),
        ),
        raw=raw,
    )

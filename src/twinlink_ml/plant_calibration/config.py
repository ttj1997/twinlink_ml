"""Settings of the plant calibration study (``conf/plant_calibration/plant_calibration.yaml``)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..plant_scale.config import _path

__all__ = [
    "KINDS",
    "CalibrationSettings",
    "ParameterSpec",
    "PlantCalibrationConfig",
    "ProfileSettings",
    "SensitivitySettings",
    "load_plant_calibration_config",
]

KINDS = ("steady", "kinetics", "dynamic")
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "plant_calibration" / "plant_calibration.yaml"


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """One candidate parameter: nominal value, normalisation scale, bounds, kind."""

    name: str
    nominal: float
    scale: float
    bounds: tuple[float, float]
    kind: str
    product: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown parameter kind {self.kind!r}")
        if not self.scale > 0.0:
            raise ValueError(f"scale of {self.name} must be positive")
        lo, hi = self.bounds
        if not lo < hi:
            raise ValueError(f"bounds of {self.name} must be increasing")
        if not lo <= self.nominal <= hi:
            raise ValueError(f"nominal of {self.name} outside its bounds")


@dataclass(frozen=True, slots=True)
class SensitivitySettings:
    """Finite-difference step and the identifiability thresholds."""

    relative_step: float
    delta_min: float
    gamma_max: float
    max_parameters: int


@dataclass(frozen=True, slots=True)
class ProfileSettings:
    """Profile likelihood grid and threshold."""

    grid_points: int
    threshold_chi2: float


@dataclass(frozen=True, slots=True)
class CalibrationSettings:
    """Tolerance of the reading, fit budget and starting values."""

    tolerance_k: float
    max_nfev: int
    start: Mapping[str, float]


@dataclass(frozen=True)
class PlantCalibrationConfig:
    """Parsed study settings."""

    version: int
    plant_scale_config: Path
    products: Mapping[str, str]
    sigma_sensor_k: float
    bed_sensors: int
    exhaust_sensors: int
    parameters: tuple[ParameterSpec, ...]
    shared_offset: ParameterSpec
    sensitivity: SensitivitySettings
    profile: ProfileSettings
    calibration: CalibrationSettings
    workers: int

    @property
    def sigma_bed_k(self) -> float:
        return self.sigma_sensor_k / math.sqrt(self.bed_sensors)

    @property
    def sigma_exhaust_k(self) -> float:
        return self.sigma_sensor_k / math.sqrt(self.exhaust_sensors)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    def spec(self, name: str) -> ParameterSpec:
        for p in (*self.parameters, self.shared_offset):
            if p.name == name:
                return p
        raise KeyError(name)

    def nominal(self) -> dict[str, float]:
        """Parameter vector of the uncalibrated model (main variant)."""
        return {p.name: p.nominal for p in self.parameters}


def _spec(name: str, raw: Mapping[str, Any], kind: str | None = None) -> ParameterSpec:
    lo, hi = (float(v) for v in raw["bounds"])
    return ParameterSpec(
        name=name,
        nominal=float(raw["nominal"]),
        scale=float(raw["scale"]),
        bounds=(lo, hi),
        kind=str(raw.get("kind", kind)),
        product=raw.get("product"),
    )


def load_plant_calibration_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> PlantCalibrationConfig:
    """Parse the YAML into :class:`PlantCalibrationConfig`."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    s, pr, c, sh = raw["sensitivity"], raw["profile"], raw["calibration"], raw["shared_offset"]
    return PlantCalibrationConfig(
        version=int(raw["version"]),
        plant_scale_config=_path(raw["plant_scale_config"]),
        products={str(k): str(v) for k, v in raw["products"].items()},
        sigma_sensor_k=float(raw["sigma"]["sensor_k"]),
        bed_sensors=int(raw["sigma"]["bed_sensors"]),
        exhaust_sensors=int(raw["sigma"]["exhaust_sensors"]),
        parameters=tuple(_spec(n, v) for n, v in raw["parameters"].items()),
        shared_offset=_spec(str(sh["name"]), sh, kind="steady"),
        sensitivity=SensitivitySettings(
            relative_step=float(s["relative_step"]),
            delta_min=float(s["delta_min"]),
            gamma_max=float(s["gamma_max"]),
            max_parameters=int(s["max_parameters"]),
        ),
        profile=ProfileSettings(
            grid_points=int(pr["grid_points"]), threshold_chi2=float(pr["threshold_chi2"])
        ),
        calibration=CalibrationSettings(
            tolerance_k=float(c["tolerance_k"]),
            max_nfev=int(c["max_nfev"]),
            start={str(k): float(v) for k, v in c["start"].items()},
        ),
        workers=int(raw["workers"]),
    )

"""Settings of the outlier robustness study, loaded from ``conf/sensor/outliers_placeholder.yaml``.

Every value is a study setting or a placeholder (see the YAML header); the dataclasses only give
them names and types.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "HampelSettings",
    "OutlierSettings",
    "OutlierStudyConfig",
    "load_outlier_config",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "sensor" / "outliers_placeholder.yaml"


@dataclass(frozen=True, slots=True)
class OutlierSettings:
    """Rates and shape of the outlier process (placeholders)."""

    rates: tuple[float, ...]
    t_degrees_of_freedom: float
    scale_factor: float
    mean_burst_length_steps: float

    def __post_init__(self) -> None:
        if any(not 0.0 <= rate < 1.0 for rate in self.rates):
            raise ValueError(f"rates must lie in [0, 1), got {self.rates}")
        if self.t_degrees_of_freedom <= 0.0 or self.scale_factor < 0.0:
            raise ValueError("t_degrees_of_freedom must be > 0 and scale_factor >= 0")
        if self.mean_burst_length_steps < 1.0:
            raise ValueError("mean_burst_length_steps must be >= 1")


@dataclass(frozen=True, slots=True)
class HampelSettings:
    """Causal Hampel filter (placeholders)."""

    window: int
    threshold: float
    side_check_windows: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OutlierStudyConfig:
    """The whole study configuration."""

    version: int
    channel: str
    outliers: OutlierSettings
    hampel: HampelSettings

    @property
    def digest(self) -> str:
        """Short SHA-256 of the settings (logged with every record)."""
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def as_parameters(self) -> dict[str, Any]:
        """For the run log."""
        return {
            "outlier_config/version": self.version,
            "outlier_config/digest": self.digest,
            "outlier_config/channel": self.channel,
            "outlier_config/t_degrees_of_freedom": self.outliers.t_degrees_of_freedom,
            "outlier_config/scale_factor": self.outliers.scale_factor,
            "outlier_config/mean_burst_length_steps": self.outliers.mean_burst_length_steps,
            "outlier_config/hampel_threshold": self.hampel.threshold,
        }


def load_outlier_config(path: Path = DEFAULT_CONFIG_PATH) -> OutlierStudyConfig:
    """Read the study settings."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    outliers = raw["outliers"]
    hampel = raw["hampel"]
    return OutlierStudyConfig(
        version=int(raw["version"]),
        channel=str(raw["channel"]),
        outliers=OutlierSettings(
            rates=tuple(float(rate) for rate in outliers["rates"]),
            t_degrees_of_freedom=float(outliers["t_degrees_of_freedom"]),
            scale_factor=float(outliers["scale_factor"]),
            mean_burst_length_steps=float(outliers["mean_burst_length_steps"]),
        ),
        hampel=HampelSettings(
            window=int(hampel["window"]),
            threshold=float(hampel["threshold"]),
            side_check_windows=tuple(int(w) for w in hampel["side_check_windows"]),
        ),
    )

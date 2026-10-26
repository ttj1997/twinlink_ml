"""Settings of the sensor-value study, loaded from ``conf/sensor_value/sensor_value.yaml``.

Every value is a study setting or a placeholder (see the YAML header); the dataclasses only give
them names and types. The truth line, the scenarios, R1 and the numerics come from the headroom
study config named there, the parameter bands from the Sobol study config and the outlier
settings from the outlier study config -- all three are read, never copied.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "FOLLOW_UP_CONFIG_PATH",
    "DrawSettings",
    "FeedbackSettings",
    "FollowUpConfig",
    "LabSampleSettings",
    "LoopSettings",
    "MarginSettings",
    "ResponseSoftSensorSettings",
    "ResponseSoftSensorVariant",
    "SensorValueConfig",
    "load_follow_up_config",
    "load_sensor_value_config",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "sensor_value" / "sensor_value.yaml"


@dataclass(frozen=True, slots=True)
class DrawSettings:
    """Parameter draws of the truth model."""

    n: int
    seed: int
    sequence_length: int
    parameters: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LoopSettings:
    """Closed-loop settings: initial phase, feedback grid and bias filter."""

    warm_up_s: float
    pre_roll_s: float
    measurement_interval_s: float
    bias_filter_time_constant_s: float


@dataclass(frozen=True, slots=True)
class FeedbackSettings:
    """Settings of the feedback variants F2 (inline sensor) and F3 (soft sensor)."""

    outlier_rate: float
    soft_sensor_training_seeds: tuple[int, ...]
    soft_sensor_n_train: int
    soft_sensor_architecture: str
    soft_sensor_max_epochs: int


@dataclass(frozen=True, slots=True)
class MarginSettings:
    """Margin criteria."""

    holding_share: float
    lower_kg_per_kg: float
    max_upper_kg_per_kg: float


@dataclass(frozen=True, slots=True)
class SensorValueConfig:
    """All settings plus the paths of the three configs it reads."""

    version: int
    headroom_config: Path
    sensitivity_config: Path
    outlier_config: Path
    draws: DrawSettings
    scenario_family: str
    scenarios_per_type: int
    loop: LoopSettings
    feedback: FeedbackSettings
    margin: MarginSettings
    noise_seed: int
    raw: dict[str, Any] = field(compare=False, repr=False)

    @property
    def fingerprint(self) -> str:
        """Hash of the YAML content (sorted JSON), recorded in every artifact."""
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:16]


def load_sensor_value_config(path: Path | str = DEFAULT_CONFIG_PATH) -> SensorValueConfig:
    """Read the study YAML."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    draws, loop, fb, margin = raw["draws"], raw["loop"], raw["feedback"], raw["margin"]
    return SensorValueConfig(
        version=int(raw["version"]),
        headroom_config=REPO_ROOT / raw["headroom_config"],
        sensitivity_config=REPO_ROOT / raw["sensitivity_config"],
        outlier_config=REPO_ROOT / raw["outlier_config"],
        draws=DrawSettings(
            n=int(draws["n"]),
            seed=int(draws["seed"]),
            sequence_length=int(draws["sequence_length"]),
            parameters=tuple(str(p) for p in draws["parameters"]),
        ),
        scenario_family=str(raw["scenarios"]["family"]),
        scenarios_per_type=int(raw["scenarios"]["per_type"]),
        loop=LoopSettings(**{k: float(v) for k, v in loop.items()}),
        feedback=FeedbackSettings(
            outlier_rate=float(fb["F2"]["outlier_rate"]),
            soft_sensor_training_seeds=tuple(int(s) for s in fb["F3"]["training_seeds"]),
            soft_sensor_n_train=int(fb["F3"]["n_train"]),
            soft_sensor_architecture=str(fb["F3"]["architecture"]),
            soft_sensor_max_epochs=int(fb["F3"]["max_epochs"]),
        ),
        margin=MarginSettings(
            holding_share=float(margin["holding_share"]),
            lower_kg_per_kg=float(margin["lower_kg_per_kg"]),
            max_upper_kg_per_kg=float(margin["max_upper_kg_per_kg"]),
        ),
        noise_seed=int(raw["seeds"]["noise"]),
        raw=raw,
    )


# ------------------------------------------------------------------------ follow-up study (docs/29)

FOLLOW_UP_CONFIG_PATH = REPO_ROOT / "conf" / "sensor_value" / "sensor_value_2.yaml"


@dataclass(frozen=True, slots=True)
class LabSampleSettings:
    """F4: periodic laboratory sample."""

    sampling_intervals_h: tuple[float, ...]
    delay_s: float
    error_std_kg_per_kg: float
    gain: float


@dataclass(frozen=True, slots=True)
class ResponseSoftSensorVariant:
    """One F5 training variant: data source and measured inputs."""

    label: str
    data: str
    inputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResponseSoftSensorSettings:
    """F5: GRU soft sensors with measured process responses."""

    training_seeds: tuple[int, ...]
    n_train: int
    architecture: str
    max_epochs: int
    variants: tuple[ResponseSoftSensorVariant, ...]
    randomisation_seed: int
    randomisation_sequence_length: int


@dataclass(frozen=True, slots=True)
class FollowUpConfig:
    """Settings of the follow-up study; the shared settings are those of ``base``."""

    version: int
    base: SensorValueConfig
    base_path: Path
    predecessor_run: str
    control_pairs: int
    lab: LabSampleSettings
    soft: ResponseSoftSensorSettings
    budget_limit_h: float
    raw: dict[str, Any] = field(compare=False, repr=False)

    @property
    def fingerprint(self) -> str:
        """Hash of this YAML and of the base config's fingerprint."""
        payload = {"follow_up": self.raw, "base": self.base.fingerprint}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def load_follow_up_config(path: Path | str = FOLLOW_UP_CONFIG_PATH) -> FollowUpConfig:
    """Read the follow-up YAML and the base config it names."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    base_path = REPO_ROOT / raw["base_config"]
    lab, soft = raw["F4"], raw["F5"]
    variants = tuple(
        ResponseSoftSensorVariant(label=str(k), data=str(v["data"]), inputs=tuple(v["inputs"]))
        for k, v in soft["variants"].items()
    )
    if any(v.data not in ("nominal", "randomised") for v in variants):
        raise ValueError("F5 data must be 'nominal' or 'randomised'")
    return FollowUpConfig(
        version=int(raw["version"]),
        base=load_sensor_value_config(base_path),
        base_path=base_path,
        predecessor_run=str(raw["predecessor"]["run"]),
        control_pairs=int(raw["predecessor"]["control_pairs"]),
        lab=LabSampleSettings(
            sampling_intervals_h=tuple(float(h) for h in lab["sampling_intervals_h"]),
            delay_s=float(lab["delay_s"]),
            error_std_kg_per_kg=float(lab["error_std_kg_per_kg"]),
            gain=float(lab["gain"]),
        ),
        soft=ResponseSoftSensorSettings(
            training_seeds=tuple(int(s) for s in soft["training_seeds"]),
            n_train=int(soft["n_train"]),
            architecture=str(soft["architecture"]),
            max_epochs=int(soft["max_epochs"]),
            variants=variants,
            randomisation_seed=int(soft["randomisation"]["seed"]),
            randomisation_sequence_length=int(soft["randomisation"]["sequence_length"]),
        ),
        budget_limit_h=float(raw["budget"]["limit_h"]),
        raw=raw,
    )

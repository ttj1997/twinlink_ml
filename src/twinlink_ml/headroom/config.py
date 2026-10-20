"""Settings of the dynamic headroom study, loaded from ``conf/headroom/dynamic_headroom.yaml``.

Every value is a study setting or a placeholder (see the YAML header); the dataclasses only give
them names and types. :func:`study_process_config` derives the process config of the lab line
from the study settings: quality limit and nozzle pressure as in the bandit studies, action-box
clipping off (the realised inlet temperature carries an additive disturbance, see
:mod:`twinlink_ml.headroom.truth`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from twinlink_ml.simulation import PlaceholderConfigWarning, ProcessConfig, load_process_config

__all__ = [
    "AmbientSettings",
    "ControlSettings",
    "HeadroomStudyConfig",
    "LineSettings",
    "ObservedSettings",
    "ProductChangeSettings",
    "RobustnessSettings",
    "ScenarioSettings",
    "StressSettings",
    "load_headroom_config",
    "study_process_config",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "headroom" / "dynamic_headroom.yaml"


@dataclass(frozen=True, slots=True)
class LineSettings:
    """Fixed settings of the lab line."""

    product_moisture_limit_kg_per_kg: float
    gas_velocity_minimum_m_per_s: float
    nozzle_pressure_bar: float
    bed_temperature_limit_celsius: float
    nominal_spray_rate_kg_per_h: float
    nominal_solid_fraction: float
    start_margin_kg_per_kg: float


@dataclass(frozen=True, slots=True)
class ControlSettings:
    """Time grid, solver and bisection settings shared by all controllers."""

    interval_s: float
    optimizer_step_s: float
    mpc_horizon_s: float
    violation_tolerance_kg_per_kg: float
    soft_constraint_weight_kwh: float
    margin_upper_kg_per_kg: float
    margin_tolerance_kg_per_kg: float
    gas_heater_efficiency: float

    @property
    def steps_per_interval(self) -> int:
        """Optimiser RK4 steps per control interval."""
        return round(self.interval_s / self.optimizer_step_s)

    @property
    def mpc_intervals(self) -> int:
        """Control intervals in the R3 horizon."""
        return round(self.mpc_horizon_s / self.interval_s)


@dataclass(frozen=True, slots=True)
class AmbientSettings:
    """Day profiles of the ambient air around the bandit weather grid."""

    grid_points_per_axis: int
    diurnal_amplitude_fraction: float
    drift_amplitude_fraction: float
    drift_correlation_time_s: float


@dataclass(frozen=True, slots=True)
class ObservedSettings:
    """Amplitudes of the family "observed" (relative, derived from plant data, placeholder)."""

    throughput_relative_amplitude: float
    throughput_hold_time_s: tuple[float, float]
    solid_fraction_relative_half_band: float
    solid_fraction_std_fraction_of_half_band: float
    solid_fraction_correlation_time_s: float
    inlet_temperature_slow: tuple[float, float]
    inlet_temperature_fast: tuple[float, float]


@dataclass(frozen=True, slots=True)
class StressSettings:
    """Amplitudes of the family "stress" (upper estimate)."""

    throughput_levels: tuple[float, ...]
    throughput_hold_time_s: tuple[float, float]
    solid_fraction_band: tuple[float, float]
    solid_fraction_hold_time_s: tuple[float, float]
    inlet_temperature_std_factor: float


@dataclass(frozen=True, slots=True)
class ProductChangeSettings:
    """Changeover between two products."""

    duration_s: float
    change_time_s: float
    moisture_limit_b_kg_per_kg: float
    solid_fraction_b: float
    per_direction: int


@dataclass(frozen=True, slots=True)
class ScenarioSettings:
    """Scenario generation."""

    seed: int
    duration_s: float
    sample_time_s: float
    ambient: AmbientSettings
    per_type: int
    observed: ObservedSettings
    stress: StressSettings
    product_change: ProductChangeSettings


@dataclass(frozen=True, slots=True)
class RobustnessSettings:
    """Isotherm spread of the truth model."""

    draws: int
    seed: int
    monolayer_factor: tuple[float, float]
    c_constant_factor_log: tuple[float, float]
    k_constant: tuple[float, float]
    enthalpy_factor: tuple[float, float]
    holding_share: float
    scenario_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HeadroomStudyConfig:
    """All settings of the study plus the path of the process config."""

    version: int
    process_config: Path
    line: LineSettings
    control: ControlSettings
    scenarios: ScenarioSettings
    robustness: RobustnessSettings
    raw: dict[str, Any] = dataclasses.field(compare=False, repr=False)

    @property
    def fingerprint(self) -> str:
        """Hash of the YAML content (sorted JSON), recorded in every artifact."""
        text = json.dumps(self.raw, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _pair(value: Any) -> tuple[float, float]:
    low, high = (float(v) for v in value)
    return low, high


def _ou(value: dict[str, Any]) -> tuple[float, float]:
    return float(value["std_k"]), float(value["correlation_time_s"])


def load_headroom_config(path: Path | str = DEFAULT_CONFIG_PATH) -> HeadroomStudyConfig:
    """Read the study YAML."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    sc = raw["scenarios"]
    obs, stress, change = sc["observed"], sc["stress"], sc["product_change"]
    rob = raw["robustness"]
    return HeadroomStudyConfig(
        version=int(raw["version"]),
        process_config=REPO_ROOT / raw["process_config"],
        line=LineSettings(**{k: float(v) for k, v in raw["line"].items()}),
        control=ControlSettings(**{k: float(v) for k, v in raw["control"].items()}),
        scenarios=ScenarioSettings(
            seed=int(sc["seed"]),
            duration_s=float(sc["duration_s"]),
            sample_time_s=float(sc["sample_time_s"]),
            ambient=AmbientSettings(
                grid_points_per_axis=int(sc["ambient"]["grid_points_per_axis"]),
                diurnal_amplitude_fraction=float(sc["ambient"]["diurnal_amplitude_fraction"]),
                drift_amplitude_fraction=float(sc["ambient"]["drift_amplitude_fraction"]),
                drift_correlation_time_s=float(sc["ambient"]["drift_correlation_time_s"]),
            ),
            per_type=int(sc["per_type"]),
            observed=ObservedSettings(
                throughput_relative_amplitude=float(obs["throughput_relative_amplitude"]),
                throughput_hold_time_s=_pair(obs["throughput_hold_time_s"]),
                solid_fraction_relative_half_band=float(obs["solid_fraction_relative_half_band"]),
                solid_fraction_std_fraction_of_half_band=float(
                    obs["solid_fraction_std_fraction_of_half_band"]
                ),
                solid_fraction_correlation_time_s=float(obs["solid_fraction_correlation_time_s"]),
                inlet_temperature_slow=_ou(obs["inlet_temperature_slow"]),
                inlet_temperature_fast=_ou(obs["inlet_temperature_fast"]),
            ),
            stress=StressSettings(
                throughput_levels=tuple(float(v) for v in stress["throughput_levels"]),
                throughput_hold_time_s=_pair(stress["throughput_hold_time_s"]),
                solid_fraction_band=_pair(stress["solid_fraction_band"]),
                solid_fraction_hold_time_s=_pair(stress["solid_fraction_hold_time_s"]),
                inlet_temperature_std_factor=float(stress["inlet_temperature_std_factor"]),
            ),
            product_change=ProductChangeSettings(
                duration_s=float(change["duration_s"]),
                change_time_s=float(change["change_time_s"]),
                moisture_limit_b_kg_per_kg=float(change["moisture_limit_b_kg_per_kg"]),
                solid_fraction_b=float(change["solid_fraction_b"]),
                per_direction=int(change["per_direction"]),
            ),
        ),
        robustness=RobustnessSettings(
            draws=int(rob["draws"]),
            seed=int(rob["seed"]),
            monolayer_factor=_pair(rob["monolayer_factor"]),
            c_constant_factor_log=_pair(rob["c_constant_factor_log"]),
            k_constant=_pair(rob["k_constant"]),
            enthalpy_factor=_pair(rob["enthalpy_factor"]),
            holding_share=float(rob["holding_share"]),
            scenario_indices=tuple(int(i) for i in rob["scenario_indices"]),
        ),
        raw=raw,
    )


def study_process_config(study: HeadroomStudyConfig) -> ProcessConfig:
    """Process config of the lab line for this study.

    Changes against the loaded YAML, each a study setting: quality limit
    (``product_moisture_limit_kg_per_kg``, as the bandit env's override), nozzle pressure nominal
    = the fixed point, ``enforce_action_bounds`` off (commands are bounded by the controllers;
    the realised inlet temperature = command + disturbance must not be clipped back into the
    command box), measurement off (controllers see the true state).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base = load_process_config(study.process_config)
    line = study.line
    if abs(base.actions.gas_velocity_m_per_s.minimum - line.gas_velocity_minimum_m_per_s) > 1e-12:
        raise ValueError("gas_velocity_minimum_m_per_s differs from the ActionBound minimum")
    actions = dataclasses.replace(
        base.actions,
        nozzle_pressure_bar=dataclasses.replace(
            base.actions.nozzle_pressure_bar, nominal=line.nozzle_pressure_bar
        ),
    )
    return dataclasses.replace(
        base,
        material=dataclasses.replace(
            base.material, product_moisture_limit_kg_per_kg=line.product_moisture_limit_kg_per_kg
        ),
        actions=actions,
        numerics=dataclasses.replace(base.numerics, enforce_action_bounds=False),
        measurement=dataclasses.replace(base.measurement, enabled=False),
    )

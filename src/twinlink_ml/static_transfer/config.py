"""Settings of the static transfer study, loaded from ``conf/static_transfer/static_transfer.yaml``.

Every value is a study setting or a placeholder (see the YAML header). The dataclasses are frozen;
``raw`` keeps the parsed YAML for the fingerprint and the record.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "INPUT_NAMES",
    "TARGET_NAMES",
    "TARGET_UNITS",
    "Box",
    "Condition",
    "EfficiencySpec",
    "InputErrorSpec",
    "MoistureSpec",
    "NoiseSpec",
    "PlantSpec",
    "PretrainingSpec",
    "SourceSpec",
    "StaticTransferConfig",
    "load_config",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "static_transfer" / "static_transfer.yaml"

INPUT_NAMES: tuple[str, ...] = (
    "inlet_celsius",
    "spray_load_kg_per_kg",
    "ambient_celsius",
    "inlet_humidity_g_per_kg",
    "solid_fraction",
)
"""Column order of every input array ``x (n, 5)``."""

TARGET_NAMES: tuple[str, ...] = ("outlet_temperature", "outlet_humidity", "heater_power")
"""Column order of every target array ``y (n, 3)``."""

TARGET_UNITS: tuple[str, ...] = ("K", "g/kg", "kW/(kg/s)")


@dataclass(frozen=True)
class Box:
    """Axis-aligned box over the five inputs (columns of :data:`INPUT_NAMES`)."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.lower) != len(INPUT_NAMES) or len(self.upper) != len(INPUT_NAMES):
            raise ValueError("a box needs one interval per input")
        if any(lo > hi for lo, hi in zip(self.lower, self.upper, strict=True)):
            raise ValueError("box with lower > upper")

    def scale(self, unit: np.ndarray) -> np.ndarray:
        """Map points of the unit cube into the box."""
        lo, hi = np.asarray(self.lower), np.asarray(self.upper)
        return np.asarray(lo + np.asarray(unit, dtype=np.float64) * (hi - lo))

    def contains(self, x: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        lo, hi = np.asarray(self.lower), np.asarray(self.upper)
        x = np.asarray(x, dtype=np.float64)
        return np.asarray(np.all((x >= lo - tol) & (x <= hi + tol), axis=1), dtype=bool)

    def to_unit_symmetric(self, x: np.ndarray) -> np.ndarray:
        """Map the box to ``[-1, 1]`` per input (the fixed a-priori input scaling)."""
        lo, hi = np.asarray(self.lower), np.asarray(self.upper)
        return np.asarray(2.0 * (np.asarray(x, dtype=np.float64) - lo) / (hi - lo) - 1.0)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> Box:
        return cls(
            tuple(float(raw[n][0]) for n in INPUT_NAMES),
            tuple(float(raw[n][1]) for n in INPUT_NAMES),
        )


@dataclass(frozen=True)
class SourceSpec:
    """Properties and fixed values of the partner balance (one input set)."""

    label: str
    cp_gas_j_per_kg_k: float
    cp_vapour_j_per_kg_k: float
    cp_liquid_j_per_kg_k: float
    cp_solid_j_per_kg_k: float
    evaporation_enthalpy_0c_j_per_kg: float
    equilibrium_moisture_kg_per_kg: float
    heater_efficiency: float
    spray_temperature_celsius: float
    pressure_pa: float
    air_kg_per_h: float
    nominal: tuple[float, ...]
    """Nominal input point, order of :data:`INPUT_NAMES`."""


@dataclass(frozen=True)
class EfficiencySpec:
    """``eta(q) = peak - curvature (q - load_at_peak)^2``, ``q`` = duty / nominal duty."""

    peak: float
    load_at_peak: float
    curvature: float


@dataclass(frozen=True)
class MoistureSpec:
    """Product moisture above equilibrium, Langmuir-shaped in the exhaust relative humidity.

    ``X_p = X_eq + excess_max RH / (RH + K)`` with ``K`` chosen so that the excess is
    ``excess_nominal`` at the source's nominal point (computed by the plant, recorded).
    """

    excess_nominal: float
    excess_max: float


@dataclass(frozen=True)
class PlantSpec:
    """One synthetic plant: the balance plus the switched-on deviations (all off = P0)."""

    name: str
    heat_loss_w_per_k_per_kg_h: float = 0.0
    atomising_air_share: float = 0.0
    atomising_air_celsius: float = 25.0
    false_air_share: float = 0.0
    efficiency: EfficiencySpec | None = None
    moisture: MoistureSpec | None = None

    @classmethod
    def from_mapping(cls, name: str, raw: dict[str, Any]) -> PlantSpec:
        eff = raw.get("efficiency")
        moist = raw.get("moisture")
        return cls(
            name=name,
            heat_loss_w_per_k_per_kg_h=float(raw.get("heat_loss_w_per_k_per_kg_h", 0.0)),
            atomising_air_share=float(raw.get("atomising_air_share", 0.0)),
            atomising_air_celsius=float(raw.get("atomising_air_celsius", 25.0)),
            false_air_share=float(raw.get("false_air_share", 0.0)),
            efficiency=None
            if eff is None
            else EfficiencySpec(**{k: float(v) for k, v in eff.items()}),
            moisture=None
            if moist is None
            else MoistureSpec(**{k: float(v) for k, v in moist.items()}),
        )


@dataclass(frozen=True)
class NoiseSpec:
    """Target noise: absolute for temperature and humidity, relative for the power."""

    outlet_temperature_k: float
    outlet_humidity_g_per_kg: float
    heater_power_relative: float

    def scaled(self, factor: float) -> NoiseSpec:
        return NoiseSpec(
            self.outlet_temperature_k * factor,
            self.outlet_humidity_g_per_kg * factor,
            self.heater_power_relative * factor,
        )

    def sigma(self, y: np.ndarray) -> np.ndarray:
        """Standard deviation per point and target for targets ``y (n, 3)``."""
        y = np.asarray(y, dtype=np.float64)
        out = np.empty_like(y)
        out[:, 0] = self.outlet_temperature_k
        out[:, 1] = self.outlet_humidity_g_per_kg
        out[:, 2] = self.heater_power_relative * np.abs(y[:, 2])
        return out


@dataclass(frozen=True)
class InputErrorSpec:
    """Magnitudes of the unknown input offsets (sign per seed)."""

    inlet_k: float
    spray_relative: float
    solid_fraction_relative: float
    inlet_humidity_g_per_kg: float
    sign_seed_offset: int


@dataclass(frozen=True)
class Condition:
    """One run condition: noise level and whether the input offsets act."""

    name: str
    noise_level: str
    input_errors: bool


@dataclass(frozen=True)
class PretrainingSpec:
    """Pretraining of the L3 nets on balance points."""

    n_train: int
    n_validation: int
    seed_offset: int
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int


@dataclass(frozen=True)
class StaticTransferConfig:
    """The YAML, typed where the study code reads it."""

    version: int
    seeds: tuple[int, ...]
    budgets: tuple[int, ...]
    source: SourceSpec
    sensitivity_source: SourceSpec
    boxes: dict[str, Box]
    max_relative_humidity: float
    test_count: int
    test_seed: int
    design_seed_offset: int
    pretraining: PretrainingSpec
    noise: NoiseSpec
    high_noise_factor: float
    input_errors: InputErrorSpec
    conditions: tuple[Condition, ...]
    plants: tuple[PlantSpec, ...]
    pilot_plant: PlantSpec
    affected_targets: dict[str, tuple[str, ...]]
    families: dict[str, str]
    raw: dict[str, Any] = field(compare=False, repr=False)

    @property
    def fingerprint(self) -> str:
        """Hash of the YAML content (sorted JSON), recorded in every artifact."""
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def fingerprint_without_selection(self) -> str:
        """Fingerprint of everything except the ``selected`` block (the pilot fills it)."""
        raw = {k: v for k, v in self.raw.items() if k != "selected"}
        return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:16]

    def plant(self, name: str) -> PlantSpec:
        for p in (*self.plants, self.pilot_plant):
            if p.name == name:
                return p
        raise KeyError(name)

    def condition(self, name: str) -> Condition:
        for c in self.conditions:
            if c.name == name:
                return c
        raise KeyError(name)

    def noise_for(self, condition: Condition) -> NoiseSpec:
        if condition.noise_level == "primary":
            return self.noise
        if condition.noise_level == "high":
            return self.noise.scaled(self.high_noise_factor)
        raise ValueError(condition.noise_level)

    @property
    def target_centre(self) -> np.ndarray:
        return np.asarray(self.raw["scales"]["target_centre"], dtype=np.float64)

    @property
    def target_scale(self) -> np.ndarray:
        return np.asarray(self.raw["scales"]["target_scale"], dtype=np.float64)

    @property
    def residual_scale(self) -> np.ndarray:
        return np.asarray(self.raw["scales"]["residual_scale"], dtype=np.float64)

    @property
    def hidden(self) -> tuple[int, ...]:
        return tuple(int(h) for h in self.raw["mlp"]["hidden"])

    @property
    def calibration_parameters(self) -> dict[str, tuple[float, float, float, float]]:
        return {
            k: (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
            for k, v in self.raw["calibration"]["parameters"].items()
        }

    @property
    def ridge_alphas(self) -> np.ndarray:
        lo, hi, n = self.raw["ridge_alphas_log10"]
        return np.logspace(float(lo), float(hi), int(n))

    def selected(self, method: str) -> dict[str, float] | None:
        value = self.raw["selected"].get(method)
        return None if value is None else {k: float(v) for k, v in value.items()}

    @property
    def alpha(self) -> float:
        return float(self.raw["statistics"]["alpha"])

    @property
    def min_sign_agreement(self) -> int:
        return int(self.raw["statistics"]["min_sign_agreement"])


def _source(raw: dict[str, Any]) -> SourceSpec:
    nominal = raw["nominal"]
    return SourceSpec(
        label=str(raw["label"]),
        cp_gas_j_per_kg_k=float(raw["cp_gas_j_per_kg_k"]),
        cp_vapour_j_per_kg_k=float(raw["cp_vapour_j_per_kg_k"]),
        cp_liquid_j_per_kg_k=float(raw["cp_liquid_j_per_kg_k"]),
        cp_solid_j_per_kg_k=float(raw["cp_solid_j_per_kg_k"]),
        evaporation_enthalpy_0c_j_per_kg=float(raw["evaporation_enthalpy_0c_j_per_kg"]),
        equilibrium_moisture_kg_per_kg=float(raw["equilibrium_moisture_kg_per_kg"]),
        heater_efficiency=float(raw["heater_efficiency"]),
        spray_temperature_celsius=float(raw["spray_temperature_celsius"]),
        pressure_pa=float(raw["pressure_pa"]),
        air_kg_per_h=float(raw["air_kg_per_h"]),
        nominal=tuple(float(nominal[n]) for n in INPUT_NAMES),
    )


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> StaticTransferConfig:
    """Read and check the study YAML."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    budgets = tuple(int(n) for n in raw["budgets"])
    if list(budgets) != sorted(budgets):
        raise ValueError("budgets must be increasing (nested prefixes)")
    source = _source(raw["source"])
    sens_raw = {**raw["source"], **raw["sensitivity_source"]}
    sensitivity = _source(sens_raw)
    conditions = tuple(
        Condition(name, str(v[0]), bool(v[1])) for name, v in raw["conditions"].items()
    )
    plants = tuple(PlantSpec.from_mapping(n, v or {}) for n, v in raw["plants"].items())
    ((pilot_name, pilot_raw),) = raw["pilot_plant"].items()
    pre = raw["pretraining"]
    err = raw["input_errors"]
    noise = raw["noise"]["primary"]
    affected = {k: tuple(v) for k, v in raw["affected_targets"].items()}
    for targets in affected.values():
        unknown = set(targets) - set(TARGET_NAMES)
        if unknown:
            raise ValueError(f"unknown targets {unknown}")
    return StaticTransferConfig(
        version=int(raw["version"]),
        seeds=tuple(int(s) for s in raw["seeds"]),
        budgets=budgets,
        source=source,
        sensitivity_source=sensitivity,
        boxes={k: Box.from_mapping(v) for k, v in raw["boxes"].items()},
        max_relative_humidity=float(raw["admissibility"]["max_relative_humidity"]),
        test_count=int(raw["test_sets"]["count"]),
        test_seed=int(raw["test_sets"]["seed"]),
        design_seed_offset=int(raw["test_sets"]["design_seed_offset"]),
        pretraining=PretrainingSpec(
            n_train=int(pre["n_train"]),
            n_validation=int(pre["n_validation"]),
            seed_offset=int(pre["seed_offset"]),
            learning_rate=float(pre["learning_rate"]),
            batch_size=int(pre["batch_size"]),
            max_epochs=int(pre["max_epochs"]),
            patience=int(pre["patience"]),
        ),
        noise=NoiseSpec(
            float(noise["outlet_temperature_k"]),
            float(noise["outlet_humidity_g_per_kg"]),
            float(noise["heater_power_relative"]),
        ),
        high_noise_factor=float(raw["noise"]["high_factor"]),
        input_errors=InputErrorSpec(
            inlet_k=float(err["inlet_k"]),
            spray_relative=float(err["spray_relative"]),
            solid_fraction_relative=float(err["solid_fraction_relative"]),
            inlet_humidity_g_per_kg=float(err["inlet_humidity_g_per_kg"]),
            sign_seed_offset=int(err["sign_seed_offset"]),
        ),
        conditions=conditions,
        plants=plants,
        pilot_plant=PlantSpec.from_mapping(pilot_name, pilot_raw),
        affected_targets=affected,
        families={k: str(v) for k, v in raw["families"].items()},
        raw=raw,
    )

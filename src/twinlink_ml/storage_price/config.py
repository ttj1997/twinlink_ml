"""Settings of the storage-price study, loaded from ``conf/storage_price/storage_price.yaml``.

Every value is a study setting or a placeholder (see the YAML header and its comments); the
dataclasses only give them names and types. A variant (``variants`` of the YAML) is the main case
with some fields replaced (:func:`load_storage_price_config` with ``variant``).
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..heatpump.config import Economics
from ..simulation.config import config_fingerprint

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ClimateSettings",
    "ControllerSettings",
    "HeatPumpSettings",
    "PlantSettings",
    "PriceSettings",
    "RLSettings",
    "StoragePriceConfig",
    "StorageSettings",
    "load_storage_price_config",
    "variant_names",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "storage_price" / "storage_price.yaml"


@dataclass(frozen=True, slots=True)
class PriceSettings:
    """Source of the day-ahead prices and the constant adder of the plant tariff."""

    source: str
    url: str
    retrieved: str
    raw_file: str
    sha256: str
    year: int
    clock_utc_offset_h: int
    adder_eur_per_kwh: float


@dataclass(frozen=True, slots=True)
class PlantSettings:
    """Streams of the plant-A-like structure (waste heat -> storage -> preheater)."""

    minimum_pinch_k: float
    air_cp_kw_per_k: float
    preheater_outlet_celsius: float
    waste_heat_kw: float
    waste_heat_in_celsius: float
    waste_heat_out_celsius: float
    weekend_factor: float

    def __post_init__(self) -> None:
        if self.waste_heat_in_celsius <= self.waste_heat_out_celsius:
            raise ValueError("waste heat must cool down (in > out)")
        if min(self.air_cp_kw_per_k, self.waste_heat_kw, self.minimum_pinch_k) <= 0.0:
            raise ValueError("capacity rates, waste heat and dT_min must be positive")
        if not 0.0 <= self.weekend_factor <= 1.0:
            raise ValueError("weekend_factor must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ClimateSettings:
    """Deterministic intake temperature (monthly means plus a daily cosine)."""

    monthly_mean_celsius: tuple[float, ...]
    daily_amplitude_k: float
    hour_of_maximum: float

    def __post_init__(self) -> None:
        if len(self.monthly_mean_celsius) != 12:
            raise ValueError("need 12 monthly means")


@dataclass(frozen=True, slots=True)
class HeatPumpSettings:
    """Design pair and size of the heat pump (model of Befund 101)."""

    evaporating_celsius: float
    condensing_celsius: float
    size_kw: float


@dataclass(frozen=True, slots=True)
class StorageSettings:
    """One storage level of Befund 99 with power limits."""

    level_celsius: float
    capacity_kwh: float
    charge_max_kw: float
    discharge_max_kw: float
    loss_per_h: float

    def __post_init__(self) -> None:
        if min(self.capacity_kwh, self.charge_max_kw, self.discharge_max_kw) < 0.0:
            raise ValueError("capacity and power limits must not be negative")
        if not 0.0 <= self.loss_per_h < 1.0:
            raise ValueError("loss_per_h must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class ControllerSettings:
    """Grids of the tuned rules, MPC horizons and the violation tolerance."""

    stored_value_epsilon: float
    r0_kappa_grid: tuple[float, ...]
    r1_quantile_grid: tuple[float, ...]
    r1_window_h: int
    mpc_horizons_h: tuple[int, ...]
    violation_tolerance_relative: float


@dataclass(frozen=True, slots=True)
class RLSettings:
    """SAC / TD3 settings (stable-baselines3)."""

    algorithms: tuple[str, ...]
    seeds: tuple[int, ...]
    total_steps: int
    episode_h: int
    eval_every_steps: int
    gamma: float
    learning_rate: float
    batch_size: int
    buffer_size: int
    learning_starts: int
    net_arch: tuple[int, ...]
    td3_action_noise_std: float
    reward_scale_eur: float


@dataclass(frozen=True, slots=True)
class StoragePriceConfig:
    """All settings of one case (the main case or one variant)."""

    version: int
    variant: str
    prices: PriceSettings
    test_months: tuple[int, ...]
    plant: PlantSettings
    climate: ClimateSettings
    heat_pump: HeatPumpSettings
    storage: StorageSettings
    economics: Economics
    controllers: ControllerSettings
    rl: RLSettings
    raw: dict[str, Any] = dataclasses.field(compare=False, repr=False)

    @property
    def fingerprint(self) -> str:
        """Fingerprint of the resolved settings of this case (variant overrides applied)."""
        return config_fingerprint(self.raw)

    @property
    def training_months(self) -> tuple[int, ...]:
        return tuple(m for m in range(1, 13) if m not in self.test_months)


def _grid(spec: dict[str, float]) -> tuple[float, ...]:
    n = round((spec["stop"] - spec["start"]) / spec["step"]) + 1
    return tuple(float(round(v, 10)) for v in np.linspace(spec["start"], spec["stop"], n))


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for section, fields in override.items():
        if section == "name":
            continue
        if section not in out or not isinstance(out[section], dict):
            raise KeyError(f"variant overrides unknown section {section!r}")
        unknown = set(fields) - set(out[section])
        if unknown:
            raise KeyError(f"variant overrides unknown fields {sorted(unknown)} of {section!r}")
        out[section].update(fields)
    return out


def variant_names(path: Path | str = DEFAULT_CONFIG_PATH) -> list[str]:
    """Names of the variants of the YAML (the main case is ``"main"``)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [str(v["name"]) for v in raw.get("variants", [])]


def load_storage_price_config(
    path: Path | str = DEFAULT_CONFIG_PATH, variant: str = "main"
) -> StoragePriceConfig:
    """Main case, or the main case with the overrides of ``variant``."""
    raw_all = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    variants = {str(v["name"]): v for v in raw_all.get("variants", [])}
    base = {k: v for k, v in raw_all.items() if k != "variants"}
    if variant == "main":
        raw = copy.deepcopy(base)
    elif variant in variants:
        raw = _merge(base, variants[variant])
    else:
        raise KeyError(f"unknown variant {variant!r}; known: {sorted(variants)}")
    raw["variant"] = variant
    c = raw["controllers"]
    rl = raw["rl"]
    eco = raw["economics"]
    return StoragePriceConfig(
        version=int(raw["version"]),
        variant=variant,
        prices=PriceSettings(**raw["prices"]),
        test_months=tuple(int(m) for m in raw["split"]["test_months"]),
        plant=PlantSettings(**{k: float(v) for k, v in raw["plant"].items()}),
        climate=ClimateSettings(
            monthly_mean_celsius=tuple(float(v) for v in raw["climate"]["monthly_mean_celsius"]),
            daily_amplitude_k=float(raw["climate"]["daily_amplitude_k"]),
            hour_of_maximum=float(raw["climate"]["hour_of_maximum"]),
        ),
        heat_pump=HeatPumpSettings(**{k: float(v) for k, v in raw["heat_pump"].items()}),
        storage=StorageSettings(**{k: float(v) for k, v in raw["storage"].items()}),
        # only the gas side and the CO2 factors are used; the electricity price is the hourly
        # series of :mod:`.prices` (the field keeps its Befund-101 default and is not read)
        economics=Economics(
            gas_eur_per_kwh=float(eco["gas_eur_per_kwh"]),
            co2_eur_per_t=float(eco["co2_eur_per_t"]),
            heater_efficiency=float(eco["heater_efficiency"]),
            electricity_co2_kg_per_kwh=float(eco["electricity_co2_kg_per_kwh"]),
            gas_co2_kg_per_kwh=float(eco["gas_co2_kg_per_kwh"]),
        ),
        controllers=ControllerSettings(
            stored_value_epsilon=float(c["stored_value_epsilon"]),
            r0_kappa_grid=_grid(c["r0_kappa_grid"]),
            r1_quantile_grid=_grid(c["r1_quantile_grid"]),
            r1_window_h=int(c["r1_window_h"]),
            mpc_horizons_h=tuple(int(h) for h in c["mpc_horizons_h"]),
            violation_tolerance_relative=float(c["violation_tolerance_relative"]),
        ),
        rl=RLSettings(
            algorithms=tuple(str(a) for a in rl["algorithms"]),
            seeds=tuple(int(s) for s in rl["seeds"]),
            total_steps=int(rl["total_steps"]),
            episode_h=int(rl["episode_h"]),
            eval_every_steps=int(rl["eval_every_steps"]),
            gamma=float(rl["gamma"]),
            learning_rate=float(rl["learning_rate"]),
            batch_size=int(rl["batch_size"]),
            buffer_size=int(rl["buffer_size"]),
            learning_starts=int(rl["learning_starts"]),
            net_arch=tuple(int(n) for n in rl["net_arch"]),
            td3_action_noise_std=float(rl["td3_action_noise_std"]),
            reward_scale_eur=float(rl["reward_scale_eur"]),
        ),
        raw=raw,
    )

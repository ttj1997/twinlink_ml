"""Settings of the campaign-network study (``conf/campaign_network/campaign_network.yaml``).

Every value is a study setting or a placeholder (see the YAML file, which names the source of each
value); the dataclasses only give them names and types. A variant (``variants`` of the YAML) is the
main case with some fields replaced (:func:`load_campaign_network_config` with ``variant``).
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..simulation.config import config_fingerprint

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "CampaignNetworkConfig",
    "CampaignSettings",
    "ClimateSettings",
    "HallSettings",
    "ProcessPlant",
    "SolverSettings",
    "SourceSettings",
    "StorageSettings",
    "WaterSettings",
    "load_campaign_network_config",
    "variant_names",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "campaign_network" / "campaign_network.yaml"


@dataclass(frozen=True, slots=True)
class SourceSettings:
    """Waste-heat source: power levels, supply temperature, return floor."""

    power_kw_levels: tuple[float, ...]
    supply_celsius: float
    floor_celsius: float

    def __post_init__(self) -> None:
        if self.supply_celsius <= self.floor_celsius:
            raise ValueError("supply temperature must lie above the floor")

    @property
    def span_k(self) -> float:
        return self.supply_celsius - self.floor_celsius


@dataclass(frozen=True, slots=True)
class ProcessPlant:
    """Process-air preheater of one plant."""

    name: str
    flow_nm3_per_h: float
    gas_efficiency: float


@dataclass(frozen=True, slots=True)
class HallSettings:
    """Hall heating (seasonal)."""

    peak_kw: float
    supply_celsius: float
    return_celsius: float
    limit_celsius: float
    design_celsius: float
    day_factor: float
    gas_efficiency: float


@dataclass(frozen=True, slots=True)
class WaterSettings:
    """Process water: constant, or zero in the months ``months_off`` (1 = January).

    ``months_off`` is optional in the YAML (absent = constant, the setting of the run v1).
    """

    power_kw: float
    supply_celsius: float
    return_celsius: float
    gas_efficiency: float
    months_off: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ClimateSettings:
    """Ambient temperature profile."""

    monthly_mean_celsius: tuple[float, ...]
    daily_amplitude_k: float
    hour_of_maximum: float


@dataclass(frozen=True, slots=True)
class CampaignSettings:
    """On/off campaigns of the process plants."""

    availability_levels: tuple[float, ...]
    on_days: tuple[float, float]
    hours: int
    burn_in_hours: int
    modes: tuple[str, ...]
    draws: int
    seed: int

    def __post_init__(self) -> None:
        if any(not 0.0 < a <= 1.0 for a in self.availability_levels):
            raise ValueError("availability must lie in (0, 1]")
        if set(self.modes) - {"independent", "joint"}:
            raise ValueError("modes must be 'independent' and/or 'joint'")


@dataclass(frozen=True, slots=True)
class StorageSettings:
    """Hot-water tank on the network loop."""

    hours_levels: tuple[float, ...]
    loss_per_h: float
    discharge_max_fraction: float


@dataclass(frozen=True, slots=True)
class SolverSettings:
    """Numerical settings of the K2 rule and the K3 LP."""

    bisection_iterations: int
    duty_tolerance_kw: float
    lp_method: str


@dataclass(frozen=True, slots=True)
class CampaignNetworkConfig:
    """All settings of one case (the main case or one variant)."""

    version: int
    variant: str
    source: SourceSettings
    approach_k: float
    pump_fraction: float
    air_cp_kw_per_k_per_nm3_h: float
    air_required_celsius: float
    plants: tuple[ProcessPlant, ...]
    hall: HallSettings
    water: WaterSettings
    climate: ClimateSettings
    campaigns: CampaignSettings
    storage: StorageSettings
    gas_co2_kg_per_kwh: float
    electricity_co2_kg_per_kwh: float
    solver: SolverSettings
    raw: dict[str, Any] = dataclasses.field(compare=False, repr=False)

    @property
    def fingerprint(self) -> str:
        """Fingerprint of the resolved settings of this case (variant overrides applied)."""
        return config_fingerprint(self.raw)

    def plant_cp_kw_per_k(self, plant: ProcessPlant) -> float:
        """Heat-capacity flow of the process air of ``plant``."""
        return plant.flow_nm3_per_h * self.air_cp_kw_per_k_per_nm3_h


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


def from_raw(raw: dict[str, Any], variant: str = "main") -> CampaignNetworkConfig:
    """Build the config from a resolved ``dict`` (tests use it to change single fields)."""
    src = raw["source"]
    air = raw["process_air"]
    low = raw["low_temperature"]
    cl = raw["climate"]
    cmp = raw["campaigns"]
    st = raw["storage"]
    car = raw["carriers"]
    sol = raw["solver"]
    cp_per_nm3_h = float(air["normal_density_kg_per_nm3"]) * float(air["cp_kj_per_kg_k"]) / 3600.0
    return CampaignNetworkConfig(
        version=int(raw["version"]),
        variant=variant,
        source=SourceSettings(
            power_kw_levels=tuple(float(v) for v in src["power_kw_levels"]),
            supply_celsius=float(src["supply_celsius"]),
            floor_celsius=float(src["floor_celsius"]),
        ),
        approach_k=float(raw["network"]["approach_k"]),
        pump_fraction=float(raw["network"]["pump_fraction"]),
        air_cp_kw_per_k_per_nm3_h=cp_per_nm3_h,
        air_required_celsius=float(air["required_celsius"]),
        plants=tuple(
            ProcessPlant(
                name=str(p["name"]),
                flow_nm3_per_h=float(p["flow_nm3_per_h"]),
                gas_efficiency=float(p["gas_efficiency"]),
            )
            for p in air["plants"]
        ),
        hall=HallSettings(**{k: float(v) for k, v in low["hall"].items()}),
        water=WaterSettings(
            power_kw=float(low["water"]["power_kw"]),
            supply_celsius=float(low["water"]["supply_celsius"]),
            return_celsius=float(low["water"]["return_celsius"]),
            gas_efficiency=float(low["water"]["gas_efficiency"]),
            months_off=tuple(int(m) for m in low["water"].get("months_off", ())),
        ),
        climate=ClimateSettings(
            monthly_mean_celsius=tuple(float(v) for v in cl["monthly_mean_celsius"]),
            daily_amplitude_k=float(cl["daily_amplitude_k"]),
            hour_of_maximum=float(cl["hour_of_maximum"]),
        ),
        campaigns=CampaignSettings(
            availability_levels=tuple(float(v) for v in cmp["availability_levels"]),
            on_days=(float(cmp["on_days"][0]), float(cmp["on_days"][1])),
            hours=int(cmp["hours"]),
            burn_in_hours=int(cmp["burn_in_hours"]),
            modes=tuple(str(m) for m in cmp["modes"]),
            draws=int(cmp["draws"]),
            seed=int(cmp["seed"]),
        ),
        storage=StorageSettings(
            hours_levels=tuple(float(v) for v in st["hours_levels"]),
            loss_per_h=float(st["loss_per_h"]),
            discharge_max_fraction=float(st["discharge_max_fraction"]),
        ),
        gas_co2_kg_per_kwh=float(car["gas_co2_kg_per_kwh"]),
        electricity_co2_kg_per_kwh=float(car["electricity_co2_kg_per_kwh"]),
        solver=SolverSettings(
            bisection_iterations=int(sol["bisection_iterations"]),
            duty_tolerance_kw=float(sol["duty_tolerance_kw"]),
            lp_method=str(sol["lp_method"]),
        ),
        raw=raw,
    )


def load_campaign_network_config(
    path: Path | str = DEFAULT_CONFIG_PATH, variant: str = "main"
) -> CampaignNetworkConfig:
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
    return from_raw(raw, variant)

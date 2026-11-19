"""Settings of the load-following study, loaded from ``conf/load_following/load_following.yaml``.

The study reuses two existing configurations unchanged and only reads them: the dynamic headroom
study (line, control grid, moisture limit, margin search, ambient day profiles; Befund 105) and
the price study (price file, tariff adder, month split; Befund 115). The fingerprint covers the
study's own YAML and the fingerprints of both, so that a change in either is visible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from twinlink_ml.evaluation.net_energy import GAS_CO2_KG_PER_KWH
from twinlink_ml.headroom.config import (
    HeadroomStudyConfig,
    load_headroom_config,
    study_process_config,
)
from twinlink_ml.simulation import ProcessConfig
from twinlink_ml.storage_price.config import StoragePriceConfig, load_storage_price_config

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DaySettings",
    "LoadFollowingConfig",
    "ThroughputSettings",
    "load_load_following_config",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "load_following" / "load_following.yaml"


@dataclass(frozen=True, slots=True)
class ThroughputSettings:
    """Band of the spray rate and the grids of stage 0 and of the schedule."""

    main_relative_band: float
    stage0_step_kg_per_h: float
    stage0_feasibility_margin_kg_per_kg: float
    schedule_levels: int


@dataclass(frozen=True, slots=True)
class DaySettings:
    """Length of a day and of its tail, selection of the days."""

    hours: int
    tail_hours: int
    seed: int
    test_per_month_by_tercile: tuple[int, int, int]
    training_per_month_by_tercile: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class LoadFollowingConfig:
    """All settings of the study (see the YAML for reasons and placeholder status)."""

    throughput: ThroughputSettings
    day: DaySettings
    tariff_floor_eur_per_kwh: float
    gas_fuel_eur_per_kwh: float
    gas_co2_eur_per_t: float
    quantile_grid: tuple[float, ...]
    spread_grid: tuple[float, ...]
    resolve_interval_h: int
    critical_days: int
    margin_maximum: float
    weeks_per_test_month: int
    budget_wall_clock_h: float
    headroom: HeadroomStudyConfig
    prices: StoragePriceConfig
    process: ProcessConfig
    raw: dict[str, Any]

    @property
    def gas_cost_eur_per_kwh_fuel(self) -> float:
        """Fuel price of gas with the CO2 price, EUR per kWh of fuel."""
        return self.gas_fuel_eur_per_kwh + self.gas_co2_eur_per_t * GAS_CO2_KG_PER_KWH / 1000.0

    @property
    def nominal_spray_rate(self) -> float:
        """Nominal spray rate (throughput) of the line, kg/h."""
        return self.headroom.line.nominal_spray_rate_kg_per_h

    @property
    def intervals_per_hour(self) -> int:
        """Control intervals per hour."""
        return round(3600.0 / self.headroom.control.interval_s)

    @property
    def samples_per_interval(self) -> int:
        """Truth samples per control interval."""
        return round(self.headroom.control.interval_s / self.headroom.scenarios.sample_time_s)

    @property
    def fingerprint(self) -> str:
        """Hash of the own YAML and of the two configurations read (recorded in every artifact)."""
        text = json.dumps(
            {
                "own": self.raw,
                "headroom": self.headroom.fingerprint,
                # only what this study reads from the price study: source file, adder, split
                "prices": self.prices.raw["prices"],
                "test_months": list(self.prices.test_months),
            },
            sort_keys=True,
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _triple(value: Any) -> tuple[int, int, int]:
    a, b, c = (int(v) for v in value)
    return a, b, c


def load_load_following_config(
    path: Path | str = DEFAULT_CONFIG_PATH, *, day_hours: int | None = None
) -> LoadFollowingConfig:
    """Load the study settings; ``day_hours`` shortens the day (dry run only)."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("version") != 1:
        raise ValueError(f"unsupported load-following config version {raw.get('version')!r}")
    headroom = load_headroom_config(REPO_ROOT / raw["headroom_config"])
    prices = load_storage_price_config(REPO_ROOT / raw["prices_config"])
    process = study_process_config(headroom)
    thr, day, days = raw["throughput"], raw["day"], raw["days"]
    hours = int(day["hours"]) if day_hours is None else int(day_hours)
    if not 1 <= hours <= 24:
        raise ValueError("day.hours must lie in 1..24")
    if abs(3600.0 / headroom.control.interval_s - round(3600.0 / headroom.control.interval_s)) > 0:
        raise ValueError("the control interval must divide one hour")
    return LoadFollowingConfig(
        throughput=ThroughputSettings(
            main_relative_band=float(thr["main_relative_band"]),
            stage0_step_kg_per_h=float(thr["stage0_step_kg_per_h"]),
            stage0_feasibility_margin_kg_per_kg=float(thr["stage0_feasibility_margin_kg_per_kg"]),
            schedule_levels=int(thr["schedule_levels"]),
        ),
        day=DaySettings(
            hours=hours,
            tail_hours=int(day["tail_hours"]),
            seed=int(days["seed"]),
            test_per_month_by_tercile=_triple(days["test_per_month_by_tercile"]),
            training_per_month_by_tercile=_triple(days["training_per_month_by_tercile"]),
        ),
        tariff_floor_eur_per_kwh=float(raw["tariff"]["floor_eur_per_kwh"]),
        gas_fuel_eur_per_kwh=float(raw["gas"]["fuel_eur_per_kwh"]),
        gas_co2_eur_per_t=float(raw["gas"]["co2_eur_per_t"]),
        quantile_grid=tuple(float(q) for q in raw["rule"]["quantile_grid"]),
        spread_grid=tuple(float(s) for s in raw["rule"]["spread_grid_fraction_of_band"]),
        resolve_interval_h=int(raw["mpc"]["resolve_interval_h"]),
        critical_days=int(raw["margins"]["critical_days"]),
        margin_maximum=float(raw["margins"]["maximum_kg_per_kg"]),
        weeks_per_test_month=int(raw["weekly"]["weeks_per_test_month"]),
        budget_wall_clock_h=float(raw["budget"]["wall_clock_h"]),
        headroom=headroom,
        prices=prices,
        process=process,
        raw=raw,
    )

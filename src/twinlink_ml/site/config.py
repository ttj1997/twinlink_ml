r"""Placeholders and scenario definition of the site pinch study (``docs/20_site_pinch.md``).

The single source of the numbers is ``conf/site/site_placeholder.yaml``; :func:`load_site_config`
reads it into :class:`SiteConfig`. The table below repeats every placeholder with its main value,
band and source (``tests/test_site.py`` checks that it agrees with the YAML file). EVERY value is
a placeholder (status ``placeholder``): partner statements of an order of magnitude, conventions
of Befunde 91/99, repository defaults or literature ranges -- none of them is a plant measurement.
The only measurements are the two days of plant A, read at run time (``storage_timeslice.plant``).

Roles, not tags: plant A (existing line with storage A1), plant C with lines C1/C2 (planned),
compressor station, hall heating, process water, generic further lines, steam boiler.

================================ ======== ============== ===========================================
placeholder                      value    band           source
================================ ======== ============== ===========================================
minimum_pinch_k                  20       10 - 30        heatnetwork default (two stages in series)
cold_temperature_celsius         15       10 - 25        heatnetwork default ``T_kalt``
compressor_station_kw            800      400 - 1200     partner: up to 1200 kW, 400 kW used today
compressor_hot_in_celsius        80       70 - 90        brief 50-80 degC; manufacturer range
compressor_hot_out_celsius       50       40 - 55        brief; band of Befund 99
compressor_used_today_kw         400      300 - 500      partner: up to about 400 kW used today
compressor_c_gwh_per_a           2.4      1.2 - 3.6      partner annual figure (Befund 99)
exhaust_c_gwh_per_a              1.8      1.8 - 1.8      partner annual figure (reading "annual")
c_line_nm3_per_h                 20000    15000 - 25000  partner: 2 lines of about 20 000 m3/h
c_low_temperature_today_kw       200      100 - 300      partner: about 200 kW to hall/process water
c_operating_fraction             1        0.5 - 1        no schedule of plant C known
generic_flow_nm3_per_h           20000    10000 - 30000  brief: further lines 10 000-30 000 m3/h
hall_kw                          200      0 - 200        partner: up to about 200 kW (peak)
hall_supply_celsius              50       40 - 60        Befund 99
hall_return_celsius              35       30 - 45        Befund 99
hall_limit_celsius               15       12 - 17        Befund 99 (heating limit)
hall_design_celsius              -10      -14 - -8       Befund 99 (design outdoor temperature)
hall_day_factor                  0.4      0.2 - 0.6      Befund 99
water_kw                         200      0 - 200        partner: up to about 200 kW
water_supply_celsius             60       45 - 70        Befund 99
water_return_celsius             15       10 - 25        Befund 99
solid_fraction                   0.634    0.6 - 0.7      balance of plant A (study 4, docs/23)
relative_humidity                0.75     0.5 - 0.9      Befund 91/99 convention
normal_density_kg_per_nm3        1.293    1.293 - 1.293  Befund 91 convention
gas_efficiency                   0.9      0.8 - 1        heatnetwork default heater efficiency
electricity_co2_kg_per_kwh       0.366    0.2 - 0.366    carrier_energy; 0.20 = variant "2030"
gas_co2_kg_per_kwh               0.201    0.201 - 0.201  carrier_energy
network_pump_fraction            0.015    0.005 - 0.03   district-heating pumping, literature
heat_pump_carnot_efficiency      0.5      0.4 - 0.6      industrial heat pumps, literature
boiler_flue_celsius              180      130 - 250      shell boiler without economiser, variant
boiler_excess_air                1.15     1.05 - 1.3     gas boiler air ratio, variant
climate_shift_k                  0        -3 - 3         uniform shift of the monthly means
================================ ======== ============== ===========================================

``solid_fraction``: the energy balance of the two measured days of plant A requires at least 0.575
(product 1) and 0.625 (product 2); 0.634 fits both (parallel study 4, branch
``plant-a-operating-point``, ``docs/23`` section 5). The 0.3 of Befund 99 is inadmissible by that
balance; the study script keeps it only as the paired variant "Befund 99 assumption".

Monthly climate: typical long-term monthly means for northern Germany (order of magnitude,
placeholder, not the site's station), see the YAML file.

Scenario axes (:class:`SiteScenario`): product day of plant A, reading of plant C (``volume``:
20 000 Nm3/h per line as stated; ``annual``: exhaust scaled to the partner's annual figure;
``none``: plant C not built), number of generic lines (0-5) and their product day, the recovery
the generic lines have today (``none`` or their ``own`` exhaust -> process air), and the steam
boiler flue gas as an optional source.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from ..simulation.config import Provenance, config_fingerprint

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "CBasis",
    "PlaceholderInfo",
    "SiteConfig",
    "SitePlaceholders",
    "SiteScenario",
    "load_site_config",
]

CBasis = Literal["volume", "annual", "none"]
GenericToday = Literal["none", "own"]

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "conf" / "site" / "site_placeholder.yaml"
)
"""``conf/site/site_placeholder.yaml`` of the checkout this package is loaded from."""


@dataclass(frozen=True, slots=True)
class PlaceholderInfo:
    """Main value, band, unit and source of one placeholder."""

    value: float
    low: float
    high: float
    unit: str
    source: str

    def __post_init__(self) -> None:
        if not self.low <= self.value <= self.high:
            raise ValueError(f"value {self.value} outside its band [{self.low}, {self.high}]")


@dataclass(frozen=True, slots=True)
class SitePlaceholders:
    """All numeric placeholders of the study (meaning and band: module docstring)."""

    minimum_pinch_k: float
    cold_temperature_celsius: float
    compressor_station_kw: float
    compressor_hot_in_celsius: float
    compressor_hot_out_celsius: float
    compressor_used_today_kw: float
    compressor_c_gwh_per_a: float
    exhaust_c_gwh_per_a: float
    c_line_nm3_per_h: float
    c_low_temperature_today_kw: float
    c_operating_fraction: float
    generic_flow_nm3_per_h: float
    hall_kw: float
    hall_supply_celsius: float
    hall_return_celsius: float
    hall_limit_celsius: float
    hall_design_celsius: float
    hall_day_factor: float
    water_kw: float
    water_supply_celsius: float
    water_return_celsius: float
    solid_fraction: float
    relative_humidity: float
    normal_density_kg_per_nm3: float
    gas_efficiency: float
    electricity_co2_kg_per_kwh: float
    gas_co2_kg_per_kwh: float
    network_pump_fraction: float
    heat_pump_carnot_efficiency: float
    boiler_flue_celsius: float
    boiler_excess_air: float
    climate_shift_k: float

    def __post_init__(self) -> None:
        if self.minimum_pinch_k <= 0.0:
            raise ValueError("minimum_pinch_k must be positive")
        if self.compressor_hot_in_celsius <= self.compressor_hot_out_celsius:
            raise ValueError("compressor_hot_in_celsius must exceed compressor_hot_out_celsius")
        if self.hall_supply_celsius <= self.hall_return_celsius:
            raise ValueError("hall_supply_celsius must exceed hall_return_celsius")
        if self.water_supply_celsius <= self.water_return_celsius:
            raise ValueError("water_supply_celsius must exceed water_return_celsius")
        if not 0.0 < self.gas_efficiency <= 1.0:
            raise ValueError("gas_efficiency must be in (0, 1]")
        if not 0.0 <= self.c_operating_fraction <= 1.0:
            raise ValueError("c_operating_fraction must be in [0, 1]")
        for name in (
            "compressor_station_kw",
            "compressor_used_today_kw",
            "compressor_c_gwh_per_a",
            "c_low_temperature_today_kw",
            "hall_kw",
            "water_kw",
            "network_pump_fraction",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must not be negative")
        if self.generic_flow_nm3_per_h <= 0.0 or self.c_line_nm3_per_h <= 0.0:
            raise ValueError("line flows must be positive")

    @property
    def network_approach_k(self) -> float:
        """Approach of each network exchanger: ``dT_min / 2``.

        ``dT_min`` of the repository is the sum of two stages in series (source -> water -> sink),
        so a water network with this approach on both sides is consistent with the direct target
        (and can never exceed it, see ``tests/test_site.py``).
        """
        return self.minimum_pinch_k / 2.0


@dataclass(frozen=True, slots=True)
class SiteConfig:
    """Placeholders (with bands and sources), monthly climate and site reference figures."""

    placeholders: SitePlaceholders
    info: Mapping[str, PlaceholderInfo]
    climate_monthly_mean_celsius: tuple[float, ...]
    climate_days: tuple[int, ...]
    site_reference: Mapping[str, Any]
    streams: Mapping[str, Mapping[str, str]]
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self) -> None:
        if len(self.climate_monthly_mean_celsius) != 12 or len(self.climate_days) != 12:
            raise ValueError("climate needs 12 monthly means and 12 day counts")
        if sum(self.climate_days) != 365:
            raise ValueError("climate days must sum to 365")
        names = {f.name for f in fields(SitePlaceholders)}
        if set(self.info) != names:
            raise ValueError(
                f"placeholder entries differ from SitePlaceholders: {set(self.info) ^ names}"
            )

    def band(self, name: str) -> tuple[float, float]:
        entry = self.info[name]
        return entry.low, entry.high


def load_site_config(path: str | Path = DEFAULT_CONFIG_PATH) -> SiteConfig:
    """Read ``conf/site/site_placeholder.yaml`` (or another file of the same form)."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    info: dict[str, PlaceholderInfo] = {}
    for name, entry in raw["placeholders"].items():
        value, low, high, unit, source = entry
        info[name] = PlaceholderInfo(float(value), float(low), float(high), str(unit), str(source))
    placeholders = SitePlaceholders(**{name: entry.value for name, entry in info.items()})
    prov = raw.get("provenance", {})
    climate = raw["climate_monthly_mean_celsius"]
    return SiteConfig(
        placeholders=placeholders,
        info=info,
        climate_monthly_mean_celsius=tuple(float(v) for v in climate["values"]),
        climate_days=tuple(int(d) for d in climate["days"]),
        site_reference=dict(raw["site_reference"]),
        streams={k: dict(v) for k, v in raw["streams"].items()},
        provenance=Provenance(
            status=prov.get("status", "placeholder"),
            description=str(prov.get("description", "")),
            open_todos=tuple(prov.get("open_todos", ())),
        ),
    )


@dataclass(frozen=True, slots=True)
class SiteScenario:
    """One scenario: placeholders plus the structural axes of the grid.

    :param day_a: label of the measured day of plant A (``product_1`` / ``product_2``).
    :param c_basis: reading of plant C (see module docstring).
    :param n_generic: number of generic further lines (scaled copies of plant A).
    :param generic_day: measured day whose state the generic lines copy.
    :param generic_today: recovery of the generic lines today (``none`` or ``own``).
    :param boiler: steam-boiler flue gas as an extra source (variant).
    """

    placeholders: SitePlaceholders
    day_a: str = "product_1"
    c_basis: CBasis = "volume"
    n_generic: int = 0
    generic_day: str = "product_1"
    generic_today: GenericToday = "none"
    boiler: bool = False

    def __post_init__(self) -> None:
        if self.c_basis not in ("volume", "annual", "none"):
            raise ValueError(f"unknown c_basis {self.c_basis!r}")
        if self.generic_today not in ("none", "own"):
            raise ValueError(f"unknown generic_today {self.generic_today!r}")
        if self.n_generic < 0:
            raise ValueError("n_generic must not be negative")

    def with_placeholders(self, **changes: Any) -> SiteScenario:
        return replace(self, placeholders=replace(self.placeholders, **changes))

    @property
    def has_c(self) -> bool:
        return self.c_basis != "none"

    @property
    def label(self) -> str:
        ph = self.placeholders
        generic = (
            f"gen{self.n_generic}x{ph.generic_flow_nm3_per_h:g}@{self.generic_day}"
            if self.n_generic
            else "gen0"
        )
        return (
            f"A:{self.day_a}|C:{self.c_basis}|comp{ph.compressor_station_kw:g}|{generic}|"
            f"hall{ph.hall_kw:g}|water{ph.water_kw:g}|dT{ph.minimum_pinch_k:g}"
        )

    @property
    def fingerprint(self) -> str:
        return config_fingerprint(asdict(self))

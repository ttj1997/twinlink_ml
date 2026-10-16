r"""Placeholders and scenario definition of the time-slice pinch study (storage test 1).

Everything that is not a measurement is a PLACEHOLDER with a source status and a band
(:data:`PLACEHOLDER_BANDS`); a scenario is REALISTIC when all of its placeholders lie inside their
bands (:func:`is_realistic`). The values enter the fingerprint
(:func:`~twinlink_ml.simulation.config.config_fingerprint`).

Roles, not tags: nothing here names a measurement point, a company or a product.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from ..simulation.config import GasConfig, config_fingerprint

__all__ = [
    "BAND_CAPACITY_MAX_KWH",
    "CAPACITIES_KWH",
    "LEVELS_CELSIUS",
    "PLACEHOLDER_BANDS",
    "Mode",
    "Placeholders",
    "Scenario",
    "Site",
    "is_realistic",
]

Site = Literal["A", "C", "A+C"]
Mode = Literal["flat", "shift", "weekly"]

CAPACITIES_KWH: tuple[float, ...] = (0.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0)
"""Capacity grid 0, 0.5, 1, 2, 5, 10, 20 MWh (storage A1 holds about 0.7 MWh)."""

BAND_CAPACITY_MAX_KWH = 10000.0
"""Largest capacity counted as realistic (the gate); 20 MWh is reported only."""

LEVELS_CELSIUS: tuple[float, ...] = tuple(float(t) for t in range(30, 85, 5))
"""Storage temperature levels ``T_S`` of the grid, degC."""

PLACEHOLDER_BANDS: dict[str, tuple[float, float]] = {
    "minimum_pinch_k": (10.0, 30.0),
    "cold_temperature_celsius": (10.0, 25.0),
    "compressor_a_kw": (400.0, 1200.0),
    "compressor_hot_in_celsius": (70.0, 90.0),
    "compressor_hot_out_celsius": (40.0, 55.0),
    "compressor_c_gwh_per_a": (1.2, 3.6),
    "low_fraction": (0.1, 0.5),
    "hall_kw": (0.0, 200.0),
    "hall_supply_celsius": (40.0, 60.0),
    "hall_return_celsius": (30.0, 45.0),
    "water_kw": (0.0, 200.0),
    "c_line_nm3_per_h": (15000.0, 25000.0),
    "solid_fraction": (0.2, 0.5),
    "relative_humidity": (0.5, 0.9),
    "loss_per_h": (0.0, 0.01),
    "ambient_shift_k": (-10.0, 5.0),
    "shift_start_h": (5.0, 7.0),
    "shift_end_h": (19.0, 23.0),
}
"""Band of every numeric placeholder: a scenario inside all bands is realistic (gate)."""


@dataclass(frozen=True, slots=True)
class Placeholders:
    """All placeholders that are not scenario axes. Defaults = main case (docstring of the gate).

    Sources of the status: ``dT_min`` and ``T_kalt`` repository defaults of the heat-network
    module; the rest partner statements of an order of magnitude or free placeholders.
    """

    minimum_pinch_k: float = 20.0
    storage_approach_k: float | None = None  # None: a = dT_min (specification); dT_min/2: variant
    cold_temperature_celsius: float = 15.0
    compressor_hot_in_celsius: float = 85.0
    compressor_hot_out_celsius: float = 45.0
    compressor_c_gwh_per_a: float = 2.4
    exhaust_c_gwh_per_a: float = 1.8
    exhaust_c_basis: Literal["volume", "annual"] = "volume"
    low_fraction: float = 0.3
    shift_start_h: float = 6.0
    shift_end_h: float = 22.0
    c2_end_h: float = 14.0
    hall_supply_celsius: float = 50.0
    hall_return_celsius: float = 35.0
    hall_limit_celsius: float = 15.0
    hall_design_celsius: float = -10.0
    hall_day_factor: float = 0.4
    water_supply_celsius: float = 60.0
    water_return_celsius: float = 15.0
    c_line_nm3_per_h: float = 20000.0
    solid_fraction: float = 0.3
    relative_humidity: float = 0.75
    normal_density_kg_per_nm3: float = 1.293
    n_segments_below_dew_point: int = 50
    gas: GasConfig = field(default_factory=GasConfig)

    def __post_init__(self) -> None:
        if self.minimum_pinch_k <= 0.0:
            raise ValueError("minimum_pinch_k must be positive")
        if (
            self.storage_approach_k is not None
            and self.storage_approach_k < self.minimum_pinch_k / 2
        ):
            raise ValueError("storage_approach_k must be at least dT_min/2")
        if self.compressor_hot_in_celsius <= self.compressor_hot_out_celsius:
            raise ValueError("compressor_hot_in_celsius must exceed compressor_hot_out_celsius")
        if self.hall_supply_celsius <= self.hall_return_celsius:
            raise ValueError("hall_supply_celsius must exceed hall_return_celsius")
        if self.water_supply_celsius <= self.water_return_celsius:
            raise ValueError("water_supply_celsius must exceed water_return_celsius")
        if not 0.0 <= self.solid_fraction < 1.0:
            raise ValueError("solid_fraction must be in [0, 1)")
        if not 0.0 <= self.relative_humidity <= 1.0:
            raise ValueError("relative_humidity must be in [0, 1]")
        if not 0.0 <= self.low_fraction <= 1.0:
            raise ValueError("low_fraction must be in [0, 1]")
        if self.exhaust_c_basis not in ("volume", "annual"):
            raise ValueError("exhaust_c_basis must be 'volume' or 'annual'")

    @property
    def approach_k(self) -> float:
        """Approach on the storage side: ``dT_min`` unless set explicitly."""
        return self.minimum_pinch_k if self.storage_approach_k is None else self.storage_approach_k


@dataclass(frozen=True, slots=True)
class Scenario:
    """One scenario of the study (all fields are labelled placeholders or scenario axes)."""

    site: Site
    day: str  # label of the measured day, e.g. "product_1"
    horizon: Literal["day", "week"]
    mode: Mode
    ambient_shift_k: float = 0.0
    compressor_a_kw: float = 800.0
    hall_kw: float = 200.0
    water_kw: float = 200.0
    loss_per_h: float = 0.0
    placeholders: Placeholders = field(default_factory=Placeholders)

    def __post_init__(self) -> None:
        if self.site not in ("A", "C", "A+C"):
            raise ValueError(f"unknown site {self.site!r}")
        if self.horizon not in ("day", "week"):
            raise ValueError(f"unknown horizon {self.horizon!r}")
        if self.mode not in ("flat", "shift", "weekly"):
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.mode == "weekly" and self.horizon != "week":
            raise ValueError("mode 'weekly' needs the week horizon")
        if min(self.compressor_a_kw, self.hall_kw, self.water_kw, self.loss_per_h) < 0.0:
            raise ValueError("powers and the loss rate must not be negative")

    @property
    def n_days(self) -> int:
        return 7 if self.horizon == "week" else 1

    def with_placeholders(self, **changes: Any) -> Scenario:
        return replace(self, placeholders=replace(self.placeholders, **changes))

    @property
    def label(self) -> str:
        return (
            f"{self.site}|{self.day}|{self.horizon}|{self.mode}|shift{self.ambient_shift_k:+g}K|"
            f"compA{self.compressor_a_kw:g}|hall{self.hall_kw:g}|water{self.water_kw:g}|"
            f"loss{self.loss_per_h:g}"
        )

    @property
    def fingerprint(self) -> str:
        return config_fingerprint(asdict(self))


def is_realistic(scenario: Scenario) -> bool:
    """All numeric placeholders of ``scenario`` inside :data:`PLACEHOLDER_BANDS`."""
    ph = scenario.placeholders
    values: dict[str, float] = {
        "minimum_pinch_k": ph.minimum_pinch_k,
        "cold_temperature_celsius": ph.cold_temperature_celsius,
        "compressor_a_kw": scenario.compressor_a_kw,
        "compressor_hot_in_celsius": ph.compressor_hot_in_celsius,
        "compressor_hot_out_celsius": ph.compressor_hot_out_celsius,
        "compressor_c_gwh_per_a": ph.compressor_c_gwh_per_a,
        "low_fraction": ph.low_fraction,
        "hall_kw": scenario.hall_kw,
        "hall_supply_celsius": ph.hall_supply_celsius,
        "hall_return_celsius": ph.hall_return_celsius,
        "water_kw": scenario.water_kw,
        "c_line_nm3_per_h": ph.c_line_nm3_per_h,
        "solid_fraction": ph.solid_fraction,
        "relative_humidity": ph.relative_humidity,
        "loss_per_h": scenario.loss_per_h,
        "ambient_shift_k": scenario.ambient_shift_k,
        "shift_start_h": ph.shift_start_h,
        "shift_end_h": ph.shift_end_h,
    }
    return all(lo <= values[name] <= hi for name, (lo, hi) in PLACEHOLDER_BANDS.items())

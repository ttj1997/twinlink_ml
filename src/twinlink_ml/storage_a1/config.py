r"""Configuration of the heat-storage model of plant A (storage A1, heat exchangers A1).

Everything that is not measured or given by a source is a PLACEHOLDER and says so: the
:class:`Provenance` of :class:`StorageSystemConfig` starts with status ``"placeholder"``, the
list :data:`PLACEHOLDERS` names every placeholder with its source status, and every value enters
the fingerprint (same mechanism as :class:`~twinlink_ml.heatnetwork.config.HeatNetworkConfig`).
Calling :meth:`StorageSystemConfig.warn_if_placeholder` raises the project-wide
:class:`~twinlink_ml.simulation.config.PlaceholderConfigWarning`.

Roles, not tags: this module (like all of ``storage_a1``) never names a measurement point. The
mapping role -> channel lives in the git-ignored ``conf/plant/plant_local.yaml``
(:mod:`twinlink_ml.storage_a1.roles`).

Layout of the system (see ``docs/18_storage_plant_a.md``)::

    compressor heat -- storage loop (storage A1 inside) -- [exchanger A1]
        -- preheater loop -- [preheater A1]
    air path: outside air -> preheater A1 -> steam heater -> fan -> supply air

``topology`` is the switch for the one open question of the storage loop (two control-system
pictures disagree): ``"storage_then_exchanger"`` (variant B, compressors -> storage -> exchanger
-> compressors; the hot side of the exchanger sees the storage temperature, measurable) or
``"exchanger_then_storage"`` (variant A, compressors -> exchanger -> storage -> compressors).
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

from ..simulation.config import (
    GasConfig,
    PlaceholderConfigWarning,
    Provenance,
    config_fingerprint,
    section_fingerprints_of,
    without_documentary_provenance,
)

__all__ = [
    "PLACEHOLDERS",
    "AirConfig",
    "CompressorHeatConfig",
    "ExchangerConfig",
    "SteamHeaterConfig",
    "StorageConfig",
    "StorageSystemConfig",
    "Topology",
    "WaterConfig",
]

Topology = Literal["storage_then_exchanger", "exchanger_then_storage"]
"""Order in the storage loop. B = ``storage_then_exchanger`` (default, hot side of the exchanger
= storage temperature, measurable), A = ``exchanger_then_storage``."""


def _positive(obj: object, *names: str) -> None:
    for name in names:
        value = getattr(obj, name)
        if not value > 0.0:
            raise ValueError(f"{type(obj).__name__}.{name} must be positive, is {value}")


@dataclass(frozen=True, slots=True)
class WaterConfig:
    """Heat-transfer medium of both water loops. ASSUMPTION: plain water (medium not confirmed)."""

    rho_kg_per_m3: float = 990.0
    cp_j_per_kg_k: float = 4180.0

    def __post_init__(self) -> None:
        _positive(self, "rho_kg_per_m3", "cp_j_per_kg_k")

    @property
    def rho_cp_kj_per_m3_k(self) -> float:
        return self.rho_kg_per_m3 * self.cp_j_per_kg_k / 1000.0


@dataclass(frozen=True, slots=True)
class AirConfig:
    """Process air. Material values of the grey-box model are reused, not copied.

    ``normal_density_kg_per_nm3`` converts the measured normal volume flow to a dry-air mass flow
    (moist-air correction of about 1 % neglected). ``relative_humidity`` of the intake air is NOT
    measured (placeholder, same assumption as the plant validation of Befund 56).
    """

    gas: GasConfig = field(default_factory=GasConfig)
    normal_density_kg_per_nm3: float = 1.293
    relative_humidity: float = 0.75

    def __post_init__(self) -> None:
        _positive(self, "normal_density_kg_per_nm3")
        if not 0.0 <= self.relative_humidity <= 1.0:
            raise ValueError(f"relative_humidity must be in [0, 1], is {self.relative_humidity}")


@dataclass(frozen=True, slots=True)
class StorageConfig:
    """Storage A1 as ONE mixed node.

    ``volume_m3``, ``ua_kw_per_k`` and ``max_temperature_celsius`` are placeholders without a
    source (volume: not identifiable from the two days, only bounded, see ``identifiability``).
    ``room_temperature_celsius`` is the temperature the heat loss refers to (unknown location).
    """

    volume_m3: float = 10.0
    ua_kw_per_k: float = 0.02
    room_temperature_celsius: float = 20.0
    max_temperature_celsius: float = 90.0

    def __post_init__(self) -> None:
        _positive(self, "volume_m3")
        if self.ua_kw_per_k < 0.0:
            raise ValueError(f"ua_kw_per_k must be >= 0, is {self.ua_kw_per_k}")


@dataclass(frozen=True, slots=True)
class ExchangerConfig:
    """Counter-current exchanger, effectiveness-NTU with one conductance ``ua_kw_per_k``.

    ``minimum_approach_k``: smallest admissible temperature difference anywhere along the
    exchanger (review finding A1 of the heat-network module, there called ``dT_min``). Placeholder.
    """

    ua_kw_per_k: float = 10.0
    minimum_approach_k: float = 2.0

    def __post_init__(self) -> None:
        _positive(self, "ua_kw_per_k")
        if self.minimum_approach_k < 0.0:
            raise ValueError(f"minimum_approach_k must be >= 0, is {self.minimum_approach_k}")


@dataclass(frozen=True, slots=True)
class SteamHeaterConfig:
    """Steam heater (steam from a gas boiler) and the fan that sits behind it.

    ``fan_heat_fraction``: share of the measured electric fan power that ends up as heat in the
    air (shaft + motor losses inside the stream) -- placeholder. ``boiler_efficiency``: placeholder.
    """

    boiler_efficiency: float = 0.90
    fan_heat_fraction: float = 0.90

    def __post_init__(self) -> None:
        if not 0.0 < self.boiler_efficiency <= 1.0:
            raise ValueError(f"boiler_efficiency must be in (0, 1], is {self.boiler_efficiency}")
        if not 0.0 <= self.fan_heat_fraction <= 1.0:
            raise ValueError(f"fan_heat_fraction must be in [0, 1], is {self.fan_heat_fraction}")


@dataclass(frozen=True, slots=True)
class CompressorHeatConfig:
    """Compressor waste heat. Only the order of magnitude given by the application partner.

    ``total_waste_heat_max_kw``: partner statement "up to about 1200 kW" (upper bound used in the
    identifiability study). ``used_max_kw``: "currently up to about 400 kW used" (partner
    statement, not checked against a measurement). ``rejected_fraction``: share of the waste heat
    discarded over the cooler -- unknown (``None``), no data.
    """

    total_waste_heat_max_kw: float = 1200.0
    used_max_kw: float = 400.0
    rejected_fraction: float | None = None

    def __post_init__(self) -> None:
        _positive(self, "total_waste_heat_max_kw", "used_max_kw")
        if self.rejected_fraction is not None and not 0.0 <= self.rejected_fraction <= 1.0:
            raise ValueError(f"rejected_fraction must be in [0, 1], is {self.rejected_fraction}")


PLACEHOLDERS: dict[str, str] = {
    "storage.volume_m3": "placeholder, no source; only bounded from the data (identifiability)",
    "storage.ua_kw_per_k": "placeholder, no source; not identifiable (Q_in unmeasured)",
    "storage.room_temperature_celsius": "placeholder, location of the storage unknown",
    "storage.max_temperature_celsius": "placeholder, no source; unused (no control in S1)",
    "storage_exchanger.ua_kw_per_k": "to be calibrated per day (identifiable for topology B)",
    "storage_exchanger.minimum_approach_k": "placeholder, physical sanity bound",
    "preheater.ua_kw_per_k": "to be calibrated per day (identifiable)",
    "preheater.minimum_approach_k": "placeholder, physical sanity bound",
    "steam.boiler_efficiency": "placeholder, no source",
    "steam.fan_heat_fraction": "placeholder, no source (fan power is measured)",
    "air.relative_humidity": "placeholder, intake humidity not measured",
    "air.normal_density_kg_per_nm3": "textbook value, moist-air correction neglected",
    "water.*": "assumption: plain water in both loops (medium not confirmed)",
    "compressor.total_waste_heat_max_kw": "partner statement (order of magnitude), no measurement",
    "compressor.used_max_kw": "partner statement (order of magnitude), no measurement",
    "compressor.rejected_fraction": "unknown (None); share discarded over the cooler, no data",
}
"""Every placeholder / unconfirmed value with its status (generated into docs/18)."""


@dataclass(frozen=True, slots=True)
class StorageSystemConfig:
    """Complete configuration of the storage / extraction model (fingerprinted)."""

    provenance: Provenance = field(
        default_factory=lambda: Provenance(
            status="placeholder",
            description="storage A1 model, S1: see PLACEHOLDERS",
            open_todos=tuple(PLACEHOLDERS),
        )
    )
    topology: Topology = "storage_then_exchanger"
    water: WaterConfig = field(default_factory=WaterConfig)
    air: AirConfig = field(default_factory=AirConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    storage_exchanger: ExchangerConfig = field(default_factory=ExchangerConfig)
    preheater: ExchangerConfig = field(default_factory=ExchangerConfig)
    steam: SteamHeaterConfig = field(default_factory=SteamHeaterConfig)
    compressor: CompressorHeatConfig = field(default_factory=CompressorHeatConfig)

    def __post_init__(self) -> None:
        if self.topology not in ("storage_then_exchanger", "exchanger_then_storage"):
            raise ValueError(f"unknown topology {self.topology!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return config_fingerprint(without_documentary_provenance(self.to_dict()))

    @property
    def section_fingerprints(self) -> dict[str, str]:
        sections = tuple(f.name for f in fields(self))
        return section_fingerprints_of(without_documentary_provenance(self.to_dict()), sections)

    def warn_if_placeholder(self) -> None:
        if self.provenance.status == "placeholder":
            warnings.warn(
                "StorageSystemConfig contains placeholder values "
                f"(open items: {len(self.provenance.open_todos)}, see PLACEHOLDERS). "
                "Absolute values are not reliable; use the bands, not single numbers.",
                PlaceholderConfigWarning,
                stacklevel=2,
            )

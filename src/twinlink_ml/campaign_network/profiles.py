"""Hourly profiles of the campaign-network study: climate, campaign schedules, consumer streams.

Every consumer is a cold stream from ``t_in`` to ``t_out`` with heat-capacity flow ``cp`` (kW/K)
per hour; ``cp`` is 0 while the consumer stands still or has no demand.

* process air of plant ``p`` (A, B, C): ``t_in`` = ambient, ``t_out`` = supply - approach (the
  highest preheat the network water can give), ``cp`` = air flow x cp while the plant runs;
* hall heating: return -> supply of the heating circuit, demand = peak x heating degree (heating
  limit to design temperature) x day/night factor (Befund 99/100 profile);
* process water: return (fresh water) -> supply, constant, or zero in the months ``months_off``
  (post-hoc extension v2; calendar of 365 days).

Campaigns: alternating renewal process of on (campaign) and off (standstill) periods, see
:func:`campaign_schedule`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import CampaignNetworkConfig

__all__ = [
    "ColdStream",
    "Consumers",
    "ambient_temperature",
    "build_consumers",
    "campaign_schedule",
    "plant_schedules",
]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

_MID_MONTH_DAY = np.array(
    [15.5, 45.0, 74.5, 105.0, 135.5, 166.0, 196.5, 227.5, 258.0, 288.5, 319.0, 349.5]
)
"""Day of the year (0-based, fractional) of the middle of each month (anchor of the means)."""


def ambient_temperature(config: CampaignNetworkConfig) -> FloatArray:
    """Ambient temperature per hour: monthly means interpolated cyclically plus a daily cosine."""
    cl = config.climate
    n = config.campaigns.hours
    t = np.arange(n, dtype=np.float64)
    hour = t % 24.0
    x = t / 24.0
    days = 365.0
    means = np.asarray(cl.monthly_mean_celsius, dtype=np.float64)
    xp = np.concatenate([_MID_MONTH_DAY - days, _MID_MONTH_DAY, _MID_MONTH_DAY + days])
    base = np.interp(x, xp, np.concatenate([means, means, means]))
    daily = cl.daily_amplitude_k * np.cos(2.0 * np.pi * (hour - cl.hour_of_maximum) / 24.0)
    return np.asarray(base + daily, dtype=np.float64)


def campaign_schedule(
    availability: float,
    on_days: tuple[float, float],
    hours: int,
    burn_in_hours: int,
    rng: np.random.Generator,
) -> BoolArray:
    """On/off schedule (True = running) of one consumer, 1 h step.

    Alternating renewal process started at ``-burn_in_hours`` in a random state (on with
    probability ``availability``): on durations uniform in ``on_days`` (days), off durations
    uniform in ``on_days x (1 - a) / a``, so the expected share of on hours is ``a`` exactly.
    Durations are rounded to whole hours (at least 1 h). ``a = 1`` gives all hours on.
    """
    if availability >= 1.0:
        return np.ones(hours, dtype=bool)
    off_scale = (1.0 - availability) / availability
    total = hours + burn_in_hours
    out = np.empty(total, dtype=bool)
    state = bool(rng.random() < availability)
    pos = 0
    while pos < total:
        lo, hi = on_days if state else (on_days[0] * off_scale, on_days[1] * off_scale)
        length = max(1, round(rng.uniform(lo, hi) * 24.0))
        out[pos : pos + length] = state
        pos += length
        state = not state
    return out[burn_in_hours:]


def plant_schedules(
    config: CampaignNetworkConfig, availability: float, mode: str, draw: int
) -> dict[str, BoolArray]:
    """Schedules of the process plants for one draw.

    ``independent``: plant ``i`` uses the random stream ``(seed, draw, a, i)``; ``joint``: every
    plant takes plant A's schedule (stream ``i = 0``), so plant A is identical in both modes.
    The streams do not depend on the source power or the configuration (paired comparison).
    """
    c = config.campaigns
    out: dict[str, BoolArray] = {}
    for i, plant in enumerate(config.plants):
        stream = 0 if mode == "joint" else i
        seq = np.random.SeedSequence([c.seed, draw, round(availability * 1000), stream])
        out[plant.name] = campaign_schedule(
            availability, c.on_days, c.hours, c.burn_in_hours, np.random.default_rng(seq)
        )
    return out


@dataclass(frozen=True, slots=True)
class ColdStream:
    """One consumer.

    :ivar stage_celsius: required temperature, the order of the cascade (highest first).
    :ivar group: consumers of the same group are parallel branches of one cascade stage.
    """

    name: str
    t_in: FloatArray
    t_out: FloatArray
    cp: FloatArray
    gas_efficiency: float
    stage_celsius: float
    group: str

    @property
    def demand_kw(self) -> FloatArray:
        return np.asarray(self.cp * np.maximum(self.t_out - self.t_in, 0.0), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Consumers:
    """All consumers of one case, hour by hour."""

    streams: tuple[ColdStream, ...]
    ambient: FloatArray

    def by_name(self, name: str) -> ColdStream:
        for s in self.streams:
            if s.name == name:
                return s
        raise KeyError(name)


def build_consumers(config: CampaignNetworkConfig, on: dict[str, BoolArray]) -> Consumers:
    """Cold streams of the process plants (with their schedules ``on``), the hall and the water."""
    amb = ambient_temperature(config)
    n = amb.size
    ones = np.ones(n, dtype=np.float64)
    air_out = config.source.supply_celsius - config.approach_k
    streams: list[ColdStream] = []
    for plant in config.plants:
        running = np.asarray(on[plant.name], dtype=np.float64)
        cp = running * config.plant_cp_kw_per_k(plant) * (amb < air_out)
        streams.append(
            ColdStream(
                name=plant.name,
                t_in=amb.copy(),
                t_out=air_out * ones,
                cp=np.asarray(cp, dtype=np.float64),
                gas_efficiency=plant.gas_efficiency,
                stage_celsius=config.air_required_celsius,
                group="process_air",
            )
        )
    h = config.hall
    hour = np.arange(n) % 24
    night = (hour >= 18) | (hour < 6)
    degree = np.clip((h.limit_celsius - amb) / (h.limit_celsius - h.design_celsius), 0.0, 1.0)
    hall_kw = h.peak_kw * degree * np.where(night, 1.0, h.day_factor)
    streams.append(
        ColdStream(
            name="hall",
            t_in=h.return_celsius * ones,
            t_out=h.supply_celsius * ones,
            cp=np.asarray(hall_kw / (h.supply_celsius - h.return_celsius), dtype=np.float64),
            gas_efficiency=h.gas_efficiency,
            stage_celsius=h.supply_celsius,
            group="hall",
        )
    )
    w = config.water
    month = np.searchsorted(np.cumsum(_MONTH_DAYS), (np.arange(n) // 24) % 365, side="right") + 1
    water_on = ~np.isin(month, w.months_off)
    streams.append(
        ColdStream(
            name="water",
            t_in=w.return_celsius * ones,
            t_out=w.supply_celsius * ones,
            cp=w.power_kw / (w.supply_celsius - w.return_celsius) * water_on.astype(np.float64),
            gas_efficiency=w.gas_efficiency,
            stage_celsius=w.supply_celsius,
            group="water",
        )
    )
    return Consumers(streams=tuple(streams), ambient=amb)

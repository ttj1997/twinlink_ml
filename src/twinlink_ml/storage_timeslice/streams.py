r"""Hot and cold streams of a scenario on the 15-minute grid.

Stream names (``StreamBank.hot`` / ``StreamBank.cold``)::

    hot   compressor_a  compressor_c  exhaust_a  exhaust_c1  exhaust_c2
    cold  air_a  air_c1  air_c2  hall  water

Each stream is an array ``(steps, segments, 3)`` of ``[high, low, capacity rate]`` in degC and kW/K
(segments with capacity 0 are padding), the format of :mod:`twinlink_ml.storage_timeslice.table`.
The exhaust streams are built with the existing condensation sampling of the heat-network module
(:func:`twinlink_ml.heatnetwork.pinch.hot_segments_for_source`, one unit of dry-air flow per step,
then scaled -- the capacity rate is linear in the flow) and cached per (day, humidity placeholders,
``T_kalt``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..heatnetwork.config import HeatSourceConfig
from ..heatnetwork.pinch import hot_segments_for_source
from .config import Placeholders, Scenario
from .plant import STEPS_PER_DAY, MeasuredDay, exhaust_humidity, intake_humidity
from .profiles import (
    Clock,
    compressor_profile,
    line_c_profile,
    night_profile,
    water_profile,
)

__all__ = [
    "COLD_NAMES",
    "HOT_NAMES",
    "StreamBank",
    "StreamFactory",
    "site_streams",
]

FloatArray = NDArray[np.float64]

HOT_NAMES = ("compressor_a", "compressor_c", "exhaust_a", "exhaust_c1", "exhaust_c2")
COLD_NAMES = ("air_a", "air_c1", "air_c2", "hall", "water")
_HOURS_PER_YEAR = 8760.0
_SITE_HOT = {
    "A": ("compressor_a", "exhaust_a"),
    "C": ("compressor_c", "exhaust_c1", "exhaust_c2"),
    "A+C": HOT_NAMES,
}
_SITE_COLD = {
    "A": ("air_a", "hall", "water"),
    "C": ("air_c1", "air_c2", "hall", "water"),
    "A+C": COLD_NAMES,
}


def site_streams(site: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Names of the hot and cold streams that exist at a site (hall and water at every site)."""
    return _SITE_HOT[site], _SITE_COLD[site]


@dataclass(frozen=True, slots=True)
class StreamBank:
    """All streams of one scenario, selectable by name."""

    hot: Mapping[str, FloatArray]
    cold: Mapping[str, FloatArray]
    dt_h: float
    n_steps: int
    notes: Mapping[str, float] = field(default_factory=dict)

    def select(
        self, hot_names: Sequence[str], cold_names: Sequence[str]
    ) -> tuple[FloatArray, FloatArray]:
        """Concatenate the chosen streams along the segment axis (``(steps, segments, 3)``)."""
        return _stack([self.hot[n] for n in hot_names], self.n_steps), _stack(
            [self.cold[n] for n in cold_names], self.n_steps
        )


def _stack(parts: list[FloatArray], n_steps: int) -> FloatArray:
    if not parts:
        return np.zeros((n_steps, 1, 3))
    return np.concatenate(parts, axis=1)


def _single(high: float, low: float, cp: FloatArray) -> FloatArray:
    """One segment of constant temperatures and a time-varying capacity rate."""
    out = np.zeros((len(cp), 1, 3))
    out[:, 0, 0] = high
    out[:, 0, 1] = low
    out[:, 0, 2] = cp
    return out


def _tile(day_values: FloatArray, n_days: int) -> FloatArray:
    return np.tile(day_values, n_days)


class StreamFactory:
    """Builds :class:`StreamBank` objects for scenarios; caches the exhaust sampling."""

    def __init__(self, days: Mapping[str, MeasuredDay]) -> None:
        self._days = dict(days)
        self._exhaust_cache: dict[
            tuple[object, ...], tuple[FloatArray, FloatArray, NDArray[np.bool_]]
        ] = {}

    def exhaust_unit(
        self, day_label: str, ph: Placeholders
    ) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
        """Exhaust of the measured day per unit of dry-air flow.

        :returns: ``(segments (96, S, 3) at 1 kg/s dry air, dry-air flow kg/s of plant A,
            capped mask)``.
        """
        key = (
            day_label,
            ph.solid_fraction,
            ph.relative_humidity,
            ph.cold_temperature_celsius,
            ph.n_segments_below_dew_point,
            ph.normal_density_kg_per_nm3,
        )
        if key in self._exhaust_cache:
            return self._exhaust_cache[key]
        day = self._days[day_label]
        y_out, m_dry, capped = exhaust_humidity(
            day,
            solid_fraction=ph.solid_fraction,
            relative_humidity=ph.relative_humidity,
            normal_density_kg_per_nm3=ph.normal_density_kg_per_nm3,
            gas=ph.gas,
        )
        rows = []
        for i in range(STEPS_PER_DAY):
            source = HeatSourceConfig("exhaust", 1.0, float(day.exhaust_out_c[i]), float(y_out[i]))
            segments = hot_segments_for_source(
                source,
                ph.gas,
                ph.cold_temperature_celsius,
                n_segments_below_dew_point=ph.n_segments_below_dew_point,
            )
            rows.append(
                [
                    (
                        s.high_temperature_celsius,
                        s.low_temperature_celsius,
                        s.capacity_rate_kw_per_k,
                    )
                    for s in segments
                ]
            )
        width = max(len(r) for r in rows)
        unit = np.zeros((STEPS_PER_DAY, max(width, 1), 3))
        for i, r in enumerate(rows):
            if r:
                unit[i, : len(r)] = np.asarray(r)
        result = (unit, m_dry, capped)
        self._exhaust_cache[key] = result
        return result

    def build(self, scenario: Scenario) -> StreamBank:
        ph = scenario.placeholders
        day = self._days[scenario.day]
        n_days = scenario.n_days
        clock = Clock.of(n_days)
        n_steps = len(clock.hour)
        rho = ph.normal_density_kg_per_nm3
        gas = ph.gas

        amb = _tile(day.ambient_c, n_days) + scenario.ambient_shift_k
        t_sup = _tile(day.supply_air_c, n_days)
        y_amb = intake_humidity(amb, ph.relative_humidity, gas)
        cp_moist = (gas.cp_dry_air_j_per_kg_k + y_amb * gas.cp_vapour_j_per_kg_k) / 1000.0

        def air_stream(flow_nm3h: FloatArray | float, factor: FloatArray | float) -> FloatArray:
            m_dry = np.asarray(flow_nm3h) * rho / 3600.0 * np.asarray(factor)
            out = np.zeros((n_steps, 1, 3))
            out[:, 0, 0] = t_sup
            out[:, 0, 1] = amb
            out[:, 0, 2] = m_dry * cp_moist
            return out

        flow_a = _tile(day.process_air_nm3h, n_days)
        line1 = line_c_profile(scenario.mode, clock, ph, 1)
        line2 = line_c_profile(scenario.mode, clock, ph, 2)
        flow_c = np.full(n_steps, ph.c_line_nm3_per_h)
        cold = {
            "air_a": air_stream(flow_a, 1.0),
            "air_c1": air_stream(flow_c, line1),
            "air_c2": air_stream(flow_c, line2),
        }
        # hall heating: heating degree x night factor; water: constant power over the window
        degree = np.clip(
            (ph.hall_limit_celsius - amb) / (ph.hall_limit_celsius - ph.hall_design_celsius),
            0.0,
            1.0,
        )
        hall_kw = scenario.hall_kw * degree * night_profile(clock, ph.hall_day_factor)
        cold["hall"] = _single(
            ph.hall_supply_celsius,
            ph.hall_return_celsius,
            hall_kw / (ph.hall_supply_celsius - ph.hall_return_celsius),
        )
        water_kw = scenario.water_kw * water_profile(scenario.mode, clock, ph)
        cold["water"] = _single(
            ph.water_supply_celsius,
            ph.water_return_celsius,
            water_kw / (ph.water_supply_celsius - ph.water_return_celsius),
        )

        span = ph.compressor_hot_in_celsius - ph.compressor_hot_out_celsius
        profile = compressor_profile(scenario.mode, clock, ph)
        comp_a_kw = scenario.compressor_a_kw * profile
        comp_c_kw = (
            ph.compressor_c_gwh_per_a * 1.0e6 / _HOURS_PER_YEAR * profile / float(np.mean(profile))
        )
        hot = {
            "compressor_a": _single(
                ph.compressor_hot_in_celsius, ph.compressor_hot_out_celsius, comp_a_kw / span
            ),
            "compressor_c": _single(
                ph.compressor_hot_in_celsius, ph.compressor_hot_out_celsius, comp_c_kw / span
            ),
        }
        unit, m_dry_a, capped = self.exhaust_unit(scenario.day, ph)
        unit_tiled = np.tile(unit, (n_days, 1, 1))
        # plant A exhaust: its own dry-air flow (process + atomising air)
        hot["exhaust_a"] = _scaled(unit_tiled, _tile(m_dry_a, n_days))
        # plant C lines: the line flow plus the atomising-air ratio of the measured day
        atom_ratio = _tile(day.atomising_nm3h / day.process_air_nm3h, n_days)
        m_dry_c = flow_c * (1.0 + atom_ratio) * rho / 3600.0
        flat_mean_kw = _mean_duty_kw(unit_tiled, m_dry_c, 2.0)
        scale = 1.0
        if ph.exhaust_c_basis == "annual":
            scale = ph.exhaust_c_gwh_per_a * 1.0e6 / _HOURS_PER_YEAR / flat_mean_kw
        hot["exhaust_c1"] = _scaled(unit_tiled, m_dry_c * line1 * scale)
        hot["exhaust_c2"] = _scaled(unit_tiled, m_dry_c * line2 * scale)
        notes = {
            "exhaust_capped_fraction": float(np.mean(capped)),
            "exhaust_c_flat_mean_kw_volume_basis": flat_mean_kw,
            "exhaust_c_scale": scale,
            "exhaust_a_mean_kw": _mean_duty_kw(unit_tiled, _tile(m_dry_a, n_days), 1.0),
        }
        return StreamBank(
            hot=hot, cold=cold, dt_h=24.0 / STEPS_PER_DAY, n_steps=n_steps, notes=notes
        )


def _scaled(unit: FloatArray, m_dry_kg_per_s: FloatArray) -> FloatArray:
    out = unit.copy()
    out[:, :, 2] = unit[:, :, 2] * m_dry_kg_per_s[:, None]
    return out


def _mean_duty_kw(unit: FloatArray, m_dry: FloatArray, lines: float) -> float:
    duty = unit[:, :, 2] * (unit[:, :, 0] - unit[:, :, 1])
    return float(np.mean(np.sum(duty, axis=1) * m_dry) * lines)

r"""Streams of the whole site for one scenario and one representative month (96 steps of 15 min).

Built on :class:`twinlink_ml.storage_timeslice.streams.StreamFactory` (Befund 99) -- the measured
days of plant A, the exhaust-humidity reconstruction, the condensation sampling of
:func:`twinlink_ml.heatnetwork.pinch.hot_segments_for_source`, plant C, compressor heat, hall
heating and process water are taken from there unchanged (mode ``flat``, horizon one day) -- plus
three additions of this study:

* **generic further lines**: ``n`` scaled copies of plant A on a chosen product day; exhaust and
  process air of that day times ``n * generic_flow / measured flow of plant A`` (one aggregated
  stream each; the problem table is linear in the capacity rates, so ``n`` identical lines and one
  line of ``n`` times the flow give the same targets);
* **steam-boiler flue gas** (variant): the fuel of the steam heater of plant A (its heating duty
  after today's recovery divided by ``gas_efficiency``), natural gas as methane (lower heating
  value 50 MJ/kg, the repository default of ``heatnetwork``; stoichiometric air 17.2 kg/kg; 2.25 kg
  water per kg), air ratio ``boiler_excess_air``; dry flue gas treated with the specific heats of
  air (``GasConfig``), the condensation sampling of ``heatnetwork.pinch`` from
  ``boiler_flue_celsius`` down to ``T_kalt``;
* **months**: the measured day is shifted so that its mean intake temperature equals the monthly
  mean of the placeholder climate (plus ``climate_shift_k``); the exhaust state is not shifted
  (Befund 99 convention).

Groups of today's recovery (:mod:`twinlink_ml.site.evaluate`): compressor -> process air of plant A
(capped at ``compressor_used_today_kw``); plant C's own exhaust and compressor heat -> its process
air, plus at most ``c_low_temperature_today_kw`` to hall heating and process water (the planned
design of plant C); generic lines: nothing, or (variant) their own exhaust -> own process air.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..heatnetwork.config import DEFAULT_FUEL_LOWER_HEATING_VALUE_J_PER_KG, HeatSourceConfig
from ..heatnetwork.pinch import hot_segments_for_source
from ..simulation.config import GasConfig
from ..storage_timeslice.config import Placeholders, Scenario
from ..storage_timeslice.plant import MeasuredDay, intake_humidity
from ..storage_timeslice.streams import StreamBank, StreamFactory
from .config import SiteConfig, SiteScenario

__all__ = [
    "AIR_PER_KG_METHANE",
    "WATER_PER_KG_METHANE",
    "SiteSnapshot",
    "SiteStreamBuilder",
    "flue_gas_state",
    "timeslice_placeholders",
]

FloatArray = NDArray[np.float64]

AIR_PER_KG_METHANE = 17.2
"""Stoichiometric dry air per kg of methane, kg/kg (2 O2 per CH4, 23.2 % O2 by mass in air)."""
WATER_PER_KG_METHANE = 2.25
"""Water formed per kg of methane, kg/kg (2 x 18 / 16)."""


def timeslice_placeholders(scenario: SiteScenario) -> Placeholders:
    """The placeholders of ``storage_timeslice`` that carry the site's values."""
    ph = scenario.placeholders
    return Placeholders(
        minimum_pinch_k=ph.minimum_pinch_k,
        cold_temperature_celsius=ph.cold_temperature_celsius,
        compressor_hot_in_celsius=ph.compressor_hot_in_celsius,
        compressor_hot_out_celsius=ph.compressor_hot_out_celsius,
        compressor_c_gwh_per_a=ph.compressor_c_gwh_per_a,
        exhaust_c_gwh_per_a=ph.exhaust_c_gwh_per_a,
        exhaust_c_basis="annual" if scenario.c_basis == "annual" else "volume",
        hall_supply_celsius=ph.hall_supply_celsius,
        hall_return_celsius=ph.hall_return_celsius,
        hall_limit_celsius=ph.hall_limit_celsius,
        hall_design_celsius=ph.hall_design_celsius,
        hall_day_factor=ph.hall_day_factor,
        water_supply_celsius=ph.water_supply_celsius,
        water_return_celsius=ph.water_return_celsius,
        c_line_nm3_per_h=ph.c_line_nm3_per_h,
        solid_fraction=ph.solid_fraction,
        relative_humidity=ph.relative_humidity,
        normal_density_kg_per_nm3=ph.normal_density_kg_per_nm3,
    )


def flue_gas_state(excess_air: float, intake_humidity_kg_per_kg: float) -> tuple[float, float]:
    """``(dry flue gas per kg fuel, absolute humidity of the flue gas)`` for methane."""
    air = AIR_PER_KG_METHANE * excess_air
    dry = 1.0 + air - WATER_PER_KG_METHANE
    return dry, (WATER_PER_KG_METHANE + air * intake_humidity_kg_per_kg) / dry


@dataclass(frozen=True, slots=True)
class SiteSnapshot:
    """All streams of one scenario-month: ``name -> (steps, segments, 3)``."""

    hot: Mapping[str, FloatArray]
    cold: Mapping[str, FloatArray]
    n_steps: int
    ambient_mean_celsius: float
    notes: Mapping[str, float] = field(default_factory=dict)

    def segments(self, names: tuple[str, ...], step: int, *, hot: bool) -> FloatArray:
        """Segments ``(n, 3)`` of the named streams at one step (missing names are skipped)."""
        bank = self.hot if hot else self.cold
        parts = [bank[n][step] for n in names if n in bank]
        if not parts:
            return np.zeros((0, 3))
        out = np.concatenate(parts, axis=0)
        return out[out[:, 2] > 0.0]


def _scaled(array: FloatArray, factor: float | FloatArray) -> FloatArray:
    out = np.array(array, dtype=np.float64, copy=True)
    out[:, :, 2] = out[:, :, 2] * (np.asarray(factor)[:, None] if np.ndim(factor) else factor)
    return out


class SiteStreamBuilder:
    """Builds :class:`SiteSnapshot` objects; the exhaust sampling is cached by the factory."""

    def __init__(self, config: SiteConfig, days: Mapping[str, MeasuredDay]) -> None:
        self._config = config
        self._days = dict(days)
        self._factory = StreamFactory(days)
        self._flue_cache: dict[tuple[float, ...], FloatArray] = {}

    @property
    def days(self) -> Mapping[str, MeasuredDay]:
        return self._days

    def month_shift_k(self, day_label: str, month: int, climate_shift_k: float) -> float:
        """Shift of the measured day's intake temperature to the mean of ``month`` (0-11)."""
        target = self._config.climate_monthly_mean_celsius[month] + climate_shift_k
        return target - float(np.mean(self._days[day_label].ambient_c))

    def _bank(
        self, scenario: SiteScenario, day: str, month: int, compressor_kw: float
    ) -> StreamBank:
        ph = scenario.placeholders
        ts = Scenario(
            site="A+C",
            day=day,
            horizon="day",
            mode="flat",
            ambient_shift_k=self.month_shift_k(day, month, ph.climate_shift_k),
            compressor_a_kw=compressor_kw,
            hall_kw=ph.hall_kw,
            water_kw=ph.water_kw,
            placeholders=timeslice_placeholders(scenario),
        )
        return self._factory.build(ts)

    def build(self, scenario: SiteScenario, month: int) -> SiteSnapshot:
        ph = scenario.placeholders
        bank = self._bank(scenario, scenario.day_a, month, ph.compressor_station_kw)
        hot: dict[str, FloatArray] = {"exhaust_a": bank.hot["exhaust_a"]}
        cold: dict[str, FloatArray] = {"air_a": bank.cold["air_a"]}
        if ph.compressor_station_kw > 0.0:
            hot["compressor_a"] = bank.hot["compressor_a"]
        if scenario.has_c:
            for name in ("compressor_c", "exhaust_c1", "exhaust_c2"):
                hot[name] = bank.hot[name]
            for name in ("air_c1", "air_c2"):
                cold[name] = bank.cold[name]
        if ph.hall_kw > 0.0:
            cold["hall"] = bank.cold["hall"]
        if ph.water_kw > 0.0:
            cold["water"] = bank.cold["water"]
        notes = {f"timeslice_{k}": float(v) for k, v in bank.notes.items()}
        if scenario.n_generic > 0:
            gbank = (
                bank
                if scenario.generic_day == scenario.day_a
                else self._bank(scenario, scenario.generic_day, month, 0.0)
            )
            measured_flow = float(np.mean(self._days[scenario.generic_day].process_air_nm3h))
            factor = scenario.n_generic * ph.generic_flow_nm3_per_h / measured_flow
            hot["exhaust_generic"] = _scaled(gbank.hot["exhaust_a"], factor)
            cold["air_generic"] = _scaled(gbank.cold["air_a"], factor)
            notes["generic_flow_factor"] = factor
        ambient = float(np.mean(self._days[scenario.day_a].ambient_c)) + self.month_shift_k(
            scenario.day_a, month, ph.climate_shift_k
        )
        return SiteSnapshot(
            hot=hot, cold=cold, n_steps=bank.n_steps, ambient_mean_celsius=ambient, notes=notes
        )

    def flue_unit(self, scenario: SiteScenario) -> FloatArray:
        """Flue-gas segments per kW of fuel (LHV), ``(segments, 3)``."""
        ph = scenario.placeholders
        key = (
            ph.boiler_excess_air,
            ph.boiler_flue_celsius,
            ph.cold_temperature_celsius,
            ph.relative_humidity,
        )
        if key not in self._flue_cache:
            gas = GasConfig()
            y_air = float(intake_humidity(np.array([10.0]), ph.relative_humidity, gas)[0])
            dry, y_flue = flue_gas_state(ph.boiler_excess_air, y_air)
            fuel_kg_per_s = 1000.0 / DEFAULT_FUEL_LOWER_HEATING_VALUE_J_PER_KG
            source = HeatSourceConfig(
                "boiler_flue", dry * fuel_kg_per_s, ph.boiler_flue_celsius, y_flue
            )
            segments = hot_segments_for_source(source, gas, ph.cold_temperature_celsius)
            self._flue_cache[key] = np.array(
                [
                    (
                        s.high_temperature_celsius,
                        s.low_temperature_celsius,
                        s.capacity_rate_kw_per_k,
                    )
                    for s in segments
                ]
            )
        return self._flue_cache[key]

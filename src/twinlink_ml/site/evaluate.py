r"""Site pinch of one scenario over a year of 12 representative days (15-minute steps).

Per step (no storage: heat is usable only when it occurs, as ``E_direct`` of Befund 99):

* **target**: minimum hot utility of ALL site streams (:func:`.curves.problem_table`), its pinch
  and the stream that sets it;
* **today**: recovery only inside the groups that exist today -- compressor -> process air of
  plant A (capped at ``compressor_used_today_kw``); plant C's own exhaust and compressor heat ->
  its process air plus at most ``c_low_temperature_today_kw`` to hall heating and process water;
  generic lines nothing (``generic_today="none"``) or their own exhaust -> own air (``"own"``);
  every other demand is fossil;
* **network levels, composite loop** (diagnostic, :func:`.curves.network_capacity`, approach
  ``dT_min/2`` per exchanger): the duty a single loop at ``(supply, return)`` could carry if
  every transfer went through it, with the source side and the sink side reported separately
  (which side limits the level); it replaces today's direct recovery inside plant C, too;
* **network levels, network + in-plant direct** (the study's network model,
  :func:`.curves.zonal_network_target`): heat moves directly (``dT_min``) inside each zone and
  between zones only through the loop, each zone on its own branch (parallel), return cooler
  allowed. Zones
  (:func:`zones_of`): compressor station (its heat reaches every consumer only through the network,
  as today through storage A1), plant A (own exhaust and process air), plant C (own exhaust and
  compressor heat, process air, hall heating and process water -- its planned design), the generic
  lines (own exhaust and air), hall heating and process water without plant C, the boiler flue gas;
* **heat pump limits**: evaporator/condenser limits from the grand composite curve
  (:func:`.curves.heat_pump_limits`) on a grid of temperatures, combined into the best pair per
  electricity CO2 factor.

Year: month ``m`` weighs ``days_m * 24`` h; plant C runs continuously in the run with C; another
operating fraction is combined exactly from the runs with and without plant C (per-step results
are additive in time). Energy balance per step: ``Q_H - Q_C = demand - supply`` (recorded).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import SiteConfig, SiteScenario
from .curves import (
    Ramps,
    TableResult,
    heat_pump_limits,
    multi_loop_capacity,
    network_capacity,
    problem_table,
    zonal_network_target,
)
from .streams import SiteSnapshot, SiteStreamBuilder

__all__ = [
    "COLD_A",
    "COLD_C_AIR",
    "COLD_C_LOW",
    "DEFAULT_CONDENSER_CELSIUS",
    "DEFAULT_EVAPORATOR_CELSIUS",
    "DEFAULT_NETWORK_LEVELS",
    "DEFAULT_TWO_LEVEL_PAIRS",
    "DEFAULT_ZONAL_LEVELS",
    "HOT_A",
    "HOT_C",
    "SiteYear",
    "annual_summary",
    "evaluate_year",
    "heat_pump_cop",
    "pinch_label",
    "zones_of",
]

FloatArray = NDArray[np.float64]

HOT_A = ("compressor_a",)
COLD_A = ("air_a",)
HOT_C = ("compressor_c", "exhaust_c1", "exhaust_c2")
COLD_C_AIR = ("air_c1", "air_c2")
COLD_C_LOW = ("hall", "water")
HOT_GENERIC = ("exhaust_generic",)
COLD_GENERIC = ("air_generic",)

DEFAULT_NETWORK_LEVELS: tuple[tuple[float, float], ...] = tuple(
    (float(s), float(s - d)) for s in range(30, 95, 5) for d in (10, 20, 30) if s - d >= 20
)
"""Levels ``(supply, return)`` in degC: supply 30-90 degC, spread 10/20/30 K, return >= 20."""

DEFAULT_ZONAL_LEVELS: tuple[tuple[float, float], ...] = (
    (40.0, 20.0),
    (45.0, 25.0),
    (50.0, 30.0),
    (55.0, 35.0),
    (60.0, 30.0),
    (60.0, 40.0),
    (65.0, 35.0),
    (70.0, 40.0),
    (70.0, 50.0),
    (75.0, 45.0),
    (80.0, 50.0),
    (80.0, 60.0),
    (90.0, 60.0),
)
"""Levels of the variant "network + in-plant direct" (one LP per step and level, so fewer)."""

DEFAULT_TWO_LEVEL_PAIRS: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = tuple(
    (low, high)
    for low in ((40.0, 30.0), (45.0, 30.0), (50.0, 30.0), (55.0, 35.0), (60.0, 40.0))
    for high in ((70.0, 50.0), (75.0, 55.0), (80.0, 60.0), (85.0, 60.0), (90.0, 60.0))
)
"""Low-temperature x high-temperature loop pairs of the two-level check (reference scenarios)."""

DEFAULT_EVAPORATOR_CELSIUS: tuple[float, ...] = tuple(float(t) for t in range(10, 65, 5))
DEFAULT_CONDENSER_CELSIUS: tuple[float, ...] = tuple(float(t) for t in range(40, 105, 5))


def heat_pump_cop(
    evaporator_celsius: FloatArray, condenser_celsius: FloatArray, eta: float
) -> FloatArray:
    """Heating COP ``eta T_c / (T_c - T_e)`` (Kelvin); ``nan`` where ``T_c <= T_e`` or COP <= 1."""
    lift = condenser_celsius - evaporator_celsius
    cop = np.where(
        lift > 0.0, eta * (condenser_celsius + 273.15) / np.where(lift > 0, lift, 1.0), np.nan
    )
    return np.where(cop > 1.0, cop, np.nan)


@dataclass(frozen=True, slots=True)
class _Labelled:
    segments: FloatArray
    names: tuple[str, ...]
    top: NDArray[np.bool_]
    bottom: NDArray[np.bool_]


def _labelled(snapshot: SiteSnapshot, step: int, *, hot: bool) -> _Labelled:
    bank = snapshot.hot if hot else snapshot.cold
    segs: list[FloatArray] = []
    names: list[str] = []
    tops: list[NDArray[np.bool_]] = []
    bottoms: list[NDArray[np.bool_]] = []
    for name, array in bank.items():
        seg = array[step]
        seg = seg[seg[:, 2] > 0.0]
        if seg.size == 0:
            continue
        segs.append(seg)
        names.extend([name] * len(seg))
        tops.append(seg[:, 0] >= seg[:, 0].max() - 1e-9)
        bottoms.append(seg[:, 1] <= seg[:, 1].min() + 1e-9)
    if not segs:
        empty = np.zeros(0, dtype=bool)
        return _Labelled(np.zeros((0, 3)), (), empty, empty)
    return _Labelled(
        np.concatenate(segs), tuple(names), np.concatenate(tops), np.concatenate(bottoms)
    )


def pinch_label(table: TableResult, hot: _Labelled, cold: _Labelled, minimum_pinch_k: float) -> str:
    """Stream boundary at the pinch, e.g. ``exhaust_a:inlet`` (several joined by ``+``)."""
    if table.threshold or not np.isfinite(table.pinch_shifted):
        return "threshold"
    half = minimum_pinch_k / 2.0
    s = table.pinch_shifted
    tol = 1e-6
    labels: set[str] = set()
    for i, name in enumerate(hot.names):
        high, low = hot.segments[i, 0] - half, hot.segments[i, 1] - half
        if abs(high - s) < tol:
            labels.add(f"{name}:{'inlet' if hot.top[i] else 'internal'}")
        if abs(low - s) < tol:
            labels.add(f"{name}:{'outlet' if hot.bottom[i] else 'internal'}")
    for i, name in enumerate(cold.names):
        high, low = cold.segments[i, 0] + half, cold.segments[i, 1] + half
        if abs(high - s) < tol:
            labels.add(f"{name}:{'target' if cold.top[i] else 'internal'}")
        if abs(low - s) < tol:
            labels.add(f"{name}:{'supply' if cold.bottom[i] else 'internal'}")
    internal_only = {lab for lab in labels if not lab.endswith(":internal")}
    return "+".join(sorted(internal_only or labels)) or "unlabelled"


def zones_of(snapshot: SiteSnapshot) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Zones of the variant "network + in-plant direct" that have streams in ``snapshot``."""
    has_c = "exhaust_c1" in snapshot.hot
    candidates: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
        (HOT_A, ()),
        (("exhaust_a",), COLD_A),
        (HOT_C, COLD_C_AIR + COLD_C_LOW) if has_c else ((), COLD_C_LOW),
        (HOT_GENERIC, COLD_GENERIC),
        (("boiler_flue",), ()),
    ]
    present = set(snapshot.hot) | set(snapshot.cold)
    zones = []
    for hot, cold in candidates:
        hot_in = tuple(n for n in hot if n in present)
        cold_in = tuple(n for n in cold if n in present)
        if hot_in or cold_in:
            zones.append((hot_in, cold_in))
    return zones


def _target(
    snapshot: SiteSnapshot, step: int, hot: Sequence[str], cold: Sequence[str], dt: float
) -> tuple[float, float]:
    """``(recovered, demand)`` of a group alone at one step."""
    h = Ramps.of_segments(snapshot.segments(tuple(hot), step, hot=True))
    c = Ramps.of_segments(snapshot.segments(tuple(cold), step, hot=False))
    demand = c.total
    if demand <= 0.0 or h.total <= 0.0:
        return 0.0, demand
    return demand - problem_table(h, c, dt).hot_utility_kw, demand


@dataclass
class SiteYear:
    """Per-step results of one scenario over the 12 representative days (arrays ``(12, 96)``)."""

    scenario: SiteScenario
    hours_per_step: FloatArray  # (12,) hours represented by one step of month m
    series: dict[str, FloatArray]
    network_src_kw: FloatArray  # (12, 96, L)
    network_sink_kw: FloatArray
    hp_evaporator_kw: FloatArray  # (12, 96, n_e)
    hp_condenser_kw: FloatArray  # (12, 96, n_c)
    pinch_labels: Counter[str]
    pinch_labels_weighted: dict[str, float]
    network_levels: tuple[tuple[float, float], ...]
    zonal_hot_utility_kw: FloatArray  # (12, 96, Lz)
    zonal_levels: tuple[tuple[float, float], ...]
    evaporator_celsius: tuple[float, ...]
    condenser_celsius: tuple[float, ...]
    detail: dict[str, Any] = field(default_factory=dict)

    def annual_mean(self, values: FloatArray) -> FloatArray:
        """Time-weighted annual mean over the (12, 96, ...) leading axes (kW)."""
        w = self.hours_per_step[:, None] * np.ones(values.shape[:2])
        w = w / w.sum()
        return np.asarray(np.tensordot(w, values, axes=([0, 1], [0, 1])), dtype=np.float64)


def evaluate_year(
    builder: SiteStreamBuilder,
    config: SiteConfig,
    scenario: SiteScenario,
    *,
    network_levels: Sequence[tuple[float, float]] = DEFAULT_NETWORK_LEVELS,
    evaporator_celsius: Sequence[float] = DEFAULT_EVAPORATOR_CELSIUS,
    condenser_celsius: Sequence[float] = DEFAULT_CONDENSER_CELSIUS,
    zonal_levels: Sequence[tuple[float, float]] = DEFAULT_ZONAL_LEVELS,
    detail: bool = False,
    two_level_pairs: Sequence[tuple[tuple[float, float], tuple[float, float]]] = (),
) -> SiteYear:
    """Evaluate one scenario (module docstring). ``detail`` adds the reference-scenario outputs."""
    ph = scenario.placeholders
    dt = ph.minimum_pinch_k
    a = ph.network_approach_k
    t_s = np.array([s for s, _ in network_levels])
    t_r = np.array([r for _, r in network_levels])
    t_e = np.asarray(evaporator_celsius, dtype=np.float64)
    t_c = np.asarray(condenser_celsius, dtype=np.float64)
    n_months, n_steps = 12, 96
    keys = (
        "demand_kw",
        "supply_kw",
        "hot_utility_kw",
        "cold_utility_kw",
        "pinch_hot_celsius",
        "threshold",
        "rec_today_a_kw",
        "rec_today_c_kw",
        "rec_today_generic_kw",
        "demand_air_a_kw",
        "demand_air_c_kw",
        "demand_air_generic_kw",
        "demand_low_kw",
        "boiler_fuel_kw",
        "balance_residual_kw",
        "ambient_celsius",
    )
    series = {k: np.zeros((n_months, n_steps)) for k in keys}
    net_src = np.zeros((n_months, n_steps, len(t_s)))
    net_sink = np.zeros_like(net_src)
    hp_e = np.zeros((n_months, n_steps, len(t_e)))
    hp_c = np.zeros((n_months, n_steps, len(t_c)))
    zonal = np.zeros((n_months, n_steps, len(zonal_levels)))
    labels: Counter[str] = Counter()
    labels_w: dict[str, float] = {}
    hours = np.array([d * 24.0 / n_steps for d in config.climate_days])
    tam_parts_hot: list[tuple[Ramps, float]] = []
    tam_parts_cold: list[tuple[Ramps, float]] = []
    stream_names: set[str] = set()
    per_stream: dict[str, dict[str, float]] = {}
    pair_rec: dict[str, float] = {}
    two_level: dict[str, list[float]] = {}
    potentials_src: dict[str, FloatArray] = {}
    potentials_sink: dict[str, FloatArray] = {}
    total_hours = float(hours.sum() * n_steps)
    flue_unit = builder.flue_unit(scenario) if scenario.boiler else None

    for m in range(n_months):
        snap = builder.build(scenario, m)
        fuel = np.zeros(n_steps)
        if flue_unit is not None:
            snap, fuel = _with_boiler(snap, scenario, flue_unit, dt)
        stream_names |= set(snap.hot) | set(snap.cold)
        zone_names = zones_of(snap)
        for i in range(n_steps):
            hl = _labelled(snap, i, hot=True)
            cl = _labelled(snap, i, hot=False)
            hot = Ramps.of_segments(hl.segments)
            cold = Ramps.of_segments(cl.segments)
            table = problem_table(hot, cold, dt)
            s = series
            s["demand_kw"][m, i] = cold.total
            s["supply_kw"][m, i] = hot.total
            s["hot_utility_kw"][m, i] = table.hot_utility_kw
            s["cold_utility_kw"][m, i] = table.cold_utility_kw
            s["pinch_hot_celsius"][m, i] = table.pinch_shifted + dt / 2.0
            s["threshold"][m, i] = float(table.threshold)
            s["balance_residual_kw"][m, i] = (
                table.hot_utility_kw - table.cold_utility_kw - (cold.total - hot.total)
            )
            s["ambient_celsius"][m, i] = snap.ambient_mean_celsius
            label = pinch_label(table, hl, cl, dt)
            labels[label] += 1
            labels_w[label] = labels_w.get(label, 0.0) + hours[m] / total_hours

            rec_a, dem_a = _target(snap, i, HOT_A, COLD_A, dt)
            s["rec_today_a_kw"][m, i] = min(rec_a, ph.compressor_used_today_kw)
            s["demand_air_a_kw"][m, i] = dem_a
            if scenario.has_c:
                rec_air, dem_c = _target(snap, i, HOT_C, COLD_C_AIR, dt)
                rec_all, _ = _target(snap, i, HOT_C, COLD_C_AIR + COLD_C_LOW, dt)
                s["rec_today_c_kw"][m, i] = min(rec_all, rec_air + ph.c_low_temperature_today_kw)
                s["demand_air_c_kw"][m, i] = dem_c
            s["demand_low_kw"][m, i] = Ramps.of_segments(
                snap.segments(COLD_C_LOW, i, hot=False)
            ).total
            if scenario.n_generic:
                rec_g, dem_g = _target(snap, i, HOT_GENERIC, COLD_GENERIC, dt)
                s["demand_air_generic_kw"][m, i] = dem_g
                if scenario.generic_today == "own":
                    s["rec_today_generic_kw"][m, i] = rec_g
            s["boiler_fuel_kw"][m, i] = fuel[i]

            q_src, q_sink = network_capacity(hot, cold, t_s, t_r, a)
            net_src[m, i] = q_src
            net_sink[m, i] = q_sink
            e_lim, u_lim = heat_pump_limits(table, t_e, t_c, dt)
            hp_e[m, i] = e_lim
            hp_c[m, i] = u_lim
            if zonal_levels:
                zones = [
                    (
                        Ramps.of_segments(snap.segments(zh, i, hot=True)),
                        Ramps.of_segments(snap.segments(zc, i, hot=False)),
                    )
                    for zh, zc in zone_names
                ]
                for j, (sup, ret) in enumerate(zonal_levels):
                    zonal[m, i, j] = zonal_network_target(zones, sup, ret, dt)[0]

            if detail:
                w = hours[m] / total_hours
                tam_parts_hot.append((hot, w))
                tam_parts_cold.append((cold, w))
                _accumulate_detail(
                    snap,
                    i,
                    table,
                    dt,
                    a,
                    w,
                    network_levels,
                    per_stream,
                    potentials_src,
                    potentials_sink,
                    pair_rec,
                )
                if two_level_pairs:
                    for low, high in two_level_pairs:
                        duties = multi_loop_capacity(hot, cold, (low, high), a)
                        key = f"{low[0]:g}/{low[1]:g}+{high[0]:g}/{high[1]:g}"
                        acc = two_level.setdefault(key, [0.0, 0.0])
                        acc[0] += w * float(duties[0])
                        acc[1] += w * float(duties[1])

    year = SiteYear(
        scenario=scenario,
        hours_per_step=hours,
        series=series,
        network_src_kw=net_src,
        network_sink_kw=net_sink,
        hp_evaporator_kw=hp_e,
        hp_condenser_kw=hp_c,
        pinch_labels=labels,
        pinch_labels_weighted=labels_w,
        network_levels=tuple(network_levels),
        zonal_hot_utility_kw=zonal,
        zonal_levels=tuple(zonal_levels),
        evaporator_celsius=tuple(float(t) for t in t_e),
        condenser_celsius=tuple(float(t) for t in t_c),
    )
    if detail:
        year.detail = _finish_detail(
            config,
            scenario,
            tam_parts_hot,
            tam_parts_cold,
            per_stream,
            potentials_src,
            potentials_sink,
            pair_rec,
            two_level,
            network_levels,
            dt,
        )
    return year


def _with_boiler(
    snap: SiteSnapshot, scenario: SiteScenario, unit: FloatArray, dt: float
) -> tuple[SiteSnapshot, FloatArray]:
    """Add the flue gas of plant A's steam heater (fuel = duty after today's recovery / eta)."""
    ph = scenario.placeholders
    fuel = np.zeros(snap.n_steps)
    for i in range(snap.n_steps):
        rec_a, dem_a = _target(snap, i, HOT_A, COLD_A, dt)
        fuel[i] = (dem_a - min(rec_a, ph.compressor_used_today_kw)) / ph.gas_efficiency
    flue = np.repeat(unit[None, :, :], snap.n_steps, axis=0).copy()
    flue[:, :, 2] = unit[None, :, 2] * fuel[:, None]
    snapshot = SiteSnapshot(
        hot={**snap.hot, "boiler_flue": flue},
        cold=snap.cold,
        n_steps=snap.n_steps,
        ambient_mean_celsius=snap.ambient_mean_celsius,
        notes=snap.notes,
    )
    return snapshot, fuel


def _accumulate_detail(
    snap: SiteSnapshot,
    step: int,
    table: TableResult,
    dt: float,
    a: float,
    weight: float,
    network_levels: Sequence[tuple[float, float]],
    per_stream: dict[str, dict[str, float]],
    potentials_src: dict[str, FloatArray],
    potentials_sink: dict[str, FloatArray],
    pair_rec: dict[str, float],
) -> None:
    """Per-stream heat above/below the pinch, network potentials, single source x sink pairs."""
    half = dt / 2.0
    t_hot = table.pinch_shifted + half
    t_cold = table.pinch_shifted - half
    t_s = np.array([s for s, _ in network_levels])
    t_r = np.array([r for _, r in network_levels])
    for name in snap.hot:
        r = Ramps.of_segments(snap.segments((name,), step, hot=True))
        entry = per_stream.setdefault(
            name, {"kind_hot": 1.0, "total_kw": 0.0, "above_pinch_kw": 0.0}
        )
        entry["total_kw"] += weight * r.total
        if not table.threshold:
            entry["above_pinch_kw"] += weight * float(r(t_hot))
        pot = r(t_r + a)  # heat this source alone could put into the loop (above its return + a)
        potentials_src[name] = potentials_src.get(name, np.zeros(len(t_s))) + weight * pot
    for name in snap.cold:
        r = Ramps.of_segments(snap.segments((name,), step, hot=False))
        entry = per_stream.setdefault(
            name, {"kind_hot": 0.0, "total_kw": 0.0, "above_pinch_kw": 0.0}
        )
        entry["total_kw"] += weight * r.total
        if not table.threshold:
            entry["above_pinch_kw"] += weight * float(r(t_cold))
        pot = r.below(t_s - a)  # demand below the loop supply - a
        potentials_sink[name] = potentials_sink.get(name, np.zeros(len(t_s))) + weight * pot
    for source in snap.hot:
        for sink in snap.cold:
            rec, _ = _target(snap, step, (source,), (sink,), dt)
            key = f"{source}->{sink}"
            pair_rec[key] = pair_rec.get(key, 0.0) + weight * rec


def _finish_detail(
    config: SiteConfig,
    scenario: SiteScenario,
    hot_parts: list[tuple[Ramps, float]],
    cold_parts: list[tuple[Ramps, float]],
    per_stream: dict[str, dict[str, float]],
    potentials_src: dict[str, FloatArray],
    potentials_sink: dict[str, FloatArray],
    pair_rec: dict[str, float],
    two_level: dict[str, list[float]],
    network_levels: Sequence[tuple[float, float]],
    dt: float,
) -> dict[str, Any]:
    """Time-averaged (TAM) composite curves and grand composite curve, and the detail tables."""
    del config, scenario
    hot = Ramps.combine(hot_parts)
    cold = Ramps.combine(cold_parts)
    tam = problem_table(hot, cold, dt)
    grid = np.unique(np.round(np.concatenate([hot.points, cold.points]), 3))
    plot_t = np.linspace(float(grid.min()), float(grid.max()), 400) if grid.size else grid
    shifted_plot = np.linspace(float(tam.shifted.min()), float(tam.shifted.max()), 400)
    gcc_plot = np.interp(shifted_plot, tam.shifted[::-1], tam.gcc_kw[::-1])
    levels = [f"{s:g}/{r:g}" for s, r in network_levels]
    return {
        "tam_hot_utility_kw": tam.hot_utility_kw,
        "tam_cold_utility_kw": tam.cold_utility_kw,
        "tam_pinch_hot_celsius": tam.pinch_shifted + dt / 2.0,
        "tam_threshold": tam.threshold,
        "composite_temperature_celsius": plot_t,
        "composite_hot_below_kw": hot.below(plot_t),
        "composite_cold_below_kw": cold.below(plot_t),
        "gcc_shifted_celsius": shifted_plot,
        "gcc_kw": gcc_plot,
        "per_stream": per_stream,
        "network_potential_source_kw": {
            k: dict(zip(levels, v.tolist(), strict=True)) for k, v in potentials_src.items()
        },
        "network_potential_sink_kw": {
            k: dict(zip(levels, v.tolist(), strict=True)) for k, v in potentials_sink.items()
        },
        "pair_recovery_kw": pair_rec,
        "two_level_kw": {
            k: {"low_kw": v[0], "high_kw": v[1], "sum_kw": v[0] + v[1]}
            for k, v in two_level.items()
        },
    }


def annual_summary(year: SiteYear, config: SiteConfig) -> dict[str, Any]:
    """Annual means (kW), MWh/a, CO2 and the network and heat-pump summaries of one scenario."""
    ph = year.scenario.placeholders
    s = year.series
    mean = {k: float(year.annual_mean(v)) for k, v in s.items() if k not in ("threshold",)}
    mean["threshold_share"] = float(year.annual_mean(s["threshold"]))
    rec_today = mean["rec_today_a_kw"] + mean["rec_today_c_kw"] + mean["rec_today_generic_kw"]
    fossil_today = mean["demand_kw"] - rec_today
    fossil_target = mean["hot_utility_kw"]
    saving = fossil_today - fossil_target
    to_mwh = 8760.0 / 1000.0
    co2_gas = ph.gas_co2_kg_per_kwh / ph.gas_efficiency
    out: dict[str, Any] = {
        "label": year.scenario.label,
        "fingerprint": year.scenario.fingerprint,
        "mean_kw": mean,
        "rec_today_kw": rec_today,
        "fossil_today_kw": fossil_today,
        "fossil_target_kw": fossil_target,
        "rec_target_kw": mean["demand_kw"] - fossil_target,
        "saving_kw": saving,
        "saving_mwh_per_a": saving * to_mwh,
        "saving_fraction_of_today": saving / fossil_today if fossil_today > 0 else float("nan"),
        "saving_co2_t_per_a": saving * to_mwh * co2_gas,
        "pump_electric_mwh_per_a": max(saving, 0.0) * ph.network_pump_fraction * to_mwh,
        "pinch_hot_celsius_median": float(np.median(s["pinch_hot_celsius"][s["threshold"] < 0.5]))
        if np.any(s["threshold"] < 0.5)
        else float("nan"),
        "pinch_labels": dict(year.pinch_labels.most_common(6)),
        "pinch_labels_weighted": dict(
            sorted(year.pinch_labels_weighted.items(), key=lambda kv: -kv[1])[:6]
        ),
        "balance_residual_max_kw": float(np.max(np.abs(s["balance_residual_kw"]))),
    }
    out["saving_co2_net_t_per_a"] = {
        f"el_{f:g}": out["saving_co2_t_per_a"] - out["pump_electric_mwh_per_a"] * f
        for f in _electricity_factors(config, ph.electricity_co2_kg_per_kwh)
    }
    # network levels
    q_n = np.minimum(year.network_src_kw, year.network_sink_kw)
    q_mean = year.annual_mean(q_n)
    src_mean = year.annual_mean(year.network_src_kw)
    sink_mean = year.annual_mean(year.network_sink_kw)
    levels = [f"{a:g}/{b:g}" for a, b in year.network_levels]
    network = {}
    for j, key in enumerate(levels):
        fossil_n = mean["demand_kw"] - float(q_mean[j])
        network[key] = {
            "q_kw": float(q_mean[j]),
            "q_source_side_kw": float(src_mean[j]),
            "q_sink_side_kw": float(sink_mean[j]),
            "fossil_kw": fossil_n,
            "fraction_of_target_saving": (fossil_today - fossil_n) / saving
            if saving > 1e-6
            else float("nan"),
        }
    out["loop_composite"] = network
    best = max(network, key=lambda k: network[k]["q_kw"])
    out["loop_composite_best_level"] = best
    out["loop_composite_best_fraction"] = network[best]["fraction_of_target_saving"]
    zonal_mean = year.annual_mean(year.zonal_hot_utility_kw)
    zonal = {}
    for j, (sup, ret) in enumerate(year.zonal_levels):
        fossil_z = float(zonal_mean[j])
        zonal[f"{sup:g}/{ret:g}"] = {
            "fossil_kw": fossil_z,
            "saving_vs_today_kw": fossil_today - fossil_z,
            "fraction_of_target_saving": (fossil_today - fossil_z) / saving
            if saving > 1e-6
            else float("nan"),
        }
    out["network"] = zonal
    if zonal:
        best_z = min(zonal, key=lambda k: zonal[k]["fossil_kw"])
        out["network_best_level"] = best_z
        out["network_best_fraction"] = zonal[best_z]["fraction_of_target_saving"]
    # heat pump (on top of the target)
    out["heat_pump"] = _heat_pump_summary(year, config)
    return out


def _electricity_factors(config: SiteConfig, base: float) -> tuple[float, ...]:
    low, _ = config.band("electricity_co2_kg_per_kwh")
    return tuple(sorted({base, low}, reverse=True))


def _heat_pump_summary(year: SiteYear, config: SiteConfig) -> dict[str, Any]:
    ph = year.scenario.placeholders
    t_e = np.asarray(year.evaporator_celsius)
    t_c = np.asarray(year.condenser_celsius)
    cop = heat_pump_cop(t_e[:, None], t_c[None, :], ph.heat_pump_carnot_efficiency)  # (n_e, n_c)
    valid = np.isfinite(cop)
    lift_factor = np.where(valid, 1.0 / (1.0 - 1.0 / np.where(valid, cop, 2.0)), 0.0)
    # per step: Q_cond = min(U_c, E_e * COP/(COP-1))
    e = year.hp_evaporator_kw[:, :, :, None] * lift_factor[None, None, :, :]
    u = year.hp_condenser_kw[:, :, None, :]
    q_cond = np.where(valid[None, None], np.minimum(e, u), 0.0)
    q_mean = year.annual_mean(q_cond)  # (n_e, n_c)
    q_peak = np.max(q_cond, axis=(0, 1))
    w_mean = np.where(valid, q_mean / np.where(valid, cop, 1.0), 0.0)
    result: dict[str, Any] = {}
    for f in _electricity_factors(config, ph.electricity_co2_kg_per_kwh):
        co2 = (q_mean * ph.gas_co2_kg_per_kwh / ph.gas_efficiency - w_mean * f) * 8.76  # t/a
        co2 = np.where(valid, co2, -np.inf)
        j_e, j_c = np.unravel_index(int(np.argmax(co2)), co2.shape)
        result[f"el_{f:g}"] = {
            "evaporator_celsius": float(t_e[j_e]),
            "condenser_celsius": float(t_c[j_c]),
            "cop": float(cop[j_e, j_c]),
            "q_condenser_mean_kw": float(q_mean[j_e, j_c]),
            "q_condenser_peak_kw": float(q_peak[j_e, j_c]),
            "electric_mean_kw": float(w_mean[j_e, j_c]),
            "co2_saving_t_per_a": float(co2[j_e, j_c]),
        }
    result["evaporator_limit_mean_kw"] = dict(
        zip([f"{t:g}" for t in t_e], year.annual_mean(year.hp_evaporator_kw).tolist(), strict=True)
    )
    result["condenser_limit_mean_kw"] = dict(
        zip([f"{t:g}" for t in t_c], year.annual_mean(year.hp_condenser_kw).tolist(), strict=True)
    )
    return result

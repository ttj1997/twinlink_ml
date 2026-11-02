r"""Problem table algorithm: the thermodynamic minimum of supplementary heat.

Answers a different question than ``balance.evaluate_cascade``: not "how much fuel
does THIS interconnection need", but "how much fuel does EVERY CONCEIVABLE
interconnection need at least" -- the yardstick against which ``evaluate_cascade`` results (and
every later controller) are measured. For the pure target problem (no network
design, only the minimum heating/cooling demand) the problem table algorithm is exact and
needs no solver -- an LP would only be needed for the question of which heat exchanger
goes where (network synthesis), not part of this module.

Condensation: no new physics
--------------------------------
The exhaust air sources have NO constant heat capacity -- below their dew point
latent heat is released, and the enthalpy-temperature curve kinks sharply there. A stream with
constant cp would be wrong here. This module introduces NO second condensation formula
for it: :func:`hot_segments_for_source` samples the curve with the existing, already tested
:func:`~.balance.evaluate_cascade` (``recovery_mode="condensing"``) -- the same code that
also drives the cascade calculation itself -- and builds a piecewise linear
approximation from it. Above the dew point the curve is exactly linear (humidity constant, no
discretisation error possible, one segment suffices); below it, it is sampled at
``n_segments_below_dew_point`` nodes, with the dew point itself as an exact
interval boundary (not left to the chance of a grid). How fine this sampling has to
be is an empirical question -- see ``docs/06_waermenetz.md`` for the
convergence study.

Problem table algorithm (standard procedure, only the short version here; textbook e.g.
Kemp, "Pinch Analysis and Process Integration")
------------------------------------------------------------------------------------------
1. Shifted temperatures: hot segments by ``dT_min/2`` downwards, cold ones by
   ``dT_min/2`` upwards (:meth:`StreamSegment.shifted`) -- turns the minimum approach
   temperature into a single common temperature scale on which hot and cold may
   touch.
2. All shifted segment boundaries (hot and cold) form the interval boundaries.
3. Per interval: heat balance = (sum of CP hot active - sum of CP cold active) * Delta T.
   Cascade from the highest temperature downwards (start value 0). The negative cascade
   value with the largest magnitude is the minimum heating demand (hot utility); adding
   it to the whole cascade makes it non-negative everywhere, its end value is the minimum
   cooling demand (cold utility), and the point at which it becomes exactly zero is the
   pinch temperature.

The lower bound, not a design rule
------------------------------------------
The heating demand computed here assumes arbitrarily many, ideally matched
heat exchangers. It is a LOWER BOUND for every real interconnection with
THE SAME approach temperature, not a statement about whether it is achievable with two exchangers.
``tests/test_heatnetwork_pinch.py::test_pinch_target_never_exceeds_the_best_of_six_cases``
makes this inequality explicitly checkable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from ..simulation.backend import NumpyBackend
from ..simulation.config import GasConfig
from ..simulation.thermo import dew_point_celsius
from .balance import SinkState, SourceState, evaluate_cascade
from .config import HeatNetworkConfig, HeatSinkConfig, HeatSourceConfig

__all__ = [
    "PinchResult",
    "StreamSegment",
    "cold_segment_for_sink",
    "composite_curve_kw",
    "compute_pinch_target",
    "dew_point_of_source_celsius",
    "evaluate_pinch_target",
    "hot_segments_for_source",
]

_BACKEND = NumpyBackend()
_BOUNDARY_TOLERANCE_K = 1e-6
"""Toleranz beim Vergleich verschobener Temperaturgrenzen (Float-Rundung nach der
Rundung auf 9 Nachkommastellen in :func:`compute_pinch_target`/:func:`composite_curve_kw`)."""

_DUMMY_SINK_NAME = "__pinch_probe_sink__"
"""Name der Wegwerf-Senke in :func:`_recoverable_power_to_floor_kw` -- taucht in keinem
Ergebnis auf, dient nur dazu, ``evaluate_cascade`` ueberhaupt aufrufen zu koennen (das Feld
``source_available_power_kw`` ist unabhaengig von der Senke, siehe dort)."""


@dataclass(frozen=True, slots=True)
class StreamSegment:
    r"""A linear sub-section of a stream with constant heat capacity rate.

    :param name: source/sink name (for diagnostics, composite curve assignment).
    :param is_hot: ``True`` for exhaust air sources (cool down), ``False`` for sinks
        (heat up).
    :param high_temperature_celsius: upper limit of this section, UNSHIFTED (real
        temperature). For a source the inlet side, for a sink the outlet side.
    :param low_temperature_celsius: lower limit, analogously.
    :param capacity_rate_kw_per_k: :math:`\dot m\,(c_{p,L}+Y\,c_{p,D})` of this section in
        kW/K, always non-negative.
    """

    name: str
    is_hot: bool
    high_temperature_celsius: float
    low_temperature_celsius: float
    capacity_rate_kw_per_k: float

    def __post_init__(self) -> None:
        if self.high_temperature_celsius <= self.low_temperature_celsius:
            raise ValueError(
                "high_temperature_celsius muss über low_temperature_celsius liegen, sind "
                f"{self.high_temperature_celsius}/{self.low_temperature_celsius}"
            )
        if self.capacity_rate_kw_per_k < 0.0:
            raise ValueError(
                f"capacity_rate_kw_per_k darf nicht negativ sein, ist {self.capacity_rate_kw_per_k}"
            )

    def duty_kw(self) -> float:
        """Heat duty of this section, kW."""
        return self.capacity_rate_kw_per_k * (
            self.high_temperature_celsius - self.low_temperature_celsius
        )

    def shifted(self, minimum_pinch_k: float) -> tuple[float, float]:
        """(high, low) shifted limit.

        Hot by ``dT_min/2`` downwards, cold upwards (step 1 of the
        problem table algorithm, see module docstring).
        """
        half = minimum_pinch_k / 2.0
        shift = -half if self.is_hot else half
        return self.high_temperature_celsius + shift, self.low_temperature_celsius + shift


def dew_point_of_source_celsius(source: HeatSourceConfig, gas: GasConfig) -> float:
    """Dew point of a source from its own exhaust air humidity.

    The same Buck correlation as ``balance.py`` (``thermo.dew_point_celsius``) -- no
    second formula.
    """
    return float(
        _BACKEND.to_numpy(
            dew_point_celsius(_BACKEND, gas, _BACKEND.asarray(source.outlet_humidity_kg_per_kg))
        )
    )


def _recoverable_power_to_floor_kw(
    source: HeatSourceConfig, gas: GasConfig, floor_temperature_celsius: float
) -> float:
    r"""ONE support point of the enthalpy-temperature curve.

    :math:`\dot Q_{rec}` when cooling ``source`` from its outlet temperature down to
    ``floor_temperature_celsius``. For this it calls ``balance.evaluate_cascade`` with
    ``recovery_mode="condensing"`` and reads ``source_available_power_kw`` --
    this field is computed there directly from
    outlet temperature and cold limit, INDEPENDENT of the sink and the stages
    (see ``balance.py``, calculation after the stage loop). The throwaway sink below
    therefore limits nothing; it exists only because ``evaluate_cascade`` requires a stage
    list. No separate condensation formula -- see module docstring.
    """
    if floor_temperature_celsius >= source.outlet_temperature_celsius:
        return 0.0
    dummy_sink = HeatSinkConfig(
        name=_DUMMY_SINK_NAME,
        dry_air_mass_flow_kg_per_s=1.0e9,
        humidity_kg_per_kg=0.0,
        intake_temperature_celsius=-40.0,
        setpoint_temperature_celsius=source.outlet_temperature_celsius + 50.0,
    )
    network = HeatNetworkConfig(
        gas=gas,
        sources=(source,),
        sinks=(dummy_sink,),
        minimum_pinch_k=1.0e-6,
        recovery_mode="condensing",
        cold_temperature_celsius=floor_temperature_celsius,
    )
    stages = ((source.name, dummy_sink.name),)
    result = evaluate_cascade(
        network,
        stages,
        (SourceState.from_config(source),),
        (SinkState.from_config(dummy_sink),),
    )
    return float(result.source_available_power_kw[0, 0])


def hot_segments_for_source(
    source: HeatSourceConfig,
    gas: GasConfig,
    floor_temperature_celsius: float,
    *,
    n_segments_below_dew_point: int = 50,
) -> tuple[StreamSegment, ...]:
    r"""Splits a source into piecewise linear segments.

    From ``outlet_temperature_celsius`` to ``floor_temperature_celsius`` (= ``T_kalt``
    of this run). If the dew point lies at or below ``floor_temperature_celsius``,
    the source does not condense at all in this run (e.g. source 2 above approx. 25 °C, see
    ``docs/06_waermenetz.md``) -- then a single, exactly linear segment. Otherwise: one
    exact linear segment above the dew point (humidity constant) plus
    ``n_segments_below_dew_point`` sampled segments below it (see module docstring).

    :raises ValueError: if ``n_segments_below_dew_point < 1``.
    """
    t_out = source.outlet_temperature_celsius
    if floor_temperature_celsius >= t_out:
        return ()
    if n_segments_below_dew_point < 1:
        raise ValueError(
            f"n_segments_below_dew_point muss >= 1 sein, ist {n_segments_below_dew_point}"
        )

    dew_point = dew_point_of_source_celsius(source, gas)
    sensible_capacity_rate_kw_per_k = (
        source.dry_air_mass_flow_kg_per_s
        * (gas.cp_dry_air_j_per_kg_k + source.outlet_humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k)
        / 1000.0
    )

    segments: list[StreamSegment] = []
    if dew_point <= floor_temperature_celsius:
        segments.append(
            StreamSegment(
                source.name, True, t_out, floor_temperature_celsius, sensible_capacity_rate_kw_per_k
            )
        )
        return tuple(segments)

    top_of_condensing_region = min(dew_point, t_out)
    if top_of_condensing_region < t_out:
        segments.append(
            StreamSegment(
                source.name, True, t_out, top_of_condensing_region, sensible_capacity_rate_kw_per_k
            )
        )

    grid = np.linspace(
        top_of_condensing_region, floor_temperature_celsius, n_segments_below_dew_point + 1
    )
    duty_at_grid_kw = np.array(
        [_recoverable_power_to_floor_kw(source, gas, float(t)) for t in grid]
    )
    for index in range(n_segments_below_dew_point):
        t_high, t_low = float(grid[index]), float(grid[index + 1])
        if t_high - t_low <= 0.0:
            continue
        capacity_rate = (duty_at_grid_kw[index + 1] - duty_at_grid_kw[index]) / (t_high - t_low)
        segments.append(StreamSegment(source.name, True, t_high, t_low, max(capacity_rate, 0.0)))
    return tuple(segments)


def cold_segment_for_sink(sink: HeatSinkConfig, gas: GasConfig) -> StreamSegment:
    """A sink as a single segment.

    Purely sensible heating, constant humidity, hence exactly linear (no
    need for discretisation, unlike the sources).
    """
    capacity_rate = (
        sink.dry_air_mass_flow_kg_per_s
        * (gas.cp_dry_air_j_per_kg_k + sink.humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k)
        / 1000.0
    )
    return StreamSegment(
        sink.name,
        False,
        sink.setpoint_temperature_celsius,
        sink.intake_temperature_celsius,
        capacity_rate,
    )


@dataclass(frozen=True, slots=True)
class PinchResult:
    """Result of a problem table run.

    :ivar minimum_hot_utility_kw: minimum heating demand (hot utility) -- the lower bound
        from the module docstring.
    :ivar minimum_cold_utility_kw: minimum cooling demand (cold utility).
    :ivar pinch_temperature_hot_side_celsius: pinch temperature on the hot scale
        (real source temperature at the bottleneck).
    :ivar pinch_temperature_cold_side_celsius: pinch temperature on the cold scale;
        the difference to ``pinch_temperature_hot_side_celsius`` is ``dT_min`` by construction.
    :ivar interval_boundaries_shifted_celsius: shifted interval boundaries, sorted in
        descending order, length ``n+1`` for ``n`` intervals.
    :ivar cascade_kw: cascade AFTER adding ``minimum_hot_utility_kw`` (non-negative
        everywhere), at the same nodes as ``interval_boundaries_shifted_celsius``
        -- this is the grand composite curve (PA4).
    """

    minimum_hot_utility_kw: float
    minimum_cold_utility_kw: float
    pinch_temperature_hot_side_celsius: float
    pinch_temperature_cold_side_celsius: float
    interval_boundaries_shifted_celsius: tuple[float, ...]
    cascade_kw: tuple[float, ...]


def _segment_active(high_low_cp: tuple[float, float, float], t_high: float, t_low: float) -> bool:
    high, low, _ = high_low_cp
    return high >= t_high - _BOUNDARY_TOLERANCE_K and low <= t_low + _BOUNDARY_TOLERANCE_K


def compute_pinch_target(
    hot_segments: Sequence[StreamSegment],
    cold_segments: Sequence[StreamSegment],
    *,
    minimum_pinch_k: float,
) -> PinchResult:
    """Problem table algorithm over already built segments.

    See module docstring, steps 1-3. Segments may come from several sources/sinks AND
    from several sub-sections per source (condensation curve) -- the algorithm
    treats every segment like its own sub-stream with constant CP, which is standard
    for non-linear streams in pinch analysis.

    :raises ValueError: if ``minimum_pinch_k`` is negative or no segments were
        passed.
    """
    if minimum_pinch_k < 0.0:
        raise ValueError(f"minimum_pinch_k darf nicht negativ sein, ist {minimum_pinch_k}")
    if not hot_segments and not cold_segments:
        raise ValueError("mindestens ein Segment (heiss oder kalt) noetig")

    shifted_hot = [
        (*seg.shifted(minimum_pinch_k), seg.capacity_rate_kw_per_k) for seg in hot_segments
    ]
    shifted_cold = [
        (*seg.shifted(minimum_pinch_k), seg.capacity_rate_kw_per_k) for seg in cold_segments
    ]

    boundary_set = {round(t, 9) for high, low, _ in shifted_hot for t in (high, low)}
    boundary_set |= {round(t, 9) for high, low, _ in shifted_cold for t in (high, low)}
    boundaries = sorted(boundary_set, reverse=True)
    if len(boundaries) < 2:
        raise ValueError("weniger als zwei unterschiedliche Temperaturgrenzen -- leere Stroeme?")

    net_kw: list[float] = []
    for t_high, t_low in pairwise(boundaries):
        delta_t = t_high - t_low
        cp_hot = sum(entry[2] for entry in shifted_hot if _segment_active(entry, t_high, t_low))
        cp_cold = sum(entry[2] for entry in shifted_cold if _segment_active(entry, t_high, t_low))
        net_kw.append((cp_hot - cp_cold) * delta_t)

    cascade = [0.0]
    for net in net_kw:
        cascade.append(cascade[-1] + net)

    minimum_hot_utility_kw = max(0.0, -min(cascade))
    feasible_cascade = tuple(value + minimum_hot_utility_kw for value in cascade)
    minimum_cold_utility_kw = feasible_cascade[-1]

    pinch_index = min(range(len(cascade)), key=lambda i: cascade[i])
    pinch_shifted_celsius = boundaries[pinch_index]
    half = minimum_pinch_k / 2.0

    return PinchResult(
        minimum_hot_utility_kw=minimum_hot_utility_kw,
        minimum_cold_utility_kw=minimum_cold_utility_kw,
        pinch_temperature_hot_side_celsius=pinch_shifted_celsius + half,
        pinch_temperature_cold_side_celsius=pinch_shifted_celsius - half,
        interval_boundaries_shifted_celsius=tuple(boundaries),
        cascade_kw=feasible_cascade,
    )


def composite_curve_kw(segments: Sequence[StreamSegment]) -> tuple[tuple[float, float], ...]:
    r"""Composite curve (hot OR cold, depending on which ``segments`` are passed).

    ``(Temperatur, kumulierte Leistung)`` nodes -- UNSHIFTED (real)
    temperatures, what is shown to the consortium partner (PA4). Accumulated from the
    lowest involved temperature upwards (:math:`H=0` at the cold end), segments as
    in the problem table algorithm additive over their respective validity intervals --
    the same interval decomposition, here without shift and without cascade.
    """
    if not segments:
        return ()
    boundary_temperatures = (
        (seg.high_temperature_celsius, seg.low_temperature_celsius) for seg in segments
    )
    boundaries = sorted({round(t, 9) for pair in boundary_temperatures for t in pair})
    cumulative_kw = 0.0
    points: list[tuple[float, float]] = [(boundaries[0], 0.0)]
    for t_low, t_high in pairwise(boundaries):
        active_cp = sum(
            seg.capacity_rate_kw_per_k
            for seg in segments
            if seg.low_temperature_celsius <= t_low + _BOUNDARY_TOLERANCE_K
            and seg.high_temperature_celsius >= t_high - _BOUNDARY_TOLERANCE_K
        )
        cumulative_kw += active_cp * (t_high - t_low)
        points.append((t_high, cumulative_kw))
    return tuple(points)


def evaluate_pinch_target(
    sources: Sequence[HeatSourceConfig],
    sinks: Sequence[HeatSinkConfig],
    gas: GasConfig,
    *,
    cold_temperature_celsius: float,
    minimum_pinch_k: float,
    n_segments_below_dew_point: int = 50,
) -> tuple[PinchResult, tuple[StreamSegment, ...], tuple[StreamSegment, ...]]:
    """Builds segments for all ``sources``/``sinks`` and evaluates the pinch target.

    Source target: ``cold_temperature_celsius``, sink target: its own set point; evaluated
    via :func:`compute_pinch_target`.

    :returns: ``(PinchResult, hot_segments, cold_segments)`` -- the segments are also
        returned so that :func:`composite_curve_kw` can reuse them without re-sampling
        (the sampling via ``evaluate_cascade`` is the expensive part).
    """
    hot_segments = tuple(
        segment
        for source in sources
        for segment in hot_segments_for_source(
            source,
            gas,
            cold_temperature_celsius,
            n_segments_below_dew_point=n_segments_below_dew_point,
        )
    )
    cold_segments = tuple(cold_segment_for_sink(sink, gas) for sink in sinks)
    result = compute_pinch_target(hot_segments, cold_segments, minimum_pinch_k=minimum_pinch_k)
    return result, hot_segments, cold_segments

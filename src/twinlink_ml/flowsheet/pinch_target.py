"""Pinch target of the two-line network: the least burner duty of any assignment.

The six scenarios of the partner model are six fixed assignments of two exhausts to two sinks.
The bound against which they are measured is the target of the problem table
(:func:`twinlink_ml.heatnetwork.pinch.compute_pinch_target`, used unchanged): any number of
exchangers, any split of the streams, any medium flows. This module only builds the stream
segments for it from the units of this package, so that the bound uses **the same release curve
and the same properties** as the counter-current stages it is compared with
(:func:`~.thermo.exhaust_profile`, the partner's ``cp`` of :func:`~.thermo.partner_props`):

* hot: each exhaust from its temperature down along its release curve (piecewise linear between
  the nodes of the curve; above the dew point one exact linear piece);
* cold: each sink's gas from its inlet up to its set point; with an upper limit of the medium
  ``T_max`` the gas can take recovered heat only up to ``T_max - dT`` (``dT`` the approach of one
  exchanger), the rest of its demand is burner duty in any case (``fixed_burner_kw``);
* approach: a circuit puts two exchangers in series, so exhaust and gas meet at least ``2 dT``
  apart at every point of the transferred heat; the problem table is run with ``2 dT``.

The result is a lower bound on the burner duty of every network of circuits with that approach
per exchanger; it is not a design and says nothing about how many exchangers reach it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from twinlink_ml.heatnetwork.pinch import PinchResult, StreamSegment, compute_pinch_target
from twinlink_ml.partner_v4 import corrected as c

from .thermo import GasProps, ProfileKind, capacity, exhaust_profile

__all__ = ["NetworkPinch", "SinkDemand", "exhaust_segments", "network_pinch_target"]

_MIN_SPAN_K = 1e-9


@dataclass(frozen=True)
class SinkDemand:
    """The gas of one sink at its inlet and the set point its burner heats it to."""

    name: str
    gas: c.Stream
    target_celsius: float


@dataclass(frozen=True)
class NetworkPinch:
    """Problem-table target plus the burner duty that no recovery can cover."""

    pinch: PinchResult
    fixed_burner_kw: float
    """Demand of the sinks above the medium limit (``T_max - dT``); 0 without a limit."""
    total_demand_kw: float
    """Burner duty without any recovery."""
    approach_k: float
    """Approach of the problem table (``2 dT``)."""

    @property
    def min_burner_kw(self) -> float:
        return self.pinch.minimum_hot_utility_kw + self.fixed_burner_kw

    @property
    def max_recovered_kw(self) -> float:
        return self.total_demand_kw - self.min_burner_kw


def exhaust_segments(
    name: str, exhaust: c.Stream, props: GasProps, kind: ProfileKind = "condensing"
) -> tuple[StreamSegment, ...]:
    """The release curve of an exhaust as hot segments (kW/K between consecutive nodes)."""
    profile = exhaust_profile(exhaust, props, kind)
    out = []
    for i in range(len(profile.q_w) - 1):
        t_high, t_low = float(profile.t_c[i]), float(profile.t_c[i + 1])
        dq = float(profile.q_w[i + 1] - profile.q_w[i])
        if t_high - t_low <= _MIN_SPAN_K or dq <= 0.0:
            continue
        out.append(StreamSegment(name, True, t_high, t_low, dq / (t_high - t_low) / 1e3))
    return tuple(out)


def network_pinch_target(
    exhausts: Sequence[tuple[str, c.Stream]],
    sinks: Sequence[SinkDemand],
    props: GasProps,
    *,
    dt_min_k_per_exchanger: float,
    htm_max_celsius: float | None = None,
    kind: ProfileKind = "condensing",
) -> NetworkPinch:
    """Least burner duty of any network of circuits (see the module doc)."""
    hot = tuple(seg for name, ex in exhausts for seg in exhaust_segments(name, ex, props, kind))
    cold: list[StreamSegment] = []
    fixed = total = 0.0
    top = math.inf if htm_max_celsius is None else htm_max_celsius - dt_min_k_per_exchanger
    for sink in sinks:
        cp_kw = capacity(sink.gas, props) / 1e3
        t_in, t_set = sink.gas.temperature_celsius, sink.target_celsius
        total += cp_kw * (t_set - t_in)
        t_rec = min(t_set, top)
        fixed += cp_kw * (t_set - max(t_rec, t_in))
        if t_rec - t_in > _MIN_SPAN_K:
            cold.append(StreamSegment(sink.name, False, t_rec, t_in, cp_kw))
    approach = 2.0 * dt_min_k_per_exchanger
    result = compute_pinch_target(hot, tuple(cold), minimum_pinch_k=approach)
    return NetworkPinch(result, fixed, total, approach)

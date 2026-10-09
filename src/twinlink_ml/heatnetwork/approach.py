r"""Minimum approach temperature along a counter-current stage (review finding A1, Befund 78/79).

``balance.evaluate_cascade`` in the mode ``approach_check="along_exchanger"`` (the default since
Befund 79) limits every stage so that the approach temperature is at least ``dT_min`` at EVERY
point of the exchanger, not only at its warm end. The tightest point can lie at the cold end
(sink with the larger heat-capacity rate) or inside the exchanger (at the dew point of a
condensing source, where the slope of the source curve changes).

Source profile
--------------
Heat released by a source cooled from its outlet temperature ``T_out`` down to ``T``:

.. math::
    q(T) = C\,(T_{out} - T) + \dot m_L\,\max(0,\, Y_{out} - Y_s(T))\,\Delta h_v(T_{kalt})

in ``recovery_mode="condensing"`` (down to ``T_kalt``), and ``q(T) = C (T_out - T)`` down to
``T_dp(Y_out) + dT_approach`` in ``recovery_mode="sensible"``. ``C = m_L (c_p,L + Y_out c_p,D)``
and ``dh_v(T_kalt)`` are those of the cascade, so the total down to the floor equals
``source_available_power_kw`` of the warm-end mode (MA1): the profile changes WHERE the heat is
released (condensation from the dew point on, not as a flash at ``T_kalt``), not how much.

Nodes (fixed count per source, so that the computation is vectorised over the time steps):
``T_out``, the dew point as an exact node (clipped into ``[T_floor, T_out]``), and
``approach_profile_nodes`` nodes uniformly in temperature from the dew point down to the floor;
linear between the nodes. Above the dew point the profile is exactly linear (one segment).

Stage duty
----------
With ``q0`` the heat already taken from the source by earlier stages, the stage duty ``Q``, the
sink entering at ``T_in`` with capacity rate ``C_s``, counter-current: at the source point ``q``
the sink temperature is ``T_in + (q0 + Q - q) / C_s``. With ``g(q) = T_src(q) + q / C_s`` the
condition "approach >= dT_min everywhere" is

.. math::
    \min_{q \in [q_0,\, q_0 + Q]} g(q) \;\ge\; dT_{min} + T_{in} + (q_0 + Q) / C_s .

The left side does not increase and the right side increases with ``Q``; the largest admissible
``Q`` follows in closed form from the running minimum of ``g`` over the nodes and one linear
segment (the function ``max_duty_along`` of ``scripts/check_review_a1_cascade_approach.py``,
vectorised here).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..simulation.backend import NumpyBackend
from ..simulation.thermo import (
    dew_point_celsius,
    latent_heat_j_per_kg,
    saturation_humidity_kg_per_kg,
)
from .config import HeatNetworkConfig

__all__ = [
    "SourceProfile",
    "build_source_profile",
    "max_duty_along",
    "profile_temperature",
    "stage_approach_diagnostics",
]

FloatArray = NDArray[np.float64]

_BACKEND = NumpyBackend()


@dataclass(frozen=True, slots=True)
class SourceProfile:
    """Released heat ``q_w`` [W] against the source temperature ``t_celsius`` [degC].

    Both arrays have the form ``(n_nodes, n_steps)``, ordered by non-decreasing ``q_w`` (the
    first node is ``q = 0`` at ``T_out``). ``capacity_w_per_k`` and ``outlet_celsius`` have the
    form ``(n_steps,)``; ``latent_j_per_kg`` is the latent heat used for the condensate
    (``dh_v(T_kalt)``; 0 in the sensible mode).
    """

    q_w: FloatArray
    t_celsius: FloatArray
    capacity_w_per_k: FloatArray
    outlet_celsius: FloatArray
    latent_j_per_kg: float

    @property
    def total_w(self) -> FloatArray:
        return self.q_w[-1]

    def latent_released_w(self, q_w: FloatArray) -> FloatArray:
        """Latent part of the heat released down to the point ``q_w`` (released minus sensible)."""
        sensible = self.capacity_w_per_k * (self.outlet_celsius - profile_temperature(self, q_w))
        return np.maximum(q_w - sensible, 0.0)


def build_source_profile(
    config: HeatNetworkConfig,
    *,
    dry_air_mass_flow_kg_per_s: FloatArray,
    outlet_temperature_celsius: FloatArray,
    outlet_humidity_kg_per_kg: FloatArray,
) -> SourceProfile:
    """Profile of one source (vectorised over the time steps), see the module docstring."""
    gas = config.gas
    m = np.asarray(dry_air_mass_flow_kg_per_s, dtype=np.float64)
    t_out = np.asarray(outlet_temperature_celsius, dtype=np.float64)
    y = np.asarray(outlet_humidity_kg_per_kg, dtype=np.float64)
    capacity = m * (gas.cp_dry_air_j_per_kg_k + y * gas.cp_vapour_j_per_kg_k)
    dew_point = np.asarray(dew_point_celsius(_BACKEND, gas, _BACKEND.asarray(y)), dtype=np.float64)
    n = config.approach_profile_nodes
    fractions = np.linspace(0.0, 1.0, n + 1)[1:, None]
    if config.recovery_mode == "condensing":
        floor = np.full_like(t_out, config.cold_temperature_celsius)
        latent = float(
            latent_heat_j_per_kg(_BACKEND, _BACKEND.asarray(config.cold_temperature_celsius))
        )
    else:
        floor = dew_point + config.dew_point_approach_k
        latent = 0.0
    top = np.maximum(t_out, floor)  # a source at or below its floor releases nothing
    knee = np.clip(dew_point, floor, top)
    t_nodes = np.vstack([top[None, :], knee[None, :], knee + (floor - knee) * fractions])
    t_nodes = np.where(t_out > floor, t_nodes, t_out)
    q_nodes = capacity * (t_out - t_nodes)
    if config.recovery_mode == "condensing":
        y_s = np.asarray(
            saturation_humidity_kg_per_kg(_BACKEND, gas, _BACKEND.asarray(t_nodes)),
            dtype=np.float64,
        )
        q_nodes = q_nodes + m * np.maximum(y - y_s, 0.0) * latent
    q_nodes = np.where(t_out > floor, q_nodes, 0.0)
    q_nodes = np.maximum.accumulate(np.maximum(q_nodes, 0.0), axis=0)
    return SourceProfile(q_nodes, t_nodes, capacity, t_out, latent)


def _take(array: FloatArray, index: NDArray[np.int64]) -> FloatArray:
    return np.take_along_axis(array, index[None, :], axis=0)[0]


def profile_temperature(profile: SourceProfile, q_w: FloatArray) -> FloatArray:
    """Source temperature at the released heat ``q_w`` (per time step, linear between nodes)."""
    q_nodes, t_nodes = profile.q_w, profile.t_celsius
    n_nodes = q_nodes.shape[0]
    index = np.clip(np.sum(q_nodes <= q_w[None, :], axis=0) - 1, 0, n_nodes - 2)
    q_a, q_b = _take(q_nodes, index), _take(q_nodes, index + 1)
    t_a, t_b = _take(t_nodes, index), _take(t_nodes, index + 1)
    span = q_b - q_a
    weight = np.clip(
        np.divide(q_w - q_a, span, out=np.zeros_like(span), where=span > 0.0), 0.0, 1.0
    )
    return t_a + weight * (t_b - t_a)


def _stage_points(
    profile: SourceProfile, q_before: FloatArray, t_start: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Points of the profile from ``q_before`` on: the start, then every node beyond it.

    Nodes at or before ``q_before`` are replaced by the start, so the sequence stays ordered.
    """
    active = profile.q_w > q_before[None, :]
    qs = np.vstack([q_before[None, :], np.where(active, profile.q_w, q_before[None, :])])
    ts = np.vstack([t_start[None, :], np.where(active, profile.t_celsius, t_start[None, :])])
    return qs, ts


def max_duty_along(
    profile: SourceProfile,
    q_before: FloatArray,
    sink_in_celsius: FloatArray,
    sink_capacity_w_per_k: FloatArray,
    minimum_approach_k: float,
    upper_w: FloatArray,
) -> FloatArray:
    """Largest duty ``Q <= upper_w`` [W] with the approach >= ``minimum_approach_k`` everywhere.

    Counter-current stage, per time step; see the module docstring.
    """
    upper = np.maximum(np.minimum(upper_w, profile.total_w - q_before), 0.0)
    c_s = sink_capacity_w_per_k
    c0 = minimum_approach_k + sink_in_celsius
    t_start = profile_temperature(profile, q_before)
    qs, ts = _stage_points(profile, q_before, t_start)
    gs = ts + qs / c_s
    running = np.minimum.accumulate(gs, axis=0)
    h = running - c0 - qs / c_s  # non-increasing along the points
    n_points = qs.shape[0]
    count = np.sum(h >= 0.0, axis=0)
    last = np.clip(count - 1, 0, n_points - 1)
    following = np.minimum(last + 1, n_points - 1)
    q_a, q_b = _take(qs, last), _take(qs, following)
    g_a, g_b = _take(gs, last), _take(gs, following)
    running_a = _take(running, last)
    q_end_upper = q_before + upper

    g_limit = profile_temperature(profile, q_end_upper) + q_end_upper / c_s
    upper_admissible = (
        (q_a >= q_end_upper)
        | (count == n_points)
        | (
            (q_b > q_end_upper)  # the limit lies on the segment after the last admissible point
            & (np.minimum(running_a, g_limit) - c0 - q_end_upper / c_s >= 0.0)
        )
    )
    span = q_b - q_a
    slope = np.divide(g_b - g_a, span, out=np.zeros_like(span), where=span > 0.0)
    root_flat = (running_a - c0) * c_s
    denominator = 1.0 / c_s - slope
    root_line = np.divide(
        g_a - slope * q_a - c0,
        denominator,
        out=np.full_like(denominator, np.inf),
        where=denominator > 0.0,
    )
    q_end = np.maximum(
        np.minimum(np.minimum(root_flat, root_line), np.minimum(q_b, q_end_upper)), q_a
    )
    duty = np.where(upper_admissible, upper, q_end - q_before)
    duty = np.where(count == 0, 0.0, duty)
    return np.clip(duty, 0.0, upper)


def stage_approach_diagnostics(
    profile: SourceProfile,
    q_before: FloatArray,
    duty_w: FloatArray,
    sink_in_celsius: FloatArray,
    sink_capacity_w_per_k: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Smallest approach [K] over the stage and its position (0 = warm end, 1 = cold end)."""
    q_end = q_before + duty_w
    t_start = profile_temperature(profile, q_before)
    t_end = profile_temperature(profile, q_end)
    qs, ts = _stage_points(profile, q_before, t_start)
    inside = qs <= q_end[None, :]
    qs = np.vstack([np.where(inside, qs, q_end[None, :]), q_end[None, :]])
    ts = np.vstack([np.where(inside, ts, t_end[None, :]), t_end[None, :]])
    approach = ts - (sink_in_celsius + (q_end - qs) / sink_capacity_w_per_k)
    index = np.argmin(approach, axis=0)
    minimum = _take(approach, index)
    position = np.divide(
        _take(qs, index) - q_before, duty_w, out=np.zeros_like(duty_w), where=duty_w > 0.0
    )
    return minimum, position

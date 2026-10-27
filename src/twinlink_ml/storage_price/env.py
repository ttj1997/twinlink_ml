r"""Gymnasium environment of the study and the policy wrapper for the evaluation protocol.

Observation (all of it is information M2 also has; demand and waste heat follow the production
plan, which every controller knows through the clock):

=====  =====================================================================================
dim    content
=====  =====================================================================================
1      content ``s / C``
1      demand: gas heat without pump and storage ``U0_t / 1000 kW``
1      waste heat offered ``W_t / 1000 kW``
H      price information of the hours ``t .. t+H-1`` (as M2: known until midnight, then the
       price of the same hour today), ``(c_el - 0.16) / 0.10``; entry 0 is the current price
1      hours to midnight / 24
4      sin / cos of the hour of the day and of the day of the week
=====  =====================================================================================

Action ``a in [-1, 1]^2``: ``Q = (a_0 + 1) / 2 Q_max``; ``a_1 > 0`` charges ``a_1 P_c``, ``a_1 < 0``
discharges ``|a_1| P_d``; the power limits therefore hold by construction, the content limits are
enforced by the projection of :func:`~twinlink_ml.storage_price.evaluate.project` (counted there).
Reward ``-dt (c_U U + c_el Q / COP) / reward_scale``. An episode covers ``episode_h`` consecutive
hours inside ONE training month from a content drawn uniformly in ``[0, C]`` and ends truncated
(time limit, so the agents bootstrap; nothing terminates). The observation returned with the last
step (the bootstrap target) is that of hour ``start + episode_h``, so this hour must lie inside the
same month too: the last start of a month is ``episode_h`` hours before its last hour (otherwise the
target of a January episode would carry the day-ahead prices of 1 February, a test month).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from .controllers import Action, Controller
from .evaluate import month_hours, project
from .market import Market
from .plant import Plant
from .prices import HOURS_PER_DAY, Clock

__all__ = ["PolicyController", "StoragePriceEnv", "action_to_duties", "observation_matrix"]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_PRICE_CENTRE = 0.16
_PRICE_SCALE = 0.10
_POWER_SCALE_KW = 1000.0


def observation_matrix(plant: Plant, market: Market, clock: Clock) -> FloatArray:
    """Every observation entry except the content, for all hours ``(hours, dim - 1)``."""
    n = len(market.electricity)
    hour = clock.hour.astype(np.float64)
    wd = clock.weekday.astype(np.float64)
    cols = [
        plant.direct_kw[:, None] / _POWER_SCALE_KW,
        plant.waste_heat_kw[:, None] / _POWER_SCALE_KW,
        (market.known - _PRICE_CENTRE) / _PRICE_SCALE,
        ((HOURS_PER_DAY - hour) / HOURS_PER_DAY)[:, None],
        np.sin(2 * np.pi * hour / 24)[:, None],
        np.cos(2 * np.pi * hour / 24)[:, None],
        np.sin(2 * np.pi * wd / 7)[:, None],
        np.cos(2 * np.pi * wd / 7)[:, None],
    ]
    out = np.concatenate(cols, axis=1).astype(np.float64)
    if out.shape[0] != n:
        raise ValueError("observation rows do not match the hours")
    return out


def action_to_duties(plant: Plant, action: FloatArray) -> Action:
    """Map ``a in [-1, 1]^2`` to pump heat and storage duties (before the content projection)."""
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    q = 0.5 * (a[0] + 1.0) * plant.size_kw
    if a[1] >= 0.0:
        return Action(float(q), float(a[1] * plant.charge_max_kw), 0.0)
    return Action(float(q), 0.0, float(-a[1] * plant.discharge_max_kw))


class StoragePriceEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Training environment on the training months (module docstring)."""

    # dict, not ClassVar: gym.Env declares ``metadata`` as an instance attribute (as rl/env.py)
    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self,
        plant: Plant,
        market: Market,
        clock: Clock,
        months: Sequence[int],
        *,
        episode_h: int,
        reward_scale: float,
    ) -> None:
        super().__init__()
        self.plant = plant
        self.market = market
        self.features = observation_matrix(plant, market, clock).astype(np.float32)
        starts: list[int] = []
        for m in months:
            h = month_hours(clock, m)
            starts.extend(int(x) for x in h[: len(h) - episode_h])
        if not starts:
            raise ValueError("no episode fits into the training months")
        self.starts = np.asarray(starts, dtype=np.int64)
        self.allowed = {int(x) for m in months for x in month_hours(clock, m)}
        self.episode_h = episode_h
        self.reward_scale = reward_scale
        dim = 1 + self.features.shape[1]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.t = 0
        self.k = 0
        self.s = 0.0
        self.cuts = 0

    def _obs(self) -> NDArray[np.float32]:
        cap = max(self.plant.capacity_kwh, 1e-9)
        return np.concatenate([[self.s / cap], self.features[self.t]]).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: ARG002 - gymnasium interface
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        self.t = int(self.np_random.choice(self.starts))
        self.k = 0
        self.s = float(self.np_random.uniform(0.0, self.plant.capacity_kwh))
        return self._obs(), {}

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self.t not in self.allowed:
            raise RuntimeError("episode left the training months")
        pl, mk = self.plant, self.market
        pr = project(pl, action_to_duties(pl, action), self.s)
        a = pr.action
        if pr.cut_content_kw > 1e-9:
            self.cuts += 1
        u = pl.utility_kw(self.t, a.heat_kw, a.charge_kw, a.discharge_kw)
        cost = pl.dt_h * (mk.heat_value * u + mk.electricity[self.t] * a.heat_kw / pl.cop)
        self.s = min(
            max(pl.keep * self.s + (a.charge_kw - a.discharge_kw) * pl.dt_h, 0.0), pl.capacity_kwh
        )
        self.t += 1
        self.k += 1
        truncated = self.k >= self.episode_h
        if self.t not in self.allowed:
            raise RuntimeError("the observation after the step lies outside the training months")
        return self._obs(), -cost / self.reward_scale, False, truncated, {"cost_eur": cost}


class PolicyController(Controller):
    """Deterministic policy of a trained agent inside the evaluation protocol."""

    def __init__(self, plant: Plant, market: Market, clock: Clock, policy: Any, name: str) -> None:
        super().__init__(plant, market)
        self.policy = policy
        self.name = name
        self.features = observation_matrix(plant, market, clock).astype(np.float32)

    def decide(self, k: int, stored_kwh: float) -> Action:
        t = int(self.hours[k])
        cap = max(self.plant.capacity_kwh, 1e-9)
        obs = np.concatenate([[stored_kwh / cap], self.features[t]]).astype(np.float32)
        act, _ = self.policy.predict(obs, deterministic=True)
        return action_to_duties(self.plant, act)

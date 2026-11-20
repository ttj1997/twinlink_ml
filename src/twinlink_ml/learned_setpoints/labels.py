"""Inputs of the policies of family P and the labels from an expert.

The information of the moment (:func:`decision_features`, names in :data:`FEATURE_NAMES`): hour of
the day, the price of the hour and its rank among the prices of the day, the 24 known prices of
the day (all relative to the reference tariff), ambient temperature and relative humidity, the
remaining production target, the last realised commands and the margin. Nothing of the true
state enters, and nothing the expert does not have either.

Labels: an expert controller (the twin, C2) is run in closed loop on days of the TRAINING months
at several margins; :class:`DecisionLogger` records the inputs and the expert's command at every
decision. The margin is an input, so a policy trained on the labels can be asked for any margin
and its margin is found by the same bisection as for every other controller (decision 9).

Label days per cell (:func:`nested_label_days`): the training days of the load-following study
are sorted by their price spread into strata; per seed the days are permuted within each stratum
and a budget takes the first days of every stratum, so the budgets of a seed are nested and each
is stratified.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from twinlink_ml.load_following.config import LoadFollowingConfig
from twinlink_ml.load_following.controllers import LoadController
from twinlink_ml.load_following.market import DayScenario

__all__ = [
    "FEATURE_NAMES",
    "HAND_MADE_FEATURES",
    "N_PRICES",
    "DecisionLogger",
    "LabelSet",
    "decision_features",
    "nested_label_days",
]

N_PRICES = 24
FEATURE_NAMES: tuple[str, ...] = (
    "hour_sin",
    "hour_cos",
    "price_of_hour",
    "price_quantile_of_hour",
    *(f"price_{h:02d}" for h in range(N_PRICES)),
    "ambient_temperature_celsius",
    "ambient_relative_humidity",
    "remaining_target_fraction",
    "required_rate_kg_per_h",
    "last_inlet_temperature_celsius",
    "last_gas_velocity_m_per_s",
    "last_spray_rate_kg_per_h",
    "margin_kg_per_kg",
)
HAND_MADE_FEATURES: tuple[str, ...] = (
    "price_quantile_of_hour",
    "ambient_temperature_celsius",
    "ambient_relative_humidity",
    "required_rate_kg_per_h",
    "margin_kg_per_kg",
)
"""Inputs of the baseline P-lin."""


def decision_features(
    config: LoadFollowingConfig,
    day: DayScenario,
    k: int,
    disturbance: np.ndarray,
    sprayed_kg: float,
    last_command: np.ndarray,
    margin: float,
    reference_tariff: float,
) -> np.ndarray:
    """Input vector of a policy for interval ``k`` of ``day`` (order of :data:`FEATURE_NAMES`)."""
    per_hour = config.intervals_per_hour
    hour = k // per_hour
    prices = np.asarray(day.price_known[: day.hours], dtype=np.float64) / reference_tariff
    padded = np.full(N_PRICES, prices[-1])
    padded[: min(len(prices), N_PRICES)] = prices[:N_PRICES]
    target = config.nominal_spray_rate * day.hours
    remaining = target - sprayed_kg
    hours_left = day.hours - k / per_hour
    angle = 2.0 * np.pi * (k / per_hour) / 24.0
    return np.array(
        [
            np.sin(angle),
            np.cos(angle),
            prices[hour],
            float(np.mean(prices <= prices[hour])),
            *padded,
            disturbance[0],
            disturbance[1],
            remaining / target,
            remaining / hours_left,
            last_command[0],
            last_command[1],
            last_command[2],
            margin,
        ],
        dtype=np.float64,
    )


class DecisionLogger(LoadController):
    """Wraps an expert controller and records inputs and commands of every decision."""

    def __init__(
        self,
        expert: LoadController,
        config: LoadFollowingConfig,
        margin: float,
        reference_tariff: float,
    ) -> None:
        super().__init__()
        self._expert = expert
        self._config = config
        self._margin = margin
        self._reference = reference_tariff
        self._day: DayScenario | None = None
        self.features: list[np.ndarray] = []
        self.commands: list[np.ndarray] = []

    def reset(self, day: DayScenario, state: np.ndarray, command: np.ndarray) -> None:
        """Start a day."""
        self._day = day
        self.features, self.commands = [], []
        self._expert.reset(day, state, command)

    def command(
        self,
        k: int,
        state: np.ndarray,
        last_command: np.ndarray,
        disturbance: np.ndarray,
        sprayed_kg: float,
    ) -> np.ndarray:
        """The expert's command, recorded with the inputs a policy would see."""
        assert self._day is not None
        out = self._expert.command(k, state, last_command, disturbance, sprayed_kg)
        self.features.append(
            decision_features(
                self._config,
                self._day,
                k,
                disturbance,
                sprayed_kg,
                last_command,
                self._margin,
                self._reference,
            )
        )
        self.commands.append(np.asarray(out, dtype=np.float64).copy())
        self.info = dict(self._expert.info)
        return out


@dataclass(frozen=True, eq=False)
class LabelSet:
    """Labels of an expert: one row per decision.

    ``features (n, F)``, ``commands (n, 3)``, ``day (n,)`` day of the year, ``margin (n,)``,
    ``step (n,)`` interval of the day. Rows of one (day, margin) are consecutive and in time order.
    """

    features: np.ndarray
    commands: np.ndarray
    day: np.ndarray
    margin: np.ndarray
    step: np.ndarray

    def subset(self, days: Sequence[int]) -> LabelSet:
        """The rows of the given days."""
        keep = np.isin(self.day, np.asarray(list(days)))
        return LabelSet(
            self.features[keep],
            self.commands[keep],
            self.day[keep],
            self.margin[keep],
            self.step[keep],
        )

    def sequences(self) -> list[np.ndarray]:
        """Row indices per (day, margin), in time order."""
        keys = np.stack([self.day.astype(np.float64), self.margin], axis=1)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).ravel()
        out = []
        for group in range(int(inverse.max()) + 1):
            rows = np.flatnonzero(inverse == group)
            out.append(rows[np.argsort(self.step[rows], kind="stable")])
        return out

    def save(self, path: Path) -> None:
        """Write as ``.npz`` (never over an existing file)."""
        if path.exists():
            raise FileExistsError(f"{path} exists -- labels are never overwritten")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features,
            commands=self.commands,
            day=self.day,
            margin=self.margin,
            step=self.step,
        )

    @classmethod
    def load(cls, path: Path) -> LabelSet:
        """Read labels written by :meth:`save`."""
        with np.load(path) as data:
            return cls(
                *(np.array(data[k]) for k in ("features", "commands", "day", "margin", "step"))
            )

    @classmethod
    def concatenate(cls, parts: Sequence[LabelSet]) -> LabelSet:
        """Rows of several label sets."""
        return cls(
            *(
                np.concatenate([getattr(p, name) for p in parts])
                for name in ("features", "commands", "day", "margin", "step")
            )
        )


def nested_label_days(
    days: Sequence[int], spread: Sequence[float], budgets: Sequence[int], strata: int, seed: int
) -> dict[int, list[int]]:
    """Label days per budget for one seed, nested and stratified by the price spread.

    ``days`` with their price ``spread`` are sorted into ``strata`` equally large strata; the days
    of each stratum are permuted with ``seed``; a budget of ``K`` days takes the first
    ``K / strata`` days of every stratum (``K`` must be a multiple of ``strata`` and at most
    ``len(days)``). The returned lists keep the draw order (stratum-major within a round).
    """
    order = sorted(range(len(days)), key=lambda i: (spread[i], days[i]))
    groups = np.array_split(np.array([days[i] for i in order]), strata)
    permuted = [
        [int(d) for d in np.random.default_rng([seed, g]).permutation(group)]
        for g, group in enumerate(groups)
    ]
    out: dict[int, list[int]] = {}
    for budget in budgets:
        per_stratum, rest = divmod(budget, strata)
        if rest or any(per_stratum > len(group) for group in permuted):
            raise ValueError(f"budget {budget} does not fit {strata} strata of {len(days)} days")
        out[budget] = [group[r] for r in range(per_stratum) for group in permuted]
    return out

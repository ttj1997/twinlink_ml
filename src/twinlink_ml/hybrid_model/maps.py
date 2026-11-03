r"""Steady operating-point maps of the models M0-M4 (docs/NN_hybrid_model.md).

Every map takes operating points (inlet temperature, gas velocity) under contexts (ambient
temperature, relative humidity) -- the other inputs where the bandit of Befund 62/93/96 holds
them -- and returns energy, moisture, its ensemble spread and the bed temperature.

* :class:`GreyBoxMap` -- the steady grey-box model of one process config (M0: nominal "prior
  knowledge"; M2: refitted kinetics; the truth). ``wide`` selects the config with the widened
  disturbance ranges for the extrapolation contexts (a); only the bracket of the inner bisection
  depends on the ranges.
* :class:`GruMap` -- one composite GRU, steady state = fixed point of its horizon map with the
  criterion of Befund 113 (M1).
* :class:`HybridMap` -- prior knowledge plus learned residual: ``nominal(u, w) + r*(u, w)`` with
  ``r*`` the fixed point of a residual GRU (M3) or the member mean of several (M4; spread
  ``std(ddof=1)`` of the moisture residual). The residual networks are trained on
  ``truth - nominal`` along the training runs, so in steady state their fixed point is the steady
  residual. With a residual of exactly zero the map IS the nominal map (test).

A point is valid only if the steady solver converged and every head of every network has a
valid fixed point.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from twinlink_ml.simulation import ProcessConfig

from .steady_batch import BatchSteadyResult, solve_steady_states

__all__ = ["GreyBoxMap", "GruMap", "HybridMap", "MapPrediction", "ResidualNetwork", "SteadyMap"]


@dataclass(frozen=True, slots=True)
class MapPrediction:
    """Steady prediction at ``n`` points (NaN where not valid)."""

    energy_kw: np.ndarray
    moisture_kg_per_kg: np.ndarray
    moisture_std_kg_per_kg: np.ndarray
    bed_temperature_celsius: np.ndarray
    valid: np.ndarray


class SteadyMap(ABC):
    """Steady map ``(action, context) -> MapPrediction``."""

    @abstractmethod
    def predict(self, actions: np.ndarray, contexts: np.ndarray, *, wide: bool) -> MapPrediction:
        """``actions (n, 2)``: T_in, u_gas; ``contexts (n, 2)``: T_amb, phi."""


class ResidualNetwork(Protocol):
    """What :class:`HybridMap` needs of a network (``GruFixedPointModel`` has it)."""

    def steady_physical(
        self, actions: np.ndarray, contexts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


class GreyBoxMap(SteadyMap):
    """Steady grey-box model; targets in the order energy, moisture, bed temperature."""

    def __init__(
        self,
        config: ProcessConfig,
        wide_config: ProcessConfig,
        *,
        nozzle_pressure_bar: float,
        quality_limit_kg_per_kg: float,
    ) -> None:
        self.config = config
        self.wide_config = wide_config
        self._nozzle = nozzle_pressure_bar
        self._limit = quality_limit_kg_per_kg

    def solve(self, actions: np.ndarray, contexts: np.ndarray, *, wide: bool) -> BatchSteadyResult:
        """Full batch result (constraint check included) at the points."""
        actions = np.atleast_2d(np.asarray(actions, dtype=np.float64))
        contexts = np.atleast_2d(np.asarray(contexts, dtype=np.float64))
        return solve_steady_states(
            self.wide_config if wide else self.config,
            actions[:, 0],
            actions[:, 1],
            contexts[:, 0],
            contexts[:, 1],
            nozzle_pressure_bar=self._nozzle,
            quality_limit_kg_per_kg=self._limit,
        )

    def predict(self, actions: np.ndarray, contexts: np.ndarray, *, wide: bool) -> MapPrediction:
        result = self.solve(actions, contexts, wide=wide)
        valid = ~result.solve_failed
        return MapPrediction(
            energy_kw=result.energy_kw,
            moisture_kg_per_kg=result.moisture_kg_per_kg,
            moisture_std_kg_per_kg=np.where(valid, 0.0, np.nan),
            bed_temperature_celsius=result.bed_temperature_celsius,
            valid=valid,
        )


class GruMap(SteadyMap):
    """One composite network of the Befund 94 pipeline (targets energy, moisture, bed)."""

    def __init__(self, network: ResidualNetwork) -> None:
        self.network = network

    def predict(self, actions: np.ndarray, contexts: np.ndarray, *, wide: bool) -> MapPrediction:
        del wide  # the network has no solver bracket
        values, ok, _ = self.network.steady_physical(actions, contexts)
        valid = ok.all(axis=1)
        return MapPrediction(
            energy_kw=np.where(valid, values[:, 0], np.nan),
            moisture_kg_per_kg=np.where(valid, values[:, 1], np.nan),
            moisture_std_kg_per_kg=np.where(valid, 0.0, np.nan),
            bed_temperature_celsius=np.where(valid, values[:, 2], np.nan),
            valid=valid,
        )


class HybridMap(SteadyMap):
    """Nominal grey-box map plus the steady residual of one (M3) or several (M4) networks."""

    def __init__(self, nominal: GreyBoxMap, residuals: Sequence[ResidualNetwork]) -> None:
        if not residuals:
            raise ValueError("at least one residual network")
        self.nominal = nominal
        self.residuals = tuple(residuals)

    def predict(self, actions: np.ndarray, contexts: np.ndarray, *, wide: bool) -> MapPrediction:
        base = self.nominal.predict(actions, contexts, wide=wide)
        results = [r.steady_physical(actions, contexts) for r in self.residuals]
        values = np.stack([v for v, _, _ in results])  # (members, n, 3)
        ok = np.stack([o for _, o, _ in results]).all(axis=(0, 2)) & base.valid
        mean = values.mean(axis=0) if len(results) > 1 else values[0]
        std = values[:, :, 1].std(axis=0, ddof=1) if len(results) > 1 else np.zeros(values.shape[1])
        return MapPrediction(
            energy_kw=np.where(ok, base.energy_kw + mean[:, 0], np.nan),
            moisture_kg_per_kg=np.where(ok, base.moisture_kg_per_kg + mean[:, 1], np.nan),
            moisture_std_kg_per_kg=np.where(ok, std, np.nan),
            bed_temperature_celsius=np.where(ok, base.bed_temperature_celsius + mean[:, 2], np.nan),
            valid=ok,
        )

r"""Operating-point models for the learned environments U1-U3 (docs/30).

An :class:`OperatingPointModel` maps operating points (inlet temperature, gas velocity) under
contexts (ambient temperature, relative humidity) to the steady energy, product moisture and bed
temperature, with the other inputs held where the bandit of Befund 62/93/96 holds them: nozzle
pressure at ``FIXED_NOZZLE_PRESSURE_BAR``, spray rate and solid fraction at their nominal values.

* :class:`GruFixedPointModel` -- one composite GRU of the Befund 94 pipeline; the steady state is
  the fixed point of its horizon map (:mod:`.fixed_point`).
* :class:`EnsembleFixedPointModel` -- several such networks trained on the same data (different
  model seeds); mean of every channel, sample standard deviation (``ddof=1``) of the moisture, and
  the moisture of the constraint ``mean + k_sigma * std`` (U3: ``k_sigma = 2``). A point is valid
  only if every head of every member has a valid fixed point.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from twinlink_ml.data.windows import WindowBatch

from .fixed_point import FIXED_POINT_STATUSES, FixedPointCriterion, solve_fixed_points

__all__ = [
    "EXOGENOUS_NAMES",
    "TARGET_NAMES",
    "EnsembleFixedPointModel",
    "FixedInputs",
    "GruFixedPointModel",
    "OperatingPointModel",
    "Prediction",
]

EXOGENOUS_NAMES: tuple[str, ...] = (
    "inlet_temperature_celsius",
    "gas_velocity_m_per_s",
    "spray_rate_kg_per_h",
    "nozzle_pressure_bar",
    "ambient_temperature_celsius",
    "ambient_relative_humidity",
    "suspension_solid_fraction",
)
"""Exogenous channels of the Befund 94 data in their order (actions, then disturbances)."""

TARGET_NAMES: tuple[str, ...] = (
    "energy_target",
    "product_moisture_kg_per_kg",
    "bed_temperature_celsius",
)


@dataclass(frozen=True, slots=True)
class FixedInputs:
    """Inputs the bandit does not choose (Befund 62: nozzle fixed point, nominal spray, solids)."""

    nozzle_pressure_bar: float
    spray_rate_kg_per_h: float
    suspension_solid_fraction: float


@dataclass(frozen=True, slots=True)
class Prediction:
    """Steady prediction at ``n`` operating points (NaN where not valid)."""

    energy_kw: np.ndarray
    moisture_kg_per_kg: np.ndarray
    bed_temperature_celsius: np.ndarray
    moisture_std_kg_per_kg: np.ndarray
    moisture_for_constraint_kg_per_kg: np.ndarray
    valid: np.ndarray
    status_counts: dict[str, int]
    """Head x member statuses over the points, keyed ``"<target>/<status>"``."""

    def point(self, index: int) -> dict[str, Any]:
        return {
            "energy_kw": float(self.energy_kw[index]),
            "moisture_kg_per_kg": float(self.moisture_kg_per_kg[index]),
            "bed_temperature_celsius": float(self.bed_temperature_celsius[index]),
            "moisture_std_kg_per_kg": float(self.moisture_std_kg_per_kg[index]),
            "moisture_for_constraint_kg_per_kg": float(
                self.moisture_for_constraint_kg_per_kg[index]
            ),
            "valid": bool(self.valid[index]),
        }


class OperatingPointModel(ABC):
    """Steady map ``(action, context) -> (energy, moisture, bed temperature)``."""

    @abstractmethod
    def predict(self, actions: np.ndarray, contexts: np.ndarray) -> Prediction:
        """``actions (n, 2)``: inlet temperature, gas velocity; ``contexts (n, 2)``: T_amb, phi."""


def _count(statuses: np.ndarray, prefix: str, out: dict[str, int]) -> None:
    for name in FIXED_POINT_STATUSES:
        key = f"{prefix}/{name}"
        out[key] = out.get(key, 0) + int(np.sum(statuses == name))


class GruFixedPointModel(OperatingPointModel):
    """One composite network with its scalers; steady state = fixed point of the horizon map."""

    def __init__(
        self,
        model: Any,
        metadata: dict[str, Any],
        fixed: FixedInputs,
        criterion: FixedPointCriterion | None = None,
    ) -> None:
        if tuple(metadata["exogenous_names"]) != EXOGENOUS_NAMES:
            raise ValueError(f"unexpected exogenous channels {metadata['exogenous_names']}")
        if tuple(metadata["target_names"]) != TARGET_NAMES:
            raise ValueError(f"unexpected targets {metadata['target_names']}")
        spec = metadata["data_spec"]
        self._model = model
        self.metadata = metadata
        self._fixed = fixed
        self._criterion = criterion if criterion is not None else FixedPointCriterion()
        self._input_scaler = metadata["input_scaler"]
        self._target_scaler = metadata["target_scaler"]
        self.history_length = int(spec["data/history_length"])
        self.horizon = int(spec["data/horizon"])
        self._n_exo = len(EXOGENOUS_NAMES)

    @classmethod
    def load(
        cls, path: Path, fixed: FixedInputs, criterion: FixedPointCriterion | None = None
    ) -> GruFixedPointModel:
        from twinlink_ml.utils.checkpoints import load_checkpoint

        model, metadata = load_checkpoint(path)
        return cls(model, metadata, fixed, criterion)

    @property
    def criterion(self) -> FixedPointCriterion:
        return self._criterion

    def _exogenous(self, actions: np.ndarray, contexts: np.ndarray) -> np.ndarray:
        n = actions.shape[0]
        values = np.empty((n, self._n_exo))
        values[:, 0] = actions[:, 0]
        values[:, 1] = actions[:, 1]
        values[:, 2] = self._fixed.spray_rate_kg_per_h
        values[:, 3] = self._fixed.nozzle_pressure_bar
        values[:, 4] = contexts[:, 0]
        values[:, 5] = contexts[:, 1]
        values[:, 6] = self._fixed.suspension_solid_fraction
        return values

    def horizon_map(self, exogenous: np.ndarray, targets_physical: np.ndarray) -> np.ndarray:
        """Predicted targets ``(n, H, 3)`` (physical) for constant inputs and constant histories.

        ``exogenous (n, 7)`` physical, ``targets_physical (n, 3)`` the constant target histories.
        """
        n = exogenous.shape[0]
        history = np.concatenate([exogenous, targets_physical], axis=-1)
        history_std = self._input_scaler.transform(history)
        exo_std = self._input_scaler.select(range(self._n_exo)).transform(exogenous)
        batch = WindowBatch(
            history_inputs=np.repeat(history_std[:, None, :], self.history_length, axis=1),
            future_exogenous=np.repeat(exo_std[:, None, :], self.horizon, axis=1),
            future_targets=np.zeros((n, self.horizon, len(TARGET_NAMES))),
            run_index=np.zeros(n),
            window_start=np.zeros(n),
        )
        prediction = np.asarray(self._model.predict(batch), dtype=np.float64)
        return self._target_scaler.inverse(prediction)

    def fixed_points(self, actions: np.ndarray, contexts: np.ndarray) -> Any:
        """:class:`~.fixed_point.FixedPointResult` in target-standardised units, all heads."""
        exogenous = self._exogenous(actions, contexts)
        mean = np.asarray(self._target_scaler.mean, dtype=np.float64)
        scale = np.asarray(self._target_scaler.scale, dtype=np.float64)
        n_points = exogenous.shape[0]

        def residual(candidates: np.ndarray) -> np.ndarray:
            _, n_cand, n_heads = candidates.shape
            physical = mean + scale * candidates.reshape(-1, n_heads)
            exo = np.repeat(exogenous, n_cand, axis=0)
            final = self.horizon_map(exo, physical)[:, -1, :]
            return ((final - mean) / scale).reshape(n_points, n_cand, n_heads) - candidates

        return solve_fixed_points(residual, n_points, len(TARGET_NAMES), self._criterion)

    def steady_physical(
        self, actions: np.ndarray, contexts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(values (n, 3) physical, ok (n, 3), statuses (n, 3))``."""
        actions = np.atleast_2d(np.asarray(actions, dtype=np.float64))
        contexts = np.atleast_2d(np.asarray(contexts, dtype=np.float64))
        result = self.fixed_points(actions, contexts)
        mean = np.asarray(self._target_scaler.mean, dtype=np.float64)
        scale = np.asarray(self._target_scaler.scale, dtype=np.float64)
        return mean + scale * result.root, result.ok, result.status

    def predict(self, actions: np.ndarray, contexts: np.ndarray) -> Prediction:
        values, ok, statuses = self.steady_physical(actions, contexts)
        counts: dict[str, int] = {}
        for k, target in enumerate(TARGET_NAMES):
            _count(statuses[:, k], target, counts)
        valid = ok.all(axis=1)
        nan = np.full(valid.shape, np.nan)
        moisture = np.where(valid, values[:, 1], np.nan)
        return Prediction(
            energy_kw=np.where(valid, values[:, 0], np.nan),
            moisture_kg_per_kg=moisture,
            bed_temperature_celsius=np.where(valid, values[:, 2], np.nan),
            moisture_std_kg_per_kg=np.where(valid, 0.0, nan),
            moisture_for_constraint_kg_per_kg=moisture,
            valid=valid,
            status_counts=counts,
        )


class EnsembleFixedPointModel(OperatingPointModel):
    """Members trained on the same data; pessimistic moisture ``mean + k_sigma * std(ddof=1)``."""

    def __init__(self, members: Sequence[GruFixedPointModel], k_sigma: float) -> None:
        if len(members) < 2:
            raise ValueError("an ensemble needs at least two members")
        if k_sigma < 0.0:
            raise ValueError("k_sigma must be non-negative")
        self.members = tuple(members)
        self.k_sigma = float(k_sigma)

    def predict(self, actions: np.ndarray, contexts: np.ndarray) -> Prediction:
        results = [member.steady_physical(actions, contexts) for member in self.members]
        values = np.stack([r[0] for r in results])  # (m, n, 3)
        ok = np.stack([r[1] for r in results]).all(axis=(0, 2))
        counts: dict[str, int] = {}
        for _, _, statuses in results:
            for k, target in enumerate(TARGET_NAMES):
                _count(statuses[:, k], target, counts)
        mean = values.mean(axis=0)
        std = values[:, :, 1].std(axis=0, ddof=1)
        pessimistic = mean[:, 1] + self.k_sigma * std
        return Prediction(
            energy_kw=np.where(ok, mean[:, 0], np.nan),
            moisture_kg_per_kg=np.where(ok, mean[:, 1], np.nan),
            bed_temperature_celsius=np.where(ok, mean[:, 2], np.nan),
            moisture_std_kg_per_kg=np.where(ok, std, np.nan),
            moisture_for_constraint_kg_per_kg=np.where(ok, pessimistic, np.nan),
            valid=ok,
            status_counts=counts,
        )

"""The two abstractions of the study: a steady problem (the "true model") and a family of them.

A :class:`SteadyProblem` maps a decision ``u`` and a disturbance ``w`` to the energy ``E`` (kW, the
objective) and one value per constraint in the convention ``g(u, w) <= 0`` (physical units, the
unit of each constraint in its :class:`ConstraintSpec`). Box bounds of ``u`` are known exactly and
are not constraints. Every call of :meth:`SteadyProblem.evaluate` counts (:attr:`calls`): the
budget of the study is the number of true evaluations.

A :class:`ProblemFamily` builds problems for parameter values. The truth of the study is the
member at the shifted values (:attr:`ProblemFamily.truth_values`), the grey-box fit (M4) works on
members whose fitted parameters it estimates while the hidden parameter stays at its nominal value
(:attr:`ParameterSpec.fitted`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Box",
    "ConstraintSpec",
    "Evaluation",
    "ParameterSpec",
    "ProblemFamily",
    "SteadyProblem",
]


@dataclass(frozen=True)
class Box:
    """Named lower and upper bounds of a vector."""

    names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        if not len(self.names) == len(self.lower) == len(self.upper):
            raise ValueError("names, lower and upper must have the same length")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper, strict=True)):
            raise ValueError("every lower bound must be below its upper bound")

    @property
    def dim(self) -> int:
        return len(self.names)

    @property
    def lo(self) -> np.ndarray:
        return np.asarray(self.lower, dtype=np.float64)

    @property
    def hi(self) -> np.ndarray:
        return np.asarray(self.upper, dtype=np.float64)

    def from_unit(self, z: np.ndarray) -> np.ndarray:
        """Map points of the unit cube (last axis) into the box."""
        out: np.ndarray = self.lo + np.asarray(z, dtype=np.float64) * (self.hi - self.lo)
        return out

    def to_unit(self, x: np.ndarray) -> np.ndarray:
        """Map points of the box (last axis) into the unit cube."""
        out: np.ndarray = (np.asarray(x, dtype=np.float64) - self.lo) / (self.hi - self.lo)
        return out

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=np.float64), self.lo, self.hi)

    def as_dict(self) -> dict[str, list[float]]:
        return {n: [lo, hi] for n, lo, hi in zip(self.names, self.lower, self.upper, strict=True)}


@dataclass(frozen=True)
class ConstraintSpec:
    """One constraint ``g <= 0``: name, unit, scale and feasibility tolerance (both in ``unit``).

    ``scale`` normalises the constraint for the optimiser and for the margin of the fixed operating
    point (``g / scale``); ``tolerance`` is the largest true value that still counts as feasible.
    """

    name: str
    unit: str
    scale: float
    tolerance: float
    description: str = ""


@dataclass(frozen=True)
class Evaluation:
    """One evaluation of a problem: energy (kW), constraint values (physical) and side metrics."""

    energy_kw: float
    constraints: np.ndarray
    side: Mapping[str, float] = field(default_factory=dict)

    @property
    def outputs(self) -> np.ndarray:
        """``[E, g_1, ..., g_m]``: the targets of the surrogate models."""
        return np.concatenate(([self.energy_kw], self.constraints))


class SteadyProblem(ABC):
    """A steady model ``(u, w) -> (E, g)``; every evaluation counts (see the module doc)."""

    name: str
    decisions: Box
    disturbances: Box
    constraints: tuple[ConstraintSpec, ...]

    def __init__(self) -> None:
        self.calls = 0

    @property
    def n_outputs(self) -> int:
        return 1 + len(self.constraints)

    @property
    def scales(self) -> np.ndarray:
        return np.array([c.scale for c in self.constraints])

    @property
    def tolerances(self) -> np.ndarray:
        return np.array([c.tolerance for c in self.constraints])

    @abstractmethod
    def _evaluate(self, u: np.ndarray, w: np.ndarray) -> Evaluation:
        """Evaluate one point (``u`` inside the decision box)."""

    def evaluate(
        self, u: Sequence[float] | np.ndarray, w: Sequence[float] | np.ndarray
    ) -> Evaluation:
        """Evaluate one point and count the call."""
        self.calls += 1
        return self._evaluate(np.asarray(u, dtype=np.float64), np.asarray(w, dtype=np.float64))

    def outputs(self, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """``(n, 1 + m)`` outputs for ``(n, du)`` decisions and ``(n, dw)`` disturbances."""
        u2, w2 = np.atleast_2d(u), np.atleast_2d(w)
        if w2.shape[0] == 1 and u2.shape[0] > 1:
            w2 = np.repeat(w2, u2.shape[0], axis=0)
        return np.array([self.evaluate(a, b).outputs for a, b in zip(u2, w2, strict=True)])

    def feasible(self, evaluation: Evaluation) -> bool:
        """All constraints within their tolerance."""
        return bool(np.all(evaluation.constraints <= self.tolerances))

    def violation(self, evaluation: Evaluation) -> float:
        """Largest normalised violation ``max(g / scale)``, clipped at 0."""
        return float(max(np.max(evaluation.constraints / self.scales), 0.0))


@dataclass(frozen=True)
class ParameterSpec:
    """An uncertain parameter: nominal value, value of the truth, fit bounds, fitted or hidden."""

    name: str
    nominal: float
    truth: float
    lower: float
    upper: float
    fitted: bool
    source: str = ""
    """Where the nominal value and the shift come from (placeholder, band of a study)."""


class ProblemFamily(ABC):
    """Problems for parameter values (see the module doc)."""

    parameters: tuple[ParameterSpec, ...]

    @abstractmethod
    def build(self, values: Mapping[str, float]) -> SteadyProblem:
        """The problem at these parameter values (missing names: nominal)."""

    @property
    def nominal_values(self) -> dict[str, float]:
        return {p.name: p.nominal for p in self.parameters}

    @property
    def truth_values(self) -> dict[str, float]:
        return {p.name: p.truth for p in self.parameters}

    @property
    def fitted(self) -> tuple[ParameterSpec, ...]:
        return tuple(p for p in self.parameters if p.fitted)

    def truth(self) -> SteadyProblem:
        return self.build(self.truth_values)

    def model(self, fitted_values: Sequence[float] | np.ndarray) -> SteadyProblem:
        """The grey-box model: fitted parameters set, hidden ones at their nominal value."""
        values = self.nominal_values
        for spec, value in zip(self.fitted, fitted_values, strict=True):
            values[spec.name] = float(value)
        return self.build(values)

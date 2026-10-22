"""A quadratic test problem with a known optimum (tests of the models, the optimiser, the budget).

``u, w`` in ``[0, 1]^2``::

    E = 1 + (u_1 - a(w))^2 + (u_2 - b(w))^2,   a = a_0 + 0.3 w_1,   b = b_0 + 0.2 w_2
    g = h(w) - u_1 <= 0,                         h = h_0 + 0.2 w_2

Optimum ``u_1* = max(a, h)``, ``u_2* = b`` (inside the box for the defaults); the constraint is
active where ``h > a``. ``a_0`` and ``b_0`` are the parameters a grey-box fit estimates, ``h_0``
the hidden one.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .problem import Box, ConstraintSpec, Evaluation, ParameterSpec, ProblemFamily, SteadyProblem

__all__ = ["QuadraticFamily", "QuadraticProblem"]


class QuadraticProblem(SteadyProblem):
    """See the module doc."""

    def __init__(self, a0: float = 0.3, b0: float = 0.5, h0: float = 0.4) -> None:
        super().__init__()
        self.name = "quadratic"
        self.a0, self.b0, self.h0 = a0, b0, h0
        self.decisions = Box(("u1", "u2"), (0.0, 0.0), (1.0, 1.0))
        self.disturbances = Box(("w1", "w2"), (0.0, 0.0), (1.0, 1.0))
        self.constraints = (ConstraintSpec("lower_u1", "-", 1.0, 1e-4),)

    def _evaluate(self, u: np.ndarray, w: np.ndarray) -> Evaluation:
        a, b, h = self.a0 + 0.3 * w[0], self.b0 + 0.2 * w[1], self.h0 + 0.2 * w[1]
        e = 1.0 + (u[0] - a) ** 2 + (u[1] - b) ** 2
        return Evaluation(float(e), np.array([h - u[0]]))

    def optimum(self, w: np.ndarray) -> np.ndarray:
        """The known optimum."""
        a, b, h = self.a0 + 0.3 * w[0], self.b0 + 0.2 * w[1], self.h0 + 0.2 * w[1]
        return np.array([max(a, h), b])


class QuadraticFamily(ProblemFamily):
    """``a_0``, ``b_0`` fitted, ``h_0`` hidden; truth values as given."""

    def __init__(self, truth: Mapping[str, float] | None = None) -> None:
        t = {"a0": 0.35, "b0": 0.45, "h0": 0.4, **(truth or {})}
        self.parameters = (
            ParameterSpec("a0", 0.3, t["a0"], 0.0, 1.0, True),
            ParameterSpec("b0", 0.5, t["b0"], 0.0, 1.0, True),
            ParameterSpec("h0", 0.4, t["h0"], 0.0, 1.0, False),
        )

    @property
    def nominal_disturbance(self) -> np.ndarray:
        return np.array([0.5, 0.5])

    def build(self, values: Mapping[str, float]) -> QuadraticProblem:
        v = {**self.nominal_values, **values}
        return QuadraticProblem(v["a0"], v["b0"], v["h0"])

"""Surrogates of a stationary flowsheet: polynomial regression and an MLP.

All models see the inputs scaled to ``[-1, 1]`` by the box (an a-priori scaling, no statistic of
the data) and predict every target at once.

* :class:`PolynomialRegression` -- ordinary least squares on all monomials up to ``degree``
  (1: linear, 2: quadratic with interactions). No regularisation parameter: with fewer points
  than coefficients the minimum-norm solution is taken (``numpy.linalg.lstsq``).
* :class:`MlpRegressor` -- ``d -> hidden (tanh) -> targets`` in float64, full-batch Adam on the
  mean squared error of targets standardised with the training set's mean and spread, optional
  weight decay. Network helper functions follow ``static_transfer.surrogates``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Any

import numpy as np

__all__ = ["MlpRegressor", "MlpSettings", "PolynomialRegression", "Surrogate"]


class Surrogate(ABC):
    """Static map from scaled inputs ``(n, d)`` to targets ``(n, m)``."""

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> Surrogate:
        """Fit on scaled inputs and targets in physical units."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Targets in physical units."""


class PolynomialRegression(Surrogate):
    """Least squares on all monomials up to ``degree``."""

    def __init__(self, degree: int) -> None:
        if degree < 1:
            raise ValueError("degree must be at least 1")
        self.degree = degree
        self._coefficients: np.ndarray | None = None

    def features(self, x: np.ndarray) -> np.ndarray:
        """Design matrix: constant, then every monomial of degree 1 to ``degree``."""
        x = np.asarray(x, dtype=np.float64)
        columns = [np.ones(len(x))]
        for order in range(1, self.degree + 1):
            for combo in combinations_with_replacement(range(x.shape[1]), order):
                columns.append(np.prod(x[:, combo], axis=1))
        return np.column_stack(columns)

    def fit(self, x: np.ndarray, y: np.ndarray) -> PolynomialRegression:
        """Solve the least-squares problem (minimum norm if underdetermined)."""
        self._coefficients, *_ = np.linalg.lstsq(self.features(x), np.asarray(y), rcond=None)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the fitted polynomial."""
        if self._coefficients is None:
            raise RuntimeError("fit first")
        return np.asarray(self.features(x) @ self._coefficients)


@dataclass(frozen=True)
class MlpSettings:
    """Architecture and optimiser of one fit."""

    hidden: tuple[int, ...]
    learning_rate: float
    steps: int
    weight_decay: float

    def as_dict(self) -> dict[str, Any]:
        """Plain mapping for records."""
        return {
            "hidden": list(self.hidden),
            "learning_rate": self.learning_rate,
            "steps": self.steps,
            "weight_decay": self.weight_decay,
        }


class MlpRegressor(Surrogate):
    """Tanh MLP trained with full-batch Adam."""

    def __init__(self, settings: MlpSettings, seed: int) -> None:
        self.settings = settings
        self.seed = seed
        self._net: Any = None
        self._centre: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self.final_loss = float("nan")

    def fit(self, x: np.ndarray, y: np.ndarray) -> MlpRegressor:
        """Train from a seeded initialisation."""
        import torch
        from torch import nn

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._centre = y.mean(axis=0)
        spread = y.std(axis=0)
        self._scale = np.where(spread > 0.0, spread, 1.0)
        torch.manual_seed(self.seed)
        layers: list[nn.Module] = []
        width = x.shape[1]
        for h in self.settings.hidden:
            layers += [nn.Linear(width, h), nn.Tanh()]
            width = h
        layers.append(nn.Linear(width, y.shape[1]))
        net = nn.Sequential(*layers).double()
        xt = torch.as_tensor(x)
        zt = torch.as_tensor((y - self._centre) / self._scale)
        params = list(net.parameters())
        n_params = sum(p.numel() for p in params)
        opt = torch.optim.Adam(params, lr=self.settings.learning_rate)
        for _ in range(self.settings.steps):
            opt.zero_grad()
            loss = torch.mean((net(xt) - zt) ** 2)
            if self.settings.weight_decay > 0.0:
                reg = sum(torch.sum(p**2) for p in params)
                loss = loss + self.settings.weight_decay * reg / n_params
            loss.backward()  # type: ignore[no-untyped-call,unused-ignore]
            opt.step()
            self.final_loss = float(loss.detach())
        self._net = net
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the trained network."""
        import torch

        if self._net is None or self._centre is None or self._scale is None:
            raise RuntimeError("fit first")
        with torch.no_grad():
            z = self._net(torch.as_tensor(np.asarray(x, dtype=np.float64))).numpy()
        return np.asarray(z * self._scale + self._centre)

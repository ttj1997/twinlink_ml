"""Black-box surrogate models M1-M3: one regression per target (energy and every constraint).

All models see the joint input ``x = (u, w)`` mapped to the unit cube and return a mean and a
spread per target. The spread is what the conservative variants put on the constraints
(``g + kappa * sigma <= 0``):

M1 :class:`QuadraticSurface`
    full quadratic polynomial in ``x``, ridge per target with the penalty chosen by exact
    leave-one-out error; the spread is the leave-one-out RMSE of the target -- a FIXED margin,
    the same everywhere (the request: "fuer M1 fester Abstand").
M2 :class:`GaussianProcess`
    one GP per target, Matern kernel with one length scale per input (ARD), constant times kernel
    plus a small white noise for conditioning (the truth is noise-free); ``normalize_y``. The
    spread is the posterior standard deviation.
M3 :class:`MlpEnsemble`
    ``members`` multi-output MLPs on standardised targets, differing only in their initialisation;
    the spread is the standard deviation over the members.

There is no feasibility classifier: feasibility is always ``g_hat <= 0``.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from itertools import combinations_with_replacement
from typing import Any

import numpy as np

__all__ = ["GaussianProcess", "MlpEnsemble", "QuadraticSurface", "Surrogate", "make_surrogate"]


class Surrogate(ABC):
    """Regression ``x -> y`` with a spread per target; inputs in the unit cube."""

    name: str

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> Surrogate:
        """Fit on ``x (n, d)``, ``y (n, k)``."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and spread, both ``(n, k)``."""


def _quadratic_features(x: np.ndarray) -> np.ndarray:
    z = 2.0 * np.asarray(x, dtype=np.float64) - 1.0
    d = z.shape[1]
    cols = [z[:, i] for i in range(d)]
    cols += [z[:, i] * z[:, j] for i, j in combinations_with_replacement(range(d), 2)]
    return np.column_stack(cols)


class QuadraticSurface(Surrogate):
    """M1 (see the module doc)."""

    name = "M1"

    def __init__(self, alphas: np.ndarray) -> None:
        self.alphas = np.asarray(alphas, dtype=np.float64)

    def fit(self, x: np.ndarray, y: np.ndarray) -> QuadraticSurface:
        f = _quadratic_features(x)
        self._f_mean = f.mean(axis=0)
        self._f_std = np.where(f.std(axis=0) > 0.0, f.std(axis=0), 1.0)
        a = (f - self._f_mean) / self._f_std
        self._y_mean = y.mean(axis=0)
        yc = y - self._y_mean
        n = a.shape[0]
        u, s, vt = np.linalg.svd(a, full_matrices=False)
        k = y.shape[1]
        self._coef = np.zeros((a.shape[1], k))
        self.sigma = np.zeros(k)
        self.alpha = np.zeros(k)
        uty = u.T @ yc
        for j in range(k):
            best: tuple[float, float, np.ndarray] | None = None
            for alpha in self.alphas:
                shrink = s**2 / (s**2 + alpha)
                fitted = u @ (shrink * uty[:, j])
                # hat diagonal incl. the intercept (centred data): 1/n + sum_i u_ki^2 shrink_i
                h = 1.0 / n + (u**2) @ shrink
                loo = (yc[:, j] - fitted) / np.maximum(1.0 - h, 1e-12)
                mse = float(np.mean(loo**2))
                if best is None or mse < best[0]:
                    coef = vt.T @ (s / (s**2 + alpha) * uty[:, j])
                    best = (mse, float(alpha), coef)
            assert best is not None
            self.sigma[j], self.alpha[j], self._coef[:, j] = np.sqrt(best[0]), best[1], best[2]
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = (_quadratic_features(x) - self._f_mean) / self._f_std
        mean = a @ self._coef + self._y_mean
        return mean, np.broadcast_to(self.sigma, mean.shape)


class GaussianProcess(Surrogate):
    """M2 (see the module doc)."""

    name = "M2"

    def __init__(
        self,
        *,
        nu: float,
        length_scale_bounds: tuple[float, float],
        noise_level: float,
        noise_bounds: tuple[float, float],
        restarts: int,
        seed: int,
    ) -> None:
        self.settings: dict[str, Any] = {
            "nu": nu,
            "length_scale_bounds": length_scale_bounds,
            "noise_level": noise_level,
            "noise_bounds": noise_bounds,
            "restarts": restarts,
        }
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> GaussianProcess:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

        s = self.settings
        d = x.shape[1]
        self.models = []
        for j in range(y.shape[1]):
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                np.ones(d), s["length_scale_bounds"], nu=s["nu"]
            ) + WhiteKernel(s["noise_level"], s["noise_bounds"])
            gp = GaussianProcessRegressor(
                kernel, normalize_y=True, n_restarts_optimizer=s["restarts"],
                random_state=self.seed + 7919 * j,
            )  # fmt: skip
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(x, y[:, j])
            self.models.append(gp)
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = [gp.predict(x, return_std=True) for gp in self.models]
        return np.column_stack([m for m, _ in out]), np.column_stack([s for _, s in out])


class MlpEnsemble(Surrogate):
    """M3 (see the module doc)."""

    name = "M3"

    def __init__(
        self,
        *,
        members: int,
        hidden: tuple[int, ...],
        activation: str,
        alpha: float,
        max_iter: int,
        seed: int,
    ) -> None:
        self.members, self.hidden, self.activation = members, hidden, activation
        self.alpha, self.max_iter, self.seed = alpha, max_iter, seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> MlpEnsemble:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.neural_network import MLPRegressor

        self._y_mean = y.mean(axis=0)
        self._y_std = np.where(y.std(axis=0) > 0.0, y.std(axis=0), 1.0)
        target = (y - self._y_mean) / self._y_std
        self.nets = []
        for m in range(self.members):
            net = MLPRegressor(
                hidden_layer_sizes=self.hidden, activation=self.activation, solver="lbfgs",
                alpha=self.alpha, max_iter=self.max_iter, random_state=1000 * self.seed + m,
            )  # fmt: skip
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                net.fit(2.0 * x - 1.0, target)
            self.nets.append(net)
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = np.stack(
            [np.asarray(net.predict(2.0 * x - 1.0)).reshape(len(x), -1) for net in self.nets]
        )
        preds = preds * self._y_std + self._y_mean
        return preds.mean(axis=0), preds.std(axis=0, ddof=1)


def make_surrogate(key: str, settings: dict[str, Any], seed: int) -> Surrogate:
    """M1/M2/M3 from the ``models`` block of the study YAML."""
    if key == "M1":
        lo, hi, n = settings["alphas_log10"]
        return QuadraticSurface(np.logspace(float(lo), float(hi), int(n)))
    if key == "M2":
        return GaussianProcess(
            nu=float(settings["nu"]),
            length_scale_bounds=tuple(float(v) for v in settings["length_scale_bounds"]),  # type: ignore[arg-type]
            noise_level=float(settings["noise_level"]),
            noise_bounds=tuple(float(v) for v in settings["noise_bounds"]),  # type: ignore[arg-type]
            restarts=int(settings["restarts"]),
            seed=seed,
        )
    if key == "M3":
        return MlpEnsemble(
            members=int(settings["members"]),
            hidden=tuple(int(h) for h in settings["hidden"]),
            activation=str(settings["activation"]),
            alpha=float(settings["alpha"]),
            max_iter=int(settings["max_iter"]),
            seed=seed,
        )
    raise ValueError(f"unknown surrogate {key!r}")

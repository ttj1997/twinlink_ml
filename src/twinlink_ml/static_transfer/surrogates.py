"""Methods of the static transfer study behind one ABC, :class:`StaticSurrogate`.

========  ========================================================================================
L0        the partner balance (:func:`.balance.source_model`); also with the v4.1_2 input set
L1        the balance with four calibrated physical parameters (heater efficiency, heat loss per
          kg/h of air, false-air share into the exhaust, constant product-moisture excess);
          bounded MAP: residuals weighted by the primary noise, Gaussian prior around the nominal
          values with ``prior_scale`` x the prior sigmas of the config
L2        L0 plus a residual MLP (plant - balance), last layer initialised at zero, weight decay
L3        MLP pretrained on balance points, fine-tuned on the plant points (all parameters,
          L2-SP towards the pretrained weights); ``L3_0`` is the pretrained net without
          fine-tuning
L4        the same MLP from scratch, weight decay
B1        L0 plus a constant bias per target (mean training residual): a residual of degree 0
B2        linear ridge on the plant points, penalty per target by exact leave-one-out error
========  ========================================================================================

Scaling (decision E3, docs/37): inputs are mapped from the PRETRAINING box to ``[-1, 1]`` and the
targets by fixed centres and scales of the config -- a-priori constants, no statistic of any
data, the same for L2, L3 and L4; so no information from test points enters a fit. B2 centres its
features and targets on the training points only.
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .balance import ConstantEfficiency, ConstantMoisture, DryerPlantModel
from .config import Box, NoiseSpec, SourceSpec

__all__ = [
    "BalanceSurrogate",
    "BiasCorrectedBalance",
    "CalibratedBalance",
    "FineTunedMlp",
    "LinearRidge",
    "MlpSettings",
    "PretrainedMlp",
    "ResidualMlp",
    "ScratchMlp",
    "StaticSurrogate",
    "TargetScaling",
    "make_net",
    "pretrain",
]


class StaticSurrogate(ABC):
    """Regression from measured inputs ``x (n, 5)`` to targets ``y (n, 3)``."""

    name: str

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> StaticSurrogate: ...

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray: ...

    def info(self) -> dict[str, Any]:
        """Fitted quantities worth recording (parameters, iterations)."""
        return {}


# ------------------------------------------------------------------------------ physics based
@dataclass
class BalanceSurrogate(StaticSurrogate):
    """L0: the balance, no data used."""

    model: DryerPlantModel
    name: str = "L0"

    def fit(
        self,
        x: np.ndarray,  # noqa: ARG002 - shared interface, no data used
        y: np.ndarray,  # noqa: ARG002
    ) -> BalanceSurrogate:
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


@dataclass
class BiasCorrectedBalance(StaticSurrogate):
    """B1: L0 plus the mean training residual per target."""

    model: DryerPlantModel
    name: str = "B1"

    def fit(self, x: np.ndarray, y: np.ndarray) -> BiasCorrectedBalance:
        self.bias = np.mean(np.asarray(y) - self.model.predict(x), axis=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(x) + self.bias)

    def info(self) -> dict[str, Any]:
        return {"bias": self.bias.tolist()}


CALIBRATED_PARAMETERS: tuple[str, ...] = (
    "heater_efficiency",
    "heat_loss_w_per_k_per_kg_h",
    "false_air_share",
    "moisture_excess_kg_per_kg",
)


def calibrated_model(source: SourceSpec, theta: np.ndarray) -> DryerPlantModel:
    """The balance with the four L1 parameters (order :data:`CALIBRATED_PARAMETERS`)."""
    eta, ua, false_share, excess = (float(v) for v in theta)
    return DryerPlantModel(
        source=source,
        efficiency=ConstantEfficiency(eta),
        moisture=ConstantMoisture(excess),
        heat_loss_w_per_k_per_kg_h=ua,
        false_air_share=false_share,
    )


@dataclass
class CalibratedBalance(StaticSurrogate):
    """L1 (see the module doc)."""

    source: SourceSpec
    parameters: dict[str, tuple[float, float, float, float]]
    """name -> (nominal, lower, upper, prior sigma), order :data:`CALIBRATED_PARAMETERS`."""
    noise: NoiseSpec
    prior_scale: float
    name: str = "L1"
    max_nfev: int | None = None
    """Cap of the solver's function evaluations (dry run only; ``None`` = solver default)."""

    def _arrays(self) -> tuple[np.ndarray, ...]:
        table = np.asarray([self.parameters[k] for k in CALIBRATED_PARAMETERS], dtype=np.float64)
        return table[:, 0], table[:, 1], table[:, 2], table[:, 3]

    def fit(self, x: np.ndarray, y: np.ndarray) -> CalibratedBalance:
        nominal, lower, upper, prior_sigma = self._arrays()
        sigma = self.noise.sigma(y)
        width = self.prior_scale * prior_sigma

        def residuals(theta: np.ndarray) -> np.ndarray:
            pred = calibrated_model(self.source, theta).predict(x)
            return np.concatenate([((y - pred) / sigma).ravel(), (theta - nominal) / width])

        span = upper - lower
        start = np.clip(nominal, lower + 1e-6 * span, upper - 1e-6 * span)
        result = least_squares(
            residuals,
            start,
            bounds=(lower, upper),
            x_scale=prior_sigma,
            method="trf",
            max_nfev=self.max_nfev,
        )
        self.theta = np.asarray(result.x)
        self._model = calibrated_model(self.source, self.theta)
        self._result = {"nfev": int(result.nfev), "status": int(result.status)}
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x)

    def info(self) -> dict[str, Any]:
        return {
            "parameters": dict(zip(CALIBRATED_PARAMETERS, self.theta.tolist(), strict=True)),
            **self._result,
        }


@dataclass
class LinearRidge(StaticSurrogate):
    """B2: linear ridge per target, penalty by exact leave-one-out error."""

    box: Box
    alphas: np.ndarray
    name: str = "B2"

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRidge:
        f = self.box.to_unit_symmetric(x)
        self._f_mean = f.mean(axis=0)
        a = f - self._f_mean
        self._y_mean = np.mean(y, axis=0)
        yc = np.asarray(y) - self._y_mean
        n = len(a)
        u, s, vt = np.linalg.svd(a, full_matrices=False)
        uty = u.T @ yc
        self._coef = np.zeros((a.shape[1], y.shape[1]))
        self.alpha = np.zeros(y.shape[1])
        for j in range(y.shape[1]):
            best: tuple[float, float, np.ndarray] | None = None
            for alpha in self.alphas:
                shrink = s**2 / (s**2 + alpha)
                fitted = u @ (shrink * uty[:, j])
                h = 1.0 / n + (u**2) @ shrink
                loo = (yc[:, j] - fitted) / np.maximum(1.0 - h, 1e-12)
                mse = float(np.mean(loo**2))
                if best is None or mse < best[0]:
                    best = (mse, float(alpha), vt.T @ (s / (s**2 + alpha) * uty[:, j]))
            assert best is not None
            self.alpha[j], self._coef[:, j] = best[1], best[2]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        a = self.box.to_unit_symmetric(x) - self._f_mean
        return np.asarray(a @ self._coef + self._y_mean)

    def info(self) -> dict[str, Any]:
        return {"alpha": self.alpha.tolist()}


# ------------------------------------------------------------------------------------- MLPs
@dataclass(frozen=True)
class TargetScaling:
    """Fixed a-priori input box and target centre/scale (no statistic of data)."""

    box: Box
    centre: np.ndarray
    scale: np.ndarray

    def x(self, x: np.ndarray) -> np.ndarray:
        return self.box.to_unit_symmetric(x)

    def y(self, y: np.ndarray) -> np.ndarray:
        return np.asarray((np.asarray(y) - self.centre) / self.scale)

    def y_back(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(np.asarray(z) * self.scale + self.centre)


@dataclass(frozen=True)
class MlpSettings:
    """Optimiser of one fit (full batch Adam)."""

    learning_rate: float
    steps: int
    penalty: float
    """Weight decay (towards 0) for L2/L4, L2-SP (towards the pretrained weights) for L3."""


def make_net(hidden: tuple[int, ...], seed: int) -> Any:
    """``5 -> hidden (tanh) -> 3`` in float64, initialised from ``seed``."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    layers: list[nn.Module] = []
    width = 5
    for h in hidden:
        layers += [nn.Linear(width, h), nn.Tanh()]
        width = h
    layers.append(nn.Linear(width, 3))
    return nn.Sequential(*layers).double()


def _anchor(net: Any) -> list[Any]:
    return [p.detach().clone() for p in net.parameters()]


def train_full_batch(
    net: Any, x: np.ndarray, z: np.ndarray, settings: MlpSettings, anchor: list[Any] | None
) -> float:
    """Full-batch Adam on mean squared error of scaled targets plus the penalty; final loss."""
    import torch

    xt = torch.as_tensor(x, dtype=torch.float64)
    zt = torch.as_tensor(z, dtype=torch.float64)
    params = list(net.parameters())
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.Adam(params, lr=settings.learning_rate)
    loss_value = math.nan
    for _ in range(settings.steps):
        opt.zero_grad()
        loss = torch.mean((net(xt) - zt) ** 2)
        if settings.penalty > 0.0:
            if anchor is None:
                reg = sum(torch.sum(p**2) for p in params)
            else:
                reg = sum(torch.sum((p - a) ** 2) for p, a in zip(params, anchor, strict=True))
            loss = loss + settings.penalty * reg / n_params
        loss.backward()  # type: ignore[no-untyped-call,unused-ignore]
        opt.step()
        loss_value = float(loss.detach())
    return loss_value


def _predict(net: Any, x: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return np.asarray(net(torch.as_tensor(x, dtype=torch.float64)).numpy())


@dataclass
class ScratchMlp(StaticSurrogate):
    """L4: MLP from scratch on the plant points."""

    scaling: TargetScaling
    hidden: tuple[int, ...]
    settings: MlpSettings
    seed: int
    name: str = "L4"

    def fit(self, x: np.ndarray, y: np.ndarray) -> ScratchMlp:
        self.net = make_net(self.hidden, self.seed)
        self.loss = train_full_batch(
            self.net, self.scaling.x(x), self.scaling.y(y), self.settings, None
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.scaling.y_back(_predict(self.net, self.scaling.x(x)))

    def info(self) -> dict[str, Any]:
        return {"final_loss": self.loss}


@dataclass
class ResidualMlp(StaticSurrogate):
    """L2: balance plus a residual MLP that starts at zero."""

    model: DryerPlantModel
    box: Box
    residual_scale: np.ndarray
    hidden: tuple[int, ...]
    settings: MlpSettings
    seed: int
    name: str = "L2"

    def fit(self, x: np.ndarray, y: np.ndarray) -> ResidualMlp:
        import torch

        self.net = make_net(self.hidden, self.seed)
        last = list(self.net.children())[-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
        r = (np.asarray(y) - self.model.predict(x)) / self.residual_scale
        self.loss = train_full_batch(
            self.net, self.box.to_unit_symmetric(x), r, self.settings, None
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        r = _predict(self.net, self.box.to_unit_symmetric(x)) * self.residual_scale
        return np.asarray(self.model.predict(x) + r)

    def info(self) -> dict[str, Any]:
        return {"final_loss": self.loss}


@dataclass
class PretrainedMlp(StaticSurrogate):
    """L3_0: the pretrained net without plant data."""

    scaling: TargetScaling
    net: Any
    name: str = "L3_0"

    def fit(
        self,
        x: np.ndarray,  # noqa: ARG002 - shared interface, no data used
        y: np.ndarray,  # noqa: ARG002
    ) -> PretrainedMlp:
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.scaling.y_back(_predict(self.net, self.scaling.x(x)))


@dataclass
class FineTunedMlp(StaticSurrogate):
    """L3: copy of the pretrained net, all parameters fine-tuned with L2-SP."""

    scaling: TargetScaling
    pretrained: Any
    settings: MlpSettings
    name: str = "L3"

    def fit(self, x: np.ndarray, y: np.ndarray) -> FineTunedMlp:
        self.net = copy.deepcopy(self.pretrained)
        anchor = _anchor(self.pretrained)
        self.loss = train_full_batch(
            self.net, self.scaling.x(x), self.scaling.y(y), self.settings, anchor
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.scaling.y_back(_predict(self.net, self.scaling.x(x)))

    def info(self) -> dict[str, Any]:
        return {"final_loss": self.loss}


def pretrain(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    scaling: TargetScaling,
    hidden: tuple[int, ...],
    *,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Minibatch Adam on balance points with early stopping on the validation MSE (best kept)."""
    import torch

    net = make_net(hidden, seed)
    xt = torch.as_tensor(scaling.x(x_train), dtype=torch.float64)
    zt = torch.as_tensor(scaling.y(y_train), dtype=torch.float64)
    xv = torch.as_tensor(scaling.x(x_val), dtype=torch.float64)
    zv = torch.as_tensor(scaling.y(y_val), dtype=torch.float64)
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    gen = torch.Generator().manual_seed(seed)
    best = (math.inf, 0, copy.deepcopy(net.state_dict()))
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        perm = torch.randperm(len(xt), generator=gen)
        for start in range(0, len(xt), batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            loss = torch.mean((net(xt[idx]) - zt[idx]) ** 2)
            loss.backward()  # type: ignore[no-untyped-call,unused-ignore]
            opt.step()
        with torch.no_grad():
            val = float(torch.mean((net(xv) - zv) ** 2))
        if val < best[0]:
            best = (val, epoch, copy.deepcopy(net.state_dict()))
        elif epoch - best[1] >= patience:
            break
    net.load_state_dict(best[2])
    val_rmse = np.sqrt(
        np.mean((scaling.y_back(_predict(net, scaling.x(x_val))) - y_val) ** 2, axis=0)
    )
    return net, {
        "epochs": epoch,
        "best_epoch": best[1],
        "best_validation_mse_scaled": best[0],
        "validation_rmse_physical": val_rmse.tolist(),
    }

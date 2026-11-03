r"""M2: refit of the drying kinetics of the grey-box model on the training runs.

As M4 of Befund 107 (``surrogate_opt.study.fit_greybox``): least squares of the fitted parameters
on all outputs, each scaled by its spread in the data, ``scipy.optimize.least_squares`` with
``x_scale="jac"``, ``diff_step`` and ``max_nfev`` from the study config, start at the nominal
values, box bounds. Fitted here: critical moisture and ``nu_max`` (``max_saturation_degree``); the
five GAB parameters of the truth stay hidden (nominal). Data: the measured targets (energy,
moisture, bed temperature) of the ``n_train`` training runs, model: the open-loop replay of the same
runs (:func:`.residual.replay`). A parameter set whose material fails the isotherm check, or whose
replay raises, gets the residual 1e3 everywhere (as Befund 107) and is counted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from twinlink_ml.evaluation.experiments import GeneratedData
from twinlink_ml.sensor_value.draws import truth_config
from twinlink_ml.simulation import ProcessConfig
from twinlink_ml.simulation.sorption import validate_isotherm_against_critical_moisture

from .residual import replay

__all__ = ["RefitSettings", "fit_kinetics", "refit_config"]


@dataclass(frozen=True, slots=True)
class RefitSettings:
    """Fitted parameters with bounds; least-squares settings (fixed before the run)."""

    parameters: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    max_nfev: int
    diff_step: float

    def __post_init__(self) -> None:
        if not (len(self.parameters) == len(self.lower) == len(self.upper)):
            raise ValueError("one bound per parameter")


def refit_config(
    nominal: ProcessConfig, names: Sequence[str], theta: Sequence[float]
) -> ProcessConfig:
    """Nominal config with the fitted material parameters replaced."""
    return truth_config(nominal, {n: float(v) for n, v in zip(names, theta, strict=True)})


def fit_kinetics(
    nominal: ProcessConfig,
    data: GeneratedData,
    rows: list[int],
    target_names: Sequence[str],
    settings: RefitSettings,
    *,
    use_noise_free: bool = False,
) -> dict[str, Any]:
    """Fit on the runs ``rows`` of ``data``; returns theta, cost, status and counters."""
    names = list(data.trajectory.metadata["output_names"])
    cols = [names.index(t) for t in target_names]
    source = data.trajectory.outputs if use_noise_free else data.trajectory.measurements
    y = source[rows][..., cols]
    flat = y.reshape(-1, len(cols))
    scale = np.maximum(flat.std(axis=0), 1e-6 * np.maximum(np.abs(flat).mean(axis=0), 1e-9))
    failures = 0
    evaluations = 0

    def residual(theta: np.ndarray) -> np.ndarray:
        nonlocal failures, evaluations
        evaluations += 1
        try:
            config = refit_config(nominal, settings.parameters, theta)
            validate_isotherm_against_critical_moisture(config)
            prediction = replay(data, config, rows).outputs[..., cols]
        except (ValueError, RuntimeError, FloatingPointError):
            failures += 1
            return np.full(y.size, 1e3)
        out: np.ndarray = ((prediction - y) / scale).ravel()
        if not np.all(np.isfinite(out)):
            failures += 1
            return np.full(y.size, 1e3)
        return out

    x0 = np.array([getattr(nominal.material, p) for p in settings.parameters], dtype=np.float64)
    res = least_squares(
        residual,
        x0,
        bounds=(list(settings.lower), list(settings.upper)),
        x_scale="jac",
        diff_step=settings.diff_step,
        max_nfev=settings.max_nfev,
    )
    return {
        "theta": [float(v) for v in res.x],
        "names": list(settings.parameters),
        "start": [float(v) for v in x0],
        "cost": float(res.cost),
        "status": int(res.status),
        "message": str(res.message),
        "nfev": int(res.nfev),
        "n_residual_calls": evaluations,
        "failures": failures,
        "rms_scaled_residual": float(np.sqrt(np.mean(res.fun**2))),
        "scale": [float(v) for v in scale],
    }

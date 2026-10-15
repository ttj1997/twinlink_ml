r"""Calibration on one plant day, validation on the other (never inside a day), baselines.

Split unit = day. :func:`require_different_days` stops a calibration/validation pair that is the
same day; there is no window-level or random split anywhere in this module.

Two model parts are evaluated, because the storage state itself cannot be validated like the rest:

* the EXTRACTION part (exchanger A1, preheater loop, preheater A1): with the measured storage
  temperature, loop flows, air flow and ambient temperature as inputs it predicts supply and return
  temperature of the preheater loop and hence ``Q_out`` and the steam share. Its "24 h rollout" is
  an open-loop run over the whole day from measured exogenous inputs (the part has no state).
  The two conductances are fitted on the calibration day.
* the STORAGE state ``T_S``: a true 24 h rollout from the initial state. ``Q_in`` is not measured,
  so the rollout needs an assumption; two are run and labelled: (a) ``Q_in`` constant = the mean
  reconstructed on the calibration day (a prediction), (b) ``Q_in`` reconstructed from the
  validation day itself (an ORACLE: shows only what the model error in ``Q_out`` adds up to, not
  a prediction).

Baselines (mandatory): persistence (initial value of the validation day) and a linear ARX
(``y_t = a y_{t-1} + b^T x_t + c`` fitted on the calibration day, free-run on the validation day,
fed with the same measured exogenous inputs the physical model gets).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from .balance import air_heating_kw, steam_share
from .config import ExchangerConfig, StorageSystemConfig
from .identifiability import heat_out_kw, smoothed_derivative_k_per_s
from .model import ExtractionChain, ExtractionResult
from .roles import PlantDay

__all__ = [
    "ArxModel",
    "ExtractionFit",
    "arx_free_run",
    "error_metrics",
    "fit_arx",
    "fit_conductances",
    "mean_q_in_kw",
    "oracle_q_in_kw",
    "persistence",
    "predict_extraction",
    "repair_glitches",
    "require_different_days",
    "rollout_storage",
]

FloatArray = NDArray[np.float64]
DERIVATIVE_WINDOW = 15
"""Samples of the moving average behind ``dT_S/dt`` (used by topology A and by ``Q_in``)."""


FLOW_ROLES = ("storage_loop_flow", "preheater_loop_flow", "process_air_flow")


def repair_glitches(
    day: PlantDay, window: int = 9, rel_tol: float = 0.2
) -> tuple[PlantDay, dict[str, int]]:
    """Replace single-sample dropouts/spikes of the flow channels by linear interpolation.

    A sample is a glitch if it deviates from the running median (``window`` samples) by more than
    ``rel_tol`` of that median. Data-quality rule, not a model choice: one product-2 sample of the
    preheater loop flow drops to half its value for one minute while both temperatures stay
    smooth, and a static model fitted with that sample is distorted. Returns the repaired day and
    the number of repaired samples per role (to be reported).
    """
    from scipy.ndimage import median_filter

    roles = dict(day.roles)
    counts: dict[str, int] = {}
    x = np.arange(len(day.time_s))
    for role in FLOW_ROLES:
        values = day[role]
        med = median_filter(values, size=window, mode="nearest")
        bad = np.abs(values - med) > rel_tol * np.abs(med)
        counts[role] = int(np.count_nonzero(bad))
        if bad.any():
            fixed = values.copy()
            fixed[bad] = np.interp(x[bad], x[~bad], values[~bad])
            roles[role] = fixed
    return PlantDay(label=day.label, time_s=day.time_s, roles=roles), counts


def require_different_days(calibration: PlantDay, validation: PlantDay) -> None:
    """Split rule: calibrate and validate on different days, never within one."""
    if calibration.label == validation.label:
        raise ValueError(f"calibration and validation are the same day ({calibration.label!r})")


def error_metrics(prediction: FloatArray, reference: FloatArray) -> dict[str, float]:
    """RMSE and maximum absolute error, in the unit of the quantity."""
    err = np.asarray(prediction) - np.asarray(reference)
    return {"rmse": float(np.sqrt(np.mean(err**2))), "max_abs": float(np.max(np.abs(err)))}


def persistence(reference: FloatArray) -> FloatArray:
    """Initial value held over the whole day."""
    return np.full_like(np.asarray(reference, dtype=np.float64), reference[0])


# -- extraction part ---------------------------------------------------------------------------
def predict_extraction(
    cfg: StorageSystemConfig, day: PlantDay, window: int = DERIVATIVE_WINDOW
) -> tuple[slice, ExtractionResult]:
    """Run the extraction chain over a day from measured inputs.

    Returns the slice of the day's samples the result refers to (the derivative window trims
    the edges; the same slice for both topologies so results are comparable).
    """
    chain = ExtractionChain.from_config(cfg)
    index, d_t = smoothed_derivative_k_per_s(day["storage_temperature"], day.step_s, window)
    result = chain.solve_given_storage_rate(
        day["storage_temperature"][index],
        d_t,
        day["storage_loop_flow"][index],
        day["preheater_loop_flow"][index],
        day["process_air_flow"][index],
        day["ambient_temperature"][index],
    )
    return index, result


@dataclass(frozen=True, slots=True)
class ExtractionFit:
    """Result of the conductance fit on one day."""

    config: StorageSystemConfig
    ua_exchanger_kw_per_k: float
    ua_preheater_kw_per_k: float
    rmse_supply_k: float
    rmse_return_k: float
    success: bool


def _with_conductances(cfg: StorageSystemConfig, ua1: float, ua2: float) -> StorageSystemConfig:
    return replace(
        cfg,
        storage_exchanger=ExchangerConfig(ua1, cfg.storage_exchanger.minimum_approach_k),
        preheater=ExchangerConfig(ua2, cfg.preheater.minimum_approach_k),
    )


def fit_conductances(
    cfg: StorageSystemConfig, day: PlantDay, window: int = DERIVATIVE_WINDOW
) -> ExtractionFit:
    """Least squares on supply and return temperature (K) over one day, log-parameters."""

    def residual(log_ua: FloatArray) -> FloatArray:
        trial = _with_conductances(cfg, float(np.exp(log_ua[0])), float(np.exp(log_ua[1])))
        index, res = predict_extraction(trial, day, window)
        return np.concatenate(
            [
                res.t_supply_celsius - day["preheater_supply_temperature"][index],
                res.t_return_celsius - day["preheater_return_temperature"][index],
            ]
        )

    start = np.log([cfg.storage_exchanger.ua_kw_per_k, cfg.preheater.ua_kw_per_k])
    sol = least_squares(residual, start, method="lm")
    fitted = _with_conductances(cfg, float(np.exp(sol.x[0])), float(np.exp(sol.x[1])))
    r = residual(sol.x)
    n = len(r) // 2
    return ExtractionFit(
        config=fitted,
        ua_exchanger_kw_per_k=fitted.storage_exchanger.ua_kw_per_k,
        ua_preheater_kw_per_k=fitted.preheater.ua_kw_per_k,
        rmse_supply_k=float(np.sqrt(np.mean(r[:n] ** 2))),
        rmse_return_k=float(np.sqrt(np.mean(r[n:] ** 2))),
        success=bool(sol.success),
    )


# -- storage state -----------------------------------------------------------------------------
def mean_q_in_kw(cfg: StorageSystemConfig, day: PlantDay, window: int = DERIVATIVE_WINDOW) -> float:
    """Mean reconstructed compressor heat of a day for the volume in ``cfg`` (storage balance)."""
    index, d_t = smoothed_derivative_k_per_s(day["storage_temperature"], day.step_s, window)
    stored = cfg.storage.volume_m3 * cfg.water.rho_cp_kj_per_m3_k * d_t
    t_s = day["storage_temperature"][index]
    loss = cfg.storage.ua_kw_per_k * (t_s - cfg.storage.room_temperature_celsius)
    return float(np.mean(stored + heat_out_kw(day, cfg)[index] + loss))


def oracle_q_in_kw(
    cfg: StorageSystemConfig, day: PlantDay, window: int = DERIVATIVE_WINDOW
) -> FloatArray:
    """``Q_in(t)`` from the day's own measurements, on all samples (edges held constant)."""
    index, d_t = smoothed_derivative_k_per_s(day["storage_temperature"], day.step_s, window)
    stored = cfg.storage.volume_m3 * cfg.water.rho_cp_kj_per_m3_k * d_t
    t_s = day["storage_temperature"][index]
    loss = cfg.storage.ua_kw_per_k * (t_s - cfg.storage.room_temperature_celsius)
    inner = stored + heat_out_kw(day, cfg)[index] + loss
    out = np.empty(len(day.time_s))
    out[index] = inner
    out[: index.start] = inner[0]
    out[index.stop :] = inner[-1]
    return out


def rollout_storage(
    cfg: StorageSystemConfig, day: PlantDay, q_in_kw: FloatArray | float
) -> tuple[FloatArray, FloatArray]:
    """24 h rollout of ``T_S`` from the first measured value (Heun scheme, inputs of the day).

    ``q_in_kw``: scalar (constant) or one value per sample. Returns ``T_S(t)`` and ``Q_out(t)``.
    """
    chain = ExtractionChain.from_config(cfg)
    n = len(day.time_s)
    q_in = np.broadcast_to(np.asarray(q_in_kw, dtype=np.float64), (n,))
    dt = day.step_s
    v1, v2 = day["storage_loop_flow"], day["preheater_loop_flow"]
    af, ta = day["process_air_flow"], day["ambient_temperature"]
    t_s = np.empty(n)
    q_out = np.empty(n)
    t_s[0] = day["storage_temperature"][0]

    def rhs(t_state: float, k: int) -> tuple[float, float]:
        res = chain.solve_given_q_in(
            np.array([t_state]),
            q_in[k : k + 1],
            v1[k : k + 1],
            v2[k : k + 1],
            af[k : k + 1],
            ta[k : k + 1],
        )
        d = chain.storage.derivative_k_per_s(np.array([t_state]), q_in[k : k + 1], res.q_out_kw)
        return float(d[0]), float(res.q_out_kw[0])

    for k in range(n - 1):
        d0, q_out[k] = rhs(t_s[k], k)
        predictor = t_s[k] + dt * d0
        d1, _ = rhs(predictor, k + 1)
        t_s[k + 1] = t_s[k] + 0.5 * dt * (d0 + d1)
    q_out[-1] = rhs(t_s[-1], n - 1)[1]
    return t_s, q_out


# -- ARX baseline ------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ArxModel:
    """``y_t = a y_{t-1} + b^T x_t + c`` on standardised columns (ridge-regularised)."""

    a: float
    b: FloatArray
    c: float
    x_mean: FloatArray
    x_std: FloatArray


def fit_arx(y: FloatArray, x: FloatArray, ridge: float = 1e-6) -> ArxModel:
    """Fit on one day. ``x`` has shape (samples, n_inputs)."""
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std = np.where(x_std < 1e-12, 1.0, x_std)
    xs = (x - x_mean) / x_std
    design = np.column_stack([y[:-1], xs[1:], np.ones(len(y) - 1)])
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    coef = np.linalg.solve(gram, design.T @ y[1:])
    return ArxModel(a=float(coef[0]), b=coef[1:-1], c=float(coef[-1]), x_mean=x_mean, x_std=x_std)


def arx_free_run(model: ArxModel, y0: float, x: FloatArray) -> FloatArray:
    """Recursive prediction from ``y0`` with the measured inputs ``x``."""
    xs = (x - model.x_mean) / model.x_std
    out = np.empty(len(x))
    out[0] = y0
    for k in range(1, len(x)):
        out[k] = model.a * out[k - 1] + float(model.b @ xs[k]) + model.c
    return out


def steam_share_from_q_out(
    cfg: StorageSystemConfig, day: PlantDay, q_out_kw: FloatArray
) -> FloatArray:
    """Steam share of the air heating for a given ``Q_out`` (one value per sample)."""
    return steam_share(air_heating_kw(day, cfg), q_out_kw)

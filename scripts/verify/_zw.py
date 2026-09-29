"""Shared helpers for the ZW series (audit of Befund 65 GA2).

Reads the raw predictions written by ``zw1_raw_predictions.py`` and offers the estimators the ZW
scripts use. Imports no GA2 code (``scripts/probe_soft_sensor_*``) and no project evaluation code.

Point layout: one point = one (window, forecast step). ``truth`` is the noise-free moisture (the
quantity classes are formed on, as in GA2), ``measured`` the noisy reference the error metric of
the project is taken against, ``pred`` the network output. ``run`` is the index into the 32
holdout test runs (identical for all sets, so class membership by truth is identical for all).
"""

from __future__ import annotations

import numpy as np
from _common import RUNS_DIR

RAW_DIR = RUNS_DIR / "raw_zw1"
ARMS = ("excited", "constant")
SEEDS = (1, 2, 3, 4, 5)
H = 36
N_RUNS = 32


def load(arm: str, seed: int, *, step: int | None = None) -> dict[str, np.ndarray]:
    """Points of one (arm, seed); ``step`` = None pools steps 1..36, ``step=1`` keeps only step 1."""
    z = np.load(RAW_DIR / f"{arm}_seed{seed}.npz")
    cols = slice(None) if step is None else slice(step - 1, step)
    n_steps = z["truth"][:, cols].shape[1]
    return {
        "truth": z["truth"][:, cols].reshape(-1),
        "measured": z["measured"][:, cols].reshape(-1),
        "pred": z["pred"][:, cols].reshape(-1),
        "run": np.repeat(z["run"].astype(np.int64), n_steps),
        "window": np.repeat(np.arange(len(z["run"])), n_steps),
    }


def in_class(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (values >= lo) & (values < hi)


def rmse_pooled(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2))) if err.size else float("nan")


def per_run_rmse(err: np.ndarray, run: np.ndarray) -> np.ndarray:
    """RMSE of each run present in ``run`` (runs without points are absent)."""
    return np.asarray([np.sqrt(np.mean(err[run == r] ** 2)) for r in np.unique(run)])


def rmse_run_median(err: np.ndarray, run: np.ndarray) -> float:
    return float(np.median(per_run_rmse(err, run))) if err.size else float("nan")


def run_stats(err: np.ndarray, run: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-run sum of squared error and point count (length ``N_RUNS``), for fast bootstraps."""
    sse = np.bincount(run, weights=err**2, minlength=N_RUNS)
    cnt = np.bincount(run, minlength=N_RUNS).astype(float)
    return sse, cnt


def quotients_from_run_stats(
    sse_l: np.ndarray, cnt_l: np.ndarray, sse_g: np.ndarray, cnt_g: np.ndarray, idx: np.ndarray
) -> tuple[float, float]:
    """(pooled/pooled, run-median/run-median) quotient over the runs ``idx`` (with repeats)."""
    cl, cg = cnt_l[idx], cnt_g[idx]
    if cl.sum() == 0:
        return float("nan"), float("nan")
    pooled = np.sqrt(sse_l[idx].sum() / cl.sum()) / np.sqrt(sse_g[idx].sum() / cg.sum())
    have = cl > 0
    med_l = np.median(np.sqrt(sse_l[idx][have] / cl[have]))
    med_g = np.median(np.sqrt(sse_g[idx] / cg))
    return float(pooled), float(med_l / med_g)

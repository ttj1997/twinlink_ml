r"""Model-free measures of the information density of an excitation (B1.1).

Two families, both computed on the *normalised* actions ``(u - lower) / span`` in ``[0, 1]``
so that channels are comparable and the numbers do not depend on the (placeholder) engineering
ranges.

1. **Spectral coverage** (:func:`spectral_coverage`). Pooled over the runs, per channel: the
   periodogram of the demeaned run in ``n_bands`` logarithmic bands between ``band_low_hz`` and
   ``band_high_hz`` (the corner frequencies of the slowest and fastest measured mode, see
   ``excitation_plans``). A band is *covered* when its power density is at least
   ``threshold_fraction`` (default 1 %) of the density of a white signal with the variance of
   a uniform signal on ``[0, 1]`` (1/12). The reference is absolute, not relative to the
   design, so that a design with little power everywhere does not look flat. Also reported:
   the spectral flatness of the band powers (geometric / arithmetic mean) and the largest
   absolute pooled correlation between two channels (demeaned per run).

2. **Fisher information of a linear model** (:func:`fir_fisher_information`,
   :func:`arx_fisher_information`). For Gaussian noise of variance :math:`\sigma^2` the Fisher
   information of the regression :math:`y = \Phi\theta + e` is :math:`\Phi^\top\Phi/\sigma^2`; it
   is additive over runs. :math:`\sigma^2 = 1` is used: the numbers compare designs at equal
   :math:`N`, they are not absolute.

   * *FIR (inputs only)*: regressors are the lagged normalised inputs (``n_lags`` lags per
     channel, centred at 0.5) and a bias. Needs no simulation, hence usable *before* a
     candidate is run (the basis of the active-learning acquisition) and structurally free of
     the truth of the candidate.
   * *ARX (needs outputs)*: the regressors of ``LinearARXModel`` (``p`` target lags,
     ``q + 1`` input lags, bias) with the simulated normalised outputs. An analysis tool only;
     never an acquisition input.

   Reported: ``logdet_per_parameter`` (D-criterion per parameter), ``lambda_min`` (the least
   informed direction), ``condition`` (cross-channel confounding) and the parameter count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import periodogram

FloatArray = NDArray[np.float64]

__all__ = [
    "FisherReport",
    "SpectralReport",
    "arx_fisher_information",
    "fir_fisher_information",
    "normalise_actions",
    "spectral_coverage",
]

_WHITE_UNIFORM_VARIANCE = 1.0 / 12.0
_FIM_RIDGE = 1e-9


def normalise_actions(actions: FloatArray, lower: FloatArray, span: FloatArray) -> FloatArray:
    """Actions ``(runs, steps, channels)`` to ``[0, 1]`` using the action bounds."""
    return np.asarray((actions - lower[None, None, :]) / span[None, None, :], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SpectralReport:
    """Spectral coverage of one design, pooled over its runs."""

    band_edges_hz: tuple[float, ...]
    band_density: tuple[tuple[float, ...], ...]
    """Pooled one-sided power density per channel and band, normalised actions."""
    coverage_per_channel: tuple[float, ...]
    flatness_per_channel: tuple[float, ...]
    max_abs_cross_correlation: float

    @property
    def coverage(self) -> float:
        return float(np.mean(self.coverage_per_channel))

    @property
    def flatness(self) -> float:
        return float(np.mean(self.flatness_per_channel))


def spectral_coverage(
    normalised_actions: FloatArray,
    sample_time_s: float,
    *,
    band_low_hz: float,
    band_high_hz: float,
    n_bands: int = 20,
    threshold_fraction: float = 0.01,
) -> SpectralReport:
    """Coverage of the band ``[band_low_hz, band_high_hz]``, see module docstring."""
    if normalised_actions.ndim != 3:
        raise ValueError("normalised_actions needs the shape (runs, steps, channels)")
    n_runs, _, n_channels = normalised_actions.shape
    edges = np.geomspace(band_low_hz, band_high_hz, n_bands + 1)
    white_density = 2.0 * _WHITE_UNIFORM_VARIANCE * sample_time_s
    density = np.zeros((n_channels, n_bands), dtype=np.float64)
    for channel in range(n_channels):
        run_density = np.zeros(n_bands, dtype=np.float64)
        for run in range(n_runs):
            signal = normalised_actions[run, :, channel]
            frequencies, power = periodogram(
                signal - signal.mean(), fs=1.0 / sample_time_s, window="boxcar", detrend=False
            )
            for band in range(n_bands):
                mask = (frequencies >= edges[band]) & (frequencies < edges[band + 1])
                if mask.any():
                    run_density[band] += float(power[mask].mean())
        density[channel] = run_density / n_runs
    covered = density >= threshold_fraction * white_density
    floor = 1e-300
    flatness = np.exp(np.log(np.maximum(density, floor)).mean(axis=1)) / np.maximum(
        density.mean(axis=1), floor
    )
    flatness = np.where(density.mean(axis=1) > 0.0, flatness, 0.0)

    centred = normalised_actions - normalised_actions.mean(axis=1, keepdims=True)
    flat = centred.reshape(-1, n_channels)
    scale = flat.std(axis=0)
    cross = 0.0
    if n_channels > 1 and np.all(scale > 0.0):
        correlation = np.corrcoef(flat, rowvar=False)
        cross = float(np.abs(correlation[~np.eye(n_channels, dtype=bool)]).max())
    return SpectralReport(
        band_edges_hz=tuple(float(e) for e in edges),
        band_density=tuple(tuple(float(v) for v in row) for row in density),
        coverage_per_channel=tuple(float(v) for v in covered.mean(axis=1)),
        flatness_per_channel=tuple(float(v) for v in flatness),
        max_abs_cross_correlation=cross,
    )


@dataclass(frozen=True, slots=True)
class FisherReport:
    """Summary of a Fisher information matrix (``sigma^2 = 1``)."""

    n_parameters: int
    n_rows: int
    logdet_per_parameter: float
    lambda_min: float
    condition: float
    kind: str


def _summarise(information: FloatArray, n_rows: int, kind: str) -> FisherReport:
    n_parameters = information.shape[0]
    regularised = information + _FIM_RIDGE * np.eye(n_parameters)
    eigenvalues = np.linalg.eigvalsh(regularised)
    smallest = float(max(eigenvalues[0], 0.0))
    largest = float(eigenvalues[-1])
    logdet = float(np.sum(np.log(np.maximum(eigenvalues, 1e-300)))) / n_parameters
    return FisherReport(
        n_parameters=n_parameters,
        n_rows=n_rows,
        logdet_per_parameter=logdet,
        lambda_min=smallest,
        condition=math.inf if smallest <= 0.0 else largest / smallest,
        kind=kind,
    )


def _lag_matrix(signal: FloatArray, n_lags: int) -> FloatArray:
    """Rows ``k = n_lags - 1 .. n-1`` with columns ``u[k], u[k-1], .. u[k-n_lags+1]``."""
    n_steps = signal.shape[0]
    return np.stack([signal[n_lags - 1 - lag : n_steps - lag] for lag in range(n_lags)], axis=1)


def fir_fisher_information(
    normalised_actions: FloatArray, *, n_lags: int = 25, centre: float = 0.5
) -> FisherReport:
    """Fisher information of an input-only FIR regression, summed over the runs.

    ``n_lags = 25`` lags at 25 s cover 600 s (the solids control loop). Inputs only: needs no
    simulation and no truth.
    """
    n_runs, n_steps, n_channels = normalised_actions.shape
    if n_steps < n_lags:
        raise ValueError("runs are shorter than n_lags")
    n_parameters = n_lags * n_channels + 1
    information = np.zeros((n_parameters, n_parameters), dtype=np.float64)
    rows = 0
    for run in range(n_runs):
        blocks = [
            _lag_matrix(normalised_actions[run, :, channel] - centre, n_lags)
            for channel in range(n_channels)
        ]
        design = np.concatenate([*blocks, np.ones((blocks[0].shape[0], 1))], axis=1)
        information += design.T @ design
        rows += design.shape[0]
    return _summarise(information, rows, "fir")


def arx_fisher_information(
    normalised_actions: FloatArray,
    normalised_outputs: FloatArray,
    *,
    p: int = 4,
    q: int = 2,
) -> FisherReport:
    """Fisher information of the ARX regression of ``LinearARXModel`` (orders ``p``, ``q``).

    Regressors: ``p`` lags of every output, the inputs at lags ``0 .. q`` and a bias; one-step
    prediction. ``normalised_outputs`` are standardised outputs ``(runs, steps, targets)``.
    Needs the simulated outputs: analysis only.
    """
    n_runs, n_steps, n_channels = normalised_actions.shape
    n_targets = normalised_outputs.shape[2]
    start = max(p, q + 1)
    n_parameters = p * n_targets + (q + 1) * n_channels + 1
    information = np.zeros((n_parameters, n_parameters), dtype=np.float64)
    rows = 0
    for run in range(n_runs):
        columns: list[FloatArray] = []
        for target in range(n_targets):
            series = normalised_outputs[run, :, target]
            columns.extend(series[start - lag : n_steps - lag] for lag in range(1, p + 1))
        for channel in range(n_channels):
            series = normalised_actions[run, :, channel] - 0.5
            columns.extend(series[start - lag : n_steps - lag] for lag in range(0, q + 1))
        design = np.stack([*columns, np.ones(n_steps - start)], axis=1)
        information += design.T @ design
        rows += design.shape[0]
    return _summarise(information, rows, "arx")

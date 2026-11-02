r"""Figures for learning curves, variance decomposition and noise floor (step 5, item 5).

Why this is a module and not notebook cells
-------------------------------------------------
The code conventions require it, but there is also a substantive reason: a learning
curve without a resolution threshold is a misleading figure. If the logic lives in the
notebook, sooner or later a variant without the threshold band appears -- and that one
then ends up in a talk. Here the band is part of the function and not of the caller.

What every curve must carry
---------------------------
1. Median over the replicates with a spread band (min/max, not ±1σ: with three
   replicates the normality assumption cannot be checked).
2. The parameter-free baselines as horizontal lines -- persistence is constant over
   :math:`n_{\text{train}}`, because it is not trained.
3. The **resolution threshold** as a band around the best curve. Anything that lies
   inside it is not a result.
4. The **saturation level** (noise floor) as the lower asymptote, where known.

Matplotlib is not a run-time dependency of the package (extra ``dev``); the import
therefore only happens in the function call, so that ``import twinlink_ml`` does not fail
on a compute node without a graphics stack.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..utils.seeding import VarianceSource
from ..utils.tracking import RunRecord
from .protocol import NoiseFloor
from .resolution import DEFAULT_METRIC, collect_spreads

__all__ = [
    "plot_learning_curve",
    "plot_noise_study",
    "plot_variance_decomposition",
]

_FLAT_LINE_LABELS: tuple[str, ...] = ("persistence",)
"""Modelle ohne Trainingsschritt -- als waagerechte Linie statt als Kurve."""


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "matplotlib fehlt. Entweder 'pip install twinlink-ml[dev]' oder die "
            "Tabellenausgabe der Skripte in scripts/ verwenden."
        ) from error
    return plt


def _curves(
    records: Sequence[RunRecord],
    *,
    target: str,
    horizon: int,
    metric: str,
    source: VarianceSource,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """``{Modell: (n_train, Median, Minimum, Maximum)}``, sorted in ascending order."""
    spreads = collect_spreads(records, target=target, horizon=horizon, metric=metric)
    per_label: dict[str, dict[int, tuple[float, float, float]]] = {}
    for (key, spread_source), spread in spreads.items():
        if spread_source is not source:
            continue
        per_label.setdefault(key.label, {})[key.n_train] = (
            spread.median,
            spread.minimum,
            spread.maximum,
        )
    curves = {}
    for label, points in sorted(per_label.items()):
        sizes = np.asarray(sorted(points), dtype=float)
        values = np.asarray([points[int(n)] for n in sizes], dtype=float)
        curves[label] = (sizes, values[:, 0], values[:, 1], values[:, 2])
    return curves


def plot_learning_curve(
    records: Sequence[RunRecord],
    *,
    target: str,
    horizon: int,
    metric: str = DEFAULT_METRIC,
    source: VarianceSource = VarianceSource.JOINT,
    resolution: Mapping[int, float] | None = None,
    noise_floor: float | None = None,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    r"""Learning curve with spread band, baselines and resolution threshold.

    Parameters
    ----------
    resolution
        ``{n_train: relative Schwelle}`` from
        :func:`~twinlink_ml.evaluation.resolution.resolution_thresholds`. The band is
        drawn around the **best** curve: differences within this band cannot be
        demonstrated with the number of replicates run.
    noise_floor
        Saturation level from :class:`~twinlink_ml.evaluation.protocol.NoiseFloor`. A
        curve that flattens there has reached the noise floor and not the model limit.

    The x-axis is logarithmic (learning-curve points are powers of two), the y-axis is
    not: the comparison with the threshold 1.0 -- better than the mean of the run itself --
    should remain readable at a glance.
    """
    plt = _pyplot()
    axis = ax if ax is not None else plt.subplots(figsize=(7.0, 4.5))[1]
    curves = _curves(records, target=target, horizon=horizon, metric=metric, source=source)
    if not curves:
        raise ValueError(f"keine Datenpunkte für {target}, h={horizon}, Achse {source.value}")

    all_sizes = sorted({int(n) for sizes, *_ in curves.values() for n in sizes})
    for label, (sizes, median, minimum, maximum) in curves.items():
        if label in _FLAT_LINE_LABELS:
            axis.axhline(
                float(np.median(median)),
                linestyle="--",
                linewidth=1.2,
                color="0.35",
                label=f"{label} (parameterfrei)",
            )
            axis.fill_between(
                [min(all_sizes), max(all_sizes)],
                float(np.min(minimum)),
                float(np.max(maximum)),
                color="0.35",
                alpha=0.10,
            )
            continue
        line = axis.plot(sizes, median, marker="o", label=label)[0]
        axis.fill_between(sizes, minimum, maximum, alpha=0.18, color=line.get_color())

    if resolution:
        best_label = min(
            (label for label in curves if label not in _FLAT_LINE_LABELS),
            key=lambda label: float(np.min(curves[label][1])),
            default=None,
        )
        if best_label is not None:
            sizes, median, *_ = curves[best_label]
            band = np.asarray([resolution.get(int(n), np.nan) for n in sizes], dtype=float)
            axis.fill_between(
                sizes,
                median * (1.0 - band),
                median * (1.0 + band),
                color="crimson",
                alpha=0.12,
                label="Auflösungsschwelle",
            )

    if noise_floor is not None:
        axis.axhline(
            noise_floor,
            color="seagreen",
            linestyle=":",
            linewidth=1.4,
            label="Rauschgrenze",
        )

    axis.set_xscale("log", base=2)
    axis.set_xticks(all_sizes)
    axis.set_xticklabels([str(n) for n in all_sizes])
    axis.set_xlabel("Trainingsläufe $n_\\mathrm{train}$")
    axis.set_ylabel(f"NRMSE (laufweiser Median), $h={horizon}$")
    axis.set_title(title if title is not None else f"{target}, Streuung: {source.value}")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize="small")
    return axis


def plot_variance_decomposition(
    records: Sequence[RunRecord],
    *,
    target: str,
    horizon: int,
    metric: str = DEFAULT_METRIC,
    ax: Any = None,
) -> Any:
    r"""Spread per random source as grouped bars, model per learning-curve point.

    Plotted is :math:`\\hat\\sigma` (range-based), not the range itself -- only then are
    axes with different numbers of replicates comparable.
    """
    plt = _pyplot()
    axis = ax if ax is not None else plt.subplots(figsize=(8.0, 4.5))[1]
    spreads = collect_spreads(records, target=target, horizon=horizon, metric=metric)
    keys = sorted({key for key, _ in spreads}, key=lambda key: (key.label, key.n_train))
    sources = [
        VarianceSource.DATA,
        VarianceSource.NOISE,
        VarianceSource.MODEL,
        VarianceSource.JOINT,
    ]
    positions = np.arange(len(keys), dtype=float)
    width = 0.8 / len(sources)
    for index, source in enumerate(sources):
        values = [
            spreads[(key, source)].sigma_from_span if (key, source) in spreads else np.nan
            for key in keys
        ]
        axis.bar(positions + index * width, values, width=width, label=source.value)
    axis.set_xticks(positions + 0.4 - width / 2)
    axis.set_xticklabels([str(key) for key in keys], rotation=30, ha="right", fontsize="small")
    axis.set_ylabel("$\\hat\\sigma$ (spannweitenbasiert)")
    axis.set_title(f"Streuung nach Quelle -- {target}, $h={horizon}$")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize="small")
    return axis


def plot_noise_study(
    records: Sequence[RunRecord],
    floors: Mapping[float, NoiseFloor],
    *,
    target: str,
    horizon: int,
    n_train: int,
    metric: str = DEFAULT_METRIC,
    ax: Any = None,
) -> Any:
    """Error over the noise level, with the noise floor as a reference line.

    The gap between curve and limit is the share that a better model can still gain at
    all. If the curve runs parallel above the limit, the model error is independent of
    the noise -- then a quieter measurement gains nothing.
    """
    plt = _pyplot()
    axis = ax if ax is not None else plt.subplots(figsize=(7.0, 4.5))[1]
    key = f"{metric}/{target}/h{horizon}"

    per_label: dict[str, dict[float, list[float]]] = {}
    for record in records:
        if key not in record.metrics or int(record.params["n_train"]) != n_train:
            continue
        label = str(record.params.get("model/label", "?"))
        level = float(record.params["noise/relative_std"])
        per_label.setdefault(label, {}).setdefault(level, []).append(float(record.metrics[key]))

    for label, by_level in sorted(per_label.items()):
        levels = sorted(by_level)
        medians = [float(np.median(by_level[level])) for level in levels]
        axis.plot(levels, medians, marker="o", label=label)

    if floors:
        levels = sorted(floors)
        axis.plot(
            levels,
            [floors[level].per_run[target][horizon][0] for level in levels],
            color="seagreen",
            linestyle=":",
            marker="s",
            label="Rauschgrenze (modellfrei)",
        )

    axis.set_xlabel("relatives Sensorrauschen $\\sigma_\\mathrm{rel}$")
    axis.set_ylabel(f"NRMSE (laufweiser Median), $h={horizon}$")
    axis.set_title(f"{target}, $n_\\mathrm{{train}} = {n_train}$")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize="small")
    return axis

r"""Abbildungen zu Lernkurven, Varianzzerlegung und Rauschgrenze (Schritt 5, Punkt 5).

Warum das ein Modul ist und keine Notebook-Zellen
-------------------------------------------------
Die Code-Konventionen verlangen es, aber es gibt auch einen inhaltlichen Grund: eine
Lernkurve ohne Auflösungsschwelle ist eine irreführende Abbildung. Liegt die Logik im
Notebook, entsteht früher oder später eine Variante ohne Schwellenband -- und die wandert
dann in einen Vortrag. Hier ist das Band Teil der Funktion und nicht des Aufrufers.

Was jede Kurve tragen muss
--------------------------
1. Median über die Replikate mit Streuband (Min/Max, nicht ±1σ: bei drei Replikaten ist
   die Normalverteilungsannahme nicht prüfbar).
2. Die parameterfreien Baselines als waagerechte Linien -- Persistenz ist über
   :math:`n_{\text{train}}` konstant, weil sie nicht trainiert wird.
3. Die **Auflösungsschwelle** als Band um die beste Kurve. Alles, was darin liegt, ist
   kein Ergebnis.
4. Die **Sättigungshöhe** (Rauschgrenze) als untere Asymptote, soweit bekannt.

Matplotlib ist keine Laufzeitabhängigkeit des Pakets (Extra ``dev``); der Import erfolgt
deshalb erst im Funktionsaufruf, damit ``import twinlink_ml`` auf einem Rechenknoten ohne
Grafikstack nicht scheitert.
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
    except ImportError as error:  # pragma: no cover - abhängig von der Umgebung
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
    """``{Modell: (n_train, Median, Minimum, Maximum)}``, aufsteigend sortiert."""
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
    r"""Lernkurve mit Streuband, Baselines und Auflösungsschwelle.

    Parameters
    ----------
    resolution
        ``{n_train: relative Schwelle}`` aus
        :func:`~twinlink_ml.evaluation.resolution.resolution_thresholds`. Gezeichnet wird
        das Band um die **beste** Kurve: Unterschiede innerhalb dieses Bandes sind mit
        der gefahrenen Replikatzahl nicht nachweisbar.
    noise_floor
        Sättigungshöhe aus :class:`~twinlink_ml.evaluation.protocol.NoiseFloor`. Eine
        Kurve, die dort abflacht, hat die Rauschgrenze erreicht und nicht die
        Modellgrenze.

    Die x-Achse ist logarithmisch (Stützstellen sind Zweierpotenzen), die y-Achse nicht:
    der Vergleich mit der Schwelle 1.0 -- besser als der Mittelwert des eigenen Laufs --
    soll auf einen Blick ablesbar bleiben.
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
    r"""Streuung je Zufallsquelle als gruppierte Balken, Modell je Stützstelle.

    Aufgetragen ist :math:`\\hat\\sigma` (spannweitenbasiert), nicht die Spannweite
    selbst -- nur so sind Achsen mit unterschiedlicher Replikatzahl vergleichbar.
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
    """Fehler über dem Rauschniveau, mit der Rauschgrenze als Referenzlinie.

    Der Abstand zwischen Kurve und Grenze ist der Anteil, den ein besseres Modell
    überhaupt noch holen kann. Läuft die Kurve parallel über der Grenze, ist der
    Modellfehler rauschunabhängig -- dann bringt eine leisere Messung nichts.
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

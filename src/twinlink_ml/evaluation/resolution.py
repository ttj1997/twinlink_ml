r"""Auflösungsvermögen und Varianzzerlegung (Arbeitsplan Schritt 5, Punkte 1 und 2).

Die Frage, die dieses Modul beantwortet
---------------------------------------
Nach Schritt 4 steht eine Lernkurve mit Median und Spannweite über drei Seeds. Was
**nicht** dasteht, ist die Schwelle, ab der ein Unterschied zwischen zwei Architekturen
überhaupt eine Aussage ist. Bei ``product_moisture_kg_per_kg`` liegt die Seed-Spannweite
der Persistenz bei :math:`\pm 10\,\%` relativ, und Persistenz hat **keine Parameter**:
diese Streuung ist reine Datensatz-Realisierung. Ein GRU, das 5 % besser ist, ist damit
nicht besser, sondern nicht unterscheidbar.

Zwei Größen, die auseinandergehalten werden müssen
-------------------------------------------------
* **Streuung** :math:`\hat\sigma` -- die Schwankung eines einzelnen Ergebnisses über die
  Replikate. Das ist die berichtete Spannweite.
* **Auflösungsvermögen** -- die kleinste Differenz zwischen zwei Modellen, die mit
  :math:`n` Replikaten von Null unterscheidbar ist. Sie fällt mit
  :math:`\sqrt{2/n}` und ist deshalb *kleiner* als die Spannweite.

.. math::
   \Delta_{\min} = z_{1-\alpha/2}\,\hat\sigma\,\sqrt{\frac{2}{n}}

Der Faktor :math:`\sqrt 2` steht dort, weil zwei unabhängig gestreute Größen verglichen
werden, nicht eine gegen eine Konstante.

Warum die Streuung aus der Spannweite und nicht aus der Standardabweichung
--------------------------------------------------------------------------
Bei drei bis fünf Replikaten ist :math:`s` mit :math:`\mathrm{ddof}=1` selbst extrem
streuungsbehaftet (relativer Standardfehler von :math:`s` bei :math:`n=3` rund 50 %). Der
spannweitenbasierte Schätzer :math:`\hat\sigma = R/d_2(n)` mit den Hartley-Konstanten aus
der Prozessregelung ist bei kleinem :math:`n` robuster und benutzt genau die Zahl, die
ohnehin berichtet wird. Beide werden mitprotokolliert; abweichen sie stark, ist ein
Replikat ein Ausreißer und die Zahl nicht zu glauben.

**Die Grenze der Methode, offen benannt:** mit drei Replikaten ist jede dieser Zahlen
eine Schätzung mit einer Unsicherheit derselben Größenordnung wie sie selbst. Sie ist als
*Größenordnung* zu lesen -- als Antwort auf "ist dieser Unterschied 3 % oder 300 %" und
nicht als Signifikanztest. Genau deshalb liefert :meth:`SeedSpread.required_replicates`
eine Zahl: wer eine Differenz von 5 % nachweisen will, sieht dort, dass drei Seeds dafür
nicht genügen.

Was die Zerlegung nach Quelle leistet und was nicht
---------------------------------------------------
Die Achsen ``DATA``, ``NOISE``, ``MODEL`` werden einzeln variiert (one-factor-at-a-time,
siehe :class:`~twinlink_ml.utils.seeding.SeedPlan`). Das sind **Haupteffekte**, keine
Varianzanalyse mit Interaktionen. Ob die Zerlegung trägt, prüft
:meth:`VarianceDecomposition.additivity_ratio` gegen die gemeinsam variierte Achse:

.. math::
   \varrho = \frac{\sqrt{\hat\sigma_{data}^2 + \hat\sigma_{noise}^2
   + \hat\sigma_{model}^2}}{\hat\sigma_{joint}}

Bei :math:`\varrho \approx 1` sind die Quellen näherungsweise unabhängig und die
Zerlegung ist lesbar. Weicht :math:`\varrho` deutlich ab, dominieren Interaktionen (oder
die Schätzung ist bei drei Replikaten zu grob) -- dann bleibt nur die gemeinsame Streuung
als berichtete Zahl. Diese Prüfung ist der Grund, warum ``JOINT`` mitgefahren wird,
anstatt die Gesamtstreuung als Summe der Einzelachsen zu behaupten.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from ..utils.seeding import VarianceSource
from ..utils.tracking import RunRecord

__all__ = [
    "PointKey",
    "SeedSpread",
    "VarianceDecomposition",
    "Verdict",
    "architecture_verdicts",
    "collect_spreads",
    "decompose_variance",
    "resolution_report",
    "resolution_thresholds",
    "select",
]

_HARTLEY_D2: Final[dict[int, float]] = {
    2: 1.128,
    3: 1.693,
    4: 2.059,
    5: 2.326,
    6: 2.534,
    7: 2.704,
    8: 2.847,
    9: 2.970,
    10: 3.078,
}
"""Erwartungswert der Spannweite in Einheiten von :math:`\\sigma`, Normalverteilung."""

_Z_FROM_CONFIDENCE: Final[dict[float, float]] = {0.68: 1.0, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}

DEFAULT_METRIC: Final[str] = "nrmse_run_median"
"""Primärmetrik: laufweise normierter Rollout-Fehler, Median über die Testläufe."""


@dataclass(frozen=True, slots=True)
class PointKey:
    """Identität eines Lernkurvenpunkts jenseits der variierten Achse."""

    label: str
    n_train: int

    def __str__(self) -> str:
        """Kompakte Darstellung für Tabellen."""
        return f"{self.label} @ n={self.n_train}"


@dataclass(frozen=True, slots=True)
class SeedSpread:
    """Streuung eines Ergebnisses über die Replikate einer Zufallsquelle."""

    source: VarianceSource
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("SeedSpread ohne Werte")

    @property
    def n_replicates(self) -> int:
        """Anzahl Replikate."""
        return len(self.values)

    @property
    def median(self) -> float:
        """Median -- die berichtete Zahl."""
        return float(np.median(self.values))

    @property
    def minimum(self) -> float:
        """Kleinster Wert."""
        return float(np.min(self.values))

    @property
    def maximum(self) -> float:
        """Größter Wert."""
        return float(np.max(self.values))

    @property
    def span(self) -> float:
        r"""Spannweite :math:`R = \max - \min`."""
        return self.maximum - self.minimum

    @property
    def relative_span(self) -> float:
        """Spannweite bezogen auf den Median."""
        return self.span / self.median if self.median != 0.0 else math.nan

    @property
    def sigma_from_span(self) -> float:
        """Spannweitenbasierter Streuungsschätzer :math:`R / d_2(n)`.

        Für :math:`n > 10` wird auf die Standardabweichung zurückgefallen -- die
        Hartley-Konstanten sind dort nicht mehr tabelliert und der übliche Schätzer ist
        bei dieser Replikatzahl ohnehin brauchbar.
        """
        if self.n_replicates == 1:
            return 0.0
        factor = _HARTLEY_D2.get(self.n_replicates)
        if factor is None:
            return self.sigma_from_std
        return self.span / factor

    @property
    def sigma_from_std(self) -> float:
        """Standardabweichung mit ``ddof=1``; nur zum Abgleich."""
        if self.n_replicates < 2:
            return 0.0
        return float(np.std(self.values, ddof=1))

    def minimum_detectable_difference(
        self, *, n_replicates: int | None = None, confidence: float = 0.95
    ) -> float:
        """Kleinste nachweisbare Differenz zweier Modelle, absolut.

        Parameters
        ----------
        n_replicates
            Replikatzahl, mit der verglichen werden soll. ``None`` nimmt die eigene.
        confidence
            Zweiseitiges Konfidenzniveau; erlaubt sind 0.68, 0.90, 0.95, 0.99.
        """
        count = self.n_replicates if n_replicates is None else n_replicates
        if count < 1:
            raise ValueError("n_replicates muss >= 1 sein")
        z = _z_value(confidence)
        return z * self.sigma_from_span * math.sqrt(2.0 / count)

    def relative_minimum_detectable_difference(
        self, *, n_replicates: int | None = None, confidence: float = 0.95
    ) -> float:
        """Dasselbe, bezogen auf den Median -- die berichtbare Prozentzahl."""
        absolute = self.minimum_detectable_difference(
            n_replicates=n_replicates, confidence=confidence
        )
        return absolute / self.median if self.median != 0.0 else math.nan

    def required_replicates(
        self, relative_difference: float, *, confidence: float = 0.95, maximum: int = 200
    ) -> int:
        r"""Replikatzahl, um eine relative Differenz nachzuweisen.

        .. math:: n \ge 2\left(\frac{z\,\hat\sigma}{\delta}\right)^2

        Nach oben begrenzt, weil das Ergebnis sonst als Rechenplan gelesen wird: wer bei
        einer Zielgröße 800 Seeds braucht, hat kein Seed-Problem, sondern die falsche
        Zielgröße oder ein zu grobes Anregungsdesign.
        """
        if relative_difference <= 0.0:
            raise ValueError("relative_difference muss > 0 sein")
        if self.sigma_from_span == 0.0:
            return 1
        absolute = relative_difference * abs(self.median)
        z = _z_value(confidence)
        needed = 2.0 * (z * self.sigma_from_span / absolute) ** 2
        return min(math.ceil(needed), maximum)

    def as_metrics(self, prefix: str) -> dict[str, float]:
        """Flache Kennzahlen für das Log."""
        return {
            f"{prefix}/median": self.median,
            f"{prefix}/min": self.minimum,
            f"{prefix}/max": self.maximum,
            f"{prefix}/sigma_span": self.sigma_from_span,
            f"{prefix}/sigma_std": self.sigma_from_std,
            f"{prefix}/n_replicates": float(self.n_replicates),
        }


def _z_value(confidence: float) -> float:
    if confidence not in _Z_FROM_CONFIDENCE:
        raise ValueError(
            f"confidence {confidence} nicht unterstützt, erlaubt: {sorted(_Z_FROM_CONFIDENCE)}"
        )
    return _Z_FROM_CONFIDENCE[confidence]


@dataclass(frozen=True, slots=True)
class VarianceDecomposition:
    """Streuung eines Punkts, aufgeschlüsselt nach Zufallsquelle."""

    key: PointKey
    target: str
    horizon: int
    per_source: Mapping[VarianceSource, SeedSpread]

    @property
    def dominant_source(self) -> VarianceSource:
        """Achse mit der größten Streuung, ``JOINT`` ausgenommen."""
        candidates = {
            source: spread.sigma_from_span
            for source, spread in self.per_source.items()
            if source is not VarianceSource.JOINT
        }
        if not candidates:
            raise ValueError("keine Einzelachse vorhanden")
        return max(candidates, key=lambda source: candidates[source])

    @property
    def additivity_ratio(self) -> float:
        r"""Quadratische Summe der Einzelachsen über der gemeinsamen Streuung.

        Näher an 1 heißt: die Quellen sind näherungsweise unabhängig und die Zerlegung
        ist lesbar. ``nan``, wenn ``JOINT`` fehlt oder dort keine Streuung auftrat.
        """
        joint = self.per_source.get(VarianceSource.JOINT)
        single = (VarianceSource.DATA, VarianceSource.NOISE, VarianceSource.MODEL)
        if joint is None or joint.sigma_from_span == 0.0:
            return math.nan
        if any(source not in self.per_source for source in single):
            # Fehlt eine Einzelachse, ist die Quadratsumme keine Zerlegung der
            # Gesamtstreuung, sondern ein Teil davon -- und eine Zahl nahe Null würde als
            # "keine Interaktionen" gelesen statt als "nicht gemessen".
            return math.nan
        total = math.sqrt(
            sum(
                spread.sigma_from_span**2
                for source, spread in self.per_source.items()
                if source is not VarianceSource.JOINT
            )
        )
        return total / joint.sigma_from_span

    def as_metrics(self) -> dict[str, float]:
        """Flache Kennzahlen für das Log."""
        metrics: dict[str, float] = {}
        for source, spread in self.per_source.items():
            metrics.update(spread.as_metrics(f"variance/{source.value}"))
        metrics["variance/additivity_ratio"] = self.additivity_ratio
        return metrics


def collect_spreads(
    records: Sequence[RunRecord],
    *,
    target: str,
    horizon: int,
    metric: str = DEFAULT_METRIC,
) -> dict[tuple[PointKey, VarianceSource], SeedSpread]:
    """Gruppiere Records nach Punkt und Zufallsquelle.

    Erwartet die Parameter ``variance/source``, ``variance/replicate``, ``model/label``
    und ``n_train``, wie sie
    :func:`~twinlink_ml.evaluation.protocol.run_variance_decomposition` schreibt.
    Mehrfach vorkommende Replikate derselben Achse werden **nicht** stillschweigend
    gemittelt, sondern als Fehler gemeldet. Das ist gleichzeitig der Schutz gegen den
    naheliegenden Bedienfehler: schiebt man ein Log hinein, in dem außer dem Seed noch
    ein zweiter Faktor variiert (Rauschniveau, Validierungs-Betriebsart, Δt), erscheint
    dasselbe Replikat mit zwei Werten -- und die "Streuung über Seeds" wäre in Wahrheit
    der Effekt dieses Faktors. Vorher mit :func:`select` filtern.
    """
    key_name = f"{metric}/{target}/h{horizon}"
    grouped: dict[tuple[PointKey, VarianceSource], dict[int, float]] = {}
    for record in records:
        if key_name not in record.metrics:
            continue
        source_name = record.params.get("variance/source")
        if source_name is None:
            continue
        source = VarianceSource(str(source_name))
        key = PointKey(
            label=str(record.params.get("model/label", record.tags.get("model", "?"))),
            n_train=int(record.params["n_train"]),
        )
        replicate = int(record.params["variance/replicate"])
        bucket = grouped.setdefault((key, source), {})
        value = float(record.metrics[key_name])
        if replicate in bucket and not math.isclose(bucket[replicate], value):
            raise ValueError(
                f"Replikat {replicate} von {key} / {source.value} kommt mit zwei "
                f"verschiedenen Werten vor ({bucket[replicate]} und {value})"
            )
        bucket[replicate] = value
    return {
        identity: SeedSpread(
            source=identity[1],
            values=tuple(values[replicate] for replicate in sorted(values)),
        )
        for identity, values in grouped.items()
    }


def decompose_variance(
    records: Sequence[RunRecord],
    *,
    target: str,
    horizon: int,
    metric: str = DEFAULT_METRIC,
) -> dict[PointKey, VarianceDecomposition]:
    """Varianzzerlegung je Modell und Stützstelle."""
    spreads = collect_spreads(records, target=target, horizon=horizon, metric=metric)
    per_key: dict[PointKey, dict[VarianceSource, SeedSpread]] = {}
    for (key, source), spread in spreads.items():
        per_key.setdefault(key, {})[source] = spread
    return {
        key: VarianceDecomposition(
            key=key, target=target, horizon=horizon, per_source=dict(sorted(sources.items()))
        )
        for key, sources in sorted(
            per_key.items(), key=lambda item: (item[0].label, item[0].n_train)
        )
    }


def resolution_thresholds(
    records: Sequence[RunRecord],
    *,
    target: str,
    horizon: int,
    reference_label: str = "persistence",
    metric: str = DEFAULT_METRIC,
    confidence: float = 0.95,
    source: VarianceSource = VarianceSource.JOINT,
) -> dict[int, float]:
    r"""Auflösungsschwelle je Stützstelle, relativ, aus einer parameterfreien Baseline.

    Warum die Referenz parameterfrei sein muss: die Streuung eines angepassten Modells
    enthält Datensatz-Realisierung **und** Schätzunsicherheit. Als Schwelle taugt nur der
    erste Anteil, denn er begrenzt jeden Vergleich, unabhängig von der Architektur. Die
    Persistenz hat keine Parameter -- ihre Streuung ist genau dieser Anteil.

    Der Preis dieser Wahl, offen benannt: die Schwelle ist damit **konstruktionsbedingt
    unabhängig von** :math:`n_{\text{train}}`, weil die Persistenz nicht trainiert wird
    und der Testsplit über alle Stützstellen derselbe ist. Sie ist eine untere Schranke.
    Die zusätzliche, mit kleinem :math:`n` wachsende Schätzstreuung steht in der
    Zerlegung der angepassten Baselines (``static``, ``arx``) und ist dort abzulesen --
    beide Zahlen gehören in den Bericht, nicht nur diese.
    """
    spreads = collect_spreads(records, target=target, horizon=horizon, metric=metric)
    thresholds: dict[int, float] = {}
    for (key, spread_source), spread in spreads.items():
        if key.label != reference_label or spread_source is not source:
            continue
        thresholds[key.n_train] = spread.relative_minimum_detectable_difference(
            confidence=confidence
        )
    if not thresholds:
        raise ValueError(
            f"keine Einträge für Referenzmodell {reference_label!r} auf Achse "
            f"{source.value!r} -- ohne parameterfreie Baseline gibt es keine Schwelle"
        )
    return dict(sorted(thresholds.items()))


def resolution_report(
    records: Sequence[RunRecord],
    *,
    target: str,
    horizon: int,
    metric: str = DEFAULT_METRIC,
    reference_label: str = "persistence",
    confidence: float = 0.95,
) -> str:
    """Textbericht: Zerlegung nach Quelle, Auflösungsschwelle, nötige Replikatzahl."""
    decomposition = decompose_variance(records, target=target, horizon=horizon, metric=metric)
    lines = [
        f"Zielgröße {target}, h = {horizon}, Metrik {metric}",
        f"Konfidenz {confidence:.0%}, Streuung spannweitenbasiert (Hartley d2)",
        "",
        f"{'Punkt':<26}{'Median':>9}{'σ data':>9}{'σ noise':>9}{'σ model':>9}"
        f"{'σ joint':>9}{'ϱ':>7}{'Δmin rel':>10}{'n für 5 %':>11}",
    ]
    for key, item in decomposition.items():
        joint = item.per_source.get(VarianceSource.JOINT)
        reference = joint if joint is not None else next(iter(item.per_source.values()))
        cells = "".join(
            f"{item.per_source[source].sigma_from_span:>9.4f}"
            if source in item.per_source
            else f"{'-':>9}"
            for source in (
                VarianceSource.DATA,
                VarianceSource.NOISE,
                VarianceSource.MODEL,
                VarianceSource.JOINT,
            )
        )
        ratio = item.additivity_ratio
        lines.append(
            f"{key!s:<26}{reference.median:>9.4f}{cells}{ratio:>7.2f}"
            f"{reference.relative_minimum_detectable_difference(confidence=confidence):>10.1%}"
            f"{reference.required_replicates(0.05, confidence=confidence):>11d}"
        )

    try:
        thresholds = resolution_thresholds(
            records,
            target=target,
            horizon=horizon,
            reference_label=reference_label,
            metric=metric,
            confidence=confidence,
        )
    except ValueError as error:
        lines += ["", f"Auflösungsschwelle: {error}"]
    else:
        band = ", ".join(f"n={n}: {value:.1%}" for n, value in thresholds.items())
        lines += [
            "",
            f"Auflösungsschwelle aus {reference_label} (parameterfrei): {band}",
            "Unterschiede unterhalb dieser Schwelle sind mit dieser Replikatzahl nicht "
            "nachweisbar.",
        ]
    return "\n".join(lines)


def select(records: Sequence[RunRecord], **parameters: object) -> list[RunRecord]:
    """Filtere Records nach Parameterwerten.

    Für Studien mit mehr als einem Faktor: erst auf eine Stufe des Faktors filtern, dann
    zerlegen. Der Vergleich in Strings, damit ``0.005`` und ``"0.005"`` aus dem
    JSONL-Wiedereinlesen dasselbe treffen.
    """

    def matches(record: RunRecord) -> bool:
        return all(str(record.params.get(key)) == str(value) for key, value in parameters.items())

    return [record for record in records if matches(record)]


@dataclass(frozen=True, slots=True)
class Verdict:
    """Vergleich zweier Modelle an einer Stützstelle, gegen die Auflösungsschwelle.

    Attributes
    ----------
    candidate, reference
        Labels. ``reference`` ist die Baseline, die geschlagen werden muss -- nicht die
        parameterfreie Referenz, aus der die Schwelle stammt.
    improvement
        Relative Verbesserung des Medians, positiv wenn der Kandidat besser (kleiner) ist.
    threshold
        Relative Auflösungsschwelle an dieser Stützstelle.
    """

    candidate: str
    reference: str
    n_train: int
    candidate_median: float
    reference_median: float
    improvement: float
    threshold: float

    @property
    def is_resolved(self) -> bool:
        """Ob der Unterschied größer als das Auflösungsvermögen ist."""
        return abs(self.improvement) > self.threshold

    @property
    def statement(self) -> str:
        """Formulierung, die berichtet werden darf -- und keine andere."""
        if not self.is_resolved:
            return (
                f"{self.candidate} vs. {self.reference}: {self.improvement:+.1%} — "
                f"unterhalb der Auflösungsschwelle ({self.threshold:.1%}), "
                "kein nachweisbarer Unterschied"
            )
        direction = "besser" if self.improvement > 0.0 else "schlechter"
        return (
            f"{self.candidate} vs. {self.reference}: {self.improvement:+.1%} — "
            f"nachweisbar {direction} (Schwelle {self.threshold:.1%})"
        )


def architecture_verdicts(
    records: Sequence[RunRecord],
    *,
    target: str,
    horizon: int,
    reference_label: str,
    resolution_label: str = "persistence",
    metric: str = DEFAULT_METRIC,
    source: VarianceSource = VarianceSource.JOINT,
    confidence: float = 0.95,
) -> list[Verdict]:
    """Jeden Kandidaten gegen eine Baseline halten -- **mit** der Schwelle daneben.

    Der Zweck dieser Funktion ist eine Formulierungsdisziplin, nicht eine Rechnung: die
    Zahl "12 % besser" ist ohne die Schwelle daneben keine Aussage, und in einem Bericht
    steht sie dann trotzdem. Hier kommen beide nur gemeinsam heraus, und
    :attr:`Verdict.statement` formuliert das Ergebnis so, wie es berichtet werden darf.

    ``reference_label`` ist die Baseline, die zu schlagen ist (auf ``product_moisture``
    die Persistenz, auf ``energy_target`` der ARX). ``resolution_label`` ist die
    parameterfreie Baseline, aus deren Seed-Streuung die Schwelle kommt -- im
    Soft-Sensor-Betrieb ``climatology`` statt ``persistence``.
    """
    spreads = collect_spreads(records, target=target, horizon=horizon, metric=metric)
    thresholds = resolution_thresholds(
        records,
        target=target,
        horizon=horizon,
        reference_label=resolution_label,
        metric=metric,
        confidence=confidence,
        source=source,
    )
    by_point: dict[int, dict[str, SeedSpread]] = {}
    for (key, spread_source), spread in spreads.items():
        if spread_source is not source:
            continue
        by_point.setdefault(key.n_train, {})[key.label] = spread

    verdicts: list[Verdict] = []
    for n_train in sorted(by_point):
        per_label = by_point[n_train]
        reference = per_label.get(reference_label)
        if reference is None or reference.median == 0.0:
            continue
        threshold = thresholds.get(n_train)
        if threshold is None:
            continue
        for label, spread in sorted(per_label.items()):
            if label in {reference_label, resolution_label}:
                continue
            improvement = (reference.median - spread.median) / abs(reference.median)
            verdicts.append(
                Verdict(
                    candidate=label,
                    reference=reference_label,
                    n_train=n_train,
                    candidate_median=spread.median,
                    reference_median=reference.median,
                    improvement=improvement,
                    threshold=threshold,
                )
            )
    return verdicts

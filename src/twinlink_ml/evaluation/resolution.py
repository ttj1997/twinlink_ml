r"""Resolving power and variance decomposition (work plan step 5, items 1 and 2).

The question this module answers
---------------------------------------
After step 4 there is a learning curve with median and range over three seeds. What is
**not** there is the threshold above which a difference between two architectures is a
statement at all. For ``product_moisture_kg_per_kg`` the seed range of persistence is
:math:`\pm 10\,\%` relative, and persistence has **no parameters**: this spread is pure
data set realisation. A GRU that is 5 % better is therefore not better but
indistinguishable.

Two quantities that must be kept apart
-------------------------------------------------
* **Spread** :math:`\hat\sigma` -- the fluctuation of a single result over the
  replicates. This is the reported range.
* **Resolving power** -- the smallest difference between two models that can be
  distinguished from zero with :math:`n` replicates. It falls with
  :math:`\sqrt{2/n}` and is therefore *smaller* than the range.

.. math::
   \Delta_{\min} = z_{1-\alpha/2}\,\hat\sigma\,\sqrt{\frac{2}{n}}

The factor :math:`\sqrt 2` is there because two independently scattered quantities are
compared, not one against a constant.

Why the spread from the range and not from the standard deviation
--------------------------------------------------------------------------
With three to five replicates, :math:`s` with :math:`\mathrm{ddof}=1` is itself extremely
noisy (relative standard error of :math:`s` at :math:`n=3` about 50 %). The range-based
estimator :math:`\hat\sigma = R/d_2(n)` with the Hartley constants from process control is
more robust at small :math:`n` and uses exactly the number that is reported anyway. Both
are logged; if they differ strongly, one replicate is an outlier and the number is not to
be believed.

**The limit of the method, stated openly:** with three replicates each of these numbers
is an estimate with an uncertainty of the same order of magnitude as itself. It is to be
read as an *order of magnitude* -- as an answer to "is this difference 3 % or 300 %" and
not as a significance test. Precisely for this reason :meth:`SeedSpread.required_replicates`
provides a number: anyone who wants to demonstrate a difference of 5 % sees there that
three seeds are not enough for it.

What the decomposition by source achieves and what it does not
---------------------------------------------------------------
The axes ``DATA``, ``NOISE``, ``MODEL`` are varied individually (one-factor-at-a-time,
see :class:`~twinlink_ml.utils.seeding.SeedPlan`). These are **main effects**, not an
analysis of variance with interactions. Whether the decomposition holds is checked by
:meth:`VarianceDecomposition.additivity_ratio` against the jointly varied axis:

.. math::
   \varrho = \frac{\sqrt{\hat\sigma_{data}^2 + \hat\sigma_{noise}^2
   + \hat\sigma_{model}^2}}{\hat\sigma_{joint}}

At :math:`\varrho \approx 1` the sources are approximately independent and the
decomposition is readable. If :math:`\varrho` deviates clearly, interactions dominate (or
the estimate is too coarse with three replicates) -- then only the joint spread remains
as the reported number. This check is the reason why ``JOINT`` is run along, instead of
claiming the total spread as the sum of the individual axes.
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
    """Identity of a learning-curve point beyond the varied axis."""

    label: str
    n_train: int

    def __str__(self) -> str:
        """Compact representation for tables."""
        return f"{self.label} @ n={self.n_train}"


@dataclass(frozen=True, slots=True)
class SeedSpread:
    """Spread of a result over the replicates of one random source."""

    source: VarianceSource
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("SeedSpread ohne Werte")

    @property
    def n_replicates(self) -> int:
        """Number of replicates."""
        return len(self.values)

    @property
    def median(self) -> float:
        """Median -- the reported number."""
        return float(np.median(self.values))

    @property
    def minimum(self) -> float:
        """Smallest value."""
        return float(np.min(self.values))

    @property
    def maximum(self) -> float:
        """Largest value."""
        return float(np.max(self.values))

    @property
    def span(self) -> float:
        r"""Range :math:`R = \max - \min`."""
        return self.maximum - self.minimum

    @property
    def relative_span(self) -> float:
        """Range relative to the median."""
        return self.span / self.median if self.median != 0.0 else math.nan

    @property
    def sigma_from_span(self) -> float:
        """Range-based spread estimator :math:`R / d_2(n)`.

        For :math:`n > 10` it falls back to the standard deviation -- the Hartley
        constants are no longer tabulated there and the usual estimator is usable anyway
        at this number of replicates.
        """
        if self.n_replicates == 1:
            return 0.0
        factor = _HARTLEY_D2.get(self.n_replicates)
        if factor is None:
            return self.sigma_from_std
        return self.span / factor

    @property
    def sigma_from_std(self) -> float:
        """Standard deviation with ``ddof=1``; only for cross-checking."""
        if self.n_replicates < 2:
            return 0.0
        return float(np.std(self.values, ddof=1))

    def minimum_detectable_difference(
        self, *, n_replicates: int | None = None, confidence: float = 0.95
    ) -> float:
        """Smallest detectable difference between two models, absolute.

        Parameters
        ----------
        n_replicates
            Number of replicates with which to compare. ``None`` takes the own one.
        confidence
            Two-sided confidence level; allowed are 0.68, 0.90, 0.95, 0.99.
        """
        count = self.n_replicates if n_replicates is None else n_replicates
        if count < 1:
            raise ValueError("n_replicates muss >= 1 sein")
        z = _z_value(confidence)
        return z * self.sigma_from_span * math.sqrt(2.0 / count)

    def relative_minimum_detectable_difference(
        self, *, n_replicates: int | None = None, confidence: float = 0.95
    ) -> float:
        """The same, relative to the median -- the reportable percentage."""
        absolute = self.minimum_detectable_difference(
            n_replicates=n_replicates, confidence=confidence
        )
        return absolute / self.median if self.median != 0.0 else math.nan

    def required_replicates(
        self, relative_difference: float, *, confidence: float = 0.95, maximum: int = 200
    ) -> int:
        r"""Number of replicates needed to demonstrate a relative difference.

        .. math:: n \ge 2\left(\frac{z\,\hat\sigma}{\delta}\right)^2

        Capped from above, because otherwise the result is read as a compute plan: anyone
        who needs 800 seeds for a target does not have a seed problem but the wrong target
        or too coarse an excitation design.
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
        """Flat figures for the log."""
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
    """Spread of a point, broken down by random source."""

    key: PointKey
    target: str
    horizon: int
    per_source: Mapping[VarianceSource, SeedSpread]

    @property
    def dominant_source(self) -> VarianceSource:
        """Axis with the largest spread, ``JOINT`` excluded."""
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
        r"""Quadratic sum of the individual axes over the joint spread.

        Closer to 1 means: the sources are approximately independent and the
        decomposition is readable. ``nan`` if ``JOINT`` is missing or showed no spread.
        """
        joint = self.per_source.get(VarianceSource.JOINT)
        single = (VarianceSource.DATA, VarianceSource.NOISE, VarianceSource.MODEL)
        if joint is None or joint.sigma_from_span == 0.0:
            return math.nan
        if any(source not in self.per_source for source in single):
            # If an individual axis is missing, the sum of squares is not a decomposition of
            # the total spread but a part of it -- and a number close to zero would be read
            # as "no interactions" instead of as "not measured".
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
        """Flat figures for the log."""
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
    """Group records by point and random source.

    Expects the parameters ``variance/source``, ``variance/replicate``, ``model/label``
    and ``n_train``, as written by
    :func:`~twinlink_ml.evaluation.protocol.run_variance_decomposition`.
    Replicates of the same axis that occur more than once are **not** silently averaged
    but reported as an error. This is at the same time the protection against the obvious
    usage error: if one feeds in a log in which a second factor varies besides the seed
    (noise level, validation mode, Δt), the same replicate appears with two values -- and
    the "spread over seeds" would in truth be the effect of that factor. Filter with
    :func:`select` beforehand.
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
    """Variance decomposition per model and learning-curve point."""
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
    r"""Resolution threshold per learning-curve point, relative, from a parameter-free baseline.

    Why the reference must be parameter-free: the spread of a fitted model contains data
    set realisation **and** estimation uncertainty. Only the first part is suitable as a
    threshold, because it limits every comparison, independently of the architecture.
    Persistence has no parameters -- its spread is exactly this part.

    The price of this choice, stated openly: the threshold is thereby **by construction
    independent of** :math:`n_{\text{train}}`, because persistence is not trained and the
    test split is the same across all learning-curve points. It is a lower bound. The
    additional estimation spread, which grows with small :math:`n`, is in the
    decomposition of the fitted baselines (``static``, ``arx``) and is to be read there --
    both numbers belong in the report, not only this one.
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
    """Text report: decomposition by source, resolution threshold, required replicates."""
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
    """Filter records by parameter values.

    For studies with more than one factor: first filter to one level of the factor, then
    decompose. The comparison is on strings, so that ``0.005`` and ``"0.005"`` from
    re-reading the JSONL match the same records.
    """

    def matches(record: RunRecord) -> bool:
        return all(str(record.params.get(key)) == str(value) for key, value in parameters.items())

    return [record for record in records if matches(record)]


@dataclass(frozen=True, slots=True)
class Verdict:
    """Comparison of two models at a learning-curve point, against the resolution threshold.

    Attributes
    ----------
    candidate, reference
        Labels. ``reference`` is the baseline that must be beaten -- not the
        parameter-free reference from which the threshold comes.
    improvement
        Relative improvement of the median, positive if the candidate is better (smaller).
    threshold
        Relative resolution threshold at this learning-curve point.
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
        """Whether the difference is larger than the resolving power."""
        return abs(self.improvement) > self.threshold

    @property
    def statement(self) -> str:
        """The wording that may be reported -- and no other."""
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
    """Hold every candidate against a baseline -- **with** the threshold next to it.

    The purpose of this function is a discipline of wording, not a calculation: the
    number "12 % better" is not a statement without the threshold next to it, and yet it
    ends up in a report. Here both only come out together, and
    :attr:`Verdict.statement` words the result the way it may be reported.

    ``reference_label`` is the baseline to be beaten (on ``product_moisture`` persistence,
    on ``energy_target`` the ARX). ``resolution_label`` is the parameter-free baseline
    from whose seed spread the threshold comes -- in soft-sensor operation
    ``climatology`` instead of ``persistence``.
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

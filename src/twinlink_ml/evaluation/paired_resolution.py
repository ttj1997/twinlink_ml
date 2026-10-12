r"""Paired resolution test over seeds (review findings C1/C2 of 2026-10-09, docs/12).

Why a second module next to :mod:`twinlink_ml.evaluation.resolution`
--------------------------------------------------------------------
The threshold of ``resolution.py`` is built from the spread of a parameter-free reference
alone, with a normal quantile and the unpaired two-sample formula. Review C1/C2 showed that
this is the wrong variance (candidate and reference never enter), the wrong scale (relative
to the median of the reference, not of the compared model) and the wrong layout (on the
``JOINT`` axis every method sees the same ``master_seed``, i.e. the design is paired), and
that ``z = 1.96`` is about 1.4 times too small at ``n = 3``. The old functions stay unchanged
so that findings of the lab book that rest on them remain reproducible; **new comparisons
use this module.**

The test
--------
All methods are run with the same ``master_seed`` set, so seed :math:`i` gives one pair
:math:`d_i = r_i - c_i` (reference minus candidate; positive = candidate better, for an
error metric). With :math:`n` pairs, mean :math:`\bar d`, sample standard deviation
:math:`s_d` (``ddof=1``):

.. math::
   t = \frac{\bar d}{s_d/\sqrt n}, \qquad \nu = n - 1, \qquad
   \Delta_{\min} = t_{\nu,\,1-\alpha/2}\,\frac{s_d}{\sqrt n}.

A difference is *resolved* when :math:`|\bar d| > \Delta_{\min}` (equivalently
:math:`p < \alpha`). Pairing removes the seed effect common to both methods (data
realisation, noise realisation) from :math:`s_d`; nothing is assumed about the spread of any
parameter-free baseline.

What this module does not do
----------------------------
* The test is on the **mean** of the paired differences. The lab book reports medians over
  the seeds; with 3-5 seeds the two can differ. Report both, say which one is tested.
* No correction for multiple comparisons (review item C-L1, open): :func:`paired_verdicts`
  returns one test per candidate; the number of tests belongs next to the result.
* ``metric`` has no default. Compare in physical units where the question is about a
  physical quantity (CLAUDE.md: R2 and normalised metrics compress the decisive range), and
  always name the test set in the report; the records carry no test-set name.
* Fewer than two pairs, or pairs whose differences are exactly identical (zero spread, e.g.
  two seed-independent methods), give **no** threshold: the result is returned as
  not resolved with ``degenerate=True`` instead of a threshold of 0 (review C4).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..utils.tracking import RunRecord

__all__ = [
    "PairedComparison",
    "paired_comparison",
    "paired_values",
    "paired_verdicts",
]

DEFAULT_SEED_PARAM = "master_seed"
"""Record parameter that identifies the seed; equal across methods of one study."""


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Result of one paired comparison; differences are ``reference - candidate``.

    Attributes
    ----------
    seeds
        Seeds that form the pairs, sorted.
    mean_difference, sd_difference
        Mean and sample standard deviation (``ddof=1``) of the paired differences, in the
        unit of the metric. Positive mean: the candidate is better (smaller).
    standard_error
        ``sd_difference / sqrt(n_pairs)``.
    degrees_of_freedom
        ``n_pairs - 1``.
    critical_value
        Two-sided t quantile at ``confidence``; ``nan`` for a degenerate comparison.
    threshold
        Smallest resolvable absolute difference ``critical_value * standard_error``;
        ``nan`` for a degenerate comparison.
    p_value
        Two-sided p-value of the paired t-test; ``nan`` for a degenerate comparison.
    degenerate
        True when fewer than two pairs or zero spread of the differences: no variance
        estimate, nothing is resolved.
    """

    candidate: str
    reference: str
    seeds: tuple[int, ...]
    mean_difference: float
    sd_difference: float
    standard_error: float
    degrees_of_freedom: int
    critical_value: float
    threshold: float
    p_value: float
    confidence: float
    mean_candidate: float
    mean_reference: float
    degenerate: bool

    @property
    def n_pairs(self) -> int:
        """Number of seeds that form pairs."""
        return len(self.seeds)

    @property
    def is_resolved(self) -> bool:
        """Whether ``|mean difference|`` exceeds the threshold."""
        if self.degenerate:
            return False
        return abs(self.mean_difference) > self.threshold

    @property
    def relative_improvement(self) -> float:
        """Mean difference relative to the mean of the reference (positive = better).

        Secondary number for the table; the test is on the absolute differences.
        """
        if self.mean_reference == 0.0:
            return math.nan
        return self.mean_difference / abs(self.mean_reference)

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """Confidence interval of the mean difference; ``(nan, nan)`` if degenerate."""
        if self.degenerate:
            return (math.nan, math.nan)
        return (self.mean_difference - self.threshold, self.mean_difference + self.threshold)

    @property
    def statement(self) -> str:
        """Wording that may be reported -- and no other."""
        head = (
            f"{self.candidate} vs. {self.reference} (n = {self.n_pairs} paired seeds): "
            f"mean difference {self.mean_difference:+.4g}"
        )
        if self.degenerate:
            return (
                head + " -- no variance estimate (fewer than 2 pairs or zero spread), not resolved"
            )
        if not self.is_resolved:
            return (
                head + f", threshold {self.threshold:.4g} (p = {self.p_value:.3f}) -- not resolved"
            )
        direction = "candidate better" if self.mean_difference > 0.0 else "candidate worse"
        return head + f", threshold {self.threshold:.4g} (p = {self.p_value:.3f}) -- {direction}"


def paired_comparison(
    candidate: Mapping[int, float],
    reference: Mapping[int, float],
    *,
    candidate_label: str = "candidate",
    reference_label: str = "reference",
    confidence: float = 0.95,
) -> PairedComparison:
    """Paired t-test of two methods over seeds (``n - 1`` degrees of freedom).

    Parameters
    ----------
    candidate, reference
        Metric value per seed (same unit, lower is better). The two mappings must have
        **exactly the same seeds**: a missing seed is an error, not silently dropped, because
        dropping it changes the sample.
    confidence
        Two-sided confidence level in ``(0, 1)``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if set(candidate) != set(reference):
        raise ValueError(
            "candidate and reference need the same seeds, got "
            f"{sorted(candidate)} and {sorted(reference)}"
        )
    seeds = tuple(sorted(candidate))
    if not seeds:
        raise ValueError("no seeds")
    cand = np.asarray([candidate[seed] for seed in seeds], dtype=np.float64)
    ref = np.asarray([reference[seed] for seed in seeds], dtype=np.float64)
    if not (np.all(np.isfinite(cand)) and np.all(np.isfinite(ref))):
        raise ValueError("non-finite metric value")
    n_pairs = len(seeds)
    differences = ref - cand
    mean_difference = float(differences.mean())
    nu = n_pairs - 1

    degenerate = n_pairs < 2
    sd = math.nan if n_pairs < 2 else float(differences.std(ddof=1))
    if not degenerate and sd == 0.0:
        degenerate = True
    if degenerate:
        return PairedComparison(
            candidate=candidate_label,
            reference=reference_label,
            seeds=seeds,
            mean_difference=mean_difference,
            sd_difference=sd,
            standard_error=math.nan,
            degrees_of_freedom=nu,
            critical_value=math.nan,
            threshold=math.nan,
            p_value=math.nan,
            confidence=confidence,
            mean_candidate=float(cand.mean()),
            mean_reference=float(ref.mean()),
            degenerate=True,
        )

    standard_error = sd / math.sqrt(n_pairs)
    critical = float(stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, nu))
    t_statistic = mean_difference / standard_error
    p_value = float(2.0 * stats.t.sf(abs(t_statistic), nu))
    return PairedComparison(
        candidate=candidate_label,
        reference=reference_label,
        seeds=seeds,
        mean_difference=mean_difference,
        sd_difference=sd,
        standard_error=standard_error,
        degrees_of_freedom=nu,
        critical_value=critical,
        threshold=critical * standard_error,
        p_value=p_value,
        confidence=confidence,
        mean_candidate=float(cand.mean()),
        mean_reference=float(ref.mean()),
        degenerate=False,
    )


def paired_values(
    records: Sequence[RunRecord],
    *,
    label: str,
    n_train: int,
    target: str,
    horizon: int,
    metric: str,
    seed_param: str = DEFAULT_SEED_PARAM,
) -> dict[int, float]:
    """Metric value per seed for one method at one learning-curve point.

    The metric key is ``{metric}/{target}/h{horizon}`` as in
    :func:`twinlink_ml.evaluation.resolution.collect_spreads`. Records of other methods or
    points are skipped. A seed that appears twice with different values is an error (a log
    that mixes a second varied factor), not averaged.
    """
    key_name = f"{metric}/{target}/h{horizon}"
    values: dict[int, float] = {}
    for record in records:
        if key_name not in record.metrics:
            continue
        if str(record.params.get("model/label", record.tags.get("model", "?"))) != label:
            continue
        if int(record.params["n_train"]) != n_train:
            continue
        if seed_param not in record.params:
            raise ValueError(f"record of {label!r} at n={n_train} has no parameter {seed_param!r}")
        seed = int(record.params[seed_param])
        value = float(record.metrics[key_name])
        if seed in values and not math.isclose(values[seed], value):
            raise ValueError(
                f"seed {seed} of {label!r} at n={n_train} occurs with two values "
                f"({values[seed]} and {value}); filter the records to one factor level first"
            )
        values[seed] = value
    return values


def paired_verdicts(
    records: Sequence[RunRecord],
    *,
    n_train: int,
    target: str,
    horizon: int,
    metric: str,
    reference_label: str,
    candidate_labels: Sequence[str] | None = None,
    seed_param: str = DEFAULT_SEED_PARAM,
    confidence: float = 0.95,
) -> list[PairedComparison]:
    """Each candidate against one reference at one point, paired over the seeds.

    ``candidate_labels=None`` takes every other label found at the point. One test per
    candidate, no multiplicity correction (see module docstring).
    """
    key_name = f"{metric}/{target}/h{horizon}"
    labels = {
        str(record.params.get("model/label", record.tags.get("model", "?")))
        for record in records
        if key_name in record.metrics and int(record.params["n_train"]) == n_train
    }
    if reference_label not in labels:
        raise ValueError(f"no records for reference {reference_label!r} at n={n_train}")
    candidates = (
        sorted(labels - {reference_label}) if candidate_labels is None else list(candidate_labels)
    )
    reference = paired_values(
        records,
        label=reference_label,
        n_train=n_train,
        target=target,
        horizon=horizon,
        metric=metric,
        seed_param=seed_param,
    )
    return [
        paired_comparison(
            paired_values(
                records,
                label=label,
                n_train=n_train,
                target=target,
                horizon=horizon,
                metric=metric,
                seed_param=seed_param,
            ),
            reference,
            candidate_label=label,
            reference_label=reference_label,
            confidence=confidence,
        )
        for label in candidates
    ]

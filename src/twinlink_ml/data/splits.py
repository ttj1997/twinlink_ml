"""Deterministic splits at run level (run-level splits).

This module knows neither windows nor tensors. It assigns only run IDs to the sets
``train`` / ``val`` / ``test``. The separation is intentional: the leakage regression
thereby checks a pure set operation and not the interplay with windowing.

Two invariants
--------------
1. **Reproducibility.** The assignment is a function of (run ID, seed, splitter
   configuration). No dependence on process salt (``PYTHONHASHSEED``), dict iteration
   order or import time.
2. **Run-count invariance of ``val`` and ``test``.** Learning-curve points arise
   exclusively by shrinking ``train`` (:meth:`RunSplit.subset`). Otherwise the y-axis of
   the learning curve has a different meaning at every point.

Three assignment strategies behind one ABC
------------------------------------------
* :class:`IndexBlockSplitter` -- default for self-generated QMC designs. ``train`` is
  a true prefix of the Sobol sequence, ``val``/``test`` are subsequent blocks.
* :class:`HashRunSplitter` -- for runs whose order is not under our own control
  (Dyssol deliveries from TUHH-A, application partner history). Stable under pool growth.
* :class:`PredefinedRunSplitter` -- for domain cuts from AP 2 on (train = grey-box,
  test = Dyssol or real data).

On the choice of the learning-curve points
------------------------------------------
``DEFAULT_LEARNING_CURVE_SIZES`` are powers of two, not 5/10/25/50/100 as in the
first version of ``05_arbeitsplan_ap1.md``. Reason: the balance property of a
Sobol sequence holds only for :math:`n = 2^k`; ``scipy.stats.qmc.Sobol.random`` warns
accordingly. A 25-point prefix is a low-discrepancy sequence, but not a balanced
one -- for our own points this would violate exactly the argument with which
Latin hypercube was rejected in step 2.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from ..simulation.excitation import ExcitationSchedule

__all__ = [
    "DEFAULT_LEARNING_CURVE_SIZES",
    "HASH_NAMESPACE",
    "HashRunSplitter",
    "IndexBlockSplitter",
    "LearningCurvePoint",
    "PredefinedRunSplitter",
    "RunSplit",
    "RunSplitter",
    "SplitConfigError",
    "SplitDegenerateError",
    "SplitError",
    "SplitName",
    "SplitRatios",
    "design_key",
    "learning_curve",
    "realized_ratios",
    "signal_kind_strata",
    "split_report",
    "split_schedule",
    "stable_unit_hash",
    "stratum_coverage",
]


HASH_NAMESPACE: Final[str] = "twinlink_ml/splits/v1"
"""Namespace aller Hashes dieses Moduls; wird bei geänderter Semantik hochgezählt."""

_SEP: Final[bytes] = b"\x1f"
_DIGEST_BYTES: Final[int] = 8
_DIGEST_SCALE: Final[float] = float(1 << (8 * _DIGEST_BYTES))
_RATIO_TOL: Final[float] = 1e-9

DEFAULT_LEARNING_CURVE_SIZES: Final[tuple[int, ...]] = (8, 16, 32, 64, 128)
"""Stützstellen der Lernkurve. Zweierpotenzen wegen der Sobol-Balance, siehe Modul-Docstring."""


class SplitError(Exception):
    """Base class for errors of this module."""


class SplitConfigError(SplitError):
    """The splitter configuration or the run pool passed in is invalid."""


class SplitDegenerateError(SplitError):
    """The assignment is formally valid but methodologically unusable.

    Typical case: an empty or single-element test set with a small pool. Deliberately
    an error and not a warning -- a learning curve whose left point is measured on a
    single test run is a random variable and not a measurement.
    """


class SplitName(str, Enum):
    """Canonical split names. ``str`` enum for YAML round trip without a converter."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


# ----------------------------------------------------------------------- Hashing


def _hash_bytes(*parts: str | int, namespace: str) -> bytes:
    """Common BLAKE2b construction: namespace, then parts concatenated with a separator byte.

    Without a separator, ``("run", "12")`` and ``("run1", "2")`` collide. Deliberately
    **not** ``hash()``: Python's string hash is salted per process (``PYTHONHASHSEED``). A
    split built with it is consistent within one process and different between two
    processes -- the error goes unnoticed in tests and invalidates every reproducibility
    claim. ``seed_everything`` does set ``PYTHONHASHSEED``, but only at run time, which no
    longer reaches the interpreter start.

    Returned raw, not as a number or hex string: :func:`stable_unit_hash` and
    :func:`_digest_of` are thin wrappers that merely represent the same bytes differently
    (unit interval vs. fingerprint string).
    """
    digest = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    digest.update(namespace.encode("utf-8"))
    for part in parts:
        digest.update(_SEP)
        digest.update(str(part).encode("utf-8"))
    return digest.digest()


def stable_unit_hash(*parts: str | int, namespace: str = HASH_NAMESPACE) -> float:
    """Process-independent hash of the parts onto the interval :math:`[0, 1)`.

    Parameters
    ----------
    parts
        Parts to hash in fixed order, see :func:`_hash_bytes`.
    namespace
        Prefix for versioning the hash semantics.

    Returns
    -------
    float
        Value in :math:`[0, 1)` from the first 64 bits of the BLAKE2b digest.
    """
    return int.from_bytes(_hash_bytes(*parts, namespace=namespace), "big") / _DIGEST_SCALE


def _digest_of(*parts: str | int) -> str:
    return _hash_bytes(*parts, namespace=HASH_NAMESPACE).hex()


def design_key(run_id: str) -> str | None:
    """Design part of a run ID in the scheme ``{name}-{fingerprint}-{index:05d}``.

    ``MixedExcitationDesign.build`` produces IDs in exactly this scheme, and the
    fingerprint contains ``n_runs``. Two separately built designs are therefore
    nested **neither in the IDs nor in content** -- only
    ``ExcitationSchedule.subset`` is. This function makes the difference
    checkable; :meth:`RunSplitter.split` uses it as a guard.

    Returns
    -------
    str or None
        ``"{name}-{fingerprint}"``, or ``None`` if the ID does not follow the scheme
        (foreign IDs from Dyssol or plant data are allowed).
    """
    parts = run_id.rsplit("-", 2)
    if len(parts) != 3 or not parts[2].isdigit():
        return None
    return f"{parts[0]}-{parts[1]}"


# ----------------------------------------------------------------- Configuration


@dataclass(frozen=True, slots=True)
class SplitRatios:
    """Fractions of the three splits in the run pool.

    Attributes
    ----------
    train, val, test
        Fractions in :math:`[0, 1]`, sum 1. ``val = 0`` is allowed (fixed epoch budget
        instead of early stopping); ``train`` and ``test`` are not.
    """

    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > _RATIO_TOL:
            raise SplitConfigError(f"Anteile müssen sich zu 1 summieren, sind {total!r}")
        if self.train <= 0.0 or self.test <= 0.0:
            raise SplitConfigError("train und test brauchen einen positiven Anteil")
        if self.val < 0.0:
            raise SplitConfigError("val darf nicht negativ sein")

    @property
    def edges(self) -> tuple[float, float]:
        """Cumulative edges for the bucket assignment: ``(train, train + val)``."""
        return self.train, self.train + self.val

    @property
    def digest(self) -> str:
        """Fingerprint of the fractions."""
        return _digest_of("ratios", f"{self.train:.12g}", f"{self.val:.12g}", f"{self.test:.12g}")


# ---------------------------------------------------------------- Result objects


@dataclass(frozen=True, slots=True)
class RunSplit:
    """Assignment of run IDs to train/val/test.

    The order of ``train`` **carries meaning**: it defines the nesting of the
    learning-curve subsets (:meth:`subset`). For a Sobol design this is the design order,
    so that ``train[:n]`` remains a prefix of the sequence. Canonical sorting would turn
    ``train[:n]`` into an arbitrary subset.
    """

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    splitter_digest: str

    def __post_init__(self) -> None:
        if not self.train:
            raise SplitDegenerateError("Trainingsmenge ist leer")
        if not self.test:
            raise SplitDegenerateError("Testmenge ist leer")
        mapping = self.as_mapping
        for name, ids in mapping.items():
            counts = Counter(ids)
            duplicates = sorted(key for key, value in counts.items() if value > 1)
            if duplicates:
                raise SplitError(f"Doppelte Lauf-IDs in {name.value}: {duplicates}")
        pairs = (
            (SplitName.TRAIN, SplitName.VAL),
            (SplitName.TRAIN, SplitName.TEST),
            (SplitName.VAL, SplitName.TEST),
        )
        for left, right in pairs:
            shared = set(mapping[left]) & set(mapping[right])
            if shared:
                raise SplitError(f"{left.value} und {right.value} teilen Läufe: {sorted(shared)}")

    @property
    def as_mapping(self) -> Mapping[SplitName, tuple[str, ...]]:
        """Split name to run IDs."""
        return {
            SplitName.TRAIN: self.train,
            SplitName.VAL: self.val,
            SplitName.TEST: self.test,
        }

    @property
    def n_runs(self) -> int:
        """Total number of assigned runs."""
        return len(self.train) + len(self.val) + len(self.test)

    @property
    def all_runs(self) -> tuple[str, ...]:
        """All runs in the order train, val, test."""
        return self.train + self.val + self.test

    @property
    def digest(self) -> str:
        """Fingerprint of the concrete assignment for experiment tracking.

        Over the **ordered** tuples, because the training order determines the
        learning-curve subsets.
        """
        return _digest_of(
            "split",
            self.splitter_digest,
            "|".join(self.train),
            "|".join(self.val),
            "|".join(self.test),
        )

    def membership(self) -> Mapping[str, SplitName]:
        """Inverse mapping run ID to split, for filtering in the data set."""
        return {run_id: name for name, ids in self.as_mapping.items() for run_id in ids}

    def subset(self, n_train: int) -> RunSplit:
        """Learning-curve point with ``n_train`` training runs.

        Shrinks **exclusively** ``train``. ``val`` and ``test`` stay unchanged, so that
        all points of the learning curve are measured on the same test set. Same name and
        same semantics as ``ExcitationSchedule.subset(n)`` -- nested prefix.
        """
        if n_train < 1:
            raise SplitConfigError(f"n_train muss >= 1 sein, ist {n_train}")
        if n_train > len(self.train):
            raise SplitConfigError(
                f"n_train={n_train} > verfügbare Trainingsläufe {len(self.train)}"
            )
        return RunSplit(
            train=self.train[:n_train],
            val=self.val,
            test=self.test,
            splitter_digest=self.splitter_digest,
        )


@dataclass(frozen=True, slots=True)
class LearningCurvePoint:
    """One point of the learning curve.

    Attributes
    ----------
    n_train
        Number of training runs -- the x-axis of the learning curve.
    split
        Complete split object. Normalisation statistics are drawn from ``split.train``
        of **this** point, never from the overall pool: otherwise the model at the
        smallest point has information from all runs and the left end of the curve is
        too optimistic -- exactly the region that AP 1 is about.
    """

    n_train: int
    split: RunSplit


def learning_curve(
    split: RunSplit,
    sizes: Sequence[int] = DEFAULT_LEARNING_CURVE_SIZES,
    *,
    skip_oversized: bool = True,
) -> tuple[LearningCurvePoint, ...]:
    """Generate nested learning-curve points from a split.

    Parameters
    ----------
    split
        Base split; ``split.train`` provides the maximum point.
    sizes
        Ascending, unique numbers of training runs.
    skip_oversized
        ``True``: points that are too large are omitted. ``False``: error. Set ``False``
        for the final report, so that an unnoticed truncated curve does not pass as
        complete.
    """
    requested = [int(size) for size in sizes]
    if requested != sorted(set(requested)):
        raise SplitConfigError("sizes müssen aufsteigend und eindeutig sein")
    points: list[LearningCurvePoint] = []
    for size in requested:
        if size > len(split.train):
            if skip_oversized:
                continue
            raise SplitConfigError(f"Stützstelle n={size} > Trainings-Pool {len(split.train)}")
        points.append(LearningCurvePoint(n_train=size, split=split.subset(size)))
    if not points:
        raise SplitDegenerateError(
            f"Keine Stützstelle passt in einen Trainings-Pool der Größe {len(split.train)}"
        )
    return tuple(points)


# --------------------------------------------------------------------- Splitter


class RunSplitter(ABC):
    """Abstract base class for the assignment of runs to splits.

    Open-closed: a new strategy (domain cut grey-box to Dyssol in AP 2, temporal
    separation for plant history) must not require any change to training or
    evaluation code.
    """

    min_per_split: int = 2
    """Mindestgröße jedes nichtleeren Splits. Darunter ist die Metrik Rauschen."""

    require_single_design: bool = True
    """Alle Läufe müssen aus demselben Anregungsdesign stammen, siehe :func:`design_key`."""

    @abstractmethod
    def split(self, run_ids: Sequence[str]) -> RunSplit:
        """Assign runs. ``run_ids`` in design order, without duplicates."""

    @property
    @abstractmethod
    def config_digest(self) -> str:
        """Fingerprint of the splitter configuration including the seed."""

    def _check_input(self, run_ids: Sequence[str]) -> tuple[str, ...]:
        ids = tuple(run_ids)
        if not ids:
            raise SplitConfigError("Leere Lauf-Liste")
        if any(not isinstance(run_id, str) or not run_id for run_id in ids):
            raise SplitConfigError("Lauf-IDs müssen nichtleere Strings sein")
        counts = Counter(ids)
        duplicates = sorted(key for key, value in counts.items() if value > 1)
        if duplicates:
            raise SplitConfigError(f"Doppelte Lauf-IDs im Pool: {duplicates}")
        if self.require_single_design:
            keys = {key for key in (design_key(run_id) for run_id in ids) if key is not None}
            if len(keys) > 1:
                raise SplitConfigError(
                    f"Pool enthält {len(keys)} verschiedene Anregungsdesigns: "
                    f"{sorted(keys)}. Zwei separat gebaute Designs sind nicht "
                    "verschachtelt -- Maximaldesign einmal bauen und subset() nutzen, "
                    "oder require_single_design=False setzen (Domänenschnitt)."
                )
        return ids

    def _finalize(self, train: Sequence[str], val: Sequence[str], test: Sequence[str]) -> RunSplit:
        for name, ids in (("train", train), ("test", test)):
            if len(ids) < self.min_per_split:
                raise SplitDegenerateError(
                    f"{name} hat {len(ids)} Läufe, Minimum ist {self.min_per_split}. "
                    "Bei kleinem Pool IndexBlockSplitter mit expliziten Blockgrößen "
                    "verwenden statt anteiliger Zuordnung."
                )
        if val and len(val) < self.min_per_split:
            raise SplitDegenerateError(
                f"val hat {len(val)} Läufe, Minimum ist {self.min_per_split}"
            )
        return RunSplit(
            train=tuple(train),
            val=tuple(val),
            test=tuple(test),
            splitter_digest=self.config_digest,
        )


@dataclass(frozen=True, slots=True)
class IndexBlockSplitter(RunSplitter):
    """Contiguous blocks in design order -- default for our own QMC designs.

    ``train = ids[0:n_train]``, then ``val``, then ``test``. The training prefix stays
    a prefix of the Sobol sequence, ``subset(n)`` inherits the nesting, and the val and
    test blocks are space-filling as well.

    Why not hashing for this case: a hash-filtered Sobol prefix is no longer a
    low-discrepancy sequence -- the thinning is not the one for which the discrepancy
    bound holds. The error that was removed from the sampler in step 2 would come back
    via the splitter.

    Block sizes should be powers of two, so that ``val`` and ``test`` also lie on
    Sobol balance boundaries.
    """

    n_train: int
    n_val: int = 0
    n_test: int = 0
    strict_pool_size: bool = True
    min_per_split: int = 2
    require_single_design: bool = True

    def __post_init__(self) -> None:
        if self.n_train < 1 or self.n_test < 1:
            raise SplitConfigError("n_train und n_test müssen >= 1 sein")
        if self.n_val < 0:
            raise SplitConfigError("n_val darf nicht negativ sein")

    @property
    def required_pool_size(self) -> int:
        """Number of runs the design must supply."""
        return self.n_train + self.n_val + self.n_test

    @property
    def config_digest(self) -> str:
        """Fingerprint of the block sizes."""
        return _digest_of("IndexBlockSplitter", self.n_train, self.n_val, self.n_test)

    def split(self, run_ids: Sequence[str]) -> RunSplit:
        """Cut the blocks."""
        ids = self._check_input(run_ids)
        need = self.required_pool_size
        if len(ids) < need:
            raise SplitConfigError(f"Pool hat {len(ids)} Läufe, benötigt werden {need}")
        if self.strict_pool_size and len(ids) > need:
            raise SplitConfigError(
                f"Pool hat {len(ids)} Läufe, Blöcke belegen {need}. "
                f"{len(ids) - need} Läufe blieben ungenutzt -- strict_pool_size=False "
                "setzen, wenn das gewollt ist."
            )
        first = self.n_train
        second = first + self.n_val
        return self._finalize(ids[:first], ids[first:second], ids[second : second + self.n_test])


@dataclass(frozen=True, slots=True)
class HashRunSplitter(RunSplitter):
    """Independent bucket assignment per run -- for externally determined run pools.

    Every run is assigned from its ID and the seed alone, without knowledge of the
    pool. The assignment is therefore **stable under pool growth**: when further Dyssol
    deliveries arrive, the assignment of no existing run changes. With "shuffle and
    cut", runs migrate between the splits as soon as the pool grows -- yesterday's test
    run becomes today's training run and all test errors reported so far are invalid.

    Price: the fractions hold only in expectation. The realised sizes scatter
    binomially and can degenerate with a small pool.
    """

    seed: int
    ratios: SplitRatios = field(default_factory=SplitRatios)
    min_per_split: int = 2
    require_single_design: bool = False

    @property
    def config_digest(self) -> str:
        """Fingerprint from seed and fractions."""
        return _digest_of("HashRunSplitter", self.seed, self.ratios.digest)

    def assign(self, run_id: str) -> SplitName:
        """Split of a single run, without knowledge of the rest of the pool."""
        value = stable_unit_hash("assign", self.seed, run_id)
        train_edge, val_edge = self.ratios.edges
        if value < train_edge:
            return SplitName.TRAIN
        if value < val_edge:
            return SplitName.VAL
        return SplitName.TEST

    def split(self, run_ids: Sequence[str]) -> RunSplit:
        """Assign all runs individually."""
        ids = self._check_input(run_ids)
        buckets: dict[SplitName, list[str]] = {name: [] for name in SplitName}
        for run_id in ids:
            buckets[self.assign(run_id)].append(run_id)
        return self._finalize(
            buckets[SplitName.TRAIN], buckets[SplitName.VAL], buckets[SplitName.TEST]
        )


@dataclass(frozen=True, slots=True)
class PredefinedRunSplitter(RunSplitter):
    """Explicitly specified assignment -- for domain cuts from AP 2 on.

    Use case: ``train`` from the grey-box model, ``test`` from Dyssol or from plant
    data. That is not a random cut but a domain change (domain shift); a hash-based
    assignment would destroy exactly that. Deliberately provided now, so that the
    training code does not have to be tinkered with later.
    """

    label: str
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    min_per_split: int = 1
    require_single_design: bool = False

    @property
    def config_digest(self) -> str:
        """Fingerprint from the label of the cut."""
        return _digest_of("PredefinedRunSplitter", self.label)

    def split(self, run_ids: Sequence[str]) -> RunSplit:
        """Check the specification against the pool and adopt it."""
        ids = set(self._check_input(run_ids))
        declared = set(self.train_ids) | set(self.val_ids) | set(self.test_ids)
        missing = declared - ids
        if missing:
            raise SplitConfigError(f"Zugeordnete, aber fehlende Läufe: {sorted(missing)}")
        unassigned = ids - declared
        if unassigned:
            raise SplitConfigError(
                f"Nicht zugeordnete Läufe: {sorted(unassigned)}. Eine unvollständige "
                "Zuordnung verschweigt, welche Daten ungenutzt bleiben."
            )
        return self._finalize(self.train_ids, self.val_ids, self.test_ids)


# ------------------------------------------------ Coupling to the excitation design


def split_schedule(schedule: ExcitationSchedule, splitter: RunSplitter) -> RunSplit:
    """Split from an :class:`~twinlink_ml.simulation.excitation.ExcitationSchedule`.

    Thin adapter function: ``schedule.run_id`` is already in design order and carries
    the design fingerprint, so that the split fingerprint alone identifies the data.
    """
    return splitter.split(schedule.run_id)


def signal_kind_strata(schedule: ExcitationSchedule, channel: int) -> Mapping[str, str]:
    """Signal kind per run for one action channel, as a stratification feature.

    The signal kind is drawn independently per run **and per channel**. There is
    therefore no single stratum per run; coverage has to be checked channel by channel.
    Relevant because for a ramp the tracking lag is directly proportional to the time
    constant -- a test block without ramps measures the multi-step rollout error on the
    least informative signal type.
    """
    kinds = schedule.signal_kinds
    if kinds and not 0 <= channel < len(kinds[0]):
        raise SplitConfigError(f"Kanalindex {channel} außerhalb von 0..{len(kinds[0]) - 1}")
    return {run_id: kinds[index][channel] for index, run_id in enumerate(schedule.run_id)}


# ------------------------------------------------------------------- Diagnostics


def realized_ratios(split: RunSplit) -> Mapping[SplitName, float]:
    """Actually realised fractions; for :class:`HashRunSplitter` not the target values."""
    total = split.n_runs
    return {name: len(ids) / total for name, ids in split.as_mapping.items()}


def stratum_coverage(
    split: RunSplit, strata: Mapping[str, str]
) -> Mapping[SplitName, Mapping[str, int]]:
    """Coverage of a stratification feature per split.

    Exact stratification would need knowledge of the pool and would thereby give up the
    stability of :class:`HashRunSplitter`. Recommendation: diagnose here, on a gap change
    the seed in a documented way, do not complicate the mechanism.
    """
    missing = set(split.all_runs) - set(strata)
    if missing:
        raise SplitConfigError(f"Kein Stratum für Läufe: {sorted(missing)}")
    return {
        name: dict(Counter(strata[run_id] for run_id in ids))
        for name, ids in split.as_mapping.items()
    }


def split_report(split: RunSplit, strata: Mapping[str, str] | None = None) -> str:
    """Short report for the run log. Text only, no side effects."""
    ratios = realized_ratios(split)
    lines = [
        f"Split-Fingerprint : {split.digest}",
        f"Splitter          : {split.splitter_digest}",
        f"Läufe gesamt      : {split.n_runs}",
    ]
    for name, ids in split.as_mapping.items():
        lines.append(f"  {name.value:<5} n={len(ids):>4}  Anteil={ratios[name]:.3f}")
    if strata is not None:
        coverage = stratum_coverage(split, strata)
        levels = sorted({level for counts in coverage.values() for level in counts})
        lines.append("Stratumabdeckung:")
        lines.append("  " + "".join(f"{level:>12}" for level in ["", *levels]))
        for name in SplitName:
            row = "".join(f"{coverage[name].get(level, 0):>12}" for level in levels)
            gap = "" if all(coverage[name].get(lv, 0) > 0 for lv in levels) else "  <- Lücke"
            lines.append(f"  {name.value:<12}{row}{gap}")
    return "\n".join(lines)

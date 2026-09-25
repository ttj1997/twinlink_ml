"""Deterministische Splits auf Lauf-Ebene (run-level splits).

Dieses Modul kennt weder Fenster noch Tensoren. Es ordnet ausschließlich Lauf-IDs den
Mengen ``train`` / ``val`` / ``test`` zu. Die Trennung ist beabsichtigt: die
Leakage-Regression prüft damit eine reine Mengenoperation und nicht das Zusammenspiel
mit der Fensterung.

Zwei Invarianten
----------------
1. **Reproduzierbarkeit.** Die Zuordnung ist eine Funktion von (Lauf-ID, Seed,
   Splitter-Konfiguration). Keine Abhängigkeit von Prozess-Salt (``PYTHONHASHSEED``),
   Dict-Iterationsreihenfolge oder Importzeitpunkt.
2. **Laufzahl-Invarianz von ``val`` und ``test``.** Lernkurven-Stützstellen entstehen
   ausschließlich durch Verkleinern von ``train`` (:meth:`RunSplit.subset`). Sonst hat
   die y-Achse der Lernkurve an jedem Punkt eine andere Bedeutung.

Drei Zuordnungsstrategien hinter einer ABC
------------------------------------------
* :class:`IndexBlockSplitter` -- Default für selbst erzeugte QMC-Designs. ``train`` ist
  ein echtes Präfix der Sobol-Folge, ``val``/``test`` sind anschließende Blöcke.
* :class:`HashRunSplitter` -- für Läufe, deren Reihenfolge nicht unter eigener Kontrolle
  steht (Dyssol-Lieferungen aus TUHH-A, Anwendungspartner-Historie). Stabil unter Pool-Wachstum.
* :class:`PredefinedRunSplitter` -- für Domänenschnitte ab AP 2 (train = Grey-Box,
  test = Dyssol oder Realdaten).

Zur Wahl der Stützstellen
-------------------------
``DEFAULT_LEARNING_CURVE_SIZES`` sind Zweierpotenzen, nicht 5/10/25/50/100 wie in der
ersten Fassung von ``05_arbeitsplan_ap1.md``. Grund: die Balance-Eigenschaft einer
Sobol-Folge gilt nur für :math:`n = 2^k`; ``scipy.stats.qmc.Sobol.random`` warnt
entsprechend. Ein 25-Punkt-Präfix ist eine Folge niedriger Diskrepanz, aber keine
balancierte -- damit wäre für die eigenen Stützstellen genau das Argument verletzt, mit
dem in Schritt 2 Latin-Hypercube verworfen wurde.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover - nur für Typprüfung
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
    """Basisklasse für Fehler dieses Moduls."""


class SplitConfigError(SplitError):
    """Die Splitter-Konfiguration oder der übergebene Lauf-Pool ist unzulässig."""


class SplitDegenerateError(SplitError):
    """Die Zuordnung ist formal gültig, aber methodisch unbrauchbar.

    Typischer Fall: eine leere oder einelementige Testmenge bei kleinem Pool. Bewusst
    ein Fehler und keine Warnung -- eine Lernkurve, deren linke Stützstelle auf einem
    einzigen Testlauf gemessen ist, ist eine Zufallsvariable und kein Messwert.
    """


class SplitName(str, Enum):
    """Kanonische Split-Namen. ``str``-Enum für YAML-Roundtrip ohne Konverter."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


# ----------------------------------------------------------------------- Hashing


def _hash_bytes(*parts: str | int, namespace: str) -> bytes:
    """Gemeinsamer BLAKE2b-Aufbau: Namespace, dann Teile mit Trennbyte verkettet.

    Ohne Trenner kollidieren ``("run", "12")`` und ``("run1", "2")``. Bewusst **nicht**
    ``hash()``: Pythons String-Hash ist pro Prozess gesalzen (``PYTHONHASHSEED``). Ein
    damit gebauter Split ist innerhalb eines Prozesses konsistent und zwischen zwei
    Prozessen verschieden -- der Fehler fällt in Tests nicht auf und macht jede
    Reproduzierbarkeitsaussage ungültig. ``seed_everything`` setzt ``PYTHONHASHSEED``
    zwar, aber erst zur Laufzeit, was den Interpreterstart nicht mehr erreicht.

    Rückgabe roh, nicht als Zahl oder Hexstring: :func:`stable_unit_hash` und
    :func:`_digest_of` sind dünne Wrapper, die dieselben Bytes nur unterschiedlich
    darstellen (Einheitsintervall vs. Fingerprint-String).
    """
    digest = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    digest.update(namespace.encode("utf-8"))
    for part in parts:
        digest.update(_SEP)
        digest.update(str(part).encode("utf-8"))
    return digest.digest()


def stable_unit_hash(*parts: str | int, namespace: str = HASH_NAMESPACE) -> float:
    """Prozessunabhängiger Hash der Teile auf das Intervall :math:`[0, 1)`.

    Parameters
    ----------
    parts
        Zu hashende Teile in fester Reihenfolge, siehe :func:`_hash_bytes`.
    namespace
        Präfix zur Versionierung der Hash-Semantik.

    Returns
    -------
    float
        Wert in :math:`[0, 1)` aus den ersten 64 Bit des BLAKE2b-Digests.
    """
    return int.from_bytes(_hash_bytes(*parts, namespace=namespace), "big") / _DIGEST_SCALE


def _digest_of(*parts: str | int) -> str:
    return _hash_bytes(*parts, namespace=HASH_NAMESPACE).hex()


def design_key(run_id: str) -> str | None:
    """Design-Anteil einer Lauf-ID im Schema ``{name}-{fingerprint}-{index:05d}``.

    ``MixedExcitationDesign.build`` erzeugt IDs in genau diesem Schema, und der
    Fingerprint enthält ``n_runs``. Zwei getrennt gebaute Designs sind deshalb
    **weder in den IDs noch im Inhalt** verschachtelt -- nur
    ``ExcitationSchedule.subset`` ist es. Diese Funktion macht den Unterschied
    prüfbar; :meth:`RunSplitter.split` nutzt sie als Wächter.

    Returns
    -------
    str or None
        ``"{name}-{fingerprint}"``, oder ``None`` wenn die ID dem Schema nicht folgt
        (fremde IDs aus Dyssol oder Anlagendaten sind erlaubt).
    """
    parts = run_id.rsplit("-", 2)
    if len(parts) != 3 or not parts[2].isdigit():
        return None
    return f"{parts[0]}-{parts[1]}"


# ----------------------------------------------------------------- Konfiguration


@dataclass(frozen=True, slots=True)
class SplitRatios:
    """Anteile der drei Splits am Lauf-Pool.

    Attributes
    ----------
    train, val, test
        Anteile in :math:`[0, 1]`, Summe 1. ``val = 0`` ist zulässig (festes
        Epochenbudget statt Early Stopping); ``train`` und ``test`` nicht.
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
        """Kumulative Grenzen für die Bucket-Zuordnung: ``(train, train + val)``."""
        return self.train, self.train + self.val

    @property
    def digest(self) -> str:
        """Fingerprint der Anteile."""
        return _digest_of("ratios", f"{self.train:.12g}", f"{self.val:.12g}", f"{self.test:.12g}")


# --------------------------------------------------------------- Ergebnisobjekte


@dataclass(frozen=True, slots=True)
class RunSplit:
    """Zuordnung von Lauf-IDs zu train/val/test.

    Die Reihenfolge von ``train`` ist **bedeutungstragend**: sie definiert die
    Verschachtelung der Lernkurven-Teilmengen (:meth:`subset`). Bei einem Sobol-Design
    ist das die Design-Reihenfolge, sodass ``train[:n]`` ein Präfix der Folge bleibt.
    Kanonisches Sortieren würde ``train[:n]`` zu einer beliebigen Teilmenge machen.
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
        """Split-Name auf Lauf-IDs."""
        return {
            SplitName.TRAIN: self.train,
            SplitName.VAL: self.val,
            SplitName.TEST: self.test,
        }

    @property
    def n_runs(self) -> int:
        """Gesamtzahl zugeordneter Läufe."""
        return len(self.train) + len(self.val) + len(self.test)

    @property
    def all_runs(self) -> tuple[str, ...]:
        """Alle Läufe in der Reihenfolge train, val, test."""
        return self.train + self.val + self.test

    @property
    def digest(self) -> str:
        """Fingerprint der konkreten Zuordnung für das Experiment-Tracking.

        Über die **geordneten** Tupel, weil die Trainingsreihenfolge die
        Lernkurven-Teilmengen bestimmt.
        """
        return _digest_of(
            "split",
            self.splitter_digest,
            "|".join(self.train),
            "|".join(self.val),
            "|".join(self.test),
        )

    def membership(self) -> Mapping[str, SplitName]:
        """Umkehrabbildung Lauf-ID auf Split, für die Filterung im Dataset."""
        return {run_id: name for name, ids in self.as_mapping.items() for run_id in ids}

    def subset(self, n_train: int) -> RunSplit:
        """Stützstelle der Lernkurve mit ``n_train`` Trainingsläufen.

        Verkleinert **ausschließlich** ``train``. ``val`` und ``test`` bleiben
        unverändert, damit alle Punkte der Lernkurve auf derselben Testmenge gemessen
        werden. Namensgleich und semantikgleich zu
        ``ExcitationSchedule.subset(n)`` -- verschachteltes Präfix.
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
    """Eine Stützstelle der Lernkurve.

    Attributes
    ----------
    n_train
        Anzahl Trainingsläufe -- die x-Achse der Lernkurve.
    split
        Vollständiges Split-Objekt. Normalisierungsstatistiken werden aus
        ``split.train`` **dieser** Stützstelle gezogen, nie aus dem Gesamt-Pool: sonst
        hat das Modell an der kleinsten Stützstelle Information aus allen Läufen und
        das linke Ende der Kurve ist zu optimistisch -- genau die Region, um die es in
        AP 1 geht.
    """

    n_train: int
    split: RunSplit


def learning_curve(
    split: RunSplit,
    sizes: Sequence[int] = DEFAULT_LEARNING_CURVE_SIZES,
    *,
    skip_oversized: bool = True,
) -> tuple[LearningCurvePoint, ...]:
    """Verschachtelte Stützstellen aus einem Split erzeugen.

    Parameters
    ----------
    split
        Basis-Split; ``split.train`` liefert die maximale Stützstelle.
    sizes
        Aufsteigende, eindeutige Anzahlen von Trainingsläufen.
    skip_oversized
        ``True``: zu große Stützstellen werden weggelassen. ``False``: Fehler. Für den
        Endbericht ``False`` setzen, damit eine unbemerkt verkürzte Kurve nicht als
        vollständig durchgeht.
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
    """Abstrakte Basisklasse für die Zuordnung von Läufen zu Splits.

    Open-Closed: eine neue Strategie (Domänenschnitt Grey-Box auf Dyssol in AP 2,
    zeitliche Trennung bei Anlagenhistorie) darf keine Änderung an Trainings- oder
    Evaluationscode erfordern.
    """

    min_per_split: int = 2
    """Mindestgröße jedes nichtleeren Splits. Darunter ist die Metrik Rauschen."""

    require_single_design: bool = True
    """Alle Läufe müssen aus demselben Anregungsdesign stammen, siehe :func:`design_key`."""

    @abstractmethod
    def split(self, run_ids: Sequence[str]) -> RunSplit:
        """Läufe zuordnen. ``run_ids`` in Design-Reihenfolge, ohne Duplikate."""

    @property
    @abstractmethod
    def config_digest(self) -> str:
        """Fingerprint der Splitter-Konfiguration inklusive Seed."""

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
    """Zusammenhängende Blöcke in Design-Reihenfolge -- Default für eigene QMC-Designs.

    ``train = ids[0:n_train]``, dann ``val``, dann ``test``. Das Trainings-Präfix bleibt
    ein Präfix der Sobol-Folge, ``subset(n)`` erbt die Verschachtelung, Val- und
    Testblock sind ebenfalls raumfüllend.

    Warum nicht Hashing für diesen Fall: ein hash-gefiltertes Sobol-Präfix ist keine
    Folge niedriger Diskrepanz mehr -- die Ausdünnung ist nicht die, für die die
    Diskrepanzschranke gilt. Der Fehler, der in Schritt 2 aus dem Sampler entfernt
    wurde, käme über den Splitter zurück.

    Blockgrößen sollten Zweierpotenzen sein, damit auch ``val`` und ``test`` auf
    Sobol-Balance-Grenzen liegen.
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
        """Anzahl Läufe, die das Design liefern muss."""
        return self.n_train + self.n_val + self.n_test

    @property
    def config_digest(self) -> str:
        """Fingerprint der Blockgrößen."""
        return _digest_of("IndexBlockSplitter", self.n_train, self.n_val, self.n_test)

    def split(self, run_ids: Sequence[str]) -> RunSplit:
        """Blöcke schneiden."""
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
    """Unabhängige Bucket-Zuordnung je Lauf -- für fremdbestimmte Lauf-Pools.

    Jeder Lauf wird allein aus seiner ID und dem Seed zugeordnet, ohne Kenntnis des
    Pools. Damit ist die Zuordnung **stabil unter Pool-Wachstum**: kommen
    Dyssol-Lieferungen nach, ändert sich die Zuordnung keines bestehenden Laufs. Bei
    "mischen und schneiden" wandern Läufe zwischen den Splits, sobald der Pool wächst --
    ein Testlauf von gestern wird zum Trainingslauf von heute und alle bisher
    berichteten Testfehler sind ungültig.

    Preis: die Anteile gelten nur im Erwartungswert. Die realisierten Größen streuen
    binomial und können bei kleinem Pool degenerieren.
    """

    seed: int
    ratios: SplitRatios = field(default_factory=SplitRatios)
    min_per_split: int = 2
    require_single_design: bool = False

    @property
    def config_digest(self) -> str:
        """Fingerprint aus Seed und Anteilen."""
        return _digest_of("HashRunSplitter", self.seed, self.ratios.digest)

    def assign(self, run_id: str) -> SplitName:
        """Split eines einzelnen Laufs, ohne Kenntnis des übrigen Pools."""
        value = stable_unit_hash("assign", self.seed, run_id)
        train_edge, val_edge = self.ratios.edges
        if value < train_edge:
            return SplitName.TRAIN
        if value < val_edge:
            return SplitName.VAL
        return SplitName.TEST

    def split(self, run_ids: Sequence[str]) -> RunSplit:
        """Alle Läufe einzeln zuordnen."""
        ids = self._check_input(run_ids)
        buckets: dict[SplitName, list[str]] = {name: [] for name in SplitName}
        for run_id in ids:
            buckets[self.assign(run_id)].append(run_id)
        return self._finalize(
            buckets[SplitName.TRAIN], buckets[SplitName.VAL], buckets[SplitName.TEST]
        )


@dataclass(frozen=True, slots=True)
class PredefinedRunSplitter(RunSplitter):
    """Explizit vorgegebene Zuordnung -- für Domänenschnitte ab AP 2.

    Anwendungsfall: ``train`` aus dem Grey-Box-Modell, ``test`` aus Dyssol oder aus
    Anlagendaten. Das ist kein Zufallsschnitt, sondern ein Domänenwechsel (domain
    shift); eine hashbasierte Zuordnung würde ihn gerade zerstören. Bewusst jetzt
    vorgesehen, damit später nicht am Trainingscode geschraubt wird.
    """

    label: str
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    min_per_split: int = 1
    require_single_design: bool = False

    @property
    def config_digest(self) -> str:
        """Fingerprint aus dem Label des Schnitts."""
        return _digest_of("PredefinedRunSplitter", self.label)

    def split(self, run_ids: Sequence[str]) -> RunSplit:
        """Vorgabe gegen den Pool prüfen und übernehmen."""
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


# ------------------------------------------------ Anbindung an das Anregungsdesign


def split_schedule(schedule: ExcitationSchedule, splitter: RunSplitter) -> RunSplit:
    """Split aus einem :class:`~twinlink_ml.simulation.excitation.ExcitationSchedule`.

    Dünne Adapterfunktion: ``schedule.run_id`` ist bereits in Design-Reihenfolge und
    trägt den Design-Fingerprint, sodass der Split-Fingerprint allein die Daten
    identifiziert.
    """
    return splitter.split(schedule.run_id)


def signal_kind_strata(schedule: ExcitationSchedule, channel: int) -> Mapping[str, str]:
    """Signalart je Lauf für einen Stellgrößenkanal, als Schichtungsmerkmal.

    Signalart wird pro Lauf **und pro Kanal** unabhängig gezogen. Es gibt daher kein
    einziges Stratum pro Lauf; die Abdeckung ist kanalweise zu prüfen. Relevant, weil
    bei einer Rampe die Schleppabweichung direkt proportional zur Zeitkonstante ist --
    ein Testblock ohne Rampen misst den Mehrschritt-Rollout-Fehler am uninformativsten
    Signaltyp.
    """
    kinds = schedule.signal_kinds
    if kinds and not 0 <= channel < len(kinds[0]):
        raise SplitConfigError(f"Kanalindex {channel} außerhalb von 0..{len(kinds[0]) - 1}")
    return {run_id: kinds[index][channel] for index, run_id in enumerate(schedule.run_id)}


# ---------------------------------------------------------------------- Diagnose


def realized_ratios(split: RunSplit) -> Mapping[SplitName, float]:
    """Tatsächlich realisierte Anteile; bei :class:`HashRunSplitter` nicht die Sollwerte."""
    total = split.n_runs
    return {name: len(ids) / total for name, ids in split.as_mapping.items()}


def stratum_coverage(
    split: RunSplit, strata: Mapping[str, str]
) -> Mapping[SplitName, Mapping[str, int]]:
    """Abdeckung eines Schichtungsmerkmals je Split.

    Exakte Stratifizierung bräuchte Pool-Wissen und gäbe damit die Stabilität von
    :class:`HashRunSplitter` auf. Empfehlung: hier diagnostizieren, bei Lücke den Seed
    dokumentiert wechseln, den Mechanismus nicht verkomplizieren.
    """
    missing = set(split.all_runs) - set(strata)
    if missing:
        raise SplitConfigError(f"Kein Stratum für Läufe: {sorted(missing)}")
    return {
        name: dict(Counter(strata[run_id] for run_id in ids))
        for name, ids in split.as_mapping.items()
    }


def split_report(split: RunSplit, strata: Mapping[str, str] | None = None) -> str:
    """Kurzbericht für das Run-Log. Nur Text, keine Nebenwirkungen."""
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

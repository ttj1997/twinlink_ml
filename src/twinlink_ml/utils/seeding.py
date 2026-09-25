"""Zentrales Seed-Handling -- eine Stelle für Python, NumPy, PyTorch, CUDA.

Warum das ab Schritt 1 dasteht und nicht später: sobald an mehreren Stellen im Code
``np.random.seed`` oder ``torch.manual_seed`` auftauchen, ist die Frage "war Lauf X
reproduzierbar" nicht mehr beantwortbar. Für Lernkurven mit 3-5 Seeds pro Punkt ist die
Antwort auf diese Frage die Grundlage der Aussage.

Wichtig für die Auswertung: der Seed betrifft im Prozessmodell **nur** das Sensormodell
und das Anregungs-Sampling. Das Grey-Box-Modell selbst ist deterministisch. Streuung in
den Lernkurven kommt damit aus drei trennbaren Quellen -- Datensatz-Realisierung
(``data_seed``), Sensorrauschen (``noise_seed``) und Initialisierung/Batch-Reihenfolge
des Netzes (``model_seed``). Wer alle drei mit einem Seed koppelt, kann später nicht
sagen, ob die Streuung aus den Daten oder aus dem Training kommt.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import random
from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = [
    "SeedBundle",
    "SeedPlan",
    "VarianceSource",
    "seed_everything",
]


def _derive(*parts: str | int) -> int:
    """32-Bit-Seed aus beliebigen Bestandteilen.

    SHA-256 statt einer linearen Transformation, damit benachbarte Master-Seeds keine
    korrelierten Teilseeds erzeugen.
    """
    payload = ":".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


class VarianceSource(str, Enum):
    """Zufallsquelle, deren Streuung in einer Studie variiert wird.

    ``JOINT`` variiert alle drei gemeinsam über einen Master-Seed. Das ist die
    konservative, also ehrlichste Angabe und die Voreinstellung der Lernkurven. Die
    Einzelquellen beantworten die Frage, die ``JOINT`` offen lässt: kommt eine Differenz
    zwischen zwei Architekturen aus dem Training oder aus der Datensatz-Realisierung?
    """

    DATA = "data"
    NOISE = "noise"
    MODEL = "model"
    JOINT = "joint"

    @property
    def is_parameter_free_relevant(self) -> bool:
        """Ob ein parameterfreies Modell auf dieser Achse überhaupt streuen kann.

        Persistenz hat keine Parameter: ihre Streuung über ``MODEL`` ist exakt Null, über
        ``DATA`` und ``NOISE`` dagegen reine Realisierung des Datensatzes. Genau deshalb
        ist sie das Maß für das Auflösungsvermögen -- siehe
        :mod:`twinlink_ml.evaluation.resolution`.
        """
        return self in {VarianceSource.DATA, VarianceSource.NOISE, VarianceSource.JOINT}


@dataclass(frozen=True, slots=True)
class SeedBundle:
    """Getrennte Seeds für die drei unabhängigen Zufallsquellen."""

    data_seed: int
    noise_seed: int
    model_seed: int

    @classmethod
    def from_master(cls, master_seed: int) -> SeedBundle:
        """Deterministische Ableitung dreier Seeds aus einem Master-Seed."""
        return cls(
            data_seed=_derive(master_seed, "data"),
            noise_seed=_derive(master_seed, "noise"),
            model_seed=_derive(master_seed, "model"),
        )

    def generator(self, role: str) -> np.random.Generator:
        """NumPy-Generator für eine Rolle. Bevorzugt gegenüber globalem Zustand."""
        seeds = {"data": self.data_seed, "noise": self.noise_seed, "model": self.model_seed}
        if role not in seeds:
            raise ValueError(f"Unbekannte Rolle {role!r}, erlaubt: {sorted(seeds)}")
        return np.random.default_rng(seeds[role])

    def replace(self, source: VarianceSource, value: int) -> SeedBundle:
        """Bundle mit einem ausgetauschten Teilseed.

        ``JOINT`` ist hier nicht zulässig -- ein gemeinsamer Master-Seed ist kein
        Teilseed, sondern erzeugt über :meth:`from_master` gleich alle drei.
        """
        field = {
            VarianceSource.DATA: "data_seed",
            VarianceSource.NOISE: "noise_seed",
            VarianceSource.MODEL: "model_seed",
        }.get(source)
        if field is None:
            raise ValueError("VarianceSource.JOINT ersetzt keinen einzelnen Teilseed")
        return dataclasses.replace(self, **{field: value})

    def as_parameters(self) -> dict[str, int]:
        """Für das Run-Log."""
        return {
            "seed/data": self.data_seed,
            "seed/noise": self.noise_seed,
            "seed/model": self.model_seed,
        }


@dataclass(frozen=True, slots=True)
class SeedPlan:
    """Welche Seeds eine Studie fährt, aufgeschlüsselt nach Zufallsquelle.

    Konstruktion **eine Quelle auf einmal** (one-factor-at-a-time): Replikat :math:`k`
    einer Achse benutzt den Teilseed von Master-Seed :math:`m_k` und für die beiden
    anderen Achsen die des Referenz-Master-Seeds :math:`m_0`.

    Zwei Eigenschaften, die dadurch entstehen und beide gebraucht werden:

    * Die Teilseeds der Achse ``DATA`` sind **dieselben** wie die von ``JOINT``. Damit
      lässt sich der simulierte Lauf-Pool zwischen den Achsen wiederverwenden -- das
      Grey-Box-Modell ist deterministisch, ein Pool hängt nur am ``data_seed``.
    * Replikat :math:`k = 0` ist auf allen Achsen identisch (das Referenz-Bundle). Es
      wird nur einmal gerechnet und in jede Achse eingehängt.

    Die Zerlegung ist damit **nicht** eine Varianzanalyse mit Interaktionen: gemessen
    werden Haupteffekte. Ob das zulässig ist, prüft
    :meth:`~twinlink_ml.evaluation.resolution.VarianceDecomposition.additivity_ratio`
    gegen die gemeinsame Streuung -- deshalb wird ``JOINT`` mitgefahren und nicht als
    Summe der Einzelachsen behauptet.
    """

    master_seeds: tuple[int, ...] = (1, 2, 3)

    def __post_init__(self) -> None:
        if len(self.master_seeds) < 2:
            raise ValueError("mindestens zwei Master-Seeds, sonst gibt es keine Streuung")
        if len(set(self.master_seeds)) != len(self.master_seeds):
            raise ValueError("Master-Seeds müssen paarweise verschieden sein")

    @property
    def n_replicates(self) -> int:
        """Anzahl Replikate je Achse."""
        return len(self.master_seeds)

    @property
    def reference(self) -> SeedBundle:
        """Referenz-Bundle, Replikat 0 aller Achsen."""
        return SeedBundle.from_master(self.master_seeds[0])

    def bundle(self, source: VarianceSource, replicate: int) -> SeedBundle:
        """Seed-Bundle für Achse und Replikatindex."""
        if not 0 <= replicate < self.n_replicates:
            raise ValueError(f"replicate {replicate} außerhalb 0..{self.n_replicates - 1}")
        variant = SeedBundle.from_master(self.master_seeds[replicate])
        if source is VarianceSource.JOINT:
            return variant
        field = {
            VarianceSource.DATA: variant.data_seed,
            VarianceSource.NOISE: variant.noise_seed,
            VarianceSource.MODEL: variant.model_seed,
        }[source]
        return self.reference.replace(source, field)

    def bundles(self, source: VarianceSource) -> tuple[SeedBundle, ...]:
        """Alle Bundles einer Achse, Replikat 0 zuerst."""
        return tuple(self.bundle(source, k) for k in range(self.n_replicates))

    def data_seeds(self, sources: tuple[VarianceSource, ...]) -> tuple[int, ...]:
        """Alle vorkommenden ``data_seed``-Werte -- die Anzahl teurer Simulationen."""
        seen = {bundle.data_seed for source in sources for bundle in self.bundles(source)}
        return tuple(sorted(seen))

    def as_parameters(self) -> dict[str, object]:
        """Für das Run-Log."""
        return {"seedplan/master_seeds": list(self.master_seeds)}


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> SeedBundle:
    """Setzt alle globalen Zufallsquellen und liefert das abgeleitete Seed-Bundle.

    :param deterministic_torch: aktiviert deterministische CUDA-Kernel. Kostet
        Rechenzeit und lässt manche Operationen fehlschlagen -- für die Lernkurven in
        AP 1 ist der Preis richtig investiert, für lange RL-Läufe später abzuwägen.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))  # noqa: NPY002 - globaler Zustand ist hier der Zweck
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False

    return SeedBundle.from_master(seed)

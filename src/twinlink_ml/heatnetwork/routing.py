r"""Verschaltung Quelle -> Senke: welche Quelle speist welche Senke, in welcher Reihenfolge.

Zwei Implementierungen liegen von Anfang an vor (:class:`FixedRouting`,
:class:`MatrixRouting`) -- dieselbe Schwelle, die ``sorption.SorptionIsotherm`` bereits über
zwei Implementierungen zieht (``docs/03_code_konventionen.md``): eine ABC ist hier keine
Vorratsabstraktion, sondern folgt der bestehenden Konvention.

Kein LP, kein Optimierer (EA6) -- ``MatrixRouting`` nimmt eine bereits entschiedene
Zuordnungsmatrix entgegen (später der RL-Aktionsraum), sie entscheidet nichts selbst.

Trennung von Topologie und Physik: eine Route entscheidet nur, WELCHE ``(Quelle, Senke)``-Paare
existieren, nicht in welcher Reihenfolge ÜBER QUELLEN HINWEG sie abgearbeitet werden -- das ist
die physikalische Regel aus EA2, Korrektur 3 (Niedertemperaturquellen zuerst) und lebt in
:func:`resolve_cascade_order`, von beiden Implementierungen unten aufgerufen statt je Klasse
neu umgesetzt. ``balance.evaluate_cascade`` kennt diese Regel nicht mehr -- sie bekommt die
schon korrekt geordnete Stufenliste übergeben (Dependency Injection).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import HeatNetworkConfig

__all__ = [
    "FixedRouting",
    "HeatNetworkRouting",
    "MatrixRouting",
    "resolve_cascade_order",
]

FloatArray = NDArray[np.float64]


def resolve_cascade_order(network: HeatNetworkConfig) -> tuple[str, ...]:
    """Bearbeitungsreihenfolge der Quellennamen, Niedertemperatur zuerst (EA2, Korrektur 3).

    ``network.cascade_order`` gewinnt, wenn gesetzt (bereits als Permutation der
    Quellennamen validiert, siehe ``HeatNetworkConfig.__post_init__``). Sonst: aufsteigend
    nach ``outlet_temperature_celsius`` sortiert, stabil -- bei gleicher Temperatur bleibt
    die in ``network.sources`` gegebene Reihenfolge erhalten, statt eine willkürliche
    weitere Sortierung (z. B. nach Namen) einzuführen.
    """
    if network.cascade_order:
        return network.cascade_order
    ordered = sorted(network.sources, key=lambda source: source.outlet_temperature_celsius)
    return tuple(source.name for source in ordered)


class HeatNetworkRouting(ABC):
    """Ordnet Quellen und Senken zu geordneten Kaskadenstufen.

    :meth:`route` liefert eine geordnete Liste von ``(Quellenname, Senkenname)``-Paaren --
    genau die Eingabe, die :func:`twinlink_ml.heatnetwork.balance.evaluate_cascade` als
    ``stages`` erwartet. Die Reihenfolge im Rückgabewert ist bereits die tatsächliche
    Abarbeitungsreihenfolge (Niedertemperaturquellen zuerst, EA2 Korrektur 3) -- eine
    Implementierung darf diese Regel nicht umgehen, deshalb rufen beide Implementierungen
    unten :func:`resolve_cascade_order` auf, statt sie zu duplizieren oder zu ignorieren.
    """

    @abstractmethod
    def route(self, network: HeatNetworkConfig) -> tuple[tuple[str, str], ...]:
        """Geordnete ``(Quellenname, Senkenname)``-Stufenliste für ``network``."""
        ...


@dataclass(frozen=True, slots=True)
class FixedRouting(HeatNetworkRouting):
    """Feste, konfigurierbare Verschaltung -- die Topologie steht in ``connections``.

    :param connections: ``(Quellenname, Senkenname)``-Paare, in beliebiger Reihenfolge
        angegeben (die tatsächliche Abarbeitungsreihenfolge legt trotzdem
        :func:`resolve_cascade_order` fest, siehe :meth:`route`). Eine Quelle darf in
        mehreren Paaren erscheinen (kaskadiert über mehrere Senken, Korrektur 2); eine
        Senke ebenso (kaskadiert über mehrere Quellen).
    """

    connections: tuple[tuple[str, str], ...]

    def route(self, network: HeatNetworkConfig) -> tuple[tuple[str, str], ...]:
        source_names = {source.name for source in network.sources}
        sink_names = {sink.name for sink in network.sinks}
        for source_name, sink_name in self.connections:
            if source_name not in source_names:
                raise ValueError(f"connections referenziert unbekannte Quelle {source_name!r}")
            if sink_name not in sink_names:
                raise ValueError(f"connections referenziert unbekannte Senke {sink_name!r}")
        order = resolve_cascade_order(network)
        rank = {name: index for index, name in enumerate(order)}
        return tuple(
            sorted(self.connections, key=lambda pair: rank[pair[0]])
        )


@dataclass(frozen=True, slots=True)
class MatrixRouting(HeatNetworkRouting):
    """Verschaltung aus einer Zuordnungsmatrix -- der spätere RL-Aktionsraum.

    :param assignment: Form ``(n_sources, n_sinks)``, Zeilen/Spalten in der Reihenfolge von
        ``network.sources``/``network.sinks``. Ein Eintrag über ``threshold`` aktiviert die
        Verbindung (Quelle i, Senke j); der Zahlenwert selbst dient nur als
        Prioritätskriterium zwischen mehreren Senken derselben Quelle (siehe
        :meth:`route`), nicht als Anteil/Split -- ein Split-Modell ist nicht Teil dieses
        Auftrags (EA6).
    :param threshold: Untergrenze, ab der ein Matrixeintrag als aktive Verbindung zählt.
    """

    assignment: FloatArray
    threshold: float = 0.0

    def route(self, network: HeatNetworkConfig) -> tuple[tuple[str, str], ...]:
        n_sources = len(network.sources)
        n_sinks = len(network.sinks)
        if self.assignment.shape != (n_sources, n_sinks):
            raise ValueError(
                f"assignment muss Form ({n_sources}, {n_sinks}) haben, ist "
                f"{self.assignment.shape}"
            )
        order = resolve_cascade_order(network)
        rank = {name: index for index, name in enumerate(order)}
        source_names = [source.name for source in network.sources]
        sink_names = [sink.name for sink in network.sinks]

        connections: list[tuple[str, str]] = []
        weights: list[float] = []
        for source_index, source_name in enumerate(source_names):
            for sink_index, sink_name in enumerate(sink_names):
                weight = float(self.assignment[source_index, sink_index])
                if weight > self.threshold:
                    connections.append((source_name, sink_name))
                    weights.append(weight)

        order_index = sorted(
            range(len(connections)),
            key=lambda i: (rank[connections[i][0]], -weights[i]),
        )
        return tuple(connections[i] for i in order_index)

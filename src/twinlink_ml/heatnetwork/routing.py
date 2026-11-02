r"""Interconnection source -> sink: which source feeds which sink, in which order.

Two implementations exist from the start (:class:`FixedRouting`,
:class:`MatrixRouting`) -- the same threshold that ``sorption.SorptionIsotherm`` already draws
over two implementations (``docs/03_code_konventionen.md``): an ABC here is not a
stockpile abstraction but follows the existing convention.

No LP, no optimizer (EA6) -- ``MatrixRouting`` takes an already decided
assignment matrix (later the RL action space); it decides nothing itself.

Separation of topology and physics: a route only decides WHICH ``(Quelle, Senke)`` pairs
exist, not in which order ACROSS SOURCES they are processed -- that is
the physical rule from EA2, correction 3 (low-temperature sources first) and lives in
:func:`resolve_cascade_order`, called by both implementations below instead of being
re-implemented per class. ``balance.evaluate_cascade`` no longer knows this rule -- it receives
the already correctly ordered stage list (dependency injection).
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
    """Processing order of the source names, low temperature first (EA2, correction 3).

    ``network.cascade_order`` wins if set (already validated as a permutation of the
    source names, see ``HeatNetworkConfig.__post_init__``). Otherwise: sorted ascending
    by ``outlet_temperature_celsius``, stable -- at equal temperature the order
    given in ``network.sources`` is kept, instead of introducing an arbitrary
    further sorting (e.g. by name).
    """
    if network.cascade_order:
        return network.cascade_order
    ordered = sorted(network.sources, key=lambda source: source.outlet_temperature_celsius)
    return tuple(source.name for source in ordered)


class HeatNetworkRouting(ABC):
    """Arranges sources and sinks into ordered cascade stages.

    :meth:`route` returns an ordered list of ``(Quellenname, Senkenname)`` pairs --
    exactly the input that :func:`twinlink_ml.heatnetwork.balance.evaluate_cascade` expects as
    ``stages``. The order in the return value is already the actual
    processing order (low-temperature sources first, EA2 correction 3) -- an
    implementation must not bypass this rule, which is why both implementations
    below call :func:`resolve_cascade_order` instead of duplicating or ignoring it.
    """

    @abstractmethod
    def route(self, network: HeatNetworkConfig) -> tuple[tuple[str, str], ...]:
        """Ordered ``(Quellenname, Senkenname)`` stage list for ``network``."""
        ...


@dataclass(frozen=True, slots=True)
class FixedRouting(HeatNetworkRouting):
    """Fixed, configurable interconnection -- the topology is in ``connections``.

    :param connections: ``(Quellenname, Senkenname)`` pairs, given in any order
        (the actual processing order is nevertheless set by
        :func:`resolve_cascade_order`, see :meth:`route`). A source may appear in
        several pairs (cascaded over several sinks, correction 2); a
        sink likewise (cascaded over several sources).
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
        return tuple(sorted(self.connections, key=lambda pair: rank[pair[0]]))


@dataclass(frozen=True, slots=True)
class MatrixRouting(HeatNetworkRouting):
    """Interconnection from an assignment matrix -- the later RL action space.

    :param assignment: shape ``(n_sources, n_sinks)``, rows/columns in the order of
        ``network.sources``/``network.sinks``. An entry above ``threshold`` activates the
        connection (source i, sink j); the numerical value itself serves only as a
        priority criterion between several sinks of the same source (see
        :meth:`route`), not as a fraction/split -- a split model is not part of this
        task (EA6).
    :param threshold: lower bound from which a matrix entry counts as an active connection.
    """

    assignment: FloatArray
    threshold: float = 0.0

    def route(self, network: HeatNetworkConfig) -> tuple[tuple[str, str], ...]:
        n_sources = len(network.sources)
        n_sinks = len(network.sinks)
        if self.assignment.shape != (n_sources, n_sinks):
            raise ValueError(
                f"assignment muss Form ({n_sources}, {n_sinks}) haben, ist {self.assignment.shape}"
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

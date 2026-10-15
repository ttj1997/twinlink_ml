"""The pre-heating train of one sink: its stages in the order of the scenario drawing.

``Preheater`` takes the process gas at ambient conditions and sends it through one
:class:`~twinlink_ml.flowsheet.loop.HeatTransferLoop` per source in the order given (the sheet's
"low" stage first, then "high"); the gas leaving one stage enters the next. A stage that cannot
recover (no driving force) leaves the gas as it is at 0 kW; a stage of the partner model with the
gas warmer than the exhaust would cool it (negative duty, kept as the sheet has it).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from twinlink_ml.partner_v4 import corrected as c

from .base import Assumption, Unit
from .loop import HeatTransferLoop, LoopInputs, StageOut

__all__ = ["Preheater", "PreheaterInputs", "PreheaterOut", "SourceFeed"]


@dataclass(frozen=True)
class SourceFeed:
    """One exhaust that feeds a stage, and the flow of the medium of its circuit."""

    source: int
    exhaust: c.Stream
    htm_kg_per_s: float


@dataclass(frozen=True)
class PreheaterInputs:
    """The gas of one sink, the sources that feed it in order, and the medium."""

    sink: int
    gas: c.Stream
    feeds: Sequence[SourceFeed]
    htm_cp_j_per_kg_k: float
    target_celsius: float


@dataclass(frozen=True)
class PreheaterOut:
    """Stages of the train and the gas that leaves it."""

    stages: tuple[StageOut, ...]
    gas: c.Stream
    """The process gas entering the burner."""


class Preheater(Unit[PreheaterInputs, PreheaterOut]):
    """Pre-heating train of one sink (zero to two stages in the scenarios of the sheet)."""

    def __init__(self, loop: HeatTransferLoop, *, name: str = "preheater") -> None:
        self.loop = loop
        self.name = name

    def assumptions(self) -> dict[str, Assumption]:
        return self.loop.assumptions()

    def solve(self, inputs: PreheaterInputs) -> PreheaterOut:
        gas = inputs.gas
        stages: list[StageOut] = []
        for feed in inputs.feeds:
            stage = self.loop.solve(
                LoopInputs(
                    source=feed.source,
                    sink=inputs.sink,
                    exhaust=feed.exhaust,
                    gas=gas,
                    htm_kg_per_s=feed.htm_kg_per_s,
                    htm_cp_j_per_kg_k=inputs.htm_cp_j_per_kg_k,
                    target_celsius=inputs.target_celsius,
                )
            )
            stages.append(stage)
            gas = c.Stream(gas.dry_air_kg_per_s, gas.humidity_kg_per_kg, stage.gas_out_celsius)
        return PreheaterOut(tuple(stages), gas)

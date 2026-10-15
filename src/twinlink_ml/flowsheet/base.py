"""The unit interface and the switches of the flowsheet.

Every unit is a frozen configuration plus a ``solve`` that is a pure function of its inputs
(no hidden state, no I/O). A unit knows nothing of the other units: the flowsheet wires them.
Each assumption of the partner model that the independent calculation replaces is one switch with
two values, ``"partner"`` (the partner workbook ``Model_v4.1.xlsx`` as built or as written in the
equation document) and ``"physical"`` (the independent calculation of Befund 90 and its
neighbours). The mapping of the values:

==================  ===============================  =======================================
switch              ``"partner"``                    ``"physical"``
==================  ===============================  =======================================
``exchanger``       equal outlet temperatures        counter-current, minimum approach
                    (co-current, unlimited area)     ``dT_min`` checked along the exchanger
``cp_gas``          1.2 / 2.0 kJ/(kg K)              1.006 / 1.86 kJ/(kg K) (gas and vapour)
``condensation``    off                              on
``heat_loss``       off                              ``UA (T - T_amb)`` of the bed (placeholder)
``product_moisture``  equilibrium moisture 2 g/kg    from the fluidised-bed model (dryer 2)
``gas_formula``     as the sheet (Scenarios row 26)  corrected (``/100``, kg/kg)
``boiling_check``   water in the circuit unlimited   flag above the boiling point at 1 atm
==================  ===============================  =======================================

The switches are values of one frozen dataclass so that a combination is hashable, comparable and
goes into an artifact as it is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Generic, Literal, TypeVar

__all__ = [
    "SWITCH_NAMES",
    "Assumption",
    "Convention",
    "PhysicalParameters",
    "Switches",
    "Unit",
]

Assumption = Literal["partner", "physical"]
Convention = Literal["per_exchanger", "through_circuit"]

SWITCH_NAMES: tuple[str, ...] = (
    "exchanger",
    "cp_gas",
    "condensation",
    "heat_loss",
    "product_moisture",
    "gas_formula",
    "boiling_check",
)
"""The seven switches in the order of the task (the forward order of the decomposition)."""

_I = TypeVar("_I")
_O = TypeVar("_O")


class Unit(ABC, Generic[_I, _O]):
    """A process unit: a configuration, and ``solve`` as a pure function of the inputs."""

    name: str

    @abstractmethod
    def solve(self, inputs: _I) -> _O:
        """Compute the outlet state and the diagnostics of the unit for ``inputs``."""

    def assumptions(self) -> dict[str, Assumption]:
        """The switches this unit reads (name -> value); empty for a unit without assumptions."""
        return {}


@dataclass(frozen=True)
class Switches:
    """One value per assumption; the default is the partner model as built."""

    exchanger: Assumption = "partner"
    cp_gas: Assumption = "partner"
    condensation: Assumption = "partner"
    heat_loss: Assumption = "partner"
    product_moisture: Assumption = "partner"
    gas_formula: Assumption = "partner"
    boiling_check: Assumption = "partner"

    def __post_init__(self) -> None:
        for name in SWITCH_NAMES:
            if getattr(self, name) not in ("partner", "physical"):
                raise ValueError(f"switch {name} must be 'partner' or 'physical'")

    @classmethod
    def all_partner(cls) -> Switches:
        return cls()

    @classmethod
    def all_physical(cls) -> Switches:
        values: dict[str, Assumption] = dict.fromkeys(SWITCH_NAMES, "physical")
        return cls(**values)

    def with_(self, name: str, value: Assumption) -> Switches:
        """The same switches with one changed."""
        if name not in SWITCH_NAMES:
            raise KeyError(name)
        return replace(self, **{name: value})

    def as_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in SWITCH_NAMES}


@dataclass(frozen=True)
class PhysicalParameters:
    """Numbers that only the ``"physical"`` side of a switch uses (all placeholders, no source).

    ``dt_min_k`` is the minimum approach of one exchanger in the main convention
    (``per_exchanger``); ``through_circuit`` takes it as the approach through the whole circuit
    and gives each of the two exchangers half of it (the convention of Befund 90 and of the
    heat-network module, whose 20 K stands for two exchangers in series). 10 K is the middle one
    of the 20 / 10 / 5 K of Befund 90.
    """

    dt_min_k: float = 10.0
    convention: Convention = "per_exchanger"
    heat_loss_ua_w_per_k: float = 2.0
    """``UA`` of the bed to the surroundings: the laboratory placeholder (docs/11, 9.2); not a
    plant value (docs/11, 9.6)."""
    boiling_celsius: float = 100.0
    """Boiling point of water at 1 atm (flag threshold of the ``boiling_check``)."""

    def __post_init__(self) -> None:
        if self.dt_min_k < 0.0:
            raise ValueError("dt_min_k must not be negative")
        if self.convention not in ("per_exchanger", "through_circuit"):
            raise ValueError("convention must be 'per_exchanger' or 'through_circuit'")

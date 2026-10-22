"""Flowsheet of the partner model (``Model_v4.1.xlsx``) from exchangeable units.

The units of the partner model existed twice: as a faithful rebuild
(:mod:`twinlink_ml.partner_v4_1`) and as physical alternatives (fluidised-bed steady state,
counter-current heat network, carrier energy). This package puts both behind one interface: a
:class:`~twinlink_ml.flowsheet.base.Unit` per process unit, a switch per assumption
(``"partner"`` | ``"physical"``, :class:`~twinlink_ml.flowsheet.base.Switches`) and a
:class:`~twinlink_ml.flowsheet.flowsheet.Flowsheet` that wires the six scenarios.

* :mod:`.base` -- ``Unit`` ABC, ``Switches``, ``PhysicalParameters``,
* :mod:`.thermo` -- gas properties and release curves (lifted from the Befund-90 script),
* :mod:`.exchanger` -- ``HeatExchanger`` and the counter-current solver,
* :mod:`.loop` -- ``HeatTransferLoop`` (recovery exchanger, medium, pre-heating exchanger),
* :mod:`.preheater` -- ``Preheater`` (the stages of one sink),
* :mod:`.dryer` -- ``DryerBlock``, :mod:`.burner` -- ``Burner``,
* :mod:`.auxiliary` -- fan and pump power (output), CO2 with the carrier factors,
* :mod:`.bed_state` -- dryer 2 in the laboratory model (physical product moisture),
* :mod:`.flowsheet` -- ``Flowsheet``; :mod:`.sheet` -- scenarios in the cell layout of the workbook,
* :mod:`.decomposition` -- switch-by-switch walk from ``"partner"`` to ``"physical"``,
* :mod:`.line` -- one dryer line (the stationary Dyssol flowsheet of the delivery 4.1_2),
* :mod:`.validators` -- plausibility checks (exhaust saturation, product mass, medium boiling).
"""

from .base import SWITCH_NAMES, PhysicalParameters, Switches, Unit
from .flowsheet import Flowsheet, FlowsheetConfig, ScenarioResult

__all__ = [
    "SWITCH_NAMES",
    "Flowsheet",
    "FlowsheetConfig",
    "PhysicalParameters",
    "ScenarioResult",
    "Switches",
    "Unit",
]

"""Evaluation of the six cases (and case 0) at one operating point.

Two variants (decision of the project lead, F6):

* ``fixed`` -- the workbook's circuit flows (40 / 30 kg/h) in every case (answers "does the
  statement of the workbook hold"),
* ``optimised`` -- the two circuit flows chosen per case and point to minimise that case's fuel,
  within a stated range, by a multistart search (:class:`FlowOptimiser`), so that every topology is
  compared at its own best operating point.

Three modes (docs/11 section 13.6; the same switch sets as ``scripts/report_partner_6cases.py``):
partner (switch set of 4.1_2, corrected gas formula, no saturation limit), physical unpressurised
(counter-current, 10 K per exchanger, condensing release curve, saturation limit, medium at most
95 degC) and physical pressurised (the same without the medium limit).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import minimize

from twinlink_ml.flowsheet import Flowsheet, FlowsheetConfig, ScenarioResult, Switches
from twinlink_ml.flowsheet.validators import MOLAR_MASS_RATIO, water_saturation_pressure_pa

from .space import ATM_PA, OperatingPoint
from .workbook import WORKBOOK, WorkbookConstants, excel_inputs

__all__ = [
    "CASES",
    "MODES",
    "CaseEvaluator",
    "CaseOutcome",
    "FlowOptimiser",
    "FlowsheetCaseEvaluator",
    "Mode",
    "PointResult",
    "exhaust_relative_humidity",
]

CASES: tuple[int, ...] = (1, 2, 3, 4, 5, 6)


@dataclass(frozen=True)
class Mode:
    """One mode of docs/11 section 13.6: switch set, saturation limit, medium limit."""

    name: str
    switches: Switches
    saturation_limit: bool
    htm_max_celsius: float | None


_PHYSICAL = Switches(exchanger="physical", condensation="physical", gas_formula="physical")
MODES: dict[str, Mode] = {
    m.name: m
    for m in (
        Mode("partner", Switches(gas_formula="physical"), False, None),
        Mode("physical_unpressurised_95", _PHYSICAL, True, 95.0),
        Mode("physical_pressurised", _PHYSICAL, True, None),
    )
}
"""The three modes of docs/11 section 13.6 (as ``scripts/report_partner_6cases.MODES``)."""


def exhaust_relative_humidity(
    temperature_celsius: float, humidity_kg_per_kg: float, pressure_pa: float = ATM_PA
) -> float:
    """Relative humidity of an exhaust, IAPWS-IF97 saturation (as the admissibility of docs/37).

    ``inf`` below 0 degC (outside the validity range; such an exhaust is never admissible).
    """
    if temperature_celsius < 0.0:
        return math.inf
    p_v = humidity_kg_per_kg * pressure_pa / (MOLAR_MASS_RATIO + humidity_kg_per_kg)
    return p_v / water_saturation_pressure_pa(temperature_celsius)


@dataclass(frozen=True)
class CaseOutcome:
    """One case at one point (case 0: no recovery)."""

    case: int
    fuel_kg_per_h: float
    recovered_kw: float
    htm_flow_kg_per_h: tuple[float, float] | None
    medium_hot_max_celsius: float | None
    stages_no_driving_force: int
    stages_negative_recovery: int
    stages_medium_limit: int
    optimiser_evaluations: int = 0


@dataclass(frozen=True)
class PointResult:
    """All cases at one point, one mode, one variant."""

    mode: str
    variant: str
    point: tuple[float, ...]
    exhaust_celsius: tuple[float, float]
    exhaust_humidity_kg_per_kg: tuple[float, float]
    exhaust_relative_humidity: tuple[float, float]
    cases: tuple[CaseOutcome, ...] = field(default_factory=tuple)

    def fuel(self) -> np.ndarray:
        """Fuel of cases 0..6 in kg/h (index = case)."""
        return np.array([c.fuel_kg_per_h for c in self.cases])

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _outcome(r: ScenarioResult, flows: tuple[float, float] | None, nfev: int = 0) -> CaseOutcome:
    stages = r.stages
    hot = [s.htm_hot_celsius for s in stages if s.htm_hot_celsius is not None]
    return CaseOutcome(
        case=r.case,
        fuel_kg_per_h=r.fuel_corrected_kg_per_h,
        recovered_kw=r.recovered_kw,
        htm_flow_kg_per_h=flows if r.case else None,
        medium_hot_max_celsius=max(hot) if hot else None,
        stages_no_driving_force=sum(s.binding == "no_driving_force" for s in stages),
        stages_negative_recovery=sum(s.q_w < 0.0 for s in stages),
        stages_medium_limit=sum(s.binding == "medium_limit" for s in stages),
        optimiser_evaluations=nfev,
    )


class CaseEvaluator(ABC):
    """Fuel of every case at one point."""

    @abstractmethod
    def evaluate(self, point: OperatingPoint) -> PointResult: ...


@dataclass(frozen=True)
class FlowOptimiser:
    """Multistart search of the two circuit flows of one case (kg/h, box-bounded).

    A ``grid`` x ``grid`` lattice over ``bounds`` (both circuits) plus the workbook's flows; bounded
    Nelder-Mead from the ``starts`` best of them; the best value found is kept. Including the
    workbook's flows among the candidates makes the optimised fuel never worse than the fixed one.
    """

    bounds_kg_per_h: tuple[float, float] = (20.0, 150.0)
    grid: int = 5
    starts: int = 3
    max_evaluations_per_start: int = 60
    xatol_kg_per_h: float = 0.05
    fatol_kg_per_h: float = 1e-7

    def search(
        self, fuel: Callable[[tuple[float, float]], float], workbook_flows: tuple[float, float]
    ) -> tuple[tuple[float, float], float, int]:
        lo, hi = self.bounds_kg_per_h
        axis = np.linspace(lo, hi, self.grid)
        candidates = [(float(a), float(b)) for a in axis for b in axis]
        candidates.append((float(workbook_flows[0]), float(workbook_flows[1])))
        values = [fuel(c) for c in candidates]
        nfev = len(candidates)
        best_x, best_f = candidates[int(np.argmin(values))], float(min(values))
        order = np.argsort(values, kind="stable")[: self.starts]
        for k in order:
            res = minimize(
                lambda x: fuel((float(x[0]), float(x[1]))),
                np.array(candidates[int(k)]),
                method="Nelder-Mead",
                bounds=[(lo, hi), (lo, hi)],
                options={
                    "maxfev": self.max_evaluations_per_start,
                    "xatol": self.xatol_kg_per_h,
                    "fatol": self.fatol_kg_per_h,
                },
            )
            nfev += int(res.nfev)
            if float(res.fun) < best_f:
                best_x, best_f = (float(res.x[0]), float(res.x[1])), float(res.fun)
        return best_x, best_f, nfev


class FlowsheetCaseEvaluator(CaseEvaluator):
    """The rebuild of :mod:`twinlink_ml.flowsheet` in one mode; ``optimiser``: the variant."""

    def __init__(
        self,
        mode: Mode,
        optimiser: FlowOptimiser | None = None,
        constants: WorkbookConstants = WORKBOOK,
        cases: Sequence[int] = CASES,
    ) -> None:
        self.mode = mode
        self.optimiser = optimiser
        self.constants = constants
        self.cases = tuple(cases)

    @property
    def variant(self) -> str:
        return "fixed" if self.optimiser is None else "optimised"

    def flowsheet(
        self,
        point: OperatingPoint,
        flows: tuple[float, float] | None = None,
        pinned: Mapping[int, tuple[float, float]] | None = None,
    ) -> Flowsheet:
        return Flowsheet(
            FlowsheetConfig(
                excel_inputs(point, flows, self.constants).to_si(),
                self.mode.switches,
                pinned_exhausts=pinned,
                saturation_limit=self.mode.saturation_limit,
                htm_max_celsius=self.mode.htm_max_celsius,
            )
        )

    def evaluate(self, point: OperatingPoint) -> PointResult:
        base_flows = self.constants.htm_flow_kg_per_h
        fs = self.flowsheet(point)
        exhausts = fs.exhausts()
        outcomes = [_outcome(fs.evaluate(0), None)]
        if self.optimiser is None:
            outcomes += [_outcome(fs.evaluate(n), base_flows) for n in self.cases]
        else:
            # The exhausts do not depend on the circuit flows: pin them (the same values) so that
            # the search does not re-solve the dryers at every evaluation.
            pinned = {
                i: (exhausts[i].temperature_celsius, exhausts[i].humidity_kg_per_kg) for i in (1, 2)
            }
            for n in self.cases:

                def fuel(flows: tuple[float, float], n: int = n) -> float:
                    return self.flowsheet(point, flows, pinned).evaluate(n).fuel_corrected_kg_per_h

                x, _, nfev = self.optimiser.search(fuel, base_flows)
                r = self.flowsheet(point, x, pinned).evaluate(n)
                outcomes.append(_outcome(r, x, nfev))
        e1, e2 = exhausts[1], exhausts[2]
        return PointResult(
            mode=self.mode.name,
            variant=self.variant,
            point=tuple(float(v) for v in point.as_array()),
            exhaust_celsius=(e1.temperature_celsius, e2.temperature_celsius),
            exhaust_humidity_kg_per_kg=(e1.humidity_kg_per_kg, e2.humidity_kg_per_kg),
            exhaust_relative_humidity=(
                exhaust_relative_humidity(e1.temperature_celsius, e1.humidity_kg_per_kg),
                exhaust_relative_humidity(e2.temperature_celsius, e2.humidity_kg_per_kg),
            ),
            cases=tuple(outcomes),
        )

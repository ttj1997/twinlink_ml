"""Switch-by-switch walk from ``"partner"`` to ``"physical"`` (U3), ranking of the scenarios.

The decomposition **depends on the order** of the switches (what one assumption does depends on
which of the others are already replaced; Befund 90, WU2), so a walk is always for one named
order and the study runs the order of the task (``SWITCH_NAMES``) and its reverse. Step 0 is the
partner model as built, step ``k`` has the first ``k`` switches of the order on ``"physical"``.

Ranking. The scenarios are ranked by the CO2 of the corrected fuel plus the placeholder
electricity (:attr:`ScenarioResult.co2_kg_per_h`), lowest first; scenarios whose values agree to
a relative ``1e-9`` are tied (the mirror pairs 1/2, 3/4, 5/6 always are). The corrected fuel is
used in every step, also before the switch ``gas_formula`` is on ``"physical"``: the as-built
formula is only reported (``fuel_kg_per_h``), never ranked or turned into CO2, because its factor
108.9 is a unit error and not a property of the plant. Rankings by fuel only and by recovered heat
are stored next to it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .base import SWITCH_NAMES, Switches
from .flowsheet import SCENARIOS, Flowsheet, FlowsheetConfig, ScenarioResult

__all__ = [
    "Step",
    "rank_groups",
    "ranking_text",
    "walk",
]

TIE_RELATIVE = 1e-9


def rank_groups(
    values: dict[int, float], *, descending: bool = False
) -> tuple[tuple[int, ...], ...]:
    """Scenarios grouped by equal value (relative ``TIE_RELATIVE``), best first."""
    ordered = sorted(values, key=lambda k: values[k], reverse=descending)
    groups: list[list[int]] = []
    for key in ordered:
        if groups and abs(values[key] - values[groups[-1][0]]) <= TIE_RELATIVE * max(
            abs(values[key]), abs(values[groups[-1][0]]), 1e-300
        ):
            groups[-1].append(key)
        else:
            groups.append([key])
    return tuple(tuple(sorted(g)) for g in groups)


def ranking_text(groups: Sequence[Sequence[int]]) -> str:
    """``"5=6 < 3=4 < 1=2"``: best (lowest) first, ``=`` for ties."""
    return " < ".join("=".join(str(k) for k in g) for g in groups)


@dataclass(frozen=True)
class Step:
    """One state of the walk."""

    index: int
    changed: str | None
    """The switch turned to ``"physical"`` in this step (``None`` for step 0)."""
    switches: Switches
    results: dict[int, ScenarioResult]
    """Cases 0-6."""

    def metric(self, name: str) -> dict[int, float]:
        return {c: getattr(self.results[c], name) for c in SCENARIOS}

    @property
    def ranking_co2(self) -> tuple[tuple[int, ...], ...]:
        return rank_groups(self.metric("co2_kg_per_h"))

    @property
    def ranking_fuel(self) -> tuple[tuple[int, ...], ...]:
        return rank_groups(self.metric("fuel_corrected_kg_per_h"))

    @property
    def ranking_recovery(self) -> tuple[tuple[int, ...], ...]:
        return rank_groups(self.metric("recovered_kw"), descending=True)

    def as_dict(self) -> dict[str, Any]:
        """Per-case metrics and the three rankings (for an artifact)."""
        cases: dict[str, Any] = {}
        for case, r in self.results.items():
            cases[str(case)] = {
                "recovered_kw": r.recovered_kw,
                "burner_duty_kw": r.burner_duty_kw,
                "fuel_kg_per_h": r.fuel_kg_per_h,
                "fuel_corrected_kg_per_h": r.fuel_corrected_kg_per_h,
                "electric_kw": r.carrier_energy.electric_kw,
                "co2_kg_per_h": r.co2_kg_per_h,
                "co2_fuel_only_kg_per_h": r.co2_fuel_only_kg_per_h,
                "htm_hot_max_celsius": r.htm_hot_max_celsius,
                "htm_boiling_stages": r.htm_boiling_stages,
                "stage_recovered_kw": [s.q_w / 1e3 for s in r.stages],
                "exhaust_celsius": {str(k): v.temperature_celsius for k, v in r.exhausts.items()},
            }
        return {
            "index": self.index,
            "changed": self.changed,
            "switches": self.switches.as_dict(),
            "cases": cases,
            "ranking_co2": ranking_text(self.ranking_co2),
            "ranking_fuel_corrected": ranking_text(self.ranking_fuel),
            "ranking_recovery": ranking_text(self.ranking_recovery),
        }


def walk(
    base: FlowsheetConfig,
    order: Sequence[str] = SWITCH_NAMES,
    *,
    factory: Callable[[FlowsheetConfig], Flowsheet] = Flowsheet,
) -> list[Step]:
    """From all ``"partner"`` to all ``"physical"``, one switch per step, in ``order``."""
    if sorted(order) != sorted(SWITCH_NAMES):
        raise ValueError(f"order must be a permutation of {SWITCH_NAMES}")
    switches = Switches.all_partner()
    steps: list[Step] = []
    for index in range(len(order) + 1):
        changed = order[index - 1] if index else None
        if changed is not None:
            switches = switches.with_(changed, "physical")
        flowsheet = factory(replace(base, switches=switches))
        steps.append(Step(index, changed, switches, flowsheet.evaluate_all()))
    return steps

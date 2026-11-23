r"""Key numbers of the findings that rest on the steady-state optimiser, and the reading O0 / O1.

The formulas are those of the analyses of the findings (``scripts/analyze_dynamic_headroom.py``,
``scripts/analyze_load_following.py``, the analysis of the learned-setpoints study), repeated
here on records so that the same function is applied to the records of the artifacts and to the
recomputed ones:

* Befund 105, ``g10``: per scenario :math:`(E_{R0} - E_{R1}) / E_{R0}`, median over the scenarios
  of the family "observed", each controller at its own margin.
* Befund 126, gain of a controller X against C0: :math:`(J_{C0} - J_X) / J_{C0}` with
  :math:`J` = sum of the cost of the days divided by the sum of the dry product (variant E).
* Learned set points, twin: :math:`J_{C2}` at the margin taken from the load-following artifact,
  and the gap of an arm against it, :math:`(J_L - J_{twin}) / J_{twin}`, median over the seeds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "cost_per_kg",
    "g10_median",
    "gain",
    "median_gap_to_twin",
    "reading",
    "relative_change",
]


def cost_per_kg(records: Sequence[Mapping[str, Any]], field: str = "cost_e") -> float:
    """Sum of the cost over the days divided by the sum of the dry product (EUR/kg)."""
    good = [r for r in records if r.get("max_violation") is not None]
    return float(sum(r[field] for r in good) / sum(r["product_dry_kg"] for r in good))


def gain(reference: float, value: float) -> float:
    """``(reference - value) / reference``."""
    return (reference - value) / reference


def g10_median(r0: Sequence[Mapping[str, Any]], r1: Sequence[Mapping[str, Any]]) -> float:
    """Median over the scenarios of ``(E_R0 - E_R1) / E_R0``; records matched by scenario name."""
    e1 = {r["scenario"]: float(r["energy_kwh"]) for r in r1}
    if set(e1) != {r["scenario"] for r in r0}:
        raise ValueError("R0 and R1 records cover different scenarios")
    gaps = [(float(r["energy_kwh"]) - e1[r["scenario"]]) / float(r["energy_kwh"]) for r in r0]
    return float(np.median(gaps))


def median_gap_to_twin(cost_per_seed: Mapping[str, float], twin: float) -> float:
    """Median over the seeds of ``(J_seed - J_twin) / J_twin``."""
    return float(np.median([(float(j) - twin) / twin for j in cost_per_seed.values()]))


def relative_change(old: float, new: float) -> float:
    """``|new - old| / |old|``."""
    return abs(new - old) / abs(old)


def reading(changes: Mapping[str, float], threshold: float) -> str:
    """``O1`` if at least one relative change exceeds ``threshold``, else ``O0``."""
    if not changes:
        raise ValueError("no key number")
    return "O1" if any(c > threshold for c in changes.values()) else "O0"

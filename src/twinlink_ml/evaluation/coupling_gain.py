r"""Coupling gain of two lines that compete for a limited resource (stage M3', verdict rule).

The coupling gain per scenario is ``G = CO2(S2) - CO2(M3')`` in kg CO2e/h of the site: the CO2 of
"each line on its own, then the best allocation under the limit" minus that of the joint optimum of
the four set points. THE RULE BELOW WAS FIXED BEFORE ANY RUN OF THE STAGE (project lead, 2026-10-06;
thresholds proposed in ``docs/09_entwurf_modellbasierte_optimierung.md``):

* the interaction check (stage 0) decides whether the joint runs are made at all: the largest
  absolute mixed second difference of the site CO2 over the 3 x 3 grid of the inlet temperatures
  must reach :data:`INTERACTION_RESOLUTION_KG_PER_H` for some limit and bound; otherwise the
  finding is "no coupling for this kind of scarcity";
* a coupling gain counts as DEMONSTRATED iff all three hold:
  (i)   the median of ``G`` over the scenarios is at least :data:`GAIN_MEDIAN_MIN_KG_PER_H`;
  (ii)  the median of ``G`` is at least :data:`GAIN_SEED_FACTOR` times the largest spread of the
        joint optimum over three DE seeds (measured on three scenarios, never assumed);
  (iii) ``G`` exceeds :data:`GAIN_PER_SCENARIO_MIN_KG_PER_H` at no fewer than
        :data:`GAIN_SCENARIO_SHARE_MIN` of the scenarios.
  Without a measured seed spread (ii) cannot be met and the gain is not demonstrated.
* ``G`` is reported additionally relative to the CO2 attributable to the two lines.

The numbers rest on the optimiser resolution of M2 (Befund 70): the joint and the line optimum
differed by at most 0.064 (1.26 m/s) and 0.179 kg CO2e/h (0.80 m/s) in the site CO2.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "GAIN_MEDIAN_MIN_KG_PER_H",
    "GAIN_PER_SCENARIO_MIN_KG_PER_H",
    "GAIN_SCENARIO_SHARE_MIN",
    "GAIN_SEED_FACTOR",
    "INTERACTION_RESOLUTION_KG_PER_H",
    "coupling_gain_verdict",
    "interaction_verdict",
]

INTERACTION_RESOLUTION_KG_PER_H = 0.5
GAIN_MEDIAN_MIN_KG_PER_H = 1.0
GAIN_SEED_FACTOR = 3.0
GAIN_PER_SCENARIO_MIN_KG_PER_H = 0.2
GAIN_SCENARIO_SHARE_MIN = 0.6


def interaction_verdict(max_abs_mixed_difference_kg_per_h: list[float]) -> dict[str, Any]:
    """Stage 0: do the joint runs follow? ``True`` iff some value reaches the resolution."""
    values = [v for v in max_abs_mixed_difference_kg_per_h if not math.isnan(v)]
    largest = max(values) if values else float("nan")
    return {
        "resolution_kg_per_h": INTERACTION_RESOLUTION_KG_PER_H,
        "largest_mixed_difference_kg_per_h": largest,
        "n_values": len(values),
        "n_at_or_above_resolution": sum(v >= INTERACTION_RESOLUTION_KG_PER_H for v in values),
        "joint_runs_justified": bool(values and largest >= INTERACTION_RESOLUTION_KG_PER_H),
    }


def coupling_gain_verdict(
    gain_kg_per_h: list[float],
    seed_spread_kg_per_h: float | None,
    line_co2_s2_kg_per_h: list[float],
) -> dict[str, Any]:
    """Verdict of the fixed rule for one limit and bound (see the module docstring)."""
    gains = np.asarray(gain_kg_per_h, dtype=np.float64)
    line_co2 = np.asarray(line_co2_s2_kg_per_h, dtype=np.float64)
    if gains.size == 0 or gains.size != line_co2.size:
        raise ValueError("gains and line CO2 must be non-empty and of the same length")
    median = float(np.median(gains))
    share = float(np.mean(gains > GAIN_PER_SCENARIO_MIN_KG_PER_H))
    crit_i = median >= GAIN_MEDIAN_MIN_KG_PER_H
    crit_ii = seed_spread_kg_per_h is not None and median >= GAIN_SEED_FACTOR * seed_spread_kg_per_h
    crit_iii = share >= GAIN_SCENARIO_SHARE_MIN
    return {
        "n_scenarios": int(gains.size),
        "gain_median_kg_per_h": median,
        "gain_min_kg_per_h": float(gains.min()),
        "gain_max_kg_per_h": float(gains.max()),
        "gain_relative_to_line_co2_median": float(np.median(gains / line_co2)),
        "share_above_per_scenario_threshold": share,
        "n_above_per_scenario_threshold": int(np.sum(gains > GAIN_PER_SCENARIO_MIN_KG_PER_H)),
        "seed_spread_kg_per_h": seed_spread_kg_per_h,
        "criterion_i_median": bool(crit_i),
        "criterion_ii_seed_spread": bool(crit_ii),
        "criterion_iii_share": bool(crit_iii),
        "demonstrated": bool(crit_i and crit_ii and crit_iii),
    }

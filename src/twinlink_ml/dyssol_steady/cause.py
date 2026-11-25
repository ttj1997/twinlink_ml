"""Add-on to the fresh-gas study: why the quadratic regression misses the thresholds.

Design and reading: ``docs/NN_dyssol_freshgas.md`` (section 17) and the docstring of
``scripts/run_dyssol_freshgas_cause.py``. Two things changed at once against Befund 128: the
fresh-gas state became an input, and the flow of the heat transfer medium went from a narrow
range to the full one. Variants of the box separate them; everything is computed with the Python
rebuild (:mod:`.reference`), which the control of the study found equal to Dyssol at every point.

**The heat recovery of the units.** By equations 4.1 / 4.2 (:func:`.reference.heat_recovery`) the
recovered heat is

    Q = (T_hot - T_cold) / (1 / A + 1 / B + 1 / C)

with the capacity rates ``A`` of the dryer exhaust, ``B`` of the medium and ``C`` of the fresh
gas. This is a smooth saturation in ``B`` (``Q -> 0`` for ``B -> 0``); the units have no kink of
the form ``min(C_hot, C_cold) x dT_max`` that an ideal counter-current exchanger would have.
:func:`capacity_ratio` gives ``B / min(A, C)``, the ratio at which such a kink would lie at 1; the
split of the errors by it is descriptive.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

import numpy as np

from .design import Box
from .materials import DyssolProperties
from .reference import LineInputs, LineOutputs

__all__ = [
    "MEDIUM",
    "RATIO_EDGES",
    "capacity_ratio",
    "cause_reading",
    "ratio_groups",
    "with_interval_of",
]

MEDIUM = "medium_kg_per_h"

RATIO_EDGES: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, math.inf)
"""Bins of the capacity ratio ``B / min(A, C)`` of the descriptive split."""


def with_interval_of(base: Box, other: Box, name: str) -> Box:
    """``base`` with the interval of the input ``name`` (of line 1) taken from ``other``."""
    column = f"line1/{name}"
    if column not in base.names or column not in other.names:
        raise ValueError(f"both boxes must hold {column}")
    i, j = base.names.index(column), other.names.index(column)
    lower, upper = list(base.lower), list(base.upper)
    lower[i], upper[i] = other.lower[j], other.upper[j]
    return Box(base.names, tuple(lower), tuple(upper))


def capacity_ratio(props: DyssolProperties, line: LineInputs, outputs: LineOutputs) -> float:
    """Capacity rate of the medium over the smaller one of the two gas streams, ``B / min(A, C)``.

    ``A``: dryer exhaust (dry gas and the vapour it carries after the dryer); ``C``: fresh gas;
    ``B``: medium (heat capacity of liquid water, as the unit takes it).
    """
    gas = line.dry_gas_kg_per_h / 3600.0
    hot = gas * (props.cp_gas + outputs.dryer_outlet_humidity_g_per_kg / 1000.0 * props.cp_vapour)
    cold = gas * (props.cp_gas + line.gas_humidity_g_per_kg / 1000.0 * props.cp_vapour)
    medium = line.medium_kg_per_h / 3600.0 * props.cp_liquid
    return medium / min(hot, cold)


def ratio_groups(ratio: np.ndarray, edges: Sequence[float] = RATIO_EDGES) -> dict[str, np.ndarray]:
    """Named masks over the points: the two sides of ratio 1, and the bins of ``edges``.

    ``medium_below_gas``: ratio below 1 (the medium is the smallest stream); ``medium_above_gas``:
    at or above 1. Bin ``ratio_<lo>_<hi>`` holds ``lo <= ratio < hi``.
    """
    ratio = np.asarray(ratio, dtype=np.float64)
    groups = {"medium_below_gas": ratio < 1.0, "medium_above_gas": ratio >= 1.0}
    for lo, hi in pairwise(edges):
        groups[f"ratio_{lo:g}_{hi:g}"] = (ratio >= lo) & (ratio < hi)
    return groups


def cause_reading(
    n_star_quadratic: Mapping[str, int | None], control_reproduces: bool
) -> dict[str, Any]:
    """Reading of the add-on from ``N*`` of the quadratic regression in the variants.

    ``V-a``: seven inputs, medium flow as in the box of Befund 128; ``V-b``: five inputs (fresh gas
    fixed as in Befund 128), full range of the medium flow; ``V-c``: the main run (control).

    ``K0``  the quadratic regression reaches the thresholds in ``V-a`` and not in ``V-b``: the
            cause is the range of the medium flow, not the additional inputs.
    ``K1``  the reverse: the cause is the fresh-gas inputs.
    ``K2``  otherwise (it reaches them in both or in neither).
    Without a reproduced control the add-on is not read.
    """
    narrow, full = n_star_quadratic["V-a"], n_star_quadratic["V-b"]
    if not control_reproduces:
        verdict = "not read (the control variant does not reproduce the main run)"
    elif narrow is not None and full is None:
        verdict = "K0"
    elif narrow is None and full is not None:
        verdict = "K1"
    else:
        verdict = "K2"
    return {
        "verdict": verdict,
        "n_star_quadratic": dict(n_star_quadratic),
        "control_reproduces": control_reproduces,
    }

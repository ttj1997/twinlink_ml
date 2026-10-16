r"""Search for the CO2-minimal operating point under the proxy constraints, and its decomposition.

Decision variables: process-air flow relative to the measured day (``air_factor``) and the
supply-air temperature, in the box of :class:`~twinlink_ml.plant_a_op.config.SearchSpace`.
Constraints (:class:`Constraints`), all evaluated with the stationary model:

* (a) outlet temperature >= ``outlet_min_celsius`` (observed minimum of the product);
* (b) relative exhaust humidity <= ``rh_max`` (observed maximum, RECONSTRUCTED -- the exhaust
  humidity is not measured);
* air factor >= ``air_factor_min`` (placeholder for the fluidisation limit).

The thresholds never exclude today's operation: if the model's own baseline lies beyond an
observed extreme (another weather, the other day's calibration), the threshold is moved to the
baseline (:func:`constraints_for`). The proxy constraints do NOT replace a product-moisture
measurement; the optimum is an upper bound of the lever, not an operating recommendation.

Search: full grid, then one local refinement; the measured point is always a candidate. The
problem is two-dimensional and cheap, so an exhaustive grid is preferred to a local solver (no
starting-point dependence, the binding constraints are read off directly).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .config import ConstraintVariant, SearchSpace
from .plant import OperatingPoints, PlantAModel

__all__ = [
    "BINDING_TOLERANCES",
    "Constraints",
    "Optimum",
    "coarse_grid",
    "constraints_for",
    "decompose",
    "optimise",
]

FloatArray = NDArray[np.float64]

BINDING_TOLERANCES: dict[str, float] = {
    "outlet_min": 0.05,  # K
    "rh_max": 0.002,  # relative humidity
    "air_factor_min": 1e-3,
    "air_factor_max": 1e-3,
    "supply_max": 0.02,  # K
    "supply_min": 0.02,  # K
}
"""Slack below which a constraint counts as binding at the optimum (after refinement)."""


@dataclass(frozen=True, slots=True)
class Constraints:
    """Thresholds of the proxy constraints and the minimum air factor (``None``: not used)."""

    outlet_min_celsius: float | None
    rh_max: float | None
    air_factor_min: float

    def admissible(self, points: OperatingPoints) -> NDArray[np.bool_]:
        ok = points.air_factor >= self.air_factor_min - 1e-12
        if self.outlet_min_celsius is not None:
            ok &= points.outlet_celsius >= self.outlet_min_celsius - 1e-9
        if self.rh_max is not None:
            ok &= points.outlet_relative_humidity <= self.rh_max + 1e-12
        # never over-saturate, whatever the variant
        ok &= points.outlet_relative_humidity <= 1.0
        return np.asarray(ok, dtype=np.bool_)


def constraints_for(
    variant: ConstraintVariant,
    *,
    observed_outlet_min_celsius: float,
    observed_rh_max: float,
    air_factor_min: float,
    baseline: OperatingPoints,
) -> Constraints:
    """Thresholds of a variant, moved to the model baseline if it lies beyond them."""
    t_base = float(baseline.outlet_celsius.ravel()[0])
    rh_base = float(baseline.outlet_relative_humidity.ravel()[0])
    t_min = min(observed_outlet_min_celsius, t_base) if variant in ("a", "a_and_b") else None
    rh_max = max(observed_rh_max, rh_base) if variant in ("b", "a_and_b") else None
    return Constraints(outlet_min_celsius=t_min, rh_max=rh_max, air_factor_min=air_factor_min)


@dataclass(frozen=True, slots=True)
class Optimum:
    """Baseline, best point, binding constraints and the decomposition of one search."""

    baseline: dict[str, float]
    best: dict[str, float]
    binding: tuple[str, ...]
    constraints: Constraints
    n_admissible: int
    decomposition: dict[str, float] = field(default_factory=dict)

    @property
    def saving_kg_h(self) -> float:
        return self.baseline["co2_kg_h"] - self.best["co2_kg_h"]

    @property
    def saving_fraction(self) -> float:
        return self.saving_kg_h / self.baseline["co2_kg_h"]

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline,
            "best": self.best,
            "binding": list(self.binding),
            "constraints": {
                "outlet_min_celsius": self.constraints.outlet_min_celsius,
                "rh_max": self.constraints.rh_max,
                "air_factor_min": self.constraints.air_factor_min,
            },
            "n_admissible": self.n_admissible,
            "saving_kg_per_day": 24.0 * self.saving_kg_h,
            "saving_fraction": self.saving_fraction,
            "decomposition_kg_per_day": self.decomposition,
        }


def _axis(lo: float, hi: float, step: float) -> FloatArray:
    n = round((hi - lo) / step)
    return np.linspace(lo, hi, n + 1)


def coarse_grid(
    model: PlantAModel, space: SearchSpace
) -> tuple[FloatArray, FloatArray, OperatingPoints]:
    """Evaluate the full box once (shared by every constraint set of the same model).

    The air axis always contains 1.0 and, with the default step, every value of the minimum-air
    grid (0.6 + k * 0.005).
    """
    f = _axis(space.air_factor_min, space.air_factor_max, space.air_factor_step)
    f = np.unique(np.concatenate([f, [1.0]]))
    t = _axis(space.supply_min_celsius, space.supply_max_celsius, space.supply_step_k)
    ff, tt = np.meshgrid(f, t, indexing="ij")
    return f, t, model.evaluate(ff, tt)


def _argbest(pts: OperatingPoints, constraints: Constraints) -> tuple[tuple[int, ...], int]:
    ok = constraints.admissible(pts)
    co2 = np.where(ok, pts.co2_kg_h, np.inf)
    index = tuple(int(k) for k in np.unravel_index(int(np.argmin(co2)), co2.shape))
    return index, int(ok.sum())


def _binding(best: dict[str, float], c: Constraints, space: SearchSpace) -> tuple[str, ...]:
    tol = BINDING_TOLERANCES
    out = []
    if (
        c.outlet_min_celsius is not None
        and best["outlet_celsius"] - c.outlet_min_celsius <= tol["outlet_min"]
    ):
        out.append("outlet_min")
    if c.rh_max is not None and c.rh_max - best["outlet_relative_humidity"] <= tol["rh_max"]:
        out.append("rh_max")
    if best["air_factor"] - c.air_factor_min <= tol["air_factor_min"]:
        out.append("air_factor_min")
    if space.air_factor_max - best["air_factor"] <= tol["air_factor_max"]:
        out.append("air_factor_max")
    if space.supply_max_celsius - best["supply_celsius"] <= tol["supply_max"]:
        out.append("supply_max")
    if best["supply_celsius"] - space.supply_min_celsius <= tol["supply_min"]:
        out.append("supply_min")
    if best["waste_heat_curtailed_kw"] > 1e-6:
        out.append("waste_heat_curtailed")
    return tuple(out)


def optimise(
    model: PlantAModel,
    constraints: Constraints,
    space: SearchSpace,
    grid: tuple[FloatArray, FloatArray, OperatingPoints] | None = None,
) -> Optimum:
    """Grid search plus one refinement; the measured point is always a candidate.

    ``grid`` is the result of :func:`coarse_grid` for the same model and space (recomputed if
    ``None``).
    """
    base_pts = model.baseline()
    baseline = base_pts.at(0)
    f, t, pts = grid if grid is not None else coarse_grid(model, space)
    f_lo = max(space.air_factor_min, constraints.air_factor_min)
    index, n_ok = _argbest(pts, constraints)
    candidates: list[dict[str, float]] = []
    if n_ok:
        i, j = index
        offsets = np.linspace(-2.0, 2.0, 4 * space.refine + 1)
        fine_f = np.unique(
            np.clip(f[i] + offsets * space.air_factor_step, f_lo, space.air_factor_max)
        )
        fine_t = np.unique(
            np.clip(
                t[j] + offsets * space.supply_step_k,
                space.supply_min_celsius,
                space.supply_max_celsius,
            )
        )
        ff, tt = np.meshgrid(fine_f, fine_t, indexing="ij")
        fine = model.evaluate(ff, tt)
        index2, n2 = _argbest(fine, constraints)
        candidates.append(pts.at(index))
        if n2:
            candidates.append(fine.at(index2))
    if bool(constraints.admissible(base_pts)[0]):
        candidates.append(baseline)
    if not candidates:
        raise RuntimeError("no admissible operating point (the baseline should always be one)")
    best = min(candidates, key=lambda d: d["co2_kg_h"])
    return Optimum(
        baseline=baseline,
        best=best,
        binding=_binding(best, constraints, space),
        constraints=constraints,
        n_admissible=n_ok,
        decomposition=decompose(baseline, best, model),
    )


def decompose(
    baseline: dict[str, float], best: dict[str, float], model: PlantAModel
) -> dict[str, float]:
    """Saving split into steam and fan electricity (kg CO2/d); steam further by its terms.

    ``steam = air heating - fan heat - waste heat``, so the steam part is exactly
    ``(dQ_air - dQ_fan_heat - dQ_waste) / eta * f_gas`` (positive = saving).
    """
    e = model.energy
    k_steam = 24.0 * e.gas_co2_kg_per_kwh / e.boiler_efficiency
    k_el = 24.0 * e.electricity_co2_kg_per_kwh
    d_air = baseline["air_heating_kw"] - best["air_heating_kw"]
    d_fan_heat = baseline["fan_heat_kw"] - best["fan_heat_kw"]
    d_waste = baseline["waste_heat_kw"] - best["waste_heat_kw"]
    d_el = (baseline["process_fan_kw"] + baseline["exhaust_fan_kw"]) - (
        best["process_fan_kw"] + best["exhaust_fan_kw"]
    )
    steam_part = k_steam * (baseline["steam_kw"] - best["steam_kw"])
    fan_part = k_el * d_el
    total = steam_part + fan_part
    return {
        "total": total,
        "steam": steam_part,
        "fan_electricity": fan_part,
        "steam_from_air_heating": k_steam * d_air,
        "steam_from_lost_fan_heat": -k_steam * d_fan_heat,
        "steam_from_lost_waste_heat": -k_steam * d_waste,
        "fan_electricity_process": k_el * (baseline["process_fan_kw"] - best["process_fan_kw"]),
        "fan_electricity_exhaust": k_el * (baseline["exhaust_fan_kw"] - best["exhaust_fan_kw"]),
        "steam_share_of_saving": steam_part / total if abs(total) > 1e-9 else float("nan"),
    }

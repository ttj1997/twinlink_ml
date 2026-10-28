r"""Steady headroom: fixed operating point (R0) against tracked operating point (R1).

Steady variant of R0 against R1 of Befund 105, on the exact steady-state solver (no surrogate, no
CasADi model; the IPOPT optimiser of :mod:`twinlink_ml.headroom.controllers` has fixed bounds of
the lab line). Decision: inlet temperature ``T`` and gas velocity ``u`` (the model's commands);
objective: the model's total power ``P`` (as the RTO of Befund 105/108); constraints: bed moisture
``X <= L`` and bed temperature ``T_bed <= T_max``.

Structure used (assumptions, checked by the search, not proven):

* at fixed ``u`` and disturbance ``d``, ``X`` falls and ``P`` and ``T_bed`` rise with ``T``; so the
  cheapest admissible ``T`` is the smallest one with ``X <= L`` (bisection to ``inlet_tolerance``),
  admissible only if ``T_bed <= T_max`` there;
* over ``u``: a grid of ``grid_points`` velocities, then a bounded Brent search between the
  neighbours of the best grid point (``scipy.optimize.minimize_scalar``).

R1 at ``d``: the optimum at ``d`` (margin 0: steady, disturbance known). R0: ONE command, the
cheapest at the nominal disturbance ``d0`` that holds both limits at every corner of the
disturbance box (the robust inlet temperature at ``u`` is the largest of the corners'). Saving at
``d``: ``(P(R0, d) - P(R1(d), d)) / P(R0, d)``.

A solver failure at a trial point counts as inadmissible there and is counted.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from ..simulation import GreyBoxFluidBedGranulator, PlaceholderConfigWarning, ProcessConfig
from ..simulation.operating_point import DEFAULT_SEARCH_MAX_ITERATIONS, DEFAULT_SEARCH_TOLERANCE
from ..simulation.steady_state import solve_steady_state
from .model import PlantParameters, PlantPoint, _bound, plant_config

__all__ = [
    "Disturbance",
    "Evaluation",
    "LabFactory",
    "PlantFactory",
    "SearchBox",
    "SteadySearch",
    "box_points",
]


@dataclass(frozen=True, slots=True)
class Disturbance:
    """Steady disturbance: spray rate (throughput), solid fraction, ambient temperature."""

    spray_kg_h: float
    solid_fraction: float
    ambient_celsius: float

    def as_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


def box_points(
    nominal: Disturbance, spray_relative: float, solid_relative: float, ambient_k: float
) -> tuple[list[Disturbance], list[Disturbance]]:
    """The 8 corners of the box and the 9 evaluation points (corners and the centre ``nominal``)."""
    corners = [
        Disturbance(
            nominal.spray_kg_h * (1.0 + a * spray_relative),
            nominal.solid_fraction * (1.0 + b * solid_relative),
            nominal.ambient_celsius + c * ambient_k,
        )
        for a, b, c in itertools.product((-1.0, 1.0), repeat=3)
    ]
    return corners, [*corners, nominal]


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Steady state at one command and disturbance."""

    moisture_kg_per_kg: float
    bed_temperature_celsius: float
    power_kw: float
    feasible: bool
    failure: str | None = None

    @property
    def ok(self) -> bool:
        return self.failure is None


@dataclass(frozen=True, slots=True)
class SearchBox:
    """Command box and search tolerances."""

    inlet_celsius: tuple[float, float]
    velocity_m_per_s: tuple[float, float]
    bed_temperature_limit_celsius: float
    grid_points: int
    inlet_tolerance_k: float
    velocity_relative_tolerance: float


ConfigFactory = Callable[[float, float, Disturbance], ProcessConfig]
"""``(inlet temperature, gas velocity, disturbance) -> process config`` with those nominals."""


@dataclass(frozen=True)
class PlantFactory:
    """Plant-near model: the operating point with the disturbance's spray, solid fraction, ambient.

    The inlet humidity stays the placeholder of the dried supply air at every ambient temperature;
    the ``evaporation`` switch is applied to the disturbed spray (:func:`.model.plant_config`).
    """

    base: ProcessConfig
    point: PlantPoint
    params: PlantParameters
    inlet_humidity_kg_per_kg: float
    material_values: tuple[tuple[str, float], ...] = ()

    def __call__(self, inlet: float, velocity: float, d: Disturbance) -> ProcessConfig:
        point = dataclasses.replace(
            self.point, spray_kg_h=d.spray_kg_h, ambient_celsius=d.ambient_celsius
        )
        params = dataclasses.replace(self.params, solid_fraction=d.solid_fraction)
        return plant_config(
            self.base,
            point,
            params,
            gas_velocity_m_per_s=velocity,
            inlet_humidity_kg_per_kg=self.inlet_humidity_kg_per_kg,
            inlet_celsius=inlet,
            material_values=dict(self.material_values),
        )


@dataclass(frozen=True)
class LabFactory:
    """Lab line (process config of the headroom study): nominals replaced, humidity kept."""

    base: ProcessConfig
    material_values: tuple[tuple[str, float], ...] = ()

    def __call__(self, inlet: float, velocity: float, d: Disturbance) -> ProcessConfig:
        a, dist = self.base.actions, self.base.disturbances
        config = dataclasses.replace(
            self.base,
            actions=dataclasses.replace(
                a,
                inlet_temperature_celsius=_bound(a.inlet_temperature_celsius, inlet),
                gas_velocity_m_per_s=_bound(a.gas_velocity_m_per_s, velocity),
                spray_rate_kg_per_h=_bound(a.spray_rate_kg_per_h, d.spray_kg_h),
            ),
            disturbances=dataclasses.replace(
                dist,
                ambient_temperature_celsius=_bound(
                    dist.ambient_temperature_celsius, d.ambient_celsius
                ),
                suspension_solid_fraction=_bound(dist.suspension_solid_fraction, d.solid_fraction),
            ),
        )
        if self.material_values:
            config = dataclasses.replace(
                config, material=dataclasses.replace(config.material, **dict(self.material_values))
            )
        return config


class SteadySearch:
    """Steady optima on one model (``factory``) at search precision (Befund 52)."""

    def __init__(self, factory: ConfigFactory, box: SearchBox) -> None:
        self.factory = factory
        self.box = box
        self.solves = 0
        self.failures = 0

    def evaluate(self, inlet: float, velocity: float, d: Disturbance) -> Evaluation:
        self.solves += 1
        try:
            config = self.factory(inlet, velocity, d)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PlaceholderConfigWarning)
                solution = solve_steady_state(
                    GreyBoxFluidBedGranulator(config),
                    max_iterations=DEFAULT_SEARCH_MAX_ITERATIONS,
                    tolerance=DEFAULT_SEARCH_TOLERANCE,
                )
        except (RuntimeError, ValueError) as error:
            self.failures += 1
            return Evaluation(
                math.nan, math.nan, math.nan, False, f"{type(error).__name__}: {error}"
            )
        return Evaluation(
            moisture_kg_per_kg=float(solution.particle_moisture_kg_per_kg),
            bed_temperature_celsius=float(solution.bed_temperature_celsius),
            power_kw=float(solution.diagnostics["power_total_kw"]),
            feasible=bool(solution.feasible),
        )

    def _admissible(self, e: Evaluation, limit: float) -> bool:
        return e.ok and e.moisture_kg_per_kg <= limit

    def inlet_for_limit(self, velocity: float, d: Disturbance, limit: float) -> float | None:
        """Smallest inlet temperature with ``X <= limit`` at ``velocity`` (None: none in box)."""
        lo, hi = self.box.inlet_celsius
        if self._admissible(self.evaluate(lo, velocity, d), limit):
            return lo
        if not self._admissible(self.evaluate(hi, velocity, d), limit):
            return None
        while hi - lo > self.box.inlet_tolerance_k:
            mid = 0.5 * (lo + hi)
            if self._admissible(self.evaluate(mid, velocity, d), limit):
                hi = mid
            else:
                lo = mid
        return hi

    def _minimise(self, objective: Callable[[float], float]) -> tuple[float, float]:
        lo, hi = self.box.velocity_m_per_s
        grid = np.linspace(lo, hi, self.box.grid_points)
        values = np.array([objective(float(u)) for u in grid])
        if not np.isfinite(values).any():
            return math.nan, math.inf
        k = int(np.nanargmin(np.where(np.isfinite(values), values, np.inf)))
        a, b = float(grid[max(k - 1, 0)]), float(grid[min(k + 1, len(grid) - 1)])
        best_u, best = float(grid[k]), float(values[k])
        if b > a:
            result = minimize_scalar(
                objective,
                bounds=(a, b),
                method="bounded",
                options={"xatol": self.box.velocity_relative_tolerance * best_u},
            )
            if np.isfinite(result.fun) and result.fun < best:
                best_u, best = float(result.x), float(result.fun)
        return best_u, best

    def optimum(self, d: Disturbance, limit: float) -> dict[str, Any]:
        """R1 at ``d``: the cheapest admissible command."""

        def objective(u: float) -> float:
            t = self.inlet_for_limit(u, d, limit)
            if t is None:
                return math.inf
            e = self.evaluate(t, u, d)
            if not e.ok or e.bed_temperature_celsius > self.box.bed_temperature_limit_celsius:
                return math.inf
            return e.power_kw

        u, power = self._minimise(objective)
        if not math.isfinite(power):
            return {"admissible": False}
        t = self.inlet_for_limit(u, d, limit)
        assert t is not None
        e = self.evaluate(t, u, d)
        return self._point(t, u, e)

    def robust(
        self, corners: Sequence[Disturbance], nominal: Disturbance, limit: float
    ) -> dict[str, Any]:
        """R0: the cheapest command at ``nominal`` that holds both limits at every corner."""

        def inlet(u: float) -> float | None:
            ts = [self.inlet_for_limit(u, d, limit) for d in corners]
            if any(t is None for t in ts):
                return None
            t = max(t for t in ts if t is not None)
            for d in corners:
                e = self.evaluate(t, u, d)
                if not e.ok or e.bed_temperature_celsius > self.box.bed_temperature_limit_celsius:
                    return None
            return t

        def objective(u: float) -> float:
            t = inlet(u)
            if t is None:
                return math.inf
            e = self.evaluate(t, u, nominal)
            return e.power_kw if e.ok else math.inf

        u, power = self._minimise(objective)
        if not math.isfinite(power):
            return {"admissible": False}
        t = inlet(u)
        assert t is not None
        return self._point(t, u, self.evaluate(t, u, nominal))

    @staticmethod
    def _point(t: float, u: float, e: Evaluation) -> dict[str, Any]:
        return {
            "admissible": True,
            "inlet_celsius": t,
            "velocity_m_per_s": u,
            "moisture_kg_per_kg": e.moisture_kg_per_kg,
            "bed_temperature_celsius": e.bed_temperature_celsius,
            "power_kw": e.power_kw,
            "steady_state_feasible": e.feasible,
        }

    def fixed_against_tracked(
        self,
        nominal: Disturbance,
        corners: Sequence[Disturbance],
        points: Sequence[Disturbance],
        limit: float,
    ) -> dict[str, Any]:
        """R0 once, R1 and the saving at every evaluation point."""
        r0 = self.robust(corners, nominal, limit)
        rows: list[dict[str, Any]] = []
        for d in points:
            r1 = self.optimum(d, limit)
            row: dict[str, Any] = {"disturbance": d.as_dict(), "r1": r1}
            if r0["admissible"]:
                e0 = self.evaluate(r0["inlet_celsius"], r0["velocity_m_per_s"], d)
                row["r0_at_d"] = {
                    "moisture_kg_per_kg": e0.moisture_kg_per_kg,
                    "bed_temperature_celsius": e0.bed_temperature_celsius,
                    "power_kw": e0.power_kw,
                    "holds_limit": bool(e0.ok and e0.moisture_kg_per_kg <= limit + 1e-6),
                }
                if r1["admissible"] and e0.ok:
                    row["saving"] = (e0.power_kw - r1["power_kw"]) / e0.power_kw
            rows.append(row)
        savings = [r["saving"] for r in rows if "saving" in r]
        return {
            "limit_kg_per_kg": limit,
            "r0": r0,
            "points": rows,
            "saving_nominal": rows[-1].get("saving", math.nan) if rows else math.nan,
            "saving_median": float(np.median(savings)) if savings else math.nan,
            "saving_min": float(np.min(savings)) if savings else math.nan,
            "saving_max": float(np.max(savings)) if savings else math.nan,
            "solves": self.solves,
            "solver_failures": self.failures,
        }

    def margin_reference(
        self, nominal: Disturbance, limit: float, margins: tuple[float, float]
    ) -> dict[str, Any]:
        """Befund 108 definition at ``nominal``: optimum with the R0 margin vs the R1 margin."""
        fixed = self.optimum(nominal, limit - margins[0])
        tracked = self.optimum(nominal, limit - margins[1])
        saving = math.nan
        if fixed["admissible"] and tracked["admissible"]:
            saving = (fixed["power_kw"] - tracked["power_kw"]) / fixed["power_kw"]
        return {
            "margins_kg_per_kg": list(margins),
            "fixed": fixed,
            "tracked": tracked,
            "saving": saving,
        }

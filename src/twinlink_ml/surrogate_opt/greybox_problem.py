"""P2: the steady grey-box model of the lab line, with the solver of the RTO R1 (docs/24).

The model is the one of :class:`~twinlink_ml.headroom.controllers.SteadyStateOptimizer`: the
unchanged physics through :func:`~twinlink_ml.headroom.symbolic.symbolic_derivative` on the reduced
state ``(m_s, m_w, T_bed)`` with ``m_s`` at its set point, the lab-line settings of
``conf/headroom/dynamic_headroom.yaml`` (:func:`~twinlink_ml.headroom.config.study_process_config`:
quality limit, nozzle pressure at the Befund-52 fixed point), no inlet-temperature offset.

Decisions ``u``: inlet temperature [degC] and gas velocity [m/s] (the box of R1, the action box of
the process config). Disturbances ``w``: ambient temperature [degC], ambient relative humidity
[-], spray rate [kg/h] (throughput), solid fraction [kg/kg] -- the order of P1, not that of
:data:`~twinlink_ml.headroom.symbolic.DISTURBANCE_CHANNELS`.

A true evaluation solves ``dm_w/dt = 0, dT_bed/dt = 0`` at fixed ``(u, w)`` (CasADi Newton,
several starts; residual checked) and returns the total power of the model [kW] (the objective
of R1) and the constraints ``X - X_lim`` [kg/kg] and ``T_bed - T_bed,max`` [K]. The optimum at the
true model (:meth:`GreyBoxProblem.optimum`) is the R1 NLP (IPOPT, same options and bounds) from
several starts.

Parameters (:class:`GreyBoxProblemFamily`): ``max_saturation_degree`` (``nu_max``), the bed heat
loss ``heat_loss_ua_w_per_k`` and one GAB parameter of the isotherm (the hidden shift of the study,
named in the YAML). CO2 (side metric) books the heater on gas as the headroom study does
(:func:`~twinlink_ml.evaluation.carrier_energy.carrier_energy`, its gas-heater efficiency).
"""

from __future__ import annotations

import dataclasses
import warnings
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import casadi as cs
import numpy as np

from twinlink_ml.evaluation.carrier_energy import carrier_energy
from twinlink_ml.headroom.config import HeadroomStudyConfig
from twinlink_ml.headroom.symbolic import symbolic_derivative
from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    PlaceholderConfigWarning,
    ProcessConfig,
)
from twinlink_ml.simulation.sorption import validate_isotherm_against_critical_moisture

from .problem import Box, ConstraintSpec, Evaluation, ParameterSpec, ProblemFamily, SteadyProblem

__all__ = [
    "GAB_PARAMETERS",
    "GreyBoxProblem",
    "GreyBoxProblemFamily",
    "OptimumResult",
    "apply_parameters",
]

_M_W = (0.0, 10.0)
"""Bounds of the water mass [kg] in the R1 NLP (``SteadyStateOptimizer``)."""
_T_BED = (-20.0, 200.0)
"""Bounds of the bed temperature [degC] in the R1 NLP."""

_CACHE_SIZE = 8
"""Problems kept per family (truth, nominal model and the last members of a fit)."""

_IPOPT_OPTIONS: dict[str, Any] = {
    "ipopt.print_level": 0,
    "print_time": False,
    "ipopt.tol": 1e-6,
    "ipopt.acceptable_tol": 1e-5,
    "ipopt.max_iter": 1000,
    "ipopt.sb": "yes",
}
"""The options of :mod:`twinlink_ml.headroom.controllers` (R1)."""

GAB_PARAMETERS = {
    "gab_monolayer_factor": "gab_monolayer_moisture_kg_per_kg",
    "gab_c_constant_factor": "gab_c_constant_at_reference_temperature",
    "gab_k_constant": "gab_k_constant_at_reference_temperature",
    "gab_c_enthalpy_factor": "gab_c_enthalpy_j_per_mol",
    "gab_k_enthalpy_factor": "gab_k_enthalpy_j_per_mol",
}
"""GAB parameters as the robustness study of docs/24 shifts them: factors on the nominal value,
except ``gab_k_constant`` (an absolute value)."""


def apply_parameters(process: ProcessConfig, values: Mapping[str, float]) -> ProcessConfig:
    """The process config with ``nu_max``, heat loss and GAB parameters set (names as above)."""
    material, geometry = process.material, process.geometry
    changes: dict[str, float] = {}
    for name, value in values.items():
        if name == "max_saturation_degree":
            changes[name] = float(value)
        elif name == "heat_loss_ua_w_per_k":
            geometry = dataclasses.replace(geometry, heat_loss_ua_w_per_k=float(value))
        elif name == "gab_k_constant":
            changes[GAB_PARAMETERS[name]] = float(value)
        elif name in GAB_PARAMETERS:
            field = GAB_PARAMETERS[name]
            base = getattr(material, field)
            if base is None:
                raise ValueError(f"{field} is not set in the process config")
            changes[field] = float(base) * float(value)
        else:
            raise ValueError(f"unknown parameter {name!r}")
    return dataclasses.replace(
        process,
        material=dataclasses.replace(material, **changes),  # type: ignore[arg-type]
        geometry=geometry,
    )


@dataclass(frozen=True)
class OptimumResult:
    """Optimum of one disturbance state: decision, energy and the multistart record."""

    u: np.ndarray
    energy_kw: float
    feasible: bool
    starts: int
    converged_starts: int
    starts_at_best: int
    """Converged starts whose objective is within ``rtol`` of the best (convergence check)."""
    spread_kw: float
    """Range of the objective over the converged starts."""


class GreyBoxProblem(SteadyProblem):
    """P2 at one process config (see the module doc)."""

    def __init__(
        self,
        study: HeadroomStudyConfig,
        process: ProcessConfig,
        disturbances: Box,
        *,
        tolerances: Mapping[str, float] | None = None,
        name: str = "P2_greybox",
    ) -> None:
        super().__init__()
        self.name = name
        self.study = study
        self.process = process
        acts = process.actions
        inlet, gas = acts.inlet_temperature_celsius, acts.gas_velocity_m_per_s
        self.decisions = Box(
            ("inlet_temperature_celsius", "gas_velocity_m_per_s"),
            (inlet.minimum, gas.minimum),
            (inlet.maximum, gas.maximum),
        )
        self.disturbances = disturbances
        tol = dict(tolerances or {})
        self.moisture_limit = study.line.product_moisture_limit_kg_per_kg
        self.bed_limit = study.line.bed_temperature_limit_celsius
        self.constraints = (
            ConstraintSpec("moisture", "kg/kg", 0.01, tol.get("moisture", 1e-4),
                           "product moisture below the quality limit"),
            ConstraintSpec("bed_temperature", "K", 10.0, tol.get("bed_temperature", 0.05),
                           "bed temperature below its limit"),
        )  # fmt: skip
        self.solid_mass = process.geometry.bed_mass_solid_setpoint_kg

    # ---------------------------------------------------------------- symbolic model
    def _derivative(self, z: cs.SX, u: cs.SX, w: cs.SX) -> tuple[cs.SX, cs.SX]:
        state = cs.vertcat(self.solid_mass, z[0], z[1])
        action = cs.vertcat(u[0], u[1], w[2], self.study.line.nozzle_pressure_bar)
        disturbance = cs.vertcat(w[0], w[1], w[3])
        derivative, power = symbolic_derivative(self.process, state, action, disturbance)
        return derivative, power

    @cached_property
    def _root(self) -> tuple[cs.Function, cs.Function]:
        z, p = cs.SX.sym("z", 2), cs.SX.sym("p", 6)
        derivative, power = self._derivative(z, p[0:2], p[2:6])
        residual = cs.vertcat(derivative[1], derivative[2])
        finder = cs.rootfinder(
            "steady_root",
            "newton",
            cs.Function("res", [z, p], [residual]),
            {"abstol": 1e-12, "max_iter": 100, "error_on_fail": False},
        )
        return finder, cs.Function("out", [z, p], [residual, power])

    @cached_property
    def _nlp(self) -> cs.Function:
        x, p = cs.SX.sym("x", 4), cs.SX.sym("p", 4)  # m_w, T_bed, T_in, u_gas | w
        derivative, power = self._derivative(x[0:2], x[2:4], p)
        g = cs.vertcat(derivative[1], derivative[2], x[0] / self.solid_mass, x[1])
        return cs.nlpsol("steady", "ipopt", {"x": x, "p": p, "f": power, "g": g}, _IPOPT_OPTIONS)

    # ---------------------------------------------------------------- evaluation
    def steady_state(self, u: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
        """``(m_w, T_bed)`` and power [kW] at ``(u, w)``; raises if no root is found.

        Newton from several starts; a root counts only inside the bounds of the R1 NLP (Newton
        can end on spurious roots of the clipped model far outside, e.g. ``T_bed < -1000``).
        Without one, the nested bisection :meth:`_bisect` (unique root by monotonicity).
        """
        finder, out = self._root
        p = np.concatenate((u, w))
        best: tuple[float, np.ndarray, float] | None = None
        for x0 in ((0.25, 45.0), (0.1, 35.0), (0.4, 30.0), (0.05, 60.0), (0.6, 25.0)):
            z = np.asarray(finder(np.array(x0), p)).ravel()
            residual, power = (np.asarray(v).ravel() for v in out(z, p))
            scaled = float(max(abs(residual[0]) * 1e4, abs(residual[1]) * 1e2))
            inside = bool(np.all(np.isfinite(z)) and _M_W[0] < z[0] <= _M_W[1])
            inside = inside and _T_BED[0] <= z[1] <= _T_BED[1]
            if inside and (best is None or scaled < best[0]):
                best = (scaled, z, float(power[0]))
            if best is not None and best[0] < 1e-8:
                break
        if best is None or best[0] > 1e-6:
            z = self._bisect(p)
            residual, power = (np.asarray(v).ravel() for v in out(z, p))
            if float(max(abs(residual[0]) * 1e4, abs(residual[1]) * 1e2)) > 1e-6:
                raise RuntimeError(f"steady state not found at u={u}, w={w}")
            best = (0.0, z, float(power[0]))
        return best[1], best[2]

    def _bisect(self, p: np.ndarray, steps: int = 80) -> np.ndarray:
        """Fallback where Newton fails (very dry bed, kinks of the model): nested bisection.

        ``dm_w/dt`` falls monotonically in ``m_w`` at fixed ``T_bed`` and ``dT_bed/dt`` (with the
        water balance solved) falls in ``T_bed`` -- the structure the exact NumPy solver of
        :mod:`twinlink_ml.simulation.steady_state` uses; brackets are the bounds of the R1 NLP.
        """
        out = self._root[1]

        def water(m_w: float, t: float) -> float:
            return float(np.asarray(out(np.array([m_w, t]), p)[0]).ravel()[0])

        def inner(t: float) -> float:
            lo, hi = _M_W
            if water(hi, t) > 0.0:
                return hi
            for _ in range(steps):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if water(mid, t) > 0.0 else (lo, mid)
            return 0.5 * (lo + hi)

        def heat(t: float) -> float:
            return float(np.asarray(out(np.array([inner(t), t]), p)[0]).ravel()[1])

        lo, hi = _T_BED
        if not (heat(lo) > 0.0 > heat(hi)):
            raise RuntimeError(f"no steady-state bracket at p={p}")
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if heat(mid) > 0.0 else (lo, mid)
        t = 0.5 * (lo + hi)
        return np.array([inner(t), t])

    def _evaluate(self, u: np.ndarray, w: np.ndarray) -> Evaluation:
        z, power = self.steady_state(u, w)
        moisture = z[0] / self.solid_mass
        g = np.array([moisture - self.moisture_limit, z[1] - self.bed_limit])
        return Evaluation(
            power, g, {"moisture_kg_per_kg": moisture, "bed_temperature_celsius": z[1]}
        )

    def side_metrics(self, u: np.ndarray, w: np.ndarray) -> dict[str, float]:
        """CO2 (gas-fired heater, headroom booking) and the power terms at ``(u, w)``."""
        z, _ = self.steady_state(u, w)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            sim = GreyBoxFluidBedGranulator(self.process)
        state = np.array([self.solid_mass, z[0], 1.0, z[1]])
        action = np.array([u[0], u[1], w[2], self.study.line.nozzle_pressure_bar])
        _, diag = sim.evaluate_derivative(state, action, np.array([w[0], w[1], w[3]]))
        carriers = carrier_energy(
            power_heater_kw=float(diag["power_heater_kw"]),
            power_fan_kw=float(diag["power_fan_kw"]),
            power_pump_kw=float(diag["power_pump_kw"]),
            power_atomiser_kw=float(diag["power_atomiser_kw"]),
            heater="gas",
            process_heater_efficiency=self.process.energy.heater_efficiency,
            gas_heater_efficiency=self.study.control.gas_heater_efficiency,
        )
        return {
            "co2_kg_per_h": carriers.co2_kg_per_h,
            "gas_kw": carriers.gas_kw,
            "electric_kw": carriers.electric_kw,
            "power_total_kw_numpy": float(diag["power_total_kw"]),
        }

    # ---------------------------------------------------------------- optimum (R1 NLP)
    def optimum(
        self,
        w: np.ndarray,
        starts: np.ndarray,
        *,
        margin: float | np.ndarray = 0.0,
        rtol: float = 1e-6,
    ) -> OptimumResult:
        """R1 NLP at ``w`` from every start ``(T_in, u_gas)`` (``m_w``, ``T_bed`` from R1's guess).

        ``margin`` (scalar or one per constraint) tightens the constraints in normalised units
        (``g / scale + margin <= 0``; for the moisture alone this is the margin ``m`` of R0/R1 in
        units of the scale 0.01 kg/kg).
        """
        nlp = self._nlp
        lbx = np.array([_M_W[0], _T_BED[0], *self.decisions.lower])
        ubx = np.array([_M_W[1], _T_BED[1], *self.decisions.upper])
        lbg = np.array([0.0, 0.0, -np.inf, -np.inf])
        sm, sb = self.scales
        mm, mb = np.broadcast_to(np.asarray(margin, dtype=np.float64), (2,))
        ubg = np.array([0.0, 0.0, self.moisture_limit - mm * sm, self.bed_limit - mb * sb])
        results: list[tuple[float, np.ndarray]] = []
        for start in np.atleast_2d(starts):
            sol = nlp(x0=np.array([0.25, 45.0, *start]), p=w, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            if nlp.stats()["success"]:
                results.append((float(sol["f"]), np.asarray(sol["x"]).ravel()[2:4]))
        if not results:
            return OptimumResult(np.full(2, np.nan), np.nan, False, len(starts), 0, 0, np.nan)
        values = np.array([r[0] for r in results])
        best = int(np.argmin(values))
        return OptimumResult(
            results[best][1],
            float(values[best]),
            True,
            len(np.atleast_2d(starts)),
            len(results),
            int(np.sum(values <= values[best] * (1.0 + rtol))),
            float(values.max() - values.min()),
        )


class GreyBoxProblemFamily(ProblemFamily):
    """P2 for values of ``nu_max``, heat loss and one GAB parameter (see the module doc)."""

    def __init__(
        self,
        study: HeadroomStudyConfig,
        process: ProcessConfig,
        disturbances: Box,
        nominal_disturbance: tuple[float, ...],
        parameters: tuple[ParameterSpec, ...],
        *,
        tolerances: Mapping[str, float] | None = None,
    ) -> None:
        self.study = study
        self.process = process
        self.disturbances = disturbances
        self._nominal_disturbance = nominal_disturbance
        self.parameters = parameters
        self.tolerances = tolerances
        self._cache: OrderedDict[tuple[tuple[str, float], ...], GreyBoxProblem] = OrderedDict()

    @property
    def nominal_disturbance(self) -> np.ndarray:
        return np.asarray(self._nominal_disturbance, dtype=np.float64)

    def build(self, values: Mapping[str, float]) -> GreyBoxProblem:
        full = {**self.nominal_values, **values}
        key = tuple(sorted((k, float(v)) for k, v in full.items()))
        if key not in self._cache:
            process = apply_parameters(self.process, full)
            validate_isotherm_against_critical_moisture(process)
            self._cache[key] = GreyBoxProblem(
                self.study, process, self.disturbances, tolerances=self.tolerances
            )
            while len(self._cache) > _CACHE_SIZE:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]

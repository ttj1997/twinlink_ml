"""Running controllers on scenarios: initial conditions, lockstep truth runs, per-scenario records.

:func:`run_controller` is the single entry point used by the driver
(``scripts/run_dynamic_headroom.py``), its worker processes and the tests. It builds the
controllers of one kind for a list of scenarios, runs them in lockstep on one
:class:`~.truth.TruthPlant` batch and returns one :class:`RunRecord` per scenario.

Initial condition (all controllers alike): the steady optimum at the first disturbance sample with
the bound ``limit - start_margin`` (state and command). With a spread isotherm (robustness
variant) the TRUE plant has been running at that command: its state is the truth model after a
warm-up of :data:`WARM_UP_S` at the command and the first disturbance sample.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.stats import qmc

from twinlink_ml.simulation import GreyBoxFluidBedGranulator, ProcessConfig
from twinlink_ml.simulation.sorption import validate_isotherm_against_critical_moisture

from .config import HeadroomStudyConfig
from .controllers import (
    Controller,
    DynamicOptimizer,
    FixedPointController,
    MpcController,
    PreviewController,
    QuasiStationaryController,
    SteadyStateOptimizer,
    nominal_disturbance,
)
from .scenarios import Scenario
from .symbolic import steady_particle_count
from .truth import TruthPlant, TruthResult

__all__ = [
    "WARM_UP_S",
    "ControllerKind",
    "IsothermDraw",
    "RunRecord",
    "Solvers",
    "initial_conditions",
    "isotherm_draws",
    "run_controller",
]

ControllerKind = Literal["R0", "R1", "R2", "R3"]
WARM_UP_S = 6 * 3600.0
"""Warm-up of the truth model with a spread isotherm (about nine moisture time constants)."""


@dataclass
class Solvers:
    """Optimisers of one process, built lazily (one per worker process)."""

    study: HeadroomStudyConfig
    process: ProcessConfig
    _steady: SteadyStateOptimizer | None = None
    _preview: dict[int, DynamicOptimizer] = field(default_factory=dict)

    @property
    def steady(self) -> SteadyStateOptimizer:
        """Steady-state optimiser (R0, R1, initial conditions)."""
        if self._steady is None:
            self._steady = SteadyStateOptimizer(self.study, self.process)
        return self._steady

    def preview(self, n_intervals: int) -> DynamicOptimizer:
        """Full-horizon optimiser for R2 (MX, see :class:`DynamicOptimizer`)."""
        if n_intervals not in self._preview:
            self._preview[n_intervals] = DynamicOptimizer(
                self.study,
                self.process,
                n_intervals,
                warm_start=False,
                expression="MX",
                guess_start=True,
            )
        return self._preview[n_intervals]

    def mpc(self) -> DynamicOptimizer:
        """A fresh receding-horizon optimiser for R3 (one per scenario: own warm start)."""
        return DynamicOptimizer(self.study, self.process, self.study.control.mpc_intervals)


@dataclass(frozen=True)
class IsothermDraw:
    """One spread of the GAB parameters of the truth model (robustness variant)."""

    index: int
    parameters: dict[str, float]
    process: ProcessConfig


@dataclass
class RunRecord:
    """Truth-model record of one controller on one scenario (JSON-serialisable via ``asdict``)."""

    scenario: str
    family: str
    kind: str
    index: int
    controller: str
    margin: float
    variant: int | None
    energy_kwh: float
    co2_kg: float
    electric_kwh: float
    gas_kwh: float
    max_violation: float
    max_bed_temperature: float
    energy_kwh_per_interval: list[float]
    moisture_max_per_interval: list[float]
    commands: list[list[float]]
    solver: dict[str, Any]
    change: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Plain dict."""
        return dataclasses.asdict(self)


def initial_conditions(
    solvers: Solvers, scenarios: Sequence[Scenario]
) -> tuple[np.ndarray, np.ndarray]:
    """Nominal initial states ``(n, 4)`` and commands ``(n, 2)``, see the module docstring."""
    start_margin = solvers.study.line.start_margin_kg_per_kg
    states, commands = [], []
    for scenario in scenarios:
        d0 = scenario.disturbances[:, 0]
        point = solvers.steady.solve(d0, float(scenario.moisture_limit[0]) - start_margin)
        count = steady_particle_count(solvers.process, float(d0[2]))
        states.append([point.state[0], point.state[1], count, point.state[2]])
        commands.append(point.command)
    return np.array(states), np.array(commands)


def warm_up(
    truth: ProcessConfig,
    states: np.ndarray,
    commands: np.ndarray,
    scenarios: Sequence[Scenario],
    duration_s: float = WARM_UP_S,
) -> np.ndarray:
    """State of the truth model after running at the commands and the first disturbance sample."""
    simulator = GreyBoxFluidBedGranulator(truth)
    d0 = np.stack([s.disturbances[:, 0] for s in scenarios])
    nozzle = truth.actions.nozzle_pressure_bar.nominal
    action = np.stack(
        [commands[:, 0] + d0[:, 4], commands[:, 1], d0[:, 3], np.full(len(d0), nozzle)], axis=-1
    )
    state = np.array(states, dtype=np.float64)
    for _ in range(round(duration_s / truth.numerics.sample_time_s)):
        state = np.asarray(simulator.step(state, action, d0[:, 0:3]).state)
    return state


def _change_metrics(
    result: TruthResult, i: int, scenario: Scenario, tolerance: float, interval_s: float
) -> dict[str, Any]:
    dt = scenario.sample_time_s
    assert scenario.change_time_s is not None
    c = round(scenario.change_time_s / dt)
    excess = result.moisture[i] - result.moisture_limit[i]
    after = excess[c:]
    bad = np.flatnonzero(after > tolerance)
    duration = 0.0 if bad.size == 0 else float((bad[-1] + 1) * dt)
    k = round(scenario.change_time_s / interval_s)
    energy = result.energy_kwh_per_interval[i]
    return {
        "change_time_s": scenario.change_time_s,
        "duration_to_spec_s": duration,
        "max_violation_before": float(max(excess[:c].max(), 0.0)),
        "max_violation_after": float(max(after.max(), 0.0)),
        "energy_before_kwh": float(energy[:k].sum()),
        "energy_after_kwh": float(energy[k:].sum()),
        "limit_before": float(result.moisture_limit[i][c - 1]),
        "limit_after": float(result.moisture_limit[i][c]),
    }


def _build_controller(
    kind: ControllerKind, margin: float, solvers: Solvers, n_intervals: int
) -> Controller:
    if kind == "R0":
        nominal = nominal_disturbance(solvers.study, solvers.process)
        limit = solvers.study.line.product_moisture_limit_kg_per_kg
        return FixedPointController(solvers.steady.solve(nominal, limit - margin))
    if kind == "R1":
        return QuasiStationaryController(solvers.steady, margin)
    if kind == "R2":
        return PreviewController(
            solvers.preview(n_intervals),
            margin,
            guess=QuasiStationaryController(solvers.steady, margin),
        )
    if kind == "R3":
        return MpcController(solvers.mpc(), margin)
    raise ValueError(f"unknown controller {kind!r}")


def run_controller(
    kind: ControllerKind,
    margin: float,
    scenarios: Sequence[Scenario],
    solvers: Solvers,
    *,
    truth: ProcessConfig | None = None,
    variant: int | None = None,
) -> list[RunRecord]:
    """Run controller ``kind`` with ``margin`` on ``scenarios`` (one lockstep truth batch).

    ``truth``: process config of the truth model (default: the controllers' own, i.e. nominal);
    with a different config the initial state is warmed up on the truth, see the module
    docstring. A failed optimisation holds the previous command and is counted.
    """
    study = solvers.study
    truth_config = solvers.process if truth is None else truth
    states, commands = initial_conditions(solvers, scenarios)
    if truth is not None:
        states = warm_up(truth_config, states, commands, scenarios)
    n_intervals = round(scenarios[0].duration_s / study.control.interval_s)
    plant = TruthPlant(
        truth_config,
        scenarios,
        states,
        commands,
        interval_s=study.control.interval_s,
        gas_heater_efficiency=study.control.gas_heater_efficiency,
    )
    controllers = [_build_controller(kind, margin, solvers, n_intervals) for _ in scenarios]
    solver_info: list[dict[str, Any]] = [
        {"failures": 0, "iterations": [], "max_slack": 0.0} for _ in scenarios
    ]
    for i, (controller, scenario) in enumerate(zip(controllers, scenarios, strict=True)):
        try:
            controller.reset(scenario, states[i], commands[i])
        except RuntimeError as error:  # R2 only: fall back to the guess (R1 schedule)
            solver_info[i]["failures"] += 1
            solver_info[i]["reset_error"] = str(error)
            controllers[i] = QuasiStationaryController(solvers.steady, margin)
        if isinstance(controllers[i], PreviewController):
            solver_info[i].update(controllers[i].solve_stats)
    last = np.array(commands, dtype=np.float64)
    for k in range(plant.n_intervals):
        disturbance = plant.current_disturbance()
        limit = plant.current_limit()
        realised = plant.last_commands
        new = np.empty_like(last)
        for i, controller in enumerate(controllers):
            try:
                new[i] = controller.command(
                    k, plant.state[i], realised[i], disturbance[i], limit[i]
                )
            except RuntimeError:
                solver_info[i]["failures"] += 1
                new[i] = last[i]
            if isinstance(controller, MpcController):
                stats = controller.optimizer.last_stats
                solver_info[i]["iterations"].append(int(stats.get("iterations", -1)))
                solver_info[i]["max_slack"] = max(solver_info[i]["max_slack"], controller.max_slack)
        plant.advance(new)
        last = new
    result = plant.result()
    tolerance = study.control.violation_tolerance_kg_per_kg
    spi = round(study.control.interval_s / result.sample_time_s)
    records = []
    for i, scenario in enumerate(scenarios):
        info = solver_info[i]
        iterations = info.pop("iterations")
        if iterations:
            info["iterations_mean"] = float(np.mean(iterations))
            info["iterations_max"] = int(np.max(iterations))
        records.append(
            RunRecord(
                scenario=scenario.name,
                family=scenario.family,
                kind=scenario.kind,
                index=scenario.index,
                controller=kind,
                margin=float(margin),
                variant=variant,
                energy_kwh=float(result.energy_kwh[i]),
                co2_kg=float(result.co2_kg[i]),
                electric_kwh=float(result.electric_kwh_per_interval[i].sum()),
                gas_kwh=float(result.gas_kwh_per_interval[i].sum()),
                max_violation=float(result.max_violation[i]),
                max_bed_temperature=float(result.bed_temperature[i].max()),
                energy_kwh_per_interval=[float(v) for v in result.energy_kwh_per_interval[i]],
                moisture_max_per_interval=[
                    float(v) for v in result.moisture[i].reshape(-1, spi).max(axis=1)
                ],
                commands=[[float(a), float(b)] for a, b in result.commands[i]],
                solver=info,
                change=None
                if scenario.change_time_s is None
                else _change_metrics(result, i, scenario, tolerance, study.control.interval_s),
            )
        )
    return records


def isotherm_draws(
    study: HeadroomStudyConfig, process: ProcessConfig
) -> tuple[list[IsothermDraw], int]:
    """``robustness.draws`` valid GAB spreads (LHS, seeded) and the number of rejected draws.

    A draw is rejected when the isotherm reaches the critical moisture on the safety grid
    (:func:`validate_isotherm_against_critical_moisture`); rejected points are replaced by the
    next points of the same LHS stream.
    """
    rob = study.robustness
    material = process.material
    if material.sorption_isotherm_kind != "gab":
        raise ValueError("the robustness variant spreads a GAB isotherm")
    sampler = qmc.LatinHypercube(d=5, seed=rob.seed)
    draws: list[IsothermDraw] = []
    rejected = 0
    while len(draws) < rob.draws:
        for unit in sampler.random(rob.draws):
            if len(draws) == rob.draws:
                break
            lo_c, hi_c = np.log(rob.c_constant_factor_log[0]), np.log(rob.c_constant_factor_log[1])
            factors = {
                "monolayer_factor": rob.monolayer_factor[0]
                + unit[0] * (rob.monolayer_factor[1] - rob.monolayer_factor[0]),
                "c_constant_factor": float(np.exp(lo_c + unit[1] * (hi_c - lo_c))),
                "k_constant": rob.k_constant[0] + unit[2] * (rob.k_constant[1] - rob.k_constant[0]),
                "c_enthalpy_factor": rob.enthalpy_factor[0]
                + unit[3] * (rob.enthalpy_factor[1] - rob.enthalpy_factor[0]),
                "k_enthalpy_factor": rob.enthalpy_factor[0]
                + unit[4] * (rob.enthalpy_factor[1] - rob.enthalpy_factor[0]),
            }
            assert material.gab_monolayer_moisture_kg_per_kg is not None
            assert material.gab_c_constant_at_reference_temperature is not None
            assert material.gab_c_enthalpy_j_per_mol is not None
            assert material.gab_k_enthalpy_j_per_mol is not None
            spread = dataclasses.replace(
                material,
                gab_monolayer_moisture_kg_per_kg=material.gab_monolayer_moisture_kg_per_kg
                * factors["monolayer_factor"],
                gab_c_constant_at_reference_temperature=material.gab_c_constant_at_reference_temperature
                * factors["c_constant_factor"],
                gab_k_constant_at_reference_temperature=factors["k_constant"],
                gab_c_enthalpy_j_per_mol=material.gab_c_enthalpy_j_per_mol
                * factors["c_enthalpy_factor"],
                gab_k_enthalpy_j_per_mol=material.gab_k_enthalpy_j_per_mol
                * factors["k_enthalpy_factor"],
            )
            config = dataclasses.replace(process, material=spread)
            try:
                validate_isotherm_against_critical_moisture(config)
            except ValueError:
                rejected += 1
                continue
            draws.append(
                IsothermDraw(
                    index=len(draws),
                    parameters={k: float(v) for k, v in factors.items()},
                    process=config,
                )
            )
    return draws, rejected

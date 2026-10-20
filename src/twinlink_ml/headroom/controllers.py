r"""Controllers R0-R3 of the dynamic headroom study and their two optimisers.

All controllers command (inlet temperature, gas velocity) piecewise constant on the control grid
(``control.interval_s``); spray rate (throughput) and nozzle pressure are not theirs. Each
controller works with a moisture margin ``m``: it plans for ``X <= limit - m``.

R0 fixed operating point
    The steady optimum at the NOMINAL disturbance with margin ``m``, held in every scenario.
R1 quasi-stationary RTO
    At every interval start the steady optimum at the CURRENT disturbance (perfect knowledge,
    including the realised inlet-temperature offset) and the current limit.
R2 dynamic optimum with perfect preview
    One multiple-shooting problem over the whole scenario with the true disturbance course
    (per-RK4-step means of the truth samples) and the limit course; applied open loop.
R3 MPC without disturbance preview
    Every interval the multiple-shooting problem over ``mpc_horizon_s`` from the TRUE state, the
    current disturbance and the current limit held constant over the horizon; the first command
    is applied (receding horizon).

Optimisers
----------
:class:`SteadyStateOptimizer`: :math:`\min P` s.t. :math:`\dot m_w = 0`, :math:`\dot T = 0` at
:math:`m_s = m_{s,set}`, :math:`X \le X_{lim} - m`, :math:`T_{bed} \le T_{max}`, action box.
:class:`DynamicOptimizer`: multiple shooting on :func:`~.symbolic.interval_function`; the path
condition at every RK4 step as a soft constraint with L1 slack (weight
``control.soft_constraint_weight_kwh``; an exact penalty, so where the hard problem is feasible
the slack is zero -- needed because R3 meets limits it cannot hold any more, e.g. right after a
product change). The bed-temperature limit is hard. Solver: IPOPT (CasADi), exact Hessian.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import casadi as cs
import numpy as np

from twinlink_ml.simulation import ProcessConfig

from .config import HeadroomStudyConfig
from .scenarios import Scenario
from .symbolic import DISTURBANCE_CHANNELS, interval_function, step_means, symbolic_derivative

__all__ = [
    "Controller",
    "DynamicOptimizer",
    "FixedPointController",
    "MpcController",
    "PreviewController",
    "QuasiStationaryController",
    "SteadyPoint",
    "SteadyStateOptimizer",
    "nominal_disturbance",
]

_IPOPT_OPTIONS = {
    "ipopt.print_level": 0,
    "print_time": False,
    "ipopt.tol": 1e-6,
    "ipopt.acceptable_tol": 1e-5,
    "ipopt.max_iter": 1000,
    "ipopt.sb": "yes",
}


def nominal_disturbance(study: HeadroomStudyConfig, process: ProcessConfig) -> np.ndarray:
    """Nominal disturbance vector in the order of :data:`DISTURBANCE_CHANNELS`."""
    dist = process.disturbances
    return np.array(
        [
            dist.ambient_temperature_celsius.nominal,
            dist.ambient_relative_humidity.nominal,
            study.line.nominal_solid_fraction,
            study.line.nominal_spray_rate_kg_per_h,
            0.0,
        ]
    )


def _action_box(process: ProcessConfig) -> tuple[np.ndarray, np.ndarray]:
    inlet, gas = process.actions.inlet_temperature_celsius, process.actions.gas_velocity_m_per_s
    return np.array([inlet.minimum, gas.minimum]), np.array([inlet.maximum, gas.maximum])


@dataclass(frozen=True)
class SteadyPoint:
    """Steady optimum: commands, reduced state (m_s, m_w, T_bed), total power in kW."""

    command: np.ndarray
    state: np.ndarray
    power_kw: float


class SteadyStateOptimizer:
    """Steady-state NLP, see the module docstring. Parameters: disturbance (5) and bound."""

    def __init__(self, study: HeadroomStudyConfig, process: ProcessConfig) -> None:
        self._process = process
        self._solid_mass = process.geometry.bed_mass_solid_setpoint_kg
        w = cs.SX.sym("w", 4)  # m_w, T_bed, T_in command, u_gas
        p = cs.SX.sym("p", 6)  # disturbance (5), moisture bound
        state = cs.vertcat(self._solid_mass, w[0], w[1])
        action = cs.vertcat(w[2] + p[4], w[3], p[3], study.line.nozzle_pressure_bar)
        derivative, power = symbolic_derivative(process, state, action, p[0:3])
        g = cs.vertcat(derivative[1], derivative[2], w[0] / self._solid_mass - p[5], w[1])
        self._solver = cs.nlpsol(
            "steady", "ipopt", {"x": w, "p": p, "f": power, "g": g}, _IPOPT_OPTIONS
        )
        low, high = _action_box(process)
        self._lbx = np.array([0.0, -20.0, low[0], low[1]])
        self._ubx = np.array([10.0, 200.0, high[0], high[1]])
        self._lbg = np.array([0.0, 0.0, -np.inf, -np.inf])
        self._ubg = np.array([0.0, 0.0, 0.0, study.line.bed_temperature_limit_celsius])
        self._guess = np.array([0.25, 45.0, 90.0, low[1]])
        self._power = cs.Function("power", [w, p], [power])

    def solve(self, disturbance: np.ndarray, moisture_bound: float) -> SteadyPoint:
        """Steady optimum for one disturbance vector and moisture bound ``limit - m``."""
        params = np.append(np.asarray(disturbance, dtype=np.float64), moisture_bound)
        solution = self._solver(
            x0=self._guess, p=params, lbx=self._lbx, ubx=self._ubx, lbg=self._lbg, ubg=self._ubg
        )
        if not self._solver.stats()["success"]:
            # one retry from the default guess before giving up
            solution = self._solver(
                x0=np.array([0.25, 45.0, 90.0, self._lbx[3]]),
                p=params,
                lbx=self._lbx,
                ubx=self._ubx,
                lbg=self._lbg,
                ubg=self._ubg,
            )
            if not self._solver.stats()["success"]:
                raise RuntimeError(
                    f"steady optimum failed: {self._solver.stats()['return_status']} at {params}"
                )
        w = np.asarray(solution["x"]).ravel()
        self._guess = w
        return SteadyPoint(
            command=w[2:4].copy(),
            state=np.array([self._solid_mass, w[0], w[1]]),
            power_kw=float(solution["f"]),
        )


class DynamicOptimizer:
    """Multiple-shooting NLP over ``n_intervals`` control intervals, see the module docstring.

    Decision variables are scaled (states by ``(2 kg, 0.3 kg, 50 K)``, commands by ``(100 K,
    1 m/s)``, slacks by ``1e-3`` kg/kg) so that IPOPT sees quantities of order one. The command
    change between intervals is bounded by ``max_rate_per_s * interval_s`` (the interval-level
    form of the rate limit; see :mod:`.symbolic`).
    """

    _STATE_SCALE = np.array([2.0, 0.3, 50.0])
    _COMMAND_SCALE = np.array([100.0, 1.0])
    _SLACK_SCALE = 1e-3

    def __init__(
        self,
        study: HeadroomStudyConfig,
        process: ProcessConfig,
        n_intervals: int,
        *,
        warm_start: bool = True,
        warm_mu_init: float = 1e-4,
        expression: str = "SX",
        guess_start: bool = False,
    ) -> None:
        control = study.control
        self.n_intervals = n_intervals
        self.n_steps = control.steps_per_interval
        interval = interval_function(
            process,
            interval_s=control.interval_s,
            step_s=control.optimizer_step_s,
            nozzle_pressure_bar=study.line.nozzle_pressure_bar,
            sample_time_s=study.scenarios.sample_time_s,
        )
        self._interval = interval
        n, steps, nd = n_intervals, self.n_steps, len(DISTURBANCE_CHANNELS)
        # SX inlines the whole horizon (fast evaluation, memory grows with the horizon): used
        # for the 2 h MPC problem. MX keeps one interval function in a map: used for the 24 h
        # preview problem, whose inlined SX Hessian would need gigabytes per process.
        sym = {"SX": cs.SX, "MX": cs.MX}[expression]
        sx = cs.DM(self._STATE_SCALE)
        su = cs.DM(self._COMMAND_SCALE)
        xs = sym.sym("X", 3, n + 1)
        us = sym.sym("U", 2, n)
        ss = sym.sym("S", 1, n * steps)
        x0 = sym.sym("x0", 3)
        u_prev = sym.sym("u_prev", 2)
        dist = sym.sym("D", nd, n * steps)
        bound = sym.sym("bound", 1, n * steps)
        x = cs.repmat(sx, 1, n + 1) * xs
        u = cs.repmat(su, 1, n) * us
        s = self._SLACK_SCALE * ss
        u_before = cs.horzcat(u_prev, u[:, : n - 1])
        x_end, moisture, temperature, energy = interval.map(n, "serial")(
            x[:, :n], u, u_before, dist
        )
        weight = control.soft_constraint_weight_kwh
        self._weight = weight
        objective = cs.sum2(energy) + weight * cs.sum2(s)
        rates = np.array(
            [
                process.actions.inlet_temperature_celsius.max_rate_per_s,
                process.actions.gas_velocity_m_per_s.max_rate_per_s,
            ]
        )
        self._max_change = rates * control.interval_s
        g = cs.vertcat(
            cs.vec((x[:, 0] - x0) / sx),
            cs.vec((x_end - x[:, 1:]) / cs.repmat(sx, 1, n)),
            cs.vec((moisture - bound - s) / self._SLACK_SCALE),
            cs.vec(temperature),
            cs.vec(u - u_before),
        )
        variables = cs.vertcat(cs.vec(xs), cs.vec(us), cs.vec(ss))
        params = cs.vertcat(x0, u_prev, cs.vec(dist), cs.vec(bound))
        nlp = {"x": variables, "p": params, "f": objective, "g": g}
        base_options = dict(_IPOPT_OPTIONS)
        if guess_start:
            # Start close to the given guess (small barrier, small push off the bounds): the
            # problem is non-convex (ramp model) and a default start can end in a worse local
            # optimum than the guess itself (seen on an undisturbed scenario, docs/24 sec. 4).
            base_options.update(
                {"ipopt.mu_init": 1e-4, "ipopt.bound_push": 1e-8, "ipopt.bound_frac": 1e-8}
            )
        self._cold_solver = cs.nlpsol("dynamic_cold", "ipopt", nlp, base_options)
        self._warm_solver = None
        if warm_start:
            warm_options = dict(base_options)
            warm_options["ipopt.warm_start_init_point"] = "yes"
            warm_options["ipopt.warm_start_bound_push"] = 1e-6
            warm_options["ipopt.warm_start_mult_bound_push"] = 1e-6
            warm_options["ipopt.mu_init"] = warm_mu_init
            self._warm_solver = cs.nlpsol("dynamic_warm", "ipopt", nlp, warm_options)
        n_x, n_u, n_s = 3 * (n + 1), 2 * n, n * steps
        self._slices = (slice(0, n_x), slice(n_x, n_x + n_u), slice(n_x + n_u, n_x + n_u + n_s))
        low, high = _action_box(process)
        state_low = np.tile(np.array([0.2, 0.0, -20.0]) / self._STATE_SCALE, n + 1)
        state_high = np.tile(np.array([20.0, 20.0, 200.0]) / self._STATE_SCALE, n + 1)
        self._lbx = np.concatenate(
            [state_low, np.tile(low / self._COMMAND_SCALE, n), np.zeros(n_s)]
        )
        self._ubx = np.concatenate(
            [state_high, np.tile(high / self._COMMAND_SCALE, n), np.full(n_s, np.inf)]
        )
        n_eq = 3 + 3 * n
        self._lbg = np.concatenate(
            [
                np.zeros(n_eq),
                np.full(n_s, -np.inf),
                np.full(n_s, -np.inf),
                np.tile(-self._max_change, n),
            ]
        )
        self._ubg = np.concatenate(
            [
                np.zeros(n_eq),
                np.zeros(n_s),
                np.full(n_s, study.line.bed_temperature_limit_celsius),
                np.tile(self._max_change, n),
            ]
        )
        self._g_blocks = (3, 3 * n, n_s, n_s, 2 * n)
        self._warm: dict[str, np.ndarray] | None = None
        self.last_stats: dict[str, object] = {}
        self.last_slack_max = 0.0

    def predict(
        self, x0: np.ndarray, u_prev: np.ndarray, commands: np.ndarray, dist: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """States ``(3, n + 1)``, moisture at every RK4 step and energy (kWh) under ``commands``."""
        states = np.empty((3, commands.shape[1] + 1))
        states[:, 0] = x0
        moisture = []
        energy = 0.0
        before = np.asarray(u_prev, dtype=np.float64)
        for k in range(commands.shape[1]):
            cols = dist[:, k * self.n_steps : (k + 1) * self.n_steps]
            out = self._interval(states[:, k], commands[:, k], before, cols)
            states[:, k + 1] = np.asarray(out[0]).ravel()
            moisture.append(np.asarray(out[1]).ravel())
            energy += float(out[3])
            before = commands[:, k]
        return states, np.concatenate(moisture), energy

    def penalised_objective(
        self,
        x0: np.ndarray,
        u_prev: np.ndarray,
        commands: np.ndarray,
        dist: np.ndarray,
        bound: np.ndarray,
    ) -> float:
        """Objective of the NLP (energy plus slack penalty) for given commands, in the model."""
        _, moisture, energy = self.predict(x0, u_prev, commands, dist)
        slack = np.maximum(moisture - np.asarray(bound).ravel(), 0.0)
        return energy + self._weight * float(slack.sum())

    def solve(
        self,
        x0: np.ndarray,
        u_prev: np.ndarray,
        dist: np.ndarray,
        bound: np.ndarray,
        *,
        guess_commands: np.ndarray | None = None,
    ) -> np.ndarray:
        """Optimal commands ``(2, n_intervals)``.

        ``dist``: ``(5, n_intervals * steps)`` disturbance per RK4 step; ``bound``: moisture bound
        ``limit - m`` at the end of every RK4 step. Warm start from the previous solution shifted
        by one interval (receding horizon) unless ``guess_commands`` is given or there is none.
        """
        n = self.n_intervals
        params = np.concatenate(
            [x0, u_prev, np.asarray(dist).ravel(order="F"), np.asarray(bound).ravel()]
        )
        if guess_commands is not None or self._warm is None:
            commands = (
                np.tile(u_prev[:, None], (1, n)) if guess_commands is None else guess_commands
            )
            states, _, _ = self.predict(x0, u_prev, commands, dist)
            init = {
                "x0": np.concatenate(
                    [
                        (states / self._STATE_SCALE[:, None]).ravel(order="F"),
                        (commands / self._COMMAND_SCALE[:, None]).ravel(order="F"),
                        np.zeros(n * self.n_steps),
                    ]
                )
            }
            solver = self._cold_solver
        else:
            init = self._shifted_warm_start(x0, u_prev, dist, np.asarray(bound).ravel())
            solver = self._warm_solver if self._warm_solver is not None else self._cold_solver
        solution = solver(
            p=params, lbx=self._lbx, ubx=self._ubx, lbg=self._lbg, ubg=self._ubg, **init
        )
        stats = solver.stats()
        if not stats["success"] and solver is not self._cold_solver:
            # warm start stalled: one cold retry from the shifted primal point
            solver = self._cold_solver
            solution = solver(
                p=params,
                lbx=self._lbx,
                ubx=self._ubx,
                lbg=self._lbg,
                ubg=self._ubg,
                x0=init["x0"],
            )
            stats = solver.stats()
        self.last_stats = {
            "success": bool(stats["success"]),
            "status": stats["return_status"],
            "iterations": int(stats["iter_count"]),
            "cold": solver is self._cold_solver,
        }
        if not stats["success"]:
            raise RuntimeError(f"dynamic optimum failed: {stats['return_status']}")
        x = np.asarray(solution["x"]).ravel()
        self._warm = {
            "x": x,
            "lam_x": np.asarray(solution["lam_x"]).ravel(),
            "lam_g": np.asarray(solution["lam_g"]).ravel(),
        }
        self.last_slack_max = float(np.max(x[self._slices[2]], initial=0.0)) * self._SLACK_SCALE
        self.last_objective = float(solution["f"])
        commands = x[self._slices[1]].reshape((2, n), order="F")
        return commands * self._COMMAND_SCALE[:, None]

    @staticmethod
    def _shift(block: np.ndarray, rows: int) -> np.ndarray:
        """Shift a column-major ``(rows, cols)`` block one column left, repeating the last."""
        matrix = block.reshape((rows, -1), order="F")
        return np.hstack([matrix[:, 1:], matrix[:, -1:]]).ravel(order="F")

    def _shift_nodes(self, block: np.ndarray) -> np.ndarray:
        steps = self.n_steps
        return np.concatenate([block[steps:], block[-steps:]])

    def _shifted_warm_start(
        self, x0: np.ndarray, u_prev: np.ndarray, dist: np.ndarray, bound: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Previous solution shifted by one interval, re-simulated under the new disturbance.

        Commands and multipliers are shifted; states and slacks are recomputed from the new
        initial state and disturbance so that the starting point is (nearly) primal feasible --
        otherwise a changed disturbance leaves the interior-point method far from feasibility.
        """
        assert self._warm is not None
        sl = self._slices

        def shift_variables(v: np.ndarray) -> np.ndarray:
            return np.concatenate(
                [self._shift(v[sl[0]], 3), self._shift(v[sl[1]], 2), self._shift_nodes(v[sl[2]])]
            )

        primal = shift_variables(self._warm["x"])
        commands = primal[sl[1]].reshape((2, -1), order="F") * self._COMMAND_SCALE[:, None]
        states, moisture, _ = self.predict(x0, u_prev, commands, dist)
        slack = np.maximum(moisture - bound, 0.0) / self._SLACK_SCALE
        primal[sl[0]] = (states / self._STATE_SCALE[:, None]).ravel(order="F")
        primal[sl[2]] = slack + 1e-6
        init = {"x0": primal}
        if self._warm_solver is None:
            return init
        init["lam_x0"] = shift_variables(self._warm["lam_x"])
        lam_g = self._warm["lam_g"]
        sizes = np.cumsum((0, *self._g_blocks))
        head, dyn, moist, temp, change = (
            lam_g[sizes[i] : sizes[i + 1]] for i in range(len(self._g_blocks))
        )
        init["lam_g0"] = np.concatenate(
            [
                head,
                self._shift(dyn, 3),
                self._shift_nodes(moist),
                self._shift_nodes(temp),
                self._shift(change, 2),
            ]
        )
        return init

    def reset(self) -> None:
        """Forget the warm start (new scenario)."""
        self._warm = None


class Controller(ABC):
    """A controller run on one scenario; the truth plant calls :meth:`command` per interval."""

    @abstractmethod
    def reset(self, scenario: Scenario, state: np.ndarray, command: np.ndarray) -> None:
        """Start a scenario from the true state ``(4,)`` with the last command ``(2,)``."""

    @abstractmethod
    def command(
        self,
        k: int,
        state: np.ndarray,
        last_command: np.ndarray,
        disturbance: np.ndarray,
        limit: float,
    ) -> np.ndarray:
        """Command ``(2,)`` for interval ``k`` from the current true state and disturbance."""


class FixedPointController(Controller):
    """R0."""

    def __init__(self, point: SteadyPoint) -> None:
        self._command = point.command

    def reset(self, scenario: Scenario, state: np.ndarray, command: np.ndarray) -> None:
        """Nothing to reset."""

    def command(
        self,
        k: int,  # noqa: ARG002 - shared interface
        state: np.ndarray,  # noqa: ARG002 - shared interface
        last_command: np.ndarray,  # noqa: ARG002 - shared interface
        disturbance: np.ndarray,  # noqa: ARG002 - shared interface
        limit: float,  # noqa: ARG002 - shared interface
    ) -> np.ndarray:
        """The fixed command."""
        return self._command


class QuasiStationaryController(Controller):
    """R1: steady optimum at the current disturbance, bound ``limit - margin``."""

    def __init__(self, optimizer: SteadyStateOptimizer, margin: float) -> None:
        self._optimizer = optimizer
        self._margin = margin
        self._cache: dict[tuple[float, ...], np.ndarray] = {}

    def reset(self, scenario: Scenario, state: np.ndarray, command: np.ndarray) -> None:
        """Nothing to reset (the cache is valid across scenarios)."""

    def command(
        self,
        k: int,  # noqa: ARG002 - shared interface
        state: np.ndarray,  # noqa: ARG002 - shared interface
        last_command: np.ndarray,  # noqa: ARG002 - shared interface
        disturbance: np.ndarray,
        limit: float,
    ) -> np.ndarray:
        """Steady optimum at ``disturbance``."""
        key = (*np.round(disturbance, 12), round(limit - self._margin, 12))
        if key not in self._cache:
            self._cache[key] = self._optimizer.solve(disturbance, limit - self._margin).command
        return self._cache[key]


class PreviewController(Controller):
    """R2: open-loop plan from one full-horizon solve with the true disturbance course."""

    def __init__(
        self,
        optimizer: DynamicOptimizer,
        margin: float,
        guess: QuasiStationaryController | None = None,
    ) -> None:
        self._optimizer = optimizer
        self._margin = margin
        self._guess = guess
        self._plan: np.ndarray | None = None
        self.solve_stats: dict[str, object] = {}

    def reset(self, scenario: Scenario, state: np.ndarray, command: np.ndarray) -> None:
        """Solve the whole scenario at once."""
        opt = self._optimizer
        n_samples = scenario.n_samples
        samples_per_step = n_samples // (opt.n_intervals * opt.n_steps)
        if samples_per_step * opt.n_intervals * opt.n_steps != n_samples:
            raise ValueError("scenario length does not match the optimiser horizon")
        dist = step_means(scenario.disturbances, samples_per_step)
        # limit at the END of each RK4 step (the node where the path condition is checked)
        limit = scenario.moisture_limit[samples_per_step - 1 :: samples_per_step]
        bound = limit - self._margin
        guess = None
        if self._guess is not None:
            spi = n_samples // opt.n_intervals
            guess = np.stack(
                [
                    self._guess.command(
                        k,
                        state,
                        command,
                        scenario.disturbances[:, k * spi],
                        float(scenario.moisture_limit[k * spi]),
                    )
                    for k in range(opt.n_intervals)
                ],
                axis=1,
            )
        opt.reset()
        x0 = state[[0, 1, 3]]
        try:
            self._plan = opt.solve(x0, command, dist, bound, guess_commands=guess)
            self.solve_stats = {**opt.last_stats, "slack_max": opt.last_slack_max}
            objective = opt.last_objective
        except RuntimeError as error:
            if guess is None:
                raise
            self._plan, objective = None, np.inf
            self.solve_stats = {"success": False, "status": str(error)}
        # Safeguard: R2 is the upper bound of the dynamic gain; it never takes a plan that its
        # own model rates worse than the R1 schedule (a worse local optimum of the non-convex
        # problem, or a failed solve). Counted in the record.
        self.solve_stats["fallback_to_guess"] = False
        if guess is not None:
            guess_objective = opt.penalised_objective(x0, command, guess, dist, bound)
            self.solve_stats["objective"] = objective
            self.solve_stats["guess_objective"] = guess_objective
            if guess_objective < objective:
                self._plan = guess
                self.solve_stats["fallback_to_guess"] = True

    def command(
        self,
        k: int,
        state: np.ndarray,  # noqa: ARG002 - shared interface
        last_command: np.ndarray,  # noqa: ARG002 - shared interface
        disturbance: np.ndarray,  # noqa: ARG002 - shared interface
        limit: float,  # noqa: ARG002 - shared interface
    ) -> np.ndarray:
        """Planned command of interval ``k``."""
        assert self._plan is not None
        return self._plan[:, k]


class MpcController(Controller):
    """R3: receding horizon, disturbance and limit held constant over the horizon."""

    def __init__(self, optimizer: DynamicOptimizer, margin: float) -> None:
        self._optimizer = optimizer
        self._margin = margin
        self.max_slack = 0.0

    @property
    def optimizer(self) -> DynamicOptimizer:
        """The receding-horizon optimiser (for its solver statistics)."""
        return self._optimizer

    def reset(self, scenario: Scenario, state: np.ndarray, command: np.ndarray) -> None:  # noqa: ARG002 - shared interface
        """Forget the warm start."""
        self._optimizer.reset()
        self.max_slack = 0.0

    def command(
        self,
        k: int,  # noqa: ARG002 - shared interface
        state: np.ndarray,
        last_command: np.ndarray,
        disturbance: np.ndarray,
        limit: float,
    ) -> np.ndarray:
        """First command of the horizon problem."""
        opt = self._optimizer
        nodes = opt.n_intervals * opt.n_steps
        dist = np.tile(np.asarray(disturbance)[:, None], (1, nodes))
        bound = np.full(nodes, limit - self._margin)
        plan = opt.solve(state[[0, 1, 3]], last_command, dist, bound)
        self.max_slack = max(self.max_slack, opt.last_slack_max)
        return plan[:, 0]

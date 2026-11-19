r"""Dynamic programme with the spray rate as a third control and a priced objective.

Same model as the truth and as the headroom study: the interval function below composes
:func:`twinlink_ml.headroom.symbolic.symbolic_derivative` (unchanged) exactly as
:func:`twinlink_ml.headroom.symbolic.interval_function` does; the only differences are

* the spray rate is a command with its own ramp (``ActionBound.max_rate_per_s`` of the spray
  rate, :func:`~twinlink_ml.headroom.symbolic.ramp_mean`) instead of a disturbance column, and
* a fifth output, the suspension mass sprayed in the interval (kg), for the production target.

``tests/test_load_following.py`` pins this function against the headroom one (spray rate held)
and the optimiser against the headroom optimiser (spray rate pinned, constant price).

Programme (:class:`LoadOptimizer`), multiple shooting over ``n`` control intervals:

.. math::
    \min \sum_k \pi_k E_k + w \sum_j s_j \quad \text{s.t.} \quad
    X_j \le b_j + s_j,\; T_j \le T_{max},\; |u_k - u_{k-1}| \le r\,\Delta t,\;
    C_{k+1} = C_k + a_k M_k,\; C_0 = 0,\; C_n = M_{target}

with the relative price :math:`\pi_k` (tariff over a reference tariff, so that with a constant
price the objective is the energy in kWh and the slack weight keeps its meaning of the headroom
study), the sprayed mass :math:`M_k` and the indicator :math:`a_k` of the intervals that count for
the production target. The target is written as a chain over the cumulative mass :math:`C_k`
(a variable per interval) and not as one sum over all intervals: a single dense row would couple
every command with every other one in the Jacobian and cost a derivative sweep per command
(measured: the Jacobian took eight times as long at 60 intervals). Bounds of the commands are
passed per solve (band of the spray rate, fixed
commands of the tail, pinned commands in tests).
"""

from __future__ import annotations

import casadi as cs
import numpy as np

from twinlink_ml.headroom.config import HeadroomStudyConfig
from twinlink_ml.headroom.symbolic import DISTURBANCE_CHANNELS, ramp_mean, symbolic_derivative
from twinlink_ml.simulation import ProcessConfig

__all__ = ["LoadOptimizer", "command_rates", "load_interval_function"]

_SECONDS_PER_HOUR = 3600.0

_IPOPT_OPTIONS = {
    "ipopt.print_level": 0,
    "print_time": False,
    "ipopt.tol": 1e-6,
    "ipopt.acceptable_tol": 1e-5,
    "ipopt.max_iter": 1000,
    "ipopt.sb": "yes",
}


def command_rates(process: ProcessConfig) -> np.ndarray:
    """Rate limits of (inlet temperature, gas velocity, spray rate) per second."""
    actions = process.actions
    return np.array(
        [
            actions.inlet_temperature_celsius.max_rate_per_s,
            actions.gas_velocity_m_per_s.max_rate_per_s,
            actions.spray_rate_kg_per_h.max_rate_per_s,
        ]
    )


def load_interval_function(
    config: ProcessConfig,
    *,
    interval_s: float,
    step_s: float,
    nozzle_pressure_bar: float,
    sample_time_s: float,
) -> cs.Function:
    """``(x0[3], u[3], u_prev[3], D[5, n]) -> (x_end, X[n], T[n], E_kwh, M_kg)``.

    ``u`` = (commanded inlet temperature, gas velocity, spray rate); ``D`` as in the headroom
    study (:data:`~twinlink_ml.headroom.symbolic.DISTURBANCE_CHANNELS`), of which the spray-rate
    row is NOT used here. Ramp and time shift as in the headroom interval function.
    """
    n_steps = round(interval_s / step_s)
    if abs(n_steps * step_s - interval_s) > 1e-9:
        raise ValueError("interval_s must be a multiple of step_s")
    x0 = cs.SX.sym("x0", 3)
    u = cs.SX.sym("u", 3)
    u_prev = cs.SX.sym("u_prev", 3)
    rates = command_rates(config)
    shift = 0.5 * sample_time_s
    disturbances = cs.SX.sym("D", len(DISTURBANCE_CHANNELS), n_steps)

    xs = cs.SX.sym("xs", 3)
    us = cs.SX.sym("us", 4)
    ds = cs.SX.sym("ds", 3)
    derivative, power = symbolic_derivative(config, xs, us, ds)
    rhs = cs.Function("rhs", [xs, us, ds], [derivative, power])

    x = x0
    moisture_nodes = []
    temperature_nodes = []
    energy = 0.0
    sprayed = 0.0
    for j in range(n_steps):
        column = disturbances[:, j]
        start, end = j * step_s + shift, (j + 1) * step_s + shift
        inlet = u_prev[0] + ramp_mean(u[0] - u_prev[0], float(rates[0]), start, end)
        gas = u_prev[1] + ramp_mean(u[1] - u_prev[1], float(rates[1]), start, end)
        spray = u_prev[2] + ramp_mean(u[2] - u_prev[2], float(rates[2]), start, end)
        action = cs.vertcat(inlet + column[4], gas, spray, nozzle_pressure_bar)
        weather = column[0:3]
        k1, p1 = rhs(x, action, weather)
        k2, _ = rhs(x + 0.5 * step_s * k1, action, weather)
        k3, _ = rhs(x + 0.5 * step_s * k2, action, weather)
        k4, _ = rhs(x + step_s * k3, action, weather)
        x = x + step_s / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        energy = energy + p1 * step_s / _SECONDS_PER_HOUR
        sprayed = sprayed + spray * step_s / _SECONDS_PER_HOUR
        moisture_nodes.append(x[1] / x[0])
        temperature_nodes.append(x[2])
    return cs.Function(
        "load_interval",
        [x0, u, u_prev, disturbances],
        [x, cs.horzcat(*moisture_nodes), cs.horzcat(*temperature_nodes), energy, sprayed],
        ["x0", "u", "u_prev", "D"],
        ["x_end", "moisture", "bed_temperature", "energy_kwh", "sprayed_kg"],
    )


class LoadOptimizer:
    """Multiple-shooting programme over ``n_intervals`` control intervals (module docstring).

    Scaling as in the headroom optimiser (states ``(2 kg, 0.3 kg, 50 K)``, commands ``(100 K,
    1 m/s, 1 kg/h)``, slacks ``1e-3`` kg/kg). The start is close to the given guess (small
    barrier parameter and bound push): the problem is non-convex, see ``docs/24`` section 4.
    """

    _STATE_SCALE = np.array([2.0, 0.3, 50.0])
    _COMMAND_SCALE = np.array([100.0, 1.0, 1.0])
    _SLACK_SCALE = 1e-3
    _MASS_SCALE = 10.0

    def __init__(
        self,
        study: HeadroomStudyConfig,
        process: ProcessConfig,
        n_intervals: int,
        *,
        expression: str = "MX",
        guess_start: bool = True,
    ) -> None:
        control = study.control
        self.n_intervals = n_intervals
        self.n_steps = control.steps_per_interval
        interval = load_interval_function(
            process,
            interval_s=control.interval_s,
            step_s=control.optimizer_step_s,
            nozzle_pressure_bar=study.line.nozzle_pressure_bar,
            sample_time_s=study.scenarios.sample_time_s,
        )
        self._interval = interval
        n, steps, nd = n_intervals, self.n_steps, len(DISTURBANCE_CHANNELS)
        sym = {"SX": cs.SX, "MX": cs.MX}[expression]
        sx = cs.DM(self._STATE_SCALE)
        su = cs.DM(self._COMMAND_SCALE)
        xs = sym.sym("X", 3, n + 1)
        us = sym.sym("U", 3, n)
        ss = sym.sym("S", 1, n * steps)
        cs_ = sym.sym("C", 1, n + 1)
        x0 = sym.sym("x0", 3)
        u_prev = sym.sym("u_prev", 3)
        dist = sym.sym("D", nd, n * steps)
        bound = sym.sym("bound", 1, n * steps)
        price = sym.sym("price", 1, n)
        counted = sym.sym("counted", 1, n)
        x = cs.repmat(sx, 1, n + 1) * xs
        u = cs.repmat(su, 1, n) * us
        s = self._SLACK_SCALE * ss
        cumulative = self._MASS_SCALE * cs_
        u_before = cs.horzcat(u_prev, u[:, : n - 1])
        x_end, moisture, temperature, energy, sprayed = interval.map(n, "serial")(
            x[:, :n], u, u_before, dist
        )
        self._weight = control.soft_constraint_weight_kwh
        objective = cs.sum2(price * energy) + self._weight * cs.sum2(s)
        self._max_change = command_rates(process) * control.interval_s
        g = cs.vertcat(
            cs.vec((x[:, 0] - x0) / sx),
            cs.vec((x_end - x[:, 1:]) / cs.repmat(sx, 1, n)),
            cs.vec((moisture - bound - s) / self._SLACK_SCALE),
            cs.vec(temperature),
            cs.vec(u - u_before),
            cs.vec(cumulative[:, 1:] - cumulative[:, :n] - counted * sprayed) / self._MASS_SCALE,
        )
        variables = cs.vertcat(cs.vec(xs), cs.vec(us), cs.vec(ss), cs.vec(cs_))
        params = cs.vertcat(x0, u_prev, cs.vec(dist), cs.vec(bound), cs.vec(price), cs.vec(counted))
        nlp = {"x": variables, "p": params, "f": objective, "g": g}
        options = dict(_IPOPT_OPTIONS)
        if guess_start:
            options.update(
                {"ipopt.mu_init": 1e-4, "ipopt.bound_push": 1e-8, "ipopt.bound_frac": 1e-8}
            )
        self._solver = cs.nlpsol("load_dynamic", "ipopt", nlp, options)
        n_x, n_u, n_s = 3 * (n + 1), 3 * n, n * steps
        self._slices = (slice(0, n_x), slice(n_x, n_x + n_u), slice(n_x + n_u, n_x + n_u + n_s))
        self._state_low = np.tile(np.array([0.2, 0.0, -20.0]) / self._STATE_SCALE, n + 1)
        self._state_high = np.tile(np.array([20.0, 20.0, 200.0]) / self._STATE_SCALE, n + 1)
        n_eq = 3 + 3 * n
        self._lbg = np.concatenate(
            [
                np.zeros(n_eq),
                np.full(n_s, -np.inf),
                np.full(n_s, -np.inf),
                np.tile(-self._max_change, n),
                np.zeros(n),
            ]
        )
        self._ubg = np.concatenate(
            [
                np.zeros(n_eq),
                np.zeros(n_s),
                np.full(n_s, study.line.bed_temperature_limit_celsius),
                np.tile(self._max_change, n),
                np.zeros(n),
            ]
        )
        self.last_stats: dict[str, object] = {}
        self.last_slack_max = 0.0
        self.last_objective = float("nan")

    # ------------------------------------------------------------------ model evaluation

    def predict(
        self, x0: np.ndarray, u_prev: np.ndarray, commands: np.ndarray, dist: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prediction of the optimisation model under ``commands`` ``(3, n)``.

        States ``(3, n + 1)``, moisture per RK4 step, energy (kWh) and sprayed mass (kg) per
        interval.
        """
        n = commands.shape[1]
        states = np.empty((3, n + 1))
        states[:, 0] = x0
        moisture = []
        energy = np.empty(n)
        sprayed = np.empty(n)
        before = np.asarray(u_prev, dtype=np.float64)
        for k in range(n):
            cols = dist[:, k * self.n_steps : (k + 1) * self.n_steps]
            out = self._interval(states[:, k], commands[:, k], before, cols)
            states[:, k + 1] = np.asarray(out[0]).ravel()
            moisture.append(np.asarray(out[1]).ravel())
            energy[k] = float(out[3])
            sprayed[k] = float(out[4])
            before = commands[:, k]
        return states, np.concatenate(moisture), energy, sprayed

    def penalised_objective(
        self,
        x0: np.ndarray,
        u_prev: np.ndarray,
        commands: np.ndarray,
        dist: np.ndarray,
        bound: np.ndarray,
        price: np.ndarray,
    ) -> float:
        """Objective of the programme (priced energy plus slack penalty) for given commands."""
        _, moisture, energy, _ = self.predict(x0, u_prev, commands, dist)
        slack = np.maximum(moisture - np.asarray(bound).ravel(), 0.0)
        return float(np.sum(np.asarray(price) * energy)) + self._weight * float(slack.sum())

    # ------------------------------------------------------------------ solve

    def solve(
        self,
        x0: np.ndarray,
        u_prev: np.ndarray,
        dist: np.ndarray,
        bound: np.ndarray,
        price: np.ndarray,
        *,
        command_low: np.ndarray,
        command_high: np.ndarray,
        guess_commands: np.ndarray,
        counted: np.ndarray | None = None,
        target_kg: float | None = None,
    ) -> np.ndarray:
        """Optimal commands ``(3, n_intervals)``.

        ``dist`` ``(5, n * steps)`` and ``bound`` ``(n * steps,)`` per RK4 step; ``price`` ``(n,)``
        relative price per interval; ``command_low`` / ``command_high`` ``(3, n)`` bounds of the
        commands (equal entries fix a command); ``counted`` ``(n,)`` and ``target_kg``: the
        production target (omitted: no target). Start at ``guess_commands`` ``(3, n)``.
        """
        n = self.n_intervals
        with_target = target_kg is not None
        counted_row = np.zeros(n) if counted is None else np.asarray(counted, dtype=np.float64)
        params = np.concatenate(
            [
                x0,
                u_prev,
                np.asarray(dist).ravel(order="F"),
                np.asarray(bound).ravel(),
                np.asarray(price, dtype=np.float64).ravel(),
                counted_row,
            ]
        )
        guess = np.clip(guess_commands, command_low, command_high)
        states, _, _, sprayed = self.predict(x0, u_prev, guess, dist)
        cumulative = np.concatenate([[0.0], np.cumsum(counted_row * sprayed)]) / self._MASS_SCALE
        init = np.concatenate(
            [
                (states / self._STATE_SCALE[:, None]).ravel(order="F"),
                (guess / self._COMMAND_SCALE[:, None]).ravel(order="F"),
                np.zeros(n * self.n_steps),
                cumulative,
            ]
        )
        scale = self._COMMAND_SCALE[:, None]
        # cumulative mass: starts at zero; its end is the target (free without a target)
        mass_low = np.full(n + 1, -np.inf)
        mass_high = np.full(n + 1, np.inf)
        mass_low[0] = mass_high[0] = 0.0
        if with_target:
            mass_low[n] = mass_high[n] = float(target_kg) / self._MASS_SCALE
        lbx = np.concatenate(
            [
                self._state_low,
                (command_low / scale).ravel(order="F"),
                np.zeros(n * self.n_steps),
                mass_low,
            ]
        )
        ubx = np.concatenate(
            [
                self._state_high,
                (command_high / scale).ravel(order="F"),
                np.full(n * self.n_steps, np.inf),
                mass_high,
            ]
        )
        solution = self._solver(x0=init, p=params, lbx=lbx, ubx=ubx, lbg=self._lbg, ubg=self._ubg)
        stats = self._solver.stats()
        self.last_stats = {
            "success": bool(stats["success"]),
            "status": stats["return_status"],
            "iterations": int(stats["iter_count"]),
        }
        if not stats["success"]:
            raise RuntimeError(f"load programme failed: {stats['return_status']}")
        x = np.asarray(solution["x"]).ravel()
        self.last_slack_max = float(np.max(x[self._slices[2]], initial=0.0)) * self._SLACK_SCALE
        self.last_objective = float(solution["f"])
        return x[self._slices[1]].reshape((3, n), order="F") * scale

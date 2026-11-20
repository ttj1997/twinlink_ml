r"""Family D: the sequence model in the loop, and its control C3-15 on the true model.

D-gru
    Throughput from the schedule on the learned steady map of M-gru (the unchanged
    :class:`~twinlink_ml.load_following.controllers.Schedule`). Every control interval the inlet
    temperature and the gas velocity come from an optimisation through the rollout of the
    composite GRU over its horizon (36 blocks of 25 s = three control intervals), started from the
    measured history of the last 32 blocks (:class:`MeasuredHistory`).

C3-15
    The same throughput law on the TRUE steady map and the lower layer of C3 (the unchanged
    :class:`~twinlink_ml.headroom.controllers.DynamicOptimizer`) with its horizon cut to the same
    three intervals (:class:`ScheduleMpc`). It separates the cost of the short horizon from the
    cost of the learned model.

Horizon problem of D-gru (:class:`HorizonOptimizer`), commands :math:`u_i` piecewise constant
over the intervals :math:`i`, one slack :math:`s_h \ge 0` per block :math:`h`:

.. math:: \min \sum_h \hat P_h \,\Delta t + w \sum_h s_h \quad 	ext{s.t.}\quad
          \hat X_h \le X_{limit} - m + s_h,\ \hat T_{bed,h} \le T_{max},\ u_i \in 	ext{box}

with the predictions :math:`\hat P, \hat X, \hat T_{bed}` of the rollout at the block ends, the
ambient state held and the spray rate at its planned level. :math:`w` is the slack weight of the
headroom config per node and kg/kg, as in the lower layer of C3 (exact penalty).
Solved by SLSQP with finite-difference gradients
from one batched rollout, so the same optimiser runs on any :class:`RolloutModel` (the test puts
the truth behind the interface). Rate limits of the commands are not part of the prediction (the
truth applies them); the first command of the solution is applied.

:class:`NumpyGruRollout` evaluates the trained networks with NumPy from their weights (the history
is encoded once per decision, the decoder runs on the batch); it equals the PyTorch model to
rounding (tested).
"""

from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from twinlink_ml.headroom.controllers import DynamicOptimizer
from twinlink_ml.load_following.config import LoadFollowingConfig
from twinlink_ml.load_following.controllers import LoadController, Schedule
from twinlink_ml.load_following.dynamic import command_rates
from twinlink_ml.load_following.market import DayScenario
from twinlink_ml.simulation import GreyBoxFluidBedGranulator, ProcessConfig
from twinlink_ml.surrogate_env.model import EXOGENOUS_NAMES, TARGET_NAMES

__all__ = [
    "HorizonMpc",
    "HorizonOptimizer",
    "InlineTargets",
    "MeasuredHistory",
    "NumpyGruRollout",
    "RolloutModel",
    "ScheduleMpc",
    "TruthRollout",
    "gru_step",
]

_N_EXO = len(EXOGENOUS_NAMES)
_N_TARGETS = len(TARGET_NAMES)


class RolloutModel(ABC):
    """Prediction of (power, moisture, bed temperature) over a horizon of blocks."""

    history_length: int
    horizon: int

    @abstractmethod
    def prepare(self, history: np.ndarray) -> Any:
        """Encode the history ``(L, 10)`` (7 exogenous channels, 3 targets, physical units)."""

    @abstractmethod
    def rollout(self, context: Any, future_exogenous: np.ndarray) -> np.ndarray:
        """Targets ``(B, H, 3)`` for future exogenous inputs ``(B, H, 7)`` (physical units)."""


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def gru_step(
    x: np.ndarray,
    h: np.ndarray,
    w_ih: np.ndarray,
    w_hh: np.ndarray,
    b_ih: np.ndarray,
    b_hh: np.ndarray,
) -> np.ndarray:
    """One GRU step ``(B, n_in), (B, H) -> (B, H)``; gates and equations of ``torch.nn.GRU``."""
    size = h.shape[1]
    gi = x @ w_ih.T + b_ih
    gh = h @ w_hh.T + b_hh
    r = _sigmoid(gi[:, :size] + gh[:, :size])
    z = _sigmoid(gi[:, size : 2 * size] + gh[:, size : 2 * size])
    n = np.tanh(gi[:, 2 * size :] + r * gh[:, 2 * size :])
    return np.asarray((1.0 - z) * n + z * h)


@dataclasses.dataclass(frozen=True, eq=False)
class _Head:
    w_ih: np.ndarray
    w_hh: np.ndarray
    b_ih: np.ndarray
    b_hh: np.ndarray
    w_out: np.ndarray
    b_out: np.ndarray

    def step(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """One step of this head's GRU."""
        return gru_step(x, h, self.w_ih, self.w_hh, self.b_ih, self.b_hh)


class NumpyGruRollout(RolloutModel):
    """Composite GRU of the Befund 94 pipeline evaluated from its weights."""

    def __init__(
        self, state_dicts: Mapping[str, Mapping[str, Any]], metadata: Mapping[str, Any]
    ) -> None:
        if tuple(metadata["exogenous_names"]) != EXOGENOUS_NAMES:
            raise ValueError(f"unexpected exogenous channels {metadata['exogenous_names']}")
        if tuple(metadata["target_names"]) != TARGET_NAMES:
            raise ValueError(f"unexpected targets {metadata['target_names']}")
        model_config = metadata["config"]
        if model_config["core"] != "gru" or int(model_config["num_layers"]) != 1:
            raise ValueError("only a single-layer GRU core is supported")
        self._heads = []
        for target in TARGET_NAMES:
            weights = {k: np.asarray(v, dtype=np.float64) for k, v in state_dicts[target].items()}
            self._heads.append(
                _Head(
                    w_ih=weights["core.weight_ih_l0"],
                    w_hh=weights["core.weight_hh_l0"],
                    b_ih=weights["core.bias_ih_l0"],
                    b_hh=weights["core.bias_hh_l0"],
                    w_out=weights["head.weight"],
                    b_out=weights["head.bias"],
                )
            )
        self._in_mean = np.asarray(metadata["input_scaler"].mean, dtype=np.float64)
        self._in_scale = np.asarray(metadata["input_scaler"].scale, dtype=np.float64)
        self._out_mean = np.asarray(metadata["target_scaler"].mean, dtype=np.float64)
        self._out_scale = np.asarray(metadata["target_scaler"].scale, dtype=np.float64)
        self.history_length = int(metadata["data_spec"]["data/history_length"])
        self.horizon = int(metadata["data_spec"]["data/horizon"])

    @classmethod
    def load(cls, path: Path) -> NumpyGruRollout:
        """From a checkpoint written by ``twinlink_ml.utils.checkpoints.save_checkpoint``."""
        from twinlink_ml.utils.checkpoints import load_checkpoint

        model, metadata = load_checkpoint(path)
        return cls(model.state_dict(), metadata)

    def prepare(self, history: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Hidden state and last (standardised) target of every head after the history."""
        standardised = (np.asarray(history, dtype=np.float64) - self._in_mean) / self._in_scale
        context = []
        for k, head in enumerate(self._heads):
            rows = standardised[:, [*range(_N_EXO), _N_EXO + k]]
            hidden = np.zeros((1, head.w_hh.shape[1]))
            for row in rows:
                hidden = head.step(row[None, :], hidden)
            context.append((hidden, rows[-1:, _N_EXO:]))
        return context

    def rollout(self, context: Any, future_exogenous: np.ndarray) -> np.ndarray:
        future = np.asarray(future_exogenous, dtype=np.float64)
        batch, horizon, _ = future.shape
        exogenous = (future - self._in_mean[:_N_EXO]) / self._in_scale[:_N_EXO]
        out = np.empty((batch, horizon, _N_TARGETS))
        for k, head in enumerate(self._heads):
            hidden = np.repeat(context[k][0], batch, axis=0)
            last = np.repeat(context[k][1], batch, axis=0)
            for step in range(horizon):
                hidden = head.step(np.concatenate([exogenous[:, step, :], last], axis=1), hidden)
                last = last + hidden @ head.w_out.T + head.b_out
                out[:, step, k] = last[:, 0]
        return np.asarray(self._out_mean + self._out_scale * out)


class TruthRollout(RolloutModel):
    """The grey-box model behind the rollout interface (test of the optimiser, not an arm).

    Simulates from the true state handed over by :meth:`set_state`; the history is ignored. The
    inputs of a block are applied without the rate limits, as in the prediction of D-gru.
    """

    def __init__(
        self, process: ProcessConfig, *, block_s: float, history_length: int, horizon: int
    ) -> None:
        self._sim = GreyBoxFluidBedGranulator(process)
        self._samples = round(block_s / process.numerics.sample_time_s)
        self.history_length = history_length
        self.horizon = horizon
        self._state: np.ndarray | None = None

    def set_state(self, state: np.ndarray) -> None:
        """True state ``(4,)`` at the decision instant."""
        self._state = np.asarray(state, dtype=np.float64).copy()

    def prepare(self, history: np.ndarray) -> None:  # noqa: ARG002 - shared interface
        """Nothing to encode."""
        return None

    def rollout(self, context: Any, future_exogenous: np.ndarray) -> np.ndarray:  # noqa: ARG002 - shared interface
        if self._state is None:
            raise RuntimeError("set_state was not called")
        future = np.asarray(future_exogenous, dtype=np.float64)
        batch, horizon, _ = future.shape
        state = np.repeat(self._state[None, :], batch, axis=0)
        out = np.empty((batch, horizon, _N_TARGETS))
        for step in range(horizon):
            for _ in range(self._samples):
                result = self._sim.step(state, future[:, step, 0:4], future[:, step, 4:7])
                state = np.asarray(result.state)
            diagnostics = result.diagnostics
            out[:, step, 0] = np.asarray(diagnostics["power_total_kw"])
            out[:, step, 1] = np.asarray(diagnostics["particle_moisture_kg_per_kg"])
            out[:, step, 2] = state[:, 3]
        return out


class InlineTargets:
    """Inline measurement of the three targets through the existing sensor model.

    First-order lag and noise of the measurement config (enabled), at the block interval; the
    random numbers come from one generator per day-run.
    """

    def __init__(self, process: ProcessConfig, block_s: float, seed: int) -> None:
        numerics = dataclasses.replace(process.numerics, sample_time_s=block_s)
        measurement = dataclasses.replace(process.measurement, enabled=True)
        self._sensor = GreyBoxFluidBedGranulator(
            dataclasses.replace(process, numerics=numerics, measurement=measurement)
        )
        self._rng = np.random.default_rng(seed)

    def __call__(self, targets: np.ndarray) -> np.ndarray:
        """Measured ``(power, moisture, bed temperature)`` of one block."""
        # grey-box output order: energy, bed temperature, moisture, median diameter (unused)
        outputs = np.array([[targets[0], targets[2], targets[1], 0.0]])
        measured = np.asarray(self._sensor.measure(outputs, self._rng))[0]
        return np.array([measured[0], measured[2], measured[1]])


class MeasuredHistory:
    """The last ``length`` blocks: exogenous block means and measured targets."""

    def __init__(self, length: int, sensor: InlineTargets | None = None) -> None:
        self._length = length
        self._sensor = sensor
        self._rows = np.empty((0, _N_EXO + _N_TARGETS))

    def reset(self, exogenous: np.ndarray, targets: np.ndarray) -> None:
        """Steady pre-history: the plant ran at these inputs and outputs before the day."""
        measured = targets if self._sensor is None else self._sensor(np.asarray(targets))
        row = np.concatenate([np.asarray(exogenous, np.float64), measured])
        self._rows = np.repeat(row[None, :], self._length, axis=0)

    def append(self, exogenous: np.ndarray, targets: np.ndarray) -> None:
        """Blocks of one interval: ``(m, 7)`` block means, ``(m, 3)`` true block-end targets."""
        measured = np.asarray(targets, dtype=np.float64)
        if self._sensor is not None:
            measured = np.stack([self._sensor(row) for row in measured])
        rows = np.concatenate([np.asarray(exogenous, np.float64), measured], axis=1)
        self._rows = np.concatenate([self._rows, rows], axis=0)[-self._length :]

    def window(self) -> np.ndarray:
        """History ``(length, 10)``."""
        if len(self._rows) != self._length:
            raise RuntimeError("the history was not reset")
        return self._rows


class HorizonOptimizer:
    """The horizon problem of the module docstring on a :class:`RolloutModel`."""

    _SLACK_SCALE = 0.01  # kg/kg per unit of the scaled slack variable
    _BED_SCALE = 10.0  # K per unit of the scaled bed-temperature constraint

    def __init__(
        self,
        model: RolloutModel,
        process: ProcessConfig,
        *,
        n_intervals: int,
        blocks_per_interval: int,
        block_s: float,
        bed_limit: float,
        slack_weight_kwh: float,
        fd_step: float,
        max_iterations: int,
    ) -> None:
        if n_intervals * blocks_per_interval != model.horizon:
            raise ValueError("the intervals must fill the horizon of the model")
        inlet, gas = process.actions.inlet_temperature_celsius, process.actions.gas_velocity_m_per_s
        self._model = model
        self.n_intervals = n_intervals
        self._blocks = blocks_per_interval
        self._centre = np.array(
            [0.5 * (inlet.minimum + inlet.maximum), 0.5 * (gas.minimum + gas.maximum)]
        )
        self._half = np.array(
            [0.5 * (inlet.maximum - inlet.minimum), 0.5 * (gas.maximum - gas.minimum)]
        )
        self._hours = block_s / 3600.0
        self._bed_limit = bed_limit
        self._weight = slack_weight_kwh
        self._fd = fd_step
        self._max_iterations = max_iterations
        self.last_stats: dict[str, Any] = {}

    def _physical(self, scaled: np.ndarray) -> np.ndarray:
        """Scaled commands ``(B, 2 n)`` (interval-major) -> ``(B, n, 2)`` physical."""
        shaped = scaled.reshape(scaled.shape[0], self.n_intervals, 2)
        return np.asarray(self._centre + self._half * shaped)

    def _predict(self, context: Any, base: np.ndarray, scaled: np.ndarray) -> np.ndarray:
        future = np.repeat(base[None, :, :], scaled.shape[0], axis=0)
        future[:, :, 0:2] = np.repeat(self._physical(scaled), self._blocks, axis=1)
        return self._model.rollout(context, future)

    def solve(
        self, context: Any, base_exogenous: np.ndarray, bound: float, guess: np.ndarray
    ) -> np.ndarray:
        """Commands ``(2, n_intervals)``; raises ``RuntimeError`` if no acceptable point is found.

        ``base_exogenous (H, 7)``: the future inputs with the commands still to be filled in;
        ``guess (2, n_intervals)``: start of the search.
        """
        n_var = 2 * self.n_intervals
        horizon = self._model.horizon
        cache: dict[bytes, tuple[float, np.ndarray, np.ndarray, np.ndarray]] = {}
        calls = 0

        def evaluate(x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
            nonlocal calls
            key = x.tobytes()
            if key in cache:
                return cache[key]
            calls += 1
            rows = np.repeat(x[None, :n_var], n_var + 1, axis=0)
            # one-sided differences, turned inwards at the upper bound of the box
            steps = np.where(x[:n_var] + self._fd > 1.0, -self._fd, self._fd)
            rows[np.arange(1, n_var + 1), np.arange(n_var)] += steps
            predicted = self._predict(context, base_exogenous, rows)
            slack = x[n_var:]
            energy = predicted[:, :, 0].sum(axis=1) * self._hours
            constraint = np.concatenate(
                [
                    (bound - predicted[:, :, 1]) / self._SLACK_SCALE + slack,
                    (self._bed_limit - predicted[:, :, 2]) / self._BED_SCALE,
                ],
                axis=1,
            )
            objective = float(energy[0] + self._weight * self._SLACK_SCALE * slack.sum())
            gradient = np.empty(n_var + horizon)
            gradient[:n_var] = (energy[1:] - energy[0]) / steps
            gradient[n_var:] = self._weight * self._SLACK_SCALE
            jacobian = np.zeros((2 * horizon, n_var + horizon))
            jacobian[:, :n_var] = ((constraint[1:] - constraint[0]) / steps[:, None]).T
            jacobian[:horizon, n_var:] = np.eye(horizon)
            cache.clear()
            cache[key] = (objective, gradient, constraint[0], jacobian)
            return cache[key]

        start = np.zeros(n_var + horizon)
        start[:n_var] = np.clip(
            ((np.asarray(guess, np.float64).T - self._centre) / self._half).ravel(), -1.0, 1.0
        )
        moisture = self._predict(context, base_exogenous, start[None, :n_var])[0, :, 1]
        start[n_var:] = np.maximum(moisture - bound, 0.0) / self._SLACK_SCALE + 1e-4
        lower = np.array([-1.0] * n_var + [0.0] * horizon)
        upper = np.array([1.0] * n_var + [np.inf] * horizon)
        result = minimize(
            lambda x: evaluate(x)[0],
            start,
            jac=lambda x: evaluate(x)[1],
            bounds=list(zip(lower, upper, strict=True)),
            constraints=[
                {"type": "ineq", "fun": lambda x: evaluate(x)[2], "jac": lambda x: evaluate(x)[3]}
            ],
            method="SLSQP",
            options={"maxiter": self._max_iterations, "ftol": 1e-9},
        )
        x = np.clip(result.x, lower, upper)
        worst = float(np.min(evaluate(x)[2]))
        self.last_stats = {
            "success": bool(result.success),
            "status": int(result.status),
            "iterations": int(result.nit),
            "model_calls": calls,
            "slack": float(x[n_var:].max() * self._SLACK_SCALE),
            "worst_constraint": worst,
        }
        if not np.all(np.isfinite(x)) or worst < -1e-3:
            raise RuntimeError(f"horizon problem failed: status {result.status}, {result.message}")
        return np.asarray(self._physical(x[None, :n_var])[0].T)


class HorizonMpc(LoadController):
    """D-gru: throughput from a schedule, set points from :class:`HorizonOptimizer`."""

    def __init__(
        self,
        schedule: Schedule,
        optimizer: HorizonOptimizer,
        model: RolloutModel,
        history: MeasuredHistory,
        config: LoadFollowingConfig,
        margin: float,
    ) -> None:
        super().__init__()
        self._schedule = schedule
        self._optimizer = optimizer
        self._model = model
        self._history = history
        self._config = config
        self._bound = config.headroom.line.product_moisture_limit_kg_per_kg - margin
        self._nozzle = config.process.actions.nozzle_pressure_bar.nominal
        self._sim = GreyBoxFluidBedGranulator(config.process)
        self._plan: np.ndarray | None = None
        self.info.update({"iterations": [], "model_calls": [], "solve_s": [], "slack_max": 0.0})

    def reset(self, day: DayScenario, state: np.ndarray, command: np.ndarray) -> None:
        """Start a day: the history is the measured steady state before it."""
        self._schedule.reset(day, state, command)
        d0 = day.disturbances[:, 0]
        exogenous = np.array([command[0] + d0[4], command[1], command[2], self._nozzle, *d0[0:3]])
        _, diagnostics = self._sim.evaluate_derivative(
            np.asarray(state, np.float64)[None, :], exogenous[None, 0:4], exogenous[None, 4:7]
        )
        targets = np.array(
            [
                float(np.asarray(diagnostics["power_total_kw"]).ravel()[0]),
                float(np.asarray(diagnostics["particle_moisture_kg_per_kg"]).ravel()[0]),
                float(state[3]),
            ]
        )
        self._history.reset(exogenous, targets)
        self._plan = None

    def observe(self, exogenous: np.ndarray, targets: np.ndarray) -> None:
        """Blocks of the interval just simulated (called by the runner)."""
        self._history.append(exogenous, targets)

    def command(
        self,
        k: int,
        state: np.ndarray,
        last_command: np.ndarray,
        disturbance: np.ndarray,
        sprayed_kg: float,
    ) -> np.ndarray:
        """Spray rate of the schedule, set points of the horizon problem."""
        steady = self._schedule.command(k, state, last_command, disturbance, sprayed_kg)
        spray = float(steady[2])
        if isinstance(self._model, TruthRollout):
            self._model.set_state(state)
        base = np.empty((self._model.horizon, _N_EXO))
        base[:, 0:2] = steady[0:2]
        base[:, 2] = spray
        base[:, 3] = self._nozzle
        base[:, 4:7] = disturbance[0:3]
        n = self._optimizer.n_intervals
        if self._plan is None:
            guess = np.repeat(np.asarray(steady[0:2], np.float64)[:, None], n, axis=1)
        else:
            guess = np.concatenate([self._plan[:, 1:], self._plan[:, -1:]], axis=1)
        started = time.perf_counter()
        try:
            self._plan = self._optimizer.solve(
                self._model.prepare(self._history.window()), base, self._bound, guess
            )
            inlet, gas = self._plan[:, 0]
            stats = self._optimizer.last_stats
            self.info["iterations"].append(stats["iterations"])
            self.info["model_calls"].append(stats["model_calls"])
            self.info["slack_max"] = max(self.info["slack_max"], stats["slack"])
        except RuntimeError:
            self.info["failures"] += 1
            self._plan = None
            inlet, gas = last_command[:2]
        self.info["solve_s"].append(time.perf_counter() - started)
        self.info["map_failures"] = self._schedule.info["failures"]
        return np.array([inlet, gas, spray])


class ScheduleMpc(LoadController):
    """C3-15: throughput from a schedule on the true map, set points from a short true-model MPC."""

    def __init__(
        self, schedule: Schedule, fast: DynamicOptimizer, config: LoadFollowingConfig, margin: float
    ) -> None:
        super().__init__()
        self._schedule = schedule
        self._fast = fast
        self._config = config
        self._margin = margin
        self._rate = float(command_rates(config.process)[2])
        self.info.update({"fast_iterations": []})

    def reset(self, day: DayScenario, state: np.ndarray, command: np.ndarray) -> None:
        """Forget the warm start."""
        self._schedule.reset(day, state, command)
        self._fast.reset()

    def _spray_course(self, level: float, realised: float) -> np.ndarray:
        """Spray rate per RK4 step of the horizon: rate limited from ``realised`` to ``level``."""
        cfg, fast = self._config, self._fast
        step = self._rate * cfg.headroom.scenarios.sample_time_s
        course = np.empty(fast.n_intervals * cfg.samples_per_interval)
        value = realised
        for i in range(len(course)):
            value += float(np.clip(level - value, -step, step))
            course[i] = value
        return course.reshape(fast.n_intervals * fast.n_steps, -1).mean(axis=1)

    def command(
        self,
        k: int,
        state: np.ndarray,
        last_command: np.ndarray,
        disturbance: np.ndarray,
        sprayed_kg: float,
    ) -> np.ndarray:
        """Spray rate of the schedule, set points of the short MPC."""
        steady = self._schedule.command(k, state, last_command, disturbance, sprayed_kg)
        spray = float(steady[2])
        fast = self._fast
        nodes = fast.n_intervals * fast.n_steps
        dist = np.tile(np.asarray(disturbance, dtype=np.float64)[:, None], (1, nodes))
        dist[3] = self._spray_course(spray, float(last_command[2]))
        limit = self._config.headroom.line.product_moisture_limit_kg_per_kg
        bound = np.full(nodes, limit - self._margin)
        try:
            plan = fast.solve(state[[0, 1, 3]], np.asarray(last_command[:2]), dist, bound)
            inlet, gas = plan[:, 0]
            self.info["fast_iterations"].append(int(fast.last_stats["iterations"]))  # type: ignore[call-overload]
        except RuntimeError:
            self.info["failures"] += 1
            inlet, gas = last_command[:2]
        self.info["map_failures"] = self._schedule.info["failures"]
        return np.array([inlet, gas, spray])

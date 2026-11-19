"""Cells of the study and their runs on days of the load-following task.

A :class:`Cell` is an arm at a feedback level, a budget and a seed. :func:`run_rows` is the
single entry point of the driver (``scripts/run_learned_setpoints.py``), of its worker processes
and of the tests: it builds the controllers of rows ``(cell, margin, day)``, runs them in lockstep
on the truth and returns one record per row. The loop is that of
:func:`twinlink_ml.load_following.study.run_days` (same initial conditions, same tail law, same
costs); for C2 it gives the records of ``run_days`` bit for bit (tested), so the numbers of the
learned arms and of the ladder are comparable.

Arms
----
``Cfix``, ``C0``, ``C2``, ``C3-15``
    Controllers with the true model (reference ladder and the control of family D).
``M-tab``
    Control: the TRUE steady response behind the table of family M (what the tabulation costs).
``M-quad``, ``M-gru``
    Family M at the feedback levels ``none``, ``fb`` and (M-gru) ``lab``.
``D-gru``
    Family D.
``P-lin``, ``P-mlp``, ``P-gru``
    Family P.

Every learned arm (and ``M-tab``) runs behind :class:`~.policy.TargetGuard`. The feedback levels
``fb`` and ``lab`` get a pre-roll before the day (:mod:`.feedback`); the records cover the day and
its tail only.
"""

from __future__ import annotations

import dataclasses
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from twinlink_ml.headroom.controllers import DynamicOptimizer, nominal_disturbance
from twinlink_ml.load_following.controllers import (
    ConstantThroughput,
    LoadController,
    Schedule,
    SetPointLaw,
    TailLaw,
)
from twinlink_ml.load_following.market import DayScenario
from twinlink_ml.load_following.steady import cost_rate
from twinlink_ml.load_following.study import Solvers, initial_conditions
from twinlink_ml.load_following.truth import LoadTruthPlant

from .config import LearnedSetpointsConfig
from .feedback import BiasedSetPointLaw, MoistureFeedback, build_feedback
from .gru_mpc import (
    HorizonMpc,
    HorizonOptimizer,
    InlineTargets,
    MeasuredHistory,
    NumpyGruRollout,
    ScheduleMpc,
)
from .labels import DecisionLogger
from .maps import SteadyTable, TableMap
from .policy import Policy, PolicyController, TargetGuard, load_policy

__all__ = [
    "FAMILIES",
    "LEARNED_ARMS",
    "REFERENCE_ARMS",
    "Assets",
    "Cell",
    "FixedSetPoints",
    "MapController",
    "RecordingPlant",
    "build_controller",
    "cell_assets_exist",
    "run_cell_days",
    "run_rows",
]

REFERENCE_ARMS: tuple[str, ...] = ("Cfix", "C0", "C2", "C3-15")
LEARNED_ARMS: tuple[str, ...] = ("M-tab", "M-quad", "M-gru", "D-gru", "P-lin", "P-mlp", "P-gru")
FAMILIES: dict[str, tuple[str, ...]] = {
    "M": ("M-quad", "M-gru"),
    "D": ("D-gru",),
    "P": ("P-lin", "P-mlp", "P-gru"),
}


@dataclass(frozen=True, slots=True)
class Cell:
    """An arm at a feedback level, a budget (``n_train`` or label days) and a seed."""

    arm: str
    level: str = "none"
    budget: int = 0
    seed: int = 0

    @property
    def key(self) -> str:
        """Identifier used in caches and artifacts."""
        return f"{self.arm}|{self.level}|{self.budget}|{self.seed}"

    @classmethod
    def from_key(cls, key: str) -> Cell:
        """Inverse of :attr:`key`."""
        arm, level, budget, seed = key.split("|")
        return cls(arm, level, int(budget), int(seed))

    @property
    def family(self) -> str:
        """``M``, ``D``, ``P`` or ``reference``."""
        for family, arms in FAMILIES.items():
            if self.arm in arms:
                return family
        return "reference"


class Assets:
    """Tables, checkpoints and policies of stage 0, loaded lazily (one object per worker)."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._tables: dict[str, SteadyTable] = {}
        self._rollouts: dict[str, NumpyGruRollout] = {}
        self._policies: dict[str, Path] = {}

    @staticmethod
    def table_name(arm: str, budget: int, seed: int) -> str:
        """File name of a table (``truth`` for the control ``M-tab``)."""
        if arm == "M-tab":
            return "table_truth.npz"
        kind = {"M-quad": "quad", "M-gru": "gru", "D-gru": "gru"}[arm]
        return f"table_{kind}_n{budget}_s{seed}.npz"

    @staticmethod
    def checkpoint_name(budget: int, seed: int) -> str:
        """File name of a GRU checkpoint."""
        return f"gru_n{budget}_s{seed}.pt"

    @staticmethod
    def policy_name(arm: str, budget: int, seed: int) -> str:
        """File name of a policy."""
        return f"policy_{arm.lower().replace('-', '_')}_k{budget}_s{seed}.npz"

    def table(self, cell: Cell) -> SteadyTable:
        """Steady table of an M arm (or of the schedule of D-gru)."""
        name = self.table_name(cell.arm, cell.budget, cell.seed)
        if name not in self._tables:
            self._tables[name] = SteadyTable.load(self.directory / name)
        return self._tables[name]

    def rollout(self, cell: Cell) -> NumpyGruRollout:
        """Composite GRU of a D-gru cell."""
        name = self.checkpoint_name(cell.budget, cell.seed)
        if name not in self._rollouts:
            if len(self._rollouts) >= 4:
                self._rollouts.pop(next(iter(self._rollouts)))
            self._rollouts[name] = NumpyGruRollout.load(self.directory / name)
        return self._rollouts[name]

    def policy(self, cell: Cell) -> Policy:
        """A fresh policy object of a P cell (policies may carry a state)."""
        return load_policy(self.directory / self.policy_name(cell.arm, cell.budget, cell.seed))


class FixedSetPoints(LoadController):
    """Cfix: nominal throughput and fixed set points (R0 of Befund 105)."""

    def __init__(self, command: np.ndarray, nominal: float) -> None:
        super().__init__()
        self._command = np.array([command[0], command[1], nominal], dtype=np.float64)

    def reset(self, day: DayScenario, state: np.ndarray, command: np.ndarray) -> None:
        """Nothing to reset."""

    def command(
        self,
        k: int,  # noqa: ARG002 - shared interface
        state: np.ndarray,  # noqa: ARG002 - shared interface
        last_command: np.ndarray,  # noqa: ARG002 - shared interface
        disturbance: np.ndarray,  # noqa: ARG002 - shared interface
        sprayed_kg: float,  # noqa: ARG002 - shared interface
    ) -> np.ndarray:
        """The fixed command."""
        return self._command.copy()


class MapController(LoadController):
    """Family M: the unchanged schedule and set-point law of C2 on a given steady map.

    With a :class:`~.feedback.MoistureFeedback` the bound of the law is lowered by its bias before
    every decision, and the feedback is fed with the blocks of every interval.
    """

    def __init__(
        self,
        steady_map: Any,
        config: LearnedSetpointsConfig,
        band: tuple[float, float],
        margin: float,
        feedback: MoistureFeedback | None = None,
    ) -> None:
        super().__init__()
        lf = config.load_following
        limit = lf.headroom.line.product_moisture_limit_kg_per_kg
        self._map = steady_map
        self._law = BiasedSetPointLaw(steady_map, limit, margin)
        self._schedule = Schedule(self._law, steady_map, lf, band, "E")
        self._feedback = feedback
        self._nominal = lf.nominal_spray_rate
        if feedback is not None:
            self.observe = self._observe  # only arms with feedback are observers

    def reset(self, day: DayScenario, state: np.ndarray, command: np.ndarray) -> None:
        """Start a day."""
        self._schedule.reset(day, state, command)

    def _apply_bias(self) -> None:
        if self._feedback is not None:
            self._law.bias = self._feedback.bias

    def _sync(self) -> None:
        self.info = dict(self._schedule.info)
        self.info["map_failures"] = int(getattr(self._map, "failures", 0))
        self.info["map_clipped"] = int(getattr(self._map, "clipped", 0))
        if self._feedback is not None:
            self.info["invalid_predictions"] = self._feedback.invalid_predictions

    def command(
        self,
        k: int,
        state: np.ndarray,
        last_command: np.ndarray,
        disturbance: np.ndarray,
        sprayed_kg: float,
    ) -> np.ndarray:
        """Command of the schedule on the map, with the current bias."""
        self._apply_bias()
        out = self._schedule.command(k, state, last_command, disturbance, sprayed_kg)
        self._sync()
        return out

    def pre_roll_command(self, disturbance: np.ndarray, last_command: np.ndarray) -> np.ndarray:
        """Pre-roll: nominal throughput, set points of the law with the current bias."""
        self._apply_bias()
        inlet, gas = self._law(disturbance, self._nominal, last_command)
        return np.array([inlet, gas, self._nominal])

    def _observe(self, exogenous: np.ndarray, targets: np.ndarray) -> None:
        assert self._feedback is not None
        self._feedback.update(exogenous, targets[:, 1])

    @property
    def bias_trace(self) -> list[float]:
        """Bias after every interval (pre-roll included); empty without feedback."""
        return [] if self._feedback is None else self._feedback.trace


class _SimRecorder:
    """Proxy of the truth simulator that keeps what every sample of an interval saw and gave."""

    def __init__(self, sim: Any) -> None:
        self._sim = sim
        self.rows: list[tuple[np.ndarray, np.ndarray]] = []

    def step(self, state: Any, action: Any, disturbance: Any) -> Any:
        result = self._sim.step(state, action, disturbance)
        exogenous = np.concatenate([np.asarray(action), np.asarray(disturbance)], axis=1)
        diagnostics = result.diagnostics
        targets = np.stack(
            [
                np.asarray(diagnostics["power_total_kw"], dtype=np.float64),
                np.asarray(diagnostics["particle_moisture_kg_per_kg"], dtype=np.float64),
                np.asarray(result.state, dtype=np.float64)[:, 3],
            ],
            axis=1,
        )
        self.rows.append((exogenous, targets))
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sim, name)


class RecordingPlant(LoadTruthPlant):
    """The truth plant of the load-following study; additionally hands out feedback blocks.

    The simulation is that of the base class (its simulator is only wrapped); :meth:`blocks`
    returns, for the interval just simulated, the block means of the seven exogenous channels and
    the true targets (power, moisture, bed temperature) at the block ends.
    """

    def __init__(
        self,
        config: Any,
        days: Sequence[DayScenario],
        initial_states: np.ndarray,
        initial_commands: np.ndarray,
        *,
        interval_s: float,
        block_s: float,
    ) -> None:
        super().__init__(config, days, initial_states, initial_commands, interval_s=interval_s)
        self._recorder = _SimRecorder(self._sim)
        self._sim = self._recorder  # type: ignore[assignment]
        self._samples_per_block = round(block_s / self._dt)
        if self._spi % self._samples_per_block:
            raise ValueError("the feedback grid must divide the control interval")

    def advance(self, commands: np.ndarray) -> None:
        """Simulate one control interval, keeping its samples."""
        self._recorder.rows.clear()
        super().advance(commands)

    def blocks(self) -> tuple[np.ndarray, np.ndarray]:
        """``exogenous (n, m, 7)`` block means and ``targets (n, m, 3)`` at the block ends."""
        rows = self._recorder.rows
        exogenous = np.stack([r[0] for r in rows], axis=1)  # (n, samples, 7)
        targets = np.stack([r[1] for r in rows], axis=1)
        n, spb = exogenous.shape[0], self._samples_per_block
        means = exogenous.reshape(n, -1, spb, exogenous.shape[2]).mean(axis=2)
        return means, targets[:, spb - 1 :: spb, :]


def _seed(*parts: int) -> int:
    return int(np.random.SeedSequence(list(parts)).generate_state(1)[0])


def build_controller(
    cell: Cell,
    margin: float,
    day: DayScenario,
    solvers: Solvers,
    assets: Assets | None,
    config: LearnedSetpointsConfig,
    band: tuple[float, float],
    *,
    steady_map: Any | None = None,
    policy: Policy | None = None,
    rollout: Any | None = None,
) -> LoadController:
    """Controller of ``cell`` for one day.

    ``steady_map``, ``policy`` and ``rollout`` replace what would be loaded from ``assets`` (tests:
    the true map behind family M, a lookup policy, the truth behind the rollout interface).
    """
    lf = config.load_following
    limit = lf.headroom.line.product_moisture_limit_kg_per_kg
    nominal = lf.nominal_spray_rate
    arm = cell.arm
    if arm == "C0":
        return ConstantThroughput(SetPointLaw(solvers.steady, limit, margin), nominal)
    if arm == "C2":
        return Schedule(SetPointLaw(solvers.steady, limit, margin), solvers.steady, lf, band, "E")
    if arm == "Cfix":
        point = solvers.steady.point(
            nominal_disturbance(lf.headroom, lf.process), nominal, limit - margin
        )
        if point is None:
            raise RuntimeError(f"no fixed point at margin {margin}")
        return FixedSetPoints(point.command, nominal)
    if arm == "C3-15":
        schedule = Schedule(
            SetPointLaw(solvers.steady, limit, margin), solvers.steady, lf, band, "E"
        )
        fast = DynamicOptimizer(
            lf.headroom, lf.process, config.short_horizon_intervals, expression="MX"
        )
        return ScheduleMpc(schedule, fast, lf, margin)
    if arm in ("M-tab", "M-quad", "M-gru"):
        if steady_map is None:
            assert assets is not None
            steady_map = TableMap(assets.table(cell), config)
        feedback = None
        if cell.level != "none":
            feedback = build_feedback(cell.level, steady_map, config, day.day)  # type: ignore[arg-type]
        return TargetGuard(MapController(steady_map, config, band, margin, feedback), lf, band)
    if arm == "D-gru":
        if steady_map is None:
            assert assets is not None
            steady_map = TableMap(assets.table(cell), config)
        if rollout is None:
            assert assets is not None
            rollout = assets.rollout(cell)
        block_s = config.feedback.measurement_interval_s
        law = SetPointLaw(steady_map, limit, margin)  # type: ignore[arg-type]
        schedule = Schedule(law, steady_map, lf, band, "E")  # type: ignore[arg-type]
        optimizer = HorizonOptimizer(
            rollout,
            lf.process,
            n_intervals=config.short_horizon_intervals,
            blocks_per_interval=round(lf.headroom.control.interval_s / block_s),
            block_s=block_s,
            bed_limit=lf.headroom.line.bed_temperature_limit_celsius,
            slack_weight_kwh=lf.headroom.control.soft_constraint_weight_kwh,
            fd_step=config.gru_mpc_fd_step,
            max_iterations=config.gru_mpc_max_iterations,
        )
        sensor = InlineTargets(lf.process, block_s, _seed(config.gru_mpc_measurement_seed, day.day))
        history = MeasuredHistory(rollout.history_length, sensor)
        return TargetGuard(HorizonMpc(schedule, optimizer, rollout, history, lf, margin), lf, band)
    if arm in ("P-lin", "P-mlp", "P-gru"):
        if policy is None:
            assert assets is not None
            policy = assets.policy(cell)
        inner = PolicyController(policy, lf, margin, solvers.reference_tariff)
        return TargetGuard(inner, lf, band)
    raise ValueError(f"unknown arm {arm!r}")


def _extended(day: DayScenario, n_samples: int) -> DayScenario:
    """``day`` with ``n_samples`` of its first sample prepended (pre-roll, disturbance held)."""
    if n_samples == 0:
        return day
    return dataclasses.replace(
        day,
        disturbances=np.concatenate(
            [np.repeat(day.disturbances[:, :1], n_samples, axis=1), day.disturbances], axis=1
        ),
        moisture_limit=np.concatenate(
            [np.repeat(day.moisture_limit[:1], n_samples), day.moisture_limit]
        ),
    )


def _summarise(values: Sequence[float]) -> dict[str, float] | None:
    if not len(values):
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": float(len(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
        "sum": float(array.sum()),
    }


def run_rows(
    rows: Sequence[tuple[Cell, float, DayScenario]],
    tail_margin: float,
    solvers: Solvers,
    assets: Assets | None,
    config: LearnedSetpointsConfig,
    band: tuple[float, float],
    *,
    controllers: Sequence[LoadController] | None = None,
    log_decisions: bool = False,
) -> list[dict[str, Any]]:
    """Run the rows ``(cell, margin, day)`` in one lockstep truth batch; one record per row.

    The truth simulation costs about the same for one row as for some hundred, so the driver
    packs rows of many cells into one batch. All rows of a batch must agree on the pre-roll
    (feedback level ``none`` or not). A row whose controller cannot be built or raises during
    the day is recorded with ``error`` (its plant row holds the last command); the other rows
    are not affected. ``controllers`` replaces the controllers built from the cells (tests).
    With ``log_decisions`` every controller is wrapped in a :class:`~.labels.DecisionLogger` and
    the record holds the inputs and commands of every decision (label generation).
    """
    lf = config.load_following
    headroom = lf.headroom
    limit = headroom.line.product_moisture_limit_kg_per_kg
    nominal = lf.nominal_spray_rate
    per_hour = lf.intervals_per_hour
    spi = lf.samples_per_interval
    days = [row[2] for row in rows]
    hours, tail_hours = days[0].hours, days[0].tail_hours
    n_day = hours * per_hour
    errors: list[str | None] = [None] * len(rows)
    built: list[LoadController | None]
    if controllers is None:
        built = []
        for cell, margin, day in rows:
            try:
                built.append(build_controller(cell, margin, day, solvers, assets, config, band))
            except Exception:  # a failed row must not stop the batch
                built.append(None)
                errors[len(built) - 1] = traceback.format_exc()
    else:
        built = list(controllers)
    if log_decisions:
        built = [
            None if c is None else DecisionLogger(c, lf, row[1], solvers.reference_tariff)
            for c, row in zip(built, rows, strict=True)
        ]
    with_pre_roll = {row[0].level != "none" for row in rows}
    if len(with_pre_roll) != 1:
        raise ValueError("rows with and without a pre-roll cannot share a batch")
    pre = 0
    if with_pre_roll.pop():
        pre = round(config.feedback.pre_roll_s / headroom.control.interval_s)
    observers = [
        c is not None and hasattr(c, "observe") and bool(getattr(c, "observes", True))
        for c in built
    ]
    states, commands = initial_conditions(solvers, days)
    plant_days = [_extended(day, pre * spi) for day in days]
    plant: LoadTruthPlant
    if any(observers):
        plant = RecordingPlant(
            lf.process,
            plant_days,
            states,
            commands,
            interval_s=headroom.control.interval_s,
            block_s=config.feedback.measurement_interval_s,
        )
    else:
        plant = LoadTruthPlant(
            lf.process, plant_days, states, commands, interval_s=headroom.control.interval_s
        )
    tails = [TailLaw(solvers.steady, limit, tail_margin, nominal) for _ in rows]
    decide_s: list[list[float]] = [[] for _ in rows]
    reset_s = [0.0] * len(rows)
    observe_s = [0.0] * len(rows)
    for i, (controller, day) in enumerate(zip(built, days, strict=True)):
        if controller is None:
            continue
        started = time.perf_counter()
        try:
            controller.reset(day, states[i], commands[i])
        except Exception:
            errors[i] = traceback.format_exc()
        reset_s[i] = time.perf_counter() - started
    last = np.array(commands, dtype=np.float64)
    sprayed_before = np.zeros(len(rows))
    for step in range(plant.n_intervals):
        k = step - pre
        disturbance = plant.current_disturbance()
        realised = plant.last_commands
        sprayed = plant.sprayed_so_far - sprayed_before
        new = np.empty_like(last)
        for i, controller in enumerate(built):
            if k >= n_day:
                new[i] = tails[i](disturbance[i], realised[i])
                continue
            if errors[i] is not None or controller is None:
                new[i] = last[i]
                continue
            try:
                if k < 0:
                    new[i] = controller.pre_roll_command(disturbance[i], realised[i])  # type: ignore[attr-defined]
                else:
                    started = time.perf_counter()
                    new[i] = controller.command(
                        k, plant.state[i], realised[i], disturbance[i], float(sprayed[i])
                    )
                    decide_s[i].append(time.perf_counter() - started)
            except Exception:
                errors[i] = traceback.format_exc()
                new[i] = last[i]
        plant.advance(new)
        last = new
        if any(observers) and k < n_day:
            exogenous, targets = plant.blocks()  # type: ignore[attr-defined]
            for i, controller in enumerate(built):
                if observers[i] and errors[i] is None:
                    started = time.perf_counter()
                    try:
                        controller.observe(exogenous[i], targets[i])  # type: ignore[union-attr]
                    except Exception:
                        errors[i] = traceback.format_exc()
                    observe_s[i] += time.perf_counter() - started
        if k == -1:
            sprayed_before = plant.sprayed_so_far.copy()
    result = plant.result()
    low, high = band
    summarised = (
        "plan_iterations",
        "plan_s",
        "fast_iterations",
        "iterations",
        "model_calls",
        "solve_s",
    )
    records = []
    for i, (cell, margin, day) in enumerate(rows):
        head = {
            "cell": cell.key,
            "day": day.name,
            "day_of_year": day.day,
            "month": day.month,
            "price_mode": day.price_mode,
            "controller": cell.arm,
            "margin": float(margin),
            "tail_margin": float(tail_margin),
        }
        if errors[i] is not None:
            records.append({**head, "error": errors[i], "max_violation": None})
            continue
        controller = built[i]
        assert controller is not None
        shape = (hours + tail_hours, per_hour)
        first = pre * spi
        energy = result.energy_kwh_per_interval[i][pre:].reshape(shape).sum(axis=1)
        heater = result.heater_kwh_per_interval[i][pre:].reshape(shape).sum(axis=1)
        sprayed_h = result.sprayed_kg_per_interval[i][pre:].reshape(shape).sum(axis=1)
        moisture = result.moisture[i][first:]
        excess = np.maximum(moisture - result.moisture_limit[i][first:], 0.0)
        day_samples = n_day * spi
        day_commands = result.commands[i, pre : pre + n_day]
        info = dict(controller.info)
        for key in summarised:
            if key in info:
                info[key] = _summarise(info[key])
        info["tail_failures"] = tails[i].failures
        trace = getattr(getattr(controller, "inner", controller), "bias_trace", [])
        if trace:
            in_day = np.asarray(trace[pre:], dtype=np.float64)
            info["bias"] = {
                "at_day_start": float(trace[pre - 1]) if pre else 0.0,
                "mean": float(in_day.mean()),
                "min": float(in_day.min()),
                "max": float(in_day.max()),
            }
        solid = float(day.disturbances[2, 0])
        record = {
            **head,
            "pre_roll_intervals": pre,
            "cost_e": float(np.sum(cost_rate(energy, heater, day.price_true, "E", lf))),
            "cost_g": float(np.sum(cost_rate(energy, heater, day.price_true, "G", lf))),
            "energy_kwh": float(energy.sum()),
            "heater_kwh": float(heater.sum()),
            "sprayed_day_kg": float(sprayed_h[:hours].sum()),
            "sprayed_total_kg": float(sprayed_h.sum()),
            "target_day_kg": float(nominal * hours),
            "product_dry_kg": float(sprayed_h.sum() * solid),
            "max_violation": float(excess.max()),
            "max_violation_tail": float(excess[day_samples:].max(initial=0.0)),
            "max_bed_temperature": float(result.bed_temperature[i][first:].max()),
            "moisture_mean": float(moisture[:day_samples].mean()),
            "share_at_low": float(np.mean(day_commands[:, 2] <= low + 1e-6)),
            "share_at_high": float(np.mean(day_commands[:, 2] >= high - 1e-6)),
            "inlet_command_mean": float(day_commands[:, 0].mean()),
            "gas_velocity_command_mean": float(day_commands[:, 1].mean()),
            "energy_kwh_per_hour": [float(f"{v:.7g}") for v in energy],
            "sprayed_kg_per_hour": [float(f"{v:.7g}") for v in sprayed_h],
            "moisture_max_per_hour": [
                float(f"{v:.6g}") for v in moisture.reshape(shape[0], -1).max(axis=1)
            ],
            "decide_s": _summarise(decide_s[i]),
            "reset_s": float(reset_s[i]),
            "observe_s": float(observe_s[i]),
            "solver": info,
        }
        if log_decisions:
            record["decisions"] = {
                "features": np.array(controller.features),  # type: ignore[attr-defined]
                "commands": np.array(controller.commands),  # type: ignore[attr-defined]
            }
        records.append(record)
    return records


def run_cell_days(
    cell: Cell,
    margin: float,
    tail_margin: float,
    days: Sequence[DayScenario],
    solvers: Solvers,
    assets: Assets | None,
    config: LearnedSetpointsConfig,
    band: tuple[float, float],
    *,
    controllers: Sequence[LoadController] | None = None,
    log_decisions: bool = False,
) -> list[dict[str, Any]]:
    """One cell at one margin on ``days`` (:func:`run_rows` with equal cells and margins)."""
    return run_rows(
        [(cell, margin, day) for day in days],
        tail_margin,
        solvers,
        assets,
        config,
        band,
        controllers=controllers,
        log_decisions=log_decisions,
    )


def cell_assets_exist(cell: Cell, directory: Path) -> bool:
    """Whether the stage-0 files a cell needs are present."""
    names = []
    if cell.arm in ("M-tab", "M-quad", "M-gru", "D-gru"):
        names.append(Assets.table_name(cell.arm, cell.budget, cell.seed))
    if cell.arm == "D-gru":
        names.append(Assets.checkpoint_name(cell.budget, cell.seed))
    if cell.arm in ("P-lin", "P-mlp", "P-gru"):
        names.append(Assets.policy_name(cell.arm, cell.budget, cell.seed))
    return all((Path(directory) / name).exists() for name in names)

r"""Test sets T1-T3 of study B1 (information per run), reported separately.

* **T1 neutral mix**: the same number of runs from each plan B0-B4, each from its own test seed.
* **T2 constant operation**: the constant-run hold-out of earlier studies
  (``MixedExcitationDesign`` with ``constant_run_fraction = 1``).
* **T3 deployment profiles** (the decisive one): runs that start in a steady state and make
  set-point changes near the oracle optimum or near the product-moisture quality limit.

No test run comes from a training run: test seeds are derived from a fixed namespace that no
master seed enters, and :func:`assert_disjoint` compares the actions of every test run with the
training runs (identical action arrays are an error, not only identical ids).

T3 definition
-------------
Anchors come from an artifact (``scripts/compute_b1_t3_anchors.py``): for each of a few ambient
conditions the constrained optimum of the steady state at the lower gas-velocity bound 1.26 m/s
(``oracle_search`` mechanics, corrected solver of Befund 78/79, quality limit of the decoupled
field ``product_moisture_limit_kg_per_kg`` = 0.15 kg/kg, not the critical moisture of the
kinetics) and a second operating point on the spray-rate line through the optimum at which the
steady-state product moisture is half the soft-sensor margin below the quality limit. A T3 run
starts in the steady state of its anchor and its own disturbances at ``t = 0`` (no start-up
transient), holds for one hour, then makes ``n_changes`` set-point changes of two randomly
chosen channels by 5-10 % of their span around the anchor (not cumulative), each held for an
equal share of the rest of the run. The gas velocity never goes below the study bound.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..simulation.base import Trajectory
from ..simulation.config import ProcessConfig
from ..simulation.excitation import (
    DiurnalDisturbanceDesign,
    ExcitationConfig,
    ExcitationSchedule,
    MixedExcitationDesign,
    apply_rate_limit,
)
from ..simulation.excitation_plans import PLAN_NAMES, make_plan
from ..simulation.greybox import GreyBoxFluidBedGranulator
from ..simulation.steady_state import SteadyStateSolution, solve_steady_state
from .experiments import DataSpec
from .growing_pool import simulate_schedule

__all__ = [
    "SEARCH_MAX_ITERATIONS",
    "SEARCH_TOLERANCE",
    "HoldoutSet",
    "assert_disjoint",
    "build_t1",
    "build_t2",
    "build_t3",
    "config_at",
    "find_limit_spray",
    "holdout_seed",
    "steady_state_at",
]

SEARCH_MAX_ITERATIONS = 30
SEARCH_TOLERANCE = 1.0e-7
"""Search precision of Befund 52 / M1 (the solver defaults do not reproduce it)."""

_CHANNELS = ("t_in", "u_gas", "spray", "p_nozzle")


@dataclass(frozen=True, slots=True)
class HoldoutSet:
    """A named test set: its inputs and the simulated trajectory."""

    name: str
    schedule: ExcitationSchedule
    trajectory: Trajectory


def holdout_seed(name: str, replicate: int = 0, part: str = "") -> int:
    """Seed of a test set from a fixed namespace (no master seed enters)."""
    digest = hashlib.sha256(f"b1-test:{name}:{part}:{replicate}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def assert_disjoint(training: Trajectory, test_sets: Sequence[HoldoutSet]) -> None:
    """No test run equals a training run (ids or action arrays)."""
    train_ids = set(training.run_id)
    train_hashes = {hashlib.sha1(a.tobytes()).hexdigest() for a in training.actions}
    for test in test_sets:
        shared = train_ids & set(test.trajectory.run_id)
        if shared:
            raise ValueError(f"{test.name} shares run ids with the training data: {sorted(shared)}")
        for run_id, actions in zip(test.trajectory.run_id, test.trajectory.actions, strict=True):
            if hashlib.sha1(actions.tobytes()).hexdigest() in train_hashes:
                raise ValueError(f"{test.name}: run {run_id} has the actions of a training run")


# ------------------------------------------------------------------------------ T1, T2


def build_t1(
    process_config: ProcessConfig,
    spec: DataSpec,
    *,
    runs_per_plan: int = 8,
    replicate: int = 0,
) -> HoldoutSet:
    """T1: ``runs_per_plan`` runs from each of B0-B4."""
    from .growing_pool import concatenate_trajectories, prepare_process_config

    config = prepare_process_config(process_config, spec)
    schedules: list[ExcitationSchedule] = []
    parts: list[Trajectory] = []
    for name in PLAN_NAMES:
        plan = make_plan(
            name,
            config,
            sample_time_s=spec.sample_time_s,
            run_duration_s=spec.run_duration_s,
            solid_fraction_process=spec.solid_fraction_process,
        )
        schedule = plan.build(runs_per_plan, seed=holdout_seed("t1", replicate, name))
        schedules.append(schedule)
        parts.append(
            simulate_schedule(
                config, schedule, noise_seed=holdout_seed("t1-noise", replicate, name)
            )
        )
    merged = ExcitationSchedule(
        run_id=tuple(r for s in schedules for r in s.run_id),
        actions=np.concatenate([s.actions for s in schedules]),
        disturbances=np.concatenate([s.disturbances for s in schedules]),
        initial_states=np.concatenate([s.initial_states for s in schedules]),  # type: ignore[misc]
        signal_kinds=tuple(k for s in schedules for k in s.signal_kinds),
        metadata={"design_name": "t1", "plans": list(PLAN_NAMES)},
    )
    return HoldoutSet("T1", merged, concatenate_trajectories(parts))


def build_t2(
    process_config: ProcessConfig, spec: DataSpec, *, n_runs: int = 32, replicate: int = 0
) -> HoldoutSet:
    """T2: constant-operation hold-out (``constant_run_fraction = 1``)."""
    from .growing_pool import prepare_process_config

    config = prepare_process_config(process_config, spec)
    design = MixedExcitationDesign(
        ExcitationConfig(run_duration_s=spec.run_duration_s, constant_run_fraction=1.0),
        config.actions,
        config.disturbances,
        sample_time_s=spec.sample_time_s,
        geometry=config.geometry,
        material=config.material,
        operation=config.operation,
        disturbance_design=(
            None
            if spec.solid_fraction_process == "diurnal"
            else DiurnalDisturbanceDesign(solid_fraction_process=spec.solid_fraction_process)
        ),
        name="t2",
    )
    schedule = design.build(n_runs, seed=holdout_seed("t2", replicate))
    trajectory = simulate_schedule(config, schedule, noise_seed=holdout_seed("t2-noise", replicate))
    return HoldoutSet("T2", schedule, trajectory)


# ---------------------------------------------------------------------------------- T3


def config_at(
    base: ProcessConfig, action: Sequence[float], disturbance: Sequence[float]
) -> ProcessConfig:
    """Config whose nominal fields are the given 4 actions and 3 disturbances.

    ``solve_steady_state`` reads actions and disturbances only through the nominal fields.
    """
    actions = dataclasses.replace(
        base.actions,
        inlet_temperature_celsius=dataclasses.replace(
            base.actions.inlet_temperature_celsius, nominal=float(action[0])
        ),
        gas_velocity_m_per_s=dataclasses.replace(
            base.actions.gas_velocity_m_per_s, nominal=float(action[1])
        ),
        spray_rate_kg_per_h=dataclasses.replace(
            base.actions.spray_rate_kg_per_h, nominal=float(action[2])
        ),
        nozzle_pressure_bar=dataclasses.replace(
            base.actions.nozzle_pressure_bar, nominal=float(action[3])
        ),
    )
    disturbances = dataclasses.replace(
        base.disturbances,
        ambient_temperature_celsius=dataclasses.replace(
            base.disturbances.ambient_temperature_celsius, nominal=float(disturbance[0])
        ),
        ambient_relative_humidity=dataclasses.replace(
            base.disturbances.ambient_relative_humidity, nominal=float(disturbance[1])
        ),
        suspension_solid_fraction=dataclasses.replace(
            base.disturbances.suspension_solid_fraction, nominal=float(disturbance[2])
        ),
    )
    return dataclasses.replace(base, actions=actions, disturbances=disturbances)


def steady_state_at(
    base: ProcessConfig, action: Sequence[float], disturbance: Sequence[float]
) -> SteadyStateSolution:
    """Steady state at an arbitrary operating point, search precision of Befund 52 / M1."""
    simulator = GreyBoxFluidBedGranulator(config_at(base, action, disturbance))
    return solve_steady_state(
        simulator, max_iterations=SEARCH_MAX_ITERATIONS, tolerance=SEARCH_TOLERANCE
    )


def find_limit_spray(
    base: ProcessConfig,
    optimum_action: Sequence[float],
    disturbance: Sequence[float],
    *,
    target_moisture_kg_per_kg: float,
    iterations: int = 40,
) -> tuple[float, float]:
    """Spray rate at which the steady-state product moisture equals the target.

    Bisection over the spray-rate bounds with the other channels at the optimum; the moisture
    must rise with the spray rate (checked at the bounds). Returns ``(spray, moisture)``.
    """
    bound = base.actions.spray_rate_kg_per_h

    def moisture(spray: float) -> float:
        action = (optimum_action[0], optimum_action[1], spray, optimum_action[3])
        return steady_state_at(base, action, disturbance).particle_moisture_kg_per_kg

    low, high = bound.minimum, bound.maximum
    m_low, m_high = moisture(low), moisture(high)
    if not m_low <= target_moisture_kg_per_kg <= m_high:
        raise ValueError(
            f"target {target_moisture_kg_per_kg} not bracketed by the spray bounds "
            f"(moisture {m_low:.4f} .. {m_high:.4f})"
        )
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        if moisture(mid) < target_moisture_kg_per_kg:
            low = mid
        else:
            high = mid
    spray = 0.5 * (low + high)
    return spray, moisture(spray)


def build_t3(
    process_config: ProcessConfig,
    spec: DataSpec,
    anchors: Mapping[str, Any],
    *,
    runs_per_cell: int = 4,
    n_changes: int = 4,
    initial_hold_s: float = 3600.0,
    replicate: int = 0,
) -> HoldoutSet:
    """T3 from the anchors artifact (see module docstring).

    ``anchors`` is the parsed JSON of ``compute_b1_t3_anchors``: ``u_gas_min``,
    ``anchors`` (each with ``ambient_unit`` and two operating points ``optimum`` and ``limit``,
    each ``{"t_in", "u_gas", "spray", "p_nozzle"}``).
    """
    from .growing_pool import prepare_process_config

    config = prepare_process_config(process_config, spec)
    dt = spec.sample_time_s
    n_steps = round(spec.run_duration_s / dt)
    families = ("optimum", "limit")
    cells = [(a, f) for a in anchors["anchors"] for f in families]
    n_runs = len(cells) * runs_per_cell
    rng = np.random.default_rng(holdout_seed("t3", replicate))

    unit = np.empty((n_runs, len(config.disturbances.names)), dtype=np.float64)
    base_actions = np.empty((n_runs, 4), dtype=np.float64)
    kinds: list[tuple[str, ...]] = []
    for cell_index, (anchor, family) in enumerate(cells):
        for k in range(runs_per_cell):
            run = cell_index * runs_per_cell + k
            unit[run] = [anchor["ambient_unit"][0], anchor["ambient_unit"][1], 0.5]
            point = anchor[family]
            base_actions[run] = [point[c] for c in _CHANNELS]
            kinds.append((f"t3-{family}",) * 4)
    disturbances = DiurnalDisturbanceDesign(
        solid_fraction_process=spec.solid_fraction_process  # type: ignore[arg-type]
    ).build(config.disturbances, unit, n_steps, dt, rng)

    bounds = config.actions.bounds
    lower = np.array([b.minimum for b in bounds])
    upper = np.array([b.maximum for b in bounds])
    span = upper - lower
    lower[1] = max(lower[1], float(anchors["u_gas_min"]))
    hold_steps = round(initial_hold_s / dt)
    change_steps = np.linspace(hold_steps, n_steps, n_changes + 1).astype(int)
    actions = np.repeat(base_actions[:, None, :], n_steps, axis=1)
    for run in range(n_runs):
        for change in range(n_changes):
            channels = rng.choice(4, size=2, replace=False)
            target = base_actions[run].copy()
            for channel in channels:
                size = rng.uniform(0.05, 0.10) * span[channel] * rng.choice([-1.0, 1.0])
                target[channel] = np.clip(
                    base_actions[run, channel] + size, lower[channel], upper[channel]
                )
            actions[run, change_steps[change] : change_steps[change + 1]] = target
    for channel, bound in enumerate(bounds):
        actions[:, :, channel] = apply_rate_limit(actions[:, :, channel], bound.max_rate_per_s, dt)

    initial = np.empty((n_runs, 4), dtype=np.float64)
    for run in range(n_runs):
        solution = steady_state_at(config, base_actions[run], disturbances[run, 0])
        if not solution.feasible:
            raise ValueError(
                f"T3 run {run}: steady state infeasible: {solution.infeasibility_reasons}"
            )
        initial[run] = solution.state

    payload = json.dumps({"anchors": anchors, "replicate": replicate, "n": n_runs}, sort_keys=True)
    fingerprint = hashlib.sha256(payload.encode()).hexdigest()[:8]
    schedule = ExcitationSchedule(
        run_id=tuple(f"t3-{fingerprint}-{i:05d}" for i in range(n_runs)),
        actions=actions,
        disturbances=disturbances,
        initial_states=None,
        signal_kinds=tuple(kinds),
        metadata={
            "design_name": "t3",
            "fingerprint": fingerprint,
            "sample_time_s": dt,
            "run_duration_s": spec.run_duration_s,
            "action_names": config.actions.names,
            "disturbance_names": config.disturbances.names,
        },
    )
    trajectory = simulate_schedule(
        config, schedule, noise_seed=holdout_seed("t3-noise", replicate), initial_states=initial
    )
    return HoldoutSet("T3", schedule, trajectory)

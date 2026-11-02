r"""CB1-CB4: checks whether Befund 52's regret survives the step from steady state to batch.

NO network training, NO reinforcement learning -- transient rollouts of the existing
grey-box simulator (``GreyBoxFluidBedGranulator.rollout``, ``simulation/base.py``) plus
evaluation, building on the committed result of Befund 52
(``artifacts/runs/bb1_bb3_operating_point_regret.json``, ``scripts/
study_operating_point_regret.py``).

Occasion: Befund 52 assesses 25 grid points (ambient temperature x relative humidity)
exclusively in the exact steady state (``solve_steady_state``). The solids mode has
``tau_4 = 16000 s`` against a batch duration of 28800 s (ratio 1.80, 83.5 %
approach to the fixed point) -- the fixed point is never reached during a batch. This
script checks whether regret and feasibility from Befund 52 hold up under this batch
reality.

Two configs, deliberately different, see ``docs/04_stand_ap1.md``, section
"Fingerprint-Historie":

* **Steady state** (reference, identical to Befund 52): ``conf/process/
  greybox_gab_placeholder.yaml`` loaded unchanged (fingerprint ``2acd06f38963c3af`` --
  before the pseudonymisation cut 2026-09-25: ``abbe8325f1326d80``, see
  ``docs/ARCHIV.md`` -- ``sample_time_s=5.0`` file default, projection OFF).
  ``solve_steady_state`` is
  algebraic (root of ``evaluate_derivative``) -- ``sample_time_s``/projection
  do not enter there; the choice of this config for the reference calculation is therefore only
  convenience (the same as Befund 52), not a physical determination.
* **Transient** (batch rollout, CB1-CB4): the same file, ``numerics.sample_time_s=25.0``
  and ``numerics.enforce_gas_velocity_projection=True`` overridden (fingerprint
  ``bddca37f716f5cfc`` -- before the cut: ``2d2679eeed716577`` --, verified below) --
  the config under which every other
  batch/excitation study in the project runs
  (``DataSpec(sample_time_s=25.0, run_duration_s=28800.0)``, among others
  ``scripts/study_learning_curve.py``, ``scripts/probe_x3_recoverable_power.py``),
  NOT the steady-state config from Befund 52. Reason for the projection: Befund 52
  optimises without it (``solve_steady_state`` knows no projection, see CB1(d)) -- its
  effect during the batch is itself part of the question, not decided in advance.

CB1 -- fixed point against batch trajectory
-------------------------------------
For each of the 25 grid points: rollout over 28800 s (Delta_t=25 s, 1152 steps) with the
set points optimal THERE (Befund 52), held constant, initial state
``simulator.initial_state(1)`` (nominal). Reported per grid point: final state against the
fixed point (d50, solids mass, bed temperature, moisture, absolute and relative),
``energy_target`` (end/batch mean/fixed point), ``Q_rec`` (end/batch mean/fixed point,
``twinlink_ml.evaluation.recoverable_power.compute_recoverable_power_kw`` reused,
reference mass flow/dew-point margin unchanged module default) and ``n_projected_steps``
(``Trajectory.n_projected_steps``, generic bookkeeping from ``base.py``).

The fixed point itself is NOT taken over from Befund 52's JSON (no
``outlet_humidity_kg_per_kg`` for Q_rec there), but computed afresh with
``solve_steady_state`` under the steady-state config, via
``scripts/study_operating_point_regret.py``'s own ``build_variant_config``/``evaluate_point``
(loaded via ``importlib``, no line changed there) -- this is at the same time a cross-check
against Befund 52 (field ``steady_state_cross_check``).

CB2 -- Are the stationary optima feasible as a batch operating mode?
---------------------------------------------------------------------
Over the ENTIRE batch trajectory (not only the end): share of time steps with
``product_moisture_kg_per_kg > critical_moisture_kg_per_kg``, maximum exceedance;
the same for the fluidization window (``fluidization_margin_lower/upper_m_per_s`` from
``greybox._rhs``, evaluated at the REALISED, possibly projected gas velocity --
with active projection structurally almost never negative, see CB1(d) for the frequency of the
intervention) and for ``bed_temperature_celsius > 80``.

CB3 -- Does the regret survive?
------------------------------
Repeats the regret calculation from Befund 52 (BB3(b)), but with the BATCH-AVERAGED
energy instead of the fixed-point value, with UNCHANGED set points: 25 rollouts with the
respective optimal set points (reused from CB1) plus 25 rollouts with the fixed
operating point from Befund 52 (``T_in=90.41 °C``, ``u_gas=0.800 m/s``,
``p_nozzle=1.003 bar``). NO transient re-optimisation -- the set points stay the
stationary optimal ones.

CB4 (optional, ``--cb4``) -- does the optimum move?
-----------------------------------------------------
Re-optimisation with batch-averaged energy as the objective function, at the four corner
points of the disturbance grid (min/max ambient temperature x min/max humidity). The same
optimiser mechanics as Befund 52 (``differential_evolution`` + Nelder-Mead), but the
objective function itself is now a 1152-step rollout (~6 s/call, measured) instead of a
single ``solve_steady_state`` call (~0.5 s) -- ``differential_evolution`` therefore runs
with ``workers=-1`` (population parallelism WITHIN one point, not, as in Befund 52,
across the points: here there are only four, not 25). Stopping criterion: 30 min per
grid point (``callback``, wall-clock), after that the intermediate solution plus a run-time
projection.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import time
import warnings
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution, minimize

from twinlink_ml.evaluation.recoverable_power import compute_recoverable_power_kw
from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    PlaceholderConfigWarning,
    ProcessConfig,
    load_process_config,
)
from twinlink_ml.simulation.backend import NumpyBackend
from twinlink_ml.simulation.base import Trajectory
from twinlink_ml.utils.fingerprint_history import fingerprints_equivalent
from twinlink_ml.utils.tracking import environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
REGRET_STUDY_PATH = REPO_ROOT / "artifacts" / "runs" / "bb1_bb3_operating_point_regret.json"
REGRET_SCRIPT_PATH = REPO_ROOT / "scripts" / "study_operating_point_regret.py"
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "study_operating_point_transient.json"

EXPECTED_STEADY_STATE_FINGERPRINT = "2acd06f38963c3af"
"""Muss mit ``config_fingerprint`` in ``bb1_bb3_operating_point_regret.json`` übereinstimmen."""
EXPECTED_TRANSIENT_FINGERPRINT = "bddca37f716f5cfc"
"""``docs/04_stand_ap1.md``, Fingerprint-Historie: ``sample_time_s=25 s`` (angewendet),
``enforce_gas_velocity_projection=True``, nach Einführung von ``ExhaustConfig``."""

BATCH_DURATION_S = 28_800.0
SAMPLE_TIME_S = 25.0
N_STEPS = round(BATCH_DURATION_S / SAMPLE_TIME_S)
assert N_STEPS * SAMPLE_TIME_S == BATCH_DURATION_S

BED_TEMPERATURE_LIMIT_C = 80.0
"""Wie ``study_operating_point_regret.BED_TEMPERATURE_LIMIT_C`` -- Platzhalter, Befund 46."""

_SOLVE_FAILURE_PENALTY_KW = 1.0e6
_PENALTY_WEIGHT_KW = 50.0
_CB4_TIME_LIMIT_S = 30.0 * 60.0
_CB4_DE_POPSIZE = 8
_CB4_DE_MAXITER = 12
_CB4_DE_SEED = 0


def _load_regret_module() -> ModuleType:
    """``study_operating_point_regret.py`` loaded by file path (no package under
    ``scripts/``) -- reuse of ``build_variant_config``/``evaluate_point``/
    ``specific_energy_kwh_per_kg``/tolerance constants, no line changed there.

    Must be entered in ``sys.modules`` BEFORE ``exec_module``: the target module
    defines ``@dataclass(frozen=True, slots=True)`` classes whose fields Python 3.12
    resolves via ``sys.modules[cls.__module__]`` (``dataclasses._is_type``) -- without the
    entry the import fails with ``AttributeError: 'NoneType' object has no attribute
    '__dict__'``, not an error of the target module itself.
    """
    spec = importlib.util.spec_from_file_location(
        "study_operating_point_regret", REGRET_SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _suppress_placeholder_warning() -> None:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)


def build_transient_config(steady_state_base: ProcessConfig) -> ProcessConfig:
    """``sample_time_s=25 s``, projection ON -- see module docstring."""
    return dataclasses.replace(
        steady_state_base,
        numerics=dataclasses.replace(
            steady_state_base.numerics,
            sample_time_s=SAMPLE_TIME_S,
            enforce_gas_velocity_projection=True,
        ),
    )


@dataclass(frozen=True, slots=True)
class GridPoint:
    """A disturbance grid point from Befund 52, with the set points optimal there."""

    index: int
    ambient_temperature_celsius: float
    ambient_relative_humidity: float
    inlet_temperature_celsius: float
    gas_velocity_m_per_s: float
    nozzle_pressure_bar: float
    steady_state_energy_target_kw: float
    steady_state_feasible: bool


def load_grid_points(regret_json: dict[str, Any]) -> list[GridPoint]:
    points: list[GridPoint] = []
    for i, result in enumerate(regret_json["grid_results"]):
        points.append(
            GridPoint(
                index=i,
                ambient_temperature_celsius=result["ambient_temperature_celsius"],
                ambient_relative_humidity=result["ambient_relative_humidity"],
                inlet_temperature_celsius=result["inlet_temperature_celsius"],
                gas_velocity_m_per_s=result["gas_velocity_m_per_s"],
                nozzle_pressure_bar=result["nozzle_pressure_bar"],
                steady_state_energy_target_kw=result["energy_target"],
                steady_state_feasible=result["feasible"],
            )
        )
    return points


def rollout_constant_setpoint(
    config: ProcessConfig,
    *,
    t_in: float,
    u_gas: float,
    p_nozzle: float,
    t_amb: float,
    phi_amb: float,
    spray_rate: float,
    solid_fraction: float,
    n_steps: int,
    run_id: str,
) -> Trajectory:
    """Rollout with set points/disturbances held constant over the entire batch.

    ``initial_state`` is the nominal initial state (task: "starting from the nominal
    initial state"), not the steady state -- exactly the difference that CB1 measures.
    """
    simulator = GreyBoxFluidBedGranulator(config)
    actions = np.tile(
        np.array([[t_in, u_gas, spray_rate, p_nozzle]], dtype=np.float64), (1, n_steps, 1)
    )
    disturbances = np.tile(
        np.array([[t_amb, phi_amb, solid_fraction]], dtype=np.float64), (1, n_steps, 1)
    )
    initial_state = simulator.initial_state(1)
    return simulator.rollout(
        actions, disturbances, run_id=(run_id,), seed=0, initial_state=initial_state
    )


def q_rec_series_kw(config: ProcessConfig, trajectory: Trajectory) -> np.ndarray:
    """``Q_rec`` per time step, reused from ``evaluation.recoverable_power``.

    ``outlet_temperature_celsius = bed_temperature_celsius - outlet_duct_heat_loss_k``
    (0.0 in this config, see ``simulation.exhaust.reconstruct_exhaust_state`` --
    identical calculation, here without the detour via the function, because
    ``outlet_humidity_kg_per_kg`` already sits unchanged in ``diagnostics``).
    """
    bed_temperature = trajectory.states[0, :, 3]
    outlet_temperature = bed_temperature - config.exhaust.outlet_duct_heat_loss_k
    outlet_humidity = trajectory.diagnostics["outlet_humidity_kg_per_kg"][0, :]
    backend = NumpyBackend()
    return np.asarray(
        compute_recoverable_power_kw(backend, config.gas, outlet_temperature, outlet_humidity)
    )


@dataclass(frozen=True, slots=True)
class SteadyStateReference:
    """Fixed point of a grid point, fresh from ``solve_steady_state`` -- see CB1 docstring."""

    solid_mass_kg: float
    d50_um: float
    particle_moisture_kg_per_kg: float
    bed_temperature_celsius: float
    energy_target_kw: float
    q_rec_kw: float
    feasible: bool


def compute_steady_state_reference(
    regret_module: ModuleType,
    steady_state_base: ProcessConfig,
    point: GridPoint,
) -> SteadyStateReference:
    # The same (relaxed) bisection precision as Befund 52's own report call
    # (``optimize_grid_point``, justified in detail there: solver defaults at the
    # search result reproduce the bracket failure found in Befund 52 itself,
    # see ``evaluate_point`` docstring) -- not the solver defaults (100/1e-13).
    solution = regret_module.evaluate_point(
        point.inlet_temperature_celsius,
        point.gas_velocity_m_per_s,
        point.nozzle_pressure_bar,
        point.ambient_temperature_celsius,
        point.ambient_relative_humidity,
        max_iterations=regret_module.DEFAULT_SEARCH_MAX_ITERATIONS,
        tolerance=regret_module.DEFAULT_SEARCH_TOLERANCE,
        base_config=steady_state_base,
    )
    backend = NumpyBackend()
    outlet_temperature = (
        solution.bed_temperature_celsius - steady_state_base.exhaust.outlet_duct_heat_loss_k
    )
    q_rec = compute_recoverable_power_kw(
        backend,
        steady_state_base.gas,
        np.asarray(outlet_temperature),
        np.asarray(solution.outlet_humidity_kg_per_kg),
    )
    return SteadyStateReference(
        solid_mass_kg=solution.solid_mass_kg,
        d50_um=solution.diameter_m * 1.0e6,
        particle_moisture_kg_per_kg=solution.particle_moisture_kg_per_kg,
        bed_temperature_celsius=solution.bed_temperature_celsius,
        energy_target_kw=solution.diagnostics["energy_target"],
        q_rec_kw=float(q_rec),
        feasible=solution.feasible,
    )


def _rel_pct(actual: float, reference: float) -> float:
    denom = abs(reference) if abs(reference) > 1e-9 else 1e-9
    return 100.0 * (actual - reference) / denom


def cb2_constraint_series(
    trajectory: Trajectory, critical_moisture_kg_per_kg: float
) -> dict[str, Any]:
    diagnostics = trajectory.diagnostics
    moisture = diagnostics["particle_moisture_kg_per_kg"][0, :]
    fluid_lower = diagnostics["fluidization_margin_lower_m_per_s"][0, :]
    fluid_upper = diagnostics["fluidization_margin_upper_m_per_s"][0, :]
    bed_temperature = trajectory.states[0, :, 3]

    moisture_violation = np.maximum(0.0, moisture - critical_moisture_kg_per_kg)
    fluid_lower_violation = np.maximum(0.0, -fluid_lower)
    fluid_upper_violation = np.maximum(0.0, -fluid_upper)
    bed_temperature_violation = np.maximum(0.0, bed_temperature - BED_TEMPERATURE_LIMIT_C)

    n_steps = moisture.shape[0]

    def _summary(violation: np.ndarray) -> dict[str, float]:
        return {
            "fraction_steps_violating": float(np.mean(violation > 0.0)),
            "max_exceedance": float(np.max(violation)),
        }

    result = {
        "n_steps": n_steps,
        "moisture": _summary(moisture_violation),
        "fluidization_lower": _summary(fluid_lower_violation),
        "fluidization_upper": _summary(fluid_upper_violation),
        "bed_temperature": _summary(bed_temperature_violation),
    }
    if trajectory.projection_active:
        assert trajectory.n_projected_steps is not None
        result["n_projected_steps"] = int(trajectory.n_projected_steps[0])
        result["fraction_projected_steps"] = float(trajectory.n_projected_steps[0]) / n_steps
    return result


@dataclass(frozen=True, slots=True)
class GridPointTask:
    """A grid-point work package for ``pool.imap_unordered`` in :func:`main`."""

    point: GridPoint
    transient_config: ProcessConfig
    steady_state_base: ProcessConfig
    fixed_setpoint: tuple[float, float, float]
    spray_rate: float
    solid_fraction: float
    critical_moisture_kg_per_kg: float


def _run_grid_point(task: GridPointTask) -> dict[str, Any]:
    """CB1+CB2+CB3 raw data for ONE grid point -- target of ``pool.imap_unordered``.

    Runs in a worker process (``fork``); the regret module is loaded ANEW here (no
    module global inherited across the fork, see ``study_operating_point_regret.py``'s own
    pattern with ``_BASE_CONFIG`` -- here instead passed through as a parameter, because each
    task carries its own grid point anyway).
    """
    regret_module = _load_regret_module()
    point = task.point
    t_in, u_gas, p_nozzle = (
        point.inlet_temperature_celsius,
        point.gas_velocity_m_per_s,
        point.nozzle_pressure_bar,
    )
    fixed_t_in, fixed_u_gas, fixed_p_nozzle = task.fixed_setpoint

    steady_state = compute_steady_state_reference(regret_module, task.steady_state_base, point)

    optimal_trajectory = rollout_constant_setpoint(
        task.transient_config,
        t_in=t_in,
        u_gas=u_gas,
        p_nozzle=p_nozzle,
        t_amb=point.ambient_temperature_celsius,
        phi_amb=point.ambient_relative_humidity,
        spray_rate=task.spray_rate,
        solid_fraction=task.solid_fraction,
        n_steps=N_STEPS,
        run_id=f"optimal_{point.index:02d}",
    )
    fixed_trajectory = rollout_constant_setpoint(
        task.transient_config,
        t_in=fixed_t_in,
        u_gas=fixed_u_gas,
        p_nozzle=fixed_p_nozzle,
        t_amb=point.ambient_temperature_celsius,
        phi_amb=point.ambient_relative_humidity,
        spray_rate=task.spray_rate,
        solid_fraction=task.solid_fraction,
        n_steps=N_STEPS,
        run_id=f"fixed_{point.index:02d}",
    )

    optimal_energy = optimal_trajectory.outputs[0, :, 0]
    optimal_q_rec = q_rec_series_kw(task.transient_config, optimal_trajectory)
    fixed_energy = fixed_trajectory.outputs[0, :, 0]

    end_solid_mass = float(optimal_trajectory.states[0, -1, 0])
    end_d50_um = float(optimal_trajectory.outputs[0, -1, 3])
    end_moisture = float(optimal_trajectory.outputs[0, -1, 2])
    end_bed_temperature = float(optimal_trajectory.outputs[0, -1, 1])

    deviation_abs = {
        "solid_mass_kg": end_solid_mass - steady_state.solid_mass_kg,
        "d50_um": end_d50_um - steady_state.d50_um,
        "particle_moisture_kg_per_kg": end_moisture - steady_state.particle_moisture_kg_per_kg,
        "bed_temperature_celsius": end_bed_temperature - steady_state.bed_temperature_celsius,
    }
    deviation_rel_pct = {
        "solid_mass_kg": _rel_pct(end_solid_mass, steady_state.solid_mass_kg),
        "d50_um": _rel_pct(end_d50_um, steady_state.d50_um),
        "particle_moisture_kg_per_kg": _rel_pct(
            end_moisture, steady_state.particle_moisture_kg_per_kg
        ),
        "bed_temperature_celsius": _rel_pct(
            end_bed_temperature, steady_state.bed_temperature_celsius
        ),
    }

    return {
        "grid_index": point.index,
        "ambient_temperature_celsius": point.ambient_temperature_celsius,
        "ambient_relative_humidity": point.ambient_relative_humidity,
        "setpoints_optimal": {
            "inlet_temperature_celsius": t_in,
            "gas_velocity_m_per_s": u_gas,
            "nozzle_pressure_bar": p_nozzle,
        },
        "steady_state_reference": dataclasses.asdict(steady_state),
        "steady_state_cross_check": {
            "energy_target_abs_diff_kw": steady_state.energy_target_kw
            - point.steady_state_energy_target_kw,
            "befund_52_energy_target_kw": point.steady_state_energy_target_kw,
            "befund_52_feasible": point.steady_state_feasible,
        },
        "transient_end_of_batch": {
            "solid_mass_kg": end_solid_mass,
            "d50_um": end_d50_um,
            "particle_moisture_kg_per_kg": end_moisture,
            "bed_temperature_celsius": end_bed_temperature,
            "energy_target_kw": float(optimal_energy[-1]),
            "q_rec_kw": float(optimal_q_rec[-1]),
        },
        "transient_batch_mean": {
            "energy_target_kw": float(optimal_energy.mean()),
            "q_rec_kw": float(optimal_q_rec.mean()),
        },
        "deviation_end_vs_fixed_point": {
            "absolute": deviation_abs,
            "relative_pct": deviation_rel_pct,
        },
        "cb2_constraint_series_optimal_setpoint": cb2_constraint_series(
            optimal_trajectory, task.critical_moisture_kg_per_kg
        ),
        "cb3": {
            "optimal_batch_mean_energy_target_kw": float(optimal_energy.mean()),
            "fixed_batch_mean_energy_target_kw": float(fixed_energy.mean()),
        },
    }


def compute_cb3_regret(cb1_records: list[dict[str, Any]]) -> dict[str, Any]:
    absolute_kw: list[float] = []
    relative_pct: list[float] = []
    for record in cb1_records:
        optimal = record["cb3"]["optimal_batch_mean_energy_target_kw"]
        fixed = record["cb3"]["fixed_batch_mean_energy_target_kw"]
        regret = fixed - optimal
        absolute_kw.append(regret)
        denom = abs(optimal) if abs(optimal) > 1e-9 else 1e-9
        relative_pct.append(100.0 * regret / denom)
    absolute_arr = np.asarray(absolute_kw)
    relative_arr = np.asarray(relative_pct)
    return {
        "n_points": len(absolute_kw),
        "absolute_kw": {
            "median": float(np.median(absolute_arr)),
            "p95": float(np.percentile(absolute_arr, 95)),
            "max": float(np.max(absolute_arr)),
        },
        "relative_pct": {
            "median": float(np.median(relative_arr)),
            "p95": float(np.percentile(relative_arr, 95)),
            "max": float(np.max(relative_arr)),
        },
    }


# ------------------------------------------------------------------------------- CB4


def _cb4_objective(
    x: np.ndarray,
    t_amb: float,
    phi_amb: float,
    transient_config: ProcessConfig,
    spray_rate: float,
    solid_fraction: float,
    critical_moisture_kg_per_kg: float,
) -> float:
    """Objective function for CB4: batch-averaged energy plus penalty function.

    Penalty function analogous to ``study_operating_point_regret._penalized_energy``, but averaged
    over the ENTIRE trajectory (not only the steady state) -- the search should
    avoid set points that violate on average during the batch, not only at the end point.
    """
    t_in, u_gas, p_nozzle = (float(v) for v in x)
    try:
        trajectory = rollout_constant_setpoint(
            transient_config,
            t_in=t_in,
            u_gas=u_gas,
            p_nozzle=p_nozzle,
            t_amb=t_amb,
            phi_amb=phi_amb,
            spray_rate=spray_rate,
            solid_fraction=solid_fraction,
            n_steps=N_STEPS,
            run_id="cb4_candidate",
        )
    except (RuntimeError, ValueError):
        return _SOLVE_FAILURE_PENALTY_KW

    diagnostics = trajectory.diagnostics
    moisture = diagnostics["particle_moisture_kg_per_kg"][0, :]
    fluid_lower = diagnostics["fluidization_margin_lower_m_per_s"][0, :]
    fluid_upper = diagnostics["fluidization_margin_upper_m_per_s"][0, :]
    bed_temperature = trajectory.states[0, :, 3]
    window = np.maximum(fluid_lower + fluid_upper, 1e-3)

    moisture_violation = np.maximum(0.0, moisture - critical_moisture_kg_per_kg) / max(
        critical_moisture_kg_per_kg, 1e-6
    )
    fluid_lower_violation = np.maximum(0.0, -fluid_lower) / window
    fluid_upper_violation = np.maximum(0.0, -fluid_upper) / window
    bed_temperature_violation = np.maximum(0.0, bed_temperature - BED_TEMPERATURE_LIMIT_C) / 10.0
    mean_violation = float(
        np.mean(
            moisture_violation
            + fluid_lower_violation
            + fluid_upper_violation
            + bed_temperature_violation
        )
    )

    energy = float(trajectory.outputs[0, :, 0].mean())
    return energy + _PENALTY_WEIGHT_KW * mean_violation


def _make_time_callback(deadline: float) -> Any:
    def _callback(intermediate_result: OptimizeResult) -> bool:
        del intermediate_result
        return time.perf_counter() >= deadline

    return _callback


def optimize_cb4_point(
    t_amb: float,
    phi_amb: float,
    bounds: list[tuple[float, float]],
    transient_config: ProcessConfig,
    spray_rate: float,
    solid_fraction: float,
    critical_moisture_kg_per_kg: float,
    workers: int,
) -> dict[str, Any]:
    args = (
        t_amb,
        phi_amb,
        transient_config,
        spray_rate,
        solid_fraction,
        critical_moisture_kg_per_kg,
    )
    start = time.perf_counter()
    deadline = start + _CB4_TIME_LIMIT_S

    de_result = differential_evolution(
        _cb4_objective,
        bounds,
        args=args,
        seed=_CB4_DE_SEED,
        polish=False,
        workers=workers,
        updating="deferred",
        popsize=_CB4_DE_POPSIZE,
        maxiter=_CB4_DE_MAXITER,
        tol=1.0e-4,
        callback=_make_time_callback(deadline),
    )
    de_elapsed = time.perf_counter() - start
    time_budget_hit = de_elapsed >= _CB4_TIME_LIMIT_S or time.perf_counter() >= deadline

    refine_result = de_result
    refine_elapsed = 0.0
    refine_converged = bool(de_result.success)
    if not time_budget_hit:
        refine_start = time.perf_counter()
        remaining = max(0.0, deadline - refine_start)
        # Nelder-Mead has no built-in wall-clock limit -- ``maxiter`` roughly estimated from the
        # remaining time and the measured cost per objective-function call,
        # conservatively (factor 2 safety margin), instead of undermining the 30-min stopping
        # criterion with an unbounded refinement.
        per_eval_s = max(de_elapsed / max(de_result.nfev, 1), 0.5)
        max_refine_evals = max(5, int(remaining / (2.0 * per_eval_s)))
        refine_result = minimize(
            _cb4_objective,
            de_result.x,
            args=args,
            method="Nelder-Mead",
            bounds=bounds,
            options={
                "xatol": 1e-4,
                "fatol": 1e-7,
                "maxiter": max_refine_evals,
                "maxfev": max_refine_evals,
                "adaptive": True,
            },
        )
        refine_elapsed = time.perf_counter() - refine_start
        refine_converged = bool(refine_result.success)

    total_elapsed = time.perf_counter() - start
    best_x = refine_result.x if refine_result.fun <= de_result.fun else de_result.x
    t_in, u_gas, p_nozzle = (float(v) for v in best_x)

    return {
        "ambient_temperature_celsius": t_amb,
        "ambient_relative_humidity": phi_amb,
        "inlet_temperature_celsius": t_in,
        "gas_velocity_m_per_s": u_gas,
        "nozzle_pressure_bar": p_nozzle,
        "batch_mean_energy_plus_penalty_kw": float(min(de_result.fun, refine_result.fun)),
        "de_nfev": int(de_result.nfev),
        "de_elapsed_s": de_elapsed,
        "refine_elapsed_s": refine_elapsed,
        "total_elapsed_s": total_elapsed,
        "time_budget_exceeded": time_budget_hit,
        "de_converged": bool(de_result.success),
        "refine_converged": refine_converged,
        "extrapolated_full_search_s": (
            None
            if not time_budget_hit
            else de_elapsed
            / max(de_result.nfev, 1)
            * _CB4_DE_POPSIZE
            * len(bounds)
            * (_CB4_DE_MAXITER + 1)
        ),
    }


def run_cb4(
    transient_config: ProcessConfig,
    steady_state_base: ProcessConfig,
    spray_rate: float,
    solid_fraction: float,
    critical_moisture_kg_per_kg: float,
    workers: int,
) -> list[dict[str, Any]]:
    disturbances = steady_state_base.disturbances
    t_amb_bounds = (
        disturbances.ambient_temperature_celsius.minimum,
        disturbances.ambient_temperature_celsius.maximum,
    )
    phi_bounds = (
        disturbances.ambient_relative_humidity.minimum,
        disturbances.ambient_relative_humidity.maximum,
    )
    corners = [
        (t_amb_bounds[0], phi_bounds[0]),
        (t_amb_bounds[0], phi_bounds[1]),
        (t_amb_bounds[1], phi_bounds[0]),
        (t_amb_bounds[1], phi_bounds[1]),
    ]
    actions = steady_state_base.actions
    bounds = [
        (actions.inlet_temperature_celsius.minimum, actions.inlet_temperature_celsius.maximum),
        (actions.gas_velocity_m_per_s.minimum, actions.gas_velocity_m_per_s.maximum),
        (actions.nozzle_pressure_bar.minimum, actions.nozzle_pressure_bar.maximum),
    ]

    results = []
    for t_amb, phi_amb in corners:
        print(f"--- CB4: Ecke T_amb={t_amb:.1f} C, phi={phi_amb:.2f} " + "-" * 20, flush=True)
        result = optimize_cb4_point(
            t_amb,
            phi_amb,
            bounds,
            transient_config,
            spray_rate,
            solid_fraction,
            critical_moisture_kg_per_kg,
            workers,
        )
        print(
            f"  -> T_in={result['inlet_temperature_celsius']:.2f} C, "
            f"u_gas={result['gas_velocity_m_per_s']:.3f} m/s, "
            f"p_nozzle={result['nozzle_pressure_bar']:.3f} bar "
            f"({result['total_elapsed_s']:.0f} s, budget_exceeded={result['time_budget_exceeded']})",
            flush=True,
        )
        results.append(result)
    return results


# ------------------------------------------------------------------------------- main


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--regret-json", type=Path, default=REGRET_STUDY_PATH)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--workers", type=int, default=-1, help="-1 = alle Kerne")
    parser.add_argument(
        "--cb4", action="store_true", help="CB4 (Reoptimierung, 4 Eckpunkte) zusaetzlich ausfuehren"
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    _suppress_placeholder_warning()

    env = environment_fingerprint()
    if env["git/dirty"]:
        print(
            "WARNUNG: Arbeitsbaum ist dirty -- git/dirty wird in jeden Record geschrieben, "
            "der Lauf ist trotzdem nicht vollständig reproduzierbar.",
            flush=True,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        steady_state_base = load_process_config(arguments.config)
    if not fingerprints_equivalent(
        steady_state_base.fingerprint, EXPECTED_STEADY_STATE_FINGERPRINT
    ):
        raise RuntimeError(
            f"Beharrungszustands-Config hat Fingerprint {steady_state_base.fingerprint!r}, "
            f"erwartet {EXPECTED_STEADY_STATE_FINGERPRINT!r} -- weicht von Befund 52 ab."
        )
    transient_config = build_transient_config(steady_state_base)
    if not fingerprints_equivalent(transient_config.fingerprint, EXPECTED_TRANSIENT_FINGERPRINT):
        raise RuntimeError(
            f"Transiente Config hat Fingerprint {transient_config.fingerprint!r}, erwartet "
            f"{EXPECTED_TRANSIENT_FINGERPRINT!r}."
        )

    regret_json = json.loads(arguments.regret_json.read_text())
    if not fingerprints_equivalent(
        regret_json["config_fingerprint"], EXPECTED_STEADY_STATE_FINGERPRINT
    ):
        raise RuntimeError(
            f"{arguments.regret_json} hat config_fingerprint "
            f"{regret_json['config_fingerprint']!r}, erwartet {EXPECTED_STEADY_STATE_FINGERPRINT!r}."
        )
    grid_points = load_grid_points(regret_json)
    fixed_point = regret_json["fixed_point"]
    fixed_setpoint = (
        fixed_point["inlet_temperature_celsius"],
        fixed_point["gas_velocity_m_per_s"],
        fixed_point["nozzle_pressure_bar"],
    )

    spray_rate = steady_state_base.actions.spray_rate_kg_per_h.nominal
    solid_fraction = steady_state_base.disturbances.suspension_solid_fraction.nominal
    critical_moisture = steady_state_base.material.critical_moisture_kg_per_kg

    workers = arguments.workers if arguments.workers > 0 else cpu_count()
    print("=" * 78)
    print(f"Beharrungszustands-Config  {steady_state_base.fingerprint}  (Befund 52)")
    print(f"Transiente Config          {transient_config.fingerprint}  (CB1-CB4)")
    print(
        f"Chargendauer               {BATCH_DURATION_S:.0f} s, Delta_t={SAMPLE_TIME_S:.0f} s, "
        f"{N_STEPS} Schritte"
    )
    print(
        f"Fester Betriebspunkt (Befund 52)  T_in={fixed_setpoint[0]:.2f} C, "
        f"u_gas={fixed_setpoint[1]:.3f} m/s, p_nozzle={fixed_setpoint[2]:.3f} bar"
    )
    print(f"git/commit={env['git/commit']}  git/dirty={env['git/dirty']}")
    print(f"Worker                     {workers}")
    print("=" * 78, flush=True)

    tasks = [
        GridPointTask(
            point=point,
            transient_config=transient_config,
            steady_state_base=steady_state_base,
            fixed_setpoint=fixed_setpoint,
            spray_rate=spray_rate,
            solid_fraction=solid_fraction,
            critical_moisture_kg_per_kg=critical_moisture,
        )
        for point in grid_points
    ]

    print(
        f"\n--- CB1/CB2/CB3: {len(tasks)} Gitterpunkte, je 2 Rollouts (optimal + fest) " + "-" * 5,
        flush=True,
    )
    cb1_records_by_index: dict[int, dict[str, Any]] = {}
    start = time.perf_counter()
    with Pool(workers) as pool:
        for n_done, record in enumerate(pool.imap_unordered(_run_grid_point, tasks), start=1):
            index = record["grid_index"]
            elapsed = time.perf_counter() - start
            print(
                f"[{n_done:2d}/{len(tasks)}] Punkt {index + 1}  (kumulativ {elapsed:6.1f} s)",
                flush=True,
            )
            cb1_records_by_index[index] = record
    cb1_records = [cb1_records_by_index[i] for i in range(len(tasks))]

    cb3_regret = compute_cb3_regret(cb1_records)
    print("\n--- CB3: Regret (chargengemittelt) " + "-" * 30)
    print(f"n = {cb3_regret['n_points']}")
    print(
        f"absolut [kW]  Median={cb3_regret['absolute_kw']['median']:.4f}  "
        f"P95={cb3_regret['absolute_kw']['p95']:.4f}  Max={cb3_regret['absolute_kw']['max']:.4f}"
    )
    print(
        f"relativ [%]   Median={cb3_regret['relative_pct']['median']:.3f}  "
        f"P95={cb3_regret['relative_pct']['p95']:.3f}  Max={cb3_regret['relative_pct']['max']:.3f}"
    )
    print(
        "Referenz Befund 52 (Beharrungszustand): Median 16.030 %, P95 22.199 %, Max 25.801 %.",
        flush=True,
    )

    n_moisture_violated = sum(
        1
        for record in cb1_records
        if record["cb2_constraint_series_optimal_setpoint"]["moisture"]["fraction_steps_violating"]
        > 0.0
    )
    print(
        f"\n--- CB2: {n_moisture_violated}/{len(cb1_records)} Gitterpunkte verletzen die "
        "Feuchtegrenze zu irgendeinem Zeitpunkt der Charge (stationäres Optimum als "
        "Sollwert) ---",
        flush=True,
    )

    output: dict[str, Any] = {
        "environment": env,
        "steady_state_config_fingerprint": steady_state_base.fingerprint,
        "transient_config_fingerprint": transient_config.fingerprint,
        "source_regret_study": str(arguments.regret_json),
        "batch_duration_s": BATCH_DURATION_S,
        "sample_time_s": SAMPLE_TIME_S,
        "n_steps": N_STEPS,
        "fixed_setpoint_befund_52": {
            "inlet_temperature_celsius": fixed_setpoint[0],
            "gas_velocity_m_per_s": fixed_setpoint[1],
            "nozzle_pressure_bar": fixed_setpoint[2],
        },
        "cb1_cb2_cb3_per_grid_point": cb1_records,
        "cb3_regret": cb3_regret,
        "cb3_regret_steady_state_reference_befund_52": regret_json["regret"],
    }

    if arguments.cb4:
        print("\n" + "=" * 78)
        print("CB4: Reoptimierung mit chargengemittelter Energie, 4 Eckpunkte")
        print("=" * 78, flush=True)
        output["cb4_reoptimized_corners"] = run_cb4(
            transient_config,
            steady_state_base,
            spray_rate,
            solid_fraction,
            critical_moisture,
            workers,
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nErgebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

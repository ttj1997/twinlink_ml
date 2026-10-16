r"""Exhaust recirculation of one fluidised-bed line: oracle, constraints, heat recovery (docs/22).

Question of the study: how much exhaust can a line recirculate into its inlet air before the
product-moisture limit is violated, and what does that save in energy and CO2? The process model is
the laboratory line of M1 (``greybox_gab_placeholder.yaml``, placeholder physics) with the optional
``OperationConfig.exhaust_recirculation_fraction`` ``r``: the heater takes the mixture of fresh air
and bed exhaust (``thermo.recirculated_inlet_humidity``, ``thermo.mixing_temperature_celsius``).

Oracle (mechanics of M1 / Befund 52, ``simulation.oracle_search.run_de_nm``): per ambient context
the set points ``(T_in, u)`` -- and ``r`` when it is a decision variable -- minimise the CO2 rate
with the gas-fired heater booking of M1 (``carrier_energy``, ``eta_gas`` 0.80, factors 0.366 /
0.201 kg CO2e/kWh), fan power included (main fan of the process model plus the cube-law
recirculation fan, ``EnergyConfig.recirculation_duct_loss_coefficient``). The nozzle pressure is
fixed at the lower box bound 1.0 bar: it acts only on the atomiser power, which is zero for every
pressure up to ambient pressure, so the bound is the exact optimum (M1 optimised it and found
1.00-1.01 bar). Constraints: those of ``operating_point.check_constraints`` (quality threshold,
fluidisation window, bed temperature 80 degC, solver validity) plus fog in the mixing chamber
(``Y_in <= Y_s(T_mix)``), all with the penalty weight of M1 (``carrier_oracle.penalty_weight``).

Heat recovery (:func:`recovery_balance`): the purge ``(1 - r)`` of the bed flow, scaled with the
constant factor ``k`` of M2 (D4), is the source of the condensing cascade of
:mod:`twinlink_ml.heatnetwork` (``dT_min``, ``T_kalt``, ``eta_H``, ``H_u`` as ``SiteSpec`` of M2).
Two layouts of the sink: ``fresh_air`` -- the exchanger preheats the fresh air BEFORE the mixing
point (sink ``(1 - r) k m_L`` from ``T_amb`` to the fresh-air temperature that gives the same heater
duty; by the linearity of the enthalpy the duty is exactly the heater duty of the line) -- and
``mixed_air`` -- the exchanger heats the mixture (sink ``k m_L`` from ``T_mix`` to ``T_in``).
Both are equal at ``r = 0``.

Every number is laboratory scale or linearly scaled laboratory scale with placeholder physics; none
is a plant statement.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Literal

import numpy as np

from twinlink_ml.heatnetwork.balance import (
    SinkState,
    SourceState,
    build_source_profiles,
    evaluate_cascade,
)
from twinlink_ml.heatnetwork.config import (
    DEFAULT_APPROACH_CHECK,
    DEFAULT_APPROACH_PROFILE_NODES,
    ApproachCheck,
    HeatNetworkConfig,
    HeatSinkConfig,
    HeatSourceConfig,
)
from twinlink_ml.heatnetwork.routing import FixedRouting
from twinlink_ml.simulation.backend import NumpyBackend
from twinlink_ml.simulation.config import GasConfig, ProcessConfig, config_fingerprint
from twinlink_ml.simulation.operating_point import check_constraints
from twinlink_ml.simulation.oracle_search import OptimizationBudget, evaluate_point, run_de_nm
from twinlink_ml.simulation.steady_state import SteadyStateSolution
from twinlink_ml.simulation.thermo import saturation_humidity_kg_per_kg

from .carrier_energy import carrier_energy_from_diagnostics
from .carrier_oracle import OracleSpec, penalty_weight, solve_failure_value
from .net_energy import ELECTRICITY_CO2_KG_PER_KWH, GAS_CO2_KG_PER_KWH

__all__ = [
    "FIXED_NOZZLE_PRESSURE_BAR",
    "RecirculationTask",
    "RecoverySpec",
    "evaluate_task",
    "fog_violation",
    "recovery_balance",
    "run_tasks",
    "with_recirculation",
]

FIXED_NOZZLE_PRESSURE_BAR = 1.0
"""Nozzle pressure of every evaluation: the lower box bound, exact optimum (module docstring)."""

FOG_TOLERANCE = 1.0e-4
"""Tolerance of the fog constraint ``Y_in / Y_s(T_mix) - 1`` (relative, dimensionless)."""

Layout = Literal["fresh_air", "mixed_air"]

_BACKEND = NumpyBackend()


@dataclass(frozen=True, slots=True)
class RecoverySpec:
    """Cascade settings of the recovery evaluation (placeholders of M2, ``SiteSpec``)."""

    scale_factor: float
    layout: Layout = "fresh_air"
    dt_min_k: float = 20.0
    t_kalt_celsius: float = 29.1
    heater_efficiency: float = 0.80
    fuel_lower_heating_value_j_per_kg: float = 45.0e6
    approach_check: ApproachCheck = DEFAULT_APPROACH_CHECK
    approach_profile_nodes: int = DEFAULT_APPROACH_PROFILE_NODES

    @property
    def fingerprint(self) -> str:
        return config_fingerprint(dataclasses.asdict(self))


@dataclass(frozen=True, slots=True)
class RecirculationTask:
    """One optimisation: an ambient context, a fixed ``r`` or an ``r`` box, the objective.

    ``fixed_r``: the recirculation fraction of this task (``None`` only together with ``r_box``,
    then ``r`` is the third decision variable). ``fresh_air_path``: evaluate with the field unset
    (the model of M1) instead of ``r = 0.0``; only for the anchor. ``recovery``: include the
    cascade in the objective (the CO2 after recovery is minimised), else the line alone.
    """

    key: str
    context_index: int
    ambient_temperature_celsius: float
    ambient_relative_humidity: float
    bounds_t_in: tuple[float, float]
    bounds_u: tuple[float, float]
    quality_threshold_kg_per_kg: float
    fixed_r: float | None = 0.0
    r_box: tuple[float, float] | None = None
    fresh_air_path: bool = False
    recovery: RecoverySpec | None = None
    gas_heater_efficiency: float = 0.80
    budget: OptimizationBudget | None = None
    config_overrides: tuple[tuple[str, str, float], ...] = ()
    """``(section, field, value)`` applied to the base config of the run (sensitivity variants)."""

    def __post_init__(self) -> None:
        if (self.fixed_r is None) == (self.r_box is None):
            raise ValueError("exactly one of fixed_r and r_box must be given")
        if self.fresh_air_path and self.fixed_r != 0.0:
            raise ValueError("the fresh-air path is the case r = 0")

    @property
    def oracle_spec(self) -> OracleSpec:
        return OracleSpec(
            self.quality_threshold_kg_per_kg, "co2_gas_heater", self.gas_heater_efficiency
        )


def with_recirculation(config: ProcessConfig, r: float | None) -> ProcessConfig:
    """Config with ``exhaust_recirculation_fraction`` set (``None``: unset, the M1 model)."""
    return dataclasses.replace(
        config, operation=dataclasses.replace(config.operation, exhaust_recirculation_fraction=r)
    )


def apply_overrides(
    config: ProcessConfig, overrides: Sequence[tuple[str, str, float]]
) -> ProcessConfig:
    """Replace single fields of config sections (``(section, field, value)`` triples)."""
    for section, name, value in overrides:
        config = dataclasses.replace(
            config, **{section: dataclasses.replace(getattr(config, section), **{name: value})}
        )
    return config


def mixing_saturation_ratio(solution: SteadyStateSolution, gas: GasConfig) -> float:
    """``Y_in / Y_s(T_mix)`` in the mixing chamber (0 without recirculation); above 1: fog."""
    d = solution.diagnostics
    if "mixing_temperature_celsius" not in d:
        return 0.0
    saturation = float(
        saturation_humidity_kg_per_kg(_BACKEND, gas, np.asarray(d["mixing_temperature_celsius"]))
    )
    return float(d["inlet_humidity_kg_per_kg"] / saturation)


def fog_violation(solution: SteadyStateSolution, gas: GasConfig) -> float:
    """``max(0, Y_in / Y_s(T_mix) - 1)``: fog in the mixing chamber."""
    return max(0.0, mixing_saturation_ratio(solution, gas) - 1.0)


# Set per process by the pool initialiser (picklable under the Windows spawn start method).
_BASE_CONFIG: ProcessConfig | None = None


def _init_worker(base_config: ProcessConfig) -> None:
    global _BASE_CONFIG
    _BASE_CONFIG = base_config


def _config_for(task: RecirculationTask, r: float) -> ProcessConfig:
    assert _BASE_CONFIG is not None
    config = apply_overrides(_BASE_CONFIG, task.config_overrides)
    return with_recirculation(config, None if task.fresh_air_path else r)


def _evaluate(task: RecirculationTask, t_in: float, u: float, r: float) -> SteadyStateSolution:
    return evaluate_point(
        t_in,
        u,
        FIXED_NOZZLE_PRESSURE_BAR,
        task.ambient_temperature_celsius,
        task.ambient_relative_humidity,
        max_iterations=task.oracle_spec.search_max_iterations,
        tolerance=task.oracle_spec.search_tolerance,
        base_config=_config_for(task, r),
    )


def recovery_balance(
    solution: SteadyStateSolution,
    config: ProcessConfig,
    spec: RecoverySpec,
    *,
    t_amb: float,
    t_in: float,
) -> dict[str, Any]:
    """Cascade of the line's purge into its own inlet air; CO2 after recovery (scaled by ``k``).

    Returns the recovered power, the fuel power of the heater, the electric power and the CO2
    rate, all scaled by ``spec.scale_factor``; ``rejected`` if the cascade refuses the stage.
    """
    d = solution.diagnostics
    k = spec.scale_factor
    m_bed = k * solution.dry_air_mass_flow_kg_per_s
    m_purge = k * solution.exhaust_dry_air_mass_flow_kg_per_s
    y_fresh = float(d.get("fresh_air_humidity_kg_per_kg", d["inlet_humidity_kg_per_kg"]))
    t_mix = float(d.get("mixing_temperature_celsius", t_amb))
    duty_kw = k * float(d["power_heater_kw"]) * config.energy.heater_efficiency
    electric_kw = k * float(d["power_fan_kw"] + d["power_pump_kw"] + d["power_atomiser_kw"])
    gas = config.gas
    result: dict[str, Any] = {
        "layout": spec.layout,
        "duty_kw": duty_kw,
        "electric_kw": electric_kw,
        "purge_dry_air_kg_per_s": m_purge,
        "rejected": False,
    }
    if spec.layout == "fresh_air":
        c_fresh = gas.cp_dry_air_j_per_kg_k + y_fresh * gas.cp_vapour_j_per_kg_k
        sink = HeatSinkConfig(
            "SA",
            m_purge,
            y_fresh,
            t_amb,
            t_amb + duty_kw * 1000.0 / (m_purge * c_fresh),
        )
    else:
        sink = HeatSinkConfig("SA", m_bed, float(d["inlet_humidity_kg_per_kg"]), t_mix, t_in)
    source = HeatSourceConfig(
        "LA",
        m_purge,
        solution.bed_temperature_celsius - config.exhaust.outlet_duct_heat_loss_k,
        solution.outlet_humidity_kg_per_kg,
    )
    network = HeatNetworkConfig(
        sources=(source,),
        sinks=(sink,),
        minimum_pinch_k=spec.dt_min_k,
        recovery_mode="condensing",
        cold_temperature_celsius=spec.t_kalt_celsius,
        fuel_lower_heating_value_j_per_kg=spec.fuel_lower_heating_value_j_per_kg,
        heater_efficiency=spec.heater_efficiency,
        approach_check=spec.approach_check,
        approach_profile_nodes=spec.approach_profile_nodes,
    )
    sources = (SourceState.from_config(source),)
    try:
        stages = FixedRouting((("LA", "SA"),)).route(network)
        profiles = (
            build_source_profiles(network, sources)
            if spec.approach_check == "along_exchanger"
            else None
        )
        cascade = evaluate_cascade(
            network, stages, sources, (SinkState.from_config(sink),), (), profiles
        )
    except (ValueError, RuntimeError) as exc:
        result.update({"rejected": True, "reason": str(exc)})
        return result
    gas_kw = float(cascade.fuel_mass_flow_kg_per_s[0, 0]) * spec.fuel_lower_heating_value_j_per_kg
    gas_kw /= 1000.0
    recovered_kw = float(np.sum(cascade.stage_recovered_power_kw[:, 0]))
    result.update(
        {
            "recovered_kw": recovered_kw,
            "gas_kw": gas_kw,
            "gas_kw_without_recovery": duty_kw / spec.heater_efficiency,
            "co2_kg_per_h": ELECTRICITY_CO2_KG_PER_KWH * electric_kw + GAS_CO2_KG_PER_KWH * gas_kw,
            "co2_without_recovery_kg_per_h": ELECTRICITY_CO2_KG_PER_KWH * electric_kw
            + GAS_CO2_KG_PER_KWH * duty_kw / spec.heater_efficiency,
            "kwh_sum_kw": electric_kw + gas_kw,
            "source_dew_point_relevant_outlet_celsius": float(
                cascade.source_outlet_temperature_celsius[0, 0]
            ),
            "condensate_kg_per_s": float(cascade.source_condensate_kg_per_s[0, 0]),
            "binding_code": int(cascade.stage_binding_code[0, 0]),
            "min_approach_k": float(cascade.stage_min_approach_k[0, 0]),
        }
    )
    return result


def _assessment(
    task: RecirculationTask, solution: SteadyStateSolution, t_in: float, u: float, r: float
) -> tuple[float, float, dict[str, Any]]:
    """Objective, normalised violation and report fields of one evaluated point."""
    config = _config_for(task, r)
    spec = task.oracle_spec
    check = check_constraints(
        solution,
        critical_moisture_kg_per_kg=spec.quality_threshold_kg_per_kg,
        gas_velocity_m_per_s=u,
    )
    ratio = mixing_saturation_ratio(solution, config.gas)
    fog = max(0.0, ratio - 1.0)
    violation = check.normalized_violation + (fog if fog > FOG_TOLERANCE else 0.0)
    carrier = carrier_energy_from_diagnostics(
        solution.diagnostics,
        heater="gas",
        process_heater_efficiency=config.energy.heater_efficiency,
        gas_heater_efficiency=spec.gas_heater_efficiency,
    )
    report: dict[str, Any] = {
        "co2_kg_per_h": carrier.co2_kg_per_h,
        "kwh_sum_kw": carrier.kwh_sum_kw,
        "gas_kw": carrier.gas_kw,
        "electric_kw": carrier.electric_kw,
        "fog_violation": fog,
        "mixing_saturation_ratio": ratio,
        "constraint_check": check.as_dict(),
    }
    objective = carrier.co2_kg_per_h
    if task.recovery is not None:
        balance = recovery_balance(
            solution, config, task.recovery, t_amb=task.ambient_temperature_celsius, t_in=t_in
        )
        report["recovery"] = balance
        if balance["rejected"]:
            return solve_failure_value(spec), violation, report
        # the recovery objective is in the units of the scaled site; the penalty is scaled alike
        objective = balance["co2_kg_per_h"] / task.recovery.scale_factor
    return objective, violation, report


def _decode(task: RecirculationTask, x: np.ndarray) -> tuple[float, float, float]:
    t_in, u = float(x[0]), float(x[1])
    r = float(x[2]) if task.fixed_r is None else task.fixed_r
    return t_in, u, r


def _search_objective(x: np.ndarray, task: RecirculationTask) -> float:
    t_in, u, r = _decode(task, x)
    try:
        solution = _evaluate(task, t_in, u, r)
    except (RuntimeError, ValueError):
        return solve_failure_value(task.oracle_spec)
    objective, violation, _ = _assessment(task, solution, t_in, u, r)
    return float(objective + penalty_weight(task.oracle_spec) * violation)


def _report_solution(solution: SteadyStateSolution, r: float) -> dict[str, Any]:
    d = solution.diagnostics
    return {
        "bed_temperature_celsius": solution.bed_temperature_celsius,
        "particle_moisture_kg_per_kg": solution.particle_moisture_kg_per_kg,
        "inlet_humidity_kg_per_kg": d["inlet_humidity_kg_per_kg"],
        "outlet_humidity_kg_per_kg": solution.outlet_humidity_kg_per_kg,
        "fresh_air_humidity_kg_per_kg": d.get(
            "fresh_air_humidity_kg_per_kg", d["inlet_humidity_kg_per_kg"]
        ),
        "dew_point_celsius": solution.dew_point_celsius,
        "mixing_temperature_celsius": d.get("mixing_temperature_celsius"),
        "dry_air_mass_flow_kg_per_s": solution.dry_air_mass_flow_kg_per_s,
        "exhaust_dry_air_mass_flow_kg_per_s": solution.exhaust_dry_air_mass_flow_kg_per_s,
        "power_heater_kw": d["power_heater_kw"],
        "power_fan_kw": d["power_fan_kw"],
        "power_recirculation_fan_kw": d.get("power_recirculation_fan_kw", 0.0),
        "power_pump_kw": d["power_pump_kw"],
        "power_atomiser_kw": d["power_atomiser_kw"],
        "exhaust_sensible_kw": solution.exhaust_sensible_kw,
        "exhaust_latent_kw": solution.exhaust_latent_kw,
        "saturation_degree": d["saturation_degree"],
        "residual_inf_norm": solution.residual_inf_norm,
        "solver_feasible": solution.feasible,
        "infeasibility_reasons": list(solution.infeasibility_reasons),
        "recirculation_fraction": r,
    }


def evaluate_task(task: RecirculationTask) -> dict[str, Any]:
    """Run one optimisation (single-threaded; the pool distributes tasks)."""
    assert task.budget is not None
    start = time.perf_counter()
    bounds = [task.bounds_t_in, task.bounds_u]
    if task.r_box is not None:
        bounds.append(task.r_box)
    de_result, refine_result, de_elapsed, refine_elapsed = run_de_nm(
        _search_objective, bounds, (task,), task.budget, de_workers=1
    )
    best_x = refine_result.x if refine_result.fun <= de_result.fun else de_result.x
    t_in, u, r = _decode(task, np.asarray(best_x))
    result: dict[str, Any] = {
        "key": task.key,
        "context_index": task.context_index,
        "ambient_temperature_celsius": task.ambient_temperature_celsius,
        "ambient_relative_humidity": task.ambient_relative_humidity,
        "inlet_temperature_celsius": t_in,
        "gas_velocity_m_per_s": u,
        "recirculation_fraction": r,
        "nozzle_pressure_bar": FIXED_NOZZLE_PRESSURE_BAR,
        "search_objective": float(min(refine_result.fun, de_result.fun)),
        "de_converged": bool(de_result.success),
        "de_nfev": int(de_result.nfev),
        "refine_converged": bool(refine_result.success),
        "de_elapsed_s": de_elapsed,
        "refine_elapsed_s": refine_elapsed,
        "solve_failed": False,
    }
    try:
        solution = _evaluate(task, t_in, u, r)
    except (RuntimeError, ValueError) as exc:
        result.update({"solve_failed": True, "reason": str(exc), "feasible": False})
        result["wall_elapsed_s"] = time.perf_counter() - start
        return result
    objective, violation, report = _assessment(task, solution, t_in, u, r)
    check = report["constraint_check"]
    result.update(report)
    result.update(_report_solution(solution, r))
    result.update(
        {
            "objective": objective,
            "normalized_violation": violation,
            "feasible": bool(check["feasible"] and report["fog_violation"] <= FOG_TOLERANCE),
            "binding": _binding(task, check, report["mixing_saturation_ratio"], t_in, u, r),
            "wall_elapsed_s": time.perf_counter() - start,
        }
    )
    return result


def _binding(
    task: RecirculationTask,
    check: dict[str, Any],
    mixing_ratio: float,
    t_in: float,
    u: float,
    r: float,
) -> dict[str, bool]:
    """Which limits are active at the reported point (tolerances of ``operating_point``)."""
    return {
        "moisture": bool(check["binding_moisture"]),
        "gas_velocity_box_min": abs(u - task.bounds_u[0]) <= 1.0e-3,
        "gas_velocity_u_mf": bool(check["binding_gas_lower"]),
        "gas_velocity_u_terminal": bool(check["binding_gas_upper"]),
        "bed_temperature": bool(check["binding_bed_temperature"]),
        "inlet_temperature_box_min": abs(t_in - task.bounds_t_in[0]) <= 0.05,
        "inlet_temperature_box_max": abs(t_in - task.bounds_t_in[1]) <= 0.05,
        "fog": mixing_ratio >= 1.0 - 1.0e-3,
        "recirculation_box_max": task.r_box is not None and abs(r - task.r_box[1]) <= 1.0e-3,
    }


def _task_worker(task: RecirculationTask) -> dict[str, Any]:
    return evaluate_task(task)


def run_tasks(
    base_config: ProcessConfig,
    tasks: Sequence[RecirculationTask],
    *,
    workers: int,
    cache_path: Path,
    label: str,
) -> list[dict[str, Any]]:
    """Run every task not yet in ``cache_path`` (JSON lines, resumable); return all, by key order.

    The cache lives OUTSIDE the repository (it is a run log, not an artifact); the artifact is
    assembled from it after the last task.
    """
    done: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done[record["key"]] = record
    keys = [t.key for t in tasks]
    if len(set(keys)) != len(keys):
        raise ValueError("task keys must be unique")
    todo = [t for t in tasks if t.key not in done]
    print(f"=== {label}: {len(tasks)} tasks, {len(todo)} to run ===", flush=True)
    start = time.perf_counter()
    if todo:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            Pool(workers, initializer=_init_worker, initargs=(base_config,)) as pool,
            cache_path.open("a", encoding="utf-8") as handle,
        ):
            for n, result in enumerate(pool.imap_unordered(_task_worker, todo), start=1):
                handle.write(json.dumps(result, default=_json_default) + "\n")
                handle.flush()
                done[result["key"]] = result
                print(
                    f"  [{n}/{len(todo)}] {result['key']} ({time.perf_counter() - start:.0f} s)",
                    flush=True,
                )
    return [done[k] for k in keys]


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not serialisable: {type(value)}")

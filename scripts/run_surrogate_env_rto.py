r"""Controller C2 (RTO), controls and diagnostics of the study docs/30_surrogate_env.md.

Pre-registration of the study (environments, controllers, metrics, reading): docstring of
``scripts/analyze_surrogate_env.py``, committed before the first run.

Parts (each writes its own artifact; an existing artifact is never overwritten):

``rto``        C2 on every environment: U0 (grey-box steady state) and U1 / U2 / U3 of the master
               seeds 1-3; margin factors 5 (main, the margin of C1) and 1 (sensitivity, no reading);
               the 25 Befund-52 grid points (test set). Optimiser: ``surrogate_env.rto``
               (multistart Nelder-Mead, 8 starts) on the penalised objective of the oracle
               (``energy_target`` + 50 kW x normalised violation against ``0.15 - k m``; a point
               without steady state / valid fixed point gets the solver-failure value 1e6 kW of the
               bandit, i.e. it is not admissible). Nozzle pressure fixed as in the bandit. Every
               result is evaluated on the grey-box model against the TRUE limit 0.15; the prediction
               of the environment at the result is stored next to it (exploitation gap).
               -> ``surrogate_env_rto_v1``
``check``      (a) the oracle of Befund 96 at k = 5 recomputed at 10 of the 25 grid points
               (``ORACLE_CHECK_INDICES``) with the unchanged machinery of
               ``run_bres_margin_references.py``, compared with ``bres_margin_ref_oracle_k5``;
               (b) U0' against U0: the grey-box steady state of every grid point at the true optimum
               (``s05_limit015_oracle_and_fixed``) and at the fixed point, simulated 900 s (h36) with
               constant inputs -- the fixed point of the h36 map of the grey-box model is its steady
               state, so the drift must be at solver precision.  -> ``surrogate_env_oracle_check_v1``
``errormap``   every learned model on the 25 grid points x a 7 x 7 grid of the residual box of C1
               (+-15 K, +-0.425 m/s around the fixed point, clipped to the box) against the grey-box
               steady state; fixed-point statuses counted.  -> ``surrogate_env_error_map_v1``

    tools/run_here.sh scripts/run_surrogate_env_rto.py --part rto --workers 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import run_rl_bandit_residual as rr  # noqa: E402
import run_surrogate_env_rl as senv  # noqa: E402

REPO_ROOT = rr.REPO_ROOT
RUNS = rr.RUNS

# --- fixed before the runs ---------------------------------------------------------------------
MARGIN_FACTORS = (5.0, 1.0)
LEARNED_ENVIRONMENTS = ("U1", "U2", "U3")
REPLICATE_SEEDS = (1, 2, 3)
ORACLE_CHECK_INDICES = (0, 2, 5, 7, 10, 12, 15, 17, 20, 24)
ERROR_MAP_SIZE = 7
U0_PRIME_STEPS_25S = 36
# -----------------------------------------------------------------------------------------------

_MODELS: dict[tuple[str, int], Any] = {}


def _bandit_env(limit: float) -> Any:
    return rr.OperatingPointBanditEnv(rr.BanditEnvConfig(product_moisture_limit_kg_per_kg=limit))


def _model(env_name: str, seed: int) -> Any:
    key = (env_name, seed)
    if key not in _MODELS:
        _MODELS[key] = senv.build_model(env_name, seed, senv.fixed_inputs())
    return _MODELS[key]


def _truth(t_in: float, u_gas: float, t_amb: float, phi: float) -> dict[str, Any]:
    env = rr.eval_env()
    _, info = env.evaluate_operating_point(
        inlet_temperature_celsius=t_in,
        gas_velocity_m_per_s=u_gas,
        ambient_temperature_celsius=t_amb,
        ambient_relative_humidity=phi,
    )
    check = info.get("constraint_check", {})
    return {
        "solve_failed": bool(info.get("solve_failed")),
        "feasible": bool(info["feasible"]),
        "energy_kw": float(info.get("energy_target", np.nan)),
        "moisture_kg_per_kg": float(check.get("moisture_kg_per_kg", np.nan)),
        "bed_temperature_celsius": float(check.get("bed_temperature_celsius", np.nan)),
    }


def rto_task(env_name: str, seed: int, factor: float, index: int) -> dict[str, Any]:
    import torch

    from twinlink_ml.rl.env import _denormalise  # noqa: F401  (import check of the bandit module)
    from twinlink_ml.simulation.operating_point import PENALTY_WEIGHT_KW, SOLVE_FAILURE_PENALTY_KW
    from twinlink_ml.surrogate_env import PointEvaluation, multistart_minimize

    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    points = rr.load_test_points(rr.REFERENCE_015)
    point = points[index]
    t_amb, phi = point.ambient_temperature_celsius, point.ambient_relative_humidity
    limit = rr.LIMIT_KG_PER_KG - factor * rr.MARGIN_M
    env = _bandit_env(limit)
    bounds = [
        (env._inlet_bound.minimum, env._inlet_bound.maximum),
        (env._gas_bound.minimum, env._gas_bound.maximum),
    ]
    model = None if env_name == "U0" else _model(env_name, seed)

    def evaluate(t_in: float, u_gas: float) -> PointEvaluation:
        if model is None:
            reward, info = env.evaluate_operating_point(
                inlet_temperature_celsius=t_in,
                gas_velocity_m_per_s=u_gas,
                ambient_temperature_celsius=t_amb,
                ambient_relative_humidity=phi,
            )
            if info.get("solve_failed"):
                return PointEvaluation(-reward, np.nan, np.nan, np.nan, False)
            check = info["constraint_check"]
            return PointEvaluation(
                -reward,
                float(info["energy_target"]),
                float(check["moisture_kg_per_kg"]),
                float(check["bed_temperature_celsius"]),
                True,
            )
        prediction = model.predict(np.array([[t_in, u_gas]]), np.array([[t_amb, phi]])).point(0)
        if not prediction["valid"]:
            return PointEvaluation(SOLVE_FAILURE_PENALTY_KW, np.nan, np.nan, np.nan, False)
        moisture = prediction["moisture_for_constraint_kg_per_kg"]
        bed = prediction["bed_temperature_celsius"]
        violation = max(0.0, moisture - limit) / limit + max(0.0, bed - 80.0) / 10.0
        return PointEvaluation(
            prediction["energy_kw"] + PENALTY_WEIGHT_KW * violation,
            prediction["energy_kw"],
            moisture,
            bed,
            True,
        )

    start = time.perf_counter()
    result = multistart_minimize(evaluate, bounds, rr.fixed_point_from_reference())
    elapsed = time.perf_counter() - start
    t_in, u_gas = result["inlet_temperature_celsius"], result["gas_velocity_m_per_s"]
    row: dict[str, Any] = {
        "env": env_name,
        "replicate_master_seed": seed if env_name != "U0" else None,
        "margin_factor": factor,
        "tightened_limit_kg_per_kg": limit,
        "grid_index": index,
        "ambient_temperature_celsius": t_amb,
        "ambient_relative_humidity": phi,
        "result": result,
        "truth": _truth(t_in, u_gas, t_amb, phi),
        "energy_oracle_limit015_kw": point.energy_oracle_kw,
        "energy_fixed_kw": point.energy_fixed_kw,
        "elapsed_s": elapsed,
    }
    if model is not None:
        prediction = model.predict(np.array([[t_in, u_gas]]), np.array([[t_amb, phi]]))
        row["predicted"] = prediction.point(0)
        row["predicted_status_counts"] = prediction.status_counts
    return row


def run_rto(workers: int) -> dict[str, Any]:
    tasks = [
        (env_name, seed, factor, index)
        for env_name in ("U3", "U0", "U2", "U1")
        for seed in (REPLICATE_SEEDS if env_name != "U0" else (0,))
        for factor in MARGIN_FACTORS
        for index in range(25)
    ]
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(rto_task, *task): task for task in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if done % 10 == 0:
                print(
                    f"  rto [{done}/{len(tasks)}] ({time.perf_counter() - start:.0f} s)", flush=True
                )
    rows.sort(
        key=lambda r: (
            r["env"],
            r["replicate_master_seed"] or 0,
            -r["margin_factor"],
            r["grid_index"],
        )
    )
    return {"tasks": rows, "n_tasks": len(rows), "multistart": _multistart_settings()}


def _multistart_settings() -> dict[str, Any]:
    import dataclasses

    from twinlink_ml.surrogate_env import MultistartConfig

    return dataclasses.asdict(MultistartConfig())


def run_check(workers: int) -> dict[str, Any]:
    import run_m1_margin_cost as m1

    from twinlink_ml.evaluation.carrier_oracle import OracleSpec, run_oracle_grid
    from twinlink_ml.simulation import GreyBoxFluidBedGranulator
    from twinlink_ml.simulation.operating_point import (
        DEFAULT_SEARCH_MAX_ITERATIONS,
        DEFAULT_SEARCH_TOLERANCE,
        build_variant_config,
    )
    from twinlink_ml.simulation.oracle_search import action_bounds, build_grid
    from twinlink_ml.simulation.steady_state import solve_steady_state

    reference = json.loads((RUNS / "bres_margin_ref_oracle_k5.json").read_text(encoding="utf-8"))
    config = m1._load_config()
    budget = m1._budget(reference["budget"]["workers"])
    grid = build_grid(config, 5)
    bounds = action_bounds(config)
    spec = OracleSpec(
        reference["tightened_limit_kg_per_kg"], "legacy_power_kw", m1.GAS_HEATER_EFFICIENCY
    )
    subset = [grid[i] for i in ORACLE_CHECK_INDICES]
    results = run_oracle_grid(
        config, subset, bounds, budget, spec, workers=workers, label="oracle check k=5"
    )
    comparison = []
    for index, new in zip(ORACLE_CHECK_INDICES, results, strict=True):
        old = reference["results"][index]
        keys = (
            "inlet_temperature_celsius",
            "gas_velocity_m_per_s",
            "nozzle_pressure_bar",
            "objective",
            "legacy_energy_target_kw",
            "particle_moisture_kg_per_kg",
        )
        comparison.append(
            {
                "grid_index": index,
                **{f"abs_diff/{k}": abs(float(new[k]) - float(old[k])) for k in keys},
                "identical": all(float(new[k]) == float(old[k]) for k in keys),
            }
        )

    # U0' against U0
    base = rr.eval_env()._base_config
    s05 = json.loads(rr.REFERENCE_015.read_text(encoding="utf-8"))
    fixed = s05["fixed_point"]
    drift = []
    spray = base.actions.spray_rate_kg_per_h.nominal
    solids = base.disturbances.suspension_solid_fraction.nominal
    steps_5s = round(U0_PRIME_STEPS_25S * 25.0 / base.numerics.sample_time_s)
    for g in s05["grid_results"]:
        t_amb, phi = g["ambient_temperature_celsius"], g["ambient_relative_humidity"]
        for label, t_in, u_gas in (
            ("true_optimum", g["inlet_temperature_celsius"], g["gas_velocity_m_per_s"]),
            ("fixed_point", fixed["inlet_temperature_celsius"], fixed["gas_velocity_m_per_s"]),
        ):
            variant = build_variant_config(
                base,
                t_in=t_in,
                u_gas=u_gas,
                p_nozzle=senv.FIXED_NOZZLE_PRESSURE_BAR,
                t_amb=t_amb,
                phi_amb=phi,
            )
            sim = GreyBoxFluidBedGranulator(variant)
            sol = solve_steady_state(
                sim,
                max_iterations=DEFAULT_SEARCH_MAX_ITERATIONS,
                tolerance=DEFAULT_SEARCH_TOLERANCE,
            )
            action_values = {
                "inlet_temperature_celsius": t_in,
                "gas_velocity_m_per_s": u_gas,
                "spray_rate_kg_per_h": spray,
                "nozzle_pressure_bar": senv.FIXED_NOZZLE_PRESSURE_BAR,
            }
            disturbance_values = {
                "ambient_temperature_celsius": t_amb,
                "ambient_relative_humidity": phi,
                "suspension_solid_fraction": solids,
            }
            action = np.array([[action_values[n] for n in base.actions.names]])
            disturbance = np.array([[disturbance_values[n] for n in base.disturbances.names]])
            state = np.asarray(sol.state)[None, :]
            for _ in range(steps_5s):
                out = sim.step(state, action, disturbance)
                state = out.state
            outputs = np.asarray(out.outputs)[0]
            drift.append(
                {
                    "ambient_temperature_celsius": t_amb,
                    "ambient_relative_humidity": phi,
                    "action": label,
                    "steady_energy_kw": float(sol.diagnostics["energy_target"]),
                    "steady_moisture_kg_per_kg": sol.particle_moisture_kg_per_kg,
                    "steady_bed_temperature_celsius": sol.bed_temperature_celsius,
                    "h36_energy_kw": float(outputs[0]),
                    "h36_moisture_kg_per_kg": float(outputs[2]),
                    "h36_bed_temperature_celsius": float(outputs[1]),
                }
            )
    worst = {
        key: max(abs(d[f"h36_{key}"] - d[f"steady_{key}"]) for d in drift)
        for key in ("energy_kw", "moisture_kg_per_kg", "bed_temperature_celsius")
    }
    return {
        "oracle_check": {
            "indices": list(ORACLE_CHECK_INDICES),
            "reference_artifact": "bres_margin_ref_oracle_k5",
            "reference_commit": reference["provenance"].get("git/commit"),
            "results": results,
            "comparison": comparison,
            "n_identical": sum(c["identical"] for c in comparison),
        },
        "u0_prime": {
            "steps_of_25_s": U0_PRIME_STEPS_25S,
            "internal_steps": steps_5s,
            "rows": drift,
            "max_abs_drift": worst,
        },
    }


def errormap_task(env_name: str, seed: int) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    model = _model(env_name, seed)
    actions, contexts, _ = _error_map_grid()
    start = time.perf_counter()
    prediction = model.predict(actions, contexts)
    return {
        "env": env_name,
        "replicate_master_seed": seed,
        "energy_kw": prediction.energy_kw.tolist(),
        "moisture_kg_per_kg": prediction.moisture_kg_per_kg.tolist(),
        "moisture_for_constraint_kg_per_kg": prediction.moisture_for_constraint_kg_per_kg.tolist(),
        "bed_temperature_celsius": prediction.bed_temperature_celsius.tolist(),
        "valid": prediction.valid.tolist(),
        "status_counts": prediction.status_counts,
        "elapsed_s": time.perf_counter() - start,
    }


def _error_map_grid() -> tuple[np.ndarray, np.ndarray, Any]:
    env = rr.eval_env()
    t_fix, u_fix = rr.fixed_point_from_reference()
    half_t = rr.RESIDUAL_SCALE * 0.5 * (env._inlet_bound.maximum - env._inlet_bound.minimum)
    half_u = rr.RESIDUAL_SCALE * 0.5 * (env._gas_bound.maximum - env._gas_bound.minimum)
    ts = np.clip(
        np.linspace(t_fix - half_t, t_fix + half_t, ERROR_MAP_SIZE),
        env._inlet_bound.minimum,
        env._inlet_bound.maximum,
    )
    us = np.clip(
        np.linspace(u_fix - half_u, u_fix + half_u, ERROR_MAP_SIZE),
        env._gas_bound.minimum,
        env._gas_bound.maximum,
    )
    points = rr.load_test_points(rr.REFERENCE_015)
    actions, contexts = [], []
    for point in points:
        for t in ts:
            for u in us:
                actions.append((t, u))
                contexts.append(
                    (point.ambient_temperature_celsius, point.ambient_relative_humidity)
                )
    return np.array(actions), np.array(contexts), points


def _truth_task(chunk: list[tuple[float, float, float, float]]) -> list[dict[str, Any]]:
    warnings.filterwarnings("ignore")
    return [_truth(*row) for row in chunk]


def run_errormap(workers: int) -> dict[str, Any]:
    actions, contexts, _ = _error_map_grid()
    rows = [(a[0], a[1], c[0], c[1]) for a, c in zip(actions, contexts, strict=True)]
    chunks = [rows[i : i + 49] for i in range(0, len(rows), 49)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        truth = [t for part in pool.map(_truth_task, chunks) for t in part]
        models = list(
            pool.map(
                errormap_task,
                [e for e in LEARNED_ENVIRONMENTS for _ in REPLICATE_SEEDS],
                [s for _ in LEARNED_ENVIRONMENTS for s in REPLICATE_SEEDS],
            )
        )
    return {
        "grid": {"actions": actions.tolist(), "contexts": contexts.tolist()},
        "truth": truth,
        "models": models,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=("rto", "check", "errormap"), required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke", action="store_true", help="two tasks, no artifact")
    args = parser.parse_args()

    from _common import provenance, require_clean_tree, write_artifact

    from twinlink_ml.utils.code_location import require_code_location

    warnings.filterwarnings("ignore")
    require_code_location(REPO_ROOT)
    name = {
        "rto": "surrogate_env_rto_v1",
        "check": "surrogate_env_oracle_check_v1",
        "errormap": "surrogate_env_error_map_v1",
    }[args.part]
    if (RUNS / f"{name}.json").exists():
        raise SystemExit(f"{name}.json exists -- artifacts are never overwritten")
    if args.smoke:
        print(json.dumps(rto_task("U0", 0, 5.0, 12), default=str)[:1500])
        print(json.dumps(rto_task("U2", 1, 5.0, 12), default=str)[:1500])
        return 0
    exempt = [*senv.study_output_paths(), RUNS / "surrogate_env_error_map_v1.json"]
    require_clean_tree(exempt)
    prov = provenance("not used", ignore_paths=exempt)
    start = time.perf_counter()
    payload = {"rto": run_rto, "check": run_check, "errormap": run_errormap}[args.part](
        args.workers
    )
    payload = {
        "_provenance": prov,
        "study": "learned sequence models as environments (docs/30_surrogate_env.md)",
        "test_set": "25 Befund-52 grid points (ambient temperature x relative humidity); truth = "
        "grey-box steady state against the limit 0.15",
        "placeholders": "quality limit 0.15 kg/kg; gas velocity lower bound 0.80 m/s; margin m of "
        "Befund 65 FA2; penalty weight 50 kW",
        "margin_m_kg_per_kg": rr.MARGIN_M,
        **payload,
        "runtime_s": time.perf_counter() - start,
    }
    write_artifact(name, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

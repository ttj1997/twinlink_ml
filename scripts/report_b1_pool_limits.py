r"""B1: limiting of the requested actions per pool, with the gas-velocity cap of preset version 2.

Before the main run (decision of the project lead, 2026-10-13): all plans B0-B4 and the candidate pool
of B5 keep the per-run cap of the gas velocity (``u_t`` at the initial diameter, Befund 86); the
projection is on as a counted guard. This script builds, for the ten master seeds of the main run, the
pools the main run will use (B0-B4: 32 runs each; B5: the candidate pool of 128 runs, simulated here for
counting only -- the acquisition never sees these outputs), simulates them as the main run does (preset
``DataSpec.new_study()``, three targets, guarded config) and reports per pool and seed:

* ``projected_steps`` / ``runs_with_projection`` (expected 0, counted, never assumed);
* the equal-cost check (``check_equal_cost``: duration, bounds, rate limits, disturbances, initial states,
  per-run caps; B0-B4, one seed at a time) -- violations are listed, not fixed;
* per plan and channel: the share of steps at which the rate limiter of the design changed the signal
  (``ratelim_changed``), the share on a box bound (``at_lower`` / ``at_upper``) and, for the gas velocity,
  the share on the per-run cap (``at_cap``). These shares are properties of the plans (decision O7:
  accepted and reported per plan).

    tools/run_here.sh scripts/report_b1_pool_limits.py --workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
from _common import (
    CONFIG_PATH,
    RUNS_DIR,
    provenance,
    require_clean_tree,
    write_artifact,
)

from twinlink_ml.evaluation.b1_learning_curve import candidate_pool
from twinlink_ml.evaluation.experiments import DataSpec
from twinlink_ml.evaluation.growing_pool import (
    GrowingPool,
    guarded_config,
    prepare_process_config,
    projection_counts,
)
from twinlink_ml.simulation import (
    PlaceholderConfigWarning,
    excitation,
    excitation_plans,
    load_process_config,
)
from twinlink_ml.simulation.excitation_plans import (
    PLAN_NAMES,
    check_equal_cost,
    make_plan,
)
from twinlink_ml.utils.seeding import SeedBundle

TARGETS = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
N_POOL = 32
_STATE: dict[str, Any] = {}


def _init() -> None:
    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = guarded_config(load_process_config(CONFIG_PATH))
    spec = DataSpec.new_study(n_train_max=N_POOL, n_val=0, n_test=0, target_names=TARGETS)
    _STATE.update(config=config, spec=spec)


def _task(task: tuple[int, str]) -> dict[str, Any]:
    seed_id, pool_name = task
    config, spec = _STATE["config"], _STATE["spec"]
    prepared = prepare_process_config(config, spec)
    seeds = SeedBundle.from_master(seed_id)
    dt = spec.sample_time_s
    recorded: list[tuple[np.ndarray, np.ndarray, float]] = []
    original = excitation.apply_rate_limit

    def record(signal: np.ndarray, rate: float, step: float) -> np.ndarray:
        out = original(signal, rate, step)
        recorded.append((signal.copy(), out.copy(), rate))
        return out

    excitation.apply_rate_limit = record
    excitation_plans.apply_rate_limit = record
    try:
        if pool_name == "b5_candidates":
            schedule = candidate_pool(config, spec, seeds.data_seed, per_plan=32)
        else:
            plan = make_plan(
                pool_name,
                prepared,
                sample_time_s=dt,
                run_duration_s=spec.run_duration_s,
                solid_fraction_process=spec.solid_fraction_process,
                gas_velocity_cap=spec.gas_velocity_cap,
            )
            schedule = plan.build(N_POOL, seed=seeds.data_seed)
    finally:
        excitation.apply_rate_limit = original
        excitation_plans.apply_rate_limit = original
    pool = GrowingPool(config, spec, schedule, noise_seed=seeds.noise_seed)
    pool.add(list(range(schedule.n_runs)))
    trajectory = pool.trajectory()
    result: dict[str, Any] = {
        "seed": seed_id,
        "pool": pool_name,
        "n_runs": schedule.n_runs,
        "projection": projection_counts(trajectory),
    }
    if pool_name != "b5_candidates":
        bounds = prepared.actions.bounds
        names = prepared.actions.names
        gas = names.index("gas_velocity_m_per_s")
        caps = np.asarray(schedule.metadata["gas_velocity_cap_m_per_s"])
        shares: dict[str, list[float]] = {
            k: [] for k in ("ratelim_changed", "at_lower", "at_upper", "at_cap")
        }
        for channel, bound in enumerate(bounds):
            pre, post, _ = recorded[channel]
            requested = schedule.actions[:, :, channel]
            tol = 1e-9 * bound.span
            shares["ratelim_changed"].append(float(np.mean(np.abs(pre - post) > tol)))
            shares["at_lower"].append(float(np.mean(requested <= bound.minimum + tol)))
            shares["at_upper"].append(float(np.mean(requested >= bound.maximum - tol)))
            shares["at_cap"].append(
                float(np.mean(requested >= caps[:, None] - tol)) if channel == gas else 0.0
            )
        result["shares"] = shares
        result["schedule_fingerprint"] = schedule.metadata["fingerprint"]
    result["_actions"] = schedule.actions if pool_name != "b5_candidates" else None
    result["_meta"] = schedule.metadata if pool_name != "b5_candidates" else None
    result["_dist"] = schedule.disturbances if pool_name != "b5_candidates" else None
    result["_init"] = schedule.initial_states if pool_name != "b5_candidates" else None
    result["_kinds"] = schedule.signal_kinds if pool_name != "b5_candidates" else None
    result["_ids"] = schedule.run_id if pool_name != "b5_candidates" else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="b1_pool_limits")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    arguments = parser.parse_args()
    if arguments.workers > 8:
        raise SystemExit("at most 8 workers")
    require_clean_tree()
    if (RUNS_DIR / f"{arguments.name}.json").exists():
        raise SystemExit("artifact exists; use a new --name")
    tasks = [(s, p) for s in arguments.seeds for p in ("b5_candidates", *PLAN_NAMES)]
    context = mp.get_context("spawn")
    start = time.perf_counter()
    with context.Pool(arguments.workers, initializer=_init) as pool:
        results = []
        for done, result in enumerate(pool.imap_unordered(_task, tasks), start=1):
            results.append(result)
            print(
                f"[{done}/{len(tasks)}] seed {result['seed']} {result['pool']}: projected "
                f"{result['projection']['projected_steps']} of {result['projection']['total_steps']}"
                f" ({time.perf_counter() - start:.0f} s)",
                flush=True,
            )
    config = guarded_config(load_process_config(CONFIG_PATH))
    spec = DataSpec.new_study(n_train_max=N_POOL, n_val=0, n_test=0, target_names=TARGETS)
    prepared = prepare_process_config(config, spec)
    from twinlink_ml.simulation.excitation import ExcitationSchedule

    equal_cost = {}
    for seed_id in arguments.seeds:
        schedules = {}
        for r in results:
            if r["seed"] == seed_id and r["pool"] in PLAN_NAMES:
                schedules[r["pool"]] = ExcitationSchedule(
                    run_id=r["_ids"],
                    actions=r["_actions"],
                    disturbances=r["_dist"],
                    initial_states=r["_init"],
                    signal_kinds=r["_kinds"],
                    metadata=r["_meta"],
                )
        equal_cost[str(seed_id)] = check_equal_cost(schedules, prepared, reference="b1")
    for r in results:
        for key in [k for k in r if k.startswith("_")]:
            del r[key]
    results.sort(key=lambda r: (r["seed"], r["pool"]))
    summary: dict[str, Any] = {}
    for pool_name in ("b5_candidates", *PLAN_NAMES):
        rows = [r for r in results if r["pool"] == pool_name]
        summary[pool_name] = {
            "projected_steps": sum(r["projection"]["projected_steps"] for r in rows),
            "total_steps": sum(r["projection"]["total_steps"] for r in rows),
            "runs_with_projection": sum(r["projection"]["runs_with_projection"] for r in rows),
            "n_runs": sum(r["n_runs"] for r in rows),
        }
        if pool_name != "b5_candidates":
            summary[pool_name]["shares_mean_over_seeds"] = {
                key: np.mean([r["shares"][key] for r in rows], axis=0).tolist()
                for key in ("ratelim_changed", "at_lower", "at_upper", "at_cap")
            }
    print("projected steps per pool (sum over seeds):")
    for name, row in summary.items():
        print(f"  {name}: {row['projected_steps']} of {row['total_steps']}")
    print("equal-cost violations:", {k: v for k, v in equal_cost.items() if v} or "none")
    write_artifact(
        arguments.name,
        {
            "_provenance": provenance(),
            "purpose": "projection counts and limit shares of the B1 pools with the per-run gas cap",
            "data_spec": spec.as_parameters(),
            "channels": list(prepared.actions.names),
            "seeds": arguments.seeds,
            "n_pool": N_POOL,
            "equal_cost_violations": equal_cost,
            "summary": summary,
            "per_pool": results,
            "wall_s": time.perf_counter() - start,
            "workers": arguments.workers,
        },
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()

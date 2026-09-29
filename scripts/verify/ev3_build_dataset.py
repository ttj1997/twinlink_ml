"""EV3 (data) -- constant runs at the instance-optimal operating points of Befund 52.

Test data set ``ev3opt``: for each of the 25 disturbance scenarios of the Befund 52 grid (ambient
temperature x relative humidity) the per-instance oracle setpoints (inlet temperature, gas velocity,
nozzle pressure; spray rate at its nominal value, as in Befund 52) are held constant for a whole run,
with constant disturbances at the grid point (suspension solid fraction nominal). Same run duration
(28800 s, 25 s steps, 1152 steps), same measurement model (the ``ProcessConfig`` of Befund 51: PT1
sensors, relative noise), same simulator configuration (gas-velocity projection on, fingerprint
``bddca37f716f5cfc``) as the constant-run holdout.

Two runs per scenario (50 runs): replicate 0 starts from the nominal initial state (as Befund 53's batch
rollouts), replicate 1 from an initial state drawn from the holdout's initial-condition design.
Rationale: the local error found in ZW4b is a steady-state offset of an operating point, so the unit
of information is the operating point (all 25 oracle points are covered, 17 of them bind the 0.20 kg/kg
limit); replicates only add transient diversity and independent sensor noise.

New run IDs (``ev3opt-<fingerprint>-<scenario>-r<replicate>``). Disjointness from every network's
training pool (5 excited + 5 constant-trained seeds, all 128 runs each) and from the holdout is checked
by ID and by content (SHA-256 of actions, disturbances and initial state per run; plus the smallest
normalised distance of the run's mean setpoint/disturbance vector to any pool run).

    python scripts/verify/ev3_build_dataset.py
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import warnings

import numpy as np
from _common import REPO_ROOT, RUNS_DIR, provenance, require_clean_tree, write_artifact
from _data import HOLDOUT_SEED, load_config, make_data

SAMPLE_TIME_S = 25.0
N_STEPS = 1152
INIT_SEED = 20260931
NOISE_SEED = 20260932
RAW_DIR = RUNS_DIR / "raw_ev3"


def run_hash(actions: np.ndarray, disturbances: np.ndarray, initial: np.ndarray | None) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(actions).tobytes())
    h.update(np.ascontiguousarray(disturbances).tobytes())
    if initial is not None:
        h.update(np.ascontiguousarray(initial).tobytes())
    return h.hexdigest()


def main() -> None:
    require_clean_tree()
    from twinlink_ml.simulation import GreyBoxFluidBedGranulator
    from twinlink_ml.simulation.excitation import InitialConditionDesign

    RAW_DIR.mkdir(exist_ok=True)
    config = load_config()
    config = dataclasses.replace(
        config, numerics=dataclasses.replace(config.numerics, sample_time_s=SAMPLE_TIME_S)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulator = GreyBoxFluidBedGranulator(config)

    grid = json.loads((RUNS_DIR / "study_operating_point_regret_win_repro.json").read_text("utf-8"))
    points = grid["grid_results"]
    assert len(points) == 25 and all(p["feasible"] for p in points)
    spray = config.actions.spray_rate_kg_per_h.nominal
    solid = config.disturbances.suspension_solid_fraction.nominal

    scen = []
    for i, p in enumerate(points):
        scen.append(
            {
                "scenario": i,
                "ambient_temperature_celsius": p["ambient_temperature_celsius"],
                "ambient_relative_humidity": p["ambient_relative_humidity"],
                "inlet_temperature_celsius": p["inlet_temperature_celsius"],
                "gas_velocity_m_per_s": p["gas_velocity_m_per_s"],
                "nozzle_pressure_bar": p["nozzle_pressure_bar"],
                "spray_rate_kg_per_h": spray,
                "suspension_solid_fraction": solid,
                "moisture_binds_at_oracle": bool(p["constraint_check"]["binding_moisture"]),
            }
        )
    payload = json.dumps([scen, INIT_SEED, NOISE_SEED, N_STEPS], sort_keys=True).encode()
    fp = hashlib.sha256(payload).hexdigest()[:8]

    n_runs = 2 * len(scen)
    actions = np.empty((n_runs, N_STEPS, 4))
    disturbances = np.empty((n_runs, N_STEPS, 3))
    run_id, scenario_of, rep_of = [], [], []
    for i, s in enumerate(scen):
        for rep in (0, 1):
            k = 2 * i + rep
            actions[k] = [
                s["inlet_temperature_celsius"],
                s["gas_velocity_m_per_s"],
                spray,
                s["nozzle_pressure_bar"],
            ]
            disturbances[k] = [
                s["ambient_temperature_celsius"],
                s["ambient_relative_humidity"],
                solid,
            ]
            run_id.append(f"ev3opt-{fp}-{i:02d}-r{rep}")
            scenario_of.append(i)
            rep_of.append(rep)
    assert list(config.actions.names) == [
        "inlet_temperature_celsius",
        "gas_velocity_m_per_s",
        "spray_rate_kg_per_h",
        "nozzle_pressure_bar",
    ]

    rng = np.random.default_rng(INIT_SEED)
    unit = rng.random((len(scen), 4))
    drawn = InitialConditionDesign().build(
        unit, geometry=config.geometry, material=config.material, operation=config.operation
    )
    nominal_state = simulator.initial_state(1)
    init_phys = np.empty((n_runs, 4))
    state0 = np.empty((n_runs, nominal_state.shape[-1]))
    for i in range(len(scen)):
        st = simulator.state_from_physical(
            drawn[i : i + 1, 0], drawn[i : i + 1, 1], drawn[i : i + 1, 2], drawn[i : i + 1, 3]
        )
        state0[2 * i] = np.asarray(nominal_state)[0]
        state0[2 * i + 1] = np.asarray(st)[0]
        init_phys[2 * i + 1] = drawn[i]
    init_phys[0::2] = np.nan  # nominal state: physical columns not drawn

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traj = simulator.rollout(
            actions, disturbances, run_id=tuple(run_id), seed=NOISE_SEED, initial_state=state0
        )

    np.savez_compressed(
        RAW_DIR / "dataset.npz",
        run_id=np.array(run_id),
        scenario=np.array(scenario_of),
        replicate=np.array(rep_of),
        actions=traj.actions,
        disturbances=traj.disturbances,
        outputs=traj.outputs,
        measurements=traj.measurements,
        initial_state=state0,
        n_projected_steps=traj.n_projected_steps,
    )
    sha = hashlib.sha256((RAW_DIR / "dataset.npz").read_bytes()).hexdigest()

    # ---- disjointness from every training pool and the holdout
    pool_ids: set[str] = set()
    pool_hashes: dict[str, str] = {}
    pool_vec = []
    lo = np.array([b.minimum for b in config.actions.bounds])
    span = np.array([b.span for b in config.actions.bounds])
    dlo = np.array([r.minimum for r in config.disturbances.ranges])
    dspan = np.array([r.maximum - r.minimum for r in config.disturbances.ranges])
    sources = [
        (f"{k}_seed{s}", make_data(k, s))
        for k in ("excited", "constant_train")
        for s in range(1, 6)
    ]
    sources.append(("holdout", make_data("holdout", HOLDOUT_SEED)))
    for name, d in sources:
        t = d.trajectory
        pool_ids.update(t.run_id)
        init = d.schedule.initial_states
        for j, rid in enumerate(t.run_id):
            pool_hashes[
                run_hash(t.actions[j], t.disturbances[j], None if init is None else init[j])
            ] = f"{name}:{rid}"
        pool_vec.append(
            np.concatenate(
                [(t.actions.mean(axis=1) - lo) / span, (t.disturbances.mean(axis=1) - dlo) / dspan],
                axis=1,
            )
        )
    pool_vec_all = np.concatenate(pool_vec)
    mine = np.concatenate(
        [(traj.actions.mean(axis=1) - lo) / span, (traj.disturbances.mean(axis=1) - dlo) / dspan],
        axis=1,
    )
    dist = np.sqrt(((mine[:, None, :] - pool_vec_all[None, :, :]) ** 2).sum(-1)).min(axis=1)
    my_hashes = [run_hash(traj.actions[k], traj.disturbances[k], state0[k]) for k in range(n_runs)]
    # the pool hashes use the schedule's physical initial states; the ev3 runs use model states, so a
    # content match is tested on actions + disturbances alone as well
    pool_ad = {
        run_hash(d.trajectory.actions[j], d.trajectory.disturbances[j], None)
        for _, d in sources
        for j in range(d.trajectory.actions.shape[0])
    }
    my_ad = [run_hash(traj.actions[k], traj.disturbances[k], None) for k in range(n_runs)]

    moist = traj.outputs[..., 2]  # product moisture (truth), TARGETS order col 2 of outputs
    per_scenario = []
    for i, s in enumerate(scen):
        m = moist[2 * i : 2 * i + 2]
        per_scenario.append(
            {
                **s,
                "moisture_last_step_r0_r1": [float(m[0, -1]), float(m[1, -1])],
                "moisture_max_r0_r1": [float(m[0].max()), float(m[1].max())],
                "share_steps_in_0.18_0.22_r0_r1": [
                    float(((m[r] >= 0.18) & (m[r] < 0.22)).mean()) for r in (0, 1)
                ],
                "n_projected_steps_r0_r1": [
                    int(traj.n_projected_steps[2 * i]),
                    int(traj.n_projected_steps[2 * i + 1]),
                ],
            }
        )
    write_artifact(
        "verify_ev3_dataset",
        {
            "_provenance": provenance(),
            "process_fingerprint": config.fingerprint,
            "n_runs": n_runs,
            "n_steps": N_STEPS,
            "sample_time_s": SAMPLE_TIME_S,
            "seeds": {"initial_states": INIT_SEED, "sensor_noise": NOISE_SEED},
            "dataset_file": "artifacts/runs/raw_ev3/dataset.npz (git-ignored)",
            "dataset_sha256": sha,
            "dataset_bytes": (RAW_DIR / "dataset.npz").stat().st_size,
            "run_ids": run_id,
            "scenarios": per_scenario,
            "disjointness": {
                "n_pool_runs_compared": len(pool_ids),
                "n_pool_sources": len(sources),
                "id_overlap": len(pool_ids & set(run_id)),
                "content_overlap_actions_disturbances_initial": len(
                    set(pool_hashes) & set(my_hashes)
                ),
                "content_overlap_actions_disturbances": len(pool_ad & set(my_ad)),
                "smallest_normalised_distance_to_any_pool_run": float(dist.min()),
                "median_normalised_distance_to_pool": float(np.median(dist)),
                "distance_note": "Euclidean, mean actions and mean disturbances scaled by bound span, 7 dims",
            },
            "share_runs_with_projection_active": float((traj.n_projected_steps > 0).mean()),
        },
    )
    print(json.dumps({"sha": sha, "dist": [float(dist.min()), float(np.median(dist))]}))
    for r in per_scenario:
        print(
            r["scenario"],
            r["moisture_binds_at_oracle"],
            [round(x, 4) for x in r["moisture_last_step_r0_r1"]],
            [round(x, 2) for x in r["share_steps_in_0.18_0.22_r0_r1"]],
            r["n_projected_steps_r0_r1"],
        )
    _ = REPO_ROOT


if __name__ == "__main__":
    main()

"""FA3 (2/2) -- EV3-analogue data sets at the tightened oracle points and the error there.

For each of the two margins the runs of EV3 are rebuilt with the setpoints of the tightened oracle
(``verify_fa3_tightened_oracle``): same 25 scenarios, same replicate structure, same initial states (seed
20260931), same sensor-noise seed (20260932), same process/measurement configuration -- only the setpoints
differ, so EV3 and the tightened sets are PAIRED. The builder is validated first by rebuilding EV3 from its
own setpoints and comparing bitwise with ``raw_ev3/dataset.npz``. New run IDs
``ev3m<tag>-<fingerprint>-<scenario>-r<replicate>``; disjointness from EV3 and from all 1316 training/holdout
runs by ID and content.

Evaluation: the ten archive networks + ensemble of each arm, moisture, h=36, error against the TRUE moisture
(FA1), population = settled windows (forecast starting in the second half of the run) of the 17 scenarios
that bind the limit at margin 0 (34 runs, paired with EV3); the arm is evaluated on the set of its own margin
(excited-trained: 0.0016282, constant-trained: 0.0044651 kg/kg), the other margin as information.

Pre-registered, before the first evaluation: the error "changes noticeably" if the run-median RMSE against
the truth (E1t) of the network median or of the ensemble differs between margin 0 and the arm's margin by
>= 20 % (relative to margin 0) AND the paired bootstrap over scenarios (10000 draws) 95 % interval of the
relative change excludes 0. Otherwise "no noticeable change". The reserve factors at the 90 % mark are
recomputed with the same E1t/pooled/E4 definitions as FA1.

    python scripts/verify/fa3_build_evaluate.py
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import warnings

import numpy as np
from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact
from _data import HOLDOUT_SEED, load_config, make_data
from _fa import ARMS, RAW_FA, SEEDS, load_dataset, per_run_rmse, points, predictions

SAMPLE_TIME_S = 25.0
N_STEPS = 1152
INIT_SEED = 20260931
NOISE_SEED = 20260932
SETTLED_START = 544
B = 10_000
TAGS = {"excited_lstm_3rmse": "exc", "constant_lstm_3rmse": "const"}
ARM_TAG = {"excited": "excited_lstm_3rmse", "constant": "constant_lstm_3rmse"}


def build(setpoints: list[dict], tag: str, fp_extra: str) -> dict[str, np.ndarray]:
    from twinlink_ml.simulation import GreyBoxFluidBedGranulator
    from twinlink_ml.simulation.excitation import InitialConditionDesign

    config = load_config()
    config = dataclasses.replace(
        config, numerics=dataclasses.replace(config.numerics, sample_time_s=SAMPLE_TIME_S)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = GreyBoxFluidBedGranulator(config)
    spray = config.actions.spray_rate_kg_per_h.nominal
    solid = config.disturbances.suspension_solid_fraction.nominal
    n_scen = len(setpoints)
    n_runs = 2 * n_scen
    actions = np.empty((n_runs, N_STEPS, 4))
    dist = np.empty((n_runs, N_STEPS, 3))
    fp = hashlib.sha256(
        json.dumps([setpoints, fp_extra, INIT_SEED, NOISE_SEED]).encode()
    ).hexdigest()[:8]
    run_id, scen, rep = [], [], []
    for i, s in enumerate(setpoints):
        for r in (0, 1):
            k = 2 * i + r
            actions[k] = [
                s["inlet_temperature_celsius"],
                s["gas_velocity_m_per_s"],
                spray,
                s["nozzle_pressure_bar"],
            ]
            dist[k] = [s["ambient_temperature_celsius"], s["ambient_relative_humidity"], solid]
            run_id.append(f"ev3m{tag}-{fp}-{i:02d}-r{r}")
            scen.append(i)
            rep.append(r)
    rng = np.random.default_rng(INIT_SEED)
    drawn = InitialConditionDesign().build(
        rng.random((n_scen, 4)),
        geometry=config.geometry,
        material=config.material,
        operation=config.operation,
    )
    nominal = np.asarray(sim.initial_state(1))
    state0 = np.empty((n_runs, nominal.shape[-1]))
    for i in range(n_scen):
        st = sim.state_from_physical(
            drawn[i : i + 1, 0], drawn[i : i + 1, 1], drawn[i : i + 1, 2], drawn[i : i + 1, 3]
        )
        state0[2 * i] = nominal[0]
        state0[2 * i + 1] = np.asarray(st)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traj = sim.rollout(
            actions, dist, run_id=tuple(run_id), seed=NOISE_SEED, initial_state=state0
        )
    return {
        "run_id": np.array(run_id),
        "scenario": np.array(scen),
        "replicate": np.array(rep),
        "actions": traj.actions,
        "disturbances": traj.disturbances,
        "outputs": traj.outputs,
        "measurements": traj.measurements,
        "initial_state": state0,
        "n_projected_steps": traj.n_projected_steps,
    }


def run_hash(a: np.ndarray, d: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(a).tobytes())
    h.update(np.ascontiguousarray(d).tobytes())
    return h.hexdigest()


def stats_truth(pred: np.ndarray, p: dict, mask: np.ndarray) -> dict[str, float]:
    e = (pred - p["truth"])[mask]
    run = p["run"][mask]
    per = per_run_rmse(e, run)
    return {
        "rm_truth": float(np.median(per)),
        "pooled_truth": float(np.sqrt(np.mean(e**2))),
        "bias": float(e.mean()),
        "spread": float(e.std()),
    }


def main() -> None:
    require_clean_tree()
    ref = json.loads((RUNS_DIR / "probe_soft_sensor_margin_requirement.json").read_text("utf-8"))
    sigma_max = ref["interpolation"]["0.90"]["sigma_max_kg_per_kg"]
    m90 = ref["interpolation"]["0.90"]["interpolated_margin_kg_per_kg"]
    oracle = json.loads((RUNS_DIR / "verify_fa3_tightened_oracle.json").read_text("utf-8"))
    ev3_info = json.loads((RUNS_DIR / "verify_ev3_dataset.json").read_text("utf-8"))["scenarios"]
    binding0 = np.array([s["moisture_binds_at_oracle"] for s in ev3_info])
    RAW_FA.mkdir(exist_ok=True)
    out: dict[str, object] = {"sigma_max_90": sigma_max, "margin_90": m90}

    # builder validation: rebuild EV3 from its own setpoints
    ev3_dev = build(ev3_info, "chk", "validation")
    z = np.load(RUNS_DIR / "raw_ev3" / "dataset.npz")
    out["builder_check_vs_ev3"] = {
        "outputs_identical": bool(np.array_equal(ev3_dev["outputs"], z["outputs"])),
        "measurements_identical": bool(np.array_equal(ev3_dev["measurements"], z["measurements"])),
    }
    print(out["builder_check_vs_ev3"], flush=True)

    # datasets
    datasets, meta = {}, {}
    for label, tag in TAGS.items():
        res = oracle["results"][label]
        grid = res["grid_results"]
        assert all(g.get("feasible") and not g.get("solve_failed") for g in grid)
        sp = [
            {
                "inlet_temperature_celsius": g["inlet_temperature_celsius"],
                "gas_velocity_m_per_s": g["gas_velocity_m_per_s"],
                "nozzle_pressure_bar": g["nozzle_pressure_bar"],
                "ambient_temperature_celsius": g["ambient_temperature_celsius"],
                "ambient_relative_humidity": g["ambient_relative_humidity"],
            }
            for g in grid
        ]
        d = build(sp, tag, label)
        np.savez_compressed(RAW_FA / f"ev3m_{tag}.npz", **d)
        datasets[tag] = d
        moist = d["outputs"][..., 2]
        meta[tag] = {
            "margin": res["margin_kg_per_kg"],
            "threshold": res["effective_threshold_kg_per_kg"],
            "sha256": hashlib.sha256((RAW_FA / f"ev3m_{tag}.npz").read_bytes()).hexdigest(),
            "binding_moisture_at_tightened_oracle": [
                bool(g["constraint_check"]["binding_moisture"]) for g in grid
            ],
            "moisture_last_step_r0": [float(moist[2 * i, -1]) for i in range(25)],
            "n_projected_steps_total": int(d["n_projected_steps"].sum()),
            "setpoint_shift_vs_margin0": {
                k: [float(g[k] - s[k]) for g, s in zip(grid, ev3_info, strict=True)]
                for k in (
                    "inlet_temperature_celsius",
                    "gas_velocity_m_per_s",
                    "nozzle_pressure_bar",
                )
            },
        }
    out["datasets"] = meta

    # disjointness (IDs and content) from EV3 and from all pools
    pool_ids, pool_hash = set(), set()
    for k in ("excited", "constant_train"):
        for s in SEEDS:
            dd = make_data(k, s)
            pool_ids.update(dd.trajectory.run_id)
            pool_hash.update(
                run_hash(dd.trajectory.actions[j], dd.trajectory.disturbances[j])
                for j in range(dd.trajectory.actions.shape[0])
            )
    hd = make_data("holdout", HOLDOUT_SEED)
    pool_ids.update(hd.trajectory.run_id)
    pool_hash.update(
        run_hash(hd.trajectory.actions[j], hd.trajectory.disturbances[j])
        for j in range(hd.trajectory.actions.shape[0])
    )
    ev3_ids = {str(r) for r in z["run_id"]}
    ev3_hash = {
        run_hash(z["actions"][j], z["disturbances"][j]) for j in range(z["actions"].shape[0])
    }
    out["disjointness"] = {
        tag: {
            "id_overlap_pools_and_ev3": len((pool_ids | ev3_ids) & {str(r) for r in d["run_id"]}),
            "content_overlap_pools": len(
                pool_hash & {run_hash(d["actions"][j], d["disturbances"][j]) for j in range(50)}
            ),
            "content_overlap_ev3": len(
                ev3_hash & {run_hash(d["actions"][j], d["disturbances"][j]) for j in range(50)}
            ),
        }
        for tag, d in datasets.items()
    }
    print(out["disjointness"], flush=True)

    rng = np.random.default_rng(20260936)
    ev3 = load_dataset("raw_ev3/dataset.npz")
    p3 = points(ev3, ev3.trajectory.run_id)
    scen3 = ev3.scenario[p3["run"]]
    mask3 = (p3["start"] >= SETTLED_START) & binding0[scen3]
    bind_scen = np.where(binding0)[0]

    for arm in ARMS:
        res: dict[str, object] = {}
        pr0 = predictions(arm, "ev3", ev3, ev3.trajectory.run_id)
        for tag in ("exc", "const"):
            ds = load_dataset(f"raw_fa/ev3m_{tag}.npz")
            ids = ds.trajectory.run_id
            pm = points(ds, ids)
            scen_m = ds.scenario[pm["run"]]
            maskm = (pm["start"] >= SETTLED_START) & binding0[scen_m]
            prm = predictions(arm, f"ev3m_{tag}", ds, ids)
            info: dict[str, object] = {
                "own_margin": ARM_TAG[arm] == next(k for k, v in TAGS.items() if v == tag)
            }
            rows = {}
            for who in ("network_median", "ensemble"):

                def get(
                    pr: np.ndarray, p: dict, mask: np.ndarray, who: str = who
                ) -> dict[str, float]:
                    if who == "ensemble":
                        return stats_truth(pr.mean(axis=0), p, mask)
                    st = [stats_truth(pr[i], p, mask) for i in range(len(SEEDS))]
                    return {k: float(np.median([s[k] for s in st])) for k in st[0]}

                s0, sm = get(pr0, p3, mask3), get(prm, pm, maskm)
                rows[who] = {
                    "margin0": s0,
                    "tightened": sm,
                    "relative_change_rm_truth": sm["rm_truth"] / s0["rm_truth"] - 1,
                    "reserve_E1t_margin0": sigma_max / s0["rm_truth"],
                    "reserve_E1t_tightened": sigma_max / sm["rm_truth"],
                    "reserve_pooled_tightened": sigma_max / sm["pooled_truth"],
                    "reserve_E4_dangerous_tightened": m90
                    / (max(0.0, -sm["bias"]) + 3 * sm["spread"]),
                    "reserve_E4_worst_tightened": m90 / (abs(sm["bias"]) + 3 * sm["spread"]),
                }
            # paired bootstrap over scenarios for the relative change of E1t
            runs0 = np.unique(p3["run"][mask3])
            runsm = np.unique(pm["run"][maskm])
            assert np.array_equal(runs0, runsm)
            scen_of = ev3.scenario[runs0]
            idx_of = {s: np.where(scen_of == s)[0] for s in bind_scen}
            per0 = [per_run_rmse((pr0[i] - p3["truth"])[mask3], p3["run"][mask3]) for i in range(5)]
            per0.append(per_run_rmse((pr0.mean(axis=0) - p3["truth"])[mask3], p3["run"][mask3]))
            perm = [per_run_rmse((prm[i] - pm["truth"])[maskm], pm["run"][maskm]) for i in range(5)]
            perm.append(per_run_rmse((prm.mean(axis=0) - pm["truth"])[maskm], pm["run"][maskm]))
            bn, be = [], []
            for _ in range(B):
                ds_ = rng.choice(bind_scen, size=len(bind_scen), replace=True)
                ie = np.concatenate([idx_of[s] for s in ds_])
                bn.append(
                    np.median([np.median(perm[i][ie]) for i in range(5)])
                    / np.median([np.median(per0[i][ie]) for i in range(5)])
                    - 1
                )
                be.append(np.median(perm[5][ie]) / np.median(per0[5][ie]) - 1)
            rows["network_median"]["bootstrap_95_relative_change"] = [
                float(np.percentile(bn, 2.5)),
                float(np.percentile(bn, 97.5)),
            ]
            rows["ensemble"]["bootstrap_95_relative_change"] = [
                float(np.percentile(be, 2.5)),
                float(np.percentile(be, 97.5)),
            ]
            for who in rows:
                rc = rows[who]["relative_change_rm_truth"]
                lo, hi = rows[who]["bootstrap_95_relative_change"]
                rows[who]["noticeable"] = bool(abs(rc) >= 0.20 and (lo > 0 or hi < 0))
            info["results"] = rows
            res[tag] = info
            print(
                arm,
                tag,
                {
                    w: (
                        round(rows[w]["relative_change_rm_truth"], 3),
                        rows[w]["bootstrap_95_relative_change"],
                        rows[w]["noticeable"],
                        round(rows[w]["reserve_E1t_tightened"], 2),
                    )
                    for w in rows
                },
                flush=True,
            )
        out[arm] = res
    write_artifact("verify_fa3_evaluate", {"_provenance": provenance(), **out})


if __name__ == "__main__":
    main()

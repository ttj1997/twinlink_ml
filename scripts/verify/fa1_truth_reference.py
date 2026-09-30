"""FA1 -- same reference on both sides: error against the noise-free TRUE moisture.

Motivation: the EV3 factors compared an error against a noisy reference at the limit points (reference
noise 0.0010 kg/kg) with an error at points with less noise -- the same class of mistake as the unequal
estimators in GA2. The margin constraint holds for the true moisture, so the error that counts is the
error against the truth.

a) The noise-free truth of the constant-run holdout and of the EV3 set is the simulator output. Check that
   a fresh regeneration is bit-identical to the stored raw data: holdout (``make_data`` vs the ZW1 files:
   truth and measured windows), EV3 (re-simulation of the stored actions, disturbances and initial states
   with the stored sensor-noise seed vs ``raw_ev3/dataset.npz``: outputs and measurements).
b) Error against the truth with the same estimator on both sides (run-median over runs of the per-run RMSE,
   pooled alongside), per arm, single networks (median, min-max over 5) and the five-network ensemble:
   global (holdout, all 32 runs, steps 1..36), holdout class [0.18, 0.22) (5 runs), and the EV3 population
   (settled windows of the 34 runs whose oracle binds the limit).
c) The EV3 factors again (EV3 / global holdout, and EV3 / holdout class), with a 95 % interval from a
   bootstrap over holdout runs (denominator) and over EV3 scenarios (numerator), and the reserve factors at
   the 90 % mark against the truth (sigma_max / run-median RMSE vs truth; sigma_max / pooled RMSE vs truth;
   margin / (max(0, -bias) + 3 spread)).
d) Dependence of the reference noise (measured - truth) on the moisture; noise floor of the estimators used
   in GA2/ZW/EV3.

Pre-registered readings, fixed before the first evaluation: a ratio EV3 / global-holdout (run-median vs
truth, same estimator) of <= 1.5 counts as "about like the global median run", >= 2.0 as "clearly higher",
in between "intermediate". As a same-population cross-check, EV3 / holdout-class within [0.67, 1.5] counts as
"consistent with the holdout class" (both are settled or near-limit runs against the truth). Same
thresholds for single networks (median) and the ensemble.

    python scripts/verify/fa1_truth_reference.py
"""

from __future__ import annotations

import itertools
import json
import warnings

import numpy as np
from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact
from _data import HOLDOUT_SEED, load_config, make_data
from _fa import ARMS, SEEDS, load_dataset, per_run_rmse, points, predictions
from _zw import RAW_DIR as ZW_RAW
from _zw import load as zw_load
from pr3_metrics import make_windows, train_scaler

SETTLED_START = 544
B = 10_000
LO, HI = 0.18, 0.22
EDGES = [*np.arange(0.0, 0.4001, 0.02), 0.6, np.inf]


def check_reproduction() -> dict[str, object]:
    out: dict[str, object] = {}
    hold = make_data("holdout", HOLDOUT_SEED)
    pool = make_data("excited", 1)
    w = make_windows(hold, hold.split.test, train_scaler(pool, 64))
    z = np.load(ZW_RAW / "excited_seed1.npz")
    out["holdout_truth_windows_identical"] = bool(np.array_equal(w["truth"][..., 1], z["truth"]))
    out["holdout_measured_windows_identical"] = bool(
        np.array_equal(w["ref"][..., 1], z["measured"])
    )

    import dataclasses

    from twinlink_ml.simulation import GreyBoxFluidBedGranulator

    config = load_config()
    config = dataclasses.replace(
        config, numerics=dataclasses.replace(config.numerics, sample_time_s=25.0)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = GreyBoxFluidBedGranulator(config)
        d = np.load(RUNS_DIR / "raw_ev3" / "dataset.npz")
        traj = sim.rollout(
            d["actions"],
            d["disturbances"],
            run_id=tuple(str(r) for r in d["run_id"]),
            seed=20260932,
            initial_state=d["initial_state"],
        )
    out["ev3_outputs_identical"] = bool(np.array_equal(traj.outputs, d["outputs"]))
    out["ev3_measurements_identical"] = bool(np.array_equal(traj.measurements, d["measurements"]))
    out["ev3_max_abs_output_diff"] = float(np.abs(traj.outputs - d["outputs"]).max())
    return out


def estimators_truth(
    pred: np.ndarray, p: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, float]:
    e = (pred - p["truth"])[mask]
    run = p["run"][mask]
    bias, spread = float(e.mean()), float(e.std())
    return {
        "n_runs": int(np.unique(run).size),
        "rm_truth": float(np.median(per_run_rmse(e, run))),
        "pooled_truth": float(np.sqrt(np.mean(e**2))),
        "bias": bias,
        "spread": spread,
    }


def main() -> None:
    require_clean_tree()
    ref = json.loads((RUNS_DIR / "probe_soft_sensor_margin_requirement.json").read_text("utf-8"))
    sigma_max = ref["interpolation"]["0.90"]["sigma_max_kg_per_kg"]
    m90 = ref["interpolation"]["0.90"]["interpolated_margin_kg_per_kg"]
    out: dict[str, object] = {"sigma_max_90": sigma_max, "margin_90": m90}
    out["a_reproduction"] = check_reproduction()
    print(out["a_reproduction"], flush=True)

    ev3 = load_dataset("raw_ev3/dataset.npz")
    ids = ev3.trajectory.run_id
    scen_info = json.loads((RUNS_DIR / "verify_ev3_dataset.json").read_text("utf-8"))["scenarios"]
    binding = np.array([s["moisture_binds_at_oracle"] for s in scen_info])
    p3 = points(ev3, ids)
    scen_pt = ev3.scenario[p3["run"]]
    settled_binding = (p3["start"] >= SETTLED_START) & binding[scen_pt]
    rng = np.random.default_rng(20260934)

    # d) noise vs moisture, holdout (all points) and EV3 (all points)
    p_h = zw_load("excited", 1)
    noise_bins = {}
    for name, truth, meas, run in (
        ("holdout", p_h["truth"], p_h["measured"], p_h["run"]),
        ("ev3", p3["truth"], p3["measured"], p3["run"]),
    ):
        rows = []
        for lo, hi in itertools.pairwise(EDGES):
            m = (truth >= lo) & (truth < hi)
            if m.sum() == 0:
                continue
            d = (meas - truth)[m]
            rows.append(
                {
                    "lo": float(lo),
                    "n": int(m.sum()),
                    "n_runs": int(np.unique(run[m]).size),
                    "mean_truth": float(truth[m].mean()),
                    "noise_rmse": float(np.sqrt(np.mean(d**2))),
                    "noise_mean": float(d.mean()),
                    "relative_rmse": float(np.sqrt(np.mean(d**2)) / truth[m].mean()),
                }
            )
        noise_bins[name] = rows
    # through-origin slope on bins with >= 3 runs, truth > 0.02
    sel = [r for r in noise_bins["holdout"] if r["n_runs"] >= 3 and r["mean_truth"] > 0.02]
    slope = float(
        np.sum([r["mean_truth"] * r["noise_rmse"] * r["n"] for r in sel])
        / np.sum([r["mean_truth"] ** 2 * r["n"] for r in sel])
    )
    noise_floor = {}
    cls_h = (p_h["truth"] >= LO) & (p_h["truth"] < HI)
    dn = p_h["measured"] - p_h["truth"]
    noise_floor["holdout_global_run_median"] = float(np.median(per_run_rmse(dn, p_h["run"])))
    noise_floor["holdout_class_run_median"] = float(
        np.median(per_run_rmse(dn[cls_h], p_h["run"][cls_h]))
    )
    dn3 = (p3["measured"] - p3["truth"])[settled_binding]
    noise_floor["ev3_settled_binding_run_median"] = float(
        np.median(per_run_rmse(dn3, p3["run"][settled_binding]))
    )
    noise_floor["ratio_class_over_global_holdout"] = (
        noise_floor["holdout_class_run_median"] / noise_floor["holdout_global_run_median"]
    )
    noise_floor["ratio_ev3_over_global_holdout"] = (
        noise_floor["ev3_settled_binding_run_median"] / noise_floor["holdout_global_run_median"]
    )
    cfg = load_config().measurement
    out["d_noise"] = {
        "bins": noise_bins,
        "through_origin_slope_noise_rmse_over_truth_holdout": slope,
        "noise_floor_run_median": noise_floor,
        "measurement_config": {
            k: (v if isinstance(v, int | float | str) else str(v)) for k, v in vars(cfg).items()
        }
        if hasattr(cfg, "__dict__")
        else str(cfg),
    }
    print("noise", noise_floor, slope, flush=True)

    for arm in ARMS:
        # holdout side
        nets_h = [zw_load(arm, s) for s in SEEDS]
        ens_h = np.mean([n["pred"] for n in nets_h], axis=0)
        all_h = np.ones_like(cls_h)
        hold_stats: dict[str, dict] = {}
        for label, pred in [
            *[(f"net{s}", n["pred"]) for s, n in zip(SEEDS, nets_h, strict=True)],
            ("ensemble", ens_h),
        ]:
            hold_stats[label] = {
                "global": estimators_truth(pred, p_h, all_h),
                "class": estimators_truth(pred, p_h, cls_h),
            }
        # EV3 side
        pr3 = predictions(arm, "ev3", ev3, ids)
        ens3 = pr3.mean(axis=0)
        ev_stats = {
            **{
                f"net{s}": estimators_truth(pr3[i], p3, settled_binding)
                for i, s in enumerate(SEEDS)
            },
            "ensemble": estimators_truth(ens3, p3, settled_binding),
        }
        nonbind = (p3["start"] >= SETTLED_START) & ~binding[scen_pt]
        ev_nonbind = {
            "network_median_rm_truth": float(
                np.median([estimators_truth(pr3[i], p3, nonbind)["rm_truth"] for i in range(5)])
            ),
            "ensemble_rm_truth": estimators_truth(ens3, p3, nonbind)["rm_truth"],
        }

        def med(stats: dict, part: str, key: str) -> float:
            return float(
                np.median(
                    [stats[f"net{s}"][part][key] if part else stats[f"net{s}"][key] for s in SEEDS]
                )
            )

        res: dict[str, object] = {
            "holdout": hold_stats,
            "ev3_settled_binding": ev_stats,
            "ev3_settled_nonbinding": ev_nonbind,
        }
        summ = {}
        for who in ("network_median", "ensemble"):

            def pick(stats: dict, part: str | None, key: str, who: str = who) -> float:
                if who == "ensemble":
                    return stats["ensemble"][part][key] if part else stats["ensemble"][key]
                return med(stats, part or "", key)

            g = pick(hold_stats, "global", "rm_truth")
            c = pick(hold_stats, "class", "rm_truth")
            e = pick(ev_stats, None, "rm_truth")
            bias = pick(ev_stats, None, "bias")
            spread = pick(ev_stats, None, "spread")
            summ[who] = {
                "holdout_global_rm_truth": g,
                "holdout_class_rm_truth": c,
                "ev3_rm_truth": e,
                "ev3_pooled_truth": pick(ev_stats, None, "pooled_truth"),
                "ev3_bias": bias,
                "ev3_spread": spread,
                "ratio_ev3_over_global": e / g,
                "ratio_ev3_over_holdout_class": e / c,
                "reserve_vs_truth_E1t": sigma_max / e,
                "reserve_vs_truth_pooled": sigma_max / pick(ev_stats, None, "pooled_truth"),
                "reserve_E4_dangerous": m90 / (max(0.0, -bias) + 3 * spread),
                "reserve_E4_worst": m90 / (abs(bias) + 3 * spread),
                "holdout_class_reserve_E1t": sigma_max / c,
                "holdout_class_reserve_pooled": sigma_max
                / pick(hold_stats, "class", "pooled_truth"),
                "holdout_global_reserve_E1t": sigma_max / g,
            }
        summ["network_range"] = {
            "ratio_ev3_over_global": [
                min(
                    ev_stats[f"net{s}"]["rm_truth"] / hold_stats[f"net{s}"]["global"]["rm_truth"]
                    for s in SEEDS
                ),
                max(
                    ev_stats[f"net{s}"]["rm_truth"] / hold_stats[f"net{s}"]["global"]["rm_truth"]
                    for s in SEEDS
                ),
            ],
            "ratio_ev3_over_global_medians": [
                min(ev_stats[f"net{s}"]["rm_truth"] for s in SEEDS)
                / summ["network_median"]["holdout_global_rm_truth"],
                max(ev_stats[f"net{s}"]["rm_truth"] for s in SEEDS)
                / summ["network_median"]["holdout_global_rm_truth"],
            ],
            "reserve_E1t": [
                sigma_max / max(ev_stats[f"net{s}"]["rm_truth"] for s in SEEDS),
                sigma_max / min(ev_stats[f"net{s}"]["rm_truth"] for s in SEEDS),
            ],
            "n_networks_reserve_E1t_below_1": int(
                sum(sigma_max / ev_stats[f"net{s}"]["rm_truth"] < 1 for s in SEEDS)
            ),
        }
        res["summary"] = summ

        # bootstrap: holdout runs (denominator) and EV3 scenarios (numerator)
        pr_h = np.array([n["pred"] for n in nets_h] + [ens_h])  # 6 x points (5 nets + ensemble)
        per_run_h_global = [per_run_rmse(pr_h[i] - p_h["truth"], p_h["run"]) for i in range(6)]
        per_run_h_class = [
            per_run_rmse((pr_h[i] - p_h["truth"])[cls_h], p_h["run"][cls_h]) for i in range(6)
        ]
        class_runs_h = np.unique(p_h["run"][cls_h])
        pr_e = np.vstack([pr3, ens3[None, :]])
        runs_e = np.unique(p3["run"][settled_binding])
        per_run_e = [
            per_run_rmse((pr_e[i] - p3["truth"])[settled_binding], p3["run"][settled_binding])
            for i in range(6)
        ]
        scen_of_run_e = ev3.scenario[runs_e]
        bind_scen = np.unique(scen_of_run_e)
        idx_of_scen = {s: np.where(scen_of_run_e == s)[0] for s in bind_scen}
        boot = {
            k: []
            for k in (
                "ratio_global_net",
                "ratio_global_ens",
                "ratio_class_net",
                "ratio_class_ens",
                "reserve_net",
                "reserve_ens",
            )
        }
        for _ in range(B):
            dh = rng.integers(0, 32, 32)
            # holdout global: per-run values indexed by run order 0..31 (all runs present)
            ds = rng.choice(bind_scen, size=len(bind_scen), replace=True)
            ie = np.concatenate([idx_of_scen[s] for s in ds])
            g = [np.median(per_run_h_global[i][dh]) for i in range(6)]
            cd = rng.integers(0, len(class_runs_h), len(class_runs_h))
            c = [np.median(per_run_h_class[i][cd]) for i in range(6)]
            e = [np.median(per_run_e[i][ie]) for i in range(6)]
            boot["ratio_global_net"].append(np.median(e[:5]) / np.median(g[:5]))
            boot["ratio_global_ens"].append(e[5] / g[5])
            boot["ratio_class_net"].append(np.median(e[:5]) / np.median(c[:5]))
            boot["ratio_class_ens"].append(e[5] / c[5])
            boot["reserve_net"].append(sigma_max / np.median(e[:5]))
            boot["reserve_ens"].append(sigma_max / e[5])
        res["bootstrap_95"] = {
            k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in boot.items()
        }
        out[arm] = res
        print(arm, json.dumps(summ["network_median"], default=float)[:900])
        print(arm, json.dumps(summ["ensemble"], default=float)[:900])
        print(arm, summ["network_range"], res["bootstrap_95"], ev_nonbind, flush=True)
    write_artifact("verify_fa1_truth_reference", {"_provenance": provenance(), **out})


if __name__ == "__main__":
    main()

"""EV3 (evaluation) -- error of the existing networks at the instance-optimal operating points.

Networks: the ten archive LSTMs of ZW1 (excited-trained and constant-trained, five seeds each),
read-only, plus the five-seed ensemble mean of each arm (EV2). Nothing is trained. Data:
``verify_ev3_dataset`` (50 runs, 25 oracle scenarios x 2 replicates). Moisture channel, n_train=64,
h=36 = forecast steps 1..36 pooled, windows as in Befund 51 (history 32, horizon 36, stride 8),
standardised with each network's own training statistics.

Definitions fixed BEFORE the first evaluation:

* "settled" windows: forecast starts in the second half of the run (window start >= 544 steps);
  "all" windows include the initial transient. Main population: settled windows of the 17
  scenarios whose oracle sits on the 0.20 kg/kg limit ("binding", 34 runs); the 8 non-binding
  scenarios are reported separately.
* Same estimator as the global figure of Befund 62 HA1: run-median over runs of the per-run RMSE
  against the measured reference (E1); pooled (E2) and pooled against the truth (E3) alongside;
  bias = mean(pred - truth) (negative = sensor reports drier than true), spread = std of
  (pred - truth); E4 as in ZW6.
* Reserve factor at the 90 % mark as in ZW6: sigma_max 0.003181 / error, margin 0.009544 /
  (max(0, -bias) + 3 spread).
* Pre-registered reading, applied to E1 (settled, binding) divided by the arm's global holdout
  run-median (5-seed median of ``global_run_median_all_runs`` of ``verify_zw3_same_estimator``):
  ratio <= 1.5 -> "about like the global median run: the sensor is good where the controller
  operates"; ratio >= 2.0 -> "clearly higher: the optimum is hard to predict"; between -> reported as
  intermediate. Same thresholds for the ensemble.
* Uncertainty: bootstrap over SCENARIOS (both replicates of a scenario move together), 10000 draws.

    python scripts/verify/ev3_evaluate.py
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from _common import ARCHIVE_ROOT, RUNS_DIR, provenance, require_clean_tree, write_artifact
from _data import make_data
from pr3_metrics import make_windows, net_predict, train_scaler
from scipy.stats import spearmanr

SEEDS = (1, 2, 3, 4, 5)
MOIST = 1
CKPT = ARCHIVE_ROOT / "artifacts" / "checkpoints"
SETTLED_START = 544
B = 10_000
PREFIX = {"excited": "lstm", "constant": "constant_trained_lstm"}
KIND = {"excited": "excited", "constant": "constant_train"}


def dataset() -> SimpleNamespace:
    z = np.load(RUNS_DIR / "raw_ev3" / "dataset.npz")
    traj = SimpleNamespace(
        run_id=tuple(str(r) for r in z["run_id"]),
        measurements=z["measurements"],
        outputs=z["outputs"],
        actions=z["actions"],
        disturbances=z["disturbances"],
        requested_actions=None,
    )
    return SimpleNamespace(trajectory=traj, scenario=z["scenario"], replicate=z["replicate"])


def _e1(st: dict[int, tuple[float, int]], runs: list[int]) -> float:
    return float(np.median([np.sqrt(st[r][0] / st[r][1]) for r in runs]))


def per_run_rmse(err: np.ndarray, run: np.ndarray) -> np.ndarray:
    runs = np.unique(run)
    return np.array([np.sqrt(np.mean(err[run == r] ** 2)) for r in runs])


def summarise(
    pred: np.ndarray, p: dict, mask: np.ndarray, sigma_max: float, m90: float
) -> dict[str, float]:
    e_m = (pred - p["measured"])[mask]
    e_t = (pred - p["truth"])[mask]
    run = p["run"][mask]
    bias, spread = float(e_t.mean()), float(e_t.std())
    e1 = float(np.median(per_run_rmse(e_m, run)))
    e2 = float(np.sqrt(np.mean(e_m**2)))
    e3 = float(np.sqrt(np.mean(e_t**2)))
    return {
        "n_points": int(mask.sum()),
        "n_runs": int(np.unique(run).size),
        "E1_run_median_measured": e1,
        "E2_pooled_measured": e2,
        "E3_pooled_truth": e3,
        "bias": bias,
        "spread": spread,
        "reserve_E1": sigma_max / e1,
        "reserve_E2": sigma_max / e2,
        "reserve_E3": sigma_max / e3,
        "m_req_dangerous": max(0.0, -bias) + 3 * spread,
        "m_req_worst": abs(bias) + 3 * spread,
        "reserve_E4_dangerous": m90 / (max(0.0, -bias) + 3 * spread),
        "reserve_E4_worst": m90 / (abs(bias) + 3 * spread),
    }


def main() -> None:
    import torch

    require_clean_tree()
    torch.set_num_threads(1)
    ref = json.loads((RUNS_DIR / "probe_soft_sensor_margin_requirement.json").read_text("utf-8"))
    sigma_max = ref["interpolation"]["0.90"]["sigma_max_kg_per_kg"]
    m90 = ref["interpolation"]["0.90"]["interpolated_margin_kg_per_kg"]
    zw3 = json.loads((RUNS_DIR / "verify_zw3_same_estimator.json").read_text("utf-8"))
    global_ref = {
        arm: zw3[arm]["aggregate_5_seeds"]["vs_measured"]["global_run_median_all_runs"]["median"]
        for arm in ("excited", "constant")
    }
    zw6 = json.loads((RUNS_DIR / "verify_zw6_reserve.json").read_text("utf-8"))
    ds = dataset()
    ids = ds.trajectory.run_id
    scen_of_run = ds.scenario
    binding = {
        s["scenario"]: s["moisture_binds_at_oracle"]
        for s in json.loads((RUNS_DIR / "verify_ev3_dataset.json").read_text("utf-8"))["scenarios"]
    }
    out: dict[str, object] = {
        "sigma_max_90": sigma_max,
        "margin_90": m90,
        "global_holdout_run_median_by_arm": global_ref,
        "holdout_class_reserve_E1_median_zw6": {
            a: zw6[a]["aggregate_5_seeds"]["reserve_E1"]["median"] for a in ("excited", "constant")
        },
        "settled_start_step": SETTLED_START,
    }
    rng = np.random.default_rng(20260933)
    n_scen = 25
    for arm in ("excited", "constant"):
        preds, p_ref = [], None
        for seed in SEEDS:
            pool = make_data(KIND[arm], seed)
            w = make_windows(ds, ids, train_scaler(pool, 64))
            pred, _ = net_predict(CKPT / f"{PREFIX[arm]}_seed{seed}_n64.pt", w)
            preds.append(pred[..., MOIST])
            if p_ref is None:
                p_ref = {
                    "truth": w["truth"][..., MOIST].reshape(-1),
                    "measured": w["ref"][..., MOIST].reshape(-1),
                    "run": np.repeat(w["run"], 36),
                    "start": np.repeat(w["start"], 36),
                    "step": np.tile(np.arange(1, 37), len(w["run"])),
                }
            print(arm, seed, flush=True)
        p = p_ref
        preds_flat = [x.reshape(-1) for x in preds]
        ens = np.mean(preds_flat, axis=0)
        spread_net = np.std(preds_flat, axis=0, ddof=1)
        scen_pt = scen_of_run[p["run"]]
        bind_pt = np.array([binding[s] for s in scen_pt])
        settled = p["start"] >= SETTLED_START
        pops = {
            "settled_binding": settled & bind_pt,
            "settled_all": settled,
            "settled_nonbinding": settled & ~bind_pt,
            "all_windows_binding": bind_pt,
            "settled_binding_step1": settled & bind_pt & (p["step"] == 1),
        }
        res: dict[str, object] = {}
        for pop, mask in pops.items():
            res[pop] = {
                "networks": [summarise(x, p, mask, sigma_max, m90) for x in preds_flat],
                "ensemble": summarise(ens, p, mask, sigma_max, m90),
            }
            keys = [k for k in res[pop]["networks"][0] if k not in ("n_points", "n_runs")]
            res[pop]["network_median"] = {
                k: float(np.median([n[k] for n in res[pop]["networks"]])) for k in keys
            }
            res[pop]["network_min"] = {
                k: float(np.min([n[k] for n in res[pop]["networks"]])) for k in keys
            }
            res[pop]["network_max"] = {
                k: float(np.max([n[k] for n in res[pop]["networks"]])) for k in keys
            }
        g = global_ref[arm]
        main_pop = res["settled_binding"]
        res["reading"] = {
            "global_holdout_run_median": g,
            "ratio_network_median_E1_over_global": main_pop["network_median"][
                "E1_run_median_measured"
            ]
            / g,
            "ratio_ensemble_E1_over_global": main_pop["ensemble"]["E1_run_median_measured"] / g,
            "ratio_network_min_max": [
                main_pop["network_min"]["E1_run_median_measured"] / g,
                main_pop["network_max"]["E1_run_median_measured"] / g,
            ],
        }

        # per scenario (settled): ensemble and network-median error/offset
        per_scen = []
        for s in range(n_scen):
            m = settled & (scen_pt == s)
            row = {"scenario": s, "binding": binding[s], "truth_mean": float(p["truth"][m].mean())}
            row["ensemble_offset"] = float((ens - p["truth"])[m].mean())
            row["ensemble_spread_between_nets"] = float(spread_net[m].mean())
            row["ensemble_rmse_measured_pooled"] = float(
                np.sqrt(np.mean((ens - p["measured"])[m] ** 2))
            )
            offs = [float((x - p["truth"])[m].mean()) for x in preds_flat]
            row["network_offsets"] = offs
            row["network_median_abs_offset"] = float(np.median(np.abs(offs)))
            per_scen.append(row)
        res["per_scenario_settled"] = per_scen
        bs = [r for r in per_scen if r["binding"]]
        res["offset_summary_binding_scenarios"] = {
            "ensemble_share_negative_offset_drier": float(
                np.mean([r["ensemble_offset"] < 0 for r in bs])
            ),
            "ensemble_median_abs_offset": float(np.median([abs(r["ensemble_offset"]) for r in bs])),
            "network_median_abs_offset": float(
                np.median([r["network_median_abs_offset"] for r in bs])
            ),
            "share_network_offsets_negative": float(
                np.mean([o < 0 for r in bs for o in r["network_offsets"]])
            ),
        }

        # spread as uncertainty on this set
        m = pops["settled_binding"]
        e_abs = np.abs(ens - p["truth"])[m]
        s_ = spread_net[m]
        res["spread_as_uncertainty_settled_binding"] = {
            "spearman_abs_error_vs_spread": float(spearmanr(e_abs, s_).statistic),
            "share_within_2_spread": float((e_abs <= 2 * s_).mean()),
            "median_spread": float(np.median(s_)),
            "median_abs_error": float(np.median(e_abs)),
        }

        # bootstrap over scenarios (binding scenarios, settled): E1/E2 reserve, ensemble and network median
        bind_scen = [s for s in range(n_scen) if binding[s]]
        run_stats = {}
        for name, pr in (("ensemble", ens), *[(f"net{i}", x) for i, x in enumerate(preds_flat)]):
            e = pr - p["measured"]
            per_run = {}
            for r in np.unique(p["run"][pops["settled_binding"]]):
                mk = pops["settled_binding"] & (p["run"] == r)
                per_run[int(r)] = (float(np.sum(e[mk] ** 2)), int(mk.sum()))
            run_stats[name] = per_run
        runs_of_scen = {s: [int(r) for r in np.where(scen_of_run == s)[0]] for s in bind_scen}
        boot: dict[str, list[float]] = {"ens_E1": [], "ens_E2": [], "net_median_E1": []}
        for _ in range(B):
            draw = rng.choice(bind_scen, size=len(bind_scen), replace=True)
            runs = [r for s in draw for r in runs_of_scen[s]]

            boot["ens_E1"].append(sigma_max / _e1(run_stats["ensemble"], runs))
            st = run_stats["ensemble"]
            boot["ens_E2"].append(
                sigma_max
                / float(np.sqrt(sum(st[r][0] for r in runs) / sum(st[r][1] for r in runs)))
            )
            boot["net_median_E1"].append(
                sigma_max
                / float(np.median([_e1(run_stats[f"net{i}"], runs) for i in range(len(SEEDS))]))
            )
        res["bootstrap_95_over_scenarios"] = {
            k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in boot.items()
        }
        out[arm] = res
        sb = res["settled_binding"]
        print(arm, "reading", res["reading"])
        print(arm, "netmed", {k: round(v, 5) for k, v in sb["network_median"].items()})
        print(arm, "ens", {k: round(v, 5) for k, v in sb["ensemble"].items()})
        print(
            arm,
            "nonbinding netmed E1",
            res["settled_nonbinding"]["network_median"]["E1_run_median_measured"],
            "offsets",
            res["offset_summary_binding_scenarios"],
        )
        print(
            arm,
            "boot",
            res["bootstrap_95_over_scenarios"],
            res["spread_as_uncertainty_settled_binding"],
        )
    write_artifact("verify_ev3_evaluate", {"_provenance": provenance(), **out})


if __name__ == "__main__":
    main()

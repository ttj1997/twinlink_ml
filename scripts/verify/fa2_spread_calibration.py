"""FA2 -- calibration of the between-network spread on VALIDATION data, checked on EV3.

The spread between the five networks covered only 68 % (excited) / 81 % (constant) of the holdout class
window-steps within twice the spread (EV2c). A scale factor k for the spread is determined on the
validation runs of the training pools, NOT on EV3 and NOT on the holdout, so that ``|ensemble - truth| <=
2 k spread`` holds for 95 % of the validation window-steps (pooled over the five pools of the arm).

Validation data: the 32 validation runs of each of the five pools of the arm (validation split of the
Befund 51/55 protocol, the block used for early stopping of the network trained on that pool). To keep the
data unseen by the ensemble the PRIMARY calibration is leave-one-out: for the validation runs of pool s the
ensemble is the mean of the four networks NOT trained on pool s (spread with ddof=1 over these four);
sensitivity: all five networks (contains the early-stopping bias of network s). The excited-trained arm's
validation runs are excitation runs (90 % dynamic), the constant-trained arm's are constant runs -- the
distribution differs from EV3 for the excited arm, which is what the check on EV3 tests. The factor is
applied to the five-network spread on EV3 (against the truth).

Pre-registered criteria for "usable", fixed before the first evaluation, on EV3 (settled windows of the 34
runs of the 17 limit-binding scenarios), calibrated band ``2 k spread``, error against the truth:
(i) overall coverage in [90 %, 99 %]; (ii) at least 80 % of the 17 scenarios have a coverage >= 85 %
(an uncertainty per operating point must hold point by point, not only on average); (iii) rank correlation
across the 17 scenarios between the scenario's absolute mean error (its offset) and its mean spread >= 0.5
(the spread must rank the operating points by their uncertainty). All three must hold.

    python scripts/verify/fa2_spread_calibration.py
"""

from __future__ import annotations

import json

import numpy as np
from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact
from _data import HOLDOUT_SEED, make_data
from _fa import ARMS, CKPT, KIND, MOIST, PREFIX, SEEDS, load_dataset, points, predictions
from _zw import load as zw_load
from pr3_metrics import make_windows, net_predict, train_scaler
from scipy.stats import spearmanr

SETTLED_START = 544
TARGET = 0.95
B = 10_000


def coverage(err: np.ndarray, spread: np.ndarray, k: float) -> float:
    return float(np.mean(np.abs(err) <= 2 * k * spread))


def solve_k(err: np.ndarray, spread: np.ndarray) -> float:
    """Smallest k with coverage >= TARGET (quantile of |err| / (2 spread))."""
    ok = spread > 0
    return float(np.quantile(np.abs(err[ok]) / (2 * spread[ok]), TARGET))


def main() -> None:
    import torch

    require_clean_tree()
    torch.set_num_threads(1)
    rng = np.random.default_rng(20260935)
    ev3 = load_dataset("raw_ev3/dataset.npz")
    ids3 = ev3.trajectory.run_id
    binding = np.array(
        [
            s["moisture_binds_at_oracle"]
            for s in json.loads((RUNS_DIR / "verify_ev3_dataset.json").read_text("utf-8"))[
                "scenarios"
            ]
        ]
    )
    p3 = points(ev3, ids3)
    scen_pt = ev3.scenario[p3["run"]]
    settled_binding = (p3["start"] >= SETTLED_START) & binding[scen_pt]
    out: dict[str, object] = {"target_coverage": TARGET, "settled_start_step": SETTLED_START}

    for arm in ARMS:
        pools = {s: make_data(KIND[arm], s) for s in SEEDS}
        scal = {s: train_scaler(pools[s], 64) for s in SEEDS}
        # predictions of every network on every pool's validation runs
        val_pred, val_truth = {}, {}
        for s in SEEDS:
            w_truth = None
            for m in SEEDS:
                w = make_windows(pools[s], pools[s].split.val, scal[m])
                pred, _ = net_predict(CKPT / f"{PREFIX[arm]}_seed{m}_n64.pt", w)
                val_pred[(s, m)] = pred[..., MOIST].reshape(-1)
                w_truth = w["truth"][..., MOIST].reshape(-1)
            val_truth[s] = w_truth
            print(arm, "val pool", s, flush=True)

        res: dict[str, object] = {}
        for label, members_of in (
            ("primary_leave_one_out", lambda s: [m for m in SEEDS if m != s]),
            ("sensitivity_all_five", lambda _s: list(SEEDS)),
        ):
            errs, sprs, per_set = [], [], {}
            for s in SEEDS:
                stack = np.stack([val_pred[(s, m)] for m in members_of(s)])
                e = stack.mean(axis=0) - val_truth[s]
                sp = stack.std(axis=0, ddof=1)
                errs.append(e)
                sprs.append(sp)
                per_set[str(s)] = {
                    "k": solve_k(e, sp),
                    "coverage_uncalibrated_2spread": coverage(e, sp, 1.0),
                }
            e_all, s_all = np.concatenate(errs), np.concatenate(sprs)
            k = solve_k(e_all, s_all)
            res[label] = {
                "k": k,
                "coverage_at_k": coverage(e_all, s_all, k),
                "coverage_uncalibrated_2spread": coverage(e_all, s_all, 1.0),
                "k_per_validation_set": per_set,
                "n_points": int(e_all.size),
            }

        # application: EV3 (five-network spread), against the truth
        pr3 = predictions(arm, "ev3", ev3, ids3)
        ens3, sp3 = pr3.mean(axis=0), pr3.std(axis=0, ddof=1)
        err3 = ens3 - p3["truth"]
        m = settled_binding
        ev: dict[str, object] = {}
        for label in ("primary_leave_one_out", "sensitivity_all_five"):
            k = res[label]["k"]
            band = np.abs(err3[m]) <= 2 * k * sp3[m]
            scen_cov = []
            scen_off, scen_spread = [], []
            for sidx in np.where(binding)[0]:
                ms = m & (scen_pt == sidx)
                scen_cov.append(float(np.mean(np.abs(err3[ms]) <= 2 * k * sp3[ms])))
                scen_off.append(abs(float(err3[ms].mean())))
                scen_spread.append(float(sp3[ms].mean()))
            rho = spearmanr(scen_off, scen_spread).statistic
            # bootstrap over scenarios for the overall coverage
            bsc = np.where(binding)[0]
            per_scen_hits = {
                s: (
                    int(
                        np.sum(np.abs(err3[m & (scen_pt == s)]) <= 2 * k * sp3[m & (scen_pt == s)])
                    ),
                    int((m & (scen_pt == s)).sum()),
                )
                for s in bsc
            }
            boots = []
            for _ in range(B):
                d = rng.choice(bsc, size=len(bsc), replace=True)
                boots.append(
                    sum(per_scen_hits[s][0] for s in d) / sum(per_scen_hits[s][1] for s in d)
                )
            crit = {
                "i_overall_in_90_99": bool(0.90 <= float(band.mean()) <= 0.99),
                "ii_share_scenarios_cov_ge_85": float(np.mean(np.array(scen_cov) >= 0.85)),
                "ii_pass": bool(np.mean(np.array(scen_cov) >= 0.85) >= 0.80),
                "iii_spearman_offset_vs_spread": float(rho),
                "iii_pass": bool(rho >= 0.5),
            }
            crit["usable"] = bool(
                crit["i_overall_in_90_99"] and crit["ii_pass"] and crit["iii_pass"]
            )
            ev[label] = {
                "k": k,
                "coverage_overall": float(band.mean()),
                "coverage_overall_bootstrap_95": [
                    float(np.percentile(boots, 2.5)),
                    float(np.percentile(boots, 97.5)),
                ],
                "coverage_uncalibrated_2spread": float(np.mean(np.abs(err3[m]) <= 2 * sp3[m])),
                "per_scenario_coverage": scen_cov,
                "per_scenario_abs_offset": scen_off,
                "per_scenario_mean_spread": scen_spread,
                "median_band_half_width_2k_spread": float(np.median(2 * k * sp3[m])),
                "median_abs_error": float(np.median(np.abs(err3[m]))),
                "criteria": crit,
            }
        res["ev3_settled_binding"] = ev

        # information: holdout test (constant-run holdout), same factor
        nets_h = [zw_load(arm, sd) for sd in SEEDS]
        pr_h = np.stack([n["pred"] for n in nets_h])
        eh, sh = pr_h.mean(axis=0) - nets_h[0]["truth"], pr_h.std(axis=0, ddof=1)
        cls = (nets_h[0]["truth"] >= 0.18) & (nets_h[0]["truth"] < 0.22)
        k = res["primary_leave_one_out"]["k"]
        res["holdout_test_information"] = {
            "coverage_global": coverage(eh, sh, k),
            "coverage_class": coverage(eh[cls], sh[cls], k),
            "coverage_uncalibrated_global": coverage(eh, sh, 1.0),
            "coverage_uncalibrated_class": coverage(eh[cls], sh[cls], 1.0),
        }
        out[arm] = res
        print(
            arm,
            json.dumps(
                {k_: v for k_, v in res.items() if k_ != "ev3_settled_binding"}, default=float
            )[:900],
        )
        print(
            arm,
            json.dumps(
                {
                    k_: {kk: vv for kk, vv in v.items() if not kk.startswith("per_scenario")}
                    for k_, v in ev.items()
                },
                default=float,
            )[:1500],
            flush=True,
        )
    write_artifact("verify_fa2_spread_calibration", {"_provenance": provenance(), **out})
    _ = HOLDOUT_SEED


if __name__ == "__main__":
    main()

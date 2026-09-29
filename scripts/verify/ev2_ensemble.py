"""EV2 -- five-seed ensemble from the ZW1 raw predictions (evaluation only, nothing is trained).

Per arm the ensemble prediction of a window-step is the mean of the five networks' predictions.
Members differ in initialisation AND in training data (each seed has its own pool).

a) error of the ensemble against the median of the single networks, same estimators (run-median
   and pooled, against the measured reference and against the truth), globally and in the class
   [0.18, 0.22) of the true moisture; the quotients A/B/C/D of ZW3 for the ensemble;
b) per run: offset (mean of ``pred - truth``) of the single networks and of the ensemble, in the
   class runs (in class) and over the 13 runs that exceed 0.10 kg/kg (whole run);
c) spread between the networks (std over the five members, ddof=1) as uncertainty estimate: rank and
   linear correlation with the absolute ensemble error (against the truth), the share of
   window-steps with ``|ensemble - truth| <= 2 spread`` (globally and in the class), the median of
   ``|error| / spread``, and the run-level correlation of |offset| with the spread;
d) reserve factor at the 90 % mark with the ensemble (E1-E4 as in ZW6; 95 % interval over runs).

    python scripts/verify/ev2_ensemble.py
"""

from __future__ import annotations

import json

import numpy as np
from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact
from _zw import ARMS, N_RUNS, SEEDS, in_class, load, rmse_pooled, rmse_run_median
from scipy.stats import pearsonr, spearmanr

LO, HI = 0.18, 0.22
B = 10_000


def estimators(pred: np.ndarray, p: dict[str, np.ndarray]) -> dict[str, float]:
    cls = in_class(p["truth"], LO, HI)
    same = np.isin(p["run"], np.unique(p["run"][cls]))
    out: dict[str, float] = {}
    for ref in ("measured", "truth"):
        e = pred - p[ref]
        out[f"global_run_median_{ref}"] = rmse_run_median(e, p["run"])
        out[f"global_pooled_{ref}"] = rmse_pooled(e)
        out[f"local_run_median_{ref}"] = rmse_run_median(e[cls], p["run"][cls])
        out[f"local_pooled_{ref}"] = rmse_pooled(e[cls])
        out[f"A_{ref}"] = out[f"local_run_median_{ref}"] / out[f"global_run_median_{ref}"]
        out[f"B_{ref}"] = out[f"local_pooled_{ref}"] / out[f"global_pooled_{ref}"]
        out[f"C_{ref}"] = out[f"local_run_median_{ref}"] / rmse_run_median(e[same], p["run"][same])
        out[f"D_{ref}"] = out[f"local_pooled_{ref}"] / rmse_pooled(e[same])
    return out


def main() -> None:
    require_clean_tree()
    ref = json.loads((RUNS_DIR / "probe_soft_sensor_margin_requirement.json").read_text("utf-8"))
    sigma_max = ref["interpolation"]["0.90"]["sigma_max_kg_per_kg"]
    m90 = ref["interpolation"]["0.90"]["interpolated_margin_kg_per_kg"]
    rng = np.random.default_rng(20260930)
    draws = rng.integers(0, N_RUNS, size=(B, N_RUNS))
    out: dict[str, object] = {"sigma_max_90": sigma_max, "margin_90": m90, "class": [LO, HI]}
    for arm in ARMS:
        nets = [load(arm, s) for s in SEEDS]
        for n in nets[1:]:
            assert np.array_equal(n["truth"], nets[0]["truth"]) and np.array_equal(
                n["run"], nets[0]["run"]
            )
        p = nets[0]
        preds = np.stack([n["pred"] for n in nets])  # 5 x points
        ens = preds.mean(axis=0)
        spread = preds.std(axis=0, ddof=1)
        cls = in_class(p["truth"], LO, HI)
        res: dict[str, object] = {}

        # a) estimators
        single = [estimators(preds[i], p) for i in range(len(SEEDS))]
        ensemble = estimators(ens, p)
        res["a_single_median"] = {k: float(np.median([s[k] for s in single])) for k in ensemble}
        res["a_single_min"] = {k: float(np.min([s[k] for s in single])) for k in ensemble}
        res["a_single_max"] = {k: float(np.max([s[k] for s in single])) for k in ensemble}
        res["a_ensemble"] = ensemble
        res["a_ratio_ensemble_over_single_median"] = {
            k: ensemble[k] / res["a_single_median"][k]
            for k in ensemble
            if k.startswith(("global", "local"))
        }

        # b) offsets per run
        runs_class = [int(r) for r in np.unique(p["run"][cls])]
        runs_dyn = [r for r in range(N_RUNS) if p["truth"][p["run"] == r].max() > 0.10]
        rows = []
        for r in sorted(set(runs_class) | set(runs_dyn)):
            m = p["run"] == r
            c = m & cls
            row = {"run": r, "in_class_run": r in runs_class, "n_class_points": int(c.sum())}
            for label, mask in (("whole_run", m), ("in_class", c)):
                if mask.sum() == 0:
                    continue
                off_single = [
                    float((preds[i][mask] - p["truth"][mask]).mean()) for i in range(len(SEEDS))
                ]
                off_ens = float((ens[mask] - p["truth"][mask]).mean())
                row[label] = {
                    "single_offsets": off_single,
                    "single_median_abs": float(np.median(np.abs(off_single))),
                    "single_rms": float(np.sqrt(np.mean(np.square(off_single)))),
                    "ensemble_offset": off_ens,
                    "ensemble_abs_over_single_median_abs": abs(off_ens)
                    / float(np.median(np.abs(off_single))),
                    "spread_of_single_offsets_std": float(np.std(off_single, ddof=1)),
                    "ensemble_abs_over_single_rms": abs(off_ens)
                    / float(np.sqrt(np.mean(np.square(off_single)))),
                }
            rows.append(row)
        res["b_offsets_per_run"] = rows
        cl_rows = [r for r in rows if r["in_class_run"]]
        res["b_class_runs_summary"] = {
            "median_single_abs_offset_in_class": float(
                np.median([r["in_class"]["single_median_abs"] for r in cl_rows])
            ),
            "median_ensemble_abs_offset_in_class": float(
                np.median([abs(r["in_class"]["ensemble_offset"]) for r in cl_rows])
            ),
            "mean_of_ensemble_over_single_rms": float(
                np.mean([r["in_class"]["ensemble_abs_over_single_rms"] for r in cl_rows])
            ),
            "note": "1/sqrt(5)=0.447 would mean fully independent member offsets, 1 = fully common offset",
        }

        # c) spread as uncertainty
        err_t = ens - p["truth"]
        c_res = {}
        for label, m in (("global", np.ones_like(cls)), ("class", cls)):
            ok = spread[m] > 0
            e, s = np.abs(err_t[m][ok]), spread[m][ok]
            c_res[label] = {
                "n": int(ok.sum()),
                "pearson_abs_error_vs_spread": float(pearsonr(e, s).statistic),
                "spearman_abs_error_vs_spread": float(spearmanr(e, s).statistic),
                "share_within_2_spread": float((e <= 2 * s).mean()),
                "median_abs_error_over_spread": float(np.median(e / s)),
                "median_spread": float(np.median(s)),
                "median_abs_error": float(np.median(e)),
            }
        # run-level: |ensemble offset| in the run vs mean spread in the run (runs with > 200 points)
        lv = [
            (abs(float(err_t[p["run"] == r].mean())), float(spread[p["run"] == r].mean()))
            for r in range(N_RUNS)
        ]
        res["c_spread"] = c_res
        res["c_run_level"] = {
            "spearman_abs_run_offset_vs_mean_spread": float(
                spearmanr(*zip(*lv, strict=True)).statistic
            ),
            "n_runs": len(lv),
        }
        # spread in the class vs outside (does the ensemble know the class is hard?)
        res["c_spread_class_vs_rest"] = {
            "median_spread_class": float(np.median(spread[cls])),
            "median_spread_rest": float(np.median(spread[~cls])),
        }

        # d) reserve with the ensemble
        e_meas, e_true = ens - p["measured"], ens - p["truth"]
        bias, sd = float(e_true[cls].mean()), float(e_true[cls].std())
        point = {
            "E1_local_run_median_measured": rmse_run_median(e_meas[cls], p["run"][cls]),
            "E2_local_pooled_measured": rmse_pooled(e_meas[cls]),
            "E3_local_pooled_truth": rmse_pooled(e_true[cls]),
            "bias_class": bias,
            "spread_class": sd,
            "m_req_dangerous_direction": max(0.0, -bias) + 3 * sd,
            "m_req_worst_case": abs(bias) + 3 * sd,
            "global_run_median_measured": rmse_run_median(e_meas, p["run"]),
        }
        rsv = {
            "reserve_E1": sigma_max / point["E1_local_run_median_measured"],
            "reserve_E2": sigma_max / point["E2_local_pooled_measured"],
            "reserve_E3": sigma_max / point["E3_local_pooled_truth"],
            "reserve_E4_dangerous": m90 / point["m_req_dangerous_direction"],
            "reserve_E4_worst_case": m90 / point["m_req_worst_case"],
            "reserve_global_run_median": sigma_max / point["global_run_median_measured"],
        }
        # bootstrap over runs for E1 and E2/E3
        sse_m = np.bincount(p["run"][cls], weights=e_meas[cls] ** 2, minlength=N_RUNS)
        sse_t = np.bincount(p["run"][cls], weights=e_true[cls] ** 2, minlength=N_RUNS)
        cnt = np.bincount(p["run"][cls], minlength=N_RUNS).astype(float)
        e1, e2, e3 = [], [], []
        for d in draws:
            have = cnt[d] > 0
            if not have.any():
                continue
            e1.append(sigma_max / np.median(np.sqrt(sse_m[d][have] / cnt[d][have])))
            e2.append(sigma_max / np.sqrt(sse_m[d].sum() / cnt[d].sum()))
            e3.append(sigma_max / np.sqrt(sse_t[d].sum() / cnt[d].sum()))
        ci = {
            k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in (("reserve_E1", e1), ("reserve_E2", e2), ("reserve_E3", e3))
        }
        # same for the single networks, for the comparison in the table
        res["d_ensemble"] = {**point, **rsv, "bootstrap_95_over_runs": ci}
        res["d_single_reserve_E1_E2_E3"] = {
            k: [
                sigma_max / s["local_run_median_measured"]
                if k == "E1"
                else sigma_max / s["local_pooled_measured" if k == "E2" else "local_pooled_truth"]
                for s in single
            ]
            for k in ("E1", "E2", "E3")
        }
        out[arm] = res
        print(arm, "ensemble", {k: round(v, 3) for k, v in rsv.items()})
        print(
            arm,
            "A..D meas ens",
            {
                k: round(ensemble[k], 3)
                for k in ("A_measured", "B_measured", "C_measured", "D_measured")
            },
        )
        print(
            arm,
            "single med",
            {
                k: round(res["a_single_median"][k], 5)
                for k in (
                    "global_run_median_measured",
                    "local_run_median_measured",
                    "local_pooled_measured",
                )
            },
            "ens",
            {
                k: round(ensemble[k], 5)
                for k in (
                    "global_run_median_measured",
                    "local_run_median_measured",
                    "local_pooled_measured",
                )
            },
        )
        print(arm, "b", res["b_class_runs_summary"])
        print(arm, "c", c_res, res["c_run_level"], res["c_spread_class_vs_rest"])
    write_artifact("verify_ev2_ensemble", {"_provenance": provenance(), **out})


if __name__ == "__main__":
    main()

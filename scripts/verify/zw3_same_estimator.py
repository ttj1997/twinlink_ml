"""ZW3 -- same estimator on both sides of the boundary/global quotient.

GA2's quotient (Befund 65) is ``local / global`` with the local value a run-median over the (few)
runs that reach the class and the global value a run-median over all 32 runs. Which estimator sits
where, and what happens if the quotient is formed with the SAME estimator and the SAME runs on
both sides? Four quotients per (arm, seed), all in physical units (kg/kg), error against the
measured reference (the metric of the project), class [0.18, 0.22) on the true moisture, steps
1..36 pooled (h=36):

* A  GA2 as published: run-median over class runs / run-median over all 32 runs
* B  pooled / pooled: RMSE over all class points / RMSE over all points
* C  run-median / run-median on the SAME runs (the runs that reach the class, both sides)
* D  pooled / pooled on the same runs

plus the same four against the noise-free truth (error of the network itself, no sensor noise).

    python scripts/verify/zw3_same_estimator.py
"""

from __future__ import annotations

import numpy as np
from _common import provenance, require_clean_tree, write_artifact
from _zw import ARMS, SEEDS, in_class, load, per_run_rmse, rmse_pooled

LO, HI = 0.18, 0.22


def quotients(err: np.ndarray, run: np.ndarray, cls: np.ndarray) -> dict[str, float]:
    runs_in = np.unique(run[cls])
    same = np.isin(run, runs_in)
    loc_rm = float(np.median(per_run_rmse(err[cls], run[cls])))
    return {
        "global_run_median_all_runs": float(np.median(per_run_rmse(err, run))),
        "global_pooled_all_points": rmse_pooled(err),
        "local_run_median": loc_rm,
        "local_pooled": rmse_pooled(err[cls]),
        "global_run_median_class_runs": float(np.median(per_run_rmse(err[same], run[same]))),
        "global_pooled_class_runs": rmse_pooled(err[same]),
        "A_GA2_runmedian_over_runmedian_all_runs": loc_rm
        / float(np.median(per_run_rmse(err, run))),
        "B_pooled_over_pooled": rmse_pooled(err[cls]) / rmse_pooled(err),
        "C_runmedian_over_runmedian_same_runs": loc_rm
        / float(np.median(per_run_rmse(err[same], run[same]))),
        "D_pooled_over_pooled_same_runs": rmse_pooled(err[cls]) / rmse_pooled(err[same]),
        "n_class_runs": int(runs_in.size),
    }


def main() -> None:
    require_clean_tree()
    out: dict[str, object] = {}
    for arm in ARMS:
        per_seed = {}
        for seed in SEEDS:
            p = load(arm, seed)
            cls = in_class(p["truth"], LO, HI)
            per_seed[seed] = {
                "vs_measured": quotients(p["pred"] - p["measured"], p["run"], cls),
                "vs_truth": quotients(p["pred"] - p["truth"], p["run"], cls),
            }
        agg = {}
        for ref in ("vs_measured", "vs_truth"):
            keys = per_seed[1][ref].keys()
            agg[ref] = {
                k: {
                    "median": float(np.median([per_seed[s][ref][k] for s in SEEDS])),
                    "min": float(np.min([per_seed[s][ref][k] for s in SEEDS])),
                    "max": float(np.max([per_seed[s][ref][k] for s in SEEDS])),
                }
                for k in keys
            }
            # GA2 aggregation: ratio of the cross-seed medians of numerator and denominator
            agg[ref]["A_ratio_of_medians"] = (
                agg[ref]["local_run_median"]["median"]
                / agg[ref]["global_run_median_all_runs"]["median"]
            )
        out[arm] = {"per_seed": per_seed, "aggregate_5_seeds": agg}
    write_artifact(
        "verify_zw3_same_estimator", {"_provenance": provenance(), "class": [LO, HI], **out}
    )
    for arm in ARMS:
        a = out[arm]["aggregate_5_seeds"]["vs_measured"]
        print(
            arm,
            {
                k: round(v["median"], 3)
                for k, v in a.items()
                if k[0] in "ABCD" and isinstance(v, dict)
            },
        )
        print("  A ratio of medians", a["A_ratio_of_medians"])


if __name__ == "__main__":
    main()

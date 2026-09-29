"""ZW4b -- the runs behind the class [0.18, 0.22): who they are and how their error behaves.

Addendum to ZW4 (the composition there is counts only). Per contributing run and (arm, seed):
points in the class and their share of the class, moisture range and mean of the whole run,
share of the run inside the class, error of the run inside the class and over the whole run
(against the measured reference, pooled within the run), signed offset ``pred - truth`` in the
class. Also the distribution of the whole-run error over all 32 holdout runs (quantiles), which is
what the global run-median is the median of.

    python scripts/verify/zw4b_class_runs.py
"""

from __future__ import annotations

import numpy as np
from _common import provenance, require_clean_tree, write_artifact
from _zw import ARMS, N_RUNS, SEEDS, in_class, load, per_run_rmse, rmse_pooled

LO, HI = 0.18, 0.22


def main() -> None:
    require_clean_tree()
    out: dict[str, object] = {"class": [LO, HI]}
    p0 = load("excited", 1)
    cls = in_class(p0["truth"], LO, HI)
    runs = [int(r) for r in np.unique(p0["run"][cls])]
    static = []
    for r in runs:
        m = p0["run"] == r
        static.append(
            {
                "run": r,
                "n_class_points": int((cls & m).sum()),
                "share_of_class": float((cls & m).sum() / cls.sum()),
                "n_run_points": int(m.sum()),
                "share_of_run_in_class": float((cls & m).sum() / m.sum()),
                "truth_min": float(p0["truth"][m].min()),
                "truth_max": float(p0["truth"][m].max()),
                "truth_mean": float(p0["truth"][m].mean()),
                "truth_std": float(p0["truth"][m].std()),
                "reaches_0.20": bool(p0["truth"][m].max() >= 0.20),
            }
        )
    out["class_runs"] = static
    out["runs_with_any_truth_above_0.10"] = [
        {"run": r, "truth_max": float(p0["truth"][p0["run"] == r].max())}
        for r in range(N_RUNS)
        if p0["truth"][p0["run"] == r].max() > 0.10
    ]
    for arm in ARMS:
        rows = []
        quantiles = {}
        for seed in SEEDS:
            p = load(arm, seed)
            e = p["pred"] - p["measured"]
            d = p["pred"] - p["truth"]
            quantiles[str(seed)] = {
                "whole_run_rmse_quantiles_0_25_50_75_90_100": [
                    float(q)
                    for q in np.percentile(per_run_rmse(e, p["run"]), [0, 25, 50, 75, 90, 100])
                ]
            }
            for r in runs:
                m = p["run"] == r
                c = m & cls
                rows.append(
                    {
                        "seed": seed,
                        "run": r,
                        "rmse_in_class": rmse_pooled(e[c]),
                        "rmse_whole_run": rmse_pooled(e[m]),
                        "offset_in_class_pred_minus_truth": float(d[c].mean()),
                        "spread_in_class": float(d[c].std()),
                    }
                )
        out[arm] = {"per_seed_run": rows, "whole_run_error_distribution": quantiles}
    write_artifact("verify_zw4b_class_runs", {"_provenance": provenance(), **out})
    for s in static:
        print(s)
    for arm in ARMS:
        for row in out[arm]["per_seed_run"]:
            if row["seed"] in (1, 2, 3):
                print(arm, row)


if __name__ == "__main__":
    main()

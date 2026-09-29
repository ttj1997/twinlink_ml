"""ZW2 -- independent recomputation of the Befund 65 GA2 quotient (3.35).

Follows the GA2 procedure as WRITTEN in Befund 65 (not its code): moisture error of the LSTM
(n_train=64, excited-trained) on the constant-run holdout, all forecast steps 1..36 pooled,
error against the measured reference, classes formed on the true (noise-free) moisture;
"global" = run-median RMSE over all points, "boundary" = run-median RMSE over the points with
true moisture in [0.18, 0.22); ratio per seed, then the cross-seed median. The published values
are read from ``probe_soft_sensor_error_by_moisture_class.json`` as DATA (no code import).

    python scripts/verify/zw2_reproduce_ratio.py
"""

from __future__ import annotations

import itertools
import json

import numpy as np
from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact
from _zw import SEEDS, in_class, load, rmse_pooled, rmse_run_median

EDGES = (0.0, 0.10, 0.14, 0.18, 0.22, 0.26, np.inf)


def main() -> None:
    require_clean_tree()
    published = json.loads(
        (RUNS_DIR / "probe_soft_sensor_error_by_moisture_class.json").read_text(encoding="utf-8")
    )
    rows = []
    for seed in SEEDS:
        p = load("excited", seed)
        err = p["pred"] - p["measured"]
        cls = in_class(p["truth"], 0.18, 0.22)
        row = {
            "seed": seed,
            "global_run_median": rmse_run_median(err, p["run"]),
            "boundary_run_median": rmse_run_median(err[cls], p["run"][cls]),
            "global_pooled": rmse_pooled(err),
            "boundary_pooled": rmse_pooled(err[cls]),
            "boundary_n_points": int(cls.sum()),
            "boundary_n_runs": int(np.unique(p["run"][cls]).size),
            "classes": [
                {
                    "class": f"[{lo:.2f},{hi:.2f})",
                    "n": int(m.sum()),
                    "n_runs": int(np.unique(p["run"][m]).size),
                    "rmse_run_median": rmse_run_median(err[m], p["run"][m]),
                    "rmse_pooled": rmse_pooled(err[m]),
                }
                for lo, hi in itertools.pairwise(EDGES)
                for m in [in_class(p["truth"], lo, hi)]
            ],
        }
        row["ratio_run_median"] = row["boundary_run_median"] / row["global_run_median"]
        row["ratio_pooled"] = row["boundary_pooled"] / row["global_pooled"]
        if str(seed) in published["per_seed"]:
            e = published["per_seed"][str(seed)]["error_by_class_test_constant_holdout"]
            row["published"] = {
                "global_run_median": e["global_rmse_per_run_median_kg_per_kg"],
                "boundary_run_median": e["boundary_class_rmse_per_run_median_kg_per_kg"],
                "boundary_pooled": e["boundary_class_rmse_pooled_kg_per_kg"],
                "global_pooled": e["global_rmse_pooled_kg_per_kg"],
                "boundary_n_points": e["boundary_class_n_points"],
                "boundary_n_runs": e["boundary_class_n_runs"],
                "class_n": [c["n_points"] for c in e["per_class"]],
            }
            row["max_rel_diff_vs_published"] = max(
                abs(row[k] - row["published"][k]) / row["published"][k]
                for k in (
                    "global_run_median",
                    "boundary_run_median",
                    "boundary_pooled",
                    "global_pooled",
                )
            )
            row["class_n_equal"] = [c["n"] for c in row["classes"]] == row["published"]["class_n"]
        rows.append(row)

    summary = {}
    for label, seeds in (("seeds_1_3_as_GA2", (1, 2, 3)), ("seeds_1_5", SEEDS)):
        sel = [r for r in rows if r["seed"] in seeds]
        med_b = float(np.median([r["boundary_run_median"] for r in sel]))
        med_g = float(np.median([r["global_run_median"] for r in sel]))
        summary[label] = {
            "median_global": med_g,
            "median_boundary": med_b,
            "ratio_of_medians": med_b / med_g,
            "median_of_ratios": float(np.median([r["ratio_run_median"] for r in sel])),
            "per_seed_ratios": [r["ratio_run_median"] for r in sel],
        }
    write_artifact(
        "verify_zw2_reproduce_ratio",
        {
            "_provenance": provenance(),
            "published_aggregate": published["aggregate_over_seeds"],
            "rows": rows,
            "summary": summary,
        },
    )
    print(json.dumps(summary, indent=1))
    for r in rows:
        print(r["seed"], r.get("max_rel_diff_vs_published"), r.get("class_n_equal"))


if __name__ == "__main__":
    main()

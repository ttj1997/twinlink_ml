"""ZW5b -- training density of the moisture classes (mechanism check for ZW5).

For every (arm, seed) network: how many training window-steps (the 64 training runs, forecast
steps 1..36 of every stride-8 window, the convention of Befund 65 GA2a) and how many distinct
training runs fall into the ZW5 classes of the TRUE moisture. Compared with the holdout error per
class from ``verify_zw5_shape.json`` (rank correlation across classes with at least three holdout
runs). Data are regenerated with the audit's ``make_data`` (same seeds as ZW1).

    python scripts/verify/zw5b_train_density.py
"""

from __future__ import annotations

import itertools
import json

import numpy as np
from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact
from _data import make_data
from _zw import ARMS, SEEDS
from pr3_metrics import make_windows, train_scaler
from scipy.stats import spearmanr

MOIST = 1
EDGES = np.concatenate([np.arange(0.0, 0.4001, 0.02), [0.6, np.inf]])
KIND = {"excited": "excited", "constant": "constant_train"}


def main() -> None:
    require_clean_tree()
    shape = json.loads((RUNS_DIR / "verify_zw5_shape.json").read_text(encoding="utf-8"))
    out: dict[str, object] = {"edges": [float(e) for e in EDGES[:-1]] + [None]}
    for arm in ARMS:
        rows = []
        for seed in SEEDS:
            data = make_data(KIND[arm], seed)
            w = make_windows(data, data.split.subset(64).train, train_scaler(data, 64))
            truth = w["truth"][..., MOIST]
            run = np.repeat(w["run"], truth.shape[1])
            flat = truth.reshape(-1)
            n = np.zeros(len(EDGES) - 1, dtype=int)
            n_runs = np.zeros(len(EDGES) - 1, dtype=int)
            for i, (lo, hi) in enumerate(itertools.pairwise(EDGES)):
                m = (flat >= lo) & (flat < hi)
                n[i] = int(m.sum())
                n_runs[i] = int(np.unique(run[m]).size)
            rows.append(
                {
                    "seed": seed,
                    "n_train_window_steps": int(flat.size),
                    "n_train_runs": int(np.unique(run).size),
                    "counts": n.tolist(),
                    "share": (n / flat.size).tolist(),
                    "n_runs_per_class": n_runs.tolist(),
                    "boundary_class_share": float(((flat >= 0.18) & (flat < 0.22)).mean()),
                    "boundary_class_n_runs": int(
                        np.unique(run[(flat >= 0.18) & (flat < 0.22)]).size
                    ),
                }
            )
            print(arm, seed, flush=True)
        curve = shape[arm]["h36_pooled"]["by_truth"]["curve"]
        share_med = np.median([r["share"] for r in rows], axis=0)
        sel = [
            i
            for i, c in enumerate(curve)
            if c.get("n_runs", 0) >= 3 and c["n"] > 0 and share_med[i] > 0 and 0.0 <= c["lo"] < 0.40
        ]
        rho = spearmanr(np.log(share_med[sel]), [curve[i]["rmse_truth"] for i in sel])
        out[arm] = {
            "per_seed": rows,
            "median_share_per_class": share_med.tolist(),
            "spearman_log_train_share_vs_holdout_rmse_truth": {
                "rho": float(rho.statistic),
                "p": float(rho.pvalue),
                "n_classes": len(sel),
                "classes_lo": [curve[i]["lo"] for i in sel],
            },
            "boundary_class_0.18_0.22": {
                "train_share_per_seed": [r["boundary_class_share"] for r in rows],
                "train_runs_in_class_per_seed": [r["boundary_class_n_runs"] for r in rows],
            },
        }
    write_artifact("verify_zw5b_train_density", {"_provenance": provenance(), **out})
    for arm in ARMS:
        print(arm, out[arm]["spearman_log_train_share_vs_holdout_rmse_truth"])
        print(arm, out[arm]["boundary_class_0.18_0.22"]["train_share_per_seed"])


if __name__ == "__main__":
    main()

"""EV3 addendum -- reference noise level on the oracle-point runs.

RMSE of (measured - truth) moisture on the settled forecast steps of the binding scenarios (same
population as the main EV3 figure), pooled and run-median, next to the same figure on the constant-run
holdout class [0.18, 0.22) (ZW4). The measured reference carries this noise, so errors against it
cannot fall below it.

    python scripts/verify/ev3_noise_floor.py
"""

from __future__ import annotations

import json

import numpy as np
from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact

STEPS_HIST, STEPS_HOR, STRIDE, SETTLED_START = 32, 36, 8, 544


def main() -> None:
    require_clean_tree()
    z = np.load(RUNS_DIR / "raw_ev3" / "dataset.npz")
    scen = json.loads((RUNS_DIR / "verify_ev3_dataset.json").read_text("utf-8"))["scenarios"]
    binding = np.array([s["moisture_binds_at_oracle"] for s in scen])[z["scenario"]]
    truth, meas = (
        z["outputs"][..., 2],
        z["measurements"][..., 2],
    )  # column 2 = product moisture in both
    n_steps = truth.shape[1]
    starts = np.arange(0, n_steps - (STEPS_HIST + STEPS_HOR) + 1, STRIDE)
    starts = starts[starts >= SETTLED_START]
    out = {}
    for label, sel in (("binding", binding), ("nonbinding", ~binding)):
        d_all, per_run = [], []
        for r in np.where(sel)[0]:
            d = np.concatenate(
                [
                    meas[r, s + STEPS_HIST : s + STEPS_HIST + STEPS_HOR]
                    - truth[r, s + STEPS_HIST : s + STEPS_HIST + STEPS_HOR]
                    for s in starts
                ]
            )
            d_all.append(d)
            per_run.append(np.sqrt(np.mean(d**2)))
        d = np.concatenate(d_all)
        out[label] = {
            "n_runs": int(sel.sum()),
            "noise_rmse_pooled": float(np.sqrt(np.mean(d**2))),
            "noise_rmse_run_median": float(np.median(per_run)),
            "noise_mean": float(d.mean()),
        }
    zw4 = json.loads((RUNS_DIR / "verify_zw4_reliability.json").read_text("utf-8"))
    out["holdout_class_0.18_0.22_noise_pooled_zw4"] = zw4["reference_noise_measured_minus_truth"][
        "class_0.18_0.22"
    ]["rmse_pooled"]
    write_artifact("verify_ev3_noise_floor", {"_provenance": provenance(), **out})
    print(out)


if __name__ == "__main__":
    main()

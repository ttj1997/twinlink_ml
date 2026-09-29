"""PR1c (supplement) -- integration error in the units and aggregation of the reported metric.

``pr1c_integration.py`` reports the worst single step, which is dominated by a few extreme
steps. The reported model errors are per-run RMSEs over forecast windows (history 32, horizon
36, stride 8), median over runs. Here the deviation between the 5- and the 10-sub-step
simulation is aggregated exactly like that: a hypothetical model that reproduced the coarse
simulation perfectly is scored against the finer simulation (same windows, RMSE over the
pooled forecast steps of each run, median over runs), and compared with the LSTM errors.

    python scripts/verify/pr1c2_integration_per_run.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
from _common import provenance, require_clean_tree, write_artifact
from _data import make_data
from pr1c_integration import CHANNELS, lstm_reference_errors, rollout_with

HISTORY, HORIZON, STRIDE = 32, 36, 8


def window_rmse_per_run(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-run RMSE of (a - b) over all forecast steps of all windows, shape (runs, channels)."""
    n_runs, n_steps, _ = a.shape
    starts = np.arange(0, n_steps - (HISTORY + HORIZON) + 1, STRIDE)
    rows = []
    for name, col in CHANNELS.items():
        del name
        d2 = (a[..., col] - b[..., col]) ** 2
        acc = np.zeros(n_runs)
        for s in starts:
            acc += d2[:, s + HISTORY : s + HISTORY + HORIZON].sum(axis=1)
        rows.append(np.sqrt(acc / (len(starts) * HORIZON)))
    return np.stack(rows, axis=1)


def main() -> None:
    require_clean_tree()
    reference = lstm_reference_errors()
    pools: list[dict[str, Any]] = []
    for kind, seed, test_set in (
        ("excited", 1, "excited"),
        ("holdout", 999, "constant_holdout"),
        ("constant_train", 1, "constant_holdout"),
    ):
        data = make_data(kind, seed)
        coarse = rollout_with(data, 5)
        fine = rollout_with(data, 10)
        # the test block of the excited pool is what the metric sees; the whole pool is reported too
        blocks = {"all_runs": np.arange(coarse.shape[0])}
        if kind == "excited":
            pos = {rid: i for i, rid in enumerate(data.trajectory.run_id)}
            blocks["test_block"] = np.asarray([pos[r] for r in data.split.test])
        if kind == "holdout":
            pos = {rid: i for i, rid in enumerate(data.trajectory.run_id)}
            blocks["test_block"] = np.asarray([pos[r] for r in data.split.test])
        entry: dict[str, Any] = {"kind": kind, "seed": seed, "lstm_reference_test_set": test_set}
        for block_name, rows in blocks.items():
            per_run = window_rmse_per_run(coarse[rows], fine[rows])
            block: dict[str, Any] = {"n_runs": len(rows)}
            for j, name in enumerate(CHANNELS):
                lstm = reference[test_set][name]
                block[name] = {
                    "median_of_per_run_rmse": float(np.median(per_run[:, j])),
                    "p90_of_per_run_rmse": float(np.percentile(per_run[:, j], 90)),
                    "max_of_per_run_rmse": float(per_run[:, j].max()),
                    "lstm_error_n64_h36_median": lstm,
                    "median_over_lstm_error": float(np.median(per_run[:, j]) / lstm),
                    "max_over_lstm_error": float(per_run[:, j].max() / lstm),
                    "n_runs_above_one_tenth_of_lstm_error": int((per_run[:, j] > lstm / 10).sum()),
                }
            entry[block_name] = block
        pools.append(entry)
        print(
            kind,
            seed,
            {n: entry["all_runs"][n]["median_over_lstm_error"] for n in CHANNELS},
            flush=True,
        )
    write_artifact(
        "verify_pr1c2_integration_per_run", {"_provenance": provenance(), "pools": pools}
    )


if __name__ == "__main__":
    main()

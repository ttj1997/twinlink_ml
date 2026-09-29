"""ZW4 -- how reliable is the boundary/global quotient.

(1) Which runs make up the class [0.18, 0.22) and how the window-steps distribute over them.
(2) Bootstrap over RUNS (32 holdout runs drawn with replacement, 10000 replicates, the same run
    draw for every seed of an arm): 95 % percentile interval of the quotients A (GA2 estimator),
    B (pooled/pooled) and C (run-median/run-median on the class runs of the draw) of ZW3, per
    seed and for the cross-seed statistic. Replicates in which no run reaches the class are
    counted and dropped.
(3) Per-seed quotient (point values, ZW3 definitions).
(4) Noise level of the moisture reference in the class: RMSE of (measured - truth) in the class
    against the same over all points, and the network error in the class against it.

    python scripts/verify/zw4_reliability.py
"""

from __future__ import annotations

import numpy as np
from _common import provenance, require_clean_tree, write_artifact
from _zw import ARMS, N_RUNS, SEEDS, in_class, load, rmse_pooled, run_stats

LO, HI = 0.18, 0.22
B = 10_000


def draw_quotients(
    stats: dict[str, tuple[np.ndarray, ...]], idx: np.ndarray
) -> tuple[float, float, float]:
    """A, B, C for the run draw ``idx`` from per-run (sse_class, n_class, sse_all, n_all)."""
    sse_l, cnt_l, sse_g, cnt_g = stats["s"]
    cl, cg = cnt_l[idx], cnt_g[idx]
    have = cl > 0
    if not have.any():
        return np.nan, np.nan, np.nan
    rm_l = np.sqrt(sse_l[idx][have] / cl[have])
    rm_g_all = np.sqrt(sse_g[idx] / cg)
    a = np.median(rm_l) / np.median(rm_g_all)
    b = np.sqrt(sse_l[idx].sum() / cl.sum()) / np.sqrt(sse_g[idx].sum() / cg.sum())
    c = np.median(rm_l) / np.median(rm_g_all[have])
    return float(a), float(b), float(c)


def interval(x: np.ndarray) -> dict[str, float]:
    x = x[~np.isnan(x)]
    return {
        "point_median": float(np.median(x)),
        "lo95": float(np.percentile(x, 2.5)),
        "hi95": float(np.percentile(x, 97.5)),
    }


def main() -> None:
    require_clean_tree()
    rng = np.random.default_rng(20260929)
    draws = rng.integers(0, N_RUNS, size=(B, N_RUNS))
    out: dict[str, object] = {"bootstrap": {"B": B, "seed": 20260929, "class": [LO, HI]}}

    # (1) composition of the class -- class membership depends on the truth only (same for all sets)
    p0 = load("excited", 1)
    cls0 = in_class(p0["truth"], LO, HI)
    counts = np.bincount(p0["run"][cls0], minlength=N_RUNS)
    windows = [int(np.unique(p0["window"][cls0 & (p0["run"] == r)]).size) for r in range(N_RUNS)]
    order = np.argsort(-counts)
    shares = counts / counts.sum()
    out["composition"] = {
        "n_points": int(counts.sum()),
        "n_runs_contributing": int((counts > 0).sum()),
        "n_runs_total": N_RUNS,
        "points_per_run_sorted": [int(counts[r]) for r in order if counts[r] > 0],
        "run_index_sorted": [int(r) for r in order if counts[r] > 0],
        "windows_per_run_sorted": [windows[r] for r in order if counts[r] > 0],
        "share_of_largest_run": float(shares.max()),
        "effective_number_of_runs_1_over_sum_share_squared": float(1.0 / np.sum(shares**2)),
        "points_per_run_all_32": [int(c) for c in counts],
        "n_points_all_per_run": int(np.bincount(p0["run"]).min()),
    }

    for arm in ARMS:
        res: dict[str, object] = {}
        per_seed_stats = {}
        for seed in SEEDS:
            p = load(arm, seed)
            cls = in_class(p["truth"], LO, HI)
            stats = {}
            for ref in ("measured", "truth"):
                err = p["pred"] - p[ref]
                s_l = run_stats(err[cls], p["run"][cls])
                s_g = run_stats(err, p["run"])
                stats[ref] = {"s": (*s_l, *s_g)}
            per_seed_stats[seed] = stats

        for ref in ("measured", "truth"):
            block: dict[str, object] = {}
            replicate = {
                s: np.array([draw_quotients(per_seed_stats[s][ref], d) for d in draws])
                for s in SEEDS
            }
            full = np.arange(N_RUNS)
            block["per_seed"] = {
                str(s): {
                    "point_A_B_C": list(draw_quotients(per_seed_stats[s][ref], full)),
                    "A": interval(replicate[s][:, 0]),
                    "B": interval(replicate[s][:, 1]),
                    "C": interval(replicate[s][:, 2]),
                }
                for s in SEEDS
            }
            # cross-seed statistic: median over seeds of each quotient, per replicate
            stack = np.stack([replicate[s] for s in SEEDS])  # seeds x B x 3
            med = np.nanmedian(stack, axis=0)
            block["cross_seed_median_of_quotients"] = {
                q: interval(med[:, i]) for i, q in enumerate("ABC")
            }
            block["cross_seed_median_point_A_B_C"] = [
                float(np.median([draw_quotients(per_seed_stats[s][ref], full)[i] for s in SEEDS]))
                for i in range(3)
            ]
            block["share_replicates_without_class_run"] = float(np.isnan(stack[0, :, 0]).mean())
            block["share_replicates_with_A_below_1"] = float(np.nanmean(med[:, 0] < 1.0))
            block["share_replicates_with_B_below_1"] = float(np.nanmean(med[:, 1] < 1.0))
            res[ref] = block
        out[arm] = res

    # (4) noise level of the reference in the class (does not depend on the network)
    noise = {}
    for lo, hi, name in ((LO, HI, "class_0.18_0.22"), (0.0, 10.0, "all_points")):
        m = in_class(p0["truth"], lo, hi)
        d = p0["measured"][m] - p0["truth"][m]
        noise[name] = {
            "rmse_pooled": rmse_pooled(d),
            "mean": float(d.mean()),
            "std": float(d.std()),
            "n": int(m.sum()),
        }
    out["reference_noise_measured_minus_truth"] = noise
    err_rows = {}
    for arm in ARMS:
        rows = []
        for seed in SEEDS:
            p = load(arm, seed)
            m = in_class(p["truth"], LO, HI)
            e_meas = rmse_pooled((p["pred"] - p["measured"])[m])
            e_true = rmse_pooled((p["pred"] - p["truth"])[m])
            rows.append(
                {
                    "seed": seed,
                    "local_pooled_vs_measured": e_meas,
                    "local_pooled_vs_truth": e_true,
                    "noise_pooled": noise["class_0.18_0.22"]["rmse_pooled"],
                    "local_vs_measured_over_noise": e_meas
                    / noise["class_0.18_0.22"]["rmse_pooled"],
                    "local_vs_truth_over_noise": e_true / noise["class_0.18_0.22"]["rmse_pooled"],
                }
            )
        err_rows[arm] = rows
    out["local_error_against_noise"] = err_rows
    write_artifact("verify_zw4_reliability", {"_provenance": provenance(), **out})
    c = out["composition"]
    print(c["n_runs_contributing"], c["points_per_run_sorted"], c["windows_per_run_sorted"])
    for arm in ARMS:
        for ref in ("measured", "truth"):
            b = out[arm][ref]
            print(arm, ref, b["cross_seed_median_point_A_B_C"], b["cross_seed_median_of_quotients"])


if __name__ == "__main__":
    main()

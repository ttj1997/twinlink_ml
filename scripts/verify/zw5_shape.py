"""ZW5 -- shape of the moisture error over the whole range (both arms, ZW6 uses the same output).

(a) error over the true moisture in narrow classes (0.02 kg/kg, 0 to 0.40, then [0.40,0.60) and
    [0.60,inf)) with the number of distinct runs and points per class;
(b) per class bias (signed, ``pred - truth``, negative = the sensor reports DRIER than true) and
    spread (std of ``pred - truth``) separately;
(c) the same, classed by the PREDICTION (what a controller sees; error is still ``pred - truth``);
(d) class bounds varied around the limit: [0.19,0.21), [0.18,0.22), [0.17,0.23), [0.15,0.25);
(e) step 1 against steps 1..36 pooled;
plus: local-peak test at 0.20 (bootstrap over runs, fine 0.01 bins), monotonicity of the curve, and
the slope of the prediction on the truth (slope < 1 = shrinkage towards the centre).

Quantities per class and seed; the aggregate is the median over the 5 seeds with min/max. All in
kg/kg. ``rmse_measured`` is the project's metric reference (noisy), ``rmse_truth`` the error of the
network itself.

    python scripts/verify/zw5_shape.py
"""

from __future__ import annotations

import itertools

import numpy as np
from _common import provenance, require_clean_tree, write_artifact
from _zw import ARMS, N_RUNS, SEEDS, in_class, load, rmse_pooled, run_stats
from scipy.stats import spearmanr

FINE = np.concatenate([np.arange(0.0, 0.4001, 0.02), [0.6, np.inf]])
VARIANTS = ((0.19, 0.21), (0.18, 0.22), (0.17, 0.23), (0.15, 0.25))
B = 5000


def class_stats(p: dict[str, np.ndarray], key: str, lo: float, hi: float) -> dict[str, float | int]:
    m = in_class(p[key], lo, hi)
    n = int(m.sum())
    if n == 0:
        return {"n": 0, "n_runs": 0}
    d = p["pred"][m] - p["truth"][m]
    run_rmse = [
        np.sqrt(np.mean((p["pred"][m & (p["run"] == r)] - p["measured"][m & (p["run"] == r)]) ** 2))
        for r in np.unique(p["run"][m])
    ]
    return {
        "n": n,
        "n_runs": int(np.unique(p["run"][m]).size),
        "rmse_truth": rmse_pooled(d),
        "bias": float(d.mean()),
        "spread": float(d.std()),
        "rmse_measured": rmse_pooled(p["pred"][m] - p["measured"][m]),
        "rmse_measured_run_median": float(np.median(run_rmse)),
    }


def aggregate(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    out: dict[str, float | int] = {"n": int(rows[0]["n"]), "n_runs": int(rows[0]["n_runs"])}
    if rows[0]["n"] == 0:
        return out
    for k in ("rmse_truth", "bias", "spread", "rmse_measured", "rmse_measured_run_median"):
        v = [r[k] for r in rows]
        out[k] = float(np.median(v))
        out[k + "_min"] = float(np.min(v))
        out[k + "_max"] = float(np.max(v))
    return out


def peak_statistic(
    sse: dict[str, np.ndarray], cnt: dict[str, np.ndarray], idx: np.ndarray
) -> float:
    def rm(name: str) -> float:
        with np.errstate(all="ignore"):
            return float(np.sqrt(sse[name][idx].sum() / cnt[name][idx].sum()))

    return rm("c") - 0.5 * (rm("lo") + rm("hi"))


def main() -> None:
    require_clean_tree()
    rng = np.random.default_rng(20260930)
    draws = rng.integers(0, N_RUNS, size=(B, N_RUNS))
    out: dict[str, object] = {"edges": [float(e) for e in FINE[:-1]] + [None], "bootstrap_B": B}
    for arm in ARMS:
        res: dict[str, object] = {}
        for step_label, step in (("h36_pooled", None), ("h1", 1)):
            data = {s: load(arm, s, step=step) for s in SEEDS}
            block: dict[str, object] = {}
            for key in ("truth", "pred"):
                curve = []
                for lo, hi in itertools.pairwise(FINE):
                    agg = aggregate([class_stats(data[s], key, lo, hi) for s in SEEDS])
                    agg["lo"], agg["hi"] = float(lo), (None if np.isinf(hi) else float(hi))
                    curve.append(agg)
                variants = {}
                for lo, hi in VARIANTS:
                    rows = [class_stats(data[s], key, lo, hi) for s in SEEDS]
                    agg = aggregate(rows)
                    glob = float(
                        np.median(
                            [rmse_pooled(data[s]["pred"] - data[s]["measured"]) for s in SEEDS]
                        )
                    )
                    agg["global_pooled_rmse_measured"] = glob
                    agg["quotient_pooled_over_pooled"] = agg["rmse_measured"] / glob
                    variants[f"{lo:.2f}-{hi:.2f}"] = agg
                block[f"by_{key}"] = {"curve": curve, "variants": variants}

            # shrinkage: slope of pred on truth (whole range 0.10-0.40 and near the limit)
            slopes = {}
            for lo, hi in ((0.10, 0.40), (0.15, 0.25)):
                v = []
                for s in SEEDS:
                    m = in_class(data[s]["truth"], lo, hi)
                    v.append(float(np.polyfit(data[s]["truth"][m], data[s]["pred"][m], 1)[0]))
                slopes[f"{lo:.2f}-{hi:.2f}"] = {"median": float(np.median(v)), "per_seed": v}
            block["slope_pred_on_truth"] = slopes

            # monotonicity of the truth-classed curve above the pool centre (0.06 .. 0.40)
            curve = block["by_truth"]["curve"]
            sel = [c for c in curve if c.get("n_runs", 0) >= 3 and 0.06 <= c["lo"] < 0.40]
            y = [c["rmse_truth"] for c in sel]
            x = [c["lo"] for c in sel]
            rho = spearmanr(x, y)
            block["monotonic_check_0.06_0.40"] = {
                "spearman_rho": float(rho.statistic),
                "n_classes": len(sel),
                "n_decreases_between_neighbours": int(np.sum(np.diff(y) < 0)),
            }

            # local peak at 0.20 (fine 0.01 bins, statistic on the pooled error against truth)
            if step is None:
                peaks = {}
                for key in ("truth", "pred"):
                    per_seed_draws = []
                    point = []
                    for s in SEEDS:
                        p = data[s]
                        d = p["pred"] - p["truth"]
                        sse, cnt = {}, {}
                        for name, (lo, hi) in (
                            ("lo", (0.17, 0.19)),
                            ("c", (0.19, 0.21)),
                            ("hi", (0.21, 0.23)),
                        ):
                            m = in_class(p[key], lo, hi)
                            sse[name], cnt[name] = run_stats(d[m], p["run"][m])
                        point.append(peak_statistic(sse, cnt, np.arange(N_RUNS)))
                        vals = []
                        for dr in draws:
                            try:
                                vals.append(peak_statistic(sse, cnt, dr))
                            except (ZeroDivisionError, FloatingPointError):
                                vals.append(np.nan)
                        per_seed_draws.append(vals)
                    with np.errstate(all="ignore"):
                        med = np.nanmedian(np.array(per_seed_draws, dtype=float), axis=0)
                    peaks[key] = {
                        "statistic": "rmse_truth[0.19,0.21) - mean(rmse_truth[0.17,0.19), rmse_truth[0.21,0.23))",
                        "point_per_seed": point,
                        "point_median": float(np.median(point)),
                        "boot_lo95": float(np.nanpercentile(med, 2.5)),
                        "boot_hi95": float(np.nanpercentile(med, 97.5)),
                        "share_boot_above_0": float(np.nanmean(med > 0)),
                        "share_boot_nan": float(np.isnan(med).mean()),
                    }
                block["peak_at_0.20"] = peaks
            res[step_label] = block
        out[arm] = res
    write_artifact("verify_zw5_shape", {"_provenance": provenance(), **out})
    for arm in ARMS:
        print(arm, "peak", out[arm]["h36_pooled"]["peak_at_0.20"])
        for c in out[arm]["h36_pooled"]["by_truth"]["curve"]:
            if c["n"]:
                print(
                    f"  {c['lo']:.2f} n={c['n']:6d} runs={c['n_runs']:2d} rmse_t={c['rmse_truth']:.5f} bias={c['bias']:+.5f} spread={c['spread']:.5f}"
                )


if __name__ == "__main__":
    main()

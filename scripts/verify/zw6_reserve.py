"""ZW6 -- reserve factor at the 90 % mark with the LOCAL error, both arms; margin coverage.

Befund 65 GA1 reads ``sigma_max = 0.003181 kg/kg`` (margin ``m = 3 sigma = 0.009544 kg/kg``) for
90 % of the control benefit; both values are read from ``probe_soft_sensor_margin_requirement.json``
(``interpolation['0.90']``). The reserve factor is ``sigma_max / sigma_local``. Local error in the
class [0.18, 0.22) of the true moisture, steps 1..36 pooled, holdout, five seeds per arm (median,
min-max), in four definitions:

* E1  run-median RMSE against the measured reference (GA2's estimator; 5 class runs)
* E2  pooled RMSE against the measured reference
* E3  pooled RMSE against the truth (error of the network alone)
* E4  bias and spread against the truth: margin needed to cover the dangerous direction,
      ``m_req = max(0, -bias) + 3 spread`` (bias = mean(pred - truth); negative = sensor drier than
      true) and the worst case ``|bias| + 3 spread``; coverage factor ``m90 / m_req``.

Controller view: for predictions in [0.19, 0.21) the quantiles of ``truth - pred`` (positive =
sensor reports drier than true), and the empirical violation rate when a controller holds the
prediction at ``0.20 - m`` (predictions in [0.20 - m - 0.002, 0.20 - m]; violation = truth > 0.20)
for the margins of Befund 64 (excited-trained 3 x global RMSE, constant-trained 3 x global RMSE,
the 90 %-mark margin).

    python scripts/verify/zw6_reserve.py
"""

from __future__ import annotations

import json

import numpy as np
from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact
from _zw import ARMS, SEEDS, in_class, load, rmse_pooled, rmse_run_median

LO, HI = 0.18, 0.22
LIMIT = 0.20


def spread_of(x: np.ndarray) -> tuple[float, float]:
    return float(x.mean()), float(x.std())


def stats_of(values: list[float]) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    require_clean_tree()
    ref = json.loads(
        (RUNS_DIR / "probe_soft_sensor_margin_requirement.json").read_text(encoding="utf-8")
    )
    sigma_max = ref["interpolation"]["0.90"]["sigma_max_kg_per_kg"]
    m90 = ref["interpolation"]["0.90"]["interpolated_margin_kg_per_kg"]
    margins_b64 = {
        row["label"]: row["margin_kg_per_kg"] for row in ref["merged_dataset_sorted_by_margin"]
    }
    margin_of_arm = {
        "excited": margins_b64["excited_lstm_3rmse"],
        "constant": margins_b64["constant_lstm_3rmse"],
    }
    out: dict[str, object] = {
        "sigma_max_90": sigma_max,
        "margin_90": m90,
        "margins_befund64": margin_of_arm,
        "class": [LO, HI],
    }
    for arm in ARMS:
        per_seed = []
        for seed in SEEDS:
            p = load(arm, seed)
            cls = in_class(p["truth"], LO, HI)
            e_meas = p["pred"] - p["measured"]
            d = p["pred"] - p["truth"]
            bias, spread = spread_of(d[cls])
            row = {
                "seed": seed,
                "global_run_median": rmse_run_median(e_meas, p["run"]),
                "E1_local_run_median": rmse_run_median(e_meas[cls], p["run"][cls]),
                "E2_local_pooled_measured": rmse_pooled(e_meas[cls]),
                "E3_local_pooled_truth": rmse_pooled(d[cls]),
                "bias_class": bias,
                "spread_class": spread,
                "m_req_dangerous_direction": max(0.0, -bias) + 3 * spread,
                "m_req_worst_case": abs(bias) + 3 * spread,
            }
            # controller view: classed by prediction
            byp = in_class(p["pred"], 0.19, 0.21)
            miss = (p["truth"] - p["pred"])[byp]
            row["by_prediction_0.19_0.21"] = {
                "n": int(byp.sum()),
                "n_runs": int(np.unique(p["run"][byp]).size),
                "mean_truth_minus_pred": float(miss.mean()),
                "quantiles_truth_minus_pred": {
                    str(q): float(np.percentile(miss, q)) for q in (50, 90, 95, 99, 99.9)
                },
            }
            viol = {}
            for name, m in (("margin_arm_3rmse_B64", margin_of_arm[arm]), ("margin_90", m90)):
                at = in_class(p["pred"], LIMIT - m - 0.002, LIMIT - m + 1e-12)
                viol[name] = {
                    "margin": m,
                    "n_points_at_margin": int(at.sum()),
                    "n_runs": int(np.unique(p["run"][at]).size) if at.any() else 0,
                    "violation_rate_truth_above_limit": float((p["truth"][at] > LIMIT).mean())
                    if at.any()
                    else None,
                }
                miss_all = (p["pred"] <= LIMIT - m) & (p["truth"] > LIMIT)
                viol[name]["missed_share_of_all_truth_above_limit"] = float(
                    miss_all.sum() / max(1, (p["truth"] > LIMIT).sum())
                )
            row["controller_view"] = viol
            for k, sig in (
                ("E1", row["E1_local_run_median"]),
                ("E2", row["E2_local_pooled_measured"]),
                ("E3", row["E3_local_pooled_truth"]),
                ("global_run_median", row["global_run_median"]),
            ):
                row[f"reserve_{k}"] = sigma_max / sig
            row["reserve_E4_dangerous_direction"] = m90 / row["m_req_dangerous_direction"]
            row["reserve_E4_worst_case"] = m90 / row["m_req_worst_case"]
            per_seed.append(row)
        keys = [
            k
            for k in per_seed[0]
            if k.startswith("reserve_")
            or k
            in (
                "global_run_median",
                "E1_local_run_median",
                "E2_local_pooled_measured",
                "E3_local_pooled_truth",
                "bias_class",
                "spread_class",
                "m_req_dangerous_direction",
                "m_req_worst_case",
            )
        ]
        out[arm] = {
            "per_seed": per_seed,
            "aggregate_5_seeds": {k: stats_of([r[k] for r in per_seed]) for k in keys},
            "aggregate_seeds_1_3": {
                k: stats_of([r[k] for r in per_seed if r["seed"] <= 3]) for k in keys
            },
            "reserve_from_ratio_of_medians_E1": sigma_max
            / float(np.median([r["E1_local_run_median"] for r in per_seed])),
            "share_seeds_with_reserve_below_1": {
                k: float(np.mean([r[k] < 1 for r in per_seed]))
                for k in keys
                if k.startswith("reserve_")
            },
        }
    write_artifact("verify_zw6_reserve", {"_provenance": provenance(), **out})
    for arm in ARMS:
        a = out[arm]["aggregate_5_seeds"]
        print(arm, {k: round(v["median"], 3) for k, v in a.items() if k.startswith("reserve_")})
        print(arm, "share below 1", out[arm]["share_seeds_with_reserve_below_1"])
        print(arm, "m_req", a["m_req_dangerous_direction"], a["m_req_worst_case"])


if __name__ == "__main__":
    main()

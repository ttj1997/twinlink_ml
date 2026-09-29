"""PR4 (records part) -- recompute the abstract's numbers from the committed records.

Independent code: plain json + numpy/scipy; no import of the project's resolution module, of
``scripts/analyze_*`` or of the evaluator. The Hartley d2 constants are *derived* here by
numerical integration (E[range of n standard normals]) and compared with the tabulated ones.

Items: K5 factors (Befund 51), persistence/static shifts (Befund 58 OA5a), energy factors and the
implied R^2 (Befund 59), resolution thresholds and verdicts (Befund 63), epoch limits of the
600-epoch probe, 0.00149 vs 0.00054 kg/kg (Befund 55/62), persistence never beaten, early-stopping
epochs, and the machine/platform behind the inference and training times (Befund 63a).

    python scripts/verify/pr4_records.py
"""

from __future__ import annotations

import collections
import math
from typing import Any

import numpy as np
from _common import TARGETS, provenance, read_jsonl, require_clean_tree, write_artifact
from scipy import integrate, stats

SEEDS = (1, 2, 3, 4, 5)
BUDGETS = (8, 16, 32, 64)
F_PROJ = "arch_learning_curve_projected"
F_EXT = "arch_learning_curve_projected_extended"
F_CONST = "arch_learning_curve_constant_trained"
F_600 = "arch_epoch_budget_probe600"
ENERGY, MOIST, TBED = TARGETS


def load() -> dict[str, list[dict[str, Any]]]:
    return {f: read_jsonl(f) for f in (F_PROJ, F_EXT, F_CONST, F_600)}


def vals(
    recs: dict[str, list[dict[str, Any]]], file: str, model: str, test_set: str, n: int, key: str
) -> list[float]:
    """Metric values over seeds (sorted by master seed)."""
    rows = [
        r
        for r in recs[file]
        if r["tags"]["model"] == model
        and r["tags"]["test_set"] == test_set
        and r["params"]["n_train"] == n
    ]
    rows.sort(key=lambda r: r["params"]["master_seed"])
    return [r["metrics"][key] for r in rows]


def phys(channel: str, h: int) -> str:
    return f"rmse_run_median_physical/{channel}/h{h}"


def nrm(channel: str, h: int) -> str:
    return f"nrmse_run_median/{channel}/h{h}"


def med(x: list[float]) -> float:
    return float(np.median(x))


# ------------------------------------------------------------------- Hartley constants
def d2(n: int) -> float:
    """E[range] of n iid standard normals, by numerical integration."""
    integrand = lambda x: 1.0 - stats.norm.cdf(x) ** n - (1.0 - stats.norm.cdf(x)) ** n  # noqa: E731
    return float(integrate.quad(integrand, -12, 12, epsabs=1e-13)[0])


def threshold_two_sample(
    cand: list[float], ref: list[float], z: float = 1.96
) -> tuple[float, float]:
    """(relative threshold, relative improvement of cand vs ref), n = number of seeds."""
    n = len(ref)
    sig = lambda v: (max(v) - min(v)) / D2[len(v)]  # noqa: E731
    ref_med = med(ref)
    threshold = z * math.sqrt(sig(cand) ** 2 + sig(ref) ** 2) / math.sqrt(n) / abs(ref_med)
    improvement = (ref_med - med(cand)) / abs(ref_med)
    return threshold, improvement


D2 = {n: d2(n) for n in range(2, 11)}


def main() -> None:
    require_clean_tree()
    recs = load()
    out: dict[str, Any] = {"_provenance": provenance()}
    out["d2_derived_vs_table"] = {
        "derived_d2": {n: round(v, 4) for n, v in D2.items()},
        "table_used_by_project_(5)": 2.326,
        "derived_d2_5": D2[5],
    }

    # ---------------- A. K5 (Befund 51): factor persistence / model on moisture ------------
    k5: dict[str, Any] = {}
    for metric_name, key_fn in (("nrmse_run_median", nrm), ("rmse_run_median_physical", phys)):
        table = {}
        for ts in ("excited", "constant_holdout"):
            per_n = {}
            for n in BUDGETS:
                pers = med(vals(recs, F_PROJ, "persistence", ts, n, key_fn(MOIST, 36)))
                per_n[n] = {
                    "persistence_median": pers,
                    "gru_median": med(vals(recs, F_PROJ, "gru", ts, n, key_fn(MOIST, 36))),
                    "lstm_median": med(vals(recs, F_PROJ, "lstm", ts, n, key_fn(MOIST, 36))),
                }
                per_n[n]["factor_gru"] = pers / per_n[n]["gru_median"]
                per_n[n]["factor_lstm"] = pers / per_n[n]["lstm_median"]
            table[ts] = per_n
        allf = [
            v
            for n in BUDGETS
            for k in ("factor_gru", "factor_lstm")
            for v in [table["excited"][n][k]]
        ]
        k5[metric_name] = {
            "cells": table,
            "excited_factor_range_over_budgets_and_gru_lstm": [min(allf), max(allf)],
            "constant_holdout_n64": {
                "gru": table["constant_holdout"][64]["factor_gru"],
                "lstm": table["constant_holdout"][64]["factor_lstm"],
            },
        }
    out["A_K5_factors_moisture_h36"] = k5

    # ---------------- B. persistence vs static shift between test sets (OA5a) ---------------
    shift: dict[str, Any] = {}
    for h in (1, 36):
        for model in ("persistence", "static"):
            ex = med(vals(recs, F_PROJ, model, "excited", 64, phys(ENERGY, h)))
            co = med(vals(recs, F_PROJ, model, "constant_holdout", 64, phys(ENERGY, h)))
            shift[f"{model}_h{h}_n64"] = {"excited": ex, "constant_holdout": co, "ratio": ex / co}
    out["B_energy_test_set_shift_(physical_kW)"] = shift

    # ---------------- C. energy factors vs static, implied R2 (Befund 59) --------------------
    energy: dict[str, Any] = {}
    factors = []
    for ts in ("excited", "constant_holdout"):
        for model, file in (("gru", F_PROJ), ("lstm", F_PROJ), ("transformer", F_EXT)):
            static = med(vals(recs, F_PROJ, "static", ts, 64, phys(ENERGY, 36)))
            m = med(vals(recs, file, model, ts, 64, phys(ENERGY, 36)))
            k = static / m
            factors.append(k)
            energy[f"{ts}/{model}"] = {
                "static_kw": static,
                "model_kw": m,
                "factor": k,
                "implied_r2_from_0.972_(1-(1-r2)/k^2)": 1.0 - (1.0 - 0.972) / k**2,
            }
    out["C_energy_factor_vs_static_h36_n64"] = {
        "cells": energy,
        "factor_range": [min(factors), max(factors)],
        "implied_r2_min_max": [
            min(v["implied_r2_from_0.972_(1-(1-r2)/k^2)"] for v in energy.values()),
            max(v["implied_r2_from_0.972_(1-(1-r2)/k^2)"] for v in energy.values()),
        ],
        "note": "implied R2 is arithmetic on an in-sample R2 of a different data set; the measured R2 "
        "on the Befund 51 windows is in verify_pr3_metrics.json",
    }

    # ---------------- D. resolution (Befund 63) ---------------------------------------------------
    def series(file: str, model: str, ts: str, ch: str) -> list[float]:
        return vals(recs, file, model, ts, 64, phys(ch, 36))

    verdicts: list[dict[str, Any]] = []

    def add(label: str, cand: list[float], ref: list[float], ch: str, ts: str) -> None:
        thr, imp = threshold_two_sample(cand, ref)
        verdicts.append(
            {
                "comparison": label,
                "channel": ch,
                "test_set": ts,
                "improvement": imp,
                "threshold": thr,
                "resolvable": abs(imp) > thr,
            }
        )

    for ts in ("excited", "constant_holdout"):
        for ch in TARGETS:
            ref = series(F_PROJ, "lstm", ts, ch)
            add("gru_vs_lstm", series(F_PROJ, "gru", ts, ch), ref, ch, ts)
            add("transformer200_vs_lstm200", series(F_EXT, "transformer", ts, ch), ref, ch, ts)
            add("xlstm_vs_lstm", series(F_EXT, "xlstm", ts, ch), ref, ch, ts)
            add("mamba_vs_lstm", series(F_EXT, "mamba", ts, ch), ref, ch, ts)
            add(
                "transformer600_vs_lstm600",
                series(F_600, "transformer", ts, ch),
                series(F_600, "lstm", ts, ch),
                ch,
                ts,
            )

    def group(name: str) -> list[dict[str, Any]]:
        return [v for v in verdicts if v["comparison"] == name]

    gru = group("gru_vs_lstm")
    xl_ma = group("xlstm_vs_lstm") + group("mamba_vs_lstm")
    slow = [
        v
        for v in verdicts
        if v["comparison"]
        in ("gru_vs_lstm", "transformer200_vs_lstm200", "transformer600_vs_lstm600")
    ]
    energy_bed = [v["threshold"] for v in slow if v["channel"] in (ENERGY, TBED)]
    moist = [v["threshold"] for v in slow if v["channel"] == MOIST]
    lstm_moist = {ts: series(F_PROJ, "lstm", ts, MOIST) for ts in ("excited", "constant_holdout")}

    def required_n(vs: list[float], rel: float = 0.125, z: float = 1.96) -> int:
        sigma = (max(vs) - min(vs)) / D2[len(vs)]
        return math.ceil(2.0 * (z * sigma / (rel * med(vs))) ** 2)

    out["D_resolution"] = {
        "verdicts": verdicts,
        "gru_vs_lstm_resolvable_cells": f"{sum(v['resolvable'] for v in gru)} of {len(gru)}",
        "xlstm_mamba_vs_lstm_resolvable_cells": f"{sum(v['resolvable'] for v in xl_ma)} of {len(xl_ma)}",
        "unresolvable_xlstm_mamba_cells": [
            (v["comparison"], v["channel"], v["test_set"]) for v in xl_ma if not v["resolvable"]
        ],
        "threshold_range_gru_transformer_energy_and_bed_temperature": [
            min(energy_bed),
            max(energy_bed),
        ],
        "threshold_range_gru_transformer_moisture": [min(moist), max(moist)],
        "threshold_range_xlstm_mamba_all": [
            min(v["threshold"] for v in xl_ma),
            max(v["threshold"] for v in xl_ma),
        ],
        "required_seeds_for_12.5pct_lstm_moisture_n64": {
            ts: required_n(v) for ts, v in lstm_moist.items()
        },
        "required_seeds_note": "n >= 2 (z*sigma/(0.125*median))^2, sigma = range/d2, z=1.96",
    }
    t200 = [v for v in verdicts if v["comparison"] == "transformer200_vs_lstm200"]
    t600 = [v for v in verdicts if v["comparison"] == "transformer600_vs_lstm600"]
    pick = lambda vs, ch, ts: next(v for v in vs if v["channel"] == ch and v["test_set"] == ts)  # noqa: E731
    out["D_transformer_vs_lstm"] = {
        "energy_constant_200ep": pick(t200, ENERGY, "constant_holdout"),
        "energy_constant_600ep": pick(t600, ENERGY, "constant_holdout"),
        "bed_temperature_constant_200ep": pick(t200, TBED, "constant_holdout"),
        "bed_temperature_constant_600ep": pick(t600, TBED, "constant_holdout"),
        "bed_temperature_constant_ratio_600ep_(transformer/lstm)": med(
            series(F_600, "transformer", "constant_holdout", TBED)
        )
        / med(series(F_600, "lstm", "constant_holdout", TBED)),
        "energy_constant_ratio_600ep_(transformer/lstm)": med(
            series(F_600, "transformer", "constant_holdout", ENERGY)
        )
        / med(series(F_600, "lstm", "constant_holdout", ENERGY)),
        "bed_temperature_constant_ratio_200ep": med(
            series(F_EXT, "transformer", "constant_holdout", TBED)
        )
        / med(series(F_PROJ, "lstm", "constant_holdout", TBED)),
    }

    # ---------------- E. epoch limit in the 600-epoch probe ------------------------------------
    limit_rows = []
    for r in recs[F_600]:
        limits = r["params"]["model/net_max_epochs"]
        for ch in TARGETS:
            e = r["params"][f"training/epochs_run/{ch}"]
            limit_rows.append(
                {
                    "model": r["tags"]["model"],
                    "seed": r["params"]["master_seed"],
                    "test_set": r["tags"]["test_set"],
                    "head": ch,
                    "epochs_run": e,
                    "max_epochs": limits,
                    "at_limit": e >= limits,
                }
            )
    seen = {
        (x["model"], x["seed"], x["head"]): x for x in limit_rows
    }  # one training per (model, seed)
    out["E_probe600_epoch_limit"] = {
        "trainings": len(seen),
        "heads_at_limit_by_model": {
            m: sum(x["at_limit"] for k, x in seen.items() if k[0] == m)
            for m in ("transformer", "lstm")
        },
        "max_epochs_run_by_model": {
            m: max(x["epochs_run"] for k, x in seen.items() if k[0] == m)
            for m in ("transformer", "lstm")
        },
        "transformer_epochs_run": {
            f"seed{k[1]}/{k[2]}": x["epochs_run"]
            for k, x in sorted(seen.items())
            if k[0] == "transformer"
        },
        "max_epochs": 600,
    }

    # ---------------- F. 0.00149 vs 0.00054 ---------------------------------------------------------
    exc = vals(recs, F_PROJ, "lstm", "constant_holdout", 64, phys(MOIST, 36))
    con = vals(recs, F_CONST, "lstm", "constant_holdout", 64, phys(MOIST, 36))
    out["F_moisture_constant_holdout_lstm_n64_h36"] = {
        "excited_trained_per_seed": exc,
        "constant_trained_per_seed": con,
        "excited_trained_median": med(exc),
        "constant_trained_median": med(con),
        "ratio": med(con) / med(exc),
        "same_test_set_estimator_horizon": "constant_holdout / rmse_run_median_physical / h36 / product_moisture_kg_per_kg / lstm n_train=64 / median over 5 seeds",
    }

    # ---------------- G. persistence never beaten (constant-trained arm) ----------------------------
    beaten: list[dict[str, Any]] = []
    for n in BUDGETS:
        for model in ("gru", "lstm"):
            for metric_name, key_fn in (
                ("nrmse_run_median", nrm),
                ("rmse_run_median_physical", phys),
            ):
                p = med(
                    vals(recs, F_CONST, "persistence", "constant_holdout", n, key_fn(MOIST, 36))
                )
                m = med(vals(recs, F_CONST, model, "constant_holdout", n, key_fn(MOIST, 36)))
                beaten.append(
                    {
                        "n_train": n,
                        "model": model,
                        "metric": metric_name,
                        "persistence": p,
                        "model_value": m,
                        "factor_persistence_over_model": p / m,
                        "persistence_beaten": m < p,
                    }
                )
    out["G_constant_trained_vs_persistence_moisture_h36"] = {
        "cells": beaten,
        "n_cells_where_model_beats_persistence": sum(b["persistence_beaten"] for b in beaten),
        "n_cells": len(beaten),
    }

    # ---------------- H. early stopping epochs ---------------------------------------------------------
    es: dict[str, Any] = {}
    for label, file in (("constant_trained", F_CONST), ("excited_trained", F_PROJ)):
        for model in ("gru", "lstm"):
            row = {}
            for n in BUDGETS:
                per_seed = []
                for seed in SEEDS:
                    r = next(
                        x
                        for x in recs[file]
                        if x["tags"]["model"] == model
                        and x["params"]["n_train"] == n
                        and x["params"]["master_seed"] == seed
                        and x["tags"]["test_set"] == "constant_holdout"
                    )
                    per_seed.append(
                        float(np.mean([r["params"][f"training/epochs_run/{ch}"] for ch in TARGETS]))
                    )
                row[n] = {
                    "median_over_seeds_of_head_mean_epochs_run": med(per_seed),
                    "implied_best_epoch_(epochs_run-patience20_if_stopped_early)": med(per_seed)
                    - 20.0,
                }
            es[f"{label}/{model}"] = row
    out["H_epochs_run"] = es

    # ---------------- I. machines behind inference / training times ------------------------------------
    times: dict[str, Any] = {}
    for file in (F_PROJ, F_EXT, F_CONST, F_600):
        for model in sorted({r["tags"]["model"] for r in recs[file]}):
            rows = [
                r
                for r in recs[file]
                if r["tags"]["model"] == model
                and r["params"]["n_train"] == 64
                and r["tags"]["test_set"] in ("excited", "own_pool")
            ]
            if not rows:
                continue
            times[f"{file}/{model}"] = {
                "platform": sorted({r["params"]["platform"][:12] for r in rows}),
                "python": sorted({r["params"]["python"] for r in rows}),
                "torch_threads": sorted({r["params"].get("torch/num_threads") for r in rows}),
                "inference_us_per_window_median": med(
                    [r["metrics"]["inference_us_per_window"] for r in rows]
                ),
                "inference_us_per_window_min_max": [
                    min(r["metrics"]["inference_us_per_window"] for r in rows),
                    max(r["metrics"]["inference_us_per_window"] for r in rows),
                ],
                "training_elapsed_s_median": med([r["params"]["training/elapsed_s"] for r in rows]),
            }
    out["I_times_as_recorded_(n_train=64)"] = times
    lstm_lx = times[f"{F_PROJ}/lstm"]
    tr = times[f"{F_EXT}/transformer"]
    out["I_ratios_as_quoted"] = {
        "transformer_over_lstm_training_time_(mixed_machines)": tr["training_elapsed_s_median"]
        / lstm_lx["training_elapsed_s_median"],
        "transformer_over_lstm_inference_(mixed_machines)": tr["inference_us_per_window_median"]
        / lstm_lx["inference_us_per_window_median"],
        "gru_over_lstm_inference_linux": times[f"{F_PROJ}/gru"]["inference_us_per_window_median"]
        / lstm_lx["inference_us_per_window_median"],
        "gru_over_lstm_inference_windows_constant_trained": times[f"{F_CONST}/gru"][
            "inference_us_per_window_median"
        ]
        / times[f"{F_CONST}/lstm"]["inference_us_per_window_median"],
        "transformer_over_lstm_inference_windows_probe600": times[f"{F_600}/transformer"][
            "inference_us_per_window_median"
        ]
        / times[f"{F_600}/lstm"]["inference_us_per_window_median"],
        "transformer_over_lstm_training_windows_probe600_(600ep_max)": times[
            f"{F_600}/transformer"
        ]["training_elapsed_s_median"]
        / times[f"{F_600}/lstm"]["training_elapsed_s_median"],
    }
    write_artifact("verify_pr4_records", out)
    print(
        {
            "K5": k5["nrmse_run_median"]["constant_holdout_n64"],
            "K5 excited range": k5["nrmse_run_median"][
                "excited_factor_range_over_budgets_and_gru_lstm"
            ],
            "shift": {k: round(v["ratio"], 2) for k, v in shift.items()},
            "energy factor range": out["C_energy_factor_vs_static_h36_n64"]["factor_range"],
            "res": {
                k: out["D_resolution"][k]
                for k in (
                    "gru_vs_lstm_resolvable_cells",
                    "xlstm_mamba_vs_lstm_resolvable_cells",
                    "threshold_range_gru_transformer_energy_and_bed_temperature",
                    "threshold_range_gru_transformer_moisture",
                    "required_seeds_for_12.5pct_lstm_moisture_n64",
                )
            },
            "moisture": out["F_moisture_constant_holdout_lstm_n64_h36"]["ratio"],
            "probe600": out["E_probe600_epoch_limit"]["heads_at_limit_by_model"],
        }
    )
    _ = collections


if __name__ == "__main__":
    main()

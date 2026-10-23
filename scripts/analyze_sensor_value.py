r"""Analysis of the sensor-value study (docs/28_sensor_value.md), reading fixed before the run.

Input: ``artifacts/runs/<name>.json.gz`` of ``scripts/run_sensor_value.py``. Output:
``artifacts/runs/<name>_analysis.json`` (never overwritten) and a text summary on stdout.

Quantities
----------
Pairs p = (draw, scenario), 20 x 20. Variant v in {F0, F1, F2, F3:1, F3:2, F3:3}; criterion
c in {(ii) at least 95 % of the pairs hold (MAIN), (i) all pairs hold}; margin m_v^c from the run
(evaluated on all pairs). Per pair, at the margin of criterion c:

* relative energy  r_v(p) = E_v(p) / E_F1(p) - 1  (F1 at ITS margin of the same criterion), and the
  same for CO2 (gas-fired heater, 0.366 / 0.201 kg/kWh); median, minimum and maximum over pairs;
* value of the measurement  V(p) = (E_F0(p) - E_F2(p)) / E_F1(p);
* soft sensor against the inline sensor  W_s(p) = (E_F3:s(p) - E_F2(p)) / E_F1(p), per training
  seed s; median over pairs per seed, and the range of these medians over the seeds;
* required margin m_v^c; maximum violation over the pairs (and the number of pairs that violate);
  solver failures.

Pre-registered reading (main: criterion (ii), median over the pairs of V)
------------------------------------------------------------------------
* **V0**  median V < 1 %: the inline measurement has no notable energy value for this line;
  partial observability does not justify a learning control.
* **V1**  1 % <= median V <= 5 %: measurable value; the question to the application partner (is
  the moisture measured inline, how well) has priority.
* **V2**  median V > 5 %: clear value; check a sensor investment and a control with state
  estimation.
* If F2 meets no margin of criterion (ii) up to the cap, there is no reading (decision to the
  project lead). If F0 meets none while F2 does, the reading is **V2 (not quantified)**: without
  feedback the limit cannot be held in 95 % of the pairs by any margin up to 0.12 kg/kg.

Reported separately, not part of the reading: the same with criterion (i); V normalised by E_F0
and by E_F2 instead of E_F1 (sensitivity of the class to the normalisation).

Position of the soft sensor (reported separately, per seed, criterion (ii))
-------------------------------------------------------------------------
Captured share  s = median_p[(E_F0 - E_F3:s) / E_F1] / median_p[V]. "Between F0 and F2" if
0 < s < 1; "at F0 or worse" if s <= 0; "at F2 or better" if s >= 1. Closeness to F2: 1 - s and the
median of W_s. Also the margins m_F3:s against m_F0 and m_F2. Range over the three seeds.

Drivers of the F0 margin (descriptive, as Befund 108 maps parameters to questions)
---------------------------------------------------------------------------------
Per draw the F0 margin it alone requires, max over its scenarios of the per-pair smallest margin.
Spearman rank correlation of that margin with each of the seven parameters; share of its variance
explained (R^2 and adjusted R^2 of an ordinary least-squares fit on the unit-cube coordinates)
by the groups of Befund 108's questions: drying kinetics (critical moisture, nu_max) and isotherm
(five GAB parameters), and by all seven. With 20 draws this is descriptive, not a Sobol index.
Also the steady model error per draw (mean over its scenarios of the F1 bias at the end of the
pre-roll).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import RUNS_DIR  # noqa: E402

READING_CRITERION = "ii"
V0_UPPER = 0.01
V1_UPPER = 0.05
DRYING_KINETICS = ("critical_moisture_kg_per_kg", "max_saturation_degree")
ISOTHERM = (
    "gab_monolayer_moisture_kg_per_kg",
    "gab_c_constant_at_reference_temperature",
    "gab_k_constant_at_reference_temperature",
    "gab_c_enthalpy_j_per_mol",
    "gab_k_enthalpy_j_per_mol",
)


def classify(median_value: float | None, f0_ok: bool, f2_ok: bool) -> str:
    """Pre-registered class of the value of the measurement."""
    if not f2_ok:
        return "none (F2 has no margin of criterion (ii); decision to the project lead)"
    if not f0_ok:
        return "V2 (not quantified)"
    assert median_value is not None
    if median_value < V0_UPPER:
        return "V0"
    if median_value <= V1_UPPER:
        return "V1"
    return "V2"


def soft_sensor_position(captured: float) -> str:
    """Position of F3 between F0 (s = 0) and F2 (s = 1)."""
    if captured <= 0.0:
        return "at F0 or worse"
    if captured >= 1.0:
        return "at F2 or better"
    return "between F0 and F2"


def _stats(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def _by_pair(records: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    return {(int(r["draw"]), r["scenario"]): r for r in records}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties)."""
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)


def r_squared(features: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """R^2 and adjusted R^2 of OLS with intercept."""
    n, k = features.shape
    design = np.column_stack([np.ones(n), features])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coef
    total = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / total if total > 0 else math.nan
    adjusted = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1) if n - k - 1 > 0 else math.nan
    return {"r2": r2, "r2_adjusted": adjusted}


def analyse(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload["results"]
    variants = list(payload["variants"])
    tol = float(payload["violation_tolerance_kg_per_kg"])
    out: dict[str, Any] = {"variants": variants, "criteria": {}}
    for criterion in ("ii", "i"):
        block: dict[str, Any] = {"per_variant": {}}
        records = {
            v: _by_pair(results[v]["criteria"][criterion].get("records", [])) for v in variants
        }
        ok = {v: bool(results[v]["criteria"][criterion]["converged"]) for v in variants}
        pairs = sorted(records["F1"]) if ok.get("F1") else []
        for v in variants:
            entry = results[v]["criteria"][criterion]
            info: dict[str, Any] = {
                "margin": entry.get("margin"),
                "converged": ok[v],
                "k_required": entry["k_required"],
                "holding": entry.get("holding"),
                "tried": entry.get("tried"),
            }
            if ok[v]:
                recs = list(records[v].values())
                violations = np.array([r["max_violation"] for r in recs])
                info["max_violation"] = float(violations.max())
                info["violating_pairs"] = int(np.sum(violations > tol))
                info["failures"] = int(sum(r["failures"] for r in recs))
                info["energy_kwh"] = _stats(np.array([r["energy_kwh"] for r in recs]))
                info["bias_mean"] = _stats(np.array([r["bias_mean"] for r in recs]))
                info["feedback_error_rmse"] = _stats(
                    np.array([r["feedback_error_rmse"] for r in recs])
                )
                shares = [
                    r["extra"]["outlier_share"] for r in recs if "outlier_share" in r["extra"]
                ]
                if shares:
                    info["outlier_share"] = _stats(np.array(shares))
                if pairs:
                    for quantity in ("energy_kwh", "co2_kg"):
                        rel = np.array(
                            [
                                records[v][p][quantity] / records["F1"][p][quantity] - 1.0
                                for p in pairs
                            ]
                        )
                        info[f"relative_{quantity}_vs_F1"] = _stats(rel)
            block["per_variant"][v] = info
        if pairs and ok.get("F0") and ok.get("F2"):

            def value(
                a: str,
                b: str,
                norm: str,
                quantity: str = "energy_kwh",
                records: dict[str, Any] = records,
                pairs: list[tuple[int, str]] = pairs,
            ) -> np.ndarray:
                return np.array(
                    [
                        (records[a][p][quantity] - records[b][p][quantity])
                        / records[norm][p][quantity]
                        for p in pairs
                    ]
                )

            v_main = value("F0", "F2", "F1")
            block["value_F0_minus_F2"] = {
                "energy_vs_F1": _stats(v_main),
                "co2_vs_F1": _stats(value("F0", "F2", "F1", "co2_kg")),
                "energy_vs_F0": _stats(value("F0", "F2", "F0")),
                "energy_vs_F2": _stats(value("F0", "F2", "F2")),
            }
            soft = {}
            for v in variants:
                if not v.startswith("F3:"):
                    continue
                if not ok[v]:
                    soft[v] = {"converged": False}
                    continue
                w = value(v, "F2", "F1")
                gain = value("F0", v, "F1")
                denominator = float(np.median(v_main))
                captured = float(np.median(gain)) / denominator if denominator != 0 else math.nan
                soft[v] = {
                    "converged": True,
                    "W_F3_minus_F2_vs_F1": _stats(w),
                    "F0_minus_F3_vs_F1": _stats(gain),
                    "captured_share": captured,
                    "position": soft_sensor_position(captured) if math.isfinite(captured) else None,
                    "margin": results[v]["criteria"][criterion]["margin"],
                }
            medians = [s["W_F3_minus_F2_vs_F1"]["median"] for s in soft.values() if s["converged"]]
            captured = [s["captured_share"] for s in soft.values() if s["converged"]]
            block["soft_sensor"] = {
                "per_seed": soft,
                "W_median_over_seeds": _stats(np.array(medians)) if medians else None,
                "captured_share_over_seeds": _stats(np.array(captured)) if captured else None,
            }
        median_v = (
            block["value_F0_minus_F2"]["energy_vs_F1"]["median"]
            if "value_F0_minus_F2" in block
            else None
        )
        block["reading"] = classify(median_v, ok.get("F0", False), ok.get("F2", False))
        out["criteria"][criterion] = block
    out["reading"] = out["criteria"][READING_CRITERION]["reading"]
    out["drivers_F0"] = drivers(payload)
    out["runtime_s"] = payload.get("runtime_s")
    out["stage_runtime_s"] = payload.get("stage_runtime_s")
    out["seconds_per_pair_and_job"] = payload.get("seconds_per_pair_and_job")
    return out


def drivers(payload: dict[str, Any]) -> dict[str, Any]:
    """Which draws drive the F0 margin (see the module docstring)."""
    draws = payload["draws"]
    per_pair = np.array(payload["results"]["F0"]["per_pair_margin"], dtype=np.float64)
    per_draw = per_pair.max(axis=1)
    names = list(draws[0]["unit"])
    unit = np.array([[d["unit"][n] for n in names] for d in draws])
    finite = np.isfinite(per_draw)
    out: dict[str, Any] = {
        "per_draw_margin": per_draw.tolist(),
        "draws_without_margin_up_to_cap": int(np.sum(~finite)),
    }
    target = per_draw[finite]
    x = unit[finite]
    out["spearman"] = {n: spearman(x[:, j], target) for j, n in enumerate(names)}
    groups = {"drying_kinetics": DRYING_KINETICS, "isotherm": ISOTHERM, "all": tuple(names)}
    out["variance_explained"] = {
        g: r_squared(x[:, [names.index(n) for n in members]], target)
        for g, members in groups.items()
    }
    bias = None
    f1 = payload["results"].get("F1", {}).get("criteria", {}).get("ii", {}).get("records")
    if f1:
        bias = [
            float(np.mean([r["bias_end_pre_roll"] for r in f1 if int(r["draw"]) == d["index"]]))
            for d in draws
        ]
    order = np.argsort(-np.where(np.isfinite(per_draw), per_draw, np.inf))
    out["ranking"] = [
        {
            "draw": int(draws[i]["index"]),
            "margin": float(per_draw[i]),
            "values": draws[i]["values"],
            "steady_model_error_F1_bias": None if bias is None else bias[i],
        }
        for i in order
    ]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    args = parser.parse_args()
    runs = Path(args.runs_dir)
    source = runs / f"{args.name}.json.gz"
    output = runs / f"{args.name}_analysis.json"
    if output.exists():
        raise SystemExit(f"{output} exists -- artifacts are never overwritten")
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    analysis = analyse(payload)
    analysis["source"] = {"name": args.name, "provenance": payload["provenance"]}
    output.write_text(json.dumps(analysis, indent=1), encoding="utf-8")
    main_block = analysis["criteria"][READING_CRITERION]
    print(f"reading (criterion ii): {analysis['reading']}")
    for v, info in main_block["per_variant"].items():
        rel = info.get("relative_energy_kwh_vs_F1", {})
        print(
            f"{v}: margin {info['margin']}, holding {info['holding']}/{info['k_required']}, "
            f"max violation {info.get('max_violation')}, dE vs F1 {rel}"
        )
    print("value F0-F2:", main_block.get("value_F0_minus_F2"))
    print("soft sensor:", json.dumps(main_block.get("soft_sensor"), indent=1))
    print("drivers:", json.dumps(analysis["drivers_F0"]["spearman"], indent=1))
    print("variance explained:", analysis["drivers_F0"]["variance_explained"])
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

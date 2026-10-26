r"""Analysis of the follow-up sensor-value study (docs/29_sensor_value_2.md), reading fixed before the run.

Input: ``artifacts/runs/<name>.json.gz`` of ``scripts/run_sensor_value_2.py`` and the predecessor's
artifact it names (SHA-256 checked); the soft-sensor training record named there. Output:
``artifacts/runs/<name>_analysis.json`` (never overwritten) and a summary on stdout.

Quantities
----------
Pairs p = (draw, scenario), 20 x 20, as in the predecessor. Variants X of this study: ``F4:<h>h``
(periodic laboratory sample every h hours) and ``F5:<v>:<s>`` (soft sensor with measured responses,
training variant v in {nom, DR, DRoY}, training seed s). Reference energies E_F0, E_F1, E_F2: the
predecessor's records at the margins of the same criterion (not recomputed). Criterion c in {(ii) at
least 95 % of the pairs hold (MAIN), (i) all pairs hold}; margin m_X^c from the run (verified on all
pairs). Per pair at the margin of criterion c:

* **share of the value captured**  A_X(p) = (E_F0(p) - E_X(p)) / (E_F0(p) - E_F2(p));
  A = 0: as without feedback, A = 1: as the inline sensor. Main statistic: the **median over all
  400 pairs** (literal request), with minimum and maximum;
* relative energy r_X(p) = E_X(p) / E_F1(p) - 1 and the same for CO2 (median and range); margin
  m_X^c, pairs holding, maximum violation, solver failures; feedback error against the true moisture
  (F4: lab error at the samples; F5: estimate error at every 25 s block).

The denominator E_F0 - E_F2 is small or negative for some pairs (predecessor: per-pair value of the
measurement from -2.1 % of E_F1), where A_X(p) is unstable or flips sign. Reported with the main
statistic, not part of the reading: the number of pairs with E_F0 - E_F2 <= 0 and with
E_F0 - E_F2 < 1 % of E_F1; **(S1)** the median of A_X over the pairs with E_F0 - E_F2 >= 1 % of
E_F1; **(S2)** the ratio of medians median_p[(E_F0 - E_X)/E_F1] / median_p[(E_F0 - E_F2)/E_F1] (the
captured share s of the predecessor). If S1 or S2 lies on the other side of 0.8 than the main
statistic, the reading is marked "sensitive to the handling of small denominators".

A variant without a margin of criterion c up to the cap has no E_X: its A counts as below every
threshold ("not quantified").

Pre-registered reading (criterion (ii), main statistic; threshold 0.8 inclusive)
-------------------------------------------------------------------------------
**La** (periodic laboratory sample):
  * if A(F4:4h) >= 0.8: **"periodic laboratory samples suffice; an inline sensor has little added
    value for this line"** (in this simulation, for the placeholder lab error 0.003 kg/kg, delay
    30 min, gain 0.5); whether 8 h also reaches 0.8 is reported;
  * otherwise: the **longest tested interval with A >= 0.8** ("from an interval of h hours or
    shorter"), or **"none"** if no tested interval reaches 0.8. Not assumed monotone in the
    interval: every interval is reported. An interval dropped by the budget rule counts as not
    tested.

**Lb** (soft sensor with measured responses):
  * per training seed s: A_s = median over the pairs of A_{F5:DR:s}; over the seeds the median and
    the range of A_s;
  * if the **median over the seeds of A_s(DR) >= 0.8**: **"a soft sensor learned with domain
    randomisation in the simulation replaces the measurement (in simulation)"**; reported with it
    whether every seed reaches 0.8. Otherwise "does not replace the measurement", with the median
    and range;
  * value of the randomisation: Delta_rand(s) = A_s(DR) - A_s(nom); value of the exhaust humidity:
    Delta_Y(s) = A_s(DR) - A_s(DRoY) (positive = the humidity helps); each per seed (paired by the
    master seed: same excitation, split and model seed), median and range over the seeds. A
    direction ("helps", "hurts") is claimed only if the paired t-test over the seeds
    (``evaluation.paired_resolution.paired_comparison``, 95 %, mean of the differences, metric
    -A_s) resolves it; otherwise "not resolved". Two tests (no multiplicity correction, named).

Reported separately, not part of the reading: the same with criterion (i); the position of every
F5 variant between F0 and F2 as in the predecessor; the soft-sensor test errors of the training
record (excited nominal, excited randomised and constant-run test sets, named).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import REPO_ROOT, RUNS_DIR  # noqa: E402

READING_CRITERION = "ii"
THRESHOLD = 0.8
SMALL_DENOMINATOR = 0.01  # of E_F1
MAIN_LAB_INTERVAL = "F4:4h"


def _stats(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def _by_pair(records: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    return {(int(r["draw"]), r["scenario"]): r for r in records}


def _ok(results: dict[str, Any], v: str, c: str) -> bool:
    return bool(results[v]["criteria"][c]["converged"])


def share_block(
    reference: dict[str, dict], records: dict[tuple[int, str], dict], pairs: list
) -> dict[str, Any]:
    """A_X per pair with the main statistic and the two sensitivity statistics."""
    e0 = np.array([reference["F0"][p]["energy_kwh"] for p in pairs])
    e1 = np.array([reference["F1"][p]["energy_kwh"] for p in pairs])
    e2 = np.array([reference["F2"][p]["energy_kwh"] for p in pairs])
    ex = np.array([records[p]["energy_kwh"] for p in pairs])
    denominator = e0 - e2
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (e0 - ex) / denominator
    finite = np.isfinite(a)
    large = denominator >= SMALL_DENOMINATOR * e1
    s2_den = float(np.median(denominator / e1))
    out = {
        "A": _stats(a[finite]),
        "pairs": len(pairs),
        "pairs_nonfinite": int(np.sum(~finite)),
        "pairs_denominator_nonpositive": int(np.sum(denominator <= 0)),
        "pairs_denominator_below_1pct_of_E_F1": int(np.sum(~large)),
        "S1_median_A_large_denominator": float(np.median(a[large])) if large.any() else None,
        "S2_ratio_of_medians": float(np.median((e0 - ex) / e1)) / s2_den if s2_den else None,
        "F0_minus_X_vs_F1": _stats((e0 - ex) / e1),
        "X_minus_F2_vs_F1": _stats((ex - e2) / e1),
        "per_pair_A": [None if not math.isfinite(x) else float(x) for x in a],
    }
    main = out["A"]["median"]
    sens = [s for s in (out["S1_median_A_large_denominator"], out["S2_ratio_of_medians"]) if s]
    out["sensitive_to_small_denominators"] = any(
        (s >= THRESHOLD) != (main >= THRESHOLD) for s in sens
    )
    return out


def position(captured: float) -> str:
    """Position between F0 (0) and F2 (1), as in the predecessor."""
    if captured <= 0.0:
        return "at F0 or worse"
    if captured >= 1.0:
        return "at F2 or better"
    return "between F0 and F2"


def analyse(payload: dict[str, Any], predecessor: dict[str, Any]) -> dict[str, Any]:
    results = payload["results"]
    variants = list(payload["variants"])
    tol = float(payload["violation_tolerance_kg_per_kg"])
    out: dict[str, Any] = {"variants": variants, "criteria": {}}
    for c in ("ii", "i"):
        reference = {
            v: _by_pair(predecessor["results"][v]["criteria"][c]["records"])
            for v in ("F0", "F1", "F2")
        }
        pairs = sorted(reference["F1"])
        block: dict[str, Any] = {
            "reference_margins": {
                v: predecessor["results"][v]["criteria"][c]["margin"] for v in ("F0", "F1", "F2")
            },
            "per_variant": {},
        }
        for v in variants:
            entry = results[v]["criteria"][c]
            info: dict[str, Any] = {
                "margin": entry.get("margin"),
                "converged": _ok(results, v, c),
                "k_required": entry["k_required"],
                "holding": entry.get("holding"),
                "tried": entry.get("tried"),
            }
            if info["converged"]:
                records = _by_pair(entry["records"])
                if sorted(records) != pairs:
                    raise SystemExit(f"{v}: pairs differ from the predecessor's")
                recs = [records[p] for p in pairs]
                violations = np.array([r["max_violation"] for r in recs])
                info["max_violation"] = float(violations.max())
                info["violating_pairs"] = int(np.sum(violations > tol))
                info["failures"] = int(sum(r["failures"] for r in recs))
                info["failures_pre_roll"] = int(sum(r["failures_pre_roll"] for r in recs))
                info["energy_kwh"] = _stats(np.array([r["energy_kwh"] for r in recs]))
                info["bias_mean"] = _stats(np.array([r["bias_mean"] for r in recs]))
                info["bias_end_pre_roll"] = _stats(np.array([r["bias_end_pre_roll"] for r in recs]))
                info["feedback_error_rmse"] = _stats(
                    np.array([r["feedback_error_rmse"] for r in recs])
                )
                for quantity in ("energy_kwh", "co2_kg"):
                    info[f"relative_{quantity}_vs_F1"] = _stats(
                        np.array(
                            [records[p][quantity] / reference["F1"][p][quantity] - 1 for p in pairs]
                        )
                    )
                info["share"] = share_block(reference, records, pairs)
                s2 = info["share"]["S2_ratio_of_medians"]
                info["position_S2"] = position(s2) if s2 is not None else None
            block["per_variant"][v] = info
        block["La"] = reading_la(block["per_variant"], variants)
        block["Lb"] = reading_lb(block["per_variant"], variants)
        out["criteria"][c] = block
    out["La"] = out["criteria"][READING_CRITERION]["La"]
    out["Lb"] = out["criteria"][READING_CRITERION]["Lb"]
    out["runtime_s"] = payload.get("runtime_s")
    out["stage_runtime_s"] = payload.get("stage_runtime_s")
    out["seconds_per_pair_and_job"] = payload.get("seconds_per_pair_and_job")
    out["budget_dropped"] = payload.get("budget_dropped")
    out["control"] = {
        v: {k: x for k, x in info.items() if k != "comparisons"}
        for v, info in (payload.get("control") or {}).get("variants", {}).items()
    }
    return out


def _median_a(info: dict[str, Any] | None) -> float:
    """Main statistic of a variant; -inf if it has no margin (counts as below every threshold)."""
    if info is None or not info["converged"]:
        return -math.inf
    return float(info["share"]["A"]["median"])


def reading_la(per_variant: dict[str, Any], variants: list[str]) -> dict[str, Any]:
    labs = [v for v in variants if v.startswith("F4:")]
    values = {v: _median_a(per_variant.get(v)) for v in labs}
    sensitive = {
        v: per_variant[v]["share"]["sensitive_to_small_denominators"]
        for v in labs
        if per_variant[v]["converged"]
    }
    hours = {v: float(v.removeprefix("F4:").removesuffix("h")) for v in labs}
    if MAIN_LAB_INTERVAL in values and values[MAIN_LAB_INTERVAL] >= THRESHOLD:
        statement = (
            "periodic laboratory samples suffice; an inline sensor has little added value for "
            "this line (simulation, placeholder lab settings)"
        )
        passing = [v for v in labs if values[v] >= THRESHOLD]
    else:
        passing = [v for v in labs if values[v] >= THRESHOLD]
        if passing:
            longest = max(passing, key=hours.get)  # type: ignore[arg-type]
            statement = f"A >= {THRESHOLD} from an interval of {hours[longest]:g} h or shorter"
        else:
            statement = "none: no tested interval reaches A >= 0.8"
    return {
        "statement": statement,
        "A_median": {v: (None if math.isinf(x) else x) for v, x in values.items()},
        "passing": passing,
        "F4_4h_tested": MAIN_LAB_INTERVAL in values,
        "sensitive_to_small_denominators": sensitive,
    }


def _seed_values(per_variant: dict[str, Any], variants: list[str], label: str) -> dict[int, float]:
    prefix = f"F5:{label}:"
    return {
        int(v.removeprefix(prefix)): _median_a(per_variant.get(v))
        for v in variants
        if v.startswith(prefix)
    }


def _difference(a: dict[int, float], b: dict[int, float], name: str) -> dict[str, Any] | None:
    """Per-seed difference a - b, its median and range, and the paired test (metric -A)."""
    from twinlink_ml.evaluation.paired_resolution import paired_comparison

    seeds = sorted(set(a) & set(b))
    if not seeds or set(a) != set(b):
        return None
    diffs = {s: a[s] - b[s] for s in seeds}
    finite = all(math.isfinite(a[s]) and math.isfinite(b[s]) for s in seeds)
    out: dict[str, Any] = {
        "name": name,
        "per_seed": {str(s): (d if math.isfinite(d) else None) for s, d in diffs.items()},
    }
    if not finite:
        out["test"] = "not computed (a variant without margin)"
        return out
    values = np.array(list(diffs.values()))
    out.update(_stats(values))
    test = paired_comparison(
        {s: -a[s] for s in seeds},
        {s: -b[s] for s in seeds},
        candidate_label="a",
        reference_label="b",
    )
    resolved = (not test.degenerate) and abs(test.mean_difference) > test.threshold
    out["test"] = {
        "mean_difference": test.mean_difference,
        "threshold": test.threshold,
        "p_value": test.p_value,
        "degenerate": test.degenerate,
        "resolved": bool(resolved),
        "direction": ("a better" if test.mean_difference > 0 else "b better")
        if resolved
        else "not resolved",
    }
    return out


def reading_lb(per_variant: dict[str, Any], variants: list[str]) -> dict[str, Any]:
    dr = _seed_values(per_variant, variants, "DR")
    nom = _seed_values(per_variant, variants, "nom")
    oy = _seed_values(per_variant, variants, "DRoY")
    if not dr:
        return {"statement": "F5-DR not run"}
    values = np.array(list(dr.values()))
    median = float(np.median(values))
    holds = median >= THRESHOLD
    return {
        "statement": (
            "a soft sensor learned with domain randomisation in the simulation replaces the "
            "measurement (in simulation)"
            if holds
            else "the domain-randomised soft sensor does not replace the measurement"
        ),
        "A_DR_per_seed": {str(s): (None if math.isinf(x) else x) for s, x in dr.items()},
        "A_DR_median_over_seeds": None if math.isinf(median) else median,
        "A_DR_range_over_seeds": [float(values.min()), float(values.max())],
        "every_seed_reaches_threshold": bool(np.all(values >= THRESHOLD)),
        "A_nom_per_seed": {str(s): (None if math.isinf(x) else x) for s, x in nom.items()},
        "A_DRoY_per_seed": {str(s): (None if math.isinf(x) else x) for s, x in oy.items()},
        "value_of_randomisation": _difference(dr, nom, "A(DR) - A(nom)"),
        "value_of_exhaust_humidity": _difference(dr, oy, "A(DR) - A(DRoY)"),
        "sensitive_to_small_denominators": {
            v: per_variant[v]["share"]["sensitive_to_small_denominators"]
            for v in variants
            if v.startswith("F5:DR:") and per_variant[v]["converged"]
        },
    }


def soft_sensor_test_errors(payload: dict[str, Any]) -> dict[str, Any] | None:
    soft = payload.get("soft_sensor")
    if not soft:
        return None
    record = json.loads((REPO_ROOT / soft["record"]).read_text(encoding="utf-8"))
    return {
        f"F5:{r['variant']}:{r['seed']}": {
            key: r[key]
            for key in (
                "test_excited_nominal",
                "test_excited_randomised",
                "test_constant_holdout_nominal",
                "training",
            )
        }
        for r in record["records"]
    }


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
    pred_path = REPO_ROOT / payload["predecessor"]["artifact"]
    if hashlib.sha256(pred_path.read_bytes()).hexdigest() != payload["predecessor"]["sha256"]:
        raise SystemExit(f"{pred_path}: SHA-256 differs from the one recorded by the run")
    with gzip.open(pred_path, "rt", encoding="utf-8") as handle:
        predecessor = json.load(handle)
    analysis = analyse(payload, predecessor)
    analysis["soft_sensor_test_errors"] = soft_sensor_test_errors(payload)
    analysis["source"] = {
        "name": args.name,
        "provenance": payload["provenance"],
        "predecessor": payload["predecessor"],
    }
    output.write_text(json.dumps(analysis, indent=1), encoding="utf-8")
    main_block = analysis["criteria"][READING_CRITERION]
    print("La:", json.dumps(analysis["La"], indent=1))
    print("Lb:", json.dumps(analysis["Lb"], indent=1))
    for v, info in main_block["per_variant"].items():
        share = info.get("share", {})
        print(
            f"{v}: margin {info['margin']}, holding {info['holding']}/{info['k_required']}, "
            f"A {share.get('A')}, S1 {share.get('S1_median_A_large_denominator')}, "
            f"S2 {share.get('S2_ratio_of_medians')}, dE vs F1 {info.get('relative_energy_kwh_vs_F1')}"
        )
    print("control:", analysis["control"])
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

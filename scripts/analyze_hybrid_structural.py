r"""Analysis and PRE-REGISTERED reading of the study docs/NN_hybrid_structural.md.

This docstring was committed before the first run of the study (scripts
``train_hybrid_structural.py`` and ``run_hybrid_structural_evaluation.py``), i.e. before any margin
result existed; it fixes the metrics and the reading.

Question (docs/12 section I, item 30)
-------------------------------------
Befund 122: with a truth of the SAME structure as the prior model, a refit of two kinetic
parameters (M2) needs margins of 0.0019-0.0039 kg/kg, and the hybrid (M3) is not resolvably better
than a pure GRU. That truth favours M2 by construction. Here the truth deviates STRUCTURALLY:
does a learned residual now pay off against a calibration of the prior model?

Set-up (fixed before the run; ``conf/hybrid_structural/hybrid_structural.yaml``)
-------------------------------------------------------------------------------
Code, task, fixed-point map, grid + zoom, test cases, inside / outside definition, margin
determination and the addendum on determinability (>= 20 pooled inside cases) are those of Befund
122, reused unchanged (``hybrid_model`` package, task functions of
``scripts/run_hybrid_evaluation.py``; see the docstring of ``scripts/analyze_hybrid_model.py``).
Changes:

* **Truth**: the 20 parameter draws of Befund 111 PLUS evaporation of a fixed share of the spray
  water in the freeboard above the bed (candidate C2 of Befund 117, emulated by the inputs,
  ``hybrid_structural.truth``). Share 0.30, fixed by a rule committed before it was computed
  (the candidate on a 0.05 grid whose bed-to-exhaust gap at the nominal point is closest to
  7.5 K): gap 7.42 K (``artifacts/runs/hybrid_structural_share_v1.json``). **Prior knowledge**
  (M0, basis of M2, M2b and M3): the model WITHOUT that term, nominal parameters.
* **Models**: M0 prior knowledge; M1 GRU; M2 refit of critical moisture and nu_max; M2b = M2 plus a
  constant bias of the moisture (mean of measured minus replay over the training runs); M3 hybrid
  (prior knowledge + GRU on the residual). No M4.
* ``n_train`` in {4, 8, 16}; master seeds 1-5; every model trained anew for the new truth.

Metrics (per model, master seed, ``n_train``; at the truth), as Befund 122
---------------------------------------------------------------------------
* **Margin** (main metric): the 95 % quantile over the INSIDE cases of the per-case needed constant
  margin m (limit 0.15 - m; bisection on [0, 0.12], tolerance 1e-4; raise rule of Befund 111).
  Not reached within 0.12: reported as "not reached"; in every comparison it counts as 0.12
  (censored, flagged). A cell with fewer than 20 pooled inside cases is "not determinable" and
  has no margin.
* Energy relative to the oracle, cases without a valid model point, moisture bias: as Befund 122.

Pre-registered reading
----------------------
Comparisons are PAIRED over the master seeds (``evaluation.paired_resolution.paired_comparison``:
t-test of the per-seed margins, candidate M3, reference M2 or M2b; difference = reference minus
candidate, positive = M3 needs the smaller margin), on the seeds whose cells are determinable
(the inside cases do not depend on the model).

Two **families** of three tests each: {M3 vs M2 at ``n_train`` 4, 8, 16} and {M3 vs M2b at
``n_train`` 4, 8, 16}. Within each family the p-values are corrected by Holm (step-down,
family-wise 5 %). A difference is **resolved** iff

1. Holm rejects it in its family (adjusted p < 0.05), AND
2. **sign consistency**: at least 4 of the 5 seeds have a paired difference of the sign of the
   mean difference (a difference of exactly 0 counts for neither sign; with fewer than 4
   determinable seeds nothing is resolved).

A degenerate comparison (fewer than 2 pairs or zero spread of the differences) is never resolved.
The resolution thresholds (two-sided 95 % t threshold of the mean paired difference, uncorrected)
are reported for every test, next to the raw and the Holm-adjusted p-values and the signs.

* **Z0** -- with a structural error the residual pays off: at AT LEAST ONE ``n_train`` in
  {4, 8, 16} the margin of M3 is resolved smaller than that of M2 AND resolved smaller than that
  of M2b (both at the same ``n_train``).
* **Z1** -- calibration (possibly with a bias) stays the first choice with a structural error too:
  for M2 or for M2b (the same model X at every ``n_train``) the margin of X is, at EVERY
  ``n_train`` in {4, 8, 16}, resolved smaller than that of M3, or of equal size. "Of equal size"
  is fixed here as: the difference is not resolved in either direction and the median margin of X
  over the seeds is not larger than that of M3. Z1 additionally needs at least one resolved
  difference among the six tests (otherwise Z2).
* **Z2** -- nothing resolved: none of the six tests is resolved. Reported as such, with the
  thresholds; it is "not resolved", not equality.

Order of the verdict: Z0 if its criterion holds; else Z2 if nothing is resolved; else Z1 if its
criterion holds; else "mixed (none of Z0, Z1, Z2)", reported with the six tests. An ``n_train``
without a determinable cell drops out of the reading and is reported; if ``n_train`` 4 is removed by
the budget rule, the reading rests on 8 and 16 with families of two tests.

Reported WITHOUT a reading
--------------------------
* One-step (h1) and rollout (h36) errors of every model, physical units, test set named (excited
  test set, constant-run holdout, constant-run holdout with widened ambient ranges).
* Violation shares outside the training range ((a) extrapolation, (b) interior outside) at the
  margin determined inside, pooled over the seeds; "holds" iff <= 5 %.
* The M2 margin against Befund 122 (cost of the structural error): medians per ``n_train`` here
  (5 seeds) and in ``hybrid_model_v1_analysis.json`` (3 seeds), their difference, and the same on
  the common seeds 1-3 with the paired differences. Descriptive: the two studies differ only in
  the truth, but no test is fixed for this comparison.
* M0 and M1 margins, energies, moisture bias, cases without a valid point, the discretisation of
  the oracle (stated next to every energy figure if above 0.2 %).

Budget rule of the request: if the projection of the main run exceeds Monday 07:00, first
``n_train`` 4 is dropped, then M0. Neither changes the reading (M0 is not part of it).

    tools/run_here.sh scripts/analyze_hybrid_structural.py --prefix hybrid_structural_v1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import analyze_hybrid_model as base  # noqa: E402 - needs the script directory on the path

ROOT = SCRIPTS.parent
RUNS = ROOT / "artifacts" / "runs"
MODELS = ("M0", "M1", "M2", "M2b", "M3")
READING_N = (4, 8, 16)
FAMILIES = {"M3_vs_M2": "M2", "M3_vs_M2b": "M2b"}
CANDIDATE = "M3"
ALPHA = 0.05
MIN_CONSISTENT = 4
OUTSIDE_LIMIT = 0.05
DISCRETISATION_THRESHOLD = 0.002
BEFUND_122 = "hybrid_model_v1_analysis"
MOISTURE = "product_moisture_kg_per_kg"


def holm_adjusted(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (NaN stays NaN and takes no part)."""
    finite = [(p, i) for i, p in enumerate(p_values) if np.isfinite(p)]
    out = [math.nan] * len(p_values)
    running = 0.0
    for rank, (p, i) in enumerate(sorted(finite)):
        running = max(running, min(1.0, (len(finite) - rank) * p))
        out[i] = running
    return out


def paired_test(cand: dict[int, float], ref: dict[int, float]) -> dict[str, Any]:
    """Paired comparison candidate vs reference over their common seeds (lower is better)."""
    from twinlink_ml.evaluation.paired_resolution import paired_comparison

    seeds = sorted(set(cand) & set(ref))
    out: dict[str, Any] = {"seeds": seeds, "n_pairs": len(seeds)}
    if not seeds:
        return {**out, "degenerate": True, "p_value": math.nan}
    c = paired_comparison({s: cand[s] for s in seeds}, {s: ref[s] for s in seeds})
    differences = [ref[s] - cand[s] for s in seeds]
    sign = np.sign(c.mean_difference)
    return {
        **out,
        "differences_ref_minus_cand": differences,
        "mean_difference_ref_minus_cand": c.mean_difference,
        "sd_difference": c.sd_difference,
        "threshold": c.threshold,
        "p_value": c.p_value,
        "degenerate": c.degenerate,
        "n_positive": int(sum(d > 0 for d in differences)),
        "n_negative": int(sum(d < 0 for d in differences)),
        "n_consistent": int(sum(np.sign(d) == sign for d in differences)) if sign != 0 else 0,
    }


def resolve_family(tests: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Holm within one family, plus the sign-consistency condition (module docstring)."""
    from twinlink_ml.evaluation.b1_analysis import holm

    order = sorted(tests)
    p = [tests[n]["p_value"] if not tests[n]["degenerate"] else math.nan for n in order]
    adjusted = holm_adjusted(p)
    rejected = holm(p, ALPHA)
    out = {}
    for n, p_adj, reject in zip(order, adjusted, rejected, strict=True):
        t = dict(tests[n])
        consistent = (not t["degenerate"]) and t.get("n_consistent", 0) >= MIN_CONSISTENT
        resolved = bool(reject and consistent)
        mean = t.get("mean_difference_ref_minus_cand", math.nan)
        t.update(
            {
                "p_holm": p_adj,
                "holm_rejects": bool(reject),
                "family_size": int(sum(np.isfinite(v) for v in p)),
                "sign_consistent": bool(consistent),
                "resolved": resolved,
                "candidate_smaller": bool(resolved and mean > 0),
                "candidate_larger": bool(resolved and mean < 0),
            }
        )
        out[n] = t
    return out


def reading(table: dict[tuple[str, int], list[dict[str, Any]]]) -> dict[str, Any]:
    """Z0 / Z1 / Z2 on the determinable cells (module docstring)."""

    def cells(model: str, n: int) -> list[dict[str, Any]]:
        return [r for r in table.get((model, n), []) if r["determinable"]]

    def per_seed(model: str, n: int) -> dict[int, float]:
        return {r["seed"]: r["margin_censored"] for r in cells(model, n)}

    def median(model: str, n: int) -> float | None:
        values = list(per_seed(model, n).values())
        return float(np.median(values)) if values else None

    assessed = [n for n in READING_N if cells(CANDIDATE, n)]
    families: dict[str, dict[int, dict[str, Any]]] = {}
    for name, reference in FAMILIES.items():
        tests = {
            n: paired_test(per_seed(CANDIDATE, n), per_seed(reference, n))
            for n in assessed
            if cells(reference, n)
        }
        families[name] = resolve_family(tests)
        for n, t in families[name].items():
            t["median_margin_candidate"] = median(CANDIDATE, n)
            t["median_margin_reference"] = median(reference, n)

    def complete(name: str) -> bool:
        return bool(assessed) and all(n in families[name] for n in assessed)

    z0_at = [
        n
        for n in assessed
        if all(n in families[f] and families[f][n]["candidate_smaller"] for f in FAMILIES)
    ]
    any_resolved = any(t["resolved"] for f in families.values() for t in f.values())

    def z1_for(name: str) -> dict[str, Any]:
        per_n = {}
        for n in assessed:
            t = families[name].get(n)
            if t is None:
                per_n[n] = None
                continue
            equal = (not t["resolved"]) and (
                t["median_margin_reference"] <= t["median_margin_candidate"]
            )
            per_n[n] = {
                "reference_resolved_smaller": t["candidate_larger"],
                "of_equal_size": bool(equal),
                "ok": bool(t["candidate_larger"] or equal),
            }
        holds = complete(name) and all(v is not None and v["ok"] for v in per_n.values())
        return {"holds_at_every_n_train": bool(holds), "per_n_train": per_n}

    z1 = {FAMILIES[name]: z1_for(name) for name in FAMILIES}
    z1_models = [m for m, v in z1.items() if v["holds_at_every_n_train"]]
    if not assessed:
        verdict = "not determinable (no n_train with a determinable cell)"
    elif z0_at:
        verdict = "Z0"
    elif not any_resolved:
        verdict = "Z2"
    elif z1_models:
        verdict = "Z1"
    else:
        verdict = "mixed (none of Z0, Z1, Z2)"
    return {
        "verdict": verdict,
        "n_train_assessed": assessed,
        "n_train_not_assessed": [n for n in READING_N if n not in assessed],
        "z0_n_train": z0_at,
        "z1_models": z1_models,
        "z1_detail": z1,
        "any_resolved": bool(any_resolved),
        "families": families,
        "settings": {
            "alpha": ALPHA,
            "min_consistent_signs": MIN_CONSISTENT,
            "correction": "Holm within each family",
        },
    }


def compare_with_befund_122(
    table: dict[tuple[str, int], list[dict[str, Any]]], reference: dict[str, Any] | None
) -> dict[str, Any] | None:
    """M2 margin here against Befund 122 (descriptive, module docstring)."""
    if reference is None:
        return None
    out: dict[str, Any] = {}
    for n in READING_N:
        here = {
            r["seed"]: r["margin_censored"] for r in table.get(("M2", n), []) if r["determinable"]
        }
        old = reference["curves"].get(f"M2/n{n}", {}).get("margin_censored")
        if not here or old is None:
            continue
        then = {int(s): v for s, v in old["per_seed"].items() if v is not None}
        common = sorted(set(here) & set(then))
        out[f"n{n}"] = {
            "median_here": float(np.median(list(here.values()))),
            "range_here": [min(here.values()), max(here.values())],
            "n_seeds_here": len(here),
            "median_befund_122": old["median"],
            "range_befund_122": [old["min"], old["max"]],
            "n_seeds_befund_122": old["n_seeds"],
            "difference_of_medians": float(np.median(list(here.values()))) - old["median"],
            "common_seeds": common,
            "median_here_common_seeds": float(np.median([here[s] for s in common]))
            if common
            else None,
            "paired_differences_here_minus_122": [here[s] - then[s] for s in common],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefix", default="hybrid_structural_v1")
    parser.add_argument("--margins", default=None, help="default <prefix>_margins_per_draw")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from _common import provenance, require_clean_tree

    from twinlink_ml.utils.code_location import require_code_location

    require_code_location(ROOT)
    output = RUNS / f"{args.output or args.prefix + '_analysis'}.json"
    if output.exists():
        raise SystemExit(f"{output} exists -- artifacts are never overwritten")
    require_clean_tree(ignore_paths=[output])
    margins_path = RUNS / f"{args.margins or args.prefix + '_margins_per_draw'}.jsonl.gz"
    oracle_path = RUNS / f"{args.prefix}_oracle.jsonl.gz"
    errors_path = RUNS / f"{args.prefix}_errors.jsonl.gz"
    margins = base.jobs(margins_path, kind="cell")
    oracle_jobs = base.jobs(oracle_path)
    errors = base.jobs(errors_path) if errors_path.exists() else []

    rows = [base.summarise_job(j) for j in margins]
    table: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        table[(r["model"], r["n_train"])].append(r)
    for v in table.values():
        v.sort(key=lambda r: r["seed"])
    keys = (
        "margin",
        "margin_censored",
        "parameter",
        "energy_relative_median",
        "energy_excess_pooled",
        "holding_share_inside",
        "n_inside",
        "moisture_bias_at_action_inside",
        "moisture_bias_at_action_b",
        "moisture_bias_at_action_a",
        "coarse_bias_inside",
        "coarse_bias_b",
        "coarse_bias_a",
    )
    curves = {
        f"{m}/n{n}": {
            **{key: base.curve(v, key) for key in keys},
            "n_not_reached": sum(r["determinable"] and not r["reached"] for r in v),
            "n_not_determinable": sum(not r["determinable"] for r in v),
            "model_invalid_at_action": sum(r["model_invalid_at_action"] for r in v),
            "model_invalid_inside": sum(r["model_invalid_inside"] for r in v),
            "bed_violations": sum(r["bed_violations"] for r in v),
        }
        for (m, n), v in sorted(table.items())
    }
    outside: dict[str, Any] = {}
    for (m, n), v in sorted(table.items()):
        entry = {}
        for set_name in ("a", "b"):
            bad = sum(r[f"violations_{set_name}"][0] for r in v)
            total = sum(r[f"violations_{set_name}"][1] for r in v)
            share = bad / total if total else None
            entry[set_name] = {
                "violations": bad,
                "cases": total,
                "share": share,
                "margin_holds_outside": None if share is None else bool(share <= OUTSIDE_LIMIT),
            }
        outside[f"{m}/n{n}"] = entry

    disc = [j for j in oracle_jobs if j["key"].startswith("discretisation_")]
    worst = max((abs(j["energy_difference_relative"]) for j in disc), default=math.nan)
    discretisation = {
        "n_cases": len(disc),
        "max_abs_energy_difference_relative": worst,
        "max_abs_slack_difference_kg_per_kg": max(
            (abs(j["slack_difference_kg_per_kg"]) for j in disc), default=math.nan
        ),
        "report_next_to_energy": bool(np.isfinite(worst) and worst > DISCRETISATION_THRESHOLD),
    }

    error_table: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for job in errors:
        for cell in job["cells"]:
            for set_name, entry in cell["test_sets"].items():
                for m, parts in entry["models"].items():
                    for part, metrics in parts.items():
                        for key, value in metrics.items():
                            error_table[f"{m}/n{cell['n_train']}/{set_name}/{part}"][key].append(
                                value
                            )
    error_summary = {
        k: {
            key: {
                "median": float(np.median(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
                "n": len(v),
            }
            for key, v in metrics.items()
        }
        for k, metrics in sorted(error_table.items())
    }
    reference_path = RUNS / f"{BEFUND_122}.json"
    reference = (
        json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.exists() else None
    )
    headers = base.headers(margins_path)
    payload = {
        "provenance": provenance("not used", ignore_paths=[output]),
        "inputs": {
            "margins": margins_path.name,
            "oracle": oracle_path.name,
            "errors": errors_path.name if errors else None,
            "befund_122": reference_path.name if reference is not None else None,
            "margin_headers": [h["provenance"] for h in headers],
            "structure": headers[0].get("structure") if headers else None,
            "selection": [h.get("selection") for h in headers],
        },
        "inside_cases_per_cell": {
            f"{r['model']}/s{r['seed']}/n{r['n_train']}": {
                "n_inside": r["n_inside"],
                "determinable": r["determinable"],
            }
            for r in rows
        },
        "reading": reading(table),
        "m2_against_befund_122": compare_with_befund_122(table, reference),
        "outside": outside,
        "discretisation": discretisation,
        "curves": curves,
        "per_job": rows,
        "errors": error_summary,
        "moisture_error_key": f"rmse/{MOISTURE}/h36",
    }
    output.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    print(json.dumps(payload["reading"], indent=1, default=float)[:6000])
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

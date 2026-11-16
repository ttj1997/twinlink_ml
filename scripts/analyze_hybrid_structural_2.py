r"""Analysis and PRE-REGISTERED cost measure and reading of docs/NN_hybrid_structural_2.md.

This docstring was committed before the first data, training or evaluation run of the study, i.e.
before any margin or energy result existed; it fixes the cost measure, the statistics and the
reading. Known at that time: only the moisture of M0 against the truth at M0's operating point
(``scripts/probe_hybrid_structural_2_strength.py``), which fixes the strength of the error.

Question
--------
Befund 124 (docs/41_hybrid_structural.md): with a structural error to the SAFE side (truth drier
than the prior model predicts) the hybrid M3 and the prior model M0 need no margin and pay in
energy; the pre-registered reading on the margin came out "mixed", and no reading on the energy
was fixed. Here the error has the OTHER sign: the bed is wetter than the prior model predicts, so
the conservatism of the prior structure turns into violations. Which model reaches the limit at
the lowest energy: calibration with a bias (M2b), the hybrid (M3) or the pure GRU (M1)?

Set-up (fixed before the run; ``conf/hybrid_structural_2/hybrid_structural_2.yaml``)
-----------------------------------------------------------------------------------
Everything is that of Befund 124, reused unchanged: code (``hybrid_model``, ``hybrid_structural``,
the drivers ``scripts/train_hybrid_structural.py`` and ``scripts/run_hybrid_structural_evaluation.py``
called with ``--config``), task, fixed-point map, grid + zoom, the 140 test cases, the inside /
outside definition, determinability (>= 20 pooled inside cases), the margin determination
(95 % quantile over the inside cases, bisection on [0, 0.12], tolerance 1e-4, raise rule of
Befund 111), the models M0, M1, M2, M2b, M3, ``n_train`` in {4, 8, 16}, master seeds 1-5 and the
20 parameter draws of Befund 111. The config differs from that of Befund 124 in the section
``structure`` only (test). The one change is the truth:

* **Truth**: the draw parameters PLUS exhaust recirculation, an existing switch of the grey-box
  model (``OperationConfig.exhaust_recirculation_fraction``; ``hybrid_structural_2.study``): a
  fraction of the bed exhaust is mixed into the inlet air ahead of the heater, the bed is wetter.
  The fraction is fixed by a rule committed before it was computed (config, section
  ``structure``): M0 underestimates the moisture at its own operating point by 0.03-0.07 kg/kg in
  the median over the 100 interior cases. **Prior knowledge** (M0, basis of M2, M2b and M3): the
  model WITHOUT the term, nominal parameters.

Cost measure (new; per model, master seed and ``n_train``)
----------------------------------------------------------
**Energy relative to the oracle at the margin that holds in >= 95 % of the inside cases.**

* The margin of a cell is the one the study determines (as Befund 124): the pooled 95 % quantile
  of the per-case needed margin over the inside cases, raised by the rule of Befund 111 until
  the evaluation at the truth holds in >= 95 % of the inside cases.
* A cell is **admissible** iff it is determinable (>= 20 inside cases), the margin is reached
  within the upper end of the search (0.12 kg/kg) AND the final evaluation holds in >= 95 % of
  the inside cases. Otherwise the cell is **not admissible**: it is counted as such per model and
  ``n_train`` and takes part in NO comparison -- no substitute value is used for it.
* Cost of an admissible cell: the median over its inside cases of ``E_truth(model's operating
  point at that margin) / E_truth(oracle operating point)``, both at the truth (the
  ``energy_relative_median`` of Befund 122/124; inside cases at which the model has no valid
  operating point give no solution and are left out, their number is reported). Lower is better;
  1.0 is the oracle.

Statistics
----------
Comparisons are PAIRED over the master seeds (``evaluation.paired_resolution.paired_comparison``:
t-test of the per-seed costs; difference = reference minus candidate, positive = the candidate
has the lower energy), on the seeds at which BOTH cells are admissible; the seeds dropped for each
side are listed. Four **families** of three tests each (``n_train`` 4, 8, 16):

    {M3 vs M2b},  {M3 vs M2},  {M1 vs M2b},  {M1 vs M3}        (candidate vs reference)

Within each family the p-values are corrected by Holm (step-down, family-wise 5 %). A difference
is **resolved** iff

1. Holm rejects it in its family (adjusted p < 0.05), AND
2. **sign consistency**: at least 4 of the 5 seeds have a paired difference of the sign of the
   mean difference (a difference of exactly 0 counts for neither sign; a test with fewer than 4
   pairs can therefore not be resolved).

A degenerate comparison (fewer than 2 pairs or zero spread of the differences) is never resolved.
The resolution thresholds (two-sided 95 % t threshold of the mean paired difference, uncorrected)
are reported for every test, next to the raw and the Holm-adjusted p-values and the signs.

Pre-registered reading (``n_train`` <= 16, i.e. 4, 8, 16)
--------------------------------------------------------
* **E0** -- calibration with a bias is the first choice in both directions of the error: at AT
  LEAST ONE ``n_train`` M2b has a resolved lower energy than M3, and at NO ``n_train`` M3 has a
  resolved lower energy than M2b.
* **E1** -- the residual pays off when the prior knowledge errs to the unsafe side: at AT LEAST
  ONE ``n_train`` M3 has a resolved lower energy than M2b AND a resolved lower energy than M2
  (both at the same ``n_train``).
* **E2** -- pure learning suffices: at AT LEAST ONE ``n_train`` M1 has a resolved lower energy
  than M2b AND a resolved lower energy than M3 (both at the same ``n_train``).
* **E3** -- nothing resolved: none of the twelve tests is resolved. Reported as such, with the
  thresholds; it is "not resolved", not equality.

Verdict: the one of E0-E3 whose criterion holds. If more than one holds or none: **"mixed"**,
reported with the criteria that hold and the twelve tests, and not reinterpreted. An ``n_train``
at which a test has fewer than 4 pairs (cells not determinable or not admissible) cannot
contribute a resolved difference; this is reported next to the verdict. If ``n_train`` 4 (or M0)
is removed by the budget rule, the reading rests on what was run (families of two tests; M0 is
not part of the reading).

Reported WITHOUT a reading
--------------------------
* Share of inside cases violated at margin 0 (the model's optimum at the plain limit, evaluated
  at the truth), per model, pooled over the seeds and per seed; M0 and M3 are named in the
  request (expectation of the project lead: above 5 %).
* Margins (median and range over the seeds, "not reached" listed), admissibility counts.
* One-step (h1) and rollout (h36) errors of every model, physical units, test set named (excited
  test set, constant-run holdout, constant-run holdout with widened ambient ranges).
* Violation shares outside the training range ((a) extrapolation, (b) interior outside) at the
  margin determined inside, pooled over the seeds; "holds" iff <= 5 %.
* Share of the M2 refits that end at their parameter bounds (training records).
* Moisture bias at the operating point, cases without a valid point, discretisation of the oracle
  (stated next to every energy figure if above 0.2 %).

POST HOC, separate and labelled as such
---------------------------------------
The same cost measure with the same statistics and the same criteria E0-E3 applied to the
existing artifact of Befund 124 (``hybrid_structural_v1_margins_per_draw``; nothing is run anew),
so that both directions of the error stand next to each other. Befund 124 pre-registered a
reading on the margin, not on the energy, and its energies are known (docs/41 section 11); this
part is therefore post hoc, written to the key ``post_hoc_befund_124`` and never merged with the
verdict of this study.

    tools/run_here.sh scripts/analyze_hybrid_structural_2.py --prefix hybrid_structural_2_v1
"""

from __future__ import annotations

import argparse
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
import analyze_hybrid_structural as prev  # noqa: E402

ROOT = SCRIPTS.parent
RUNS = ROOT / "artifacts" / "runs"
MODELS = ("M0", "M1", "M2", "M2b", "M3")
READING_N = (4, 8, 16)
FAMILIES = {  # name -> (candidate, reference)
    "M3_vs_M2b": ("M3", "M2b"),
    "M3_vs_M2": ("M3", "M2"),
    "M1_vs_M2b": ("M1", "M2b"),
    "M1_vs_M3": ("M1", "M3"),
}
HOLDING_SHARE = 0.95
OUTSIDE_LIMIT = 0.05
DISCRETISATION_THRESHOLD = 0.002
BEFUND_124_PREFIX = "hybrid_structural_v1"
MOISTURE = "product_moisture_kg_per_kg"
BOUND_TOLERANCE = 1.0e-6


def cost_row(job: dict[str, Any]) -> dict[str, Any]:
    """The summary of Befund 122 for one cell plus admissibility, cost and violations at margin 0."""
    row = base.summarise_job(job)
    inside = [c for c in job["cases"] if c["inside"]]
    n_holding = sum(c["holds"] for c in inside)
    holds_enough = bool(inside) and n_holding >= HOLDING_SHARE * len(inside) - 1e-9
    admissible = bool(row["determinable"] and row["reached"] and holds_enough)
    energy = row["energy_relative_median"] if admissible else None
    needed = job["needed"]
    row.update(
        {
            "admissible": admissible,
            "not_admissible_reason": None
            if admissible
            else "not determinable"
            if not row["determinable"]
            else "margin not reached within the upper end"
            if not row["reached"]
            else "final evaluation holds in fewer than 95 % of the inside cases",
            "energy_cost": energy,
            # needed == 0: the truth holds at the model's optimum for the plain limit
            "violations_at_margin_0": [sum(r["needed"] != 0.0 for r in needed), len(needed)],
        }
    )
    return row


def tabulate(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    table: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        table[(r["model"], r["n_train"])].append(r)
    for v in table.values():
        v.sort(key=lambda r: r["seed"])
    return dict(table)


def admissibility(table: dict[tuple[str, int], list[dict[str, Any]]]) -> dict[str, Any]:
    """Counts per model and ``n_train``: cells, determinable, admissible, and why not."""
    out = {}
    for (m, n), v in sorted(table.items()):
        out[f"{m}/n{n}"] = {
            "n_cells": len(v),
            "n_determinable": sum(r["determinable"] for r in v),
            "n_admissible": sum(r["admissible"] for r in v),
            "admissible_seeds": [r["seed"] for r in v if r["admissible"]],
            "not_admissible": {
                str(r["seed"]): r["not_admissible_reason"] for r in v if not r["admissible"]
            },
        }
    return out


def reading(table: dict[tuple[str, int], list[dict[str, Any]]]) -> dict[str, Any]:
    """E0 / E1 / E2 / E3 on the admissible cells (module docstring)."""

    def per_seed(model: str, n: int) -> dict[int, float]:
        return {
            r["seed"]: r["energy_cost"]
            for r in table.get((model, n), [])
            if r["admissible"] and r["energy_cost"] is not None
        }

    def median(model: str, n: int) -> float | None:
        values = list(per_seed(model, n).values())
        return float(np.median(values)) if values else None

    models = {m for pair in FAMILIES.values() for m in pair}
    assessed = [n for n in READING_N if any(table.get((m, n)) for m in models)]
    families: dict[str, dict[int, dict[str, Any]]] = {}
    for name, (candidate, reference) in FAMILIES.items():
        tests = {
            n: prev.paired_test(per_seed(candidate, n), per_seed(reference, n)) for n in assessed
        }
        families[name] = prev.resolve_family(tests)
        for n, t in families[name].items():
            run = {m: [r["seed"] for r in table.get((m, n), [])] for m in (candidate, reference)}
            t.update(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "median_energy_candidate": median(candidate, n),
                    "median_energy_reference": median(reference, n),
                    "seeds_dropped_candidate": sorted(
                        set(run[candidate]) - set(per_seed(candidate, n))
                    ),
                    "seeds_dropped_reference": sorted(
                        set(run[reference]) - set(per_seed(reference, n))
                    ),
                    "resolvable": t["n_pairs"] >= prev.MIN_CONSISTENT,
                }
            )

    def lower(name: str, n: int) -> bool:  # the candidate has the resolved lower energy
        return bool(families[name][n]["candidate_smaller"])

    def higher(name: str, n: int) -> bool:  # the reference has the resolved lower energy
        return bool(families[name][n]["candidate_larger"])

    e0_at = [n for n in assessed if higher("M3_vs_M2b", n)]
    e0_against = [n for n in assessed if lower("M3_vs_M2b", n)]
    e1_at = [n for n in assessed if lower("M3_vs_M2b", n) and lower("M3_vs_M2", n)]
    e2_at = [n for n in assessed if lower("M1_vs_M2b", n) and lower("M1_vs_M3", n)]
    any_resolved = any(t["resolved"] for f in families.values() for t in f.values())
    criteria = {
        "E0": bool(e0_at) and not e0_against,
        "E1": bool(e1_at),
        "E2": bool(e2_at),
        "E3": bool(assessed) and not any_resolved,
    }
    holding = [name for name, ok in criteria.items() if ok]
    if not assessed:
        verdict = "not assessable (no n_train with a cell)"
    elif len(holding) == 1:
        verdict = holding[0]
    elif holding:
        verdict = f"mixed ({' and '.join(holding)} hold)"
    else:
        verdict = "mixed (none of E0, E1, E2, E3)"
    return {
        "verdict": verdict,
        "criteria": criteria,
        "n_train_assessed": assessed,
        "n_train_not_assessed": [n for n in READING_N if n not in assessed],
        "e0_m2b_resolved_lower_than_m3_at": e0_at,
        "e0_m3_resolved_lower_than_m2b_at": e0_against,
        "e1_n_train": e1_at,
        "e2_n_train": e2_at,
        "any_resolved": bool(any_resolved),
        "tests_not_resolvable_fewer_than_4_pairs": [
            f"{name}/n{n}"
            for name, f in families.items()
            for n, t in f.items()
            if not t["resolvable"]
        ],
        "families": families,
        "settings": {
            "alpha": prev.ALPHA,
            "min_consistent_signs": prev.MIN_CONSISTENT,
            "correction": "Holm within each family",
            "holding_share": HOLDING_SHARE,
            "cost": "median over the inside cases of E_truth(model point) / E_truth(oracle point) "
            "at the margin holding in >= 95 % of the inside cases; not admissible cells excluded",
        },
    }


def curves(table: dict[tuple[str, int], list[dict[str, Any]]]) -> dict[str, Any]:
    keys = (
        "energy_cost",
        "margin",
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
    return {
        f"{m}/n{n}": {
            **{key: base.curve(v, key) for key in keys},
            "n_not_reached": sum(r["determinable"] and not r["reached"] for r in v),
            "n_not_determinable": sum(not r["determinable"] for r in v),
            "n_not_admissible": sum(not r["admissible"] for r in v),
            "model_invalid_at_action": sum(r["model_invalid_at_action"] for r in v),
            "model_invalid_inside": sum(r["model_invalid_inside"] for r in v),
            "bed_violations": sum(r["bed_violations"] for r in v),
        }
        for (m, n), v in sorted(table.items())
    }


def violations_at_margin_0(table: dict[tuple[str, int], list[dict[str, Any]]]) -> dict[str, Any]:
    """Share of inside cases violated at margin 0, pooled over the determinable seeds, per seed."""
    out = {}
    for (m, n), v in sorted(table.items()):
        cells = [r for r in v if r["determinable"]]
        bad = sum(r["violations_at_margin_0"][0] for r in cells)
        total = sum(r["violations_at_margin_0"][1] for r in cells)
        out[f"{m}/n{n}"] = {
            "violations": bad,
            "cases": total,
            "share": bad / total if total else None,
            "per_seed": {
                str(r["seed"]): (r["violations_at_margin_0"][0] / r["violations_at_margin_0"][1])
                if r["violations_at_margin_0"][1]
                else None
                for r in v
            },
        }
    return out


def outside(table: dict[tuple[str, int], list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
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
        out[f"{m}/n{n}"] = entry
    return out


def refit_at_bounds(training: list[dict[str, Any]], header: dict[str, Any]) -> dict[str, Any]:
    """Per ``n_train``: number of M2 refits ending at a bound of each parameter, and at any."""
    settings = header["config"]["refit"]
    names = list(settings["parameters"])
    lower, upper = list(settings["lower"]), list(settings["upper"])
    out: dict[str, Any] = {}
    for record in training:
        if record.get("role") != "M2" or record.get("status") != "ok":
            continue
        entry = out.setdefault(
            f"n{record['n_train']}",
            {
                "n_refits": 0,
                "at_any_bound": 0,
                "at_a_bound_of_every_parameter": 0,
                "per_parameter": {p: {"at_lower": 0, "at_upper": 0} for p in names},
            },
        )
        theta = dict(zip(record["refit"]["names"], record["refit"]["theta"], strict=True))
        hits = []
        for p, lo, hi in zip(names, lower, upper, strict=True):
            scale = max(abs(hi - lo), 1e-12)
            at_lower = abs(theta[p] - lo) <= BOUND_TOLERANCE * scale
            at_upper = abs(theta[p] - hi) <= BOUND_TOLERANCE * scale
            entry["per_parameter"][p]["at_lower"] += at_lower
            entry["per_parameter"][p]["at_upper"] += at_upper
            hits.append(at_lower or at_upper)
        entry["n_refits"] += 1
        entry["at_any_bound"] += any(hits)
        entry["at_a_bound_of_every_parameter"] += all(hits)
    return out


def error_summary(errors: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for job in errors:
        for cell in job["cells"]:
            for set_name, entry in cell["test_sets"].items():
                for m, parts in entry["models"].items():
                    for part, metrics in parts.items():
                        for key, value in metrics.items():
                            table[f"{m}/n{cell['n_train']}/{set_name}/{part}"][key].append(value)
    return {
        k: {
            key: {
                "median": float(np.median(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
                "n": len(v),
            }
            for key, v in metrics.items()
        }
        for k, metrics in sorted(table.items())
    }


def cost_analysis(margins_path: Path) -> dict[str, Any]:
    """Cost table, admissibility and reading E0-E3 of one margins artifact."""
    rows = [cost_row(j) for j in base.jobs(margins_path, kind="cell")]
    table = tabulate(rows)
    headers = base.headers(margins_path)
    return {
        "margins": margins_path.name,
        "structure": headers[0].get("structure") if headers else None,
        "margin_headers": [h["provenance"] for h in headers],
        "selection": [h.get("selection") for h in headers],
        "reading": reading(table),
        "admissibility": admissibility(table),
        "violations_at_margin_0": violations_at_margin_0(table),
        "curves": curves(table),
        "outside": outside(table),
        "per_job": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefix", default="hybrid_structural_2_v1")
    parser.add_argument("--margins", default=None, help="default <prefix>_margins_per_draw")
    parser.add_argument("--training", default=None, help="default hybrid_structural_2_training_v1")
    parser.add_argument("--posthoc-prefix", default=BEFUND_124_PREFIX)
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
    training_path = RUNS / f"{args.training or 'hybrid_structural_2_training_v1'}.jsonl.gz"
    posthoc_path = RUNS / f"{args.posthoc_prefix}_margins_per_draw.jsonl.gz"

    study = cost_analysis(margins_path)
    oracle_jobs = base.jobs(oracle_path)
    errors = base.jobs(errors_path) if errors_path.exists() else []
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
    bounds = None
    if training_path.exists():
        training_headers = base.headers(training_path)
        bounds = refit_at_bounds(base.jobs(training_path, kind="task"), training_headers[0])
    post_hoc = None
    if posthoc_path.exists():
        post_hoc = {
            "label": "POST HOC: the cost measure and the criteria E0-E3 of this study applied to "
            "the existing artifact of Befund 124 (structural error to the safe side); nothing "
            "run anew; not part of the pre-registered verdict",
            **cost_analysis(posthoc_path),
        }
        del post_hoc["per_job"]  # the cells of Befund 124 are in its own analysis artifact
    payload = {
        "provenance": provenance("not used", ignore_paths=[output]),
        "inputs": {
            "margins": margins_path.name,
            "oracle": oracle_path.name,
            "errors": errors_path.name if errors else None,
            "training": training_path.name if bounds is not None else None,
            "post_hoc": posthoc_path.name if post_hoc is not None else None,
            "margin_headers": study["margin_headers"],
            "structure": study["structure"],
            "selection": study["selection"],
        },
        "inside_cases_per_cell": {
            f"{r['model']}/s{r['seed']}/n{r['n_train']}": {
                "n_inside": r["n_inside"],
                "determinable": r["determinable"],
                "admissible": r["admissible"],
            }
            for r in study["per_job"]
        },
        "reading": study["reading"],
        "admissibility": study["admissibility"],
        "violations_at_margin_0": study["violations_at_margin_0"],
        "outside": study["outside"],
        "refit_at_bounds": bounds,
        "discretisation": discretisation,
        "curves": study["curves"],
        "per_job": study["per_job"],
        "errors": error_summary(errors),
        "moisture_error_key": f"rmse/{MOISTURE}/h36",
        "post_hoc_befund_124": post_hoc,
    }
    import json

    output.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    print(json.dumps(payload["reading"], indent=1, default=float)[:8000])
    if post_hoc is not None:
        print(
            "POST HOC (Befund 124):",
            post_hoc["reading"]["verdict"],
            post_hoc["reading"]["criteria"],
        )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

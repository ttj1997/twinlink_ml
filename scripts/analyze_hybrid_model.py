r"""Analysis and PRE-REGISTERED reading of the study docs/NN_hybrid_model.md.

This docstring was committed before the first run of the study (scripts
``train_hybrid_model.py`` and ``run_hybrid_evaluation.py``); it fixes the metrics and the reading.

Question
--------
With few runs, how much safety margin does a hybrid model (grey-box prior knowledge plus a GRU on
the residual) save against a pure GRU, and against a simple refit of the grey-box parameters?

Set-up (fixed before the run; settings ``conf/hybrid_model/hybrid_model.yaml``)
------------------------------------------------------------------------------------
* **Truth**: the grey-box model of the Befund 94 pipeline with the material parameters of one of the
  20 draws of Befund 111 (critical moisture, nu_max, five GAB parameters; read from
  ``conf/sensor_value/sensor_value.yaml``). **Prior knowledge**: the same model, nominal parameters.
* **Data** per draw and master seed 1-3: the pool of Befund 94 (arm N, preset version 2) generated
  with the truth; run-level splits as Befund 94; ``n_train`` in {4, 8, 16, 32}.
* **Models**: M0 prior knowledge only; M1 GRU of the Befund 94 pipeline; M2 least-squares refit of
  critical moisture and nu_max (as M4 of Befund 107); M3 prior knowledge + GRU on the residual
  ``truth - nominal`` (same architecture and budget as M1; steady state = nominal steady state +
  fixed point of the residual network); M4 ensemble of 5 residual networks (M3 + 4 training seeds),
  constraint ``mean + k std``.
* **Task**: the one-step operating-point problem of Befund 113 (actions inlet temperature, gas
  velocity; context ambient temperature and humidity; limits moisture 0.15 kg/kg, bed 80 degC);
  steady state of a network = fixed point of its h36 map with the criterion of Befund 113; RTO by
  grid (21 x 21) + zoom (9 x 9, three levels, recentred, from the 4 best coarse points; set after
  the pilot's discretisation check, before the main run) on the penalised objective of the oracle,
  the same for every model and the oracle (decision of the project lead, 2026-11-03). Oracle = the
  truth with margin 0.
* **Cases** (the same for every model, seed and ``n_train``; rotation committed in the config):
  per draw 5 interior contexts of the 25 Befund-52 grid points (100 cases) and 2 extrapolation
  contexts (a) outside the disturbance box of the data design (40 cases).
* **Inside / outside**: an interior case is *inside* the training range of a (draw, seed,
  ``n_train``) cell iff the oracle operating point (T_in, u_gas, T_amb, phi) lies within 0.2 of the
  range of some training sample in all four channels (Chebyshev, ``hybrid_model/coverage.py``),
  else *outside (b)*; the (a) cases are always outside.

Metrics, per (model, master seed, ``n_train``), at the truth
--------------------------------------------------------------
* **Margin** (main metric). Per inside case the smallest constraint parameter at which the true
  moisture at the model's optimum stays <= 0.15 + 1e-4 (bisection; M0-M3 a constant margin m on
  [0, 0.12], tolerance 1e-4; M4 the factor k on [0, 10], tolerance 0.01). The model's parameter is
  the 95 % quantile over its inside cases (the ``ceil(0.95 N)``-th smallest; raise rule of Befund 111
  if the share holding at it is below 95 %). Not reached within the upper end: reported as "not
  reached", never an abort; in comparisons it counts as the upper end (censored, flagged). For M4
  the margin compared is the *effective* margin, median over its inside cases of ``k std`` at the
  chosen points; k is reported next to it.
* **Energy relative to the oracle**: true energy at the chosen point at the model's parameter
  divided by the oracle energy of the case; median over the inside cases (main), also pooled sums.
* **No valid point** (added after the pilot, before the main run): a model without a single valid
  grid point in a case (no steady state / no valid fixed point anywhere -- in the pilot one member
  of M4 at ``n_train`` 32 had an unstable moisture fixed point everywhere, which under the rule of
  Befund 113 makes the ensemble invalid) gives no solution there: the case counts as NOT holding
  and its energy is left out of the energy figures; the number of such cases is reported per model
  and ``n_train``.
* **Outside**: violation share (true moisture > 0.15 + 1e-4) at exactly the model's parameter on
  the outside cases (b) and on the extrapolation cases (a), separately; no margin search outside.
* **Bias of the moisture**: predicted minus true moisture at the chosen points (median over cases)
  and box-wide on the coarse grid (mean over valid grid points), inside / (b) / (a).
* **One-step and rollout error**: RMSE and bias at h1 and h36 in physical units on the excited test
  set, the constant-run holdout (runs split inside / outside (b)) and a constant-run holdout with the
  widened ambient ranges (runs outside the data box = (a)). For M0/M2 the open-loop replay from the
  run start (their h1 and h36 errors are the same quantity at different windows).
* Learning curves over ``n_train``: median and range over the 3 seeds.

Pre-registered reading (margins are the medians over the 3 seeds of the per-seed margins)
-----------------------------------------------------------------------------------------
Paired comparisons: ``evaluation.paired_resolution.paired_comparison`` over the master seeds
(t with 2 degrees of freedom, 95 %), candidate against reference, lower is better.

**Y0** -- hybrid recommended for few data: at EVERY ``n_train`` in {4, 8, 16} the margin of M3 is at
most half the margin of M1 (``median m_M3 <= 0.5 median m_M1``) AND M3 is resolvably better than M1
in the paired comparison of the margins at that ``n_train``. If the half-criterion holds everywhere
but a comparison is not resolved: "Y0 not resolved" (not Y0).

**Y1** -- a simple parameter refit suffices, no residual network needed: at EVERY ``n_train`` in
{4, 8, 16} both ``|median m_M2 - median m_M3| <= 0.10 median m_M3`` (absolute floor 1e-4 when
``median m_M3`` is 0) and ``|median R_M2 - median R_M3| <= 0.10 median R_M3`` with ``R`` the energy
relative to the oracle (median over the inside cases). Equivalence statement, no rank claim; the
paired differences are reported next to it. (Note fixed before the run: the energy part is weak,
since R is near 1; the excess over the oracle in percentage points is reported descriptively.)

**Y2** -- no advantage of the hybrid in this setting: at NO ``n_train`` in {4, 8, 16} is the margin of
M3 resolvably smaller than that of M1 (paired). Y0 and Y2 exclude each other; if neither holds
(M3 resolvably better somewhere but not by half everywhere), the reading is "partial advantage
(neither Y0 nor Y2)". Y1 is assessed independently of Y0/Y2.

**Extrapolation** (per model and ``n_train``): "margin holds outside" = yes iff the violation share at
the model's parameter, pooled over the 3 seeds, is <= 5 % -- separately for (a) and (b). Reported for
every model; not part of Y0-Y2.

**Discretisation** (addendum 1): if the largest relative energy difference of the oracle by grid +
zoom against the 8-start Nelder-Mead at the 10 check cases exceeds 0.2 %, that number is stated next
to every energy figure of the report.

All ``n_train`` = 32 values are reported (learning curves); the readings use ``n_train`` <= 16. If the
budget rule removes ``n_train`` 32 or M4, the reading is unchanged (it does not use them).

**Addendum 2026-11-05 (decision of the project lead; committed after the failed margins run of
a343840 and before any margin result exists).** A margin of a cell (model, master seed,
``n_train``) counts as *determinable* only if at least 20 inside cases are pooled over the 20 draws;
otherwise it is reported as "not determinable", not as a number. The number of inside cases is
reported for every cell. Y0 rests on the determinable cells with ``n_train`` <= 16: medians over the
determinable seeds; the paired comparison over the seeds determinable for both M1 and M3 (the
inside cases do not depend on the model, so these are the same seeds), with at least 2 seeds,
otherwise "not resolved" at that ``n_train``; an ``n_train`` without any determinable cell is reported
as such and drops out of Y0. Y1 and Y2 use the same determinable cells. A draw without any inside
case contributes nothing to the search (no search there); its cases are evaluated at the chosen
parameter like all others.

Implementation of the addendum: a cell with fewer than ``MIN_INSIDE`` pooled inside cases has
``determinable`` False and no margin, energy or holding share as a number; medians and paired
comparisons use the determinable cells only (paired: at least 2 seeds). The margins are read from
the cell records of ``<prefix>_margins_per_draw.jsonl`` (driver split per draw, same format).

    tools/run_here.sh scripts/analyze_hybrid_model.py --prefix hybrid_model_v1
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

ROOT = SCRIPTS.parent
RUNS = ROOT / "artifacts" / "runs"
MODELS = ("M0", "M1", "M2", "M3", "M4")
READING_N = (4, 8, 16)
HALF = 0.5
EQUIVALENCE = 0.10
ABSOLUTE_FLOOR = 1.0e-4
OUTSIDE_LIMIT = 0.05
DISCRETISATION_THRESHOLD = 0.002
MIN_INSIDE = 20  # addendum 2026-11-05: a margin is determinable from >= 20 pooled inside cases
MIN_PAIRED_SEEDS = 2


def jobs(path: Path, kind: str = "job") -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("kind") == kind:
                out.append(row)
    return out


def headers(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("kind") == "header"
    ]


def _median(values: list[float]) -> float | None:
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(finite)) if finite else None


def summarise_job(job: dict[str, Any]) -> dict[str, Any]:
    """Per (model, seed, n): margin, energy, outside violation, biases."""
    cases = job["cases"]
    inside = [c for c in cases if c["inside"]]
    out_b = [c for c in cases if c["kind"] == "interior" and not c["inside"]]
    out_a = [c for c in cases if c["kind"] == "a"]
    upper = job["search"]["upper"]
    determinable = job["n_inside"] >= MIN_INSIDE
    if not determinable:
        margin = None
    elif job["parameter_kind"] == "k":
        margin = _median([c["effective_margin"] for c in inside]) if job["reached"] else None
    else:
        margin = job["parameter"] if job["reached"] else None
    # cases without a valid model point give no solution: left out of the energy figures
    priced = [
        c
        for c in inside
        if c["predicted"]["valid"]
        and c["energy_excess_relative"] is not None
        and c["truth"]["energy_kw"] is not None
    ]
    r = [1.0 + c["energy_excess_relative"] for c in priced]
    pooled_true = sum(c["truth"]["energy_kw"] for c in priced)
    pooled_oracle = sum(c["oracle_energy_kw"] for c in priced)

    def share(rows: list[dict[str, Any]]) -> tuple[int, int]:
        return sum(not c["holds"] for c in rows), len(rows)

    def bias(kind_filter: Any) -> float | None:
        return _median([b["bias"] for b in job["coarse_bias"] if kind_filter(b)])

    return {
        "model": job["model"],
        "seed": job["master_seed"],
        "n_train": job["n_train"],
        "reached": job["reached"],
        "parameter": job["parameter"],
        "parameter_kind": job["parameter_kind"],
        "raised": job["raised"],
        "n_inside": job["n_inside"],
        "determinable": determinable,
        "margin": margin,
        "margin_censored": margin
        if margin is not None
        else upper
        if job["parameter_kind"] == "margin" and determinable
        else None,
        "holding_share_inside": float(np.mean([c["holds"] for c in inside]))
        if inside and determinable
        else None,
        "energy_relative_median": _median(r) if determinable else None,
        "energy_excess_pooled": (pooled_true / pooled_oracle - 1.0)
        if pooled_oracle and determinable
        else None,
        "violations_b": share(out_b),
        "violations_a": share(out_a),
        "bed_violations": sum(c["bed_violated"] for c in cases),
        "moisture_bias_at_action_inside": _median([c["moisture_bias_at_action"] for c in inside]),
        "moisture_bias_at_action_b": _median([c["moisture_bias_at_action"] for c in out_b]),
        "moisture_bias_at_action_a": _median([c["moisture_bias_at_action"] for c in out_a]),
        "coarse_bias_inside": bias(lambda b: b["inside"]),
        "coarse_bias_b": bias(lambda b: b["kind"] == "interior" and not b["inside"]),
        "coarse_bias_a": bias(lambda b: b["kind"] == "a"),
        "model_invalid_at_action": sum(not c["predicted"]["valid"] for c in cases),
        "model_invalid_inside": sum(not c["predicted"]["valid"] for c in inside),
    }


def curve(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [r[key] for r in rows]
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return {
        "median": float(np.median(finite)) if finite else None,
        "min": float(np.min(finite)) if finite else None,
        "max": float(np.max(finite)) if finite else None,
        "n_seeds": len(finite),
        "per_seed": {r["seed"]: r[key] for r in rows},
    }


def paired(cand: dict[int, float], ref: dict[int, float]) -> dict[str, Any] | None:
    from twinlink_ml.evaluation.paired_resolution import paired_comparison

    if len(cand) < MIN_PAIRED_SEEDS or set(cand) != set(ref):
        return None
    if any(v is None for v in [*cand.values(), *ref.values()]):
        return None
    c = paired_comparison(cand, ref, candidate_label="candidate", reference_label="reference")
    resolved = (not c.degenerate) and abs(c.mean_difference) > c.threshold
    return {
        "mean_difference_ref_minus_cand": c.mean_difference,
        "threshold": c.threshold,
        "p_value": c.p_value,
        "degenerate": c.degenerate,
        "resolved": bool(resolved),
        "candidate_better": bool(resolved and c.mean_difference > 0),
        "candidate_worse": bool(resolved and c.mean_difference < 0),
    }


def reading(table: dict[tuple[str, int], list[dict[str, Any]]]) -> dict[str, Any]:
    # addendum 2026-11-05: determinable cells only
    def cells(model: str, n: int) -> list[dict[str, Any]]:
        return [r for r in table.get((model, n), []) if r["determinable"]]

    def per_seed(model: str, n: int, key: str) -> dict[int, float]:
        return {r["seed"]: r[key] for r in cells(model, n)}

    def med(model: str, n: int, key: str) -> float | None:
        return curve(cells(model, n), key)["median"] if cells(model, n) else None

    y0_half, y0_resolved, y2_better, y1 = {}, {}, {}, {}
    for n in READING_N:
        m1, m3, m2 = (
            med("M1", n, "margin_censored"),
            med("M3", n, "margin_censored"),
            med("M2", n, "margin_censored"),
        )
        y0_half[n] = None if (m1 is None or m3 is None) else bool(m3 <= HALF * m1)
        comp = paired(per_seed("M3", n, "margin_censored"), per_seed("M1", n, "margin_censored"))
        y0_resolved[n] = comp
        y2_better[n] = bool(comp and comp["candidate_better"])
        r2, r3 = med("M2", n, "energy_relative_median"), med("M3", n, "energy_relative_median")
        if None in (m2, m3, r2, r3):
            y1[n] = None
        else:
            y1[n] = {
                "margin_ok": bool(abs(m2 - m3) <= max(EQUIVALENCE * m3, ABSOLUTE_FLOOR)),
                "energy_ok": bool(abs(r2 - r3) <= EQUIVALENCE * r3),
                "margin_m2": m2,
                "margin_m3": m3,
                "energy_relative_m2": r2,
                "energy_relative_m3": r3,
                "excess_difference_pp": 100.0 * (r2 - r3),
                "paired_margin_m2_vs_m3": paired(
                    per_seed("M2", n, "margin_censored"), per_seed("M3", n, "margin_censored")
                ),
            }
    # an n_train without any determinable cell drops out (reported as such)
    assessable = [n for n in READING_N if y0_half[n] is not None]
    y0_half = {n: y0_half[n] for n in assessable}
    y0_resolved = {n: y0_resolved[n] for n in assessable}
    y2_better = {n: y2_better[n] for n in assessable}
    complete = bool(assessable)
    y0 = (
        complete
        and all(y0_half.values())
        and all(bool(c and c["candidate_better"]) for c in y0_resolved.values())
    )
    y0_unresolved = complete and all(y0_half.values()) and not y0
    y2 = complete and not any(y2_better.values())
    y1_holds = all(v is not None and v["margin_ok"] and v["energy_ok"] for v in y1.values())
    if not complete:
        verdict = "not determinable (no n_train <= 16 with a determinable cell)"
    elif y0:
        verdict = "Y0"
    elif y2:
        verdict = "Y2"
    elif y0_unresolved:
        verdict = "Y0 not resolved"
    else:
        verdict = "partial advantage (neither Y0 nor Y2)"
    return {
        "verdict_y0_y2": verdict,
        "n_train_assessed": assessable,
        "n_train_not_determinable": [n for n in READING_N if n not in assessable],
        "y1": bool(y1_holds),
        "y0_half_criterion": y0_half,
        "y0_paired_m3_vs_m1": y0_resolved,
        "y2_m3_resolved_better": y2_better,
        "y1_detail": y1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefix", default="hybrid_model_v1")
    parser.add_argument("--training", default="hybrid_model_training_v1")
    parser.add_argument("--margins", default=None, help="default <prefix>_margins_per_draw")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from _common import provenance, require_clean_tree
    from run_hybrid_evaluation import clean_tree_exceptions

    output = RUNS / f"{args.output or args.prefix + '_analysis'}.json"
    if output.exists():
        raise SystemExit(f"{output} exists -- artifacts are never overwritten")
    # the uncommitted training records are the only input exempt here (not read by the analysis)
    exempt = clean_tree_exceptions(RUNS, output, args.training, None)
    require_clean_tree(ignore_paths=exempt)
    margins_name = args.margins or f"{args.prefix}_margins_per_draw"
    margins_path = RUNS / f"{margins_name}.jsonl"
    margins = jobs(margins_path, kind="cell")
    oracle_jobs = jobs(RUNS / f"{args.prefix}_oracle.jsonl")
    errors_path = RUNS / f"{args.prefix}_errors.jsonl"
    errors = jobs(errors_path) if errors_path.exists() else []

    rows = [summarise_job(j) for j in margins]
    table: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        table[(r["model"], r["n_train"])].append(r)
    curves = {
        f"{m}/n{n}": {
            key: curve(v, key)
            for key in (
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
        "per_case": [
            {
                k: j[k]
                for k in (
                    "draw",
                    "slot",
                    "energy_difference_relative",
                    "objective_difference_relative",
                    "slack_difference_kg_per_kg",
                )
            }
            for j in disc
        ],
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
    payload = {
        "provenance": provenance("not used", ignore_paths=exempt),
        "inputs": {
            "margins": margins_path.name,
            "oracle": f"{args.prefix}_oracle.jsonl",
            "errors": f"{args.prefix}_errors.jsonl" if errors else None,
            "margin_headers": [h["provenance"] for h in headers(margins_path)],
        },
        "inside_cases_per_cell": {
            f"{r['model']}/s{r['seed']}/n{r['n_train']}": {
                "n_inside": r["n_inside"],
                "determinable": r["determinable"],
            }
            for r in rows
        },
        "reading": reading(table),
        "outside": outside,
        "discretisation": discretisation,
        "curves": curves,
        "per_job": rows,
        "errors": error_summary,
    }
    output.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    print(json.dumps(payload["reading"], indent=1, default=float)[:4000])
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

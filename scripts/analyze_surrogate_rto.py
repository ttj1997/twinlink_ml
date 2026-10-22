"""Analysis of the surrogate-RTO study (docs/25): learning curves, pre-registered reading.

Input: the artifact of ``scripts/run_surrogate_rto.py`` (``artifacts/runs/<name>.json.gz``).
Output: tables on stdout, ``artifacts/runs/<name>_analysis.json`` and the figures
``docs/figures/<name>_<problem>_{eta,regret,infeasible}.png`` (never overwritten).

Test set: the 50 test states of each problem (Sobol over the disturbance box, fixed seed, an own
sequence disjoint from every training design). There is no other test set in this study.

Metrics per (problem, model, kappa, N, seed) on the 50 test states, all at the TRUE model
-----------------------------------------------------------------------------------------
* feasible state: every true constraint ``g_j(u_hat, w) <= tol_j`` (tolerances of the YAML).
* infeasibility rate: share of the 50 states that are not feasible; maximum violation: largest
  ``max_j g_j / scale_j`` over the states (normalised; also per constraint in its unit).
* regret ``R = (E(u_hat) - E(u*)) / E(u*)`` per feasible state; per seed the MEAN over the
  feasible states (median over states reported as well).
* pooled regret ``R_pool = sum_f (E(u_hat) - E(u*)) / sum_f E(u*)`` over the feasible states.
* captured share ``eta = sum_f (E(u0) - E(u_hat)) / sum_f (E(u0) - E(u*))`` over the feasible
  states ``f`` (headroom-weighted, so that states where ``u0`` is already near ``u*`` do not
  dominate). ``u0`` is the robust fixed point (least energy at the nominal disturbance subject to
  feasibility in every test state; decision of the project lead, 2026-10-22). It is determined on
  the 50 test states and knows their disturbance envelope: this favours the baseline, ``eta`` is
  conservative.
* headroom ``H = sum (E(u0) - E(u*)) / sum E(u*)`` per problem. With ``H_f`` the same over the
  feasible states, ``eta >= 0.9`` is exactly ``R_pool <= 0.1 H_f``: the reading is also given in
  regret percent. With a small headroom ``eta`` reacts strongly to small regrets, so regret and
  ``eta`` are always reported side by side.
* secondary (report only): ``eta`` against the R0-style fixed point (one normalised margin on the
  constraints active at the nominal optimum, bisected).
* conservative captured share ``eta_c``: as ``eta``, but the denominator runs over ALL states (an
  infeasible state counts as zero captured headroom). Used for every paired comparison.
* Per (problem, model, kappa, N): median and range over the 5 seeds.

Pre-registered reading (fixed BEFORE the first run; applied to each problem separately)
------------------------------------------------------------------------------------------
A variant (model and kappa; M1-M3 with kappa 0/1/2, M4, M2a with kappa 0/1/2) *succeeds at N*
when the median over seeds of ``eta`` is >= 0.9 AND the median over seeds of the infeasibility
rate is <= 0.05. ``N_min`` of a variant is the smallest budget from which on it succeeds at that
budget and at every larger budget of the grid (8, 16, 32, 64, 128, 256); the first budget at
which it succeeds is reported too.

S1  some variant has ``N_min <= 32``
    -> steady RTO with a surrogate is data-efficient.
S2  the smallest ``N_min`` over all variants lies in 64 <= N <= 256
    -> feasible, but costly with an expensive Dyssol model; adaptive design is decisive if it
    lowers N clearly: the best ``N_min`` of M2a is at least one budget step (factor 2) below the
    best ``N_min`` of the Sobol variants (M1-M4).
S3  no variant succeeds at N = 256
    -> physics-based RTO needed.

(If the smallest ``N_min`` is <= 32, S1 applies; the three cases are exhaustive. A variant that
succeeds at 256 but not from some N on is covered by S2 via N_min = 256.)

Additional reports (no gate)
----------------------------
* M4 against the black-box models: at each N, a paired t-test over the 5 seeds
  (``twinlink_ml.evaluation.paired_resolution.paired_comparison``, metric ``1 - eta_c``, lower is
  better) of every black-box variant against M4. M4 is *overtaken at N* when at least one
  black-box variant is better and the difference is resolved (p < 0.05). Reported: the smallest
  such N, and whether M4 stays overtaken at every larger N. Number of tests reported, no
  multiplicity correction. **The error floor of M4 and the overtake point hold for the shift of
  the hidden parameter of this study (P1: exchanger approach 10 -> 20 K; P2: GAB monolayer
  factor 1.0 -> 1.2), not in general.**
* Adaptive against Sobol: M2a against M2 at the same kappa and N, same paired test.
* Effect of kappa: eta, infeasibility rate and maximum violation per kappa.
* Run time: seconds per true evaluation against seconds per surrogate prediction.
* Side metrics at the points found: CO2 (0.366 electricity, 0.201 gas; reported, not optimised);
  P1 condensate in the recovery exchanger.
* Reference quality: ``u*`` convergence (starts at the best value, check run) and the margin of
  ``u0``.

Addendum (2026-10-22, written after run v1, before run v2): POST HOC, DECLARED variant M4b
------------------------------------------------------------------------------------------------
Added after the results of ``surrogate_rto_v1`` were known (there M4 is infeasible on P2 by about
0.001 kg/kg). It is reported separately as "post hoc, declared"; **the pre-registered reading
above, computed from the v1 variants only, is the main result of the study and is not
rewritten.**

* Rule (decision of the project lead, 2026-10-22): M4 unchanged (same fit, same optimiser), plus a
  back-off per constraint ``b_N = kappa * RMS(g_true - g_model)``, the RMS over the N training runs
  at the fitted parameters, ``kappa in {1, 2}``; the fitted model is optimised with
  ``g_model + b_N <= 0``. Only the N training runs enter ``b_N``; no information from the test
  states, no second rule. Applied to both problems and every constraint.
* Same N, seeds, test states, metrics and the same reading S1-S3 (gate per kappa: median ``eta``
  >= 0.9 and median infeasibility <= 5 %, ``N_min`` as above), applied to M4b on its own.
* Kappa 0 of the M4b arm reproduces M4 of v1 (a check, not a variant).
* Reported in addition, descriptive only, without effect on the rule: ``b_N`` per N against the
  actual deviation ``g_true - g_model`` of the fitted model at its own decisions on the test
  states (per constraint, median and maximum over the states; for P2 the moisture). If ``b_N``
  underestimates the error at the limit, that is a result, not a reason for another variant.
* Holds for the hidden shift of this study only.
* Input: the v1 artifact plus the v2 artifact via ``--extra``; output named by ``--name``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import provenance  # noqa: E402

from twinlink_ml.evaluation.paired_resolution import paired_comparison  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"
FIG_DIR = REPO_ROOT / "docs" / "figures"

ETA_GATE = 0.9
INFEASIBLE_GATE = 0.05
SMALL_N = 32
CONFIDENCE = 0.95


def _variant(model: str, kappa: float) -> str:
    return "M4" if model == "M4" else f"{model}/k{kappa:g}"


def _posthoc(var: str) -> bool:
    """Variants added after run v1 (post hoc, declared; never part of the main reading)."""
    return var.startswith("M4b")


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def reference_of(art: dict[str, Any], problem: str) -> dict[str, Any]:
    rec = next(r for r in art["records"] if r["task"] == ["reference", problem])
    ref = rec["result"]
    star, u0 = ref["score_star"], ref["score_u0"]
    return {
        "e_star": np.array(star["energy"]),
        "star_feasible": np.array(star["feasible"]),
        "e0": np.array(u0["energy"]),
        "e0_margin": np.array(ref["score_u0_margin"]["energy"]),
        "u0_margin": ref["u0_margin"],
        "u0_margin_feasible": np.array(ref["score_u0_margin"]["feasible"]),
        "margin_constraints": ref["margin_constraints"],
        "u0_feasible": np.array(u0["feasible"]),
        "u0": ref["u0"],
        "margin": ref["margin"],
        "margin_converged": ref["margin_converged"],
        "star": ref["star"],
        "star_check": ref["star_check"],
        "star_source": ref["star_source"],
        "star_side": star["side"],
        "u0_side": u0["side"],
        "seconds_star": ref["seconds_star"],
    }


def metrics(row: dict[str, Any], ref: dict[str, Any], scales: np.ndarray) -> dict[str, float]:
    s = row["score"]
    e = np.array(s["energy"])
    feas = np.array(s["feasible"])
    g = np.array(s["constraints"])
    valid = ref["star_feasible"]
    f = feas & valid
    head = ref["e0"] - ref["e_star"]
    regret = (e - ref["e_star"]) / ref["e_star"]
    viol = np.max(g / scales, axis=1)
    head_m = ref["e0_margin"] - ref["e_star"]
    nan = float("nan")
    return {
        "eta": float(np.sum(ref["e0"][f] - e[f]) / np.sum(head[f])) if f.any() else nan,
        "regret_pooled": (
            float(np.sum(e[f] - ref["e_star"][f]) / np.sum(ref["e_star"][f])) if f.any() else nan
        ),
        "headroom_feasible": (
            float(np.sum(head[f]) / np.sum(ref["e_star"][f])) if f.any() else nan
        ),
        "eta_r0_margin": (
            float(np.sum(ref["e0_margin"][f] - e[f]) / np.sum(head_m[f])) if f.any() else nan
        ),
        "eta_c": float(np.sum(ref["e0"][f] - e[f]) / np.sum(head[valid])),
        "regret_mean": float(np.mean(regret[f])) if f.any() else float("nan"),
        "regret_median": float(np.median(regret[f])) if f.any() else float("nan"),
        "infeasible": float(np.mean(~feas[valid])),
        "max_violation": float(max(np.max(viol[valid]), 0.0)),
        "co2_median": float(np.median([d["co2_kg_per_h"] for d in s["side"]])),
        "condensate_median": float(
            np.median([d.get("condensate_kg_per_h", np.nan) for d in s["side"]])
        ),
    }


def summarise(art: dict[str, Any], problem: str) -> dict[str, Any]:
    ref = reference_of(art, problem)
    cons = art["problems"][problem]["constraints"]
    scales = np.array([c["scale"] for c in cons])
    per: dict[tuple[str, int], dict[int, dict[str, float]]] = defaultdict(dict)
    times: dict[str, list[float]] = defaultdict(list)
    backoff: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for rec in art["records"]:
        if rec["task"][0] == "reference" or rec["task"][1] != problem:
            continue
        for row in rec["result"]:
            var = _variant(row["model"], row["kappa"])
            m = metrics(row, ref, scales)
            m["true_evaluations"] = row["true_evaluations"]
            per[(var, row["N"])][row["seed"]] = m
            if "backoff" in row:
                dev = np.array(row["score"]["constraints"]) - np.array(
                    row["model_constraints_at_decisions"]
                )
                bn = np.array(row["backoff"])
                backoff[(var, row["N"])].append(
                    {
                        "seed": row["seed"],
                        "backoff": bn,
                        "dev_max": dev.max(axis=0),
                        "dev_median": np.median(dev, axis=0),
                        "share_dev_above_backoff": np.mean(dev > bn, axis=0),
                    }
                )
            times["truth_s_per_eval"].append(row["truth_s_per_eval"])
            if "predict_single_s" in row:
                times[f"{row['model']}_predict_single_s"].append(row["predict_single_s"])
            if "predict_batch_s_per_point" in row:
                times[f"{row['model']}_predict_batch_s"].append(row["predict_batch_s_per_point"])
            times[f"{row['model']}_fit_s"].append(row["fit_s"])
    budgets = sorted({n for _, n in per})
    variants = sorted({v for v, _ in per})
    table: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for (var, n), seeds in per.items():
        agg: dict[str, Any] = {"seeds": len(seeds)}
        for key in next(iter(seeds.values())):
            vals = np.array([seeds[s][key] for s in sorted(seeds)], dtype=float)
            agg[key] = {
                "median": float(np.nanmedian(vals)) if np.isfinite(vals).any() else float("nan"),
                "min": float(np.nanmin(vals)) if np.isfinite(vals).any() else float("nan"),
                "max": float(np.nanmax(vals)) if np.isfinite(vals).any() else float("nan"),
                "per_seed": vals.tolist(),
            }
        agg["success"] = bool(
            agg["eta"]["median"] >= ETA_GATE and agg["infeasible"]["median"] <= INFEASIBLE_GATE
        )
        table[var][n] = agg
    n_min, n_first = {}, {}
    for var in variants:
        ok = [table[var][n]["success"] for n in budgets]
        n_first[var] = next((n for n, o in zip(budgets, ok, strict=True) if o), None)
        n_min[var] = next((n for i, n in enumerate(budgets) if all(ok[i:])), None)
    pre = [v for v in variants if not _posthoc(v)]
    reading = _reading({v: n_min[v] for v in pre}, budgets)
    overtake = _overtake(per, pre, budgets)
    adaptive = _adaptive(per, pre, budgets)
    posthoc = _posthoc_report(per, variants, budgets, n_min, n_first, backoff, cons)
    return {
        "reference": _reference_report(ref),
        "budgets": budgets,
        "table": table,
        "n_min": n_min,
        "n_first": n_first,
        "reading": reading,
        "overtake": overtake,
        "adaptive": adaptive,
        "posthoc": posthoc,
        "times": {k: {"median": float(np.median(v)), "n": len(v)} for k, v in times.items()},
    }


def _posthoc_report(
    per: Any,
    variants: list[str],
    budgets: list[int],
    n_min: dict[str, int | None],
    n_first: dict[str, int | None],
    backoff: dict[tuple[str, int], list[dict[str, Any]]],
    cons: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """M4b (post hoc, declared): gate per kappa, reproduction of M4, b_N against the deviation."""
    post = [v for v in variants if _posthoc(v) and not v.endswith("/k0")]
    if not post:
        return None
    reproduction = None
    if ("M4b/k0", budgets[0]) in per:
        reproduction = float(
            max(
                abs(per[("M4b/k0", n)][s]["eta_c"] - per[("M4", n)][s]["eta_c"])
                for n in budgets
                for s in per[("M4", n)]
            )
        )
    tests = []
    for v in post:
        for n in budgets:
            t = _paired(per, v, "M4", n)
            if t is not None:
                tests.append({"N": n, **t})
    report = {}
    for (v, n), rows in sorted(backoff.items()):
        report[f"{v}@{n}"] = {
            key: np.median([r[key] for r in rows], axis=0).tolist()
            for key in ("backoff", "dev_max", "dev_median", "share_dev_above_backoff")
        }
    return {
        "label": "post hoc, declared",
        "n_min": {v: n_min[v] for v in post},
        "n_first": {v: n_first[v] for v in post},
        "case_if_alone": _reading({v: n_min[v] for v in post}, budgets)["case"],
        "reproduction_k0_vs_M4_max_abs_eta_c": reproduction,
        "vs_M4": tests,
        "backoff": report,
        "constraints": [c["name"] for c in cons],
    }


def _reading(n_min: dict[str, int | None], budgets: list[int]) -> dict[str, Any]:
    found = {v: n for v, n in n_min.items() if n is not None}
    best = min(found.values()) if found else None
    sobol = [n for v, n in found.items() if not v.startswith("M2a")]
    adapt = [n for v, n in found.items() if v.startswith("M2a")]
    best_sobol, best_adapt = (min(sobol) if sobol else None), (min(adapt) if adapt else None)
    if best is not None and best <= SMALL_N:
        case = "S1"
    elif best is not None:
        case = "S2"
    else:
        case = "S3"
    decisive = (
        best_adapt is not None
        and (best_sobol is None or best_adapt * 2 <= best_sobol)
        and case == "S2"
    )
    return {
        "case": case,
        "best_n_min": best,
        "best_variants": sorted(v for v, n in found.items() if n == best),
        "best_n_min_sobol": best_sobol,
        "best_n_min_adaptive": best_adapt,
        "adaptive_decisive": bool(decisive),
        "budgets": budgets,
    }


def _paired(
    per: dict[tuple[str, int], dict[int, dict[str, float]]], cand: str, ref: str, n: int
) -> dict[str, Any] | None:
    a, b = per.get((cand, n)), per.get((ref, n))
    if not a or not b or set(a) != set(b):
        return None
    c = paired_comparison(
        {s: 1.0 - a[s]["eta_c"] for s in a},
        {s: 1.0 - b[s]["eta_c"] for s in b},
        candidate_label=cand,
        reference_label=ref,
        confidence=CONFIDENCE,
    )
    return {
        "candidate": cand,
        "reference": ref,
        "mean_difference": c.mean_difference,
        "threshold": c.threshold,
        "p_value": c.p_value,
        "degenerate": c.degenerate,
        "resolved_better": bool(
            not c.degenerate and c.mean_difference > 0.0 and c.p_value < 1.0 - CONFIDENCE
        ),
        "resolved_worse": bool(
            not c.degenerate and c.mean_difference < 0.0 and c.p_value < 1.0 - CONFIDENCE
        ),
    }


def _overtake(per: Any, variants: list[str], budgets: list[int]) -> dict[str, Any]:
    black = [v for v in variants if v.split("/")[0] in ("M1", "M2", "M3", "M2a")]
    tests, first = [], None
    by_n: dict[int, list[str]] = {}
    for n in budgets:
        better = []
        for v in black:
            t = _paired(per, v, "M4", n)
            if t is None:
                continue
            tests.append({"N": n, **t})
            if t["resolved_better"]:
                better.append(v)
        by_n[n] = better
        if better and first is None:
            first = n
    stays = first is not None and all(by_n[n] for n in budgets if n >= first)
    return {"first_n": first, "stays_overtaken": stays, "by_n": by_n, "tests": tests}


def _adaptive(per: Any, variants: list[str], budgets: list[int]) -> list[dict[str, Any]]:
    out = []
    for v in variants:
        if not v.startswith("M2a/"):
            continue
        ref = "M2/" + v.split("/")[1]
        for n in budgets:
            t = _paired(per, v, ref, n)
            if t is not None:
                out.append({"N": n, **t})
    return out


def _reference_report(ref: dict[str, Any]) -> dict[str, Any]:
    star, check = ref["star"], ref["star_check"]
    e_check = np.array([c["energy"] for c in check], dtype=float)
    e_main = np.array([s["energy"] for s in star], dtype=float)
    rel = (e_check - e_main) / e_main
    return {
        "u_star_feasible": int(np.sum(ref["star_feasible"])),
        "states": len(e_main),
        "starts": [s["starts"] for s in star][:1],
        "at_best_min": int(min(s["at_best"] for s in star)),
        "at_best_median": float(np.median([s["at_best"] for s in star])),
        "check_rel_diff_min": float(np.nanmin(rel)),
        "check_rel_diff_max": float(np.nanmax(rel)),
        "check_better_states": int(np.sum(rel < -1e-6)),
        "u_star_from_check": int(sum(x == "check" for x in ref["star_source"])),
        "u0": ref["u0"],
        "margin": ref["margin"],
        "margin_converged": ref["margin_converged"],
        "u0_feasible": int(np.sum(ref["u0_feasible"])),
        "headroom": float(np.sum(ref["e0"] - ref["e_star"]) / np.sum(ref["e_star"])),
        "headroom_state_min": float(np.min((ref["e0"] - ref["e_star"]) / ref["e_star"])),
        "headroom_state_max": float(np.max((ref["e0"] - ref["e_star"]) / ref["e_star"])),
        "regret_gate_equivalent": float(
            (1.0 - ETA_GATE) * np.sum(ref["e0"] - ref["e_star"]) / np.sum(ref["e_star"])
        ),
        "u0_margin": ref["u0_margin"],
        "margin_constraints": ref["margin_constraints"],
        "u0_margin_feasible": int(np.sum(ref["u0_margin_feasible"])),
        "headroom_r0_margin": float(
            np.sum(ref["e0_margin"] - ref["e_star"]) / np.sum(ref["e_star"])
        ),
        "e_star_median": float(np.median(ref["e_star"])),
        "e0_median": float(np.median(ref["e0"])),
        "headroom_median_rel": float(np.median((ref["e0"] - ref["e_star"]) / ref["e0"])),
        "co2_star_median": float(np.median([d["co2_kg_per_h"] for d in ref["star_side"]])),
        "co2_u0_median": float(np.median([d["co2_kg_per_h"] for d in ref["u0_side"]])),
        "condensate_star_median": float(
            np.median([d.get("condensate_kg_per_h", np.nan) for d in ref["star_side"]])
        ),
        "condensate_star_max": float(
            np.max([d.get("condensate_kg_per_h", np.nan) for d in ref["star_side"]])
        ),
        "seconds_star": ref["seconds_star"],
    }


def _print(problem: str, s: dict[str, Any]) -> None:
    r = s["reference"]
    print(f"\n=== {problem}: test set = 50 test states (Sobol, own sequence) ===")
    print(
        f"u*: {r['u_star_feasible']}/{r['states']} feasible, starts at best min "
        f"{r['at_best_min']} median {r['at_best_median']:.0f}, check run rel. diff "
        f"[{r['check_rel_diff_min']:.2e}, {r['check_rel_diff_max']:.2e}] "
        f"(check better in {r['check_better_states']}, u* taken from it in "
        f"{r['u_star_from_check']})"
    )
    print(
        f"u0 (robust) = {np.round(r['u0'], 3).tolist()}, feasible {r['u0_feasible']}/{r['states']}; "
        f"median E* {r['e_star_median']:.4f} kW, E0 {r['e0_median']:.4f} kW"
    )
    print(
        f"headroom H = {100 * r['headroom']:.2f} % of E* (states {100 * r['headroom_state_min']:.2f}"
        f" to {100 * r['headroom_state_max']:.2f} %); eta >= {ETA_GATE} <=> pooled regret <= "
        f"{100 * r['regret_gate_equivalent']:.3f} % (all states feasible)"
    )
    print(
        f"R0-style u0 = {np.round(r['u0_margin'], 3).tolist()} margin {r['margin']:.4f} on "
        f"{r['margin_constraints']} (converged {r['margin_converged']}), feasible "
        f"{r['u0_margin_feasible']}/{r['states']}, headroom {100 * r['headroom_r0_margin']:.2f} %"
    )
    for key, label in (
        ("eta", "eta"),
        ("regret_pooled", "pooled regret"),
        ("regret_mean", "regret (mean feasible)"),
        ("eta_r0_margin", "eta against the R0-style u0 (secondary)"),
        ("infeasible", "infeasible"),
        ("max_violation", "max viol"),
    ):
        print(f"\n{label}: median [min, max] over seeds")
        print(f"{'variant':10s} " + " ".join(f"{n:>22d}" for n in s["budgets"]))
        for var, rows in sorted(s["table"].items()):
            cells = []
            for n in s["budgets"]:
                a = rows.get(n, {}).get(key)
                cells.append(
                    "-" if a is None else f"{a['median']:.3f} [{a['min']:.3f},{a['max']:.3f}]"
                )
            print(f"{var:10s} " + " ".join(f"{c:>22s}" for c in cells))
    print("\nN_min (first success):")
    for v in sorted(s["n_min"]):
        print(f"  {v:10s} {s['n_min'][v]} ({s['n_first'][v]})")
    print(f"reading: {s['reading']}")
    print(
        f"  in regret percent: success <=> median pooled regret <= 0.1 H_f (all feasible: "
        f"{100 * r['regret_gate_equivalent']:.3f} %) and median infeasibility <= "
        f"{100 * INFEASIBLE_GATE:.0f} %"
    )
    o = s["overtake"]
    print(
        f"M4 overtaken first at N = {o['first_n']} (stays: {o['stays_overtaken']}); "
        f"{len(o['tests'])} paired tests"
    )
    print(f"  by N: {o['by_n']}")
    for a in s["adaptive"]:
        print(
            f"  adaptive {a['candidate']} vs {a['reference']} N={a['N']}: "
            f"d={a['mean_difference']:+.4f} p={a['p_value']:.3f} "
            f"better={a['resolved_better']} worse={a['resolved_worse']}"
        )
    ph = s.get("posthoc")
    if ph:
        print(f"\n--- M4b ({ph['label']}; not part of the reading above) ---")
        for v in ph["n_min"]:
            print(f"  {v}: N_min {ph['n_min'][v]} (first success {ph['n_first'][v]})")
        print(f"  case if M4b alone: {ph['case_if_alone']}")
        print(f"  kappa 0 vs M4, max |d eta_c|: {ph['reproduction_k0_vs_M4_max_abs_eta_c']}")
        for t in ph["vs_M4"]:
            print(
                f"  {t['candidate']} vs M4 N={t['N']}: d={t['mean_difference']:+.4f} "
                f"p={t['p_value']:.3f} better={t['resolved_better']} worse={t['resolved_worse']}"
            )
        print(
            f"  b_N against g_true - g_model at the decisions (median over seeds), {ph['constraints']}:"
        )
        for key, rep in ph["backoff"].items():
            print(
                f"  {key}: b_N {np.round(rep['backoff'], 5).tolist()} | dev median "
                f"{np.round(rep['dev_median'], 5).tolist()} max {np.round(rep['dev_max'], 5).tolist()}"
                f" | share dev > b_N {np.round(rep['share_dev_above_backoff'], 2).tolist()}"
            )
    print("times (median):", {k: f"{v['median']:.2e}" for k, v in sorted(s["times"].items())})


def _figures(name: str, problem: str, s: dict[str, Any]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for key, ylabel, gate in (
        ("eta", "captured share eta (axis cut at -0.5)", ETA_GATE),
        ("regret_mean", "regret (mean over feasible states)", None),
        ("infeasible", "infeasibility rate", INFEASIBLE_GATE),
    ):
        path = FIG_DIR / f"{name}_{problem}_{key}.png"
        if path.exists():
            raise SystemExit(f"{path} exists -- figures are never overwritten")
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for ax, kappa in zip(axes, ("k0", "k1", "k2"), strict=True):
            for var, rows in sorted(s["table"].items()):
                if var == "M4b/k0" or (var != "M4" and not var.endswith(kappa)):
                    continue
                ns = [n for n in s["budgets"] if n in rows]
                med = [rows[n][key]["median"] for n in ns]
                lo = [rows[n][key]["min"] for n in ns]
                hi = [rows[n][key]["max"] for n in ns]
                (line,) = ax.plot(ns, med, marker="o", label=var)
                ax.fill_between(ns, lo, hi, alpha=0.15, color=line.get_color())
            if gate is not None:
                ax.axhline(gate, color="grey", ls="--", lw=1)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("true evaluations N")
            ax.set_title(f"{problem}, kappa = {kappa[1:]}")
            if key == "regret_mean":
                ax.set_yscale("symlog", linthresh=1e-3)
            if key == "eta":
                ax.set_ylim(-0.5, 1.05)  # far negative values at small N would hide the gate
            ax.legend(fontsize=7)
        axes[0].set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path.relative_to(REPO_ROOT)))
    return written


def _check_compatible(art: dict[str, Any], extra: dict[str, Any]) -> None:
    """Same settings apart from the block of the added arm (test states, designs, problems)."""

    def strip(cfg: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = json.loads(json.dumps(cfg))
        out["models"].pop("m4b", None)
        return out

    if strip(art["config"]) != strip(extra["config"]):
        raise SystemExit("the extra artifact was run with other settings")
    if any(r["task"][0] == "reference" for r in extra["records"]):
        raise SystemExit("the extra artifact must not carry its own reference")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("artifact", type=Path)
    ap.add_argument("--no-write", action="store_true", help="print only (smoke runs)")
    ap.add_argument("--extra", type=Path, nargs="*", default=[], help="artifacts of added arms")
    ap.add_argument("--name", help="output name (default: the artifact's)")
    args = ap.parse_args()
    art = load(args.artifact)
    name = args.name or args.artifact.name.removesuffix(".json.gz")
    extras = []
    for path in args.extra:
        extra = load(path)
        _check_compatible(art, extra)
        art["records"] = art["records"] + extra["records"]
        extras.append(
            {
                "artifact": path.name,
                "provenance": extra["provenance"],
                "config_fingerprint": extra["config_fingerprint"],
            }
        )
    out: dict[str, Any] = {
        "provenance": None if args.no_write else provenance(),
        "artifact": args.artifact.name,
        "provenance_of_artifact": art["provenance"],
        "config_fingerprint": art["config_fingerprint"],
        "extra_artifacts": extras,
        "problems": {},
    }
    for problem in sorted(art["problems"]):
        s = summarise(art, problem)
        _print(problem, s)
        out["problems"][problem] = s
        if not args.no_write:
            s["figures"] = _figures(name, problem, s)
    if args.no_write:
        return
    path = RUNS_DIR / f"{name}_analysis.json"
    if path.exists():
        sys.exit(f"{path} exists -- artifacts are never overwritten")
    path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

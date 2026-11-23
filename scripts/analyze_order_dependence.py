"""Analysis of the order-dependence measurement and its PRE-REGISTERED reading.

Study document: ``docs/46_order_dependence.md``. Input: ``artifacts/runs/<name>.json.gz`` written
by ``scripts/run_order_dependence.py --stage main``, the artifacts of the findings
(``dynamic_headroom_v2.json.gz``, ``load_following_v1.json.gz``) and, if given, the two artifacts
of the learned-setpoints study (branch ``learned-setpoints``, not on master: passed as paths,
recorded with their SHA-256). Output: ``artifacts/runs/<name>_analysis.json`` and a report on
stdout.

PRE-REGISTERED READING (committed before the dry run, the probe and the main run; task of the
project lead of 2026-11-20, definitions of this session where the task leaves room are marked
"(session)")
------------------------------------------------------------------------------------------------
Question: do the key numbers that rest on the steady-state optimiser change when its order
dependence is removed? Removed means: every steady optimum from the cold deterministic
multi-start (8 starts, fixed seed; switch ``cold_starts`` of ``SteadyStateOptimizer``) with
unrounded cache keys. Only the reference controllers are recomputed (R0, R1; C0, C1, C2); no
controller with a dynamic optimiser, no learned arm, no part of stage 0.

Key numbers. "Old" is computed from the records of the artifact of the finding with the function
that is applied to the recomputed records (``twinlink_ml.order_dependence.keynumbers``); the
value printed in the finding is given in brackets:

* **K1**, Befund 105: median over the 105 scenarios of the family "observed" of
  ``(E_R0 - E_R1) / E_R0``, each controller at its own margin (8.93 %).
* **K2**, Befund 126: ``(J_C0 - J_C1) / J_C0`` (2.07 %); **K3**: ``(J_C0 - J_C2) / J_C0``
  (2.06 %); ``J`` = cost per kg of dry product over the 40 test days, variant E, main band, true
  prices, the tail of every run at the margin of C0.
* **K4**, learned set points: the cost of the twin, ``J_C2`` at the margin of C2 with the tail at
  the margin of C0 (0.39794 EUR/kg). **K5** (session): the smallest median gap of a learned arm
  against the twin that the study reports (D-gru at the largest budget, 2.98 %), recomputed as
  the median over the seeds of ``(J_seed - J_twin) / J_twin`` with the costs of the arm held and
  the new twin. K5 is added because a relative change of a cost of 10 % cannot occur, while the
  statements of that study are gaps of a few per cent against the twin (session).

"New", two variants (session; both count):

* **held**: the reference controllers with the cold multi-start at the margins of the artifacts
  (part ``cold_held``). If the largest violation of a controller then exceeds the tolerance of
  the headroom config, this is reported next to the number.
* **searched**: the margin searches of the findings repeated with the cold multi-start (same
  bisection, bracket and tolerance; part ``cold_search``), every controller at the margin found.
  For K4 and K5: C2 at its newly found margin, tail at the newly found margin of C0.

``change = |new - old| / |old|`` for every key number in both variants.

* **O0** if no change exceeds 10 %: the effect is immaterial for the reported statements; noted
  in the lab book.
* **O1** if at least one change exceeds 10 %: a dated addendum to the finding concerned is due
  (append-only, by the managing session). The key numbers and variants concerned are named.

The reading is INVALID (reported as such, not replaced) if a unit of ``cold_held`` or
``cold_search`` ended with an error or a search of ``cold_search`` did not converge. Without the
artifacts of the learned-setpoints study the reading is taken on K1-K3 and marked "without the
twin".

No reading is taken from (descriptive, all reported):

* ``warm_rerun``: the key numbers of a second run of the unchanged code and the units whose
  records differ from the artifact (relative difference of energy or cost above 1e-9);
* ``cold_repeat``: whether the records with the switch are bit-equal for other chunks and orders;
* ``queries``: per source (R1, C0, C1, C2) and pooled, at the thresholds 1e-9, 1e-6, 1e-4, 1e-3
  and 1e-2 of the relative power difference: the share of queries that differ between the warm
  orders; the share for which the warm start (natural order; any order) returns a WORSE optimum
  than the cold multi-start, and by how much (power; inlet temperature and gas velocity); the
  share for which it returns a better one; the cold single start; the dense reference (64
  starts) against the multi-start; the number of local optima per query;
* ``stage0_check``: admissible range, bands and tuned rule of C1 on the map with the cold
  multi-start against the stage-0 artifact; set points of R1 per interval, warm against cold.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import RUNS_DIR, provenance, require_clean_tree  # noqa: E402

from twinlink_ml.order_dependence.keynumbers import (  # noqa: E402
    cost_per_kg,
    g10_median,
    gain,
    median_gap_to_twin,
    reading,
    relative_change,
)
from twinlink_ml.order_dependence.settings import OrderDependenceSettings  # noqa: E402
from twinlink_ml.order_dependence.statistics import pool_units, query_statistics  # noqa: E402

HEADROOM_ARTIFACT = "dynamic_headroom_v2.json.gz"
LOAD_ARTIFACT = "load_following_v1.json.gz"
FAMILY = "observed"
PRINTED = {"K1": 0.0893, "K2": 0.0207, "K3": 0.0206, "K4": 0.39794, "K5": 0.0298}
FINDING = {
    "K1": "Befund 105",
    "K2": "Befund 126",
    "K3": "Befund 126",
    "K4": "learned set points",
    "K5": "learned set points",
}
SMALLEST_GAP_ARM = ("D-gru|none", "64")
NOT_EQUAL = 1.0e-9


def _load(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)  # type: ignore[no-any-return]
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def key_numbers(
    records: dict[str, list[dict[str, Any]]], arm_costs: dict[str, float] | None
) -> dict[str, float]:
    """K1-K3 from the records of R0, R1, C0, C1, C2; K4 and K5 if the arm costs are given."""
    j = {kind: cost_per_kg(records[kind]) for kind in ("C0", "C1", "C2")}
    out = {
        "K1": g10_median(records["R0"], records["R1"]),
        "K2": gain(j["C0"], j["C1"]),
        "K3": gain(j["C0"], j["C2"]),
    }
    if arm_costs is not None:
        out["K4"] = j["C2"]
        out["K5"] = median_gap_to_twin(arm_costs, j["C2"])
    return out


def largest_violation(records: list[dict[str, Any]]) -> float:
    """Largest violation over the units without an error."""
    return float(max(r["max_violation"] for r in records if r["max_violation"] is not None))


def unit_differences(
    old: list[dict[str, Any]], new: list[dict[str, Any]], unit_field: str, value_field: str
) -> dict[str, Any]:
    """Relative difference of ``value_field`` per unit, new against old."""
    by_unit = {r[unit_field]: r[value_field] for r in old}
    rel = np.array(
        [(r[value_field] - by_unit[r[unit_field]]) / by_unit[r[unit_field]] for r in new]
    )
    return {
        "units": int(rel.size),
        "units_differing": int(np.sum(np.abs(rel) > NOT_EQUAL)),
        "min": float(rel.min()),
        "max": float(rel.max()),
        "median_abs": float(np.median(np.abs(rel))),
    }


def set_point_differences(
    first: list[dict[str, Any]], second: list[dict[str, Any]], settings: OrderDependenceSettings
) -> dict[str, Any]:
    """Share of the intervals of R1 whose set points differ between two runs."""
    a_in = np.array([r["inlet_command_per_interval"] for r in first])
    b_in = np.array([r["inlet_command_per_interval"] for r in second])
    a_gas = np.array([r["gas_command_per_interval"] for r in first])
    b_gas = np.array([r["gas_command_per_interval"] for r in second])
    differs = (np.abs(a_in - b_in) > settings.same_optimum_inlet_k) | (
        np.abs(a_gas - b_gas) > settings.same_optimum_gas_m_per_s
    )
    return {
        "share_of_intervals_differing": float(np.mean(differs)),
        "largest_abs_d_inlet_k": float(np.max(np.abs(a_in - b_in))),
        "largest_abs_d_gas_m_per_s": float(np.max(np.abs(a_gas - b_gas))),
    }


def stage0_comparison(check: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    """Bands and rule of the cold map against the stage-0 artifact."""
    if "error" in check:
        return {"error": check["error"]}
    rule = {
        carrier: {
            "quantile": [artifact["rule"][carrier]["quantile"], check["rule"][carrier]["quantile"]],
            "spread_kg_per_h": [
                artifact["rule"][carrier]["spread_kg_per_h"],
                check["rule"][carrier]["spread_kg_per_h"],
            ],
            "relative_change_of_cost": relative_change(
                artifact["rule"][carrier]["cost"], check["rule"][carrier]["cost"]
            ),
            "equal": bool(
                artifact["rule"][carrier]["quantile"] == check["rule"][carrier]["quantile"]
                and np.isclose(
                    artifact["rule"][carrier]["spread_kg_per_h"],
                    check["rule"][carrier]["spread_kg_per_h"],
                    rtol=1e-12,
                )
            ),
        }
        for carrier in ("E", "G")
    }
    return {
        "bands_artifact": artifact["bands"],
        "bands_cold": check["bands"],
        "bands_equal": bool(
            all(
                np.allclose(artifact["bands"][name], check["bands"][name], rtol=0.0, atol=1e-12)
                for name in ("main", "box")
            )
        ),
        "rule": rule,
        "steady_failures_cold": check["steady_failures"],
    }


def map_comparison(check: dict[str, Any], stage0_file: dict[str, Any]) -> dict[str, Any]:
    """Map table of stage 0 with the cold multi-start against the artifact (margin 0)."""
    old, new = stage0_file["map"], check["map"]
    p_old = np.array(old["total_kw"], dtype=np.float64)
    p_new = np.array(new["total_kw"], dtype=np.float64)
    both = np.isfinite(p_old) & np.isfinite(p_new)
    rel = (p_old[both] - p_new[both]) / p_new[both]
    feasible_old = np.array(old["feasible_with_margin"], dtype=bool)
    feasible_new = np.array(new["feasible_with_margin"], dtype=bool)
    return {
        "points": int(p_old.size),
        "points_with_an_optimum_in_both": int(both.sum()),
        "optimum_only_in_artifact": int(np.sum(np.isfinite(p_old) & ~np.isfinite(p_new))),
        "optimum_only_cold": int(np.sum(~np.isfinite(p_old) & np.isfinite(p_new))),
        "feasibility_flags_differing": int(np.sum(feasible_old != feasible_new)),
        "relative_power_artifact_minus_cold": {
            "max": float(rel.max()),
            "min": float(rel.min()),
            "share_above_1e-6": float(np.mean(rel > 1e-6)),
            "share_above_1e-3": float(np.mean(rel > 1e-3)),
            "share_below_-1e-6": float(np.mean(rel < -1e-6)),
        },
    }


def analyse(
    run: dict[str, Any],
    headroom: dict[str, Any],
    load: dict[str, Any],
    stage0_file: dict[str, Any] | None,
    learned: tuple[dict[str, Any], dict[str, Any]] | None,
) -> dict[str, Any]:
    """Everything of the analysis, see the module docstring."""
    settings = OrderDependenceSettings(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in run["settings"].items()}
    )
    tolerance = float(headroom["study_config"]["control"]["violation_tolerance_kg_per_kg"])
    old_records = {
        "R0": headroom[FAMILY]["records"]["R0"],
        "R1": headroom[FAMILY]["records"]["R1"],
        **{kind: load["main"][kind]["records"] for kind in ("C0", "C1", "C2")},
    }
    arm_costs = None
    twin_artifact = None
    if learned is not None:
        learned_main, learned_analysis = learned
        arm, budget = SMALLEST_GAP_ARM
        arm_costs = learned_analysis["tables"][arm][budget]["J_per_seed"]
        twin_artifact = {
            "J_twin": learned_analysis["J"]["twin"],
            "gap_printed_source": learned_analysis["tables"][arm][budget]["gap"]["median"],
            "margin": learned_main["reference"]["C2"]["margin"],
        }
    old = key_numbers(old_records, arm_costs)
    if learned is not None:
        # the twin of the learned-setpoints study is its own run of C2, not the load-following one
        old["K4"] = cost_per_kg(learned[0]["reference"]["C2"]["records"])
        assert arm_costs is not None
        old["K5"] = median_gap_to_twin(arm_costs, old["K4"])
    out: dict[str, Any] = {
        "old": old,
        "printed_in_the_findings": {k: PRINTED[k] for k in old},
        "twin_artifact": twin_artifact,
        "variants": {},
    }
    changes: dict[str, float] = {}
    invalid: list[str] = []
    for variant, part in (("held", "cold_held"), ("searched", "cold_search")):
        block = run[part]
        records = {kind: block[kind]["records"] for kind in old_records}
        for kind, recs in records.items():
            errors = sum(1 for r in recs if r.get("max_violation") is None)
            if errors:
                invalid.append(f"{part} {kind}: {errors} units with an error")
            if part == "cold_search" and not block[kind]["search"]["converged"]:
                invalid.append(f"{part} {kind}: margin search did not converge")
        new = key_numbers(records, arm_costs)
        change = {k: relative_change(old[k], new[k]) for k in old}
        changes.update({f"{k}/{variant}": c for k, c in change.items()})
        out["variants"][variant] = {
            "new": new,
            "relative_change": change,
            "margins": {kind: block[kind]["margin"] for kind in records},
            "largest_violation": {kind: largest_violation(r) for kind, r in records.items()},
            "holds_the_limit": {
                kind: bool(largest_violation(r) <= tolerance) for kind, r in records.items()
            },
            "cost_per_kg": {k: cost_per_kg(records[k]) for k in ("C0", "C1", "C2")},
        }
    out["old_margins"] = {
        **{k: headroom[FAMILY]["margins"][k] for k in ("R0", "R1")},
        **{k: load["margins"][k] for k in ("C0", "C1", "C2")},
    }
    out["old_cost_per_kg"] = {k: cost_per_kg(old_records[k]) for k in ("C0", "C1", "C2")}
    exceeded = sorted(k for k, c in changes.items() if c > settings.reading_threshold)
    out["reading"] = {
        "threshold": settings.reading_threshold,
        "changes": changes,
        "largest_change": max(changes.values()),
        "reading": reading(changes, settings.reading_threshold),
        "exceeded": exceeded,
        "findings_concerned": sorted({FINDING[k.split("/")[0]] for k in exceeded}),
        "valid": not invalid,
        "invalid_because": invalid,
        "with_the_twin": learned is not None,
    }

    # ------------------------------------------------------------------ descriptive
    warm = {kind: run["warm_rerun"][kind]["records"] for kind in old_records}
    held = {kind: run["cold_held"][kind]["records"] for kind in old_records}
    warm_numbers = key_numbers(warm, arm_costs)
    out["warm_rerun"] = {
        "key_numbers": warm_numbers,
        "relative_change": {k: relative_change(old[k], warm_numbers[k]) for k in old},
        "largest_violation": {kind: largest_violation(r) for kind, r in warm.items()},
        "against_the_artifact": {
            "R0": unit_differences(old_records["R0"], warm["R0"], "scenario", "energy_kwh"),
            "R1": unit_differences(old_records["R1"], warm["R1"], "scenario", "energy_kwh"),
            **{
                kind: unit_differences(old_records[kind], warm[kind], "day_of_year", "cost_e")
                for kind in ("C0", "C1", "C2")
            },
        },
        "cold_held_against_the_artifact": {
            "R0": unit_differences(old_records["R0"], held["R0"], "scenario", "energy_kwh"),
            "R1": unit_differences(old_records["R1"], held["R1"], "scenario", "energy_kwh"),
            **{
                kind: unit_differences(old_records[kind], held[kind], "day_of_year", "cost_e")
                for kind in ("C0", "C1", "C2")
            },
        },
        "r1_set_points_warm_against_cold": set_point_differences(warm["R1"], held["R1"], settings),
    }
    if learned is not None:
        twin_records = learned[0]["reference"]["C2"]["records"]
        out["warm_rerun"]["twin_records_against_load_following"] = unit_differences(
            old_records["C2"], twin_records, "day_of_year", "cost_e"
        )
        out["warm_rerun"]["cold_held_against_the_twin_records"] = unit_differences(
            twin_records, held["C2"], "day_of_year", "cost_e"
        )
    if "cold_repeat" in run:
        out["cold_repeat"] = run["cold_repeat"]
    if "queries" in run:
        units = run["queries"]["units"]
        sources = {kind: list(by_unit.values()) for kind, by_unit in units.items()}
        out["queries"] = {
            kind: query_statistics(pool_units(u), settings) for kind, u in sources.items()
        }
        everything = [u for group in sources.values() for u in group]
        out["queries"]["pooled"] = query_statistics(pool_units(everything), settings)
        out["queries"]["units_with_an_error"] = sum(1 for u in everything if "error" in u)
    if "stage0_check" in run:
        out["stage0_check"] = stage0_comparison(run["stage0_check"], run["stage0_of_the_artifact"])
        if stage0_file is not None and "error" not in run["stage0_check"]:
            out["stage0_check"]["map"] = map_comparison(run["stage0_check"], stage0_file["stage0"])
    out["errors_of_the_run"] = len(run.get("errors", []))
    return out


def report(analysis: dict[str, Any]) -> str:
    """Markdown report."""
    lines = ["# Order dependence of the steady-state optimiser", ""]
    r = analysis["reading"]
    lines.append(
        f"Reading: **{r['reading']}** (valid: {r['valid']}; with the twin: {r['with_the_twin']}); "
        f"largest relative change {100 * r['largest_change']:.2f} % (threshold "
        f"{100 * r['threshold']:.0f} %)."
    )
    if r["exceeded"]:
        lines.append(f"Exceeded: {', '.join(r['exceeded'])}; {', '.join(r['findings_concerned'])}.")
    for reason in r["invalid_because"]:
        lines.append(f"INVALID: {reason}")
    lines += [
        "",
        "| key number | old | held | change % | searched | change % |",
        "|---|---|---|---|---|---|",
    ]
    held, searched = analysis["variants"]["held"], analysis["variants"]["searched"]
    for k, old in analysis["old"].items():
        lines.append(
            f"| {k} | {old:.6f} | {held['new'][k]:.6f} | "
            f"{100 * held['relative_change'][k]:.3f} | {searched['new'][k]:.6f} | "
            f"{100 * searched['relative_change'][k]:.3f} |"
        )
    lines += [
        "",
        "| controller | margin old | margin searched | violation held | holds |",
        "|---|---|---|---|---|",
    ]
    for kind, margin in analysis["old_margins"].items():
        lines.append(
            f"| {kind} | {margin:.6f} | {searched['margins'][kind]:.6f} | "
            f"{held['largest_violation'][kind]:.2e} | {held['holds_the_limit'][kind]} |"
        )
    w = analysis["warm_rerun"]
    lines += [
        "",
        "Warm rerun against the artifact (units differing / units; range of the relative difference):",
    ]
    for kind, d in w["against_the_artifact"].items():
        lines.append(
            f"* {kind}: {d['units_differing']} / {d['units']}; "
            f"{100 * d['min']:+.4f} % .. {100 * d['max']:+.4f} %"
        )
    lines.append(
        "Key numbers of the warm rerun, relative change: "
        + ", ".join(f"{k} {100 * c:.4f} %" for k, c in w["relative_change"].items())
    )
    if "cold_repeat" in analysis:
        lines += [
            "",
            "Cold repeat (bit-equal units / units): "
            + ", ".join(
                f"{kind} {c['equal_units']}/{c['units']}"
                for kind, c in analysis["cold_repeat"].items()
            ),
        ]
    if "queries" in analysis:
        lines += [
            "",
            "Queries (shares in %):",
            "",
            "| source | n | tau | differing between orders | warm natural worse | warm natural better | any order worse | cold single worse | reference better |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for kind, s in analysis["queries"].items():
            if not isinstance(s, dict):
                continue
            for tau, b in s["by_threshold"].items():
                lines.append(
                    f"| {kind} | {s['n_queries']} | {tau} | "
                    f"{100 * b['share_differing_between_orders']:.2f} | "
                    f"{100 * b['share_warm_natural_worse']:.2f} | "
                    f"{100 * b['share_warm_natural_better']:.2f} | "
                    f"{100 * b['share_warm_any_order_worse']:.2f} | "
                    f"{100 * b['share_cold_single_start_worse']:.2f} | "
                    f"{100 * b['share_reference_better_than_multi_start']:.2f} |"
                )
    if "stage0_check" in analysis:
        lines += [
            "",
            f"Stage-0 check: {json.dumps({k: v for k, v in analysis['stage0_check'].items() if k != 'error'})}",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--learned-main", help="learned_setpoints_v1.json.gz (optional)")
    parser.add_argument("--learned-analysis", help="learned_setpoints_v1_analysis.json (optional)")
    parser.add_argument("--learned-commit", help="commit of the branch the two files come from")
    parser.add_argument("--in-dir", default=str(RUNS_DIR))
    args = parser.parse_args()
    in_dir = Path(args.in_dir).resolve()
    smoke = in_dir != RUNS_DIR.resolve()
    output = in_dir / f"{args.name}_analysis.json"
    if output.exists():
        raise SystemExit(f"{output} exists -- artifacts are never overwritten; use a new name")
    if not smoke:
        require_clean_tree([output])
    run = _load(in_dir / f"{args.name}.json.gz")
    headroom = _load(RUNS_DIR / HEADROOM_ARTIFACT)
    load = _load(RUNS_DIR / LOAD_ARTIFACT)
    stage0_path = RUNS_DIR / "load_following_v1_stage0.json.gz"
    learned = None
    learned_source = None
    if bool(args.learned_main) != bool(args.learned_analysis):
        raise SystemExit("pass both --learned-main and --learned-analysis, or neither")
    if args.learned_main:
        paths = [Path(args.learned_main), Path(args.learned_analysis)]
        learned = (_load(paths[0]), _load(paths[1]))
        learned_source = {
            "commit": args.learned_commit,
            "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        }
    analysis = analyse(run, headroom, load, _load(stage0_path), learned)
    payload = {
        "provenance": provenance("not used", ignore_paths=[output]),
        "source": f"{args.name}.json.gz",
        "source_commit": run["provenance"]["git/commit"],
        "smoke": smoke or bool(run.get("smoke")),
        "learned_setpoints_source": learned_source,
        **analysis,
    }
    output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(report(analysis))
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

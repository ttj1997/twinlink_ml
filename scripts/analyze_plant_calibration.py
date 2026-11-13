r"""Analysis and pre-registered reading of the plant calibration study (docs/NN_plant_calibration.md).

Input: the record of ``scripts/run_plant_calibration.py`` (``artifacts/runs/<name>.json.gz``).
Output: ``artifacts/runs/<name>_analysis.json`` and a summary on stdout. No model runs.

Pre-registered reading (fixed and committed BEFORE the run on plant data)
==========================================================================

Basis: main variant (one bed-to-exhaust distance per product; a parameter not identifiable from the
calibration day stays at its nominal value, for the distance of the other product 0 = switch off),
reference solid fraction 0.65, nominal kinetics. "Calibrated model" = the model with the parameter
set selected by the identifiability rule on the calibration day (largest identifiable subset of the
"steady" candidates, at most 3, rules in ``conf/plant_calibration/plant_calibration.yaml``) fitted
by bounded weighted least squares on the stationary segments of that day. "Constant bias" (B1 of
Befund 120) = the uncalibrated model plus the mean residual (measured - model) per measured
quantity on the calibration day. Cross error = model - measured on every stationary segment of the
OTHER day; directions day 1 -> day 2 and day 2 -> day 1; quantities bed temperature (mean of six
sensors) and exhaust temperature.

* **K0** (stationary calibration usable): the calibrated model meets bed AND exhaust within 2 K on
  every segment of the other day in BOTH directions, AND its largest absolute cross error (over
  both directions, both quantities, all segments) is smaller than that of the constant bias.
* **K1** (data do not suffice for a physical calibration; the bias suffices until transient data
  are available): the largest absolute cross error of the calibrated model is NOT smaller than
  that of the constant bias.
* **between** (case not named in the request, added before the run): smaller than the bias but
  above 2 K somewhere. Reported as such; no reading of its own beyond "not K0, not K1".

Without reading (reported): the uncalibrated model on both days; the identifiability results
(delta_msqr, SVD, condition number, collinearity, selected sets per data set, profiles of the three
most sensitive candidates); the sensitivity at the estimates; the kinetics (critical moisture,
nu_max) are NEVER calibrated, their identifiability from temperatures is reported only; the
descriptive variant with ONE shared distance for both products (decision of the project lead
2026-11-13), with the same tables and the same comparison with the bias, explicitly "without
reading". Day = product: the cross-check between the days is a check across products; a check
within one product needs several days per product.

Profile verdict: profile of the weighted SSR over 11 grid points of the parameter's bounds plus the
joint optimum (a grid point with a lower SSR than the joint fit becomes the optimum, reported);
95 % threshold chi2(1) = 3.84 on (a) the SSR with the sensor sigma and (b) the SSR
rescaled with s^2 = SSR_min / (n - p); "practically identifiable" (b) if the threshold is crossed on
every side where the optimum is not on a bound.
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

from twinlink_ml.plant_calibration import profile_verdict, reading  # noqa: E402

DAYS = ("day_1", "day_2")


def load(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def _num(v: Any) -> float:
    return math.inf if v is None else float(v)


def _metrics(block: dict[str, Any]) -> dict[str, Any]:
    """None (from inf in JSON) back to inf."""
    out = {q: dict(block[q]) for q in ("bed", "exhaust")}
    for q in out:
        for k in ("max_abs_k", "rms_k", "mean_k"):
            out[q][k] = _num(out[q][k])
    out["max_abs_k"] = _num(block["max_abs_k"])
    return out


def cross_table(record: dict[str, Any], variant: str) -> dict[str, Any]:
    cal = record["calibration"]
    rows: dict[str, Any] = {}
    for d in DAYS:
        v = cal["variants"][variant][d]
        other = DAYS[1 - DAYS.index(d)]
        rows[f"{d}->{other}"] = {
            "selected": v.get("selected"),
            "theta": v.get("fit", {}).get("theta"),
            "at_bound": v.get("fit", {}).get("at_bound"),
            "calibrated_other_day": _metrics(v["other_day"]) if "other_day" in v else None,
            "calibrated_own_day": _metrics(v["own_day"]) if "own_day" in v else None,
            "bias_other_day": _metrics(cal["bias"][d]["other_day"]),
            "bias_offsets_k": cal["bias"][d]["offsets_k"],
            "uncalibrated_other_day": _metrics(cal["uncalibrated"][other]),
        }
    return rows


def apply_reading(rows: dict[str, Any], tolerance: float) -> dict[str, Any]:
    if any(r["calibrated_other_day"] is None for r in rows.values()):
        return {"reading": "not evaluable (failed fit)"}
    return reading(
        {k: r["calibrated_other_day"] for k, r in rows.items()},
        {k: r["bias_other_day"] for k, r in rows.items()},
        tolerance,
    )


def profiles(record: dict[str, Any]) -> dict[str, Any]:
    cfg = record["config"]
    threshold = float(cfg["profile"]["threshold_chi2"])
    joint = record["calibration"].get("joint_main", {})
    if "fit" not in joint:
        return {"failure": "joint fit failed"}
    fit = joint["fit"]
    out: dict[str, Any] = {}
    nominal = {p["name"]: p["nominal"] for p in cfg["parameters"]}
    for m, prof in record["profiles"].items():
        x_opt = float(fit["theta"].get(m, nominal[m]))
        ok = [g for g in prof["grid"] if g.get("ssr") is not None]
        grid = [g["value"] for g in ok] + [x_opt]
        ssr = [g["ssr"] for g in ok] + [fit["ssr"]]
        # A grid point below the joint fit (the fit found a local minimum only) becomes the
        # optimum of the profile; reported as `optimum_from_grid`.
        best = int(np.argmin(ssr))
        from_grid = ssr[best] < float(fit["ssr"])
        ssr_opt = float(ssr[best]) if from_grid else float(fit["ssr"])
        x_opt = float(grid[best]) if from_grid else x_opt
        verdict = profile_verdict(
            grid,
            ssr,
            x_opt=x_opt,
            ssr_opt=ssr_opt,
            bounds=tuple(prof["bounds"]),
            n_residuals=int(fit["n_residuals"]),
            n_free=len(fit["parameters"]),
            threshold_chi2=threshold,
        )
        verdict["profile"] = [
            {"value": g, "delta_ssr": s - ssr_opt} for g, s in zip(grid, ssr, strict=True)
        ]
        verdict["optimum_from_grid"] = bool(from_grid)
        verdict["free"] = prof["free"]
        verdict["failed_grid_points"] = len(prof["grid"]) - len(ok)
        out[m] = verdict
    return out


def kinetics_statement(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    nom = record["identifiability"]["nominal"]["main"]
    for label, block in nom.items():
        out[label] = {
            m: block["delta_msqr"][m]
            for m in ("critical_moisture_kg_per_kg", "max_saturation_degree", "cp_solid_j_per_kg_k")
        }
        out[label]["largest_identifiable_any"] = block["largest_identifiable_any"]
        kin_subsets = [
            s
            for s in block["subsets_all"]
            if s["identifiable"]
            and any(
                p in ("critical_moisture_kg_per_kg", "max_saturation_degree")
                for p in s["parameters"]
            )
        ]
        out[label]["identifiable_subsets_with_kinetics"] = [s["parameters"] for s in kin_subsets]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("record", help="record path or name in artifacts/runs")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    path = Path(args.record)
    if not path.exists():
        path = RUNS_DIR / f"{args.record}.json.gz"
    record = load(path)
    out_path = Path(args.output) if args.output else RUNS_DIR / f"{record['name']}_analysis.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; artifacts are never overwritten")
    tolerance = float(record["config"]["calibration"]["tolerance_k"])
    main_rows = cross_table(record, "main")
    shared_rows = cross_table(record, "shared")
    ident = record["identifiability"]
    summary = {
        "record": record["name"],
        "record_commit": record["provenance"].get("git/commit"),
        "record_dirty": record["provenance"].get("git/dirty"),
        "selected_sets": {
            v: {d: b["selected"] for d, b in ident["nominal"][v].items()} for v in ident["nominal"]
        },
        "condition_numbers": {
            v: {d: b["svd"].get("condition_number") for d, b in ident["nominal"][v].items()}
            for v in ident["nominal"]
        },
        "delta_msqr": {d: b["delta_msqr"] for d, b in ident["nominal"]["main"].items()},
        "selected_at_estimates": {
            k: {d: b["selected"] for d, b in v["main"].items()}
            for k, v in ident.items()
            if k.startswith("estimate_")
        },
        "kinetics": kinetics_statement(record),
        "profiles": profiles(record),
        "main": {"cross": main_rows, "reading": apply_reading(main_rows, tolerance)},
        "shared_without_reading": {
            "cross": shared_rows,
            "comparison_as_reading_rule_no_reading": apply_reading(shared_rows, tolerance),
        },
        "uncalibrated": record["calibration"]["uncalibrated"],
        "joint_main": record["calibration"].get("joint_main"),
        "failures": record.get("failures"),
    }
    out_path.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    r = summary["main"]["reading"]
    print(
        f"reading (main variant): {r.get('reading')}  "
        f"calibrated {r.get('largest_cross_error_calibrated_k')} K vs bias "
        f"{r.get('largest_cross_error_bias_k')} K"
    )
    for variant, rows in (("main", main_rows), ("shared (no reading)", shared_rows)):
        print(f"--- {variant}")
        for k, row in rows.items():
            c, b, u = (
                row["calibrated_other_day"],
                row["bias_other_day"],
                row["uncalibrated_other_day"],
            )
            if c is None:
                print(f"{k}: failed")
                continue
            print(
                f"{k}: set {row['selected']}; cal bed/exh {c['bed']['max_abs_k']:.2f}/"
                f"{c['exhaust']['max_abs_k']:.2f}; bias {b['bed']['max_abs_k']:.2f}/"
                f"{b['exhaust']['max_abs_k']:.2f}; uncal {u['bed']['max_abs_k']:.2f}/"
                f"{u['exhaust']['max_abs_k']:.2f} K"
            )
    print(f"selected: {summary['selected_sets']}")
    print(f"analysis {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

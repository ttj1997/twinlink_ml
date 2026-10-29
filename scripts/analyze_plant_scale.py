r"""Analysis and pre-registered reading of the plant-near study (docs/34_plant_scale.md).

Reads ``artifacts/runs/<name>.json`` of ``scripts/run_plant_scale.py`` and writes
``artifacts/runs/<name>_analysis.json``. No plant data are read here.

Pre-registered reading (fixed in this docstring and committed BEFORE the run)
------------------------------------------------------------------------------

Wording of the study request: P1 -- the calibrated model reproduces bed and exhaust temperature on
the respective other day within 2 K: the plant-near model is usable for energy questions. P2 --
deviation above 2 K: structural finding; the results of step 4 hold only under reservation.

Operationalised:

* **Error** = model minus measurement, bed temperature = mean of the six bed sensors, exhaust
  temperature = the sensor at the granulator outlet; per stationary segment of the check day
  (segment means); calibration on all stationary segments of the other day.
* **Reference case of the reading:** solid fraction 0.65 on both days (docs/23 reference), kinetics
  and isotherm nominal (lab line), dried supply air at the placeholder humidity, every other
  placeholder of ``conf/plant_scale/plant_scale.yaml`` and of the plant-scale process config.
* **P1** if at least one calibration candidate (C0 switch off; C3 offset; C1 heat loss; C2
  evaporation above the bed; each with the bed heat-loss UA) has ``|error| <= 2 K`` for BOTH
  temperatures at EVERY segment of the check day in BOTH directions (day 1 -> day 2 and day 2 ->
  day 1). The plant-near model is then the passing candidate with the smallest largest error
  (selection rule of the driver docstring).
* **P2** otherwise. Step 4 then uses the candidate with the smallest largest cross error, and every
  step-4 number carries the reservation.
* **Structural statement** (independent of P1/P2): "the structure T_out = T_bed cannot represent
  the data" if C0, fitted on a day, leaves ``|error| > 2 K`` for at least one temperature at at
  least one segment OF THAT SAME DAY (in-sample), in either direction. Parameter-free bound
  reported with it: with ``T_out = T_bed`` every parameter set has ``max(|e_bed|, |e_exh|) >=
  gap / 2`` at each segment.
* Selection among four candidates on the cross-check is a selection, not a validation (two days,
  both used); stated with the reading.

Reported WITHOUT reading: the balance numbers of the candidate explanations; sensitivity of the
calibration over the solid-fraction band and the 20 kinetics draws; the SNR plant-near against
the lab line (as docs/33 and with dried supply air); the steady headroom fixed against tracked
operating point at plant scale against the lab line under the same definition and against the
9.67 % of Befund 108 (its definition reproduced on the lab nominal as a check of the optimiser).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import REPO_ROOT, RUNS_DIR, git  # noqa: E402

TOLERANCE_K = 2.0
CANDIDATES = ("C0_none", "C3_offset", "C1_heat_loss", "C2_evaporation")
MEASUREMENTS = ("Y_out", "Y_out_balance", "T_out", "T_bed", "T_bed_x6")


def clean(obj: Any) -> Any:
    """JSON-safe copy (NaN/inf -> None)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def max_abs(rows: list[dict[str, Any]], key: str) -> float:
    if any("failure" in r for r in rows):
        return math.inf
    return max(abs(r[key]) for r in rows)


def summarise_fit(f: dict[str, Any]) -> dict[str, Any]:
    return {
        "parameters": f["fit"]["parameters"],
        "at_bound": f["fit"]["at_bound"],
        "status": f["fit"]["status"],
        "in_sample_max_bed_k": max_abs(f["in_sample"], "error_bed_k"),
        "in_sample_max_exhaust_k": max_abs(f["in_sample"], "error_exhaust_k"),
        "cross_max_bed_k": max_abs(f["cross"], "error_bed_k"),
        "cross_max_exhaust_k": max_abs(f["cross"], "error_exhaust_k"),
        "cross_mean_bed_k": float(
            np.mean([r["error_bed_k"] for r in f["cross"] if "failure" not in r])
        ),
        "cross_mean_exhaust_k": float(
            np.mean([r["error_exhaust_k"] for r in f["cross"] if "failure" not in r])
        ),
        "cross_errors": [
            {k: r.get(k) for k in ("point", "error_bed_k", "error_exhaust_k", "moisture_kg_per_kg")}
            for r in f["cross"]
        ],
    }


def calibration_tables(record: dict[str, Any]) -> dict[str, Any]:
    fits = record["calibration"]["fits"]
    ref_x = 0.65
    out: dict[str, Any] = {"reference": {}, "band": {}, "draws": {}}
    for c in CANDIDATES:
        for f in fits:
            if f["candidate"] != c:
                continue
            key = f"{c}:{f['calibration_day']}->{f['check_day']}"
            if f["draw"] is None and f["solid_fraction"] == ref_x:
                out["reference"][key] = summarise_fit(f)
            elif f["draw"] is None:
                out["band"].setdefault(key, {})[f"{f['solid_fraction']:g}"] = summarise_fit(f)
            else:
                out["draws"].setdefault(key, []).append(summarise_fit(f))
    draws_summary = {}
    for key, rows in out["draws"].items():
        worst = [max(r["cross_max_bed_k"], r["cross_max_exhaust_k"]) for r in rows]
        draws_summary[key] = {
            "n": len(rows),
            "cross_worst_min_k": float(np.min(worst)),
            "cross_worst_median_k": float(np.median(worst)),
            "cross_worst_max_k": float(np.max(worst)),
            "passing_draws": int(sum(w <= TOLERANCE_K for w in worst)),
            "ua_range_w_per_k": [
                float(min(r["parameters"]["heat_loss_ua_w_per_k"] for r in rows)),
                float(max(r["parameters"]["heat_loss_ua_w_per_k"] for r in rows)),
            ],
            "freeboard_value_range": [
                float(min(r["parameters"]["freeboard_value"] for r in rows)),
                float(max(r["parameters"]["freeboard_value"] for r in rows)),
            ],
        }
    out["draws"] = draws_summary
    return out


def reading(record: dict[str, Any], tables: dict[str, Any]) -> dict[str, Any]:
    ref = tables["reference"]
    worst = {}
    for c in CANDIDATES:
        rows = [v for k, v in ref.items() if k.startswith(c + ":")]
        worst[c] = max(max(r["cross_max_bed_k"], r["cross_max_exhaust_k"]) for r in rows)
    passing = [c for c in CANDIDATES if worst[c] <= TOLERANCE_K]
    pool = passing or list(CANDIDATES)
    selected = min(pool, key=lambda c: (worst[c], CANDIDATES.index(c)))
    c0 = [v for k, v in ref.items() if k.startswith("C0_none:")]
    c0_in = max(max(r["in_sample_max_bed_k"], r["in_sample_max_exhaust_k"]) for r in c0)
    gaps = [g for d in record["survey"]["days"].values() for g in d["segment_gaps_k"]]
    driver = record["calibration"]["selection"]
    return {
        "worst_cross_error_k": worst,
        "passing": passing,
        "reading": "P1" if passing else "P2",
        "selected_for_step_4": selected,
        "driver_selection_agrees": driver["selected"] == selected
        and driver["reading"] == ("P1" if passing else "P2"),
        "structure_t_out_eq_t_bed_cannot_represent": bool(c0_in > TOLERANCE_K),
        "c0_in_sample_worst_k": c0_in,
        "parameter_free_bound_k": {
            "min_half_gap": min(gaps) / 2.0,
            "max_half_gap": max(gaps) / 2.0,
        },
    }


def snr_summary(record: dict[str, Any]) -> dict[str, Any]:
    obs = record["observability"]
    out: dict[str, Any] = {}
    for key, block in obs.items():
        v = block["snr"].get("variants", {})
        out[key] = {
            "snr": {m: v[m]["snr"] for m in MEASUREMENTS if m in v},
            "r2": {m: v[m]["fit"]["r2"] for m in MEASUREMENTS if m in v},
            "n_failed": block["snr"]["n_failed"],
            "n_infeasible": block["snr"]["n_infeasible"],
            "moisture_range_kg_per_kg": block["snr"]["moisture_range_kg_per_kg"],
            "solid_to_air_ratio": block["nominal"].get("solid_to_air_ratio"),
            "nominal_moisture_kg_per_kg": block["nominal"].get("moisture_kg_per_kg"),
        }
    # reproduction of docs/33 (lab as there) against the Befund 116 analysis
    path = REPO_ROOT / "artifacts" / "runs" / "observability_v1_analysis.json"
    check: dict[str, float] = {}
    if path.exists():
        ref = json.loads(path.read_text(encoding="utf-8"))["lab"]
        for name, block in ref.items():
            key = f"lab_as_docs33:{name}"
            if key in out:
                diffs = [
                    abs(out[key]["snr"][m] - block["variants"][m]["snr"])
                    for m in MEASUREMENTS
                    if m in block["variants"]
                ]
                check[name] = max(diffs)
    return {"per_case": out, "reproduction_max_abs_snr_difference": check}


def headroom_summary(record: dict[str, Any]) -> dict[str, Any]:
    res = record["headroom"]["results"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for label, r in res.items():
        if r["kind"] in ("plant", "lab"):
            groups.setdefault(label.split(":")[0], []).append({"label": label, **r})
    out: dict[str, Any] = {"amplitudes": record["headroom"]["amplitudes"], "groups": {}}
    for g, rows in groups.items():
        nominal = next((r for r in rows if r["label"].endswith(":nominal")), None)
        draws = [r for r in rows if not r["label"].endswith(":nominal")]
        nom_sav = [r["saving_nominal"] for r in draws if r.get("saving_nominal") is not None]
        med_sav = [r["saving_median"] for r in draws if r.get("saving_median") is not None]
        entry: dict[str, Any] = {
            "nominal_kinetics": None
            if nominal is None
            else {
                k: nominal.get(k)
                for k in (
                    "limit_kg_per_kg",
                    "saving_nominal",
                    "saving_median",
                    "saving_min",
                    "saving_max",
                )
            },
            "draws": len(draws),
            "draws_r0_inadmissible": sum(not r["r0"]["admissible"] for r in draws),
            "draws_saving_nominal": [
                float(np.min(nom_sav)),
                float(np.median(nom_sav)),
                float(np.max(nom_sav)),
            ]
            if nom_sav
            else None,
            "draws_saving_median": [
                float(np.min(med_sav)),
                float(np.median(med_sav)),
                float(np.max(med_sav)),
            ]
            if med_sav
            else None,
            "solver_failures": int(sum(r.get("solver_failures", 0) for r in rows)),
        }
        if nominal is not None and nominal["r0"]["admissible"]:
            entry["r0_nominal_kinetics"] = nominal["r0"]
            entry["r1_at_nominal_disturbance"] = nominal["points"][-1]["r1"]
            if "today" in nominal:
                entry["today"] = nominal["today"]
        out["groups"][g] = entry
    ref = res.get("lab:befund108")
    if ref is not None:
        out["lab_befund108_reproduction"] = {
            "saving": ref["saving"],
            "reference": 0.0967,
            "fixed": ref["fixed"],
            "tracked": ref["tracked"],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", default="plant_scale_v1")
    parser.add_argument("--record", default=None)
    args = parser.parse_args()
    path = Path(args.record) if args.record else RUNS_DIR / f"{args.name}.json"
    out_path = path.with_name(path.stem + "_analysis.json")
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; artifacts are never overwritten")
    record = json.loads(path.read_text(encoding="utf-8"))
    tables = calibration_tables(record)
    result = {
        "source": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "source_commit": record["provenance"]["git/commit"],
        "source_dirty": record["provenance"]["git/dirty"],
        "analysis_commit": git("rev-parse", "HEAD"),
        "smoke": record.get("smoke", False),
        "reading": reading(record, tables),
        "calibration": tables,
        "observability": snr_summary(record),
        "headroom": headroom_summary(record),
        "balances": record["balances"],
        "consistent_solid_fraction": record["balances"]["consistent_solid_fraction"],
    }
    out_path.write_text(json.dumps(clean(result), indent=1), encoding="utf-8")
    r = result["reading"]
    print(
        f"reading {r['reading']}; selected {r['selected_for_step_4']}; worst cross {r['worst_cross_error_k']}"
    )
    print(
        f"structure T_out = T_bed cannot represent: {r['structure_t_out_eq_t_bed_cannot_represent']}"
    )
    for key, v in tables["reference"].items():
        print(
            f"{key:34s} in {v['in_sample_max_bed_k']:.2f}/{v['in_sample_max_exhaust_k']:.2f} "
            f"cross {v['cross_max_bed_k']:.2f}/{v['cross_max_exhaust_k']:.2f} K  {v['parameters']}"
        )
    for key, v in result["observability"]["per_case"].items():
        print(f"{key:24s} " + " ".join(f"{m} {s:.3f}" for m, s in v["snr"].items()))
    print(json.dumps(result["headroom"]["groups"], indent=1, default=str)[:3000])
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

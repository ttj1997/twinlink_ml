"""Analysis of the dynamic headroom study -- with the reading fixed BEFORE the first run.

Input: ``artifacts/runs/<name>.json.gz`` of ``scripts/run_dynamic_headroom.py``. Output:
``artifacts/runs/<name>_analysis.json`` and a Markdown report on stdout. Study document:
``docs/24_dynamic_headroom.md``.

Quantities (per scenario ``s``, truth model, whole scenario)
------------------------------------------------------------
``E_Rx(s)``
    Energy of controller Rx at its required margin: model total power (``power_total_kw``,
    objective of the bandit studies) integrated over the scenario, kWh. ``CO2_Rx(s)``: gas-fired
    heater booking (0.366 / 0.201 kg CO2e/kWh), reported alongside, not used for the reading.
Gaps
    ``g13(s) = (E_R1 - E_R3) / E_R1`` (realistically reachable: MPC without disturbance preview),
    ``g12(s) = (E_R1 - E_R2) / E_R1`` (upper bound: perfect preview), ``g10 = (E_R0 - E_R1) / E_R0``
    (what quasi-stationary RTO saves over the fixed point). Positive = the dynamic controller
    needs less energy.
Decomposition of ``g13`` (and analogously ``g12``)
    ``margin share = (E_R1(m1) - E_R1(m3)) / E_R1(m1)``, ``trajectory share = (E_R1(m3) -
    E_R3(m3)) / E_R1(m1)``; the two add up to ``g13``. ``R1(m3)`` is R1 run with R3's margin -- only
    a point of the decomposition; it may violate the path condition.
Statistics
    Median and range (min, max) over scenarios; mean only as an extra. Per family, per scenario
    type, and pooled over the family.

Pre-registered reading (fixed in this docstring before the first run)
--------------------------------------------------------------------
Applies to the family ``observed`` ONLY, pooled over all its scenarios (all types):

* **L0** -- median of ``g13`` < 2 % energy: dynamic control does not pay in continuous operation;
  close the RL line for the granulator.
* **L1** -- 2 % <= median of ``g13`` <= 5 %: MPC suffices; pursue RL only with a robustness
  argument.
* **L2** -- median of ``g13`` > 5 %: an RL-against-MPC study is justified.

Independent of L0-L2: **robustness cost** ``c_Rx(s) = (E_Rx(m_Rx,95) - E_Rx(m_Rx)) / E_Rx(m_Rx)``
on the NOMINAL truth, over the robustness subset (``robustness.scenario_indices`` of every
observed type), ``m_Rx,95`` = smallest margin at which the path condition holds in at least 95 %
of the isotherm draws. If the median of ``c_R1`` or of ``c_R3`` is > 5 %: **candidate "robust
control" open**, independently of L0-L2.

Further fixed rules: the family ``stress`` is an upper estimate and gets no reading; the
upper bound ``g12`` is reported but does not decide the reading; per-type medians are reported,
and a type whose median would fall in another class is named, without changing the reading; for
``inlet_temperature_fast`` the decomposition (margin share) is reported explicitly. Product change
is reported per changeover (energy and duration to specification), not extrapolated. A margin
search that did not converge, a solver failure, or an R2 fallback to the R1 schedule is listed
next to the result; the reading is still computed and marked as affected.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import RUNS_DIR, provenance  # noqa: E402

THRESHOLD_L1 = 0.02
THRESHOLD_L2 = 0.05
THRESHOLD_ROBUST = 0.05


def reading(median_gap: float) -> str:
    """L0 / L1 / L2 from the pooled median of ``g13`` (boundaries 2 % and 5 % belong to L1)."""
    if median_gap < THRESHOLD_L1:
        return "L0"
    if median_gap <= THRESHOLD_L2:
        return "L1"
    return "L2"


def stats(values: np.ndarray) -> dict[str, float]:
    """Median, range and mean."""
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "n": int(values.size),
    }


def _by_name(records: list[dict]) -> dict[str, dict]:
    return {r["scenario"]: r for r in records}


def family_analysis(family: dict[str, Any]) -> dict[str, Any]:
    """Gaps, decomposition and their statistics for one family."""
    rec = {k: _by_name(v) for k, v in family["records"].items()}
    names = family["scenarios"]
    kinds = [rec["R1"][n]["kind"] for n in names]

    def energy(key: str) -> np.ndarray:
        return np.array([rec[key][n]["energy_kwh"] for n in names])

    def co2(key: str) -> np.ndarray:
        return np.array([rec[key][n]["co2_kg"] for n in names])

    e0, e1, e2, e3 = energy("R0"), energy("R1"), energy("R2"), energy("R3")
    e1_m3, e1_m2 = energy("R1_at_R3_margin"), energy("R1_at_R2_margin")
    per_scenario = {
        "g13": (e1 - e3) / e1,
        "g12": (e1 - e2) / e1,
        "g10": (e0 - e1) / e0,
        "g13_margin_share": (e1 - e1_m3) / e1,
        "g13_trajectory_share": (e1_m3 - e3) / e1,
        "g12_margin_share": (e1 - e1_m2) / e1,
        "g12_trajectory_share": (e1_m2 - e2) / e1,
        "g13_co2": (co2("R1") - co2("R3")) / co2("R1"),
        "g12_co2": (co2("R1") - co2("R2")) / co2("R1"),
    }
    pooled = {k: stats(v) for k, v in per_scenario.items()}
    by_type: dict[str, dict[str, dict[str, float]]] = {}
    for kind in sorted(set(kinds)):
        mask = np.array([k == kind for k in kinds])
        by_type[kind] = {k: stats(v[mask]) for k, v in per_scenario.items()}
    issues = []
    for kind_name, search in family["searches"].items():
        if not search["converged"]:
            issues.append(f"{kind_name}: margin search did not converge")
    for key in ("R0", "R1", "R2", "R3"):
        for n in names:
            solver = rec[key][n]["solver"]
            if solver.get("failures", 0):
                issues.append(f"{key} {n}: {solver['failures']} solver failures")
            if solver.get("fallback_to_guess"):
                issues.append(f"{key} {n}: R2 used the R1 schedule (fallback)")
    max_violation = {
        key: float(max(rec[key][n]["max_violation"] for n in names))
        for key in ("R0", "R1", "R2", "R3")
    }
    return {
        "margins": family["margins"],
        "energy_kwh": {k: stats(v) for k, v in (("R0", e0), ("R1", e1), ("R2", e2), ("R3", e3))},
        "pooled": pooled,
        "by_type": by_type,
        "max_violation_at_margin": max_violation,
        "issues": issues,
        "per_scenario": {k: [float(x) for x in v] for k, v in per_scenario.items()},
        "scenarios": names,
    }


def product_change_analysis(stage: dict[str, Any]) -> dict[str, Any]:
    """Per changeover: energy (whole scenario and after the change) and duration to spec."""
    rec = {k: _by_name(v) for k, v in stage["records"].items()}
    names = stage["scenarios"]
    out: dict[str, Any] = {"margins": stage["margins"], "by_direction": {}}
    for kind in sorted({rec["R1"][n]["kind"] for n in names}):
        subset = [n for n in names if rec["R1"][n]["kind"] == kind]
        row: dict[str, Any] = {"n": len(subset)}
        for key in ("R1", "R2", "R3"):
            row[key] = {
                "energy_kwh": stats(np.array([rec[key][n]["energy_kwh"] for n in subset])),
                "energy_after_change_kwh": stats(
                    np.array([rec[key][n]["change"]["energy_after_kwh"] for n in subset])
                ),
                "duration_to_spec_s": stats(
                    np.array([rec[key][n]["change"]["duration_to_spec_s"] for n in subset])
                ),
                "max_violation_after": stats(
                    np.array([rec[key][n]["change"]["max_violation_after"] for n in subset])
                ),
            }
        for key in ("R2", "R3"):
            row[f"delta_energy_{key}_vs_R1_kwh"] = stats(
                np.array([rec[key][n]["energy_kwh"] - rec["R1"][n]["energy_kwh"] for n in subset])
            )
            row[f"delta_energy_{key}_vs_R1_rel"] = stats(
                np.array(
                    [
                        (rec[key][n]["energy_kwh"] - rec["R1"][n]["energy_kwh"])
                        / rec["R1"][n]["energy_kwh"]
                        for n in subset
                    ]
                )
            )
        out["by_direction"][kind] = row
    return out


def robustness_analysis(stage: dict[str, Any]) -> dict[str, Any]:
    """Robust margins and their energy cost on the nominal truth."""
    rec = {k: _by_name(v) for k, v in stage["records_nominal_truth"].items()}
    names = list(rec["R1_nominal_margin"])
    costs = {}
    for key in ("R1", "R3"):
        nominal = np.array([rec[f"{key}_nominal_margin"][n]["energy_kwh"] for n in names])
        robust = np.array([rec[f"{key}_robust_margin"][n]["energy_kwh"] for n in names])
        costs[key] = stats((robust - nominal) / nominal)
    candidate = max(costs["R1"]["median"], costs["R3"]["median"]) > THRESHOLD_ROBUST
    return {
        "nominal_margins": stage["nominal_margins"],
        "robust_margins": {"R1": stage["R1"]["robust_margin"], "R3": stage["R3"]["robust_margin"]},
        "k_required": stage["k_required"],
        "n_draws": len(stage["draws"]),
        "rejected_draws": stage["rejected_draws"],
        "r1_per_draw_margins": [d["margin"] for d in stage["R1"]["per_draw"]],
        "r3_holds_at_robust_margin": stage["R3"]["holds_at_robust_margin"],
        "cost": costs,
        "candidate_robust_control_open": bool(candidate),
        "subset": names,
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f} %"


def report(result: dict[str, Any]) -> str:
    """Markdown summary."""
    lines = [f"# Dynamic headroom -- analysis of `{result['artifact']}`", ""]
    for family in ("observed", "stress"):
        if family not in result:
            continue
        fa = result[family]
        lines += [f"## Family `{family}`", ""]
        lines.append(
            "Margins (kg/kg): "
            + ", ".join(f"{k} {v:.5f}" for k, v in sorted(fa["margins"].items()))
        )
        lines.append("")
        lines.append("| quantity | median | min | max |")
        lines.append("|---|---|---|---|")
        for key in (
            "g13",
            "g12",
            "g10",
            "g13_margin_share",
            "g13_trajectory_share",
            "g12_margin_share",
            "g12_trajectory_share",
            "g13_co2",
            "g12_co2",
        ):
            s = fa["pooled"][key]
            lines.append(f"| {key} | {_pct(s['median'])} | {_pct(s['min'])} | {_pct(s['max'])} |")
        lines.append("")
        lines.append("| type | g13 median (range) | margin share | trajectory share | g12 median |")
        lines.append("|---|---|---|---|---|")
        for kind, s in fa["by_type"].items():
            lines.append(
                f"| {kind} | {_pct(s['g13']['median'])} ({_pct(s['g13']['min'])} .. "
                f"{_pct(s['g13']['max'])}) | {_pct(s['g13_margin_share']['median'])} | "
                f"{_pct(s['g13_trajectory_share']['median'])} | {_pct(s['g12']['median'])} |"
            )
        lines.append("")
        if fa["issues"]:
            lines += ["Issues: " + "; ".join(fa["issues"]), ""]
    if "reading" in result:
        r = result["reading"]
        lines += [
            f"**Reading (observed, pooled median g13 = {_pct(r['median_g13'])}): {r['class']}**",
            "",
        ]
        if r["types_in_other_class"]:
            lines += [
                "Types whose own median falls in another class: "
                + ", ".join(f"{k} ({v})" for k, v in r["types_in_other_class"].items()),
                "",
            ]
    if "robustness" in result:
        rb = result["robustness"]
        lines += [
            "## Robustness",
            "",
            f"Margins nominal -> 95 %: R1 {rb['nominal_margins']['R1']:.5f} -> "
            f"{rb['robust_margins']['R1']:.5f}, R3 {rb['nominal_margins']['R3']:.5f} -> "
            f"{rb['robust_margins']['R3']:.5f} (k = {rb['k_required']} of {rb['n_draws']})",
            f"Energy cost: R1 median {_pct(rb['cost']['R1']['median'])} "
            f"({_pct(rb['cost']['R1']['min'])} .. {_pct(rb['cost']['R1']['max'])}), "
            f"R3 median {_pct(rb['cost']['R3']['median'])} "
            f"({_pct(rb['cost']['R3']['min'])} .. {_pct(rb['cost']['R3']['max'])})",
            f"Candidate robust control open: {rb['candidate_robust_control_open']}",
            "",
        ]
    if "product_change" in result:
        lines += [
            "## Product change (per changeover)",
            "",
            "| direction | dE R2-R1 kWh | dE R3-R1 kWh | duration to spec R1 / R2 / R3 (min, median) |",
            "|---|---|---|---|",
        ]
        for kind, row in result["product_change"]["by_direction"].items():
            d = [row[k]["duration_to_spec_s"]["median"] / 60.0 for k in ("R1", "R2", "R3")]
            lines.append(
                f"| {kind} | {row['delta_energy_R2_vs_R1_kwh']['median']:.4f} | "
                f"{row['delta_energy_R3_vs_R1_kwh']['median']:.4f} | "
                f"{d[0]:.1f} / {d[1]:.1f} / {d[2]:.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="artifact name without .json.gz")
    parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    args = parser.parse_args()
    runs = Path(args.runs_dir)
    source = runs / f"{args.name}.json.gz"
    target = runs / f"{args.name}_analysis.json"
    if target.exists():
        raise SystemExit(f"{target} exists -- artifacts are never overwritten")
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    result: dict[str, Any] = {"artifact": source.name, "source_provenance": data["provenance"]}
    for family in ("observed", "stress"):
        if family in data:
            result[family] = family_analysis(data[family])
    if "observed" in result:
        obs = result["observed"]
        median = obs["pooled"]["g13"]["median"]
        cls = reading(median)
        other = {
            kind: reading(s["g13"]["median"])
            for kind, s in obs["by_type"].items()
            if reading(s["g13"]["median"]) != cls
        }
        result["reading"] = {
            "median_g13": median,
            "class": cls,
            "types_in_other_class": other,
            "affected_by_issues": bool(obs["issues"]),
        }
    if "product_change" in data:
        result["product_change"] = product_change_analysis(data["product_change"])
    if "robustness" in data:
        result["robustness"] = robustness_analysis(data["robustness"])
    smoke = runs.resolve() != RUNS_DIR.resolve()
    result["provenance"] = (
        provenance("not used", ignore_paths=[target]) if not smoke else {"smoke": True}
    )
    text = report(result)
    print(text)
    target.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

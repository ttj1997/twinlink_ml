r"""M1: cost of the soft-sensor margin in CO2 (by carrier) and kWh, single line A (docs/09 MO2).

Question: what does a constant soft-sensor margin ``m`` at the moisture limit cost, and how much
of the control benefit (fixed operating point -> per-instance optimum) survives, in CO2 by carrier
and in the unweighted kWh sum from the SAME solutions? Single line (A, spray rate 1.5 kg/h, config
nominal), no network, so ``Q_recovered`` = 0 (the network enters in M2).

Settings (confirmed by the project lead): quality limit ``L_q`` = 0.15 kg/kg main case, 0.20 kg/kg
separately as the stress case (``--quality-limit``); the kinetics keep the critical moisture 0.20
(decoupled mechanics of Befund 64); ``m`` = 0.005961819615175404 kg/kg (median half-width of the
calibrated ensemble band, Befund 65 FA2, as in S0.3); objective ``co2_gas_heater`` (D5 main case:
gas-fired air heater, ``eta_gas`` 0.80 of the partner data case, factors 0.366 / 0.201 kg CO2e/kWh
of ``net_energy.py``); the electric-heater booking is a sensitivity of M5 and is only REPORTED here
(both bookings of every solution are in the artifact; the optimisation is on the gas booking).

Arms, on the 5 x 5 ambient grid of Befund 52, same optimiser budget as Befund 52 (DE popsize 10,
maxiter 50, tolerance 1e-4, seed 0, Nelder-Mead refinement, search precision 30 / 1e-7, three
variables: inlet temperature, gas velocity, nozzle pressure; spray rate fixed):

  F   fixed operating point: one set point, feasible at all 25 points against ``L_q``, minimal mean
      objective (BB3(a) of Befund 52; the fixed point never sees a moisture measurement, hence no
      margin, as in Befund 64);
  M0  per-instance oracle against ``L_q`` (margin 0);
  M   per-instance oracle against ``L_q - m``.

Reported (descriptive, no gate): per point and pooled, over the points feasible in all three arms:
cost of the margin ``M - M0`` (absolute and relative, CO2 and kWh sum), the remaining fraction of
the control benefit ``(F - M) / (F - M0)`` (median of the per-point ratios and pooled sums, as in
Befund 64 KA2; on all points and on the points where M0 binds the limit), the distance of M and M0
to the validity boundary 0.20 (kg/kg and in units of ``m``), binding counts, convergence flags. The
objective differs from Befund 64/65 (CO2 with a gas heater instead of the electric gross power), so
the numbers are NOT the same quantity as the 95.35 % / 89.52 % of the Befund 65 table; the anchor
stage checks the machinery against Befund 52-level artifacts instead (see below).

Anchor stage (``--stage anchor``, always before the arms; limit 0.15, margin 0): the legacy
objective (Befund 52 ``energy_target``, electric booking, threshold 0.15) of the new code path at
grid points 0, 12 and 24 must reproduce the optimum of the committed artifact
``s05_limit015_oracle_and_fixed.json`` (set points and energy to 1e-9); the arms are only run if
the anchor holds, and a failing anchor is reported, not repaired silently.

Runtime estimate (before the run, 16 logical CPUs; oracle 17 min and fixed-point search 18 min per
limit in S0.5, measured): anchor about 9 min (3 points in one wave); arms per limit about 17 + 17 +
18 = 52 min, so about 1 h per limit, 2 h for both limits when run one after the other.

    PYTHONPATH=<worktree>/src python scripts/run_m1_margin_cost.py --stage anchor
    PYTHONPATH=<worktree>/src python scripts/run_m1_margin_cost.py --stage arms --quality-limit 0.15
    PYTHONPATH=<worktree>/src python scripts/run_m1_margin_cost.py --stage arms --quality-limit 0.20
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import warnings
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import (  # noqa: E402
    REPO_ROOT,
    RUNS_DIR,
    provenance,
    require_clean_tree,
    write_artifact,
)

import twinlink_ml  # noqa: E402
from twinlink_ml.evaluation.carrier_oracle import (  # noqa: E402
    OracleSpec,
    find_fixed_point,
    run_oracle_grid,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config  # noqa: E402
from twinlink_ml.simulation.operating_point import (  # noqa: E402
    DEFAULT_SEARCH_MAX_ITERATIONS,
    DEFAULT_SEARCH_TOLERANCE,
)
from twinlink_ml.simulation.oracle_search import (  # noqa: E402
    OptimizationBudget,
    action_bounds,
    build_grid,
    with_gas_velocity_minimum,
)

MARGIN_KG_PER_KG = 0.005961819615175404
GAS_HEATER_EFFICIENCY = 0.80
KINETICS_BOUNDARY_KG_PER_KG = 0.20
ANCHOR_INDICES = (0, 12, 24)
ANCHOR_TOLERANCE = 1.0e-9
ANCHOR_LIMIT_KG_PER_KG = 0.15
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"


def _budget(workers: int) -> OptimizationBudget:
    grid52 = json.loads(
        (RUNS_DIR / "study_operating_point_regret_win_repro.json").read_text("utf-8")
    )
    budget = OptimizationBudget(
        seed=0,
        popsize=10,
        maxiter=50,
        de_tol=1.0e-4,
        search_max_iterations=DEFAULT_SEARCH_MAX_ITERATIONS,
        search_tolerance=DEFAULT_SEARCH_TOLERANCE,
        workers=workers,
    )
    committed = grid52["budget"]
    for key in (
        "de_tol",
        "maxiter",
        "popsize",
        "search_max_iterations",
        "search_tolerance",
        "seed",
    ):
        assert dataclasses.asdict(budget)[key] == committed[key], key
    return budget


def _load_config() -> Any:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    config = load_process_config(CONFIG_PATH)
    assert config.material.critical_moisture_kg_per_kg == KINETICS_BOUNDARY_KG_PER_KG
    return config


def run_anchor() -> int:
    config = _load_config()
    workers = cpu_count()
    budget = _budget(workers)
    grid = build_grid(config, 5)
    points = [grid[i] for i in ANCHOR_INDICES]
    spec = OracleSpec(ANCHOR_LIMIT_KG_PER_KG, "legacy_power_kw", GAS_HEATER_EFFICIENCY)
    results = run_oracle_grid(
        config, points, action_bounds(config), budget, spec, workers=workers, label="anchor"
    )
    committed = json.loads((RUNS_DIR / "s05_limit015_oracle_and_fixed.json").read_text("utf-8"))[
        "grid_results"
    ]
    comparisons = []
    for index, result in zip(ANCHOR_INDICES, results, strict=True):
        ref = next(r for r in committed if r.get("_grid_index") == index)
        diffs = {
            "inlet_temperature_celsius": abs(
                result["inlet_temperature_celsius"] - ref["inlet_temperature_celsius"]
            ),
            "gas_velocity_m_per_s": abs(
                result["gas_velocity_m_per_s"] - ref["gas_velocity_m_per_s"]
            ),
            "nozzle_pressure_bar": abs(result["nozzle_pressure_bar"] - ref["nozzle_pressure_bar"]),
            "energy_target": abs(result["legacy_energy_target_kw"] - ref["energy_target"]),
        }
        comparisons.append(
            {
                "grid_index": index,
                "differences": diffs,
                "reproduced": bool(all(v <= ANCHOR_TOLERANCE for v in diffs.values())),
            }
        )
    reproduced = all(c["reproduced"] for c in comparisons)
    write_artifact(
        "m1_anchor_s05",
        {
            "_provenance": provenance(),
            "tolerance": ANCHOR_TOLERANCE,
            "limit_kg_per_kg": ANCHOR_LIMIT_KG_PER_KG,
            "indices": list(ANCHOR_INDICES),
            "comparisons": comparisons,
            "reproduced": reproduced,
            "results": results,
        },
    )
    print("anchor reproduced:", reproduced, comparisons)
    return 0 if reproduced else 1


def _pooled(num: list[float], den: list[float]) -> float | None:
    return float(sum(num) / sum(den)) if den and sum(den) != 0.0 else None


def analyse(
    f_points: list[dict[str, Any]], m0: list[dict[str, Any]], m: list[dict[str, Any]]
) -> dict[str, Any]:
    """Cost of the margin and remaining control benefit, CO2 and kWh sum from the same solutions."""
    rows = []
    for f, a, b in zip(f_points, m0, m, strict=True):
        ok = all((not x.get("solve_failed")) and x.get("feasible", False) for x in (f, a, b))
        row: dict[str, Any] = {
            "ambient_temperature_celsius": a["ambient_temperature_celsius"],
            "ambient_relative_humidity": a["ambient_relative_humidity"],
            "feasible_in_all_arms": ok,
        }
        if ok:
            for key in ("co2_kg_per_h", "kwh_sum_kw"):
                fv, av, bv = (x["gas_heater"][key] for x in (f, a, b))
                row[key] = {
                    "F": fv,
                    "M0": av,
                    "M": bv,
                    "cost_abs": bv - av,
                    "cost_rel": (bv - av) / av,
                    "remaining_fraction": (fv - bv) / (fv - av) if fv != av else None,
                    "benefit_M0": fv - av,
                    "benefit_M": fv - bv,
                }
            row["m0_binding_moisture"] = a["constraint_check"]["binding_moisture"]
            row["m_binding_moisture"] = b["constraint_check"]["binding_moisture"]
            row["distance_to_validity_boundary_m0"] = a["distance_to_validity_boundary_kg_per_kg"]
            row["distance_to_validity_boundary_m"] = b["distance_to_validity_boundary_kg_per_kg"]
            row["moisture_m"] = b["particle_moisture_kg_per_kg"]
            row["setpoints_m0"] = [
                a["inlet_temperature_celsius"],
                a["gas_velocity_m_per_s"],
                a["nozzle_pressure_bar"],
            ]
            row["setpoints_m"] = [
                b["inlet_temperature_celsius"],
                b["gas_velocity_m_per_s"],
                b["nozzle_pressure_bar"],
            ]
        rows.append(row)
    ok_rows = [r for r in rows if r["feasible_in_all_arms"]]
    binding = [r for r in ok_rows if r["m0_binding_moisture"]]
    summary: dict[str, Any] = {
        "n_points": len(rows),
        "n_feasible_in_all_arms": len(ok_rows),
        "n_m0_binding": len(binding),
    }
    for key in ("co2_kg_per_h", "kwh_sum_kw"):
        entry: dict[str, Any] = {}
        for label, subset in (("all", ok_rows), ("m0_binding", binding)):
            ratios = [
                r[key]["remaining_fraction"]
                for r in subset
                if r[key]["remaining_fraction"] is not None
            ]
            entry[label] = {
                "n": len(subset),
                "remaining_fraction_median": float(np.median(ratios)) if ratios else None,
                "remaining_fraction_pooled": _pooled(
                    [r[key]["benefit_M"] for r in subset], [r[key]["benefit_M0"] for r in subset]
                ),
                "cost_abs_median": float(np.median([r[key]["cost_abs"] for r in subset]))
                if subset
                else None,
                "cost_rel_median": float(np.median([r[key]["cost_rel"] for r in subset]))
                if subset
                else None,
                "cost_rel_max": float(np.max([r[key]["cost_rel"] for r in subset]))
                if subset
                else None,
            }
        summary[key] = entry
    if binding:
        d_m = [r["distance_to_validity_boundary_m"] for r in binding]
        summary["distance_to_validity_boundary_at_m0_binding_points"] = {
            "m_arm_min_kg_per_kg": float(np.min(d_m)),
            "m_arm_median_kg_per_kg": float(np.median(d_m)),
            "m_arm_min_in_units_of_m": float(np.min(d_m)) / MARGIN_KG_PER_KG,
            "m_arm_median_in_units_of_m": float(np.median(d_m)) / MARGIN_KG_PER_KG,
        }
    return {"per_point": rows, "summary": summary}


def run_arms(quality_limit: float, gas_velocity_min: float | None) -> int:
    config = _load_config()
    workers = cpu_count()
    budget = _budget(workers)
    grid = build_grid(config, 5)
    bounds = action_bounds(config)
    config_default_min = bounds[1][0]
    tag = f"l{round(quality_limit * 100):03d}"
    if gas_velocity_min is not None:
        # run parameter, not a changed default: only the search box of this run
        bounds = with_gas_velocity_minimum(bounds, gas_velocity_min)
        tag += f"_umin{round(gas_velocity_min * 100):03d}"
    spec_m0 = OracleSpec(quality_limit, "co2_gas_heater", GAS_HEATER_EFFICIENCY)
    spec_m = OracleSpec(quality_limit - MARGIN_KG_PER_KG, "co2_gas_heater", GAS_HEATER_EFFICIENCY)

    m0 = run_oracle_grid(config, grid, bounds, budget, spec_m0, workers=workers, label=f"M0 {tag}")
    m = run_oracle_grid(config, grid, bounds, budget, spec_m, workers=workers, label=f"M {tag}")
    fixed = find_fixed_point(config, grid, bounds, budget, spec_m0)
    analysis = analyse(fixed["per_grid_point"], m0, m)
    write_artifact(
        f"m1_margin_cost_{tag}",
        {
            "_provenance": provenance(),
            "quality_limit_kg_per_kg": quality_limit,
            "gas_velocity_lower_bound_m_per_s": bounds[1][0],
            "gas_velocity_lower_bound_config_default_m_per_s": config_default_min,
            "gas_velocity_lower_bound_is_run_parameter": gas_velocity_min is not None,
            "margin_kg_per_kg": MARGIN_KG_PER_KG,
            "kinetics_boundary_kg_per_kg": KINETICS_BOUNDARY_KG_PER_KG,
            "gas_heater_efficiency": GAS_HEATER_EFFICIENCY,
            "objective": "co2_gas_heater",
            "budget": dataclasses.asdict(budget),
            "config_fingerprint": config.fingerprint,
            "fixed_point_feasible_everywhere": fixed["feasible_everywhere"],
            "summary": analysis["summary"],
            "per_point": analysis["per_point"],
            "arm_F_fixed_point": fixed,
            "arm_M0_oracle": m0,
            "arm_M_oracle": m,
        },
    )
    print(json.dumps(analysis["summary"], indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=("anchor", "arms"), required=True)
    parser.add_argument("--quality-limit", type=float, choices=(0.15, 0.20), default=0.15)
    parser.add_argument(
        "--gas-velocity-min",
        type=float,
        default=None,
        help="lower bound of the gas velocity for this run in m/s (default: the config value)",
    )
    arguments = parser.parse_args()
    require_clean_tree()
    assert Path(twinlink_ml.__file__).resolve().is_relative_to(REPO_ROOT), (
        "twinlink_ml is imported from outside this checkout; set PYTHONPATH=<checkout>/src"
    )
    if arguments.stage == "anchor":
        return run_anchor()
    return run_arms(arguments.quality_limit, arguments.gas_velocity_min)


if __name__ == "__main__":
    raise SystemExit(main())

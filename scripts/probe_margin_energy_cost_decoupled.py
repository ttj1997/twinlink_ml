r"""MA2: repeat KA1 decoupled -- drying kinetics unchanged (0.20 kg/kg), only the
threshold of the quality constraint shifted.

Motivation (MA1, see Befund 64 addendum): `material.critical_moisture_kg_per_kg` has a
DUAL ROLE in the model. It is (a) `X_cr` in `thermo.normalised_moisture` -- the denominator of
the normalised drying curve, `span = X_cr - X_eq`, enters `evaporation_rate_kg_per_s`
directly, i.e. the physics of `solve_steady_state` itself (comment in `config.py`:
"Drying kinetics: normalised drying curve, linearly falling rate") -- AND (b) the
threshold of the quality constraint in `study_operating_point_regret.py::check_
constraints` (an ORDINARY function argument there, independent of the config).
`probe_margin_energy_cost.py` (KA1), like `probe_critical_moisture_sensitivity.py`
(Befund 54 DA1), varied `material.critical_moisture_kg_per_kg` itself -- thereby shifting
both roles at once, not only the quality limit. For the RL agent margin
(0.0605 kg/kg), a pure kinetics validation (`sorption.py::validate_
isotherm_against_critical_moisture`) therefore appeared as "infeasible" before any point was
solved -- although Befund 62 GA1 in the SAME model (kinetics unchanged at 0.20 kg/kg) actually
measured product moistures of 0.055-0.070 kg/kg.

This script decouples the two roles WITHOUT changing `study_operating_point_regret.py`
(still loaded via `importlib`, no line changed there) and WITHOUT changing the
ProcessConfig (`material.critical_moisture_kg_per_kg` stays at 0.20 kg/kg throughout
-- the same fingerprint as the base config of Befund 52/54/KA1). Only the value passed to `check_
constraints(..., critical_moisture_kg_per_kg=...)` -- an ordinary
function argument of this function, independent of `_BASE_CONFIG` -- is set to the
respective effective limit.

Technically: `study_operating_point_regret.py`'s own orchestration (`_penalized_energy`,
`optimize_grid_point`, `_fixed_point_worker`) ALWAYS reads the threshold from `_BASE_CONFIG.
material.critical_moisture_kg_per_kg` -- hard-wired, no injection point. These three
functions are therefore NOT used here; instead this script composes
`evaluate_point`, `check_constraints`, `_objective_components` and `_optimize` (all four
imported unchanged from the module) into its own objective function living here, which takes
the threshold as an explicit parameter -- the same building-block principle as
`_OBJECTIVE_VALUE_FN` in `study_operating_point_regret_net.py` (MA3/MA4), only for the
constraint instead of the target.

Only BB2 (per-instance optima), as in KA1 -- no BB3(a). The fixed operating point (`E_fest`)
and the oracle at the true limit (`E_Orakel(0,20)`, "oracle_anchor") are NOT affected by the
dual role: at margin 0, kinetics parameter and threshold coincide, there is no
difference from the decoupled computation -- both are reused unchanged from Befund 52/54 (DA1).
The remaining six limits (both soft-sensor margins, 0.18/0.15/0.12,
agent margin) are recomputed here -- unlike in KA1 also 0.15 kg/kg, because DA1's
0.15 point had also shifted the kinetics along with it.

Reused from `probe_margin_energy_cost.py` (KA1, unchanged): margin provenance
(`load_margin_sources`), limit definition (`build_boundary_definitions`), FA3 reference
(`load_fa3_reference`), KA2 aggregation (`compute_ka2_for_boundary`, `_pooled`,
`_median_or_none`), grid/budget matching (`_assert_same_grid`, `_assert_same_budget`,
`_grid_coordinates`). Only the grid point solver is new.

Usage::

    python scripts/probe_margin_energy_cost_decoupled.py \
        --output artifacts/runs/probe_margin_energy_cost_decoupled.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from types import ModuleType
from typing import Any

import probe_margin_energy_cost as ka1

from twinlink_ml.simulation import PlaceholderConfigWarning, ProcessConfig, load_process_config
from twinlink_ml.utils.fingerprint_history import fingerprints_equivalent
from twinlink_ml.utils.tracking import environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "probe_margin_energy_cost_decoupled.json"

RECOMPUTED_LABELS = (
    "excited_lstm_3rmse",
    "constant_lstm_3rmse",
    "grid_0.18",
    "grid_0.15",
    "grid_0.12",
    "rl_agent_margin",
)
"""Alle sieben KA1-Grenzen ausser dem Anker (Marge 0 -- Kinetik und Schwelle fallen dort
zusammen, keine Entkopplung noetig, unveraendert aus Befund 52/54 wiederverwendet)."""


def _penalized_energy_decoupled(
    x: Any,
    t_amb: float,
    phi_amb: float,
    max_iterations: int,
    tolerance: float,
    effective_critical_moisture_kg_per_kg: float,
    regret_module: ModuleType,
) -> float:
    """Like `study_operating_point_regret.py::_penalized_energy`, but with a DECOUPLED
    threshold: `evaluate_point` solves against `regret_module._BASE_CONFIG` (kinetics, always
    0.20 kg/kg), `check_constraints` checks against `effective_critical_moisture_kg_per_kg`
    (the margin of this run) -- two different values, where `_penalized_energy` knows only
    one. Module level for picklability (Windows `spawn`) would not be necessary here (only
    needed within this script as a closure substitute), but it lives at module level anyway,
    because `_optimize` passes it on to `differential_evolution`/`minimize`,
    which -- depending on `workers` -- may also pickle."""
    assert regret_module._BASE_CONFIG is not None
    t_in, u_gas, p_nozzle = x
    try:
        solution = regret_module.evaluate_point(
            t_in,
            u_gas,
            p_nozzle,
            t_amb,
            phi_amb,
            max_iterations=max_iterations,
            tolerance=tolerance,
            base_config=regret_module._BASE_CONFIG,
        )
    except (RuntimeError, ValueError):
        return regret_module._SOLVE_FAILURE_PENALTY_KW
    check = regret_module.check_constraints(
        solution,
        critical_moisture_kg_per_kg=effective_critical_moisture_kg_per_kg,
        gas_velocity_m_per_s=u_gas,
    )
    objective = regret_module._objective_components(solution, regret_module._BASE_CONFIG)[
        "objective_value"
    ]
    return float(objective + regret_module._PENALTY_WEIGHT_KW * check.normalized_violation)


def _penalized_energy_decoupled_toplevel(
    x: Any,
    t_amb: float,
    phi_amb: float,
    max_iterations: int,
    tolerance: float,
    effective_critical_moisture_kg_per_kg: float,
) -> float:
    """Top-level wrapper (picklable under `spawn`) -- `regret_module` is resolved via the
    module global `_REGRET_MODULE`, which the pool initializer sets in every worker
    (the same pattern as `regret_module._BASE_CONFIG`)."""
    return _penalized_energy_decoupled(
        x,
        t_amb,
        phi_amb,
        max_iterations,
        tolerance,
        effective_critical_moisture_kg_per_kg,
        _REGRET_MODULE,
    )


_REGRET_MODULE: ModuleType | None = None


def _init_worker(base_config: ProcessConfig) -> None:
    """Pool initializer: loads `study_operating_point_regret.py` freshly in EVERY worker
    (independent of the start method, see the `_load_regret_module` docstring there) and sets
    `_BASE_CONFIG` -- identical to `regret_module._set_base_config`, only additionally sets the
    module global `_REGRET_MODULE` in this script, so that the top-level objective function
    finds it."""
    global _REGRET_MODULE
    _REGRET_MODULE = ka1._load_regret_module()
    _REGRET_MODULE._set_base_config(base_config)


def optimize_grid_point_decoupled(
    t_amb: float,
    phi_amb: float,
    bounds: list[tuple[float, float]],
    budget: Any,
    effective_critical_moisture_kg_per_kg: float,
) -> dict[str, Any]:
    """Like `study_operating_point_regret.py::optimize_grid_point`, but with a decoupled
    threshold -- see `_penalized_energy_decoupled`. Result schema deliberately identical to
    `optimize_grid_point`'s return value, so that `probe_margin_energy_cost.compute_ka2_for_
    boundary` can be reused unchanged."""
    regret_module = _REGRET_MODULE
    assert regret_module is not None and regret_module._BASE_CONFIG is not None
    args = (
        t_amb,
        phi_amb,
        budget.search_max_iterations,
        budget.search_tolerance,
        effective_critical_moisture_kg_per_kg,
    )
    de_result, refine_result, de_elapsed, refine_elapsed = regret_module._optimize(
        _penalized_energy_decoupled_toplevel, bounds, args, budget, de_workers=1
    )
    best_x = refine_result.x if refine_result.fun <= de_result.fun else de_result.x
    t_in, u_gas, p_nozzle = (float(v) for v in best_x)

    solve_failed = False
    try:
        solution = regret_module.evaluate_point(
            t_in,
            u_gas,
            p_nozzle,
            t_amb,
            phi_amb,
            max_iterations=budget.search_max_iterations,
            tolerance=budget.search_tolerance,
            base_config=regret_module._BASE_CONFIG,
        )
    except (RuntimeError, ValueError) as exc:
        solution = None
        solve_failed = True
        solve_failure_reason = str(exc)

    result: dict[str, Any] = {
        "ambient_temperature_celsius": t_amb,
        "ambient_relative_humidity": phi_amb,
        "inlet_temperature_celsius": t_in,
        "gas_velocity_m_per_s": u_gas,
        "nozzle_pressure_bar": p_nozzle,
        "de_converged": bool(de_result.success),
        "de_message": str(de_result.message),
        "de_nit": int(de_result.nit),
        "de_nfev": int(de_result.nfev),
        "de_elapsed_s": de_elapsed,
        "refine_converged": bool(refine_result.success),
        "refine_message": str(refine_result.message),
        "refine_nit": int(refine_result.nit),
        "refine_elapsed_s": refine_elapsed,
        "converged": bool(refine_result.success),
        "solve_failed": solve_failed,
    }
    if solve_failed:
        result["solve_failure_reason"] = solve_failure_reason
        result["feasible"] = False
        return result

    check = regret_module.check_constraints(
        solution,
        critical_moisture_kg_per_kg=effective_critical_moisture_kg_per_kg,
        gas_velocity_m_per_s=u_gas,
    )
    components = regret_module._objective_components(solution, regret_module._BASE_CONFIG)
    result.update(
        {
            "energy_target": components["objective_value"],
            "energy_target_unit": regret_module._BASE_CONFIG.energy.target,
            "gross_energy_target_kw": components["gross_energy_target_kw"],
            "q_rec_kw": components["q_rec_kw"],
            "recovery_credit_kw": components["recovery_credit_kw"],
            "specific_energy_kwh_per_kg": regret_module.specific_energy_kwh_per_kg(solution),
            "bed_temperature_celsius": solution.bed_temperature_celsius,
            "particle_moisture_kg_per_kg": solution.particle_moisture_kg_per_kg,
            "diameter_um": solution.diameter_m * 1e6,
            "residual_inf_norm": solution.residual_inf_norm,
            "feasible": check.feasible,
            "infeasibility_reasons": list(solution.infeasibility_reasons),
            "constraint_check": check.as_dict(),
        }
    )
    return result


def _grid_point_task_decoupled(
    task: tuple[int, float, float, list[tuple[float, float]], Any, float],
) -> dict[str, Any]:
    index, t_amb, phi_amb, bounds, budget, effective_critical_moisture_kg_per_kg = task
    start = time.perf_counter()
    result = optimize_grid_point_decoupled(
        t_amb, phi_amb, bounds, budget, effective_critical_moisture_kg_per_kg
    )
    result["_grid_index"] = index
    result["_wall_elapsed_s"] = time.perf_counter() - start
    return result


def run_oracle_grid_decoupled(
    base_config: ProcessConfig,
    regret_module: ModuleType,
    *,
    grid_size: int,
    workers: int,
    budget: Any,
    effective_critical_moisture_kg_per_kg: float,
    label: str,
) -> list[dict[str, Any]]:
    """`regret_module` is passed through by the caller instead of being reloaded here: every
    call of `_load_regret_module()` replaces `sys.modules["study_operating_point_regret"]` with
    a FRESH module object (see the docstring there) -- an `OptimizationBudget` that
    was created against an earlier load can no longer be pickled afterwards
    (`PicklingError: nicht dasselbe Objekt`, observed empirically in the smoke test). In the
    main process the module is therefore loaded EXACTLY ONCE (`main()`), and only
    reused here."""
    bounds = regret_module.action_bounds(base_config)
    grid_points = regret_module.build_grid(base_config, grid_size)

    start = time.perf_counter()
    print(
        f"\n=== {label}: Gitter {grid_size}x{grid_size} = {len(grid_points)} Punkte "
        f"(entkoppelt, Schwelle={effective_critical_moisture_kg_per_kg:.6f} kg/kg) ===",
        flush=True,
    )
    with Pool(workers, initializer=_init_worker, initargs=(base_config,)) as pool:
        tasks = [
            (index, t_amb, phi_amb, bounds, budget, effective_critical_moisture_kg_per_kg)
            for index, (t_amb, phi_amb) in enumerate(grid_points)
        ]
        grid_results_by_index: dict[int, dict[str, Any]] = {}
        for n_done, result in enumerate(
            pool.imap_unordered(_grid_point_task_decoupled, tasks), start=1
        ):
            index = result["_grid_index"]
            status = (
                "OK"
                if not result.get("solve_failed") and result.get("feasible")
                else "UNZULAESSIG/FEHLER"
            )
            print(
                f"  [{n_done}/{len(grid_points)}] Punkt {index + 1} -> {status:20s} "
                f"({result['_wall_elapsed_s']:5.1f}s)",
                flush=True,
            )
            grid_results_by_index[index] = result
    grid_results = [grid_results_by_index[i] for i in range(len(grid_points))]
    elapsed = time.perf_counter() - start
    n_infeasible = sum(
        1 for r in grid_results if r.get("solve_failed") or not r.get("feasible", False)
    )
    print(f"  {label}: {elapsed:.0f}s, {n_infeasible}/{len(grid_results)} unzulaessig", flush=True)
    return grid_results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=ka1.CONFIG_PATH)
    parser.add_argument("--workers", type=int, default=-1, help="-1 = alle Kerne")
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)

    env = environment_fingerprint()
    if env["git/dirty"]:
        print(
            "WARNUNG: Arbeitsbaum ist dirty -- git/dirty wird im Artefakt vermerkt.",
            flush=True,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base_config = load_process_config(arguments.config)
    # Kinetics stays UNCHANGED at 0.20 kg/kg -- the same fingerprint as Befund 52/54/KA1.
    assert (
        base_config.material.critical_moisture_kg_per_kg
        == ka1.CRITICAL_MOISTURE_REFERENCE_KG_PER_KG
    )

    da1 = ka1.load_da1_artifact()
    if not fingerprints_equivalent(base_config.fingerprint, da1["base_config_fingerprint"]):
        raise RuntimeError(
            f"Aktuelle Config hat Fingerprint {base_config.fingerprint!r}, DA1-Artefakt "
            f"{da1['base_config_fingerprint']!r} -- keine gemeinsame Basis."
        )
    r_reference = da1["results_by_value"]["0.20"]
    if not r_reference["fixed_point"]["feasible_everywhere"]:
        raise RuntimeError("DA1s Fixpunkt bei 0,20 kg/kg ist nicht ueberall zulaessig.")

    margins = ka1.load_margin_sources()
    fa3_reference = ka1.load_fa3_reference()
    boundaries = ka1.build_boundary_definitions(margins)

    regret_module = ka1._load_regret_module()
    workers = arguments.workers if arguments.workers > 0 else cpu_count()
    budget = regret_module.OptimizationBudget(
        seed=0,
        popsize=regret_module.DEFAULT_POPSIZE,
        maxiter=regret_module.DEFAULT_MAXITER,
        de_tol=regret_module.DEFAULT_DE_TOL,
        search_max_iterations=regret_module.DEFAULT_SEARCH_MAX_ITERATIONS,
        search_tolerance=regret_module.DEFAULT_SEARCH_TOLERANCE,
        workers=workers,
    )
    budget_dict = dataclasses.asdict(budget)
    ka1._assert_same_budget(budget_dict, r_reference["budget"], context="Budget vs. DA1 0,20 kg/kg")

    reference_grid_coords = ka1._grid_coordinates(r_reference["grid_results"])
    fresh_grid_coords = ka1._grid_coordinates(
        [
            {"ambient_temperature_celsius": t, "ambient_relative_humidity": phi}
            for t, phi in regret_module.build_grid(base_config, ka1.GRID_SIZE)
        ]
    )
    ka1._assert_same_grid(
        reference_grid_coords, fresh_grid_coords, context="frisches Gitter vs. DA1 0,20 kg/kg"
    )

    binding_at_reference = [
        bool(r["constraint_check"]["binding_moisture"]) for r in r_reference["grid_results"]
    ]

    print("=" * 88)
    print("MA2: KA1 entkoppelt -- Kinetik fest bei 0,20 kg/kg, nur die Schwelle variiert")
    print(f"Config       {arguments.config}  (Fingerprint {base_config.fingerprint})")
    print(f"Gitter       {ka1.GRID_SIZE}x{ka1.GRID_SIZE} (wie Befund 54 DA1)")
    print(f"Worker       {workers}")
    print(f"git/commit={env['git/commit']}  git/dirty={env['git/dirty']}")
    for label, spec in boundaries.items():
        recompute = label in RECOMPUTED_LABELS
        print(
            f"  {label:22s} Schwelle={spec['critical_moisture_kg_per_kg']:.6f} kg/kg "
            f"(Marge {spec['margin_kg_per_kg']:.6f} kg/kg){'  [neu, entkoppelt]' if recompute else '  [Anker, unveraendert]'}"
        )
    print("=" * 88, flush=True)

    total_start = time.perf_counter()
    grid_results_by_boundary: dict[str, list[dict[str, Any]]] = {}
    reused_flags: dict[str, str | None] = {}
    for label, spec in boundaries.items():
        if label not in RECOMPUTED_LABELS:
            grid_results_by_boundary[label] = r_reference["grid_results"]
            reused_flags[label] = (
                "probe_critical_moisture_sensitivity.json#results_by_value.0.20.grid_results (Marge 0 -- Kinetik und Schwelle fallen zusammen)"
            )
            print(f"\n{label}: unveraendert (Marge 0) -- keine neue Rechnung.")
            continue
        grid_results_by_boundary[label] = run_oracle_grid_decoupled(
            base_config,
            regret_module,
            grid_size=ka1.GRID_SIZE,
            workers=workers,
            budget=budget,
            effective_critical_moisture_kg_per_kg=spec["critical_moisture_kg_per_kg"],
            label=label,
        )
        reused_flags[label] = None
        new_coords = ka1._grid_coordinates(grid_results_by_boundary[label])
        ka1._assert_same_grid(
            reference_grid_coords, new_coords, context=f"Gitter {label} vs. Referenz"
        )
    total_elapsed = time.perf_counter() - total_start

    ka1_output: dict[str, Any] = {}
    for label, spec in boundaries.items():
        ka1_output[label] = {
            **spec,
            "reused_from": reused_flags[label],
            "decoupled": label in RECOMPUTED_LABELS,
            "grid_results": grid_results_by_boundary[label],
        }
        n_infeasible = sum(
            1
            for r in grid_results_by_boundary[label]
            if r.get("solve_failed") or not r.get("feasible", False)
        )
        ka1_output[label]["n_infeasible"] = n_infeasible
        ka1_output[label]["infeasible_points"] = [
            {
                "ambient_temperature_celsius": r["ambient_temperature_celsius"],
                "ambient_relative_humidity": r["ambient_relative_humidity"],
            }
            for r in grid_results_by_boundary[label]
            if r.get("solve_failed") or not r.get("feasible", False)
        ]

    print("\n" + "=" * 88)
    print("KA2 (entkoppelt): Kosten und verbleibender Regelungsgewinn")
    print("=" * 88)
    ka2_output: dict[str, Any] = {}
    for label in boundaries:
        result = ka1.compute_ka2_for_boundary(
            grid_results_by_boundary[label],
            r_reference["grid_results"],
            r_reference["fixed_point"]["per_grid_point"],
            binding_at_reference,
        )
        ka2_output[label] = result
        agg = result["aggregate"]["all"]
        print(
            f"\n{label} (n_feasible={agg['n_feasible']}/{agg['n_points']}): "
            f"Median-Kosten={agg['median_cost_rel_pct']}%  Gepoolte-Kosten={agg['pooled_cost_rel_pct']}%  "
            f"Median-verbleibend={agg['median_remaining_fraction']}  Gepoolt-verbleibend={agg['pooled_remaining_fraction']}"
        )
        for group_name in ("binding_at_reference", "non_binding_at_reference"):
            g = result["aggregate"][group_name]
            print(
                f"    {group_name:24s} n_feasible={g['n_feasible']}/{g['n_points']}  "
                f"Gepoolte-Kosten={g['pooled_cost_rel_pct']}%  Gepoolt-verbleibend={g['pooled_remaining_fraction']}"
            )

    print("\n" + "=" * 88)
    print("KA3 (entkoppelt): Gegenprobe zum RL-Agenten")
    print("=" * 88)
    agent_all = ka2_output["rl_agent_margin"]["aggregate"]["all"]
    print(
        f"Diese Studie (entkoppelt), rl_agent_margin, gepoolt (alle 9 Punkte): {agent_all['pooled_remaining_fraction']}"
    )
    print(
        f"Befund 62 FA3, RL-Agent, gepoolt (25 Punkte, Schritt 10000, Median ueber 3 Seeds): "
        f"{fa3_reference['pooled_closed_regret_fraction_step10000_median']}"
    )

    output: dict[str, Any] = {
        "environment": env,
        "config_path": str(arguments.config),
        "base_config_fingerprint": base_config.fingerprint,
        "grid_size": ka1.GRID_SIZE,
        "budget": budget_dict,
        "critical_moisture_reference_kg_per_kg": ka1.CRITICAL_MOISTURE_REFERENCE_KG_PER_KG,
        "decoupling_note": (
            "material.critical_moisture_kg_per_kg bleibt 0.20 (Kinetik unveraendert, "
            "identischer Fingerprint) fuer ALLE Grenzen dieses Laufs -- nur die an "
            "check_constraints uebergebene Schwelle variiert. Siehe Befund 64 Nachtrag "
            "und scripts/probe_margin_energy_cost.py (KA1, dort war die Kinetik "
            "mitverschoben)."
        ),
        "margin_sources": margins,
        "fa3_reference": fa3_reference,
        "binding_moisture_at_reference": binding_at_reference,
        "ka1_boundaries": ka1_output,
        "ka2_cost_and_remaining_fraction": ka2_output,
        "ka3_rl_agent_counterfactual": {
            "boundary_label": "rl_agent_margin",
            "this_study_pooled_remaining_fraction_all_points": agent_all[
                "pooled_remaining_fraction"
            ],
            "fa3_pooled_closed_regret_fraction_step10000_median": fa3_reference[
                "pooled_closed_regret_fraction_step10000_median"
            ],
        },
        "total_elapsed_s": total_elapsed,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nGesamtlaufzeit (nur neu gerechnete Grenzen): {total_elapsed / 60.0:.1f} min")
    print(f"Ergebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

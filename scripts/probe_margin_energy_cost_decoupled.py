r"""MA2: KA1 entkoppelt wiederholen -- Trocknungskinetik unveraendert (0,20 kg/kg), nur die
Schwelle der Qualitaets-Nebenbedingung verschoben.

Anlass (MA1, siehe Befund 64 Nachtrag): `material.critical_moisture_kg_per_kg` hat im Modell
eine DOPPELROLLE. Es ist (a) `X_cr` in `thermo.normalised_moisture` -- der Nenner der
normierten Trocknungskurve, `span = X_cr - X_eq`, geht direkt in `evaporation_rate_kg_per_s`
ein, also in die Physik von `solve_steady_state` selbst (Kommentar in `config.py`:
"Trocknungskinetik: normierte Trocknungskurve, linear fallende Rate") -- UND (b) die
Schwelle der Qualitaets-Nebenbedingung in `study_operating_point_regret.py::check_
constraints` (dort ein GEWOEHNLICHES Funktionsargument, unabhaengig von der Config).
`probe_margin_energy_cost.py` (KA1) hat wie `probe_critical_moisture_sensitivity.py`
(Befund 54 DA1) `material.critical_moisture_kg_per_kg` selbst variiert -- damit beide Rollen
gleichzeitig verschoben, nicht nur die Qualitaetsgrenze. Bei der RL-Agenten-Marge
(0,0605 kg/kg) erschien deshalb eine reine Kinetik-Validierung (`sorption.py::validate_
isotherm_against_critical_moisture`) als "unzulaessig", bevor irgendein Punkt geloest wurde
-- obwohl Befund 62 GA1 im SELBEN Modell (Kinetik unveraendert bei 0,20 kg/kg) tatsaechlich
Produktfeuchten von 0,055-0,070 kg/kg gemessen hat.

Dieses Skript entkoppelt die beiden Rollen, OHNE `study_operating_point_regret.py` zu
aendern (weiterhin per `importlib` geladen, keine Zeile dort veraendert) und OHNE die
ProcessConfig zu aendern (`material.critical_moisture_kg_per_kg` bleibt durchgehend 0,20 kg/kg
-- derselbe Fingerprint wie Befund 52/54/KA1s Basis-Config). Nur der Wert, der an `check_
constraints(..., critical_moisture_kg_per_kg=...)` uebergeben wird -- ein gewoehnliches
Funktionsargument dieser Funktion, unabhaengig von `_BASE_CONFIG` -- wird auf die jeweilige
effektive Grenze gesetzt.

Technisch: `study_operating_point_regret.py`s eigene Orchestrierung (`_penalized_energy`,
`optimize_grid_point`, `_fixed_point_worker`) liest die Schwelle IMMER aus `_BASE_CONFIG.
material.critical_moisture_kg_per_kg` -- fest verdrahtet, kein Injektionspunkt. Diese drei
Funktionen werden deshalb hier NICHT verwendet; stattdessen komponiert dieses Skript
`evaluate_point`, `check_constraints`, `_objective_components` und `_optimize` (alle vier
unveraendert aus dem Modul importiert) zu einer eigenen, hier lebenden Zielfunktion, die die
Schwelle als expliziten Parameter nimmt -- dasselbe Baukasten-Prinzip wie `_OBJECTIVE_VALUE_FN`
in `study_operating_point_regret_net.py` (MA3/MA4), nur fuer die Nebenbedingung statt die
Zielgroesse.

Nur BB2 (instanzweise Optima), wie KA1 -- kein BB3(a). Der feste Betriebspunkt (`E_fest`) und
das Orakel bei der wahren Grenze (`E_Orakel(0,20)`, "oracle_anchor") sind von der Doppelrolle
NICHT betroffen: bei Marge 0 fallen Kinetik-Parameter und Schwelle zusammen, es gibt keinen
Unterschied zur entkoppelten Rechnung -- beide bleiben unveraendert aus Befund 52/54 (DA1)
wiederverwendet. Die uebrigen sechs Grenzen (beide Softsensor-Margen, 0,18/0,15/0,12,
Agentenmarge) werden hier neu gerechnet -- anders als in KA1 auch 0,15 kg/kg, weil DA1s
0,15-Punkt ebenfalls die Kinetik mitverschoben hatte.

Wiederverwendet aus `probe_margin_energy_cost.py` (KA1, unveraendert): Margen-Herkunft
(`load_margin_sources`), Grenzendefinition (`build_boundary_definitions`), FA3-Referenz
(`load_fa3_reference`), KA2-Aggregation (`compute_ka2_for_boundary`, `_pooled`,
`_median_or_none`), Gitter-/Budget-Abgleich (`_assert_same_grid`, `_assert_same_budget`,
`_grid_coordinates`). Nur der Gitterpunkt-Solver ist neu.

Aufruf::

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
    """Wie `study_operating_point_regret.py::_penalized_energy`, aber mit ENTKOPPELTER
    Schwelle: `evaluate_point` loest gegen `regret_module._BASE_CONFIG` (Kinetik, immer
    0,20 kg/kg), `check_constraints` prueft gegen `effective_critical_moisture_kg_per_kg`
    (die Marge dieses Laufs) -- zwei verschiedene Werte, wo `_penalized_energy` nur einen
    kennt. Modul-Ebene fuer Picklebarkeit (Windows-`spawn`) waere hier nicht noetig (nur
    innerhalb dieses Skripts als Closure-Ersatz gebraucht), lebt aber trotzdem auf
    Modulebene, weil `_optimize` sie an `differential_evolution`/`minimize` durchreicht,
    die -- je nach `workers` -- ebenfalls pickeln koennen."""
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
    """Top-Level-Wrapper (picklebar unter `spawn`) -- `regret_module` wird ueber das
    Modul-Global `_REGRET_MODULE` aufgeloest, das der Pool-Initializer in jedem Worker
    setzt (dasselbe Muster wie `regret_module._BASE_CONFIG`)."""
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
    """Pool-Initializer: laedt `study_operating_point_regret.py` in JEDEM Worker frisch
    (unabhaengig von der Startmethode, siehe `_load_regret_module`-Docstring dort) und setzt
    `_BASE_CONFIG` -- identisch zu `regret_module._set_base_config`, nur zusaetzlich das
    Modul-Global `_REGRET_MODULE` in diesem Skript gesetzt, damit die Top-Level-Zielfunktion
    es findet."""
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
    """Wie `study_operating_point_regret.py::optimize_grid_point`, aber mit entkoppelter
    Schwelle -- siehe `_penalized_energy_decoupled`. Ergebnis-Schema bewusst identisch zu
    `optimize_grid_point`s Rueckgabe, damit `probe_margin_energy_cost.compute_ka2_for_
    boundary` unveraendert weiterverwendet werden kann."""
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
    """`regret_module` wird vom Aufrufer durchgereicht statt hier neu geladen: jeder Aufruf
    von `_load_regret_module()` ersetzt `sys.modules["study_operating_point_regret"]` durch
    ein FRISCHES Modulobjekt (siehe dortigen Docstring) -- ein `OptimizationBudget`, das
    gegen einen frueheren Ladevorgang erzeugt wurde, laesst sich danach nicht mehr pickeln
    (`PicklingError: nicht dasselbe Objekt`, empirisch im Smoketest aufgetreten). Im
    Hauptprozess wird das Modul deshalb GENAU EINMAL geladen (`main()`), hier nur
    weiterverwendet."""
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
    # Kinetik bleibt UNVERAENDERT bei 0,20 kg/kg -- derselbe Fingerprint wie Befund 52/54/KA1.
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

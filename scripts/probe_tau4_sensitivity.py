r"""DA2: haengen Befund 52/53 an der Feststoff-Verweilzeit ``tau_4`` (Bettmasse/Produktabzug).

Anlass: ``tau_4 = bed_mass_solid_setpoint_kg / solid_inflow`` betraegt im aktuellen Modell
16000 s (4.4 h, siehe ``docs/04_stand_ap1.md``, geschlossene Form aus Befund 47/M1).
``geometry.bed_mass_solid_setpoint_kg`` ist ein ungepruefter Platzhalter (Dateikopf
``conf/process/greybox_gab_placeholder.yaml``, TODO "SPE: Geometrie und Bettmasse der
Laboranlage"). Befund 53 fand, dass die Charge (28800 s) den Beharrungszustand bei diesem
tau_4 nur zu 83.5% erreicht und die chargengemittelte rueckgewinnbare Leistung Q_rec um 36%
(Median) vom stationaeren Wert abweicht. Dieses Skript misst, wie stark DIESE Befunde von
``tau_4`` abhaengen.

Der einzige Knopf, um ``tau_4`` zu aendern, ohne den Feststoffzulauf selbst zu veraendern
(``spray_rate`` ist laut Befund 52 bewusst FEST, siehe dortiger Docstring), ist
``geometry.bed_mass_solid_setpoint_kg`` -- siehe ``_product_mass_flow`` in
``src/twinlink_ml/simulation/greybox.py`` (woertlich unten zitiert) und die geschlossene
Form in ``diagnostics._solid_residence_time_s``:

    tau_4 = bed_mass_solid_setpoint_kg / (spray_rate_kg_per_h/3600 * solid_fraction + seed_flow)

Geprueft (Nutzerrueckmeldung, vor diesem Lauf eingeholt): am Nominalpunkt bleiben d50,
Fluidisierungsfenster (u_mf/u_terminal) und Gutsfeuchte EXAKT gleich, wenn NUR
``bed_mass_solid_setpoint_kg`` variiert wird (solid_inflow unveraendert) -- die Kopplung ist
strukturell, nicht numerisches Rauschen. ABER: die Bettmasse geht linear in den
Schuettbett-Druckverlust ein (``bed_pressure_drop_pa``, ``simulation/energy.py``) und damit
in die Ventilatorleistung -- am Nominalpunkt schwankt ``power_total_kw`` dadurch um bis zu
4.6% ueber den vollen ``tau_4``-Bereich (4000..32000 s). Diese Kopplung ist reale
Modellphysik (mehr Bettmasse -> mehr Druckverlust), kein Artefakt, und wird hier als
NIVEAUVERSCHIEBUNG explizit mitberichtet (absolutes Energieniveau je Wert), nicht
verschwiegen. Der Regret ist eine Differenz zweier Energien BEIM GLEICHEN ``tau_4`` --
der gemeinsame Niveauversatz kuerzt sich darin weitgehend heraus; zur Kontrolle wird der
relative Regret (bereits energie-normiert) mitgefuehrt.

DA2(a) -- Stationaerer Regret je ``tau_4``
--------------------------------------------
Wiederholt BB1-BB3 aus ``study_operating_point_regret.py`` (per ``importlib`` geladen,
keine Zeile dort veraendert) fuer vier ``tau_4``-Werte: 4000, 8000, 16000 (aktuell),
32000 s, auf einem 3x3- statt 5x5-Gitter (Laufzeitgrund, siehe Auftrag). Fuer den
aktuellen Wert (16000 s, Config byte-identisch zur Basis-Config) wird zusaetzlich das
committete 5x5-Ergebnis aus Befund 52 als Anker verglichen.

DA2(b)/(c) -- Chargenrollout je ``tau_4``
--------------------------------------------
Wiederholt CB1 aus ``study_operating_point_transient.py`` (per ``importlib`` geladen,
keine Zeile dort veraendert) fuer dieselben vier ``tau_4``-Werte: Rollout ueber 28800 s
(Delta_t=25 s, Projektion an) je Gitterpunkt MIT DEN DORT (DA2a) optimalen Sollwerten,
Endzustand gegen den frischen Beharrungszustand (b), ``Q_rec`` Chargenmittel gegen
stationaer (c) -- dieselbe Rechnung wie Befund 53 (CB1), hier ueber vier ``tau_4``-Werte
statt einem.

Aufruf::

    python scripts/probe_tau4_sensitivity.py \
        --output artifacts/runs/probe_tau4_sensitivity.json
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from twinlink_ml.simulation import PlaceholderConfigWarning, ProcessConfig, load_process_config
from twinlink_ml.utils.fingerprint_history import fingerprints_equivalent
from twinlink_ml.utils.tracking import environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
REGRET_SCRIPT_PATH = REPO_ROOT / "scripts" / "study_operating_point_regret.py"
TRANSIENT_SCRIPT_PATH = REPO_ROOT / "scripts" / "study_operating_point_transient.py"
ANCHOR_5X5_PATH = REPO_ROOT / "artifacts" / "runs" / "bb1_bb3_operating_point_regret.json"
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "probe_tau4_sensitivity.json"

TAU4_VALUES_S = (4000.0, 8000.0, 16000.0, 32000.0)
CURRENT_TAU4_S = 16000.0
DEFAULT_GRID_SIZE = 3

EXPECTED_BASE_FINGERPRINT = "2acd06f38963c3af"
"""Muss mit ``config_fingerprint`` in ``bb1_bb3_operating_point_regret.json`` (Befund 52,
5x5-Anker) uebereinstimmen."""


def _load_module(name: str, path: Path) -> ModuleType:
    """Wie ``study_operating_point_transient.py::_load_regret_module`` -- ``sys.modules``-
    Eintrag VOR ``exec_module`` noetig, weil Dataclass-Felder ueber
    ``sys.modules[cls.__module__]`` aufgeloest werden (Python 3.12)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _suppress_placeholder_warning() -> None:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)


def solid_inflow_kg_per_s(config: ProcessConfig) -> float:
    spray_solid = (
        config.actions.spray_rate_kg_per_h.nominal
        / 3600.0
        * (config.disturbances.suspension_solid_fraction.nominal)
    )
    seed = config.operation.seed_mass_flow_kg_per_h / 3600.0
    return spray_solid + seed


def build_variant_config(base: ProcessConfig, *, tau4_s: float) -> tuple[ProcessConfig, float]:
    """``geometry.bed_mass_solid_setpoint_kg`` so gesetzt, dass ``tau_4`` (geschlossene
    Form) exakt ``tau4_s`` ergibt -- solid_inflow (spray_rate, solid_fraction, seed)
    bleibt unveraendert (siehe Moduldocstring, Kopplungspruefung)."""
    inflow = solid_inflow_kg_per_s(base)
    setpoint_kg = tau4_s * inflow
    geometry = dataclasses.replace(base.geometry, bed_mass_solid_setpoint_kg=setpoint_kg)
    return dataclasses.replace(base, geometry=geometry), setpoint_kg


def run_regret_study(
    regret_module: ModuleType,
    variant_config: ProcessConfig,
    *,
    grid_size: int,
    workers: int,
    label: str,
) -> dict[str, Any]:
    """Identisch zu ``probe_critical_moisture_sensitivity.py::run_regret_study`` --
    BB1-BB3-Orchestrierung nach dem Muster von ``study_operating_point_regret.main()``,
    keine Budget-/Toleranzkonstante der wiederverwendeten Funktionen veraendert."""
    regret_module._set_base_config(variant_config)
    budget = regret_module.OptimizationBudget(
        seed=0,
        popsize=regret_module.DEFAULT_POPSIZE,
        maxiter=regret_module.DEFAULT_MAXITER,
        de_tol=regret_module.DEFAULT_DE_TOL,
        search_max_iterations=regret_module.DEFAULT_SEARCH_MAX_ITERATIONS,
        search_tolerance=regret_module.DEFAULT_SEARCH_TOLERANCE,
        workers=workers,
    )
    bounds = regret_module.action_bounds(variant_config)
    grid_points = regret_module.build_grid(variant_config, grid_size)

    start = time.perf_counter()
    print(
        f"\n=== {label}: Gitter {grid_size}x{grid_size} = {len(grid_points)} Punkte (BB1-BB3) ===",
        flush=True,
    )

    with Pool(workers) as pool:
        tasks = [
            (index, t_amb, phi_amb, bounds, budget)
            for index, (t_amb, phi_amb) in enumerate(grid_points)
        ]
        grid_results_by_index: dict[int, dict[str, Any]] = {}
        for n_done, result in enumerate(
            pool.imap_unordered(regret_module._grid_point_task, tasks), start=1
        ):
            index = result["_grid_index"]
            print(
                f"  [{n_done:2d}/{len(grid_points)}] Punkt {index + 1} "
                f"({result['_wall_elapsed_s']:5.1f}s)",
                flush=True,
            )
            grid_results_by_index[index] = result
        grid_results = [grid_results_by_index[i] for i in range(len(grid_points))]

        output: dict[str, Any] = {
            "label": label,
            "config_fingerprint": variant_config.fingerprint,
            "grid_size": grid_size,
            "budget": dataclasses.asdict(budget),
            "grid_results": grid_results,
        }

        fixed_point = regret_module.find_fixed_operating_point(grid_points, bounds, budget, pool)
        output["fixed_point"] = fixed_point
        if fixed_point["feasible_everywhere"]:
            output["regret"] = regret_module.compute_regret(grid_results, fixed_point)
            output["setpoint_spread"] = regret_module.compute_setpoint_spread(
                grid_results, variant_config
            )
        else:
            output["fixed_point_verification"] = (
                regret_module.verify_no_feasible_fixed_point_exists(
                    grid_points, bounds, budget, pool
                )
            )

    output["elapsed_s"] = time.perf_counter() - start
    regret = output.get("regret")
    if regret and regret.get("n_points", 0) > 0:
        print(
            f"  Regret: Median={regret['relative_pct']['median']:.3f}%  "
            f"Max={regret['relative_pct']['max']:.3f}%  | {output['elapsed_s']:.0f}s",
            flush=True,
        )
    else:
        print(
            f"  Kein ueberall zulaessiger fester Betriebspunkt | {output['elapsed_s']:.0f}s",
            flush=True,
        )
    return output


def run_transient_study(
    regret_module: ModuleType,
    transient_module: ModuleType,
    variant_config: ProcessConfig,
    stationary_result: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """CB1 aus ``study_operating_point_transient.py`` nachgebildet: je Gitterpunkt ein
    28800s-Rollout (Delta_t=25s, Projektion an) mit den dort (DA2a) stationaer optimalen
    Sollwerten, Endzustand + Q_rec gegen den frischen Beharrungszustand."""
    transient_config = transient_module.build_transient_config(variant_config)
    spray_rate = variant_config.actions.spray_rate_kg_per_h.nominal
    solid_fraction = variant_config.disturbances.suspension_solid_fraction.nominal

    start = time.perf_counter()
    print(f"\n=== {label}: {len(stationary_result['grid_results'])} Rollouts (CB1) ===", flush=True)

    per_point: list[dict[str, Any]] = []
    for grid_result in stationary_result["grid_results"]:
        if grid_result.get("solve_failed"):
            continue
        point = transient_module.GridPoint(
            index=grid_result["_grid_index"],
            ambient_temperature_celsius=grid_result["ambient_temperature_celsius"],
            ambient_relative_humidity=grid_result["ambient_relative_humidity"],
            inlet_temperature_celsius=grid_result["inlet_temperature_celsius"],
            gas_velocity_m_per_s=grid_result["gas_velocity_m_per_s"],
            nozzle_pressure_bar=grid_result["nozzle_pressure_bar"],
            steady_state_energy_target_kw=grid_result["energy_target"],
            steady_state_feasible=grid_result["feasible"],
        )
        steady_state = transient_module.compute_steady_state_reference(
            regret_module, variant_config, point
        )
        trajectory = transient_module.rollout_constant_setpoint(
            transient_config,
            t_in=point.inlet_temperature_celsius,
            u_gas=point.gas_velocity_m_per_s,
            p_nozzle=point.nozzle_pressure_bar,
            t_amb=point.ambient_temperature_celsius,
            phi_amb=point.ambient_relative_humidity,
            spray_rate=spray_rate,
            solid_fraction=solid_fraction,
            n_steps=transient_module.N_STEPS,
            run_id=f"{label}_{point.index:02d}",
        )
        energy = trajectory.outputs[0, :, 0]
        q_rec = transient_module.q_rec_series_kw(transient_config, trajectory)

        end_solid_mass = float(trajectory.states[0, -1, 0])
        end_d50_um = float(trajectory.outputs[0, -1, 3])
        end_moisture = float(trajectory.outputs[0, -1, 2])
        end_bed_temperature = float(trajectory.outputs[0, -1, 1])

        deviation_rel_pct = {
            "solid_mass_kg": transient_module._rel_pct(end_solid_mass, steady_state.solid_mass_kg),
            "d50_um": transient_module._rel_pct(end_d50_um, steady_state.d50_um),
            "particle_moisture_kg_per_kg": transient_module._rel_pct(
                end_moisture, steady_state.particle_moisture_kg_per_kg
            ),
            "bed_temperature_celsius": transient_module._rel_pct(
                end_bed_temperature, steady_state.bed_temperature_celsius
            ),
        }
        q_rec_batch_mean = float(q_rec.mean())
        q_rec_end = float(q_rec[-1])
        per_point.append(
            {
                "grid_index": point.index,
                "ambient_temperature_celsius": point.ambient_temperature_celsius,
                "ambient_relative_humidity": point.ambient_relative_humidity,
                "steady_state_reference": dataclasses.asdict(steady_state),
                "end_of_batch": {
                    "solid_mass_kg": end_solid_mass,
                    "d50_um": end_d50_um,
                    "particle_moisture_kg_per_kg": end_moisture,
                    "bed_temperature_celsius": end_bed_temperature,
                    "energy_target_kw": float(energy[-1]),
                    "q_rec_kw": q_rec_end,
                },
                "batch_mean": {
                    "energy_target_kw": float(energy.mean()),
                    "q_rec_kw": q_rec_batch_mean,
                },
                "deviation_end_vs_steady_state_relative_pct": deviation_rel_pct,
                "q_rec_deviation": {
                    "batch_mean_vs_steady_state_abs_kw": q_rec_batch_mean - steady_state.q_rec_kw,
                    "batch_mean_vs_steady_state_relative_pct": transient_module._rel_pct(
                        q_rec_batch_mean, steady_state.q_rec_kw
                    ),
                    "end_vs_steady_state_relative_pct": transient_module._rel_pct(
                        q_rec_end, steady_state.q_rec_kw
                    ),
                },
            }
        )

    def _agg(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values)
        return {"median": float(np.median(arr)), "max": float(np.max(np.abs(arr)))}

    channels = ("solid_mass_kg", "d50_um", "particle_moisture_kg_per_kg", "bed_temperature_celsius")
    state_channel_summary = {
        channel: _agg([p["deviation_end_vs_steady_state_relative_pct"][channel] for p in per_point])
        for channel in channels
    }
    q_rec_summary = {
        "batch_mean_vs_steady_state_relative_pct": _agg(
            [p["q_rec_deviation"]["batch_mean_vs_steady_state_relative_pct"] for p in per_point]
        ),
        "batch_mean_vs_steady_state_abs_kw": _agg(
            [p["q_rec_deviation"]["batch_mean_vs_steady_state_abs_kw"] for p in per_point]
        ),
    }

    elapsed = time.perf_counter() - start
    print(
        f"  Q_rec-Abweichung (Chargenmittel vs. stationaer): "
        f"Median={q_rec_summary['batch_mean_vs_steady_state_relative_pct']['median']:.2f}%  "
        f"Max(|.|)={q_rec_summary['batch_mean_vs_steady_state_relative_pct']['max']:.2f}%  "
        f"| {elapsed:.0f}s",
        flush=True,
    )
    return {
        "label": label,
        "transient_config_fingerprint": transient_config.fingerprint,
        "batch_duration_s": transient_module.BATCH_DURATION_S,
        "sample_time_s": transient_module.SAMPLE_TIME_S,
        "n_steps": transient_module.N_STEPS,
        "per_grid_point": per_point,
        "state_channel_deviation_relative_pct": state_channel_summary,
        "q_rec_deviation_summary": q_rec_summary,
        "elapsed_s": elapsed,
    }


def load_anchor_5x5() -> dict[str, Any]:
    anchor = json.loads(ANCHOR_5X5_PATH.read_text())
    if not fingerprints_equivalent(anchor["config_fingerprint"], EXPECTED_BASE_FINGERPRINT):
        raise RuntimeError(
            f"{ANCHOR_5X5_PATH} hat config_fingerprint {anchor['config_fingerprint']!r}, "
            f"erwartet {EXPECTED_BASE_FINGERPRINT!r}."
        )
    return {
        "source": str(ANCHOR_5X5_PATH),
        "grid_size": anchor["grid_size"],
        "config_fingerprint": anchor["config_fingerprint"],
        "regret": anchor["regret"],
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--values", type=float, nargs="+", default=list(TAU4_VALUES_S))
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--workers", type=int, default=-1, help="-1 = alle Kerne")
    parser.add_argument(
        "--skip-transient",
        action="store_true",
        help="nur DA2(a) (stationaerer Regret), kein Chargenrollout -- fuer Laufzeitnot",
    )
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    _suppress_placeholder_warning()

    env = environment_fingerprint()
    if env["git/dirty"]:
        print("WARNUNG: Arbeitsbaum ist dirty.", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base_config = load_process_config(arguments.config)
    if not fingerprints_equivalent(base_config.fingerprint, EXPECTED_BASE_FINGERPRINT):
        raise RuntimeError(
            f"Basis-Config hat Fingerprint {base_config.fingerprint!r}, erwartet "
            f"{EXPECTED_BASE_FINGERPRINT!r} -- weicht von Befund 52 ab."
        )

    regret_module = _load_module("study_operating_point_regret", REGRET_SCRIPT_PATH)
    transient_module = _load_module("study_operating_point_transient", TRANSIENT_SCRIPT_PATH)
    workers = arguments.workers if arguments.workers > 0 else cpu_count()

    print("=" * 78)
    print("DA2: Sensitivitaet gegen tau_4 (Bettmasse-Sollwert / Feststoffzulauf)")
    print(f"Config       {arguments.config}  (Fingerprint {base_config.fingerprint})")
    print(f"tau_4-Werte  {arguments.values} s")
    print(f"Gitter       {arguments.grid_size}x{arguments.grid_size}")
    print(f"Worker       {workers}")
    print(f"git/commit={env['git/commit']}  git/dirty={env['git/dirty']}")
    print("=" * 78, flush=True)

    total_start = time.perf_counter()
    results: dict[str, dict[str, Any]] = {}
    for tau4 in arguments.values:
        variant_config, setpoint_kg = build_variant_config(base_config, tau4_s=tau4)
        label = f"tau4={tau4:.0f}s"
        stationary = run_regret_study(
            regret_module,
            variant_config,
            grid_size=arguments.grid_size,
            workers=workers,
            label=label,
        )
        stationary["bed_mass_solid_setpoint_kg"] = setpoint_kg
        stationary["solid_inflow_kg_per_s"] = solid_inflow_kg_per_s(variant_config)
        entry: dict[str, Any] = {"stationary": stationary}
        if not arguments.skip_transient and stationary.get("regret") is not None:
            entry["transient"] = run_transient_study(
                regret_module, transient_module, variant_config, stationary, label=label
            )
        results[f"{tau4:.0f}"] = entry
    total_elapsed = time.perf_counter() - total_start

    anchor_comparison = None
    current_key = f"{CURRENT_TAU4_S:.0f}"
    if current_key in results:
        anchor = load_anchor_5x5()
        new_regret = results[current_key]["stationary"].get("regret")
        anchor_comparison = {
            "anchor_5x5": anchor,
            "new_3x3_at_current_value": {
                "grid_size": results[current_key]["stationary"]["grid_size"],
                "regret": new_regret,
            },
        }
        if new_regret and new_regret.get("n_points", 0) > 0:
            anchor_comparison["median_relative_pct_diff"] = (
                new_regret["relative_pct"]["median"] - anchor["regret"]["relative_pct"]["median"]
            )
        print("\n" + "=" * 78)
        print("Anker-Vergleich am aktuellen Wert (tau_4=16000s): 3x3 (neu) vs. 5x5 (Befund 52)")
        if new_regret and new_regret.get("n_points", 0) > 0:
            print(
                f"  5x5 Median={anchor['regret']['relative_pct']['median']:.3f}%  "
                f"3x3 Median={new_regret['relative_pct']['median']:.3f}%  "
                f"Differenz={anchor_comparison['median_relative_pct_diff']:+.3f} Prozentpunkte"
            )
        print("=" * 78, flush=True)

    output: dict[str, Any] = {
        "environment": env,
        "config_path": str(arguments.config),
        "base_config_fingerprint": base_config.fingerprint,
        "grid_size": arguments.grid_size,
        "tau4_values_s": arguments.values,
        "results_by_value": results,
        "anchor_comparison": anchor_comparison,
        "total_elapsed_s": total_elapsed,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nGesamtlaufzeit: {total_elapsed / 60.0:.1f} min")
    print(f"Ergebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

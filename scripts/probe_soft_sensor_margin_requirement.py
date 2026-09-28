r"""GA1: Zwei zusaetzliche Stuetzstellen (0,19 / 0,185 kg/kg) fuer die 90-%-Interpolation aus
Befund 64s entkoppeltem KA2-Ergebnis, und die Interpolation selbst (90 / 50 / 0 % Anteil des
Regelungsgewinns).

Anlass: neue Frage -- welche Genauigkeit braucht der Feuchte-Soft-Sensor, damit der
Regelungsgewinn erhalten bleibt, und erreicht er sie im Bereich, in dem das Optimum liegt?
Befund 64s entkoppeltes KA2-Ergebnis (`artifacts/runs/probe_margin_energy_cost_decoupled.json`,
committed `dc5c1c4`/`daccb4f`) gibt den gepoolten verbleibenden Anteil des Regelungsgewinns an
sieben Margen. Sortiert nach Marge:

    Marge [kg/kg]   verbleibend
    0.0              1.0000   (oracle_anchor)
    0.0016282        0.9798   (excited_lstm_3rmse)
    0.0044651        0.9535   (constant_lstm_3rmse)
    0.02             0.7837   (grid_0.18)
    0.05             0.3575   (grid_0.15)
    0.08            -0.1980   (grid_0.12)
    0.1395451       -2.2371   (rl_agent_margin)

Die 90-%-Marke (linear interpoliert) faellt zwischen der Marge des konstant trainierten
Softsensors (0,0044651 kg/kg, 95,35 % verbleibend) und dem Gitterpunkt bei 0,02 kg/kg
(78,37 % verbleibend) -- eine Luecke von 0,0155 kg/kg Marge ohne Stuetzstelle dazwischen, und
genau der Fall, den der Auftrag vorwegnimmt ("Stuetzstellen liegen zwischen 0.1955 und 0.18
kg/kg"). Dieses Skript rechnet zwei weitere Stuetzstellen (`critical_moisture_kg_per_kg` =
0,19 und 0,185, Margen 0,01 und 0,015 kg/kg) mit DERSELBEN entkoppelten Mechanik wie Befund 64
MA2 (`probe_margin_energy_cost_decoupled.py`, unveraendert importiert -- keine Zeile dort
geaendert) und interpoliert danach linear ueber alle neun Stuetzstellen (sieben aus MA2, zwei
neu) fuer die 90-/50-/0-%-Marken.

Materialkonfiguration unveraendert: `material.critical_moisture_kg_per_kg` bleibt 0,20 kg/kg
(derselbe Fingerprint wie Befund 52/54/62/64) -- nur die an `check_constraints` uebergebene
Schwelle variiert, exakt wie in MA2. Referenz (`E_Orakel(0,20)`, `E_fest`,
`binding_at_reference`) unveraendert aus Befund 54 DA1 wiederverwendet, wie in KA1/MA2.

Laufzeitschaetzung (vor dem Rechnen, wie im Auftrag verlangt): MA2s sechs neu gerechnete
Stuetzstellen brauchten je ~430-510s Wandzeit (9-Punkte-Gitter, 16 Worker parallel, `max`
ueber die 9 Punkte dominiert wegen `Pool.imap_unordered`) -- zwei weitere, aehnlich kleine
Margen also ~15-20 min insgesamt, deutlich unter der 1-2h-Schwelle fuer "lange Laeufe"
(CLAUDE.md).

Nach dem Rechnen: `sigma_max = margin / 3` (dieselbe 3x-RMSE-Konvention wie Befund 62 HA1),
und Reserve-Faktor `sigma_max(90%) / gemessener_Fehler` gegen die in Befund 64
`margin_sources` bereits committeten RMSE-Mediane (angeregt trainiert: 0,000542738535806486
kg/kg; konstant trainiert / aus Betriebsdaten: 0,001488358944129813 kg/kg).

Aufruf::

    python scripts/probe_soft_sensor_margin_requirement.py \
        --output artifacts/runs/probe_soft_sensor_margin_requirement.json
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import time
import warnings
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import probe_margin_energy_cost as ka1
import probe_margin_energy_cost_decoupled as ma2

from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.fingerprint_history import fingerprints_equivalent
from twinlink_ml.utils.tracking import environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "probe_soft_sensor_margin_requirement.json"
MA2_ARTIFACT_PATH = REPO_ROOT / "artifacts" / "runs" / "probe_margin_energy_cost_decoupled.json"

EXTRA_BOUNDARIES: dict[str, dict[str, Any]] = {
    "grid_0.19": {
        "margin_kg_per_kg": ka1.CRITICAL_MOISTURE_REFERENCE_KG_PER_KG - 0.19,
        "critical_moisture_kg_per_kg": 0.19,
        "margin_source": (
            "GA1-Interpolationsstuetze (dieses Skript) -- fuellt die Luecke zwischen "
            "Befund 64s constant_lstm_3rmse- (0,19553 kg/kg) und grid_0.18-Stuetzstelle."
        ),
    },
    "grid_0.185": {
        "margin_kg_per_kg": ka1.CRITICAL_MOISTURE_REFERENCE_KG_PER_KG - 0.185,
        "critical_moisture_kg_per_kg": 0.185,
        "margin_source": (
            "GA1-Interpolationsstuetze (dieses Skript) -- fuellt die Luecke zwischen "
            "Befund 64s constant_lstm_3rmse- (0,19553 kg/kg) und grid_0.18-Stuetzstelle."
        ),
    },
}

INTERPOLATION_TARGETS = (0.90, 0.50, 0.00)


def _linear_interp_margin(points: list[tuple[float, float]], target_remaining: float) -> dict[str, Any] | None:
    """`points`: (margin, remaining), aufsteigend nach Marge sortiert (= absteigend nach
    verbleibendem Anteil, wie im gesamten Datensatz beobachtet). Sucht das Intervall, in dem
    `target_remaining` liegt, und interpoliert linear. `None`, falls ausserhalb des
    abgedeckten Bereichs."""
    for (m0, r0), (m1, r1) in itertools.pairwise(points):
        if (r0 >= target_remaining >= r1) or (r1 >= target_remaining >= r0):
            if r0 == r1:
                margin = m0
            else:
                t = (r0 - target_remaining) / (r0 - r1)
                margin = m0 + t * (m1 - m0)
            return {
                "target_remaining_fraction": target_remaining,
                "bracket_lower": {"margin_kg_per_kg": m0, "pooled_remaining_fraction": r0},
                "bracket_upper": {"margin_kg_per_kg": m1, "pooled_remaining_fraction": r1},
                "interpolated_margin_kg_per_kg": margin,
            }
    return None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ka1.CONFIG_PATH)
    parser.add_argument("--workers", type=int, default=-1, help="-1 = alle Kerne")
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)

    env = environment_fingerprint()
    if env["git/dirty"]:
        print("WARNUNG: Arbeitsbaum ist dirty -- git/dirty wird im Artefakt vermerkt.", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base_config = load_process_config(arguments.config)
    assert base_config.material.critical_moisture_kg_per_kg == ka1.CRITICAL_MOISTURE_REFERENCE_KG_PER_KG

    da1 = ka1.load_da1_artifact()
    if not fingerprints_equivalent(base_config.fingerprint, da1["base_config_fingerprint"]):
        raise RuntimeError(
            f"Aktuelle Config hat Fingerprint {base_config.fingerprint!r}, DA1-Artefakt "
            f"{da1['base_config_fingerprint']!r} -- keine gemeinsame Basis."
        )
    r_reference = da1["results_by_value"]["0.20"]
    if not r_reference["fixed_point"]["feasible_everywhere"]:
        raise RuntimeError("DA1s Fixpunkt bei 0,20 kg/kg ist nicht ueberall zulaessig.")

    if not MA2_ARTIFACT_PATH.exists():
        raise RuntimeError(f"MA2-Artefakt fehlt: {MA2_ARTIFACT_PATH} -- Befund 64 zuerst reproduzieren.")
    ma2_artifact = json.loads(MA2_ARTIFACT_PATH.read_text())
    if not fingerprints_equivalent(base_config.fingerprint, ma2_artifact["base_config_fingerprint"]):
        raise RuntimeError("Aktuelle Config-Fingerprint weicht vom MA2-Artefakt ab.")

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
    ka1._assert_same_budget(budget_dict, ma2_artifact["budget"], context="Budget vs. MA2-Artefakt")

    reference_grid_coords = ka1._grid_coordinates(r_reference["grid_results"])
    fresh_grid_coords = ka1._grid_coordinates(
        [
            {"ambient_temperature_celsius": t, "ambient_relative_humidity": phi}
            for t, phi in regret_module.build_grid(base_config, ka1.GRID_SIZE)
        ]
    )
    ka1._assert_same_grid(reference_grid_coords, fresh_grid_coords, context="frisches Gitter vs. DA1 0,20 kg/kg")
    binding_at_reference = [bool(r["constraint_check"]["binding_moisture"]) for r in r_reference["grid_results"]]

    print("=" * 88)
    print("GA1: zwei zusaetzliche Stuetzstellen (0,19 / 0,185 kg/kg), entkoppelt wie MA2")
    print(f"Config       {arguments.config}  (Fingerprint {base_config.fingerprint})")
    print(f"Gitter       {ka1.GRID_SIZE}x{ka1.GRID_SIZE} (wie Befund 54 DA1)")
    print(f"Worker       {workers}")
    print(f"git/commit={env['git/commit']}  git/dirty={env['git/dirty']}")
    for label, spec in EXTRA_BOUNDARIES.items():
        print(
            f"  {label:22s} Schwelle={spec['critical_moisture_kg_per_kg']:.6f} kg/kg "
            f"(Marge {spec['margin_kg_per_kg']:.6f} kg/kg)"
        )
    print("=" * 88, flush=True)

    total_start = time.perf_counter()
    grid_results_by_boundary: dict[str, list[dict[str, Any]]] = {}
    for label, spec in EXTRA_BOUNDARIES.items():
        grid_results_by_boundary[label] = ma2.run_oracle_grid_decoupled(
            base_config,
            regret_module,
            grid_size=ka1.GRID_SIZE,
            workers=workers,
            budget=budget,
            effective_critical_moisture_kg_per_kg=spec["critical_moisture_kg_per_kg"],
            label=label,
        )
        new_coords = ka1._grid_coordinates(grid_results_by_boundary[label])
        ka1._assert_same_grid(reference_grid_coords, new_coords, context=f"Gitter {label} vs. Referenz")
    total_elapsed = time.perf_counter() - total_start

    ka1_output: dict[str, Any] = {}
    ka2_output: dict[str, Any] = {}
    print("\n" + "=" * 88)
    print("KA2 (GA1-Stuetzstellen): Kosten und verbleibender Regelungsgewinn")
    print("=" * 88)
    for label, spec in EXTRA_BOUNDARIES.items():
        n_infeasible = sum(
            1 for r in grid_results_by_boundary[label] if r.get("solve_failed") or not r.get("feasible", False)
        )
        ka1_output[label] = {
            **spec,
            "decoupled": True,
            "grid_results": grid_results_by_boundary[label],
            "n_infeasible": n_infeasible,
            "infeasible_points": [
                {
                    "ambient_temperature_celsius": r["ambient_temperature_celsius"],
                    "ambient_relative_humidity": r["ambient_relative_humidity"],
                }
                for r in grid_results_by_boundary[label]
                if r.get("solve_failed") or not r.get("feasible", False)
            ],
        }
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
            f"Gepoolte-Kosten={agg['pooled_cost_rel_pct']}%  Gepoolt-verbleibend={agg['pooled_remaining_fraction']}"
        )

    # Gesamter Datensatz: sieben MA2-Stuetzstellen + zwei neue, aufsteigend nach Marge.
    merged_points: list[tuple[str, float, float]] = []
    for label, spec in ma2_artifact["ka1_boundaries"].items():
        remaining = ma2_artifact["ka2_cost_and_remaining_fraction"][label]["aggregate"]["all"]["pooled_remaining_fraction"]
        merged_points.append((label, spec["margin_kg_per_kg"], remaining))
    for label, spec in EXTRA_BOUNDARIES.items():
        remaining = ka2_output[label]["aggregate"]["all"]["pooled_remaining_fraction"]
        merged_points.append((label, spec["margin_kg_per_kg"], remaining))
    merged_points.sort(key=lambda item: item[1])

    print("\n" + "=" * 88)
    print("Gesamter Datensatz (neun Stuetzstellen, aufsteigend nach Marge)")
    print("=" * 88)
    for label, margin, remaining in merged_points:
        print(f"  {label:22s} Marge={margin:.7f} kg/kg  verbleibend={remaining:.4f}")

    excited_rmse = ma2_artifact["margin_sources"]["excited_rmse_median_kg_per_kg"]
    constant_rmse = ma2_artifact["margin_sources"]["constant_rmse_median_kg_per_kg"]

    interpolation: dict[str, Any] = {}
    xy_points = [(m, r) for _, m, r in merged_points]
    for target in INTERPOLATION_TARGETS:
        hit = _linear_interp_margin(xy_points, target)
        key = f"{target:.2f}"
        if hit is None:
            interpolation[key] = None
            continue
        sigma_max = hit["interpolated_margin_kg_per_kg"] / 3.0
        hit["sigma_max_kg_per_kg"] = sigma_max
        hit["reserve_factor_vs_excited_trained_rmse"] = sigma_max / excited_rmse
        hit["reserve_factor_vs_operating_data_rmse"] = sigma_max / constant_rmse
        interpolation[key] = hit

    print("\n" + "=" * 88)
    print("GA1: Interpolierte Margen / sigma_max / Reserve-Faktoren")
    print("=" * 88)
    for key, hit in interpolation.items():
        if hit is None:
            print(f"  {key}: ausserhalb des abgedeckten Bereichs")
            continue
        print(
            f"  {key}: Marge={hit['interpolated_margin_kg_per_kg']:.6f} kg/kg  "
            f"sigma_max={hit['sigma_max_kg_per_kg']:.6f} kg/kg  "
            f"Reserve(angeregt)={hit['reserve_factor_vs_excited_trained_rmse']:.3f}x  "
            f"Reserve(Betriebsdaten)={hit['reserve_factor_vs_operating_data_rmse']:.3f}x"
        )

    output: dict[str, Any] = {
        "environment": env,
        "config_path": str(arguments.config),
        "base_config_fingerprint": base_config.fingerprint,
        "grid_size": ka1.GRID_SIZE,
        "budget": budget_dict,
        "critical_moisture_reference_kg_per_kg": ka1.CRITICAL_MOISTURE_REFERENCE_KG_PER_KG,
        "ma2_source": str(MA2_ARTIFACT_PATH.relative_to(REPO_ROOT)),
        "extra_boundaries": ka1_output,
        "extra_ka2": ka2_output,
        "merged_dataset_sorted_by_margin": [
            {"label": label, "margin_kg_per_kg": margin, "pooled_remaining_fraction": remaining}
            for label, margin, remaining in merged_points
        ],
        "interpolation_targets": {"90%": 0.90, "50%": 0.50, "0%": 0.00},
        "interpolation": interpolation,
        "rmse_reference": {
            "excited_trained_rmse_median_kg_per_kg": excited_rmse,
            "operating_data_rmse_median_kg_per_kg": constant_rmse,
            "source": "probe_margin_energy_cost_decoupled.json#margin_sources (Befund 64, via rl_bandit_ha1_margin_vs_prediction_error.json)",
        },
        "total_elapsed_s": total_elapsed,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nGesamtlaufzeit (zwei neue Stuetzstellen): {total_elapsed / 60.0:.1f} min")
    print(f"Ergebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

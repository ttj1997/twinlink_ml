r"""DA1: haengt Befund 52 (Regret eines festen Betriebspunkts) an ``critical_moisture_kg_per_kg``.

Anlass: ``material.critical_moisture_kg_per_kg`` (aktuell 0.20 kg/kg, ``conf/process/
greybox_gab_placeholder.yaml``) ist ein Platzhalter ohne Anlagendaten (siehe Dateikopf,
offener TODO "SPE: Stoffsystem, ... kritische Feuchte") und bindet in Befund 52 im kalten
Regime exakt am gefundenen Optimum. Statt den Wert zu erfragen, wird hier gemessen, wie
stark Befund 52 (Regret-Groessenordnung, Bindungsverhalten der Feuchtegrenze, Existenz
eines ueberall zulaessigen festen Betriebspunkts) von ihm abhaengt. Bleibt das Ergebnis
ueber den plausiblen Bereich stabil, ist es staerker als mit einem exakten Wert, weil es
nicht daran haengt -- siehe ``docs/04_stand_ap1.md``, Befund 54.

KEIN Netztraining, KEIN Reinforcement Learning, KEIN neuer Optimierer: dieses Skript
wiederholt exakt BB1-BB3 aus ``study_operating_point_regret.py`` (per ``importlib``
geladen, keine Zeile dort veraendert -- gleiches Muster wie ``study_operating_point_
transient.py``), fuer fuenf Werte von ``critical_moisture_kg_per_kg``:

    0.10, 0.15, 0.20 (aktuell), 0.25, 0.30  kg/kg

Toleranzkonvention: Zulaessigkeits- und Bindungspruefung fuer die Feuchtegrenze verwenden
dieselbe Toleranz (``_TOLERANCE_MOISTURE_KG_PER_KG``, 1e-4 kg/kg) -- aus dem Regret-Modul
uebernommen, hier nicht neu gesetzt (siehe dortiger Docstring fuer die Begruendung).

Gitterreduktion (Laufzeit): Befund 52 kostete ~2.5 h fuer EIN 5x5-Gitter; fuenf Werte
waeren damit grob 12.5 h -- zu teuer fuer eine Sensitivitaetspruefung. Standardgitter hier
daher 3x3 statt 5x5 (``--grid-size``). Fuer den AKTUELLEN Wert (0.20) wird zusaetzlich das
committete 5x5-Ergebnis aus Befund 52 (``artifacts/runs/bb1_bb3_operating_point_regret.
json``) als Anker geladen und mit dem hier neu gerechneten 3x3-Ergebnis am selben Wert
verglichen (Feld ``anchor_comparison``) -- weichen beide deutlich voneinander ab, ist das
VOR der Interpretation der uebrigen vier Werte zu melden (Auftrag).

Aufruf::

    python scripts/probe_critical_moisture_sensitivity.py \
        --output artifacts/runs/probe_critical_moisture_sensitivity.json
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

from twinlink_ml.simulation import PlaceholderConfigWarning, ProcessConfig, load_process_config
from twinlink_ml.utils.fingerprint_history import fingerprints_equivalent
from twinlink_ml.utils.tracking import environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
REGRET_SCRIPT_PATH = REPO_ROOT / "scripts" / "study_operating_point_regret.py"
ANCHOR_5X5_PATH = REPO_ROOT / "artifacts" / "runs" / "bb1_bb3_operating_point_regret.json"
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "probe_critical_moisture_sensitivity.json"

CRITICAL_MOISTURE_VALUES_KG_PER_KG = (0.10, 0.15, 0.20, 0.25, 0.30)
CURRENT_VALUE_KG_PER_KG = 0.20
DEFAULT_GRID_SIZE = 3

EXPECTED_BASE_FINGERPRINT = "2acd06f38963c3af"
"""Muss mit ``config_fingerprint`` in ``bb1_bb3_operating_point_regret.json`` (Befund 52,
5x5-Anker) uebereinstimmen -- sonst ist der Anker nicht am selben Ausgangspunkt gerechnet."""


def _load_regret_module() -> ModuleType:
    """Wie ``study_operating_point_transient.py::_load_regret_module`` -- siehe dort fuer
    die Begruendung des ``sys.modules``-Eintrags vor ``exec_module`` (Dataclass-
    Feldaufloesung ueber ``sys.modules[cls.__module__]``)."""
    spec = importlib.util.spec_from_file_location(
        "study_operating_point_regret", REGRET_SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _suppress_placeholder_warning() -> None:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)


def build_variant_config(
    base: ProcessConfig, *, critical_moisture_kg_per_kg: float
) -> ProcessConfig:
    material = dataclasses.replace(
        base.material, critical_moisture_kg_per_kg=critical_moisture_kg_per_kg
    )
    return dataclasses.replace(base, material=material)


def run_regret_study(
    regret_module: ModuleType,
    variant_config: ProcessConfig,
    *,
    grid_size: int,
    workers: int,
    label: str,
) -> dict[str, Any]:
    """BB1-BB3 fuer EINE Config, EIN Gittergroesse -- Orchestrierung von ``main()`` aus
    ``study_operating_point_regret.py`` nachgebildet (gleiche Funktionen, gleiche
    Reihenfolge), nur ohne dessen argparse/CLI-Druck. Kein Aufruf-Parameter der
    wiederverwendeten Funktionen wird geaendert (Budget = Modul-Defaults)."""
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
        f"\n=== {label}: Gitter {grid_size}x{grid_size} = {len(grid_points)} Punkte ===", flush=True
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
            verification = regret_module.verify_no_feasible_fixed_point_exists(
                grid_points, bounds, budget, pool
            )
            output["fixed_point_verification"] = verification

    n_binding_moisture = sum(
        1
        for r in grid_results
        if not r.get("solve_failed")
        and r.get("constraint_check", {}).get("binding_moisture", False)
    )
    output["n_binding_moisture"] = n_binding_moisture
    output["n_grid_points"] = len(grid_results)
    output["binding_moisture_points"] = [
        {
            "ambient_temperature_celsius": r["ambient_temperature_celsius"],
            "ambient_relative_humidity": r["ambient_relative_humidity"],
        }
        for r in grid_results
        if not r.get("solve_failed")
        and r.get("constraint_check", {}).get("binding_moisture", False)
    ]
    output["elapsed_s"] = time.perf_counter() - start

    regret = output.get("regret")
    if regret and regret.get("n_points", 0) > 0:
        print(
            f"  Regret: Median={regret['relative_pct']['median']:.3f}%  "
            f"P95={regret['relative_pct']['p95']:.3f}%  Max={regret['relative_pct']['max']:.3f}%  "
            f"| bindet Feuchte an {n_binding_moisture}/{len(grid_results)} Punkten "
            f"| {output['elapsed_s']:.0f}s",
            flush=True,
        )
    else:
        print(
            f"  Kein ueberall zulaessiger fester Betriebspunkt "
            f"| bindet Feuchte an {n_binding_moisture}/{len(grid_results)} Punkten "
            f"| {output['elapsed_s']:.0f}s",
            flush=True,
        )
    return output


def load_anchor_5x5() -> dict[str, Any]:
    anchor = json.loads(ANCHOR_5X5_PATH.read_text())
    if not fingerprints_equivalent(anchor["config_fingerprint"], EXPECTED_BASE_FINGERPRINT):
        raise RuntimeError(
            f"{ANCHOR_5X5_PATH} hat config_fingerprint {anchor['config_fingerprint']!r}, "
            f"erwartet {EXPECTED_BASE_FINGERPRINT!r} -- kein gueltiger Anker fuer den "
            "aktuellen Betriebspunkt."
        )
    n_binding = sum(
        1
        for r in anchor["grid_results"]
        if not r.get("solve_failed")
        and r.get("constraint_check", {}).get("binding_moisture", False)
    )
    return {
        "source": str(ANCHOR_5X5_PATH),
        "grid_size": anchor["grid_size"],
        "config_fingerprint": anchor["config_fingerprint"],
        "regret": anchor["regret"],
        "n_binding_moisture": n_binding,
        "n_grid_points": len(anchor["grid_results"]),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--values",
        type=float,
        nargs="+",
        default=list(CRITICAL_MOISTURE_VALUES_KG_PER_KG),
        help="critical_moisture_kg_per_kg-Werte, Default 0.10 0.15 0.20 0.25 0.30",
    )
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--workers", type=int, default=-1, help="-1 = alle Kerne")
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    _suppress_placeholder_warning()

    env = environment_fingerprint()
    if env["git/dirty"]:
        print(
            "WARNUNG: Arbeitsbaum ist dirty -- git/dirty wird im Artefakt vermerkt, der "
            "Lauf ist trotzdem nicht vollstaendig reproduzierbar.",
            flush=True,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base_config = load_process_config(arguments.config)
    if not fingerprints_equivalent(base_config.fingerprint, EXPECTED_BASE_FINGERPRINT):
        raise RuntimeError(
            f"Basis-Config hat Fingerprint {base_config.fingerprint!r}, erwartet "
            f"{EXPECTED_BASE_FINGERPRINT!r} -- weicht von Befund 52 ab."
        )

    regret_module = _load_regret_module()
    workers = arguments.workers if arguments.workers > 0 else cpu_count()

    print("=" * 78)
    print("DA1: Sensitivitaet gegen critical_moisture_kg_per_kg")
    print(f"Config       {arguments.config}  (Fingerprint {base_config.fingerprint})")
    print(f"Werte        {arguments.values}")
    print(f"Gitter       {arguments.grid_size}x{arguments.grid_size}")
    print(f"Worker       {workers}")
    print(f"git/commit={env['git/commit']}  git/dirty={env['git/dirty']}")
    print("=" * 78, flush=True)

    total_start = time.perf_counter()
    results: dict[str, dict[str, Any]] = {}
    for value in arguments.values:
        variant_config = build_variant_config(base_config, critical_moisture_kg_per_kg=value)
        label = f"critical_moisture={value:.2f}kg/kg"
        results[f"{value:.2f}"] = run_regret_study(
            regret_module,
            variant_config,
            grid_size=arguments.grid_size,
            workers=workers,
            label=label,
        )
    total_elapsed = time.perf_counter() - total_start

    anchor_comparison = None
    current_key = f"{CURRENT_VALUE_KG_PER_KG:.2f}"
    if current_key in results:
        anchor = load_anchor_5x5()
        new_regret = results[current_key].get("regret")
        anchor_comparison = {
            "anchor_5x5": anchor,
            "new_3x3_at_current_value": {
                "grid_size": results[current_key]["grid_size"],
                "regret": new_regret,
                "n_binding_moisture": results[current_key]["n_binding_moisture"],
                "n_grid_points": results[current_key]["n_grid_points"],
            },
        }
        if new_regret and new_regret.get("n_points", 0) > 0:
            anchor_comparison["median_relative_pct_diff"] = (
                new_regret["relative_pct"]["median"] - anchor["regret"]["relative_pct"]["median"]
            )
        print("\n" + "=" * 78)
        print("Anker-Vergleich am aktuellen Wert (0.20 kg/kg): 3x3 (neu) vs. 5x5 (Befund 52)")
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
        "critical_moisture_values_kg_per_kg": arguments.values,
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

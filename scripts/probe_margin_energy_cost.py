r"""KA1-KA3: Kosten einer Sicherheitsmarge an der Feuchtegrenze und verbleibender Regelungsgewinn
bei realistischer Soft-Sensor-Guete.

Anlass: Befund 62 (GA1/HA1) zeigt, dass ein strafbasierter RL-Agent an der Feuchtegrenze
eine Marge von median 0,139545 kg/kg haelt -- 31- bis 86-mal mehr, als der Vorhersagefehler
des Feuchte-Softsensors (3x RMSE-Konvention) rechtfertigen wuerde. Offen blieb dort: was
kostet eine SOLCHE Marge an Energie, wenn man sie nicht dem RL-Agenten ueberlaesst, sondern
einem Regler unterstellt, der (bei perfektem Prozessmodell) exakt am verschaerften
Sollwert-Grenzwert `critical_moisture - margin` optimiert -- also einem Orakel mit
verschobener Nebenbedingung. Diese Studie beantwortet das.

KEIN Reinforcement Learning, KEIN Netztraining. Nur `solve_steady_state` plus die
unveraenderte Optimierermechanik aus `study_operating_point_regret.py` (per `importlib`
geladen, keine Zeile dort veraendert -- exakt das Muster aus `probe_critical_moisture_
sensitivity.py`, Befund 54 DA1). Anders als DA1 wird hier NUR BB2 wiederholt (instanzweise
Optima je Gitterpunkt, "Orakel"), NICHT BB3(a) (fester Betriebspunkt) -- der Auftrag verlangt
ausdruecklich, den festen Betriebspunkt aus Befund 52/54 (bei `critical_moisture=0,20`)
wiederzuverwenden statt neu zu suchen: eine fest verdrahtete Anlage, die nie eine
Feuchtemessung sieht, kennt keine Marge und aendert ihren Sollwert nicht.

KA1 -- Effektive Grenzen (`X_krit = 0,20 kg/kg`)
--------------------------------------------------
Sieben Werte von `critical_moisture_kg_per_kg`, absteigend:

    0,20              Orakel, Marge 0 (MUSS Befund 54 DA1 bei 0,20 reproduzieren -- Anker)
    0,20 - 3xRMSE     angeregt trainierter LSTM-Softsensor (Befund 62 HA1)
    0,20 - 3xRMSE     konstant trainierter LSTM-Softsensor (Befund 62 HA1, schwaecherer Fall)
    0,18 / 0,15 / 0,12  Gitterpunkte ohne Softsensor-Bezug (Interpolationsstuetzen)
    0,20 - Agentenmarge  RL-Agenten-Marge aus Befund 62 GA1 (Gegenprobe, KA3)

Zwei Werte sind aus Befund 54 DA1 wiederverwendbar (gleiches 3x3-Gitter, gleiches
Suchbudget, gleicher Seed, gleicher Commit-Stand -- ein Commit ist in diesem frisch
pseudonymisierten Repository der einzig vorhandene): `0,20` (Anker) und `0,15` (trifft
DA1s Gitterpunkt exakt). Die uebrigen fuenf Werte (zwei Softsensor-Margen, zwei blanke
Gitterpunkte, die Agentenmarge) hat DA1 nicht geliefert und werden hier neu gerechnet --
gleiches Gitter (3x3, `ambient_temperature_celsius` x `ambient_relative_humidity`),
gleiches Budget (`OptimizationBudget`-Defaults aus `study_operating_point_regret.py`),
gleicher Seed (0).

Die beiden Softsensor-Margen und die Agentenmarge kommen NICHT aus abgetippten Zahlen,
sondern werden aus den bereits committeten Artefakten `rl_bandit_ha1_margin_vs_prediction_
error.json` (3x RMSE je Trainingsfall) und darin wiederum `rl_bandit_ga1_safety_margin.json`
(Agentenmarge) gelesen -- volle Fliesskommapraezision, keine Rundung auf die im Auftrag
genannten Anzeige-Werte (0,1984/0,1955/0,0605 kg/kg).

Unzulaessige Gitterpunkte (die verschaerfte Grenze ist mit den verfuegbaren Stellgroessen
nicht erreichbar, `feasible=False` in `constraint_check`) werden AUSGEWIESEN, nicht
stillschweigend aus dem Gitter entfernt -- sie tragen nur nicht zu den KA2-Aggregaten bei
(dieselbe Konvention wie `compute_regret` im Regret-Modul).

KA2 -- Kosten der Marge und verbleibender Regelungsgewinn
-------------------------------------------------------------
Je Grenze und Gitterpunkt (nur an Gitterpunkten, die bei DIESER Grenze zulaessig sind):

    Kosten            = E_Orakel(Grenze) - E_Orakel(0,20)                       [kW, %]
    verbleibender Anteil = (E_fest - E_Orakel(Grenze)) / (E_fest - E_Orakel(0,20))

mit `E_fest` = Befund-52/54-Fixpunkt bei `critical_moisture=0,20` (DA1-Artefakt,
`results_by_value["0.20"]["fixed_point"]["per_grid_point"]`) -- UNVERAENDERT ueber alle
Grenzen, weil ein fester Betriebspunkt konstruktionsbedingt keine Marge haelt und nie eine
Feuchtemessung sieht.

Aggregation je Gruppe (alle 9 Punkte / die an `critical_moisture=0,20` feuchte-bindenden
Punkte / die uebrigen) auf ZWEI Arten, dieselbe Unterscheidung wie
`twinlink_ml.rl.evaluation` (XA1) und Befund 62:

  - Median ueber die Einzelquotienten je Punkt.
  - GEPOOLT, Summe durch Summe (Zaehler- und Nennersumme getrennt, dann dividiert) --
    robust gegen einzelne kleine Nenner an Punkten, wo die Feuchtegrenze bei 0,20 nicht
    bindet (dort ist `E_fest - E_Orakel(0,20)` klein bis nahe null) -- exakt dasselbe
    Problem, das in Befund 62 den gepoolten Regret-Anteil motiviert hat.

KA3 -- Gegenprobe zum RL-Agenten
------------------------------------
Bei der Agentenmarge (Befund 62 GA1) wird VORAB ein NEGATIVER gepoolter verbleibender
Anteil erwartet (ein Regler mit dieser Marge waere schlechter als der feste Betriebspunkt),
zum Vergleich mit dem in Befund 62 FA3 gemessenen gepoolten Anteil des RL-Agenten bei
Schritt 10000 (Median ueber 3 Seeds, `rl_bandit_fa3_seed_evaluation.json`). Nur eine
Groessenordnungspruefung: Befund 62 FA3 lief auf dem 5x5-Gitter (25 Kontexte) aus Befund 52,
diese Studie auf dem 3x3-Gitter (9 Kontexte) aus Befund 54 DA1 -- unterschiedliche
Populationen -- und der Agent haelt keine einheitliche Marge an allen Punkten (Befund 62 GA1
selbst: Median 0,1395 kg/kg, seedabhaengige Spanne 0,1299-0,1454 kg/kg).

Aufruf::

    python scripts/probe_margin_energy_cost.py \
        --output artifacts/runs/probe_margin_energy_cost.json
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
DA1_ARTIFACT_PATH = REPO_ROOT / "artifacts" / "runs" / "probe_critical_moisture_sensitivity.json"
HA1_ARTIFACT_PATH = (
    REPO_ROOT / "artifacts" / "runs" / "rl_bandit_ha1_margin_vs_prediction_error.json"
)
FA3_ARTIFACT_PATH = REPO_ROOT / "artifacts" / "runs" / "rl_bandit_fa3_seed_evaluation.json"
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "probe_margin_energy_cost.json"

CRITICAL_MOISTURE_REFERENCE_KG_PER_KG = 0.20
GRID_SIZE = 3
"""Wie Befund 54 DA1 -- nicht das 5x5-Gitter aus Befund 52/62."""

DA1_REUSABLE_LABELS = ("oracle_anchor", "grid_0.15")
"""Diese zwei Grenzen liegen exakt auf DA1-Gitterpunkten (0,20 und 0,15 kg/kg) -- gleiches
Gitter, gleiches Budget, gleicher Seed, gleicher (einziger) Commit-Stand dieses Repos.
Wiederverwendet statt neu gerechnet."""

_TOLERANCE_GRID_MATCH = 1e-9
"""Fuer den Abgleich der Stoergroessengitter-Koordinaten zwischen frisch gebautem Gitter und
dem in DA1 committeten -- reines Gleitkommarauschen erwartet, kein Toleranzbedarf."""


def _load_regret_module() -> ModuleType:
    """Wie in `probe_critical_moisture_sensitivity.py`/`study_operating_point_transient.py`
    -- siehe dort fuer die Begruendung des `sys.modules`-Eintrags vor `exec_module`."""
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
    """Wie in `probe_critical_moisture_sensitivity.py` -- nur `material.critical_moisture_
    kg_per_kg` ueberschrieben, sonst unveraendert."""
    material = dataclasses.replace(
        base.material, critical_moisture_kg_per_kg=critical_moisture_kg_per_kg
    )
    return dataclasses.replace(base, material=material)


def run_oracle_grid(
    regret_module: ModuleType,
    variant_config: ProcessConfig,
    *,
    grid_size: int,
    workers: int,
    budget: Any,
    label: str,
) -> list[dict[str, Any]]:
    """KA1: NUR BB2 (instanzweise Optima je Gitterpunkt) -- kein BB3(a) (Auftrag: der feste
    Betriebspunkt aus Befund 52/54 wird wiederverwendet, nicht neu gesucht)."""
    regret_module._set_base_config(variant_config)
    bounds = regret_module.action_bounds(variant_config)
    grid_points = regret_module.build_grid(variant_config, grid_size)

    start = time.perf_counter()
    print(
        f"\n=== {label}: Gitter {grid_size}x{grid_size} = {len(grid_points)} Punkte (nur Orakel) ===",
        flush=True,
    )
    with Pool(
        workers, initializer=regret_module._set_base_config, initargs=(variant_config,)
    ) as pool:
        tasks = [
            (index, t_amb, phi_amb, bounds, budget)
            for index, (t_amb, phi_amb) in enumerate(grid_points)
        ]
        grid_results_by_index: dict[int, dict[str, Any]] = {}
        for n_done, result in enumerate(
            pool.imap_unordered(regret_module._grid_point_task, tasks), start=1
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


def _grid_coordinates(grid_results: list[dict[str, Any]]) -> list[tuple[float, float]]:
    return [
        (r["ambient_temperature_celsius"], r["ambient_relative_humidity"]) for r in grid_results
    ]


def _assert_same_grid(
    a: list[tuple[float, float]], b: list[tuple[float, float]], *, context: str
) -> None:
    if len(a) != len(b):
        raise RuntimeError(f"{context}: unterschiedliche Gittergroesse ({len(a)} vs {len(b)})")
    for (t1, p1), (t2, p2) in zip(a, b, strict=True):
        if abs(t1 - t2) > _TOLERANCE_GRID_MATCH or abs(p1 - p2) > _TOLERANCE_GRID_MATCH:
            raise RuntimeError(
                f"{context}: Gitterkoordinaten weichen ab ({t1},{p1}) vs ({t2},{p2})"
            )


def _assert_same_budget(a: dict[str, Any], b: dict[str, Any], *, context: str) -> None:
    keys = ("seed", "popsize", "maxiter", "de_tol", "search_max_iterations", "search_tolerance")
    for key in keys:
        if a[key] != b[key]:
            raise RuntimeError(
                f"{context}: Budgetfeld {key!r} weicht ab ({a[key]!r} vs {b[key]!r})"
            )


def load_da1_artifact() -> dict[str, Any]:
    return json.loads(DA1_ARTIFACT_PATH.read_text())


def load_margin_sources() -> dict[str, Any]:
    ha1 = json.loads(HA1_ARTIFACT_PATH.read_text())
    return {
        "excited_3x_rmse_kg_per_kg": ha1["cases"]["excited_trained"]["3x_rmse_kg_per_kg"],
        "excited_rmse_median_kg_per_kg": ha1["cases"]["excited_trained"][
            "rmse_run_median_physical_kg_per_kg"
        ],
        "constant_3x_rmse_kg_per_kg": ha1["cases"]["constant_trained"]["3x_rmse_kg_per_kg"],
        "constant_rmse_median_kg_per_kg": ha1["cases"]["constant_trained"][
            "rmse_run_median_physical_kg_per_kg"
        ],
        "agent_margin_kg_per_kg": ha1["ga1_margin_kg_per_kg"],
        "ha1_source": str(HA1_ARTIFACT_PATH.relative_to(REPO_ROOT)),
    }


def load_fa3_reference() -> dict[str, Any]:
    fa3 = json.loads(FA3_ARTIFACT_PATH.read_text())
    median_at_10000 = fa3["per_checkpoint"]["10000"]["median"]["pooled_closed_regret_fraction"]
    return {
        "pooled_closed_regret_fraction_step10000_median": median_at_10000,
        "source": str(FA3_ARTIFACT_PATH.relative_to(REPO_ROOT)),
        "note": "5x5-Gitter (25 Kontexte, Befund 52), nicht das 3x3-Gitter dieser Studie -- nur Groessenordnungsvergleich (Auftrag KA3).",
    }


def build_boundary_definitions(margins: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ref = CRITICAL_MOISTURE_REFERENCE_KG_PER_KG
    return {
        "oracle_anchor": {
            "margin_kg_per_kg": 0.0,
            "critical_moisture_kg_per_kg": ref,
            "margin_source": "Orakel-Anker, Marge 0 -- muss Befund 54 DA1 bei 0,20 kg/kg reproduzieren.",
        },
        "excited_lstm_3rmse": {
            "margin_kg_per_kg": margins["excited_3x_rmse_kg_per_kg"],
            "critical_moisture_kg_per_kg": ref - margins["excited_3x_rmse_kg_per_kg"],
            "margin_source": (
                "3x RMSE, angeregt trainierter LSTM-Feuchte-Softsensor (Befund 51/58, "
                f"{margins['ha1_source']})"
            ),
        },
        "constant_lstm_3rmse": {
            "margin_kg_per_kg": margins["constant_3x_rmse_kg_per_kg"],
            "critical_moisture_kg_per_kg": ref - margins["constant_3x_rmse_kg_per_kg"],
            "margin_source": (
                "3x RMSE, konstant trainierter LSTM-Feuchte-Softsensor (Befund 55, "
                f"{margins['ha1_source']})"
            ),
        },
        "grid_0.18": {
            "margin_kg_per_kg": ref - 0.18,
            "critical_moisture_kg_per_kg": 0.18,
            "margin_source": "Gitterpunkt ohne Softsensor-Bezug (Interpolationsstuetze).",
        },
        "grid_0.15": {
            "margin_kg_per_kg": ref - 0.15,
            "critical_moisture_kg_per_kg": 0.15,
            "margin_source": "Gitterpunkt ohne Softsensor-Bezug -- identisch zu Befund 54 DA1s 0,15-kg/kg-Punkt, von dort wiederverwendet.",
        },
        "grid_0.12": {
            "margin_kg_per_kg": ref - 0.12,
            "critical_moisture_kg_per_kg": 0.12,
            "margin_source": "Gitterpunkt ohne Softsensor-Bezug (Interpolationsstuetze).",
        },
        "rl_agent_margin": {
            "margin_kg_per_kg": margins["agent_margin_kg_per_kg"],
            "critical_moisture_kg_per_kg": ref - margins["agent_margin_kg_per_kg"],
            "margin_source": f"RL-Agenten-Marge, Median ueber 3 Seeds (Befund 62 GA1, {margins['ha1_source']})",
        },
    }


def _pooled(numerators: list[float], denominators: list[float]) -> float | None:
    denom_sum = float(np.sum(denominators))
    if abs(denom_sum) <= 1e-12:
        return None
    return float(np.sum(numerators)) / denom_sum


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values)))


def compute_ka2_for_boundary(
    grid_results_boundary: list[dict[str, Any]],
    grid_results_reference: list[dict[str, Any]],
    fixed_point_per_point: list[dict[str, Any]],
    binding_at_reference: list[bool],
) -> dict[str, Any]:
    per_point: list[dict[str, Any]] = []
    for i, (rb, rr, fp, binds) in enumerate(
        zip(
            grid_results_boundary,
            grid_results_reference,
            fixed_point_per_point,
            binding_at_reference,
            strict=True,
        )
    ):
        feasible = not rb.get("solve_failed") and rb.get("feasible", False)
        entry: dict[str, Any] = {
            "grid_index": i,
            "ambient_temperature_celsius": rb["ambient_temperature_celsius"],
            "ambient_relative_humidity": rb["ambient_relative_humidity"],
            "binding_moisture_at_reference": binds,
            "feasible_at_boundary": feasible,
        }
        if feasible:
            e_oracle_b = rb["energy_target"]
            e_oracle_ref = rr["energy_target"]
            e_fest = fp["energy_target"]
            cost_abs = e_oracle_b - e_oracle_ref
            entry.update(
                {
                    "energy_oracle_boundary_kw": e_oracle_b,
                    "energy_oracle_reference_kw": e_oracle_ref,
                    "energy_fixed_kw": e_fest,
                    "cost_abs_kw": cost_abs,
                    "cost_rel_pct": 100.0 * cost_abs / e_oracle_ref,
                    "remaining_fraction": (
                        (e_fest - e_oracle_b) / (e_fest - e_oracle_ref)
                        if abs(e_fest - e_oracle_ref) > 1e-12
                        else None
                    ),
                }
            )
        per_point.append(entry)

    def _aggregate(subset: list[dict[str, Any]]) -> dict[str, Any]:
        feasible_points = [p for p in subset if p["feasible_at_boundary"]]
        cost_rel = [p["cost_rel_pct"] for p in feasible_points]
        cost_abs = [p["cost_abs_kw"] for p in feasible_points]
        e_oracle_ref = [p["energy_oracle_reference_kw"] for p in feasible_points]
        remaining_num = [
            p["energy_fixed_kw"] - p["energy_oracle_boundary_kw"] for p in feasible_points
        ]
        remaining_denom = [
            p["energy_fixed_kw"] - p["energy_oracle_reference_kw"] for p in feasible_points
        ]
        remaining_finite = [
            p["remaining_fraction"] for p in feasible_points if p["remaining_fraction"] is not None
        ]
        return {
            "n_points": len(subset),
            "n_feasible": len(feasible_points),
            "median_cost_abs_kw": _median_or_none(cost_abs),
            "median_cost_rel_pct": _median_or_none(cost_rel),
            "pooled_cost_rel_pct": (
                100.0 * _pooled(cost_abs, e_oracle_ref)
                if _pooled(cost_abs, e_oracle_ref) is not None
                else None
            ),
            "median_remaining_fraction": _median_or_none(remaining_finite),
            "pooled_remaining_fraction": _pooled(remaining_num, remaining_denom),
        }

    binding_points = [p for p in per_point if p["binding_moisture_at_reference"]]
    non_binding_points = [p for p in per_point if not p["binding_moisture_at_reference"]]

    return {
        "per_point": per_point,
        "aggregate": {
            "all": _aggregate(per_point),
            "binding_at_reference": _aggregate(binding_points),
            "non_binding_at_reference": _aggregate(non_binding_points),
        },
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
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

    da1 = load_da1_artifact()
    if da1["grid_size"] != GRID_SIZE:
        raise RuntimeError(
            f"DA1-Artefakt hat Gitter {da1['grid_size']}x{da1['grid_size']}, erwartet {GRID_SIZE}x{GRID_SIZE}"
        )
    if not fingerprints_equivalent(base_config.fingerprint, da1["base_config_fingerprint"]):
        raise RuntimeError(
            f"Aktuelle Config hat Fingerprint {base_config.fingerprint!r}, DA1-Artefakt "
            f"{da1['base_config_fingerprint']!r} -- kein gemeinsamer Ausgangspunkt fuer die "
            "Wiederverwendung von 0,20/0,15 kg/kg."
        )

    r_reference = da1["results_by_value"]["0.20"]
    r_015 = da1["results_by_value"]["0.15"]
    if not r_reference["fixed_point"]["feasible_everywhere"]:
        raise RuntimeError(
            "DA1s Fixpunkt bei 0,20 kg/kg ist nicht ueberall zulaessig -- E_fest nicht verwendbar."
        )

    margins = load_margin_sources()
    fa3_reference = load_fa3_reference()
    boundaries = build_boundary_definitions(margins)

    regret_module = _load_regret_module()
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
    _assert_same_budget(budget_dict, r_reference["budget"], context="Budget vs. DA1 0,20 kg/kg")
    _assert_same_budget(budget_dict, r_015["budget"], context="Budget vs. DA1 0,15 kg/kg")

    reference_grid_coords = _grid_coordinates(r_reference["grid_results"])
    fresh_grid_coords = _grid_coordinates(
        [
            {"ambient_temperature_celsius": t, "ambient_relative_humidity": phi}
            for t, phi in regret_module.build_grid(base_config, GRID_SIZE)
        ]
    )
    _assert_same_grid(
        reference_grid_coords, fresh_grid_coords, context="frisches Gitter vs. DA1 0,20 kg/kg"
    )
    fixed_point_coords = [
        (p["ambient_temperature_celsius"], p["ambient_relative_humidity"])
        for p in r_reference["fixed_point"]["per_grid_point"]
    ]
    _assert_same_grid(
        reference_grid_coords,
        fixed_point_coords,
        context="E_fest-Gitter vs. Orakel-Gitter (0,20 kg/kg)",
    )

    binding_at_reference = [
        bool(r["constraint_check"]["binding_moisture"]) for r in r_reference["grid_results"]
    ]

    print("=" * 88)
    print("KA1-KA3: Kosten einer Sicherheitsmarge an der Feuchtegrenze")
    print(f"Config       {arguments.config}  (Fingerprint {base_config.fingerprint})")
    print(f"Gitter       {GRID_SIZE}x{GRID_SIZE} (wie Befund 54 DA1)")
    print(f"Worker       {workers}")
    print(f"git/commit={env['git/commit']}  git/dirty={env['git/dirty']}")
    for label, spec in boundaries.items():
        reused = label in DA1_REUSABLE_LABELS
        print(
            f"  {label:22s} critical_moisture={spec['critical_moisture_kg_per_kg']:.6f} kg/kg "
            f"(Marge {spec['margin_kg_per_kg']:.6f} kg/kg){'  [DA1 wiederverwendet]' if reused else ''}"
        )
    print("=" * 88, flush=True)

    total_start = time.perf_counter()
    grid_results_by_boundary: dict[str, list[dict[str, Any]]] = {}
    reused_flags: dict[str, str | None] = {}
    for label, spec in boundaries.items():
        if label == "oracle_anchor":
            grid_results_by_boundary[label] = r_reference["grid_results"]
            reused_flags[label] = (
                "probe_critical_moisture_sensitivity.json#results_by_value.0.20.grid_results"
            )
            print(f"\n{label}: wiederverwendet von DA1 (0,20 kg/kg, Anker) -- keine neue Rechnung.")
            continue
        if label == "grid_0.15":
            grid_results_by_boundary[label] = r_015["grid_results"]
            reused_flags[label] = (
                "probe_critical_moisture_sensitivity.json#results_by_value.0.15.grid_results"
            )
            print(f"\n{label}: wiederverwendet von DA1 (0,15 kg/kg) -- keine neue Rechnung.")
            continue
        variant_config = build_variant_config(
            base_config, critical_moisture_kg_per_kg=spec["critical_moisture_kg_per_kg"]
        )
        grid_results_by_boundary[label] = run_oracle_grid(
            regret_module,
            variant_config,
            grid_size=GRID_SIZE,
            workers=workers,
            budget=budget,
            label=label,
        )
        reused_flags[label] = None
        new_coords = _grid_coordinates(grid_results_by_boundary[label])
        _assert_same_grid(reference_grid_coords, new_coords, context=f"Gitter {label} vs. Referenz")
    total_elapsed = time.perf_counter() - total_start

    ka1_output: dict[str, Any] = {}
    for label, spec in boundaries.items():
        ka1_output[label] = {
            **spec,
            "reused_from": reused_flags[label],
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
    print("KA2: Kosten und verbleibender Regelungsgewinn")
    print("=" * 88)
    ka2_output: dict[str, Any] = {}
    for label in boundaries:
        result = compute_ka2_for_boundary(
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
    print("KA3: Gegenprobe zum RL-Agenten")
    print("=" * 88)
    agent_all = ka2_output["rl_agent_margin"]["aggregate"]["all"]
    print(
        f"Diese Studie, rl_agent_margin, gepoolt (alle 9 Punkte): {agent_all['pooled_remaining_fraction']}"
    )
    print(
        f"Befund 62 FA3, RL-Agent, gepoolt (25 Punkte, Schritt 10000, Median ueber 3 Seeds): "
        f"{fa3_reference['pooled_closed_regret_fraction_step10000_median']}"
    )

    output: dict[str, Any] = {
        "environment": env,
        "config_path": str(arguments.config),
        "base_config_fingerprint": base_config.fingerprint,
        "grid_size": GRID_SIZE,
        "budget": budget_dict,
        "critical_moisture_reference_kg_per_kg": CRITICAL_MOISTURE_REFERENCE_KG_PER_KG,
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

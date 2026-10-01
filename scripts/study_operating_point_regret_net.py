r"""MA4: Befund 52 mit Netto-Zielfunktion -- Bruttoverbrauch minus Rueckgewinnungsgutschrift.

KEIN neuer Optimierer, KEINE neue Optimierungsmechanik: dieses Skript wiederholt exakt
BB1-BB3 aus ``study_operating_point_regret.py`` (Gitter, differential_evolution +
Nelder-Mead, Nebenbedingungen, Regret-Berechnung -- ALLE unveraendert importiert, keine
Zeile dort dupliziert), fuer FUeNF Zielfunktionsvarianten statt einer. Nur die Zielgroesse
je Beharrungszustand wird ersetzt -- ueber den in ``study_operating_point_regret.py``
(Commit 0307f91, MA4-Vorbereitung) bereitgestellten Override-Hook
(``_set_objective_value_fn``), demselben Pool-Initializer-Muster wie ``_BASE_CONFIG``.

Verschaltung (MA4-Auftrag, woertlich gefordert)
--------------------------------------------------
EINE Linie: die Rueckgewinnung waermt ihre EIGENE Zuluft vor (Quelle = Abluft DIESES
Prozesses, Senke = seine eigene Ansaugluft). Im Einzelanlagenfall (kein Wärmeverbund
zwischen mehreren Linien) ist das die einzige sinnvolle Konfiguration -- ein Prozess kann
nur seine eigene Abwaerme nutzen, es gibt keine zweite Linie, der er sie abgeben koennte.
Umgesetzt als EINZELPUNKT-Formel (wie MA2, ``compute_recoverable_power_kw``/die MA1-
Kondensationsformel direkt aus ``thermo.py``), NICHT als volle ``heatnetwork.balance.
evaluate_cascade``-Kaskade mit HeatSourceConfig/HeatSinkConfig: bei einer Quelle und einer
Senke ohne mehrstufige Kaskadierung faellt die Kaskade auf dieselbe Formel zurueck, die
Mehrstufigkeit von EA2/Korrektur 2 waere hier Vorratsbau ohne zweite Senke, die sie
brauchen koennte.

Skalenvorbehalt (WICHTIG, siehe MA6): Q_rec wird hier mit
``solution.dry_air_mass_flow_kg_per_s`` (Laborsimulator-Massenstrom, ca. 0.03 kg/s)
berechnet, NICHT mit ``DISTRICT_MASS_FLOW_KG_PER_S`` (8 kg/s) -- "ein Prozess, kein
Verbund" (Befund 52d) heisst hier: DIESER Prozess recycelt nur seine EIGENE, kleine
Abwaerme, nicht die eines Verbunds. Die Rueckgewinnungsgutschrift ist deshalb absolut
KLEIN gegen den Bruttoverbrauch (Groessenordnung siehe MA2/``evaluation.
recoverable_power``-Skalenvorbehalt) -- ob sich der optimale Betriebspunkt spuerbar
verschiebt, ist damit auch eine Frage dieses Massstabs, nicht nur der Gewichtung (MA3).

Fuenf Varianten (MA3-Nutzerentscheidung: BEIDE Gewichtungen nebeneinander, keine ersetzt
die andere; Prioritaet bei Laufzeitdruck: Kondensation vor ungewichteter Variante)
--------------------------------------------------------------------------------------
    V1  Brutto            (Referenz, unveraendert -- muss den 15.22-%-Anker treffen)
    V2  Netto, sensibel,   kW-Summe
    V3  Netto, sensibel,   CO2-gewichtet
    V4  Netto, kondensierend, CO2-gewichtet
    V5  Netto, kondensierend, kW-Summe        (niedrigste Prioritaet, nur wenn Zeit reicht)

Anker (MA4-Auftrag, woertlich): die 3x3-Rechnung bei ``critical_moisture_kg_per_kg=0.20``
(Default-Wert, unveraendert) ergab in Befund 54 (DA1) bereits 15.22 % Regret-Median gegen
16.03 % auf 5x5 (Befund 52) -- V1 hier MUSS denselben Wert treffen (± Gitterrauschen, siehe
Befund 54: 0.81 Prozentpunkte zwischen 3x3 und 5x5 sind KEINE Diskrepanz), sonst ist die
wiederverwendete Mechanik veraendert und V2-V5 sind nicht interpretierbar. Gleiches Budget
wie ``probe_critical_moisture_sensitivity.py`` (DA1, Commit siehe ``docs/04_stand_ap1.md``
Befund 54): Modul-Default-``OptimizationBudget``, ``seed=0``, Gitter 3x3.

Laufzeitschaetzung: Befund 54 (DA1, 5 Werte auf 3x3) brauchte 110.6 min gesamt, also
~22 min je Variante -- 5 Varianten hier folglich ueberschlaegig ~1.8 h, klar unter der
4-h-Meldegrenze aus dem Auftrag. Nach V1 wird die TATSAECHLICHE Laufzeit gegen diese
Schaetzung geprueft und eine Hochrechnung fuer alle 5 Varianten gemeldet (Auftrag: bei
Ueberschreitung von 4 h ist das zu melden).

Aufruf::

    python scripts/study_operating_point_regret_net.py \
        --output artifacts/runs/study_operating_point_regret_net.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import study_operating_point_regret as regret_module

from twinlink_ml.evaluation.net_energy import (
    fuel_equivalent_credit_kw,
    net_objective_co2_kg_per_h,
    net_objective_kw,
)
from twinlink_ml.evaluation.recoverable_power import (
    DEW_POINT_MARGIN_K,
    compute_recoverable_power_kw,
)
from twinlink_ml.heatnetwork.config import (
    DEFAULT_COLD_TEMPERATURE_CELSIUS,
    DEFAULT_HEATER_EFFICIENCY,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, ProcessConfig, load_process_config
from twinlink_ml.simulation.backend import NumpyBackend
from twinlink_ml.simulation.steady_state import SteadyStateSolution
from twinlink_ml.simulation.thermo import latent_heat_j_per_kg, saturation_humidity_kg_per_kg
from twinlink_ml.utils.fingerprint_history import fingerprints_equivalent
from twinlink_ml.utils.tracking import environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "study_operating_point_regret_net.json"

EXPECTED_BASE_FINGERPRINT = "2acd06f38963c3af"
"""Muss mit ``config_fingerprint`` aus Befund 52/54 uebereinstimmen -- siehe
``probe_critical_moisture_sensitivity.py``, gleiche Konstante."""

ANCHOR_3X3_REGRET_MEDIAN_PCT = 15.22
"""Befund 54 (DA1), Zeile '0.20 (aktuell, 3x3)' -- V1 hier muss diesen Wert treffen."""

ANCHOR_TOLERANCE_PCT = 1.0
"""Grosszuegiger als das beobachtete Gitterrauschen zwischen 3x3 und 5x5 in Befund 54
(0.81 Prozentpunkte) -- ein Treffer innerhalb dieser Toleranz gilt als Bestaetigung der
wiederverwendeten Mechanik, kein Bit-fuer-Bit-Anspruch (float-/Plattform-Rauschen der
DE-Suche ist nicht ausgeschlossen)."""

DEFAULT_GRID_SIZE = 3
RUNTIME_ESTIMATE_PER_VARIANT_S = 22.0 * 60.0
"""Befund 54 (DA1): 110.6 min / 5 Werte = 22.1 min je 3x3-Lauf -- Vorab-Schaetzung."""

RUNTIME_WARNING_THRESHOLD_S = 4.0 * 3600.0
"""Auftrag: Hochrechnung ueber 4 h ist nach der ersten Variante zu melden."""

VARIANTS: tuple[tuple[str, str | None, str | None], ...] = (
    ("V1_gross", None, None),
    ("V2_net_kw_sensible", "sensible", "kw_sum"),
    ("V3_net_co2_sensible", "sensible", "co2_weighted"),
    ("V4_net_co2_condensing", "condensing", "co2_weighted"),
    ("V5_net_kw_condensing", "condensing", "kw_sum"),
)
"""Reihenfolge = Prioritaet (MA3-Antwort): V4 (Kondensation) vor V5 (ungewichtet), falls
die Laufzeit knapp wird -- siehe ``main()``."""

_VARIANT: dict[str, str] | None = None
"""Modul-globaler Zustand fuer die aktive Netto-Variante, gesetzt im Pool-Initializer --
gleiches Muster wie ``regret_module._BASE_CONFIG``/``_OBJECTIVE_VALUE_FN`` (Picklebarkeit
unter Windows-``spawn``: kein Funktionsabschluss/Closure, nur ein Modul-Top-Level-Wert)."""


def _set_variant(recovery_mode: str | None, weighting: str | None) -> None:
    global _VARIANT
    if recovery_mode is None:
        _VARIANT = None
    else:
        assert weighting is not None
        _VARIANT = {"recovery_mode": recovery_mode, "weighting": weighting}


def _q_rec_kw(
    solution: SteadyStateSolution, base_config: ProcessConfig, *, recovery_mode: str
) -> float:
    """Rueckgewinnbare Leistung aus DIESEM Beharrungszustand, Einzelpunkt-Formel (siehe
    Moduldocstring, Abschnitt Verschaltung) -- Laborsimulator-Massenstrom, kein
    Verbund-Massenstrom (Skalenvorbehalt)."""
    backend = NumpyBackend()
    t_out = solution.bed_temperature_celsius - base_config.exhaust.outlet_duct_heat_loss_k
    y_out = solution.outlet_humidity_kg_per_kg
    mass_flow = solution.dry_air_mass_flow_kg_per_s

    if recovery_mode == "sensible":
        return float(
            compute_recoverable_power_kw(
                backend,
                base_config.gas,
                t_out,
                y_out,
                mass_flow_kg_per_s=mass_flow,
                dew_point_margin_k=DEW_POINT_MARGIN_K,
            )
        )

    assert recovery_mode == "condensing"
    # MA1-Formel (heatnetwork.balance.evaluate_cascade, recovery_mode="condensing"),
    # hier als Einzelpunkt-Formel direkt aus thermo.py -- siehe Moduldocstring.
    y_sat_cold = float(
        backend.to_numpy(
            saturation_humidity_kg_per_kg(
                backend, base_config.gas, backend.asarray(DEFAULT_COLD_TEMPERATURE_CELSIUS)
            )
        )
    )
    condensate_kg_per_s = max(0.0, y_out - y_sat_cold) * mass_flow
    dh_v = float(
        backend.to_numpy(
            latent_heat_j_per_kg(backend, backend.asarray(DEFAULT_COLD_TEMPERATURE_CELSIUS))
        )
    )
    q_latent_w = condensate_kg_per_s * dh_v
    q_sensible_w = (
        mass_flow
        * (t_out - DEFAULT_COLD_TEMPERATURE_CELSIUS)
        * (base_config.gas.cp_dry_air_j_per_kg_k + y_out * base_config.gas.cp_vapour_j_per_kg_k)
    )
    return (q_sensible_w + q_latent_w) / 1000.0


def _net_objective(solution: SteadyStateSolution, base_config: ProcessConfig) -> dict[str, float]:
    """Override fuer ``regret_module._objective_components`` (MA4) -- liest die aktive
    Variante aus ``_VARIANT`` (im Pool-Initializer gesetzt)."""
    assert _VARIANT is not None
    gross_kw = float(solution.diagnostics["energy_target"])
    q_rec_kw = _q_rec_kw(solution, base_config, recovery_mode=_VARIANT["recovery_mode"])
    credit_kw = fuel_equivalent_credit_kw(q_rec_kw, DEFAULT_HEATER_EFFICIENCY)
    if _VARIANT["weighting"] == "kw_sum":
        objective = net_objective_kw(gross_kw, q_rec_kw, DEFAULT_HEATER_EFFICIENCY)
    else:
        assert _VARIANT["weighting"] == "co2_weighted"
        objective = net_objective_co2_kg_per_h(gross_kw, q_rec_kw, DEFAULT_HEATER_EFFICIENCY)
    return {
        "objective_value": objective,
        "gross_energy_target_kw": gross_kw,
        "q_rec_kw": q_rec_kw,
        "recovery_credit_kw": credit_kw,
    }


def _pool_initializer(
    base_config: ProcessConfig, recovery_mode: str | None, weighting: str | None
) -> None:
    """Ein Initializer je Pool-Worker: setzt Basis-Config, aktive Variante UND den
    Override-Hook in ``study_operating_point_regret`` -- gleiches Muster wie dessen
    eigener ``_set_base_config``-Initializer (siehe dortigen Docstring)."""
    regret_module._set_base_config(base_config)
    _set_variant(recovery_mode, weighting)
    regret_module._set_objective_value_fn(_net_objective if recovery_mode is not None else None)


def _suppress_placeholder_warning() -> None:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)


def run_variant_study(
    base_config: ProcessConfig,
    *,
    grid_size: int,
    workers: int,
    label: str,
    recovery_mode: str | None,
    weighting: str | None,
) -> dict[str, Any]:
    """Ein vollstaendiger BB1-BB3-Lauf (EINE Variante) -- Orchestrierung von
    ``study_operating_point_regret.main()`` nachgebildet (gleiche Funktionen, gleiche
    Reihenfolge, gleiches Muster wie ``probe_critical_moisture_sensitivity.
    run_regret_study``), nur mit dem Netto-Objective-Hook statt einer variierten Config.
    """
    bounds = regret_module.action_bounds(base_config)
    grid_points = regret_module.build_grid(base_config, grid_size)
    budget = regret_module.OptimizationBudget(
        seed=0,
        popsize=regret_module.DEFAULT_POPSIZE,
        maxiter=regret_module.DEFAULT_MAXITER,
        de_tol=regret_module.DEFAULT_DE_TOL,
        search_max_iterations=regret_module.DEFAULT_SEARCH_MAX_ITERATIONS,
        search_tolerance=regret_module.DEFAULT_SEARCH_TOLERANCE,
        workers=workers,
    )

    start = time.perf_counter()
    print(
        f"\n=== {label}: Gitter {grid_size}x{grid_size} = {len(grid_points)} Punkte ===", flush=True
    )

    # Gilt auch fuer den MAIN-Prozess selbst: find_fixed_operating_point() ruft dessen
    # abschliessende Pro-Gitterpunkt-Auswertung direkt hier auf, nicht ueber den Pool (siehe
    # study_operating_point_regret.find_fixed_operating_point) -- ohne diesen Aufruf bliebe
    # regret_module._BASE_CONFIG im Hauptprozess auf None (nur die Pool-WORKER bekommen ihn
    # ueber den Initializer gesetzt).
    _pool_initializer(base_config, recovery_mode, weighting)

    with Pool(
        workers,
        initializer=_pool_initializer,
        initargs=(base_config, recovery_mode, weighting),
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
            print(
                f"  [{n_done:2d}/{len(grid_points)}] Punkt {index + 1} "
                f"({result['_wall_elapsed_s']:5.1f}s)",
                flush=True,
            )
            grid_results_by_index[index] = result
        grid_results = [grid_results_by_index[i] for i in range(len(grid_points))]

        output: dict[str, Any] = {
            "label": label,
            "recovery_mode": recovery_mode,
            "weighting": weighting,
            "config_fingerprint": base_config.fingerprint,
            "grid_size": grid_size,
            "budget": dataclasses.asdict(budget),
            "grid_results": grid_results,
        }

        fixed_point = regret_module.find_fixed_operating_point(grid_points, bounds, budget, pool)
        output["fixed_point"] = fixed_point
        if fixed_point["feasible_everywhere"]:
            output["regret"] = regret_module.compute_regret(grid_results, fixed_point)
            output["setpoint_spread"] = regret_module.compute_setpoint_spread(
                grid_results, base_config
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
            f"P95={regret['relative_pct']['p95']:.3f}%  Max={regret['relative_pct']['max']:.3f}%  "
            f"| {output['elapsed_s']:.0f}s",
            flush=True,
        )
        feasible_points = [
            r for r in grid_results if not r.get("solve_failed") and r.get("feasible", False)
        ]
        if feasible_points:
            q_rec_values = [r["q_rec_kw"] for r in feasible_points]
            print(
                f"  Q_rec am Optimum je Gitterpunkt: Median={sorted(q_rec_values)[len(q_rec_values) // 2]:.4f} kW  "
                f"Min={min(q_rec_values):.4f} kW  Max={max(q_rec_values):.4f} kW",
                flush=True,
            )
    else:
        print(
            f"  Kein ueberall zulaessiger fester Betriebspunkt | {output['elapsed_s']:.0f}s",
            flush=True,
        )
    return output


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--workers", type=int, default=-1, help="-1 = alle Kerne")
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--skip-v5",
        action="store_true",
        help="nur V1-V4 (Kondensation hat Prioritaet vor der ungewichteten V5, s. Auftrag)",
    )
    return parser.parse_args()


def _write_output(path: Path, output: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True))


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
            f"{EXPECTED_BASE_FINGERPRINT!r} -- weicht von Befund 52/54 ab, V1 waere gegen "
            "den 15.22-%-Anker nicht interpretierbar."
        )

    workers = arguments.workers if arguments.workers > 0 else cpu_count()
    variants = list(VARIANTS)
    if arguments.skip_v5:
        variants = [v for v in variants if v[0] != "V5_net_kw_condensing"]

    print("=" * 78)
    print("MA4: Befund 52 mit Netto-Zielfunktion")
    print(f"Config       {arguments.config}  (Fingerprint {base_config.fingerprint})")
    print(f"Gitter       {arguments.grid_size}x{arguments.grid_size}")
    print(f"Varianten    {[v[0] for v in variants]}")
    print(f"Worker       {workers}")
    print(
        f"Vorab-Schaetzung: {len(variants)} x ~{RUNTIME_ESTIMATE_PER_VARIANT_S / 60.0:.0f} min "
        f"(Befund 54, DA1) = ~{len(variants) * RUNTIME_ESTIMATE_PER_VARIANT_S / 3600.0:.1f} h"
    )
    print(f"git/commit={env['git/commit']}  git/dirty={env['git/dirty']}")
    print("=" * 78, flush=True)

    total_start = time.perf_counter()
    results: dict[str, dict[str, Any]] = {}
    for label, recovery_mode, weighting in variants:
        results[label] = run_variant_study(
            base_config,
            grid_size=arguments.grid_size,
            workers=workers,
            label=label,
            recovery_mode=recovery_mode,
            weighting=weighting,
        )
        _write_output(
            arguments.output,
            {
                "environment": env,
                "config_path": str(arguments.config),
                "config_fingerprint": base_config.fingerprint,
                "grid_size": arguments.grid_size,
                "variants": results,
                "total_elapsed_s_so_far": time.perf_counter() - total_start,
            },
        )

        if label == "V1_gross":
            v1_elapsed = results[label]["elapsed_s"]
            projected_total_s = v1_elapsed * len(variants)
            print(
                f"\n*** Nach V1: {v1_elapsed:.0f}s gemessen. Hochrechnung fuer "
                f"{len(variants)} Varianten: {projected_total_s / 3600.0:.2f} h. ***",
                flush=True,
            )
            if projected_total_s > RUNTIME_WARNING_THRESHOLD_S:
                print(
                    "*** WARNUNG (Auftrag): Hochrechnung liegt ueber 4 h. ***",
                    flush=True,
                )
            v1_regret = results[label].get("regret")
            if v1_regret and v1_regret.get("n_points", 0) > 0:
                median_pct = v1_regret["relative_pct"]["median"]
                hit = abs(median_pct - ANCHOR_3X3_REGRET_MEDIAN_PCT) <= ANCHOR_TOLERANCE_PCT
                status = "TRIFFT" if hit else "TRIFFT NICHT"
                print(
                    f"*** V1-Anker-Vergleich: Median={median_pct:.3f} %, Befund-54-Anker="
                    f"{ANCHOR_3X3_REGRET_MEDIAN_PCT} % (+-{ANCHOR_TOLERANCE_PCT} pp) -- {status} ***",
                    flush=True,
                )
                if not hit:
                    print(
                        "*** Die wiederverwendete Mechanik weicht vom Befund-54-Stand ab -- "
                        "V2-V5 sind ohne Klaerung dieser Abweichung nicht interpretierbar. ***",
                        flush=True,
                    )
            else:
                print(
                    "*** V1 hat keinen ueberall zulaessigen festen Betriebspunkt gefunden -- "
                    "kein Regret-Anker-Vergleich moeglich. ***",
                    flush=True,
                )

    total_elapsed = time.perf_counter() - total_start
    print(f"\nGesamtlaufzeit: {total_elapsed / 3600.0:.2f} h", flush=True)
    print(f"Ergebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

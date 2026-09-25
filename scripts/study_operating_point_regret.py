r"""BB1-BB3: verschiebt sich der energieoptimale Betriebspunkt über den Störgrößenbereich.

Voruntersuchung zur RL-Eignung -- KEIN Reinforcement Learning, KEIN Netztraining, KEIN
Datensatz-Sweep. Nur ``solve_steady_state`` (``simulation/steady_state.py``) plus klassische
Optimierung je Gitterpunkt. Die Frage: hängt der Nutzen einer gelernten Policy (Abbildung
Störgrößen -> Sollwerte) überhaupt an etwas Messbarem, oder tut es ein fester Betriebspunkt?

BB1 -- Problemdefinition
-------------------------
Optimierungsvariablen (die späteren RL-Aktionen), Grenzen aus ``ActionSpaceConfig``
(``conf/process/greybox_gab_placeholder.yaml``, Abschnitt ``actions``), NICHT neu gesetzt:

    inlet_temperature_celsius, gas_velocity_m_per_s, nozzle_pressure_bar

FEST, nicht optimiert: ``spray_rate_kg_per_h`` am Nominalwert. Begründung: bei konstantem
Suspensionsfeststoffanteil bestimmt die Sprührate den Durchsatz. Ohne fixierten Durchsatz
ist "Energie minimieren" entartet -- das Optimum wäre Sprührate null, also keine Produktion.

Störgrößengitter: ``ambient_temperature_celsius`` x ``ambient_relative_humidity``, Bereiche
aus ``DisturbanceSpaceConfig``, äquidistant. ``suspension_solid_fraction`` fest am
Nominalwert (dritte Störgröße, hier nicht Teil des Gitters -- eine 3D-Störgrößenkarte wäre
für eine Vorabschätzung Überbau). Standardgitter 5x5 (Laufzeit gemessen: siehe unten);
7x7 über ``--grid-size`` erreichbar, laut Laufzeitmessung deutlich teurer (Faktor ~2, siehe
``--grid-size``-Hilfetext) -- "akzeptable Laufzeit" ist damit eine explizite Entscheidung
beim Aufruf, keine im Skript vorentschiedene.

Zielfunktion: ``diagnostics["energy_target"]`` im Beharrungszustand -- respektiert die im
Projekt noch offene Formulierung (``energy.target: power_kw | specific_kwh_per_kg`` in der
Config, siehe ``simulation/energy.py``), ohne sie hier neu zu entscheiden. Zusätzlich wird
IMMER der spezifische Verbrauch ``power_total_kw / (3600 * product_mass_flow_kg_per_s)``
berichtet, unabhängig davon, was ``energy.target`` gerade ist -- welche der beiden
Formulierungen am Ende die Zielgröße wird, ist im Projekt offen (siehe ``energy.py``-
Docstring), diese Studie beantwortet nur, wie sich der Betriebspunkt je nach Wahl anfühlt.

Harte Nebenbedingungen (ein Punkt, der eine davon verletzt, ist UNZULÄSSIG -- keine
Straffunktion verwischt das in der Berichterstattung; der interne Optimierer nutzt eine
Straffunktion nur als Suchheuristik, siehe ``_normalized_violation`` unten, aber jede
berichtete Zulässigkeit kommt aus einer eigenen, ungewichteten Prüfung am Endpunkt):

1. ``product_moisture_kg_per_kg <= material.critical_moisture_kg_per_kg`` (vorhandenes Feld).
2. ``u_mf(d50) <= gas_velocity <= u_terminal(d50)``, ausgewertet am Beharrungszustand mit
   ``simulation/fluidization.py`` (identische Rechnung wie die Diagnosegrößen aus
   ``solve_steady_state``).
3. ``bed_temperature_celsius <= 80`` -- ein frei gesetzter Platzhalter (Befund 46, siehe
   ``docs/04_stand_ap1.md``), separat berichtet, OB er bindet: bindet er, ist das ein
   Artefakt der (unbelegten) Grenze und keine Aussage über die reale Anlage.

Zusätzlich muss ``SteadyStateSolution.feasible`` (der Löser-eigene Gültigkeitscheck:
Sättigungsgrad <= 1, Partikelfeuchte im durch die Sorptionsisotherme plausiblen Bereich)
erfüllt sein -- das ist keine vierte Prozess-Nebenbedingung, sondern die Bedingung, dass der
gefundene "Beharrungszustand" überhaupt ein gültiger ist (siehe ``steady_state.py``-
Moduldocstring, Abschnitt "Zulässigkeit ist ein Flag, keine Exception"). Ohne sie ließe sich
ein numerisch entarteter Punkt für eine Optimierung "gewinnen".

BB2 -- Optimierung je Gitterpunkt
-----------------------------------
Drei Variablen, also ``scipy.optimize.differential_evolution`` (robust gegen nichtkonvexe
zulässige Bereiche -- das Fluidisierungsfenster und die Feuchtegrenze schneiden die Box auf
eine im Allgemeinen nicht konvexe Teilmenge), gefolgt von einer lokalen Verfeinerung
(Nelder-Mead, nicht gradientenbasiert: ``evaluate_derivative`` enthält ``clip``/``max``-
Terme, an denen ein Gradientenverfahren wie in ``steady_state.py`` begründet vermieden
werden sollte). Seed fest (``--seed``, Default 0).

Laufzeit-Kompromiss (gemessen auf dieser Maschine, 16 Kerne): ein einzelner
``solve_steady_state``-Aufruf mit den Löser-Defaults (100 Bisektionsschritte, Toleranz
1e-13) kostet ~0.46 s. ``differential_evolution`` mit Default-Einstellungen (popsize=15)
bräuchte dafür pro Gitterpunkt mehrere tausend Aufrufe -- unpraktikabel. Zwei Anpassungen,
beide NUR für die Suche, nicht für das berichtete Ergebnis:

* Die Bisektion läuft während der Suche mit gelockerter Toleranz
  (``--search-max-iterations``/``--search-tolerance``, Default 30 / 1e-7 statt 100 / 1e-13
  -- Residuum am Nominalpunkt damit ~2e-9 statt ~5e-15, für eine Optimierungssuche
  irrelevant). Der ENDGÜLTIG berichtete Punkt wird immer mit den Löser-Defaults neu
  gelöst.
* Parallelisiert wird ÜBER DIE GITTERPUNKTE, nicht innerhalb der DE-Population eines
  einzelnen Punkts (``differential_evolution(workers=...)``). Erster Entwurf machte es
  umgekehrt -- Populations-Parallelität über alle Kerne, dann eine (grundsätzlich
  serielle) Nelder-Mead-Verfeinerung. Gemessen: die Verfeinerung allein kostete dabei so
  viel Wandzeit wie die parallele DE-Phase davor, während 15 von 16 Kernen währenddessen
  nichts taten -- pro Gitterpunkt exakt das Muster "N Kandidaten parallel, dann einer
  allein, der so lange braucht wie die parallele Phase". Da die 25-49 Gitterpunkte
  echt unabhängig sind, verteilt ``main()`` sie stattdessen über einen Prozess-Pool
  (``pool.imap_unordered``, ``optimize_grid_point`` läuft je Punkt einsträngig), was
  ALLE Kerne über die GESAMTE Rechenzeit jedes Punkts (DE **und** Verfeinerung) auslastet
  und zusätzlich die Synchronisationsbarriere entfernt, die ``differential_evolution``
  sonst am Ende jeder Generation einlegt (eine langsame Population verzögert dort die
  gesamte Generation, hier nur ihren eigenen Gitterpunkt). Gemessene Kosten EINES
  Gitterpunkts, einsträngig, mit popsize=10/maxiter=50: DE ~90 s, Verfeinerung
  (Nelder-Mead, ``maxiter=200``) zusätzlich bis zu ~30-60 s -- macht mit 16 parallelen
  Gitterpunkten (``ceil(25/16)=2`` Runden) ein Vielfaches weniger Wandzeit für das
  gesamte Gitter als die anfängliche Population-Parallelität pro Punkt.

Nicht konvergierte Gitterpunkte (``differential_evolution``- oder Nelder-Mead-Ergebnis mit
``success=False``) werden GEMELDET (Feld ``de_converged``/``refine_converged`` je
Gitterpunkt), nicht übersprungen -- der beste gefundene Punkt wird trotzdem verwendet und
ausgewiesen.

BB3 -- Regret eines festen Betriebspunkts
--------------------------------------------
(a) Fester Betriebspunkt = der Sollwertsatz, der die MITTLERE Energie über das Gitter
    minimiert, UNTER DER NEBENBEDINGUNG, an JEDEM Gitterpunkt zulässig zu sein (sonst ist er
    als Vergleich untauglich, siehe Auftrag). Gleiche Optimierungsmaschinerie wie BB2, nur
    mit über das Gitter gemittelter Zielfunktion und über alle Gitterpunkte summierter
    Verletzung; die Gitter-Auswertung je Kandidat wird über einen eigenen, wiederverwendeten
    Multiprocessing-Pool parallelisiert (``differential_evolution`` selbst läuft dafür mit
    ``workers=1`` -- verschachtelte Pools sind nicht vorgesehen).

    Existiert kein überall zulässiger fester Punkt, wird das SOFORT und VOR allen weiteren
    Zahlen gemeldet (Kriterium: die beste im Hauptlauf gefundene Kandidatenlösung ist an
    mindestens einem Gitterpunkt unzulässig; zur Erhärtung läuft dann zusätzlich eine zweite
    Suche, die ausschließlich die summierte Verletzung minimiert -- bleibt auch deren
    Minimum > 0, ist das ein empirischer Befund dieser Suche, kein Beweis, aber die
    bestmögliche Aussage ohne erschöpfende Suche über eine kontinuierliche Box).

(b) Regret je Gitterpunkt = Energie(fester Punkt) - Energie(instanzweises Optimum aus BB2),
    Median/95-Perzentil/Maximum, absolut (kW bzw. kWh/kg) und relativ (%).

(c) Spannweite der optimalen Sollwerte über das Gitter je Variable, absolut und als Anteil
    der jeweiligen Stellgrößenspanne (``ActionBound.span``).

Einordnung (keine Empfehlung): der Regret ist die OBERGRENZE dessen, was eine adaptive
Regelung gegenüber einem festen Betriebspunkt einsparen kann. Im Promillebereich genügt ein
fester Betriebspunkt; im Prozentbereich ist eine Policy begründet.

Aufruf::

    python scripts/study_operating_point_regret.py --output artifacts/runs/bb1_bb3.json

BB4 (Sensitivität gegen die 80-°C-Bettemperaturgrenze) ist NICHT Teil dieses Skripts -- laut
Auftrag nur relevant, falls BB3 zeigt, dass ``bed_temperature_celsius <= 80`` an relevanten
Gitterpunkten bindet. Das Skript berichtet dafür die nötige Vorbedingung (Feld
``binding_bed_temperature`` je Gitterpunkt); die konkrete Sensitivitätsanalyse folgt separat.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, minimize

from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    PlaceholderConfigWarning,
    ProcessConfig,
    load_process_config,
)
from twinlink_ml.simulation.steady_state import SteadyStateSolution, solve_steady_state
from twinlink_ml.utils.tracking import environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "study_operating_point_regret.json"

BED_TEMPERATURE_LIMIT_C = 80.0
"""Frei gesetzter Platzhalter, Befund 46 -- siehe Moduldocstring."""

_PENALTY_WEIGHT_KW = 50.0
"""Skaliert die normierte Verletzung in die Größenordnung der Zielfunktion (~0.5-3 kW am
Nominalpunkt), damit sie in der Suche dominiert, ohne Fließkommaauflösung zu verlieren."""

_SOLVE_FAILURE_PENALTY_KW = 1.0e6
"""Rückgabewert, wenn die Bisektion in ``solve_steady_state`` keinen Vorzeichenwechsel
findet (``RuntimeError``) oder der Batch-Fall greift (``ValueError``) -- drängt den
Optimierer weg, ohne den Lauf abzubrechen."""

_TOLERANCE_MOISTURE_KG_PER_KG = 1.0e-4
"""EINE Toleranz für Zulässigkeit UND Bindung (kein zweiter, abweichender Wert an
anderer Stelle -- ein erster Gitterlauf zeigte, dass zwei verschiedene Toleranzen für
dieselbe Größe genau den Fehler erzeugen, den eine gemeinsame Konstante strukturell
ausschließt: 8 von 14 als "unzulässig" markierte Punkte hingen NUR daran, dass die
Zulässigkeitsprüfung strikte Gleichheit (1e-9) verlangte, während der Optimierer
das Optimum -- korrekt -- exakt auf die Feuchtegrenze treibt und sie nie exakter als
auf die eigene Suchpräzision genau trifft.

Bemessung von 1e-4 kg/kg: eine Größenordnung UNTER jeder real erreichbaren
Messauflösung für Partikelfeuchte (Labor-Feuchtebestimmung, Trockenschrank- oder
NIR-Verfahren, liegt praktisch nie besser als 1e-3 kg/kg absolut) -- die Toleranz
verwischt also keine an der Anlage messbare Verletzung. Gleichzeitig eine
Größenordnung ÜBER dem numerischen Restfehler des Lösers selbst
(``residual_inf_norm`` liegt am Report-Punkt typisch bei 1e-10 oder besser, siehe
``solve_steady_state``) -- die Toleranz blendet also keinen Löserfehler aus, sondern
ausschließlich die Differenz zwischen "auf die Nebenbedingung optimiert" und
"exakt auf ihr", die aus der endlichen Auflösung des Optimierers selbst folgt
(Nelder-Mead ``xatol=1e-4`` auf den AKTIONEN, siehe ``_optimize``)."""

_TOLERANCE_VELOCITY_M_PER_S = 1.0e-3
"""Analog: eine Größenordnung unter der Unsicherheit der Fluidisierungskorrelationen
selbst (10-30 %, siehe ``fluidization.py``-Docstring zu ``minimum_fluidization_velocity_
m_per_s``), eine Größenordnung über der Optimierer-Auflösung."""

_TOLERANCE_TEMPERATURE_K = 0.05
"""Analog: unter typischer Thermoelement-Auflösung (>=0.1 K), über der
Optimierer-Auflösung."""

DEFAULT_POPSIZE = 10
DEFAULT_MAXITER = 50
DEFAULT_DE_TOL = 1.0e-4
DEFAULT_SEARCH_MAX_ITERATIONS = 30
DEFAULT_SEARCH_TOLERANCE = 1.0e-7

_BASE_CONFIG: ProcessConfig | None = None
"""Modul-globaler Zustand -- vermeidet, die Config bei jedem Zielfunktionsaufruf neu zu
bepicken. Im Hauptprozess EINMAL in ``main()`` gesetzt; in den Worker-Prozessen über
``Pool(..., initializer=_set_base_config, initargs=(base_config,))`` (s. ``main()``), NICHT
per ``fork``-Vererbung -- das ursprüngliche Verfahren (Vererbung des im Hauptprozess
gesetzten Globals) setzte ``fork`` als Worker-Startmethode voraus und lief deshalb nie unter
Windows (``spawn``-Default dort: jeder Worker importiert das Modul frisch, ``_BASE_CONFIG``
bleibt ``None``). Der ``initializer`` funktioniert unter beiden Startmethoden identisch.
Siehe ``_set_base_config``."""


def _set_base_config(config: ProcessConfig) -> None:
    global _BASE_CONFIG
    _BASE_CONFIG = config


_OBJECTIVE_VALUE_FN: Callable[[SteadyStateSolution, ProcessConfig], dict[str, float]] | None = None
"""Optionaler Override der Zielgröße (MA4, ``study_operating_point_regret_net.py``) --
``None`` (Default) reproduziert die bisherige reine Bruttoleistung bitgleich, siehe
:func:`_objective_components`. Gleiches Modul-global-plus-Pool-Initializer-Muster wie
``_BASE_CONFIG`` (siehe dort): ein Funktionsobjekt ist unter Windows-``spawn`` nur
picklebar, wenn es ein Modul-Top-Level-Objekt ist (nicht in ``main()`` definiert) --
``study_operating_point_regret_net.py`` erfüllt das, seine Override-Funktion lebt auf
Modulebene dieses importierenden Skripts."""


def _set_objective_value_fn(
    fn: Callable[[SteadyStateSolution, ProcessConfig], dict[str, float]] | None,
) -> None:
    global _OBJECTIVE_VALUE_FN
    _OBJECTIVE_VALUE_FN = fn


def _objective_components(solution: SteadyStateSolution, base_config: ProcessConfig) -> dict[str, float]:
    """Zielgröße plus Diagnosefelder für einen Beharrungszustand.

    Ohne Override (``_OBJECTIVE_VALUE_FN is None``, der Stand vor MA4): reine
    Bruttoleistung, bitgleich zu ``solution.diagnostics["energy_target"]`` wie zuvor direkt
    inline gelesen -- ``objective_value == gross_energy_target_kw``,
    ``q_rec_kw``/``recovery_credit_kw`` beide 0.0. Mit Override (MA4-Netto-Varianten)
    liefert die injizierte Funktion dieselben vier Schlüssel, mit
    ``objective_value != gross_energy_target_kw``.
    """
    if _OBJECTIVE_VALUE_FN is not None:
        return _OBJECTIVE_VALUE_FN(solution, base_config)
    gross_kw = float(solution.diagnostics["energy_target"])
    return {
        "objective_value": gross_kw,
        "gross_energy_target_kw": gross_kw,
        "q_rec_kw": 0.0,
        "recovery_credit_kw": 0.0,
    }


def _suppress_placeholder_warning() -> None:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)


def build_variant_config(
    base: ProcessConfig, *, t_in: float, u_gas: float, p_nozzle: float, t_amb: float, phi_amb: float
) -> ProcessConfig:
    """Config mit genau den fünf hier variierten Nominalwerten überschrieben.

    ``solve_steady_state`` liest Stell-/Störgrößen ausschließlich über
    ``simulator.nominal_action``/``nominal_disturbance``, also aus den ``nominal``-Feldern
    der Config (siehe ``greybox.py``) -- es gibt keinen Parameterpfad, der einen konkreten
    Aktions-/Störvektor an den Löser durchreicht. Eine Variante je Auswertung ist deshalb
    der einzige Weg, den Löser an einem beliebigen Betriebspunkt auszuwerten; alle übrigen
    Felder (inkl. Grenzen und ``max_rate_per_s``) bleiben unverändert.
    """
    actions = dataclasses.replace(
        base.actions,
        inlet_temperature_celsius=dataclasses.replace(
            base.actions.inlet_temperature_celsius, nominal=t_in
        ),
        gas_velocity_m_per_s=dataclasses.replace(
            base.actions.gas_velocity_m_per_s, nominal=u_gas
        ),
        nozzle_pressure_bar=dataclasses.replace(
            base.actions.nozzle_pressure_bar, nominal=p_nozzle
        ),
    )
    disturbances = dataclasses.replace(
        base.disturbances,
        ambient_temperature_celsius=dataclasses.replace(
            base.disturbances.ambient_temperature_celsius, nominal=t_amb
        ),
        ambient_relative_humidity=dataclasses.replace(
            base.disturbances.ambient_relative_humidity, nominal=phi_amb
        ),
    )
    return dataclasses.replace(base, actions=actions, disturbances=disturbances)


def evaluate_point(
    t_in: float,
    u_gas: float,
    p_nozzle: float,
    t_amb: float,
    phi_amb: float,
    *,
    max_iterations: int,
    tolerance: float,
    base_config: ProcessConfig,
) -> SteadyStateSolution:
    """Bewusst OHNE eigenes ``temperature_bracket_celsius`` -- ein erster Versuch, das
    Modul-Default (``ambient+10`` / ``max(inlet, susp)+5``, siehe ``steady_state.py``)
    durch eine breitere, "physikalisch begründete" Marge zu ersetzen, hat sich als
    Fehlschlag erwiesen: ``ambient+10`` ist keine willkürliche Wahl, sondern hält die
    äußere Bisektion aktiv im Trocknungsregime (siehe dortiger Docstring, Abschnitt zur
    unteren Bracket-Grenze) -- eine niedrigere Grenze reicht in die Kondensationsbranche
    hinein, in der die INNERE Feuchte-Bisektion strukturell keinen Vorzeichenwechsel
    findet. Das brach nicht nur die wenigen Randfälle, die es beheben sollte, sondern
    praktisch jede Kandidatenauswertung (siehe Modul-Docstring, Abschnitt
    "Laufzeit-Kompromiss"). Der eigentliche Befund aus dem ersten vollen Gitterlauf --
    Such- und Report-Präzision fanden am Bracket-Rand unterschiedliches Vorzeichen, weil
    sie MIT UNTERSCHIEDLICHER Bisektionspräzision liefen -- wird stattdessen dadurch
    behoben, dass ``optimize_grid_point``/``find_fixed_operating_point`` für den
    ENDGÜLTIGEN Report dieselbe (Such-)Präzision verwenden wie während der Suche, siehe
    dort. Ein danach verbleibender Bracket-Fehlschlag ist damit ein echter Befund
    (der angenommene Beharrungszustand liegt außerhalb des dokumentierten Gültigkeits-
    bereichs des Lösers), keine Artefakt-Diagnose mehr.
    """
    config = build_variant_config(
        base_config, t_in=t_in, u_gas=u_gas, p_nozzle=p_nozzle, t_amb=t_amb, phi_amb=phi_amb
    )
    simulator = GreyBoxFluidBedGranulator(config)
    return solve_steady_state(simulator, max_iterations=max_iterations, tolerance=tolerance)


@dataclass(frozen=True, slots=True)
class ConstraintCheck:
    """BB1s drei harte Nebenbedingungen plus Löser-Gültigkeit, ausgewertet an einer Lösung."""

    moisture_kg_per_kg: float
    moisture_violation_kg_per_kg: float
    gas_velocity_m_per_s: float
    minimum_fluidization_velocity_m_per_s: float
    terminal_velocity_m_per_s: float
    bed_temperature_celsius: float
    bed_temperature_violation_k: float
    solver_feasible: bool
    feasible: bool
    binding_moisture: bool
    binding_gas_lower: bool
    binding_gas_upper: bool
    binding_bed_temperature: bool
    normalized_violation: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def check_constraints(
    solution: SteadyStateSolution, *, critical_moisture_kg_per_kg: float, gas_velocity_m_per_s: float
) -> ConstraintCheck:
    moisture = solution.particle_moisture_kg_per_kg
    moisture_violation = max(0.0, moisture - critical_moisture_kg_per_kg)

    u_mf = solution.diagnostics["minimum_fluidization_velocity_m_per_s"]
    u_terminal = solution.diagnostics["terminal_velocity_m_per_s"]
    gas_lower_violation = max(0.0, u_mf - gas_velocity_m_per_s)
    gas_upper_violation = max(0.0, gas_velocity_m_per_s - u_terminal)

    bed_temperature_violation = max(0.0, solution.bed_temperature_celsius - BED_TEMPERATURE_LIMIT_C)

    feasible = (
        moisture_violation <= _TOLERANCE_MOISTURE_KG_PER_KG
        and gas_lower_violation <= _TOLERANCE_VELOCITY_M_PER_S
        and gas_upper_violation <= _TOLERANCE_VELOCITY_M_PER_S
        and bed_temperature_violation <= _TOLERANCE_TEMPERATURE_K
        and solution.feasible
    )

    window = max(u_terminal - u_mf, 1e-3)
    normalized_violation = (
        moisture_violation / max(critical_moisture_kg_per_kg, 1e-6)
        + gas_lower_violation / window
        + gas_upper_violation / window
        + bed_temperature_violation / 10.0
        + (0.0 if solution.feasible else 1.0)
    )

    return ConstraintCheck(
        moisture_kg_per_kg=moisture,
        moisture_violation_kg_per_kg=moisture_violation,
        gas_velocity_m_per_s=gas_velocity_m_per_s,
        minimum_fluidization_velocity_m_per_s=u_mf,
        terminal_velocity_m_per_s=u_terminal,
        bed_temperature_celsius=solution.bed_temperature_celsius,
        bed_temperature_violation_k=bed_temperature_violation,
        solver_feasible=solution.feasible,
        feasible=feasible,
        binding_moisture=abs(moisture - critical_moisture_kg_per_kg) <= _TOLERANCE_MOISTURE_KG_PER_KG,
        binding_gas_lower=abs(gas_velocity_m_per_s - u_mf) <= _TOLERANCE_VELOCITY_M_PER_S,
        binding_gas_upper=abs(u_terminal - gas_velocity_m_per_s) <= _TOLERANCE_VELOCITY_M_PER_S,
        binding_bed_temperature=abs(BED_TEMPERATURE_LIMIT_C - solution.bed_temperature_celsius)
        <= _TOLERANCE_TEMPERATURE_K,
        normalized_violation=normalized_violation,
    )


def specific_energy_kwh_per_kg(solution: SteadyStateSolution) -> float:
    product_flow = solution.diagnostics["product_mass_flow_kg_per_s"]
    total_power_kw = solution.diagnostics["power_total_kw"]
    return total_power_kw / (3600.0 * max(product_flow, 1e-12))


def _penalized_energy(
    x: np.ndarray, t_amb: float, phi_amb: float, max_iterations: int, tolerance: float
) -> float:
    """Zielfunktion für die Suche (BB2, je Gitterpunkt). Modul-Ebene für Picklebarkeit."""
    assert _BASE_CONFIG is not None
    t_in, u_gas, p_nozzle = x
    try:
        solution = evaluate_point(
            t_in,
            u_gas,
            p_nozzle,
            t_amb,
            phi_amb,
            max_iterations=max_iterations,
            tolerance=tolerance,
            base_config=_BASE_CONFIG,
        )
    except (RuntimeError, ValueError):
        return _SOLVE_FAILURE_PENALTY_KW
    check = check_constraints(
        solution,
        critical_moisture_kg_per_kg=_BASE_CONFIG.material.critical_moisture_kg_per_kg,
        gas_velocity_m_per_s=u_gas,
    )
    objective = _objective_components(solution, _BASE_CONFIG)["objective_value"]
    return float(objective + _PENALTY_WEIGHT_KW * check.normalized_violation)


@dataclass(frozen=True, slots=True)
class OptimizationBudget:
    """Suchbudget für ``differential_evolution`` + Nelder-Mead, geteilt von BB2 und BB3(a)."""

    seed: int
    popsize: int
    maxiter: int
    de_tol: float
    search_max_iterations: int
    search_tolerance: float
    workers: int


def _optimize(
    objective: Callable[..., float],
    bounds: list[tuple[float, float]],
    args: tuple,
    budget: OptimizationBudget,
    *,
    de_workers: int | Callable[[Callable, list], list],
) -> tuple[Any, Any, float, float]:
    """DE + Nelder-Mead-Verfeinerung, gemeinsam für BB2 und BB3(a) verwendet."""
    de_start = time.perf_counter()
    de_result = differential_evolution(
        objective,
        bounds,
        args=args,
        seed=budget.seed,
        polish=False,
        workers=de_workers,
        updating="deferred" if de_workers != 1 else "immediate",
        popsize=budget.popsize,
        maxiter=budget.maxiter,
        tol=budget.de_tol,
    )
    de_elapsed = time.perf_counter() - de_start

    refine_start = time.perf_counter()
    refine_result = minimize(
        objective,
        de_result.x,
        args=args,
        method="Nelder-Mead",
        bounds=bounds,
        options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 200, "adaptive": True},
    )
    refine_elapsed = time.perf_counter() - refine_start

    return de_result, refine_result, de_elapsed, refine_elapsed


def optimize_grid_point(
    t_amb: float, phi_amb: float, bounds: list[tuple[float, float]], budget: OptimizationBudget
) -> dict[str, Any]:
    """Ein Gitterpunkt (BB2) -- LÄUFT EINSTRÄNGIG (``de_workers=1``), absichtlich.

    Parallelisiert wird NICHT hier (innerhalb der DE-Population eines einzelnen
    Gitterpunkts), sondern in ``main()`` ÜBER die Gitterpunkte hinweg (``pool.
    imap_unordered`` auf dieser Funktion) -- siehe Modul-Docstring, Abschnitt
    "Laufzeit-Kompromiss", für die Messung, die diese Entscheidung begründet: Populations-
    Parallelität beschleunigt nur die DE-Phase und lässt alle Kerne bis auf einen während
    der (grundsätzlich seriellen) Nelder-Mead-Verfeinerung ungenutzt -- pro Gitterpunkt
    genau das Muster "N Kandidaten parallel, danach einer allein, der so lange braucht wie
    die parallele Phase zuvor". 25-49 Gitterpunkte sind dagegen echt unabhängig
    voneinander: sie über den Pool zu verteilen hält alle Kerne während des GESAMTEN
    Laufs (DE **und** Verfeinerung jedes einzelnen Punkts) beschäftigt und vermeidet
    zusätzlich die Synchronisationsbarriere, die ``differential_evolution`` sonst am Ende
    jeder Generation einlegt (eine langsame Population verzögert dort die gesamte
    Generation, hier nur ihren eigenen Gitterpunkt).
    """
    assert _BASE_CONFIG is not None
    args = (t_amb, phi_amb, budget.search_max_iterations, budget.search_tolerance)
    de_result, refine_result, de_elapsed, refine_elapsed = _optimize(
        _penalized_energy, bounds, args, budget, de_workers=1
    )
    best_x = refine_result.x if refine_result.fun <= de_result.fun else de_result.x

    t_in, u_gas, p_nozzle = (float(v) for v in best_x)
    # Gleiche Präzision wie während der Suche (siehe ``evaluate_point``-Docstring) --
    # eine höhere Präzision NUR für den Report-Aufruf war die Ursache der
    # Bracket-Fehlschläge im ersten vollen Gitterlauf, nicht deren Heilmittel.
    solve_failed = False
    try:
        solution = evaluate_point(
            t_in,
            u_gas,
            p_nozzle,
            t_amb,
            phi_amb,
            max_iterations=budget.search_max_iterations,
            tolerance=budget.search_tolerance,
            base_config=_BASE_CONFIG,
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
        # DE erschöpft bei diesem Budget planmäßig ``maxiter`` (siehe Modul-Docstring,
        # "Laufzeit-Kompromiss") -- das ist KEIN Fehlschlag, solange die anschließende
        # Verfeinerung konvergiert. ``converged`` bewertet deshalb nur die Verfeinerung
        # (den Stand, der tatsächlich berichtet wird); ``de_converged`` bleibt als
        # Diagnosefeld erhalten, ist aber für sich genommen kein Warnsignal.
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

    check = check_constraints(
        solution,
        critical_moisture_kg_per_kg=_BASE_CONFIG.material.critical_moisture_kg_per_kg,
        gas_velocity_m_per_s=u_gas,
    )
    components = _objective_components(solution, _BASE_CONFIG)
    result.update(
        {
            "energy_target": components["objective_value"],
            "energy_target_unit": _BASE_CONFIG.energy.target,
            "gross_energy_target_kw": components["gross_energy_target_kw"],
            "q_rec_kw": components["q_rec_kw"],
            "recovery_credit_kw": components["recovery_credit_kw"],
            "specific_energy_kwh_per_kg": specific_energy_kwh_per_kg(solution),
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


def _grid_point_task(
    task: tuple[int, float, float, list[tuple[float, float]], OptimizationBudget],
) -> dict[str, Any]:
    """Ein Element für ``pool.imap_unordered`` in ``main()`` -- ein Gitterpunkt komplett
    (DE + Verfeinerung + finaler Löser-Aufruf), einsträngig in EINEM Worker-Prozess.
    Der Index geht mit durch, weil ``imap_unordered`` die Ankunftsreihenfolge NICHT
    garantiert -- ``compute_regret``/``compute_setpoint_spread`` erwarten aber dieselbe
    Reihenfolge wie ``grid_points``.
    """
    index, t_amb, phi_amb, bounds, budget = task
    start = time.perf_counter()
    result = optimize_grid_point(t_amb, phi_amb, bounds, budget)
    result["_grid_index"] = index
    result["_wall_elapsed_s"] = time.perf_counter() - start
    return result


def _fixed_point_worker(
    x: tuple[float, float, float], t_amb: float, phi_amb: float, max_iterations: int, tolerance: float
) -> tuple[float, float] | None:
    """Ein Gitterpunkt für einen festen Kandidaten -- Ziel des Pools in BB3(a)."""
    assert _BASE_CONFIG is not None
    t_in, u_gas, p_nozzle = x
    try:
        solution = evaluate_point(
            t_in,
            u_gas,
            p_nozzle,
            t_amb,
            phi_amb,
            max_iterations=max_iterations,
            tolerance=tolerance,
            base_config=_BASE_CONFIG,
        )
    except (RuntimeError, ValueError):
        return None
    check = check_constraints(
        solution,
        critical_moisture_kg_per_kg=_BASE_CONFIG.material.critical_moisture_kg_per_kg,
        gas_velocity_m_per_s=u_gas,
    )
    objective = _objective_components(solution, _BASE_CONFIG)["objective_value"]
    return float(objective), check.normalized_violation


def _fixed_point_objective(
    x: np.ndarray,
    grid_points: list[tuple[float, float]],
    pool: Pool,
    max_iterations: int,
    tolerance: float,
    energy_weight: float,
) -> float:
    """``energy_weight`` ist absichtlich NICHT keyword-only: ``differential_evolution``
    reicht ``args`` rein positional an die Zielfunktion durch (``func(x, *args)``,
    siehe ``scipy._lib._util._FunctionWrapper``) -- ein keyword-only-Parameter hier
    führte zu ``TypeError: takes 5 positional arguments but 6 were given``, weil das
    fünfte ``args``-Element (der Gewichtungsfaktor) keinen keyword-Slot findet."""
    tasks = [(tuple(float(v) for v in x), t_amb, phi_amb, max_iterations, tolerance) for t_amb, phi_amb in grid_points]
    results = pool.starmap(_fixed_point_worker, tasks)
    energies: list[float] = []
    violation_sum = 0.0
    for outcome in results:
        if outcome is None:
            energies.append(_SOLVE_FAILURE_PENALTY_KW)
            violation_sum += 5.0
        else:
            energy, violation = outcome
            energies.append(energy)
            violation_sum += violation
    return energy_weight * float(np.mean(energies)) + _PENALTY_WEIGHT_KW * violation_sum


def find_fixed_operating_point(
    grid_points: list[tuple[float, float]],
    bounds: list[tuple[float, float]],
    budget: OptimizationBudget,
    pool: Pool,
) -> dict[str, Any]:
    """BB3(a): fester Betriebspunkt, an jedem Gitterpunkt zulässig, minimale Mittelenergie."""
    args = (grid_points, pool, budget.search_max_iterations, budget.search_tolerance, 1.0)
    de_result, refine_result, de_elapsed, refine_elapsed = _optimize(
        _fixed_point_objective, bounds, args, budget, de_workers=1
    )
    best_x = refine_result.x if refine_result.fun <= de_result.fun else de_result.x
    t_in, u_gas, p_nozzle = (float(v) for v in best_x)

    per_point: list[dict[str, Any]] = []
    all_feasible = True
    for t_amb, phi_amb in grid_points:
        try:
            solution = evaluate_point(
                t_in,
                u_gas,
                p_nozzle,
                t_amb,
                phi_amb,
                max_iterations=budget.search_max_iterations,
                tolerance=budget.search_tolerance,
                base_config=_BASE_CONFIG,
            )
        except (RuntimeError, ValueError) as exc:
            per_point.append(
                {
                    "ambient_temperature_celsius": t_amb,
                    "ambient_relative_humidity": phi_amb,
                    "feasible": False,
                    "solve_failed": True,
                    "solve_failure_reason": str(exc),
                }
            )
            all_feasible = False
            continue
        check = check_constraints(
            solution,
            critical_moisture_kg_per_kg=_BASE_CONFIG.material.critical_moisture_kg_per_kg,
            gas_velocity_m_per_s=u_gas,
        )
        all_feasible = all_feasible and check.feasible
        components = _objective_components(solution, _BASE_CONFIG)
        per_point.append(
            {
                "ambient_temperature_celsius": t_amb,
                "ambient_relative_humidity": phi_amb,
                "feasible": check.feasible,
                "solve_failed": False,
                "energy_target": components["objective_value"],
                "gross_energy_target_kw": components["gross_energy_target_kw"],
                "q_rec_kw": components["q_rec_kw"],
                "recovery_credit_kw": components["recovery_credit_kw"],
                "specific_energy_kwh_per_kg": specific_energy_kwh_per_kg(solution),
                "infeasibility_reasons": list(solution.infeasibility_reasons),
                "constraint_check": check.as_dict(),
            }
        )

    return {
        "inlet_temperature_celsius": t_in,
        "gas_velocity_m_per_s": u_gas,
        "nozzle_pressure_bar": p_nozzle,
        "feasible_everywhere": all_feasible,
        "de_converged": bool(de_result.success),
        "de_message": str(de_result.message),
        "de_elapsed_s": de_elapsed,
        "refine_converged": bool(refine_result.success),
        "refine_elapsed_s": refine_elapsed,
        "per_grid_point": per_point,
    }


def verify_no_feasible_fixed_point_exists(
    grid_points: list[tuple[float, float]],
    bounds: list[tuple[float, float]],
    budget: OptimizationBudget,
    pool: Pool,
) -> dict[str, Any]:
    """Zweite Suche, NUR gegen die summierte Verletzung, zur Erhärtung von BB3(a).

    Kein Beweis (kontinuierliche Box, keine erschöpfende Suche) -- aber die bestmögliche
    Aussage ohne sie: bleibt das Minimum der Verletzung > 0, hat auch eine ausschließlich
    auf Zulässigkeit zielende globale Suche keinen überall zulässigen Punkt gefunden.
    """
    args = (grid_points, pool, budget.search_max_iterations, budget.search_tolerance, 0.0)
    de_result, refine_result, de_elapsed, refine_elapsed = _optimize(
        _fixed_point_objective, bounds, args, budget, de_workers=1
    )
    best_fun = min(de_result.fun, refine_result.fun)
    return {
        "minimum_violation_found": best_fun / _PENALTY_WEIGHT_KW,
        "de_elapsed_s": de_elapsed,
        "refine_elapsed_s": refine_elapsed,
    }


def build_grid(base: ProcessConfig, grid_size: int) -> list[tuple[float, float]]:
    disturbances = base.disturbances
    t_amb_values = np.linspace(
        disturbances.ambient_temperature_celsius.minimum,
        disturbances.ambient_temperature_celsius.maximum,
        grid_size,
    )
    phi_values = np.linspace(
        disturbances.ambient_relative_humidity.minimum,
        disturbances.ambient_relative_humidity.maximum,
        grid_size,
    )
    return [(float(t), float(phi)) for t in t_amb_values for phi in phi_values]


def action_bounds(base: ProcessConfig) -> list[tuple[float, float]]:
    actions = base.actions
    return [
        (actions.inlet_temperature_celsius.minimum, actions.inlet_temperature_celsius.maximum),
        (actions.gas_velocity_m_per_s.minimum, actions.gas_velocity_m_per_s.maximum),
        (actions.nozzle_pressure_bar.minimum, actions.nozzle_pressure_bar.maximum),
    ]


def compute_regret(grid_results: list[dict[str, Any]], fixed_point: dict[str, Any]) -> dict[str, Any]:
    absolute_kw: list[float] = []
    relative_pct: list[float] = []
    for grid_point, fixed in zip(grid_results, fixed_point["per_grid_point"], strict=True):
        if grid_point.get("solve_failed") or not grid_point.get("feasible", False):
            continue
        if fixed.get("solve_failed") or not fixed.get("feasible", False):
            continue
        instance_optimum = grid_point["energy_target"]
        fixed_energy = fixed["energy_target"]
        regret = fixed_energy - instance_optimum
        absolute_kw.append(regret)
        denom = abs(instance_optimum) if abs(instance_optimum) > 1e-9 else 1e-9
        relative_pct.append(100.0 * regret / denom)

    if not absolute_kw:
        return {"n_points": 0}

    absolute_arr = np.asarray(absolute_kw)
    relative_arr = np.asarray(relative_pct)
    return {
        "n_points": len(absolute_kw),
        "absolute_kw": {
            "median": float(np.median(absolute_arr)),
            "p95": float(np.percentile(absolute_arr, 95)),
            "max": float(np.max(absolute_arr)),
        },
        "relative_pct": {
            "median": float(np.median(relative_arr)),
            "p95": float(np.percentile(relative_arr, 95)),
            "max": float(np.max(relative_arr)),
        },
    }


def compute_setpoint_spread(grid_results: list[dict[str, Any]], base: ProcessConfig) -> dict[str, Any]:
    feasible = [r for r in grid_results if not r.get("solve_failed") and r.get("feasible", False)]
    variables = {
        "inlet_temperature_celsius": base.actions.inlet_temperature_celsius,
        "gas_velocity_m_per_s": base.actions.gas_velocity_m_per_s,
        "nozzle_pressure_bar": base.actions.nozzle_pressure_bar,
    }
    spread: dict[str, Any] = {"n_feasible_points": len(feasible)}
    if not feasible:
        return spread
    for name, bound in variables.items():
        values = np.asarray([r[name] for r in feasible])
        span = float(np.max(values) - np.min(values))
        spread[name] = {
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "span_absolute": span,
            "span_fraction_of_bound": span / bound.span,
        }
    return spread


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=5,
        choices=[5, 7],
        help="5x5 (~1h Laufzeit, Default) oder 7x7 (~2-2.5h, gemessen mit --workers=-1)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--popsize", type=int, default=DEFAULT_POPSIZE)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--de-tol", type=float, default=DEFAULT_DE_TOL)
    parser.add_argument("--search-max-iterations", type=int, default=DEFAULT_SEARCH_MAX_ITERATIONS)
    parser.add_argument("--search-tolerance", type=float, default=DEFAULT_SEARCH_TOLERANCE)
    parser.add_argument("--workers", type=int, default=-1, help="-1 = alle Kerne (os.cpu_count())")
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--skip-fixed-point",
        action="store_true",
        help="nur BB1/BB2 (je-Gitterpunkt-Optima), kein Regret -- fuer schnelle Testlaeufe",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    _suppress_placeholder_warning()

    env = environment_fingerprint()
    if env["git/dirty"]:
        print(
            "WARNUNG: Arbeitsbaum ist dirty -- git/dirty wird in jeden Record geschrieben, "
            "der Lauf ist trotzdem nicht vollständig reproduzierbar.",
            flush=True,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base_config = load_process_config(arguments.config)
    _set_base_config(base_config)

    workers = arguments.workers if arguments.workers > 0 else cpu_count()
    budget = OptimizationBudget(
        seed=arguments.seed,
        popsize=arguments.popsize,
        maxiter=arguments.maxiter,
        de_tol=arguments.de_tol,
        search_max_iterations=arguments.search_max_iterations,
        search_tolerance=arguments.search_tolerance,
        workers=workers,
    )
    bounds = action_bounds(base_config)
    grid_points = build_grid(base_config, arguments.grid_size)

    print("=" * 78)
    print(f"Config              {arguments.config}")
    print(f"Fingerprint         {base_config.fingerprint}")
    print(f"Gitter              {arguments.grid_size}x{arguments.grid_size} = {len(grid_points)} Punkte")
    print(f"Fest (Nominalwert)  spray_rate={base_config.actions.spray_rate_kg_per_h.nominal} kg/h, "
          f"suspension_solid_fraction={base_config.disturbances.suspension_solid_fraction.nominal}")
    print(f"Zielgröße           energy_target = {base_config.energy.target}")
    print(f"Worker              {workers}")
    print(f"git/commit={env['git/commit']}  git/dirty={env['git/dirty']}")
    print("=" * 78)

    # EIN Pool für den gesamten Lauf (BB2-Verteilung + BB3), niemals pro Aufruf neu erzeugt --
    # siehe Docstring von ``optimize_grid_point``: mehrfach angelegte Pools über viele
    # sequentielle ``differential_evolution``-Aufrufe hinweg haben im Test einen Deadlock
    # verursacht (fork() mit Thread-Altlasten des vorherigen Pools).
    #
    # initializer/initargs statt Vererbung über fork(): setzt ``_BASE_CONFIG`` in jedem
    # Worker explizit, unabhängig von der Startmethode (fork unter Linux, spawn unter
    # Windows) -- s. Docstring von ``_BASE_CONFIG``.
    with Pool(workers, initializer=_set_base_config, initargs=(base_config,)) as pool:
        print(f"\n--- BB2: Optimierung je Gitterpunkt ({len(grid_points)} Punkte, über {workers} "
              f"Worker verteilt) " + "-" * 10)
        print(
            "Hinweis: ``differential_evolution`` erschöpft bei diesem Suchbudget planmäßig "
            "sein Iterationslimit (kein Fehlschlag, siehe Modul-Docstring) -- konvergiert "
            "werten muss nur die anschließende Nelder-Mead-Verfeinerung, die den tatsächlich "
            "berichteten Punkt liefert. Nur DIESE Konvergenz wird unten als "
            "'nicht konvergiert' geflaggt; ``de_converged`` bleibt separat im Artefakt."
        )
        tasks = [
            (index, t_amb, phi_amb, bounds, budget)
            for index, (t_amb, phi_amb) in enumerate(grid_points)
        ]
        grid_results_by_index: dict[int, dict[str, Any]] = {}
        for n_done, result in enumerate(pool.imap_unordered(_grid_point_task, tasks), start=1):
            index = result["_grid_index"]
            t_amb, phi_amb = grid_points[index]
            status = "OK" if not result.get("solve_failed") and result.get("feasible") else "UNZULAESSIG/FEHLER"
            conv = "" if result["converged"] else "  [VERFEINERUNG NICHT KONVERGIERT]"
            print(
                f"[{n_done:2d}/{len(grid_points)}] (Punkt {index + 1}) T_amb={t_amb:5.1f} °C "
                f"phi={phi_amb:.2f}  -> {status:20s} ({result['_wall_elapsed_s']:5.1f}s){conv}"
            )
            grid_results_by_index[index] = result
        grid_results = [grid_results_by_index[i] for i in range(len(grid_points))]

        n_infeasible = sum(1 for r in grid_results if r.get("solve_failed") or not r.get("feasible", False))
        n_not_converged = sum(1 for r in grid_results if not r["converged"])
        n_de_budget_exhausted = sum(1 for r in grid_results if not r["de_converged"])
        print(f"\nGitterpunkte gesamt: {len(grid_results)}, unzulässig/Fehler: {n_infeasible}, "
              f"Verfeinerung nicht konvergiert: {n_not_converged} "
              f"(DE-Suchbudget erschöpft, planmäßig: {n_de_budget_exhausted}/{len(grid_results)})")

        output: dict[str, Any] = {
            "environment": env,
            "config_path": str(arguments.config),
            "config_fingerprint": base_config.fingerprint,
            "grid_size": arguments.grid_size,
            "budget": dataclasses.asdict(budget),
            "grid_results": grid_results,
        }

        if not arguments.skip_fixed_point:
            print(
                f"\n--- BB3(a): fester Betriebspunkt (über {len(grid_points)} Gitterpunkte gemittelt) "
                + "-" * 5
            )
            fixed_point = find_fixed_operating_point(grid_points, bounds, budget, pool)
            if not fixed_point["feasible_everywhere"]:
                print(
                    "\n*** BEFUND: kein an JEDEM Gitterpunkt zulässiger fester Betriebspunkt "
                    "gefunden. ***"
                )
                verification = verify_no_feasible_fixed_point_exists(grid_points, bounds, budget, pool)
                print(
                    f"Verifikations-Suche (nur Verletzung minimiert): "
                    f"minimale gefundene Verletzung = {verification['minimum_violation_found']:.4f} "
                    "(0.0 würde Zulässigkeit bedeuten)."
                )
                print(
                    "Kein Beweis (kontinuierliche Box, keine erschöpfende Suche), aber auch eine "
                    "zweite, ausschließlich auf Zulässigkeit zielende globale Suche findet keinen "
                    "Punkt -- ein fester Betriebspunkt ist als Vergleich für BB3(b)/(c) damit "
                    "nicht zu bilden; diese Zahlen entfallen."
                )
                output["fixed_point"] = fixed_point
                output["fixed_point_verification"] = verification
            else:
                print(
                    f"Fester Punkt: T_in={fixed_point['inlet_temperature_celsius']:.2f} °C, "
                    f"u_gas={fixed_point['gas_velocity_m_per_s']:.3f} m/s, "
                    f"p_nozzle={fixed_point['nozzle_pressure_bar']:.3f} bar "
                    f"-- an allen {len(grid_points)} Gitterpunkten zulässig."
                )
                regret = compute_regret(grid_results, fixed_point)
                spread = compute_setpoint_spread(grid_results, base_config)
                output["fixed_point"] = fixed_point
                output["regret"] = regret
                output["setpoint_spread"] = spread

                print("\n--- BB3(b): Regret ------------------------------------------------")
                if regret["n_points"] == 0:
                    print("Keine gemeinsam zulässigen Punkte für einen Regret-Vergleich.")
                else:
                    print(f"n = {regret['n_points']} Gitterpunkte")
                    print(
                        f"absolut [kW]   Median={regret['absolute_kw']['median']:.4f}  "
                        f"P95={regret['absolute_kw']['p95']:.4f}  Max={regret['absolute_kw']['max']:.4f}"
                    )
                    print(
                        f"relativ [%]    Median={regret['relative_pct']['median']:.3f}  "
                        f"P95={regret['relative_pct']['p95']:.3f}  Max={regret['relative_pct']['max']:.3f}"
                    )

                print("\n--- BB3(c): Spannweite optimaler Sollwerte über das Gitter ---------")
                for name in (
                    "inlet_temperature_celsius",
                    "gas_velocity_m_per_s",
                    "nozzle_pressure_bar",
                ):
                    entry = spread.get(name)
                    if entry is None:
                        continue
                    print(
                        f"{name:28s} [{entry['minimum']:7.3f}, {entry['maximum']:7.3f}]  "
                        f"Spanne={entry['span_absolute']:7.3f}  "
                        f"({100 * entry['span_fraction_of_bound']:5.1f} % der Stellgrößenspanne)"
                    )

                print(
                    "\nEinordnung: Regret ist die OBERGRENZE dessen, was eine adaptive Regelung "
                    "gegenüber diesem festen Betriebspunkt einsparen kann. Promillebereich -> "
                    "fester Betriebspunkt genügt; Prozentbereich -> Policy begründet. "
                    "(Keine Empfehlung -- reine Einordnung.)"
                )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nErgebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

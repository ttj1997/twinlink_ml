r"""VA3/XA1: Bewertungsmetrik -- Anteil des geschlossenen Regrets.

.. math:: \text{geschlossen} = \frac{E_{fest} - E_{Agent}}{E_{fest} - E_{Orakel}}

0 heisst: nicht besser als der feste Betriebspunkt. 1 heisst: das Optimum erreicht.
Negativ heisst: schlechter als ein fester Betriebspunkt.

Testpunkte: die 25 Befund-52-Gitterpunkte aus ``artifacts/runs/bb1_bb3_operating_point_
regret.json`` (:data:`~twinlink_ml.rl.env.ANCHOR_ARTIFACT_PATH`), DISJUNKT von den
kontinuierlichen Trainingsziehungen gehalten -- siehe :func:`assert_disjoint_from_test_grid`
fuer die Laufzeitpruefung und deren Docstring fuer die "Mass-Null"-Begruendung.

XA1 -- Korrektur der urspruenglichen VA3-Metrik
--------------------------------------------------
Die urspruengliche Fassung (``closed_fraction_median``/``p95``/``min``, weiterhin
verfuegbar, siehe unten) mischte zwei Fehlerarten in einer Zahl: der PER-PUNKT-Quotient
``(E_fest - E_Agent) / (E_fest - E_Orakel)`` wird auch fuer UNZULAESSIGE Aktionen gebildet
und geht mit denselben in den Median wie zulaessige ein -- an Punkten mit ``E_fest`` nahe
``E_Orakel`` (kleiner Nenner) kann ein einzelner Punkt den Median/das Minimum dominieren,
ohne dass das an der Zahl selbst sichtbar waere (siehe WA3: Minimum -78786 bei einem
einzigen Ausreisser).

Drei Zahlen NEBENEINANDER, NIE EINZELN berichtet (:func:`evaluate_policy` liefert sie
gemeinsam in :class:`ClosedRegretResult`, siehe dessen Docstring fuer die Begruendung,
warum (b) ohne (a) irrefuehrend ist):

a) ``feasibility_rate`` -- Zulaessigkeitsrate.
b) ``pooled_closed_regret_fraction`` -- GEPOOLT, NUR ueber zulaessige Aktionen, auf der
   ROHEN Energie ohne Strafterm:

   .. math:: \frac{\sum_i (E_{fest,i} - E_{Agent,i})}{\sum_i (E_{fest,i} - E_{Orakel,i})}
       \quad \text{Summe nur ueber zulaessige } i

   Eine SUMME im Zaehler und Nenner ist gegen einzelne kleine Nenner robust (ein Ausreisser
   kann die Summe nicht dominieren, wie er es bei einem Einzelquotienten koennte) --
   deshalb gepoolt, nicht gemittelte Einzelquotienten.
c) ``infeasible_violations`` -- je unzulaessiger (oder loeserfehlgeschlagener) Aktion,
   welche Nebenbedingung um wie viel verletzt ist (siehe :class:`InfeasibleViolation`).

WARUM (b) NIE OHNE (a): (b) zaehlt nur zulaessige Punkte -- eine Politik, die nur an den
zwei leichtesten Punkten ueberhaupt zulaessig ist und dort zufaellig nah am Orakel liegt,
zeigt ein perfektes (b), obwohl sie an 23 von 25 Punkten unbrauchbar ist. (a) macht diesen
Fall sichtbar; ohne (a) ist (b) daher fuer sich genommen kein Ergebnis, sondern nur ein
Teilausschnitt.

XA3 -- Trennung nach Lage des Orakel-Optimums
-------------------------------------------------
Ein Teil der Befund-52-Optima liegt an einer Aktionsschranke (im warmen Regime: die
Gasgeschwindigkeit an ihrer UNTEREN Schranke, 23 von 25 Punkten, siehe
``test_bandit_env_evaluation.py``). Eine tanh-gequetschte Politik (SAC/TD3-Standard)
erreicht eine Schranke nur asymptotisch -- ein geschlossener Anteil von 1.0 ist an diesen
Punkten fuer diese Politikklasse STRUKTURELL nicht erreichbar, unabhaengig von der
Trainingsqualitaet. (b) wird deshalb zusaetzlich getrennt fuer Punkte mit Orakel-Optimum im
Inneren der Aktionsbox und Punkte mit Orakel-Optimum an einer Schranke berichtet
(``pooled_closed_regret_fraction_interior``/``_boundary``), damit ein niedriger Gesamtwert
nicht vorschnell als Politikversagen gelesen wird, wenn er tatsaechlich aus den
Randpunkten kommt.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .env import ANCHOR_ARTIFACT_PATH, OperatingPointBanditEnv, _denormalise

PolicyFn = Callable[[np.ndarray], np.ndarray]
"""Beobachtung ``[ambient_temperature_celsius, ambient_relative_humidity]`` -> normierte
Aktion in ``[-1, 1]^2`` -- derselbe Vertrag wie ``OperatingPointBanditEnv.action_space``."""

_BOUNDARY_RELATIVE_TOLERANCE = 0.01
"""Toleranz fuer "Orakel-Aktion liegt an einer Aktionsschranke", als Anteil der jeweiligen
``ActionBound.span`` -- 1 % ist grosszuegig ueber dem beobachteten Suchrauschen (Befund 52s
Nelder-Mead-Verfeinerung, ``xatol=1e-4`` auf den Aktionen) und eng genug, um die klare
Luecke zwischen Rand- und Innenpunkten in den Befund-52-Daten nicht zu verwischen (naechste
Innenpunkte liegen > 7 % der Spanne vom Rand entfernt, siehe
``test_bandit_env_evaluation.py``)."""


@dataclass(frozen=True, slots=True)
class TestPoint:
    """Ein Befund-52-Gitterpunkt mit den Referenzenergien und der Orakel-Aktion fuer VA3."""

    ambient_temperature_celsius: float
    ambient_relative_humidity: float
    energy_oracle_kw: float
    energy_fixed_kw: float
    oracle_inlet_temperature_celsius: float
    oracle_gas_velocity_m_per_s: float


def load_test_points(artifact_path: Path = ANCHOR_ARTIFACT_PATH) -> tuple[TestPoint, ...]:
    """Die 25 Befund-52-Gitterpunkte als fester, VORAB festgelegter Testsatz.

    Nur Punkte, an denen SOWOHL das Orakel ALS AUCH der feste Betriebspunkt zulaessig
    geloest haben (in Befund 52 alle 25 -- siehe VA2-Tests), sonst gaebe es keine
    wohldefinierte ``E_fest - E_Orakel``-Differenz im Nenner der Metrik.
    """
    anchor = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    fixed_per_point = anchor["fixed_point"]["per_grid_point"]
    points: list[TestPoint] = []
    for grid_point, fixed_point_at_grid in zip(
        anchor["grid_results"], fixed_per_point, strict=True
    ):
        if grid_point.get("solve_failed") or not grid_point.get("feasible", False):
            continue
        if fixed_point_at_grid.get("solve_failed") or not fixed_point_at_grid.get(
            "feasible", False
        ):
            continue
        points.append(
            TestPoint(
                ambient_temperature_celsius=grid_point["ambient_temperature_celsius"],
                ambient_relative_humidity=grid_point["ambient_relative_humidity"],
                energy_oracle_kw=grid_point["energy_target"],
                energy_fixed_kw=fixed_point_at_grid["energy_target"],
                oracle_inlet_temperature_celsius=grid_point["inlet_temperature_celsius"],
                oracle_gas_velocity_m_per_s=grid_point["gas_velocity_m_per_s"],
            )
        )
    return tuple(points)


def is_boundary_optimum(point: TestPoint, env: OperatingPointBanditEnv) -> bool:
    """Ob die Orakel-Aktion dieses Testpunkts an einer Aktionsschranke liegt (XA3).

    Geprueft werden BEIDE Aktionsdimensionen (nicht nur Gasgeschwindigkeit, obwohl das in
    den aktuellen Platzhalterwerten die einzige betroffene ist) -- generisch, falls
    kuenftige Config-Werte auch ``inlet_temperature_celsius`` an den Rand treiben.
    """
    gas_tol = _BOUNDARY_RELATIVE_TOLERANCE * env._gas_bound.span
    inlet_tol = _BOUNDARY_RELATIVE_TOLERANCE * env._inlet_bound.span
    at_gas_bound = (
        abs(point.oracle_gas_velocity_m_per_s - env._gas_bound.minimum) < gas_tol
        or abs(point.oracle_gas_velocity_m_per_s - env._gas_bound.maximum) < gas_tol
    )
    at_inlet_bound = (
        abs(point.oracle_inlet_temperature_celsius - env._inlet_bound.minimum) < inlet_tol
        or abs(point.oracle_inlet_temperature_celsius - env._inlet_bound.maximum) < inlet_tol
    )
    return at_gas_bound or at_inlet_bound


def assert_disjoint_from_test_grid(
    training_disturbances: np.ndarray,
    test_points: tuple[TestPoint, ...] | None = None,
    *,
    atol: float = 1.0e-9,
) -> None:
    r"""Prueft, dass gezogene Trainingspunkte keinen der 25 Testpunkte exakt treffen.

    Bei stetiger Gleichverteilung (``OperatingPointBanditEnv.reset()`` ohne ``options``)
    hat ein exaktes Treffen eines der endlich vielen, diskreten Testpunkte
    Wahrscheinlichkeit null (Mass-Null-Ereignis auf einer kontinuierlichen Box) -- das ist
    eine analytische Tatsache, keine, die ein einzelner Lauf beweisen koennte. Diese
    Funktion macht daraus trotzdem eine LAUFZEITPRUEFUNG statt einer reinen Behauptung: sie
    prueft die tatsaechlich gezogenen Trainingspunkte eines konkreten Laufs gegen den
    Testsatz und schlaegt fehl, wenn einer davon (numerisch, innerhalb ``atol``)
    zusammenfaellt -- z. B. bei einem Seed-Handling-Fehler, der versehentlich den
    Testsatz statt einer Ziehung liefert.

    :param training_disturbances: Form ``(n, 2)``, Spalten ``[ambient_temperature_celsius,
        ambient_relative_humidity]`` -- die tatsaechlich waehrend eines Trainingslaufs
        gezogenen Stoergroessenpunkte.
    :raises ValueError: wenn ein Trainingspunkt einem Testpunkt numerisch entspricht.
    """
    points = test_points if test_points is not None else load_test_points()
    test_array = np.array(
        [[p.ambient_temperature_celsius, p.ambient_relative_humidity] for p in points]
    )
    training_array = np.asarray(training_disturbances, dtype=np.float64)
    for row in training_array:
        distances = np.linalg.norm(test_array - row[None, :], axis=1)
        if np.any(distances < atol):
            raise ValueError(
                f"Trainingspunkt {row.tolist()} liegt (numerisch, atol={atol}) auf einem "
                "der 25 Befund-52-Testpunkte -- Verletzung der in VA3 geforderten "
                "Disjunktheit von Trainings- und Testmenge."
            )


@dataclass(frozen=True, slots=True)
class InfeasibleViolation:
    """XA1(c): welche Nebenbedingung an einem unzulaessigen Testpunkt um wie viel verletzt ist.

    Alle ``*_violation``-Felder sind ``0.0``, wenn diese Nebenbedingung nicht bindet --
    ``solve_failed=True`` bedeutet, dass gar keine Loesung existiert (Felder dann alle
    ``0.0``, die Ursache steht in ``solve_failure_reason``).
    """

    ambient_temperature_celsius: float
    ambient_relative_humidity: float
    solve_failed: bool
    solve_failure_reason: str
    moisture_violation_kg_per_kg: float
    bed_temperature_violation_k: float
    gas_velocity_lower_violation_m_per_s: float
    gas_velocity_upper_violation_m_per_s: float
    solver_infeasible: bool
    """``True``, wenn ``SteadyStateSolution.feasible`` False ist (Saettigungsgrad > 1 oder
    Feuchte ausserhalb des isothermenplausiblen Bereichs) -- ohne quantifizierbare
    "um wie viel"-Groesse in ``check_constraints``, siehe ``infeasibility_reasons``."""
    infeasibility_reasons: tuple[str, ...]


def _group_metrics(
    entries: list[tuple[bool, float | None, float, float]],
) -> tuple[float, float]:
    """Zulaessigkeitsrate und gepoolter geschlossener Anteil (XA1 a/b) fuer eine Gruppe.

    ``entries``: je Punkt ``(feasible, energy_agent_kw_oder_None, energy_fixed_kw,
    energy_oracle_kw)``. Nicht geloeste Punkte (``energy_agent is None``) zaehlen als
    unzulaessig und tragen nichts zur Summe bei -- exakt wie ein regulaer unzulaessiger
    Punkt.
    """
    if not entries:
        return float("nan"), float("nan")
    feasible_flags = [feasible for feasible, *_ in entries]
    numerator = 0.0
    denominator = 0.0
    for feasible, energy_agent, energy_fixed, energy_oracle in entries:
        if not feasible or energy_agent is None:
            continue
        numerator += energy_fixed - energy_agent
        denominator += energy_fixed - energy_oracle
    pooled = numerator / denominator if abs(denominator) > 1.0e-12 else float("nan")
    return float(np.mean(feasible_flags)), pooled


@dataclass(frozen=True, slots=True)
class ClosedRegretResult:
    """Ergebnis von :func:`evaluate_policy` -- IMMER als Ganzes lesen, siehe Moduldocstring.

    ``pooled_closed_regret_fraction`` (XA1 b) ist ohne ``feasibility_rate`` (XA1 a)
    irrefuehrend -- beide gehoeren zusammen berichtet. ``closed_fraction_median``/``p95``/
    ``min`` sind die URSPRUENGLICHE VA3-Metrik, NENNEREMPFINDLICH (siehe Moduldocstring,
    Abschnitt "XA1"): ein einzelner Testpunkt mit kleinem ``E_fest - E_Orakel`` kann sie
    dominieren, ohne dass das an der Zahl selbst sichtbar waere. Weiterhin berichtet, aber
    NICHT die primaere Kennzahl.
    """

    n_points: int
    n_solve_failed: int
    feasibility_rate: float
    pooled_closed_regret_fraction: float
    n_interior_points: int
    feasibility_rate_interior: float
    pooled_closed_regret_fraction_interior: float
    n_boundary_points: int
    feasibility_rate_boundary: float
    pooled_closed_regret_fraction_boundary: float
    closed_fraction_median: float
    closed_fraction_p95: float
    closed_fraction_min: float
    mean_moisture_violation_of_infeasible_kg_per_kg: float | None
    infeasible_violations: tuple[InfeasibleViolation, ...]
    per_point: tuple[dict[str, Any], ...]


def evaluate_policy(
    env: OperatingPointBanditEnv,
    policy: PolicyFn,
    test_points: tuple[TestPoint, ...] | None = None,
) -> ClosedRegretResult:
    """VA3/XA1-Metrik fuer eine Policy, ausgewertet an den 25 Befund-52-Testpunkten.

    ``policy`` bekommt die Beobachtung direkt (kein ``env.reset()``/``env.step()``-
    Rundlauf, um den Normierungs-Rundtrip nicht zweimal zu durchlaufen, siehe
    ``env.py``s ``evaluate_operating_point``-Docstring fuer dieselbe Begruendung).
    """
    points = test_points if test_points is not None else load_test_points()
    closed_fractions: list[float] = []
    moisture_violations: list[float] = []
    infeasible_violations: list[InfeasibleViolation] = []
    per_point: list[dict[str, Any]] = []
    n_solve_failed = 0

    all_entries: list[tuple[bool, float | None, float, float]] = []
    interior_entries: list[tuple[bool, float | None, float, float]] = []
    boundary_entries: list[tuple[bool, float | None, float, float]] = []

    for point in points:
        boundary = is_boundary_optimum(point, env)
        observation = np.array(
            [point.ambient_temperature_celsius, point.ambient_relative_humidity],
            dtype=np.float32,
        )
        requested_normalised = np.asarray(policy(observation), dtype=np.float64)
        applied_normalised = np.clip(requested_normalised, -1.0, 1.0)
        t_in = _denormalise(
            float(applied_normalised[0]), env._inlet_bound.minimum, env._inlet_bound.maximum
        )
        u_gas = _denormalise(
            float(applied_normalised[1]), env._gas_bound.minimum, env._gas_bound.maximum
        )
        _, info = env.evaluate_operating_point(
            inlet_temperature_celsius=t_in,
            gas_velocity_m_per_s=u_gas,
            ambient_temperature_celsius=point.ambient_temperature_celsius,
            ambient_relative_humidity=point.ambient_relative_humidity,
        )
        feasible = bool(info["feasible"])

        denominator = point.energy_fixed_kw - point.energy_oracle_kw
        if info.get("solve_failed"):
            n_solve_failed += 1
            closed = float("nan")
            energy_agent: float | None = None
            infeasible_violations.append(
                InfeasibleViolation(
                    ambient_temperature_celsius=point.ambient_temperature_celsius,
                    ambient_relative_humidity=point.ambient_relative_humidity,
                    solve_failed=True,
                    solve_failure_reason=str(info.get("solve_failure_reason", "")),
                    moisture_violation_kg_per_kg=0.0,
                    bed_temperature_violation_k=0.0,
                    gas_velocity_lower_violation_m_per_s=0.0,
                    gas_velocity_upper_violation_m_per_s=0.0,
                    solver_infeasible=True,
                    infeasibility_reasons=(),
                )
            )
        else:
            energy_agent = float(info["energy_target"])
            closed = (
                (point.energy_fixed_kw - energy_agent) / denominator
                if abs(denominator) > 1.0e-12
                else float("nan")
            )
            if not feasible:
                check = info["constraint_check"]
                moisture_violations.append(float(check["moisture_violation_kg_per_kg"]))
                gas_lower = max(
                    0.0,
                    check["minimum_fluidization_velocity_m_per_s"] - check["gas_velocity_m_per_s"],
                )
                gas_upper = max(
                    0.0,
                    check["gas_velocity_m_per_s"] - check["terminal_velocity_m_per_s"],
                )
                infeasible_violations.append(
                    InfeasibleViolation(
                        ambient_temperature_celsius=point.ambient_temperature_celsius,
                        ambient_relative_humidity=point.ambient_relative_humidity,
                        solve_failed=False,
                        solve_failure_reason="",
                        moisture_violation_kg_per_kg=float(check["moisture_violation_kg_per_kg"]),
                        bed_temperature_violation_k=float(check["bed_temperature_violation_k"]),
                        gas_velocity_lower_violation_m_per_s=gas_lower,
                        gas_velocity_upper_violation_m_per_s=gas_upper,
                        solver_infeasible=not bool(check["solver_feasible"]),
                        infeasibility_reasons=tuple(info.get("infeasibility_reasons", ())),
                    )
                )
        closed_fractions.append(closed)

        entry = (feasible, energy_agent, point.energy_fixed_kw, point.energy_oracle_kw)
        all_entries.append(entry)
        (boundary_entries if boundary else interior_entries).append(entry)

        per_point.append(
            {
                "ambient_temperature_celsius": point.ambient_temperature_celsius,
                "ambient_relative_humidity": point.ambient_relative_humidity,
                "energy_oracle_kw": point.energy_oracle_kw,
                "energy_fixed_kw": point.energy_fixed_kw,
                "energy_agent_kw": energy_agent,
                "feasible": feasible,
                "solve_failed": bool(info.get("solve_failed")),
                "boundary_optimum": boundary,
                "closed_regret_fraction": closed,
            }
        )

    closed_array = np.asarray(closed_fractions, dtype=np.float64)
    finite = closed_array[np.isfinite(closed_array)]

    feasibility_rate, pooled = _group_metrics(all_entries)
    feasibility_rate_interior, pooled_interior = _group_metrics(interior_entries)
    feasibility_rate_boundary, pooled_boundary = _group_metrics(boundary_entries)

    return ClosedRegretResult(
        n_points=len(points),
        n_solve_failed=n_solve_failed,
        feasibility_rate=feasibility_rate,
        pooled_closed_regret_fraction=pooled,
        n_interior_points=len(interior_entries),
        feasibility_rate_interior=feasibility_rate_interior,
        pooled_closed_regret_fraction_interior=pooled_interior,
        n_boundary_points=len(boundary_entries),
        feasibility_rate_boundary=feasibility_rate_boundary,
        pooled_closed_regret_fraction_boundary=pooled_boundary,
        closed_fraction_median=float(np.median(finite)) if finite.size else float("nan"),
        closed_fraction_p95=float(np.percentile(finite, 95)) if finite.size else float("nan"),
        closed_fraction_min=float(np.min(finite)) if finite.size else float("nan"),
        mean_moisture_violation_of_infeasible_kg_per_kg=(
            float(np.mean(moisture_violations)) if moisture_violations else None
        ),
        infeasible_violations=tuple(infeasible_violations),
        per_point=tuple(per_point),
    )


def flat_metrics(result: ClosedRegretResult) -> dict[str, float]:
    """Skalare Sicht auf :class:`ClosedRegretResult`, fuer flache Ablagen (``RunRecord.metrics``).

    Gemeinsam verwendet von ``scripts/run_rl_bandit_learning_curve.py`` (WA3/XA3) und
    ``scripts/probe_bandit_env_random_baseline.py`` (YA1) -- eine Stelle statt zweimal
    dieselbe Aggregation.
    """
    moisture_violation = result.mean_moisture_violation_of_infeasible_kg_per_kg
    n_bed_temp = sum(1 for v in result.infeasible_violations if v.bed_temperature_violation_k > 0)
    n_moisture = sum(
        1 for v in result.infeasible_violations if v.moisture_violation_kg_per_kg > 0
    )
    n_gas = sum(
        1
        for v in result.infeasible_violations
        if v.gas_velocity_lower_violation_m_per_s > 0 or v.gas_velocity_upper_violation_m_per_s > 0
    )
    n_solve_failed_violations = sum(1 for v in result.infeasible_violations if v.solve_failed)
    return {
        "n_points": float(result.n_points),
        "n_solve_failed": float(result.n_solve_failed),
        # (a) und (b), IMMER zusammen -- siehe Moduldocstring, Abschnitt XA1.
        "feasibility_rate": result.feasibility_rate,
        "pooled_closed_regret_fraction": result.pooled_closed_regret_fraction,
        "n_interior_points": float(result.n_interior_points),
        "feasibility_rate_interior": result.feasibility_rate_interior,
        "pooled_closed_regret_fraction_interior": result.pooled_closed_regret_fraction_interior,
        "n_boundary_points": float(result.n_boundary_points),
        "feasibility_rate_boundary": result.feasibility_rate_boundary,
        "pooled_closed_regret_fraction_boundary": result.pooled_closed_regret_fraction_boundary,
        # Urspruengliche VA3-Metrik, nennerempfindlich (siehe Moduldocstring) -- weiterhin
        # berichtet, nicht mehr primaer.
        "closed_regret_fraction_median": result.closed_fraction_median,
        "closed_regret_fraction_p95": result.closed_fraction_p95,
        "closed_regret_fraction_min": result.closed_fraction_min,
        "mean_moisture_violation_of_infeasible_kg_per_kg": (
            moisture_violation if moisture_violation is not None else float("nan")
        ),
        "n_infeasible_violations": float(len(result.infeasible_violations)),
        "n_violations_bed_temperature": float(n_bed_temp),
        "n_violations_moisture": float(n_moisture),
        "n_violations_gas_velocity_window": float(n_gas),
        "n_violations_solve_failed": float(n_solve_failed_violations),
    }

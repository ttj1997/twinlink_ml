r"""VA3/XA1: evaluation metric -- fraction of the closed regret.

.. math:: \text{closed} = \frac{E_{fest} - E_{Agent}}{E_{fest} - E_{Orakel}}

0 means: no better than the fixed operating point. 1 means: the optimum reached.
Negative means: worse than a fixed operating point.

Test points: the 25 Befund 52 grid points from ``artifacts/runs/bb1_bb3_operating_point_
regret.json`` (:data:`~twinlink_ml.rl.env.ANCHOR_ARTIFACT_PATH`), kept DISJOINT from the
continuous training draws -- see :func:`assert_disjoint_from_test_grid`
for the run-time check and its docstring for the "measure zero" justification.

XA1 -- correction of the original VA3 metric
--------------------------------------------
The original version (``closed_fraction_median``/``p95``/``min``, still
available, see below) mixed two kinds of error in one number: the PER-POINT quotient
``(E_fest - E_Agent) / (E_fest - E_Orakel)`` is also formed for INFEASIBLE actions
and enters the median with the same weight as feasible ones -- at points with ``E_fest`` close to
``E_Orakel`` (small denominator) a single point can dominate the median/the minimum
without this being visible in the number itself (see WA3: minimum -78786 from a
single outlier).

Three numbers reported SIDE BY SIDE, NEVER SINGLY (:func:`evaluate_policy` returns them
together in :class:`ClosedRegretResult`, see its docstring for the justification
why (b) without (a) is misleading):

a) ``feasibility_rate`` -- feasibility rate.
b) ``pooled_closed_regret_fraction`` -- POOLED, ONLY over feasible actions, on the
   RAW energy without penalty term:

   .. math:: \frac{\sum_i (E_{fest,i} - E_{Agent,i})}{\sum_i (E_{fest,i} - E_{Orakel,i})}
       \quad \text{sum only over feasible } i

   A SUM in numerator and denominator is robust against individual small denominators (an outlier
   cannot dominate the sum, as it could with a single quotient) --
   hence pooled, not averaged single quotients.
c) ``infeasible_violations`` -- per infeasible (or solver-failed) action,
   which constraint is violated by how much (see :class:`InfeasibleViolation`).

WHY (b) NEVER WITHOUT (a): (b) counts only feasible points -- a policy that is feasible at all
only at the two easiest points and happens to lie close to the oracle there
shows a perfect (b), although it is useless at 23 of 25 points. (a) makes this
case visible; without (a), (b) on its own is therefore not a result, but only a
partial view.

XA3 -- separation by location of the oracle optimum
---------------------------------------------------
Some of the Befund 52 optima lie at an action bound (in the warm regime: the
gas velocity at its LOWER bound, 23 of 25 points, see
``test_bandit_env_evaluation.py``). A tanh-squashed policy (SAC/TD3 standard)
reaches a bound only asymptotically -- a closed fraction of 1.0 is STRUCTURALLY unreachable
at these points for this policy class, independently of the
training quality. (b) is therefore additionally reported separately for points with the oracle
optimum in the interior of the action box and points with the oracle optimum at a bound
(``pooled_closed_regret_fraction_interior``/``_boundary``), so that a low overall value
is not hastily read as policy failure when it actually comes from the
boundary points.
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
    """A Befund 52 grid point with the reference energies and the oracle action for VA3."""

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
    """Whether the oracle action of this test point lies at an action bound (XA3).

    BOTH action dimensions are checked (not only gas velocity, although in
    the current placeholder values that is the only one affected) -- generic, in case
    future config values also drive ``inlet_temperature_celsius`` to the edge.
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
    r"""Checks that drawn training points do not exactly hit any of the 25 test points.

    Under a continuous uniform distribution (``OperatingPointBanditEnv.reset()`` without
    ``options``) exactly hitting one of the finitely many, discrete test points has
    probability zero (a measure-zero event on a continuous box) -- that is
    an analytical fact, not one a single run could prove. This
    function nevertheless turns it into a RUN-TIME CHECK instead of a mere claim: it
    checks the actually drawn training points of a concrete run against the
    test set and fails if one of them (numerically, within ``atol``)
    coincides -- e.g. in the case of a seed-handling error that accidentally returns the
    test set instead of a draw.

    :param training_disturbances: shape ``(n, 2)``, columns ``[ambient_temperature_celsius,
        ambient_relative_humidity]`` -- the disturbance points actually drawn during a
        training run.
    :raises ValueError: if a training point numerically equals a test point.
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
    """XA1(c): which constraint is violated by how much at an infeasible test point.

    All ``*_violation`` fields are ``0.0`` if this constraint does not bind --
    ``solve_failed=True`` means that no solution exists at all (fields then all
    ``0.0``, the cause is in ``solve_failure_reason``).
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
    """Feasibility rate and pooled closed fraction (XA1 a/b) for a group.

    ``entries``: per point ``(feasible, energy_agent_kw_oder_None, energy_fixed_kw,
    energy_oracle_kw)``. Unsolved points (``energy_agent is None``) count as
    infeasible and contribute nothing to the sum -- exactly like a regularly infeasible
    point.
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
    """Result of :func:`evaluate_policy` -- ALWAYS read as a whole, see module docstring.

    ``pooled_closed_regret_fraction`` (XA1 b) is misleading without ``feasibility_rate``
    (XA1 a) -- both belong reported together. ``closed_fraction_median``/``p95``/
    ``min`` are the ORIGINAL VA3 metric, DENOMINATOR-SENSITIVE (see module docstring,
    section "XA1"): a single test point with a small ``E_fest - E_Orakel`` can
    dominate them without this being visible in the number itself. Still reported, but
    NOT the primary metric.
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
    """VA3/XA1 metric for a policy, evaluated at the 25 Befund 52 test points.

    ``policy`` receives the observation directly (no ``env.reset()``/``env.step()``
    round trip, so as not to go through the normalisation round trip twice, see the
    ``evaluate_operating_point`` docstring in ``env.py`` for the same justification).
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


def moisture_margin_at_oracle_binding_points(
    env: OperatingPointBanditEnv,
    policy: PolicyFn,
    test_points: tuple[TestPoint, ...] | None = None,
) -> tuple[float, int]:
    """Median distance of the product moisture to the TRUE quality limit (S0.3 gate G3).

    Evaluated only at the test points where the Befund 52 oracle binds the moisture limit,
    with the same definition as ``probe_bandit_env_ga1_safety_margin.py`` (limit - moisture).
    Solver failures are skipped. Returns ``(median, number of binding contexts used)``;
    the median is NaN when no context is usable.
    """
    points = test_points if test_points is not None else load_test_points()
    limit = env._base_config.material.quality_moisture_limit_kg_per_kg
    margins: list[float] = []
    for point in points:
        _, oracle_info = env.evaluate_operating_point(
            inlet_temperature_celsius=point.oracle_inlet_temperature_celsius,
            gas_velocity_m_per_s=point.oracle_gas_velocity_m_per_s,
            ambient_temperature_celsius=point.ambient_temperature_celsius,
            ambient_relative_humidity=point.ambient_relative_humidity,
        )
        if (
            oracle_info.get("solve_failed")
            or not oracle_info["constraint_check"]["binding_moisture"]
        ):
            continue
        observation = np.array(
            [point.ambient_temperature_celsius, point.ambient_relative_humidity],
            dtype=np.float32,
        )
        applied = np.clip(np.asarray(policy(observation), dtype=np.float64), -1.0, 1.0)
        _, info = env.evaluate_operating_point(
            inlet_temperature_celsius=_denormalise(
                float(applied[0]), env._inlet_bound.minimum, env._inlet_bound.maximum
            ),
            gas_velocity_m_per_s=_denormalise(
                float(applied[1]), env._gas_bound.minimum, env._gas_bound.maximum
            ),
            ambient_temperature_celsius=point.ambient_temperature_celsius,
            ambient_relative_humidity=point.ambient_relative_humidity,
        )
        if info.get("solve_failed"):
            continue
        margins.append(limit - float(info["constraint_check"]["moisture_kg_per_kg"]))
    return (float(np.median(margins)) if margins else float("nan")), len(margins)


def flat_metrics(result: ClosedRegretResult) -> dict[str, float]:
    """Scalar view of :class:`ClosedRegretResult`, for flat storage (``RunRecord.metrics``).

    Shared by ``scripts/run_rl_bandit_learning_curve.py`` (WA3/XA3) and
    ``scripts/probe_bandit_env_random_baseline.py`` (YA1) -- one place instead of the same
    aggregation twice.
    """
    moisture_violation = result.mean_moisture_violation_of_infeasible_kg_per_kg
    n_bed_temp = sum(1 for v in result.infeasible_violations if v.bed_temperature_violation_k > 0)
    n_moisture = sum(1 for v in result.infeasible_violations if v.moisture_violation_kg_per_kg > 0)
    n_gas = sum(
        1
        for v in result.infeasible_violations
        if v.gas_velocity_lower_violation_m_per_s > 0 or v.gas_velocity_upper_violation_m_per_s > 0
    )
    n_solve_failed_violations = sum(1 for v in result.infeasible_violations if v.solve_failed)
    return {
        "n_points": float(result.n_points),
        "n_solve_failed": float(result.n_solve_failed),
        # (a) and (b), ALWAYS together -- see module docstring, section XA1.
        "feasibility_rate": result.feasibility_rate,
        "pooled_closed_regret_fraction": result.pooled_closed_regret_fraction,
        "n_interior_points": float(result.n_interior_points),
        "feasibility_rate_interior": result.feasibility_rate_interior,
        "pooled_closed_regret_fraction_interior": result.pooled_closed_regret_fraction_interior,
        "n_boundary_points": float(result.n_boundary_points),
        "feasibility_rate_boundary": result.feasibility_rate_boundary,
        "pooled_closed_regret_fraction_boundary": result.pooled_closed_regret_fraction_boundary,
        # Original VA3 metric, denominator-sensitive (see module docstring) -- still
        # reported, no longer primary.
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

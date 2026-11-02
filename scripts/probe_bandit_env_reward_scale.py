r"""YA2: quantify the reward scale -- pure measurement, NO training, NO change.

Three quantities, all in kW, so that they are directly comparable:

a) Energy difference oracle versus fixed operating point per test point (``E_fest -
   E_Orakel``) -- the signal that an agent must resolve in order to close the regret.
b) Range of the energy over FEASIBLE random actions (see YA1, the same 20
   repetitions/seeds) -- how much feasible actions differ at all.
c) Penalty for the bed temperature violations ACTUALLY observed in XA3,
   ``_PENALTY_WEIGHT_KW * normalized_violation`` -- verbatim from
   ``study_operating_point_regret.py`` (see :data:`NORMALIZED_VIOLATION_FORMULA` below):

   .. code-block:: python

       normalized_violation = (
           moisture_violation / max(critical_moisture_kg_per_kg, 1e-6)
           + gas_lower_violation / window
           + gas_upper_violation / window
           + bed_temperature_violation / 10.0
           + (0.0 if solution.feasible else 1.0)
       )

   For a PURE bed temperature violation (moisture/fluidization window not binding,
   solver feasible -- according to XA3 ALL 25/25/15/11 violations at all four checkpoints)
   this reduces to ``normalized_violation = bed_temperature_violation_k / 10.0``,
   i.e. ``Strafe_kW = _PENALTY_WEIGHT_KW * bed_temperature_violation_k / 10.0``.

Reported: ratio (c)/(a). At more than one order of magnitude: the likely
cause is a gradient dominated by the penalty, in which the energy signal is drowned
out -- the agent learns (if at all) feasibility first, then energy, see YA1
(feasibility likewise below random level -- even that is not learned after 10^4
steps).

Options (the task: do NOT choose, only name them, with their effect on the comparability to
Befund 52 -- ``_PENALTY_WEIGHT_KW`` is currently deliberately taken over from Befund 52, because
the VA2 oracle check requires bit-identity; every change touches that):

1. Normalise the reward (energy relative to a reference value). Changes the reward SCALE,
   not the VA2 bit-identity itself (which checks ``reward == -energy_target`` at the
   oracle point, where ``normalized_violation`` is ~0 anyway) -- but every later
   learning curve would no longer be directly readable in kW against Befund 52, a new
   reference value would have to be documented and versioned.
2. Set the penalty weight for RL separately from the optimizer. Breaks the VA2 bit-identity
   EXPLICITLY (the oracle test compares the reward against ``-energy_target``, which holds
   exactly only for ``_PENALTY_WEIGHT_KW=50``, because the optimum of Befund 52 is ~feasible
   anyway -- a different weight does not change that directly, but every future calibration
   decouples itself from Befund 52 as the reference).
3. Lagrangian method (constraint as a quantity of its own instead of a penalty). Largest
   intervention: replaces the penalty formulation completely, and with it the VA1 decision
   ("penalty function for all three constraints"). The reward would no longer be directly
   comparable with the objective function of the Befund 52 optimizer, because the latter
   itself uses a penalty function.
4. Longer training, formulation unchanged. Changes NOTHING about the comparability
   (exactly the same reward, the same fixed point) -- but YA1 shows that 10^4 steps
   do not even reach random level; whether more steps overcome the scale problem
   or only train more slowly past the same distortion is unclear.

Usage::

    python scripts/probe_bandit_env_reward_scale.py \
        --output artifacts/runs/rl_bandit_reward_scale.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from twinlink_ml.rl import OperatingPointBanditEnv, evaluate_policy, load_test_points
from twinlink_ml.simulation import PlaceholderConfigWarning

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "rl_bandit_reward_scale.json"
XA3_ARTIFACT_PATH = REPO_ROOT / "artifacts" / "runs" / "rl_bandit_sac_learning_curve_xa3.jsonl"
RANDOM_BASELINE_N_REPETITIONS = 20
"""Dieselbe Anzahl/Seeds wie YA1 -- ``_make_random_policy`` ist deterministisch je Seed,
die Wiederholung hier reproduziert YA1s Ziehungen exakt, ohne das Artefakt erneut lesen zu
muessen."""

NORMALIZED_VIOLATION_FORMULA = """\
normalized_violation = (
    moisture_violation / max(critical_moisture_kg_per_kg, 1e-6)
    + gas_lower_violation / window
    + gas_upper_violation / window
    + bed_temperature_violation / 10.0
    + (0.0 if solution.feasible else 1.0)
)"""
"""Wortgleich aus ``scripts/study_operating_point_regret.py::check_constraints`` --
NICHT hier neu formuliert, siehe Moduldocstring."""


def _suppress_placeholder_warning() -> None:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)


def _make_random_policy(seed: int) -> Any:
    """Identical to ``scripts/probe_bandit_env_random_baseline.py::_make_random_policy``
    -- deliberately duplicated (one line) instead of imported, so that this script stays
    readable without a dependency on the YA1 script; both must be adapted together
    when one changes.
    """
    rng = np.random.default_rng(seed)

    def policy(observation: np.ndarray) -> np.ndarray:
        del observation
        return rng.uniform(-1.0, 1.0, size=2)

    return policy


def measure_oracle_fixed_gap_kw() -> dict[str, float]:
    """(a) ``E_fest - E_Orakel`` per test point, in kW."""
    points = load_test_points()
    gaps = np.array([p.energy_fixed_kw - p.energy_oracle_kw for p in points])
    return {
        "n_points": float(len(points)),
        "median_kw": float(np.median(gaps)),
        "p95_kw": float(np.percentile(gaps, 95)),
        "max_kw": float(np.max(gaps)),
        "min_kw": float(np.min(gaps)),
    }


def measure_feasible_random_energy_spread_kw(
    env: OperatingPointBanditEnv, n_repetitions: int
) -> dict[str, float]:
    """(b) Range of the raw energy over FEASIBLE random actions (YA1 draws)."""
    feasible_energies: list[float] = []
    for seed in range(n_repetitions):
        result = evaluate_policy(env, _make_random_policy(seed))
        for point in result.per_point:
            if point["feasible"] and point["energy_agent_kw"] is not None:
                feasible_energies.append(float(point["energy_agent_kw"]))
    energies = np.asarray(feasible_energies)
    return {
        "n_feasible_samples": float(energies.size),
        "n_total_samples": float(n_repetitions * len(load_test_points())),
        "median_kw": float(np.median(energies)) if energies.size else float("nan"),
        "min_kw": float(np.min(energies)) if energies.size else float("nan"),
        "max_kw": float(np.max(energies)) if energies.size else float("nan"),
        "span_kw": float(np.max(energies) - np.min(energies)) if energies.size else float("nan"),
    }


def measure_observed_penalty_kw(xa3_path: Path, penalty_weight_kw: float) -> dict[str, Any]:
    """(c) Penalty for the bed temperature violations ACTUALLY observed in XA3."""
    if not xa3_path.exists():
        return {"available": False, "reason": f"{xa3_path} nicht gefunden"}
    records = [json.loads(line) for line in xa3_path.read_text(encoding="utf-8").splitlines()]
    per_checkpoint: dict[str, dict[str, float]] = {}
    all_penalties: list[float] = []
    for record in records:
        checkpoint = record["tags"]["checkpoint"]
        violations = record["params"].get("infeasible_violations", [])
        penalties = [
            penalty_weight_kw * (v["bed_temperature_violation_k"] / 10.0)
            for v in violations
            if v["bed_temperature_violation_k"] > 0
        ]
        all_penalties.extend(penalties)
        if penalties:
            arr = np.asarray(penalties)
            per_checkpoint[checkpoint] = {
                "n_violations": float(len(penalties)),
                "median_penalty_kw": float(np.median(arr)),
                "min_penalty_kw": float(np.min(arr)),
                "max_penalty_kw": float(np.max(arr)),
            }
        else:
            per_checkpoint[checkpoint] = {
                "n_violations": 0.0,
                "median_penalty_kw": float("nan"),
                "min_penalty_kw": float("nan"),
                "max_penalty_kw": float("nan"),
            }
    overall = np.asarray(all_penalties)
    return {
        "available": True,
        "penalty_weight_kw": penalty_weight_kw,
        "per_checkpoint": per_checkpoint,
        "overall_n_violations": float(overall.size),
        "overall_median_penalty_kw": float(np.median(overall)) if overall.size else float("nan"),
        "overall_min_penalty_kw": float(np.min(overall)) if overall.size else float("nan"),
        "overall_max_penalty_kw": float(np.max(overall)) if overall.size else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--xa3-artifact", type=Path, default=XA3_ARTIFACT_PATH)
    parser.add_argument("--n-repetitions", type=int, default=RANDOM_BASELINE_N_REPETITIONS)
    arguments = parser.parse_args()
    _suppress_placeholder_warning()

    env = OperatingPointBanditEnv()

    print("=" * 78)
    print("Normierung (woertlich aus study_operating_point_regret.py::check_constraints):")
    print(NORMALIZED_VIOLATION_FORMULA)
    print("=" * 78)

    gap = measure_oracle_fixed_gap_kw()
    print(
        f"\n(a) E_fest - E_Orakel je Testpunkt [kW]: median={gap['median_kw']:.4f}  "
        f"p95={gap['p95_kw']:.4f}  max={gap['max_kw']:.4f}  min={gap['min_kw']:.4f}"
    )

    print(
        f"\n(b) Spannweite Energie ueber zulaessige Zufallsaktionen [kW] "
        f"({arguments.n_repetitions} Wiederholungen, dieselben Seeds wie YA1) ..."
    )
    spread = measure_feasible_random_energy_spread_kw(env, arguments.n_repetitions)
    print(
        f"    n_feasible={int(spread['n_feasible_samples'])}/{int(spread['n_total_samples'])}  "
        f"median={spread['median_kw']:.4f}  span=[{spread['min_kw']:.4f}, "
        f"{spread['max_kw']:.4f}] ({spread['span_kw']:.4f} kW)"
    )

    penalty = measure_observed_penalty_kw(arguments.xa3_artifact, env._penalty_weight_kw)
    print(
        f"\n(c) Strafe fuer XA3s beobachtete Bettemperaturverletzungen [kW] "
        f"(_PENALTY_WEIGHT_KW={env._penalty_weight_kw}):"
    )
    if penalty["available"]:
        for checkpoint, stats in penalty["per_checkpoint"].items():
            print(
                f"    Schritt {checkpoint:>6s}: n={int(stats['n_violations']):2d}  "
                f"median={stats['median_penalty_kw']:.2f}  "
                f"[{stats['min_penalty_kw']:.2f}, {stats['max_penalty_kw']:.2f}]"
            )
        print(
            f"    GESAMT ({int(penalty['overall_n_violations'])} Verletzungen): "
            f"median={penalty['overall_median_penalty_kw']:.2f}  "
            f"[{penalty['overall_min_penalty_kw']:.2f}, {penalty['overall_max_penalty_kw']:.2f}]"
        )
    else:
        print(f"    nicht verfuegbar: {penalty['reason']}")

    print("\n--- Verhaeltnis (c)/(a) " + "-" * 20)
    if penalty["available"] and np.isfinite(penalty["overall_median_penalty_kw"]):
        ratio = penalty["overall_median_penalty_kw"] / gap["median_kw"]
        print(f"median(c) / median(a) = {ratio:.1f}")
        if ratio > 10.0:
            print(
                "-> ueber einer Groessenordnung: wahrscheinliche Ursache ist ein von der "
                "Strafe dominierter Gradient, der das Energiesignal darin untergehen laesst "
                "-- der Agent lernt (wenn ueberhaupt) zuerst Zulaessigkeit, dann Energie."
            )
        else:
            print("-> unter einer Groessenordnung, keine dominante Verzerrung durch die Strafe.")
    else:
        ratio = float("nan")
        print("nicht berechenbar (keine XA3-Verletzungen verfuegbar).")

    output = {
        "oracle_fixed_gap_kw": gap,
        "feasible_random_energy_spread_kw": spread,
        "observed_bed_temperature_penalty_kw": penalty,
        "ratio_penalty_to_signal_median": ratio,
        "normalized_violation_formula": NORMALIZED_VIOLATION_FORMULA,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nErgebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

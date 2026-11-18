r"""Fix the strength of the structural error of docs/42_hybrid_structural_2.md.

Run BEFORE any data, training or margin run of the study; the result is written into
``conf/hybrid_structural_2/hybrid_structural_2.yaml`` (``structure.fraction``) by hand and checked
by a test against this artifact.

Structural term: exhaust recirculation of the truth (``hybrid_structural_2.study``), which makes
the bed wetter than the prior model predicts. The request fixes the size by what M0 does: the
prior model has to UNDERestimate the moisture at its own operating point by 0.03-0.07 kg/kg in the
median (mirror image of Befund 124, where M0 overestimated it by 0.070).

Rule (committed in the config before this script ran)
-----------------------------------------------------
* M0 = the nominal steady grey-box model without the term. At each of the 100 interior cases of
  the study (5 per draw, 20 draws) M0 is optimised with the optimiser of the study (grid + zoom,
  settings of the config) at margin 0, limit 0.15 kg/kg. M0 does not depend on the truth, so its
  operating points are computed once.
* Bias of a case = M0's predicted moisture at that point minus the moisture of the truth of the
  case's draw (draw parameters + recirculation at the candidate fraction) at the same point.
* Among ``structure.fraction_candidates`` the fraction whose MEDIAN bias over the cases at which
  the truth has a steady state is closest to ``structure.target_bias_kg_per_kg`` (-0.05, the
  middle of the band); ties: the smaller fraction. The script fails if the selected median lies
  outside ``structure.accepted_bias_kg_per_kg``.

Amendment of the rule (cb10bfd -> this commit, AFTER a first run that wrote no artifact): the
rule as first committed admitted only candidates at which the truth has a steady state at every
one of the 100 points. No candidate meets that, and neither does the truth without the term (4 of
the 100 points: at the lower end of the gas velocity the water balance of some draws has no
steady state). The first run therefore stopped without a selection, after printing the medians of
the candidates. The condition is dropped; the statistic is the median over the cases with a
steady truth, which is how the +0.070 of Befund 124 was computed (median of the defined biases).
A point without a steady truth lies on the wet side (the bed cannot evaporate the sprayed water);
the artifact also reports the median with these points counted as the wettest
(``median_missing_as_wettest``), without any rule.

Only the moisture of M0 against the truth is computed and written: no energy of the truth, no
oracle, no margin. Reported next to the choice, without any rule: per candidate the range and the
quartiles of the bias, the number of points without a steady truth, the bias without the term
(parameter draws alone), and at the nominal point (nominal commands, disturbances and material
parameters) the steady moisture, bed temperature and inlet humidity with and without the term.
The bed-to-exhaust gap of this truth is zero by construction: the purge leaves the ideally mixed
bed at the bed temperature (``greybox._rhs``), the recirculation changes the inlet state only.

    tools/run_here.sh scripts/probe_hybrid_structural_2_strength.py --name hybrid_structural_2_strength_v1
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

ROOT = SCRIPTS.parent
BED_EXHAUST_GAP_K = 0.0  # by construction (module docstring)


def grey_box_map(setup: Any, config: Any) -> Any:
    from twinlink_ml.hybrid_model.maps import GreyBoxMap

    return GreyBoxMap(
        config,
        setup.wide(config),
        nozzle_pressure_bar=setup.nozzle_pressure_bar,
        quality_limit_kg_per_kg=setup.config.quality_limit_kg_per_kg,
    )


def interior_cases(setup: Any, draw: int) -> list[Any]:
    return [c for c in setup.draw_cases(draw) if c.kind == "interior"]


def m0_points(setup: Any, max_batch: int = 256) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per draw: M0's operating points ``(k, 2)`` and its predicted moisture ``(k,)``, margin 0."""
    from twinlink_ml.hybrid_model.grid_rto import CaseEvaluator, grid_zoom
    from twinlink_ml.hybrid_model.study import case_arrays

    limit = setup.config.quality_limit_kg_per_kg
    nominal = grey_box_map(setup, setup.steady_nominal)
    by_context: dict[tuple[float, float], tuple[np.ndarray, float]] = {}
    out = {}
    for d in range(len(setup.draws)):
        cases = interior_cases(setup, d)
        contexts, wide = case_arrays(cases)
        new = [i for i, c in enumerate(map(tuple, contexts)) if c not in by_context]
        if new:  # the prior model is the same for every draw: one optimisation per context
            evaluator = CaseEvaluator(nominal, contexts[new], wide[new], max_batch=max_batch)
            idx = np.arange(len(new))
            actions, _ = grid_zoom(
                evaluator,
                idx,
                np.full(len(new), limit),
                np.zeros(len(new)),
                setup.bounds,
                setup.config.grid,
            )
            predicted = evaluator.values(idx, actions)
            if not predicted["valid"].all():
                raise RuntimeError("M0 has no valid steady state at its own operating point")
            for j, i in enumerate(new):
                by_context[tuple(contexts[i])] = (
                    actions[j],
                    float(predicted["moisture_kg_per_kg"][j]),
                )
        out[d] = (
            np.array([by_context[tuple(c)][0] for c in contexts]),
            np.array([by_context[tuple(c)][1] for c in contexts]),
        )
    return out


def truth_moisture(setup: Any, config: Any, draw: int, actions: np.ndarray) -> np.ndarray:
    """Steady moisture of the truth ``config`` at the points (NaN without a steady state)."""
    from twinlink_ml.hybrid_model.study import case_arrays

    contexts, wide = case_arrays(interior_cases(setup, draw))
    prediction = grey_box_map(setup, config).predict(actions, contexts, wide=bool(wide.any()))
    moisture: np.ndarray = np.where(prediction.valid, prediction.moisture_kg_per_kg, np.nan)
    return moisture


def summary(bias: np.ndarray) -> dict[str, Any]:
    finite = bias[np.isfinite(bias)]
    q = np.quantile(finite, [0.0, 0.25, 0.5, 0.75, 1.0]) if finite.size else [np.nan] * 5
    wettest = np.where(np.isfinite(bias), bias, -np.inf)  # no steady truth: wetter than any
    return {
        "n_cases": int(bias.size),
        "n_without_steady_truth": int(bias.size - finite.size),
        "median": float(q[2]),
        "median_missing_as_wettest": float(np.median(wettest)) if bias.size else float("nan"),
        "min": float(q[0]),
        "q25": float(q[1]),
        "q75": float(q[3]),
        "max": float(q[4]),
    }


def nominal_point(config: Any) -> dict[str, Any]:
    """Steady state at the nominal point (moisture, bed temperature, inlet humidity; no energy)."""
    from twinlink_ml.simulation import GreyBoxFluidBedGranulator
    from twinlink_ml.simulation.steady_state import solve_steady_state

    solution = solve_steady_state(GreyBoxFluidBedGranulator(config))
    return {
        "feasible": bool(solution.feasible),
        "moisture_kg_per_kg": float(solution.particle_moisture_kg_per_kg),
        "bed_temperature_celsius": float(solution.bed_temperature_celsius),
        "inlet_humidity_kg_per_kg": float(solution.diagnostics["inlet_humidity_kg_per_kg"]),
    }


def select(rows: list[dict[str, Any]], target: float) -> dict[str, Any]:
    """The rule of the module docstring on the candidate rows."""
    eligible = [r for r in rows if np.isfinite(r["bias"]["median"])]
    if not eligible:
        raise SystemExit("no candidate with a steady truth at any point")
    return min(eligible, key=lambda r: (abs(r["bias"]["median"] - target), r["fraction"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    from _common import provenance, require_clean_tree, write_artifact

    from twinlink_ml.hybrid_model.study import build_setup
    from twinlink_ml.hybrid_structural_2.study import load_recirculation_config, with_recirculation
    from twinlink_ml.utils.code_location import require_code_location

    require_code_location(ROOT)
    require_clean_tree()
    warnings.filterwarnings("ignore")
    started = time.time()
    config = load_recirculation_config()
    structure = config.raw["structure"]
    setup = build_setup(config)  # the set-up of Befund 122: configs, the 20 draws, the cases
    candidates = [float(v) for v in structure["fraction_candidates"]]
    target = float(structure["target_bias_kg_per_kg"])
    low, high = (float(v) for v in structure["accepted_bias_kg_per_kg"])
    draws = range(len(setup.draws))

    points = m0_points(setup)
    print(f"M0 optimised at {sum(len(a) for a, _ in points.values())} cases", flush=True)

    def bias_of(truth_of: Any) -> tuple[np.ndarray, list[dict[str, Any]]]:
        values, per_draw = [], []
        for d in draws:
            actions, predicted = points[d]
            b = predicted - truth_moisture(setup, truth_of(d), d, actions)
            values.append(b)
            per_draw.append(  # None: the truth has no steady state at that point
                {"draw": d, "bias_kg_per_kg": [float(v) if np.isfinite(v) else None for v in b]}
            )
        return np.concatenate(values), per_draw

    without, without_per_draw = bias_of(setup.steady_truth)
    reference = nominal_point(setup.steady_nominal)
    rows = []
    for fraction in candidates:
        bias, per_draw = bias_of(lambda d, f=fraction: with_recirculation(setup.steady_truth(d), f))
        at_nominal = nominal_point(with_recirculation(setup.steady_nominal, fraction))
        rows.append(
            {
                "fraction": fraction,
                "bias": summary(bias),
                "nominal_point": {
                    **at_nominal,
                    "moisture_shift_kg_per_kg": at_nominal["moisture_kg_per_kg"]
                    - reference["moisture_kg_per_kg"],
                    "bed_temperature_shift_k": at_nominal["bed_temperature_celsius"]
                    - reference["bed_temperature_celsius"],
                    "bed_exhaust_gap_k": BED_EXHAUST_GAP_K,
                },
                "per_draw": per_draw,
            }
        )
        b = rows[-1]["bias"]
        print(
            f"fraction {fraction:.2f}: M0 minus truth median {b['median']:+.4f} "
            f"(min {b['min']:+.4f}, max {b['max']:+.4f}), without steady truth "
            f"{b['n_without_steady_truth']}; nominal point X "
            f"{at_nominal['moisture_kg_per_kg']:.4f}, T_bed "
            f"{at_nominal['bed_temperature_celsius']:.2f} degC",
            flush=True,
        )
    selected = select(rows, target)
    within = low <= selected["bias"]["median"] <= high
    payload = {
        "_provenance": provenance("not used"),
        "study": "hybrid against GRU against refit, structural error to the unsafe side",
        "rule": {
            "candidates": candidates,
            "target_bias_kg_per_kg": target,
            "accepted_bias_kg_per_kg": [low, high],
            "quantity": "median over the 100 interior cases of M0's predicted moisture minus the "
            "truth, at M0's operating point (margin 0, limit of the config)",
        },
        "selected_fraction": selected["fraction"],
        "selected_median_bias_kg_per_kg": selected["bias"]["median"],
        "selected_within_accepted_band": bool(within),
        "bed_exhaust_gap_k": BED_EXHAUST_GAP_K,
        "bed_exhaust_gap_note": "zero by construction: the purge leaves the bed at T_bed",
        "m0_points": [
            {
                "draw": d,
                "actions": points[d][0].tolist(),
                "predicted_moisture_kg_per_kg": points[d][1].tolist(),
            }
            for d in draws
        ],
        "without_term": {"bias": summary(without), "per_draw": without_per_draw},
        "nominal_point_without_term": reference,
        "candidates": rows,
        "runtime_s": time.time() - started,
    }
    write_artifact(args.name, payload)
    print(
        f"selected fraction {selected['fraction']:.2f}: median bias "
        f"{selected['bias']['median']:+.4f} kg/kg (within {low}..{high}: {within})"
    )
    return 0 if within else 1


if __name__ == "__main__":
    raise SystemExit(main())

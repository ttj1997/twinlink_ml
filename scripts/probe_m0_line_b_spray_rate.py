r"""M0: choice of the spray rate of the synthetic line B (docs/09, decision D4).

Line B is a defined variation of line A: same geometry, same critical moisture, and a HIGHER
spray rate (other product, higher water load). Its purpose in the model-based optimisation is a
line whose exhaust sits at a clearly different temperature level than line A's, so that the heat
allocation between the two lines is a real decision. The value is labelled synthetic; it is not
taken from the retracted parametrisation of Befund 56 and not from plant data. The qualitative
reason is the project lead's: in the plant data two products had very different operating points.

What is run: the per-instance oracle of Befund 52 (DE + Nelder-Mead, popsize 10, maxiter 50,
tolerance 1e-4, seed 0, three variables: inlet temperature, gas velocity, nozzle pressure) on the
3 x 3 ambient grid (5/20/35 degC x 0.25/0.55/0.85; a subset of the Befund 52 grid), for the spray
rates 1.5 (line A, the config nominal), 2.0, 2.5 and 3.0 kg/h (3.0 is the upper action bound),
against the TIGHTENED quality limit ``0.15 - m`` with ``m = 0.005961819615175404`` kg/kg (main
case D3; the kinetics keep the critical moisture 0.20, decoupled mechanics of Befund 64,
``probe_margin_energy_cost_decoupled`` imported unchanged). Only the spray rate differs between
the four arms; the spray rate is not an optimisation variable (as in Befund 52).

Decision rule, fixed here before the run. A candidate spray rate s in {2.0, 2.5, 3.0} qualifies
as line B iff
  (F) the optimum is solver-feasible and constraint-feasible at all 9 scenarios, and
  (L) the median over the 9 scenarios of ``|T_out(s) - T_out(1.5)|`` at the optima is at least
      10 K, with ``T_out = bed temperature - outlet_duct_heat_loss_k`` (the exhaust channel of
      the network interface; the duct loss is 0.0 in the config).
The line-B value is the SMALLEST qualifying s (the smallest departure from line A that makes the
allocation a real decision). If none qualifies, nothing is chosen and the result is reported to
the project lead. The 10 K threshold is a setting by the author, to be confirmed; it was fixed
before the run. The distance of each optimum to the validity boundary 0.20 kg/kg is reported
(``0.20 - moisture`` in kg/kg and in units of m), as are the set points and the convergence flags.

Runtime estimate (before the run, 16 logical CPUs): 9 oracle points per spray rate fill one wave
of the pool (520-580 s per point on this machine, Befund 65 FA3 text), so about 9-10 min per rate,
about 40 min for the four rates run one after the other.

    PYTHONPATH=<worktree>/src python scripts/probe_m0_line_b_spray_rate.py
"""

from __future__ import annotations

import dataclasses
import sys
import warnings
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import probe_margin_energy_cost as ka1  # noqa: E402
import probe_margin_energy_cost_decoupled as dec  # noqa: E402
from _common import REPO_ROOT, provenance, require_clean_tree, write_artifact  # noqa: E402

import twinlink_ml  # noqa: E402
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config  # noqa: E402

ARTIFACT_NAME = "m0_line_b_spray_rate"
QUALITY_LIMIT_KG_PER_KG = 0.15
MARGIN_KG_PER_KG = 0.005961819615175404
TIGHTENED_LIMIT_KG_PER_KG = QUALITY_LIMIT_KG_PER_KG - MARGIN_KG_PER_KG
KINETICS_BOUNDARY_KG_PER_KG = 0.20
LINE_A_SPRAY_KG_PER_H = 1.5
CANDIDATE_SPRAY_KG_PER_H = (2.0, 2.5, 3.0)
LEVEL_THRESHOLD_K = 10.0


def _with_spray_rate(base: Any, spray_kg_per_h: float) -> Any:
    actions = dataclasses.replace(
        base.actions,
        spray_rate_kg_per_h=dataclasses.replace(
            base.actions.spray_rate_kg_per_h, nominal=spray_kg_per_h
        ),
    )
    return dataclasses.replace(base, actions=actions)


def _outlet_temperature(result: dict[str, Any], duct_loss_k: float) -> float | None:
    if result.get("solve_failed") or "bed_temperature_celsius" not in result:
        return None
    return float(result["bed_temperature_celsius"]) - duct_loss_k


def main() -> int:
    require_clean_tree()
    assert Path(twinlink_ml.__file__).resolve().is_relative_to(REPO_ROOT), (
        "twinlink_ml is imported from outside this checkout; set PYTHONPATH=<checkout>/src"
    )
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    base_config = load_process_config(
        REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
    )
    assert base_config.material.critical_moisture_kg_per_kg == KINETICS_BOUNDARY_KG_PER_KG
    assert base_config.actions.spray_rate_kg_per_h.nominal == LINE_A_SPRAY_KG_PER_H
    duct_loss_k = base_config.exhaust.outlet_duct_heat_loss_k

    regret = ka1._load_regret_module()
    workers = cpu_count()
    budget = regret.OptimizationBudget(
        seed=0,
        popsize=regret.DEFAULT_POPSIZE,
        maxiter=regret.DEFAULT_MAXITER,
        de_tol=regret.DEFAULT_DE_TOL,
        search_max_iterations=regret.DEFAULT_SEARCH_MAX_ITERATIONS,
        search_tolerance=regret.DEFAULT_SEARCH_TOLERANCE,
        workers=workers,
    )

    arms: dict[str, Any] = {}
    for spray in (LINE_A_SPRAY_KG_PER_H, *CANDIDATE_SPRAY_KG_PER_H):
        config = _with_spray_rate(base_config, spray)
        grid = dec.run_oracle_grid_decoupled(
            config,
            regret,
            grid_size=3,
            workers=workers,
            budget=budget,
            effective_critical_moisture_kg_per_kg=TIGHTENED_LIMIT_KG_PER_KG,
            label=f"line_spray_{spray}",
        )
        arms[str(spray)] = {
            "spray_rate_kg_per_h": spray,
            "config_fingerprint": config.fingerprint,
            "grid_results": grid,
        }

    reference = arms[str(LINE_A_SPRAY_KG_PER_H)]["grid_results"]
    verdicts: dict[str, Any] = {}
    for spray in CANDIDATE_SPRAY_KG_PER_H:
        results = arms[str(spray)]["grid_results"]
        feasible_all = len(results) == 9 and all(
            (not r.get("solve_failed")) and r.get("feasible", False) for r in results
        )
        differences = []
        for ref, cand in zip(reference, results, strict=True):
            t_ref = _outlet_temperature(ref, duct_loss_k)
            t_cand = _outlet_temperature(cand, duct_loss_k)
            if t_ref is not None and t_cand is not None:
                differences.append(abs(t_cand - t_ref))
        median_difference = float(np.median(differences)) if differences else None
        level_ok = median_difference is not None and median_difference >= LEVEL_THRESHOLD_K
        verdicts[str(spray)] = {
            "feasible_at_all_9": feasible_all,
            "n_compared": len(differences),
            "median_abs_outlet_temperature_difference_k": median_difference,
            "level_criterion_met": level_ok,
            "qualifies": bool(feasible_all and level_ok),
        }
    qualifying = [s for s in CANDIDATE_SPRAY_KG_PER_H if verdicts[str(s)]["qualifies"]]
    chosen = min(qualifying) if qualifying else None

    write_artifact(
        ARTIFACT_NAME,
        {
            "_provenance": provenance(),
            "quality_limit_kg_per_kg": QUALITY_LIMIT_KG_PER_KG,
            "margin_kg_per_kg": MARGIN_KG_PER_KG,
            "tightened_limit_kg_per_kg": TIGHTENED_LIMIT_KG_PER_KG,
            "kinetics_boundary_kg_per_kg": KINETICS_BOUNDARY_KG_PER_KG,
            "level_threshold_k": LEVEL_THRESHOLD_K,
            "outlet_duct_heat_loss_k": duct_loss_k,
            "budget": dataclasses.asdict(budget),
            "arms": arms,
            "verdicts": verdicts,
            "chosen_spray_rate_kg_per_h": chosen,
        },
    )
    print("verdicts:", verdicts)
    print("chosen spray rate:", chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

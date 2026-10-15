r"""References of the margin study T1b (Befund 95): oracle and fixed point at margin k * m.

Pre-registration of the study (margins, seeds, metrics, gate): docstring of
``scripts/analyze_bres_margin.py``. This script only computes the optimiser references that the
analysis needs; it runs no RL.

PLACEHOLDERS. The quality limit 0.15 kg/kg (main case) and the lower gas-velocity bound 0.80 m/s
(the config default, unchanged) are placeholders without source; the margin m
= 0.005961819615175404 kg/kg is an earlier setting (Befund 65 FA2), not a result.

References (existing optimiser mechanics of ``run_m1_margin_cost.py`` and
``twinlink_ml.evaluation.carrier_oracle``, no line changed there):

  oracle_k{j}   per-instance oracle against the limit ``0.15 - j * m`` on the 5 x 5 ambient grid of
                Befund 52, for j in {1, 2, 3, 5}; j = 1 is the common reference "oracle with m" of
                the study, j in {2, 3, 5} the oracle with the margin of the RL cell;
  fixed_k{j}    fixed operating point against the limit ``0.15 - j * m`` (one set point, feasible at
                all 25 contexts, minimal mean objective; BB3(a) of Befund 52), for j in {2, 3, 5}.

Objective: the legacy objective of the RL environment (``energy_target``, kW, penalty 50 kW per unit
of normalised violation), so that the references are the same quantity as the RL reward energy and as
the test-set references of Befund 93 (``s05_limit015_oracle_and_fixed``); the gas-heater CO2 booking
(eta_gas 0.80, factors of ``net_energy.py``) of every solution is stored next to it. Optimiser budget
of Befund 52 (asserted by ``run_m1_margin_cost._budget``), three variables (inlet temperature, gas
velocity, nozzle pressure), seed 0. A reference that is infeasible at a context is reported, not
removed. Anchor of this machinery: ``m1_anchor_s05_k1`` (legacy objective; grid points 0 and 12 bit
identical to the Befund 52 optimum, point 24 +2.4e-5 kW, Befund 80).

Provenance is taken at the START of the script (clean tree required then); the RL runs of the same
study write their records while this script runs, which is why they are exempt from the dirty check
(``ignore_paths``) and listed in the artifact.

    tools/run_here.sh scripts/run_bres_margin_references.py --workers 8
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from multiprocessing import cpu_count
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import run_m1_margin_cost as m1  # noqa: E402
import run_rl_bandit_residual as rr  # noqa: E402
from _common import (  # noqa: E402
    NO_HEAT_NETWORK,
    REPO_ROOT,
    RUNS_DIR,
    provenance,
    require_clean_tree,
    write_artifact,
)

import twinlink_ml  # noqa: E402
from twinlink_ml.evaluation.carrier_oracle import (  # noqa: E402
    OracleSpec,
    find_fixed_point,
    run_oracle_grid,
)
from twinlink_ml.simulation.oracle_search import action_bounds, build_grid  # noqa: E402

LIMIT_KG_PER_KG = rr.LIMIT_KG_PER_KG  # PLACEHOLDER
MARGIN_M = rr.MARGIN_M
ORACLE_FACTORS = (1, 2, 3, 5)
FIXED_FACTORS = (2, 3, 5)
OBJECTIVE = "legacy_power_kw"


def artifact_name(kind: str, factor: int) -> str:
    return f"bres_margin_ref_{kind}_k{factor}"


def own_outputs() -> list[Path]:
    names = [artifact_name("oracle", j) for j in ORACLE_FACTORS]
    names += [artifact_name("fixed", j) for j in FIXED_FACTORS]
    return [RUNS_DIR / f"{name}.json" for name in names]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() // 2))
    arguments = parser.parse_args()
    assert Path(twinlink_ml.__file__).resolve().is_relative_to(REPO_ROOT), (
        "twinlink_ml is imported from outside this checkout; use tools/run_here.sh"
    )
    exempt = [*own_outputs(), *rr.sweep_output_paths(rr.MARGIN_STUDY_PREFIX)]
    require_clean_tree(exempt)
    prov = provenance(NO_HEAT_NETWORK, ignore_paths=exempt)

    config = m1._load_config()
    budget = m1._budget(arguments.workers)
    grid = build_grid(config, 5)
    bounds = action_bounds(config)
    common = {
        "_provenance": prov,
        "quality_limit_kg_per_kg": LIMIT_KG_PER_KG,
        "margin_m_kg_per_kg": MARGIN_M,
        "objective": OBJECTIVE,
        "gas_velocity_lower_bound_m_per_s": bounds[1][0],
        "placeholders": "quality limit 0.15 kg/kg; gas velocity lower bound 0.80 m/s; "
        "margin m of Befund 65 FA2",
        "kinetics_boundary_kg_per_kg": m1.KINETICS_BOUNDARY_KG_PER_KG,
        "gas_heater_efficiency": m1.GAS_HEATER_EFFICIENCY,
        "budget": dataclasses.asdict(budget),
        "config_fingerprint": config.fingerprint,
    }
    for j in ORACLE_FACTORS:
        path = RUNS_DIR / f"{artifact_name('oracle', j)}.json"
        if path.exists():
            print(f"skip {path.name} (exists)")
            continue
        limit = LIMIT_KG_PER_KG - j * MARGIN_M
        spec = OracleSpec(limit, OBJECTIVE, m1.GAS_HEATER_EFFICIENCY)
        results = run_oracle_grid(
            config, grid, bounds, budget, spec, workers=arguments.workers, label=f"oracle k={j}"
        )
        write_artifact(
            artifact_name("oracle", j),
            {**common, "margin_factor": j, "tightened_limit_kg_per_kg": limit, "results": results},
        )
    for j in FIXED_FACTORS:
        path = RUNS_DIR / f"{artifact_name('fixed', j)}.json"
        if path.exists():
            print(f"skip {path.name} (exists)")
            continue
        limit = LIMIT_KG_PER_KG - j * MARGIN_M
        spec = OracleSpec(limit, OBJECTIVE, m1.GAS_HEATER_EFFICIENCY)
        fixed = find_fixed_point(config, grid, bounds, budget, spec)
        write_artifact(
            artifact_name("fixed", j),
            {**common, "margin_factor": j, "tightened_limit_kg_per_kg": limit, "fixed": fixed},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

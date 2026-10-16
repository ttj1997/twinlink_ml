r"""Exhaust recirculation of one fluidised-bed line (docs/22): grid, optimisation, recovery, sensitivity.

Question: how much exhaust can a line recirculate into its inlet air before the product-moisture
limit is violated, and what does it save in energy and CO2? Laboratory line of M1
(``greybox_gab_placeholder.yaml``, placeholder physics, lab scale; the recovery numbers are lab
numbers scaled linearly by the constant ``k`` of M2). No plant statement anywhere.

Settings (fixed here, before any run; written down in docs/22 section 2):

* Contexts: the 25 ambient points of Befund 52 (5 x 5 grid of the disturbance box, T_amb 5-35 degC,
  phi 0.25-0.85). Sensitivity: the 3 x 3 sub-grid (T_amb 5/20/35, phi 0.25/0.55/0.85; indices
  0, 2, 4, 10, 12, 14, 20, 22, 24) to bound the run time.
* Objective as M1: CO2 with the gas-fired heater booking (``eta_gas`` 0.80, 0.366 / 0.201
  kg CO2e/kWh), fan power included; recirculation fan cube law with the placeholder
  ``recirculation_duct_loss_coefficient`` = 350 (equal to the distributor coefficient; band
  100-1000, no source). Quality limit 0.15 with the margin m = 0.005961819615175404 (the M arm of
  M1/M2, Befund 65 FA2); kinetics keep the critical moisture 0.20. Lower bound of the gas velocity
  1.26 m/s (laboratory minimum, main case); 0.80 m/s as a sensitivity. Inlet box 60-120 degC
  (config). Nozzle pressure fixed at 1.0 bar (exact optimum, see the module).
* Optimiser: DE + Nelder-Mead of M1 (seed 0, popsize 10, tolerance 1e-4, search precision 30 /
  1e-7) with ``maxiter`` reduced from 50 to 20 (run time: one steady state costs about three times
  the fresh-air model). The stage ``anchor`` checks this before any other stage at the contexts
  0, 12, 24: (i) the fresh-air path (field unset) with the reduced budget against the M arm of
  ``m1_margin_cost_l015_umin126_k1`` (CO2 per point), (ii) ``r = 0.0`` against the fresh-air
  path, (iii) the reduced against the full budget (maxiter 50) for ``r = 0.0`` and for ``r`` free.
  Gate, fixed now: every CO2 difference of (i)-(iii) below 0.5 % relative; otherwise the main
  stage runs with maxiter 50 (``--full-budget``).
* Arms of ``main`` (u_min 1.26): G -- ``r`` fixed on {0, 0.1, ..., 0.8} (the requested grid) plus
  0.9 (extension, reported separately), set points (T_in, u) optimised, 25 contexts each; R -- ``r``
  as third decision variable in [0, 0.95] (the config maximum); J0 / J -- as G at r = 0 and
  as R, but the objective is
  the CO2 AFTER the recovery cascade (layout ``fresh_air``: the exchanger preheats the fresh air
  before the mixing point), k of M2.
* Stage ``sensitivity`` (3 x 3 contexts, arms G at r = 0 and R): kinetics critical moisture 0.17
  and 0.25; quality limit 0.20 (stress case of M1, kinetics 0.20); GAB monolayer x0.7 and x1.3,
  GAB K 0.80 instead of 0.75 (uncalibrated isotherm); maximum saturation degree 0.45 instead of 0.60
  (gas-solid transfer proxy, literature 0.4-0.8); lower gas-velocity bound 0.80 m/s. The ambient
  humidity is an axis of the contexts themselves.

Run log (resumable, outside the repository): ``--cache-dir`` (default ``C:/twinlink_runs/rec``),
one JSON line per optimisation; the artifact is written when all tasks of a stage are done::

    tools/run_here.sh scripts/run_exhaust_recirc.py --stage anchor
    tools/run_here.sh scripts/run_exhaust_recirc.py --stage main
    tools/run_here.sh scripts/run_exhaust_recirc.py --stage sensitivity
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import warnings
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import (  # noqa: E402
    REPO_ROOT,
    RUNS_DIR,
    provenance,
    require_clean_tree,
    write_artifact,
)

from twinlink_ml.evaluation.exhaust_recirculation import (  # noqa: E402
    RecirculationTask,
    RecoverySpec,
    run_tasks,
)
from twinlink_ml.evaluation.site_network import scale_factor  # noqa: E402
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config  # noqa: E402
from twinlink_ml.simulation.config import ProcessConfig  # noqa: E402
from twinlink_ml.simulation.operating_point import (  # noqa: E402
    DEFAULT_SEARCH_MAX_ITERATIONS,
    DEFAULT_SEARCH_TOLERANCE,
)
from twinlink_ml.simulation.oracle_search import (  # noqa: E402
    OptimizationBudget,
    build_grid,
    evaluate_point,
)

CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
MARGIN_KG_PER_KG = 0.005961819615175404
QUALITY_LIMIT = 0.15
GAS_VELOCITY_MIN_MAIN = 1.26
GAS_VELOCITY_MIN_SENSITIVITY = 0.80
DUCT_LOSS_COEFFICIENT = 350.0
R_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
R_EXTENSION = (0.9,)
R_BOX = (0.0, 0.95)
ANCHOR_CONTEXTS = (0, 12, 24)
SUBGRID = (0, 2, 4, 10, 12, 14, 20, 22, 24)
ANCHOR_GATE_RELATIVE = 0.005
M1_ARTIFACT = "m1_margin_cost_l015_umin126_k1"

SENSITIVITY_VARIANTS: dict[str, dict[str, Any]] = {
    "xcr017": {"overrides": (("material", "critical_moisture_kg_per_kg", 0.17),)},
    "xcr025": {"overrides": (("material", "critical_moisture_kg_per_kg", 0.25),)},
    "lq020": {"overrides": (), "quality_limit": 0.20},
    "gab_m070": {"overrides": (("material", "gab_monolayer_moisture_kg_per_kg", 0.014),)},
    "gab_m130": {"overrides": (("material", "gab_monolayer_moisture_kg_per_kg", 0.026),)},
    "gab_k080": {"overrides": (("material", "gab_k_constant_at_reference_temperature", 0.80),)},
    "nu045": {"overrides": (("material", "max_saturation_degree", 0.45),)},
    "umin080": {"overrides": (), "gas_velocity_min": GAS_VELOCITY_MIN_SENSITIVITY},
}


def _config() -> ProcessConfig:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    config = load_process_config(CONFIG_PATH)
    assert config.material.critical_moisture_kg_per_kg == 0.20
    return dataclasses.replace(
        config,
        energy=dataclasses.replace(
            config.energy, recirculation_duct_loss_coefficient=DUCT_LOSS_COEFFICIENT
        ),
    )


def _budget(maxiter: int) -> OptimizationBudget:
    return OptimizationBudget(
        seed=0,
        popsize=10,
        maxiter=maxiter,
        de_tol=1.0e-4,
        search_max_iterations=DEFAULT_SEARCH_MAX_ITERATIONS,
        search_tolerance=DEFAULT_SEARCH_TOLERANCE,
        workers=1,
    )


def _scale_factor(config: ProcessConfig) -> float:
    """K of M2: 8 kg/s over the dry-air flow of line A at the config nominal point (fresh air)."""
    nominal = evaluate_point(
        config.actions.inlet_temperature_celsius.nominal,
        config.actions.gas_velocity_m_per_s.nominal,
        config.actions.nozzle_pressure_bar.nominal,
        config.disturbances.ambient_temperature_celsius.nominal,
        config.disturbances.ambient_relative_humidity.nominal,
        max_iterations=DEFAULT_SEARCH_MAX_ITERATIONS,
        tolerance=DEFAULT_SEARCH_TOLERANCE,
        base_config=dataclasses.replace(
            config,
            energy=dataclasses.replace(config.energy, recirculation_duct_loss_coefficient=None),
        ),
    )
    return scale_factor(nominal.dry_air_mass_flow_kg_per_s)


def _task(
    config: ProcessConfig,
    key: str,
    index: int,
    budget: OptimizationBudget,
    *,
    fixed_r: float | None = 0.0,
    r_box: tuple[float, float] | None = None,
    quality_limit: float = QUALITY_LIMIT,
    gas_velocity_min: float = GAS_VELOCITY_MIN_MAIN,
    overrides: tuple[tuple[str, str, float], ...] = (),
    recovery: RecoverySpec | None = None,
    fresh_air_path: bool = False,
) -> RecirculationTask:
    t_amb, phi = build_grid(config, 5)[index]
    actions = config.actions
    return RecirculationTask(
        key=key,
        context_index=index,
        ambient_temperature_celsius=t_amb,
        ambient_relative_humidity=phi,
        bounds_t_in=(
            actions.inlet_temperature_celsius.minimum,
            actions.inlet_temperature_celsius.maximum,
        ),
        bounds_u=(gas_velocity_min, actions.gas_velocity_m_per_s.maximum),
        quality_threshold_kg_per_kg=quality_limit - MARGIN_KG_PER_KG,
        fixed_r=fixed_r,
        r_box=r_box,
        fresh_air_path=fresh_air_path,
        recovery=recovery,
        budget=budget,
        config_overrides=overrides,
    )


def _settings(config: ProcessConfig, budget: OptimizationBudget, k: float) -> dict[str, Any]:
    return {
        "config_fingerprint": config.fingerprint,
        "config_path": str(CONFIG_PATH.relative_to(REPO_ROOT)),
        "margin_kg_per_kg": MARGIN_KG_PER_KG,
        "quality_limit_kg_per_kg": QUALITY_LIMIT,
        "kinetics_critical_moisture_kg_per_kg": config.material.critical_moisture_kg_per_kg,
        "recirculation_duct_loss_coefficient": DUCT_LOSS_COEFFICIENT,
        "gas_velocity_min_main_m_per_s": GAS_VELOCITY_MIN_MAIN,
        "budget": dataclasses.asdict(budget),
        "scale_factor_k": k,
        "r_grid": list(R_GRID),
        "r_extension": list(R_EXTENSION),
        "r_box": list(R_BOX),
        "scale_note": "laboratory line, placeholder physics; recovery figures scaled by k (D4)",
    }


def run_anchor(cache_dir: Path, workers: int) -> int:
    config = _config()
    reduced, full = _budget(20), _budget(50)
    tasks = []
    for i in ANCHOR_CONTEXTS:
        tasks += [
            _task(config, f"anchor_fresh_reduced_{i}", i, reduced, fresh_air_path=True),
            _task(config, f"anchor_r0_reduced_{i}", i, reduced),
            _task(config, f"anchor_r0_full_{i}", i, full),
            _task(config, f"anchor_rfree_reduced_{i}", i, reduced, fixed_r=None, r_box=R_BOX),
            _task(config, f"anchor_rfree_full_{i}", i, full, fixed_r=None, r_box=R_BOX),
        ]
    results = {
        r["key"]: r
        for r in run_tasks(
            config, tasks, workers=workers, cache_path=cache_dir / "anchor.jsonl", label="anchor"
        )
    }
    m1 = json.loads((RUNS_DIR / f"{M1_ARTIFACT}.json").read_text("utf-8"))["arm_M_oracle"]
    rows = []
    for i in ANCHOR_CONTEXTS:
        m1_co2 = next(p for p in m1 if p["_grid_index"] == i)["gas_heater"]["co2_kg_per_h"]

        def co2(key: str, i: int = i) -> float:
            return float(results[f"{key}_{i}"]["co2_kg_per_h"])

        diffs = {
            "fresh_reduced_vs_m1": co2("anchor_fresh_reduced") / m1_co2 - 1.0,
            "r0_vs_fresh": co2("anchor_r0_reduced") / co2("anchor_fresh_reduced") - 1.0,
            "r0_reduced_vs_full": co2("anchor_r0_reduced") / co2("anchor_r0_full") - 1.0,
            "rfree_reduced_vs_full": co2("anchor_rfree_reduced") / co2("anchor_rfree_full") - 1.0,
        }
        rows.append({"context_index": i, "m1_co2_kg_per_h": m1_co2, "relative_differences": diffs})
    passed = all(
        abs(v) < ANCHOR_GATE_RELATIVE for row in rows for v in row["relative_differences"].values()
    )
    write_artifact(
        "exhaust_recirc_anchor",
        {
            "_provenance": provenance("not used"),
            "settings": _settings(config, reduced, float("nan")),
            "gate_relative": ANCHOR_GATE_RELATIVE,
            "gate_passed": passed,
            "rows": rows,
            "results": list(results.values()),
        },
    )
    print(json.dumps(rows, indent=1), "\ngate passed:", passed)
    return 0 if passed else 1


def run_main(cache_dir: Path, workers: int, maxiter: int) -> int:
    config = _config()
    budget = _budget(maxiter)
    k = _scale_factor(config)
    recovery = RecoverySpec(scale_factor=k, layout="fresh_air")
    tasks = []
    for i in range(25):
        for r in R_GRID + R_EXTENSION:
            tasks.append(_task(config, f"G_r{round(r * 100):03d}_{i}", i, budget, fixed_r=r))
        tasks.append(_task(config, f"R_{i}", i, budget, fixed_r=None, r_box=R_BOX))
        tasks.append(_task(config, f"J0_{i}", i, budget, recovery=recovery))
        tasks.append(
            _task(config, f"J_{i}", i, budget, fixed_r=None, r_box=R_BOX, recovery=recovery)
        )
    results = run_tasks(
        config, tasks, workers=workers, cache_path=cache_dir / "main.jsonl", label="main"
    )
    write_artifact(
        f"exhaust_recirc_main{'' if maxiter == 20 else f'_maxiter{maxiter}'}",
        {
            "_provenance": provenance(recovery),
            "settings": _settings(config, budget, k),
            "recovery_spec": dataclasses.asdict(recovery),
            "results": results,
        },
    )
    return 0


def run_sensitivity(cache_dir: Path, workers: int, maxiter: int) -> int:
    config = _config()
    budget = _budget(maxiter)
    tasks = []
    for name, variant in SENSITIVITY_VARIANTS.items():
        common = {
            "quality_limit": variant.get("quality_limit", QUALITY_LIMIT),
            "gas_velocity_min": variant.get("gas_velocity_min", GAS_VELOCITY_MIN_MAIN),
            "overrides": variant["overrides"],
        }
        for i in SUBGRID:
            tasks.append(_task(config, f"S_{name}_r0_{i}", i, budget, **common))
            tasks.append(
                _task(config, f"S_{name}_R_{i}", i, budget, fixed_r=None, r_box=R_BOX, **common)
            )
    results = run_tasks(
        config, tasks, workers=workers, cache_path=cache_dir / "sensitivity.jsonl", label="sens"
    )
    write_artifact(
        f"exhaust_recirc_sensitivity{'' if maxiter == 20 else f'_maxiter{maxiter}'}",
        {
            "_provenance": provenance("not used"),
            "settings": _settings(config, budget, float("nan")),
            "variants": {
                name: {
                    "overrides": [list(o) for o in v["overrides"]],
                    "quality_limit": v.get("quality_limit", QUALITY_LIMIT),
                    "gas_velocity_min": v.get("gas_velocity_min", GAS_VELOCITY_MIN_MAIN),
                }
                for name, v in SENSITIVITY_VARIANTS.items()
            },
            "subgrid": list(SUBGRID),
            "results": results,
        },
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--stage", choices=("anchor", "main", "sensitivity"), required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("C:/twinlink_runs/rec"))
    parser.add_argument("--workers", type=int, default=cpu_count())
    parser.add_argument(
        "--full-budget", action="store_true", help="maxiter 50 instead of 20 (anchor gate failed)"
    )
    args = parser.parse_args()
    require_clean_tree()
    maxiter = 50 if args.full_budget else 20
    if args.stage == "anchor":
        return run_anchor(args.cache_dir, args.workers)
    if args.stage == "main":
        return run_main(args.cache_dir, args.workers, maxiter)
    return run_sensitivity(args.cache_dir, args.workers, maxiter)


if __name__ == "__main__":
    raise SystemExit(main())

r"""M3' stage 0b: how much of the stage-0 interaction is a switch of the best allocation.

Added AFTER the result of stage 0 (``m3_interaction_check``, rule met: joint runs justified), as a
DESCRIPTIVE addition without a decision rule, because a spot check had shown that the mixed second
difference of the site CO2 at the best allocation (a minimum over discrete allocations, which is not
separable even when every single allocation is) is mostly due to the best allocation changing with
the set points when no limit is set. Same grids, bounds, limits and states as stage 0 (imported from
``probe_m3_interaction_check``: same ``T_GRID``, spray rates, scale factor, 441 allocations,
``Q_cap`` = c * U, control as there); no optimisation.

Per scenario, bound and case: the best allocation at the CENTRE of the 3 x 3 grid is held fixed and
the mixed second difference of the site CO2 of THAT allocation is formed on the four blocks (the
smooth coupling inside one allocation); and the number of the 9 grid points whose best allocation
differs from the centre one. Reported per case: median and maximum over the scenarios of the largest
fixed-allocation difference, the number of scenarios at or above the stage-0 resolution (0.5 kg
CO2e/h), and the median number of switching grid points.

Run-time estimate (before the run): the same states and cascades as stage 0 without a search; about
3 min on 16 workers.

    PYTHONPATH=<worktree>/src python scripts/probe_m3_interaction_fixed_allocation.py
"""

from __future__ import annotations

import json
import sys
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import probe_m3_interaction_check as stage0  # noqa: E402
from _common import (  # noqa: E402
    REPO_ROOT,
    RUNS_DIR,
    provenance,
    require_clean_tree,
    write_artifact,
)

import twinlink_ml  # noqa: E402
from twinlink_ml.evaluation import coupling_gain as cg  # noqa: E402
from twinlink_ml.evaluation import site_network as sn  # noqa: E402
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config  # noqa: E402
from twinlink_ml.simulation.oracle_search import build_grid, evaluate_point  # noqa: E402

CENTRE = (1, 1)


def _task(task: tuple[float, int, float, float, float, float]) -> dict[str, Any]:
    u, index, t_amb, phi, k, p = task
    grid_t = stage0.T_GRID
    a_states = [stage0._line("A", t, u, p, t_amb, phi) for t in grid_t]
    b_states = [stage0._line("B", t, u, p, t_amb, phi) for t in grid_t]
    u_a = stage0._uptake(a_states[CENTRE[0]], k)
    allocations = stage0._STATE["allocations"]
    out: dict[str, Any] = {"grid_index": index, "cases": {}}
    free_demand = 0.0
    for a in a_states:
        for b in b_states:
            best = stage0._best(a, b, k, None)
            if best is not None:
                free_demand = max(free_demand, best[1])
    for name, factor in stage0.CASES:
        if factor is None or u_a is None:
            cap_kw = None
        elif name.startswith("control"):
            cap_kw = max(factor * u_a, 1.05 * free_demand)
        else:
            cap_kw = factor * u_a
        spec = sn.SiteSpec(k, shared_cap_kw=cap_kw)
        table: list[list[dict[Any, float] | None]] = []
        for a in a_states:
            row: list[dict[Any, float] | None] = []
            for b in b_states:
                if a is None or b is None:
                    row.append(None)
                    continue
                results, _ = sn.evaluate_site((a, b), spec, allocations)
                row.append({r.connections: r.co2_kg_per_h for r in results})
            table.append(row)
        centre = table[CENTRE[0]][CENTRE[1]]
        if centre is None:
            out["cases"][name] = {"max_abs_fixed_allocation_difference": None}
            continue
        centre_best = min(centre, key=lambda c: centre[c])
        blocks = []
        for i in range(len(grid_t) - 1):
            for j in range(len(grid_t) - 1):
                cells = (table[i + 1][j + 1], table[i + 1][j], table[i][j + 1], table[i][j])
                if any(c is None for c in cells):
                    continue
                c11, c10, c01, c00 = (c[centre_best] for c in cells if c is not None)
                blocks.append(c11 - c10 - c01 + c00)
        switching = sum(
            1
            for row in table
            for cell in row
            if cell is not None and min(cell, key=lambda c: cell[c]) != centre_best
        )
        out["cases"][name] = {
            "cap_kw": cap_kw,
            "centre_allocation": [list(c) for c in centre_best],
            "n_blocks": len(blocks),
            "max_abs_fixed_allocation_difference": max(abs(v) for v in blocks) if blocks else None,
            "n_grid_points_with_another_best_allocation": switching,
        }
    return out


def main() -> int:
    require_clean_tree()
    assert Path(twinlink_ml.__file__).resolve().is_relative_to(REPO_ROOT)
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    base = load_process_config(stage0.CONFIG_PATH)
    configs = {n: sn.with_spray_rate(base, r) for n, r in stage0.SPRAY_KG_PER_H.items()}
    p_bound = base.actions.nozzle_pressure_bar.minimum
    nominal = evaluate_point(
        base.actions.inlet_temperature_celsius.nominal,
        base.actions.gas_velocity_m_per_s.nominal,
        base.actions.nozzle_pressure_bar.nominal,
        base.disturbances.ambient_temperature_celsius.nominal,
        base.disturbances.ambient_relative_humidity.nominal,
        max_iterations=30,
        tolerance=1e-7,
        base_config=base,
    )
    k = sn.scale_factor(nominal.dry_air_mass_flow_kg_per_s)
    stored = json.loads((RUNS_DIR / "m3_interaction_check.json").read_text("utf-8"))
    assert abs(k - stored["scale_factor_k"]) < 1e-9 * k
    grid = build_grid(base, 5)
    per_bound: dict[str, Any] = {}
    for u in stage0.BOUNDS_M_PER_S:
        tasks = [(u, i, t, phi, k, p_bound) for i, (t, phi) in enumerate(grid)]
        with Pool(cpu_count(), initializer=stage0._init, initargs=(configs,)) as pool:
            scenarios = pool.map(_task, tasks, chunksize=1)
        summary: dict[str, Any] = {}
        for name, _ in stage0.CASES:
            values = [
                s["cases"][name]["max_abs_fixed_allocation_difference"]
                for s in scenarios
                if s["cases"][name].get("max_abs_fixed_allocation_difference") is not None
            ]
            switching = [
                s["cases"][name]["n_grid_points_with_another_best_allocation"]
                for s in scenarios
                if "n_grid_points_with_another_best_allocation" in s["cases"][name]
            ]
            resolution = cg.INTERACTION_RESOLUTION_KG_PER_H
            summary[name] = {
                "n_scenarios_with_a_value": len(values),
                "median_kg_per_h": float(np.median(values)) if values else None,
                "max_kg_per_h": max(values) if values else None,
                "n_at_or_above_resolution": sum(v >= resolution for v in values),
                "median_n_grid_points_with_another_best_allocation": (
                    float(np.median(switching)) if switching else None
                ),
            }
        per_bound[f"{u:.2f}"] = {"summary_by_case": summary, "scenarios": scenarios}
    write_artifact(
        "m3_interaction_fixed_allocation",
        {
            "_provenance": provenance(),
            "added_after": "m3_interaction_check (descriptive, no decision rule)",
            "resolution_kg_per_h": cg.INTERACTION_RESOLUTION_KG_PER_H,
            "per_bound": per_bound,
        },
    )
    for bound, block in per_bound.items():
        print(bound, json.dumps(block["summary_by_case"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

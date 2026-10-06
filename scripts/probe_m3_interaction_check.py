r"""M3' stage 0: interaction check of two lines that share a limited exchanger.

Docs/09 "Draft of the next stage", released by the project lead on 2026-10-06. Two lines (A: spray
1.5 kg/h, B: 2.0 kg/h, LB3) in the scaled site of M2 (fixed external source Q1 and sink S1, condensing
cascade, ``T_kalt`` 29.1 degC, ``dT_min`` 20 K, scale factor ``k`` = 271.797, a scaling computation),
with ONE exchanger of limited duty ``Q_cap`` (synthetic, no figure exists in the repository) for the
heat of Q1 into the inlet air of both lines (``SharedCap`` of the cascade, E1). No optimisation here.

For every one of the 25 Befund 52 ambient grid points (test set) and at each lower gas-velocity
bound (1.26 m/s main case, 0.80 m/s comparison; both lines at the bound, nozzle pressure at its
lower bound 1.0 bar, D6) the site CO2 of the BEST of the enumerated allocations (every source
feeds at least one sink; for the two stages Q1 -> SA, Q1 -> SB both orders: 441 sets) is evaluated on
a 3 x 3 grid of the two inlet temperatures (``T_GRID`` for each line) and the mixed second
difference ``f(a1,b1) - f(a1,b0) - f(a0,b1) + f(a0,b0)`` is formed on the four blocks of the grid.
Separable means: the site CO2 is the sum of a function of line A alone and one of line B alone, so
that "each line on its own, then the best allocation" loses nothing against the joint optimum
(at the set points tested). The objective is the physical site CO2; the quality limit couples
nothing between the lines and is not part of it. A solver failure at a grid point masks the blocks that
contain it (counted, never replaced).

Limits, per scenario: ``Q_cap = c * U`` with ``U`` the uptake of ONE line from Q1 without a limit
(power from Q1 into the inlet air of line A in the best single-line allocation of M2, Q1 -> S1,
Q1 -> SA, LA -> S1, at ``T_in`` 95 degC; line B reported alongside) and ``c`` in {0.5, 1.0, 1.5};
reference: no limit. CONTROL: the limit must lie above the common demand of the two lines, where
nothing may differ from the case without a limit (check of the build, bit for bit): ``Q_cap`` =
``max(2.5 * U, 1.05 * D)`` with ``D`` the largest power from Q1 into the two lines that the best
unlimited allocation takes at any point of the 3 x 3 grid of that scenario (found by the smoke test:
the two lines take 3-6 times the uptake of one line, because the exhaust of both lines warms S1 first
and so leaves the exhaust of Q1 hotter; 2.5 * U alone would bind); the realised factor
``Q_cap / U`` is reported.

Decision rule, fixed before the run (``twinlink_ml.evaluation.coupling_gain``): the largest absolute
mixed difference over all scenarios, bounds and limits c in {0.5, 1.0, 1.5} must reach 0.5 kg CO2e/h,
otherwise the joint runs are NOT made and the result is "no coupling for this kind of scarcity"
(reported as such). Also reported: how often the limit binds, the control check, and the D6 check at
one scenario (index 12, the central ambient point): the site CO2 of the line at its M2 optimum with the
M2 nozzle pressure against 1.0 bar.

Run-time estimate (before the run, 16 logical CPUs): about 7 solves per scenario (3 + 3 inlet
temperatures plus the reference), 25 scenarios x 2 bounds = 350 solves at 0.17-0.21 s; per scenario 5
limits x 9 grid points x 441 allocations of the cascade at 0.06-0.08 ms plus the network
construction, a few seconds. About 3 min wall on 16 workers, 10 min at the outside.

    PYTHONPATH=<worktree>/src python scripts/probe_m3_interaction_check.py
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
sys.path.insert(0, str(SCRIPTS / "verify"))

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
from twinlink_ml.evaluation.site_oracle import SiteOracleSpec, evaluate_site_point  # noqa: E402
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config  # noqa: E402
from twinlink_ml.simulation.oracle_search import build_grid, evaluate_point  # noqa: E402

CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
BOUNDS_M_PER_S = (1.26, 0.80)
T_GRID = (75.0, 95.0, 115.0)
REFERENCE_T_IN = 95.0
CAP_FACTORS = (0.5, 1.0, 1.5)
CONTROL_FACTOR = 2.5
SPRAY_KG_PER_H = {"A": 1.5, "B": 2.0}
QUALITY_LIMIT = 0.15
MARGIN = 0.005961819615175404
D6_SCENARIO_INDEX = 12
M2_ARTIFACTS = {1.26: "m2_site_network_l015_umin126", 0.80: "m2_site_network_l015_umin080"}
CASES: tuple[tuple[str, float | None], ...] = (
    ("none", None),
    *((f"c{c}", c) for c in CAP_FACTORS),
    (f"control_c{CONTROL_FACTOR}", CONTROL_FACTOR),
)

_STATE: dict[str, Any] = {}


def _init(configs: dict[str, Any]) -> None:
    _STATE["configs"] = configs
    _STATE["allocations"] = sn.all_allocations(2, order_variants=True)


def _line(name: str, t_in: float, u: float, p: float, t_amb: float, phi: float) -> Any:
    config = _STATE["configs"][name]
    try:
        solution = evaluate_point(
            t_in, u, p, t_amb, phi, max_iterations=30, tolerance=1e-7, base_config=config
        )
    except (RuntimeError, ValueError):
        return None
    return sn.line_state(name, solution, config, t_amb=t_amb, phi_amb=phi, t_in=t_in)


def _best(a: Any, b: Any, k: float, cap_kw: float | None) -> tuple[float, float] | None:
    """Site CO2 of the best allocation and the power from Q1 into the lines in it."""
    if a is None or b is None:
        return None
    results, _ = sn.evaluate_site(
        (a, b), sn.SiteSpec(k, shared_cap_kw=cap_kw), _STATE["allocations"]
    )
    if not results:
        return None
    best = min(results, key=lambda r: r.co2_kg_per_h)
    return best.co2_kg_per_h, best.recovered_kw_external_to_lines


def _uptake(line: Any, k: float) -> float | None:
    """Power from Q1 into the inlet air of ONE line, best single-line allocation of M2."""
    if line is None:
        return None
    n = line.name
    results, _ = sn.evaluate_site(
        (line,),
        sn.SiteSpec(k),
        [(("Q1", "S1"), ("Q1", f"S{n}"), (f"L{n}", "S1"))],
    )
    return results[0].recovered_kw_external_to_lines if results else None


def _scenario_task(task: tuple[float, int, float, float, float, float]) -> dict[str, Any]:
    u, index, t_amb, phi, k, p = task
    a_states = [_line("A", t, u, p, t_amb, phi) for t in T_GRID]
    b_states = [_line("B", t, u, p, t_amb, phi) for t in T_GRID]
    ref = T_GRID.index(REFERENCE_T_IN)
    u_a = _uptake(a_states[ref], k)
    u_b = _uptake(b_states[ref], k)
    out: dict[str, Any] = {
        "grid_index": index,
        "t_amb": t_amb,
        "phi": phi,
        "n_failed_grid_points": sum(s is None for s in a_states + b_states),
        "uptake_one_line_kw": {"A": u_a, "B": u_b},
        "cases": {},
    }
    base_f: list[list[float | None]] | None = None
    demand = 0.0  # largest power from Q1 into the two lines in the best unlimited allocation
    for a in a_states:
        for b in b_states:
            free = _best(a, b, k, None)
            if free is not None:
                demand = max(demand, free[1])
    out["largest_unlimited_two_line_uptake_kw"] = demand
    for name, factor in CASES:
        if factor is None or u_a is None:
            cap_kw = None
        elif name.startswith("control"):
            cap_kw = max(factor * u_a, 1.05 * demand)
        else:
            cap_kw = factor * u_a
        f = [
            [None if (r := _best(a, b, k, cap_kw)) is None else r[0] for b in b_states]
            for a in a_states
        ]
        blocks = []
        for i in range(len(T_GRID) - 1):
            for j in range(len(T_GRID) - 1):
                corners = (f[i + 1][j + 1], f[i + 1][j], f[i][j + 1], f[i][j])
                blocks.append(
                    None
                    if any(v is None for v in corners)
                    else corners[0] - corners[1] - corners[2] + corners[3]
                )
        entry: dict[str, Any] = {
            "cap_kw": cap_kw,
            "cap_over_one_line_uptake": None if cap_kw is None or not u_a else cap_kw / u_a,
            "site_co2": f,
            "mixed_differences": blocks,
            "max_abs_mixed_difference": (
                None
                if all(v is None for v in blocks)
                else max(abs(v) for v in blocks if v is not None)
            ),
        }
        if name == "none":
            base_f = f
        elif base_f is not None:
            diffs = [
                abs(f[i][j] - base_f[i][j])
                for i in range(len(T_GRID))
                for j in range(len(T_GRID))
                if f[i][j] is not None and base_f[i][j] is not None
            ]
            entry["n_grid_points_changed_by_the_limit"] = sum(d > 1.0e-9 for d in diffs)
            entry["max_abs_difference_to_no_limit"] = max(diffs) if diffs else None
        out["cases"][name] = entry
    return out


def _d6_check(configs: dict[str, Any], site_k: float) -> dict[str, Any]:
    """Site CO2 of line A at its M2 optimum with the M2 nozzle pressure against 1.0 bar."""
    result: dict[str, Any] = {}
    for bound, name in M2_ARTIFACTS.items():
        artifact = json.loads((RUNS_DIR / f"{name}.json").read_text("utf-8"))
        s = artifact["arm_S"][D6_SCENARIO_INDEX]
        spec = SiteOracleSpec(QUALITY_LIMIT - MARGIN, sn.SiteSpec(site_k))
        common = {
            "t_in": s["inlet_temperature_celsius"],
            "u_gas": s["gas_velocity_m_per_s"],
            "t_amb": s["ambient_temperature_celsius"],
            "phi_amb": s["ambient_relative_humidity"],
        }
        free = evaluate_site_point(configs["A"], spec, p_nozzle=s["nozzle_pressure_bar"], **common)
        fixed = evaluate_site_point(configs["A"], spec, p_nozzle=1.0, **common)
        result[f"{bound:.2f}"] = {
            "grid_index": D6_SCENARIO_INDEX,
            "p_m2_optimum_bar": s["nozzle_pressure_bar"],
            "site_co2_m2_pressure_kg_per_h": free["routing_best"]["co2_kg_per_h"],
            "site_co2_pressure_1bar_kg_per_h": fixed["routing_best"]["co2_kg_per_h"],
            "difference_kg_per_h": fixed["routing_best"]["co2_kg_per_h"]
            - free["routing_best"]["co2_kg_per_h"],
            "feasible_at_1bar_tightened_limit": fixed["feasible"],
            "moisture_at_1bar": fixed["particle_moisture_kg_per_kg"],
            "resolution_kg_per_h": cg.INTERACTION_RESOLUTION_KG_PER_H,
        }
    return result


def main() -> int:
    require_clean_tree()
    assert Path(twinlink_ml.__file__).resolve().is_relative_to(REPO_ROOT)
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    base = load_process_config(CONFIG_PATH)
    configs = {name: sn.with_spray_rate(base, rate) for name, rate in SPRAY_KG_PER_H.items()}
    assert base.actions.spray_rate_kg_per_h.nominal == SPRAY_KG_PER_H["A"]
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
    stored_k = json.loads((RUNS_DIR / "m0_line_b_network.json").read_text("utf-8"))[
        "scale_factor_k"
    ]
    assert abs(k - stored_k) < 1e-9 * k
    grid = build_grid(base, 5)

    per_bound: dict[str, Any] = {}
    all_values: list[float] = []
    for u in BOUNDS_M_PER_S:
        tasks = [(u, i, t, phi, k, p_bound) for i, (t, phi) in enumerate(grid)]
        with Pool(cpu_count(), initializer=_init, initargs=(configs,)) as pool:
            scenarios = pool.map(_scenario_task, tasks, chunksize=1)
        summary: dict[str, Any] = {}
        for name, factor in CASES:
            values = [
                s["cases"][name]["max_abs_mixed_difference"]
                for s in scenarios
                if s["cases"][name]["max_abs_mixed_difference"] is not None
            ]
            entry: dict[str, Any] = {
                "n_scenarios_with_a_value": len(values),
                "max_abs_mixed_difference_kg_per_h": max(values) if values else None,
                "median_kg_per_h": float(np.median(values)) if values else None,
                "n_at_or_above_resolution": sum(
                    v >= cg.INTERACTION_RESOLUTION_KG_PER_H for v in values
                ),
            }
            if factor is not None:
                entry["n_scenarios_where_the_limit_binds"] = sum(
                    s["cases"][name].get("n_grid_points_changed_by_the_limit", 0) > 0
                    for s in scenarios
                )
                diffs = [
                    s["cases"][name]["max_abs_difference_to_no_limit"]
                    for s in scenarios
                    if s["cases"][name].get("max_abs_difference_to_no_limit") is not None
                ]
                entry["max_abs_difference_to_no_limit_kg_per_h"] = max(diffs) if diffs else None
            if factor in CAP_FACTORS:
                all_values.extend(values)
            summary[name] = entry
        ups = [
            s["uptake_one_line_kw"]["A"]
            for s in scenarios
            if s["uptake_one_line_kw"]["A"] is not None
        ]
        upb = [
            s["uptake_one_line_kw"]["B"]
            for s in scenarios
            if s["uptake_one_line_kw"]["B"] is not None
        ]
        per_bound[f"{u:.2f}"] = {
            "gas_velocity_lower_bound_m_per_s": u,
            "n_scenarios_with_failed_grid_points": sum(
                s["n_failed_grid_points"] > 0 for s in scenarios
            ),
            "uptake_one_line_kw": {
                "A_median": float(np.median(ups)),
                "A_min": min(ups),
                "A_max": max(ups),
                "B_median": float(np.median(upb)),
                "max_abs_A_minus_B": max(abs(x - y) for x, y in zip(ups, upb, strict=False)),
                "n_zero_uptake": sum(x <= 0.0 for x in ups),
            },
            "summary_by_case": summary,
            "scenarios": scenarios,
        }
    verdict = cg.interaction_verdict(all_values)
    payload = {
        "_provenance": provenance(),
        "test_set": "25 Befund-52 grid points; lines A (1.5 kg/h) and B (2.0 kg/h), scaled site",
        "scale_factor_k": k,
        "t_grid_celsius": list(T_GRID),
        "nozzle_pressure_bar": p_bound,
        "cap_factors": list(CAP_FACTORS),
        "control_factor": CONTROL_FACTOR,
        "rule": {
            "resolution_kg_per_h": cg.INTERACTION_RESOLUTION_KG_PER_H,
            "joint_runs_iff": "largest |mixed difference| over scenarios, bounds and c in "
            f"{list(CAP_FACTORS)} >= resolution",
        },
        "verdict": verdict,
        "d6_check": _d6_check(configs, k),
        "per_bound": per_bound,
    }
    write_artifact("m3_interaction_check", payload)
    print(json.dumps(verdict, indent=1))
    for bound, block in per_bound.items():
        print(bound, json.dumps(block["summary_by_case"], indent=1))
    print(json.dumps(payload["d6_check"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

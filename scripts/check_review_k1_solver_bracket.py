r"""Check of review finding K1 (2026-10-09): the lower bed-temperature bound ``T_amb + 10 K`` of the steady-state solver.

Review 2 (``docs/reviews/2026-10-09/pruefung2.md``, K1; also the cross-cutting finding of review 1): the outer
bisection of ``simulation/steady_state.py`` (about l. 300-302) searches the bed temperature only from
``T_amb + 10 K``; ``oracle_search.evaluate_point`` passes no bracket of its own. At warm ambient air this may cut
off valid drying states, and several reported optima are said to sit exactly on this bound. This script CHECKS
that; it changes nothing in ``src/``, re-optimises nothing and is not a correction.

(a) Every reported solution of the artifacts of Befunde 69, 73, 76, 77 (list below) is re-solved at its stored
    set points with the search precision of the runs (30 bisection steps, 1e-7), and its bed temperature is
    compared with ``T_amb + 10``. Where the artifact stores the bed temperature (M1 arms, line B optimum) or the
    product moisture (S2/M3' records), the reproduction is asserted. A solution counts as ON THE BOUND iff
    ``|T_bed - T_amb - 10| <= 1e-3 K``; the distances are reported for every solution.
(b) Every distinct solution on the bound (all of them, not a sample) is probed with wider brackets of the outer
    bisection, lower edge ``T_amb + 5, + 3, + 0, - 5 K`` (upper edge as the default,
    ``max(T_in, T_suspension) + 5``), at the stored set points and with the inlet temperature of that line
    lowered by 1, 2 and 5 K (other set points held; a probe below the box minimum 60 degC is not evaluated:
    the process config admits inlet temperatures only inside [60, 120] degC). The
    regime of each solution is checked: DRYING iff ``Y_in < Y_out < Y_s(T_bed)`` (exhaust humidity below
    saturation at the bed temperature, the air takes up water), with the residual, the product moisture, the
    solver's own feasibility flags and the CO2 of the line alone (gas heater, ``eta_gas`` 0.80, as in M1/M2).
    The probes are fixed points, not a search; whether an optimum would move there is NOT computed.

Reported solutions read (artifact: entries):
  Befund 69: ``m1_margin_cost_l015``, ``m1_margin_cost_l020``, ``m1_margin_cost_l015_umin126`` (arms M0, M and
    the fixed point F at every grid point; line A, three set points), ``m1_anchor_s05`` (legacy anchor);
  Befund 73: ``m3_line_b_optimum_umin126`` (line B alone), ``m3_joint_{control,c0.5,c1.0,c1.5}_umin126`` (S2 and
    M3' per scenario, both lines), ``m3_joint_seeds_umin126`` (S2 and M3' of the seed check),
    ``m3_check_umin126`` (direct searches);
  Befund 76: ``m3x_confirm_evaluation_umin126``, ``m3x_confirm_s18_19_evaluation_umin126``,
    ``m3x_sens_tmax100_evaluation_umin126``, ``m3x_robust_rest_evaluation_umin126`` (best S2, M3' per seed);
  Befund 77: ``m4c_confirm_lab_umin126``, ``m4c_confirm_box_umin126`` (best S2, M3' per seed).

    PYTHONPATH=src python scripts/check_review_k1_solver_bracket.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import run_m3_joint as m3  # noqa: E402
from _common import (  # noqa: E402
    REPO_ROOT,
    RUNS_DIR,
    provenance,
    require_clean_tree,
    write_artifact,
)

import twinlink_ml  # noqa: E402
from twinlink_ml.evaluation.carrier_energy import carrier_energy_from_diagnostics  # noqa: E402
from twinlink_ml.simulation import PlaceholderConfigWarning  # noqa: E402
from twinlink_ml.simulation.backend import NumpyBackend  # noqa: E402
from twinlink_ml.simulation.greybox import GreyBoxFluidBedGranulator  # noqa: E402
from twinlink_ml.simulation.operating_point import (  # noqa: E402
    DEFAULT_SEARCH_MAX_ITERATIONS,
    DEFAULT_SEARCH_TOLERANCE,
)
from twinlink_ml.simulation.oracle_search import build_variant_config  # noqa: E402
from twinlink_ml.simulation.steady_state import solve_steady_state  # noqa: E402
from twinlink_ml.simulation.thermo import saturation_humidity_kg_per_kg  # noqa: E402

ARTIFACT = "review_k1_solver_bracket"
BRACKET_MARGIN_K = 10.0
ON_BOUND_TOLERANCE_K = 1.0e-3
WIDER_LOWER_EDGES_K = (5.0, 3.0, 0.0, -5.0)
INLET_PROBES_K = (0.0, -1.0, -2.0, -5.0)
BOX_INLET_MINIMUM_CELSIUS = 60.0
GAS_HEATER_EFFICIENCY = 0.80
TIGHTENED_LIMIT_KG_PER_KG = 0.15 - 0.005961819615175404
REPRODUCTION_TOLERANCE_K = 1.0e-9
REPRODUCTION_TOLERANCE_MOISTURE = 1.0e-9
_BACKEND = NumpyBackend()
_STATE: dict[str, Any] = {}


# ------------------------------------------------------------------------------------- collection
def _load(name: str) -> dict[str, Any]:
    return json.loads((RUNS_DIR / f"{name}.json").read_text("utf-8"))


def _entry(
    befund: int,
    artifact: str,
    label: str,
    t_amb: float,
    phi: float,
    line: str,
    t_in: float,
    u_gas: float,
    p_nozzle: float,
    *,
    stored_bed: float | None = None,
    stored_moisture: float | None = None,
) -> dict[str, Any]:
    return {
        "befund": befund,
        "artifact": artifact,
        "label": label,
        "t_amb": float(t_amb),
        "phi": float(phi),
        "line": line,
        "set_point": [float(t_in), float(u_gas), float(p_nozzle)],
        "stored_bed_temperature_celsius": stored_bed,
        "stored_moisture_kg_per_kg": stored_moisture,
    }


def _joint_entries(
    befund: int,
    artifact: str,
    label: str,
    t_amb: float,
    phi: float,
    x: list[float],
    moisture: tuple[float | None, float | None] = (None, None),
) -> list[dict[str, Any]]:
    if len(x) == 6:  # six-variable check of Befund 73 g: (T_A, u_A, p_A, T_B, u_B, p_B)
        a, b = (x[0], x[1], x[2]), (x[3], x[4], x[5])
    else:  # four set points, nozzle pressure 1.0 bar (D6)
        a, b = (x[0], x[1], 1.0), (x[2], x[3], 1.0)
    return [
        _entry(befund, artifact, label, t_amb, phi, "A", *a, stored_moisture=moisture[0]),
        _entry(befund, artifact, label, t_amb, phi, "B", *b, stored_moisture=moisture[1]),
    ]


def collect() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in ("m1_margin_cost_l015", "m1_margin_cost_l020", "m1_margin_cost_l015_umin126"):
        art = _load(name)
        for arm in ("M0", "M"):
            for row in art[f"arm_{arm}_oracle"]:
                out.append(
                    _entry(
                        69,
                        name,
                        arm,
                        row["ambient_temperature_celsius"],
                        row["ambient_relative_humidity"],
                        "A",
                        row["inlet_temperature_celsius"],
                        row["gas_velocity_m_per_s"],
                        row["nozzle_pressure_bar"],
                        stored_bed=row["bed_temperature_celsius"],
                    )
                )
        f = art["arm_F_fixed_point"]
        for row in f["per_grid_point"]:
            out.append(
                _entry(
                    69,
                    name,
                    "F",
                    row["ambient_temperature_celsius"],
                    row["ambient_relative_humidity"],
                    "A",
                    f["inlet_temperature_celsius"],
                    f["gas_velocity_m_per_s"],
                    f["nozzle_pressure_bar"],
                    stored_bed=row["bed_temperature_celsius"],
                )
            )
    for row in _load("m1_anchor_s05")["results"]:
        out.append(
            _entry(
                69,
                "m1_anchor_s05",
                "anchor",
                row["ambient_temperature_celsius"],
                row["ambient_relative_humidity"],
                "A",
                row["inlet_temperature_celsius"],
                row["gas_velocity_m_per_s"],
                row["nozzle_pressure_bar"],
                stored_bed=row["bed_temperature_celsius"],
            )
        )
    for row in _load("m3_line_b_optimum_umin126")["grid_results"]:
        out.append(
            _entry(
                73,
                "m3_line_b_optimum_umin126",
                "line B alone",
                row["ambient_temperature_celsius"],
                row["ambient_relative_humidity"],
                "B",
                row["inlet_temperature_celsius"],
                row["gas_velocity_m_per_s"],
                row["nozzle_pressure_bar"],
                stored_bed=row["bed_temperature_celsius"],
            )
        )
    for case in ("control", "c0.5", "c1.0", "c1.5"):
        name = f"m3_joint_{case}_umin126"
        for s in _load(name)["scenarios"]:
            for arm, rec in (("S2", s["s2"]), ("M3'", s["search"]["m3"])):
                out += _joint_entries(
                    73,
                    name,
                    f"{arm} scenario {s['grid_index']}",
                    s["t_amb"],
                    s["phi"],
                    rec["set_points"],
                    (rec.get("moisture_a"), rec.get("moisture_b")),
                )
    for s in _load("m3_joint_seeds_umin126")["records"]:
        for arm, rec in (("S2", s["s2"]), ("M3'", s["search"]["m3"])):
            out += _joint_entries(
                73,
                "m3_joint_seeds_umin126",
                f"{arm} scenario {s['grid_index']} {s['case']} seed {s['seed']}",
                s["t_amb"],
                s["phi"],
                rec["set_points"],
                (rec.get("moisture_a"), rec.get("moisture_b")),
            )
    grid = {i: (t, p) for i, (t, p) in enumerate(m3.build_grid(_STATE["configs"]["A"], 5))}
    for r in _load("m3_check_umin126")["results"]:
        t, p = grid[r["grid_index"]]
        d = r["detail"]
        out += _joint_entries(
            73,
            "m3_check_umin126",
            f"{r['kind']} scenario {r['grid_index']} {r['case']}",
            t,
            p,
            d["set_points"],
            (d.get("moisture_a"), d.get("moisture_b")),
        )
    for name in (
        "m3x_confirm_evaluation_umin126",
        "m3x_confirm_s18_19_evaluation_umin126",
        "m3x_sens_tmax100_evaluation_umin126",
        "m3x_robust_rest_evaluation_umin126",
    ):
        for sid, s in _load(name)["scenarios"].items():
            t, p = s["ambient"]
            out += _joint_entries(76, name, f"S2 scenario {sid}", t, p, s["best_s2"]["set_points"])
            for seed, rec in s["per_seed_m3"].items():
                out += _joint_entries(
                    76,
                    name,
                    f"M3' scenario {sid} seed {seed}",
                    t,
                    p,
                    rec["set_points"],
                    (rec.get("moisture_a"), rec.get("moisture_b")),
                )
    for name in ("m4c_confirm_lab_umin126", "m4c_confirm_box_umin126"):
        for s in _load(name)["scenarios"].values():
            tag = f"{s['group']} ({s['t_amb']}; {s['phi']})"
            out += _joint_entries(
                77, name, f"S2 {tag}", s["t_amb"], s["phi"], s["best_s2"]["set_points"]
            )
            for seed, rec in s["per_seed_m3"].items():
                out += _joint_entries(
                    77,
                    name,
                    f"M3' {tag} seed {seed}",
                    s["t_amb"],
                    s["phi"],
                    rec["set_points"],
                    (rec.get("moisture_a"), rec.get("moisture_b")),
                )
    return out


# ------------------------------------------------------------------------------------- solving
def _init(configs: dict[str, Any]) -> None:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    _STATE["configs"] = configs


def _solve(
    line: str,
    t_in: float,
    u_gas: float,
    p_nozzle: float,
    t_amb: float,
    phi: float,
    lower_edge_k: float | None,
) -> dict[str, Any]:
    """Steady state with the default bracket (``lower_edge_k`` None) or a wider one; regime and line CO2."""
    base = _STATE["configs"][line]
    config = build_variant_config(
        base, t_in=t_in, u_gas=u_gas, p_nozzle=p_nozzle, t_amb=t_amb, phi_amb=phi
    )
    simulator = GreyBoxFluidBedGranulator(config)
    bracket = None
    if lower_edge_k is not None:
        upper = max(t_in, config.operation.suspension_temperature_celsius) + 5.0
        bracket = (t_amb + lower_edge_k, upper)
    try:
        solution = solve_steady_state(
            simulator,
            max_iterations=DEFAULT_SEARCH_MAX_ITERATIONS,
            tolerance=DEFAULT_SEARCH_TOLERANCE,
            temperature_bracket_celsius=bracket,
        )
    except (RuntimeError, ValueError) as error:
        return {"ok": False, "error": str(error)[:200]}
    d = solution.diagnostics
    t_bed = solution.bed_temperature_celsius
    y_s = float(saturation_humidity_kg_per_kg(_BACKEND, config.gas, _BACKEND.asarray(t_bed)))
    y_in, y_out = float(d["inlet_humidity_kg_per_kg"]), solution.outlet_humidity_kg_per_kg
    co2 = carrier_energy_from_diagnostics(
        d,
        heater="gas",
        process_heater_efficiency=config.energy.heater_efficiency,
        gas_heater_efficiency=GAS_HEATER_EFFICIENCY,
    ).co2_kg_per_h
    return {
        "ok": True,
        "bed_temperature_celsius": t_bed,
        "bed_minus_ambient_k": t_bed - t_amb,
        "moisture_kg_per_kg": solution.particle_moisture_kg_per_kg,
        "inlet_humidity_kg_per_kg": y_in,
        "outlet_humidity_kg_per_kg": y_out,
        "saturation_humidity_at_bed_kg_per_kg": y_s,
        "drying_regime": bool(y_in < y_out < y_s),
        "residual_inf_norm": solution.residual_inf_norm,
        "solver_feasible": solution.feasible,
        "infeasibility_reasons": list(solution.infeasibility_reasons),
        "below_tightened_limit": bool(
            solution.particle_moisture_kg_per_kg <= TIGHTENED_LIMIT_KG_PER_KG
        ),
        "line_co2_gas_heater_kg_per_h": co2,
    }


def _bed_task(key: tuple[str, float, float, float, float, float]) -> tuple[Any, dict[str, Any]]:
    return key, _solve(*key, None)


def _probe_task(key: tuple[str, float, float, float, float, float]) -> dict[str, Any]:
    line, t_in, u_gas, p_nozzle, t_amb, phi = key
    probes = []
    for d_in in INLET_PROBES_K:
        t_probe = t_in + d_in
        if t_probe < BOX_INLET_MINIMUM_CELSIUS - 1e-9:
            # the process config admits inlet temperatures only inside the box [60, 120] degC
            probes.append(
                {
                    "inlet_change_k": d_in,
                    "inlet_temperature_celsius": t_probe,
                    "inside_box": False,
                    "evaluated": False,
                }
            )
            continue
        row: dict[str, Any] = {
            "inlet_change_k": d_in,
            "inlet_temperature_celsius": t_probe,
            "inside_box": True,
            "evaluated": True,
            "default_bracket": _solve(line, t_probe, u_gas, p_nozzle, t_amb, phi, None),
            "wider_brackets": {
                f"T_amb{edge:+g}K": _solve(line, t_probe, u_gas, p_nozzle, t_amb, phi, edge)
                for edge in WIDER_LOWER_EDGES_K
            },
        }
        found = [v["bed_temperature_celsius"] for v in row["wider_brackets"].values() if v["ok"]]
        row["wider_brackets_agree_k"] = float(max(found) - min(found)) if len(found) > 1 else None
        probes.append(row)
    return {
        "line": line,
        "set_point": [t_in, u_gas, p_nozzle],
        "t_amb": t_amb,
        "phi": phi,
        "probes": probes,
    }


def main() -> int:
    require_clean_tree()
    assert Path(twinlink_ml.__file__).resolve().is_relative_to(REPO_ROOT)
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    configs, _, _, _ = m3._setup()
    _init(configs)
    start = time.perf_counter()
    entries = collect()
    keys = sorted(
        {(e["line"], *e["set_point"], e["t_amb"], e["phi"]) for e in entries},
        key=lambda k: (k[4], k[5], k[0], k[1], k[2], k[3]),
    )
    with Pool(cpu_count(), initializer=_init, initargs=(configs,)) as pool:
        solved = dict(pool.map(_bed_task, keys, chunksize=4))
        failed = [k for k, v in solved.items() if not v["ok"]]
        if failed:
            raise SystemExit(
                f"{len(failed)} stored solutions do not solve with the default bracket: {failed[:3]}"
            )
        reproduction_bed = 0.0
        reproduction_moisture = 0.0
        for e in entries:
            res = solved[(e["line"], *e["set_point"], e["t_amb"], e["phi"])]
            e["bed_temperature_celsius"] = res["bed_temperature_celsius"]
            e["distance_to_bound_k"] = (
                res["bed_temperature_celsius"] - e["t_amb"] - BRACKET_MARGIN_K
            )
            e["on_bound"] = bool(abs(e["distance_to_bound_k"]) <= ON_BOUND_TOLERANCE_K)
            e["moisture_kg_per_kg"] = res["moisture_kg_per_kg"]
            e["drying_regime"] = res["drying_regime"]
            if e["stored_bed_temperature_celsius"] is not None:
                reproduction_bed = max(
                    reproduction_bed,
                    abs(res["bed_temperature_celsius"] - e["stored_bed_temperature_celsius"]),
                )
            if e["stored_moisture_kg_per_kg"] is not None:
                reproduction_moisture = max(
                    reproduction_moisture,
                    abs(res["moisture_kg_per_kg"] - e["stored_moisture_kg_per_kg"]),
                )
        if (
            reproduction_bed > REPRODUCTION_TOLERANCE_K
            or reproduction_moisture > REPRODUCTION_TOLERANCE_MOISTURE
        ):
            raise SystemExit(
                f"reproduction failed: bed {reproduction_bed:.3e} K, moisture {reproduction_moisture:.3e}"
            )
        pinned_keys = sorted(
            {(e["line"], *e["set_point"], e["t_amb"], e["phi"]) for e in entries if e["on_bound"]},
            key=lambda k: (k[4], k[5], k[0], k[1]),
        )
        probes = pool.map(_probe_task, pinned_keys, chunksize=1)

    by_artifact: dict[str, Any] = {}
    for e in entries:
        row = by_artifact.setdefault(
            e["artifact"],
            {"befund": e["befund"], "n_solutions": 0, "n_on_bound": 0, "on_bound": []},
        )
        row["n_solutions"] += 1
        if e["on_bound"]:
            row["n_on_bound"] += 1
            row["on_bound"].append(f"{e['label']} | ({e['t_amb']}; {e['phi']}) | line {e['line']}")
    distances = np.array([e["distance_to_bound_k"] for e in entries])
    write_artifact(
        ARTIFACT,
        {
            "_provenance": provenance(),
            "purpose": "check of review finding K1 (2026-10-09) on stored solutions; no correction, no re-optimisation",
            "bound": "T_bed >= T_amb + 10 K (default lower edge of the outer bisection, steady_state.py)",
            "on_bound_tolerance_k": ON_BOUND_TOLERANCE_K,
            "precision": {
                "max_iterations": DEFAULT_SEARCH_MAX_ITERATIONS,
                "tolerance": DEFAULT_SEARCH_TOLERANCE,
            },
            "reproduction_max_abs": {
                "bed_temperature_k": reproduction_bed,
                "moisture_kg_per_kg": reproduction_moisture,
            },
            "regime_rule": "drying iff Y_in < Y_out < Y_s(T_bed)",
            "wider_lower_edges_k": list(WIDER_LOWER_EDGES_K),
            "inlet_probes_k": list(INLET_PROBES_K),
            "tightened_limit_kg_per_kg": TIGHTENED_LIMIT_KG_PER_KG,
            "line_co2": "line alone, gas heater, eta_gas 0.80 (carrier_energy), lab scale, no network",
            "n_solutions": len(entries),
            "n_distinct_line_solves": len(keys),
            "n_on_bound": int(sum(e["on_bound"] for e in entries)),
            "n_distinct_on_bound": len(pinned_keys),
            "distance_counts": {
                "abs_le_1e-6_k": int(np.sum(np.abs(distances) <= 1e-6)),
                "abs_le_1e-3_k": int(np.sum(np.abs(distances) <= 1e-3)),
                "between_1e-3_and_0.1_k": int(
                    np.sum((np.abs(distances) > 1e-3) & (np.abs(distances) <= 0.1))
                ),
                "min_distance_k": float(distances.min()),
            },
            "by_artifact": by_artifact,
            "wall_s": time.perf_counter() - start,
            "solutions": entries,
            "probes": probes,
        },
    )
    print(
        f"{sum(e['on_bound'] for e in entries)} of {len(entries)} solutions on the bound; {len(pinned_keys)} distinct"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

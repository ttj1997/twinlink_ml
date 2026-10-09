r"""Inventory for review finding A1 (Befund 78): every Befund that uses the heat-network cascade, re-evaluated on its
stored solutions with the approach temperature checked along the whole exchanger. FIXED BEFORE THE RUN.

Decision of the project lead (2026-10-09): compute only. Nothing in ``src/`` changes, nothing is re-optimised. For
every Befund whose reported figures come from the cascade of ``heatnetwork/balance.py`` (``evaluate_cascade``),
from ``evaluation/site_network.py``, or from a copy of the same algorithm, the reported figures are recomputed at
the STORED set points with three cascades:

* ``as_run_20K`` (and ``as_run_10K`` where the Befund reported a 10 K sensitivity): the check at the warm end of a
  stage on the model profile, i.e. the cascade as it ran; it must reproduce the stored figures (asserted);
* ``along_20K`` and ``along_10K``: ``dT_min`` enforced as the minimum along the whole exchanger, on the physical
  profile with condensation from the dew point on (the cascade of ``scripts/check_review_a1_cascade_approach.py``,
  whose functions ``SourceProfile`` and ``max_duty_along`` are reused unchanged, here generalised to a source-own
  cold-side temperature, the sensible recovery mode and the shared exchanger limit of E1).

Interconnections: where the Befund chose the best interconnection at given set points (S2, F best, S, the grid of
stage 0), the best is chosen anew under each cascade; where it reported a fixed interconnection, it is held (and,
for the joint optima M and M3', both variants are given: ``fixed`` and ``best`` of the enumeration). Set points are
never changed. Figures that rest on a JOINT optimum (M of Befund 70, M3' of Befunde 73, 76, 77) are estimates at
held set points and need a re-optimisation to become results; figures at network-independent set points (F, S of
Befund 70; S2 of Befunde 73-77; the six cases of Befunde 57 and 60; LB2 of Befund 69 b; the grid of Befund 71) are
exact under the corrected check.

Befunde and figures (artifact -> figures):

* 57 d (``probe_six_case_reconciliation.py``, an own copy of the algorithm with a cold-side temperature per source:
  Q1 10 degC, Q2 15 degC): fuel of the six interconnection cases at ``dT_min`` 20 and 10 K, the best case;
* 60 (``run_pinch_benchmark``): fuel of the six cases and the gap of the best case to the pinch target (problem
  table, ``heatnetwork/pinch.py``, which handles ``dT_min`` correctly and is recomputed unchanged) over
  ``T_kalt`` 15 / 25 / 29.1 / 35 degC and ``dT_min`` 10 / 20 K;
* 69 b (``m0_line_b_network``): LB2, median relative best-to-worst difference of the site CO2 over the 343
  interconnections and the spread of the interconnections of the two lines alone, per spray rate of line B;
* 70 (``m2_site_network_l015_umin126``, ``..._umin080``): routing effect F -> F best, set-point effect
  F best -> S, joint effect S -> M, total F -> M (site CO2, absolute and relative; line CO2 relative), medians;
  the best interconnection;
* 71 (``m3_interaction_check``): largest mixed difference of stage 0 and the number of values at or above the
  resolution 0.5 kg/h (rule of ``coupling_gain.interaction_verdict``), limits held at their stored values;
* 73 (``m3_joint_{control,c0.5,c1.0,c1.5}_umin126``, ``m3_joint_seeds_umin126``): G per scenario, the verdict of
  ``coupling_gain_verdict`` (median, seed spread from the seed check, share above 0.2 kg/h) per case;
* 76 (``m3x_confirm_umin126``, ``m3x_confirm_s18_19_umin126``, ``m3x_robust_rest_umin126``,
  ``m3x_sens_tmax100_umin126``): G over the five seeds per protected scenario at the boxes 60-120 and 60-100 degC,
  the re-measured verdict over the 25 scenarios;
* 77 (``m4c_confirm_lab_umin126``, ``m4c_confirm_box_umin126``): K_lab, K_box (G over the seeds of W), N, H.

Befunde that do NOT use the stage check and are not computed here: 52 (text only), 57 c / MA4 (``net_energy``,
review finding A2), 57 h (available heat per source, no stages), 61 (pinch only, suspended), 72 / 74
(``partner_v4``, an own module), 75 (single-stream formula).

Second run (artifact ``review_a1_inventory_2``): the first artifact ``review_a1_inventory`` (commit ``9e4d07a``)
differs only in the relative figures of the summary of Befund 70, which it divided by F for every effect instead of
by the starting arm of each effect as ``m2_site_network`` does; every other figure and every raw value is the same
computation.

    PYTHONPATH=src python scripts/check_review_a1_inventory.py
"""

from __future__ import annotations

import json
import math
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import analyze_m0_line_b_network as lb  # noqa: E402
import check_review_a1_cascade_approach as a1  # noqa: E402
import probe_six_case_reconciliation as p57  # noqa: E402
import run_m3_joint as m3  # noqa: E402
import run_pinch_benchmark as p60  # noqa: E402
from _common import (  # noqa: E402
    REPO_ROOT,
    RUNS_DIR,
    provenance,
    require_clean_tree,
    write_artifact,
)

import twinlink_ml  # noqa: E402
from twinlink_ml.evaluation import site_network as sn  # noqa: E402
from twinlink_ml.evaluation.coupling_gain import (  # noqa: E402
    coupling_gain_verdict,
    interaction_verdict,
)
from twinlink_ml.evaluation.net_energy import (  # noqa: E402
    ELECTRICITY_CO2_KG_PER_KWH,
    GAS_CO2_KG_PER_KWH,
)
from twinlink_ml.evaluation.site_oracle import SiteOracleSpec, penalty_weight  # noqa: E402
from twinlink_ml.heatnetwork.config import (  # noqa: E402
    HeatNetworkConfig,
    HeatSinkConfig,
    HeatSourceConfig,
)
from twinlink_ml.heatnetwork.pinch import evaluate_pinch_target  # noqa: E402
from twinlink_ml.heatnetwork.routing import FixedRouting  # noqa: E402
from twinlink_ml.simulation import PlaceholderConfigWarning  # noqa: E402
from twinlink_ml.simulation.backend import NumpyBackend  # noqa: E402
from twinlink_ml.simulation.operating_point import check_constraints  # noqa: E402
from twinlink_ml.simulation.oracle_search import evaluate_point  # noqa: E402
from twinlink_ml.simulation.thermo import (  # noqa: E402
    dew_point_celsius,
    latent_heat_j_per_kg,
    saturation_humidity_kg_per_kg,
)

ARTIFACT = "review_a1_inventory_2"
SITE_MODES: tuple[tuple[str, str, float], ...] = (
    ("as_run_20K", "warm_end", 20.0),
    ("along_20K", "along", 20.0),
    ("along_10K", "along", 10.0),
)
PARTNER_MODES: tuple[tuple[str, str, float], ...] = (
    ("as_run_20K", "warm_end", 20.0),
    ("as_run_10K", "warm_end", 10.0),
    ("along_20K", "along", 20.0),
    ("along_10K", "along", 10.0),
)
REPRODUCTION_TOLERANCE = 1.0e-6
PRECISION = {"max_iterations": 30, "tolerance": 1.0e-7}
_BACKEND = NumpyBackend()
_STATE: dict[str, Any] = {}

Allocation = tuple[tuple[str, str], ...]


# ------------------------------------------------------------------------------------- cascade
def profiles(
    network: HeatNetworkConfig, kind: str, cold_by_source: dict[str, float] | None = None
) -> dict[str, a1.SourceProfile]:
    """Model or physical profile of every source; ``cold_by_source`` overrides ``T_kalt`` per source."""
    gas = network.gas
    out = {}
    for source in network.sources:
        m = source.dry_air_mass_flow_kg_per_s
        y = source.outlet_humidity_kg_per_kg
        cap = m * (gas.cp_dry_air_j_per_kg_k + y * gas.cp_vapour_j_per_kg_k)
        t_in = source.outlet_temperature_celsius
        if network.recovery_mode == "condensing":
            floor = (cold_by_source or {}).get(source.name, network.cold_temperature_celsius)
            latent = float(latent_heat_j_per_kg(_BACKEND, _BACKEND.asarray(floor)))
            y_s_floor = float(saturation_humidity_kg_per_kg(_BACKEND, gas, _BACKEND.asarray(floor)))
            q_lat = max(y - y_s_floor, 0.0) * m * latent
        else:
            floor = (
                float(dew_point_celsius(_BACKEND, gas, _BACKEND.asarray(y)))
                + network.dew_point_approach_k
            )
            latent, q_lat = 0.0, 0.0
        if t_in <= floor:
            out[source.name] = a1.SourceProfile(
                source.name, cap, t_in, np.array([0.0, 0.0]), np.array([t_in, t_in])
            )
            continue
        if kind == "model" or q_lat == 0.0:
            q = [0.0, cap * (t_in - floor)] + (
                [cap * (t_in - floor) + q_lat] if q_lat > 0.0 else []
            )
            t = [t_in, floor] + ([floor] if q_lat > 0.0 else [])
            q_arr, t_arr = np.asarray(q), np.asarray(t)
        else:
            n = math.ceil((t_in - floor) / a1.PROFILE_STEP_K) + 1
            t_arr = np.linspace(t_in, floor, n)
            y_s = np.asarray(saturation_humidity_kg_per_kg(_BACKEND, gas, _BACKEND.asarray(t_arr)))
            q_arr = cap * (t_in - t_arr) + m * np.maximum(y - y_s, 0.0) * latent
        out[source.name] = a1.SourceProfile(source.name, cap, t_in, q_arr, t_arr)
    return out


def cascade(
    network: HeatNetworkConfig,
    stages: Allocation,
    profs: dict[str, a1.SourceProfile],
    check: str,
    dt_min: float,
    cap: tuple[frozenset[tuple[str, str]], float] | None = None,
) -> dict[str, float]:
    """Stages in the given order; returns the fuel power [kW] per sink (duty after recovery / efficiency)."""
    gas = network.gas
    sinks = {s.name: s for s in network.sinks}
    c_snk = {
        s.name: s.dry_air_mass_flow_kg_per_s
        * (gas.cp_dry_air_j_per_kg_k + s.humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k)
        for s in network.sinks
    }
    t_snk = {s.name: float(s.intake_temperature_celsius) for s in network.sinks}
    used = dict.fromkeys(profs, 0.0)
    cap_left = cap[1] * 1000.0 if cap is not None else 0.0
    for source_name, sink_name in stages:
        prof = profs[source_name]
        q_before = used[source_name]
        q_set = max(
            c_snk[sink_name] * (sinks[sink_name].setpoint_temperature_celsius - t_snk[sink_name]),
            0.0,
        )
        upper = min(q_set, max(prof.total_w - q_before, 0.0))
        if check == "warm_end":
            q = min(
                upper,
                max(
                    c_snk[sink_name] * (prof.temperature_at(q_before) - dt_min - t_snk[sink_name]),
                    0.0,
                ),
            )
        else:
            q = a1.max_duty_along(prof, q_before, t_snk[sink_name], c_snk[sink_name], dt_min, upper)
        q = max(q, 0.0)
        if cap is not None and (source_name, sink_name) in cap[0]:
            q = min(q, cap_left)
            cap_left -= q
        used[source_name] = q_before + q
        t_snk[sink_name] += q / c_snk[sink_name]
    return {
        name: max(c_snk[name] * (sinks[name].setpoint_temperature_celsius - t_snk[name]), 0.0)
        / network.heater_efficiency
        / 1000.0
        for name in sinks
    }


# ------------------------------------------------------------------------------------- site
@dataclass(frozen=True, slots=True)
class Site:
    """One set of line states in the scaled site: network, both profiles, routes per interconnection."""

    lines: tuple[Any, ...]
    network: HeatNetworkConfig
    model: dict[str, a1.SourceProfile]
    physical: dict[str, a1.SourceProfile]
    routes: dict[Allocation, Allocation]


def site(lines: tuple[Any, ...], allocations: list[Allocation]) -> Site:
    k = _STATE["k"]
    network, _, _ = sn._network(lines, sn.SiteSpec(k))
    return Site(
        lines,
        network,
        profiles(network, "model"),
        profiles(network, "physical"),
        {a: FixedRouting(a).route(network) for a in allocations},
    )


def site_co2(
    s: Site, allocation: Allocation, mode: tuple[str, str, float], cap_kw: float | None = None
) -> dict[str, float]:
    _, check, dt = mode
    cap = None
    if cap_kw is not None:
        cap = (frozenset(("Q1", f"S{x.name}") for x in s.lines), cap_kw)
    fuel = cascade(
        s.network,
        s.routes[allocation],
        s.model if check == "warm_end" else s.physical,
        check,
        dt,
        cap,
    )
    electric = _STATE["k"] * sum(x.electric_kw for x in s.lines)
    lines_fuel = sum(v for name, v in fuel.items() if name != "S1")
    return {
        "site": ELECTRICITY_CO2_KG_PER_KWH * electric + GAS_CO2_KG_PER_KWH * sum(fuel.values()),
        "lines": ELECTRICITY_CO2_KG_PER_KWH * electric + GAS_CO2_KG_PER_KWH * lines_fuel,
    }


def best_site(
    s: Site, allocations: list[Allocation], mode: Any, cap_kw: float | None = None
) -> tuple[float, Allocation]:
    values = [site_co2(s, a, mode, cap_kw)["site"] for a in allocations]
    i = int(np.argmin(values))
    return values[i], allocations[i]


def _solve_line(
    name: str, t_in: float, u: float, p: float, t_amb: float, phi: float
) -> tuple[Any, float]:
    """Line state and the normalised violation of the tightened limit (``None`` on a solver failure)."""
    key = (name, t_in, u, p, t_amb, phi)
    cache = _STATE["lines"]
    if key not in cache:
        config = _STATE["configs"][name]
        try:
            solution = evaluate_point(t_in, u, p, t_amb, phi, base_config=config, **PRECISION)
            check = check_constraints(
                solution, critical_moisture_kg_per_kg=m3.THRESHOLD, gas_velocity_m_per_s=u
            )
            cache[key] = (
                sn.line_state(name, solution, config, t_amb=t_amb, phi_amb=phi, t_in=t_in),
                check.normalized_violation,
            )
        except (RuntimeError, ValueError):
            cache[key] = (None, math.nan)
    return cache[key]


def line(name: str, t_in: float, u: float, p: float, t_amb: float, phi: float) -> Any:
    return _solve_line(name, t_in, u, p, t_amb, phi)[0]


def pair(x: list[float], t_amb: float, phi: float) -> tuple[Any, Any]:
    return line("A", x[0], x[1], 1.0, t_amb, phi), line("B", x[2], x[3], 1.0, t_amb, phi)


def pair_penalty(x: list[float], t_amb: float, phi: float) -> float:
    """Penalty of the site objective at the set points ``x`` (as ``joint_search.evaluate_true``)."""
    violations = (
        _solve_line("A", x[0], x[1], 1.0, t_amb, phi)[1]
        + _solve_line("B", x[2], x[3], 1.0, t_amb, phi)[1]
    )
    return penalty_weight(SiteOracleSpec(m3.THRESHOLD, sn.SiteSpec(_STATE["k"]))) * violations


def _check(old: float, stored: float, what: str) -> None:
    if abs(old - stored) > REPRODUCTION_TOLERANCE:
        raise SystemExit(f"{what}: stored {stored!r} not reproduced ({old!r})")


def _self_check(s: Site, allocations: list[Allocation], cap_kw: float | None = None) -> float:
    """|CO2(script, as run) - CO2(evaluate_cascade)| over the interconnections; must stay below the tolerance."""
    spec = sn.SiteSpec(_STATE["k"], shared_cap_kw=cap_kw)
    reference, rejected = sn.evaluate_site(s.lines, spec, allocations)
    assert rejected == 0
    worst = max(
        abs(site_co2(s, a, SITE_MODES[0], cap_kw)["site"] - r.co2_kg_per_h)
        for a, r in zip(allocations, reference, strict=True)
    )
    if worst > REPRODUCTION_TOLERANCE:
        raise SystemExit(f"self-check failed: {worst:.3e} kg/h")
    return worst


# ------------------------------------------------------------------------------------- tasks
def _init(configs: dict[str, Any], k: float) -> None:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    _STATE["configs"] = configs
    _STATE["k"] = k
    _STATE["two"] = [tuple(tuple(p) for p in a) for a in sn.all_allocations(2, order_variants=True)]
    _STATE["one"] = [tuple(tuple(p) for p in a) for a in sn.all_allocations(1)]
    _STATE["lines"] = {}


def task_57(_: Any) -> dict[str, Any]:
    network = HeatNetworkConfig(
        sources=tuple(
            HeatSourceConfig(
                s.name,
                s.mass_flow_kg_per_s,
                s.outlet_temperature_celsius,
                s.outlet_humidity_kg_per_kg,
            )
            for s in p57.SOURCES.values()
        ),
        sinks=tuple(
            HeatSinkConfig(
                s.name,
                s.mass_flow_kg_per_s,
                s.humidity_kg_per_kg,
                s.intake_temperature_celsius,
                s.setpoint_temperature_celsius,
            )
            for s in p57.SINKS.values()
        ),
        minimum_pinch_k=20.0,
        recovery_mode="condensing",
        cold_temperature_celsius=15.0,
        fuel_lower_heating_value_j_per_kg=p57.FUEL_LOWER_HEATING_VALUE_J_PER_KG,
        heater_efficiency=p57.HEATER_EFFICIENCY,
    )
    cold = {name: s.cold_temperature_celsius for name, s in p57.SOURCES.items()}
    profs = {
        "model": profiles(network, "model", cold),
        "physical": profiles(network, "physical", cold),
    }
    out: dict[str, Any] = {}
    for key, check, dt in PARTNER_MODES:
        fuel = {}
        for case, stages in p57.CASES.items():
            kw = sum(
                cascade(
                    network,
                    stages,
                    profs["model" if check == "warm_end" else "physical"],
                    check,
                    dt,
                ).values()
            )
            fuel[case] = kw * 1000.0 / p57.FUEL_LOWER_HEATING_VALUE_J_PER_KG * 3600.0
            if check == "warm_end":
                _check(
                    fuel[case],
                    p57.evaluate_case(stages, minimum_pinch_k=dt)["total_fuel_kg_per_h"],
                    f"57d {case} {key}",
                )
        out[key] = {"fuel_kg_per_h": fuel, "best_case": min(fuel, key=fuel.get)}
    return {"befund": "57d", "result": out}


def task_60(_: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for t_kalt in p60.COLD_TEMPERATURES_CELSIUS:
        for dt in (10.0, 20.0):
            network = HeatNetworkConfig(
                sources=(p60.SOURCE_1, p60.SOURCE_2),
                sinks=(p60.SINK_1, p60.SINK_2),
                minimum_pinch_k=dt,
                recovery_mode="condensing",
                cold_temperature_celsius=t_kalt,
                fuel_lower_heating_value_j_per_kg=p60.FUEL_LOWER_HEATING_VALUE_J_PER_KG,
                heater_efficiency=p60.HEATER_EFFICIENCY,
            )
            pinch, _, _ = evaluate_pinch_target(
                (p60.SOURCE_1, p60.SOURCE_2),
                (p60.SINK_1, p60.SINK_2),
                p60.GAS,
                cold_temperature_celsius=t_kalt,
                minimum_pinch_k=dt,
                n_segments_below_dew_point=p60.N_SEGMENTS_BELOW_DEW_POINT,
            )
            target = p60._kw_to_fuel_kg_per_h(pinch.minimum_hot_utility_kw)
            reference = p60._six_case_fuel_kg_per_h(
                cold_temperature_celsius=t_kalt, minimum_pinch_k=dt
            )
            row: dict[str, Any] = {"pinch_target_kg_per_h": target}
            for check in ("warm_end", "along"):
                profs = profiles(network, "model" if check == "warm_end" else "physical")
                fuel = {
                    case: sum(cascade(network, stages, profs, check, dt).values())
                    * 1000.0
                    / p60.FUEL_LOWER_HEATING_VALUE_J_PER_KG
                    * 3600.0
                    for case, stages in p60.CASES.items()
                }
                if check == "warm_end":
                    for case in fuel:
                        _check(fuel[case], reference[case], f"60 {case} {t_kalt} {dt}")
                best = min(fuel, key=fuel.get)
                row["as_run" if check == "warm_end" else "along"] = {
                    "fuel_kg_per_h": fuel,
                    "best_case": best,
                    "gap_of_best_to_pinch": fuel[best] / target - 1.0,
                }
            out[f"T_kalt {t_kalt:g} / dT_min {dt:g}"] = row
    return {"befund": "60", "result": out}


def task_69b(spray: float) -> dict[str, Any]:
    stored = json.loads((RUNS_DIR / "m0_line_b_network.json").read_text("utf-8"))
    source = json.loads((RUNS_DIR / f"{lb.SOURCE_ARTIFACT}.json").read_text("utf-8"))
    base = _STATE["configs"]["base"]
    states_a = lb._states(base, source["arms"][str(lb.LINE_A_SPRAY)], lb.LINE_A_SPRAY)
    states_b = lb._states(base, source["arms"][str(spray)], spray)
    allocations = lb._allocations()
    k = stored["scale_factor_k"]
    out: dict[str, Any] = {}
    for key, check, dt in SITE_MODES:
        per = []
        for a, b in zip(states_a, states_b, strict=True):
            sources = (
                lb.partner.SOURCE_1,
                HeatSourceConfig("LA", k * a["m_dry"], a["t_out"], a["y_out"]),
                HeatSourceConfig("LB", k * b["m_dry"], b["t_out"], b["y_out"]),
            )
            sinks = (
                HeatSinkConfig("S1", 20.0, 0.005, 15.0, 300.0),
                HeatSinkConfig("SA", k * a["m_dry"], a["y_in"], a["t_amb"], a["t_in"]),
                HeatSinkConfig("SB", k * b["m_dry"], b["y_in"], b["t_amb"], b["t_in"]),
            )
            network = HeatNetworkConfig(
                sources=sources,
                sinks=sinks,
                minimum_pinch_k=lb.DT_MIN_K,
                recovery_mode="condensing",
                cold_temperature_celsius=lb.T_KALT_C,
                fuel_lower_heating_value_j_per_kg=lb.H_U_J_PER_KG,
                heater_efficiency=lb.ETA_H,
            )
            profs = profiles(network, "model" if check == "warm_end" else "physical")
            electric = k * (a["electric_kw"] + b["electric_kw"])
            co2, lines_only, q1_sets = [], [], []
            for allocation in allocations:
                fuel = cascade(network, FixedRouting(allocation).route(network), profs, check, dt)
                co2.append(
                    ELECTRICITY_CO2_KG_PER_KWH * electric + GAS_CO2_KG_PER_KWH * sum(fuel.values())
                )
                lines_only.append(
                    ELECTRICITY_CO2_KG_PER_KWH * electric
                    + GAS_CO2_KG_PER_KWH * (fuel["SA"] + fuel["SB"])
                )
                q1_sets.append(tuple(c for c in allocation if c[0] == "Q1"))
            co2_arr = np.asarray(co2)
            i_best, i_worst = int(np.argmin(co2_arr)), int(np.argmax(co2_arr))
            fixed = co2_arr[[q == q1_sets[i_best] for q in q1_sets]]
            per.append(
                {
                    "relative_difference_to_best": float(
                        (co2_arr[i_worst] - co2_arr[i_best]) / co2_arr[i_best]
                    ),
                    "relative_difference_lines_only_allocations": float(
                        (fixed.max() - fixed.min()) / fixed.min()
                    ),
                    "best_connections": [list(c) for c in allocations[i_best]],
                }
            )
        figures = {
            "median_relative_difference_to_best": float(
                np.median([p["relative_difference_to_best"] for p in per])
            ),
            "median_relative_difference_lines_only_allocations": float(
                np.median([p["relative_difference_lines_only_allocations"] for p in per])
            ),
            "lb2_met": bool(
                np.median([p["relative_difference_to_best"] for p in per]) >= lb.LB2_THRESHOLD
            ),
            "n_distinct_best": len({json.dumps(p["best_connections"]) for p in per}),
        }
        if check == "warm_end":
            ref = stored["lb2"][str(spray)]
            for name in (
                "median_relative_difference_to_best",
                "median_relative_difference_lines_only_allocations",
            ):
                _check(figures[name], ref[name], f"69b {spray} {name}")
        out[key] = figures
    return {"befund": "69b", "spray": spray, "result": out}


def task_70(args: tuple[str, int]) -> dict[str, Any]:
    name, index = args
    art = json.loads((RUNS_DIR / f"{name}.json").read_text("utf-8"))
    point = art["per_point"][index]
    t_amb, phi = point["ambient_temperature_celsius"], point["ambient_relative_humidity"]
    one = _STATE["one"]
    states = {arm: (line("A", *point["setpoints"][arm], t_amb, phi),) for arm in ("F", "S", "M")}
    sites = {arm: site(states[arm], one) for arm in states}
    _self_check(sites["F"], one)
    fixed = sn.fixed_allocation(1)
    all_to_all = sn.all_to_all_allocation(1)
    out: dict[str, Any] = {}
    for mode in SITE_MODES:
        values: dict[str, dict[str, float]] = {}
        values["F"] = site_co2(sites["F"], fixed, mode)
        values["F+R"] = site_co2(sites["F"], all_to_all, mode)
        for arm, label in (("F", "F_best"), ("S", "S"), ("M", "M")):
            _, best = best_site(sites[arm], one, mode)
            values[label] = site_co2(sites[arm], best, mode)
            values[label]["best"] = [list(c) for c in best]
        if mode[0] == "as_run_20K":
            for label in ("F", "F+R", "F_best", "S", "M"):
                _check(
                    values[label]["site"], point["site_co2"][label], f"70 {name} {index} {label}"
                )
                _check(
                    values[label]["lines"],
                    point["line_co2"][label],
                    f"70 {name} {index} {label} line",
                )
        out[mode[0]] = values
    return {"befund": "70", "artifact": name, "index": index, "result": out}


def task_71(args: tuple[str, int]) -> dict[str, Any]:
    bound, index = args
    art = json.loads((RUNS_DIR / "m3_interaction_check.json").read_text("utf-8"))
    scenario = art["per_bound"][bound]["scenarios"][index]
    u = art["per_bound"][bound]["gas_velocity_lower_bound_m_per_s"]
    t_amb, phi = scenario["t_amb"], scenario["phi"]
    grid = art["t_grid_celsius"]
    two = _STATE["two"]
    a_states = [line("A", t, u, 1.0, t_amb, phi) for t in grid]
    b_states = [line("B", t, u, 1.0, t_amb, phi) for t in grid]
    sites = [
        [None if a is None or b is None else site((a, b), two) for b in b_states] for a in a_states
    ]
    centre = sites[1][1]
    if centre is not None:
        _self_check(centre, two, scenario["cases"]["c1.0"]["cap_kw"])
    out: dict[str, Any] = {}
    for case, entry in scenario["cases"].items():
        out[case] = {}
        for mode in SITE_MODES:
            f = [
                [None if s is None else best_site(s, two, mode, entry["cap_kw"])[0] for s in row]
                for row in sites
            ]
            if mode[0] == "as_run_20K":
                for i in range(3):
                    for j in range(3):
                        if entry["site_co2"][i][j] is not None:
                            _check(f[i][j], entry["site_co2"][i][j], f"71 {bound} {index} {case}")
            blocks = []
            for i in range(2):
                for j in range(2):
                    corners = (f[i + 1][j + 1], f[i + 1][j], f[i][j + 1], f[i][j])
                    blocks.append(
                        None
                        if any(v is None for v in corners)
                        else corners[0] - corners[1] - corners[2] + corners[3]
                    )
            valid = [abs(v) for v in blocks if v is not None]
            out[case][mode[0]] = {
                "site_co2": f,
                "max_abs_mixed_difference": max(valid) if valid else None,
            }
    return {"befund": "71", "bound": bound, "index": index, "result": out}


def _joint_values(
    x: list[float], allocation: Allocation, t_amb: float, phi: float, cap_kw: float | None
) -> dict[str, Any]:
    lines = pair(x, t_amb, phi)
    s = site(lines, _STATE["two"])
    out = {}
    for mode in SITE_MODES:
        best, best_alloc = best_site(s, _STATE["two"], mode, cap_kw)
        out[mode[0]] = {"fixed": site_co2(s, allocation, mode, cap_kw)["site"], "best": best}
    return out


def task_73(args: tuple[str, int]) -> dict[str, Any]:
    name, index = args
    art = json.loads((RUNS_DIR / f"{name}.json").read_text("utf-8"))
    rows = art["records"] if name == "m3_joint_seeds_umin126" else art["scenarios"]
    row = rows[index]
    cap = row["cap_kw"]
    t_amb, phi = row["t_amb"], row["phi"]
    s2 = row["s2"]
    m3_ = row["search"]["m3"]
    s2_lines = pair(s2["set_points"], t_amb, phi)
    s2_site = site(s2_lines, _STATE["two"])
    _self_check(s2_site, _STATE["two"], cap)
    m3_alloc = tuple(tuple(c) for c in m3_["allocation"])
    m3v = _joint_values(m3_["set_points"], m3_alloc, t_amb, phi, cap)
    s2_penalty = s2["value"] - s2["co2_kg_per_h"]
    m3_penalty = m3_["value"] - m3_["co2_kg_per_h"]
    out: dict[str, Any] = {}
    for mode in SITE_MODES:
        s2_value = best_site(s2_site, _STATE["two"], mode, cap)[0] + s2_penalty
        m3_fixed = m3v[mode[0]]["fixed"] + m3_penalty
        m3_best = m3v[mode[0]]["best"] + m3_penalty
        out[mode[0]] = {
            "s2": s2_value,
            "m3_fixed": m3_fixed,
            "m3_best": m3_best,
            "gain_fixed": s2_value - m3_fixed,
            "gain_best": s2_value - m3_best,
        }
    _check(out["as_run_20K"]["s2"], row["s2_value"], f"73 {name} {index} S2")
    _check(out["as_run_20K"]["m3_fixed"], row["m3_value"], f"73 {name} {index} M3'")
    return {
        "befund": "73",
        "artifact": name,
        "index": index,
        "grid_index": row["grid_index"],
        "case": row["case"],
        "seed": row["seed"],
        "s2_co2_lines": row["s2_co2_lines_kg_per_h"],
        "stored_gain": row["gain_kg_per_h"],
        "result": out,
    }


def _records_task(
    raw_name: str,
    scenario_key: Any,
    t_amb: float,
    phi: float,
    records: list[dict[str, Any]],
    group: str,
) -> dict[str, Any]:
    two = _STATE["two"]
    lines_by = {"A": {}, "B": {}}
    s2_records = []
    for r in records:
        if r["kind"] == "line":
            lines_by[r["line"]][r["seed"]] = r["set_points"]
        elif r["kind"] == "s2":
            s2_records.append(r)
    if s2_records:  # M4-C stores the S2 pairs
        pairs = [r["set_points"] for r in s2_records]
    else:  # M3x stores the line optima per seed; S2 of a pair at 1.0 bar
        pairs = [
            [a[0], a[1], b[0], b[1]] for a in lines_by["A"].values() for b in lines_by["B"].values()
        ]
    s2_values = {mode[0]: [] for mode in SITE_MODES}
    first = True
    for x in pairs:
        s = site(pair(x, t_amb, phi), two)
        penalty = pair_penalty(x, t_amb, phi)
        if first:
            _self_check(s, two)
            first = False
        for mode in SITE_MODES:
            s2_values[mode[0]].append(best_site(s, two, mode)[0] + penalty)
    joint: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        if r["kind"] != "joint":
            continue
        d = r["detail"]
        v = _joint_values(
            d["set_points"], tuple(tuple(c) for c in d["allocation"]), t_amb, phi, None
        )
        _check(v["as_run_20K"]["fixed"], d["co2_kg_per_h"], f"{raw_name} {scenario_key} joint")
        penalty = r["value"] - d["co2_kg_per_h"]
        joint.setdefault(r["seed"], []).append(
            {m: {k_: x + penalty for k_, x in vv.items()} for m, vv in v.items()}
        )
    out: dict[str, Any] = {}
    for mode in SITE_MODES:
        s2 = min(s2_values[mode[0]])
        per_seed = {
            variant: {
                seed: min(e[mode[0]][variant] for e in entries) for seed, entries in joint.items()
            }
            for variant in ("fixed", "best")
        }
        out[mode[0]] = {
            "best_s2": s2,
            "m3_per_seed": per_seed,
            "gain_per_seed": {
                v: {seed: s2 - m for seed, m in per_seed[v].items()} for v in per_seed
            },
        }
    return {
        "artifact": raw_name,
        "scenario": scenario_key,
        "group": group,
        "t_amb": t_amb,
        "phi": phi,
        "result": out,
    }


def task_76(args: tuple[str, str, int]) -> dict[str, Any]:
    raw_name, eval_name, scenario = args
    raw = json.loads((RUNS_DIR / f"{raw_name}.json").read_text("utf-8"))
    evaluation = json.loads((RUNS_DIR / f"{eval_name}.json").read_text("utf-8"))
    entry = evaluation["scenarios"][str(scenario)]
    t_amb, phi = entry["ambient"]
    records = [r for r in raw["records"] if r["scenario"] == scenario]
    out = _records_task(raw_name, scenario, t_amb, phi, records, "")
    _check(
        out["result"]["as_run_20K"]["best_s2"],
        entry["best_s2"]["value"],
        f"76 {raw_name} {scenario} S2",
    )
    for seed, rec in entry["per_seed_m3"].items():
        _check(
            out["result"]["as_run_20K"]["m3_per_seed"]["fixed"][int(seed)],
            rec["value"],
            f"76 {raw_name} {scenario} M3' {seed}",
        )
    out["befund"] = "76"
    return out


def task_77(args: tuple[str, str]) -> dict[str, Any]:
    name, sid = args
    art = json.loads((RUNS_DIR / f"{name}.json").read_text("utf-8"))
    entry = art["scenarios"][sid]
    records = [
        r for r in art["records"] if r["scenario"] == int(sid) and r["kind"] in ("s2", "joint")
    ]
    out = _records_task(name, sid, entry["t_amb"], entry["phi"], records, entry["group"])
    _check(out["result"]["as_run_20K"]["best_s2"], entry["best_s2"]["value"], f"77 {name} {sid} S2")
    for seed, rec in entry["per_seed_m3"].items():
        _check(
            out["result"]["as_run_20K"]["m3_per_seed"]["fixed"][int(seed)],
            rec["value"],
            f"77 {name} {sid} M3' {seed}",
        )
    out["befund"] = "77"
    return out


def _dispatch(task: tuple[str, Any]) -> dict[str, Any]:
    kind, args = task
    return {
        "57": task_57,
        "60": task_60,
        "69b": task_69b,
        "70": task_70,
        "71": task_71,
        "73": task_73,
        "76": task_76,
        "77": task_77,
    }[kind](args)


# ------------------------------------------------------------------------------------- summaries
def _median(values: list[float]) -> float:
    return float(np.median(values))


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    by = {}
    for r in results:
        by.setdefault(r["befund"], []).append(r)
    summary: dict[str, Any] = {}
    summary["57d"] = by["57d"][0]["result"]
    summary["60"] = by["60"][0]["result"]
    summary["69b"] = {str(r["spray"]): r["result"] for r in by["69b"]}

    s70: dict[str, Any] = {}
    for name in ("m2_site_network_l015_umin126", "m2_site_network_l015_umin080"):
        rows = sorted((r for r in by["70"] if r["artifact"] == name), key=lambda r: r["index"])
        s70[name] = {}
        for mode, _, _ in SITE_MODES:
            v = [r["result"][mode] for r in rows]
            effects = {
                "routing F -> F best": [x["F_best"]["site"] - x["F"]["site"] for x in v],
                "set points F best -> S": [x["S"]["site"] - x["F_best"]["site"] for x in v],
                "joint S -> M": [x["M"]["site"] - x["S"]["site"] for x in v],
                "total F -> M": [x["M"]["site"] - x["F"]["site"] for x in v],
            }
            # relative to the starting arm of each effect, as in ``m2_site_network`` (Befund 70)
            starts = {
                "routing F -> F best": "F",
                "set points F best -> S": "F_best",
                "joint S -> M": "S",
                "total F -> M": "F",
            }
            ends = {
                "routing F -> F best": "F_best",
                "set points F best -> S": "S",
                "joint S -> M": "M",
                "total F -> M": "M",
            }
            rel_site = {
                key: [
                    (x[ends[key]]["site"] - x[starts[key]]["site"]) / x[starts[key]]["site"]
                    for x in v
                ]
                for key in effects
            }
            line_effects = {
                key: [
                    (x[ends[key]]["lines"] - x[starts[key]]["lines"]) / x[starts[key]]["lines"]
                    for x in v
                ]
                for key in effects
            }
            s70[name][mode] = {
                "site_abs_median_kg_per_h": {k_: _median(x) for k_, x in effects.items()},
                "site_rel_median": {k_: _median(x) for k_, x in rel_site.items()},
                "line_rel_median": {k_: _median(x) for k_, x in line_effects.items()},
                "f_plus_r_equals_f_best": int(
                    sum(abs(x["F+R"]["site"] - x["F_best"]["site"]) < 1e-9 for x in v)
                ),
                "best_uses_la_to_sa_s": int(sum(["LA", "SA"] in x["S"]["best"] for x in v)),
                "best_allocations_s": Counter(json.dumps(x["S"]["best"]) for x in v).most_common(),
            }
    summary["70"] = s70

    s71: dict[str, Any] = {}
    for mode, _, _ in SITE_MODES:
        values = []
        per_case: dict[str, Any] = {}
        for r in by["71"]:
            for case in ("c0.5", "c1.0", "c1.5"):
                v = r["result"][case][mode]["max_abs_mixed_difference"]
                if v is not None:
                    values.append(v)
            for case, entry in r["result"].items():
                v = entry[mode]["max_abs_mixed_difference"]
                if v is not None:
                    per_case.setdefault(f"{r['bound']} {case}", []).append(v)
        s71[mode] = {
            "verdict": interaction_verdict(values),
            "per_bound_and_case": {
                k_: {
                    "median": _median(x),
                    "max": max(x),
                    "n_at_or_above_0.5": sum(y >= 0.5 for y in x),
                    "n": len(x),
                }
                for k_, x in per_case.items()
            },
        }
    summary["71"] = s71

    s73: dict[str, Any] = {}
    main_rows = [r for r in by["73"] if r["artifact"] != "m3_joint_seeds_umin126"]
    seed_rows = [r for r in by["73"] if r["artifact"] == "m3_joint_seeds_umin126"]
    for case in ("control", "c0.5", "c1.0", "c1.5"):
        rows = sorted((r for r in main_rows if r["case"] == case), key=lambda r: r["grid_index"])
        s73[case] = {}
        for mode, _, _ in SITE_MODES:
            s73[case][mode] = {}
            for variant in ("fixed", "best"):
                gains = [r["result"][mode][f"gain_{variant}"] for r in rows]
                spreads = []
                for g in sorted({r["grid_index"] for r in seed_rows}):
                    m = [r["result"][mode][f"m3_{variant}"] for r in rows if r["grid_index"] == g]
                    m += [
                        r["result"][mode][f"m3_{variant}"]
                        for r in seed_rows
                        if r["grid_index"] == g and r["case"] == case
                    ]
                    spreads.append(max(m) - min(m))
                verdict = coupling_gain_verdict(
                    gains, max(spreads), [r["s2_co2_lines"] for r in rows]
                )
                verdict["gain_by_scenario"] = {
                    r["grid_index"]: g for r, g in zip(rows, gains, strict=True)
                }
                s73[case][mode][variant] = verdict
    summary["73"] = s73

    s76: dict[str, Any] = {}
    rows76 = by["76"]
    for mode, _, _ in SITE_MODES:
        s76[mode] = {}
        for variant in ("fixed", "best"):
            protected = {}
            for r in rows76:
                g = r["result"][mode]["gain_per_seed"][variant]
                box = "60-100" if r["artifact"] == "m3x_sens_tmax100_umin126" else "60-120"
                protected[f"{box} scenario {r['scenario']}"] = {
                    "min": min(g.values()),
                    "max": max(g.values()),
                }
            robust = [r for r in rows76 if r["artifact"] != "m3x_sens_tmax100_umin126"]
            gains, spreads, line_co2 = [], [], []
            stored = json.loads((RUNS_DIR / "m3x_robust_verdict_umin126.json").read_text("utf-8"))[
                "rows"
            ]
            for r in sorted(robust, key=lambda r: r["scenario"]):
                m = r["result"][mode]["m3_per_seed"][variant]
                gains.append(r["result"][mode]["best_s2"] - min(m.values()))
                spreads.append(max(m.values()) - min(m.values()))
                line_co2.append(stored[str(r["scenario"])]["s2_co2_lines_befund73"])
            s100 = [protected[f"60-100 scenario {s}"]["min"] for s in (18, 19, 22, 23, 24)]
            s76[mode][variant] = {
                "g_min_over_seeds": protected,
                "sensitivity_100_n_of_5_at_least_1": int(sum(x >= 1.0 for x in s100)),
                "robust_verdict_25": coupling_gain_verdict(gains, max(spreads), line_co2),
                "n_scenarios_robust": len(gains),
            }
    summary["76"] = s76

    s77: dict[str, Any] = {}
    for mode, _, _ in SITE_MODES:
        s77[mode] = {}
        for variant in ("fixed", "best"):
            rows = {}
            for r in by["77"]:
                g = r["result"][mode]["gain_per_seed"][variant]
                box = "lab" if "lab" in r["artifact"] else "box"
                rows[f"{box} {r['group']} ({r['t_amb']}; {r['phi']})"] = {
                    "min": min(g.values()),
                    "max": max(g.values()),
                }
            s77[mode][variant] = {
                "g_over_seeds": rows,
                "K_lab_n_of_6": int(
                    sum(v["min"] >= 1.0 for k_, v in rows.items() if k_.startswith("lab W"))
                ),
                "K_box_n_of_6": int(
                    sum(v["min"] >= 1.0 for k_, v in rows.items() if k_.startswith("box W"))
                ),
            }
    summary["77"] = s77
    return summary


def tasks() -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = [("57", None), ("60", None)]
    out += [("69b", s) for s in (lb.LINE_A_SPRAY, *lb.CANDIDATES)]
    out += [
        ("70", (n, i))
        for n in ("m2_site_network_l015_umin126", "m2_site_network_l015_umin080")
        for i in range(25)
    ]
    out += [("71", (b, i)) for b in ("1.26", "0.80") for i in range(25)]
    out += [
        ("73", (f"m3_joint_{c}_umin126", i))
        for c in ("control", "c0.5", "c1.0", "c1.5")
        for i in range(25)
    ]
    seeds = json.loads((RUNS_DIR / "m3_joint_seeds_umin126.json").read_text("utf-8"))
    out += [("73", ("m3_joint_seeds_umin126", i)) for i in range(len(seeds["records"]))]
    for raw, ev in (
        ("m3x_confirm_umin126", "m3x_confirm_evaluation_umin126"),
        ("m3x_confirm_s18_19_umin126", "m3x_confirm_s18_19_evaluation_umin126"),
        ("m3x_robust_rest_umin126", "m3x_robust_rest_evaluation_umin126"),
        ("m3x_sens_tmax100_umin126", "m3x_sens_tmax100_evaluation_umin126"),
    ):
        for s in json.loads((RUNS_DIR / f"{ev}.json").read_text("utf-8"))["scenarios"]:
            out.append(("76", (raw, ev, int(s))))
    for name in ("m4c_confirm_lab_umin126", "m4c_confirm_box_umin126"):
        for sid in json.loads((RUNS_DIR / f"{name}.json").read_text("utf-8"))["scenarios"]:
            out.append(("77", (name, sid)))
    return out


def main() -> int:
    require_clean_tree()
    assert Path(twinlink_ml.__file__).resolve().is_relative_to(REPO_ROOT)
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    configs, k, _, _ = m3._setup()
    configs = {**configs, "base": m3.load_process_config(m3.CONFIG_PATH)}
    start = time.perf_counter()
    todo = tasks()
    # longest tasks first
    weight = {"71": 0, "76": 1, "77": 2, "73": 3, "70": 4, "69b": 5, "60": 6, "57": 7}
    todo.sort(key=lambda t: weight[t[0]])
    with Pool(cpu_count(), initializer=_init, initargs=(configs, k)) as pool:
        results = pool.map(_dispatch, todo, chunksize=1)
    summary = summarise(results)
    write_artifact(
        ARTIFACT,
        {
            "_provenance": provenance(),
            "purpose": "inventory of review finding A1: Befunde using the cascade, re-evaluated at stored set points",
            "modes": {
                "as_run_20K": "warm end only, model profile, dT_min 20 K (as run; reproduces the stored figures)",
                "as_run_10K": "warm end only, model profile, dT_min 10 K (where the Befund reported 10 K)",
                "along_20K": "minimum along the exchanger >= 20 K, physical profile",
                "along_10K": "minimum along the exchanger >= 10 K, physical profile",
            },
            "variants": {
                "fixed": "stored interconnection held",
                "best": "best of the enumeration at the held set points",
            },
            "reoptimisation_needed": {
                "57d": False,
                "60": False,
                "69b": False,
                "70": "joint effect S -> M only (M is a joint optimum); F, F best, S are exact",
                "71": False,
                "73": "G: yes (M3' joint); S2 exact",
                "76": "G: yes (M3' joint); S2 exact",
                "77": "G: yes (M3' joint); S2 exact",
            },
            "scale_factor_k": k,
            "n_tasks": len(todo),
            "wall_s": time.perf_counter() - start,
            "summary": summary,
            "results": results,
        },
    )
    print(f"{len(todo)} tasks in {time.perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

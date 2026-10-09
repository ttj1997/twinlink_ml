r"""Check of review finding A1 (2026-10-09): minimum approach temperature of the cascade along the exchanger.

Review 3 (``docs/reviews/2026-10-09/pruefung3.md``, A1): ``heatnetwork/balance.py`` (about l. 545-547) limits a
stage only by ``T_sink,out <= T_source,in - dT_min`` (warm end). Whether ``T_source,out >= T_sink,in + dT_min``
holds at the cold end, or along the exchanger, is never checked. The gain of Befund 77 runs through condensation
of line exhaust at the external sink S1. This script CHECKS that on the stored solutions of Befund 77; it changes
nothing in ``src/``, re-optimises nothing and is not a correction.

Input: ``artifacts/runs/m4c_confirm_lab_umin126.json`` (Befund 77, box 60-100 degC; groups W, N, H; every S2
record (25 pairs of seeds) and every joint record (3 candidate interconnections x 5 seeds) with set points and
interconnection). The line states are re-solved at the stored set points with the search precision of the run;
the reproduction of every stored CO2 value is asserted.

Profiles of a source (constant heat-capacity rate ``C = m_L (c_p,L + Y c_p,D)`` and latent heat
``dh_v(T_kalt)`` as in the cascade, so that both profiles release the same total heat down to ``T_kalt``):

* model profile (what ``evaluate_cascade`` assumes): sensible cooling to ``T_kalt``, then all latent heat at
  ``T_kalt`` (flash condensation at the cold end);
* physical profile (condensation along the exchanger): the heat released down to ``T`` is
  ``C (T_out - T) + m_L max(0, Y - Y_s(T)) dh_v(T_kalt)``, i.e. condensation starts at the dew point of the
  exhaust. Discretised in steps of at most 0.01 K, linear between the nodes.

Counter-current stage, with ``q`` the heat already taken from the source by earlier stages and ``Q`` the stage
duty: the source runs from ``q`` to ``q + Q`` along its profile, the sink rises linearly from its current
temperature. Approach temperature at a point = source temperature minus sink temperature at that point.

(a) For the best S2 and the best M3' of every seed (box 60-100 degC, all groups; W is the decision group of
    Befund 77): every stage of the cascade AS RUN, with the approach temperature at the warm end, at the cold end
    (model profile and physical profile) and the minimum along the exchanger (physical profile, discretised).
(b) CO2 of S2 and M3' with a cascade that enforces ``dT_min`` as the minimum along the whole exchanger (physical
    profile), for ``dT_min`` = 20 K (the value of the runs) and 10 K, SET POINTS HELD at the stored values. Two
    variants of the interconnection: ``fixed`` (the stored interconnection of each solution) and ``best`` (the
    best of the 441 interconnections at the held set points, the S2 rule applied to both arms). Best S2 = minimum
    over the 25 pairs of seeds, M3'(seed) = minimum over the seed's three candidate records,
    ``G_s = S2 - M3'(seed)``, ``G = min_s G_s`` (rule of Befund 77). Reference rows: the cascade as run (warm end
    only, model profile) at 20 K, which must reproduce the artifact, and at 10 K.
    CAVEAT: at held set points this is NOT a new coupling gain (that would need a re-optimisation of the set
    points under the corrected cascade); it only estimates the effect of the check on the reported solutions.
(c) Scenario (30.0; 0.82): does the hot line (the line with the higher inlet temperature) still condense into its
    sinks with the check along the exchanger? Condensate per stage of its exhaust source (in the stage tables).

Self-check (aborts the run if violated): with the check at the warm end and the model profile the cascade of this
script reproduces ``evaluate_cascade`` for all 441 interconnections at every reported solution (CO2 within 1e-6
kg/h).

    PYTHONPATH=src python scripts/check_review_a1_cascade_approach.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import run_m3_joint as m3  # noqa: E402
import run_m3x_confirm as c  # noqa: E402
from _common import (  # noqa: E402
    REPO_ROOT,
    RUNS_DIR,
    provenance,
    require_clean_tree,
    write_artifact,
)

import twinlink_ml  # noqa: E402
from twinlink_ml.evaluation import joint_search as js  # noqa: E402
from twinlink_ml.evaluation import site_network as sn  # noqa: E402
from twinlink_ml.evaluation.net_energy import (  # noqa: E402
    ELECTRICITY_CO2_KG_PER_KWH,
    GAS_CO2_KG_PER_KWH,
)
from twinlink_ml.evaluation.site_oracle import SiteOracleSpec  # noqa: E402
from twinlink_ml.heatnetwork.routing import FixedRouting  # noqa: E402
from twinlink_ml.simulation.backend import NumpyBackend  # noqa: E402
from twinlink_ml.simulation.oracle_search import evaluate_point  # noqa: E402
from twinlink_ml.simulation.thermo import (  # noqa: E402
    dew_point_celsius,
    latent_heat_j_per_kg,
    saturation_humidity_kg_per_kg,
)

SOURCE_ARTIFACT = "m4c_confirm_lab_umin126"
ARTIFACT = "review_a1_cascade_approach_m4c_lab"
MODES: tuple[tuple[str, str, float], ...] = (
    ("as_run_20K", "warm_end", 20.0),
    ("as_run_10K", "warm_end", 10.0),
    ("along_20K", "along", 20.0),
    ("along_10K", "along", 10.0),
)
PROFILE_STEP_K = 0.01
SELF_CHECK_TOLERANCE_KG_PER_H = 1.0e-6
REPRODUCTION_TOLERANCE_KG_PER_H = 1.0e-9
GAIN_MIN_KG_PER_H = 1.0
HOT_LINE_SCENARIO = (30.0, 0.82)
_BACKEND = NumpyBackend()
_STATE: dict[str, Any] = {}

Allocation = tuple[tuple[str, str], ...]


# ------------------------------------------------------------------------------------------- profiles
@dataclass(frozen=True, slots=True)
class SourceProfile:
    """Released heat ``q`` [W] against the source temperature [degC], nodes ordered by increasing ``q``."""

    name: str
    capacity_w_per_k: float
    inlet_celsius: float
    q_w: np.ndarray
    t_celsius: np.ndarray

    @property
    def total_w(self) -> float:
        return float(self.q_w[-1])

    def temperature_at(self, q_w: float) -> float:
        return float(np.interp(q_w, self.q_w, self.t_celsius))

    def latent_w(self, q_w: float) -> float:
        """Latent part of the heat released down to the point ``q_w`` (released minus sensible)."""
        return q_w - self.capacity_w_per_k * (self.inlet_celsius - self.temperature_at(q_w))


def source_profiles(
    network: Any, t_kalt: float, kind: str
) -> tuple[dict[str, SourceProfile], float]:
    """Model or physical profile of every source of the network; returns them and ``dh_v(T_kalt)``."""
    gas = network.gas
    latent = float(latent_heat_j_per_kg(_BACKEND, _BACKEND.asarray(t_kalt)))
    y_s_kalt = float(saturation_humidity_kg_per_kg(_BACKEND, gas, _BACKEND.asarray(t_kalt)))
    out = {}
    for source in network.sources:
        m = source.dry_air_mass_flow_kg_per_s
        y = source.outlet_humidity_kg_per_kg
        cap = m * (gas.cp_dry_air_j_per_kg_k + y * gas.cp_vapour_j_per_kg_k)
        t_in = source.outlet_temperature_celsius
        if not t_in > t_kalt:
            raise SystemExit(f"source {source.name} at {t_in} degC is not above T_kalt {t_kalt}")
        if kind == "model":
            q_sens = cap * (t_in - t_kalt)
            q_lat = max(y - y_s_kalt, 0.0) * m * latent
            q = [0.0, q_sens] + ([q_sens + q_lat] if q_lat > 0.0 else [])
            t = [t_in, t_kalt] + ([t_kalt] if q_lat > 0.0 else [])
            q_arr, t_arr = np.asarray(q), np.asarray(t)
        else:
            n = math.ceil((t_in - t_kalt) / PROFILE_STEP_K) + 1
            t_arr = np.linspace(t_in, t_kalt, n)
            y_s = np.asarray(saturation_humidity_kg_per_kg(_BACKEND, gas, _BACKEND.asarray(t_arr)))
            q_arr = cap * (t_in - t_arr) + m * np.maximum(y - y_s, 0.0) * latent
        out[source.name] = SourceProfile(source.name, cap, t_in, q_arr, t_arr)
    return out, latent


def max_duty_along(
    profile: SourceProfile,
    q_before: float,
    sink_in: float,
    sink_cap: float,
    dt_min: float,
    upper: float,
) -> float:
    """Largest duty ``Q <= upper`` with an approach of at least ``dt_min`` at every point of the stage.

    Approach at the source point ``q``: ``g(q) - sink_in - (q_before + Q) / C_sink`` with
    ``g(q) = T_src(q) + q / C_sink``; its minimum over ``[q_before, q_before + Q]`` falls with ``Q``.
    """
    upper = min(upper, profile.total_w - q_before)
    if upper <= 0.0:
        return 0.0
    c0 = dt_min + sink_in
    inside = profile.q_w > q_before
    qs = np.concatenate(([q_before], profile.q_w[inside]))
    gs = (
        np.concatenate(([profile.temperature_at(q_before)], profile.t_celsius[inside]))
        + qs / sink_cap
    )
    running = np.minimum.accumulate(gs)
    h = running - c0 - qs / sink_cap  # strictly decreasing along the nodes
    if h[0] < 0.0:
        return 0.0
    q_end_upper = q_before + upper
    last = int(np.searchsorted(-h, 0.0, side="right")) - 1  # last node with h >= 0
    if qs[last] >= q_end_upper or last == len(qs) - 1:
        return upper
    q_a, q_b = qs[last], qs[last + 1]
    if q_b > q_end_upper:
        g_lim = profile.temperature_at(q_end_upper) + q_end_upper / sink_cap
        if min(running[last], g_lim) - c0 - q_end_upper / sink_cap >= 0.0:
            return upper
    # root on [q_a, q_b]: the minimum of two decreasing functions; the smaller root counts
    slope = (gs[last + 1] - gs[last]) / (q_b - q_a)
    root_flat = (running[last] - c0) * sink_cap
    denominator = 1.0 / sink_cap - slope
    root_line = (gs[last] - slope * q_a - c0) / denominator if denominator > 0.0 else math.inf
    q_end = min(root_flat, root_line, q_b, q_end_upper)
    return max(q_end, q_a) - q_before


def stage_row(
    profile: SourceProfile,
    q_before: float,
    q: float,
    sink_in: float,
    sink_cap: float,
    latent: float,
) -> dict[str, float]:
    """Temperatures, approach and condensate of one stage read on ``profile``."""
    t_src_in = profile.temperature_at(q_before)
    t_src_out = profile.temperature_at(q_before + q)
    t_snk_out = sink_in + q / sink_cap
    inside = (profile.q_w > q_before) & (profile.q_w < q_before + q)
    qs = np.concatenate(([q_before], profile.q_w[inside], [q_before + q]))
    ts = np.concatenate(([t_src_in], profile.t_celsius[inside], [t_src_out]))
    approach = ts - (sink_in + (q_before + q - qs) / sink_cap)
    i_min = int(np.argmin(approach))
    condensate = max(profile.latent_w(q_before + q) - profile.latent_w(q_before), 0.0) / latent
    return {
        "source_in_celsius": t_src_in,
        "source_out_celsius": t_src_out,
        "approach_warm_end_k": t_src_in - t_snk_out,
        "approach_cold_end_k": t_src_out - sink_in,
        "approach_min_along_k": float(approach[i_min]),
        "position_of_min_from_warm_end": float((qs[i_min] - q_before) / q) if q > 0.0 else 0.0,
        "condensate_kg_per_h_scaled": condensate * 3600.0,
    }


@dataclass(frozen=True, slots=True)
class Prepared:
    """Everything of one pair of line states that does not depend on the interconnection."""

    lines: tuple[Any, Any]
    network: Any
    model: dict[str, SourceProfile]
    physical: dict[str, SourceProfile]
    latent: float
    routes: dict[Allocation, Allocation]


def cascade(
    prep: Prepared, allocation: Allocation, check: str, dt_min: float, detail: bool = False
) -> dict[str, Any]:
    """The cascade of ``balance.evaluate_cascade`` (order, set-point cap, available heat) with the approach
    checked at the warm end on the model profile (``check="warm_end"``, as run) or along the exchanger on the
    physical profile (``check="along"``)."""
    network = prep.network
    profiles = prep.model if check == "warm_end" else prep.physical
    gas = network.gas
    sinks = {s.name: s for s in network.sinks}
    sink_cap = {
        s.name: s.dry_air_mass_flow_kg_per_s
        * (gas.cp_dry_air_j_per_kg_k + s.humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k)
        for s in network.sinks
    }
    sink_t = {s.name: float(s.intake_temperature_celsius) for s in network.sinks}
    used = dict.fromkeys(profiles, 0.0)
    rows = []
    for source_name, sink_name in prep.routes[allocation]:
        profile = profiles[source_name]
        q_before = used[source_name]
        t_snk_in = sink_t[sink_name]
        c_snk = sink_cap[sink_name]
        q_set = max(c_snk * (sinks[sink_name].setpoint_temperature_celsius - t_snk_in), 0.0)
        upper = min(q_set, max(profile.total_w - q_before, 0.0))
        if check == "warm_end":
            q_pinch = max(c_snk * (profile.temperature_at(q_before) - dt_min - t_snk_in), 0.0)
            q = min(upper, q_pinch)
        else:
            q = max_duty_along(profile, q_before, t_snk_in, c_snk, dt_min, upper)
        q = max(q, 0.0)
        if detail:
            row: dict[str, Any] = {
                "source": source_name,
                "sink": sink_name,
                "duty_kw": q / 1000.0,
                "sink_in_celsius": t_snk_in,
                "sink_out_celsius": t_snk_in + q / c_snk,
                "profile_of_the_check": stage_row(
                    profile, q_before, q, t_snk_in, c_snk, prep.latent
                ),
            }
            if check == "warm_end":
                # the same duties read on the physical profile: what the stage would look like with
                # condensation along the exchanger
                row["physical_profile"] = stage_row(
                    prep.physical[source_name], q_before, q, t_snk_in, c_snk, prep.latent
                )
            rows.append(row)
        used[source_name] = q_before + q
        sink_t[sink_name] = t_snk_in + q / c_snk
    gas_kw = sum(
        max(sink_cap[name] * (sinks[name].setpoint_temperature_celsius - sink_t[name]), 0.0)
        / network.heater_efficiency
        / 1000.0
        for name in sinks
    )
    k = _STATE["spec"].site.scale_factor
    electric = k * sum(x.electric_kw for x in prep.lines)
    co2 = ELECTRICITY_CO2_KG_PER_KWH * electric + GAS_CO2_KG_PER_KWH * gas_kw
    return {"co2": co2, "stages": rows}


# ------------------------------------------------------------------------------------------- worker
def _init(configs: dict[str, Any], k: float) -> None:
    _STATE["configs"] = configs
    _STATE["spec"] = SiteOracleSpec(c.THRESHOLD, js.site_spec_with_cap(sn.SiteSpec(k), None))
    _STATE["allocations"] = [
        tuple(tuple(p) for p in a) for a in sn.all_allocations(2, order_variants=True)
    ]
    _STATE["prepared"] = {}


def prepared(set_points: list[float], t_amb: float, phi: float) -> Prepared:
    key = (tuple(set_points), t_amb, phi)
    cache = _STATE["prepared"]
    if key in cache:
        return cache[key]
    spec: SiteOracleSpec = _STATE["spec"]
    states = []
    for name, (t_in, u) in (("A", set_points[:2]), ("B", set_points[2:])):
        solution = evaluate_point(
            t_in,
            u,
            1.0,
            t_amb,
            phi,
            max_iterations=spec.search_max_iterations,
            tolerance=spec.search_tolerance,
            base_config=_STATE["configs"][name],
        )
        states.append(
            sn.line_state(
                name, solution, _STATE["configs"][name], t_amb=t_amb, phi_amb=phi, t_in=t_in
            )
        )
    lines = (states[0], states[1])
    network, _, _ = sn._network(lines, spec.site)
    model, latent = source_profiles(network, spec.site.t_kalt_celsius, "model")
    physical, _ = source_profiles(network, spec.site.t_kalt_celsius, "physical")
    routes = {a: FixedRouting(a).route(network) for a in _STATE["allocations"]}
    cache[key] = Prepared(lines, network, model, physical, latent, routes)
    return cache[key]


def _stored(r: dict[str, Any]) -> tuple[list[float], Allocation, float, float]:
    """Set points, interconnection, value and CO2 of an S2 or joint record."""
    if r["kind"] == "s2":
        return r["set_points"], tuple(tuple(p) for p in r["allocation"]), r["value"], r["co2"]
    d = r["detail"]
    return d["set_points"], tuple(tuple(p) for p in d["allocation"]), r["value"], d["co2_kg_per_h"]


def _scenario_task(scenario: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    t_amb, phi = scenario["t_amb"], scenario["phi"]
    records = scenario["_records"]
    allocations = _STATE["allocations"]
    spec: SiteOracleSpec = _STATE["spec"]

    # reproduction of every stored value with the repository cascade (stored interconnection)
    reproduction = 0.0
    for r in records:
        x, allocation, _, co2 = _stored(r)
        prep = prepared(x, t_amb, phi)
        result, rejected = sn.evaluate_site(prep.lines, spec.site, [allocation])
        assert rejected == 0
        reproduction = max(reproduction, abs(result[0].co2_kg_per_h - co2))
    if reproduction > REPRODUCTION_TOLERANCE_KG_PER_H:
        raise SystemExit(
            f"scenario {scenario['id']}: stored CO2 not reproduced ({reproduction:.3e})"
        )

    # every record under every mode, stored interconnection and best of the 441
    values: list[dict[str, Any]] = []
    for r in records:
        x, allocation, value, co2 = _stored(r)
        penalty = value - co2  # depends on the line states only, not on the cascade
        prep = prepared(x, t_amb, phi)
        row: dict[str, Any] = {"kind": r["kind"], "seed": r["seed"]}
        for key, check, dt in MODES:
            all_values = [cascade(prep, a, check, dt)["co2"] for a in allocations]
            i_best = int(np.argmin(all_values))
            row[key] = {
                "fixed": cascade(prep, allocation, check, dt)["co2"] + penalty,
                "best": all_values[i_best] + penalty,
                "best_allocation": allocations[i_best],
            }
        values.append(row)

    # gains by the rule of Befund 77 per mode and variant
    gains: dict[str, Any] = {}
    for key, _, _ in MODES:
        gains[key] = {}
        for variant in ("fixed", "best"):
            s2_best = min(v[key][variant] for v in values if v["kind"] == "s2")
            per_seed = {
                seed: min(
                    v[key][variant] for v in values if v["kind"] == "joint" and v["seed"] == seed
                )
                for seed in c.SEEDS
            }
            g = {seed: s2_best - m for seed, m in per_seed.items()}
            gains[key][variant] = {
                "best_s2": s2_best,
                "m3_per_seed": per_seed,
                "gain_per_seed": g,
                "gain_min_over_seeds": min(g.values()),
                "gain_median_over_seeds": float(np.median(list(g.values()))),
            }

    # (a) and (c): stage tables of the reported solutions (best S2, best M3' of every seed)
    indexed = list(enumerate(records))
    reported = {
        "s2": min((p for p in indexed if p[1]["kind"] == "s2"), key=lambda p: p[1]["value"])
    }
    for seed in c.SEEDS:
        reported[f"m3_seed{seed}"] = min(
            (p for p in indexed if p[1]["kind"] == "joint" and p[1]["seed"] == seed),
            key=lambda p: p[1]["value"],
        )
    tables: dict[str, Any] = {}
    self_check = 0.0
    for label, (index, r) in reported.items():
        x, allocation, _, _ = _stored(r)
        prep = prepared(x, t_amb, phi)
        reference, rejected = sn.evaluate_site(prep.lines, spec.site, allocations)
        assert rejected == 0
        for a, ref in zip(allocations, reference, strict=True):
            mine = cascade(prep, a, "warm_end", spec.site.dt_min_k)["co2"]
            self_check = max(self_check, abs(mine - ref.co2_kg_per_h))
        entry: dict[str, Any] = {
            "set_points": x,
            "allocation": [list(p) for p in allocation],
            "hot_line": "A" if x[0] > x[2] else "B",
            "lines": {
                line.name: {
                    "outlet_celsius": line.outlet_temperature_celsius,
                    "outlet_humidity_kg_per_kg": line.outlet_humidity_kg_per_kg,
                    "dew_point_celsius": float(
                        dew_point_celsius(
                            _BACKEND,
                            prep.network.gas,
                            _BACKEND.asarray(line.outlet_humidity_kg_per_kg),
                        )
                    ),
                }
                for line in prep.lines
            },
        }
        for key, check, dt in MODES:
            for variant in ("fixed", "best"):
                chosen = allocation if variant == "fixed" else values[index][key]["best_allocation"]
                result = cascade(prep, chosen, check, dt, detail=True)
                entry[f"{key}_{variant}"] = {
                    "allocation": [list(p) for p in chosen],
                    "co2_kg_per_h": result["co2"],
                    "stages": result["stages"],
                }
        tables[label] = entry
    if self_check > SELF_CHECK_TOLERANCE_KG_PER_H:
        raise SystemExit(f"scenario {scenario['id']}: self-check failed ({self_check:.3e} kg/h)")

    return {
        "id": scenario["id"],
        "group": scenario["group"],
        "t_amb": t_amb,
        "phi": phi,
        "absolute_humidity_ambient_kg_per_kg": scenario["absolute_humidity_ambient_kg_per_kg"],
        "stored_gain_min_over_seeds": scenario["gain_min_over_seeds"],
        "stored_gain_per_seed": scenario["gain_per_seed"],
        "reproduction_max_abs_kg_per_h": reproduction,
        "self_check_max_abs_kg_per_h": self_check,
        "gains": gains,
        "reported_solutions": tables,
        "wall_s": time.perf_counter() - start,
    }


def main() -> int:
    require_clean_tree()
    assert Path(twinlink_ml.__file__).resolve().is_relative_to(REPO_ROOT)
    art = json.loads((RUNS_DIR / f"{SOURCE_ARTIFACT}.json").read_text("utf-8"))
    configs, k, _, _ = m3._setup()
    assert abs(k - art["scale_factor_k"]) < 1e-9 * k
    scenarios = []
    for sid, s in art["scenarios"].items():
        rows = [
            r for r in art["records"] if r["scenario"] == int(sid) and r["kind"] in ("s2", "joint")
        ]
        scenarios.append({**s, "_records": rows})
    start = time.perf_counter()
    with Pool(min(cpu_count(), len(scenarios)), initializer=_init, initargs=(configs, k)) as pool:
        results = pool.map(_scenario_task, scenarios, chunksize=1)
    results.sort(key=lambda r: r["id"])

    summary: dict[str, Any] = {}
    for key, _, _ in MODES:
        summary[key] = {}
        for variant in ("fixed", "best"):
            summary[key][variant] = {
                "gain_min_over_seeds_by_scenario": {
                    f"{r['group']} ({r['t_amb']}; {r['phi']})": r["gains"][key][variant][
                        "gain_min_over_seeds"
                    ]
                    for r in results
                },
                "n_W_with_gain_at_least_1": sum(
                    r["gains"][key][variant]["gain_min_over_seeds"] >= GAIN_MIN_KG_PER_H
                    for r in results
                    if r["group"] == "W"
                ),
            }
    hot = next(r for r in results if (r["t_amb"], r["phi"]) == HOT_LINE_SCENARIO)
    write_artifact(
        ARTIFACT,
        {
            "_provenance": provenance(),
            "purpose": "check of review finding A1 (2026-10-09) on stored solutions; no correction, no re-optimisation",
            "source_artifact": SOURCE_ARTIFACT,
            "caveat": (
                "at held set points the corrected-cascade gains are NOT a new coupling gain (that needs a "
                "re-optimisation of the set points); they estimate the effect of the check on the reported solutions"
            ),
            "profiles": {
                "model": "sensible to T_kalt, then all latent heat at T_kalt (as evaluate_cascade)",
                "physical": "condensation from the dew point on: q(T) = C (T_out - T) + m max(0, Y - Y_s(T)) dh_v(T_kalt)",
                "step_k": PROFILE_STEP_K,
            },
            "modes": {
                "as_run_20K": "warm end only, model profile, dT_min 20 K (the cascade of the runs)",
                "as_run_10K": "warm end only, model profile, dT_min 10 K (reference)",
                "along_20K": "minimum along the exchanger >= 20 K, physical profile",
                "along_10K": "minimum along the exchanger >= 10 K, physical profile",
            },
            "variants": {
                "fixed": "the stored interconnection of each solution",
                "best": "the best of the 441 interconnections at the held set points",
            },
            "rule": "G_s = best S2 - M3'(seed s), G = min over the seeds (Befund 77); n_W counts G >= 1.0 kg/h",
            "scale_factor_k": k,
            "hot_line_scenario": list(HOT_LINE_SCENARIO),
            "hot_line_scenario_id": hot["id"],
            "summary": summary,
            "wall_s": time.perf_counter() - start,
            "scenarios": results,
        },
    )
    for key, by_variant in summary.items():
        print(key, {v: by_variant[v]["n_W_with_gain_at_least_1"] for v in by_variant})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

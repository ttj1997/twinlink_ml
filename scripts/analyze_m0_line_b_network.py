r"""M0: replacement criterion for the spray rate of line B (LB1-LB3), evaluation only.

Replaces the 10 K outlet-temperature criterion of ``probe_m0_line_b_spray_rate.py``, which was an
unsuitable proxy: with condensing recovery (Befund 57) the moisture of the exhaust, hence its dew
point, decides, not the dry temperature. The earlier script and its artifact stay unchanged; this
script reads the artifact ``m0_line_b_spray_rate.json`` and re-evaluates (no optimisation) the
stored optima: one steady-state solve per stored optimum at the stored set points (search
precision of Befund 52) to obtain ``Y_out``, the dry-air flow and the power components, then the
heat network for every enumerated allocation. The criteria below were fixed and committed before
this script ran.

Scale (D4, project lead): the lab lines stay lab models; their stationary flows are scaled
linearly to plant size with ONE constant factor ``k = 8 kg/s / (dry-air flow of line A at the config
nominal operating point)``, so that the exhaust flow of a line equals that of the fluidised-bed
source of the partner model (Q2, 8 kg/s, Befund 60). Everything below that uses ``k`` is a scaling
computation, not a plant statement.

Network: sources Q1 (partner spray tower, fixed: 20 kg/s, 125.233 degC, Y 0.06482), LA (line A
exhaust), LB (line B exhaust); sinks S1 (partner, fixed: 20 kg/s, 15 -> 300 degC, Y 0.005), SA and
SB (the inlet air of line A and B: scaled dry-air flow, ambient intake temperature and humidity of
the scenario, set point = the line's inlet temperature). Both lines have gas-fired air heaters
(D5 main case). Condensing recovery, ``T_kalt`` 29.1 degC, ``dT_min`` 20 K, ``eta_H`` 0.80, ``H_u``
45 MJ/kg (the partner data case of Befund 60); the CO2 factors of ``net_energy.py`` (0.366 kg/kWh
electricity, 0.201 kg/kWh gas). Per scenario and allocation:
``CO2 = 0.366 * E_el + 0.201 * E_gas`` [kg CO2e/h] with ``E_el`` = fan + pump + atomiser of both
lines (scaled) and ``E_gas`` = the fuel power of all three sinks after recovery.

Allocations (D7): every set of (source, sink) connections in which each of the three sources feeds
at least one sink (7^3 = 343 sets); the cascade order is the network's own rule (low-temperature
sources first). A set that the cascade rejects is counted and left out.

Criteria, fixed before the run.
  LB1  Descriptive, per spray rate (1.5 line A; 2.0, 2.5, 3.0 line B) and scenario: dew point of
       the exhaust, ``T_out``, ``Y_out``, and the heat recoverable down to ``T_kalt + dT_min``
       (49.1 degC, as requested) and, as a second column, down to ``T_kalt`` (29.1 degC, the cold end
       of the condensing cascade), each split into a sensible and a latent part (latent = condensate
       between the dew point and the floor times the latent heat at the floor; zero when the exhaust
       is not above the floor or the floor is not below the dew point). Lab scale, not scaled.
  LB2  Decisive: does the allocation make a difference? For each of the 9 scenarios, line A at
       spray 1.5 and line B at the candidate spray rate, evaluate all allocations, CO2-weighted. Met
       iff the median over the 9 scenarios of ``(CO2_worst - CO2_best) / CO2_best`` is at least
       0.02 (the "CO2 target of the site" is the CO2 of the best allocation, whole site incl. Q1
       and S1). Also reported: in how many scenarios the best allocation changes (number of distinct
       best allocations; scenarios whose best differs from the most frequent best; ties within
       relative 1e-9 are resolved by enumeration order and counted). Descriptive extras, not part
       of the criterion: the same difference in kg CO2e/h and relative to the CO2 of the two line
       sinks plus the line electricity only; and the difference between the best and the worst
       allocation when the connections of Q1 are held at those of the globally best allocation
       (only the allocations of the two lines vary, 49 sets), which isolates what the lines
       themselves decide (added before the run, after a one-scenario smoke test showed that the
       allocation of Q1 alone, independent of line B, already changes the site CO2 by tens of
       percent; LB2 as fixed is therefore expected not to discriminate between spray rates, and the
       control shows it). A control (line B = line A at 1.5) is evaluated too.
       Caveat for the finding: the lines are optimised individually, not jointly with the
       allocation; sufficient for this pre-check.
  LB3  Choice: the SMALLEST candidate spray rate in {2.0, 2.5, 3.0} that meets LB2 AND whose
       line-B optimum has an interior inlet temperature (below 120 degC minus
       ``TOLERANCE_TEMPERATURE_K`` = 0.05 K) in the majority (at least 5 of 9) of the scenarios.
       If there is none, nothing is chosen and the result is reported.

    PYTHONPATH=<worktree>/src python scripts/analyze_m0_line_b_network.py
"""

from __future__ import annotations

import itertools
import json
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import run_pinch_benchmark as partner  # noqa: E402
from _common import (  # noqa: E402
    REPO_ROOT,
    RUNS_DIR,
    provenance,
    require_clean_tree,
    write_artifact,
)
from probe_m0_line_b_spray_rate import _with_spray_rate  # noqa: E402

import twinlink_ml  # noqa: E402
from twinlink_ml.evaluation.net_energy import (  # noqa: E402
    ELECTRICITY_CO2_KG_PER_KWH,
    GAS_CO2_KG_PER_KWH,
)
from twinlink_ml.heatnetwork.balance import SinkState, SourceState, evaluate_cascade  # noqa: E402
from twinlink_ml.heatnetwork.config import (  # noqa: E402
    HeatNetworkConfig,
    HeatSinkConfig,
    HeatSourceConfig,
)
from twinlink_ml.heatnetwork.routing import FixedRouting  # noqa: E402
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config  # noqa: E402
from twinlink_ml.simulation.backend import NumpyBackend  # noqa: E402
from twinlink_ml.simulation.operating_point import TOLERANCE_TEMPERATURE_K  # noqa: E402
from twinlink_ml.simulation.oracle_search import evaluate_point  # noqa: E402
from twinlink_ml.simulation.thermo import (  # noqa: E402
    absolute_humidity_kg_per_kg,
    latent_heat_j_per_kg,
    saturation_humidity_kg_per_kg,
)

SOURCE_ARTIFACT = "m0_line_b_spray_rate"
ARTIFACT_NAME = "m0_line_b_network"
LINE_A_SPRAY = 1.5
CANDIDATES = (2.0, 2.5, 3.0)
T_KALT_C = 29.1
DT_MIN_K = 20.0
ETA_H = partner.HEATER_EFFICIENCY
H_U_J_PER_KG = partner.FUEL_LOWER_HEATING_VALUE_J_PER_KG
PARTNER_Q2_FLOW_KG_PER_S = partner.SOURCE_2.dry_air_mass_flow_kg_per_s
LB2_THRESHOLD = 0.02
INLET_UPPER_BOUND_C = 120.0
INTERIOR_MAJORITY = 5
TIE_RELATIVE = 1.0e-9
BACKEND = NumpyBackend()


def _states(base_config: Any, arm: dict[str, Any], spray: float) -> list[dict[str, Any]]:
    config = _with_spray_rate(base_config, spray)
    assert config.fingerprint == arm["config_fingerprint"], "config fingerprint differs"
    duct_loss = config.exhaust.outlet_duct_heat_loss_k
    out = []
    for r in sorted(arm["grid_results"], key=lambda x: x["_grid_index"]):
        sol = evaluate_point(
            r["inlet_temperature_celsius"],
            r["gas_velocity_m_per_s"],
            r["nozzle_pressure_bar"],
            r["ambient_temperature_celsius"],
            r["ambient_relative_humidity"],
            max_iterations=30,
            tolerance=1e-7,
            base_config=config,
        )
        assert abs(sol.bed_temperature_celsius - r["bed_temperature_celsius"]) < 1e-6
        d = sol.diagnostics
        out.append(
            {
                "t_amb": r["ambient_temperature_celsius"],
                "phi_amb": r["ambient_relative_humidity"],
                "t_in": r["inlet_temperature_celsius"],
                "t_out": sol.bed_temperature_celsius - duct_loss,
                "y_out": sol.outlet_humidity_kg_per_kg,
                "m_dry": sol.dry_air_mass_flow_kg_per_s,
                "dew_point": sol.dew_point_celsius,
                "y_in": float(
                    BACKEND.to_numpy(
                        absolute_humidity_kg_per_kg(
                            BACKEND,
                            config.gas,
                            BACKEND.asarray(r["ambient_temperature_celsius"]),
                            BACKEND.asarray(r["ambient_relative_humidity"]),
                        )
                    )
                ),
                "electric_kw": float(
                    d["power_fan_kw"] + d["power_pump_kw"] + d["power_atomiser_kw"]
                ),
                "heater_kw": float(d["power_heater_kw"]),
                "moisture": sol.particle_moisture_kg_per_kg,
                "feasible": sol.feasible,
            }
        )
    return out


def _recoverable(
    gas: Any, t_out: float, y_out: float, m_dry: float, floor: float
) -> dict[str, float]:
    if t_out <= floor:
        return {"sensible_kw": 0.0, "latent_kw": 0.0, "condensate_kg_per_s": 0.0}
    y_sat = float(
        BACKEND.to_numpy(saturation_humidity_kg_per_kg(BACKEND, gas, BACKEND.asarray(floor)))
    )
    condensate = max(0.0, y_out - y_sat) * m_dry
    dh = float(BACKEND.to_numpy(latent_heat_j_per_kg(BACKEND, BACKEND.asarray(floor))))
    sensible_w = (
        m_dry * (t_out - floor) * (gas.cp_dry_air_j_per_kg_k + y_out * gas.cp_vapour_j_per_kg_k)
    )
    return {
        "sensible_kw": sensible_w / 1000.0,
        "latent_kw": condensate * dh / 1000.0,
        "condensate_kg_per_s": condensate,
    }


def _allocations() -> list[tuple[tuple[str, str], ...]]:
    sinks = ("S1", "SA", "SB")
    nonempty = [subset for size in (1, 2, 3) for subset in itertools.combinations(sinks, size)]
    allocations = []
    for q1, la, lb in itertools.product(nonempty, repeat=3):
        allocations.append(
            tuple(
                (src, snk) for src, subset in (("Q1", q1), ("LA", la), ("LB", lb)) for snk in subset
            )
        )
    return allocations


def _evaluate_scenario(
    a: dict[str, Any], b: dict[str, Any], k: float, allocations: list[Any]
) -> dict[str, Any]:
    sources = (
        partner.SOURCE_1,
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
        minimum_pinch_k=DT_MIN_K,
        recovery_mode="condensing",
        cold_temperature_celsius=T_KALT_C,
        fuel_lower_heating_value_j_per_kg=H_U_J_PER_KG,
        heater_efficiency=ETA_H,
    )
    source_states = tuple(SourceState.from_config(c) for c in sources)
    sink_states = tuple(SinkState.from_config(c) for c in sinks)
    electric_kw = k * (a["electric_kw"] + b["electric_kw"])
    rows = []
    rejected = 0
    for connections in allocations:
        try:
            stages = FixedRouting(connections).route(network)
            result = evaluate_cascade(network, stages, source_states, sink_states)
        except (ValueError, RuntimeError):
            rejected += 1
            continue
        fuel_w = result.fuel_mass_flow_kg_per_s[:, 0] * H_U_J_PER_KG
        gas_kw_per_sink = fuel_w / 1000.0
        gas_kw = float(gas_kw_per_sink.sum())
        lines_gas_kw = float(gas_kw_per_sink[1:].sum())
        rows.append(
            {
                "connections": connections,
                "gas_kw": gas_kw,
                "co2_kg_per_h": ELECTRICITY_CO2_KG_PER_KWH * electric_kw
                + GAS_CO2_KG_PER_KWH * gas_kw,
                "co2_lines_only_kg_per_h": ELECTRICITY_CO2_KG_PER_KWH * electric_kw
                + GAS_CO2_KG_PER_KWH * lines_gas_kw,
            }
        )
    co2 = np.array([r["co2_kg_per_h"] for r in rows])
    best_i = int(np.argmin(co2))
    worst_i = int(np.argmax(co2))
    best = rows[best_i]
    ties = int(np.sum(co2 <= co2[best_i] * (1.0 + TIE_RELATIVE)))
    lines = np.array([r["co2_lines_only_kg_per_h"] for r in rows])
    q1_best = tuple(c for c in best["connections"] if c[0] == "Q1")
    q1_fixed = np.array(
        [
            r["co2_kg_per_h"]
            for r in rows
            if tuple(c for c in r["connections"] if c[0] == "Q1") == q1_best
        ]
    )
    return {
        "n_allocations_evaluated": len(rows),
        "n_rejected": rejected,
        "best_connections": [list(c) for c in best["connections"]],
        "n_tied_with_best": ties,
        "co2_best_kg_per_h": float(co2[best_i]),
        "co2_worst_kg_per_h": float(co2[worst_i]),
        "difference_kg_per_h": float(co2[worst_i] - co2[best_i]),
        "relative_difference_to_best": float((co2[worst_i] - co2[best_i]) / co2[best_i]),
        "lines_only_best_kg_per_h": float(lines[best_i]),
        "relative_difference_to_lines_only": float((co2[worst_i] - co2[best_i]) / lines[best_i]),
        "n_allocations_with_q1_fixed": len(q1_fixed),
        "relative_difference_lines_only_allocations": float(
            (q1_fixed.max() - q1_fixed.min()) / q1_fixed.min()
        ),
        "gas_kw_best": best["gas_kw"],
        "electric_kw_scaled": electric_kw,
    }


def main() -> int:
    require_clean_tree()
    assert Path(twinlink_ml.__file__).resolve().is_relative_to(REPO_ROOT)
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    base_config = load_process_config(
        REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
    )
    source = json.loads((RUNS_DIR / f"{SOURCE_ARTIFACT}.json").read_text("utf-8"))
    arms = source["arms"]

    nominal = evaluate_point(
        base_config.actions.inlet_temperature_celsius.nominal,
        base_config.actions.gas_velocity_m_per_s.nominal,
        base_config.actions.nozzle_pressure_bar.nominal,
        base_config.disturbances.ambient_temperature_celsius.nominal,
        base_config.disturbances.ambient_relative_humidity.nominal,
        max_iterations=30,
        tolerance=1e-7,
        base_config=base_config,
    )
    k = PARTNER_Q2_FLOW_KG_PER_S / nominal.dry_air_mass_flow_kg_per_s

    states = {s: _states(base_config, arms[str(s)], s) for s in (LINE_A_SPRAY, *CANDIDATES)}

    lb1: dict[str, Any] = {}
    for s, rows in states.items():
        lb1[str(s)] = [
            {
                "t_amb": r["t_amb"],
                "phi_amb": r["phi_amb"],
                "t_in": r["t_in"],
                "t_out": r["t_out"],
                "y_out": r["y_out"],
                "dew_point": r["dew_point"],
                "dry_air_flow_lab_kg_per_s": r["m_dry"],
                "to_t_kalt_plus_dt_min": _recoverable(
                    base_config.gas, r["t_out"], r["y_out"], r["m_dry"], T_KALT_C + DT_MIN_K
                ),
                "to_t_kalt": _recoverable(
                    base_config.gas, r["t_out"], r["y_out"], r["m_dry"], T_KALT_C
                ),
            }
            for r in rows
        ]

    allocations = _allocations()
    lb2: dict[str, Any] = {}
    for s in (LINE_A_SPRAY, *CANDIDATES):
        per_scenario = [
            _evaluate_scenario(a, b, k, allocations)
            for a, b in zip(states[LINE_A_SPRAY], states[s], strict=True)
        ]
        relative = [x["relative_difference_to_best"] for x in per_scenario]
        bests = [json.dumps(x["best_connections"]) for x in per_scenario]
        counts = Counter(bests)
        modal = counts.most_common(1)[0][1]
        lb2[str(s)] = {
            "per_scenario": per_scenario,
            "median_relative_difference_to_best": float(np.median(relative)),
            "median_relative_difference_to_lines_only": float(
                np.median([x["relative_difference_to_lines_only"] for x in per_scenario])
            ),
            "median_relative_difference_lines_only_allocations": float(
                np.median([x["relative_difference_lines_only_allocations"] for x in per_scenario])
            ),
            "criterion_met": bool(np.median(relative) >= LB2_THRESHOLD),
            "n_distinct_best_allocations": len(counts),
            "n_scenarios_best_differs_from_modal": len(per_scenario) - modal,
            "n_scenarios_with_ties": int(sum(x["n_tied_with_best"] > 1 for x in per_scenario)),
            "is_control": s == LINE_A_SPRAY,
        }

    lb3: dict[str, Any] = {}
    for s in CANDIDATES:
        interior = sum(r["t_in"] < INLET_UPPER_BOUND_C - TOLERANCE_TEMPERATURE_K for r in states[s])
        lb3[str(s)] = {
            "n_interior_inlet_temperature": interior,
            "interior_majority": interior >= INTERIOR_MAJORITY,
            "lb2_met": lb2[str(s)]["criterion_met"],
            "qualifies": bool(lb2[str(s)]["criterion_met"] and interior >= INTERIOR_MAJORITY),
        }
    qualifying = [s for s in CANDIDATES if lb3[str(s)]["qualifies"]]
    chosen = min(qualifying) if qualifying else None

    write_artifact(
        ARTIFACT_NAME,
        {
            "_provenance": provenance(),
            "source_artifact": SOURCE_ARTIFACT,
            "scale_factor_k": k,
            "scale_basis": {
                "partner_q2_dry_air_flow_kg_per_s": PARTNER_Q2_FLOW_KG_PER_S,
                "line_a_nominal_dry_air_flow_kg_per_s": nominal.dry_air_mass_flow_kg_per_s,
            },
            "network": {
                "t_kalt_celsius": T_KALT_C,
                "dt_min_k": DT_MIN_K,
                "eta_h": ETA_H,
                "h_u_j_per_kg": H_U_J_PER_KG,
                "recovery_mode": "condensing",
                "n_allocations": len(allocations),
                "co2_electricity_kg_per_kwh": ELECTRICITY_CO2_KG_PER_KWH,
                "co2_gas_kg_per_kwh": GAS_CO2_KG_PER_KWH,
            },
            "lb1": lb1,
            "lb2": lb2,
            "lb3": lb3,
            "chosen_spray_rate_kg_per_h": chosen,
        },
    )
    print("k:", k)
    for s in (LINE_A_SPRAY, *CANDIDATES):
        e = lb2[str(s)]
        print(
            s,
            "median rel diff",
            round(e["median_relative_difference_to_best"], 5),
            "lines-only",
            round(e["median_relative_difference_to_lines_only"], 4),
            "q1 fixed",
            round(e["median_relative_difference_lines_only_allocations"], 5),
            "distinct best",
            e["n_distinct_best_allocations"],
        )
    print("lb3:", lb3)
    print("chosen:", chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

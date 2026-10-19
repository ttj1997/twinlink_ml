r"""Analysis of the exhaust-recirculation study (docs/22): tables, limits, recovery, sensitivity, transfer.

Reads the committed artifacts ``exhaust_recirc_anchor``, ``exhaust_recirc_main`` and
``exhaust_recirc_sensitivity`` (or their ``_maxiter50`` variants) and writes
``exhaust_recirc_analysis``. Everything here is fixed before the analysis is run (the script is
committed first); the descriptive re-evaluations (recovery in both layouts on the grid solutions,
the recirculation limit at unchanged set points) re-solve the steady state at stored set points and
change no optimum.

Labels: every table row carries its scale -- ``lab`` (laboratory line, placeholder physics),
``lab x k`` (lab flows scaled linearly with the constant k of M2, D4) or ``transfer`` (a
placeholder grid at partner orders of magnitude, no model of the plant).

Parts:

1. Grid G (r fixed, set points optimised): feasibility, CO2 change against r = 0 per context,
   binding limits, humidity and dew point; CO2 also re-weighted with the electricity factor
   "2030" (0.20 kg/kWh, placeholder) at the same solutions (not re-optimised).
2. Recirculation limit at UNCHANGED set points: per context, the largest r for which the steady
   state at the r = 0 optimum stays below the threshold ``L - m`` (and, labelled, below ``L``,
   i.e. consuming the soft-sensor margin); bisection on r in [0, 0.95], 20 steps.
3. Arm R (r free): r*, saving, binding limit.
4. Recovery: J0 / J (CO2 after the cascade minimised) and, descriptively, the cascade at the grid
   solutions in both layouts (``fresh_air``, ``mixed_air``).
5. Sensitivity: per variant the median r* and saving on the 3 x 3 sub-grid against the main run on
   the same contexts; ranking by the absolute change of the median saving.
6. Transfer grid (placeholders, ``transfer``): plant A and plant C (both readings of Befund 99),
   annual gas, CO2 and value at 55 and 300 EUR/t for the lab saving fractions (p10 / median / p90
   of the heater-gas saving of the envelope, below); the pure mixing saving at plant temperatures;
   the humidity headroom of the exhaust.

DECLARED POST HOC (written after the main artifact had been looked at, before this script ran):
the envelope of part 3 (best feasible point per context over R and the grid arms G), the envelope
of the joint arm (best of J and J0) and of the sensitivity arms (best of R and r = 0), because the
three-variable search with the reduced budget returned, at some warm contexts, a CO2 above that of
``r = 0``, which lies inside its box. The raw R / J values are kept and reported next to it, with
the count of contexts where R was beaten.

    tools/run_here.sh scripts/analyze_exhaust_recirc.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
import warnings
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

from twinlink_ml.evaluation.exhaust_recirculation import (  # noqa: E402
    FIXED_NOZZLE_PRESSURE_BAR,
    RecoverySpec,
    recovery_balance,
    with_recirculation,
)
from twinlink_ml.evaluation.net_energy import (  # noqa: E402
    ELECTRICITY_CO2_KG_PER_KWH,
    GAS_CO2_KG_PER_KWH,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config  # noqa: E402
from twinlink_ml.simulation.backend import NumpyBackend  # noqa: E402
from twinlink_ml.simulation.config import ProcessConfig  # noqa: E402
from twinlink_ml.simulation.operating_point import (  # noqa: E402
    DEFAULT_SEARCH_MAX_ITERATIONS,
    DEFAULT_SEARCH_TOLERANCE,
)
from twinlink_ml.simulation.oracle_search import evaluate_point  # noqa: E402
from twinlink_ml.simulation.thermo import (  # noqa: E402
    absolute_humidity_kg_per_kg,
    saturation_humidity_kg_per_kg,
)

CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
ELECTRICITY_2030_CO2_KG_PER_KWH = 0.20
"""Electricity factor "2030", PLACEHOLDER (instruction of the study; no source)."""
R_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
SUBGRID = (0, 2, 4, 10, 12, 14, 20, 22, 24)
CONTEXTS = range(25)
LIMIT_BISECTION_STEPS = 20
LIMIT_R_MAX = 0.95
BINDING_KEYS = (
    "moisture",
    "gas_velocity_box_min",
    "gas_velocity_u_mf",
    "gas_velocity_u_terminal",
    "bed_temperature",
    "inlet_temperature_box_min",
    "inlet_temperature_box_max",
    "fog",
    "recirculation_box_max",
)
_BACKEND = NumpyBackend()

# ------------------------------------------------------------------ transfer placeholders
# value, band, source, status -- every number of the transfer grid comes from here.
TRANSFER: dict[str, dict[str, Any]] = {
    "ambient_annual_mean_celsius": {
        "value": 10.0,
        "band": (8.0, 12.0),
        "source": "long-term annual mean of the air temperature in Germany, order of magnitude",
        "status": "placeholder",
    },
    "ambient_relative_humidity": {
        "value": 0.75,
        "band": (0.5, 0.9),
        "source": "as storage test 1 (docs/19), annual mean order of magnitude",
        "status": "placeholder",
    },
    "normal_density_kg_per_nm3": {
        "value": 1.293,
        "band": (1.293, 1.293),
        "source": "dry air at 0 degC, 1.01325 bar (definition); volumes read as Nm3/h",
        "status": "placeholder (the reading m3/h = Nm3/h is an assumption)",
    },
    "plant_a_air_nm3_per_h": {
        "value": 30000.0,
        "band": (25000.0, 35000.0),
        "source": "partner, order of magnitude (about 30 000 m3/h process air)",
        "status": "partner statement, order of magnitude",
    },
    "plant_a_inlet_celsius": {
        "value": 82.5,
        "band": (80.0, 85.0),
        "source": "partner (inlet air about 80-85 degC)",
        "status": "partner statement, order of magnitude",
    },
    "plant_a_heat_supply_efficiency": {
        "value": 0.85,
        "band": (0.75, 0.92),
        "source": "steam heater: gas boiler (about 0.90-0.95) times steam distribution losses; "
        "typical literature range",
        "status": "placeholder",
    },
    "plant_a_hours_per_year": {
        "value": 8760.0,
        "band": (8000.0, 8760.0),
        "source": "partner (operation 24/7)",
        "status": "partner statement",
    },
    "plant_c_air_nm3_per_h_per_line": {
        "value": 20000.0,
        "band": (15000.0, 25000.0),
        "source": "partner (2 lines of about 20 000 m3/h)",
        "status": "partner statement, order of magnitude",
    },
    "plant_c_lines": {
        "value": 2.0,
        "band": (2.0, 2.0),
        "source": "partner",
        "status": "partner statement",
    },
    "plant_c_inlet_celsius": {
        "value": 80.0,
        "band": (70.0, 90.0),
        "source": "no partner figure; same process as plant A assumed (as docs/19)",
        "status": "placeholder",
    },
    "plant_c_burner_efficiency": {
        "value": 0.95,
        "band": (0.90, 1.00),
        "source": "direct-fired gas burner: the flue gas enters the process air, only radiation "
        "and casing losses; typical manufacturer range",
        "status": "placeholder",
    },
    "plant_c_hours_volume_reading": {
        "value": 8760.0,
        "band": (6000.0, 8760.0),
        "source": "volume reading: lines run at the stated flow all year (planned plant)",
        "status": "placeholder",
    },
    "plant_c_reading_factor": {
        "value": 4.9,
        "band": (4.9, 4.9),
        "source": "Befund 99: exhaust of plant C by volume flow is 4.9 times the partner's annual "
        "figure (999 against 205 kW mean); annual reading = volume reading / 4.9",
        "status": "derived (contradiction of the partner data, both readings carried)",
    },
    "site_gas_gwh_per_year": {
        "value": 30.0,
        "band": (30.0, 30.0),
        "source": "partner (site about 30 GWh/a gas)",
        "status": "partner statement, order of magnitude",
    },
    "site_co2_t_per_year": {
        "value": 11500.0,
        "band": (11500.0, 11500.0),
        "source": "partner (about 11 500 t CO2/a)",
        "status": "partner statement, order of magnitude",
    },
    "co2_price_eur_per_t": {
        "value": (55.0, 300.0),
        "band": (55.0, 300.0),
        "source": "partner (55 EUR/t now, 300 EUR/t assumed for 2030)",
        "status": "partner statement",
    },
    "exhaust_temperatures_celsius": {
        "value": (34.4, 45.0, 60.7),
        "band": (30.0, 65.0),
        "source": "34.4 and 60.7: measured exhaust outlet of plant A on the two days of docs/19 "
        "(product 2 / product 1); 45: middle placeholder",
        "status": "measurement (two days) and placeholder",
    },
    "humidity_pickup_kg_per_kg": {
        "value": (0.005, 0.010, 0.020),
        "band": (0.005, 0.020),
        "source": "evaporated water per kg dry air of a spray granulator; no plant figure (the "
        "exhaust humidity of plant A is reconstructed, docs/19)",
        "status": "placeholder",
    },
    "exhaust_relative_humidity_limit": {
        "value": (0.6, 0.9),
        "band": (0.5, 0.95),
        "source": "a design limit of the exhaust relative humidity (product moisture, condensation "
        "in ducts and filter); no partner figure",
        "status": "placeholder",
    },
}


def _config() -> ProcessConfig:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    return load_process_config(CONFIG_PATH)


def _load(name: str) -> dict[str, Any] | None:
    for suffix in ("", "_maxiter50"):
        path = RUNS_DIR / f"{name}{suffix}.json"
        if path.exists():
            data: dict[str, Any] = json.loads(path.read_text("utf-8"))
            data["_artifact"] = path.stem
            return data
    return None


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _co2_2030(result: dict[str, Any]) -> float:
    return float(
        ELECTRICITY_2030_CO2_KG_PER_KWH * result["electric_kw"]
        + GAS_CO2_KG_PER_KWH * result["gas_kw"]
    )


def _ok(result: dict[str, Any]) -> bool:
    return (not result.get("solve_failed")) and bool(result.get("feasible"))


def _by_key(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["key"]: r for r in results}


NEAR_KEYS = (
    "moisture",
    "gas_velocity_box_min",
    "inlet_temperature_box_min",
    "inlet_temperature_box_max",
    "bed_temperature",
    "fog",
    "recirculation_box_max",
)


def _near(result: dict[str, Any], settings: dict[str, Any]) -> dict[str, bool]:
    """Looser "at the limit" flags than the run's (1e-3 m/s, 0.05 K): the reduced DE budget
    places box optima only to about 1e-2. Thresholds: moisture within 1e-3 kg/kg of ``L - m``,
    gas velocity within 0.01 m/s of the box minimum, inlet within 0.5 K of a box bound, bed within
    0.5 K of 80 degC, mixing ratio above 0.99, r within 0.01 of the box maximum 0.95."""
    threshold = settings["quality_limit_kg_per_kg"] - settings["margin_kg_per_kg"]
    u_min = settings["gas_velocity_min_main_m_per_s"]
    return {
        "moisture": result["particle_moisture_kg_per_kg"] >= threshold - 1.0e-3,
        "gas_velocity_box_min": result["gas_velocity_m_per_s"] <= u_min + 0.01,
        "inlet_temperature_box_min": result["inlet_temperature_celsius"] <= 60.5,
        "inlet_temperature_box_max": result["inlet_temperature_celsius"] >= 119.5,
        "bed_temperature": result["bed_temperature_celsius"] >= 79.5,
        "fog": result.get("mixing_saturation_ratio", 0.0) >= 0.99,
        "recirculation_box_max": result["recirculation_fraction"] >= 0.94,
    }


# ----------------------------------------------------------------------------- part 1, 3


def grid_tables(main: dict[str, Any]) -> dict[str, Any]:
    res = _by_key(main["results"])
    rows = []
    for r in (*R_GRID, 0.9):
        tag = f"G_r{round(r * 100):03d}"
        savings, savings_2030, savings_kwh, saving_gas = [], [], [], []
        binding = dict.fromkeys(BINDING_KEYS, 0)
        near = dict.fromkeys(NEAR_KEYS, 0)
        n_ok = 0
        t_in, y_in, dew, moisture, t_mix = [], [], [], [], []
        for i in CONTEXTS:
            base, cur = res[f"G_r000_{i}"], res[f"{tag}_{i}"]
            if not _ok(cur):
                continue
            n_ok += 1
            for key in BINDING_KEYS:
                binding[key] += int(cur["binding"][key])
            for key, flag in _near(cur, main["settings"]).items():
                near[key] += int(flag)
            t_in.append(cur["inlet_temperature_celsius"])
            y_in.append(cur["inlet_humidity_kg_per_kg"])
            dew.append(cur["dew_point_celsius"])
            moisture.append(cur["particle_moisture_kg_per_kg"])
            if cur.get("mixing_temperature_celsius") is not None:
                t_mix.append(cur["mixing_temperature_celsius"])
            if _ok(base):
                savings.append(100.0 * (1.0 - cur["co2_kg_per_h"] / base["co2_kg_per_h"]))
                savings_2030.append(100.0 * (1.0 - _co2_2030(cur) / _co2_2030(base)))
                savings_kwh.append(100.0 * (1.0 - cur["kwh_sum_kw"] / base["kwh_sum_kw"]))
                saving_gas.append(100.0 * (1.0 - cur["gas_kw"] / base["gas_kw"]))
        rows.append(
            {
                "scale": "lab",
                "r": r,
                "extension": r > 0.8,
                "n_feasible": n_ok,
                "co2_saving_pct": _stats(savings),
                "co2_saving_2030_pct": _stats(savings_2030),
                "kwh_saving_pct": _stats(savings_kwh),
                "heater_gas_saving_pct": _stats(saving_gas),
                "binding_counts": binding,
                "near_limit_counts": near,
                "inlet_temperature_celsius": _stats(t_in),
                "inlet_humidity_kg_per_kg": _stats(y_in),
                "exhaust_dew_point_celsius": _stats(dew),
                "product_moisture_kg_per_kg": _stats(moisture),
                "mixing_temperature_celsius": _stats(t_mix),
            }
        )
    largest_feasible = []
    for i in CONTEXTS:
        feasible = [r for r in (*R_GRID, 0.9) if _ok(res[f"G_r{round(r * 100):03d}_{i}"])]
        largest_feasible.append(max(feasible) if feasible else None)
    return {"rows": rows, "largest_feasible_grid_r_per_context": largest_feasible}


def r_free_table(main: dict[str, Any]) -> dict[str, Any]:
    res = _by_key(main["results"])
    rows = []
    for i in CONTEXTS:
        base, opt = res[f"G_r000_{i}"], res[f"R_{i}"]
        row: dict[str, Any] = {
            "scale": "lab",
            "context_index": i,
            "ambient_temperature_celsius": opt["ambient_temperature_celsius"],
            "ambient_relative_humidity": opt["ambient_relative_humidity"],
            "feasible": _ok(opt) and _ok(base),
        }
        if row["feasible"]:
            row.update(
                {
                    "ambient_humidity_kg_per_kg": opt["fresh_air_humidity_kg_per_kg"],
                    "r_star": opt["recirculation_fraction"],
                    "t_in_r0": base["inlet_temperature_celsius"],
                    "t_in_star": opt["inlet_temperature_celsius"],
                    "u_r0": base["gas_velocity_m_per_s"],
                    "u_star": opt["gas_velocity_m_per_s"],
                    "co2_r0_kg_per_h": base["co2_kg_per_h"],
                    "co2_star_kg_per_h": opt["co2_kg_per_h"],
                    "co2_saving_pct": 100.0 * (1.0 - opt["co2_kg_per_h"] / base["co2_kg_per_h"]),
                    "co2_saving_2030_pct": 100.0 * (1.0 - _co2_2030(opt) / _co2_2030(base)),
                    "kwh_saving_pct": 100.0 * (1.0 - opt["kwh_sum_kw"] / base["kwh_sum_kw"]),
                    "heater_gas_saving_pct": 100.0 * (1.0 - opt["gas_kw"] / base["gas_kw"]),
                    "electric_change_kw": opt["electric_kw"] - base["electric_kw"],
                    "recirculation_fan_kw": opt["power_recirculation_fan_kw"],
                    "binding": opt["binding"],
                    "binding_r0": base["binding"],
                    "near_star": _near(opt, main["settings"]),
                    "near_r0": _near(base, main["settings"]),
                    "inlet_humidity_kg_per_kg": opt["inlet_humidity_kg_per_kg"],
                    "outlet_humidity_kg_per_kg": opt["outlet_humidity_kg_per_kg"],
                    "dew_point_r0_celsius": base["dew_point_celsius"],
                    "dew_point_star_celsius": opt["dew_point_celsius"],
                    "mixing_saturation_ratio": opt["mixing_saturation_ratio"],
                    "bed_temperature_celsius": opt["bed_temperature_celsius"],
                }
            )
        rows.append(row)
    ok = [r for r in rows if r["feasible"]]
    summary = {
        "n_feasible": len(ok),
        "r_star": _stats([r["r_star"] for r in ok]),
        "co2_saving_pct": _stats([r["co2_saving_pct"] for r in ok]),
        "co2_saving_2030_pct": _stats([r["co2_saving_2030_pct"] for r in ok]),
        "kwh_saving_pct": _stats([r["kwh_saving_pct"] for r in ok]),
        "heater_gas_saving_pct": _stats([r["heater_gas_saving_pct"] for r in ok]),
        "binding_counts_star": {k: sum(int(r["binding"][k]) for r in ok) for k in BINDING_KEYS},
        "binding_counts_r0": {k: sum(int(r["binding_r0"][k]) for r in ok) for k in BINDING_KEYS},
        "near_counts_star": {k: sum(int(r["near_star"][k]) for r in ok) for k in NEAR_KEYS},
        "near_counts_r0": {k: sum(int(r["near_r0"][k]) for r in ok) for k in NEAR_KEYS},
        "dew_point_rise_k": _stats(
            [r["dew_point_star_celsius"] - r["dew_point_r0_celsius"] for r in ok]
        ),
    }
    # correlation of r* with the ambient absolute humidity (descriptive)
    if len(ok) >= 3:
        summary["spearman_r_star_vs_ambient_humidity"] = _spearman(
            [r["ambient_humidity_kg_per_kg"] for r in ok], [r["r_star"] for r in ok]
        )
        summary["spearman_saving_vs_ambient_temperature"] = _spearman(
            [r["ambient_temperature_celsius"] for r in ok], [r["co2_saving_pct"] for r in ok]
        )
    return {"rows": rows, "summary": summary, "envelope": envelope_table(main)}


def envelope_table(main: dict[str, Any]) -> dict[str, Any]:
    """DECLARED POST HOC (added after the main artifact was seen, before this script ran).

    The arm R (r free, reduced DE budget) is not always the best point it could reach: at some
    warm contexts it returns a CO2 above that of r = 0, which lies inside its box (a local optimum
    of the three-variable search; the anchor checked cold and medium contexts only). The envelope
    takes per context the best feasible point over R and every grid arm G (same objective,
    same constraints); ``source`` says which arm supplied it, and the count of contexts where R
    was beaten by the grid is reported as a measure of the optimiser's deficit. No optimum is
    re-computed.
    """
    res = _by_key(main["results"])
    rows: list[dict[str, Any]] = []
    for i in CONTEXTS:
        base = res[f"G_r000_{i}"]
        candidates = [res[f"R_{i}"]] + [res[f"G_r{round(r * 100):03d}_{i}"] for r in (*R_GRID, 0.9)]
        feasible = [c for c in candidates if _ok(c)]
        if not (_ok(base) and feasible):
            rows.append({"scale": "lab", "context_index": i, "feasible": False})
            continue
        best = min(feasible, key=lambda c: c["co2_kg_per_h"])
        r_opt = res[f"R_{i}"]
        rows.append(
            {
                "scale": "lab",
                "context_index": i,
                "feasible": True,
                "ambient_temperature_celsius": base["ambient_temperature_celsius"],
                "ambient_relative_humidity": base["ambient_relative_humidity"],
                "source": best["key"].rsplit("_", 1)[0],
                "r_env": best["recirculation_fraction"],
                "co2_saving_pct": 100.0 * (1.0 - best["co2_kg_per_h"] / base["co2_kg_per_h"]),
                "heater_gas_saving_pct": 100.0 * (1.0 - best["gas_kw"] / base["gas_kw"]),
                "co2_saving_2030_pct": 100.0 * (1.0 - _co2_2030(best) / _co2_2030(base)),
                "r_free_deficit_pct": 100.0
                * (r_opt["co2_kg_per_h"] - best["co2_kg_per_h"])
                / base["co2_kg_per_h"]
                if _ok(r_opt)
                else None,
                "binding": best["binding"],
                "near": _near(best, main["settings"]),
            }
        )
    ok = [r for r in rows if r["feasible"]]
    return {
        "rows": rows,
        "summary": {
            "n_feasible": len(ok),
            "n_r_free_beaten_by_grid": sum(r["source"] != "R" for r in ok),
            "n_r_free_beaten_by_more_than_0p1_pct": sum(
                (r["r_free_deficit_pct"] or 0.0) > 0.1 for r in ok
            ),
            "r_free_deficit_pct": _stats(
                [r["r_free_deficit_pct"] for r in ok if r["r_free_deficit_pct"] is not None]
            ),
            "r_env": _stats([r["r_env"] for r in ok]),
            "co2_saving_pct": _stats([r["co2_saving_pct"] for r in ok]),
            "heater_gas_saving_pct": _stats([r["heater_gas_saving_pct"] for r in ok]),
            "co2_saving_2030_pct": _stats([r["co2_saving_2030_pct"] for r in ok]),
            "near_counts": {k: sum(int(r["near"][k]) for r in ok) for k in NEAR_KEYS},
            "n_r_env_zero": sum(r["r_env"] == 0.0 for r in ok),
        },
    }


def _spearman(x: list[float], y: list[float]) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------------------------- part 2


def _moisture_at(config: ProcessConfig, row: dict[str, Any], r: float) -> float | None:
    try:
        solution = evaluate_point(
            row["inlet_temperature_celsius"],
            row["gas_velocity_m_per_s"],
            FIXED_NOZZLE_PRESSURE_BAR,
            row["ambient_temperature_celsius"],
            row["ambient_relative_humidity"],
            max_iterations=DEFAULT_SEARCH_MAX_ITERATIONS,
            tolerance=DEFAULT_SEARCH_TOLERANCE,
            base_config=with_recirculation(config, r),
        )
    except (RuntimeError, ValueError):
        return None
    if not solution.feasible:
        return None
    return solution.particle_moisture_kg_per_kg


def limit_at_unchanged_set_points(config: ProcessConfig, main: dict[str, Any]) -> dict[str, Any]:
    """Largest r keeping the moisture below a threshold at the r = 0 optimum's set points."""
    res = _by_key(main["results"])
    margin = main["settings"]["margin_kg_per_kg"]
    limit = main["settings"]["quality_limit_kg_per_kg"]
    rows = []
    for i in CONTEXTS:
        base = res[f"G_r000_{i}"]
        row: dict[str, Any] = {"scale": "lab", "context_index": i}
        if not _ok(base):
            rows.append(row)
            continue
        for label, threshold in (("limit_minus_margin", limit - margin), ("limit", limit)):

            def admissible(
                r: float, threshold: float = threshold, base: dict[str, Any] = base
            ) -> bool:
                moisture = _moisture_at(config, base, r)
                return moisture is not None and moisture <= threshold + 1.0e-4

            if not admissible(0.0):
                row[label] = 0.0
                continue
            if admissible(LIMIT_R_MAX):
                row[label] = LIMIT_R_MAX
                continue
            low, high = 0.0, LIMIT_R_MAX
            for _ in range(LIMIT_BISECTION_STEPS):
                mid = 0.5 * (low + high)
                if admissible(mid):
                    low = mid
                else:
                    high = mid
            row[label] = low
        row["r0_moisture_kg_per_kg"] = base["particle_moisture_kg_per_kg"]
        row["r0_binding_moisture"] = base["binding"]["moisture"]
        rows.append(row)
    ok = [r for r in rows if "limit" in r]
    return {
        "rows": rows,
        "summary": {
            "limit_minus_margin": _stats([r["limit_minus_margin"] for r in ok]),
            "limit": _stats([r["limit"] for r in ok]),
            "n_zero_limit_minus_margin": sum(r["limit_minus_margin"] == 0.0 for r in ok),
        },
    }


# ------------------------------------------------------------------------------- part 4


def _solve_at(config: ProcessConfig, point: dict[str, Any], r: float) -> Any:
    """Steady state at the stored set points of one result (search precision of the run)."""
    return evaluate_point(
        point["inlet_temperature_celsius"],
        point["gas_velocity_m_per_s"],
        FIXED_NOZZLE_PRESSURE_BAR,
        point["ambient_temperature_celsius"],
        point["ambient_relative_humidity"],
        max_iterations=DEFAULT_SEARCH_MAX_ITERATIONS,
        tolerance=DEFAULT_SEARCH_TOLERANCE,
        base_config=with_recirculation(config, r),
    )


def recovery_tables(config: ProcessConfig, main: dict[str, Any]) -> dict[str, Any]:
    res = _by_key(main["results"])
    k = main["settings"]["scale_factor_k"]
    spec = RecoverySpec(**main["recovery_spec"])
    joint = []
    for i in CONTEXTS:
        j0, j, g0, rr = res[f"J0_{i}"], res[f"J_{i}"], res[f"G_r000_{i}"], res[f"R_{i}"]
        row: dict[str, Any] = {"scale": "lab x k", "context_index": i}
        if all(_ok(x) for x in (j0, j, g0, rr)):
            row.update(
                {
                    "r_star_with_recovery": j["recirculation_fraction"],
                    "r_star_without_recovery": rr["recirculation_fraction"],
                    "co2_j0_kg_per_h": j0["recovery"]["co2_kg_per_h"],
                    "co2_j_kg_per_h": j["recovery"]["co2_kg_per_h"],
                    "co2_g0_kg_per_h": k * g0["co2_kg_per_h"],
                    "co2_r_kg_per_h": k * rr["co2_kg_per_h"],
                    "recovered_j0_kw": j0["recovery"]["recovered_kw"],
                    "recovered_j_kw": j["recovery"]["recovered_kw"],
                    "saving_recirculation_with_recovery_pct": 100.0
                    * (1.0 - j["recovery"]["co2_kg_per_h"] / j0["recovery"]["co2_kg_per_h"]),
                    "saving_recirculation_without_recovery_pct": 100.0
                    * (1.0 - rr["co2_kg_per_h"] / g0["co2_kg_per_h"]),
                    "saving_recovery_without_recirculation_pct": 100.0
                    * (1.0 - j0["recovery"]["co2_kg_per_h"] / (k * g0["co2_kg_per_h"])),
                    "saving_both_pct": 100.0
                    * (1.0 - j["recovery"]["co2_kg_per_h"] / (k * g0["co2_kg_per_h"])),
                    # post-hoc envelope (see envelope_table): J0 lies inside the box of J
                    "j_beaten_by_j0": j["recovery"]["co2_kg_per_h"]
                    > j0["recovery"]["co2_kg_per_h"],
                    "saving_recirculation_with_recovery_env_pct": 100.0
                    * (
                        1.0
                        - min(j["recovery"]["co2_kg_per_h"], j0["recovery"]["co2_kg_per_h"])
                        / j0["recovery"]["co2_kg_per_h"]
                    ),
                    "saving_both_env_pct": 100.0
                    * (
                        1.0
                        - min(j["recovery"]["co2_kg_per_h"], j0["recovery"]["co2_kg_per_h"])
                        / (k * g0["co2_kg_per_h"])
                    ),
                    "binding_j": j["binding"],
                    "dew_point_j0_celsius": j0["dew_point_celsius"],
                    "dew_point_j_celsius": j["dew_point_celsius"],
                }
            )
        joint.append(row)
    ok = [r for r in joint if "co2_j_kg_per_h" in r]
    joint_summary = {
        key: _stats([r[key] for r in ok])
        for key in (
            "r_star_with_recovery",
            "r_star_without_recovery",
            "saving_recirculation_with_recovery_pct",
            "saving_recirculation_without_recovery_pct",
            "saving_recovery_without_recirculation_pct",
            "saving_both_pct",
            "saving_recirculation_with_recovery_env_pct",
            "saving_both_env_pct",
            "recovered_j0_kw",
            "recovered_j_kw",
        )
    }
    # descriptive: cascade at the grid solutions, both layouts
    grid: list[dict[str, Any]] = []
    base_solutions: dict[int, Any] = {}
    for r in R_GRID:
        tag = f"G_r{round(r * 100):03d}"
        per_layout: dict[str, dict[str, list[float]]] = {
            layout: {"recovered_kw": [], "co2_after_kg_per_h": [], "saving_vs_r0_after_pct": []}
            for layout in ("fresh_air", "mixed_air")
        }
        for i in CONTEXTS:
            cur, base = res[f"{tag}_{i}"], res[f"G_r000_{i}"]
            if not (_ok(cur) and _ok(base)):
                continue
            if i not in base_solutions:
                base_solutions[i] = _solve_at(config, base, 0.0)
            current = _solve_at(config, cur, r)
            for layout in ("fresh_air", "mixed_air"):
                after = [
                    recovery_balance(
                        solution,
                        with_recirculation(config, rr_),
                        dataclasses.replace(spec, layout=layout),
                        t_amb=point["ambient_temperature_celsius"],
                        t_in=point["inlet_temperature_celsius"],
                    )
                    for solution, point, rr_ in (
                        (current, cur, r),
                        (base_solutions[i], base, 0.0),
                    )
                ]
                if after[0]["rejected"] or after[1]["rejected"]:
                    continue
                per_layout[layout]["recovered_kw"].append(after[0]["recovered_kw"])
                per_layout[layout]["co2_after_kg_per_h"].append(after[0]["co2_kg_per_h"])
                per_layout[layout]["saving_vs_r0_after_pct"].append(
                    100.0 * (1.0 - after[0]["co2_kg_per_h"] / after[1]["co2_kg_per_h"])
                )
        grid.append(
            {
                "scale": "lab x k",
                "r": r,
                **{
                    layout: {key: _stats(vals) for key, vals in values.items()}
                    for layout, values in per_layout.items()
                },
            }
        )
    return {"joint_rows": joint, "joint_summary": joint_summary, "grid_descriptive": grid}


# ------------------------------------------------------------------------------- part 5


def sensitivity_tables(main: dict[str, Any], sens: dict[str, Any]) -> dict[str, Any]:
    res_main = _by_key(main["results"])
    res = _by_key(sens["results"])

    def summary(get: Any) -> dict[str, Any]:
        r_star, saving, binding = [], [], dict.fromkeys(BINDING_KEYS, 0)
        saving_env: list[float] = []
        beaten = 0
        n = 0
        for i in SUBGRID:
            base, opt = get(i)
            if not (_ok(base) and _ok(opt)):
                continue
            n += 1
            r_star.append(opt["recirculation_fraction"])
            saving.append(100.0 * (1.0 - opt["co2_kg_per_h"] / base["co2_kg_per_h"]))
            # post-hoc envelope over R and r = 0 (r = 0 lies inside the box of R)
            saving_env.append(
                100.0
                * (1.0 - min(opt["co2_kg_per_h"], base["co2_kg_per_h"]) / base["co2_kg_per_h"])
            )
            beaten += int(opt["co2_kg_per_h"] > base["co2_kg_per_h"])
            for key in BINDING_KEYS:
                binding[key] += int(opt["binding"][key])
        return {
            "n_feasible": n,
            "r_star": _stats(r_star),
            "co2_saving_pct": _stats(saving),
            "co2_saving_env_pct": _stats(saving_env),
            "n_r_free_beaten_by_r0": beaten,
            "binding_counts": binding,
        }

    reference = summary(lambda i: (res_main[f"G_r000_{i}"], res_main[f"R_{i}"]))
    variants: dict[str, Any] = {}
    for name in sens["variants"]:
        variants[name] = summary(lambda i, n=name: (res[f"S_{n}_r0_{i}"], res[f"S_{n}_R_{i}"]))
        ref_saving = reference["co2_saving_env_pct"]["median"]
        ref_r = reference["r_star"]["median"]
        cur_saving = variants[name]["co2_saving_env_pct"]["median"]
        cur_r = variants[name]["r_star"]["median"]
        variants[name]["delta_median_saving_pp"] = (
            None if cur_saving is None or ref_saving is None else cur_saving - ref_saving
        )
        variants[name]["delta_median_r_star"] = (
            None if cur_r is None or ref_r is None else cur_r - ref_r
        )
    ranking = sorted(
        (n for n in variants if variants[n]["delta_median_saving_pp"] is not None),
        key=lambda n: -abs(variants[n]["delta_median_saving_pp"]),
    )
    return {
        "scale": "lab",
        "subgrid": list(SUBGRID),
        "reference_main_on_subgrid": reference,
        "variants": variants,
        "ranking_by_abs_delta_saving": ranking,
    }


# ------------------------------------------------------------------------------- part 6


def _humidity(t: float, phi: float, config: ProcessConfig) -> float:
    return float(absolute_humidity_kg_per_kg(_BACKEND, config.gas, np.asarray(t), np.asarray(phi)))


def transfer_tables(config: ProcessConfig, saving_fractions: dict[str, float]) -> dict[str, Any]:
    """Placeholder grid at partner orders of magnitude; ``saving_fractions``: heater-gas saving."""
    tp = {k: v["value"] for k, v in TRANSFER.items()}
    gas = config.gas
    t_a = tp["ambient_annual_mean_celsius"]
    y_a = _humidity(t_a, tp["ambient_relative_humidity"], config)
    c_a = gas.cp_dry_air_j_per_kg_k + y_a * gas.cp_vapour_j_per_kg_k
    rho_n = tp["normal_density_kg_per_nm3"]
    prices = tp["co2_price_eur_per_t"]

    def plant(flow_nm3h: float, t_in: float, eff: float, hours: float) -> dict[str, float]:
        m = flow_nm3h * rho_n / 3600.0 / (1.0 + y_a)  # dry air, kg/s
        duty_kw = m * c_a * (t_in - t_a) / 1000.0
        return {
            "dry_air_kg_per_s": m,
            "heater_duty_kw": duty_kw,
            "fuel_mwh_per_year": duty_kw / eff * hours / 1000.0,
        }

    plants = {
        "A": plant(
            tp["plant_a_air_nm3_per_h"],
            tp["plant_a_inlet_celsius"],
            tp["plant_a_heat_supply_efficiency"],
            tp["plant_a_hours_per_year"],
        ),
        "C_volume_reading": plant(
            tp["plant_c_air_nm3_per_h_per_line"] * tp["plant_c_lines"],
            tp["plant_c_inlet_celsius"],
            tp["plant_c_burner_efficiency"],
            tp["plant_c_hours_volume_reading"],
        ),
    }
    c_annual = dict(plants["C_volume_reading"])
    c_annual["fuel_mwh_per_year"] /= tp["plant_c_reading_factor"]
    c_annual["heater_duty_kw"] /= tp["plant_c_reading_factor"]
    plants["C_annual_reading"] = c_annual
    rows = []
    for plant_name, p in plants.items():
        for label, fraction in saving_fractions.items():
            fuel = p["fuel_mwh_per_year"] * fraction
            co2_t = fuel * GAS_CO2_KG_PER_KWH  # MWh * kg/kWh = t
            rows.append(
                {
                    "scale": "transfer",
                    "plant": plant_name,
                    "saving_fraction_label": label,
                    "heater_gas_saving_fraction": fraction,
                    "fuel_without_mwh_per_year": p["fuel_mwh_per_year"],
                    "fuel_saving_mwh_per_year": fuel,
                    "co2_saving_t_per_year": co2_t,
                    "share_of_site_gas_pct": 100.0 * fuel / (tp["site_gas_gwh_per_year"] * 1000.0),
                    "share_of_site_co2_pct": 100.0 * co2_t / tp["site_co2_t_per_year"],
                    "value_keur_per_year": {f"{int(pr)}": co2_t * pr / 1000.0 for pr in prices},
                }
            )
    # pure mixing saving at plant temperatures (upper bound without the moisture constraint)
    mixing = []
    for t_out in tp["exhaust_temperatures_celsius"]:
        for t_in_name, t_in in (
            ("A", tp["plant_a_inlet_celsius"]),
            ("C", tp["plant_c_inlet_celsius"]),
        ):
            for r in (0.2, 0.4, 0.6):
                # exhaust humidity: saturated at 34.4 degC (product-2 day), else pickup 0.010
                y_o = min(
                    y_a + 0.010,
                    float(saturation_humidity_kg_per_kg(_BACKEND, gas, np.asarray(t_out))),
                )
                c_o = gas.cp_dry_air_j_per_kg_k + y_o * gas.cp_vapour_j_per_kg_k
                t_mix = t_a + r * c_o * (t_out - t_a) / ((1.0 - r) * c_a + r * c_o)
                mixing.append(
                    {
                        "scale": "transfer",
                        "inlet": t_in_name,
                        "exhaust_celsius": t_out,
                        "r": r,
                        "mixing_celsius": t_mix,
                        "duty_saving_pct_upper_bound": 100.0 * (t_mix - t_a) / (t_in - t_a),
                    }
                )
    # humidity headroom of the exhaust: Y_out(r) = Y_a + dY / (1 - r) <= phi_lim Y_s(T_out)
    headroom = []
    for t_out in tp["exhaust_temperatures_celsius"]:
        y_s = float(saturation_humidity_kg_per_kg(_BACKEND, gas, np.asarray(t_out)))
        for phi_lim in tp["exhaust_relative_humidity_limit"]:
            y_lim = _humidity(t_out, phi_lim, config)
            for d_y in tp["humidity_pickup_kg_per_kg"]:
                fresh_out = y_a + d_y
                r_max = 0.0 if fresh_out >= y_lim else max(0.0, 1.0 - d_y / (y_lim - y_a))
                headroom.append(
                    {
                        "scale": "transfer",
                        "exhaust_celsius": t_out,
                        "exhaust_rh_limit": phi_lim,
                        "humidity_pickup_kg_per_kg": d_y,
                        "fresh_air_exhaust_rh": min(1.0, fresh_out / y_s),
                        "r_max": r_max,
                    }
                )
    return {
        "placeholders": TRANSFER,
        "ambient_humidity_kg_per_kg": y_a,
        "plants": plants,
        "rows": rows,
        "mixing_upper_bound": mixing,
        "humidity_headroom": headroom,
        "note": "transfer of LAB saving fractions to partner orders of magnitude; no plant model",
    }


def main() -> int:
    require_clean_tree()
    config = _config()
    main_art = _load("exhaust_recirc_main")
    sens_art = _load("exhaust_recirc_sensitivity")
    anchor = _load("exhaust_recirc_anchor")
    assert main_art is not None and sens_art is not None and anchor is not None
    run_config = dataclasses.replace(
        config,
        energy=dataclasses.replace(
            config.energy,
            recirculation_duct_loss_coefficient=main_art["settings"][
                "recirculation_duct_loss_coefficient"
            ],
        ),
    )
    assert run_config.fingerprint == main_art["settings"]["config_fingerprint"]
    grid = grid_tables(main_art)
    r_free = r_free_table(main_art)
    print("grid and r-free done", flush=True)
    limit = limit_at_unchanged_set_points(run_config, main_art)
    print("limit done", flush=True)
    recovery = recovery_tables(run_config, main_art)
    print("recovery done", flush=True)
    sensitivity = sensitivity_tables(main_art, sens_art)
    s = r_free["envelope"]["summary"]["heater_gas_saving_pct"]
    fractions = {
        "lab_r_star_p10": s["p10"] / 100.0,
        "lab_r_star_median": s["median"] / 100.0,
        "lab_r_star_p90": s["p90"] / 100.0,
    }
    transfer = transfer_tables(run_config, fractions)
    write_artifact(
        "exhaust_recirc_analysis",
        {
            "_provenance": provenance("not used"),
            "inputs": {
                "anchor": anchor["_artifact"],
                "main": main_art["_artifact"],
                "sensitivity": sens_art["_artifact"],
            },
            "anchor_gate_passed": anchor["gate_passed"],
            "anchor_rows": anchor["rows"],
            "electricity_2030_co2_kg_per_kwh": ELECTRICITY_2030_CO2_KG_PER_KWH,
            "electricity_co2_kg_per_kwh": ELECTRICITY_CO2_KG_PER_KWH,
            "grid": grid,
            "r_free": r_free,
            "limit_at_unchanged_set_points": limit,
            "recovery": recovery,
            "sensitivity": sensitivity,
            "transfer": transfer,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

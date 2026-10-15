r"""Dryer block of the partner's equation document (1.3, 1.5) against the fluidised-bed model.

Read-only study, no training, one process. VG0-VG4 of the task (docs/11, section 9):

VG0  property sets of both models, reference states, mapping of the bed model's quantities to
     the block's inputs (written to the artifact; the tables in docs/11 are made from it).
VG1  steady states of the bed model, and the block with the same inputs, at
     * the 25 disturbance scenarios of Befund 52 at the fixed operating point F and at the
       oracle M0 of M1, both at the gas-velocity lower bound 1.26 m/s (artifact
       ``m1_margin_cost_l015_umin126``, quality limit 0.15; the set points are read from it,
       the steady states are solved again at the search precision of Befund 52 and checked
       against the stored ones),
     * (extra) the 25 scenarios at the corner 60 degC, 1.26 m/s of the grid below,
     * a 5 x 5 grid over inlet temperature 60-120 degC and gas velocity 1.26-1.88 m/s at the
       nominal ambient state (20 degC, 0.5) and nominal spray (1.5 kg/h, solid fraction 0.3).
VG2  stepwise decomposition from the block as delivered to the bed model (a: delivered; b: gas
     properties; b2: the remaining properties; c: product moisture of the bed instead of the
     equilibrium moisture; d: heat loss; e: remainder = structure), in the order given and in
     the reversed order (the decomposition is sequential, so the order is varied).
VG3  sensitivities (finite differences, one-sided, same stencil for both models) of exhaust
     temperature and heating power to inlet temperature and gas velocity on the grid; and the
     points where the bed model reports a violated moisture limit or no steady state while the
     block shows normal operation.
VG4  the inputs of the partner's fluidised bed (300 degC, 12 kg/h spray, 100 kg/h air) in the
     bed model, only if the solver converges, marked as extrapolation; else reported as outside
     the range.

All powers are thermal (no efficiency). ``T_cold`` is a placeholder (29.1 degC of the site
model, 15 degC of the heat-network default); both are reported.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq

from twinlink_ml.evaluation import dryer_block_comparison as dbc
from twinlink_ml.partner_v4_1.dryer_block import BlockInputs, BlockProperties, dryer_block
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.backend import NumpyBackend
from twinlink_ml.simulation.config import ProcessConfig
from twinlink_ml.simulation.greybox import GreyBoxFluidBedGranulator
from twinlink_ml.simulation.operating_point import (
    DEFAULT_SEARCH_MAX_ITERATIONS,
    DEFAULT_SEARCH_TOLERANCE,
    TOLERANCE_MOISTURE_KG_PER_KG,
    build_variant_config,
)
from twinlink_ml.simulation.oracle_search import build_grid
from twinlink_ml.simulation.steady_state import SteadyStateSolution, solve_steady_state
from twinlink_ml.simulation.thermo import (
    absolute_humidity_kg_per_kg,
    relative_humidity_from_absolute_humidity,
)
from twinlink_ml.utils.tracking import environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
M1_ARTIFACT = REPO_ROOT / "artifacts" / "runs" / "m1_margin_cost_l015_umin126.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "runs" / "dryer_block_vs_bed.json"

T_COLDS = (dbc.T_COLD_SITE_CELSIUS, dbc.T_COLD_HEATNETWORK_DEFAULT_CELSIUS)
QUALITY_LIMIT_KG_PER_KG = 0.15
"""Quality limit of the M1 main case (Befund 69); the kinetic boundary 0.20 is the critical moisture."""
GRID_T_IN = (60.0, 120.0)
GRID_U = (1.26, 1.88)
GRID_N = 5
NOMINAL_T_AMB = 20.0
NOMINAL_PHI = 0.5
NOZZLE_BAR = 1.0
DELTA_T_IN_K = 1.0
DELTA_U_M_PER_S = 0.02
BED_TEMPERATURE_VALIDITY_C = 80.0
"""Above it the drying curve is outside its documented validity (``max_driving_humidity``)."""

COMPARED = (
    "outlet_temperature_celsius",
    "outlet_humidity_kg_per_kg",
    "product_moisture_kg_per_kg",
    "heater_power_kw",
    "dew_point_celsius",
    *(
        f"recovery_{kind}_kw_tcold_{t:g}"
        for t in T_COLDS
        for kind in ("sensible", "latent", "total")
    ),
)


def _clean(value: Any) -> Any:
    """JSON-safe: NaN/inf to None, numpy to Python."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.floating | float):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _solve(
    config: ProcessConfig, t_in: float, u: float, p_nozzle: float, t_amb: float, phi: float
) -> tuple[ProcessConfig, SteadyStateSolution | None, str | None]:
    variant = build_variant_config(
        config, t_in=t_in, u_gas=u, p_nozzle=p_nozzle, t_amb=t_amb, phi_amb=phi
    )
    try:
        solution = solve_steady_state(
            GreyBoxFluidBedGranulator(variant),
            max_iterations=DEFAULT_SEARCH_MAX_ITERATIONS,
            tolerance=DEFAULT_SEARCH_TOLERANCE,
        )
    except (RuntimeError, ValueError) as error:
        return variant, None, f"{type(error).__name__}: {str(error)[:200]}"
    return variant, solution, None


def _chain(
    order: tuple[str, ...], own_inputs: BlockInputs, own_props: BlockProperties, gas: Any
) -> list[dict[str, Any]]:
    """Quantities after 0, 1, ..., len(order) steps (entry 0: the block as delivered)."""
    chain = []
    for k in range(len(order) + 1):
        inputs, props = dbc.apply_steps(order[:k], own_inputs, own_props)
        chain.append(
            {
                "after_step": "delivered" if k == 0 else order[k - 1],
                "q": dbc.evaluate_block(gas, inputs, props, T_COLDS),
            }
        )
    return chain


def point_record(
    config: ProcessConfig, variant: ProcessConfig, solution: SteadyStateSolution
) -> dict[str, Any]:
    """Own steady state, the block chain in both orders, flags."""
    gas = config.gas
    own_props = dbc.own_properties(variant)
    own_inputs = dbc.block_inputs(variant, solution)
    own_res = dbc.own_result(own_inputs, solution, own_props)
    own_q = dbc.quantities(gas, own_inputs, own_props, own_res, T_COLDS)
    diagnostics = solution.diagnostics
    mass_closure = solution.outlet_humidity_kg_per_kg - (
        own_inputs.air_humidity_kg_per_kg
        + diagnostics["evaporation_rate_kg_per_s"] / own_inputs.dry_air_mass_flow_kg_per_s
    )
    delivered_in, delivered_props = dbc.apply_steps((), own_inputs, own_props)
    delivered_flags = dbc.block_is_normal(gas, dryer_block(delivered_in, delivered_props))
    full_in, full_props = dbc.apply_steps(dbc.STEP_ORDER, own_inputs, own_props)
    full_flags = dbc.block_is_normal(gas, dryer_block(full_in, full_props))
    equilibrium = dbc.own_equilibrium_moisture(
        variant, solution.outlet_humidity_kg_per_kg, solution.bed_temperature_celsius
    )
    alt_in = dataclasses.replace(full_in, product_moisture_kg_per_kg=equilibrium)
    alt_q = dbc.evaluate_block(gas, alt_in, full_props, T_COLDS)
    x_p = solution.particle_moisture_kg_per_kg
    return {
        "own": own_q,
        "own_inputs": dataclasses.asdict(own_inputs),
        "own_equilibrium_moisture_kg_per_kg": equilibrium,
        "own_outlet_humidity_minus_mass_closure_kg_per_kg": mass_closure,
        "own_flags": {
            "solver_feasible": solution.feasible,
            "infeasibility_reasons": list(solution.infeasibility_reasons),
            "above_quality_limit": x_p > QUALITY_LIMIT_KG_PER_KG + TOLERANCE_MOISTURE_KG_PER_KG,
            "above_critical_moisture": x_p
            > variant.material.critical_moisture_kg_per_kg + TOLERANCE_MOISTURE_KG_PER_KG,
            "bed_temperature_above_validity": (
                solution.bed_temperature_celsius > BED_TEMPERATURE_VALIDITY_C
            ),
            "residual_inf_norm": solution.residual_inf_norm,
        },
        "block_delivered_flags": delivered_flags,
        "block_full_flags": full_flags,
        "chain_forward": _chain(dbc.STEP_ORDER, own_inputs, own_props, gas),
        "chain_reverse": _chain(dbc.STEP_ORDER_REVERSED, own_inputs, own_props, gas),
        "after_slope_check": dbc.evaluate_block(
            gas,
            *dbc.apply_steps((*dbc.STEP_ORDER, "e_latent_slope"), own_inputs, own_props),
            T_COLDS,
        ),
        "alt_block_with_own_equilibrium_moisture": alt_q,
    }


def _stats(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0, "median": None, "min": None, "max": None}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Median/range over the points: own, delivered, differences, contributions per step."""
    valid = [r for r in records if r.get("solved") and r["own_flags"]["solver_feasible"]]
    out: dict[str, Any] = {"n_points": len(records), "n_valid": len(valid)}
    for name in COMPARED:
        own = [r["own"][name] for r in valid]
        entry: dict[str, Any] = {
            "own": _stats(own),
            "delivered": _stats([r["chain_forward"][0]["q"][name] for r in valid]),
            "difference_delivered_minus_own": _stats(
                [r["chain_forward"][0]["q"][name] - r["own"][name] for r in valid]
            ),
            "relative_difference_delivered_minus_own": _stats(
                [
                    (r["chain_forward"][0]["q"][name] - r["own"][name]) / r["own"][name]
                    for r in valid
                    if abs(r["own"][name]) > 1e-12
                ]
            ),
        }
        for label, key in (("forward", "chain_forward"), ("reverse", "chain_reverse")):
            steps = [step["after_step"] for step in valid[0][key]][1:] if valid else []
            contributions = {}
            for k, step in enumerate(steps, start=1):
                contributions[step] = _stats(
                    [r[key][k]["q"][name] - r[key][k - 1]["q"][name] for r in valid]
                )
            contributions["structure_remainder"] = _stats(
                [r[key][-1]["q"][name] - r["own"][name] for r in valid]
            )
            entry[f"contributions_{label}"] = contributions
        entry["after_latent_slope_minus_own"] = _stats(
            [r["after_slope_check"][name] - r["own"][name] for r in valid]
        )
        entry["alt_own_equilibrium_moisture_minus_own"] = _stats(
            [r["alt_block_with_own_equilibrium_moisture"][name] - r["own"][name] for r in valid]
        )
        out[name] = entry
    out["own_outlet_humidity_minus_mass_closure"] = _stats(
        [r["own_outlet_humidity_minus_mass_closure_kg_per_kg"] for r in valid]
    )
    return out


def reproduce_m1_points(
    config: ProcessConfig, m1: dict[str, Any], arm: str
) -> list[dict[str, Any]]:
    """The 25 scenarios at the set points of arm 'F' or 'M0' of the M1 artifact."""
    grid = build_grid(config, 5)
    records = []
    for index, (t_amb, phi) in enumerate(grid):
        if arm == "F":
            fixed = m1["arm_F_fixed_point"]
            setpoint = (
                fixed["inlet_temperature_celsius"],
                fixed["gas_velocity_m_per_s"],
                fixed["nozzle_pressure_bar"],
            )
            stored = fixed["per_grid_point"][index]
        else:
            stored = m1["arm_M0_oracle"][index]
            setpoint = (
                stored["inlet_temperature_celsius"],
                stored["gas_velocity_m_per_s"],
                stored["nozzle_pressure_bar"],
            )
        assert abs(stored["ambient_temperature_celsius"] - t_amb) < 1e-9
        assert abs(stored["ambient_relative_humidity"] - phi) < 1e-9
        variant, solution, error = _solve(config, *setpoint, t_amb, phi)
        record: dict[str, Any] = {
            "index": index,
            "ambient_temperature_celsius": t_amb,
            "ambient_relative_humidity": phi,
            "setpoint": {
                "inlet_temperature_celsius": setpoint[0],
                "gas_velocity_m_per_s": setpoint[1],
                "nozzle_pressure_bar": setpoint[2],
            },
            "solved": solution is not None,
            "error": error,
        }
        if solution is not None:
            record.update(point_record(config, variant, solution))
            record["stored_in_m1"] = {
                "bed_temperature_celsius": stored["bed_temperature_celsius"],
                "outlet_humidity_kg_per_kg": stored["outlet_humidity_kg_per_kg"],
                "particle_moisture_kg_per_kg": stored["particle_moisture_kg_per_kg"],
                "dry_air_mass_flow_kg_per_s": stored["dry_air_mass_flow_kg_per_s"],
            }
            record["difference_to_stored_in_m1"] = {
                "bed_temperature_celsius": solution.bed_temperature_celsius
                - stored["bed_temperature_celsius"],
                "outlet_humidity_kg_per_kg": solution.outlet_humidity_kg_per_kg
                - stored["outlet_humidity_kg_per_kg"],
                "particle_moisture_kg_per_kg": solution.particle_moisture_kg_per_kg
                - stored["particle_moisture_kg_per_kg"],
                "dry_air_mass_flow_kg_per_s": solution.dry_air_mass_flow_kg_per_s
                - stored["dry_air_mass_flow_kg_per_s"],
            }
        records.append(record)
    return records


def corner_scenarios(config: ProcessConfig) -> list[dict[str, Any]]:
    """Extra set: the 25 disturbance scenarios at the corner (60 degC, 1.26 m/s) of the grid.

    The oracle of M1 sits on this corner wherever the moisture limit does not bind; the grid of
    VG1 shows its (60, 1.26) point at the quality limit at the nominal ambient state, so this set
    shows how that depends on the ambient state.
    """
    records = []
    for index, (t_amb, phi) in enumerate(build_grid(config, 5)):
        variant, solution, error = _solve(config, GRID_T_IN[0], GRID_U[0], NOZZLE_BAR, t_amb, phi)
        record: dict[str, Any] = {
            "index": index,
            "ambient_temperature_celsius": t_amb,
            "ambient_relative_humidity": phi,
            "solved": solution is not None,
            "error": error,
        }
        if solution is not None:
            record.update(point_record(config, variant, solution))
        records.append(record)
    return records


def grid_points(config: ProcessConfig) -> list[dict[str, Any]]:
    """5 x 5 grid over inlet temperature and gas velocity, with the derivative neighbours."""
    gas = config.gas
    t_values = np.linspace(*GRID_T_IN, GRID_N)
    u_values = np.linspace(*GRID_U, GRID_N)
    records = []
    for t_in in t_values:
        for u in u_values:
            variant, solution, error = _solve(
                config, float(t_in), float(u), NOZZLE_BAR, NOMINAL_T_AMB, NOMINAL_PHI
            )
            record: dict[str, Any] = {
                "inlet_temperature_celsius": float(t_in),
                "gas_velocity_m_per_s": float(u),
                "solved": solution is not None,
                "error": error,
            }
            if solution is None:
                records.append(record)
                continue
            record.update(point_record(config, variant, solution))
            own_props = dbc.own_properties(variant)
            base_inputs = dbc.block_inputs(variant, solution)
            frozen_x_p = base_inputs.product_moisture_kg_per_kg
            sensitivities: dict[str, Any] = {}
            for axis, step, base_value in (
                ("inlet_temperature", DELTA_T_IN_K, float(t_in)),
                ("gas_velocity", DELTA_U_M_PER_S, float(u)),
            ):
                sign = -1.0 if axis == "inlet_temperature" and base_value >= GRID_T_IN[1] else 1.0
                shifted = base_value + sign * step
                args = (
                    (shifted, float(u)) if axis == "inlet_temperature" else (float(t_in), shifted)
                )
                v2, s2, e2 = _solve(config, *args, NOZZLE_BAR, NOMINAL_T_AMB, NOMINAL_PHI)
                if s2 is None:
                    sensitivities[axis] = {"error": e2}
                    continue
                p2 = dbc.block_inputs(v2, s2)
                own2 = dbc.quantities(
                    gas, p2, own_props, dbc.own_result(p2, s2, own_props), T_COLDS
                )
                delivered_in, delivered_props = dbc.apply_steps((), p2, own_props)
                delivered2 = dbc.evaluate_block(gas, delivered_in, delivered_props, T_COLDS)
                # full properties and heat loss, X_p frozen at the base point's value
                full_in, full_props = dbc.apply_steps(dbc.STEP_ORDER, p2, own_props)
                full_in = dataclasses.replace(full_in, product_moisture_kg_per_kg=frozen_x_p)
                frozen2 = dbc.evaluate_block(gas, full_in, full_props, T_COLDS)
                base_delivered = record["chain_forward"][0]["q"]
                base_full_in, base_full_props = dbc.apply_steps(
                    dbc.STEP_ORDER, base_inputs, own_props
                )
                base_frozen = dbc.evaluate_block(gas, base_full_in, base_full_props, T_COLDS)
                derivatives: dict[str, dict[str, float]] = {
                    "own": {},
                    "block_delivered": {},
                    "block_full_frozen_x_p": {},
                }
                for name in (
                    "outlet_temperature_celsius",
                    "outlet_humidity_kg_per_kg",
                    "heater_power_kw",
                    f"recovery_total_kw_tcold_{T_COLDS[0]:g}",
                    f"recovery_total_kw_tcold_{T_COLDS[1]:g}",
                ):
                    h = sign * step
                    derivatives["own"][name] = (own2[name] - record["own"][name]) / h
                    derivatives["block_delivered"][name] = (
                        delivered2[name] - base_delivered[name]
                    ) / h
                    derivatives["block_full_frozen_x_p"][name] = (
                        frozen2[name] - base_frozen[name]
                    ) / h
                sensitivities[axis] = {
                    "step": sign * step,
                    "derivatives": derivatives,
                    "own_moisture_change_kg_per_kg": s2.particle_moisture_kg_per_kg
                    - solution.particle_moisture_kg_per_kg,
                }
            record["sensitivities"] = sensitivities
            records.append(record)
    return records


def sensitivity_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    valid = [r for r in records if r.get("solved") and r["own_flags"]["solver_feasible"]]
    for axis in ("inlet_temperature", "gas_velocity"):
        per_quantity: dict[str, Any] = {}
        names = (
            "outlet_temperature_celsius",
            "outlet_humidity_kg_per_kg",
            "heater_power_kw",
            f"recovery_total_kw_tcold_{T_COLDS[0]:g}",
            f"recovery_total_kw_tcold_{T_COLDS[1]:g}",
        )
        for name in names:
            entry: dict[str, Any] = {}
            usable = [r for r in valid if "derivatives" in r.get("sensitivities", {}).get(axis, {})]
            own = [r["sensitivities"][axis]["derivatives"]["own"][name] for r in usable]
            entry["n"] = len(usable)
            entry["own"] = _stats(own)
            for model in ("block_delivered", "block_full_frozen_x_p"):
                other = [r["sensitivities"][axis]["derivatives"][model][name] for r in usable]
                diff = [o - w for o, w in zip(other, own, strict=True)]
                same_sign = [
                    (o > 0) == (w > 0)
                    for o, w in zip(other, own, strict=True)
                    if abs(o) > 1e-12 and abs(w) > 1e-12
                ]
                entry[model] = {
                    "derivative": _stats(other),
                    "difference_to_own": _stats(diff),
                    "ratio_to_own": _stats(
                        [o / w for o, w in zip(other, own, strict=True) if abs(w) > 1e-9]
                    ),
                    "same_sign_count": int(sum(same_sign)),
                    "same_sign_of": len(same_sign),
                }
            per_quantity[name] = entry
        out[axis] = per_quantity
    return out


def violation_report(
    label: str, records: list[dict[str, Any]], coordinates: list[str]
) -> dict[str, Any]:
    """Points where the bed model reports a violation or no steady state, block normal."""
    rows = []
    n_violation = 0
    for r in records:
        coords = {c: r.get(c) for c in coordinates}
        if not r.get("solved"):
            rows.append(
                {**coords, "own": "no steady state", "error": r["error"], "block_normal": None}
            )
            n_violation += 1
            continue
        flags = r["own_flags"]
        reasons = []
        if flags["above_quality_limit"]:
            reasons.append("moisture above the quality limit 0.15")
        if flags["above_critical_moisture"]:
            reasons.append("moisture above the critical moisture 0.20")
        if not flags["solver_feasible"]:
            reasons.extend(flags["infeasibility_reasons"])
        if reasons:
            n_violation += 1
            rows.append(
                {
                    **coords,
                    "own": reasons,
                    "own_moisture_kg_per_kg": r["own"]["product_moisture_kg_per_kg"],
                    "block_delivered_normal": r["block_delivered_flags"]["normal"],
                    "block_full_normal": r["block_full_flags"]["normal"],
                }
            )
    return {
        "set": label,
        "n_points": len(records),
        "n_with_violation_or_no_steady_state": n_violation,
        "n_violation_block_delivered_normal": sum(
            1 for row in rows if row.get("block_delivered_normal")
        ),
        "points": rows,
    }


def partner_inputs(config: ProcessConfig) -> dict[str, Any]:
    """VG4: the partner's fluidised-bed inputs in the bed model, only if the solver converges."""
    gas = config.gas
    t_in, spray, air_kg_h, y_in, xs, t_s, t_amb = 300.0, 12.0, 100.0, 0.006, 0.4, 15.0, 15.0
    backend = NumpyBackend()
    phi = float(
        relative_humidity_from_absolute_humidity(backend, gas, np.array(y_in), np.array(t_amb))
    )
    check_y = float(absolute_humidity_kg_per_kg(backend, gas, np.array(t_amb), np.array(phi)))
    assert abs(check_y - y_in) < 1e-9

    def variant_for(u: float) -> ProcessConfig:
        actions = dataclasses.replace(
            config.actions,
            inlet_temperature_celsius=dataclasses.replace(
                config.actions.inlet_temperature_celsius, maximum=t_in, nominal=t_in
            ),
            gas_velocity_m_per_s=dataclasses.replace(
                config.actions.gas_velocity_m_per_s, nominal=u
            ),
            spray_rate_kg_per_h=dataclasses.replace(
                config.actions.spray_rate_kg_per_h, maximum=spray, nominal=spray
            ),
            nozzle_pressure_bar=dataclasses.replace(
                config.actions.nozzle_pressure_bar, nominal=NOZZLE_BAR
            ),
        )
        disturbances = dataclasses.replace(
            config.disturbances,
            ambient_temperature_celsius=dataclasses.replace(
                config.disturbances.ambient_temperature_celsius, nominal=t_amb
            ),
            ambient_relative_humidity=dataclasses.replace(
                config.disturbances.ambient_relative_humidity, nominal=phi
            ),
            suspension_solid_fraction=dataclasses.replace(
                config.disturbances.suspension_solid_fraction, nominal=xs
            ),
        )
        operation = dataclasses.replace(config.operation, suspension_temperature_celsius=t_s)
        return dataclasses.replace(
            config, actions=actions, disturbances=disturbances, operation=operation
        )

    result: dict[str, Any] = {
        "inputs": {
            "inlet_temperature_celsius": t_in,
            "spray_kg_per_h": spray,
            "dry_air_kg_per_h": air_kg_h,
            "air_humidity_kg_per_kg": y_in,
            "ambient_relative_humidity_for_that_humidity": phi,
            "solid_fraction": xs,
            "spray_temperature_celsius": t_s,
            "ambient_temperature_celsius": t_amb,
        },
        "ranges_of_the_bed_model": {
            "inlet_temperature_celsius": [
                config.actions.inlet_temperature_celsius.minimum,
                config.actions.inlet_temperature_celsius.maximum,
            ],
            "spray_kg_per_h": [
                config.actions.spray_rate_kg_per_h.minimum,
                config.actions.spray_rate_kg_per_h.maximum,
            ],
            "bed_temperature_validity_celsius": BED_TEMPERATURE_VALIDITY_C,
        },
        "label": "extrapolation (every input outside the validated or configured range)",
    }
    target = air_kg_h / 3600.0
    scan = []
    for u in (1.0, 1.3, 1.6, 1.9, 2.2, 2.5):
        _, solution, error = _solve(variant_for(u), t_in, u, NOZZLE_BAR, t_amb, phi)
        scan.append(
            {
                "gas_velocity_m_per_s": u,
                "converged": solution is not None,
                "error": error,
                "dry_air_kg_per_h": None
                if solution is None
                else solution.dry_air_mass_flow_kg_per_s * 3600.0,
                "bed_temperature_celsius": None
                if solution is None
                else solution.bed_temperature_celsius,
            }
        )
    result["scan_over_gas_velocity"] = scan
    converged = [s for s in scan if s["converged"]]
    if not converged:
        result["status"] = (
            "outside the range: the solver finds no steady state at any gas velocity tried"
        )
        result["block"] = _block_for_partner(config, t_in, spray, air_kg_h, y_in, xs, t_s, t_amb)
        return result

    def air_gap(u: float) -> float:
        _, solution, _ = _solve(variant_for(u), t_in, u, NOZZLE_BAR, t_amb, phi)
        if solution is None:
            raise RuntimeError("no steady state")
        return solution.dry_air_mass_flow_kg_per_s - target

    brackets = [
        (a["gas_velocity_m_per_s"], b["gas_velocity_m_per_s"])
        for a, b in itertools.pairwise(converged)
        if (a["dry_air_kg_per_h"] / 3600.0 - target) * (b["dry_air_kg_per_h"] / 3600.0 - target) < 0
    ]
    if not brackets:
        result["status"] = (
            "outside the range: the solver converges at some gas velocities, but none gives the "
            "100 kg/h of dry air (no bracket)"
        )
        result["block"] = _block_for_partner(config, t_in, spray, air_kg_h, y_in, xs, t_s, t_amb)
        return result
    u_star = brentq(air_gap, *brackets[0], xtol=1e-6)
    variant, solution, error = _solve(variant_for(u_star), t_in, u_star, NOZZLE_BAR, t_amb, phi)
    assert solution is not None
    record = point_record(config, variant, solution)
    result["status"] = "converged (extrapolation)"
    result["gas_velocity_for_100_kg_per_h_m_per_s"] = u_star
    result["own"] = {
        "bed_temperature_celsius": solution.bed_temperature_celsius,
        "outlet_humidity_kg_per_kg": solution.outlet_humidity_kg_per_kg,
        "particle_moisture_kg_per_kg": solution.particle_moisture_kg_per_kg,
        "dry_air_kg_per_h": solution.dry_air_mass_flow_kg_per_s * 3600.0,
        "feasible": solution.feasible,
        "infeasibility_reasons": list(solution.infeasibility_reasons),
        "residual_inf_norm": solution.residual_inf_norm,
        "diameter_m": solution.diameter_m,
        "saturation_degree": solution.diagnostics["saturation_degree"],
    }
    result["record"] = record
    result["block"] = _block_for_partner(config, t_in, spray, air_kg_h, y_in, xs, t_s, t_amb)
    return result


def _block_for_partner(
    config: ProcessConfig,
    t_in: float,
    spray: float,
    air_kg_h: float,
    y_in: float,
    xs: float,
    t_s: float,
    t_amb: float,
) -> dict[str, Any]:
    inputs = BlockInputs(
        dry_air_mass_flow_kg_per_s=air_kg_h / 3600.0,
        air_humidity_kg_per_kg=y_in,
        air_temperature_celsius=t_in,
        spray_mass_flow_kg_per_s=spray / 3600.0,
        spray_solid_fraction_kg_per_kg=xs,
        spray_temperature_celsius=t_s,
        product_moisture_kg_per_kg=dbc.PARTNER_EQUILIBRIUM_MOISTURE_KG_PER_KG,
        ambient_temperature_celsius=t_amb,
    )
    delivered = dbc.evaluate_block(config.gas, inputs, dbc.partner_properties(), T_COLDS)
    with_own = dbc.evaluate_block(
        config.gas, *dbc.apply_steps(dbc.STEP_ORDER, inputs, dbc.own_properties(config)), T_COLDS
    )
    return {"delivered": delivered, "own_properties_and_loss_equilibrium_x_p": with_own}


def vg0_tables(config: ProcessConfig) -> dict[str, Any]:
    partner, own = dbc.partner_properties(), dbc.own_properties(config)

    def props(p: BlockProperties) -> dict[str, float | None]:
        return {
            "cp_gas_j_per_kg_k": p.cp_gas_j_per_kg_k,
            "cp_vapour_j_per_kg_k": p.cp_vapour_j_per_kg_k,
            "cp_liquid_j_per_kg_k": p.cp_liquid_j_per_kg_k,
            "cp_solid_j_per_kg_k": p.cp_solid_j_per_kg_k,
            "evaporation_enthalpy_0c_j_per_kg": p.evaporation_enthalpy_0c_j_per_kg,
            "latent_heat_slope_j_per_kg_k": p.latent_heat_slope,
            "evaporation_enthalpy_at_40c_j_per_kg": p.latent_heat_j_per_kg(40.0),
            "suspension_cp_at_solid_fraction_0.3_j_per_kg_k": p.suspension_cp(0.3),
            "suspension_cp_at_solid_fraction_0.4_j_per_kg_k": p.suspension_cp(0.4),
            "heat_loss_ua_w_per_k": p.heat_loss_ua_w_per_k,
        }

    return {
        "partner_block_properties": props(partner),
        "partner_equilibrium_moisture_kg_per_kg": dbc.PARTNER_EQUILIBRIUM_MOISTURE_KG_PER_KG,
        "own_properties_in_block_form": props(own),
        "own_cp_vapour_implied_by_latent_heat_slope_j_per_kg_k": own.latent_heat_slope
        + own.cp_liquid_j_per_kg_k,
        "own_heater_efficiency_electric": config.energy.heater_efficiency,
        "own_pressure_pa": config.gas.pressure_pa,
        "own_suspension_temperature_celsius": config.operation.suspension_temperature_celsius,
        "own_noise_std_temperature_k_placeholder": config.measurement.absolute_noise_std_temperature_k,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; artifacts are never overwritten (use a new name)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    m1 = json.loads(M1_ARTIFACT.read_text(encoding="utf-8"))
    assert m1["gas_velocity_lower_bound_m_per_s"] == 1.26

    record: dict[str, Any] = {
        "environment": environment_fingerprint(),
        "config_fingerprint": config.fingerprint,
        "config_fingerprint_of_m1_artifact": m1["config_fingerprint"],
        "t_cold_celsius": list(T_COLDS),
        "step_order": list(dbc.STEP_ORDER),
        "step_order_reversed": list(dbc.STEP_ORDER_REVERSED),
        "search_precision": {
            "max_iterations": DEFAULT_SEARCH_MAX_ITERATIONS,
            "tolerance": DEFAULT_SEARCH_TOLERANCE,
        },
        "quality_limit_kg_per_kg": QUALITY_LIMIT_KG_PER_KG,
        "vg0": vg0_tables(config),
    }
    if record["config_fingerprint"] != record["config_fingerprint_of_m1_artifact"]:
        raise SystemExit("config fingerprint differs from the M1 artifact: stop and report")

    sets: dict[str, list[dict[str, Any]]] = {
        "scenarios_fixed_point_F": reproduce_m1_points(config, m1, "F"),
        "scenarios_oracle_M0": reproduce_m1_points(config, m1, "M0"),
        "grid_inlet_temperature_x_gas_velocity": grid_points(config),
        "scenarios_corner_T60_u126": corner_scenarios(config),
    }
    record["vg1_vg2"] = {
        name: {"points": pts, "summary": summarise(pts)} for name, pts in sets.items()
    }
    record["vg3_sensitivity"] = sensitivity_summary(sets["grid_inlet_temperature_x_gas_velocity"])
    record["vg3_violations"] = [
        violation_report(
            "scenarios_fixed_point_F",
            sets["scenarios_fixed_point_F"],
            ["ambient_temperature_celsius", "ambient_relative_humidity"],
        ),
        violation_report(
            "scenarios_oracle_M0",
            sets["scenarios_oracle_M0"],
            ["ambient_temperature_celsius", "ambient_relative_humidity"],
        ),
        violation_report(
            "grid_inlet_temperature_x_gas_velocity",
            sets["grid_inlet_temperature_x_gas_velocity"],
            ["inlet_temperature_celsius", "gas_velocity_m_per_s"],
        ),
    ]
    record["vg3_violations"].append(
        violation_report(
            "scenarios_corner_T60_u126",
            sets["scenarios_corner_T60_u126"],
            ["ambient_temperature_celsius", "ambient_relative_humidity"],
        )
    )
    reproduction = {}
    for name in ("scenarios_fixed_point_F", "scenarios_oracle_M0"):
        diffs = [
            max(abs(v) for v in p["difference_to_stored_in_m1"].values())
            for p in sets[name]
            if p["solved"]
        ]
        reproduction[name] = {
            "n_solved": len(diffs),
            "max_abs_difference_to_stored": max(diffs) if diffs else None,
        }
    record["reproduction_of_m1_steady_states"] = reproduction
    record["vg4"] = partner_inputs(config)

    args.output.write_text(json.dumps(_clean(record), indent=1), encoding="utf-8")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

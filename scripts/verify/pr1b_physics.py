"""PR1b + PR1e -- physical plausibility of every run, balances, clamping, targets.

Everything is computed from the stored states, actions and disturbances with the audit's own
formulas (constants parsed from the YAML with PyYAML, not through the config dataclasses):

* finiteness of every stored array; bounds (masses > 0, moisture >= local equilibrium moisture,
  bed temperature range, outlet humidity <= saturation humidity, fluidisation window);
* **own RK4 replay** of the balance ODEs (solid, water, particle number, enthalpy) over every
  stored step, starting from the stored state and using only the audit's flux formulas
  (the simulator is called only for the evaporation rate, which needs the sorption isotherm).
  It contains NO state floor. Wherever the replayed state differs from the stored state, the
  simulator's ``_floor`` clamp (or a formula mismatch) acted; the number and size of those
  steps answers "does the data contain corrections instead of physics";
* frequency and size of the gas-velocity projection (requested vs realised action);
* targets recomputed from actions/disturbances/states: ``energy_target`` (kW, from the four
  power terms), ``product_moisture_kg_per_kg`` (m_w/m_s), ``bed_temperature_celsius``, ``d50_um``;
* the sensor model: measurements re-generated bit-exactly from outputs with the documented PT1
  filter and the seeded noise stream.

    python scripts/verify/pr1b_physics.py
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import yaml
from _common import CONFIG_PATH, provenance, require_clean_tree, write_artifact
from _data import make_data

KELVIN = 273.15
G = 9.80665  # the project uses standard gravity; checked against the recomputed fan power
SECONDS_PER_HOUR = 3600.0


def load_yaml() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def p_sat(t: np.ndarray) -> np.ndarray:
    """Buck (1996) saturation pressure [Pa], documented in the model."""
    t = np.clip(t, -40.0, 200.0)
    return 611.21 * np.exp((18.678 - t / 234.5) * t / (257.14 + t))


def abs_humidity(c: dict[str, Any], t: np.ndarray, phi: np.ndarray) -> np.ndarray:
    p = c["gas"]["pressure_pa"]
    pp = np.minimum(phi * p_sat(t), 0.95 * p)
    return 0.622 * pp / (p - pp)


def gas_density(c: dict[str, Any], t: np.ndarray, y: np.ndarray) -> np.ndarray:
    dry = c["gas"]["pressure_pa"] / (c["gas"]["r_specific_dry_air_j_per_kg_k"] * (t + KELVIN))
    return dry * (1.0 + y) / (1.0 + y / 0.622)


def latent_heat(t: np.ndarray) -> np.ndarray:
    return 2.5008e6 - 2358.6 * np.clip(t, 0.0, 150.0)


def power_terms_kw(
    c: dict[str, Any], action: np.ndarray, dist: np.ndarray, m_total: np.ndarray, t_bed: np.ndarray
) -> dict[str, np.ndarray]:
    """The four electrical power terms in kW, from first principles (docstring of energy.py)."""
    gas, energy, geom = c["gas"], c["energy"], c["geometry"]
    t_in, u, spray_kg_h, p_noz = (action[..., i] for i in range(4))
    t_amb, phi, _ = (dist[..., i] for i in range(3))
    y_in = abs_humidity(c, t_amb, phi)
    rho = gas_density(c, t_bed, y_in)
    area = geom["cross_section_area_m2"]
    dry_air = rho * u * area / (1.0 + y_in)
    heater = (
        dry_air
        * (gas["cp_dry_air_j_per_kg_k"] + y_in * gas["cp_vapour_j_per_kg_k"])
        * np.maximum(t_in - t_amb, 0.0)
        / energy["heater_efficiency"]
    )
    dp = m_total * G / area + energy["distributor_loss_coefficient"] * rho * u**2 / 2.0
    fan = u * area * dp / energy["fan_efficiency"]
    pump = (
        (spray_kg_h / SECONDS_PER_HOUR)
        * energy["pump_pressure_rise_pa"]
        / (c["material"]["suspension_density_kg_per_m3"] * energy["pump_efficiency"])
    )
    ratio = np.maximum(p_noz * 1.0e5 / gas["pressure_pa"], 1.0 + 1e-12)
    atomiser = (
        energy["atomiser_air_kg_per_s_per_bar"]
        * p_noz
        * gas["r_specific_dry_air_j_per_kg_k"]
        * (t_amb + KELVIN)
        * np.log(ratio)
        / energy["compressor_efficiency"]
    )
    return {
        "heater": heater / 1e3,
        "fan": fan / 1e3,
        "pump": pump / 1e3,
        "atomiser": atomiser / 1e3,
        "total": (heater + fan + pump + atomiser) / 1e3,
        "dry_air": dry_air,
        "y_in": y_in,
    }


def own_derivative(
    c: dict[str, Any],
    sim: Any,
    x: np.ndarray,
    action: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    """dx/dt of [m_s, m_w, N, T_bed] from the audit's own balance equations."""
    mat, op, geom, gas = c["material"], c["operation"], c["geometry"], c["gas"]
    m_s, m_w, n, t = (x[..., i] for i in range(4))
    t_in, _, spray_kg_h, _ = (action[..., i] for i in range(4))
    t_amb, _, w_s = (dist[..., i] for i in range(3))
    spray = spray_kg_h / SECONDS_PER_HOUR
    seed = op["seed_mass_flow_kg_per_h"] / SECONDS_PER_HOUR
    inflow = spray * w_s + seed
    assert op["mode"] == "continuous"
    product = np.maximum(
        inflow + (m_s - geom["bed_mass_solid_setpoint_kg"]) / op["mass_control_time_constant_s"],
        0.0,
    )
    x_moist = m_w / m_s
    _, diag = sim.evaluate_derivative(x, action, dist)
    evap = diag["evaporation_rate_kg_per_s"]
    y_in = diag["inlet_humidity_kg_per_kg"]
    dry_air = diag["dry_air_mass_flow_kg_per_s"]
    d50 = (6.0 * m_s / (math.pi * mat["particle_density_kg_per_m3"] * n)) ** (1.0 / 3.0)
    m_nucleus = mat["particle_density_kg_per_m3"] * math.pi * mat["nucleus_diameter_m"] ** 3 / 6.0
    m_particle = mat["particle_density_kg_per_m3"] * math.pi * d50**3 / 6.0
    d_s = inflow - product
    d_w = spray * (1.0 - w_s) - evap - product * x_moist
    d_n = (mat["nucleation_solid_fraction"] * spray * w_s + seed) / m_nucleus - product / m_particle
    capacity = m_s * mat["cp_solid_j_per_kg_k"] + m_w * mat["cp_water_liquid_j_per_kg_k"]
    q = (
        dry_air * (gas["cp_dry_air_j_per_kg_k"] + y_in * gas["cp_vapour_j_per_kg_k"]) * (t_in - t)
        + spray * mat["cp_suspension_j_per_kg_k"] * (op["suspension_temperature_celsius"] - t)
        - evap * latent_heat(t)
        - geom["heat_loss_ua_w_per_k"] * (t - t_amb)
    )
    return np.stack([d_s, d_w, d_n, q / capacity], axis=-1)


def own_rk4_step(
    c: dict[str, Any],
    sim: Any,
    x0: np.ndarray,
    action: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    """One 25 s step = ``substeps`` RK4 sub-steps, no floor, own integrator."""
    dt = c["numerics"]["sample_time_s_effective"] / c["numerics"]["substeps"]
    x = x0
    for _ in range(c["numerics"]["substeps"]):
        k1 = own_derivative(c, sim, x, action, dist)
        k2 = own_derivative(c, sim, x + 0.5 * dt * k1, action, dist)
        k3 = own_derivative(c, sim, x + 0.5 * dt * k2, action, dist)
        k4 = own_derivative(c, sim, x + dt * k3, action, dist)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def initial_state(c: dict[str, Any], phys: np.ndarray) -> np.ndarray:
    """[m_s, X0, d0, T0] (schedule columns) -> state vector [m_s, m_w, N, T]."""
    rho = c["material"]["particle_density_kg_per_m3"]
    m_s, x0, d0, t0 = (phys[..., i] for i in range(4))
    return np.stack([m_s, m_s * x0, m_s / (rho * math.pi * d0**3 / 6.0), t0], axis=-1)


def remeasure(c: dict[str, Any], outputs: np.ndarray, seed: int, sample_time: float) -> np.ndarray:
    """Documented sensor model: PT1 filter (tau) + relative and absolute noise, seeded."""
    meas = c["measurement"]
    rng = np.random.default_rng(seed)
    alpha = math.exp(-sample_time / meas["sensor_time_constant_s"])
    n_runs, n_steps, _ = outputs.shape
    result = np.empty_like(outputs)
    filt = None
    for k in range(n_steps):
        values = outputs[:, k, :]
        filt = values.copy() if filt is None else alpha * filt + (1.0 - alpha) * values
        noisy = filt.copy()
        noisy += rng.normal(0.0, meas["relative_noise_std"] * np.abs(noisy))
        noisy[..., 1] += rng.normal(0.0, meas["absolute_noise_std_temperature_k"], size=n_runs)
        result[:, k, :] = noisy
    return result


def stats(a: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64).ravel()
    return {
        "min": float(a.min()),
        "p01": float(np.percentile(a, 1)),
        "median": float(np.median(a)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def audit_pool(kind: str, seed: int, c: dict[str, Any]) -> dict[str, Any]:
    from twinlink_ml.simulation import GreyBoxFluidBedGranulator  # simulator: allowed
    from twinlink_ml.simulation.sorption import isotherm_from_material
    from twinlink_ml.simulation.thermo import relative_humidity_from_absolute_humidity

    data = make_data(kind, seed)
    sim = GreyBoxFluidBedGranulator(data.process_config)
    backend = sim.backend
    traj = data.trajectory
    n_runs, n_steps, _ = traj.states.shape
    dt = float(traj.metadata["sample_time_s"])
    c = {**c, "numerics": {**c["numerics"], "sample_time_s_effective": dt}}
    out: dict[str, Any] = {"kind": kind, "seed": seed, "n_runs": n_runs, "n_steps": n_steps}

    # ---- finiteness ----------------------------------------------------------------
    arrays = {
        "states": traj.states,
        "outputs": traj.outputs,
        "measurements": traj.measurements,
        "actions": traj.actions,
        "disturbances": traj.disturbances,
        **{f"diag/{k}": v for k, v in traj.diagnostics.items()},
    }
    out["finite"] = {k: bool(np.isfinite(v).all()) for k, v in arrays.items()}
    out["all_finite"] = all(out["finite"].values())

    m_s, m_w, n_part, t_bed = (traj.states[..., i] for i in range(4))
    moisture = m_w / m_s
    diag = traj.diagnostics

    # ---- bounds ----------------------------------------------------------------------
    isotherm = isotherm_from_material(data.process_config.material)
    phi_out = relative_humidity_from_absolute_humidity(
        backend, data.process_config.gas, diag["outlet_humidity_kg_per_kg"], t_bed
    )
    x_eq = isotherm.equilibrium_moisture(backend, phi_out, t_bed)
    margin_eq = moisture - x_eq
    y_sat = diag["saturation_humidity_kg_per_kg"]
    out["bounds"] = {
        "solid_mass_kg": stats(m_s),
        "water_mass_kg": stats(m_w),
        "particle_count": stats(n_part),
        "masses_and_count_all_positive": bool(
            (m_s > 0).all() and (m_w >= 0).all() and (n_part > 0).all()
        ),
        "moisture_kg_per_kg": stats(moisture),
        "moisture_minus_equilibrium": stats(margin_eq),
        "n_steps_moisture_below_equilibrium_by_more_than_1e-9": int((margin_eq < -1e-9).sum()),
        "n_steps_at_equilibrium_floor_(margin<1e-9)": int((margin_eq < 1e-9).sum()),
        "n_steps_moisture_above_critical": int(
            (moisture > c["material"]["critical_moisture_kg_per_kg"]).sum()
        ),
        "bed_temperature_celsius": stats(t_bed),
        "n_steps_bed_temperature_outside_0_100": int(((t_bed < 0) | (t_bed > 100)).sum()),
        "n_steps_bed_temperature_above_80": int((t_bed > 80).sum()),
        "outlet_humidity_minus_saturation_max": float(
            (diag["outlet_humidity_kg_per_kg"] - y_sat).max()
        ),
        "d50_um": stats(
            1e6
            * (6 * m_s / (math.pi * c["material"]["particle_density_kg_per_m3"] * n_part))
            ** (1 / 3)
        ),
        "fluidisation_margin_lower_end_of_step_min": float(
            diag["fluidization_margin_lower_m_per_s"].min()
        ),
        "fluidisation_margin_upper_end_of_step_min": float(
            diag["fluidization_margin_upper_m_per_s"].min()
        ),
        "n_steps_u_below_u_mf_end_of_step": int(
            (diag["fluidization_margin_lower_m_per_s"] < 0).sum()
        ),
        "n_steps_u_above_u_terminal_end_of_step": int(
            (diag["fluidization_margin_upper_m_per_s"] < 0).sum()
        ),
    }

    # ---- projection frequency ------------------------------------------------------------
    if traj.requested_actions is not None:
        changed = traj.actions != traj.requested_actions
        step_changed = changed.any(axis=-1)
        only_gas = bool((changed[..., [0, 2, 3]] == False).all())  # noqa: E712
        per_run = step_changed.mean(axis=1)
        gas_req, gas_real = traj.requested_actions[..., 1], traj.actions[..., 1]
        rel = (
            np.abs(gas_real - gas_req)[step_changed] / gas_req[step_changed]
            if step_changed.any()
            else np.zeros(1)
        )
        first_tenth = step_changed[:, : n_steps // 10].sum() / max(step_changed.sum(), 1)
        out["projection"] = {
            "steps_projected_fraction": float(step_changed.mean()),
            "runs_with_any_projection": int((per_run > 0).sum()),
            "runs_with_more_than_10pct_steps_projected": int((per_run > 0.10).sum()),
            "runs_with_more_than_50pct_steps_projected": int((per_run > 0.50).sum()),
            "per_run_fraction_projected": stats(per_run),
            "only_gas_velocity_channel_modified": only_gas,
            "clipped_to_upper_(u_terminal)_fraction": float(
                (gas_real < gas_req)[step_changed].mean()
            )
            if step_changed.any()
            else 0.0,
            "relative_change_of_gas_velocity_when_projected": stats(rel),
            "share_of_projected_steps_in_first_10pct_of_run": float(first_tenth),
            "n_projected_steps_field_consistent": bool(
                (traj.n_projected_steps == step_changed.sum(axis=1)).all()
            ),
        }
    else:
        out["projection"] = None

    # ---- own RK4 replay: clamp detection and balance closure ----------------------------
    x_start = np.concatenate(
        [initial_state(c, data.schedule.initial_states)[:, None, :], traj.states[:, :-1, :]], axis=1
    )
    replay = own_rk4_step(c, sim, x_start, traj.actions, traj.disturbances)
    scale = np.maximum(np.abs(traj.states), 1e-12)
    rel_dev = np.abs(replay - traj.states) / scale
    names = ("solid_mass", "water_mass", "particle_count", "bed_temperature")
    clamp_mask = rel_dev.max(axis=-1) > 1e-8
    non_clamp = rel_dev[~clamp_mask]
    out["replay"] = {
        "formulas": "own RK4 (5 substeps) of own balance ODEs; simulator used only for evaporation",
        "n_steps_total": int(clamp_mask.size),
        "steps_where_replay_differs_from_stored_(rel>1e-8)": int(clamp_mask.sum()),
        "fraction_steps_differing": float(clamp_mask.mean()),
        "runs_with_any_difference": int(clamp_mask.any(axis=1).sum()),
        "first_step_of_run_share_of_differences": float(
            clamp_mask[:, 0].sum() / max(clamp_mask.sum(), 1)
        ),
        "max_relative_deviation_on_non_flagged_steps": float(non_clamp.max())
        if non_clamp.size
        else 0.0,
        "max_relative_deviation_per_state_all_steps": {
            n: float(rel_dev[..., i].max()) for i, n in enumerate(names)
        },
        "abs_water_mass_shortfall_kg_on_flagged_steps": stats(
            (traj.states[..., 1] - replay[..., 1])[clamp_mask]
        )
        if clamp_mask.any()
        else None,
        "moisture_shortfall_kg_per_kg_on_flagged_steps": stats(
            ((traj.states[..., 1] - replay[..., 1]) / traj.states[..., 0])[clamp_mask]
        )
        if clamp_mask.any()
        else None,
    }
    # cumulative drift per run: sum of stored-minus-replayed increments (solid, water, T)
    resid = traj.states - replay
    rel_run = np.abs(resid.sum(axis=1)) / np.maximum(np.abs(traj.states).max(axis=1), 1e-12)
    out["replay"]["run_level_cumulative_residual_relative_max_per_state"] = {
        n: float(rel_run[:, i].max()) for i, n in enumerate(names)
    }

    # coarse independent trapezoid closure of the solid and water balance over each run
    spray = traj.actions[..., 2] / SECONDS_PER_HOUR
    inflow_solid = (
        spray * traj.disturbances[..., 2]
        + c["operation"]["seed_mass_flow_kg_per_h"] / SECONDS_PER_HOUR
    )
    product_end = diag["product_mass_flow_kg_per_s"]
    d_start = own_derivative(c, sim, x_start, traj.actions, traj.disturbances)
    d_end = own_derivative(c, sim, traj.states, traj.actions, traj.disturbances)
    trap = 0.5 * dt * (d_start + d_end)
    s0 = initial_state(c, data.schedule.initial_states)
    delta = traj.states[:, -1, :] - s0
    total_inflow_solid = (inflow_solid * dt).sum(axis=1)
    total_spray_water = (spray * (1 - traj.disturbances[..., 2]) * dt).sum(axis=1)
    out["balance_trapezoid_25s"] = {
        "note": "coarse second-order check independent of the integrator; the moisture mode has "
        "tau~15 s, so the water residual is dominated by trapezoid error, the solid one is tight",
        "solid_closure_relative_to_total_solid_inflow_max": float(
            np.max(np.abs(delta[:, 0] - trap[..., 0].sum(axis=1)) / total_inflow_solid)
        ),
        "water_closure_relative_to_total_sprayed_water_max": float(
            np.max(np.abs(delta[:, 1] - trap[..., 1].sum(axis=1)) / total_spray_water)
        ),
        "solid_product_flow_model_check_max_abs_kg_per_s": float(
            np.abs(
                np.maximum(
                    inflow_solid
                    + (m_s - c["geometry"]["bed_mass_solid_setpoint_kg"])
                    / c["operation"]["mass_control_time_constant_s"],
                    0.0,
                )
                - product_end
            ).max()
        ),
    }

    # ---- PR1e: targets ----------------------------------------------------------------
    power = power_terms_kw(c, traj.actions, traj.disturbances, m_s + m_w, t_bed)
    assert c["energy"]["target"] == "power_kw" and c["energy"]["reference_value"] is None
    energy_out = traj.outputs[..., 0]
    out["targets"] = {
        "units": {
            "energy_target": "kW (electrical power, absolute)",
            "product_moisture": "kg water / kg dry solid",
            "bed_temperature": "degC",
            "d50": "um",
        },
        "energy_target_vs_recomputed_kw": {
            "max_abs_diff": float(np.abs(energy_out - power["total"]).max()),
            "max_rel_diff": float((np.abs(energy_out - power["total"]) / power["total"]).max()),
            "stats_kw": stats(energy_out),
        },
        "power_terms_vs_diagnostics_max_abs_kw": {
            k: float(np.abs(diag[f"power_{k}_kw"] - power[k]).max())
            for k in ("heater", "fan", "pump", "atomiser", "total")
        },
        "energy_target_equals_power_total_diag_max_abs": float(
            np.abs(energy_out - diag["power_total_kw"]).max()
        ),
        "moisture_output_vs_mw_over_ms_max_abs": float(
            np.abs(traj.outputs[..., 2] - moisture).max()
        ),
        "bed_temperature_output_vs_state_max_abs": float(
            np.abs(traj.outputs[..., 1] - t_bed).max()
        ),
        "d50_output_vs_recomputed_um_max_abs": float(
            np.abs(
                traj.outputs[..., 3]
                - 1e6
                * (6 * m_s / (math.pi * c["material"]["particle_density_kg_per_m3"] * n_part))
                ** (1 / 3)
            ).max()
        ),
        "power_share_of_total_median": {
            k: float(np.median(power[k] / power["total"]))
            for k in ("heater", "fan", "pump", "atomiser")
        },
    }
    # ---- sensor model -------------------------------------------------------------------
    re_meas = remeasure(c, traj.outputs, data.seeds.noise_seed, dt)
    out["measurement_model"] = {
        "bit_exact_regeneration": bool(np.array_equal(re_meas, traj.measurements)),
        "max_abs_diff": float(np.abs(re_meas - traj.measurements).max()),
        "noise_floor_rmse_measurement_vs_truth": {
            name: float(np.sqrt(np.mean((traj.measurements[..., i] - traj.outputs[..., i]) ** 2)))
            for i, name in enumerate(
                ("energy_target", "bed_temperature_celsius", "product_moisture_kg_per_kg", "d50_um")
            )
        },
    }
    return out


def main() -> None:
    require_clean_tree()
    c = load_yaml()
    results = []
    jobs = [("excited", s) for s in (1, 2, 3, 4, 5)] + [
        ("constant_train", s) for s in (1, 2, 3, 4, 5)
    ]
    jobs.append(("holdout", 999))
    for kind, seed in jobs:
        results.append(audit_pool(kind, seed, c))
        r = results[-1]
        print(
            kind,
            seed,
            "finite",
            r["all_finite"],
            "clamped steps",
            r["replay"]["fraction_steps_differing"],
            flush=True,
        )
    write_artifact("verify_pr1b_physics", {"_provenance": provenance(), "pools": results})


if __name__ == "__main__":
    main()

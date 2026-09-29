"""PR1d -- is the excitation design what the documentation claims.

Independent signal classification from the *requested* action series (the design output, after
the documented rate limiter), compared with the labels the design attached
(``schedule.signal_kinds``); bounds, rate limits, constant runs, disturbance ranges and initial
conditions against the ranges parsed from the YAML with PyYAML (initial-condition ranges from the
documented design: d50 250-900 um log-uniform, X between equilibrium and critical moisture,
T_bed 35-80 degC, solid mass 0.8-1.2 x set point).

    python scripts/verify/pr1d_excitation.py
"""

from __future__ import annotations

import collections
from typing import Any

import numpy as np
import yaml
from _common import CONFIG_PATH, provenance, require_clean_tree, write_artifact
from _data import make_data

ACTIONS = (
    "inlet_temperature_celsius",
    "gas_velocity_m_per_s",
    "spray_rate_kg_per_h",
    "nozzle_pressure_bar",
)
DISTURBANCES = (
    "ambient_temperature_celsius",
    "ambient_relative_humidity",
    "suspension_solid_fraction",
)
FLAT = 1e-12


def classify(z: np.ndarray) -> tuple[str, dict[str, float]]:
    """Own classification of one normalised action series (values in [0, 1])."""
    if np.ptp(z) < 1e-9:
        return "constant", {}
    d = np.abs(np.diff(z))
    flat = d < FLAT
    frac_flat = float(flat.mean())
    # plateau levels = values at samples that sit on a flat segment
    on_plateau = np.concatenate([flat, [False]]) | np.concatenate([[False], flat])
    levels = np.unique(np.round(z[on_plateau], 6))
    info = {"frac_flat": frac_flat, "n_levels": float(len(levels))}
    if frac_flat > 0.5:
        return ("prbs" if len(levels) <= 2 else "step"), info
    return "ramp", info


def audit_pool(kind: str, seed: int, c: dict[str, Any]) -> dict[str, Any]:
    data = make_data(kind, seed)
    sched = data.schedule
    dt = float(sched.metadata["sample_time_s"])
    a_req = sched.actions
    a_real = data.trajectory.actions
    dist = sched.disturbances
    n_runs, n_steps, _ = a_req.shape
    labels = sched.signal_kinds
    out: dict[str, Any] = {"kind": kind, "seed": seed, "n_runs": n_runs, "n_steps": n_steps}

    # --- bounds and rate limits -----------------------------------------------------------
    bounds_ok, rate_ok, worst_rate = {}, {}, {}
    for i, name in enumerate(ACTIONS):
        b = c["actions"][name]
        for label, arr in (("requested", a_req), ("realised", a_real)):
            bounds_ok[f"{name}/{label}"] = bool(
                arr[..., i].min() >= b["minimum"] - 1e-12
                and arr[..., i].max() <= b["maximum"] + 1e-12
            )
        rate = np.abs(np.diff(a_req[..., i], axis=1)) / dt
        worst_rate[name] = float(rate.max() / b["max_rate_per_s"])
        rate_ok[name] = bool(rate.max() <= b["max_rate_per_s"] * (1 + 1e-9))
    out["action_bounds_ok"] = bounds_ok
    out["rate_limit_ok_(requested)"] = rate_ok
    out["max_rate_over_limit"] = worst_rate

    # --- classification vs labels ------------------------------------------------------------
    confusion: collections.Counter[tuple[str, str]] = collections.Counter()
    disagreements: list[dict[str, Any]] = []
    for r in range(n_runs):
        for i, name in enumerate(ACTIONS):
            b = c["actions"][name]
            z = (a_req[r, :, i] - b["minimum"]) / (b["maximum"] - b["minimum"])
            own, _ = classify(z)
            confusion[(labels[r][i], own)] += 1
            if own != labels[r][i]:
                slew_steps = (b["maximum"] - b["minimum"]) / b["max_rate_per_s"] / dt
                disagreements.append(
                    {
                        "run": r,
                        "channel": name,
                        "label": labels[r][i],
                        "own": own,
                        "amplitude_band_width": float(np.ptp(z)),
                        "full_scale_slew_time_steps": float(slew_steps),
                    }
                )
    out["label_vs_own_classification"] = {f"{a}->{b}": n for (a, b), n in sorted(confusion.items())}
    out["disagreement_by_channel_and_label"] = dict(
        collections.Counter(f"{d['channel']}:{d['label']}->{d['own']}" for d in disagreements)
    )
    out["disagreement_mean_band_width"] = (
        float(np.mean([d["amplitude_band_width"] for d in disagreements]))
        if disagreements
        else None
    )
    out["label_counts"] = dict(collections.Counter(k for run in labels for k in run))
    out["agreement_fraction"] = float(
        sum(n for (a, b), n in confusion.items() if a == b) / sum(confusion.values())
    )

    # --- signal-type specifics --------------------------------------------------------------
    prbs_runs = 0
    high_fraction, run_lengths, phases = [], [], []
    clock_steps = max(1, round(float(c["_excitation"]["prbs_clock_s"]) / dt))
    for r in range(n_runs):
        for i, name in enumerate(ACTIONS):
            if labels[r][i] != "prbs":
                continue
            b = c["actions"][name]
            z = (a_req[r, :, i] - b["minimum"]) / (b["maximum"] - b["minimum"])
            lo, hi = z.min(), z.max()
            high = z > 0.5 * (lo + hi)
            high_fraction.append(float(high.mean()))
            edges = np.flatnonzero(np.diff(high.astype(int)) != 0) + 1
            phases.extend((edges % clock_steps).tolist())
            bits = high[::clock_steps]
            change = np.flatnonzero(np.diff(bits.astype(int)) != 0)
            lens = np.diff(np.concatenate([[-1], change, [len(bits) - 1]]))
            run_lengths.extend(lens.tolist())
            prbs_runs += 1
    if prbs_runs:
        rl = np.asarray(run_lengths)
        out["prbs"] = {
            "n_series": prbs_runs,
            "clock_steps_assumed": clock_steps,
            "fraction_high_mean": float(np.mean(high_fraction)),
            "fraction_high_min_max": [float(np.min(high_fraction)), float(np.max(high_fraction))],
            "mean_run_length_bits_(m-sequence ~2)": float(rl.mean()),
            "share_run_length_1_(m-sequence 0.5)": float((rl == 1).mean()),
            "share_run_length_2_(m-sequence 0.25)": float((rl == 2).mean()),
            "modal_edge_phase_mod_clock_share": float(
                collections.Counter(phases).most_common(1)[0][1] / max(len(phases), 1)
            ),
        }
    n_step_series = sum(1 for run in labels for k in run if k == "step")
    n_ramp_series = sum(1 for run in labels for k in run if k == "ramp")
    out["n_step_series"], out["n_ramp_series"] = n_step_series, n_ramp_series

    # --- constant runs --------------------------------------------------------------------------
    is_const_label = np.array([all(k == "constant" for k in run) for run in labels])
    is_const_data = np.array(
        [all(np.ptp(a_req[r, :, i]) < 1e-9 for i in range(4)) for r in range(n_runs)]
    )
    idx = np.flatnonzero(is_const_label)
    out["constant_runs"] = {
        "n_flagged": int(is_const_label.sum()),
        "fraction_flagged": float(is_const_label.mean()),
        "flagged_indices_first": idx[:15].tolist(),
        "label_equals_data_constant_for_all_runs": bool((is_const_label == is_const_data).all()),
        "requested_actions_exactly_constant": bool(is_const_data[idx].all()) if len(idx) else None,
    }
    if len(idx):
        gas_real_ptp = np.ptp(a_real[idx, :, 1], axis=1)
        out["constant_runs"].update(
            {
                "realised_gas_velocity_varies_in_n_runs_(projection)": int(
                    (gas_real_ptp > 1e-12).sum()
                ),
                "realised_other_channels_constant": bool(
                    all(np.ptp(a_real[idx, :, i], axis=1).max() < 1e-12 for i in (0, 2, 3))
                ),
                "disturbance_ptp_median_in_constant_runs": {
                    n: float(np.median(np.ptp(dist[idx, :, j], axis=1)))
                    for j, n in enumerate(DISTURBANCES)
                },
                "disturbance_ptp_as_share_of_range": {
                    n: float(
                        np.median(np.ptp(dist[idx, :, j], axis=1))
                        / (c["disturbances"][n]["maximum"] - c["disturbances"][n]["minimum"])
                    )
                    for j, n in enumerate(DISTURBANCES)
                },
            }
        )
    # --- disturbances ------------------------------------------------------------------------------
    d_report = {}
    for j, n in enumerate(DISTURBANCES):
        lo, hi = c["disturbances"][n]["minimum"], c["disturbances"][n]["maximum"]
        d_report[n] = {
            "configured": [lo, hi],
            "observed": [float(dist[..., j].min()), float(dist[..., j].max())],
            "within_range": bool(
                dist[..., j].min() >= lo - 1e-12 and dist[..., j].max() <= hi + 1e-12
            ),
            "coverage_of_range": float((dist[..., j].max() - dist[..., j].min()) / (hi - lo)),
            "fraction_of_samples_exactly_at_a_bound": float(
                ((dist[..., j] <= lo + 1e-12) | (dist[..., j] >= hi - 1e-12)).mean()
            ),
        }
    out["disturbances"] = d_report
    # --- initial conditions ----------------------------------------------------------------
    ic = sched.initial_states
    mat, geom = c["material"], c["geometry"]
    expected = {
        "solid_mass_kg": (
            0.8 * geom["bed_mass_solid_setpoint_kg"],
            1.2 * geom["bed_mass_solid_setpoint_kg"],
        ),
        "moisture_kg_per_kg": (
            mat["equilibrium_moisture_kg_per_kg"],
            mat["critical_moisture_kg_per_kg"],
        ),
        "diameter_m": (250e-6, 900e-6),
        "bed_temperature_celsius": (35.0, 80.0),
    }
    out["initial_conditions"] = {
        name: {
            "documented_range": list(expected[name]),
            "observed": [float(ic[:, k].min()), float(ic[:, k].max())],
            "within_range": bool(
                ic[:, k].min() >= expected[name][0] - 1e-12
                and ic[:, k].max() <= expected[name][1] + 1e-12
            ),
        }
        for k, name in enumerate(expected)
    }
    return out


def main() -> None:
    require_clean_tree()
    c = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    # documented default excitation configuration (ExcitationConfig defaults in the design docs)
    c["_excitation"] = {"prbs_clock_s": 120.0, "constant_run_fraction_excited": 0.10}
    pools = []
    for kind, seed in [("excited", s) for s in (1, 2, 3, 4, 5)] + [
        ("constant_train", 1),
        ("holdout", 999),
    ]:
        pools.append(audit_pool(kind, seed, c))
        print(kind, seed, pools[-1]["agreement_fraction"], flush=True)
    write_artifact("verify_pr1d_excitation", {"_provenance": provenance(), "pools": pools})


if __name__ == "__main__":
    main()

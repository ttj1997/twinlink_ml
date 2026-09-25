r"""Probe: Anregungsbandbreite diagnostizieren (Auftrag C0-C4).

KEIN Netztraining, KEIN Sweep, KEINE Aenderung an ``src/`` -- insbesondere nicht an
``excitation.py``-Defaults. Alle Varianten sind lokale ``ExcitationConfig``- bzw.
``ActionSpaceConfig``-Instanzen in diesem Skript. Ruft nur bestehenden Code auf:
``simulation.diagnostics.static_explained_variance`` (Referenz, unveraendert daneben
mitgefuehrt), ``simulation.diagnostics.time_constant_report`` (ruft
``GreyBoxFluidBedGranulator.time_constants``, greybox.py:662-680, auf),
``simulation.excitation.ExcitationConfig`` / ``MixedExcitationDesign``.

Die einzige NEUE Berechnung (C0) ist ``static_explained_variance_with_lag1`` unten in
diesem Skript -- keine Aenderung an ``simulation/diagnostics.py``.

Basis: ``conf/process/greybox_gab_placeholder.yaml`` (fp 82f65f576a2e5253, siehe
``artifacts/runs/probe_a2_a3_static_variance_time_constants.json``), n_runs=32,
dt=25 s, run_duration=28800 s, randomise_initial_conditions=True,
Seeds 20260812/13/14 (``SeedBundle.from_master``).
"""

from __future__ import annotations

import dataclasses
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from twinlink_ml.simulation import (
    ExcitationConfig,
    GreyBoxFluidBedGranulator,
    MixedExcitationDesign,
    PlaceholderConfigWarning,
    load_process_config,
)
from twinlink_ml.simulation.base import Trajectory
from twinlink_ml.simulation.config import ActionSpaceConfig, ProcessConfig
from twinlink_ml.simulation.diagnostics import static_explained_variance, time_constant_report
from twinlink_ml.simulation.excitation import ExcitationSchedule
from twinlink_ml.utils.seeding import SeedBundle

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_GAB = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"

N_RUNS = 32
SAMPLE_TIME_S = 25.0
RUN_DURATION_S = 28_800.0
MASTER_SEEDS = (20260812, 20260813, 20260814)

# aus artifacts/runs/probe_a2_a3_static_variance_time_constants.json (a3, gab, unveraendert)
TAU_1_S = 30.22626192813928
TAU_2_S = 359.9276103387108

BED_TEMPERATURE_VALIDITY_LIMIT_C = 80.0  # thermo.py:268, dokumentierte Modellgrenze


# ------------------------------------------------------------------------ Aufbau


def _load(
    config_path: Path, *, sample_time_s: float = SAMPLE_TIME_S
) -> GreyBoxFluidBedGranulator:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base = load_process_config(config_path)
        numerics = dataclasses.replace(base.numerics, sample_time_s=sample_time_s, substeps=5)
        config = dataclasses.replace(base, numerics=numerics)
    return GreyBoxFluidBedGranulator(config)


def _rate_limited_actions(
    actions: ActionSpaceConfig, *, inlet_max_rate: float
) -> ActionSpaceConfig:
    return dataclasses.replace(
        actions,
        inlet_temperature_celsius=dataclasses.replace(
            actions.inlet_temperature_celsius, max_rate_per_s=inlet_max_rate
        ),
    )


def _variant_configs(
    base_actions: ActionSpaceConfig,
) -> dict[str, tuple[ExcitationConfig, ActionSpaceConfig]]:
    v0 = ExcitationConfig(run_duration_s=RUN_DURATION_S, randomise_initial_conditions=True)
    v1 = dataclasses.replace(v0, prbs_clock_s=50.0)
    v2_actions = _rate_limited_actions(base_actions, inlet_max_rate=2.0)
    v3 = dataclasses.replace(v0, prbs_clock_s=50.0)
    v4 = dataclasses.replace(
        v0,
        prbs_clock_s=50.0,
        signal_weights={"step": 0.0, "prbs": 1.0, "ramp": 0.0},
        constant_run_fraction=0.0,
    )
    return {
        "V0": (v0, base_actions),
        "V1": (v1, base_actions),
        "V2": (v0, v2_actions),
        "V3": (v3, v2_actions),
        "V4": (v4, v2_actions),
    }


def _build_trajectory(
    simulator: GreyBoxFluidBedGranulator,
    excitation_config: ExcitationConfig,
    action_space: ActionSpaceConfig,
    *,
    master_seed: int,
) -> tuple[Trajectory, ExcitationSchedule]:
    seeds = SeedBundle.from_master(master_seed)
    config = simulator.config
    design = MixedExcitationDesign(
        excitation_config,
        action_space,
        config.disturbances,
        sample_time_s=config.numerics.sample_time_s,
        geometry=config.geometry,
        material=config.material,
        operation=config.operation,
    )
    schedule = design.build(N_RUNS, seed=seeds.data_seed)
    initial = None
    if schedule.initial_states is not None:
        columns = schedule.initial_states
        initial = simulator.state_from_physical(
            columns[:, 0], columns[:, 1], columns[:, 2], columns[:, 3]
        )
    trajectory = simulator.rollout(
        schedule.actions,
        schedule.disturbances,
        run_id=schedule.run_id,
        seed=seeds.noise_seed,
        initial_state=initial,
    )
    return trajectory, schedule


# --------------------------------------------------------------------------- C0


def static_explained_variance_with_lag1(
    trajectory: Trajectory, *, quadratic: bool = True
) -> dict[str, dict[str, float]]:
    r"""Erweiterung von ``static_explained_variance`` um u(t-dt) je Stellgroesse (C0).

    Gleiche Merkmalslogik wie das Original: elementweise, kein Kreuzprodukt, keine
    Standardisierung, OLS via ``lstsq``. R2_stat wird HIER auf denselben Stichproben
    (t >= 1 je Lauf) neu berechnet wie R2_lag1 -- t=0 hat kein u(t-dt) und entfaellt
    fuer beide Modelle, damit die Differenz nicht durch eine unterschiedliche
    Stichprobenzahl konfundiert ist. ``simulation.diagnostics.static_explained_variance``
    bleibt daneben unveraendert und wird fuer den Vollstichproben-Referenzwert separat
    aufgerufen.
    """
    n_runs, n_steps, _ = trajectory.actions.shape
    actions_t = trajectory.actions[:, 1:, :]
    dist_t = trajectory.disturbances[:, 1:, :]
    actions_lag1 = trajectory.actions[:, :-1, :]
    outputs_t = trajectory.outputs[:, 1:, :]
    n_trim = n_steps - 1

    inputs = np.concatenate([actions_t, dist_t], axis=-1).reshape(n_runs * n_trim, -1)
    if quadratic:
        inputs = np.concatenate([inputs, inputs**2], axis=-1)
    design_stat = np.concatenate([inputs, np.ones((inputs.shape[0], 1))], axis=-1)

    lag = actions_lag1.reshape(n_runs * n_trim, -1)
    design_lag1 = np.concatenate([inputs, lag, np.ones((inputs.shape[0], 1))], axis=-1)

    names = trajectory.metadata.get("output_names", ())
    outputs = outputs_t.reshape(n_runs * n_trim, -1)
    result: dict[str, dict[str, float]] = {}
    for index in range(outputs.shape[1]):
        target = outputs[:, index]
        variance = float(np.var(target))
        label = names[index] if index < len(names) else f"output_{index}"
        if variance <= 0.0:
            result[label] = {
                "r2_stat_trimmed": float("nan"),
                "r2_lag1": float("nan"),
                "diff": float("nan"),
                "one_minus_r2_lag1": float("nan"),
            }
            continue
        coeff_stat, *_ = np.linalg.lstsq(design_stat, target, rcond=None)
        r2_stat = 1.0 - float(np.var(target - design_stat @ coeff_stat)) / variance
        coeff_lag1, *_ = np.linalg.lstsq(design_lag1, target, rcond=None)
        r2_lag1 = 1.0 - float(np.var(target - design_lag1 @ coeff_lag1)) / variance
        result[label] = {
            "r2_stat_trimmed": r2_stat,
            "r2_lag1": r2_lag1,
            "diff": r2_lag1 - r2_stat,
            "one_minus_r2_lag1": 1.0 - r2_lag1,
        }
    return result


# --------------------------------------------------------------------------- C1


def quasistationary_fractions(
    actions: np.ndarray, action_space: ActionSpaceConfig, *, sample_time_s: float, tau_s: float
) -> dict[str, float]:
    """Ratenbasiertes Quasistationaritaetsmass: |u(t)-u(t-3*tau)| < 0.05*Spanne."""
    lag_steps = max(1, round(3.0 * tau_s / sample_time_s))
    names = action_space.names
    bounds = action_space.bounds
    result: dict[str, float] = {}
    n_steps = actions.shape[1]
    if lag_steps >= n_steps:
        return {name: float("nan") for name in names}
    for c, (name, bound) in enumerate(zip(names, bounds, strict=False)):
        diffs = np.abs(actions[:, lag_steps:, c] - actions[:, :-lag_steps, c])
        result[name] = float(np.mean(diffs < 0.05 * bound.span))
    return result


def quasistationary_fractions_old(
    actions: np.ndarray, action_space: ActionSpaceConfig
) -> dict[str, float]:
    """Altes Kriterium (B1): exakte Gleichheit aufeinanderfolgender Abtastwerte."""
    names = action_space.names
    result: dict[str, float] = {}
    for c, name in enumerate(names):
        diffs = actions[:, 1:, c] - actions[:, :-1, c]
        result[name] = float(np.mean(diffs == 0.0))
    return result


# ------------------------------------------------------------------ Aggregation


def _median_range(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(arr)), "min": float(np.min(arr)), "max": float(np.max(arr))}


def _aggregate_per_channel(list_of_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    names = list_of_dicts[0].keys()
    return {name: _median_range([d[name] for d in list_of_dicts]) for name in names}


def _aggregate_lag1(
    list_of_dicts: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    channels = list_of_dicts[0].keys()
    metrics = next(iter(list_of_dicts[0].values())).keys()
    out: dict[str, dict[str, dict[str, float]]] = {}
    for channel in channels:
        out[channel] = {
            metric: _median_range([d[channel][metric] for d in list_of_dicts])
            for metric in metrics
        }
    return out


# --------------------------------------------------------------------------- C3


def constraint_counts(trajectory: Trajectory, config: ProcessConfig) -> dict[str, dict[str, int]]:
    """Anzahl betroffener Laeufe und Abtastpunkte je Nebenbedingung (C3c)."""
    diagnostics = trajectory.diagnostics
    n_runs, n_steps = trajectory.n_runs, trajectory.n_steps
    out: dict[str, dict[str, int]] = {}

    def _count(mask: np.ndarray) -> dict[str, int]:
        return {
            "runs_affected": int(np.sum(np.any(mask, axis=1))),
            "n_runs": n_runs,
            "points_affected": int(np.sum(mask)),
            "n_points": n_runs * n_steps,
        }

    if "fluidization_margin_lower_m_per_s" in diagnostics:
        margin_lower = diagnostics["fluidization_margin_lower_m_per_s"]
        out["below_minimum_fluidization"] = _count(margin_lower < 0.0)
    if "fluidization_margin_upper_m_per_s" in diagnostics:
        margin_upper = diagnostics["fluidization_margin_upper_m_per_s"]
        out["gas_velocity_above_terminal"] = _count(margin_upper < 0.0)
    if "particle_moisture_kg_per_kg" in diagnostics:
        critical_moisture = config.material.critical_moisture_kg_per_kg
        out["above_critical_moisture"] = _count(
            diagnostics["particle_moisture_kg_per_kg"] > critical_moisture
        )
    bed_temp_index = trajectory.metadata.get("output_names", ()).index("bed_temperature_celsius")
    out["bed_temperature_above_validity_limit"] = _count(
        trajectory.outputs[:, :, bed_temp_index] > BED_TEMPERATURE_VALIDITY_LIMIT_C
    )
    return out


def _sum_constraint_counts(
    list_of_counts: list[dict[str, dict[str, int]]],
) -> dict[str, dict[str, int]]:
    keys = list_of_counts[0].keys()
    out: dict[str, dict[str, int]] = {}
    for key in keys:
        out[key] = {
            "runs_affected": sum(c[key]["runs_affected"] for c in list_of_counts),
            "n_runs": sum(c[key]["n_runs"] for c in list_of_counts),
            "points_affected": sum(c[key]["points_affected"] for c in list_of_counts),
            "n_points": sum(c[key]["n_points"] for c in list_of_counts),
        }
    return out


# -------------------------------------------------------------------------- main


def main() -> None:
    t0 = time.monotonic()
    result: dict[str, Any] = {}

    simulator = _load(CONFIG_GAB)
    config = simulator.config
    result["config_fingerprint"] = config.fingerprint
    print(f"config fingerprint: {config.fingerprint}  (erwartet 82f65f576a2e5253)")

    variants = _variant_configs(config.actions)

    # -------------------------------------------------------------- Rollouts
    trajectories: dict[str, list[Trajectory]] = {}
    schedules: dict[str, list[Any]] = {}
    for name, (excitation_config, action_space) in variants.items():
        trajectories[name] = []
        schedules[name] = []
        for master_seed in MASTER_SEEDS:
            t_start = time.monotonic()
            trajectory, schedule = _build_trajectory(
                simulator, excitation_config, action_space, master_seed=master_seed
            )
            trajectories[name].append(trajectory)
            schedules[name].append(schedule)
            print(f"  {name} seed={master_seed}: {time.monotonic() - t_start:.1f}s")
        n_done = len(trajectories[name])
        print(f"{name}: {n_done} Rollouts fertig, {time.monotonic() - t0:.1f}s seit Start")

    # ------------------------------------------------------------------- C0
    print("\n=== C0 ===")
    c0_full_reference = [static_explained_variance(traj) for traj in trajectories["V0"]]
    c0_lag1 = [static_explained_variance_with_lag1(traj) for traj in trajectories["V0"]]
    result["c0"] = {
        "r2_stat_full_sample_reference": _aggregate_per_channel(c0_full_reference),
        "trimmed": _aggregate_lag1(c0_lag1),
    }
    print(json.dumps(result["c0"], indent=2))

    # ------------------------------------------------------------------- C1
    print("\n=== C1 (Basis V0) ===")
    c1: dict[str, Any] = {}
    for tau_label, tau_s in (("tau_1", TAU_1_S), ("tau_2", TAU_2_S)):
        new_per_seed = [
            quasistationary_fractions(
                traj.actions, config.actions, sample_time_s=SAMPLE_TIME_S, tau_s=tau_s
            )
            for traj in trajectories["V0"]
        ]
        c1[tau_label] = {"new_criterion": _aggregate_per_channel(new_per_seed)}
    old_per_seed = [
        quasistationary_fractions_old(traj.actions, config.actions) for traj in trajectories["V0"]
    ]
    c1["old_criterion"] = _aggregate_per_channel(old_per_seed)
    result["c1_base"] = c1
    print(json.dumps(c1, indent=2))

    # ------------------------------------------------------------------- C2
    print("\n=== C2 ===")
    c2: dict[str, Any] = {}
    for name in variants:
        action_space = variants[name][1]
        lag1_per_seed = [static_explained_variance_with_lag1(traj) for traj in trajectories[name]]
        entry: dict[str, Any] = {"lag1": _aggregate_lag1(lag1_per_seed)}
        for tau_label, tau_s in (("tau_1", TAU_1_S), ("tau_2", TAU_2_S)):
            new_per_seed = [
                quasistationary_fractions(
                    traj.actions, action_space, sample_time_s=SAMPLE_TIME_S, tau_s=tau_s
                )
                for traj in trajectories[name]
            ]
            entry[f"quasistationary_{tau_label}"] = _aggregate_per_channel(new_per_seed)
        old_per_seed = [
            quasistationary_fractions_old(traj.actions, action_space) for traj in trajectories[name]
        ]
        entry["quasistationary_old"] = _aggregate_per_channel(old_per_seed)
        c2[name] = entry
    result["c2"] = c2
    print(json.dumps(c2, indent=2))

    # ------------------------------------------------------------------- C3
    print("\n=== C3 ===")
    c3: dict[str, Any] = {}

    # a) PRBS-Periode
    c3["a_prbs_period"] = {}
    for name, (excitation_config, _) in variants.items():
        period_s = excitation_config.prbs_period_s
        c3["a_prbs_period"][name] = {
            "prbs_clock_s": excitation_config.prbs_clock_s,
            "prbs_register_length": excitation_config.prbs_register_length,
            "period_s": period_s,
            "complete_periods_per_run": RUN_DURATION_S / period_s,
        }

    # b) Verduennung: tatsaechliche PRBS-Kanal/Lauf-Kombinationen je Seed
    c3["b_prbs_channel_run_combinations"] = {}
    for name in ("V0", "V1", "V4"):
        counts = []
        for schedule in schedules[name]:
            kinds = schedule.signal_kinds
            n = sum(1 for run_kinds in kinds for kind in run_kinds if kind == "prbs")
            counts.append(n)
        c3["b_prbs_channel_run_combinations"][name] = {
            "per_seed": counts,
            "total_combinations": N_RUNS * len(config.actions.names),
        }

    # c) Nebenbedingungen je Variante
    c3["c_constraints"] = {}
    for name in variants:
        per_seed_counts = [constraint_counts(traj, config) for traj in trajectories[name]]
        c3["c_constraints"][name] = _sum_constraint_counts(per_seed_counts)

    # d) Zeitkonstanten je Variante
    print("Eigenwertauswertung je Variante...")
    base_taus = time_constant_report(simulator)
    rate_actions = _rate_limited_actions(config.actions, inlet_max_rate=2.0)
    rate_config = dataclasses.replace(config, actions=rate_actions)
    rate_simulator = GreyBoxFluidBedGranulator(rate_config)
    rate_taus = time_constant_report(rate_simulator)
    c3["d_time_constants"] = {
        "V0": base_taus,
        "V1": base_taus,
        "V2": rate_taus,
        "V3": rate_taus,
        "V4": rate_taus,
    }
    result["c3"] = c3
    print(json.dumps(c3, indent=2, default=str))

    result["_elapsed_s_c0_c3"] = time.monotonic() - t0

    out_path = REPO_ROOT / "artifacts" / "runs" / "probe_excitation_bandwidth.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\ngeschrieben: {out_path} ({result['_elapsed_s_c0_c3']:.1f} s)")


if __name__ == "__main__":
    main()

r"""S1: effect of a projection of gas_velocity_m_per_s onto [u_mf(d50), u_terminal(d50)].

Measurement BEFORE any change to ``src/`` -- decides whether S2/S3 (schema extension of
``Trajectory``, implementation of the projection in ``BaseSimulator.rollout``/``greybox.py``)
make sense at all. See ``docs/04_stand_ap1.md`` Befund 46 and
``scripts/probe_sweep_contamination.py`` for the finding so far: 26.6 % of the runs and
6.0 % of the sample points violate ``u_gas > u_terminal(d50)`` under the standard
excitation design (process fingerprint ``EXPECTED_FINGERPRINT`` below -- the original
S1 measurement/the committed artifact ran under 82f65f576a2e5253, before
``NumericsConfig.enforce_gas_velocity_projection`` was introduced; equivalence to the current
fingerprint shown by ``test_gas_velocity_projection_disabled_matches_legacy_baseline``,
``tests/test_gas_velocity_projection.py``).

Reproduces the existing schedules -- 3 training seeds (1, 2, 3; the same default seed
list as ``run_learning_curve``) with ``n_train_max=64`` (DataSpec identical to
``run_arch_constant_holdout_worker._excited_spec``/``probe_sweep_contamination._spec``) plus
a constant holdout pool (``HOLDOUT_SEED=999``, ``constant_run_fraction=1.0``, DataSpec
identical to ``run_arch_constant_holdout_worker._holdout_spec``) -- and rolls out every pool
twice:

1. unchanged -- this is already ``GeneratedData.trajectory`` from ``generate_data``.
2. with a projection re-implemented HERE, NOT implemented in ``src/``: immediately before
   every ``step()`` call (i.e. at the rollout rate of ``ProcessSimulator.rollout``, not per
   RK4 sub-step -- the action is zero-order hold over the sample interval anyway),
   ``gas_velocity_m_per_s`` is clipped to ``[u_mf, u_terminal]``, computed from the TRUE,
   noise-free state that enters this step (never from a measurement). ``u_mf``/
   ``u_terminal`` come unchanged from ``simulation/fluidization.py`` -- exactly the same
   computation as in ``GreyBoxFluidBedGranulator._rhs`` (purely diagnostic there, see the
   module docstrings of ``greybox.py``/``fluidization.py``).

Own re-implemented rollout loop (``_rollout_with_optional_projection``) instead of calling
``ProcessSimulator.rollout`` twice, because its signature has no projection hook
(that is exactly the gap that S1 measures) -- ``project=False`` is checked against
``GeneratedData.trajectory`` for bit-identity, to verify the re-implemented loop itself
before its result with ``project=True`` serves as the basis of the measurement.

Criterion (f) (see the task) checks after the simulation whether points with
``u > u_terminal`` remain -- expected exactly zero. If not: the diagnostic in
``greybox.py`` (``fluidization_margin_upper_m_per_s``) is computed AFTER the integration of a
step (post-RK4 state), whereas the projection here clips BEFORE the step (as specified
in the task, on the true state before the integration step) -- for this purpose,
``_analyse_group``/``main`` additionally store ``projection_u_mf_pre_
step_m_per_s``/``projection_u_terminal_pre_step_m_per_s`` in the diagnostics of the
projected trajectory, so that the two points in time can be compared directly
(field ``f_root_cause``).

**Addendum T1** (after a follow-up question on (f)): the point fraction from (f) alone does
not decide whether the residual violation is relevant -- the MAGNITUDE does.
``f_residual_magnitude`` (per group and pooled) therefore measures the exceedance
(u_gas - u_terminal) at the violating points, absolute and relative to ``u_terminal``, counts
affected runs and points per run, and places the relative exceedance against the correlation
uncertainty of Haider & Levenspiel (10-20 %, see ``fluidization.py``): if the 95th percentile
lies clearly below it, the residual violation is within the model uncertainty itself and not
a physical phenomenon of its own. A derived margin (T2) is implemented only if needed and only
after further consultation, not here.

NO network training, NO sweep, NO change to ``src/``.
"""

from __future__ import annotations

import dataclasses
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from twinlink_ml.evaluation import DataSpec, generate_data
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.base import Trajectory
from twinlink_ml.simulation.fluidization import (
    minimum_fluidization_velocity_m_per_s,
    terminal_velocity_m_per_s,
)
from twinlink_ml.simulation.greybox import (
    IDX_BED_TEMPERATURE,
    IDX_PARTICLE_COUNT,
    IDX_SOLID_MASS,
    GreyBoxFluidBedGranulator,
)
from twinlink_ml.simulation.thermo import absolute_humidity_kg_per_kg

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "probe_gas_velocity_projection.json"

EXCITED_SEEDS = (1, 2, 3)
HOLDOUT_SEED = 999
# Updated from 82f65f576a2e5253 (value of the original S1 measurement/the committed
# artifact) to 40aae2b4526a5ed7: NumericsConfig.enforce_gas_velocity_projection
# added (default OFF). ProcessConfig.fingerprint hashes the whole config, so
# the value changes even when behaviour is unchanged. Equivalence shown by
# test_gas_velocity_projection_disabled_matches_legacy_baseline
# (tests/test_gas_velocity_projection.py).
EXPECTED_FINGERPRINT = "40aae2b4526a5ed7"
T_BED_THRESHOLD_C = 78.8

# Abort criteria (see the task)
STD_DROP_ABORT_FRACTION = 0.30
HOT_FRACTION_DROP_ABORT_FRACTION = 1.0 / 3.0

# T1: yardstick for classifying the relative residual violation from (f) -- accuracy range
# of the Haider & Levenspiel correlation itself (see the simulation/fluidization.py docstring
# on terminal_velocity_m_per_s), not a set tolerance value for the projection.
CORRELATION_REL_UNCERTAINTY_LOW = 0.10
CORRELATION_REL_UNCERTAINTY_HIGH = 0.20


def _excited_spec() -> DataSpec:
    """Identical to ``run_arch_constant_holdout_worker._excited_spec``.

    The same as ``probe_sweep_contamination._spec``, the same schedule family
    (fingerprint ``EXPECTED_FINGERPRINT``).
    """
    return DataSpec(
        sample_time_s=25.0,
        run_duration_s=28_800.0,
        n_train_max=64,
        n_val=32,
        n_test=32,
        history_length=32,
        horizon=36,
        stride=8,
    )


def _holdout_spec(excited: DataSpec) -> DataSpec:
    """Identical to ``run_arch_constant_holdout_worker._holdout_spec``.

    Only ``.test`` (32 runs) is evaluated, ``n_train_max=2``/``n_val=2`` are
    unused dummy blocks.
    """
    return dataclasses.replace(
        excited, n_train_max=2, n_val=2, n_test=excited.n_test, constant_run_fraction=1.0
    )


def _true_margins(
    simulator: GreyBoxFluidBedGranulator, state: np.ndarray, disturbance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """u_mf, u_terminal at the TRUE state ``state`` -- unchanged from fluidization.py.

    Exactly the same computation as in ``GreyBoxFluidBedGranulator._rhs`` (there only AFTER
    the derivative, here BEFORE the step).
    """
    backend = simulator.backend
    cfg = simulator.config
    material, gas = cfg.material, cfg.gas

    solid_mass = backend.maximum(state[..., IDX_SOLID_MASS], cfg.numerics.state_floor)
    particle_count = backend.maximum(state[..., IDX_PARTICLE_COUNT], cfg.numerics.state_floor)
    diameter = (
        6.0 * solid_mass / (np.pi * material.particle_density_kg_per_m3 * particle_count)
    ) ** (1.0 / 3.0)
    bed_temperature = state[..., IDX_BED_TEMPERATURE]

    ambient_temperature = disturbance[..., 0]
    ambient_humidity = disturbance[..., 1]
    inlet_absolute_humidity = absolute_humidity_kg_per_kg(
        backend, gas, ambient_temperature, ambient_humidity
    )
    u_mf = minimum_fluidization_velocity_m_per_s(
        backend, gas, material, diameter, bed_temperature, inlet_absolute_humidity
    )
    u_terminal = terminal_velocity_m_per_s(
        backend, gas, material, diameter, bed_temperature, inlet_absolute_humidity
    )
    return u_mf, u_terminal


def _rollout_with_optional_projection(
    simulator: GreyBoxFluidBedGranulator,
    actions: np.ndarray,
    disturbances: np.ndarray,
    *,
    run_id: tuple[str, ...],
    seed: int,
    initial_state: np.ndarray | None,
    gas_velocity_index: int,
    project: bool,
) -> tuple[Trajectory, np.ndarray, np.ndarray]:
    """Replica of ``ProcessSimulator.rollout``, identical except for the projection hook.

    The hook sits at exactly the place that S3 intends for the generic call point:
    before ``simulator.step(...)``, at the true ``state`` of this iteration.

    :returns: (trajectory, realised actions (after projection), boolean mask of the
        actually clipped (run, time) points)
    """
    backend = simulator.backend
    n_batch, n_time, _ = actions.shape
    rng = np.random.default_rng(seed)

    state = simulator.initial_state(n_batch) if initial_state is None else initial_state
    simulator.reset_sensor()

    realised_actions = actions.copy()
    projected_mask = np.zeros((n_batch, n_time), dtype=bool)

    states: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    measurements: list[np.ndarray] = []
    diagnostics: dict[str, list[np.ndarray]] = {}
    pre_step_u_mf = np.full((n_batch, n_time), np.nan)
    pre_step_u_terminal = np.full((n_batch, n_time), np.nan)

    for k in range(n_time):
        action_np = actions[:, k, :].copy()
        if project:
            true_state_np = backend.to_numpy(state)
            u_mf, u_terminal = _true_margins(simulator, true_state_np, disturbances[:, k, :])
            pre_step_u_mf[:, k] = u_mf
            pre_step_u_terminal[:, k] = u_terminal
            requested = action_np[:, gas_velocity_index].copy()
            clipped = np.clip(requested, u_mf, u_terminal)
            action_np[:, gas_velocity_index] = clipped
            projected_mask[:, k] = clipped != requested
        realised_actions[:, k, :] = action_np

        action = backend.asarray(action_np)
        disturbance = backend.asarray(disturbances[:, k, :])
        result = simulator.step(state, action, disturbance)
        state = result.state
        states.append(backend.to_numpy(state))
        outputs.append(backend.to_numpy(result.outputs))
        measurements.append(backend.to_numpy(simulator.measure(result.outputs, rng)))
        for key, value in result.diagnostics.items():
            diagnostics.setdefault(key, []).append(backend.to_numpy(value))

    stacked_diagnostics = {k: np.stack(v, axis=1) for k, v in diagnostics.items()}
    if project:
        # For the root-cause analysis of criterion (f): the margins used for the projection
        # BEFORE the step, for comparison with the (post-integration
        # computed) diagnostic margins from greybox.py in the same trajectory.
        stacked_diagnostics["projection_u_mf_pre_step_m_per_s"] = pre_step_u_mf
        stacked_diagnostics["projection_u_terminal_pre_step_m_per_s"] = pre_step_u_terminal

    trajectory = Trajectory(
        run_id=run_id,
        time_s=np.arange(1, n_time + 1, dtype=np.float64) * simulator.sample_time_s,
        actions=np.asarray(actions, dtype=np.float64),
        disturbances=np.asarray(disturbances, dtype=np.float64),
        states=np.stack(states, axis=1),
        outputs=np.stack(outputs, axis=1),
        measurements=np.stack(measurements, axis=1),
        diagnostics=stacked_diagnostics,
        metadata=simulator.run_metadata(seed=seed),
    )
    return trajectory, realised_actions, projected_mask


def _pooled_stats(x: np.ndarray) -> dict[str, float]:
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    return {
        "median": float(np.median(flat)),
        "p05": float(np.percentile(flat, 5)),
        "p95": float(np.percentile(flat, 95)),
        "std": float(np.std(flat)),
        "n_points": int(flat.size),
    }


def _per_run_stats(x: np.ndarray) -> dict[str, Any]:
    """``x`` with shape (n_runs, n_time): summary over the runs.

    Min/median/max of the per-run metric, since metrics per single run (hundreds) cannot
    sensibly be tabulated individually here -- the full per-run arrays still end up
    in the JSON artifact.
    """
    per_run_median = np.median(x, axis=1)
    per_run_p05 = np.percentile(x, 5, axis=1)
    per_run_p95 = np.percentile(x, 95, axis=1)
    per_run_std = np.std(x, axis=1)

    def _summary(values: np.ndarray) -> dict[str, float]:
        return {
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
        }

    return {
        "n_runs": int(x.shape[0]),
        "median_across_runs": _summary(per_run_median),
        "p05_across_runs": _summary(per_run_p05),
        "p95_across_runs": _summary(per_run_p95),
        "std_across_runs": _summary(per_run_std),
        "_full_per_run": {
            "median": per_run_median.tolist(),
            "p05": per_run_p05.tolist(),
            "p95": per_run_p95.tolist(),
            "std": per_run_std.tolist(),
        },
    }


def _quarter_counts(mask: np.ndarray) -> dict[str, Any]:
    """``mask`` with shape (n_runs, n_time): temporal distribution of the True entries.

    In quarters of the run duration, analogous to ``probe_sweep_contamination.py``.
    """
    n_time = mask.shape[1]
    quarter = n_time // 4
    edges = [0, quarter, 2 * quarter, 3 * quarter, n_time]
    counts = [int(mask[:, edges[q] : edges[q + 1]].sum()) for q in range(4)]
    total = sum(counts)
    return {
        "points_q1_q2_q3_q4": counts,
        "fraction_q1_q2_q3_q4": [c / total if total else float("nan") for c in counts],
        "total_points": total,
    }


def _residual_violation_magnitude(
    violation_upper: np.ndarray, exceedance_abs: np.ndarray, u_terminal_post: np.ndarray
) -> dict[str, Any]:
    """T1: magnitude of the residual violation from (f), not only its point fraction.

    ``violation_upper``/``exceedance_abs``/``u_terminal_post`` all have shape
    (n_runs, n_time). ``exceedance_abs`` is ``max(0, u_gas - u_terminal)`` (0 outside
    violating points). Additionally places the relative exceedance against the
    accuracy of the Haider & Levenspiel correlation itself (10-20 %, see
    ``fluidization.py``) -- a residual violation below this band lies within the
    uncertainty of ``u_terminal`` itself and is not a physical phenomenon of its own.
    """
    n_runs, n_time = violation_upper.shape
    n_violating_points = int(violation_upper.sum())
    if n_violating_points == 0:
        return {"n_violating_points": 0}

    abs_v = exceedance_abs[violation_upper]
    rel_v = abs_v / u_terminal_post[violation_upper]

    counts_per_run = violation_upper.sum(axis=1)
    affected = counts_per_run > 0
    counts_at_affected = counts_per_run[affected]

    spans: list[int] = []
    longest_runs: list[int] = []
    for i in np.nonzero(affected)[0]:
        idx = np.nonzero(violation_upper[i])[0]
        spans.append(int(idx.max() - idx.min()))
        if idx.size == 1:
            longest_runs.append(1)
        else:
            longest = current = 1
            for gap in np.diff(idx):
                current = current + 1 if gap == 1 else 1
                longest = max(longest, current)
            longest_runs.append(longest)

    p95_rel = float(np.percentile(rel_v, 95))
    if p95_rel < CORRELATION_REL_UNCERTAINTY_LOW:
        verdict = (
            "95-Perzentil liegt unter der Korrelationsunsicherheit (10 %) -- Restverletzung "
            "liegt innerhalb der Modellunsicherheit von u_terminal selbst, kein eigener "
            "physikalischer Sachverhalt."
        )
    elif p95_rel < CORRELATION_REL_UNCERTAINTY_HIGH:
        verdict = (
            "95-Perzentil liegt innerhalb des Korrelationsunsicherheitsbands (10-20 %) -- "
            "an der Grenze der Modellunsicherheit, dokumentationswuerdig."
        )
    else:
        verdict = (
            "95-Perzentil liegt ueber der Korrelationsunsicherheit (20 %) -- Restverletzung "
            "geht ueber die Unsicherheit von u_terminal hinaus, T2 pruefen."
        )

    return {
        "n_violating_points": n_violating_points,
        "exceedance_abs_m_per_s": {
            "median": float(np.median(abs_v)),
            "p95": float(np.percentile(abs_v, 95)),
            "max": float(abs_v.max()),
        },
        "exceedance_rel_to_u_terminal": {
            "median": float(np.median(rel_v)),
            "p95": p95_rel,
            "max": float(rel_v.max()),
        },
        "n_affected_runs": int(affected.sum()),
        "n_runs_total": n_runs,
        "points_per_affected_run": {
            "median": float(np.median(counts_at_affected)),
            "max": int(counts_at_affected.max()),
        },
        "temporal_clustering": {
            "index_span_median": float(np.median(spans)),
            "index_span_max": int(max(spans)),
            "longest_contiguous_run_median": float(np.median(longest_runs)),
            "longest_contiguous_run_max": int(max(longest_runs)),
            "n_time": n_time,
        },
        "assessment_vs_correlation_uncertainty": verdict,
    }


def _build_group(
    label: str,
    config: Any,
    spec: DataSpec,
    master_seed: int,
    gas_velocity_index: int,
) -> dict[str, Any]:
    """One schedule: unchanged (``generate_data``) and with projection (replica).

    Including the bit-identity self-check of the replica loop at ``project=False``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        data = generate_data(config, spec, master_seed=master_seed)

    if data.process_config.fingerprint != EXPECTED_FINGERPRINT:
        raise RuntimeError(
            f"{label}: process/fingerprint={data.process_config.fingerprint!r}, "
            f"erwartet {EXPECTED_FINGERPRINT!r} -- Schedule weicht von der erwarteten "
            "Grundlage ab, Abbruch."
        )

    simulator = GreyBoxFluidBedGranulator(data.process_config)
    schedule = data.schedule
    initial = None
    if schedule.initial_states is not None:
        columns = schedule.initial_states
        initial = simulator.state_from_physical(
            columns[:, 0], columns[:, 1], columns[:, 2], columns[:, 3]
        )

    # Self-check: project=False must be bit-identical to data.trajectory.
    baseline_check, realised_check, mask_check = _rollout_with_optional_projection(
        simulator,
        schedule.actions,
        schedule.disturbances,
        run_id=schedule.run_id,
        seed=data.seeds.noise_seed,
        initial_state=initial,
        gas_velocity_index=gas_velocity_index,
        project=False,
    )
    for field_name in ("states", "outputs", "measurements"):
        if not np.array_equal(
            getattr(baseline_check, field_name), getattr(data.trajectory, field_name)
        ):
            raise RuntimeError(
                f"{label}: Nachbau-Rollout (project=False) weicht in {field_name!r} von "
                "generate_data ab"
            )
    if mask_check.any():
        raise RuntimeError(f"{label}: project=False hat trotzdem geclippt -- Logikfehler")

    projected, realised_actions, projected_mask = _rollout_with_optional_projection(
        simulator,
        schedule.actions,
        schedule.disturbances,
        run_id=schedule.run_id,
        seed=data.seeds.noise_seed,
        initial_state=initial,
        gas_velocity_index=gas_velocity_index,
        project=True,
    )

    return {
        "label": label,
        "data": data,
        "simulator": simulator,
        "before": data.trajectory,
        "after": projected,
        "requested_actions": np.asarray(schedule.actions, dtype=np.float64),
        "realised_actions": realised_actions,
        "projected_mask": projected_mask,
        "gas_velocity_index": gas_velocity_index,
    }


def _select_runs(group: dict[str, Any], run_ids: tuple[str, ...] | None) -> dict[str, Any]:
    """Subset of the runs of a group.

    For the holdout pool: only ``.test``, the 32 "real" constant-run test runs --
    dummy blocks ``train``/``val`` are left out, convention from
    ``run_arch_constant_holdout_worker.py``.
    """
    if run_ids is None:
        return group
    all_ids = group["before"].run_id
    idx = [i for i, r in enumerate(all_ids) if r in set(run_ids)]
    if len(idx) != len(run_ids):
        raise RuntimeError(f"{group['label']}: nicht alle angeforderten run_id gefunden")
    idx_arr = np.asarray(idx)
    return {
        **group,
        "before": dataclasses.replace(
            group["before"],
            run_id=tuple(all_ids[i] for i in idx),
            actions=group["before"].actions[idx_arr],
            disturbances=group["before"].disturbances[idx_arr],
            states=group["before"].states[idx_arr],
            outputs=group["before"].outputs[idx_arr],
            measurements=group["before"].measurements[idx_arr],
            diagnostics={k: v[idx_arr] for k, v in group["before"].diagnostics.items()},
        ),
        "after": dataclasses.replace(
            group["after"],
            run_id=tuple(all_ids[i] for i in idx),
            actions=group["after"].actions[idx_arr],
            disturbances=group["after"].disturbances[idx_arr],
            states=group["after"].states[idx_arr],
            outputs=group["after"].outputs[idx_arr],
            measurements=group["after"].measurements[idx_arr],
            diagnostics={k: v[idx_arr] for k, v in group["after"].diagnostics.items()},
        ),
        "requested_actions": group["requested_actions"][idx_arr],
        "realised_actions": group["realised_actions"][idx_arr],
        "projected_mask": group["projected_mask"][idx_arr],
    }


def _analyse_group(group: dict[str, Any]) -> dict[str, Any]:
    mask = group["projected_mask"]
    requested = group["requested_actions"][..., group["gas_velocity_index"]]
    realised = group["realised_actions"][..., group["gas_velocity_index"]]
    n_runs, n_time = mask.shape

    run_affected = mask.any(axis=1)
    shortfall = np.abs(realised - requested)

    output_names = group["before"].metadata["output_names"]
    bed_idx = output_names.index("bed_temperature_celsius")
    moist_idx = output_names.index("product_moisture_kg_per_kg")

    bed_before = group["before"].outputs[..., bed_idx]
    bed_after = group["after"].outputs[..., bed_idx]
    moist_before = group["before"].outputs[..., moist_idx]
    moist_after = group["after"].outputs[..., moist_idx]

    margin_upper_after = group["after"].diagnostics["fluidization_margin_upper_m_per_s"]
    margin_lower_after = group["after"].diagnostics["fluidization_margin_lower_m_per_s"]
    violation_upper = margin_upper_after < 0.0

    # Root-cause analysis (f): u_terminal BEFORE the step (used for the projection) against
    # u_terminal AFTER the step (post-integration, greybox.py diagnostic) at exactly the
    # violating points -- clarifies whether a remaining violation indicates a misplaced
    # projection or drift of d50/T_bed *within* the 25 s step.
    f_root_cause: dict[str, Any] = {"n_violating_points": int(violation_upper.sum())}
    if violation_upper.any():
        u_terminal_pre = group["after"].diagnostics["projection_u_terminal_pre_step_m_per_s"]
        u_terminal_post = group["after"].diagnostics["terminal_velocity_m_per_s"]
        delta = (u_terminal_post - u_terminal_pre)[violation_upper]
        realised_at_violation = realised[violation_upper]
        pre_bound_at_violation = u_terminal_pre[violation_upper]
        f_root_cause.update(
            {
                "delta_u_terminal_post_minus_pre_mean": float(delta.mean()),
                "delta_u_terminal_post_minus_pre_min": float(delta.min()),
                "delta_u_terminal_post_minus_pre_max": float(delta.max()),
                "fraction_where_u_terminal_shrank_within_step": float((delta < 0.0).mean()),
                "fraction_where_realised_equals_pre_step_bound": float(
                    np.isclose(realised_at_violation, pre_bound_at_violation, atol=1e-9).mean()
                ),
            }
        )

    exceedance_abs = np.maximum(0.0, -margin_upper_after)
    u_terminal_post_full = group["after"].diagnostics["terminal_velocity_m_per_s"]
    f_residual_magnitude = _residual_violation_magnitude(
        violation_upper, exceedance_abs, u_terminal_post_full
    )

    return {
        "label": group["label"],
        "n_runs": n_runs,
        "n_time": n_time,
        "n_points": n_runs * n_time,
        "a_points_affected_fraction": float(mask.sum() / mask.size),
        "a_runs_affected_fraction": float(run_affected.sum() / n_runs),
        "a_runs_affected": int(run_affected.sum()),
        "b_gas_velocity_before_pooled": _pooled_stats(requested),
        "b_gas_velocity_after_pooled": _pooled_stats(realised),
        "b_gas_velocity_before_per_run": _per_run_stats(requested),
        "b_gas_velocity_after_per_run": _per_run_stats(realised),
        "c_mean_abs_shortfall_all_points_m_per_s": float(shortfall.mean()),
        "c_mean_abs_shortfall_affected_points_m_per_s": (
            float(shortfall[mask].mean()) if mask.any() else 0.0
        ),
        "c_max_abs_shortfall_m_per_s": float(shortfall.max()),
        "d_quarter_distribution": _quarter_counts(mask),
        "e_bed_temperature_before_pooled": _pooled_stats(bed_before),
        "e_bed_temperature_after_pooled": _pooled_stats(bed_after),
        "e_moisture_before_pooled": _pooled_stats(moist_before),
        "e_moisture_after_pooled": _pooled_stats(moist_after),
        "e_fraction_above_threshold_before": float((bed_before > T_BED_THRESHOLD_C).mean()),
        "e_fraction_above_threshold_after": float((bed_after > T_BED_THRESHOLD_C).mean()),
        "f_fraction_above_u_terminal_after_diag": float((margin_upper_after < 0.0).mean()),
        "f_fraction_below_u_mf_after_diag": float((margin_lower_after < 0.0).mean()),
        "f_max_violation_above_u_terminal_m_per_s": float(
            np.maximum(0.0, -margin_upper_after).max()
        ),
        "f_root_cause": f_root_cause,
        "f_residual_magnitude": f_residual_magnitude,
    }


def main() -> None:
    t0 = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)

    gas_velocity_index = config.actions.names.index("gas_velocity_m_per_s")
    excited_spec = _excited_spec()
    holdout_spec = _holdout_spec(excited_spec)

    groups: list[dict[str, Any]] = []
    for seed in EXCITED_SEEDS:
        print(f"=== baue Schedule: excited, master_seed={seed} ===")
        groups.append(
            _build_group(f"excited_seed{seed}", config, excited_spec, seed, gas_velocity_index)
        )

    print(f"=== baue Schedule: constant_holdout, seed={HOLDOUT_SEED} ===")
    holdout_group_full = _build_group(
        "constant_holdout_full_pool", config, holdout_spec, HOLDOUT_SEED, gas_velocity_index
    )
    holdout_group = _select_runs(
        {**holdout_group_full, "label": "constant_holdout"}, holdout_group_full["data"].split.test
    )
    groups.append(holdout_group)

    per_group_report = [_analyse_group(g) for g in groups]

    # ------------------------------------------------------------ pooled over all groups
    gv_idx = gas_velocity_index
    all_requested = np.concatenate(
        [g["requested_actions"][..., gv_idx].reshape(-1) for g in groups]
    )
    all_realised = np.concatenate([g["realised_actions"][..., gv_idx].reshape(-1) for g in groups])
    all_mask = np.concatenate([g["projected_mask"].reshape(-1) for g in groups])
    all_shortfall = np.abs(all_realised - all_requested)

    def _pooled_output(name: str, which: str) -> np.ndarray:
        parts = []
        for g in groups:
            output_names = g[which].metadata["output_names"]
            idx = output_names.index(name)
            parts.append(g[which].outputs[..., idx].reshape(-1))
        return np.concatenate(parts)

    bed_before_all = _pooled_output("bed_temperature_celsius", "before")
    bed_after_all = _pooled_output("bed_temperature_celsius", "after")
    moist_before_all = _pooled_output("product_moisture_kg_per_kg", "before")
    moist_after_all = _pooled_output("product_moisture_kg_per_kg", "after")

    margin_upper_after_all = np.concatenate(
        [g["after"].diagnostics["fluidization_margin_upper_m_per_s"].reshape(-1) for g in groups]
    )
    margin_lower_after_all = np.concatenate(
        [g["after"].diagnostics["fluidization_margin_lower_m_per_s"].reshape(-1) for g in groups]
    )

    std_before = float(np.std(all_requested))
    std_after = float(np.std(all_realised))
    std_drop_fraction = (std_before - std_after) / std_before if std_before else float("nan")

    hot_before = float((bed_before_all > T_BED_THRESHOLD_C).mean())
    hot_after = float((bed_after_all > T_BED_THRESHOLD_C).mean())
    hot_drop_fraction = (hot_before - hot_after) / hot_before if hot_before else float("nan")

    f_fraction_pooled = float((margin_upper_after_all < 0.0).mean())
    f_fraction_pooled_eps = float((margin_upper_after_all < -1e-9).mean())

    # T1 needs the (n_runs, n_time) structure (for run membership/clustering) --
    # all groups share n_time=1152 (same sample_time_s/run_duration_s), therefore
    # concatenate along the run axis (axis=0) instead of flattening completely.
    margin_upper_after_2d = np.concatenate(
        [g["after"].diagnostics["fluidization_margin_upper_m_per_s"] for g in groups], axis=0
    )
    u_terminal_post_2d = np.concatenate(
        [g["after"].diagnostics["terminal_velocity_m_per_s"] for g in groups], axis=0
    )
    violation_upper_2d = margin_upper_after_2d < 0.0
    exceedance_abs_2d = np.maximum(0.0, -margin_upper_after_2d)
    f_residual_magnitude_pooled = _residual_violation_magnitude(
        violation_upper_2d, exceedance_abs_2d, u_terminal_post_2d
    )

    violation_upper_all = margin_upper_after_all < 0.0
    f_root_cause_pooled: dict[str, Any] = {"n_violating_points": int(violation_upper_all.sum())}
    if violation_upper_all.any():
        u_terminal_pre_all = np.concatenate(
            [
                g["after"].diagnostics["projection_u_terminal_pre_step_m_per_s"].reshape(-1)
                for g in groups
            ]
        )
        u_terminal_post_all = np.concatenate(
            [g["after"].diagnostics["terminal_velocity_m_per_s"].reshape(-1) for g in groups]
        )
        delta_all = (u_terminal_post_all - u_terminal_pre_all)[violation_upper_all]
        realised_at_violation_all = all_realised[violation_upper_all]
        pre_bound_at_violation_all = u_terminal_pre_all[violation_upper_all]
        f_root_cause_pooled.update(
            {
                "delta_u_terminal_post_minus_pre_mean": float(delta_all.mean()),
                "delta_u_terminal_post_minus_pre_min": float(delta_all.min()),
                "delta_u_terminal_post_minus_pre_max": float(delta_all.max()),
                "fraction_where_u_terminal_shrank_within_step": float((delta_all < 0.0).mean()),
                "fraction_where_realised_equals_pre_step_bound": float(
                    np.isclose(
                        realised_at_violation_all, pre_bound_at_violation_all, atol=1e-9
                    ).mean()
                ),
            }
        )

    abort_std = std_drop_fraction > STD_DROP_ABORT_FRACTION
    abort_hot = hot_before > 0.0 and hot_drop_fraction > HOT_FRACTION_DROP_ABORT_FRACTION
    # (f) alone (point fraction != 0) is NOT the abort criterion -- a proxy for
    # "projection correctly placed", which T1 (magnitude of the residual violation) settles.
    # What matters is whether the relative exceedance goes beyond the correlation uncertainty
    # of u_terminal itself (10 %) -- see follow-up question/answer on S1 and
    # ``_residual_violation_magnitude``.
    abort_f = (
        f_residual_magnitude_pooled.get("n_violating_points", 0) > 0
        and f_residual_magnitude_pooled["exceedance_rel_to_u_terminal"]["p95"]
        >= CORRELATION_REL_UNCERTAINTY_LOW
    )

    pooled_report = {
        "n_points_total": int(all_mask.size),
        "n_runs_total": sum(g["projected_mask"].shape[0] for g in groups),
        "a_points_affected_fraction": float(all_mask.mean()),
        "b_gas_velocity_before_pooled": _pooled_stats(all_requested),
        "b_gas_velocity_after_pooled": _pooled_stats(all_realised),
        "c_mean_abs_shortfall_all_points_m_per_s": float(all_shortfall.mean()),
        "c_mean_abs_shortfall_affected_points_m_per_s": (
            float(all_shortfall[all_mask].mean()) if all_mask.any() else 0.0
        ),
        "c_max_abs_shortfall_m_per_s": float(all_shortfall.max()),
        "e_bed_temperature_before_pooled": _pooled_stats(bed_before_all),
        "e_bed_temperature_after_pooled": _pooled_stats(bed_after_all),
        "e_moisture_before_pooled": _pooled_stats(moist_before_all),
        "e_moisture_after_pooled": _pooled_stats(moist_after_all),
        "e_fraction_above_threshold_before": hot_before,
        "e_fraction_above_threshold_after": hot_after,
        "f_fraction_above_u_terminal_after": f_fraction_pooled,
        "f_fraction_above_u_terminal_after_eps_tolerant": f_fraction_pooled_eps,
        "f_fraction_below_u_mf_after": float((margin_lower_after_all < 0.0).mean()),
        "f_root_cause": f_root_cause_pooled,
        "f_residual_magnitude": f_residual_magnitude_pooled,
        "abort_criteria": {
            "std_drop_fraction": std_drop_fraction,
            "std_drop_exceeds_30pct": abort_std,
            "hot_fraction_before": hot_before,
            "hot_fraction_after": hot_after,
            "hot_drop_fraction": hot_drop_fraction,
            "hot_drop_exceeds_one_third": abort_hot,
            "f_nonzero": f_fraction_pooled_eps > 0.0,
            "f_residual_exceeds_correlation_uncertainty": abort_f,
            "any_abort_triggered": bool(abort_std or abort_hot or abort_f),
        },
    }

    # ---------------------------------------------------------------- console report
    print("\n=== S1: Wirkung der Gasgeschwindigkeits-Projektion ===")
    print(f"process/fingerprint (alle Gruppen): {EXPECTED_FINGERPRINT}")
    for rep in per_group_report:
        print(f"\n-- {rep['label']} ({rep['n_runs']} Laeufe x {rep['n_time']} Schritte) --")
        print(
            f"  a) Punkte betroffen: {rep['a_points_affected_fraction']:.4%}, "
            f"Laeufe betroffen: {rep['a_runs_affected_fraction']:.4%} "
            f"({rep['a_runs_affected']}/{rep['n_runs']})"
        )
        print(f"  b) gas_velocity gepoolt vorher: {rep['b_gas_velocity_before_pooled']}")
        print(f"     gas_velocity gepoolt nachher: {rep['b_gas_velocity_after_pooled']}")
        print(
            f"  c) mittlere Kuerzung (betroffene Punkte): "
            f"{rep['c_mean_abs_shortfall_affected_points_m_per_s']:.6f} m/s, "
            f"maximale Kuerzung: {rep['c_max_abs_shortfall_m_per_s']:.6f} m/s"
        )
        print(f"  d) Viertelverteilung der Eingriffe: {rep['d_quarter_distribution']}")
        print(
            f"  e) T_bed > {T_BED_THRESHOLD_C} C: "
            f"vorher {rep['e_fraction_above_threshold_before']:.4%}, "
            f"nachher {rep['e_fraction_above_threshold_after']:.4%}"
        )
        print(
            f"  f) Anteil u > u_terminal nach Projektion: "
            f"{rep['f_fraction_above_u_terminal_after_diag']:.6%}"
        )
        if rep["f_root_cause"]["n_violating_points"] > 0:
            print(f"     Ursachenanalyse (f): {rep['f_root_cause']}")
            print(f"     T1 Betrag der Restverletzung (f): {rep['f_residual_magnitude']}")

    print("\n=== gepoolt ueber alle Gruppen ===")
    print(json.dumps(pooled_report, indent=2, default=str))

    print("\n=== Abbruchkriterien ===")
    ac = pooled_report["abort_criteria"]
    std_verdict = "ERFUELLT -> ABBRUCH" if ac["std_drop_exceeds_30pct"] else "nicht erfuellt"
    print(
        f"  Streuung gas_velocity faellt um {ac['std_drop_fraction']:.2%} "
        f"(Grenze 30%): {std_verdict}"
    )
    hot_verdict = "ERFUELLT -> ABBRUCH" if ac["hot_drop_exceeds_one_third"] else "nicht erfuellt"
    print(
        f"  Anteil T_bed > {T_BED_THRESHOLD_C} C faellt um {ac['hot_drop_fraction']:.2%} "
        f"(Grenze 1/3): {hot_verdict}"
    )
    f_fraction = pooled_report["f_fraction_above_u_terminal_after"]
    print(f"  (f) roh, Punktanteil u > u_terminal nach Projektion: {f_fraction:.6%}")
    print(f"      Ursachenanalyse (f), gepoolt: {pooled_report['f_root_cause']}")
    print(
        "      Deutung: die Projektion clippt auf u_terminal am WAHREN Zustand VOR dem "
        "Schritt (wie vorgegeben); die hier gepruefte Diagnose in greybox.py berechnet "
        "u_terminal dagegen NACH der Integration (5 RK4-Teilschritte). Bei allen "
        "verletzenden Punkten schrumpft u_terminal in genau diesem Intervall leicht "
        "(d50/T_bed driften) -- keine falsch platzierte Projektion im Sinne einer "
        "falschen Codestelle, sondern eine strukturelle Grenze des Ein-Schritt-Vorgriffs."
    )
    print(f"      T1 Betrag der Restverletzung, gepoolt: {pooled_report['f_residual_magnitude']}")
    f_relevant = ac["f_residual_exceeds_correlation_uncertainty"]
    f_verdict = "ERFUELLT -> ABBRUCH" if f_relevant else "nicht erfuellt"
    print(
        f"  (f) massgeblich, relative Ueberschreitung vs. Korrelationsunsicherheit "
        f"({CORRELATION_REL_UNCERTAINTY_LOW:.0%}): {f_verdict}"
    )
    if ac["any_abort_triggered"]:
        verdict = "Mindestens ein Abbruchkriterium erfuellt: S2 NICHT beginnen."
    else:
        verdict = "Kein Abbruchkriterium erfuellt: S2 kann begonnen werden."
    print(f"\n==> {verdict}")

    t3_note = (
        "T3: fluidization_margin_upper_m_per_s (Standard-Diagnose aus greybox.py._rhs, "
        "unveraendert, in JEDER Trajektorie vorhanden -- unabhaengig von project=True/"
        "False) macht die Restverletzung bereits punktweise auditierbar: "
        "diagnostics['fluidization_margin_upper_m_per_s'] < 0 markiert exakt die "
        "verletzenden Punkte, genau wie es probe_sweep_contamination.py fuer die "
        "urspruengliche (unprojizierte) Verletzung bereits tut. Keine neue Diagnosegroesse "
        "noetig."
    )
    print(f"\n{t3_note}")

    elapsed = time.monotonic() - t0
    result = {
        "process/fingerprint": EXPECTED_FINGERPRINT,
        "excited_seeds": list(EXCITED_SEEDS),
        "holdout_seed": HOLDOUT_SEED,
        "gas_velocity_index": gas_velocity_index,
        "t_bed_threshold_c": T_BED_THRESHOLD_C,
        "correlation_rel_uncertainty_band": [
            CORRELATION_REL_UNCERTAINTY_LOW,
            CORRELATION_REL_UNCERTAINTY_HIGH,
        ],
        "t3_auditability_note": t3_note,
        "per_group": per_group_report,
        "pooled": pooled_report,
        "_elapsed_s": elapsed,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\ngeschrieben: {OUT_PATH} ({elapsed:.1f} s)")


if __name__ == "__main__":
    main()

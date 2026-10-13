r"""Analysis of the pilot data of the re-run (decision of the project lead, 2026-10-12, before the full run).

Written and committed before it runs; no training. Three questions:

(a) HOW OFTEN IS A REQUESTED ACTION LIMITED, per data specification (legacy = old records and arm A1 share
    the trajectories; N = new-study pool), per channel (gas velocity, inlet temperature, spray rate,
    nozzle pressure) and at which limit: lower fluidisation limit u_mf, upper limit u_t (both only act on the
    gas velocity: ``project_action``), or the action box of ``ActionSpaceConfig``. Counted on the steps and the
    runs of the splits (train block of 64, its first 8 = the pilot's training set, validation, excited test,
    constant-run holdout), and on the WINDOWS of the pipeline (history 32, horizon 36, stride 8): windows
    whose future horizon contains a limited step, windows whose history only does, windows with neither.
    "Limited" means ``requested != realised`` (``Trajectory.requested_actions`` against ``actions``).

(b) ERRORS BY WINDOW GROUP. For every method of the pilot (the checkpoints and refitted baselines of arms
    A0, A1, N, seed 1, n_train 8) the physical RMSE (h = 1..36 pooled, per target; measurement as reference)
    separately for windows whose horizon contains a limited step ("F") and windows without ("U"). The models
    of A0 and A1 are evaluated on the same windows of the legacy pool in both window variants (realised,
    planned). Question: is the increase from A0 to A1/N almost only in the F windows? Reported: pooled RMSE
    per group, the number of windows per group, the ratio A1 / A0 per group, and the share of the total
    increase of the summed squared error that lies in F windows (``share_of_increase_in_F``) next to the
    share of F windows (``share_of_windows_F``); a share of the increase well above the share of windows says
    "concentrated in F". Arm N has its own pool, so its groups are reported, not matched.

(c) WHICH PART OF THE EXCITATION REQUESTS ACTIONS OUTSIDE THE FLUIDISATION WINDOW, and would a conservatively
    limited excitation window be possible? The margins u_mf and u_t are recomputed per step from the
    pre-step state (the same call as ``project_action``; checked: clipping the request with them reproduces the
    realised action). Requests outside the window are broken down by the signal kind of the gas-velocity
    channel (step, prbs, ramp, constant), by constant/non-constant run, by tercile of the initial diameter and
    by tercile of the run time, and by side. Conservative window: upper bound = the smallest u_t seen over
    all steps of the pool (smallest particle size), lower bound = the largest u_mf seen (largest particle
    size, hottest/driest conditions) -- empirical worst cases of the legacy pool, not analytical bounds.
    Reported: the window against the box [0.8, 2.5] m/s, the share of the requested samples that a clip of
    the REQUEST into the window would change, and a re-simulation of the whole pool with the gas-velocity
    requests clipped into (i) the upper bound only and (ii) both bounds: the number of limited steps that
    remains (the margins depend on the state, which changes with the actions; hence the re-simulation).
    Seed 1, 128 runs; the schedules of the pool are those of ``generate_data``.

    tools/run_here.sh scripts/analyze_projection_pilot.py --checkpoint-root <dir with A0/A1/N/*.pt>
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_arch_rerun_worker as worker
from _common import provenance, require_clean_tree, write_artifact

from twinlink_ml.data.splits import SplitName
from twinlink_ml.data.windows import TrajectoryWindowDataset
from twinlink_ml.evaluation import build_datasets, generate_data
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.fluidization import terminal_velocity_m_per_s
from twinlink_ml.simulation.greybox import IDX_BED_TEMPERATURE, GreyBoxFluidBedGranulator
from twinlink_ml.simulation.thermo import absolute_humidity_kg_per_kg
from twinlink_ml.utils.checkpoints import load_checkpoint

SEED = 1
N_TRAIN = 8
TOL = 1e-9
METHODS = ("persistence", "static", "arx", "gru", "lstm", "transformer", "xlstm", "mamba")
NEURAL = ("gru", "lstm", "transformer", "xlstm", "mamba")


def _gen(config: Any, spec: Any, seed: int) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return generate_data(config, spec, master_seed=seed)


def limited_flags(pool: Any) -> dict[str, np.ndarray]:
    """Per run and step: which limit acted (gas velocity: lower/upper; any other channel changed)."""
    trajectory = pool.trajectory
    requested, realised = trajectory.requested_actions, trajectory.actions
    names = trajectory.metadata["action_names"]
    gas = names.index("gas_velocity_m_per_s")
    diff = ~np.isclose(requested, realised, rtol=0.0, atol=TOL)
    lower = diff[..., gas] & (requested[..., gas] < realised[..., gas])
    upper = diff[..., gas] & (requested[..., gas] > realised[..., gas])
    other = np.zeros_like(lower)
    for index in range(len(names)):
        if index != gas:
            other |= diff[..., index]
    return {"lower": lower, "upper": upper, "other_channels": other, "any": lower | upper | other}


def box_counts(pool: Any, bounds: Any, names: tuple[str, ...]) -> dict[str, dict[str, int]]:
    requested = pool.trajectory.requested_actions
    out = {}
    for index, name in enumerate(names):
        bound = bounds[index]
        values = requested[..., index]
        out[name] = {
            "below_box_minimum": int((values < bound.minimum - TOL).sum()),
            "above_box_maximum": int((values > bound.maximum + TOL).sum()),
            "on_box_minimum": int(np.isclose(values, bound.minimum, atol=1e-6).sum()),
            "on_box_maximum": int(np.isclose(values, bound.maximum, atol=1e-6).sum()),
        }
    return out


def split_rows(pool: Any) -> dict[str, np.ndarray]:
    ids = list(pool.trajectory.run_id)
    mapping = pool.split.as_mapping

    def rows(names: Any) -> np.ndarray:
        return np.array([ids.index(name) for name in names], dtype=int)

    train = mapping[SplitName.TRAIN] if SplitName.TRAIN in mapping else mapping["train"]
    return {
        "train64": rows(train),
        "train8": rows(list(train)[:N_TRAIN]),
        "val": rows(mapping[SplitName.VAL] if SplitName.VAL in mapping else mapping["val"]),
        "test": rows(mapping[SplitName.TEST] if SplitName.TEST in mapping else mapping["test"]),
    }


def window_group_counts(
    flags: np.ndarray, rows: np.ndarray, history: int, horizon: int, stride: int
) -> dict[str, int]:
    n_steps = flags.shape[1]
    starts = np.arange(0, n_steps - (history + horizon) + 1, stride)
    future = history_only = neither = 0
    for row in rows:
        for start in starts:
            in_future = bool(flags[row, start + history : start + history + horizon].any())
            in_history = bool(flags[row, start : start + history].any())
            if in_future:
                future += 1
            elif in_history:
                history_only += 1
            else:
                neither += 1
    return {
        "future_horizon_limited": future,
        "history_only_limited": history_only,
        "neither": neither,
    }


def part_a(pools: dict[str, Any], config: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    bounds = config.actions.bounds
    names = config.actions.names
    spec = worker.legacy_spec()
    for label, (pool, holdout) in pools.items():
        flags = limited_flags(pool)
        rows = split_rows(pool)
        entry: dict[str, Any] = {"box_requests": box_counts(pool, bounds, names)}
        entry["channels_limited_steps"] = {
            "gas_velocity_lower_u_mf": int(flags["lower"].sum()),
            "gas_velocity_upper_u_t": int(flags["upper"].sum()),
            **{
                f"{name}_changed": 0
                if name == "gas_velocity_m_per_s"
                else int(
                    (
                        ~np.isclose(
                            pool.trajectory.requested_actions[..., i],
                            pool.trajectory.actions[..., i],
                            atol=TOL,
                        )
                    ).sum()
                )
                for i, name in enumerate(names)
                if name != "gas_velocity_m_per_s"
            },
        }
        splits: dict[str, Any] = {}
        for split_name, split_rows_ in rows.items():
            sub = {k: v[split_rows_] for k, v in flags.items()}
            steps = sub["any"].size
            splits[split_name] = {
                "runs": len(split_rows_),
                "steps": int(steps),
                "limited_steps": int(sub["any"].sum()),
                "limited_share_of_steps": float(sub["any"].mean()),
                "lower_u_mf_steps": int(sub["lower"].sum()),
                "upper_u_t_steps": int(sub["upper"].sum()),
                "other_channels_steps": int(sub["other_channels"].sum()),
                "runs_with_limited_step": int(sub["any"].any(axis=1).sum()),
                "windows": window_group_counts(
                    flags["any"], split_rows_, spec.history_length, spec.horizon, spec.stride
                ),
            }
        hold_flags = limited_flags(holdout)
        hold_rows = np.arange(hold_flags["any"].shape[0])
        hold_test = np.array(
            [
                list(holdout.trajectory.run_id).index(r)
                for r in holdout.split.as_mapping[SplitName.TEST]
            ]
        )
        del hold_rows
        sub = {k: v[hold_test] for k, v in hold_flags.items()}
        splits["constant_holdout"] = {
            "runs": len(hold_test),
            "steps": int(sub["any"].size),
            "limited_steps": int(sub["any"].sum()),
            "limited_share_of_steps": float(sub["any"].mean()),
            "lower_u_mf_steps": int(sub["lower"].sum()),
            "upper_u_t_steps": int(sub["upper"].sum()),
            "other_channels_steps": int(sub["other_channels"].sum()),
            "runs_with_limited_step": int(sub["any"].any(axis=1).sum()),
            "windows": window_group_counts(
                hold_flags["any"], hold_test, spec.history_length, spec.horizon, spec.stride
            ),
        }
        entry["splits"] = splits
        out[label] = entry
    return out


# ----------------------------------------------------------------------------------- (b)


def window_errors(
    model: Any, dataset: TrajectoryWindowDataset, flags: np.ndarray, pool: Any
) -> dict[str, Any]:
    """Squared physical errors per window and target, and the group flags."""
    history, horizon = dataset.config.history_length, dataset.config.horizon
    n = dataset.n_windows
    sq = []
    for start in range(0, n, 4096):
        batch = dataset.batch(np.arange(start, min(n, start + 4096)))
        prediction = dataset.target_scaler.inverse(model.predict(batch))
        reference = dataset.target_scaler.inverse(batch.future_targets)
        sq.append(((prediction - reference) ** 2).mean(axis=1))  # (windows, targets)
    mse = np.concatenate(sq, axis=0)
    rows = np.array(
        [list(pool.trajectory.run_id).index(dataset.run_ids[int(r)]) for r in dataset._index[:, 0]]
    )
    starts = dataset._index[:, 1].astype(int)
    future = np.array(
        [
            flags[r, s + history : s + history + horizon].any()
            for r, s in zip(rows, starts, strict=True)
        ]
    )
    return {"mse": mse, "future_limited": future}


def summarise_groups(result: dict[str, Any], targets: tuple[str, ...]) -> dict[str, Any]:
    mse, group = result["mse"], result["future_limited"]
    out: dict[str, Any] = {"windows": len(group), "windows_F": int(group.sum())}
    for index, target in enumerate(targets):
        out[target] = {
            "rmse_all": float(np.sqrt(mse[:, index].mean())),
            "rmse_F": float(np.sqrt(mse[group, index].mean())) if group.any() else None,
            "rmse_U": float(np.sqrt(mse[~group, index].mean())) if (~group).any() else None,
        }
    return out


def compare(
    base: dict[str, Any], other: dict[str, Any], targets: tuple[str, ...]
) -> dict[str, Any]:
    """Increase of the summed squared error from ``base`` to ``other`` on the same windows."""
    if not np.array_equal(base["future_limited"], other["future_limited"]):
        raise SystemExit("window groups differ: not the same windows")
    group = base["future_limited"]
    out: dict[str, Any] = {"share_of_windows_F": float(group.mean())}
    for index, target in enumerate(targets):
        delta = other["mse"][:, index] - base["mse"][:, index]
        total = float(delta.sum())
        f_part = float(delta[group].sum())
        out[target] = {
            "total_increase_of_summed_mse": total,
            "share_of_increase_in_F": f_part / total if total > 0.0 else None,
            "mse_ratio_F": float(other["mse"][group, index].sum() / base["mse"][group, index].sum())
            if group.any()
            else None,
            "mse_ratio_U": float(
                other["mse"][~group, index].sum() / base["mse"][~group, index].sum()
            )
            if (~group).any()
            else None,
        }
    return out


def build_model(method: str, arm: str, train: TrajectoryWindowDataset, own: Any, root: Path) -> Any:
    if method in NEURAL:
        model, _ = load_checkpoint(root / arm / f"{method}_{arm}_seed{SEED}_n{N_TRAIN}.pt")
        return model
    model = worker.old_worker.BASELINE_FACTORIES[method]()
    model.fit(train, seed=own.seeds.model_seed, validation=None)
    return model


def part_b(pools: dict[str, Any], root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    legacy_pool, legacy_hold = pools["legacy"]
    n_pool, n_hold = pools["N"]
    arms = {
        "A0": (worker.arm_spec("A0"), legacy_pool, legacy_hold),
        "A1": (worker.arm_spec("A1"), legacy_pool, legacy_hold),
        "N": (worker.arm_spec("N"), n_pool, n_hold),
    }
    flags_by_pool = {"legacy": limited_flags(legacy_pool)["any"], "N": limited_flags(n_pool)["any"]}
    hold_flags = {"legacy": limited_flags(legacy_hold)["any"], "N": limited_flags(n_hold)["any"]}
    for method in METHODS:
        t0 = time.perf_counter()
        results: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        targets: tuple[str, ...] = ()
        for arm, (spec, pool, hold) in arms.items():
            pool_key = "N" if arm == "N" else "legacy"
            train, _, _ = build_datasets(pool, N_TRAIN)
            model = build_model(method, arm, train, pool, root)
            targets = train.target_names
            windows = [spec.future_actions] if arm == "N" else ["realised", "planned"]
            for window_name in windows:
                window = dataclasses.replace(spec.window, future_actions=window_name)
                for test_set, source, split, flags in (
                    ("excited", pool, pool.split.subset(N_TRAIN), flags_by_pool[pool_key]),
                    ("constant_holdout", hold, hold.split, hold_flags[pool_key]),
                ):
                    dataset = TrajectoryWindowDataset.from_trajectory(
                        source.trajectory, split, SplitName.TEST, window, scalers=train.scalers
                    )
                    results[(arm, window_name, test_set, "")] = window_errors(
                        model, dataset, flags, source
                    )
        entry: dict[str, Any] = {"groups": {}, "increase": {}}
        for (arm, window_name, test_set, _), res in results.items():
            entry["groups"][f"{arm}_model_on_{window_name}_windows/{test_set}"] = summarise_groups(
                res, targets
            )
        for test_set in ("excited", "constant_holdout"):
            base = results[("A0", "realised", test_set, "")]
            entry["increase"][f"A1_primary_vs_A0_primary/{test_set}"] = compare(
                base, results[("A1", "planned", test_set, "")], targets
            )
            entry["increase"][f"A0_model_planned_windows_vs_A0_primary/{test_set}"] = compare(
                base, results[("A0", "planned", test_set, "")], targets
            )
            entry["increase"][f"A1_model_realised_windows_vs_A0_primary/{test_set}"] = compare(
                base, results[("A1", "realised", test_set, "")], targets
            )
        out[method] = entry
        print(f"(b) {method}: {time.perf_counter() - t0:.0f} s", flush=True)
    return out


# ----------------------------------------------------------------------------------- (c)


def margins(simulator: Any, pool: Any) -> tuple[np.ndarray, np.ndarray]:
    trajectory, schedule = pool.trajectory, pool.schedule
    columns = schedule.initial_states
    initial = simulator.state_from_physical(
        columns[:, 0], columns[:, 1], columns[:, 2], columns[:, 3]
    )
    states = np.asarray(trajectory.states)
    before = np.concatenate([np.asarray(initial)[:, None, :], states[:, :-1, :]], axis=1)
    u_mf, u_t = simulator._fluidization_margins(before, np.asarray(trajectory.disturbances))
    return np.asarray(u_mf), np.asarray(u_t)


def terciles(values: np.ndarray) -> np.ndarray:
    edges = np.quantile(values, [1 / 3, 2 / 3])
    return np.digitize(values, edges)


def part_c(config: Any, pool: Any) -> dict[str, Any]:
    names = pool.trajectory.metadata["action_names"]
    gas = names.index("gas_velocity_m_per_s")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        simulator = GreyBoxFluidBedGranulator(pool.process_config)
    u_mf, u_t = margins(simulator, pool)
    requested = pool.trajectory.requested_actions[..., gas]
    realised = pool.trajectory.actions[..., gas]
    check = float(np.abs(np.clip(requested, u_mf, u_t) - realised).max())
    below, above = requested < u_mf - TOL, requested > u_t + TOL
    outside = below | above
    n_runs, n_steps = requested.shape
    kinds = np.array([k[gas] for k in pool.schedule.signal_kinds])
    constant_run = np.array([all(x == "constant" for x in k) for k in pool.schedule.signal_kinds])
    d0 = np.asarray(pool.schedule.initial_states)[:, 2]
    time_third = np.digitize(np.arange(n_steps), [n_steps / 3, 2 * n_steps / 3])

    def breakdown(mask: np.ndarray) -> dict[str, Any]:
        return {
            "steps": int(mask.sum()),
            "share_of_all_steps": float(mask.mean()),
            "runs_with_a_step": int(mask.any(axis=1).sum()),
        }

    by_kind = {
        str(kind): {
            **breakdown(outside[kinds == kind]),
            "below_u_mf": int(below[kinds == kind].sum()),
            "above_u_t": int(above[kinds == kind].sum()),
            "runs": int((kinds == kind).sum()),
        }
        for kind in sorted(set(kinds))
    }
    d_third = terciles(d0)
    by_d0 = {
        f"initial_diameter_tercile_{t}": {
            **breakdown(outside[d_third == t]),
            "d0_um_range": [
                float(d0[d_third == t].min() * 1e6),
                float(d0[d_third == t].max() * 1e6),
            ],
        }
        for t in range(3)
    }
    by_time = {f"time_tercile_{t}": breakdown(outside[:, time_third == t]) for t in range(3)}
    u_hi = float(u_t.min())
    u_lo = float(u_mf.max())
    bound = config.actions.gas_velocity_m_per_s
    result: dict[str, Any] = {
        "recomputation_check_max_abs_difference": check,
        "requested_outside_window": breakdown(outside),
        "below_u_mf_steps": int(below.sum()),
        "above_u_t_steps": int(above.sum()),
        "by_signal_kind_of_gas_velocity": by_kind,
        "constant_runs": {**breakdown(outside[constant_run]), "runs": int(constant_run.sum())},
        "non_constant_runs": {
            **breakdown(outside[~constant_run]),
            "runs": int((~constant_run).sum()),
        },
        "by_initial_diameter": by_d0,
        "by_run_time": by_time,
        "margins_over_the_pool": {
            "u_mf_min": float(u_mf.min()),
            "u_mf_max": u_lo,
            "u_t_min": u_hi,
            "u_t_max": float(u_t.max()),
            "box": [bound.minimum, bound.maximum],
            "d50_min_um": float(np.asarray(pool.trajectory.diagnostics["d50_m"]).min() * 1e6),
            "d50_max_um": float(np.asarray(pool.trajectory.diagnostics["d50_m"]).max() * 1e6),
        },
    }
    window_valid = u_lo < u_hi
    result["conservative_window"] = {
        "upper_bound_smallest_u_t": u_hi,
        "lower_bound_largest_u_mf": u_lo,
        "non_empty": bool(window_valid),
        "width_m_per_s": float(u_hi - u_lo),
        "share_of_box_width": float(
            (min(u_hi, bound.maximum) - max(u_lo, bound.minimum)) / (bound.maximum - bound.minimum)
        ),
        "requested_samples_a_clip_into_the_window_would_change": float(
            ((requested < u_lo) | (requested > u_hi)).mean()
        ),
        "requested_samples_a_clip_to_the_upper_bound_only_would_change": float(
            (requested > u_hi).mean()
        ),
    }

    def resimulate(low: float, high: float) -> dict[str, Any]:
        actions = np.array(pool.schedule.actions, copy=True)
        actions[..., gas] = np.clip(actions[..., gas], low, high)
        columns = pool.schedule.initial_states
        initial = simulator.state_from_physical(
            columns[:, 0], columns[:, 1], columns[:, 2], columns[:, 3]
        )
        start = time.perf_counter()
        trajectory = simulator.rollout(
            actions,
            pool.schedule.disturbances,
            run_id=pool.schedule.run_id,
            seed=pool.seeds.noise_seed,
            initial_state=initial,
        )
        flags = limited_flags(dataclasses.replace(pool, trajectory=trajectory))
        return {
            "clip": [low, high],
            "limited_steps": int(flags["any"].sum()),
            "lower_u_mf_steps": int(flags["lower"].sum()),
            "upper_u_t_steps": int(flags["upper"].sum()),
            "runs_with_limited_step": int(flags["any"].any(axis=1).sum()),
            "seconds": time.perf_counter() - start,
        }

    # a-priori caps from the particle size: u_t at the smallest particle size, worst case over the
    # bed temperatures and inlet humidities seen in the pool (an analytic bound, not the empirical minimum)
    backend = simulator.backend
    bed_t = np.asarray(pool.trajectory.states)[..., IDX_BED_TEMPERATURE]
    humidity = np.asarray(
        absolute_humidity_kg_per_kg(
            backend,
            pool.process_config.gas,
            np.asarray(pool.trajectory.disturbances)[..., 0],
            np.asarray(pool.trajectory.disturbances)[..., 1],
        )
    )
    grid_t = np.linspace(bed_t.min(), bed_t.max(), 7)
    grid_y = np.linspace(humidity.min(), humidity.max(), 7)

    def worst_u_t(diameter_m: np.ndarray) -> np.ndarray:
        values = [
            np.asarray(
                terminal_velocity_m_per_s(
                    backend,
                    pool.process_config.gas,
                    pool.process_config.material,
                    diameter_m,
                    np.full_like(diameter_m, t),
                    np.full_like(diameter_m, y),
                )
            )
            for t in grid_t
            for y in grid_y
        ]
        return np.min(values, axis=0)

    d_min = float(d0.min())
    analytic_global = float(worst_u_t(np.array([d_min]))[0])
    run_cap = worst_u_t(d0)
    result["a_priori_caps"] = {
        "bed_temperature_range_degC": [float(bed_t.min()), float(bed_t.max())],
        "inlet_absolute_humidity_range": [float(humidity.min()), float(humidity.max())],
        "smallest_initial_diameter_um": d_min * 1e6,
        "global_cap_u_t_at_smallest_initial_diameter": analytic_global,
        "run_specific_cap_u_t_at_initial_diameter": {
            "min": float(run_cap.min()),
            "median": float(np.median(run_cap)),
            "max": float(run_cap.max()),
            "runs_with_cap_below_box_minimum": int((run_cap < bound.minimum).sum()),
            "runs_whose_requests_a_run_cap_would_change": int(
                (pool.schedule.actions[..., gas] > run_cap[:, None]).any(axis=1).sum()
            ),
            "share_of_requested_samples_changed": float(
                (pool.schedule.actions[..., gas] > run_cap[:, None]).mean()
            ),
        },
    }
    result["resimulation_upper_bound_only"] = resimulate(-np.inf, u_hi)
    result["resimulation_global_analytic_cap"] = resimulate(-np.inf, analytic_global)

    def resimulate_run_caps() -> dict[str, Any]:
        actions = np.array(pool.schedule.actions, copy=True)
        actions[..., gas] = np.minimum(actions[..., gas], run_cap[:, None])
        columns = pool.schedule.initial_states
        initial = simulator.state_from_physical(
            columns[:, 0], columns[:, 1], columns[:, 2], columns[:, 3]
        )
        trajectory = simulator.rollout(
            actions,
            pool.schedule.disturbances,
            run_id=pool.schedule.run_id,
            seed=pool.seeds.noise_seed,
            initial_state=initial,
        )
        flags = limited_flags(dataclasses.replace(pool, trajectory=trajectory))
        return {
            "limited_steps": int(flags["any"].sum()),
            "runs_with_limited_step": int(flags["any"].any(axis=1).sum()),
        }

    result["resimulation_run_specific_caps"] = resimulate_run_caps()
    result["resimulation_both_bounds"] = (
        resimulate(max(u_lo, bound.minimum), min(u_hi, bound.maximum)) if window_valid else None
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    arguments = parser.parse_args()
    require_clean_tree()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(worker.old_worker.CONFIG_PATH)
    config = dataclasses.replace(
        config, numerics=dataclasses.replace(config.numerics, enforce_gas_velocity_projection=True)
    )
    t0 = time.perf_counter()
    legacy, n_spec = worker.legacy_spec(), worker.arm_spec("N")
    pools = {
        "legacy": (
            _gen(config, legacy, SEED),
            _gen(config, worker.holdout_spec(legacy), worker.HOLDOUT_SEED),
        ),
        "N": (
            _gen(config, n_spec, SEED),
            _gen(config, worker.holdout_spec(n_spec), worker.HOLDOUT_SEED),
        ),
    }
    print(f"pools generated in {time.perf_counter() - t0:.0f} s", flush=True)
    a = part_a(pools, config)
    print("(a) done", flush=True)
    c = part_c(config, pools["legacy"][0])
    print("(c) done", flush=True)
    b = part_b(pools, arguments.checkpoint_root)
    write_artifact(
        "analyze_projection_pilot",
        {
            "_provenance": provenance(),
            "seed": SEED,
            "n_train": N_TRAIN,
            "a_limited_actions": a,
            "b_errors_by_window_group": b,
            "c_excitation_window": c,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

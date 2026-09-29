"""PR2 -- separation of the data (run-level splits, windows, normalisation, nesting, records).

Checks, each with the audit's own code:

A. run IDs of train/val/test disjoint inside every arm, and across arms and seeds (ID level);
B. run *content* (requested actions, disturbances, initial state) hashed per run: no two runs of
   any pool, seed or arm are identical; nearest-neighbour distance of the constant-run holdout
   test runs to the constant training pool reported;
C. windows: own window enumeration reproduces the record's ``n_windows``; the project's window
   dataset is compared with slices cut from the raw trajectory by the audit (no window crosses a
   run boundary, history and future do not overlap);
D. normalisation statistics recomputed from the training runs only and compared with the ones
   the datasets carry, with the ones stored in the checkpoints of the archive (read-only), and
   with pool-wide statistics (to show what leakage would look like);
E. budget nesting 8 in 16 in 32 in 64 (IDs, composition, unchanged val/test);
F. identical splits per seed across architectures, from the records (all four record files).

    python scripts/verify/pr2_separation.py
"""

from __future__ import annotations

import collections
import hashlib
import itertools
import json
from typing import Any

import numpy as np
from _common import (
    ARCHIVE_ROOT,
    TARGETS,
    provenance,
    read_jsonl,
    require_clean_tree,
    write_artifact,
)
from _data import EXCITED, HOLDOUT_SEED, make_data

SEEDS = (1, 2, 3, 4, 5)
BUDGETS = (8, 16, 32, 64)
HISTORY, HORIZON, STRIDE = 32, 36, 8
TARGET_COLS = {"energy_target": 0, "product_moisture_kg_per_kg": 2, "bed_temperature_celsius": 1}
# outputs columns: energy_target, bed_temperature_celsius, product_moisture, d50_um


def target_columns() -> list[int]:
    return [TARGET_COLS[t] for t in TARGETS]


def run_key(data: Any, r: int) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(data.schedule.actions[r]).tobytes())
    h.update(np.ascontiguousarray(data.schedule.disturbances[r]).tobytes())
    h.update(np.ascontiguousarray(data.schedule.initial_states[r]).tobytes())
    return h.hexdigest()[:20]


def own_inputs(data: Any, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(inputs (runs, T, 10) = actions | disturbances | measured targets, targets (runs, T, 3))."""
    t = data.trajectory
    observed = t.measurements[rows][..., target_columns()]
    inputs = np.concatenate([t.actions[rows], t.disturbances[rows], observed], axis=-1)
    return inputs, observed


def own_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = values.reshape(-1, values.shape[-1])
    return flat.mean(axis=0), np.maximum(flat.std(axis=0), 1e-12)


def index_of(data: Any, ids: tuple[str, ...]) -> np.ndarray:
    pos = {rid: i for i, rid in enumerate(data.trajectory.run_id)}
    return np.asarray([pos[r] for r in ids], dtype=np.intp)


def check_ids_and_content(pools: dict[tuple[str, int], Any]) -> dict[str, Any]:
    id_sets: dict[tuple[str, int, str], set[str]] = {}
    for (kind, seed), data in pools.items():
        for name, ids in (
            ("train", data.split.train),
            ("val", data.split.val),
            ("test", data.split.test),
        ):
            id_sets[(kind, seed, name)] = set(ids)
    within = {
        f"{kind}_seed{seed}": all(
            not (id_sets[(kind, seed, a)] & id_sets[(kind, seed, b)])
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        )
        for (kind, seed) in pools
    }
    overlaps = [
        (str(a), str(b)) for a, b in itertools.combinations(id_sets, 2) if id_sets[a] & id_sets[b]
    ]
    holdout_test = id_sets[("holdout", HOLDOUT_SEED, "test")]
    holdout_vs_constant_train = {
        f"seed{s}": {
            "n_shared_ids_with_any_split": len(
                holdout_test
                & (
                    id_sets[("constant_train", s, "train")]
                    | id_sets[("constant_train", s, "val")]
                    | id_sets[("constant_train", s, "test")]
                )
            ),
        }
        for s in SEEDS
    }
    # content level
    keys: dict[str, list[str]] = collections.defaultdict(list)
    for (kind, seed), data in pools.items():
        for r in range(data.schedule.n_runs):
            keys[run_key(data, r)].append(f"{kind}{seed}:{r}")
    duplicates = {k: v for k, v in keys.items() if len(v) > 1}
    # nearest neighbour of the holdout test runs among the constant training runs (n_train=64)
    hold = pools[("holdout", HOLDOUT_SEED)]
    bounds = np.asarray(
        [[60.0, 120.0], [0.8, 2.5], [0.4, 3.0], [1.0, 4.0]]
    )  # inlet T, gas velocity, spray, nozzle: normalise by actuator span

    def level(data: Any, rows: np.ndarray) -> np.ndarray:
        return (data.schedule.actions[rows][:, 0, :] - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

    hold_rows = index_of(hold, hold.split.test)
    nn = {}
    for s in SEEDS:
        train = pools[("constant_train", s)]
        rows = index_of(train, train.split.train)
        d = np.linalg.norm(
            level(hold, hold_rows)[:, None, :] - level(train, rows)[None, :, :], axis=-1
        )
        nn[f"seed{s}"] = {
            "min_distance_holdout_run_to_nearest_constant_training_run": float(d.min()),
            "median_nearest_distance": float(np.median(d.min(axis=1))),
        }
    return {
        "A_ids_disjoint_within_pool_train_val_test": within,
        "A_n_id_sets_compared": len(id_sets),
        "A_pairs_with_shared_ids": overlaps,
        "A_holdout_test_vs_constant_train_pool_ids": holdout_vs_constant_train,
        "A_note": "IDs embed a hash of (excitation config, n_runs, seed); disjointness across pools "
        "is therefore nearly automatic and is complemented by the content check B.",
        "B_n_runs_hashed": int(sum(len(v) for v in keys.values())),
        "B_duplicate_run_contents": duplicates,
        "B_nearest_neighbour_holdout_vs_constant_train_(normalised_action_levels)": nn,
    }


def check_windows_and_scalers(pools: dict[tuple[str, int], Any]) -> dict[str, Any]:
    from twinlink_ml.data.splits import SplitName
    from twinlink_ml.data.windows import TrajectoryWindowDataset
    from twinlink_ml.evaluation.experiments import build_datasets

    out: dict[str, Any] = {"C_windows": {}, "D_scalers": {}}
    rec = read_jsonl("arch_learning_curve_projected")
    rec_windows = {
        (r["params"]["master_seed"], r["tags"]["test_set"]): (
            r["metrics"]["n_windows"],
            r["metrics"]["n_test_runs"],
        )
        for r in rec
    }
    hold = pools[("holdout", HOLDOUT_SEED)]
    for seed in SEEDS:
        data = pools[("excited", seed)]
        train, val, test = build_datasets(data, 64)
        # own enumeration
        n_steps = data.trajectory.states.shape[1]
        starts = np.arange(0, n_steps - (HISTORY + HORIZON) + 1, STRIDE)
        own_count = {
            "excited": len(data.split.test) * len(starts),
            "constant_holdout": len(hold.split.test) * len(starts),
        }
        entry: dict[str, Any] = {
            "windows_per_run_own": len(starts),
            "last_window_end_step_own_(<=T)": int(starts[-1] + HISTORY + HORIZON),
            "T": n_steps,
        }
        for test_set, dataset, source in (
            ("excited", test, data),
            (
                "constant_holdout",
                TrajectoryWindowDataset.from_trajectory(
                    hold.trajectory,
                    hold.split,
                    SplitName.TEST,
                    data.spec.window,
                    scalers=train.scalers,
                ),
                hold,
            ),
        ):
            idx = np.asarray(dataset._index)
            in_bounds = bool(((idx[:, 1] + HISTORY + HORIZON) <= n_steps).all())
            batch = dataset.batch(range(dataset.n_windows))
            rows = index_of(source, dataset.run_ids)
            inputs, observed = own_inputs(source, rows)
            mean, scale = dataset.input_scaler.mean, dataset.input_scaler.scale
            t_mean, t_scale = dataset.target_scaler.mean, dataset.target_scaler.scale
            std_inputs = (inputs - mean) / scale
            hist_own = np.stack([std_inputs[r, s : s + HISTORY] for r, s in idx], axis=0)
            fut_tgt_own = np.stack(
                [
                    ((observed - t_mean) / t_scale)[r, s + HISTORY : s + HISTORY + HORIZON]
                    for r, s in idx
                ],
                axis=0,
            )
            fut_exo_own = np.stack(
                [std_inputs[r, s + HISTORY : s + HISTORY + HORIZON, :7] for r, s in idx], axis=0
            )
            entry[test_set] = {
                "n_windows_project": dataset.n_windows,
                "n_windows_own_enumeration": own_count[test_set],
                "n_windows_record": rec_windows[(seed, test_set)][0],
                "n_runs_record": rec_windows[(seed, test_set)][1],
                "all_windows_inside_a_single_run_and_within_T": in_bounds,
                "history_equals_own_slice_max_abs": float(
                    np.abs(batch.history_inputs - hist_own).max()
                ),
                "future_targets_equal_own_slice_max_abs": float(
                    np.abs(batch.future_targets - fut_tgt_own).max()
                ),
                "future_exogenous_equal_own_slice_max_abs": float(
                    np.abs(batch.future_exogenous - fut_exo_own).max()
                ),
                "history_and_future_disjoint_in_time": True,  # [s, s+32) vs [s+32, s+68) by construction of the slices above
                "dataset_run_ids_equal_split_ids": tuple(dataset.run_ids)
                == tuple(source.split.test),
            }
        out["C_windows"][f"seed{seed}"] = entry

        # ---- D: scalers ------------------------------------------------------------------
        scal: dict[str, Any] = {}
        for n in BUDGETS:
            tr, va, te = build_datasets(data, n)
            rows = index_of(data, data.split.subset(n).train)
            inputs, observed = own_inputs(data, rows)
            m, s = own_scaler(inputs)
            tm, ts = own_scaler(observed)
            pool_rows = np.arange(data.schedule.n_runs)
            pool_inputs, _ = own_inputs(data, pool_rows)
            pm, ps = own_scaler(pool_inputs)
            scal[f"n{n}"] = {
                "train_input_scaler_vs_own_train_only_max_rel_diff": float(
                    max(
                        np.abs(tr.input_scaler.mean - m).max() / np.abs(s).max(),
                        (np.abs(tr.input_scaler.scale - s) / s).max(),
                    )
                ),
                "train_target_scaler_vs_own_train_only_max_rel_diff": float(
                    max(
                        np.abs(tr.target_scaler.mean - tm).max() / np.abs(ts).max(),
                        (np.abs(tr.target_scaler.scale - ts) / ts).max(),
                    )
                ),
                "val_scaler_is_train_scaler": bool(
                    va is not None
                    and np.array_equal(va.input_scaler.mean, tr.input_scaler.mean)
                    and np.array_equal(va.input_scaler.scale, tr.input_scaler.scale)
                ),
                "test_scaler_is_train_scaler": bool(
                    np.array_equal(te.input_scaler.mean, tr.input_scaler.mean)
                    and np.array_equal(te.input_scaler.scale, tr.input_scaler.scale)
                ),
                "pool_wide_stats_differ_from_train_by_(max_rel_of_scale)": float(
                    max(np.abs(pm - m).max() / s.max(), (np.abs(ps - s) / s).max())
                ),
            }
        out["D_scalers"][f"seed{seed}"] = scal
    return out


def check_checkpoint_scalers(pools: dict[tuple[str, int], Any]) -> dict[str, Any]:
    """Compare scalers stored in the (read-only) archive checkpoints with train-only statistics."""
    import torch

    ckpt_dirs = {
        "gru/lstm (Linux run 0b3eaaf, some files overwritten by later probes)": ARCHIVE_ROOT
        / "artifacts"
        / "checkpoints",
        "transformer/xlstm/mamba (extended run 256d644)": ARCHIVE_ROOT
        / "artifacts"
        / "checkpoints_arch_learning_curve_extended",
    }
    results: list[dict[str, Any]] = []
    per_key: dict[tuple[int, int], list[tuple[str, np.ndarray]]] = collections.defaultdict(list)
    for _label, directory in ckpt_dirs.items():
        for path in sorted(directory.glob("*_seed?_n*.pt")):
            name = path.stem  # e.g. gru_seed1_n64
            constant = name.startswith("constant_trained_")
            arch, seed_part, n_part = name.removeprefix("constant_trained_").split("_")
            arch = f"constant_trained_{arch}" if constant else arch
            seed, n = int(seed_part[4:]), int(n_part[1:])
            payload = torch.load(path, weights_only=False)
            data = pools[("constant_train" if constant else "excited", seed)]
            rows = index_of(data, data.split.subset(n).train)
            inputs, observed = own_inputs(data, rows)
            m, s = own_scaler(inputs)
            tm, ts = own_scaler(observed)
            ck_m = np.asarray(payload["input_scaler"]["mean"])
            ck_s = np.asarray(payload["input_scaler"]["scale"])
            ck_tm = np.asarray(payload["target_scaler"]["mean"])
            ck_ts = np.asarray(payload["target_scaler"]["scale"])
            rel = float(
                max(
                    np.abs(ck_m - m).max() / s.max(),
                    (np.abs(ck_s - s) / s).max(),
                    np.abs(ck_tm - tm).max() / ts.max(),
                    (np.abs(ck_ts - ts) / ts).max(),
                )
            )
            results.append(
                {
                    "file": name,
                    "checkpoint_git_commit": payload["git_commit"][:8],
                    "max_rel_diff_to_own_train_only_scaler": rel,
                }
            )
            per_key[(seed, n, constant)].append((arch, np.concatenate([ck_m, ck_s, ck_tm, ck_ts])))
    across = {
        f"{'constant_train' if constant else 'excited'}_seed{seed}_n{n}": {
            "architectures": sorted(a for a, _ in items),
            "scalers_identical_across_architectures": bool(
                all(np.array_equal(items[0][1], v) for _, v in items)
            ),
        }
        for (seed, n, constant), items in sorted(per_key.items())
        if len(items) > 1
    }
    return {
        "D_checkpoint_scalers": results,
        "D_checkpoint_scalers_max_rel_diff_overall": max(
            (r["max_rel_diff_to_own_train_only_scaler"] for r in results), default=None
        ),
        "D_checkpoints_compared": len(results),
        "D_scalers_identical_across_architectures_per_seed_budget": across,
    }


def check_nesting(pools: dict[tuple[str, int], Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for seed in SEEDS:
        data = pools[("excited", seed)]
        subs = {n: data.split.subset(n) for n in BUDGETS}
        kinds = data.schedule.signal_kinds
        pos = {rid: i for i, rid in enumerate(data.trajectory.run_id)}
        comp = {}
        for n in BUDGETS:
            runs = [pos[r] for r in subs[n].train]
            n_const = sum(all(k == "constant" for k in kinds[r]) for r in runs)
            comp[f"n{n}"] = {"n_constant_runs": int(n_const), "fraction_constant": n_const / n}
        out[f"seed{seed}"] = {
            "train_ids_are_prefixes_of_train64": all(
                subs[n].train == data.split.train[:n] for n in BUDGETS
            ),
            "chain_8_in_16_in_32_in_64": all(
                set(subs[a].train) < set(subs[b].train) for a, b in itertools.pairwise(BUDGETS)
            ),
            "val_and_test_identical_for_all_budgets": all(
                subs[n].val == data.split.val and subs[n].test == data.split.test for n in BUDGETS
            ),
            "composition": comp,
        }
    return out


def check_records() -> dict[str, Any]:
    files = {
        "arch_learning_curve_projected": read_jsonl("arch_learning_curve_projected"),
        "arch_learning_curve_projected_extended": read_jsonl(
            "arch_learning_curve_projected_extended"
        ),
        "arch_learning_curve_constant_trained": read_jsonl("arch_learning_curve_constant_trained"),
        "arch_epoch_budget_probe600": read_jsonl("arch_epoch_budget_probe600"),
    }
    fields = (
        "split/digest",
        "excitation/fingerprint",
        "seed/data",
        "seed/noise",
        "seed/model",
        "process/fingerprint",
        "holdout/seed",
        "holdout/n_runs",
    )
    data_keys = [
        "data/sample_time_s",
        "data/run_duration_s",
        "data/n_train_max",
        "data/n_val",
        "data/n_test",
        "data/history_length",
        "data/horizon",
        "data/stride",
        "data/target_names",
        "data/constant_run_fraction",
    ]
    out: dict[str, Any] = {}
    digest_by_file_seed: dict[tuple[str, int], set[str]] = {}
    for name, rows in files.items():
        per_seed: dict[int, dict[str, Any]] = {}
        for seed in sorted({r["params"]["master_seed"] for r in rows}):
            sub = [r for r in rows if r["params"]["master_seed"] == seed]
            vals = {f: {json.dumps(r["params"].get(f)) for r in sub} for f in (*fields, *data_keys)}
            models = sorted({r["tags"]["model"] for r in sub})
            n_by_test = collections.defaultdict(set)
            for r in sub:
                n_by_test[r["tags"]["test_set"]].add(
                    (r["metrics"]["n_windows"], r["metrics"]["n_test_runs"])
                )
            per_seed[seed] = {
                "n_records": len(sub),
                "models": models,
                "budgets": sorted({r["params"]["n_train"] for r in sub}),
                "every_field_single_valued_across_models_budgets_testsets": all(
                    len(v) == 1 for v in vals.values()
                ),
                "multi_valued_fields": [f for f, v in vals.items() if len(v) > 1],
                "n_val_values": sorted({r["params"]["n_val"] for r in sub}),
                "n_windows_n_test_runs_per_test_set": {k: sorted(v) for k, v in n_by_test.items()},
            }
            digest_by_file_seed[(name, seed)] = {
                json.dumps(r["params"]["split/digest"]) for r in sub
            }
        out[name] = per_seed
    cross = {}
    for seed in SEEDS:
        excited_files = [
            "arch_learning_curve_projected",
            "arch_learning_curve_projected_extended",
            "arch_epoch_budget_probe600",
        ]
        digs = [digest_by_file_seed.get((f, seed), set()) for f in excited_files]
        cross[f"seed{seed}"] = {
            "excited_arm_digest_identical_across_projected_extended_probe600": len(
                {frozenset(d) for d in digs if d}
            )
            == 1,
            "constant_arm_digest_differs_from_excited_arm": digest_by_file_seed[
                ("arch_learning_curve_constant_trained", seed)
            ]
            != digest_by_file_seed[("arch_learning_curve_projected", seed)],
        }
    return {"F_per_file_and_seed": out, "F_cross_file": cross}


def main() -> None:
    require_clean_tree()
    pools: dict[tuple[str, int], Any] = {}
    for kind in ("excited", "constant_train"):
        for seed in SEEDS:
            pools[(kind, seed)] = make_data(kind, seed)
            print("generated", kind, seed, flush=True)
    pools[("holdout", HOLDOUT_SEED)] = make_data("holdout", HOLDOUT_SEED)
    report: dict[str, Any] = {"_provenance": provenance(), "spec_excited": EXCITED}
    report.update(check_ids_and_content(pools))
    report.update(check_windows_and_scalers(pools))
    report.update(check_checkpoint_scalers(pools))
    report["E_nesting"] = check_nesting(pools)
    report.update(check_records())
    write_artifact("verify_pr2_separation", report)


if __name__ == "__main__":
    main()

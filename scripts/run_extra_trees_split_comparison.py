r"""Split comparison: window-/point-level vs. run-level split, Extra Trees.

Responds to the proposal from ``docs/04_stand_ap1.md``, section "Was das Paper des
Vorgängers ändert": Kräuter, Zhou & Heinrich (ABS23) split their training/test data
randomly over individual observations, not over batches -- a window leak that almost
completely explains their reported R² = 0.994 (against NRMSE ≈ 0.97 in the soft-sensor operation
of this project, see there). This script measures BOTH splits on **the same**
dataset with **the same** model (:class:`~twinlink_ml.models.extra_trees.ExtraTreesModel`,
the methodological reproduction of Kräuter's feature space) -- the gap between the
two numbers is the result, not a single accuracy.

**No architecture comparison.** ``extra_trees`` is deliberately not in
``models/registry.py``/``MODEL_SLOTS`` (see the ``models/extra_trees.py`` module docstring) and
this script is no replacement for ``study_architectures.py``.

Same run pool in both arms
-----------------------------------
Both arms use exactly the same 96 runs (``data.split.train + data.split.test``, no
``val`` block -- Extra Trees has no early stopping that would need one). Arm A
(run-level) assigns whole runs, as is standard in the rest of the project
(:class:`~twinlink_ml.data.splits.IndexBlockSplitter`, already part of
``data.split``). Arm B (window-/point-level) windows the same run pool **together**
and distributes individual windows via :class:`~twinlink_ml.data.point_split.PointSplitter`,
with the same target shares (train : test = 64 : 32 runs corresponds to 2:1, see
``_POINT_RATIOS``). The split is thus the only variable between the two numbers.

Window geometry: ``history_length=3, horizon=1`` (value + 1st/2nd difference over
:math:`t, t{-}1, t{-}2`, one-step prediction) -- Kräuter's feature space, not the
32-step geometry of the GRU/LSTM/Transformer/Mamba comparison.

Costs (measured empirically, see session log before this run, ``master_seed=1``,
production size 128 runs × 1152 steps): physics rollout ≈ 8 s, window construction per arm
≤ 0.3 s, ``ExtraTreesModel`` fit (100 trees, ~73,000 windows) ≈ 3-4 s per arm. No
gradient training, no GPU. With ``N_SEEDS`` seeds thus well under one minute
total run time.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

from twinlink_ml.data import PointSplitter, SplitRatios, windowed_point_split
from twinlink_ml.evaluation import DataSpec, ValidationPolicy, build_datasets, generate_data
from twinlink_ml.evaluation.evaluator import RolloutEvaluator
from twinlink_ml.models import ExtraTreesModel
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.tracking import JsonlExperimentLogger, RunRecord, environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "runs" / "extra_trees_split_comparison.jsonl"

MASTER_SEEDS = (1, 2, 3, 4, 5)
N_ESTIMATORS = 100

# Run shares of arm A (IndexBlockSplitter standard: 64 train, 32 val, 32 test of 128)
# without the val block: 64 : 32 = 2 : 1 -- the same proportion for arm B, see module docstring.
_POINT_RATIOS = SplitRatios(train=64 / 96, val=0.0, test=32 / 96)


def _spec() -> DataSpec:
    """Kräuter's window geometry: value + differences over three steps, h=1."""
    return DataSpec(
        sample_time_s=25.0,
        run_duration_s=28_800.0,
        n_train_max=64,
        n_val=32,
        n_test=32,
        history_length=3,
        horizon=1,
        stride=1,
    )


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)

    spec = _spec()
    evaluator = RolloutEvaluator(horizons=(1,))
    logger = JsonlExperimentLogger(OUTPUT_PATH)

    _log(
        f"Start: Splitvergleich Extra Trees, {len(MASTER_SEEDS)} Seeds, n_estimators={N_ESTIMATORS}"
    )

    for master_seed in MASTER_SEEDS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            data = generate_data(config, spec, master_seed=master_seed)

        pool_run_ids = data.split.train + data.split.test
        common_params = {
            **spec.as_parameters(),
            **data.fingerprints,
            **data.seeds.as_parameters(),
            "master_seed": master_seed,
            "n_train_runs": len(data.split.train),
            "n_test_runs": len(data.split.test),
            **environment_fingerprint(),
        }

        arms: dict[str, tuple[object, object]] = {}

        train_a, _val_a, test_a = build_datasets(
            data, spec.n_train_max, policy=ValidationPolicy.NONE
        )
        # Explicit inventory check, not just trust in build_datasets() (Befund 36):
        # test_a MUST carry the same scaler object as train_a (identity, not just
        # equality) -- otherwise the run-level arm would no longer be free of scaler leaks and the
        # split comparison would mix two different leak sources.
        assert test_a.input_scaler is train_a.input_scaler
        assert test_a.target_scaler is train_a.target_scaler
        arms["run_level"] = (train_a, test_a)

        point_splitter = PointSplitter(seed=master_seed, ratios=_POINT_RATIOS)
        train_b, _val_b, test_b = windowed_point_split(
            data.trajectory, pool_run_ids, spec.window, point_splitter
        )
        # The same inventory check for the point-level arm, symmetric to the above.
        assert test_b.input_scaler is train_b.input_scaler
        assert test_b.target_scaler is train_b.target_scaler
        arms["point_level"] = (train_b, test_b)

        for split_kind, (train, test) in arms.items():
            model = ExtraTreesModel(n_estimators=N_ESTIMATORS)
            fit_start = time.perf_counter()
            model.fit(train, seed=data.seeds.model_seed)
            fit_elapsed = time.perf_counter() - fit_start

            result = evaluator.evaluate(model, test)
            record = RunRecord(
                group="extra_trees_split_comparison",
                params={
                    **common_params,
                    "split/kind": split_kind,
                    "model/label": "extra_trees",
                    **{f"model/{key}": value for key, value in model.describe().items()},
                    "n_train_windows": train.n_windows,
                    "n_test_windows": test.n_windows,
                    "fit_seconds": fit_elapsed,
                },
                metrics=result.as_flat_metrics(),
                tags={
                    "model": "extra_trees",
                    "split_kind": split_kind,
                    "group": "extra_trees_split_comparison",
                },
            )
            logger.log(record)

            energy = result.nrmse_run_spread["energy_target"][1][0]
            moisture = result.nrmse_run_spread["product_moisture_kg_per_kg"][1][0]
            _log(
                f"seed={master_seed} split={split_kind:<11} fit={fit_elapsed:5.2f}s  "
                f"nrmse_run_median  energy_target={energy:.4f}  "
                f"product_moisture_kg_per_kg={moisture:.4f}"
            )

    _log(f"Fertig. Ergebnisse in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

r"""Step B (session on Befund 37): reduced comparison, 1 seed, n_train=16.

Explicitly reduced compared with the full learning curve (3 seeds, n_train=32) -- a single
data point that sets the Transformer architecture (Befund 37, ``composite.py``) directly against
baselines/GRU/LSTM on **the same** data. n_train=16 is the next smaller
support point of the established learning curve ``{8,16,32,64}`` (``study_learning_curve.py``),
not chosen arbitrarily.

Standard production configuration, NOT shortened (explicit requirement): all factories without
config override, i.e. ``max_epochs=200``, ``patience=20`` for GRU/LSTM/Transformer --
identical to Befund 28/36. An asymmetric shortcut only for the Transformer would be
an uncontrolled methodological deviation.

The same dataset construction as in the constant-run holdout worker (``n_val=32, n_test=32`` fixed,
independent of ``n_train`` -- ``build_datasets``/``evaluate_point`` in
``twinlink_ml.evaluation.experiments`` take ``n_train`` as a prefix from the fixed
``n_train_max=64`` pool, ``data.split.subset(n_train)``). The same pattern that
confirmed the calibration of the cost model in the step-A diagnosis (11.55s
predicted against 11.6s measured at n_train=32, ``n_val=32``).

Progress log: ``verbose=True`` to ``CompositeSequenceModel.fit`` for GRU/LSTM/
Transformer (one line per epoch, incl. ``val_loss``/``stale``, see
``TorchSequenceModel.fit``) plus own start/end markers per model -- convention from
``run_arch_constant_holdout_worker.py``/``_sweep.py`` (there a ``.log`` file per job).
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import torch

from twinlink_ml.evaluation import DataSpec, ValidationPolicy, build_datasets, generate_data
from twinlink_ml.evaluation.evaluator import RolloutEvaluator
from twinlink_ml.models import LinearARXModel, PersistenceModel, StaticRegressionModel
from twinlink_ml.models.composite import (
    CompositeGruSequenceModel,
    CompositeLstmSequenceModel,
    CompositeTransformerSequenceModel,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.tracking import JsonlExperimentLogger, RunRecord, environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "runs" / "transformer_reduced_comparison.jsonl"
SEED = 1
N_TRAIN = 16

TORCH_LABELS = ("gru", "lstm", "transformer")
FACTORIES = {
    "persistence": PersistenceModel,
    "static": StaticRegressionModel,
    "arx": LinearARXModel,
    "gru": CompositeGruSequenceModel,
    "lstm": CompositeLstmSequenceModel,
    "transformer": CompositeTransformerSequenceModel,
}


def _spec() -> DataSpec:
    """Like ``_excited_spec`` in the constant-run holdout worker -- the same pool construction.

    ``n_val=32``/``n_test=32`` fixed, ``n_train_max=64`` (prefix ``n_train=16`` from it).
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


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def main() -> None:
    torch.set_num_threads(1)
    _log("Start: verkleinerter Vergleich, 1 Seed, n_train=16, Standard-Produktionskonfiguration")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)

    spec = _spec()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        data = generate_data(config, spec, master_seed=SEED)
    _log(
        f"Daten generiert: Pool {spec.pool_size} Laeufe (n_train_max={spec.n_train_max}, "
        f"n_val={spec.n_val}, n_test={spec.n_test}), Praefix n_train={N_TRAIN}"
    )

    train, validation, test = build_datasets(data, N_TRAIN, policy=ValidationPolicy.FIXED_POOL)
    _log(
        f"Datensaetze gebaut: {train.n_windows} Trainingsfenster, "
        f"{validation.n_windows if validation is not None else 0} Validierungsfenster, "
        f"{test.n_windows} Testfenster"
    )

    evaluator = RolloutEvaluator(horizons=(1, spec.horizon))
    logger = JsonlExperimentLogger(OUTPUT_PATH)

    common_params = {
        **spec.as_parameters(),
        "process/fingerprint": data.process_config.fingerprint,
        "split/digest": data.split.digest,
        **data.seeds.as_parameters(),
        "n_train": N_TRAIN,
        "n_val": spec.n_val,
        "validation_policy": ValidationPolicy.FIXED_POOL.value,
        "study/note": "schritt_b_reduziert_1seed_n16",
        **environment_fingerprint(),
    }

    overall_start = time.perf_counter()
    for label, factory in FACTORIES.items():
        _log(f"-- Modell {label!r}: Start --")
        model = factory()
        model_start = time.perf_counter()
        if label in TORCH_LABELS:
            model.fit(train, seed=data.seeds.model_seed, validation=validation, verbose=True)
        else:
            model.fit(train, seed=data.seeds.model_seed, validation=validation)
        model_elapsed = time.perf_counter() - model_start
        _log(f"-- Modell {label!r}: Training fertig ({model_elapsed:.1f}s) --")

        result = evaluator.evaluate(model, test)
        record = RunRecord(
            group="transformer_reduced_comparison",
            params={
                **common_params,
                "model/label": label,
                **{f"model/{key}": value for key, value in model.describe().items()},
            },
            metrics=result.as_flat_metrics(),
            tags={"model": label, "group": "transformer_reduced_comparison"},
        )
        logger.log(record)
        _log(f"-- Modell {label!r}: ausgewertet und geloggt --\n{result.format()}")

    logger.close()
    total_elapsed = time.perf_counter() - overall_start
    _log(
        f"Fertig. Gesamtzeit: {total_elapsed:.1f}s ({total_elapsed / 60.0:.1f} Min). "
        f"Log: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()

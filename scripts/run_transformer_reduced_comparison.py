r"""Schritt B (Session zu Befund 37): verkleinerter Vergleich, 1 Seed, n_train=16.

Explizit reduziert gegenueber der vollen Lernkurve (3 Seeds, n_train=32) -- ein einzelner
Datenpunkt, der die Transformer-Architektur (Befund 37, ``composite.py``) direkt gegen
Baselines/GRU/LSTM auf **denselben** Daten stellt. n_train=16 ist die naechst kleinere
Stuetzstelle der etablierten Lernkurve ``{8,16,32,64}`` (``study_learning_curve.py``),
nicht willkuerlich gewaehlt.

Standard-Produktionskonfiguration, NICHT verkuerzt (explizite Vorgabe): alle Fabriken ohne
Config-Override, also ``max_epochs=200``, ``patience=20`` fuer GRU/LSTM/Transformer --
identisch zu Befund 28/36. Eine asymmetrische Abkuerzung nur fuer den Transformer waere
eine unkontrollierte methodische Abweichung.

Derselbe Datensatz-Aufbau wie im Konstantlauf-Holdout-Worker (``n_val=32, n_test=32`` fest,
unabhaengig von ``n_train`` -- ``build_datasets``/``evaluate_point`` in
``twinlink_ml.evaluation.experiments`` ziehen ``n_train`` als Praefix aus dem festen
``n_train_max=64``-Pool, ``data.split.subset(n_train)``). Dasselbe Muster, das die
Kalibrierung des Kostenmodells in der Schritt-A-Diagnose bestaetigt hat (11.55s
vorhergesagt gegen 11.6s gemessen bei n_train=32, ``n_val=32``).

Fortschritts-Log: ``verbose=True`` an ``CompositeSequenceModel.fit`` fuer GRU/LSTM/
Transformer (eine Zeile je Epoche, inkl. ``val_loss``/``stale``, siehe
``TorchSequenceModel.fit``) plus eigene Start/Ende-Marker je Modell -- Konvention aus
``run_arch_constant_holdout_worker.py``/``_sweep.py`` (dort ``.log``-Datei je Job).
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
    """Wie ``_excited_spec`` im Konstantlauf-Holdout-Worker -- derselbe Pool-Aufbau.

    ``n_val=32``/``n_test=32`` fest, ``n_train_max=64`` (Praefix ``n_train=16`` daraus).
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
    _log(f"Daten generiert: Pool {spec.pool_size} Laeufe (n_train_max={spec.n_train_max}, "
         f"n_val={spec.n_val}, n_test={spec.n_test}), Praefix n_train={N_TRAIN}")

    train, validation, test = build_datasets(data, N_TRAIN, policy=ValidationPolicy.FIXED_POOL)
    _log(f"Datensaetze gebaut: {train.n_windows} Trainingsfenster, "
         f"{validation.n_windows if validation is not None else 0} Validierungsfenster, "
         f"{test.n_windows} Testfenster")

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
    _log(f"Fertig. Gesamtzeit: {total_elapsed:.1f}s ({total_elapsed / 60.0:.1f} Min). "
         f"Log: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

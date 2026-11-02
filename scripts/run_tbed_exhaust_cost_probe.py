r"""Cost probe (step 1, Befund 41 follow-up task): T_bed at production size, plus
end-to-end quality of the exhaust air back-calculation.

Befund 41g explicitly left open whether ``bed_temperature_celsius`` is predicted well enough
at real training size (not the 15-epoch/n_train=8 correctness run) to make the
exhaust air back-calculation (``simulation.exhaust.reconstruct_exhaust_state()``)
actually usable. This script delivers exactly this one measurement: one
GRU run, one seed, standard production configuration, four targets (the three established
plus ``bed_temperature_celsius`` as an explicit override -- ``DataSpec.target_names``
stays at the three-target default, Befund 41f), then both the normal
four-target NRMSE and the end-to-end exhaust metric from
``evaluation.exhaust_reconstruction`` (mass-flow leak correction, see there).

Standard production configuration identical to ``run_mamba_reduced_comparison.py``
(Befund 40): ``greybox_gab_placeholder.yaml``, ``sample_time_s=25.0``,
``run_duration_s=28_800.0``, ``n_train_max=64``, ``n_val=32``, ``n_test=32``,
``history_length=32``, ``horizon=36``, ``stride=8`` -- so that the epoch/time values are
comparable against the GRU/LSTM runs already measured there.

Only a cost probe, no sweep: one seed, only GRU (Transformer/Mamba already documented as
unsuitable, Befund 37/38/40 -- not checked again here).
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import torch

from twinlink_ml.evaluation import DataSpec, ValidationPolicy, build_datasets, generate_data
from twinlink_ml.evaluation.evaluator import RolloutEvaluator
from twinlink_ml.evaluation.exhaust_reconstruction import evaluate_exhaust_reconstruction
from twinlink_ml.models.composite import CompositeGruSequenceModel
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
SEED = 1
N_TRAIN = 64

THREE_TARGETS = ("energy_target", "product_moisture_kg_per_kg", "d50_um")
FOUR_TARGETS = (*THREE_TARGETS, "bed_temperature_celsius")


def _spec() -> DataSpec:
    return DataSpec(
        sample_time_s=25.0,
        run_duration_s=28_800.0,
        n_train_max=64,
        n_val=32,
        n_test=32,
        history_length=32,
        horizon=36,
        stride=8,
        target_names=FOUR_TARGETS,
    )


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def main() -> None:
    torch.set_num_threads(1)
    _log(
        "Start: T_bed-Kostenprobe -- GRU, 1 Seed, n_train=64, vier Ziele inkl. "
        "bed_temperature_celsius (expliziter Override, siehe Befund 41f)"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)

    spec = _spec()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        data = generate_data(config, spec, master_seed=SEED)

    _log(
        f"Datensatz: process/fingerprint={data.process_config.fingerprint} "
        f"split/digest={data.split.digest} seed/model={data.seeds.model_seed}"
    )

    train, validation, test = build_datasets(data, N_TRAIN, policy=ValidationPolicy.FIXED_POOL)
    _log(
        f"Datensaetze gebaut: {train.n_windows} Trainingsfenster, "
        f"{validation.n_windows if validation is not None else 0} Validierungsfenster, "
        f"{test.n_windows} Testfenster"
    )

    evaluator = RolloutEvaluator(horizons=(1, spec.horizon))

    _log("-- Modell 'gru': Start --")
    model = CompositeGruSequenceModel()
    model_start = time.perf_counter()
    model.fit(train, seed=data.seeds.model_seed, validation=validation, verbose=True)
    model_elapsed = time.perf_counter() - model_start
    _log(
        f"-- Modell 'gru': Training fertig ({model_elapsed:.1f}s, {model_elapsed / 60.0:.1f} Min) --"
    )

    epoch_counts: dict[str, int] = {}
    for target, history in model.training_history.items():
        epoch_counts[target] = len(history)
        _log(f"   Teilnetz {target!r}: {len(history)}/{model.config.max_epochs} Epochen")

    result = evaluator.evaluate(model, test)
    _log(f"-- Vier-Ziele-NRMSE --\n{result.format()}")

    exhaust_result = evaluate_exhaust_reconstruction(model, data, test, horizons=(1, spec.horizon))
    _log(
        f"-- End-zu-End-Exhaust-Metrik (korrigiert, ohne Massenstrom-Leck) --\n{exhaust_result.format()}"
    )

    summary = {
        "config": {
            "process_config": str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "master_seed": SEED,
            "n_train": N_TRAIN,
            **spec.as_parameters(),
        },
        "fingerprints": data.fingerprints,
        "timing": {
            "model_fit_seconds": model_elapsed,
        },
        "epoch_counts": epoch_counts,
        "max_epochs": model.config.max_epochs,
        "nrmse_run_spread": {
            target: {str(h): list(values) for h, values in by_horizon.items()}
            for target, by_horizon in result.nrmse_run_spread.items()
        },
        "exhaust_nrmse_run_spread": {
            target: {str(h): list(values) for h, values in by_horizon.items()}
            for target, by_horizon in exhaust_result.nrmse_run_spread.items()
        },
    }
    output_path = Path(
        "/tmp/claude-1000/-home-johncox-PhD-twinlink-ml--2--twinlink-ml--1--twinlink-ml/7142dce7-c169-4e7c-b6c7-b52081c05dea/scratchpad/tbed_exhaust_cost_probe.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    _log(f"Zusammenfassung geschrieben nach {output_path}")

    _log("Fertig.")


if __name__ == "__main__":
    main()

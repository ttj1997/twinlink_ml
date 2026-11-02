r"""Step 2 (Mamba work strand): reduced comparison, 1 seed, n_train=16.

**Only Mamba is trained here.** Before this script it was checked (plain-text discussion
in the session on this finding) that ``run_transformer_reduced_comparison.py``
(Befund 37d, ``artifacts/runs/transformer_reduced_comparison.jsonl``) uses exactly the same
``DataSpec``, the same ``master_seed=1`` and the same ``greybox_gab_placeholder.yaml``
-- verified bit-identical on ``process/fingerprint``, ``split/digest`` and all
three derived seeds (``seed/data``, ``seed/noise``, ``seed/model``). The six records
already logged there (persistence, static, arx, gru, lstm, transformer) are
therefore taken over UNCHANGED, not recomputed -- this saves the ~55 minutes of
GRU/LSTM/Transformer training time already measured in Befund 37d.

Output: ``artifacts/runs/mamba_reduced_comparison.jsonl`` with all seven records
(six copied unchanged from Befund 37d, plus the new Mamba record) --
``transformer_reduced_comparison.jsonl`` itself stays untouched.

Standard production configuration for Mamba, NOT shortened: ``CompositeMambaSequenceModel()``
without config override except its own capacity default (``hidden_size=16,
state_size=16, num_layers=1``, see the ``composite.py`` docstring) -- ``max_epochs=200``,
``patience=20`` (dataclass default, identical to Befund 28/36/37).
"""

from __future__ import annotations

import shutil
import time
import warnings
from collections.abc import Mapping
from pathlib import Path

import torch

from twinlink_ml.evaluation import DataSpec, ValidationPolicy, build_datasets, generate_data
from twinlink_ml.evaluation.evaluator import RolloutEvaluator
from twinlink_ml.models.composite import CompositeMambaSequenceModel
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.tracking import JsonlExperimentLogger, RunRecord, environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
REUSED_LOG_PATH = REPO_ROOT / "artifacts" / "runs" / "transformer_reduced_comparison.jsonl"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "runs" / "mamba_reduced_comparison.jsonl"
SEED = 1
N_TRAIN = 16

# Verified bit-identical with the values logged in transformer_reduced_comparison.jsonl
# (session plain-text check before this script) -- the cross-check below in main() repeats
# the same comparison, so that a later call (e.g. after a config change)
# does not silently merge incompatible records.
#
# Fingerprint updated from 82f65f576a2e5253 to 40aae2b4526a5ed7: NumericsConfig.
# enforce_gas_velocity_projection added (default OFF). ProcessConfig.fingerprint
# hashes the entire config, so the value changes even with unchanged behaviour.
# Equivalence shown by test_gas_velocity_projection_disabled_matches_legacy_baseline
# (tests/test_gas_velocity_projection.py) -- see there for the bit-identity evidence.
EXPECTED = {
    "process/fingerprint": "40aae2b4526a5ed7",
    "split/digest": "5b9628a151344f37",
    "seed/data": 57731891,
    "seed/noise": 1868603354,
    "seed/model": 2509151119,
    "data/target_names": ("energy_target", "product_moisture_kg_per_kg", "d50_um"),
}
"""``data/target_names`` bewusst hier fixiert, NICHT aus dem ``DataSpec``-Default gelesen:
seit G5 hat der Default nur noch zwei Zielgroessen (``d50_um`` entfernt,
``docs/04_stand_ap1.md`` Befund H2). Ein Log mit uneinheitlicher Kanalzahl ueber seine
Records ist wertlos und faellt spaeter niemandem auf -- deshalb hart pruefen, nicht nur
die drei bisherigen Schluessel."""


def _spec() -> DataSpec:
    """Identical to ``run_transformer_reduced_comparison._spec`` (Befund 37d)."""
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


def _check_expected(actual: Mapping[str, object]) -> None:
    """Check ``EXPECTED`` against actual values, abort hard on a deviation.

    Separate function (not inline in ``main()``) so that the schema check is testable
    without a full data-generation/training run.
    """
    mismatches = {k: (v, actual[k]) for k, v in EXPECTED.items() if v != actual[k]}
    if mismatches:
        raise RuntimeError(
            f"Daten/Seeds/Zielkanaele weichen von transformer_reduced_comparison.jsonl ab, "
            f"Wiederverwendung der sechs alten Records waere UNGUELTIG: {mismatches}"
        )


def main() -> None:
    torch.set_num_threads(1)
    _log(
        "Start: Mamba-Trainingslauf, n_train=16, Standard-Produktionskonfiguration -- "
        "GRU/LSTM/Transformer/Baselines werden NICHT neu gerechnet, siehe Moduldocstring"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)

    spec = _spec()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        data = generate_data(config, spec, master_seed=SEED)

    actual = {
        "process/fingerprint": data.process_config.fingerprint,
        "split/digest": data.split.digest,
        "seed/data": data.seeds.data_seed,
        "seed/noise": data.seeds.noise_seed,
        "seed/model": data.seeds.model_seed,
        "data/target_names": tuple(spec.target_names),
    }
    _check_expected(actual)
    _log(
        "Fingerprint/Split/Seeds/Zielkanaele bitgleich zu "
        "transformer_reduced_comparison.jsonl bestaetigt."
    )

    train, validation, test = build_datasets(data, N_TRAIN, policy=ValidationPolicy.FIXED_POOL)
    _log(
        f"Datensaetze gebaut: {train.n_windows} Trainingsfenster, "
        f"{validation.n_windows if validation is not None else 0} Validierungsfenster, "
        f"{test.n_windows} Testfenster"
    )

    evaluator = RolloutEvaluator(horizons=(1, spec.horizon))

    common_params = {
        **spec.as_parameters(),
        "process/fingerprint": data.process_config.fingerprint,
        "split/digest": data.split.digest,
        **data.seeds.as_parameters(),
        "n_train": N_TRAIN,
        "n_val": spec.n_val,
        "validation_policy": ValidationPolicy.FIXED_POOL.value,
        "study/note": "schritt2_mamba_reduziert_1seed_n16_reuse_gru_lstm_transformer",
        **environment_fingerprint(),
    }

    _log("-- Modell 'mamba': Start --")
    model = CompositeMambaSequenceModel()
    model_start = time.perf_counter()
    model.fit(train, seed=data.seeds.model_seed, validation=validation, verbose=True)
    model_elapsed = time.perf_counter() - model_start
    _log(f"-- Modell 'mamba': Training fertig ({model_elapsed:.1f}s) --")

    for target, history in model.training_history.items():
        _log(f"   Teilnetz {target!r}: {len(history)}/{model.config.max_epochs} Epochen")

    result = evaluator.evaluate(model, test)
    record = RunRecord(
        group="mamba_reduced_comparison",
        params={
            **common_params,
            "model/label": "mamba",
            **{f"model/{key}": value for key, value in model.describe().items()},
        },
        metrics=result.as_flat_metrics(),
        tags={"model": "mamba", "group": "mamba_reduced_comparison"},
    )
    _log(f"-- Modell 'mamba': ausgewertet --\n{result.format()}")

    # Copy six existing records unchanged (transformer_reduced_comparison.jsonl
    # itself stays untouched), then append the new Mamba record (the logger is
    # appending, see the tracking.JsonlExperimentLogger docstring).
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REUSED_LOG_PATH, OUTPUT_PATH)
    with REUSED_LOG_PATH.open("r", encoding="utf-8") as handle:
        n_reused = sum(1 for line in handle if line.strip())
    logger = JsonlExperimentLogger(OUTPUT_PATH)
    logger.log(record)
    _log(
        f"{n_reused} bestehende Records aus {REUSED_LOG_PATH.name} uebernommen, "
        f"1 neuer Mamba-Record angehaengt -> {OUTPUT_PATH}"
    )

    _log(f"Fertig. Mamba-Trainingszeit: {model_elapsed:.1f}s ({model_elapsed / 60.0:.1f} Min).")


if __name__ == "__main__":
    main()

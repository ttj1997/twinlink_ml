r"""Schritt 2 (Mamba-Arbeitsstrang): verkleinerter Vergleich, 1 Seed, n_train=16.

**Nur Mamba wird hier trainiert.** Vor diesem Skript wurde geprueft (Klartext-Diskussion
in der Session zu diesem Befund), dass ``run_transformer_reduced_comparison.py``
(Befund 37d, ``artifacts/runs/transformer_reduced_comparison.jsonl``) exakt dieselbe
``DataSpec``, denselben ``master_seed=1`` und dieselbe ``greybox_gab_placeholder.yaml``
verwendet -- bitgleich verifiziert an ``process/fingerprint``, ``split/digest`` und allen
drei abgeleiteten Seeds (``seed/data``, ``seed/noise``, ``seed/model``). Die sechs dort
bereits geloggten Records (persistence, static, arx, gru, lstm, transformer) werden
deshalb UNVERAENDERT uebernommen, nicht neu berechnet -- das spart die ~55 Minuten
GRU/LSTM/Transformer-Trainingszeit, die in Befund 37d bereits gemessen wurden.

Ausgabe: ``artifacts/runs/mamba_reduced_comparison.jsonl`` mit allen sieben Records
(sechs unveraendert aus Befund 37d kopiert, plus der neue Mamba-Record) --
``transformer_reduced_comparison.jsonl`` selbst bleibt unangetastet.

Standard-Produktionskonfiguration fuer Mamba, NICHT verkuerzt: ``CompositeMambaSequenceModel()``
ohne Config-Override ausser dem eigenen Kapazitaets-Standard (``hidden_size=16,
state_size=16, num_layers=1``, siehe ``composite.py``-Docstring) -- ``max_epochs=200``,
``patience=20`` (Dataclass-Standard, identisch zu Befund 28/36/37).
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

# Bitgleich mit den in transformer_reduced_comparison.jsonl geloggten Werten verifiziert
# (Session-Klartext-Check vor diesem Skript) -- Gegenprobe unten in main() wiederholt
# denselben Vergleich, damit ein spaeterer Aufruf (z. B. nach einer Config-Aenderung)
# nicht stillschweigend inkompatible Records zusammenfuehrt.
#
# Fingerprint aktualisiert von 82f65f576a2e5253 auf 40aae2b4526a5ed7: NumericsConfig.
# enforce_gas_velocity_projection hinzugefuegt (Default AUS). ProcessConfig.fingerprint
# hasht die gesamte Config, daher aendert sich der Wert auch bei unveraendertem Verhalten.
# Aequivalenz belegt durch test_gas_velocity_projection_disabled_matches_legacy_baseline
# (tests/test_gas_velocity_projection.py) -- siehe dort fuer den Bitgleichheits-Beleg.
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
    """Identisch zu ``run_transformer_reduced_comparison._spec`` (Befund 37d)."""
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
    """``EXPECTED`` gegen tatsaechliche Werte pruefen, hart abbrechen bei Abweichung.

    Eigene Funktion (nicht inline in ``main()``), damit die Schemapruefung ohne vollen
    Datengenerierungs-/Trainingslauf testbar ist.
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

    # Sechs bestehende Records unveraendert kopieren (transformer_reduced_comparison.jsonl
    # bleibt selbst unangetastet), dann den neuen Mamba-Record anhaengen (Logger ist
    # anfuegend, siehe tracking.JsonlExperimentLogger-Docstring).
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

r"""Schritt 3 (xLSTM-Wiederaufnahme): verkleinerter Vergleich, 1 Seed, n_train=16.

**Nur xLSTM wird hier trainiert.** Analog zu ``run_mamba_reduced_comparison.py``: vor
diesem Skript wurde geprueft, dass ``mamba_reduced_comparison.jsonl`` (Befund 40)
exakt dieselbe ``DataSpec``, denselben ``master_seed=1`` und dieselbe
``greybox_gab_placeholder.yaml`` verwendet wie ``transformer_reduced_comparison.jsonl``
-- bitgleich verifiziert an ``process/fingerprint``, ``split/digest`` und allen drei
abgeleiteten Seeds. Die sieben dort bereits geloggten Records (persistence, static, arx,
gru, lstm, transformer, mamba) werden deshalb UNVERAENDERT uebernommen, nicht neu
berechnet.

Ausgabe: ``artifacts/runs/xlstm_reduced_comparison.jsonl`` mit allen acht Records
(sieben unveraendert aus ``mamba_reduced_comparison.jsonl`` kopiert, plus der neue
xLSTM-Record) -- ``mamba_reduced_comparison.jsonl`` selbst bleibt unangetastet.

Standard-Produktionskonfiguration fuer xLSTM, NICHT verkuerzt:
``CompositeXLSTMSequenceModel()`` ohne Config-Override -- ``composite.py`` setzt fuer
diesen Kern KEINEN eigenen Kapazitaets-Standard (anders als bei Mamba/Transformer), der
Kern laeuft also mit dem generischen ``TorchSequenceConfig``-Standard
``hidden_size=64`` -- derselbe Wert wie GRU/LSTM (Befund 28), eine echte
"gleiche Breite"-Kapazitaetsparitaet. Schritt-0/2-Messung (``diag_xlstm_memory.py``,
``diag_xlstm_cost.py``) bestaetigt, dass das bei Produktionsgroesse unproblematisch ist
(674 MB Speicher-Delta, ~0.34s/Trainingsschritt, geschaetzt ~30 Minuten Gesamtlaufzeit
fuer alle drei Teilnetze -- weit unter dem historischen >12h-Fall, der bei einem viel
groesseren, parameterzahl-angepassten ``hidden_size`` auftrat, siehe
``CompositeTransformerSequenceModel``-Docstring). ``max_epochs=200``, ``patience=20``
(Dataclass-Standard, identisch zu Befund 28/36/37/40).
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
from twinlink_ml.models.composite import CompositeXLSTMSequenceModel
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.tracking import JsonlExperimentLogger, RunRecord, environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
REUSED_LOG_PATH = REPO_ROOT / "artifacts" / "runs" / "mamba_reduced_comparison.jsonl"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "runs" / "xlstm_reduced_comparison.jsonl"
SEED = 1
N_TRAIN = 16

# Bitgleich mit den in mamba_reduced_comparison.jsonl/transformer_reduced_comparison.jsonl
# geloggten Werten verifiziert -- Gegenprobe unten in main() wiederholt denselben
# Vergleich, damit ein spaeterer Aufruf (z. B. nach einer Config-Aenderung) nicht
# stillschweigend inkompatible Records zusammenfuehrt.
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
    """Identisch zu ``run_mamba_reduced_comparison._spec``/``_transformer_reduced_comparison``."""
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
            f"Daten/Seeds/Zielkanaele weichen von mamba_reduced_comparison.jsonl ab, "
            f"Wiederverwendung der sieben alten Records waere UNGUELTIG: {mismatches}"
        )


def main() -> None:
    torch.set_num_threads(1)
    _log(
        "Start: xLSTM-Trainingslauf, n_train=16, Standard-Produktionskonfiguration -- "
        "GRU/LSTM/Transformer/Mamba/Baselines werden NICHT neu gerechnet, siehe Moduldocstring"
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
        "Fingerprint/Split/Seeds/Zielkanaele bitgleich zu mamba_reduced_comparison.jsonl "
        "bestaetigt."
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
        "study/note": "schritt3_xlstm_reduziert_1seed_n16_reuse_gru_lstm_transformer_mamba",
        **environment_fingerprint(),
    }

    _log("-- Modell 'xlstm': Start --")
    model = CompositeXLSTMSequenceModel()
    model_start = time.perf_counter()
    model.fit(train, seed=data.seeds.model_seed, validation=validation, verbose=True)
    model_elapsed = time.perf_counter() - model_start
    _log(f"-- Modell 'xlstm': Training fertig ({model_elapsed:.1f}s) --")

    for target, history in model.training_history.items():
        _log(f"   Teilnetz {target!r}: {len(history)}/{model.config.max_epochs} Epochen")

    # inference_us_per_window ist bereits Teil von result.as_flat_metrics() (Evaluator misst
    # intern, siehe evaluator.py) -- kein eigenes Timing hier, sonst zwei leicht
    # unterschiedliche Zahlen fuer dieselbe Groesse (Konvention aus Mamba-/Transformer-Skript).
    result = evaluator.evaluate(model, test)
    record = RunRecord(
        group="xlstm_reduced_comparison",
        params={
            **common_params,
            "model/label": "xlstm",
            **{f"model/{key}": value for key, value in model.describe().items()},
        },
        metrics=result.as_flat_metrics(),
        tags={"model": "xlstm", "group": "xlstm_reduced_comparison"},
    )
    _log(f"-- Modell 'xlstm': ausgewertet --\n{result.format()}")

    # Sieben bestehende Records unveraendert kopieren (mamba_reduced_comparison.jsonl
    # selbst bleibt unangetastet), dann den neuen xLSTM-Record anhaengen.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REUSED_LOG_PATH, OUTPUT_PATH)
    with REUSED_LOG_PATH.open("r", encoding="utf-8") as handle:
        n_reused = sum(1 for line in handle if line.strip())
    logger = JsonlExperimentLogger(OUTPUT_PATH)
    logger.log(record)
    _log(
        f"{n_reused} bestehende Records aus {REUSED_LOG_PATH.name} uebernommen, "
        f"1 neuer xLSTM-Record angehaengt -> {OUTPUT_PATH}"
    )

    _log(f"Fertig. xLSTM-Trainingszeit: {model_elapsed:.1f}s ({model_elapsed / 60.0:.1f} Min).")


if __name__ == "__main__":
    main()

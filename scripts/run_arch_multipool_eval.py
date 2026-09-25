r"""Mehrfach-Pool-Auswertung bereits trainierter Checkpoints -- kein Neutraining.

Ergaenzt ``scripts/run_arch_constant_holdout_worker.py``: statt des einen dort
verwendeten Konstantlauf-Holdout-Pools (``HOLDOUT_SEED=999``) werden hier zwei
weitere, unabhaengig gezogene Pools erzeugt (997, 998) und **alle 20 bereits
gespeicherten Checkpoints** (``load_checkpoint()``, kein ``fit()``) dagegen
ausgewertet. Ziel: pruefen, ob die Verbesserung gegenueber Persistenz auf dem
Konstantlauf-Holdout von der zufaelligen Wahl des einen bisher verwendeten
Pools abhaengt, oder ueber mehrere unabhaengige Pools stabil bleibt.

Braucht **keine** Trainings-/angeregten Daten -- ``load_checkpoint()`` liefert
Modellgewichte und die zum Training gehoerigen Skalierer direkt aus der
``.pt``-Datei zurueck (siehe ``utils/checkpoints.py``). Fuer jeden neuen Pool
wird die Rohtrajektorie **einmal** erzeugt und dann 20-mal gefenstert, je mit
dem checkpoint-eigenen Skalarpaar -- genau wie im Worker-Skript fuer den
999er-Pool, nur ohne den vorausgehenden Trainingsschritt.

Persistenz wird zum Vergleich mitgeloggt: ihr NRMSE in physikalischen
Einheiten ist skalierer-invariant (Rundreise-Argument, Befund 5-Vorstufe),
deshalb genuegt ``PersistenceModel.fit()`` auf der Testmenge selbst -- die
Methode liest dort nur die strukturelle Kanalzuordnung, keine Trainingsdaten.

Aufruf::

    python scripts/run_arch_multipool_eval.py

Schreibt nach ``artifacts/runs/arch_gab_constant_holdout_multipool.jsonl``
(bestehende ``arch_gab_constant_holdout.jsonl`` bleibt unangetastet).
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twinlink_ml.data.splits import SplitName
from twinlink_ml.data.windows import SequenceDataset, TrajectoryWindowDataset
from twinlink_ml.evaluation import DataSpec, generate_data
from twinlink_ml.evaluation.evaluator import RolloutEvaluator
from twinlink_ml.models import PersistenceModel
from twinlink_ml.models.base import SequenceModel
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.checkpoints import load_checkpoint
from twinlink_ml.utils.tracking import JsonlExperimentLogger, RunRecord, environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "runs" / "arch_gab_constant_holdout_multipool.jsonl"

NEW_HOLDOUT_SEEDS = (997, 998)
"""Zwei zusaetzliche, unabhaengig gezogene Konstantlauf-Pools -- verschieden vom
bisher verwendeten ``HOLDOUT_SEED=999`` in ``run_arch_constant_holdout_worker.py``."""

ARCHITECTURE_LABELS = ("gru", "lstm")
TRAIN_SEEDS = range(1, 11)


class CheckpointSpecMismatchError(RuntimeError):
    """Zwei Checkpoints dieses Sweeps tragen unterschiedliche ``data_spec``-Parameter.

    Kann u. a. entstehen, wenn Checkpoints aus unterschiedlichen Ständen des
    ``DataSpec``-Defaults stammen (siehe G5, ``docs/04_stand_ap1.md`` Befund H2) -- ein
    Auswertungslauf über mehrere solcher Checkpoints hinweg wäre nicht mehr
    vergleichbar, das Fenster für den einen Checkpoint passt nicht zum anderen.
    """


def _spec_from_checkpoint(data_spec_params: Mapping[str, Any]) -> DataSpec:
    """``DataSpec`` unverändert aus den im Checkpoint gespeicherten Parametern rekonstruieren.

    Der Modul-Docstring dieses Skripts (und von ``run_arch_constant_holdout_worker.py``)
    fordert exakte Übereinstimmung mit der Sweep-Spec, gegen die der geladene Checkpoint
    trainiert wurde -- Fenstergeometrie **und** Kanalzahl. Ein hier hart kodiertes
    ``DataSpec()`` würde bei einer künftigen Änderung des ``DataSpec``-Defaults (wie G5)
    unbemerkt vom tatsächlich trainierten Checkpoint abweichen (H2). Das Auslesen aus dem
    Checkpoint selbst macht diese Abhängigkeit strukturell unmöglich, statt sie erneut zu
    behaupten.
    """
    prefix = "data/"
    field_names = {f.name for f in dataclasses.fields(DataSpec)}
    kwargs: dict[str, Any] = {}
    for key, value in data_spec_params.items():
        if not key.startswith(prefix):
            continue
        name = key[len(prefix) :]
        if name not in field_names:
            continue
        kwargs[name] = tuple(value) if isinstance(value, list | tuple) else value
    return DataSpec(**kwargs)


def _verify_matching_spec(
    reference: DataSpec, candidate: DataSpec, *, checkpoint_path: Path
) -> None:
    """Bricht hart ab, wenn ein Checkpoint von der Referenz-Spec des Sweeps abweicht.

    Harter Fehler statt Warnung (Auftragsvorgabe J2): eine Auswertung, die zwei
    Checkpoints mit unterschiedlicher Kanalzahl stillschweigend gegen dasselbe Fenster
    liefe, würde für den einen Checkpoint falsch (oder mit ``IndexError``) rechnen, ohne
    dass das später auffiele.
    """
    if reference != candidate:
        raise CheckpointSpecMismatchError(
            f"{checkpoint_path}: data_spec weicht vom ersten geladenen Checkpoint ab "
            f"({candidate} != {reference}) -- dieser Sweep ist nicht homogen, eine "
            "gemeinsame Auswertung wäre nicht vergleichbar."
        )


def _holdout_spec(excited: DataSpec) -> DataSpec:
    """Konstantlauf-Pool, ``n_train_max=2, n_val=2`` sind Dummy-Bloecke, nie gelesen."""
    return dataclasses.replace(
        excited, n_train_max=2, n_val=2, n_test=excited.n_test, constant_run_fraction=1.0
    )


def _log_result(
    logger: JsonlExperimentLogger,
    evaluator: RolloutEvaluator,
    dataset: SequenceDataset,
    *,
    excited_spec: DataSpec,
    seed: int,
    holdout_seed: int,
    common_extra: dict[str, Any],
    label: str,
    model: SequenceModel,
    checkpoint_path: str | None,
) -> None:
    """Ein Modell auf einem Pool auswerten und als ``RunRecord`` loggen."""
    result = evaluator.evaluate(model, dataset)
    test_set = f"constant_holdout_{holdout_seed}"
    params: dict[str, object] = {
        **excited_spec.as_parameters(),
        # data/train_constant_run_fraction wiederholt data/constant_run_fraction (aus
        # derselben excited_spec) bewusst redundant -- NICHT als ueberfluessig entfernen.
        # data/constant_run_fraction ist ein generisches Spec-Attribut und bleibt fuer
        # JEDEN test_set-Wert in diesem Record identisch (auch fuer
        # test_set="constant_holdout_*"), sagt also fuer sich genommen nichts ueber die
        # Trainingsart aus. data/train_constant_run_fraction ist die explizite, eindeutig
        # benannte Aussage "so trainiert" -- ohne sie waere ein konstant-trainierter Lauf
        # von einem angeregt-trainierten anhand des Records nicht unterscheidbar (siehe
        # Befund-Eintrag zu diesem Feld in docs/04_stand_ap1.md).
        "data/train_constant_run_fraction": excited_spec.constant_run_fraction,
        "master_seed": seed,
        "test_set": test_set,
        "model/label": label,
        **environment_fingerprint(),
        **common_extra,
    }
    if checkpoint_path is not None:
        params["checkpoint/path"] = checkpoint_path
    logger.log(
        RunRecord(
            group="arch_gab_constant_holdout_multipool",
            params=params,
            metrics=result.as_flat_metrics(),
            tags={
                "model": label,
                "group": "arch_gab_constant_holdout_multipool",
                "test_set": test_set,
            },
        )
    )


def _load_all_checkpoints() -> dict[tuple[str, int], tuple[SequenceModel, dict[str, Any]]]:
    """Alle 20 Checkpoints einmalig laden, bevor die Pools erzeugt werden.

    Ermoeglicht, die Referenz-Spec (Zielkanaele, Fenstergeometrie) aus den Checkpoints
    selbst abzuleiten und ihre gegenseitige Konsistenz zu pruefen, bevor irgendeine
    ``TrajectoryWindowDataset``-Fensterung stattfindet.
    """
    loaded: dict[tuple[str, int], tuple[SequenceModel, dict[str, Any]]] = {}
    for architecture in ARCHITECTURE_LABELS:
        for seed in TRAIN_SEEDS:
            ckpt_path = CHECKPOINT_DIR / f"{architecture}_seed{seed}.pt"
            loaded[(architecture, seed)] = load_checkpoint(ckpt_path)
    return loaded


def main() -> None:
    """Zwei neue Pools erzeugen, alle 20 Checkpoints je Pool auswerten, JSONL schreiben."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)

    checkpoints = _load_all_checkpoints()
    (first_arch, first_seed), (_first_model, first_meta) = next(iter(checkpoints.items()))
    excited_spec = _spec_from_checkpoint(first_meta["data_spec"])
    reference_path = CHECKPOINT_DIR / f"{first_arch}_seed{first_seed}.pt"
    for (architecture, seed), (_model, meta) in checkpoints.items():
        ckpt_path = CHECKPOINT_DIR / f"{architecture}_seed{seed}.pt"
        candidate = _spec_from_checkpoint(meta["data_spec"])
        _verify_matching_spec(excited_spec, candidate, checkpoint_path=ckpt_path)
    print(
        f"Referenz-Spec aus {reference_path.name} abgeleitet, "
        f"target_names={excited_spec.target_names}, "
        f"gegen alle {len(checkpoints)} Checkpoints geprueft."
    )

    holdout_spec = _holdout_spec(excited_spec)
    evaluator = RolloutEvaluator(horizons=(1, excited_spec.horizon))
    logger = JsonlExperimentLogger(OUTPUT_PATH)

    for holdout_seed in NEW_HOLDOUT_SEEDS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            holdout = generate_data(config, holdout_spec, master_seed=holdout_seed)
        print(
            f"[pool {holdout_seed}] Daten generiert, "
            f"{len(holdout.split.test)} Konstantlauf-Testlaeufe",
            flush=True,
        )

        common_extra = {
            "holdout/seed": holdout_seed,
            "holdout/n_runs": len(holdout.split.test),
            "process/fingerprint": holdout.process_config.fingerprint,
        }

        for architecture in ARCHITECTURE_LABELS:
            for seed in TRAIN_SEEDS:
                ckpt_path = CHECKPOINT_DIR / f"{architecture}_seed{seed}.pt"
                model, metadata = checkpoints[(architecture, seed)]
                scalers = (metadata["input_scaler"], metadata["target_scaler"])
                test_constant = TrajectoryWindowDataset.from_trajectory(
                    holdout.trajectory,
                    holdout.split,
                    SplitName.TEST,
                    excited_spec.window,
                    scalers=scalers,
                )

                _log_result(
                    logger,
                    evaluator,
                    test_constant,
                    excited_spec=excited_spec,
                    seed=seed,
                    holdout_seed=holdout_seed,
                    common_extra=common_extra,
                    label=architecture,
                    model=model,
                    checkpoint_path=str(ckpt_path),
                )

                persistence = PersistenceModel()
                persistence.fit(test_constant, seed=0)
                _log_result(
                    logger,
                    evaluator,
                    test_constant,
                    excited_spec=excited_spec,
                    seed=seed,
                    holdout_seed=holdout_seed,
                    common_extra=common_extra,
                    label="persistence",
                    model=persistence,
                    checkpoint_path=None,
                )

        print(
            f"[pool {holdout_seed}] fertig, 20 Checkpoints + Persistenz ausgewertet",
            flush=True,
        )

    logger.close()
    print(f"Fertig. Ergebnis: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

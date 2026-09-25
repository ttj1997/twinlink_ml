r"""Splitvergleich: fenster-/punktweise vs. laufweise Aufteilung, Extra Trees.

Antwortet auf den Vorschlag aus ``docs/04_stand_ap1.md``, Abschnitt "Was das Paper des
Vorgängers ändert": Kräuter, Zhou & Heinrich (ABS23) teilen ihre Trainings-/Testdaten
zufällig über Einzelbeobachtungen auf, nicht über Chargen -- ein Fenster-Leck, das ihr
berichtetes R² = 0.994 fast vollständig erklärt (gegen NRMSE ≈ 0.97 im Soft-Sensor-Betrieb
dieses Projekts, siehe dort). Dieses Skript misst BEIDE Aufteilungen auf **demselben**
Datensatz mit **demselben** Modell (:class:`~twinlink_ml.models.extra_trees.ExtraTreesModel`,
die methodische Reproduktion von Kräuters Merkmalsraum) -- der Abstand zwischen den
beiden Zahlen ist das Ergebnis, nicht eine einzelne Genauigkeit.

**Kein Architekturvergleich.** ``extra_trees`` steht bewusst nicht in
``models/registry.py``/``MODEL_SLOTS`` (siehe ``models/extra_trees.py``-Moduldocstring) und
dieses Skript ist kein Ersatz für ``study_architectures.py``.

Gleicher Lauf-Pool in beiden Armen
-----------------------------------
Beide Arme verwenden exakt dieselben 96 Läufe (``data.split.train + data.split.test``, kein
``val``-Block -- Extra Trees hat kein Early Stopping, das einen bräuchte). Arm A
(laufweise) ordnet ganze Läufe zu, wie im übrigen Projekt Standard
(:class:`~twinlink_ml.data.splits.IndexBlockSplitter`, bereits Teil von
``data.split``). Arm B (fenster-/punktweise) fenstert denselben Lauf-Pool **zusammen**
und verteilt einzelne Fenster über :class:`~twinlink_ml.data.point_split.PointSplitter`,
mit denselben Zielanteilen (train : test = 64 : 32 Läufe entspricht 2:1, siehe
``_POINT_RATIOS``). Damit ist der Split die einzige Variable zwischen beiden Zahlen.

Fenstergeometrie: ``history_length=3, horizon=1`` (Wert + 1./2. Differenz über
:math:`t, t{-}1, t{-}2`, Ein-Schritt-Vorhersage) -- Kräuters Merkmalsraum, nicht die
32-Schritte-Geometrie des GRU/LSTM/Transformer/Mamba-Vergleichs.

Kosten (empirisch gemessen, s. Session-Log vor diesem Lauf, ``master_seed=1``,
Produktionsgröße 128 Läufe × 1152 Schritte): Physik-Rollout ≈ 8 s, Fensterbau je Arm
≤ 0.3 s, ``ExtraTreesModel``-Fit (100 Bäume, ~73 000 Fenster) ≈ 3-4 s je Arm. Kein
Gradiententraining, keine GPU. Bei ``N_SEEDS`` Seeds also deutlich unter einer Minute
Gesamtlaufzeit.
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

# Läuft-Anteile von Arm A (IndexBlockSplitter-Standard: 64 train, 32 val, 32 test von 128)
# ohne den val-Block: 64 : 32 = 2 : 1 -- dieselbe Proportion für Arm B, siehe Moduldocstring.
_POINT_RATIOS = SplitRatios(train=64 / 96, val=0.0, test=32 / 96)


def _spec() -> DataSpec:
    """Kräuters Fenstergeometrie: Wert + Differenzen über drei Schritte, h=1."""
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
        f"Start: Splitvergleich Extra Trees, {len(MASTER_SEEDS)} Seeds, "
        f"n_estimators={N_ESTIMATORS}"
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
        # Explizite Bestandsprüfung, nicht nur Vertrauen in build_datasets() (Befund 36):
        # test_a MUSS denselben Skalierer-Objekt wie train_a tragen (Identität, nicht nur
        # Gleichheit) -- sonst wäre der laufweise Arm nicht mehr skalierer-leck-frei und der
        # Splitvergleich würde zwei verschiedene Leck-Quellen mischen.
        assert test_a.input_scaler is train_a.input_scaler
        assert test_a.target_scaler is train_a.target_scaler
        arms["run_level"] = (train_a, test_a)

        point_splitter = PointSplitter(seed=master_seed, ratios=_POINT_RATIOS)
        train_b, _val_b, test_b = windowed_point_split(
            data.trajectory, pool_run_ids, spec.window, point_splitter
        )
        # Dieselbe Bestandsprüfung für den punktweisen Arm, symmetrisch zu oben.
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

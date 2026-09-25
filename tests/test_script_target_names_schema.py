"""J2: Schemapruefung ``target_names`` in Log-anhaengenden Skripten.

``scripts/`` ist kein Package (kein ``__init__.py``, nicht auf ``sys.path``) -- die
betroffenen Skripte werden deshalb per ``importlib.util.spec_from_file_location`` direkt
aus ihrer Datei geladen, wie an keiner anderen Stelle dieses Projekts. Jeder Test prueft
nur die Schemapruefungs-Funktion selbst (synthetischer Eingang mit abweichenden
``target_names``), nicht den vollen Skriptlauf (kein Netztraining, keine Datenerzeugung).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
HAS_TORCH = importlib.util.find_spec("torch") is not None


def _load_script(name: str) -> types.ModuleType:
    """``scripts/<name>.py`` als eigenstaendiges Modul laden (kein Package-Import moeglich).

    Fuegt ``SCRIPTS_DIR`` einmalig zu ``sys.path`` hinzu: die Constant-Holdout-Worker
    importieren top-level voneinander (``from run_arch_constant_holdout_worker import
    ...``), was ohne ``scripts/`` auf dem Pfad mit ``ModuleNotFoundError`` scheitert.
    """
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_under_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_existing_record(
    path: Path, *, seed: int, target_names: tuple[str, ...]
) -> None:
    """Einen synthetischen Befund-36-artigen Record schreiben (``test_set=excited``)."""
    record = {
        "params": {
            "master_seed": seed,
            "test_set": "excited",
            "process/fingerprint": "82f65f576a2e5253",
            "split/digest": "5b9628a151344f37",
            "seed/data": 57731891,
            "seed/noise": 1868603354,
            "seed/model": 2509151119,
            "data/target_names": list(target_names),
        },
        "metrics": {},
        "tags": {},
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


@pytest.mark.skipif(not HAS_TORCH, reason="Skript importiert torch-abhaengige Module")
def test_run_arch_multipool_eval_rejects_mismatched_checkpoint_spec() -> None:
    module = _load_script("run_arch_multipool_eval")

    reference = module._spec_from_checkpoint(
        {
            "data/sample_time_s": 25.0,
            "data/run_duration_s": 28_800.0,
            "data/n_train_max": 64,
            "data/n_val": 32,
            "data/n_test": 32,
            "data/history_length": 32,
            "data/horizon": 36,
            "data/stride": 8,
            "data/target_names": ("energy_target", "product_moisture_kg_per_kg", "d50_um"),
            "data/observed_target_names": None,
            "data/constant_run_fraction": 0.10,
        }
    )
    candidate = module._spec_from_checkpoint(
        {
            "data/sample_time_s": 25.0,
            "data/run_duration_s": 28_800.0,
            "data/n_train_max": 64,
            "data/n_val": 32,
            "data/n_test": 32,
            "data/history_length": 32,
            "data/horizon": 36,
            "data/stride": 8,
            "data/target_names": ("energy_target", "product_moisture_kg_per_kg"),
            "data/observed_target_names": None,
            "data/constant_run_fraction": 0.10,
        }
    )

    with pytest.raises(module.CheckpointSpecMismatchError):
        module._verify_matching_spec(reference, candidate, checkpoint_path=Path("dummy.pt"))


class _StubEvaluationResult:
    """Ersetzt ``RolloutEvaluator.evaluate(...)``-Rueckgabe: nur ``as_flat_metrics()`` noetig."""

    def as_flat_metrics(self) -> dict[str, float]:
        return {}


class _StubEvaluator:
    """Ersetzt ``RolloutEvaluator``.

    Kein echtes Rollout, nur die Record-Konstruktion in ``_log_result`` wird geprueft.
    """

    def evaluate(self, model: object, dataset: object) -> _StubEvaluationResult:
        del model, dataset
        return _StubEvaluationResult()


@pytest.mark.skipif(not HAS_TORCH, reason="Skript importiert torch-abhaengige Module")
def test_run_arch_multipool_eval_train_constant_run_fraction_reflects_training_not_test_set(
    tmp_path: Path,
) -> None:
    """LA1: ``data/train_constant_run_fraction`` muss die TRAININGS-Spec ausweisen.

    Nicht die Testmenge: ``run_arch_multipool_eval.py`` wertet ausschliesslich gegen
    Konstantlauf-Holdouts aus
    (``test_set`` beginnt immer mit ``"constant_holdout_"``) -- ``excited_spec`` hier traegt
    dennoch ``constant_run_fraction=0.10``, weil sie aus einem angeregt trainierten
    Checkpoint stammt. Genau diese Kombination (angeregt trainiert, auf Konstantlauf
    ausgewertet) ist der Fall, den das generische ``data/constant_run_fraction`` (identisch
    fuer jeden ``test_set``-Wert desselben Records) nicht von einem konstant-trainierten Lauf
    unterscheidbar macht -- das neue Feld muss trotz ``test_set="constant_holdout_999"`` bei
    0.10 bleiben, nicht auf 1.0 kippen.
    """
    module = _load_script("run_arch_multipool_eval")

    excited_spec = module.DataSpec(constant_run_fraction=0.10)
    log_path = tmp_path / "multipool.jsonl"
    logger = module.JsonlExperimentLogger(log_path)

    module._log_result(
        logger,
        _StubEvaluator(),
        object(),
        excited_spec=excited_spec,
        seed=1,
        holdout_seed=999,
        common_extra={},
        label="gru",
        model=object(),
        checkpoint_path=None,
    )
    logger.close()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    params = json.loads(lines[0])["params"]
    assert params["test_set"] == "constant_holdout_999"
    assert params["data/constant_run_fraction"] == 0.10
    assert params["data/train_constant_run_fraction"] == 0.10


@pytest.mark.skipif(not HAS_TORCH, reason="Skript importiert torch-abhaengige Module")
def test_run_mamba_reduced_comparison_rejects_mismatched_target_names() -> None:
    module = _load_script("run_mamba_reduced_comparison")

    actual = {
        "process/fingerprint": module.EXPECTED["process/fingerprint"],
        "split/digest": module.EXPECTED["split/digest"],
        "seed/data": module.EXPECTED["seed/data"],
        "seed/noise": module.EXPECTED["seed/noise"],
        "seed/model": module.EXPECTED["seed/model"],
        "data/target_names": ("energy_target", "product_moisture_kg_per_kg"),
    }
    assert actual["data/target_names"] != module.EXPECTED["data/target_names"]

    with pytest.raises(RuntimeError, match="Zielkanaele"):
        module._check_expected(actual)


@pytest.mark.skipif(not HAS_TORCH, reason="Skript importiert torch-abhaengige Module")
def test_run_xlstm_reduced_comparison_rejects_mismatched_target_names() -> None:
    module = _load_script("run_xlstm_reduced_comparison")

    actual = {
        "process/fingerprint": module.EXPECTED["process/fingerprint"],
        "split/digest": module.EXPECTED["split/digest"],
        "seed/data": module.EXPECTED["seed/data"],
        "seed/noise": module.EXPECTED["seed/noise"],
        "seed/model": module.EXPECTED["seed/model"],
        "data/target_names": ("energy_target", "product_moisture_kg_per_kg"),
    }
    assert actual["data/target_names"] != module.EXPECTED["data/target_names"]

    with pytest.raises(RuntimeError, match="Zielkanaele"):
        module._check_expected(actual)


@pytest.mark.skipif(not HAS_TORCH, reason="Skript importiert torch-abhaengige Module")
def test_study_xlstm_wide_rejects_mismatched_target_names(tmp_path: Path) -> None:
    module = _load_script("study_xlstm_wide")

    log_path = tmp_path / "existing_log.jsonl"
    existing_record = {
        "params": {
            "data/target_names": ["energy_target", "product_moisture_kg_per_kg", "d50_um"]
        },
        "metrics": {},
        "tags": {},
    }
    log_path.write_text(json.dumps(existing_record) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="uneinheitlicher"):
        module._check_target_names_consistency(
            log_path, ("energy_target", "product_moisture_kg_per_kg")
        )


@pytest.mark.skipif(not HAS_TORCH, reason="Skript importiert torch-abhaengige Module")
def test_study_xlstm_wide_accepts_matching_or_missing_log(tmp_path: Path) -> None:
    module = _load_script("study_xlstm_wide")

    missing_log = tmp_path / "does_not_exist.jsonl"
    module._check_target_names_consistency(
        missing_log, ("energy_target", "product_moisture_kg_per_kg")
    )

    matching_log = tmp_path / "matching_log.jsonl"
    matching_record = {
        "params": {"data/target_names": ["energy_target", "product_moisture_kg_per_kg"]},
        "metrics": {},
        "tags": {},
    }
    matching_log.write_text(json.dumps(matching_record) + "\n", encoding="utf-8")
    module._check_target_names_consistency(
        matching_log, ("energy_target", "product_moisture_kg_per_kg")
    )


@pytest.mark.skipif(not HAS_TORCH, reason="Skript importiert torch-abhaengige Module")
def test_run_mamba_constant_holdout_worker_rejects_mismatched_target_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("run_mamba_constant_holdout_worker")

    existing_path = tmp_path / "arch_gab_constant_holdout.jsonl"
    _write_existing_record(
        existing_path,
        seed=1,
        target_names=("energy_target", "product_moisture_kg_per_kg", "d50_um"),
    )
    monkeypatch.setattr(module, "EXISTING_RECORDS_PATH", existing_path)

    expected = module._expected_from_existing(1)
    actual = {
        **{k: v for k, v in expected.items() if k != "data/target_names"},
        "data/target_names": ("energy_target", "product_moisture_kg_per_kg"),
    }

    with pytest.raises(RuntimeError, match="Zielkanaele"):
        module._check_expected(1, expected, actual)


@pytest.mark.skipif(not HAS_TORCH, reason="Skript importiert torch-abhaengige Module")
def test_run_mamba_constant_holdout_worker_accepts_matching_target_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("run_mamba_constant_holdout_worker")

    existing_path = tmp_path / "arch_gab_constant_holdout.jsonl"
    _write_existing_record(
        existing_path, seed=1, target_names=("energy_target", "product_moisture_kg_per_kg")
    )
    monkeypatch.setattr(module, "EXISTING_RECORDS_PATH", existing_path)

    expected = module._expected_from_existing(1)
    module._check_expected(1, expected, dict(expected))  # keine Exception


@pytest.mark.skipif(not HAS_TORCH, reason="Skript importiert torch-abhaengige Module")
def test_run_xlstm_constant_holdout_worker_rejects_mismatched_target_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("run_xlstm_constant_holdout_worker")

    existing_path = tmp_path / "arch_gab_constant_holdout.jsonl"
    _write_existing_record(
        existing_path,
        seed=1,
        target_names=("energy_target", "product_moisture_kg_per_kg", "d50_um"),
    )
    monkeypatch.setattr(module, "EXISTING_RECORDS_PATH", existing_path)

    expected = module._expected_from_existing(1)
    actual = {
        **{k: v for k, v in expected.items() if k != "data/target_names"},
        "data/target_names": ("energy_target", "product_moisture_kg_per_kg"),
    }

    with pytest.raises(RuntimeError, match="Zielkanaele"):
        module._check_expected(1, expected, actual)


@pytest.mark.skipif(not HAS_TORCH, reason="Skript importiert torch-abhaengige Module")
def test_run_xlstm_constant_holdout_worker_accepts_matching_target_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("run_xlstm_constant_holdout_worker")

    existing_path = tmp_path / "arch_gab_constant_holdout.jsonl"
    _write_existing_record(
        existing_path, seed=1, target_names=("energy_target", "product_moisture_kg_per_kg")
    )
    monkeypatch.setattr(module, "EXISTING_RECORDS_PATH", existing_path)

    expected = module._expected_from_existing(1)
    module._check_expected(1, expected, dict(expected))  # keine Exception

"""J2: schema check of ``target_names`` in log-appending scripts.

``scripts/`` is not a package (no ``__init__.py``, not on ``sys.path``) -- the
affected scripts are therefore loaded directly from their file via
``importlib.util.spec_from_file_location``, as nowhere else in this project. Each test checks
only the schema check function itself (synthetic input with deviating
``target_names``), not the full script run (no network training, no data generation).
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
    """Load ``scripts/<name>.py`` as a standalone module (no package import possible).

    Adds ``SCRIPTS_DIR`` to ``sys.path`` once: the constant-holdout workers
    import each other at top level (``from run_arch_constant_holdout_worker import
    ...``), which fails with ``ModuleNotFoundError`` without ``scripts/`` on the path.
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


def _write_existing_record(path: Path, *, seed: int, target_names: tuple[str, ...]) -> None:
    """Write a synthetic Befund-36-like record (``test_set=excited``)."""
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
    """Replaces the ``RolloutEvaluator.evaluate(...)`` return: only ``as_flat_metrics()`` needed."""

    def as_flat_metrics(self) -> dict[str, float]:
        return {}


class _StubEvaluator:
    """Replaces ``RolloutEvaluator``.

    No real rollout, only the record construction in ``_log_result`` is checked.
    """

    def evaluate(self, model: object, dataset: object) -> _StubEvaluationResult:
        del model, dataset
        return _StubEvaluationResult()


@pytest.mark.skipif(not HAS_TORCH, reason="Skript importiert torch-abhaengige Module")
def test_run_arch_multipool_eval_train_constant_run_fraction_reflects_training_not_test_set(
    tmp_path: Path,
) -> None:
    """LA1: ``data/train_constant_run_fraction`` must report the TRAINING spec.

    Not the test set: ``run_arch_multipool_eval.py`` evaluates exclusively against
    constant-run holdouts
    (``test_set`` always starts with ``"constant_holdout_"``) -- ``excited_spec`` here nevertheless
    carries ``constant_run_fraction=0.10``, because it comes from a checkpoint trained on
    excited data. Exactly this combination (trained on excited data, evaluated on constant runs)
    is the case that the generic ``data/constant_run_fraction`` (identical
    for every ``test_set`` value of the same record) does not make distinguishable from a
    constant-trained run -- the new field must stay at 0.10 despite
    ``test_set="constant_holdout_999"``, not flip to 1.0.
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
        "params": {"data/target_names": ["energy_target", "product_moisture_kg_per_kg", "d50_um"]},
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
    module._check_expected(1, expected, dict(expected))  # no exception


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
    module._check_expected(1, expected, dict(expected))  # no exception

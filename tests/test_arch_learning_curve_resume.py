"""OA2b: Fortsetzbarkeit von ``run_arch_learning_curve_sweep.py``.

``scripts/`` ist kein Package -- Laden per ``importlib.util.spec_from_file_location``,
wie in ``test_script_target_names_schema.py``.
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
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_under_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(*, seed: int, n_train: int, label: str, test_set: str, commit: str) -> dict:
    return {
        "run_uid": f"{label}-{seed}-{n_train}-{test_set}",
        "params": {
            "master_seed": seed,
            "n_train": n_train,
            "model/label": label,
            "test_set": test_set,
            "git/commit": commit,
            "git/commit_available": True,
        },
        "metrics": {},
        "tags": {},
    }


@pytest.mark.skipif(not HAS_TORCH, reason="Skript importiert torch-abhaengige Module")
def test_read_existing_records_tolerates_truncated_last_line(tmp_path: Path) -> None:
    module = _load_script("run_arch_learning_curve_sweep")

    log_path = tmp_path / "log.jsonl"
    complete = _record(seed=1, n_train=8, label="gru", test_set="excited", commit="abc123")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(complete) + "\n")
        handle.write('{"run_uid": "broken", "params": {"master_se')  # abgeschnitten, kein \n

    records, keys, commit = module._read_existing_records(log_path)

    assert records == [complete]
    assert keys == {(1, 8, "gru", "excited")}
    assert commit == "abc123"


@pytest.mark.skipif(not HAS_TORCH, reason="Skript importiert torch-abhaengige Module")
def test_read_existing_records_raises_on_non_trailing_corruption(tmp_path: Path) -> None:
    module = _load_script("run_arch_learning_curve_sweep")

    log_path = tmp_path / "log.jsonl"
    complete = _record(seed=1, n_train=8, label="gru", test_set="excited", commit="abc123")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write('{"run_uid": "broken", "params": {"master_se\n')  # NICHT die letzte Zeile
        handle.write(json.dumps(complete) + "\n")

    with pytest.raises(ValueError, match="echter Datei-Defekt"):
        module._read_existing_records(log_path)


@pytest.mark.skipif(not HAS_TORCH, reason="Skript importiert torch-abhaengige Module")
def test_resume_skips_exactly_the_complete_seed(tmp_path: Path) -> None:
    module = _load_script("run_arch_learning_curve_sweep")

    labels = module._expected_labels(None, skip_baselines=False)
    sizes = [8, 16, 32, 64]
    test_sets = module.TEST_SETS

    log_path = tmp_path / "log.jsonl"
    records = [
        _record(seed=1, n_train=n_train, label=label, test_set=test_set, commit="deadbeef")
        for n_train in sizes
        for label in labels
        for test_set in test_sets
    ]
    # Seed 2 bleibt absichtlich unvollstaendig (ein Record fehlt).
    records += [
        _record(seed=2, n_train=n_train, label=label, test_set=test_set, commit="deadbeef")
        for n_train in sizes
        for label in labels
        for test_set in test_sets
    ][:-1]
    with log_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    _existing_records, existing_keys, existing_commit = module._read_existing_records(log_path)

    assert existing_commit == "deadbeef"
    seeds = [1, 2]
    complete_seeds = [
        seed for seed in seeds if module._expected_keys(seed, sizes, labels) <= existing_keys
    ]
    assert complete_seeds == [1]


@pytest.mark.skipif(not HAS_TORCH, reason="Skript importiert torch-abhaengige Module")
def test_expected_labels_reflects_architecture_selection() -> None:
    """Auswahl per --architectures/--skip-baselines muss in den Auftragsschluessel einfliessen.

    Sonst haelt der Resume-Check einen absichtlich schlanken Lauf (z. B. nur die drei
    neuen Architekturen, ohne GRU/LSTM/Baselines erneut zu rechnen) faelschlich fuer
    unvollstaendig oder eine echte Luecke faelschlich fuer vollstaendig.
    """
    module = _load_script("run_arch_learning_curve_sweep")

    all_labels = module._expected_labels(None, skip_baselines=False)
    assert set(all_labels) == {
        "persistence",
        "static",
        "arx",
        "gru",
        "lstm",
        "transformer",
        "xlstm",
        "mamba",
    }

    only_new = module._expected_labels(
        ["transformer", "xlstm", "mamba"], skip_baselines=True
    )
    assert set(only_new) == {"transformer", "xlstm", "mamba"}

    new_with_baselines = module._expected_labels(["xlstm"], skip_baselines=False)
    assert set(new_with_baselines) == {"persistence", "static", "arx", "xlstm"}


@pytest.mark.skipif(not HAS_TORCH, reason="Skript importiert torch-abhaengige Module")
def test_worker_architectures_flag_parses_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--architectures``/``--skip-baselines`` muessen ankommen und korrekt geparst werden.

    ``main()`` liest sie direkt aus ``arguments`` (``for architecture in
    arguments.architectures``, ``if not arguments.skip_baselines``), ein Parse-Fehler
    waere hier schon sichtbar.
    """
    module = _load_script("run_arch_learning_curve_worker")

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_arch_learning_curve_worker.py",
            "--seed",
            "1",
            "--output",
            "unused.jsonl",
            "--architectures",
            "xlstm",
            "mamba",
            "--skip-baselines",
        ],
    )
    arguments = module.parse_arguments()
    assert arguments.architectures == ["xlstm", "mamba"]
    assert arguments.skip_baselines is True

    monkeypatch.setattr(
        "sys.argv",
        ["run_arch_learning_curve_worker.py", "--seed", "1", "--output", "unused.jsonl"],
    )
    default_arguments = module.parse_arguments()
    assert default_arguments.architectures == list(module.ARCHITECTURE_LABELS)
    assert default_arguments.skip_baselines is False


@pytest.mark.skipif(not HAS_TORCH, reason="Skript importiert torch-abhaengige Module")
def test_provenance_guard_rejects_commit_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script("run_arch_learning_curve_sweep")

    monkeypatch.setattr(
        module,
        "environment_fingerprint",
        lambda: {"git/dirty": False, "git/commit": "new-head"},
    )

    with pytest.raises(RuntimeError, match="new-head"):
        module._check_provenance_for_resume("old-commit", allow_mismatch=False)

    # Mit explizitem Override keine Exception.
    module._check_provenance_for_resume("old-commit", allow_mismatch=True)


@pytest.mark.skipif(not HAS_TORCH, reason="Skript importiert torch-abhaengige Module")
def test_provenance_guard_rejects_dirty_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script("run_arch_learning_curve_sweep")

    monkeypatch.setattr(
        module,
        "environment_fingerprint",
        lambda: {"git/dirty": True, "git/commit": "whatever"},
    )

    with pytest.raises(RuntimeError, match="dirty"):
        module._check_provenance_for_resume(None, allow_mismatch=False)

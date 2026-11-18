"""Model bundle of :mod:`twinlink_ml.api`: round trip, integrity, no paths, equal model."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

from _api_common import N_TRAIN, SAMPLE_TIME_S, datasets, generated
from twinlink_ml.api import BundleIntegrityError, ModelBundle, build_bundle, fit_bundle, load_bundle
from twinlink_ml.api.base import LAB_LINE_SCOPE as LAB
from twinlink_ml.api.bundle import (
    BUNDLE_FORMAT_VERSION,
    MANIFEST_NAME,
    WEIGHTS_NAME,
    BundledARXModel,
    instantiate_model,
    weights_digest,
)
from twinlink_ml.evaluation.experiments import NEW_STUDY_PRESET_VERSION
from twinlink_ml.models import LinearARXModel, PersistenceModel

HAS_TORCH = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def arx_bundle() -> ModelBundle:
    return fit_bundle(generated(), n_train=N_TRAIN, model_seed=3, kind="linear_arx")


def test_manifest_describes_data_and_code(arx_bundle: ModelBundle) -> None:
    manifest = arx_bundle.manifest
    data = generated()
    assert manifest.bundle_format_version == BUNDLE_FORMAT_VERSION
    assert manifest.model_kind == "linear_arx"
    assert manifest.sample_time_s == SAMPLE_TIME_S
    assert manifest.model_scope == LAB == "lab line, grey-box"
    assert manifest.window.future_actions == "planned"
    assert manifest.data_spec["data/new_study_version"] == NEW_STUDY_PRESET_VERSION
    assert manifest.data_fingerprints == data.fingerprints
    assert manifest.parameter_status == data.process_config.provenance.status == "placeholder"
    assert manifest.training["n_train_runs"] == N_TRAIN
    assert manifest.code.root_checked is True
    assert manifest.code.commit is not None
    assert manifest.code.preset_version == NEW_STUDY_PRESET_VERSION
    assert set(manifest.input_ranges) == {*manifest.exogenous_names, *manifest.target_names}
    low, high = (
        manifest.input_ranges["inlet_temperature_celsius"].minimum,
        manifest.input_ranges["inlet_temperature_celsius"].maximum,
    )
    box = data.process_config.actions.inlet_temperature_celsius
    assert box.minimum <= low < high <= box.maximum


def test_manifest_contains_no_local_path(arx_bundle: ModelBundle) -> None:
    """No absolute path, interpreter path or user directory leaves the machine with a bundle."""
    text = json.dumps(arx_bundle.manifest.model_dump(mode="json"))
    assert str(Path.home()) not in text
    assert Path.home().name not in text
    assert str(ROOT) not in text and ROOT.as_posix() not in text
    assert re.search(r"[A-Za-z]:[\\/]", text) is None
    assert "\\\\" not in text and "/home/" not in text and "/Users/" not in text
    assert "package_dir" not in text and "interpreter" not in text


def test_round_trip_as_directory_and_archive(arx_bundle: ModelBundle, tmp_path: Path) -> None:
    directory = arx_bundle.save(tmp_path / "bundle")
    archive = arx_bundle.save_archive(tmp_path / "bundle.zip")
    assert sorted(p.name for p in directory.iterdir()) == [MANIFEST_NAME, WEIGHTS_NAME]
    for path in (directory, archive):
        loaded = load_bundle(path, expected_fingerprint=arx_bundle.fingerprint)
        assert loaded.manifest == arx_bundle.manifest
        assert weights_digest(loaded.weights) == arx_bundle.manifest.weights_sha256
        assert all(
            np.array_equal(loaded.weights[name], array)
            for name, array in arx_bundle.weights.items()
        )


def test_weights_are_plain_arrays_without_pickle(arx_bundle: ModelBundle, tmp_path: Path) -> None:
    directory = arx_bundle.save(tmp_path / "bundle")
    with np.load(directory / WEIGHTS_NAME, allow_pickle=False) as archive:
        assert archive.files
        assert all(archive[name].dtype.kind in "fiu" for name in archive.files)


def test_save_never_overwrites(arx_bundle: ModelBundle, tmp_path: Path) -> None:
    arx_bundle.save(tmp_path / "bundle")
    arx_bundle.save_archive(tmp_path / "bundle.zip")
    with pytest.raises(FileExistsError):
        arx_bundle.save(tmp_path / "bundle")
    with pytest.raises(FileExistsError):
        arx_bundle.save_archive(tmp_path / "bundle.zip")
    with pytest.raises(ValueError, match=r"\.zip"):
        arx_bundle.save_archive(tmp_path / "bundle.tar")


def _edit_manifest(directory: Path, **changes: object) -> None:
    path = directory / MANIFEST_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_wrong_fingerprint_is_rejected(arx_bundle: ModelBundle, tmp_path: Path) -> None:
    """A bundle whose stored fingerprint is not the hash of its content is not loaded."""
    directory = arx_bundle.save(tmp_path / "bundle")
    _edit_manifest(directory, fingerprint="0" * 64)
    with pytest.raises(BundleIntegrityError, match="manifest was altered") as caught:
        load_bundle(directory)
    assert caught.value.code == "bundle_integrity"


def test_altered_manifest_is_rejected(arx_bundle: ModelBundle, tmp_path: Path) -> None:
    directory = arx_bundle.save(tmp_path / "bundle")
    _edit_manifest(directory, sample_time_s=60.0)
    with pytest.raises(BundleIntegrityError, match="manifest was altered"):
        load_bundle(directory)


def test_altered_weights_are_rejected(arx_bundle: ModelBundle, tmp_path: Path) -> None:
    directory = arx_bundle.save(tmp_path / "bundle")
    weights = {name: array.copy() for name, array in arx_bundle.weights.items()}
    name = next(iter(weights))
    weights[name].flat[0] = np.nextafter(weights[name].flat[0], np.inf)  # one bit
    np.savez(directory / WEIGHTS_NAME, **weights)
    with pytest.raises(BundleIntegrityError, match="weights do not match"):
        load_bundle(directory)


def test_unexpected_bundle_is_rejected(arx_bundle: ModelBundle, tmp_path: Path) -> None:
    directory = arx_bundle.save(tmp_path / "bundle")
    with pytest.raises(BundleIntegrityError, match="expected the bundle"):
        load_bundle(directory, expected_fingerprint="f" * 64)


def test_incomplete_or_foreign_files_are_rejected(arx_bundle: ModelBundle, tmp_path: Path) -> None:
    directory = arx_bundle.save(tmp_path / "bundle")
    (directory / WEIGHTS_NAME).unlink()
    with pytest.raises(BundleIntegrityError, match="lacks"):
        load_bundle(directory)
    foreign = tmp_path / "foreign.zip"
    foreign.write_bytes(b"not a zip archive")
    with pytest.raises(BundleIntegrityError, match="not a bundle archive"):
        load_bundle(foreign)
    with pytest.raises(BundleIntegrityError, match="neither"):
        load_bundle(tmp_path / "missing")


def test_unknown_format_version_is_rejected(arx_bundle: ModelBundle) -> None:
    newer = arx_bundle.manifest.model_copy(update={"bundle_format_version": 99})
    with pytest.raises(BundleIntegrityError, match="format 99"):
        ModelBundle(manifest=newer, weights=arx_bundle.weights).verify()


def test_restored_arx_predicts_like_the_research_model(
    arx_bundle: ModelBundle, tmp_path: Path
) -> None:
    """Bit for bit: bundle -> disk -> model against ``LinearARXModel`` fitted on the same runs."""
    train, _, test = datasets()
    research = LinearARXModel()
    research.fit(train, seed=0)
    restored = instantiate_model(load_bundle(arx_bundle.save(tmp_path / "bundle")))
    assert isinstance(restored, BundledARXModel)
    batch = test.full_batch()
    assert np.array_equal(restored.predict(batch), research.predict(batch))
    assert restored.name == research.name
    assert restored.n_parameters == research.n_parameters


def test_a_fitted_research_arx_can_be_bundled() -> None:
    train, _, test = datasets()
    research = LinearARXModel(n_lags_target=3, n_lags_exogenous=1)
    research.fit(train, seed=0)
    bundle = build_bundle(
        research,
        train,
        sample_time_s=SAMPLE_TIME_S,
        parameter_status="placeholder",
        model_scope=LAB,
    )
    assert bundle.manifest.model_settings == {
        "n_lags_target": 3,
        "n_lags_exogenous": 1,
        "ridge": 1e-4,
    }
    batch = test.full_batch()
    assert np.array_equal(instantiate_model(bundle).predict(batch), research.predict(batch))


def test_unsupported_model_is_rejected() -> None:
    train, _, _ = datasets()
    model = PersistenceModel()
    model.fit(train, seed=0)
    with pytest.raises(TypeError, match="cannot be bundled"):
        build_bundle(
            model,
            train,
            sample_time_s=SAMPLE_TIME_S,
            parameter_status="placeholder",
            model_scope=LAB,
        )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_restored_network_predicts_like_the_trained_one(tmp_path: Path) -> None:
    """Bit for bit: the composite GRU after bundle -> disk -> model against the trained object."""
    from twinlink_ml.models.composite import CompositeGruSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, validation, test = datasets()
    trained = CompositeGruSequenceModel(
        TorchSequenceConfig(core="gru", hidden_size=8, max_epochs=2, patience=0, device="cpu")
    )
    trained.fit(train, seed=5, validation=validation)
    bundle = build_bundle(
        trained, train, sample_time_s=SAMPLE_TIME_S, parameter_status="placeholder", model_scope=LAB
    )
    assert bundle.manifest.model_kind == "composite_torch"
    assert bundle.manifest.model_label == trained.name
    assert all(array.dtype == np.float32 for array in bundle.weights.values())
    restored = instantiate_model(load_bundle(bundle.save_archive(tmp_path / "gru.zip")))
    batch = test.full_batch()
    assert np.array_equal(restored.predict(batch), trained.predict(batch))
    assert restored.n_parameters == trained.n_parameters

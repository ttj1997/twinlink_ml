r"""Model bundle: a trained forecast model as a transferable package.

Format (version 1)
------------------
A directory, or a ``.zip`` archive of it, with exactly two files:

``manifest.json``
    :class:`BundleManifest`: kind and settings of the model, channel names, window, sample time,
    both scalers, the ranges of the training inputs, data specification and data fingerprints,
    status of the process parameters, code commit and versions, the hash of the weights and the
    fingerprint of the bundle.
``weights.npz``
    The weights as plain NumPy arrays (``numpy.savez``), read with ``allow_pickle=False``.

Why not the checkpoint format of ``twinlink_ml.utils.checkpoints``: a checkpoint is a pickled
``torch.save`` dictionary, read with ``weights_only=False``. That is acceptable for a file that
never leaves the machine that wrote it and not for a package handed to another system. A bundle
contains no pickled object: arrays and JSON only.

Fingerprints
------------
``weights_sha256`` is computed over names, dtypes, shapes and bytes of the arrays in sorted order,
not over the archive file (an archive carries time stamps). ``fingerprint`` is the SHA-256 of the
canonical JSON of the manifest without the field ``fingerprint`` itself, and therefore covers the
weights hash, the scalers, the settings and the provenance. :func:`load_bundle` recomputes both
and rejects a bundle in which either differs; a caller can in addition pin the fingerprint it
expects.

Provenance without paths
------------------------
A bundle records the code commit, the dirty flag, the versions and whether the code location was
checked, but **no** absolute path and no interpreter path (the fields ``code/package_dir`` and
``code/interpreter`` of other artifacts): those name a local user directory and the bundle leaves
the machine. This is an exception to the provenance rule of ``CLAUDE.md``, decided by the project
lead on 2026-11-18. The check itself is not weakened: building a bundle calls
``environment_fingerprint()``, which aborts when the loaded package is not the one of the checkout.

Research code
-------------
Only imported. The composite network is exported and restored through its public
``state_dict`` / ``load_checkpoint``; the linear ARX has no such pair, so
:class:`BundledARXModel` sub-classes it and adds one (a test pins that it predicts bit for bit like
the research class).
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import JsonValue, ValidationError

from ..data.windows import SequenceDataset, Standardizer
from ..models.base import SequenceModel
from ..models.baselines import LinearARXModel
from ..simulation.backend import FloatArray
from .base import (
    ApiModel,
    BundleIntegrityError,
    ParameterStatus,
    ValueRange,
    require_extra,
)

if TYPE_CHECKING:
    from ..evaluation.experiments import GeneratedData
    from ..models.torch_sequence import TorchSequenceConfig

__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "MANIFEST_NAME",
    "WEIGHTS_NAME",
    "BundleManifest",
    "BundledARXModel",
    "CodeProvenance",
    "ModelBundle",
    "ModelKind",
    "ScalerPayload",
    "WindowPayload",
    "build_bundle",
    "fit_bundle",
    "instantiate_model",
    "load_bundle",
    "weights_digest",
]

BUNDLE_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
WEIGHTS_NAME = "weights.npz"

ModelKind = Literal["composite_torch", "linear_arx"]

_SUBMODEL_SEPARATOR = "/"
_ARX_COEFFICIENTS = "coefficients"


# ---------------------------------------------------------------------- manifest


class ScalerPayload(ApiModel):
    """Channel-wise standardisation: mean and scale per channel."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]

    def to_standardizer(self) -> Standardizer:
        """The scaler of the research code with exactly these numbers."""
        return Standardizer(
            mean=np.asarray(self.mean, dtype=np.float64),
            scale=np.asarray(self.scale, dtype=np.float64),
        )

    @classmethod
    def from_standardizer(cls, scaler: Standardizer) -> ScalerPayload:
        """Store a scaler; the floats survive the JSON round trip bit for bit."""
        return cls(mean=tuple(scaler.mean.tolist()), scale=tuple(scaler.scale.tolist()))


class WindowPayload(ApiModel):
    """Window the model was trained with."""

    history_length: int
    horizon: int
    future_actions: Literal["realised", "planned"]


class CodeProvenance(ApiModel):
    """Where the code of a bundle came from. No paths, see the module docstring."""

    commit: str | None
    dirty: bool
    diff_digest: str | None
    root_checked: bool
    package_version: str
    python: str
    numpy: str
    torch: str | None
    preset_version: int


class BundleManifest(ApiModel):
    """Everything needed to rebuild the model and to check where it came from."""

    bundle_format_version: int
    model_kind: ModelKind
    model_label: str
    model_settings: dict[str, JsonValue]
    exogenous_names: tuple[str, ...]
    target_names: tuple[str, ...]
    observed_target_names: tuple[str, ...]
    measured_input_names: tuple[str, ...]
    window: WindowPayload
    sample_time_s: float
    input_scaler: ScalerPayload
    target_scaler: ScalerPayload
    input_ranges: dict[str, ValueRange]
    data_spec: dict[str, JsonValue]
    data_fingerprints: dict[str, str]
    parameter_status: ParameterStatus
    training: dict[str, JsonValue]
    code: CodeProvenance
    weights_sha256: str
    fingerprint: str

    @property
    def unobserved_target_names(self) -> tuple[str, ...]:
        """Predicted quantities without a measured history (the soft-sensor task)."""
        observed = set(self.observed_target_names)
        return tuple(name for name in self.target_names if name not in observed)


def weights_digest(weights: Mapping[str, np.ndarray]) -> str:
    """SHA-256 over names, dtypes, shapes and bytes of the arrays, in sorted order of the names."""
    digest = hashlib.sha256()
    for name in sorted(weights):
        array = np.ascontiguousarray(weights[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\x00")
        digest.update(",".join(str(n) for n in array.shape).encode("ascii"))
        digest.update(b"\x00")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _manifest_fingerprint(payload: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a manifest without its own ``fingerprint`` field."""
    body = {key: value for key, value in payload.items() if key != "fingerprint"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _jsonable(value: Any) -> JsonValue:
    """Tuples to lists, NumPy scalars to Python numbers, recursively."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if value is None or isinstance(value, str | bool | int | float):
        return value
    return str(value)


# ------------------------------------------------------------------------ bundle


@dataclass(frozen=True)
class ModelBundle:
    """A manifest and the weights it describes."""

    manifest: BundleManifest
    weights: Mapping[str, np.ndarray]

    @property
    def fingerprint(self) -> str:
        """Fingerprint of the bundle."""
        return self.manifest.fingerprint

    def verify(self) -> None:
        """Recompute both hashes; raise :class:`BundleIntegrityError` on any difference."""
        manifest = self.manifest
        if manifest.bundle_format_version != BUNDLE_FORMAT_VERSION:
            raise BundleIntegrityError(
                f"bundle format {manifest.bundle_format_version} is not supported "
                f"(this code reads format {BUNDLE_FORMAT_VERSION})"
            )
        actual_weights = weights_digest(self.weights)
        if actual_weights != manifest.weights_sha256:
            raise BundleIntegrityError(
                "the weights do not match the manifest: weights_sha256 is "
                f"{manifest.weights_sha256}, the arrays hash to {actual_weights}"
            )
        actual = _manifest_fingerprint(manifest.model_dump(mode="json"))
        if actual != manifest.fingerprint:
            raise BundleIntegrityError(
                f"the manifest was altered: fingerprint is {manifest.fingerprint}, "
                f"the content hashes to {actual}"
            )

    def _manifest_text(self) -> str:
        return json.dumps(self.manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"

    def _weights_bytes(self) -> bytes:
        buffer = io.BytesIO()
        arrays = {name: self.weights[name] for name in sorted(self.weights)}
        np.savez(buffer, **arrays)  # type: ignore[arg-type]  # numpy stub: keywords typed as bool
        return buffer.getvalue()

    def save(self, directory: Path | str) -> Path:
        """Write the bundle as a directory; an existing, non-empty one is never overwritten."""
        target = Path(directory)
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise FileExistsError(f"{target} exists and is not an empty directory")
        target.mkdir(parents=True, exist_ok=True)
        (target / MANIFEST_NAME).write_text(self._manifest_text(), encoding="utf-8")
        (target / WEIGHTS_NAME).write_bytes(self._weights_bytes())
        return target

    def save_archive(self, path: Path | str) -> Path:
        """Write the bundle as one ``.zip`` archive. An existing file is never overwritten."""
        target = Path(path)
        if target.suffix != ".zip":
            raise ValueError("a bundle archive is a .zip file")
        if target.exists():
            raise FileExistsError(f"{target} exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(MANIFEST_NAME, self._manifest_text())
            archive.writestr(WEIGHTS_NAME, self._weights_bytes())
        return target


def _read_members(path: Path) -> tuple[bytes, bytes]:
    """Bytes of the manifest and of the weights from a directory or a ``.zip`` archive.

    From an archive only the two named members are read into memory; nothing is extracted.
    """
    if path.is_dir():
        manifest_path, weights_path = path / MANIFEST_NAME, path / WEIGHTS_NAME
        if not manifest_path.is_file() or not weights_path.is_file():
            raise BundleIntegrityError(f"{path} lacks {MANIFEST_NAME} or {WEIGHTS_NAME}")
        return manifest_path.read_bytes(), weights_path.read_bytes()
    if path.is_file() and path.suffix == ".zip":
        try:
            with zipfile.ZipFile(path) as archive:
                return archive.read(MANIFEST_NAME), archive.read(WEIGHTS_NAME)
        except (KeyError, zipfile.BadZipFile) as error:
            raise BundleIntegrityError(f"{path} is not a bundle archive: {error}") from error
    raise BundleIntegrityError(f"{path} is neither a bundle directory nor a .zip archive")


def load_bundle(path: Path | str, *, expected_fingerprint: str | None = None) -> ModelBundle:
    """Read a bundle and check it.

    Raises :class:`BundleIntegrityError` when the files are incomplete, the manifest does not
    follow the format, a hash differs, or the fingerprint is not ``expected_fingerprint``.
    """
    manifest_bytes, weights_bytes = _read_members(Path(path))
    try:
        manifest = BundleManifest.model_validate_json(manifest_bytes)
    except ValidationError as error:
        raise BundleIntegrityError(f"the manifest does not follow the format: {error}") from error
    try:
        with np.load(io.BytesIO(weights_bytes), allow_pickle=False) as archive:
            weights = {name: np.array(archive[name]) for name in archive.files}
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        raise BundleIntegrityError(f"the weights cannot be read: {error}") from error
    bundle = ModelBundle(manifest=manifest, weights=weights)
    bundle.verify()
    if expected_fingerprint is not None and manifest.fingerprint != expected_fingerprint:
        raise BundleIntegrityError(
            f"expected the bundle {expected_fingerprint}, found {manifest.fingerprint}"
        )
    return bundle


# --------------------------------------------------------------------- ARX state


class BundledARXModel(LinearARXModel):
    """The linear ARX of the research code with an export and a restore of its fitted state.

    Adds no behaviour: ``fit`` and ``predict`` are inherited unchanged.
    """

    @classmethod
    def adopt(cls, model: LinearARXModel) -> BundledARXModel:
        """A copy of a fitted research model that can be exported."""
        settings = model.describe()
        adopted = cls(
            n_lags_target=int(settings["n_lags_target"]),
            n_lags_exogenous=int(settings["n_lags_exogenous"]),
            ridge=float(settings["ridge"]),
        )
        adopted._coefficients = None if model._coefficients is None else model._coefficients.copy()
        adopted._autoregressive_index = tuple(model._autoregressive_index)
        adopted._n_exogenous = model._n_exogenous
        return adopted

    @property
    def coefficients(self) -> FloatArray:
        """The fitted coefficient matrix."""
        if self._coefficients is None:
            raise RuntimeError("the model is not fitted")
        return self._coefficients

    @classmethod
    def from_state(
        cls,
        *,
        n_lags_target: int,
        n_lags_exogenous: int,
        ridge: float,
        coefficients: FloatArray,
        n_exogenous: int,
        n_targets: int,
    ) -> BundledARXModel:
        """Rebuild a fitted model from its coefficients, without training."""
        model = cls(n_lags_target=n_lags_target, n_lags_exogenous=n_lags_exogenous, ridge=ridge)
        model._coefficients = np.asarray(coefficients, dtype=np.float64)
        model._autoregressive_index = tuple(n_exogenous + i for i in range(n_targets))
        model._n_exogenous = n_exogenous
        return model


# ------------------------------------------------------------------------- build


def _code_provenance() -> CodeProvenance:
    """Commit, dirty flag and versions; aborts when the loaded package is not the checkout's."""
    from ..utils.tracking import environment_fingerprint

    environment = environment_fingerprint()
    try:
        package_version = metadata.version("twinlink-ml")
    except metadata.PackageNotFoundError:
        package_version = "unknown"
    torch_version = environment.get("version/torch")
    return CodeProvenance(
        commit=environment["git/commit"],
        dirty=bool(environment["git/dirty"]),
        diff_digest=environment["git/diff_digest"],
        root_checked=bool(environment["code/root_checked"]),
        package_version=package_version,
        python=str(environment["python"]),
        numpy=str(environment["version/numpy"]),
        torch=None if torch_version is None else str(torch_version),
        preset_version=int(environment["preset/new_study_version"]),
    )


def _input_ranges(train: SequenceDataset) -> dict[str, ValueRange]:
    """Smallest and largest training value of every input and of every measured target."""
    batch = train.full_batch()
    names = train.exogenous_names
    n_exogenous = len(names)
    exogenous_scaler = train.input_scaler.select(range(n_exogenous))
    history = exogenous_scaler.inverse(batch.history_inputs[..., :n_exogenous])
    future = exogenous_scaler.inverse(batch.future_exogenous)
    values = np.concatenate(
        [history.reshape(-1, n_exogenous), future.reshape(-1, n_exogenous)], axis=0
    )
    ranges = {
        name: ValueRange(minimum=float(values[:, i].min()), maximum=float(values[:, i].max()))
        for i, name in enumerate(names)
    }
    observed = set(train.observed_target_names)
    for position, name in zip(train.autoregressive_index, train.target_names, strict=True):
        if name not in observed:
            continue
        scaler = train.input_scaler.select([position])
        measured = scaler.inverse(batch.history_inputs[..., [position]])
        ranges[name] = ValueRange(minimum=float(measured.min()), maximum=float(measured.max()))
    return ranges


def _export_model(
    model: SequenceModel, train: SequenceDataset
) -> tuple[ModelKind, dict[str, JsonValue], dict[str, np.ndarray]]:
    """Kind, settings and weights of a fitted model."""
    if isinstance(model, LinearARXModel):
        exportable = model if isinstance(model, BundledARXModel) else BundledARXModel.adopt(model)
        settings = exportable.describe()
        arx_settings: dict[str, JsonValue] = {
            "n_lags_target": int(settings["n_lags_target"]),
            "n_lags_exogenous": int(settings["n_lags_exogenous"]),
            "ridge": float(settings["ridge"]),
        }
        return "linear_arx", arx_settings, {_ARX_COEFFICIENTS: exportable.coefficients.copy()}

    from ..models.composite import CompositeSequenceModel

    if isinstance(model, CompositeSequenceModel):
        weights: dict[str, np.ndarray] = {}
        for target, state in model.state_dict().items():
            for key, tensor in state.items():
                weights[f"{target}{_SUBMODEL_SEPARATOR}{key}"] = np.array(tensor.numpy())
        missing = set(train.target_names) - {name.split(_SUBMODEL_SEPARATOR)[0] for name in weights}
        if missing:
            raise ValueError(f"the model has no sub-model for {sorted(missing)}")
        composite_settings = _jsonable(dataclasses.asdict(model.config))
        assert isinstance(composite_settings, dict)
        return "composite_torch", composite_settings, weights

    raise TypeError(
        f"{type(model).__name__} cannot be bundled: supported are CompositeSequenceModel "
        "(one network per target) and LinearARXModel"
    )


def build_bundle(
    model: SequenceModel,
    train: SequenceDataset,
    *,
    sample_time_s: float,
    parameter_status: ParameterStatus,
    data_spec: Mapping[str, Any] | None = None,
    data_fingerprints: Mapping[str, str] | None = None,
    training: Mapping[str, Any] | None = None,
) -> ModelBundle:
    """Pack a fitted model with the scalers and the description of its training data set.

    Parameters
    ----------
    model
        A fitted ``CompositeSequenceModel`` or ``LinearARXModel``.
    train
        The training data set of the model: scalers, channel names, window and the ranges of the
        training inputs come from it and from nothing else.
    sample_time_s
        Sample time of the data (a data set does not know it).
    parameter_status
        Status of the process parameters behind the data (``ProcessConfig.provenance.status``).
    data_spec, data_fingerprints, training
        Description of the data and of the fit for the manifest (for example
        ``DataSpec.as_parameters()``, ``GeneratedData.fingerprints``, number of training runs and
        model seed).
    """
    kind, settings, weights = _export_model(model, train)
    window = train.config
    payload: dict[str, Any] = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "model_kind": kind,
        "model_label": model.name,
        "model_settings": settings,
        "exogenous_names": list(train.exogenous_names),
        "target_names": list(train.target_names),
        "observed_target_names": list(train.observed_target_names),
        "measured_input_names": list(window.measured_input_names),
        "window": {
            "history_length": window.history_length,
            "horizon": window.horizon,
            "future_actions": window.future_actions,
        },
        "sample_time_s": float(sample_time_s),
        "input_scaler": ScalerPayload.from_standardizer(train.input_scaler).model_dump(mode="json"),
        "target_scaler": ScalerPayload.from_standardizer(train.target_scaler).model_dump(
            mode="json"
        ),
        "input_ranges": {
            name: value.model_dump(mode="json") for name, value in _input_ranges(train).items()
        },
        "data_spec": _jsonable(dict(data_spec or {})),
        "data_fingerprints": {str(k): str(v) for k, v in (data_fingerprints or {}).items()},
        "parameter_status": parameter_status,
        "training": _jsonable(dict(training or {})),
        "code": _code_provenance().model_dump(mode="json"),
        "weights_sha256": weights_digest(weights),
    }
    # The fingerprint is formed over the JSON form the manifest has on disk, so that a bundle
    # read back hashes to the same value.
    normalised = BundleManifest.model_validate({**payload, "fingerprint": ""}).model_dump(
        mode="json"
    )
    manifest = BundleManifest.model_validate(
        {**normalised, "fingerprint": _manifest_fingerprint(normalised)}
    )
    bundle = ModelBundle(manifest=manifest, weights=weights)
    bundle.verify()
    return bundle


def fit_bundle(
    data: GeneratedData,
    *,
    n_train: int,
    model_seed: int,
    kind: ModelKind = "composite_torch",
    torch_config: TorchSequenceConfig | None = None,
) -> ModelBundle:
    """From runs to a bundle: build the data sets of one budget, fit, pack.

    ``kind="composite_torch"`` fits one network per target (GRU with the default settings of the
    architecture comparison unless ``torch_config`` says otherwise); ``kind="linear_arx"`` fits the
    linear ARX baseline, which needs neither PyTorch nor a model seed. Scalers come from the
    training runs of this budget only (run-level split of ``data``).
    """
    from ..evaluation.experiments import build_datasets

    train, validation, _ = build_datasets(data, n_train)
    model: SequenceModel
    if kind == "linear_arx":
        model = BundledARXModel()
    elif kind == "composite_torch":
        require_extra("torch", "torch")
        from ..models.composite import CompositeGruSequenceModel

        model = CompositeGruSequenceModel(torch_config)
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    model.fit(train, seed=model_seed, validation=validation)
    return build_bundle(
        model,
        train,
        sample_time_s=data.spec.sample_time_s,
        parameter_status=data.process_config.provenance.status,
        data_spec=data.spec.as_parameters(),
        data_fingerprints=data.fingerprints,
        training={
            "n_train_runs": n_train,
            "n_validation_runs": 0 if validation is None else len(data.split.val),
            "model_seed": model_seed,
            "n_parameters": model.n_parameters,
        },
    )


# ------------------------------------------------------------------- instantiate


def _split_submodel_weights(
    weights: Mapping[str, np.ndarray], target_names: Sequence[str]
) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, np.ndarray]] = {name: {} for name in target_names}
    for name, array in weights.items():
        target, separator, key = name.partition(_SUBMODEL_SEPARATOR)
        if not separator or target not in grouped:
            raise BundleIntegrityError(f"unexpected weight array {name!r}")
        grouped[target][key] = array
    return grouped


def instantiate_model(bundle: ModelBundle) -> SequenceModel:
    """Rebuild the fitted model of a bundle. Loads PyTorch only for a network.

    A network is always rebuilt on the CPU, whatever device it was trained on, so that the same
    bundle gives the same numbers on every host.
    """
    manifest = bundle.manifest
    settings = manifest.model_settings
    n_exogenous = len(manifest.exogenous_names)
    if manifest.model_kind == "linear_arx":
        return BundledARXModel.from_state(
            n_lags_target=int(settings["n_lags_target"]),  # type: ignore[arg-type]
            n_lags_exogenous=int(settings["n_lags_exogenous"]),  # type: ignore[arg-type]
            ridge=float(settings["ridge"]),  # type: ignore[arg-type]
            coefficients=bundle.weights[_ARX_COEFFICIENTS],
            n_exogenous=n_exogenous,
            n_targets=len(manifest.target_names),
        )

    require_extra("torch", "torch")
    import torch

    from ..models.composite import CompositeSequenceModel
    from ..models.torch_sequence import TorchSequenceConfig

    config = TorchSequenceConfig(**{**settings, "device": "cpu"})  # type: ignore[arg-type]
    grouped = _split_submodel_weights(bundle.weights, manifest.target_names)
    model = CompositeSequenceModel(config)
    model.load_checkpoint(
        submodel_state_dicts={
            target: {key: torch.from_numpy(np.array(array)) for key, array in arrays.items()}
            for target, arrays in grouped.items()
        },
        target_names=manifest.target_names,
        n_exogenous=n_exogenous,
    )
    return model

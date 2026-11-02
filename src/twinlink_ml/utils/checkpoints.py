r"""Checkpoint persistence for torch models (step 6, part 2).

Previously there were no saved weights -- every result needed full retraining.
This module saves and loads :class:`~twinlink_ml.models.torch_sequence.TorchSequenceModel`
and :class:`~twinlink_ml.models.composite.CompositeSequenceModel` (Befund 26: one network per
target) so that a run can later be reloaded exactly reproducibly -- weights,
scalers and provenance in one file, referenced from the JSONL log via
``checkpoint/path``.

Schema version 1
-----------------
One ``torch.save`` dict per ``run_uid`` under ``artifacts/checkpoints/<run_uid>.pt``.
Composite models save one state dict per submodel (``submodel_state_dicts``); the
scalers are stored **once for the whole** in the dict, because all submodels derive their view
deterministically via :meth:`Standardizer.select` from the same base pair (see the
``composite.py`` module docstring) -- storing them redundantly per submodel would express
nothing additional.

``target_names``/``exogenous_names``/``n_exogenous`` come from the training dataset of the
run, not from the model itself -- the model only knows them after
``fit()``/``load_checkpoint()``, the caller (which has just built the training dataset)
knows them beforehand. ``exogenous_names`` is not needed when loading (the
network architecture depends only on the channel **count**), but is stored in the checkpoint
so that a later caller without the corresponding JSONL line can make sure that its
input channel schema matches the trained model.

``torch.load(..., weights_only=False)``: deliberately, not the safer default. These
checkpoints are exclusively locally generated, own artifacts (no download, no
transport across a trust boundary) -- ``weights_only=True`` would reject the embedded
provenance (tuples, dicts, strings), which is not a tensor.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..data.windows import Standardizer
from ..models.composite import CompositeSequenceModel
from ..models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "checkpoint_path",
    "load_checkpoint",
    "save_checkpoint",
]

CHECKPOINT_SCHEMA_VERSION = 1


def _scaler_to_payload(scaler: Standardizer) -> dict[str, list[float]]:
    return {"mean": scaler.mean.tolist(), "scale": scaler.scale.tolist()}


def _scaler_from_payload(payload: Mapping[str, Any]) -> Standardizer:
    return Standardizer(
        mean=np.asarray(payload["mean"], dtype=np.float64),
        scale=np.asarray(payload["scale"], dtype=np.float64),
    )


def checkpoint_path(directory: Path, run_uid: str) -> Path:
    """Canonical path ``<directory>/<run_uid>.pt``.

    Creates ``directory`` with ``exist_ok=True`` -- several processes (step 3, parallel
    sweep) can attempt this at the same time without one of them aborting with
    ``FileExistsError``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{run_uid}.pt"


def save_checkpoint(
    path: Path,
    model: TorchSequenceModel | CompositeSequenceModel,
    *,
    model_label: str,
    run_uid: str,
    target_names: Sequence[str],
    exogenous_names: Sequence[str],
    n_exogenous: int,
    input_scaler: Standardizer,
    target_scaler: Standardizer,
    process_fingerprint: str,
    data_spec_params: Mapping[str, Any],
    seed_bundle_params: Mapping[str, Any],
    git_commit: str,
) -> None:
    """Save model, scalers and provenance in one ``.pt`` file."""
    import torch

    is_composite = isinstance(model, CompositeSequenceModel)
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_label": model_label,
        "composite": is_composite,
        "config": dataclasses.asdict(model.config),
        "target_names": tuple(target_names),
        "exogenous_names": tuple(exogenous_names),
        "n_exogenous": n_exogenous,
        "input_scaler": _scaler_to_payload(input_scaler),
        "target_scaler": _scaler_to_payload(target_scaler),
        "run_uid": run_uid,
        "process_fingerprint": process_fingerprint,
        "data_spec": dict(data_spec_params),
        "seed_bundle": dict(seed_bundle_params),
        "git_commit": git_commit,
    }
    if is_composite:
        payload["submodel_state_dicts"] = model.state_dict()
    else:
        payload["state_dict"] = model.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
) -> tuple[TorchSequenceModel | CompositeSequenceModel, dict[str, Any]]:
    """Reconstruct a model from a ``.pt`` file.

    Returns: ``(Modell, Metadaten)`` -- the metadata contains everything from the saved
    dict except the weights themselves, plus the reconstructed ``Standardizer`` objects
    under ``input_scaler``/``target_scaler``.
    """
    import torch

    payload: dict[str, Any] = torch.load(path, weights_only=False)
    version = payload.get("schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint-Schema {version!r} unbekannt, erwartet {CHECKPOINT_SCHEMA_VERSION}"
        )
    config = TorchSequenceConfig(**payload["config"])
    model: TorchSequenceModel | CompositeSequenceModel
    if payload["composite"]:
        model = CompositeSequenceModel(config)
        model.load_checkpoint(
            submodel_state_dicts=payload["submodel_state_dicts"],
            target_names=payload["target_names"],
            n_exogenous=payload["n_exogenous"],
        )
    else:
        model = TorchSequenceModel(config)
        model.load_checkpoint(
            state_dict=payload["state_dict"],
            target_names=payload["target_names"],
            n_exogenous=payload["n_exogenous"],
        )
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"state_dict", "submodel_state_dicts"}
    }
    metadata["input_scaler"] = _scaler_from_payload(payload["input_scaler"])
    metadata["target_scaler"] = _scaler_from_payload(payload["target_scaler"])
    return model, metadata

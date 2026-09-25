r"""Checkpoint-Persistenz für Torch-Modelle (Schritt 6, Teil 2).

Bisher gab es keine gespeicherten Gewichte -- jedes Ergebnis brauchte volles Neutraining.
Dieses Modul speichert und lädt :class:`~twinlink_ml.models.torch_sequence.TorchSequenceModel`
und :class:`~twinlink_ml.models.composite.CompositeSequenceModel` (Befund 26: ein Netz je
Zielgröße) so, dass ein Lauf später exakt reproduzierbar neu geladen werden kann -- Gewichte,
Skalierer und Provenienz in einer Datei, referenziert aus dem JSONL-Log über
``checkpoint/path``.

Schema Version 1
-----------------
Ein ``torch.save``-Dict je ``run_uid`` unter ``artifacts/checkpoints/<run_uid>.pt``.
Composite-Modelle speichern ein State-Dict je Teilmodell (``submodel_state_dicts``); die
Skalierer stehen **einmal fürs Ganze** im Dict, weil alle Teilmodelle ihre Sicht
deterministisch über :meth:`Standardizer.select` vom selben Basis-Paar ableiten (siehe
``composite.py``-Moduldocstring) -- sie redundant je Teilmodell zu speichern würde nichts
zusätzliches ausdrücken.

``target_names``/``exogenous_names``/``n_exogenous`` kommen vom Trainings-Datensatz des
Laufs, nicht aus dem Modell selbst -- das Modell kennt sie erst nach
``fit()``/``load_checkpoint()``, der Aufrufer (der den Trainings-Datensatz gerade gebaut
hat) kennt sie vorher. ``exogenous_names`` wird beim Laden nicht gebraucht (die
Netzarchitektur hängt nur von der Kanal**zahl** ab), steht aber im Checkpoint, damit sich
ein späterer Aufrufer ohne die zugehörige JSONL-Zeile vergewissern kann, dass sein
Eingangskanal-Schema zum trainierten Modell passt.

``torch.load(..., weights_only=False)``: bewusst, nicht der sicherere Default. Diese
Checkpoints sind ausschließlich lokal erzeugte, eigene Artefakte (kein Download, kein
Transport über eine Vertrauensgrenze) -- ``weights_only=True`` würde die eingebettete
Provenienz (Tupel, Dicts, Strings) zurückweisen, die kein Tensor ist.
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
    """Kanonischer Pfad ``<directory>/<run_uid>.pt``.

    Legt ``directory`` mit ``exist_ok=True`` an -- mehrere Prozesse (Schritt 3, paralleler
    Sweep) können das gleichzeitig versuchen, ohne dass einer davon mit ``FileExistsError``
    abbricht.
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
    """Modell, Skalierer und Provenienz in einer ``.pt``-Datei sichern."""
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
    """Modell aus einer ``.pt``-Datei rekonstruieren.

    Rückgabe: ``(Modell, Metadaten)`` -- Metadaten enthält alles aus dem gespeicherten
    Dict außer den Gewichten selbst, plus die rekonstruierten ``Standardizer``-Objekte
    unter ``input_scaler``/``target_scaler``.
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

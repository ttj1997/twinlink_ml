"""Modelle hinter dem gemeinsamen :class:`SequenceModel`-Interface.

``GruSequenceModel`` wird bewusst **nicht** hier importiert: das Paket muss ohne PyTorch
importierbar bleiben (Schritte 1-4 brauchen es nicht). Import bei Bedarf::

    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

``ExtraTreesModel`` ist hier trotzdem direkt importiert (anders als die Torch-Modelle):
sklearn wird erst in ``fit()`` importiert, die Klasse selbst ist torch-artig lazy und
verlangt beim reinen Import kein sklearn. Sie ist **nicht** in
``models/registry.py``/``MODEL_SLOTS`` registriert -- kein Kandidat des
Architekturvergleichs, siehe ``extra_trees.py``-Moduldocstring.
"""

from .base import SequenceModel
from .baselines import (
    ClimatologyModel,
    LinearARXModel,
    PersistenceModel,
    StaticRegressionModel,
)
from .extra_trees import ExtraTreesModel
from .narx import RandomFeatureNARX

__all__ = [
    "ClimatologyModel",
    "ExtraTreesModel",
    "LinearARXModel",
    "PersistenceModel",
    "RandomFeatureNARX",
    "SequenceModel",
    "StaticRegressionModel",
]

"""Models behind the common :class:`SequenceModel` interface.

``GruSequenceModel`` is deliberately **not** imported here: the package must remain
importable without PyTorch (steps 1-4 do not need it). Import when needed::

    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

``ExtraTreesModel`` is nevertheless imported directly here (unlike the Torch models):
sklearn is only imported in ``fit()``, the class itself is lazy in the torch manner and
does not require sklearn on a mere import. It is **not** registered in
``models/registry.py``/``MODEL_SLOTS`` -- not a candidate of the architecture
comparison, see the ``extra_trees.py`` module docstring.
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

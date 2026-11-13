"""Hybrid model against GRU against parameter refit with a STRUCTURALLY deviating truth.

Study docs/NN_hybrid_structural.md (docs/12 section I, item 30); pre-registration in the docstring
of ``scripts/analyze_hybrid_structural.py``. The code of Befund 122 (``hybrid_model``) is reused
unchanged; this package adds the truth and the model M2b.
"""

from .bias import BiasedMap, training_bias
from .study import (
    MODELS,
    StructuralSetup,
    build_structural_setup,
    freeboard_share,
    load_structural_config,
)
from .truth import (
    FreeboardEvaporationGranulator,
    bed_feed,
    exhaust_temperature_celsius,
    steady_config,
)

__all__ = [
    "MODELS",
    "BiasedMap",
    "FreeboardEvaporationGranulator",
    "StructuralSetup",
    "bed_feed",
    "build_structural_setup",
    "exhaust_temperature_celsius",
    "freeboard_share",
    "load_structural_config",
    "steady_config",
    "training_bias",
]

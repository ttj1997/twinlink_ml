"""Hybrid against GRU against calibration with a structural error to the UNSAFE side.

Study docs/NN_hybrid_structural_2.md; pre-registration in the docstring of
``scripts/analyze_hybrid_structural_2.py``. The code of Befund 122 (``hybrid_model``) and of
Befund 124 (``hybrid_structural``: model M2b, records, drivers) is reused; this package adds the
truth (exhaust recirculation, an existing switch of the grey-box model).
"""

from .study import (
    DEFAULT_CONFIG_PATH,
    RecirculationSetup,
    build_recirculation_setup,
    load_recirculation_config,
    recirculation_fraction,
    with_recirculation,
)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "RecirculationSetup",
    "build_recirculation_setup",
    "load_recirculation_config",
    "recirculation_fraction",
    "with_recirculation",
]

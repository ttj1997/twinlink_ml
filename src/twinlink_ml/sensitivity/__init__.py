"""Global sensitivity analysis (Sobol indices) of the steady-state models (docs/25).

``SALib`` is an optional dependency (extra ``sensitivity``); :mod:`.sobol` imports it lazily, the
P2 model needs ``casadi`` (extra ``optimal-control``).
"""

from .config import DesignSpec, InputSpec, SensitivityStudyConfig, load_sensitivity_config
from .sobol import SobolIndices, block_valid, rank_change, saltelli_unit, sobol_indices, verdict

__all__ = [
    "DesignSpec",
    "InputSpec",
    "SensitivityStudyConfig",
    "SobolIndices",
    "block_valid",
    "load_sensitivity_config",
    "rank_change",
    "saltelli_unit",
    "sobol_indices",
    "verdict",
]

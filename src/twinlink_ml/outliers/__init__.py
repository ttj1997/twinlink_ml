"""Sensor outliers on top of the existing sensor model, and a causal Hampel filter (docs/27).

Applied to already simulated data through ``remeasure`` (no new simulation); only one channel.
"""

from .config import DEFAULT_CONFIG_PATH, OutlierStudyConfig, load_outlier_config
from .contamination import (
    Condition,
    ContaminatedData,
    apply_condition,
    existing_noise_std,
    outlier_process,
    outlier_seed,
)
from .evaluation import REFERENCES, evaluate_against_references
from .hampel import MAD_SCALE, CausalHampelFilter, FilterResult
from .process import (
    BurstOutliers,
    IndependentOutliers,
    NoOutliers,
    OutlierProcess,
    OutlierSample,
)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "MAD_SCALE",
    "REFERENCES",
    "BurstOutliers",
    "CausalHampelFilter",
    "Condition",
    "ContaminatedData",
    "FilterResult",
    "IndependentOutliers",
    "NoOutliers",
    "OutlierProcess",
    "OutlierSample",
    "OutlierStudyConfig",
    "apply_condition",
    "evaluate_against_references",
    "existing_noise_std",
    "load_outlier_config",
    "outlier_process",
    "outlier_seed",
]

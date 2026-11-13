"""Calibration of the plant-near grey-box model of plant A: identifiability first (docs/40).

Basis: the plant-near parametrisation, data, stationary segments and roles of Befund 117
(:mod:`twinlink_ml.plant_scale`, only read). Candidate parameters: bed heat-loss UA, the
bed-to-exhaust distance per product and the freeboard heat transfer (switch of Befund 117, default
off), the effective (solid) heat capacity, the critical moisture and nu_max. Steps: normalised
sensitivity matrix with SVD, collinearity and profile likelihood (:mod:`.identifiability`), then
the stationary calibration of at most three identifiable parameters with a cross-check between the
two days against a constant bias and the uncalibrated model (:mod:`.calibration`).
"""

from .calibration import (
    FitResult,
    apply_bias,
    bias_offsets,
    error_metrics,
    fit_parameters,
    reading,
    weighted_residuals,
)
from .config import (
    ParameterSpec,
    PlantCalibrationConfig,
    load_plant_calibration_config,
)
from .identifiability import (
    SubsetScore,
    collinearity_index,
    delta_msqr,
    determinant_measure,
    fd_steps,
    finite_difference_thetas,
    identifiable_subsets,
    profile_verdict,
    select_subset,
    sensitivity_from_outputs,
    svd_summary,
)
from .model import (
    MATERIAL_PARAMETERS,
    ModelContext,
    Prediction,
    evaluate,
    evaluate_rows,
    offset_for,
)

__all__ = [
    "MATERIAL_PARAMETERS",
    "FitResult",
    "ModelContext",
    "ParameterSpec",
    "PlantCalibrationConfig",
    "Prediction",
    "SubsetScore",
    "apply_bias",
    "bias_offsets",
    "collinearity_index",
    "delta_msqr",
    "determinant_measure",
    "error_metrics",
    "evaluate",
    "evaluate_rows",
    "fd_steps",
    "finite_difference_thetas",
    "fit_parameters",
    "identifiable_subsets",
    "load_plant_calibration_config",
    "offset_for",
    "profile_verdict",
    "reading",
    "select_subset",
    "sensitivity_from_outputs",
    "svd_summary",
    "weighted_residuals",
]

"""Learned sequence models as environments for control (docs/30_surrogate_env.md).

Package of the study "how strongly do RL and an optimiser exploit the errors of a learned model?":
the steady state of a composite GRU of the Befund 94 pipeline as a fixed point of its horizon map
(:mod:`.fixed_point`, :mod:`.model`), the bandit of Befund 93/96 on such a model (:mod:`.env`) and
the multistart optimiser C2 (:mod:`.rto`). Truth and evaluation are always the grey-box model.
"""

from .env import LearnedSeparatedLagrangeBanditEnv, learned_info
from .fixed_point import (
    FIXED_POINT_STATUSES,
    FixedPointCriterion,
    FixedPointResult,
    solve_fixed_points,
)
from .model import (
    EXOGENOUS_NAMES,
    TARGET_NAMES,
    EnsembleFixedPointModel,
    FixedInputs,
    GruFixedPointModel,
    OperatingPointModel,
    Prediction,
)
from .rto import MultistartConfig, PointEvaluation, multistart_minimize, start_points

__all__ = [
    "EXOGENOUS_NAMES",
    "FIXED_POINT_STATUSES",
    "TARGET_NAMES",
    "EnsembleFixedPointModel",
    "FixedInputs",
    "FixedPointCriterion",
    "FixedPointResult",
    "GruFixedPointModel",
    "LearnedSeparatedLagrangeBanditEnv",
    "MultistartConfig",
    "OperatingPointModel",
    "PointEvaluation",
    "Prediction",
    "learned_info",
    "multistart_minimize",
    "solve_fixed_points",
    "start_points",
]

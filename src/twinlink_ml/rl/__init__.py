"""Reinforcement learning environments -- bandit formulation of Befund 52 first.

See :mod:`twinlink_ml.rl.env` for scope and deliberate omissions (storage, several
lines, net objective function, plant scale).
"""

from .env import (
    ACTION_NAMES,
    DEFAULT_REWARD_CENTERING_SCALE_KW,
    DISTURBANCE_NAMES,
    FIXED_NOZZLE_PRESSURE_BAR,
    FIXED_OPERATING_POINT_GAS_VELOCITY_M_PER_S,
    FIXED_OPERATING_POINT_INLET_TEMPERATURE_CELSIUS,
    BanditEnvConfig,
    OperatingPointBanditEnv,
    RewardCentering,
)
from .evaluation import (
    ClosedRegretResult,
    InfeasibleViolation,
    TestPoint,
    assert_disjoint_from_test_grid,
    evaluate_policy,
    flat_metrics,
    is_boundary_optimum,
    load_test_points,
)

__all__ = [
    "ACTION_NAMES",
    "DEFAULT_REWARD_CENTERING_SCALE_KW",
    "DISTURBANCE_NAMES",
    "FIXED_NOZZLE_PRESSURE_BAR",
    "FIXED_OPERATING_POINT_GAS_VELOCITY_M_PER_S",
    "FIXED_OPERATING_POINT_INLET_TEMPERATURE_CELSIUS",
    "BanditEnvConfig",
    "ClosedRegretResult",
    "InfeasibleViolation",
    "OperatingPointBanditEnv",
    "RewardCentering",
    "TestPoint",
    "assert_disjoint_from_test_grid",
    "evaluate_policy",
    "flat_metrics",
    "is_boundary_optimum",
    "load_test_points",
]

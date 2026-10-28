"""Operating point and heat recovery, jointly or one after the other (docs/31_joint_recovery.md).

P1 of Befund 107 (line heater -> dryer -> recovery of the flowsheet package, physical mode) with a
bypass of the recovery and a pressurised medium as a second variant (:mod:`.problem`); settings,
problem families and the strategies S0-S3 (:mod:`.study`). Existing modules are only read; the
model, its ranges, constraints, parameters and the 50 test states come from
``conf/surrogate_opt/surrogate_rto.yaml``.
"""

from .problem import BINDINGS, MEDIUM_INDEX, JointLineFamily, JointLineProblem, RestrictedProblem
from .study import (
    DEFAULT_CONFIG_PATH,
    JointRecoveryConfig,
    assemble,
    better,
    build_families,
    joint_test_states,
    load_joint_config,
    medium_optimum,
    record,
    strategy_s0,
    strategy_s2,
)

__all__ = [
    "BINDINGS",
    "DEFAULT_CONFIG_PATH",
    "MEDIUM_INDEX",
    "JointLineFamily",
    "JointLineProblem",
    "JointRecoveryConfig",
    "RestrictedProblem",
    "assemble",
    "better",
    "build_families",
    "joint_test_states",
    "load_joint_config",
    "medium_optimum",
    "record",
    "strategy_s0",
    "strategy_s2",
]

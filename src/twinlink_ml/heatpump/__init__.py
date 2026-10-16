"""Heat pump on low-temperature waste heat (Stage A) and with a storage (Stage B, gate).

Stage A: pinch-consistent, step by step. Stage B: heat pump plus storage under a time-varying
electricity signal; gate for RL/MPC.

See ``docs/21_heat_pump.md`` and the pre-registration in the docstring of
``scripts/gate_heat_pump.py``. Streams and site come from storage test 1
(:mod:`twinlink_ml.storage_timeslice`, Befund 99).
"""

from .config import (
    CONDENSING_GRID_CELSIUS,
    ELECTRICITY_CO2_2030_KG_PER_KWH,
    ELECTRICITY_CO2_KG_PER_KWH,
    EVAPORATING_GRID_CELSIUS,
    GAS_CO2_KG_PER_KWH,
    HEAT_PUMP_SIZES_KW,
    PLACEHOLDER_BANDS,
    SITE_CO2_T_PER_A,
    Capital,
    Criterion,
    Economics,
    HeatPumpSpec,
    Signal,
    co2_criterion,
    cost_criterion,
    is_realistic,
)
from .cop import DesignPair, carnot_cop, design_pairs, heat_pump_cop, pseudo_stream_points
from .dispatch import HeatPumpLines, StepDispatch, dispatch_steps, heat_pump_lines
from .evaluate import (
    StageAResult,
    StageBResult,
    evaluate_stage_a,
    evaluate_stage_b,
    site_cascade,
)
from .pinch import PLACEMENTS, PinchSummary, classify_placement, grand_composite_kw, pinch_summary
from .signal import daily_shape, signal_series, trailing_mean
from .storage import (
    RULES,
    CoupledPieces,
    DispatchResult,
    RuleRun,
    coupled_pieces,
    simulate_rule,
    solve_foresight,
    step_optimum,
)

__all__ = [
    "CONDENSING_GRID_CELSIUS",
    "ELECTRICITY_CO2_2030_KG_PER_KWH",
    "ELECTRICITY_CO2_KG_PER_KWH",
    "EVAPORATING_GRID_CELSIUS",
    "GAS_CO2_KG_PER_KWH",
    "HEAT_PUMP_SIZES_KW",
    "PLACEHOLDER_BANDS",
    "PLACEMENTS",
    "RULES",
    "SITE_CO2_T_PER_A",
    "Capital",
    "CoupledPieces",
    "Criterion",
    "DesignPair",
    "DispatchResult",
    "Economics",
    "HeatPumpLines",
    "HeatPumpSpec",
    "PinchSummary",
    "RuleRun",
    "Signal",
    "StageAResult",
    "StageBResult",
    "StepDispatch",
    "carnot_cop",
    "classify_placement",
    "co2_criterion",
    "cost_criterion",
    "coupled_pieces",
    "daily_shape",
    "design_pairs",
    "dispatch_steps",
    "evaluate_stage_a",
    "evaluate_stage_b",
    "grand_composite_kw",
    "heat_pump_cop",
    "heat_pump_lines",
    "is_realistic",
    "pinch_summary",
    "pseudo_stream_points",
    "signal_series",
    "simulate_rule",
    "site_cascade",
    "solve_foresight",
    "step_optimum",
    "trailing_mean",
]

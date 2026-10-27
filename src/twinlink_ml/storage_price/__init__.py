"""Heat pump and storage under an hourly day-ahead price: rule, MPC or RL (storage-price study).

Plant-A-like structure (waste heat -> storage -> preheater, residual demand by gas) with the
heat-pump model of Befund 101 and the storage model of Befund 99 (both only read), power limits,
the hourly DE-LU day-ahead prices of 2024. See ``docs/NN_storage_price.md``, the settings in
``conf/storage_price/storage_price.yaml`` and the pre-registered reading in the docstring of
``scripts/analyze_storage_price.py``.
"""

from .config import StoragePriceConfig, load_storage_price_config, variant_names
from .controllers import MPC, Action, Controller, Hindsight, NoPump, StepRule, r0_rule, r1_rule
from .evaluate import PeriodResult, month_hours, project, run_months, run_period
from .lp import WindowSolution, solve_window
from .market import Market, build_market
from .plant import Plant, build_plant
from .prices import Clock, PriceYear, known_price_forecast, load_price_year, year_clock
from .study import (
    Case,
    build_case,
    evaluate_controllers,
    rule_and_mpc_controllers,
    tune_r0,
    tune_r1,
)

__all__ = [
    "MPC",
    "Action",
    "Case",
    "Clock",
    "Controller",
    "Hindsight",
    "Market",
    "NoPump",
    "PeriodResult",
    "Plant",
    "PriceYear",
    "StepRule",
    "StoragePriceConfig",
    "WindowSolution",
    "build_case",
    "build_market",
    "build_plant",
    "evaluate_controllers",
    "known_price_forecast",
    "load_price_year",
    "load_storage_price_config",
    "month_hours",
    "project",
    "r0_rule",
    "r1_rule",
    "rule_and_mpc_controllers",
    "run_months",
    "run_period",
    "solve_window",
    "tune_r0",
    "tune_r1",
    "variant_names",
    "year_clock",
]

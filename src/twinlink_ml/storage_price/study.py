"""One case of the study: build plant and market, tune the rules on the training months, evaluate.

Tuning (training months only): ``kappa`` of R0 and ``q`` of R1 minimise the summed cost of the
month runs of the training months over their grids (first minimum on a tie). Nothing of a test month
enters: the grids are fixed in the config, the end value is the mean price of the training months,
and the price information at an hour never reaches past the end of its day.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import StoragePriceConfig
from .controllers import MPC, Controller, Hindsight, NoPump, pump_only, r0_rule, r1_rule
from .evaluate import PeriodResult, run_months
from .market import Market, build_market
from .plant import Plant, build_plant
from .prices import PriceYear, load_price_year

__all__ = [
    "Case",
    "build_case",
    "evaluate_controllers",
    "rule_and_mpc_controllers",
    "tune_r0",
    "tune_r1",
]


@dataclass(frozen=True, slots=True)
class Case:
    """Plant, market and prices of one configuration."""

    config: StoragePriceConfig
    prices: PriceYear
    plant: Plant
    market: Market


def build_case(config: StoragePriceConfig, root: Path) -> Case:
    """Load the prices and build plant and market (price information over the longest horizon)."""
    prices = load_price_year(config.prices, root)
    plant = build_plant(config, prices.clock)
    horizon = max(max(config.controllers.mpc_horizons_h), config.controllers.r1_window_h)
    market = build_market(config, prices, cop=plant.cop, horizon_h=horizon)
    return Case(config=config, prices=prices, plant=plant, market=market)


def _total(results: Sequence[PeriodResult]) -> float:
    return float(sum(r.cost_eur for r in results))


def tune_r0(case: Case) -> dict[str, Any]:
    """Grid search of ``kappa`` on the training months."""
    cfg = case.config
    tol = cfg.controllers.violation_tolerance_relative
    costs = []
    for kappa in cfg.controllers.r0_kappa_grid:
        rule = r0_rule(case.plant, case.market, kappa)
        res = run_months(
            rule, case.plant, case.market, case.prices.clock, cfg.training_months, tolerance=tol
        )
        costs.append(_total(res))
    best = int(np.argmin(costs))
    return {
        "grid": list(cfg.controllers.r0_kappa_grid),
        "training_cost_eur": costs,
        "kappa": cfg.controllers.r0_kappa_grid[best],
        "at_grid_edge": best in (0, len(costs) - 1),
    }


def tune_r1(case: Case) -> dict[str, Any]:
    """Grid search of ``q`` on the training months."""
    cfg = case.config
    c = cfg.controllers
    costs = []
    for q in c.r1_quantile_grid:
        rule = r1_rule(case.plant, case.market, q, c.r1_window_h, c.stored_value_epsilon)
        res = run_months(
            rule,
            case.plant,
            case.market,
            case.prices.clock,
            cfg.training_months,
            tolerance=c.violation_tolerance_relative,
        )
        costs.append(_total(res))
    best = int(np.argmin(costs))
    return {
        "grid": list(c.r1_quantile_grid),
        "training_cost_eur": costs,
        "quantile": c.r1_quantile_grid[best],
        "at_grid_edge": best in (0, len(costs) - 1),
    }


def rule_and_mpc_controllers(case: Case, kappa: float, quantile: float) -> list[Controller]:
    """N0, P0, R0, R1, M1 and M2 for every horizon, P*."""
    c = case.config.controllers
    pl, mk = case.plant, case.market
    out: list[Controller] = [
        NoPump(pl, mk),
        pump_only(pl, mk, c.stored_value_epsilon),
        r0_rule(pl, mk, kappa),
        r1_rule(pl, mk, quantile, c.r1_window_h, c.stored_value_epsilon),
    ]
    for h in c.mpc_horizons_h:
        out.append(MPC(pl, mk, horizon_h=h, information="perfect"))
        out.append(MPC(pl, mk, horizon_h=h, information="known"))
    out.append(Hindsight(pl, mk))
    return out


def evaluate_controllers(
    case: Case, controllers: Sequence[Controller], months: Sequence[int]
) -> dict[str, list[dict[str, Any]]]:
    """Month runs of every controller: ``{name: [PeriodResult.as_dict(), ...]}``."""
    tol = case.config.controllers.violation_tolerance_relative
    return {
        ctrl.name: [
            r.as_dict()
            for r in run_months(
                ctrl, case.plant, case.market, case.prices.clock, months, tolerance=tol
            )
        ]
        for ctrl in controllers
    }

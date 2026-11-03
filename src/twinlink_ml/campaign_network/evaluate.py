"""Key figures of one run and the evaluation of one schedule draw over the whole grid.

Key figures per configuration (annual sums over the 8760 h): heat used (kWh) and its share of the
source energy ``P x 8760 h``, gas replaced (heat / efficiency of the replaced supply), CO2 of the
replaced gas, pump electricity (``pump_fraction`` x heat delivered) and its CO2, net CO2 saving.
``V_N = (use K1 - use K0) / source energy``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .cascade import CascadeResult, run_k0, run_k1, run_k2
from .config import CampaignNetworkConfig
from .profiles import Consumers, build_consumers, plant_schedules
from .target import TargetResult, run_k3

__all__ = ["evaluate_draw", "key_figures"]


def key_figures(
    result: CascadeResult | TargetResult,
    consumers: Consumers,
    power_kw: float,
    config: CampaignNetworkConfig,
) -> dict[str, Any]:
    """Annual key figures of one run."""
    n = consumers.ambient.size
    source_kwh = power_kw * n
    per = {s.name: float(np.sum(result.duty_kw[s.name])) for s in consumers.streams}
    use = float(sum(per.values()))
    gas = float(sum(per[s.name] / s.gas_efficiency for s in consumers.streams))
    pump_el = config.pump_fraction * use
    co2_gas = config.gas_co2_kg_per_kwh * gas
    co2_pump = config.electricity_co2_kg_per_kwh * pump_el
    out: dict[str, Any] = {
        "source_kwh": source_kwh,
        "use_kwh": use,
        "utilisation": use / source_kwh,
        "use_by_consumer_kwh": per,
        "gas_kwh": gas,
        "co2_gas_kg": co2_gas,
        "pump_electricity_kwh": pump_el,
        "co2_pump_kg": co2_pump,
        "co2_net_kg": co2_gas - co2_pump,
        "charge_kwh": float(np.sum(result.charge_kw)),
        "discharge_kwh": float(np.sum(result.discharge_kw)),
        "content_end_kwh": float(result.content_kwh[-1]),
    }
    if isinstance(result, TargetResult):
        out["lp_status"] = result.status
        out["lp_message"] = result.message
        out["lp_seconds"] = result.seconds
    return out


def evaluate_draw(
    config: CampaignNetworkConfig,
    availability: float,
    mode: str,
    draw: int,
    power_levels: tuple[float, ...] | None = None,
    storage_levels: tuple[float, ...] | None = None,
    with_k3: bool = True,
) -> dict[str, Any]:
    """All configurations for every source power and storage size on one schedule draw."""
    powers = config.source.power_kw_levels if power_levels is None else power_levels
    sizes = config.storage.hours_levels if storage_levels is None else storage_levels
    on = plant_schedules(config, availability, mode, draw)
    consumers = build_consumers(config, on)
    out: dict[str, Any] = {
        "availability": availability,
        "mode": mode,
        "draw": draw,
        "realised_availability": {k: float(np.mean(v)) for k, v in on.items()},
        "all_down_share": float(np.mean(~np.any(np.stack(list(on.values())), axis=0))),
        "demand_kwh": {s.name: float(np.sum(s.demand_kw)) for s in consumers.streams},
        "powers": {},
    }
    for p in powers:
        k0 = key_figures(run_k0(consumers, p, config), consumers, p, config)
        k1 = key_figures(run_k1(consumers, p, config), consumers, p, config)
        row: dict[str, Any] = {
            "K0": k0,
            "K1": k1,
            "V_N": (k1["use_kwh"] - k0["use_kwh"]) / k0["source_kwh"],
            "K2": {},
            "K3": {},
        }
        for s in sizes:
            if s > 0.0:
                r2 = run_k2(consumers, p, s, config)
                row["K2"][f"{s:g}"] = key_figures(r2, consumers, p, config)
            if with_k3:
                r3 = run_k3(consumers, p, s, config)
                row["K3"][f"{s:g}"] = key_figures(r3, consumers, p, config)
        out["powers"][f"{p:g}"] = row
    return out

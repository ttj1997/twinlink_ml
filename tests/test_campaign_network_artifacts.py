"""The campaign-network code after two changes still reproduces the artifacts of Befund 118 / 119.

Changes (decision of the project lead, 2026-11-18): the arithmetic of ``planning.cut_data`` is the
pure function ``cut_data_for_streams`` (called by the study and by the module interface alike),
and ``run_k2`` / ``rolling_plan`` take an optional initial content of the tank (default: empty).

What is checked: with the defaults the key figures of one cell with standstills (availability
0.57, independent schedules, draw 0, 400 kW) equal the stored ones **bit for bit** in the three
artifacts ``campaign_network_v1`` (Befund 118: K2), ``campaign_network_v2`` (addendum: K2b) and
``campaign_network_2_v1`` (Befund 119: K2b, P1, P2); the initial content does what it says; and a
run over consecutive hours equals two runs with the content handed over.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from twinlink_ml.campaign_network import (
    CampaignNetworkConfig,
    build_consumers,
    key_figures,
    load_campaign_network_config,
    plant_schedules,
    run_k1,
    run_k2,
)
from twinlink_ml.campaign_network.planning import (
    CutData,
    PlanForecast,
    PlanResult,
    TrueForecast,
    cut_data,
    cut_data_for_streams,
    planned_transitions,
    rolling_plan,
)
from twinlink_ml.campaign_network.profiles import ColdStream, Consumers

RUNS = Path(__file__).resolve().parents[1] / "artifacts" / "runs"
AVAILABILITY, MODE, DRAW, POWER_KW = 0.57, "independent", 0, 400.0
SIZES = (2.0, 8.0, 24.0)
HORIZON_H, REPLAN_EVERY_H = 168, 24  # scripts/run_campaign_network_2.py
PLAN_KEYS = (
    "source_kwh",
    "use_kwh",
    "utilisation",
    "gas_kwh",
    "co2_gas_kg",
    "pump_electricity_kwh",
    "co2_pump_kg",
    "co2_net_kg",
    "charge_kwh",
    "discharge_kwh",
    "content_end_kwh",
    "plans",
    "lp_failures",
)


def _artifact(name: str) -> dict[str, Any]:
    with gzip.open(RUNS / f"{name}.json.gz", "rt", encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded


def _cell(artifact: dict[str, Any], level: str) -> dict[str, Any]:
    key = f"draw:{level}:{AVAILABILITY:g}:{MODE}:{DRAW}"
    cell: dict[str, Any] = artifact["draws"][key]["powers"][f"{POWER_KW:g}"]
    return cell


def _plan_figures(res: PlanResult, cfg: CampaignNetworkConfig) -> dict[str, Any]:
    """``plan_figures`` of ``scripts/run_campaign_network_2.py``, without the solve time."""
    eta = cfg.plants[0].gas_efficiency
    use = float(res.use_kw.sum())
    source = POWER_KW * res.use_kw.size
    pump = cfg.pump_fraction * use
    gas = use / eta
    return {
        "source_kwh": source,
        "use_kwh": use,
        "utilisation": use / source,
        "gas_kwh": gas,
        "co2_gas_kg": cfg.gas_co2_kg_per_kwh * gas,
        "pump_electricity_kwh": pump,
        "co2_pump_kg": cfg.electricity_co2_kg_per_kwh * pump,
        "co2_net_kg": cfg.gas_co2_kg_per_kwh * gas - cfg.electricity_co2_kg_per_kwh * pump,
        "charge_kwh": float(res.charge_kw.sum()),
        "discharge_kwh": float(res.discharge_kw.sum()),
        "content_end_kwh": float(res.content_kwh[-1]),
        "plans": res.plans,
        "lp_failures": res.lp_failures,
    }


@pytest.fixture(scope="module")
def cfg() -> CampaignNetworkConfig:
    return load_campaign_network_config()


@pytest.fixture(scope="module")
def on(cfg: CampaignNetworkConfig) -> dict[str, np.ndarray]:
    return plant_schedules(cfg, AVAILABILITY, MODE, DRAW)


@pytest.fixture(scope="module")
def consumers(cfg: CampaignNetworkConfig, on: dict[str, np.ndarray]) -> Consumers:
    return build_consumers(cfg, on)


# ------------------------------------------------------------------ artifacts, bit for bit


def test_k2_reproduces_befund_118(cfg: CampaignNetworkConfig, consumers: Consumers) -> None:
    v1 = _artifact("campaign_network_v1")
    assert v1["fingerprints"]["main"] == cfg.fingerprint
    old = _cell(v1, "main")
    assert key_figures(run_k1(consumers, POWER_KW, cfg), consumers, POWER_KW, cfg) == old["K1"]
    for size in SIZES:
        new = key_figures(run_k2(consumers, POWER_KW, size, cfg), consumers, POWER_KW, cfg)
        assert new == old["K2"][f"{size:g}"]
        assert new["charge_kwh"] > 0.0


def test_k2b_reproduces_the_addendum_and_befund_119(
    cfg: CampaignNetworkConfig, consumers: Consumers
) -> None:
    v2, second = _artifact("campaign_network_v2"), _artifact("campaign_network_2_v1")
    assert v2["fingerprints"]["water_200"] == second["fingerprints"]["water_200"] == cfg.fingerprint
    for size in SIZES:
        new = key_figures(
            run_k2(consumers, POWER_KW, size, cfg, rule="k2b"), consumers, POWER_KW, cfg
        )
        assert new == _cell(v2, "water_200")["K2b"][f"{size:g}"]
        assert new == _cell(second, "water_200")["K2b"][f"{size:g}"]


def test_rolling_planning_reproduces_befund_119(
    cfg: CampaignNetworkConfig, on: dict[str, np.ndarray]
) -> None:
    """P1 (exact plan) and P2 (plan with deviations) of one cell: every figure but the time."""
    old = _cell(_artifact("campaign_network_2_v1"), "water_200")
    cuts = cut_data(cfg)
    size = 8.0
    exact = rolling_plan(cfg, cuts, on, TrueForecast(on), POWER_KW, size, HORIZON_H, REPLAN_EVERY_H)
    shifted = rolling_plan(
        cfg,
        cuts,
        on,
        PlanForecast(on, planned_transitions(cfg, AVAILABILITY, MODE, DRAW)),
        POWER_KW,
        size,
        HORIZON_H,
        REPLAN_EVERY_H,
    )
    for name, result in (("P1", exact), ("P2", shifted)):
        stored = old[name][f"{size:g}"]
        assert set(stored) == {*PLAN_KEYS, "solve_seconds"}
        assert _plan_figures(result, cfg) == {key: stored[key] for key in PLAN_KEYS}
    assert old["P2"]["8"]["use_kwh"] < old["P1"]["8"]["use_kwh"]  # the cell has plan deviations


# --------------------------------------------------------------------- the pure function


def test_cut_data_is_the_pure_function_on_the_consumers_of_the_configuration(
    cfg: CampaignNetworkConfig,
) -> None:
    plants = {plant.name for plant in cfg.plants}
    full = build_consumers(cfg, {name: np.ones(cfg.campaigns.hours, dtype=bool) for name in plants})
    direct = cut_data_for_streams(
        full.streams,
        plants,
        supply_celsius=cfg.source.supply_celsius,
        floor_celsius=cfg.source.floor_celsius,
        approach_k=cfg.approach_k,
    )
    wrapped = cut_data(cfg)
    assert np.array_equal(direct.f, wrapped.f)
    assert np.array_equal(direct.c_low, wrapped.c_low)
    assert set(direct.c_plant) == set(wrapped.c_plant) == plants
    assert all(np.array_equal(direct.c_plant[name], wrapped.c_plant[name]) for name in plants)


def test_cut_data_for_streams_by_hand() -> None:
    """One always-on and one schedulable stream, one hour: pieces at the four boundaries."""
    ones = np.ones(1)
    base = ColdStream("base", 20.0 * ones, 40.0 * ones, 2.0 * ones, 0.9, 40.0, "base")
    plant = ColdStream("plant", 10.0 * ones, 70.0 * ones, 3.0 * ones, 0.9, 150.0, "air")
    cuts = cut_data_for_streams(
        [base, plant], {"plant"}, supply_celsius=80.0, floor_celsius=50.0, approach_k=10.0
    )
    # shifted boundaries: supply 75, floor 45, base 25 / 45, plant 15 / 75
    assert cuts.f[0].tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    assert cuts.c_low[0].tolist() == [40.0, 40.0, 0.0, 40.0, 0.0, 40.0]
    assert cuts.c_plant["plant"][0].tolist() == [180.0, 90.0, 30.0, 90.0, 0.0, 180.0]
    with pytest.raises(ValueError, match="same efficiency"):
        cut_data_for_streams(
            [base, dataclasses.replace(plant, gas_efficiency=0.8)],
            {"plant"},
            supply_celsius=80.0,
            floor_celsius=50.0,
            approach_k=10.0,
        )


# --------------------------------------------------------------------- initial content


def _slice(consumers: Consumers, rows: slice) -> Consumers:
    return Consumers(
        streams=tuple(
            dataclasses.replace(s, t_in=s.t_in[rows], t_out=s.t_out[rows], cp=s.cp[rows])
            for s in consumers.streams
        ),
        ambient=consumers.ambient[rows],
    )


@pytest.mark.parametrize("rule", ["k2", "k2b"])
def test_k2_default_is_the_empty_tank(
    cfg: CampaignNetworkConfig, consumers: Consumers, rule: str
) -> None:
    window = _slice(consumers, slice(0, 24 * 21))
    default = run_k2(window, POWER_KW, 8.0, cfg, rule=rule)
    explicit = run_k2(window, POWER_KW, 8.0, cfg, rule=rule, initial_content_kwh=0.0)
    assert default.content_kwh[0] == 0.0
    assert np.array_equal(default.content_kwh, explicit.content_kwh)
    assert np.array_equal(default.charge_kw, explicit.charge_kw)
    assert np.array_equal(default.discharge_kw, explicit.discharge_kw)
    assert all(np.array_equal(default.duty_kw[k], explicit.duty_kw[k]) for k in default.duty_kw)


@pytest.mark.parametrize("rule", ["k2", "k2b"])
def test_k2_continues_from_a_handed_over_content(
    cfg: CampaignNetworkConfig, consumers: Consumers, rule: str
) -> None:
    """No foresight: one run over three weeks equals two runs with the content handed over."""
    hours, cut = 24 * 21, 24 * 9 + 5
    whole = run_k2(_slice(consumers, slice(0, hours)), POWER_KW, 8.0, cfg, rule=rule)
    first = run_k2(_slice(consumers, slice(0, cut)), POWER_KW, 8.0, cfg, rule=rule)
    handed_over = float(first.content_kwh[-1])
    second = run_k2(
        _slice(consumers, slice(cut, hours)),
        POWER_KW,
        8.0,
        cfg,
        rule=rule,
        initial_content_kwh=handed_over,
    )
    assert whole.charge_kw.sum() > 0.0 and whole.discharge_kw.sum() > 0.0
    assert second.content_kwh[0] == handed_over == whole.content_kwh[cut]
    assert np.array_equal(np.concatenate([first.charge_kw, second.charge_kw]), whole.charge_kw)
    assert np.array_equal(
        np.concatenate([first.discharge_kw, second.discharge_kw]), whole.discharge_kw
    )
    assert np.array_equal(
        np.concatenate([first.content_kwh, second.content_kwh[1:]]), whole.content_kwh
    )
    for name, duty in whole.duty_kw.items():
        assert np.array_equal(np.concatenate([first.duty_kw[name], second.duty_kw[name]]), duty)


def test_k2_initial_content_is_lost_and_used_as_stated(cfg: CampaignNetworkConfig) -> None:
    """One hour by hand (K2b): a deficit of 100 kW, 500 kWh in the tank at the start."""
    ones = np.ones(1)
    # demand 3 kW/K x 60 K = 180 kW; the source of 80 kW covers 80 kW of it
    stream = ColdStream("plant", 10.0 * ones, 70.0 * ones, 3.0 * ones, 0.9, 150.0, "air")
    consumers = Consumers(streams=(stream,), ambient=10.0 * ones)
    keep = 1.0 - cfg.storage.loss_per_h
    empty = run_k2(consumers, 80.0, 10.0, cfg, rule="k2b")
    filled = run_k2(consumers, 80.0, 10.0, cfg, rule="k2b", initial_content_kwh=500.0)
    assert empty.discharge_kw[0] == 0.0
    deficit = 180.0 - empty.duty_kw["plant"][0]
    assert deficit > 0.0
    expected = min(deficit, keep * 500.0, cfg.storage.discharge_max_fraction * 80.0)
    assert filled.discharge_kw[0] == pytest.approx(expected)
    assert filled.content_kwh.tolist() == pytest.approx([500.0, keep * 500.0 - expected])
    assert filled.duty_kw["plant"][0] > empty.duty_kw["plant"][0]


def test_initial_content_outside_the_tank_is_refused(
    cfg: CampaignNetworkConfig, consumers: Consumers, on: dict[str, np.ndarray]
) -> None:
    window = _slice(consumers, slice(0, 48))
    capacity = 8.0 * POWER_KW
    for bad in (-1.0, capacity + 1.0):
        with pytest.raises(ValueError, match="initial_content_kwh"):
            run_k2(window, POWER_KW, 8.0, cfg, initial_content_kwh=bad)
    cuts = cut_data(cfg)
    for bad in (-1.0, capacity + 1.0):
        with pytest.raises(ValueError, match="initial_content_kwh"):
            rolling_plan(
                cfg, cuts, on, TrueForecast(on), POWER_KW, 8.0, 168, 24, initial_content_kwh=bad
            )


def _slice_cuts(cuts: CutData, rows: slice) -> CutData:
    return CutData(
        f=cuts.f[rows],
        c_low=cuts.c_low[rows],
        c_plant={name: part[rows] for name, part in cuts.c_plant.items()},
    )


def test_rolling_plan_continues_from_a_handed_over_content(
    cfg: CampaignNetworkConfig, on: dict[str, np.ndarray]
) -> None:
    """Cut at a planning hour, with the horizon reaching the end in both parts: the same plans."""
    hours, cut, horizon, every = 24 * 21, 24 * 9, 24 * 21, 24
    cuts = cut_data(cfg)

    def run(rows: slice, start: float) -> PlanResult:
        states = {name: state[rows] for name, state in on.items()}
        return rolling_plan(
            cfg,
            _slice_cuts(cuts, rows),
            states,
            TrueForecast(states),
            POWER_KW,
            8.0,
            horizon,
            every,
            initial_content_kwh=start,
        )

    whole = run(slice(0, hours), 0.0)
    default = rolling_plan(
        cfg,
        _slice_cuts(cuts, slice(0, hours)),
        {name: state[:hours] for name, state in on.items()},
        TrueForecast({name: state[:hours] for name, state in on.items()}),
        POWER_KW,
        8.0,
        horizon,
        every,
    )
    assert np.array_equal(default.use_kw, whole.use_kw)
    assert np.array_equal(default.content_kwh, whole.content_kwh)

    first = run(slice(0, cut), 0.0)
    assert first.content_kwh[0] == 0.0
    second = run(slice(cut, hours), float(whole.content_kwh[cut]))
    assert second.content_kwh[0] == whole.content_kwh[cut]
    assert whole.charge_kw.sum() > 0.0
    # the second part plans over the same hours with the same horizon as the whole run does from
    # the cut on, and starts with the same content
    assert np.array_equal(second.use_kw, whole.use_kw[cut:])
    assert np.array_equal(second.charge_kw, whole.charge_kw[cut:])
    assert np.array_equal(second.discharge_kw, whole.discharge_kw[cut:])
    assert np.array_equal(second.content_kwh, whole.content_kwh[cut:])
    assert first.use_kw.size == cut

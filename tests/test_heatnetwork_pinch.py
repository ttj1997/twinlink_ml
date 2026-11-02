"""Tests for ``heatnetwork/pinch.py`` -- problem table algorithm (PA5).

Covers: agreement with a four-stream problem worked through by hand (constant
cp, no phase change -- checks the cascade core in isolation from the condensation physics),
the limiting case ``dT_min = 0``, two monotonicities (``dT_min`` and ``T_kalt`` must never
lower the heating demand) and the most important property of all: the pinch target must
never be larger than the best of the six interconnection cases of the partner model (PA2/PA3)
-- if this test fails, the implementation is wrong, not the cases.

The four-stream problem in ``test_matches_hand_calculation`` is SELF-constructed (round
numbers, the result derived step by step below), not quoted from a particular
textbook -- a misremembered literature number would be a worse yardstick
than a traceable calculation of our own.
"""

from __future__ import annotations

import pytest

from twinlink_ml.heatnetwork.balance import SinkState, SourceState, evaluate_cascade
from twinlink_ml.heatnetwork.config import HeatNetworkConfig, HeatSinkConfig, HeatSourceConfig
from twinlink_ml.heatnetwork.pinch import (
    StreamSegment,
    cold_segment_for_sink,
    compute_pinch_target,
    dew_point_of_source_celsius,
    evaluate_pinch_target,
    hot_segments_for_source,
)
from twinlink_ml.simulation.config import GasConfig

GAS = GasConfig()

# --- PA2 data case (corrected values, Befund NN draft see docs/04_stand_ap1.md) ------------

SOURCE_1 = HeatSourceConfig("Q1", 20.0, 125.233, 0.06482)
SOURCE_2 = HeatSourceConfig("Q2", 8.0, 81.416, 0.01998)
SINK_1 = HeatSinkConfig("S1", 20.0, 0.005, 15.0, 300.0)
SINK_2 = HeatSinkConfig("S2", 8.0, 0.005, 15.0, 120.0)

FUEL_LOWER_HEATING_VALUE_J_PER_KG = 45.0e6
HEATER_EFFICIENCY = 0.80

CASES: dict[str, tuple[tuple[str, str], ...]] = {
    "Fall1": (("Q1", "S1"), ("Q2", "S2")),
    "Fall2": (("Q2", "S1"), ("Q1", "S2")),
    "Fall3": (("Q2", "S1"), ("Q1", "S1")),
    "Fall4": (("Q2", "S2"), ("Q1", "S2")),
    "Fall5": (("Q1", "S1"), ("Q2", "S1")),
    "Fall6": (("Q1", "S2"),),
}


def _fuel_kg_per_h_for_case(
    stages: tuple[tuple[str, str], ...], *, cold_temperature_celsius: float, minimum_pinch_k: float
) -> float:
    network = HeatNetworkConfig(
        sources=(SOURCE_1, SOURCE_2),
        sinks=(SINK_1, SINK_2),
        minimum_pinch_k=minimum_pinch_k,
        recovery_mode="condensing",
        cold_temperature_celsius=cold_temperature_celsius,
        fuel_lower_heating_value_j_per_kg=FUEL_LOWER_HEATING_VALUE_J_PER_KG,
        heater_efficiency=HEATER_EFFICIENCY,
    )
    sources = tuple(SourceState.from_config(cfg) for cfg in network.sources)
    sinks = tuple(SinkState.from_config(cfg) for cfg in network.sinks)
    result = evaluate_cascade(network, stages, sources, sinks)
    return float(result.total_fuel_mass_flow_kg_per_s()[0]) * 3600.0


def _pinch_fuel_kg_per_h(*, cold_temperature_celsius: float, minimum_pinch_k: float) -> float:
    result, _, _ = evaluate_pinch_target(
        (SOURCE_1, SOURCE_2),
        (SINK_1, SINK_2),
        GAS,
        cold_temperature_celsius=cold_temperature_celsius,
        minimum_pinch_k=minimum_pinch_k,
    )
    return (
        result.minimum_hot_utility_kw
        * 1000.0
        / (FUEL_LOWER_HEATING_VALUE_J_PER_KG * HEATER_EFFICIENCY)
        * 3600.0
    )


# --- Textbook case: four-stream problem, constant cp -----------------------------------
#
# Hand calculation (dT_min = 10 K, half shift = 5 K):
#
#   H1  200 -> 100 C   CP=1.0 kW/K   (shifted [195,95])
#   H2  150 ->  50 C   CP=2.0 kW/K   (shifted [145,45])
#   C1   50 -> 190 C   CP=1.5 kW/K   (shifted [55,195])
#   C2   20 -> 120 C   CP=1.0 kW/K   (shifted [25,125])
#
# Shifted boundaries in descending order: 195, 145, 125, 95, 55, 45, 25
#
#   Interval      dT   CP_hot    CP_cold   net (hot-cold)*dT
#   195-145       50    1.0       1.5      -25
#   145-125       20    3.0       1.5      +30
#   125-95        30    3.0       2.5      +15
#    95-55        40    2.0       2.5      -20
#    55-45        10    2.0       1.0      +10
#    45-25        20    0.0       1.0      -20
#
#   Cascade (start 0 at 195 C):
#     195:  0
#     145: -25   <- minimum
#     125:   5
#     95:   20
#     55:    0
#     45:   10
#     25:  -10
#
#   minimum heating demand = 25 kW (magnitude of the minimum at 145 C shifted)
#   cascade + 25:  25, 0, 30, 45, 25, 35, 15
#   minimum cooling demand = 15 kW (final value)
#   pinch at 145 C shifted -> hot 150 C, cold 140 C (difference = dT_min, check)
#
#   Energy balance check: sum CP*dT hot = 100+200=300 kW,
#   sum CP*dT cold = 210+100=310 kW; Hot - Cold Utility = 25-15=10 = 310-300 (check).


def _textbook_segments() -> tuple[tuple[StreamSegment, ...], tuple[StreamSegment, ...]]:
    hot = (
        StreamSegment("H1", True, 200.0, 100.0, 1.0),
        StreamSegment("H2", True, 150.0, 50.0, 2.0),
    )
    cold = (
        StreamSegment("C1", False, 190.0, 50.0, 1.5),
        StreamSegment("C2", False, 120.0, 20.0, 1.0),
    )
    return hot, cold


def test_matches_hand_calculation() -> None:
    hot, cold = _textbook_segments()
    result = compute_pinch_target(hot, cold, minimum_pinch_k=10.0)

    assert result.minimum_hot_utility_kw == pytest.approx(25.0)
    assert result.minimum_cold_utility_kw == pytest.approx(15.0)
    assert result.pinch_temperature_hot_side_celsius == pytest.approx(150.0)
    assert result.pinch_temperature_cold_side_celsius == pytest.approx(140.0)
    pinch_hot = result.pinch_temperature_hot_side_celsius
    pinch_cold = result.pinch_temperature_cold_side_celsius
    assert pinch_hot - pinch_cold == pytest.approx(10.0)
    # Energy balance check (see hand calculation above)
    utility_gap_kw = result.minimum_hot_utility_kw - result.minimum_cold_utility_kw
    assert utility_gap_kw == pytest.approx(310.0 - 300.0)


# --- Limiting case dT_min = 0 ----------------------------------------------------------


def test_zero_pinch_gives_total_demand_minus_total_offer_when_offer_is_smaller() -> None:
    # One hot and one cold stream with an IDENTICAL temperature range (150 -> 30 C):
    # at dT_min=0 there is only a single interval, the cascade falls monotonically, its
    # minimum lies exactly at the lower end -- the minimum heating demand is then by
    # construction exactly the difference between total supply and total demand.
    hot = (StreamSegment("H", True, 150.0, 30.0, 1.0),)  # supply = 120 kW
    cold = (StreamSegment("C", False, 150.0, 30.0, 1.5),)  # demand = 180 kW
    result = compute_pinch_target(hot, cold, minimum_pinch_k=0.0)

    assert result.minimum_hot_utility_kw == pytest.approx(180.0 - 120.0)
    assert result.minimum_cold_utility_kw == pytest.approx(0.0)


# --- Monotonicity ------------------------------------------------------------------------


def test_larger_minimum_pinch_never_lowers_hot_utility() -> None:
    hot, cold = _textbook_segments()
    utilities = [
        compute_pinch_target(hot, cold, minimum_pinch_k=dtmin).minimum_hot_utility_kw
        for dtmin in (0.0, 5.0, 10.0, 20.0, 30.0)
    ]
    assert utilities == sorted(utilities)


def test_higher_cold_side_temperature_never_lowers_hot_utility() -> None:
    utilities = []
    for cold_temperature_celsius in (15.0, 25.0, 29.1, 35.0):
        result, _, _ = evaluate_pinch_target(
            (SOURCE_1, SOURCE_2),
            (SINK_1, SINK_2),
            GAS,
            cold_temperature_celsius=cold_temperature_celsius,
            minimum_pinch_k=15.0,
        )
        utilities.append(result.minimum_hot_utility_kw)
    assert utilities == sorted(utilities)


# --- Core statement: pinch target <= best of the six cases ------------------------------


@pytest.mark.parametrize("cold_temperature_celsius", [15.0, 25.0, 29.1, 35.0])
@pytest.mark.parametrize("minimum_pinch_k", [10.0, 15.0, 20.0])
def test_pinch_target_never_exceeds_the_best_of_six_cases(
    cold_temperature_celsius: float, minimum_pinch_k: float
) -> None:
    pinch_fuel = _pinch_fuel_kg_per_h(
        cold_temperature_celsius=cold_temperature_celsius, minimum_pinch_k=minimum_pinch_k
    )
    best_case_fuel = min(
        _fuel_kg_per_h_for_case(
            stages,
            cold_temperature_celsius=cold_temperature_celsius,
            minimum_pinch_k=minimum_pinch_k,
        )
        for stages in CASES.values()
    )
    assert pinch_fuel <= best_case_fuel + 1e-6


# --- Sampling: dew point as exact interval boundary, resolution sensitivity --------------


def test_dew_point_is_an_exact_segment_boundary() -> None:
    dew_point = dew_point_of_source_celsius(SOURCE_1, GAS)
    segments = hot_segments_for_source(SOURCE_1, GAS, 15.0, n_segments_below_dew_point=5)
    boundaries = {round(seg.high_temperature_celsius, 6) for seg in segments} | {
        round(seg.low_temperature_celsius, 6) for seg in segments
    }
    assert round(dew_point, 6) in boundaries


def test_resolution_below_dew_point_converges() -> None:
    # Coarse vs. fine sampling below the dew point -- the minimum heating demand must no
    # longer change appreciably once the resolution is fine enough (PA1: show the
    # change between coarse and fine, not just claim a value).
    coarse_hot = tuple(
        segment
        for source in (SOURCE_1, SOURCE_2)
        for segment in hot_segments_for_source(source, GAS, 15.0, n_segments_below_dew_point=2)
    )
    fine_hot = tuple(
        segment
        for source in (SOURCE_1, SOURCE_2)
        for segment in hot_segments_for_source(source, GAS, 15.0, n_segments_below_dew_point=200)
    )
    cold = (cold_segment_for_sink(SINK_1, GAS), cold_segment_for_sink(SINK_2, GAS))
    coarse_result = compute_pinch_target(coarse_hot, cold, minimum_pinch_k=15.0)
    fine_result = compute_pinch_target(fine_hot, cold, minimum_pinch_k=15.0)

    coarse_kw = coarse_result.minimum_hot_utility_kw
    fine_kw = fine_result.minimum_hot_utility_kw
    relative_change = abs(coarse_kw - fine_kw) / fine_kw
    assert relative_change < 0.02

"""E1: duty limit of a shared exchanger in the cascade; without a limit nothing changes."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from twinlink_ml.heatnetwork import BINDING_EXCHANGER_CAP, SharedCap, binding_label
from twinlink_ml.heatnetwork.balance import SinkState, SourceState, evaluate_cascade
from twinlink_ml.heatnetwork.config import HeatNetworkConfig, HeatSinkConfig, HeatSourceConfig
from twinlink_ml.heatnetwork.routing import FixedRouting

Q1 = HeatSourceConfig("Q1", 20.0, 125.233, 0.06482)
S1 = HeatSinkConfig("S1", 20.0, 0.005, 15.0, 300.0)
SA = HeatSinkConfig("SA", 8.0, 0.008, 12.0, 90.0)
SB = HeatSinkConfig("SB", 8.0, 0.008, 12.0, 100.0)


def _run(connections, shared_caps=None, recovery_mode="condensing"):
    network = HeatNetworkConfig(
        sources=(Q1,),
        sinks=(S1, SA, SB),
        minimum_pinch_k=20.0,
        recovery_mode=recovery_mode,
        cold_temperature_celsius=29.1,
        fuel_lower_heating_value_j_per_kg=45.0e6,
        heater_efficiency=0.80,
    )
    stages = FixedRouting(connections).route(network)
    sources = (SourceState.from_config(Q1),)
    sinks = tuple(SinkState.from_config(c) for c in (S1, SA, SB))
    if shared_caps is None:
        return evaluate_cascade(network, stages, sources, sinks)
    return evaluate_cascade(network, stages, sources, sinks, shared_caps)


ALL = (("Q1", "S1"), ("Q1", "SA"), ("Q1", "SB"))
CAPPED = (("Q1", "SA"), ("Q1", "SB"))


def _equal(a, b) -> bool:
    return all(
        np.array_equal(getattr(a, f.name), getattr(b, f.name))
        for f in dataclasses.fields(a)
        if isinstance(getattr(a, f.name), np.ndarray)
    )


def test_no_limit_is_identical_to_the_call_without_the_argument() -> None:
    assert _equal(_run(ALL), _run(ALL, shared_caps=()))


def test_a_limit_above_the_demand_changes_nothing_bit_for_bit() -> None:
    free = _run(ALL)
    capped = _run(ALL, (SharedCap(CAPPED, 1.0e9),))
    assert _equal(free, capped)


def test_a_binding_limit_is_exhausted_by_the_stages_in_order() -> None:
    free = _run(ALL)
    uptake_free = [p[0] for p in free.stage_recovered_power_kw[1:]]
    assert sum(uptake_free) > 20.0
    cap_kw = 0.5 * sum(uptake_free)
    capped = _run(ALL, (SharedCap(CAPPED, cap_kw),))
    group = capped.stage_recovered_power_kw[1:, 0]
    assert group.sum() == pytest.approx(cap_kw, rel=1e-9)
    # the first line stage takes what it can up to the limit, the second gets the rest
    assert group[0] == pytest.approx(min(uptake_free[0], cap_kw), rel=1e-9)
    assert capped.stage_recovered_power_kw[0, 0] == free.stage_recovered_power_kw[0, 0]
    assert BINDING_EXCHANGER_CAP in capped.stage_binding_code[1:, 0]
    assert binding_label(BINDING_EXCHANGER_CAP) == "exchanger_cap"
    # less recovery means more fuel at the sinks of the group, none less
    assert capped.fuel_mass_flow_kg_per_s[1:, 0].sum() > free.fuel_mass_flow_kg_per_s[1:, 0].sum()
    assert capped.fuel_mass_flow_kg_per_s[0, 0] == free.fuel_mass_flow_kg_per_s[0, 0]


def test_a_zero_limit_blocks_the_group() -> None:
    capped = _run(ALL, (SharedCap(CAPPED, 0.0),))
    assert np.all(capped.stage_recovered_power_kw[1:] == 0.0)


def test_stages_that_are_not_in_the_routing_are_unaffected() -> None:
    only_s1 = (("Q1", "S1"),)
    assert _equal(_run(only_s1), _run(only_s1, (SharedCap(CAPPED, 0.0),)))


def test_unknown_names_and_negative_limits_are_rejected() -> None:
    with pytest.raises(ValueError):
        SharedCap((), 1.0)
    with pytest.raises(ValueError):
        SharedCap(CAPPED, -1.0)
    with pytest.raises(ValueError):
        _run(ALL, (SharedCap((("Q1", "SX"),), 1.0),))

"""Tests für ``heatnetwork/routing.py`` -- Verschaltung Quelle -> Senke."""

from __future__ import annotations

import numpy as np
import pytest

from twinlink_ml.heatnetwork.config import HeatNetworkConfig, HeatSinkConfig, HeatSourceConfig
from twinlink_ml.heatnetwork.routing import FixedRouting, MatrixRouting, resolve_cascade_order


def _source(name: str, outlet_temperature_celsius: float) -> HeatSourceConfig:
    return HeatSourceConfig(
        name=name,
        dry_air_mass_flow_kg_per_s=8.0,
        outlet_temperature_celsius=outlet_temperature_celsius,
        outlet_humidity_kg_per_kg=0.02,
    )


def _sink(name: str) -> HeatSinkConfig:
    return HeatSinkConfig(
        name=name,
        dry_air_mass_flow_kg_per_s=8.0,
        humidity_kg_per_kg=0.008,
        intake_temperature_celsius=15.0,
        setpoint_temperature_celsius=90.0,
    )


def _network(**overrides: object) -> HeatNetworkConfig:
    values: dict[str, object] = {
        "sources": (_source("cold", 60.0), _source("hot", 95.0)),
        "sinks": (_sink("a"), _sink("b")),
    }
    values.update(overrides)
    return HeatNetworkConfig(**values)  # type: ignore[arg-type]


def test_resolve_cascade_order_defaults_to_ascending_outlet_temperature() -> None:
    network = _network()
    assert resolve_cascade_order(network) == ("cold", "hot")


def test_resolve_cascade_order_respects_explicit_override() -> None:
    network = _network(cascade_order=("hot", "cold"))
    assert resolve_cascade_order(network) == ("hot", "cold")


def test_resolve_cascade_order_is_stable_for_equal_temperatures() -> None:
    network = _network(sources=(_source("first", 80.0), _source("second", 80.0)))
    assert resolve_cascade_order(network) == ("first", "second")


def test_fixed_routing_reorders_by_ascending_source_temperature_regardless_of_input_order() -> (
    None
):
    network = _network()
    routing = FixedRouting(connections=(("hot", "a"), ("cold", "a")))
    assert routing.route(network) == (("cold", "a"), ("hot", "a"))


def test_fixed_routing_preserves_relative_order_of_connections_from_the_same_source() -> None:
    network = _network()
    routing = FixedRouting(connections=(("cold", "b"), ("cold", "a"), ("hot", "a")))
    assert routing.route(network) == (("cold", "b"), ("cold", "a"), ("hot", "a"))


def test_fixed_routing_rejects_unknown_source_name() -> None:
    network = _network()
    routing = FixedRouting(connections=(("unknown", "a"),))
    with pytest.raises(ValueError, match="unbekannte Quelle"):
        routing.route(network)


def test_fixed_routing_rejects_unknown_sink_name() -> None:
    network = _network()
    routing = FixedRouting(connections=(("cold", "unknown"),))
    with pytest.raises(ValueError, match="unbekannte Senke"):
        routing.route(network)


def test_matrix_routing_selects_entries_above_threshold_only() -> None:
    network = _network()
    assignment = np.array([[1.0, 0.0], [0.0, 1.0]])
    routing = MatrixRouting(assignment=assignment, threshold=0.5)
    assert set(routing.route(network)) == {("cold", "a"), ("hot", "b")}


def test_matrix_routing_orders_by_ascending_source_temperature_first() -> None:
    network = _network()
    assignment = np.array([[1.0, 1.0], [1.0, 1.0]])
    stages = MatrixRouting(assignment=assignment).route(network)
    source_order = [source_name for source_name, _ in stages]
    cold_positions = [i for i, name in enumerate(source_order) if name == "cold"]
    hot_positions = [i for i, name in enumerate(source_order) if name == "hot"]
    assert max(cold_positions) < min(hot_positions)


def test_matrix_routing_breaks_ties_by_descending_weight_within_a_source() -> None:
    network = _network()
    assignment = np.array([[0.9, 0.1], [0.0, 0.0]])
    stages = MatrixRouting(assignment=assignment, threshold=0.0).route(network)
    assert stages == (("cold", "a"), ("cold", "b"))


def test_matrix_routing_rejects_wrong_shape() -> None:
    network = _network()
    routing = MatrixRouting(assignment=np.zeros((1, 1)))
    with pytest.raises(ValueError, match="assignment"):
        routing.route(network)

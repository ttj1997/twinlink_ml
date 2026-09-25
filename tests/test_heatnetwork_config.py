"""Tests für ``heatnetwork/config.py`` -- Validierung, Fingerprint-Mechanik, Platzhalterwarnung."""

from __future__ import annotations

import pytest

from twinlink_ml.heatnetwork.config import (
    DEFAULT_COLD_TEMPERATURE_CELSIUS,
    DEFAULT_MINIMUM_PINCH_K,
    DEFAULT_RECOVERY_MODE,
    HeatNetworkConfig,
    HeatSinkConfig,
    HeatSourceConfig,
)
from twinlink_ml.simulation.config import PlaceholderConfigWarning, Provenance


def _source(name: str = "line_1_exhaust", **overrides: float) -> HeatSourceConfig:
    values: dict[str, float] = {
        "dry_air_mass_flow_kg_per_s": 8.0,
        "outlet_temperature_celsius": 85.0,
        "outlet_humidity_kg_per_kg": 0.02,
    }
    values.update(overrides)
    return HeatSourceConfig(name=name, **values)  # type: ignore[arg-type]


def _sink(name: str = "line_1_heater", **overrides: float) -> HeatSinkConfig:
    values: dict[str, float] = {
        "dry_air_mass_flow_kg_per_s": 8.0,
        "humidity_kg_per_kg": 0.008,
        "intake_temperature_celsius": 15.0,
        "setpoint_temperature_celsius": 90.0,
    }
    values.update(overrides)
    return HeatSinkConfig(name=name, **values)  # type: ignore[arg-type]


def test_heat_source_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        _source(name="")


def test_heat_source_config_rejects_non_positive_mass_flow() -> None:
    with pytest.raises(ValueError, match="dry_air_mass_flow_kg_per_s"):
        _source(dry_air_mass_flow_kg_per_s=0.0)


def test_heat_source_config_rejects_negative_humidity() -> None:
    with pytest.raises(ValueError, match="outlet_humidity_kg_per_kg"):
        _source(outlet_humidity_kg_per_kg=-0.001)


def test_heat_sink_config_rejects_setpoint_at_or_below_intake() -> None:
    with pytest.raises(ValueError, match="setpoint_temperature_celsius"):
        _sink(setpoint_temperature_celsius=15.0, intake_temperature_celsius=15.0)


def test_heat_sink_config_rejects_negative_humidity() -> None:
    with pytest.raises(ValueError, match="humidity_kg_per_kg"):
        _sink(humidity_kg_per_kg=-0.001)


def test_heat_network_config_requires_at_least_one_source_and_sink() -> None:
    with pytest.raises(ValueError, match="sources"):
        HeatNetworkConfig(sources=(), sinks=(_sink(),))
    with pytest.raises(ValueError, match="sinks"):
        HeatNetworkConfig(sources=(_source(),), sinks=())


def test_heat_network_config_rejects_duplicate_source_names() -> None:
    with pytest.raises(ValueError, match="Quellennamen"):
        HeatNetworkConfig(sources=(_source(name="a"), _source(name="a")), sinks=(_sink(),))


def test_heat_network_config_rejects_duplicate_sink_names() -> None:
    with pytest.raises(ValueError, match="Senkennamen"):
        HeatNetworkConfig(sources=(_source(),), sinks=(_sink(name="a"), _sink(name="a")))


def test_heat_network_config_rejects_cascade_order_not_matching_sources() -> None:
    with pytest.raises(ValueError, match="cascade_order"):
        HeatNetworkConfig(
            sources=(_source(name="a"), _source(name="b")),
            sinks=(_sink(),),
            cascade_order=("a", "c"),
        )


def test_heat_network_config_default_minimum_pinch_is_twenty_kelvin() -> None:
    network = HeatNetworkConfig(sources=(_source(),), sinks=(_sink(),))
    assert network.minimum_pinch_k == DEFAULT_MINIMUM_PINCH_K == 20.0


def test_heat_network_config_rejects_non_positive_minimum_pinch() -> None:
    with pytest.raises(ValueError, match="minimum_pinch_k"):
        HeatNetworkConfig(sources=(_source(),), sinks=(_sink(),), minimum_pinch_k=0.0)


def test_heat_network_config_rejects_heater_efficiency_out_of_range() -> None:
    with pytest.raises(ValueError, match="heater_efficiency"):
        HeatNetworkConfig(sources=(_source(),), sinks=(_sink(),), heater_efficiency=1.5)


def test_heat_network_config_default_recovery_mode_is_sensible() -> None:
    network = HeatNetworkConfig(sources=(_source(),), sinks=(_sink(),))
    assert network.recovery_mode == DEFAULT_RECOVERY_MODE == "sensible"


def test_heat_network_config_default_cold_temperature_is_fifteen_celsius() -> None:
    network = HeatNetworkConfig(sources=(_source(),), sinks=(_sink(),))
    assert network.cold_temperature_celsius == DEFAULT_COLD_TEMPERATURE_CELSIUS == 15.0


def test_heat_network_config_rejects_unknown_recovery_mode() -> None:
    with pytest.raises(ValueError, match="recovery_mode"):
        HeatNetworkConfig(
            sources=(_source(),), sinks=(_sink(),), recovery_mode="both"  # type: ignore[arg-type]
        )


def test_heat_network_config_rejects_cold_temperature_out_of_range() -> None:
    with pytest.raises(ValueError, match="cold_temperature_celsius"):
        HeatNetworkConfig(sources=(_source(),), sinks=(_sink(),), cold_temperature_celsius=150.0)


def test_fingerprint_changes_with_recovery_mode() -> None:
    base = HeatNetworkConfig(sources=(_source(),), sinks=(_sink(),))
    changed = HeatNetworkConfig(
        sources=(_source(),), sinks=(_sink(),), recovery_mode="condensing"
    )
    assert base.fingerprint != changed.fingerprint


def test_fingerprint_changes_with_cold_temperature() -> None:
    base = HeatNetworkConfig(sources=(_source(),), sinks=(_sink(),))
    changed = HeatNetworkConfig(
        sources=(_source(),), sinks=(_sink(),), cold_temperature_celsius=10.0
    )
    assert base.fingerprint != changed.fingerprint


def test_fingerprint_is_stable_and_deterministic() -> None:
    network_a = HeatNetworkConfig(sources=(_source(),), sinks=(_sink(),))
    network_b = HeatNetworkConfig(sources=(_source(),), sinks=(_sink(),))
    assert network_a.fingerprint == network_b.fingerprint
    assert len(network_a.fingerprint) == 16


def test_fingerprint_changes_with_minimum_pinch_k() -> None:
    base = HeatNetworkConfig(sources=(_source(),), sinks=(_sink(),))
    changed = HeatNetworkConfig(sources=(_source(),), sinks=(_sink(),), minimum_pinch_k=15.0)
    assert base.fingerprint != changed.fingerprint


def test_section_fingerprints_cover_all_eleven_fields() -> None:
    network = HeatNetworkConfig(sources=(_source(),), sinks=(_sink(),))
    sections = network.section_fingerprints
    assert set(sections) == {
        "provenance",
        "gas",
        "sources",
        "sinks",
        "minimum_pinch_k",
        "dew_point_approach_k",
        "fuel_lower_heating_value_j_per_kg",
        "heater_efficiency",
        "recovery_mode",
        "cold_temperature_celsius",
        "cascade_order",
    }
    assert all(isinstance(value, str) and value for value in sections.values())


def test_section_fingerprints_do_not_change_overall_fingerprint() -> None:
    network = HeatNetworkConfig(sources=(_source(),), sinks=(_sink(),))
    fingerprint_before = network.fingerprint
    _ = network.section_fingerprints
    assert network.fingerprint == fingerprint_before


def test_section_fingerprints_isolate_the_changed_section() -> None:
    base = HeatNetworkConfig(sources=(_source(),), sinks=(_sink(),))
    changed = HeatNetworkConfig(sources=(_source(),), sinks=(_sink(),), minimum_pinch_k=15.0)
    before = base.section_fingerprints
    after = changed.section_fingerprints
    differing = {key for key in before if before[key] != after[key]}
    assert differing == {"minimum_pinch_k"}


def test_warn_if_placeholder_warns_by_default() -> None:
    network = HeatNetworkConfig(sources=(_source(),), sinks=(_sink(),))
    with pytest.warns(PlaceholderConfigWarning):
        network.warn_if_placeholder()


def test_warn_if_placeholder_silent_when_calibrated(recwarn: pytest.WarningsRecorder) -> None:
    network = HeatNetworkConfig(
        provenance=Provenance(status="calibrated"), sources=(_source(),), sinks=(_sink(),)
    )
    network.warn_if_placeholder()
    assert len(recwarn) == 0

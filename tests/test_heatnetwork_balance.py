"""Tests for ``heatnetwork/balance.py`` -- cascade physics (EA5).

Covers: energy conservation per source, approach-temperature and dew-point invariants per stage,
agreement with the single-stage formula (EA1) in the one-source/one-sink case,
the ordering rule (low temperature first must never be worse), bit-identical
vectorisation over time steps, and the evidence that the cascading (correction 2)
needs strictly less fuel than the uncorrected Excel template.
"""

from __future__ import annotations

import numpy as np
import pytest

from twinlink_ml.heatnetwork.balance import (
    BINDING_DEW_POINT,
    HeatNetworkResult,
    SinkState,
    SourceState,
    evaluate_cascade,
)
from twinlink_ml.heatnetwork.config import HeatNetworkConfig, HeatSinkConfig, HeatSourceConfig
from twinlink_ml.simulation.backend import NumpyBackend
from twinlink_ml.simulation.config import GasConfig
from twinlink_ml.simulation.thermo import (
    dew_point_celsius,
    latent_heat_j_per_kg,
    saturation_humidity_kg_per_kg,
)

GAS = GasConfig()
BACKEND = NumpyBackend()


def _source(name: str, temp: float, humidity: float, mass_flow: float = 8.0) -> HeatSourceConfig:
    return HeatSourceConfig(
        name=name,
        dry_air_mass_flow_kg_per_s=mass_flow,
        outlet_temperature_celsius=temp,
        outlet_humidity_kg_per_kg=humidity,
    )


def _sink(
    name: str,
    setpoint: float,
    *,
    intake: float = 15.0,
    humidity: float = 0.008,
    mass_flow: float = 8.0,
) -> HeatSinkConfig:
    return HeatSinkConfig(
        name=name,
        dry_air_mass_flow_kg_per_s=mass_flow,
        humidity_kg_per_kg=humidity,
        intake_temperature_celsius=intake,
        setpoint_temperature_celsius=setpoint,
    )


def _dew_point(humidity: float) -> float:
    return float(
        BACKEND.to_numpy(dew_point_celsius(BACKEND, GAS, BACKEND.asarray(np.array([humidity]))))[0]
    )


def _saturation_humidity(temperature_celsius: float) -> float:
    return float(
        BACKEND.to_numpy(
            saturation_humidity_kg_per_kg(BACKEND, GAS, BACKEND.asarray(temperature_celsius))
        )
    )


def _latent_heat(temperature_celsius: float) -> float:
    return float(
        BACKEND.to_numpy(latent_heat_j_per_kg(BACKEND, BACKEND.asarray(temperature_celsius)))
    )


def _states(
    network: HeatNetworkConfig, n_steps: int = 1
) -> tuple[tuple[SourceState, ...], tuple[SinkState, ...]]:
    sources = tuple(SourceState.from_config(cfg, n_steps) for cfg in network.sources)
    sinks = tuple(SinkState.from_config(cfg, n_steps) for cfg in network.sinks)
    return sources, sinks


# --- Energy conservation ------------------------------------------------------------


def test_used_plus_discarded_equals_available_per_source() -> None:
    network = HeatNetworkConfig(
        sources=(_source("cold", 60.0, 0.02), _source("hot", 95.0, 0.02, mass_flow=1.0)),
        sinks=(_sink("a", 45.0, mass_flow=2.0), _sink("b", 90.0)),
    )
    sources, sinks = _states(network)
    stages = (("cold", "a"), ("cold", "b"), ("hot", "b"))
    result = evaluate_cascade(network, stages, sources, sinks)

    total = result.source_used_power_kw + result.source_discarded_power_kw
    assert total == pytest.approx(result.source_available_power_kw, rel=1e-9, abs=1e-9)


# --- Approach temperature and dew-point limit ------------------------------------------


def test_no_stage_violates_the_minimum_pinch() -> None:
    network = HeatNetworkConfig(
        sources=(_source("cold", 55.0, 0.02), _source("hot", 100.0, 0.02)),
        sinks=(_sink("only", 95.0, intake=5.0),),
        minimum_pinch_k=20.0,
    )
    sources, sinks = _states(network)
    stages = (("cold", "only"), ("hot", "only"))
    result = evaluate_cascade(network, stages, sources, sinks)

    initial_source_temperature = {
        "cold": np.array([55.0]),
        "hot": np.array([100.0]),
    }
    last_seen_source_temperature = dict(initial_source_temperature)
    for stage_index, source_name in enumerate(result.stage_source_names):
        source_temperature_before_stage = last_seen_source_temperature[source_name]
        sink_temperature_after_stage = result.stage_sink_outlet_temperature_celsius[stage_index]
        assert np.all(
            sink_temperature_after_stage
            <= source_temperature_before_stage - network.minimum_pinch_k + 1e-9
        )
        last_seen_source_temperature[source_name] = result.stage_source_outlet_temperature_celsius[
            stage_index
        ]


def test_no_source_is_cooled_below_its_own_dew_point_floor() -> None:
    network = HeatNetworkConfig(
        sources=(_source("cold", 55.0, 0.02), _source("hot", 100.0, 0.035)),
        sinks=(_sink("only", 95.0, intake=5.0, mass_flow=20.0),),
        minimum_pinch_k=1.0,
    )
    sources, sinks = _states(network)
    stages = (("cold", "only"), ("hot", "only"))
    result = evaluate_cascade(network, stages, sources, sinks)

    for source_config, outlet_temperature in zip(
        network.sources, result.source_outlet_temperature_celsius, strict=True
    ):
        floor = _dew_point(source_config.outlet_humidity_kg_per_kg) + network.dew_point_approach_k
        assert np.all(outlet_temperature >= floor - 1e-9)


def test_each_source_uses_its_own_dew_point_not_another_sources() -> None:
    """Correction 1: different humidities -- different dew-point limits per source."""
    network = HeatNetworkConfig(
        sources=(_source("dry", 100.0, 0.01), _source("humid", 100.0, 0.04)),
        sinks=(_sink("only", 200.0, mass_flow=50.0),),
        minimum_pinch_k=1.0,
    )
    sources, sinks = _states(network)
    result_dry = evaluate_cascade(network, (("dry", "only"),), sources, sinks)
    result_humid = evaluate_cascade(network, (("humid", "only"),), sources, sinks)

    floor_dry = _dew_point(0.01) + network.dew_point_approach_k
    floor_humid = _dew_point(0.04) + network.dew_point_approach_k
    assert floor_dry < floor_humid  # drier source has the lower dew point
    dry_index = result_dry.source_names.index("dry")
    humid_index = result_humid.source_names.index("humid")
    assert result_dry.source_outlet_temperature_celsius[dry_index, 0] == pytest.approx(
        floor_dry, abs=1e-6
    )
    assert result_humid.source_outlet_temperature_celsius[humid_index, 0] == pytest.approx(
        floor_humid, abs=1e-6
    )


# --- Cascade against single stage (EA1 formula) ---------------------------------------


def test_single_source_single_sink_matches_ea1_formula_when_dew_point_binds() -> None:
    mass_flow_source = 8.0
    mass_flow_sink = 8.0
    source_temperature = 60.0
    source_humidity = 0.025
    sink_humidity = 0.008
    intake_temperature = 10.0
    setpoint_temperature = 200.0  # unreachable -- isolates the dew-point-limited branch
    minimum_pinch_k = 1.0  # deliberately loose, so that the approach temperature does not bind here

    source_config = _source("s", source_temperature, source_humidity, mass_flow_source)
    sink_config = _sink(
        "k",
        setpoint_temperature,
        intake=intake_temperature,
        humidity=sink_humidity,
        mass_flow=mass_flow_sink,
    )
    network = HeatNetworkConfig(
        sources=(source_config,), sinks=(sink_config,), minimum_pinch_k=minimum_pinch_k
    )
    sources, sinks = _states(network)
    result = evaluate_cascade(network, (("s", "k"),), sources, sinks)

    floor = _dew_point(source_humidity) + network.dew_point_approach_k
    capacity_source = mass_flow_source * (
        GAS.cp_dry_air_j_per_kg_k + source_humidity * GAS.cp_vapour_j_per_kg_k
    )
    capacity_sink = mass_flow_sink * (
        GAS.cp_dry_air_j_per_kg_k + sink_humidity * GAS.cp_vapour_j_per_kg_k
    )
    q_rec_w = capacity_source * (source_temperature - floor)
    expected_available_kw = q_rec_w / 1000.0
    expected_sink_outlet = intake_temperature + q_rec_w / capacity_sink
    expected_fuel = (capacity_sink * (setpoint_temperature - expected_sink_outlet)) / (
        network.fuel_lower_heating_value_j_per_kg * network.heater_efficiency
    )

    assert result.source_available_power_kw[0, 0] == pytest.approx(expected_available_kw, rel=1e-9)
    assert result.source_used_power_kw[0, 0] == pytest.approx(expected_available_kw, rel=1e-6)
    assert result.source_discarded_power_kw[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert result.sink_outlet_temperature_celsius[0, 0] == pytest.approx(
        expected_sink_outlet, rel=1e-9
    )
    assert result.fuel_mass_flow_kg_per_s[0, 0] == pytest.approx(expected_fuel, rel=1e-9)
    assert result.stage_binding_code[0, 0] == BINDING_DEW_POINT


# --- Ordering rule: low temperature first must never be worse --------------------------


def test_swapping_to_hot_source_first_never_improves_and_can_worsen_the_result() -> None:
    """EA2 correction 3: cold source first may heat the sink only equally well or better.

    Construction: ``hot`` has little enthalpy (small, dew-point-limited) and contributes the
    same fixed temperature rise in any order; ``cold`` is approach-limited (its
    contribution is capped at ``cold_temp - dT_min``, independent of the start temperature of
    the sink). If ``hot`` runs first, it uses up part of the headroom that ``cold``
    would otherwise have had exclusively, before ``cold`` hits its own approach limit --
    the result can thereby only get worse, never better.
    """
    network = HeatNetworkConfig(
        sources=(
            _source("cold", 50.0, 0.02, mass_flow=8.0),
            _source("hot", 90.0, 0.02, mass_flow=0.5),
        ),
        sinks=(_sink("only", 100.0, intake=10.0, mass_flow=8.0),),
        minimum_pinch_k=20.0,
    )
    sources, sinks = _states(network)

    correct_order = evaluate_cascade(network, (("cold", "only"), ("hot", "only")), sources, sinks)
    swapped_order = evaluate_cascade(network, (("hot", "only"), ("cold", "only")), sources, sinks)

    correct_final = correct_order.sink_outlet_temperature_celsius[0, 0]
    swapped_final = swapped_order.sink_outlet_temperature_celsius[0, 0]
    assert correct_final >= swapped_final - 1e-9
    assert correct_final > swapped_final + 1.0, (
        "Testkonstruktion sollte einen echten, nicht nur einen Rundungsunterschied zeigen"
    )


# --- Vectorisation: trajectory == loop-wise single evaluation, bit-identical -----------


def test_vectorised_trajectory_matches_stepwise_loop_bit_exactly() -> None:
    network = HeatNetworkConfig(
        sources=(_source("cold", 55.0, 0.02), _source("hot", 90.0, 0.025, mass_flow=2.0)),
        sinks=(_sink("a", 45.0, mass_flow=2.0), _sink("b", 90.0)),
        minimum_pinch_k=15.0,
    )
    stages = (("cold", "a"), ("cold", "b"), ("hot", "b"))
    n_steps = 5

    source_states = (
        SourceState(
            name="cold",
            dry_air_mass_flow_kg_per_s=np.full(n_steps, 8.0),
            outlet_temperature_celsius=np.array([50.0, 55.0, 60.0, 65.0, 52.0]),
            outlet_humidity_kg_per_kg=np.array([0.018, 0.02, 0.022, 0.019, 0.021]),
        ),
        SourceState(
            name="hot",
            dry_air_mass_flow_kg_per_s=np.full(n_steps, 2.0),
            outlet_temperature_celsius=np.array([85.0, 90.0, 95.0, 88.0, 92.0]),
            outlet_humidity_kg_per_kg=np.full(n_steps, 0.025),
        ),
    )
    sink_states = (
        SinkState(
            name="a",
            dry_air_mass_flow_kg_per_s=np.full(n_steps, 2.0),
            humidity_kg_per_kg=np.full(n_steps, 0.008),
            intake_temperature_celsius=np.array([10.0, 12.0, 15.0, 11.0, 13.0]),
            setpoint_temperature_celsius=np.full(n_steps, 45.0),
        ),
        SinkState(
            name="b",
            dry_air_mass_flow_kg_per_s=np.full(n_steps, 8.0),
            humidity_kg_per_kg=np.full(n_steps, 0.008),
            intake_temperature_celsius=np.full(n_steps, 15.0),
            setpoint_temperature_celsius=np.array([80.0, 90.0, 85.0, 95.0, 88.0]),
        ),
    )

    vectorised = evaluate_cascade(network, stages, source_states, sink_states)

    stepwise_results: list[HeatNetworkResult] = []
    for step in range(n_steps):
        step_sources = tuple(
            SourceState(
                name=state.name,
                dry_air_mass_flow_kg_per_s=state.dry_air_mass_flow_kg_per_s[step : step + 1],
                outlet_temperature_celsius=state.outlet_temperature_celsius[step : step + 1],
                outlet_humidity_kg_per_kg=state.outlet_humidity_kg_per_kg[step : step + 1],
            )
            for state in source_states
        )
        step_sinks = tuple(
            SinkState(
                name=state.name,
                dry_air_mass_flow_kg_per_s=state.dry_air_mass_flow_kg_per_s[step : step + 1],
                humidity_kg_per_kg=state.humidity_kg_per_kg[step : step + 1],
                intake_temperature_celsius=state.intake_temperature_celsius[step : step + 1],
                setpoint_temperature_celsius=state.setpoint_temperature_celsius[step : step + 1],
            )
            for state in sink_states
        )
        stepwise_results.append(evaluate_cascade(network, stages, step_sources, step_sinks))

    def _concatenated(field: str) -> np.ndarray:
        return np.concatenate([getattr(r, field) for r in stepwise_results], axis=-1)

    for field in (
        "stage_recovered_power_kw",
        "stage_binding_code",
        "stage_source_outlet_temperature_celsius",
        "stage_sink_outlet_temperature_celsius",
        "source_available_power_kw",
        "source_used_power_kw",
        "source_discarded_power_kw",
        "source_outlet_temperature_celsius",
        "sink_outlet_temperature_celsius",
        "fuel_mass_flow_kg_per_s",
    ):
        vectorised_value = getattr(vectorised, field)
        stepwise_value = _concatenated(field)
        assert np.array_equal(vectorised_value, stepwise_value), field


# --- Excel template uncorrected vs. corrected (evidence for correction 2) --------------


def _naive_excel_recovery(
    source_line1: HeatSourceConfig,
    source_line2: HeatSourceConfig,
    sink_heater1: HeatSinkConfig,
    sink_heater2: HeatSinkConfig,
    dew_point_approach_k: float,
    fuel_lower_heating_value_j_per_kg: float,
    heater_efficiency: float,
) -> tuple[float, float]:
    """Uncorrected template, deliberately rebuilt with defect 1 and defect 2 (EA5).

    ONLY for this test -- this logic must not appear anywhere in ``src/`` (EA2). Defect 1:
    both "heat exchangers" compute against the dew point of ``source_line1`` (copy error
    between the worksheets). Defect 2: the sink temperature is capped at its set point,
    the unused source enthalpy is discarded, no cascade to the respective
    other sink.
    """
    shared_dew_point = _dew_point(source_line1.outlet_humidity_kg_per_kg)
    shared_floor = shared_dew_point + dew_point_approach_k

    def capacity_of_source(source: HeatSourceConfig) -> float:
        return source.dry_air_mass_flow_kg_per_s * (
            GAS.cp_dry_air_j_per_kg_k + source.outlet_humidity_kg_per_kg * GAS.cp_vapour_j_per_kg_k
        )

    def capacity_of_sink(sink: HeatSinkConfig) -> float:
        return sink.dry_air_mass_flow_kg_per_s * (
            GAS.cp_dry_air_j_per_kg_k + sink.humidity_kg_per_kg * GAS.cp_vapour_j_per_kg_k
        )

    def naive_fuel_for_pair(source: HeatSourceConfig, sink: HeatSinkConfig) -> float:
        q_rec_w = max(
            capacity_of_source(source) * (source.outlet_temperature_celsius - shared_floor), 0.0
        )
        capacity_sink = capacity_of_sink(sink)
        naive_outlet = sink.intake_temperature_celsius + q_rec_w / capacity_sink
        capped_outlet = min(naive_outlet, sink.setpoint_temperature_celsius)  # defect 2
        return max(capacity_sink * (sink.setpoint_temperature_celsius - capped_outlet), 0.0) / (
            fuel_lower_heating_value_j_per_kg * heater_efficiency
        )

    fuel1 = naive_fuel_for_pair(source_line1, sink_heater1)
    fuel2 = naive_fuel_for_pair(source_line2, sink_heater2)
    return fuel1, fuel2


def test_cascading_beats_the_uncorrected_excel_template() -> None:
    """Evidence for correction 2 (and side effect of correction 1): less fuel overall.

    ``line1`` is assigned plenty of enthalpy and a small, quickly saturated sink
    (``heater1``) -- in the uncorrected template the bulk of its
    enthalpy remains unused (defect 2, the documented 891.8 kW example). In the
    corrected model the remainder cascades to ``heater2``, which would otherwise depend
    on ``line2`` alone.
    """
    fuel_lower_heating_value_j_per_kg = 50.0e6
    heater_efficiency = 0.9
    dew_point_approach_k = 3.0

    source_line1 = _source("line1", 95.0, 0.025, mass_flow=8.0)
    source_line2 = _source("line2", 60.0, 0.015, mass_flow=8.0)
    sink_heater1 = _sink("heater1", 45.0, intake=15.0, mass_flow=1.5)
    sink_heater2 = _sink("heater2", 90.0, intake=15.0, mass_flow=8.0)

    naive_fuel1, naive_fuel2 = _naive_excel_recovery(
        source_line1,
        source_line2,
        sink_heater1,
        sink_heater2,
        dew_point_approach_k,
        fuel_lower_heating_value_j_per_kg,
        heater_efficiency,
    )
    naive_total_fuel = naive_fuel1 + naive_fuel2

    network = HeatNetworkConfig(
        sources=(source_line1, source_line2),
        sinks=(sink_heater1, sink_heater2),
        dew_point_approach_k=dew_point_approach_k,
        fuel_lower_heating_value_j_per_kg=fuel_lower_heating_value_j_per_kg,
        heater_efficiency=heater_efficiency,
        minimum_pinch_k=20.0,
    )
    sources, sinks = _states(network)
    corrected = evaluate_cascade(
        network,
        (("line1", "heater1"), ("line2", "heater2"), ("line1", "heater2")),
        sources,
        sinks,
    )
    corrected_total_fuel = float(corrected.total_fuel_mass_flow_kg_per_s()[0])

    assert corrected_total_fuel < naive_total_fuel, (
        f"korrigiert {corrected_total_fuel} kg/s sollte unter naiv {naive_total_fuel} kg/s liegen"
    )
    # Substantial effect, not a rounding artifact (analogous to the documented order of magnitude).
    assert (naive_total_fuel - corrected_total_fuel) / naive_total_fuel > 0.05


# --- MA1: second recovery mode (condensation) ------------------------------------------


def test_cold_temperature_is_a_no_op_in_sensible_mode() -> None:
    """Regression test.

    ``cold_temperature_celsius`` must have no effect whatsoever in ``recovery_mode='sensible'``
    (default) -- bit-identical, whatever value is set there.
    """
    source = _source("s", 70.0, 0.03, mass_flow=8.0)
    sink = _sink("k", 200.0, intake=5.0, humidity=0.006, mass_flow=8.0)

    network_default = HeatNetworkConfig(sources=(source,), sinks=(sink,), minimum_pinch_k=1.0)
    network_explicit = HeatNetworkConfig(
        sources=(source,),
        sinks=(sink,),
        minimum_pinch_k=1.0,
        recovery_mode="sensible",
        cold_temperature_celsius=-30.0,  # deliberately extremely different from the default (15 °C)
    )
    sources, sinks = _states(network_default)
    result_default = evaluate_cascade(network_default, (("s", "k"),), sources, sinks)
    result_explicit = evaluate_cascade(network_explicit, (("s", "k"),), sources, sinks)

    for field_name in (
        "stage_recovered_power_kw",
        "stage_binding_code",
        "stage_source_outlet_temperature_celsius",
        "stage_sink_outlet_temperature_celsius",
        "source_available_power_kw",
        "source_used_power_kw",
        "source_discarded_power_kw",
        "source_condensate_kg_per_s",
        "source_outlet_temperature_celsius",
        "sink_outlet_temperature_celsius",
        "fuel_mass_flow_kg_per_s",
    ):
        assert np.array_equal(
            getattr(result_default, field_name), getattr(result_explicit, field_name)
        ), field_name
    assert np.all(result_default.source_condensate_kg_per_s == 0.0)


def test_sensible_mode_still_matches_ea1_formula() -> None:
    """MA1 regression anchor.

    ``recovery_mode='sensible'`` (default) reproduces the EA1 single-stage formula exactly --
    the same setup as
    ``test_single_source_single_sink_matches_ea1_formula_when_dew_point_binds``.
    """
    mass_flow = 8.0
    source_temperature = 70.0
    source_humidity = 0.028
    network = HeatNetworkConfig(
        sources=(_source("s", source_temperature, source_humidity, mass_flow),),
        sinks=(_sink("k", 250.0, intake=5.0, humidity=0.006, mass_flow=mass_flow),),
        minimum_pinch_k=1.0,
    )
    sources, sinks = _states(network)
    result = evaluate_cascade(network, (("s", "k"),), sources, sinks)

    floor = _dew_point(source_humidity) + network.dew_point_approach_k
    capacity_source = mass_flow * (
        GAS.cp_dry_air_j_per_kg_k + source_humidity * GAS.cp_vapour_j_per_kg_k
    )
    expected_available_kw = capacity_source * (source_temperature - floor) / 1000.0
    assert result.source_available_power_kw[0, 0] == pytest.approx(expected_available_kw, rel=1e-9)
    assert result.source_condensate_kg_per_s[0, 0] == 0.0


def test_condensing_mode_recovers_at_least_as_much_as_sensible_mode() -> None:
    """MA1 requirement 2.

    At ``T_kalt <= dew point``, Q_rec(condensing) >= Q_rec(non-condensing) -- the
    condensing source cools further (down to ``T_kalt`` instead of only down to
    ``T_dp + dT_approach``) AND additionally supplies latent heat.
    """
    source_temperature = 65.0
    source_humidity = 0.03
    mass_flow = 8.0
    dew_point = _dew_point(source_humidity)
    cold_temperature = dew_point - 5.0  # clearly below the dew point
    assert cold_temperature < dew_point

    source = _source("s", source_temperature, source_humidity, mass_flow)
    sink = _sink("k", 250.0, intake=5.0, humidity=0.006, mass_flow=mass_flow)

    network_sensible = HeatNetworkConfig(sources=(source,), sinks=(sink,), minimum_pinch_k=1.0)
    network_condensing = HeatNetworkConfig(
        sources=(source,),
        sinks=(sink,),
        minimum_pinch_k=1.0,
        recovery_mode="condensing",
        cold_temperature_celsius=cold_temperature,
    )
    sources, sinks = _states(network_sensible)
    result_sensible = evaluate_cascade(network_sensible, (("s", "k"),), sources, sinks)
    result_condensing = evaluate_cascade(network_condensing, (("s", "k"),), sources, sinks)

    available_condensing = result_condensing.source_available_power_kw[0, 0]
    available_sensible = result_sensible.source_available_power_kw[0, 0]
    assert available_condensing > available_sensible
    assert result_condensing.source_condensate_kg_per_s[0, 0] > 0.0


def test_condensing_mode_falls_back_continuously_above_dew_point() -> None:
    """MA1 requirement 3.

    If ``T_kalt`` lies above the dew point, the condensate is zero and the mode falls back
    continuously to pure sensible cooling down to ``T_kalt`` (no latent jump at the
    dew-point limit).
    """
    source_temperature = 65.0
    source_humidity = 0.03
    mass_flow = 8.0
    dew_point = _dew_point(source_humidity)
    cold_temperature = dew_point + 10.0  # clearly above the dew point
    assert cold_temperature > dew_point
    assert _saturation_humidity(cold_temperature) >= source_humidity  # condensate must be 0

    network = HeatNetworkConfig(
        sources=(_source("s", source_temperature, source_humidity, mass_flow),),
        sinks=(_sink("k", 250.0, intake=5.0, humidity=0.006, mass_flow=mass_flow),),
        minimum_pinch_k=1.0,
        recovery_mode="condensing",
        cold_temperature_celsius=cold_temperature,
    )
    sources, sinks = _states(network)
    result = evaluate_cascade(network, (("s", "k"),), sources, sinks)

    capacity_source = mass_flow * (
        GAS.cp_dry_air_j_per_kg_k + source_humidity * GAS.cp_vapour_j_per_kg_k
    )
    expected_pure_sensible_kw = capacity_source * (source_temperature - cold_temperature) / 1000.0

    assert result.source_available_power_kw[0, 0] == pytest.approx(
        expected_pure_sensible_kw, rel=1e-9
    )
    assert result.source_condensate_kg_per_s[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_condensing_mode_matches_ma1_formula_and_conserves_mass() -> None:
    """MA1 formula recomputed literally and mass balance checked.

    Condensate, Q_latent, Q_sensible, Q_rec against the MA1 formula; the condensate leaves the
    balance as liquid water, the sum of residual moisture and condensate again gives the
    original moisture load. Mode ``approach_check="warm_end"`` (flash at ``T_kalt``,
    state before Befund 79); the profile with condensation from the dew point on is checked by
    ``tests/test_heatnetwork_approach.py``.
    """
    source_temperature = 65.0
    source_humidity = 0.03
    mass_flow = 8.0
    cold_temperature = 12.0
    assert cold_temperature < _dew_point(source_humidity)  # condensation must take place

    network = HeatNetworkConfig(
        sources=(_source("s", source_temperature, source_humidity, mass_flow),),
        # Large sink: its demand (capacity- and pinch-limited against the
        # INITIAL source temperature of 65 C) must lie above the ENTIRE available power
        # (sensible + latent), otherwise the sink binds before the source is really exhausted down
        # to T_kalt including condensate (see the docstring note in the module on the
        # stage-wise release of the latent reserve).
        sinks=(_sink("k", 250.0, intake=5.0, humidity=0.006, mass_flow=20.0),),
        minimum_pinch_k=1.0,
        recovery_mode="condensing",
        approach_check="warm_end",
        cold_temperature_celsius=cold_temperature,
    )
    sources, sinks = _states(network)
    result = evaluate_cascade(network, (("s", "k"),), sources, sinks)

    y_sat_cold = _saturation_humidity(cold_temperature)
    expected_condensate_kg_per_s = max(0.0, source_humidity - y_sat_cold) * mass_flow
    expected_q_latent_w = expected_condensate_kg_per_s * _latent_heat(cold_temperature)
    capacity_source = mass_flow * (
        GAS.cp_dry_air_j_per_kg_k + source_humidity * GAS.cp_vapour_j_per_kg_k
    )
    expected_q_sensible_w = capacity_source * (source_temperature - cold_temperature)
    expected_q_rec_kw = (expected_q_sensible_w + expected_q_latent_w) / 1000.0

    # Set point/approach temperature are chosen so loosely here that the source is fully
    # exhausted down to T_kalt -- the formula therefore applies uncapped.
    assert result.source_condensate_kg_per_s[0, 0] == pytest.approx(
        expected_condensate_kg_per_s, rel=1e-9
    )
    assert result.source_available_power_kw[0, 0] == pytest.approx(expected_q_rec_kw, rel=1e-9)
    assert result.source_used_power_kw[0, 0] == pytest.approx(expected_q_rec_kw, rel=1e-6)

    # Mass balance: residual moisture in the air stream + condensed-out water == initial moisture.
    outlet_humidity = source_humidity - result.source_condensate_kg_per_s[0, 0] / mass_flow
    assert outlet_humidity == pytest.approx(y_sat_cold, rel=1e-9)
    assert outlet_humidity * mass_flow + result.source_condensate_kg_per_s[0, 0] == pytest.approx(
        source_humidity * mass_flow, rel=1e-9
    )


def test_condensate_released_only_after_sensible_reserve_exhausted() -> None:
    """Multi-stage.

    The latent heat of a source may only arise once its sensible cooling reserve has
    already been exhausted in a stage (condensation only sets in on reaching
    ``T_kalt``, not gradually before). Applies only in the mode ``approach_check="warm_end"``
    (state before Befund 79); in the default the source condenses from its dew point on.
    """
    source_temperature = 80.0
    source_humidity = 0.03
    mass_flow = 8.0
    cold_temperature = 10.0
    assert cold_temperature < _dew_point(source_humidity)

    network = HeatNetworkConfig(
        sources=(_source("s", source_temperature, source_humidity, mass_flow),),
        # Small first sink: saturates before the source has exhausted its sensible
        # reserve -- no condensate may arise there yet.
        # As in the single-stage test, "big" needs plenty of capacity, so that its demand
        # lies above the entire available power (sensible + latent) of the source.
        sinks=(
            _sink("small", 40.0, intake=5.0, humidity=0.006, mass_flow=0.5),
            _sink("big", 250.0, intake=5.0, humidity=0.006, mass_flow=20.0),
        ),
        minimum_pinch_k=1.0,
        recovery_mode="condensing",
        approach_check="warm_end",
        cold_temperature_celsius=cold_temperature,
    )
    sources, sinks = _states(network)
    result = evaluate_cascade(network, (("s", "small"), ("s", "big")), sources, sinks)

    # First stage: source stays clearly above T_kalt, purely sensible -- no condensate.
    assert result.stage_source_outlet_temperature_celsius[0, 0] > cold_temperature + 1.0
    # Cumulatively (after both stages) the condensate is identical to the single-stage formula,
    # because the source is again fully exhausted down to T_kalt at the end.
    y_sat_cold = _saturation_humidity(cold_temperature)
    expected_condensate_kg_per_s = max(0.0, source_humidity - y_sat_cold) * mass_flow
    assert result.source_condensate_kg_per_s[0, 0] == pytest.approx(
        expected_condensate_kg_per_s, rel=1e-6
    )

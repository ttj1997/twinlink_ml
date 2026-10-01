"""Tests für ``heatnetwork/balance.py`` -- Kaskadenphysik (EA5).

Deckt: Energieerhaltung je Quelle, Grädigkeits- und Taupunkt-Invarianten je Stufe,
Übereinstimmung mit der Einzelstufen-Formel (EA1) im Ein-Quelle/Ein-Senke-Fall,
die Reihenfolgeregel (Niedertemperatur zuerst darf nie schlechter sein), bitgleiche
Vektorisierung über Zeitschritte, und den Nachweis, dass die Kaskadierung (Korrektur 2)
gegenüber der unkorrigierten Excel-Vorlage strikt weniger Brennstoff braucht.
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


# --- Energieerhaltung ---------------------------------------------------------------


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


# --- Grädigkeit und Taupunktgrenze ----------------------------------------------------


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
    """Korrektur 1: unterschiedliche Feuchten -- unterschiedliche Taupunktgrenzen je Quelle."""
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
    assert floor_dry < floor_humid  # trockenere Quelle hat den niedrigeren Taupunkt
    dry_index = result_dry.source_names.index("dry")
    humid_index = result_humid.source_names.index("humid")
    assert result_dry.source_outlet_temperature_celsius[dry_index, 0] == pytest.approx(
        floor_dry, abs=1e-6
    )
    assert result_humid.source_outlet_temperature_celsius[humid_index, 0] == pytest.approx(
        floor_humid, abs=1e-6
    )


# --- Kaskade gegen Einzelstufe (EA1-Formel) -------------------------------------------


def test_single_source_single_sink_matches_ea1_formula_when_dew_point_binds() -> None:
    mass_flow_source = 8.0
    mass_flow_sink = 8.0
    source_temperature = 60.0
    source_humidity = 0.025
    sink_humidity = 0.008
    intake_temperature = 10.0
    setpoint_temperature = 200.0  # unerreichbar -- isoliert den taupunkt-limitierten Zweig
    minimum_pinch_k = 1.0  # bewusst locker, damit die Grädigkeit hier nicht bindet

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


# --- Reihenfolgeregel: Niedertemperatur zuerst darf nie schlechter sein ----------------


def test_swapping_to_hot_source_first_never_improves_and_can_worsen_the_result() -> None:
    """EA2 Korrektur 3: kalte Quelle zuerst darf die Senke nur genauso gut oder besser aufheizen.

    Konstruktion: ``hot`` hat wenig Enthalpie (klein, taupunktlimitiert) und trägt in jeder
    Reihenfolge denselben festen Temperaturhub bei; ``cold`` ist grädigkeitslimitiert (ihr
    Beitrag ist auf ``cold_temp - dT_min`` gedeckelt, unabhängig von der Starttemperatur der
    Senke). Läuft ``hot`` zuerst, verbraucht sie einen Teil des Spielraums, den ``cold``
    sonst exklusiv gehabt hätte, bevor ``cold`` an ihre eigene Grädigkeitsgrenze stößt --
    das Ergebnis kann dadurch nur schlechter, nie besser werden.
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


# --- Vektorisierung: Trajektorie == schleifenweise Einzelauswertung, bitgleich ---------


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


# --- Excel-Vorlage unkorrigiert vs. korrigiert (Nachweis für Korrektur 2) --------------


def _naive_excel_recovery(
    source_line1: HeatSourceConfig,
    source_line2: HeatSourceConfig,
    sink_heater1: HeatSinkConfig,
    sink_heater2: HeatSinkConfig,
    dew_point_approach_k: float,
    fuel_lower_heating_value_j_per_kg: float,
    heater_efficiency: float,
) -> tuple[float, float]:
    """Unkorrigierte Vorlage, absichtlich mit Defekt 1 und Defekt 2 nachgebaut (EA5).

    NUR für diesen Test -- diese Logik darf nirgends in ``src/`` auftauchen (EA2). Defekt 1:
    beide "Wärmeübertrager" rechnen gegen den Taupunkt von ``source_line1`` (Kopierfehler
    zwischen den Tabellenblättern). Defekt 2: die Senkentemperatur wird auf ihren Sollwert
    gekappt, die nicht genutzte Quellenenthalpie wird verworfen, keine Kaskade zur jeweils
    anderen Senke.
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
        capped_outlet = min(naive_outlet, sink.setpoint_temperature_celsius)  # Defekt 2
        return max(capacity_sink * (sink.setpoint_temperature_celsius - capped_outlet), 0.0) / (
            fuel_lower_heating_value_j_per_kg * heater_efficiency
        )

    fuel1 = naive_fuel_for_pair(source_line1, sink_heater1)
    fuel2 = naive_fuel_for_pair(source_line2, sink_heater2)
    return fuel1, fuel2


def test_cascading_beats_the_uncorrected_excel_template() -> None:
    """Nachweis für Korrektur 2 (und Nebeneffekt von Korrektur 1): weniger Brennstoff insgesamt.

    ``line1`` bekommt reichlich Enthalpie und eine kleine, schnell gesättigte Senke
    (``heater1``) zugeordnet -- in der unkorrigierten Vorlage bleibt der Großteil ihrer
    Enthalpie ungenutzt liegen (Defekt 2, das dokumentierte 891,8-kW-Beispiel). Im
    korrigierten Modell kaskadiert der Rest zu ``heater2``, die sonst allein auf
    ``line2`` angewiesen wäre.
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
    # Substanzieller Effekt, kein Rundungsartefakt (analog zur dokumentierten Größenordnung).
    assert (naive_total_fuel - corrected_total_fuel) / naive_total_fuel > 0.05


# --- MA1: zweiter Rückgewinnungsmodus (Kondensation) -----------------------------------


def test_cold_temperature_is_a_no_op_in_sensible_mode() -> None:
    """Regressionstest.

    ``cold_temperature_celsius`` darf in ``recovery_mode='sensible'`` (Default) keinerlei
    Wirkung haben -- bitgleich, egal welcher Wert dort steht.
    """
    source = _source("s", 70.0, 0.03, mass_flow=8.0)
    sink = _sink("k", 200.0, intake=5.0, humidity=0.006, mass_flow=8.0)

    network_default = HeatNetworkConfig(sources=(source,), sinks=(sink,), minimum_pinch_k=1.0)
    network_explicit = HeatNetworkConfig(
        sources=(source,),
        sinks=(sink,),
        minimum_pinch_k=1.0,
        recovery_mode="sensible",
        cold_temperature_celsius=-30.0,  # bewusst extrem verschieden vom Default (15 °C)
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
    """MA1-Regressionsanker.

    ``recovery_mode='sensible'`` (Default) reproduziert die EA1-Einzelstufenformel exakt --
    derselbe Aufbau wie
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
    """MA1-Anforderung 2.

    Bei ``T_kalt <= Taupunkt`` ist Q_rec(kondensierend) >= Q_rec(nicht kondensierend) -- die
    kondensierende Quelle kühlt weiter (bis ``T_kalt`` statt nur bis
    ``T_dp + dT_approach``) UND liefert zusätzlich latente Wärme.
    """
    source_temperature = 65.0
    source_humidity = 0.03
    mass_flow = 8.0
    dew_point = _dew_point(source_humidity)
    cold_temperature = dew_point - 5.0  # klar unter dem Taupunkt
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
    """MA1-Anforderung 3.

    Liegt ``T_kalt`` über dem Taupunkt, ist das Kondensat null und der Modus fällt stetig
    auf reine sensible Kühlung bis ``T_kalt`` zurück (kein latenter Sprung an der
    Taupunktgrenze).
    """
    source_temperature = 65.0
    source_humidity = 0.03
    mass_flow = 8.0
    dew_point = _dew_point(source_humidity)
    cold_temperature = dew_point + 10.0  # klar über dem Taupunkt
    assert cold_temperature > dew_point
    assert _saturation_humidity(cold_temperature) >= source_humidity  # Kondensat muss 0 sein

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
    """MA1-Formel wörtlich nachgerechnet und Massenbilanz geprüft.

    Kondensat, Q_latent, Q_sensibel, Q_rec gegen die MA1-Formel; das Kondensat verlässt die
    Bilanz als Flüssigwasser, die Summe aus Restfeuchte und Kondensat ergibt wieder die
    ursprüngliche Feuchtefracht.
    """
    source_temperature = 65.0
    source_humidity = 0.03
    mass_flow = 8.0
    cold_temperature = 12.0
    assert cold_temperature < _dew_point(source_humidity)  # Kondensation muss stattfinden

    network = HeatNetworkConfig(
        sources=(_source("s", source_temperature, source_humidity, mass_flow),),
        # Grosse Senke: ihre Nachfrage (kapazitaets- und pinchbegrenzt gegen die
        # ANFANGS-Quellentemperatur von 65 C) muss ueber der GESAMTEN verfuegbaren Leistung
        # (sensibel + latent) liegen, sonst bindet die Senke, bevor die Quelle wirklich bis
        # T_kalt samt Kondensat ausgeschoepft ist (siehe Docstring-Anmerkung im Modul zur
        # stufenweisen Freigabe des latenten Vorrats).
        sinks=(_sink("k", 250.0, intake=5.0, humidity=0.006, mass_flow=20.0),),
        minimum_pinch_k=1.0,
        recovery_mode="condensing",
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

    # Sollwert/Grädigkeit sind hier so locker gewählt, dass die Quelle vollständig bis
    # T_kalt ausgeschöpft wird -- die Formel gilt also ungekappt.
    assert result.source_condensate_kg_per_s[0, 0] == pytest.approx(
        expected_condensate_kg_per_s, rel=1e-9
    )
    assert result.source_available_power_kw[0, 0] == pytest.approx(expected_q_rec_kw, rel=1e-9)
    assert result.source_used_power_kw[0, 0] == pytest.approx(expected_q_rec_kw, rel=1e-6)

    # Massenbilanz: Restfeuchte im Luftstrom + ausgekondensiertes Wasser == Ausgangsfeuchte.
    outlet_humidity = source_humidity - result.source_condensate_kg_per_s[0, 0] / mass_flow
    assert outlet_humidity == pytest.approx(y_sat_cold, rel=1e-9)
    assert outlet_humidity * mass_flow + result.source_condensate_kg_per_s[0, 0] == pytest.approx(
        source_humidity * mass_flow, rel=1e-9
    )


def test_condensate_released_only_after_sensible_reserve_exhausted() -> None:
    """Mehrstufig.

    Die latente Wärme einer Quelle darf erst anfallen, wenn ihre sensible Kühlreserve in
    einer Stufe bereits ausgeschöpft ist (Kondensation setzt erst bei Erreichen von
    ``T_kalt`` ein, nicht graduell vorher).
    """
    source_temperature = 80.0
    source_humidity = 0.03
    mass_flow = 8.0
    cold_temperature = 10.0
    assert cold_temperature < _dew_point(source_humidity)

    network = HeatNetworkConfig(
        sources=(_source("s", source_temperature, source_humidity, mass_flow),),
        # Kleine erste Senke: sättigt sich, bevor die Quelle ihre sensible Reserve
        # ausgeschöpft hat -- dort darf noch kein Kondensat anfallen.
        # "big" braucht wie im Einzelstufentest reichlich Kapazitaet, damit ihre Nachfrage
        # ueber der gesamten verfuegbaren Leistung (sensibel + latent) der Quelle liegt.
        sinks=(
            _sink("small", 40.0, intake=5.0, humidity=0.006, mass_flow=0.5),
            _sink("big", 250.0, intake=5.0, humidity=0.006, mass_flow=20.0),
        ),
        minimum_pinch_k=1.0,
        recovery_mode="condensing",
        cold_temperature_celsius=cold_temperature,
    )
    sources, sinks = _states(network)
    result = evaluate_cascade(network, (("s", "small"), ("s", "big")), sources, sinks)

    # Erste Stufe: Quelle bleibt klar über T_kalt, rein sensibel -- kein Kondensat.
    assert result.stage_source_outlet_temperature_celsius[0, 0] > cold_temperature + 1.0
    # Kumulativ (nach beiden Stufen) ist das Kondensat identisch zur Einzelstufenformel,
    # weil die Quelle am Ende wieder vollständig bis T_kalt ausgeschöpft wird.
    y_sat_cold = _saturation_humidity(cold_temperature)
    expected_condensate_kg_per_s = max(0.0, source_humidity - y_sat_cold) * mass_flow
    assert result.source_condensate_kg_per_s[0, 0] == pytest.approx(
        expected_condensate_kg_per_s, rel=1e-6
    )

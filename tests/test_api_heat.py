"""Heat-distribution service and recovery-case rule: equal to the campaign-network code."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from twinlink_ml.api import (
    API_VERSION,
    SCHEMA_VERSION,
    HeatDistributionRequest,
    HeatDistributionResponse,
    HeatDistributionService,
    RecoveryCaseRequest,
    RecoveryCaseResponse,
    RecoveryCaseService,
    RequestRejectedError,
)
from twinlink_ml.api import heat as api_heat
from twinlink_ml.api.base import PARTNER_WORKBOOK_SCOPE, SITE_NETWORK_SCOPE
from twinlink_ml.campaign_network import (
    CampaignNetworkConfig,
    build_consumers,
    key_figures,
    load_campaign_network_config,
    plant_schedules,
    run_k1,
    run_k2,
)
from twinlink_ml.campaign_network.planning import TrueForecast, cut_data, rolling_plan
from twinlink_ml.campaign_network.profiles import Consumers
from twinlink_ml.case_ranking.config import load_config as load_case_ranking_config
from twinlink_ml.case_ranking.rule import APPROACH_TOTAL_K

ROOT = Path(__file__).resolve().parents[1]
HOURS = 14 * 24  # two weeks from the start of the year (heating season)
POWER_KW = 1200.0  # above the base demand, so that the tank charges and discharges
STORAGE_HOURS = 8.0
CAPACITY_KWH = STORAGE_HOURS * POWER_KW
AVAILABILITY = 0.4


@pytest.fixture(scope="module")
def year() -> CampaignNetworkConfig:
    return load_campaign_network_config()


@pytest.fixture(scope="module")
def short(year: CampaignNetworkConfig) -> CampaignNetworkConfig:
    """The study configuration with a profile of ``HOURS`` hours instead of a year."""
    return dataclasses.replace(year, campaigns=dataclasses.replace(year.campaigns, hours=HOURS))


@pytest.fixture(scope="module")
def on(short: CampaignNetworkConfig) -> dict[str, np.ndarray]:
    schedules = plant_schedules(short, AVAILABILITY, "independent", 0)
    assert any(not state.all() for state in schedules.values()), "a standstill is needed"
    return schedules


@pytest.fixture(scope="module")
def consumers(short: CampaignNetworkConfig, on: dict[str, np.ndarray]) -> Consumers:
    return build_consumers(short, on)


@pytest.fixture(scope="module")
def service(short: CampaignNetworkConfig) -> HeatDistributionService:
    return HeatDistributionService(short)


def _payload(
    short: CampaignNetworkConfig,
    on: dict[str, np.ndarray],
    rows: slice | None = None,
    **overrides: object,
) -> dict:
    """The consumers of the study configuration as a request: flows while running, and the plan."""
    rows = slice(None) if rows is None else rows
    full = build_consumers(short, {name: np.ones(HOURS, dtype=bool) for name in on})
    sinks = []
    for stream in full.streams:
        sink = {
            "name": stream.name,
            "group": stream.group,
            "required_temperature_celsius": stream.stage_celsius,
            "inlet_temperature_celsius": stream.t_in[rows].tolist(),
            "outlet_temperature_celsius": stream.t_out[rows].tolist(),
            "heat_capacity_flow_kw_per_k": stream.cp[rows].tolist(),
            "replaced_supply_efficiency": stream.gas_efficiency,
        }
        if stream.name in on:
            sink["running"] = on[stream.name][rows].tolist()
        sinks.append(sink)
    return {"source_power_kw": POWER_KW, "sinks": sinks, **overrides}


def _figures(response: HeatDistributionResponse) -> dict[str, float]:
    figures = response.key_figures
    return {
        "source_kwh": figures.source_energy_kwh,
        "use_kwh": figures.heat_used_kwh,
        "utilisation": figures.utilisation,
        "gas_kwh": figures.replaced_gas_kwh,
        "co2_gas_kg": figures.co2_replaced_gas_kg,
        "pump_electricity_kwh": figures.pump_electricity_kwh,
        "co2_pump_kg": figures.co2_pump_kg,
        "co2_net_kg": figures.co2_net_saving_kg,
        "charge_kwh": figures.storage_charged_kwh,
        "discharge_kwh": figures.storage_discharged_kwh,
        "content_end_kwh": figures.storage_content_end_kwh,
    }


def test_cascade_rule_equals_the_research_code(
    service: HeatDistributionService,
    short: CampaignNetworkConfig,
    on: dict[str, np.ndarray],
    consumers: Consumers,
) -> None:
    """No tank: heat per consumer and key figures of ``run_k1``, bit for bit, also through JSON."""
    expected = run_k1(consumers, POWER_KW, short)
    request = HeatDistributionRequest.model_validate(_payload(short, on))
    response = service.handle(request)
    through_json = HeatDistributionResponse.model_validate_json(
        service.handle_json(request.model_dump_json())
    )
    assert through_json == response
    assert response.mode == "cascade_rule"
    assert response.sink_heat is not None
    assert [sink.name for sink in response.sink_heat] == [s.name for s in consumers.streams]
    for sink in response.sink_heat:
        assert np.array_equal(np.asarray(sink.heat_delivered_kw), expected.duty_kw[sink.name])
    figures = key_figures(expected, consumers, POWER_KW, short)
    assert _figures(response) == {name: figures[name] for name in _figures(response)}
    assert np.array_equal(
        np.asarray(response.total_heat_used_kw),
        np.sum([expected.duty_kw[s.name] for s in consumers.streams], axis=0),
    )
    assert not any(response.storage_charge_kw)
    assert len(response.storage_content_kwh) == HOURS + 1
    assert response.meta.notices == ()


@pytest.mark.parametrize("rule", ["k2", "k2b"])
@pytest.mark.parametrize("content", [0.0, 1500.0])
def test_storage_rules_equal_the_research_code(
    service: HeatDistributionService,
    short: CampaignNetworkConfig,
    on: dict[str, np.ndarray],
    consumers: Consumers,
    rule: str,
    content: float,
) -> None:
    """With a tank, empty or holding heat at the first hour: ``run_k2``, bit for bit."""
    expected = run_k2(
        consumers, POWER_KW, STORAGE_HOURS, short, rule=rule, initial_content_kwh=content
    )
    response = service.handle(
        HeatDistributionRequest.model_validate(
            _payload(
                short,
                on,
                storage_capacity_kwh=CAPACITY_KWH,
                storage_content_kwh=content,
                storage_rule=rule,
            )
        )
    )
    assert response.sink_heat is not None
    for sink in response.sink_heat:
        assert np.array_equal(np.asarray(sink.heat_delivered_kw), expected.duty_kw[sink.name])
    assert np.array_equal(np.asarray(response.storage_charge_kw), expected.charge_kw)
    assert np.array_equal(np.asarray(response.storage_discharge_kw), expected.discharge_kw)
    assert np.array_equal(np.asarray(response.storage_content_kwh), expected.content_kwh)
    assert response.storage_content_kwh[0] == content
    figures = key_figures(expected, consumers, POWER_KW, short)
    assert _figures(response) == {name: figures[name] for name in _figures(response)}
    assert response.key_figures.storage_charged_kwh > 0.0


@pytest.mark.parametrize("rule", ["k2", "k2b"])
def test_calls_with_the_content_handed_over_equal_one_call(
    service: HeatDistributionService,
    short: CampaignNetworkConfig,
    on: dict[str, np.ndarray],
    rule: str,
) -> None:
    """The tank is state of the caller: three calls in a row give the operation of one call."""
    settings = {"storage_capacity_kwh": CAPACITY_KWH, "storage_rule": rule}
    whole = service.handle(HeatDistributionRequest.model_validate(_payload(short, on, **settings)))
    content = 0.0
    used: list[float] = []
    contents: list[float] = [content]
    for rows in (slice(0, 100), slice(100, 101), slice(101, HOURS)):
        part = service.handle(
            HeatDistributionRequest.model_validate(
                _payload(short, on, rows, storage_content_kwh=content, **settings)
            )
        )
        used.extend(part.total_heat_used_kw)
        contents.extend(part.storage_content_kwh[1:])
        content = part.storage_content_kwh[-1]
    assert tuple(used) == whole.total_heat_used_kw
    assert tuple(contents) == whole.storage_content_kwh
    assert whole.key_figures.storage_discharged_kwh > 0.0


@pytest.mark.parametrize("content", [0.0, 1500.0])
def test_planning_equals_the_research_code(
    service: HeatDistributionService,
    short: CampaignNetworkConfig,
    on: dict[str, np.ndarray],
    consumers: Consumers,
    content: float,
) -> None:
    """One window over the request with the plan known: ``rolling_plan`` of the research code."""
    expected = rolling_plan(
        short,
        cut_data(short),
        on,
        TrueForecast(on),
        POWER_KW,
        STORAGE_HOURS,
        horizon_h=HOURS,
        replan_every_h=HOURS,
        initial_content_kwh=content,
    )
    payload = _payload(
        short, on, mode="planning", storage_capacity_kwh=CAPACITY_KWH, storage_content_kwh=content
    )
    response = service.handle(HeatDistributionRequest.model_validate(payload))
    assert response.mode == "planning"
    assert response.sink_heat is None
    assert response.lp_failures == expected.lp_failures == 0
    assert np.array_equal(np.asarray(response.total_heat_used_kw), expected.use_kw)
    assert np.array_equal(np.asarray(response.storage_charge_kw), expected.charge_kw)
    assert np.array_equal(np.asarray(response.storage_discharge_kw), expected.discharge_kw)
    assert np.array_equal(np.asarray(response.storage_content_kwh), expected.content_kwh)
    assert response.storage_content_kwh[0] == content

    use = float(np.sum(expected.use_kw))
    figures = response.key_figures
    assert figures.heat_used_kwh == use
    assert figures.source_energy_kwh == POWER_KW * HOURS
    assert figures.utilisation == use / (POWER_KW * HOURS)
    efficiency = consumers.streams[0].gas_efficiency
    assert figures.replaced_gas_kwh == use / efficiency
    assert figures.co2_net_saving_kg == pytest.approx(
        short.gas_co2_kg_per_kwh * use / efficiency
        - short.electricity_co2_kg_per_kwh * short.pump_fraction * use
    )
    # planning with the plan known is at least as good as the rules without foresight
    rule = service.handle(
        HeatDistributionRequest.model_validate({**payload, "mode": "cascade_rule"})
    )
    assert figures.heat_used_kwh >= rule.key_figures.heat_used_kwh - 1e-6


def test_response_says_what_it_is(
    service: HeatDistributionService, short: CampaignNetworkConfig, on: dict[str, np.ndarray]
) -> None:
    response = service.handle(HeatDistributionRequest.model_validate(_payload(short, on)))
    meta = response.meta
    assert (meta.service, meta.api_version, meta.schema_version) == (
        "heat_distribution",
        API_VERSION,
        SCHEMA_VERSION,
    )
    assert meta.model_scope == SITE_NETWORK_SCOPE == "site heat network, campaign-network model"
    assert meta.parameter_status == "placeholder"
    assert meta.advisory_only is True
    assert meta.model_fingerprint == f"campaign_network:{short.fingerprint}"
    assert response.sample_time_s == 3600.0
    assert 0.0 < response.key_figures.utilisation <= 1.0


def test_service_is_stateless(
    service: HeatDistributionService, short: CampaignNetworkConfig, on: dict[str, np.ndarray]
) -> None:
    tank = _payload(short, on, storage_capacity_kwh=CAPACITY_KWH)
    first = HeatDistributionRequest.model_validate(tank)
    before = service.handle(first)
    service.handle(HeatDistributionRequest.model_validate(_payload(short, on)))
    assert service.handle(first) == before


def test_wrong_grid_is_rejected(
    service: HeatDistributionService, short: CampaignNetworkConfig, on: dict[str, np.ndarray]
) -> None:
    request = HeatDistributionRequest.model_validate(_payload(short, on, sample_time_s=900.0))
    with pytest.raises(RequestRejectedError, match="hourly grid"):
        service.handle(request)


def test_planning_needs_equal_efficiencies(
    service: HeatDistributionService, short: CampaignNetworkConfig, on: dict[str, np.ndarray]
) -> None:
    payload = _payload(short, on, mode="planning")
    payload["sinks"][0]["replaced_supply_efficiency"] = 0.8
    with pytest.raises(RequestRejectedError, match="same replaced_supply_efficiency"):
        service.handle(HeatDistributionRequest.model_validate(payload))
    payload["mode"] = "cascade_rule"
    assert service.handle(HeatDistributionRequest.model_validate(payload)).sink_heat is not None


def test_branches_of_one_stage_need_the_same_temperatures(
    service: HeatDistributionService, short: CampaignNetworkConfig, on: dict[str, np.ndarray]
) -> None:
    payload = _payload(short, on)
    payload["sinks"][1]["required_temperature_celsius"] = 120.0
    with pytest.raises(RequestRejectedError, match="same required temperature"):
        service.handle(HeatDistributionRequest.model_validate(payload))


def test_malformed_requests_fail_validation(
    short: CampaignNetworkConfig, on: dict[str, np.ndarray]
) -> None:
    payload = _payload(short, on)
    twice = {**payload, "sinks": [payload["sinks"][0], payload["sinks"][0]]}
    with pytest.raises(ValidationError, match="distinct"):
        HeatDistributionRequest.model_validate(twice)
    crooked = dict(payload["sinks"][0])
    crooked["outlet_temperature_celsius"] = crooked["outlet_temperature_celsius"][1:]
    with pytest.raises(ValidationError, match="same, positive length"):
        HeatDistributionRequest.model_validate({**payload, "sinks": [crooked]})
    shorter = {
        key: (value[1:] if isinstance(value, list) else value)
        for key, value in payload["sinks"][0].items()
    }
    with pytest.raises(ValidationError, match="same length"):
        HeatDistributionRequest.model_validate({**payload, "sinks": [shorter, payload["sinks"][1]]})
    with pytest.raises(ValidationError):
        HeatDistributionRequest.model_validate(
            {**payload, "sinks": [{**payload["sinks"][0], "name": "a name with blanks"}]}
        )
    with pytest.raises(ValidationError, match="exceeds storage_capacity_kwh"):
        HeatDistributionRequest.model_validate(
            {**payload, "storage_capacity_kwh": 100.0, "storage_content_kwh": 200.0}
        )
    with pytest.raises(ValidationError, match="exceeds storage_capacity_kwh"):
        HeatDistributionRequest.model_validate({**payload, "storage_content_kwh": 1.0})


def test_description(service: HeatDistributionService, short: CampaignNetworkConfig) -> None:
    description = service.describe()
    assert description.service == "heat_distribution"
    assert description.model_scope == SITE_NETWORK_SCOPE
    assert description.stateless is True and description.advisory_only is True
    assert description.required_extras == ("api",)
    assert description.modes == ("cascade_rule", "planning")
    assert description.storage_rules == ("k2", "k2b")
    assert description.sample_time_s == 3600.0
    assert description.source_supply_temperature_celsius == short.source.supply_celsius
    assert description.source_return_floor_celsius == short.source.floor_celsius
    assert description.minimum_approach_k == short.approach_k


def test_from_config_is_the_study_configuration(year: CampaignNetworkConfig) -> None:
    assert HeatDistributionService.from_config().model_fingerprint == (
        f"campaign_network:{year.fingerprint}"
    )


# ---------------------------------------------------------------- recovery case


@pytest.fixture(scope="module")
def rule_service() -> RecoveryCaseService:
    return RecoveryCaseService.from_config()


def test_rule_constants_are_those_of_the_study_artifact() -> None:
    """Thresholds and hit rates are read off the trees of Befund 121, not chosen here."""
    path = ROOT / "artifacts" / "runs" / "case_ranking_v1_analysis.json"
    rules = json.loads(path.read_text(encoding="utf-8"))["rule"]
    unpressurised = rules["fixed/physical_unpressurised_95"]["strict"]
    pressurised = rules["fixed/physical_pressurised"]["strict"]
    assert unpressurised["test_accuracy"] == api_heat.HIT_RATE_PHYSICAL_UNPRESSURISED
    assert pressurised["test_accuracy"] == api_heat.HIT_RATE_PHYSICAL_PRESSURISED
    split = f"exhaust_1_minus_2_k <= {api_heat.DIFFERENCE_THRESHOLD_K:.3f}"
    for tree in (unpressurised["text"], pressurised["text"]):
        assert tree.startswith(f"|--- {split}")
        assert "exhaust_available_2_kw <= 0.000" in tree
    assert (
        f"exhaust_2_celsius >  {api_heat.UNPRESSURISED_BRANCH_CELSIUS:.3f}"
        in (unpressurised["text"])
    )
    # the pressurised tree is exactly the rule: nothing but 'False' left of the first split, and
    # right of it 'True' exactly where exhaust 2 has usable heat
    left, right = pressurised["text"].split(f"|--- exhaust_1_minus_2_k >  {split.split()[-1]}")
    assert "class: True" not in left
    without_heat, with_heat = right.split("|   |--- exhaust_available_2_kw >  0.000")
    assert "class: True" not in without_heat and "class: False" not in with_heat
    assert api_heat.USABLE_EXCESS_OFFSET_K == APPROACH_TOTAL_K


@pytest.mark.parametrize(
    ("exhaust_1", "exhaust_2", "ambient", "case", "best", "codes"),
    [
        (70.0, 60.0, 20.0, 1, True, []),
        (60.0, 60.0, 20.0, 1, True, []),
        (59.5, 60.0, 20.0, 1, True, []),  # 0.5 K colder is inside the threshold of the trees
        (59.4, 60.0, 20.0, 2, False, []),
        (50.0, 80.0, 20.0, 2, False, []),
        (45.0, 38.0, 20.0, None, False, ["no_case_named"]),  # exhaust 2 below ambient + 20 K
        (130.0, 120.0, 20.0, 1, True, ["unpressurised_branch_not_represented"]),
        (70.0, 60.0, -15.0, 1, True, ["outside_investigated_box"]),
    ],
)
def test_recovery_case_follows_the_trees_of_the_study(
    rule_service: RecoveryCaseService,
    exhaust_1: float,
    exhaust_2: float,
    ambient: float,
    case: int | None,
    best: bool,
    codes: list[str],
) -> None:
    request = RecoveryCaseRequest(
        exhaust_1_temperature_celsius=exhaust_1,
        exhaust_2_temperature_celsius=exhaust_2,
        ambient_temperature_celsius=ambient,
    )
    response = rule_service.handle(request)
    through_json = RecoveryCaseResponse.model_validate_json(
        rule_service.handle_json(request.model_dump_json())
    )
    assert through_json == response
    assert response.recommended_case == case
    assert response.case_1_expected_best is best
    assert response.exhaust_temperature_difference_k == exhaust_1 - exhaust_2
    assert response.exhaust_2_usable_excess_k == exhaust_2 - ambient - 20.0
    assert [notice.code for notice in response.meta.notices] == codes


def test_recovery_case_answer_carries_reference_validity_and_hit_rate(
    rule_service: RecoveryCaseService,
) -> None:
    response = rule_service.handle(
        RecoveryCaseRequest(
            exhaust_1_temperature_celsius=70.0,
            exhaust_2_temperature_celsius=60.0,
            ambient_temperature_celsius=20.0,
        )
    )
    meta = response.meta
    assert meta.service == "recovery_case"
    assert meta.model_scope == PARTNER_WORKBOOK_SCOPE == "partner workbook rebuild"
    assert meta.parameter_status == "placeholder"
    assert meta.advisory_only is True
    assert "Befund 121" in response.reference
    assert "physical modes" in response.valid_for and "box" in response.valid_for
    assert "not a plant" in response.valid_for
    assert round(response.hit_rate_physical_unpressurised, 3) == 0.966
    assert round(response.hit_rate_physical_pressurised, 3) == 0.971
    assert "case 1 is strictly best" in response.hit_rate_basis


def test_recovery_case_description(rule_service: RecoveryCaseService) -> None:
    description = rule_service.describe()
    box = load_case_ranking_config().box.interval("ambient_celsius")
    assert description.model_scope == PARTNER_WORKBOOK_SCOPE
    assert description.valid_input_ranges["ambient_temperature_celsius"].minimum == box.low == -10.0
    assert description.valid_input_ranges["ambient_temperature_celsius"].maximum == box.high == 40.0
    assert description.cases == (1, 2)
    assert description.difference_threshold_k == -0.541
    assert description.usable_excess_offset_k == 20.0
    assert description.reference == api_heat.REFERENCE
    assert description.hit_rate_physical_pressurised == api_heat.HIT_RATE_PHYSICAL_PRESSURISED
    with pytest.raises(ValueError, match="increasing"):
        RecoveryCaseService((40.0, -10.0))

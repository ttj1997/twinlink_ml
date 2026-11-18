"""Set-point service: end to end, equal to the steady-state optimiser of the headroom study."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("casadi")

from twinlink_ml.api import (
    API_VERSION,
    SCHEMA_VERSION,
    NoSolutionError,
    RequestRejectedError,
    SetpointRequest,
    SetpointResponse,
    SetpointService,
)
from twinlink_ml.api.setpoint import DISTURBANCE_FIELDS
from twinlink_ml.headroom.config import (
    HeadroomStudyConfig,
    load_headroom_config,
    study_process_config,
)
from twinlink_ml.headroom.controllers import SteadyPoint, SteadyStateOptimizer
from twinlink_ml.headroom.symbolic import DISTURBANCE_CHANNELS
from twinlink_ml.simulation import ProcessConfig

BASE = {
    "ambient_temperature_celsius": 20.0,
    "ambient_relative_humidity": 0.5,
    "suspension_solid_fraction": 0.30,
    "spray_rate_kg_per_h": 1.5,
    "product_moisture_limit_kg_per_kg": 0.15,
    "moisture_margin_kg_per_kg": 0.006,
}
WARM_HUMID = {**BASE, "ambient_temperature_celsius": 31.0, "ambient_relative_humidity": 0.8}


@pytest.fixture(scope="module")
def study() -> HeadroomStudyConfig:
    return load_headroom_config()


@pytest.fixture(scope="module")
def process(study: HeadroomStudyConfig) -> ProcessConfig:
    return study_process_config(study)


@pytest.fixture(scope="module")
def service(study: HeadroomStudyConfig, process: ProcessConfig) -> SetpointService:
    return SetpointService(study, process)


def _direct(study: HeadroomStudyConfig, process: ProcessConfig, request: dict) -> SteadyPoint:
    """The research code called directly, on a new optimiser."""
    disturbance = np.array(
        [request.get(name, 0.0) for name in DISTURBANCE_CHANNELS], dtype=np.float64
    )
    bound = request["product_moisture_limit_kg_per_kg"] - request["moisture_margin_kg_per_kg"]
    return SteadyStateOptimizer(study, process).solve(disturbance, bound)


def test_disturbance_order_is_the_one_of_the_optimiser() -> None:
    assert DISTURBANCE_FIELDS == DISTURBANCE_CHANNELS


@pytest.mark.parametrize("payload", [BASE, WARM_HUMID, {**BASE, "inlet_temperature_offset_k": 1.5}])
def test_end_to_end_equals_the_research_code(
    service: SetpointService, study: HeadroomStudyConfig, process: ProcessConfig, payload: dict
) -> None:
    """Bit for bit, also through JSON."""
    expected = _direct(study, process, payload)
    request = SetpointRequest.model_validate(payload)
    response = service.handle(request)
    through_json = SetpointResponse.model_validate_json(
        service.handle_json(request.model_dump_json())
    )
    assert through_json == response
    (setpoint,) = response.setpoints
    assert setpoint.inlet_temperature_celsius == expected.command[0]
    assert setpoint.gas_velocity_m_per_s == expected.command[1]
    assert response.expected_total_power_kw == expected.power_kw
    assert response.expected_bed_temperature_celsius == expected.state[2]
    assert response.expected_product_moisture_kg_per_kg == expected.state[1] / expected.state[0]


def test_response_says_what_it_is(service: SetpointService, process: ProcessConfig) -> None:
    response = service.handle(SetpointRequest.model_validate(BASE))
    meta = response.meta
    assert (meta.service, meta.api_version, meta.schema_version) == (
        "setpoint",
        API_VERSION,
        SCHEMA_VERSION,
    )
    assert meta.parameter_status == process.provenance.status == "placeholder"
    assert meta.advisory_only is True
    assert meta.notices == ()
    assert process.fingerprint in meta.model_fingerprint
    (setpoint,) = response.setpoints
    assert setpoint.time_offset_s == 0.0
    assert setpoint.spray_rate_kg_per_h is None  # the throughput is given, not decided
    limits = response.applied_limits
    assert limits.moisture_bound_kg_per_kg == 0.15 - 0.006
    assert response.expected_product_moisture_kg_per_kg <= limits.moisture_bound_kg_per_kg + 1e-6
    assert response.expected_bed_temperature_celsius <= limits.bed_temperature_limit_celsius + 1e-6
    box = process.actions
    assert (
        limits.inlet_temperature_minimum_celsius - 1e-6
        <= setpoint.inlet_temperature_celsius
        <= limits.inlet_temperature_maximum_celsius + 1e-6
    )
    assert limits.gas_velocity_minimum_m_per_s == box.gas_velocity_m_per_s.minimum
    assert limits.gas_velocity_maximum_m_per_s == box.gas_velocity_m_per_s.maximum


def test_service_is_stateless(service: SetpointService) -> None:
    """The answer does not depend on earlier requests (the optimiser alone would warm start)."""
    first = SetpointRequest.model_validate(BASE)
    before = service.handle(first)
    service.handle(SetpointRequest.model_validate(WARM_HUMID))
    assert service.handle(first) == before


def test_narrowed_bounds_are_applied(
    service: SetpointService, study: HeadroomStudyConfig, process: ProcessConfig
) -> None:
    free = service.handle(SetpointRequest.model_validate(BASE))
    assert free.setpoints[0].gas_velocity_m_per_s < 1.0  # the lower bound is the lever
    payload = {**BASE, "gas_velocity_minimum_m_per_s": 1.0, "bed_temperature_limit_celsius": 70.0}
    response = service.handle(SetpointRequest.model_validate(payload))
    assert response.applied_limits.gas_velocity_minimum_m_per_s == 1.0
    assert response.applied_limits.bed_temperature_limit_celsius == 70.0
    assert response.setpoints[0].gas_velocity_m_per_s >= 1.0 - 1e-6
    assert response.expected_total_power_kw > free.expected_total_power_kw

    gas = process.actions.gas_velocity_m_per_s
    narrowed_process = dataclasses.replace(
        process,
        actions=dataclasses.replace(
            process.actions, gas_velocity_m_per_s=dataclasses.replace(gas, minimum=1.0)
        ),
    )
    narrowed_study = dataclasses.replace(
        study, line=dataclasses.replace(study.line, bed_temperature_limit_celsius=70.0)
    )
    expected = _direct(narrowed_study, narrowed_process, payload)
    assert response.setpoints[0].gas_velocity_m_per_s == expected.command[1]
    assert response.setpoints[0].inlet_temperature_celsius == expected.command[0]


@pytest.mark.parametrize(
    "override",
    [
        {"gas_velocity_minimum_m_per_s": 0.5},
        {"inlet_temperature_maximum_celsius": 150.0},
        {"bed_temperature_limit_celsius": 95.0},
    ],
)
def test_bounds_can_only_be_narrowed(service: SetpointService, override: dict) -> None:
    with pytest.raises(RequestRejectedError, match="only be narrowed"):
        service.handle(SetpointRequest.model_validate({**BASE, **override}))


def test_disturbance_outside_the_model_range_is_remarked(service: SetpointService) -> None:
    response = service.handle(
        SetpointRequest.model_validate({**BASE, "ambient_temperature_celsius": 38.0})
    )
    assert [notice.code for notice in response.meta.notices] == ["input_outside_model_range"]
    assert "ambient_temperature_celsius 38.0" in response.meta.notices[0].message


class _Failing:
    """An optimiser that finds no admissible point."""

    def solve(self, disturbance: np.ndarray, moisture_bound: float) -> SteadyPoint:
        del disturbance, moisture_bound
        raise RuntimeError("steady optimum failed: Infeasible_Problem_Detected")


def test_no_solution_is_an_error_with_a_code(
    study: HeadroomStudyConfig, process: ProcessConfig
) -> None:
    """The optimiser is injected; its failure becomes ``no_solution``."""
    service = SetpointService(study, process, optimizer_factory=lambda _study, _process: _Failing())
    with pytest.raises(NoSolutionError, match="Infeasible") as caught:
        service.handle(SetpointRequest.model_validate(BASE))
    assert caught.value.code == "no_solution"


def test_description(service: SetpointService, study: HeadroomStudyConfig) -> None:
    description = service.describe()
    assert description.service == "setpoint"
    assert description.stateless is True and description.advisory_only is True
    assert description.required_extras == ("api", "optimal-control")
    assert description.decided_fields == ("inlet_temperature_celsius", "gas_velocity_m_per_s")
    assert description.fixed_nozzle_pressure_bar == study.line.nozzle_pressure_bar
    assert description.setpoints_per_response == 1
    assert description.valid_input_ranges["gas_velocity_m_per_s"].minimum == 0.8
    assert (
        description.model_fingerprint
        == service.handle(SetpointRequest.model_validate(BASE)).meta.model_fingerprint
    )


def test_from_config_is_the_laboratory_line(service: SetpointService) -> None:
    assert SetpointService.from_config().model_fingerprint == service.model_fingerprint

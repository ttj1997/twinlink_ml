"""Calibration service: bias and optional fit on synthetic sections, equal to the research code."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from pydantic import ValidationError

from twinlink_ml.api import (
    API_VERSION,
    SCHEMA_VERSION,
    CalibrationRequest,
    CalibrationResponse,
    CalibrationService,
    NoSolutionError,
    RequestRejectedError,
)
from twinlink_ml.plant_calibration import (
    ModelContext,
    PlantCalibrationConfig,
    apply_bias,
    bias_offsets,
    error_metrics,
    evaluate_rows,
    fit_parameters,
    load_plant_calibration_config,
)
from twinlink_ml.plant_scale.model import PlantPoint

# Synthetic sections (invented round numbers of plant-like size, not plant data; the operating
# points of tests/test_plant_calibration.py).
SECTION_A = {
    "product": "product_1",
    "ambient_temperature_celsius": 10.0,
    "process_air_flow_nm3_per_h": 30000.0,
    "atomising_air_flow_nm3_per_h": 600.0,
    "supply_air_temperature_celsius": 85.0,
    "spray_rate_kg_per_h": 1500.0,
    "bed_temperature_celsius": 55.0,
    "exhaust_temperature_celsius": 47.0,
}
SECTION_B = {
    "product": "product_2",
    "ambient_temperature_celsius": 15.0,
    "process_air_flow_nm3_per_h": 28000.0,
    "atomising_air_flow_nm3_per_h": 600.0,
    "supply_air_temperature_celsius": 82.0,
    "spray_rate_kg_per_h": 900.0,
    "bed_temperature_celsius": 62.0,
    "exhaust_temperature_celsius": 56.0,
}


@pytest.fixture(scope="module")
def service() -> CalibrationService:
    return CalibrationService.from_config()


@pytest.fixture(scope="module")
def config() -> PlantCalibrationConfig:
    return load_plant_calibration_config()


@pytest.fixture(scope="module")
def context(service: CalibrationService) -> ModelContext:
    return service._context


def _rows(sections: list[dict]) -> list[tuple[PlantPoint, str]]:
    """The operating points as the research code takes them."""
    return [
        (
            PlantPoint(
                f"section_{index}",
                s["ambient_temperature_celsius"],
                s["process_air_flow_nm3_per_h"],
                s["atomising_air_flow_nm3_per_h"],
                s["supply_air_temperature_celsius"],
                s["spray_rate_kg_per_h"],
                s["bed_temperature_celsius"],
                s["exhaust_temperature_celsius"],
            ),
            s["product"],
        )
        for index, s in enumerate(sections)
    ]


def _measured(sections: list[dict]) -> np.ndarray:
    return np.array(
        [
            v
            for s in sections
            for v in (s["bed_temperature_celsius"], s["exhaust_temperature_celsius"])
        ]
    )


def test_bias_equals_the_research_code(
    service: CalibrationService, config: PlantCalibrationConfig, context: ModelContext
) -> None:
    """Bit for bit, also through JSON: uncalibrated model, bias, errors before and after."""
    sections = [SECTION_A, SECTION_B]
    model = evaluate_rows(context, _rows(sections), config.nominal(), {})
    measured = _measured(sections)
    bias = bias_offsets(model, measured)
    before = error_metrics(model, measured)
    after = error_metrics(apply_bias(model, bias), measured)

    request = CalibrationRequest.model_validate({"sections": sections})
    response = service.handle(request)
    assert (
        CalibrationResponse.model_validate_json(service.handle_json(request.model_dump_json()))
        == response
    )
    assert response.model_bed_temperature_celsius == tuple(model[0::2].tolist())
    assert response.model_exhaust_temperature_celsius == tuple(model[1::2].tolist())
    assert response.bias.bed_temperature_bias_k == bias["bed"]
    assert response.bias.exhaust_temperature_bias_k == bias["exhaust"]
    assert response.errors_uncalibrated.bed_temperature_rms_k == before["bed"]["rms_k"]
    assert (
        response.errors_uncalibrated.exhaust_temperature_maximum_k
        == (before["exhaust"]["max_abs_k"])
    )
    assert response.errors_with_bias.bed_temperature_rms_k == after["bed"]["rms_k"]
    assert response.errors_with_bias.exhaust_temperature_maximum_k == after["exhaust"]["max_abs_k"]
    assert response.errors_with_bias.bed_temperature_rms_k < before["bed"]["rms_k"]


def test_response_says_what_it_is(service: CalibrationService, context: ModelContext) -> None:
    response = service.handle(
        CalibrationRequest.model_validate({"sections": [SECTION_A, SECTION_B]})
    )
    meta = response.meta
    assert (meta.service, meta.api_version, meta.schema_version) == (
        "calibration",
        API_VERSION,
        SCHEMA_VERSION,
    )
    assert meta.model_scope == service.describe().model_scope == "plant A, plant-near model"
    assert meta.parameter_status == context.base.provenance.status == "placeholder"
    assert meta.advisory_only is True
    assert meta.notices == ()
    assert context.base.fingerprint in meta.model_fingerprint
    assert response.n_sections == 2
    assert response.parameters is None  # the bias is the default; nothing is fitted unasked
    assert len(response.calibration_fingerprint) == 64


def test_service_is_stateless(service: CalibrationService) -> None:
    first = CalibrationRequest.model_validate({"sections": [SECTION_A, SECTION_B]})
    before = service.handle(first)
    service.handle(CalibrationRequest.model_validate({"sections": [SECTION_B]}))
    assert service.handle(first) == before


def test_one_section_is_remarked(service: CalibrationService) -> None:
    response = service.handle(CalibrationRequest.model_validate({"sections": [SECTION_A]}))
    assert [notice.code for notice in response.meta.notices] == ["bias_from_one_section"]
    assert response.errors_with_bias.bed_temperature_maximum_k == pytest.approx(0.0, abs=1e-12)
    assert response.bias.bed_temperature_bias_k == pytest.approx(
        SECTION_A["bed_temperature_celsius"] - response.model_bed_temperature_celsius[0]
    )


def test_fit_equals_the_research_code(
    service: CalibrationService, config: PlantCalibrationConfig, context: ModelContext
) -> None:
    """Synthetic measurements from the model with a known UA and distance: same fit, bit for bit."""
    free = ("ua_bed_w_per_k", "offset_product_1_k")
    truth = config.nominal() | {"ua_bed_w_per_k": 1500.0, "offset_product_1_k": 6.0}
    synthetic = evaluate_rows(context, _rows([SECTION_A]), truth)
    section = SECTION_A | {
        "bed_temperature_celsius": float(synthetic[0]),
        "exhaust_temperature_celsius": float(synthetic[1]),
    }
    rows, measured = _rows([section]), _measured([section])
    sigma = np.array([config.sigma_bed_k, config.sigma_exhaust_k])
    guesses: dict[str, float] = {}
    fit = fit_parameters(
        lambda theta: evaluate_rows(context, rows, theta, guesses),
        measured,
        sigma,
        config.nominal(),
        free,
        scales={p.name: p.scale for p in config.parameters},
        bounds={p.name: p.bounds for p in config.parameters},
        start={name: config.calibration.start[name] for name in free},
        max_nfev=config.calibration.max_nfev,
    )
    after = error_metrics(evaluate_rows(context, rows, fit.theta, {}), measured)

    response = service.handle(
        CalibrationRequest.model_validate({"sections": [section], "calibrate_parameters": free})
    )
    parameters = response.parameters
    assert parameters is not None
    assert parameters.values == {name: fit.theta[name] for name in free}
    assert parameters.weighted_squared_residual_sum == fit.ssr
    assert (parameters.n_residuals, parameters.evaluations) == (fit.n_residuals, fit.evaluations)
    assert parameters.at_bound == fit.at_bound
    assert parameters.errors.exhaust_temperature_maximum_k == after["exhaust"]["max_abs_k"]
    # the synthetic truth is recovered (as tests/test_plant_calibration.py)
    assert parameters.values["ua_bed_w_per_k"] == pytest.approx(1500.0, rel=1e-3)
    assert parameters.values["offset_product_1_k"] == pytest.approx(6.0, abs=1e-3)
    codes = [notice.code for notice in response.meta.notices]
    assert codes[:2] == ["bias_from_one_section", "calibration_not_validated"]
    plain = service.handle(CalibrationRequest.model_validate({"sections": [section]}))
    assert plain.bias == response.bias  # the bias does not depend on the fit
    assert plain.calibration_fingerprint != response.calibration_fingerprint


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"sections": [SECTION_A | {"product": "product_9"}]}, "unknown product role"),
        (
            {"sections": [SECTION_A, SECTION_B], "calibrate_parameters": ["max_saturation_degree"]},
            "not calibratable",
        ),
        (
            {"sections": [SECTION_A, SECTION_B], "calibrate_parameters": ["cp_solid_j_per_kg_k"]},
            "not calibratable",
        ),
        (
            {
                "sections": [SECTION_A, SECTION_B],
                "calibrate_parameters": [
                    "ua_bed_w_per_k",
                    "offset_product_1_k",
                    "offset_product_2_k",
                    "freeboard_ua_w_per_k",
                ],
            },
            "at most 3 parameters",
        ),
        (
            {
                "sections": [SECTION_A],
                "calibrate_parameters": [
                    "ua_bed_w_per_k",
                    "offset_product_1_k",
                    "freeboard_ua_w_per_k",
                ],
            },
            "cannot be fitted to 2 measured values",
        ),
        (
            {"sections": [SECTION_A], "calibrate_parameters": ["offset_product_2_k"]},
            "no section of the product",
        ),
    ],
)
def test_what_cannot_be_calibrated_is_rejected(
    service: CalibrationService, payload: dict, message: str
) -> None:
    with pytest.raises(RequestRejectedError, match=message):
        service.handle(CalibrationRequest.model_validate(payload))


def test_malformed_requests_fail_validation() -> None:
    with pytest.raises(ValidationError):
        CalibrationRequest.model_validate({"sections": []})
    with pytest.raises(ValidationError, match="twice"):
        CalibrationRequest.model_validate(
            {"sections": [SECTION_A], "calibrate_parameters": ["ua_bed_w_per_k", "ua_bed_w_per_k"]}
        )
    with pytest.raises(ValidationError):
        CalibrationRequest.model_validate({"sections": [SECTION_A | {"plant_tag": 1.0}]})


def test_missing_steady_state_is_an_error_with_a_code(
    config: PlantCalibrationConfig, context: ModelContext
) -> None:
    """A model context whose velocity iteration cannot converge: ``no_solution``."""
    strict = dataclasses.replace(
        context, steady_kwargs={**context.steady_kwargs, "iterations": 1, "relative_tolerance": 0.0}
    )
    with pytest.raises(NoSolutionError, match=r"sections \[0\]") as caught:
        CalibrationService(strict, config).handle(
            CalibrationRequest.model_validate({"sections": [SECTION_A]})
        )
    assert caught.value.code == "no_solution"


def test_description(service: CalibrationService, config: PlantCalibrationConfig) -> None:
    description = service.describe()
    assert description.service == "calibration"
    assert description.stateless is True and description.advisory_only is True
    assert description.required_extras == ("api",)
    assert description.products == ("product_1", "product_2")
    assert description.calibratable_parameters == (
        "ua_bed_w_per_k",
        "offset_product_1_k",
        "offset_product_2_k",
        "freeboard_ua_w_per_k",
    )
    assert description.maximum_parameters == config.sensitivity.max_parameters == 3
    assert description.bed_temperature_sigma_k == config.sigma_bed_k
    assert description.valid_input_ranges["offset_product_1_k"].maximum == 30.0
    assert description.model_fingerprint == service.model_fingerprint

"""Schemas of :mod:`twinlink_ml.api`: round trip, strictness, units in the names, exported files."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

import twinlink_ml
from twinlink_ml.api import (
    SCHEMA_VERSION,
    ForecastRequest,
    MissingExtraError,
    SetpointRequest,
)
from twinlink_ml.api import schemas as api_schemas
from twinlink_ml.api.base import ApiModel, require_extra
from twinlink_ml.api.schema_export import collect_schemas, render_schema, service_classes
from twinlink_ml.api.schemas import EXOGENOUS_FIELDS, TARGET_FIELD_OF_MODEL_NAME
from twinlink_ml.simulation import ProcessConfig

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "docs" / "api"

UNIT_SUFFIXES = (
    "_celsius",
    "_k",
    "_m_per_s",
    "_kg_per_h",
    "_nm3_per_h",
    "_bar",
    "_kg_per_kg",
    "_kw",
    "_kwh",
    "_kw_per_k",
    "_kg",
    "_s",
    "_per_h",
)
DIMENSIONLESS = {
    "ambient_relative_humidity",
    "suspension_solid_fraction",
    "utilisation",
    "replaced_supply_efficiency",
    "pump_electricity_fraction",
    "weighted_squared_residual_sum",
    "hit_rate_physical_unpressurised",
    "hit_rate_physical_pressurised",
}

SERIES = {name: [1.0, 2.0, 3.0] for name in EXOGENOUS_FIELDS}


def _forecast_request(**overrides: object) -> dict:
    return {
        "sample_time_s": 25.0,
        "observation_mode": "soft_sensor",
        "history": SERIES,
        "planned": SERIES,
        **overrides,
    }


def _setpoint_request(**overrides: object) -> dict:
    return {
        "ambient_temperature_celsius": 20.0,
        "ambient_relative_humidity": 0.5,
        "suspension_solid_fraction": 0.3,
        "spray_rate_kg_per_h": 1.5,
        "product_moisture_limit_kg_per_kg": 0.15,
        "moisture_margin_kg_per_kg": 0.006,
        **overrides,
    }


def test_import_loads_neither_torch_nor_casadi() -> None:
    """The package is importable on a host that has only the extra ``api``."""
    code = (
        "import sys, twinlink_ml.api; "
        "print(','.join(m for m in ('torch', 'casadi') if m in sys.modules))"
    )
    source = str(Path(twinlink_ml.__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": source},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ""


def test_missing_extra_names_the_extra() -> None:
    with pytest.raises(MissingExtraError, match=r"twinlink-ml\[some-extra\]") as caught:
        require_extra("a_module_that_does_not_exist", "some-extra")
    assert caught.value.code == "missing_extra"
    assert caught.value.extra == "some-extra"


@pytest.mark.parametrize(
    ("model", "payload"),
    [(ForecastRequest, _forecast_request()), (SetpointRequest, _setpoint_request())],
)
def test_request_round_trip(model: type[ApiModel], payload: dict) -> None:
    request = model.model_validate(payload)
    assert request.schema_version == SCHEMA_VERSION
    again = model.model_validate_json(request.model_dump_json())
    assert again == request
    assert again.model_dump() == request.model_dump()


def test_float_survives_the_json_round_trip_bit_for_bit() -> None:
    value = 0.1 + 0.2  # not representable as a short decimal
    request = SetpointRequest.model_validate(_setpoint_request(ambient_temperature_celsius=value))
    again = SetpointRequest.model_validate_json(request.model_dump_json())
    assert again.ambient_temperature_celsius == value


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        SetpointRequest.model_validate(_setpoint_request(some_plant_tag=1.0))


def test_non_finite_value_is_rejected() -> None:
    text = SetpointRequest.model_validate(_setpoint_request()).model_dump_json()
    with pytest.raises(ValidationError):
        SetpointRequest.model_validate_json(text.replace("20.0", "NaN", 1))
    with pytest.raises(ValidationError):
        SetpointRequest.model_validate(_setpoint_request(spray_rate_kg_per_h=float("inf")))


def test_schema_version_major_must_match_minor_may_differ() -> None:
    assert SetpointRequest.model_validate(_setpoint_request(schema_version="1.7"))
    with pytest.raises(ValidationError, match="major version"):
        SetpointRequest.model_validate(_setpoint_request(schema_version="2.0"))
    with pytest.raises(ValidationError, match=r"major\.minor"):
        SetpointRequest.model_validate(_setpoint_request(schema_version="one"))


def test_series_of_unequal_length_are_rejected() -> None:
    crooked = {**SERIES, "gas_velocity_m_per_s": [1.0, 2.0]}
    with pytest.raises(ValidationError, match="same length"):
        ForecastRequest.model_validate(_forecast_request(history=crooked))


def test_margin_must_stay_below_the_limit() -> None:
    with pytest.raises(ValidationError, match="smaller than the moisture limit"):
        SetpointRequest.model_validate(_setpoint_request(moisture_margin_kg_per_kg=0.15))


def test_inputs_are_the_channels_of_the_process_configuration() -> None:
    config = ProcessConfig()
    assert (*config.actions.names, *config.disturbances.names) == EXOGENOUS_FIELDS


def _schema_models() -> list[type[ApiModel]]:
    return [
        value
        for name in api_schemas.__all__
        if isinstance(value := getattr(api_schemas, name), type) and issubclass(value, ApiModel)
    ]


def test_every_physical_field_carries_its_unit() -> None:
    """A numeric field of a request or response ends in a unit or is a named fraction."""
    checked = 0
    for model in _schema_models():
        for name, field in model.model_fields.items():
            annotation = str(field.annotation)
            if "float" not in annotation or "dict" in annotation:
                continue  # the keys of a dictionary carry the units
            checked += 1
            assert name.endswith(UNIT_SUFFIXES) or name in DIMENSIONLESS, (
                f"{model.__name__}.{name} names no unit"
            )
    assert checked >= 40
    assert set(TARGET_FIELD_OF_MODEL_NAME.values()) <= {
        name for model in _schema_models() for name in model.model_fields
    }


def test_exported_schema_files_are_current() -> None:
    """``docs/api`` holds exactly what ``scripts/export_api_schemas.py`` writes now."""
    expected = {name: render_schema(schema) for name, schema in collect_schemas().items()}
    assert len(expected) == 3 * len(service_classes()) + 1
    found = {path.name: path.read_text(encoding="utf-8") for path in SCHEMA_DIR.glob("*.json")}
    assert sorted(found) == sorted(expected)
    for name, text in expected.items():
        assert found[name] == text, f"docs/api/{name} is stale: run scripts/export_api_schemas.py"

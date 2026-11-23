"""Common parts of the module interface: service ABC, response metadata, errors.

The interface is framework independent on purpose (``docs/47_module_api.md``): a service is a
stateless mapping from one validated request to one response. A transport layer (REST, an asset
administration shell, a message bus) is an adapter around :meth:`Service.handle_json` and is not
part of this package; the technique is the decision of the consortium coordinator.

Three properties hold for every service and are part of the contract, not of an implementation:

* **Stateless.** A response depends on the request and on the injected model only, never on
  earlier requests.
* **Advisory.** Every response carries ``advisory_only = true``: the digital twin recommends, it
  does not change operating parameters of a plant.
* **Labelled.** Every response names the schema version, the fingerprint of the model that
  produced it, the plant and model it stands for (``model_scope``) and the status of the process
  parameters behind that model (``placeholder`` for every model in the repository at the time of
  writing).
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, field_validator

__all__ = [
    "API_VERSION",
    "LAB_LINE_SCOPE",
    "PARTNER_WORKBOOK_SCOPE",
    "PLANT_A_SCOPE",
    "SCHEMA_VERSION",
    "SITE_NETWORK_SCOPE",
    "ApiModel",
    "BundleIntegrityError",
    "MissingExtraError",
    "ModelFailureError",
    "NoSolutionError",
    "Notice",
    "ParameterStatus",
    "RequestRejectedError",
    "ResponseMeta",
    "Service",
    "ServiceDescription",
    "ServiceError",
    "ValueRange",
    "VersionedRequest",
    "require_extra",
]

API_VERSION = "1.0.0"
"""Version of the package :mod:`twinlink_ml.api` (services and model-bundle code)."""

SCHEMA_VERSION = "1.0"
"""Version of the request and response schemas, ``major.minor``. A request is accepted when its
major version equals the major version of the service; a minor version only adds optional fields."""

LAB_LINE_SCOPE = "lab line, grey-box"
PLANT_A_SCOPE = "plant A, plant-near model"
SITE_NETWORK_SCOPE = "site heat network, campaign-network model"
PARTNER_WORKBOOK_SCOPE = "partner workbook rebuild"
"""Which plant or line and which model a service computes with (``model_scope`` of every
description and response). The services do not share one model: a result of one scope must not be
applied to another (a calibration of plant A is not a correction of the laboratory line)."""

ParameterStatus = Literal["placeholder", "partially_calibrated", "calibrated"]
"""Status of the process parameters behind a model, the values of
``twinlink_ml.simulation.config.ConfigStatus``."""


# ------------------------------------------------------------------------ errors


class ServiceError(Exception):
    """Base class of the errors a service raises on purpose; ``code`` is machine readable."""

    code: ClassVar[str] = "service_error"


class RequestRejectedError(ServiceError):
    """The request is well formed but does not fit the model (grid, history length, mode, box)."""

    code: ClassVar[str] = "request_rejected"


class NoSolutionError(ServiceError):
    """The optimiser found no admissible point for the request."""

    code: ClassVar[str] = "no_solution"


class ModelFailureError(ServiceError):
    """The model returned no usable result (non-finite values) and no fallback exists."""

    code: ClassVar[str] = "model_failure"


class BundleIntegrityError(ServiceError):
    """A model bundle is damaged, altered or not the one that was expected."""

    code: ClassVar[str] = "bundle_integrity"


class MissingExtraError(ServiceError):
    """An optional dependency of the service is not installed; names the extra to install."""

    code: ClassVar[str] = "missing_extra"

    def __init__(self, module: str, extra: str) -> None:
        super().__init__(
            f"'{module}' is not installed; this service needs the extra '{extra}': "
            f"pip install 'twinlink-ml[{extra}]'"
        )
        self.module = module
        self.extra = extra


def require_extra(module: str, extra: str) -> None:
    """Raise :class:`MissingExtraError` when ``module`` cannot be imported.

    The package itself imports neither PyTorch nor CasADi; a service asks for its extra at the
    moment it needs it, so that a host without one of them can still use the other services.
    """
    if importlib.util.find_spec(module) is None:
        raise MissingExtraError(module, extra)


# ------------------------------------------------------------------------ models


class ApiModel(BaseModel):
    """Base of every schema: unknown fields are an error, values are immutable, no NaN or inf."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class VersionedRequest(ApiModel):
    """Base of every request: carries the schema version it was written against."""

    schema_version: str = SCHEMA_VERSION

    @field_validator("schema_version")
    @classmethod
    def _major_version_matches(cls, value: str) -> str:
        major, _, minor = value.partition(".")
        if not (major.isdigit() and minor.isdigit()):
            raise ValueError(f"schema_version must be 'major.minor', got {value!r}")
        expected = SCHEMA_VERSION.partition(".")[0]
        if major != expected:
            raise ValueError(
                f"schema_version {value!r} has major version {major}, this service speaks "
                f"{SCHEMA_VERSION}"
            )
        return value


class ValueRange(ApiModel):
    """Closed interval of a quantity, in the unit named by the key it is stored under."""

    minimum: float
    maximum: float


class Notice(ApiModel):
    """A remark on a response that does not make it invalid (for example an input out of range)."""

    code: str
    message: str


class ResponseMeta(ApiModel):
    """What every response says about itself."""

    schema_version: str
    service: str
    api_version: str
    model_fingerprint: str
    model_label: str
    model_scope: str
    parameter_status: ParameterStatus
    advisory_only: Literal[True] = True
    notices: tuple[Notice, ...] = ()


class ServiceDescription(ApiModel):
    """Self-description of a service: what an adapter needs to publish it."""

    service: str
    api_version: str
    schema_version: str
    model_fingerprint: str
    model_label: str
    model_scope: str
    parameter_status: ParameterStatus
    advisory_only: Literal[True] = True
    stateless: Literal[True] = True
    required_extras: tuple[str, ...]
    valid_input_ranges: dict[str, ValueRange]


# ----------------------------------------------------------------------- service

RequestT = TypeVar("RequestT", bound=ApiModel)
ResponseT = TypeVar("ResponseT", bound=ApiModel)


class Service(ABC, Generic[RequestT, ResponseT]):
    """A stateless mapping from a request to a response.

    Sub-classes set ``name``, ``request_model``, ``response_model`` and ``description_model`` and
    receive everything they compute with (a model bundle, a process configuration, an optimiser
    factory) through their constructor.
    """

    name: ClassVar[str]
    request_model: ClassVar[type[ApiModel]]
    response_model: ClassVar[type[ApiModel]]
    description_model: ClassVar[type[ServiceDescription]] = ServiceDescription

    @abstractmethod
    def describe(self) -> ServiceDescription:
        """Version, model fingerprint, parameter status and valid input ranges."""

    @abstractmethod
    def handle(self, request: RequestT) -> ResponseT:
        """Answer one request. Raises a :class:`ServiceError` when it cannot."""

    def handle_json(self, payload: str | bytes) -> str:
        """Validate a JSON request, answer it and return the response as JSON.

        This is the whole surface an adapter needs. A ``pydantic.ValidationError`` means the
        payload does not follow the schema; a :class:`ServiceError` carries a ``code``.
        """
        request = self.request_model.model_validate_json(payload)
        response = self.handle(request)  # type: ignore[arg-type]
        return response.model_dump_json()

    @classmethod
    def json_schemas(cls) -> dict[str, dict[str, Any]]:
        """JSON Schema of the request, the response and the self-description."""
        return {
            "request": cls.request_model.model_json_schema(),
            "response": cls.response_model.model_json_schema(),
            "description": cls.description_model.model_json_schema(),
        }

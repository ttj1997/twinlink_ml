"""Framework-independent module interface of the ML and control pipeline.

A narrow, stable layer above the research code (``docs/NN_module_api.md``): services behind one
ABC, requests and responses as versioned pydantic models with JSON Schemas, and a model-bundle
format for the trained models. The research code is only imported, never changed.

Needs the extra ``api`` (pydantic). Importing this package loads neither PyTorch nor CasADi; a
service asks for its own extra when it needs it and names it in the error
(:class:`~twinlink_ml.api.base.MissingExtraError`).

Stage 1 (this package): :class:`ForecastService`, :class:`SetpointService`, the model bundle.
A calibration service and a heat-distribution service are planned as stage 2.
"""

try:
    import pydantic as _pydantic  # noqa: F401
except ImportError as _error:  # pragma: no cover - depends on the environment
    raise ImportError("twinlink_ml.api needs pydantic: pip install 'twinlink-ml[api]'") from _error

from .base import (
    API_VERSION,
    SCHEMA_VERSION,
    BundleIntegrityError,
    MissingExtraError,
    ModelFailureError,
    NoSolutionError,
    Notice,
    RequestRejectedError,
    ResponseMeta,
    Service,
    ServiceDescription,
    ServiceError,
    ValueRange,
)
from .bundle import (
    BundleManifest,
    ModelBundle,
    build_bundle,
    fit_bundle,
    instantiate_model,
    load_bundle,
)
from .forecast import ForecastService
from .schemas import (
    ForecastDescription,
    ForecastRequest,
    ForecastResponse,
    MeasuredTargetSeries,
    ProcessInputSeries,
    SetpointDescription,
    SetpointRequest,
    SetpointResponse,
    TimedSetpoint,
)
from .setpoint import SetpointService

__all__ = [
    "API_VERSION",
    "SCHEMA_VERSION",
    "BundleIntegrityError",
    "BundleManifest",
    "ForecastDescription",
    "ForecastRequest",
    "ForecastResponse",
    "ForecastService",
    "MeasuredTargetSeries",
    "MissingExtraError",
    "ModelBundle",
    "ModelFailureError",
    "NoSolutionError",
    "Notice",
    "ProcessInputSeries",
    "RequestRejectedError",
    "ResponseMeta",
    "Service",
    "ServiceDescription",
    "ServiceError",
    "SetpointDescription",
    "SetpointRequest",
    "SetpointResponse",
    "SetpointService",
    "TimedSetpoint",
    "ValueRange",
    "build_bundle",
    "fit_bundle",
    "instantiate_model",
    "load_bundle",
]

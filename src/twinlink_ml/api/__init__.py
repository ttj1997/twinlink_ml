"""Framework-independent module interface of the ML and control pipeline.

A narrow, stable layer above the research code (``docs/47_module_api.md``): services behind one
ABC, requests and responses as versioned pydantic models with JSON Schemas, and a model-bundle
format for the trained models. The package imports the research code and holds no copy of
its arithmetic.

Needs the extra ``api`` (pydantic). Importing this package loads neither PyTorch nor CasADi; a
service asks for its own extra when it needs it and names it in the error
(:class:`~twinlink_ml.api.base.MissingExtraError`).

Services: :class:`ForecastService`, :class:`SetpointService` (stage 1, with the model bundle),
:class:`CalibrationService`, :class:`HeatDistributionService` and :class:`RecoveryCaseService`
(stage 2).
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
from .calibration import CalibrationService
from .forecast import ForecastService
from .heat import HeatDistributionService, RecoveryCaseService
from .schemas import (
    CalibrationDescription,
    CalibrationRequest,
    CalibrationResponse,
    ForecastDescription,
    ForecastRequest,
    ForecastResponse,
    HeatDistributionDescription,
    HeatDistributionRequest,
    HeatDistributionResponse,
    HeatSink,
    MeasuredTargetSeries,
    ProcessInputSeries,
    RecoveryCaseDescription,
    RecoveryCaseRequest,
    RecoveryCaseResponse,
    SetpointDescription,
    SetpointRequest,
    SetpointResponse,
    SteadySection,
    TimedSetpoint,
)
from .setpoint import SetpointService

__all__ = [
    "API_VERSION",
    "SCHEMA_VERSION",
    "BundleIntegrityError",
    "BundleManifest",
    "CalibrationDescription",
    "CalibrationRequest",
    "CalibrationResponse",
    "CalibrationService",
    "ForecastDescription",
    "ForecastRequest",
    "ForecastResponse",
    "ForecastService",
    "HeatDistributionDescription",
    "HeatDistributionRequest",
    "HeatDistributionResponse",
    "HeatDistributionService",
    "HeatSink",
    "MeasuredTargetSeries",
    "MissingExtraError",
    "ModelBundle",
    "ModelFailureError",
    "NoSolutionError",
    "Notice",
    "ProcessInputSeries",
    "RecoveryCaseDescription",
    "RecoveryCaseRequest",
    "RecoveryCaseResponse",
    "RecoveryCaseService",
    "RequestRejectedError",
    "ResponseMeta",
    "Service",
    "ServiceDescription",
    "ServiceError",
    "SetpointDescription",
    "SetpointRequest",
    "SetpointResponse",
    "SetpointService",
    "SteadySection",
    "TimedSetpoint",
    "ValueRange",
    "build_bundle",
    "fit_bundle",
    "instantiate_model",
    "load_bundle",
]

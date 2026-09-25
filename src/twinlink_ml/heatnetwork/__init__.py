"""Wärmenetz-Modul: kaskadierte Wärmerückgewinnung Abluft -> Prozessluft.

Siehe ``docs/06_waermenetz.md`` für Zweck, Abgrenzung und die drei Korrekturen gegenüber
der Excel-Vorlage des Verbundpartners.
"""

from .balance import (
    BINDING_DEW_POINT,
    BINDING_LABELS,
    BINDING_PINCH,
    BINDING_SINK_SATISFIED,
    HeatNetworkResult,
    SinkState,
    SourceState,
    binding_label,
    evaluate_cascade,
)
from .config import (
    DEFAULT_COLD_TEMPERATURE_CELSIUS,
    DEFAULT_DEW_POINT_APPROACH_K,
    DEFAULT_FUEL_LOWER_HEATING_VALUE_J_PER_KG,
    DEFAULT_HEATER_EFFICIENCY,
    DEFAULT_MINIMUM_PINCH_K,
    DEFAULT_RECOVERY_MODE,
    HeatNetworkConfig,
    HeatSinkConfig,
    HeatSourceConfig,
    RecoveryMode,
)
from .pinch import (
    PinchResult,
    StreamSegment,
    cold_segment_for_sink,
    composite_curve_kw,
    compute_pinch_target,
    dew_point_of_source_celsius,
    evaluate_pinch_target,
    hot_segments_for_source,
)
from .routing import FixedRouting, HeatNetworkRouting, MatrixRouting, resolve_cascade_order

__all__ = [
    "BINDING_DEW_POINT",
    "BINDING_LABELS",
    "BINDING_PINCH",
    "BINDING_SINK_SATISFIED",
    "DEFAULT_COLD_TEMPERATURE_CELSIUS",
    "DEFAULT_DEW_POINT_APPROACH_K",
    "DEFAULT_FUEL_LOWER_HEATING_VALUE_J_PER_KG",
    "DEFAULT_HEATER_EFFICIENCY",
    "DEFAULT_MINIMUM_PINCH_K",
    "DEFAULT_RECOVERY_MODE",
    "FixedRouting",
    "HeatNetworkConfig",
    "HeatNetworkResult",
    "HeatNetworkRouting",
    "HeatSinkConfig",
    "HeatSourceConfig",
    "MatrixRouting",
    "PinchResult",
    "RecoveryMode",
    "SinkState",
    "SourceState",
    "StreamSegment",
    "binding_label",
    "cold_segment_for_sink",
    "composite_curve_kw",
    "compute_pinch_target",
    "dew_point_of_source_celsius",
    "evaluate_cascade",
    "evaluate_pinch_target",
    "hot_segments_for_source",
    "resolve_cascade_order",
]

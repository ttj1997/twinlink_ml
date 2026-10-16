"""Time-slice pinch: upper bound for the benefit of a heat storage (storage test 1).

See ``docs/19_storage_time_slice.md`` and the pre-registration in the docstring of
``scripts/gate_storage_timeslice.py``.
"""

from .config import (
    BAND_CAPACITY_MAX_KWH,
    CAPACITIES_KWH,
    LEVELS_CELSIUS,
    PLACEHOLDER_BANDS,
    Placeholders,
    Scenario,
    is_realistic,
)
from .evaluate import CapacityRow, ScenarioResult, evaluate_scenario
from .plant import MeasuredDay, load_measured_day
from .storage import RuleResult, StorageResult, simulate_rule, solve_ideal_storage
from .streams import StreamBank, StreamFactory, site_streams
from .table import (
    Cascade,
    LevelGroups,
    cascade_profile,
    direct_hot_utility_kw,
    level_groups,
    storage_stream_temperatures,
)

__all__ = [
    "BAND_CAPACITY_MAX_KWH",
    "CAPACITIES_KWH",
    "LEVELS_CELSIUS",
    "PLACEHOLDER_BANDS",
    "CapacityRow",
    "Cascade",
    "LevelGroups",
    "MeasuredDay",
    "Placeholders",
    "RuleResult",
    "Scenario",
    "ScenarioResult",
    "StorageResult",
    "StreamBank",
    "StreamFactory",
    "cascade_profile",
    "direct_hot_utility_kw",
    "evaluate_scenario",
    "is_realistic",
    "level_groups",
    "load_measured_day",
    "simulate_rule",
    "site_streams",
    "solve_ideal_storage",
    "storage_stream_temperatures",
]

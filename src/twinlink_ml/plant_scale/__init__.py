"""Plant-near model of plant A (docs/NN_plant_scale.md).

The unchanged grey-box model parametrised at the scale of plant A: stationary segments of the two
measured days, balance numbers for the gap between bed and exhaust temperature, calibration of at
most two temperature-identifiable parameters (heat-loss UA and an optional bed-to-exhaust term,
switched off by default), and with the result the observability of docs/33 and the steady
fixed-against-tracked headroom of Befund 105 at plant scale. Existing modules are only read.
"""

from .balance import gap_balance, overall_balance
from .config import FREEBOARD_MODES, PlantScaleConfig, load_plant_scale_config
from .data import (
    DaySeries,
    Segment,
    block_statistics,
    day_series_from_columns,
    load_day_series,
    rolling_std,
    segment_statistics,
    stationary_segments,
    window_ratios,
)
from .headroom import Disturbance, LabFactory, PlantFactory, SearchBox, SteadySearch, box_points
from .model import (
    CalibrationFit,
    Freeboard,
    PlantParameters,
    PlantPoint,
    PlantSteady,
    calibrate,
    exhaust_after_freeboard,
    plant_config,
    predict,
    steady_plant,
)

__all__ = [
    "FREEBOARD_MODES",
    "CalibrationFit",
    "DaySeries",
    "Disturbance",
    "Freeboard",
    "LabFactory",
    "PlantFactory",
    "PlantParameters",
    "PlantPoint",
    "PlantScaleConfig",
    "PlantSteady",
    "SearchBox",
    "Segment",
    "SteadySearch",
    "block_statistics",
    "box_points",
    "calibrate",
    "day_series_from_columns",
    "exhaust_after_freeboard",
    "gap_balance",
    "load_day_series",
    "load_plant_scale_config",
    "overall_balance",
    "plant_config",
    "predict",
    "rolling_std",
    "segment_statistics",
    "stationary_segments",
    "steady_plant",
    "window_ratios",
]

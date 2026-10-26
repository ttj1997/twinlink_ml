"""Energetic value of an inline moisture measurement (docs/28_sensor_value.md).

Quasi-stationary RTO (R1 of Befund 105) with a zeroth-order bias correction from four feedback
variants (none, ideal measurement, realistic inline sensor, GRU soft sensor), run on the grey-box
truth with drawn model parameters. Existing models are only read.
"""

from .config import SensorValueConfig, load_sensor_value_config
from .draws import ParameterDraw, parameter_draws, scenario_subset, truth_config
from .feedback import (
    BiasFilter,
    Feedback,
    IdealMeasurement,
    InlineSensor,
    NoFeedback,
    SoftSensorFeedback,
)
from .loop import BiasCorrectedController, PairRecord, extend_scenario, run_pairs, warmed_up_states
from .search import PairBisection, criterion_margin, next_candidate, required_count
from .softsensor import SOFT_SENSOR_EXOGENOUS, SoftSensor

__all__ = [
    "SOFT_SENSOR_EXOGENOUS",
    "BiasCorrectedController",
    "BiasFilter",
    "Feedback",
    "IdealMeasurement",
    "InlineSensor",
    "NoFeedback",
    "PairBisection",
    "PairRecord",
    "ParameterDraw",
    "SensorValueConfig",
    "SoftSensor",
    "SoftSensorFeedback",
    "criterion_margin",
    "extend_scenario",
    "load_sensor_value_config",
    "next_candidate",
    "parameter_draws",
    "required_count",
    "run_pairs",
    "scenario_subset",
    "truth_config",
    "warmed_up_states",
]

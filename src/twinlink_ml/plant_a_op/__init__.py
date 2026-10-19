r"""Operating point of plant A: air flow and supply-air temperature against CO2 (steam and fans).

Stationary balance model of the fluidised-bed dryer of plant A at plant scale, calibrated on the
two measured days, with a fan model, a waste-heat model of the preheater (storage loop of
Befund 91) and proxy constraints for the unmeasured product moisture. See
``docs/23_plant_a_operating_point.md``. Every unmeasured input is a placeholder
(:data:`~twinlink_ml.plant_a_op.config.PLACEHOLDERS`); results are upper bounds of the lever,
not operating recommendations.
"""

from .config import (
    PLACEHOLDERS,
    DryerPlaceholders,
    EnergyPlaceholders,
    PlaceholderSpec,
    SearchSpace,
)
from .data import DayState, MeasuredDay, day_state, load_measured_day
from .dryer import (
    DryerModel,
    consistent_solid_fraction,
    loss_conductance_kw_per_k,
    minimum_admissible_solid_fraction,
)
from .fans import FanLaw, PowerLawFan, fan_law, two_point_exponent, within_day_exponent
from .optimize import Constraints, Optimum, constraints_for, optimise
from .plant import OperatingPoints, PlantAModel
from .waste_heat import DutyHeld, FixedWaterInlet, fit_preheater_conductance

__all__ = [
    "PLACEHOLDERS",
    "Constraints",
    "DayState",
    "DryerModel",
    "DryerPlaceholders",
    "DutyHeld",
    "EnergyPlaceholders",
    "FanLaw",
    "FixedWaterInlet",
    "MeasuredDay",
    "OperatingPoints",
    "Optimum",
    "PlaceholderSpec",
    "PlantAModel",
    "PowerLawFan",
    "SearchSpace",
    "consistent_solid_fraction",
    "constraints_for",
    "day_state",
    "fan_law",
    "fit_preheater_conductance",
    "load_measured_day",
    "loss_conductance_kw_per_k",
    "minimum_admissible_solid_fraction",
    "optimise",
    "two_point_exponent",
    "within_day_exponent",
]

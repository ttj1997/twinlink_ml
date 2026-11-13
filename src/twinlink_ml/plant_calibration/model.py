r"""Plant-near model of Befund 117 with the candidate parameters of the calibration study.

The grey-box model is only PARAMETRISED (via :func:`twinlink_ml.plant_scale.model.steady_plant`,
unchanged): the bed heat-loss UA, the solid heat capacity and the two kinetics parameters enter the
process config; the bed-to-exhaust terms are applied after the steady state, outside the
grey-box code, as the switch of Befund 117:

.. math:: T_{exh} = T_{amb} + (T^{model}_{exh} - T_{amb})\,e^{-UA_{fb}/(\dot m_L c_h)} - \Delta T_p

with :math:`T^{model}_{exh} = T_{bed}` (the model's exhaust, outlet-duct loss 0),
:math:`c_h = c_{p,a} + Y_{out} c_{p,v}` (the ``heat_loss`` term of Befund 117) and the
bed-to-exhaust distance :math:`\Delta T_p` of the product ``p`` (plus a shared distance in the
descriptive variant). With both terms at 0 (the default, "switch off") the exhaust is the model's
own value, bit for bit (no arithmetic is applied; tested). Each term is applied only if its
parameter is non-zero; negative values are allowed for the finite differences around 0.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..plant_scale.model import PlantParameters, PlantPoint, steady_plant
from ..simulation import PlaceholderConfigWarning, ProcessConfig

__all__ = [
    "MATERIAL_PARAMETERS",
    "ModelContext",
    "Prediction",
    "evaluate",
    "evaluate_rows",
    "offset_for",
]

# candidate name -> field of MaterialConfig
MATERIAL_PARAMETERS: dict[str, str] = {
    "cp_solid_j_per_kg_k": "cp_solid_j_per_kg_k",
    "critical_moisture_kg_per_kg": "critical_moisture_kg_per_kg",
    "max_saturation_degree": "max_saturation_degree",
}
UA_BED = "ua_bed_w_per_k"
FREEBOARD_UA = "freeboard_ua_w_per_k"
SHARED_OFFSET = "offset_shared_k"


def offset_for(theta: Mapping[str, float], product: str) -> float:
    """Bed-to-exhaust distance of ``product``: its own plus the shared one (K)."""
    return float(theta.get(f"offset_{product}_k", 0.0)) + float(theta.get(SHARED_OFFSET, 0.0))


@dataclass(frozen=True)
class ModelContext:
    """Everything a steady evaluation needs besides the point and the parameters."""

    base: ProcessConfig
    solid_fraction: float
    steady_kwargs: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Prediction:
    """Steady model outputs at one point."""

    point: str
    bed_celsius: float
    exhaust_celsius: float
    gas_velocity_m_per_s: float
    moisture_kg_per_kg: float
    feasible: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "bed_celsius": self.bed_celsius,
            "exhaust_celsius": self.exhaust_celsius,
            "gas_velocity_m_per_s": self.gas_velocity_m_per_s,
            "moisture_kg_per_kg": self.moisture_kg_per_kg,
            "feasible": self.feasible,
        }


def evaluate(
    ctx: ModelContext,
    point: PlantPoint,
    product: str,
    theta: Mapping[str, float],
    *,
    velocity_guess: float | None = None,
) -> Prediction:
    """Steady bed and exhaust temperature at ``point`` for the parameters ``theta``.

    Parameters absent from ``theta`` keep the value of the process config (UA: the config's,
    terms: 0). Raises ``RuntimeError``/``ValueError`` of the steady solver.
    """
    ua = float(theta.get(UA_BED, ctx.base.geometry.heat_loss_ua_w_per_k))
    material = {
        field: float(theta[name]) for name, field in MATERIAL_PARAMETERS.items() if name in theta
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        steady = steady_plant(
            ctx.base,
            point,
            PlantParameters(ua, ctx.solid_fraction),
            velocity_guess=velocity_guess,
            material_values=material or None,
            **ctx.steady_kwargs,
        )
    t_exh = steady.exhaust_temperature_celsius
    fb = float(theta.get(FREEBOARD_UA, 0.0))
    if fb != 0.0:
        gas = ctx.base.gas
        c_h = gas.cp_dry_air_j_per_kg_k + (
            steady.outlet_humidity_balance_kg_per_kg * gas.cp_vapour_j_per_kg_k
        )
        m = steady.bed.dry_air_flow_kg_per_s
        t_amb = point.ambient_celsius
        t_exh = t_amb + (t_exh - t_amb) * math.exp(-fb / (m * c_h))
    offset = offset_for(theta, product)
    if offset != 0.0:
        t_exh = t_exh - offset
    return Prediction(
        point=point.label,
        bed_celsius=steady.bed_temperature_celsius,
        exhaust_celsius=t_exh,
        gas_velocity_m_per_s=steady.gas_velocity_m_per_s,
        moisture_kg_per_kg=steady.moisture_kg_per_kg,
        feasible=bool(steady.bed.feasible),
    )


def evaluate_rows(
    ctx: ModelContext,
    points: Sequence[tuple[PlantPoint, str]],
    theta: Mapping[str, float],
    guesses: dict[str, float] | None = None,
) -> np.ndarray:
    """Model outputs ``[bed_1, exh_1, bed_2, exh_2, ...]``; NaN pair where the solver fails.

    ``guesses`` (point label -> gas velocity) is used as start and updated.
    """
    out: list[float] = []
    for point, product in points:
        guess = None if guesses is None else guesses.get(point.label)
        try:
            pred = evaluate(ctx, point, product, theta, velocity_guess=guess)
        except (RuntimeError, ValueError):
            out.extend([math.nan, math.nan])
            continue
        if guesses is not None:
            guesses[point.label] = pred.gas_velocity_m_per_s
        out.extend([pred.bed_celsius, pred.exhaust_celsius])
    return np.array(out, dtype=np.float64)

"""Gas properties, saturation and release curves of an exhaust, shared by the units.

Lifted unchanged from ``scripts/report_heat_recovery_comparison.py`` (Befund 90) so that the
units of :mod:`twinlink_ml.flowsheet` and that script compute with the same code. The saturation
state and the latent heat are those of the laboratory model (:mod:`twinlink_ml.simulation.thermo`),
the partner's Antoine coefficients are not used here (``partner_v4.corrected`` keeps them for the
dryer-side dew point of the as-built reproduction).

Placeholders of the physical side (named, no source): the dew-point margin of the "no
condensation" limit (3 K, module default of :mod:`twinlink_ml.heatnetwork`), the cold end of the
condensing release curve ``T_kalt`` (15 degC, module default) and the number of profile nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from twinlink_ml.partner_v4 import corrected as c
from twinlink_ml.partner_v4.inputs import SiMaterial
from twinlink_ml.simulation.backend import NumpyBackend
from twinlink_ml.simulation.config import GasConfig
from twinlink_ml.simulation.thermo import (
    dew_point_celsius,
    latent_heat_j_per_kg,
    saturation_humidity_kg_per_kg,
)

__all__ = [
    "DEW_POINT_MARGIN_K",
    "HTM_BOILING_CELSIUS",
    "PROFILE_NODES",
    "REPO_PROPS",
    "T_KALT_CELSIUS",
    "GasProps",
    "Profile",
    "ProfileKind",
    "capacity",
    "condensate_kg_per_h",
    "dew_point",
    "exhaust_profile",
    "latent_heat",
    "partner_props",
    "saturation_humidity",
]

DEW_POINT_MARGIN_K = 3.0
"""Module default ``DEFAULT_DEW_POINT_APPROACH_K`` (placeholder)."""
T_KALT_CELSIUS = 15.0
"""Module default ``DEFAULT_COLD_TEMPERATURE_CELSIUS`` (placeholder)."""
PROFILE_NODES = 400
"""Module default ``DEFAULT_APPROACH_PROFILE_NODES``."""
HTM_BOILING_CELSIUS = 100.0
"""Boiling point of water at 1 atm, the limit of a liquid heat transfer medium without pressure."""

_BACKEND = NumpyBackend()
_GAS = GasConfig()

ProfileKind = Literal["linear", "sensible", "condensing"]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class GasProps:
    """Gas properties of a model, J/(kg K)."""

    name: str
    cp_air: float
    cp_vapour: float


REPO_PROPS = GasProps("repository", _GAS.cp_dry_air_j_per_kg_k, _GAS.cp_vapour_j_per_kg_k)
"""Laboratory model's properties (1.006, 1.86 kJ/(kg K)): the "physical" side of ``cp_gas``."""


def partner_props(material: SiMaterial) -> GasProps:
    """The partner's gas properties (``Inputs!I4``, ``I9``): the "partner" side of ``cp_gas``."""
    return GasProps("partner", material.cp_gas_j_per_kg_k, material.cp_vapour_j_per_kg_k)


def capacity(stream: c.Stream, props: GasProps) -> float:
    """``M (cp_air + Y cp_vapour)`` [W/K]."""
    return stream.dry_air_kg_per_s * (props.cp_air + stream.humidity_kg_per_kg * props.cp_vapour)


def dew_point(y: float) -> float:
    """Dew point [degC] of moist air of humidity ``y`` (laboratory model's correlation)."""
    return float(dew_point_celsius(_BACKEND, _GAS, _BACKEND.asarray(np.asarray(y))))


def saturation_humidity(t: FloatArray) -> FloatArray:
    """Saturation humidity ratio [kg/kg] at ``t`` [degC]."""
    return np.asarray(
        saturation_humidity_kg_per_kg(_BACKEND, _GAS, _BACKEND.asarray(np.asarray(t))),
        dtype=np.float64,
    )


def latent_heat(t: float) -> float:
    """Evaporation enthalpy [J/kg] at ``t`` [degC]."""
    return float(latent_heat_j_per_kg(_BACKEND, _BACKEND.asarray(np.asarray(t))))


@dataclass(frozen=True)
class Profile:
    """Heat released by the exhaust ``q_w`` [W] against its temperature ``t_c`` [degC]."""

    q_w: FloatArray
    t_c: FloatArray

    @property
    def total_w(self) -> float:
        return float(self.q_w[-1])

    def temperature_at(self, q_w: float) -> float:
        return float(np.interp(q_w, self.q_w, self.t_c))


def exhaust_profile(
    stream: c.Stream, props: GasProps, kind: ProfileKind, *, nodes: int = PROFILE_NODES
) -> Profile:
    """Release curve of an exhaust.

    ``linear``: sensible heat down to ``T_kalt`` (the dew point is not a limit: what the partner
    model assumes); ``sensible``: sensible heat down to dew point + margin (no condensation);
    ``condensing``: as :mod:`twinlink_ml.heatnetwork.approach`, condensation from the dew point on,
    latent heat at ``T_kalt``, down to ``T_kalt``.
    """
    m, y, t_out = stream.dry_air_kg_per_s, stream.humidity_kg_per_kg, stream.temperature_celsius
    cap = capacity(stream, props)
    dew = dew_point(y)
    floor = dew + DEW_POINT_MARGIN_K if kind == "sensible" else T_KALT_CELSIUS
    if t_out <= floor:
        return Profile(np.zeros(2), np.array([t_out, t_out]))
    if kind != "condensing":
        return Profile(np.array([0.0, cap * (t_out - floor)]), np.array([t_out, floor]))
    knee = float(np.clip(dew, floor, t_out))
    fractions = np.linspace(0.0, 1.0, nodes + 1)[1:]
    t_nodes = np.concatenate([[t_out, knee], knee + (floor - knee) * fractions])
    y_s = saturation_humidity(t_nodes)
    q_nodes = cap * (t_out - t_nodes) + m * np.maximum(y - y_s, 0.0) * latent_heat(T_KALT_CELSIUS)
    if q_nodes[0] > 0.0:
        # an exhaust at or above saturation by this correlation (e.g. saturated by IAPWS-IF97 in
        # the dryer's saturation limit) releases latent heat at its own temperature first; the
        # curve has to start at q = 0. Never the case for an unsaturated exhaust.
        q_nodes = np.concatenate([[0.0], q_nodes])
        t_nodes = np.concatenate([[t_out], t_nodes])
    return Profile(np.maximum.accumulate(np.maximum(q_nodes, 0.0)), t_nodes)


def condensate_kg_per_h(stream: c.Stream, t_exhaust_out: float) -> float:
    """Water that would leave the exhaust below its saturation humidity [kg/h]."""
    y_s = float(saturation_humidity(np.asarray(t_exhaust_out)))
    return stream.dry_air_kg_per_s * max(stream.humidity_kg_per_kg - y_s, 0.0) * 3600.0

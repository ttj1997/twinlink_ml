r"""Truth with a structural deviation: a fixed share of the spray water evaporates above the bed.

Candidate C2 of Befund 117 (``plant_scale.model.Freeboard``, mode ``evaporation``), with the same
emulation by the inputs and the same exhaust balance, here for the lab-line process of the Befund 94
pipeline. A share ``f`` of the spray water ``W = s (1 - x_s)`` evaporates in the freeboard instead
of in the bed:

* **Bed.** The bed receives the spray rate ``s - f W`` with the solid fraction
  ``x_s s / (s - f W)``; its solid feed ``x_s s`` is unchanged. Nothing else of the grey-box model
  changes (:func:`bed_feed`).
* **Exhaust.** The gas leaving the bed evaporates ``f W`` (liquid at the suspension temperature,
  vapour at the exhaust temperature):
  ``m_L c_h (T_bed - T_exh) = f W (r_0 + c_v T_exh - c_w T_feed)`` with
  ``c_h = c_{p,a} + Y_out c_{p,v}`` (:func:`exhaust_temperature_celsius`, the expression of
  ``plant_scale.model.exhaust_after_freeboard``). The exhaust is not a target of the study; the
  bed-to-exhaust gap only fixes the size of the deviation (docs/41_hybrid_structural.md).

The grey-box code is not changed. Two forms of the same truth:

* :class:`FreeboardEvaporationGranulator` -- the simulator whose right-hand side sees the bed
  feed; commands, disturbances, clipping, projection and the sensor model stay those of the
  commanded inputs. Used for every data pool. With ``share = 0`` the right-hand side is the
  parent's call (bit-identical, tested).
* :func:`steady_config` -- the process config whose NOMINAL spray rate and solid fraction are the
  bed feed of the nominal commands. The steady solvers of the study hold spray rate and solids
  at their nominal, so the steady state of this config is the steady state of the simulator above
  (tested); with ``share = 0`` the config is returned as it is.

Known side effect of the emulation (as in Befund 117): pump and atomiser power in the energy target
are computed with the bed's spray rate ``s - f W``; their share of the total power is reported by
``scripts/probe_hybrid_structural_share.py``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from twinlink_ml.simulation import GreyBoxFluidBedGranulator, ProcessConfig
from twinlink_ml.simulation.backend import ArrayBackend, NumpyBackend
from twinlink_ml.simulation.thermo import latent_heat_j_per_kg

__all__ = [
    "FreeboardEvaporationGranulator",
    "bed_feed",
    "exhaust_temperature_celsius",
    "steady_config",
]

IDX_SPRAY_RATE = 2  # action column: spray rate kg/h
IDX_SOLID_FRACTION = 2  # disturbance column: solid fraction of the suspension
_R0_J_PER_KG = float(latent_heat_j_per_kg(NumpyBackend(), 0.0))


def _check_share(share: float) -> float:
    share = float(share)
    if not 0.0 <= share < 1.0:
        raise ValueError("the freeboard share must lie in [0, 1)")
    return share


def bed_feed(spray: Any, solid_fraction: Any, share: float) -> tuple[Any, Any, Any]:
    """Spray rate and solid fraction seen by the bed, and the freeboard water.

    Works on floats and arrays; the unit of the freeboard water is that of ``spray``.
    """
    freeboard_water = share * spray * (1.0 - solid_fraction)
    bed_spray = spray - freeboard_water
    return bed_spray, solid_fraction * spray / bed_spray, freeboard_water


class FreeboardEvaporationGranulator(GreyBoxFluidBedGranulator):
    """The grey-box simulator with a share of the spray water evaporating above the bed."""

    def __init__(
        self, config: ProcessConfig, share: float, backend: ArrayBackend | None = None
    ) -> None:
        super().__init__(config, backend)
        self._share = _check_share(share)

    @property
    def share(self) -> float:
        return self._share

    def _rhs(self, state: Any, action: Any, disturbance: Any) -> tuple[Any, dict[str, Any]]:
        if self._share == 0.0:
            return super()._rhs(state, action, disturbance)
        backend = self._backend
        bed_spray, bed_solid_fraction, _ = bed_feed(
            action[..., IDX_SPRAY_RATE], disturbance[..., IDX_SOLID_FRACTION], self._share
        )
        n_action, n_disturbance = action.shape[-1], disturbance.shape[-1]
        bed_action = backend.stack(
            [bed_spray if i == IDX_SPRAY_RATE else action[..., i] for i in range(n_action)],
            axis=-1,
        )
        bed_disturbance = backend.stack(
            [
                bed_solid_fraction if i == IDX_SOLID_FRACTION else disturbance[..., i]
                for i in range(n_disturbance)
            ],
            axis=-1,
        )
        return super()._rhs(state, bed_action, bed_disturbance)


def _contain(bound: Any, nominal: float) -> Any:
    """``bound`` with the new nominal, its limits widened to contain it (sampling limits only)."""
    return dataclasses.replace(
        bound,
        nominal=nominal,
        minimum=min(bound.minimum, nominal),
        maximum=max(bound.maximum, nominal),
    )


def steady_config(config: ProcessConfig, share: float) -> ProcessConfig:
    """``config`` with the nominal spray rate and solid fraction replaced by the bed feed."""
    share = _check_share(share)
    if share == 0.0:
        return config
    a, d = config.actions, config.disturbances
    bed_spray, bed_solid_fraction, _ = bed_feed(
        a.spray_rate_kg_per_h.nominal, d.suspension_solid_fraction.nominal, share
    )
    return dataclasses.replace(
        config,
        actions=dataclasses.replace(
            a, spray_rate_kg_per_h=_contain(a.spray_rate_kg_per_h, float(bed_spray))
        ),
        disturbances=dataclasses.replace(
            d,
            suspension_solid_fraction=_contain(
                d.suspension_solid_fraction, float(bed_solid_fraction)
            ),
        ),
    )


def exhaust_temperature_celsius(
    config: ProcessConfig,
    *,
    bed_temperature_celsius: float,
    dry_air_flow_kg_per_s: float,
    outlet_humidity_kg_per_kg: float,
    freeboard_water_kg_per_s: float,
) -> float:
    """Exhaust temperature after the evaporation of the freeboard water (module docstring).

    ``outlet_humidity_kg_per_kg``: humidity of the gas leaving the bed (water balance of the bed).
    """
    gas = config.gas
    m, w = dry_air_flow_kg_per_s, freeboard_water_kg_per_s
    c_h = gas.cp_dry_air_j_per_kg_k + outlet_humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k
    c_w = config.material.cp_water_liquid_j_per_kg_k
    feed = config.operation.suspension_temperature_celsius
    return float(
        (m * c_h * bed_temperature_celsius - w * (_R0_J_PER_KG - c_w * feed))
        / (m * c_h + w * gas.cp_vapour_j_per_kg_k)
    )

r"""X3: recoverable power ``Q_rec`` from PREDICTED exhaust air quantities (Befund 50).

.. math:: \dot Q_{rec} = \dot m_L \left(T_{out} - (T_{dp}(Y_{out}) + 3\,\mathrm K)\right)
    \left(c_{p,L} + Y_{out}\, c_{p,D}\right)

``T_out``/``Y_out`` are ``outlet_temperature_celsius``/``Y_gas_out_kg_per_kg`` from
:func:`~twinlink_ml.evaluation.exhaust_reconstruction.reconstruct_exhaust_arrays` -- **the same**
algebraic path as X1/X2 (``simulation.exhaust.reconstruct_exhaust_state``), here only with
one further, nonlinear quantity on top. No new reconstruction code: this module calls
exclusively ``reconstruct_exhaust_arrays`` and applies the Q_rec formula element-wise to
its ``reconstructed``/``reference`` arrays -- predicted Q_rec against true Q_rec, exactly
the same prediction/truth comparison as for ``T_gas_out_celsius`` itself.

3 K margin to the dew point: prevents condensation in the heat exchanger (not modelled
here) -- ``Q_rec`` is the power sensibly recoverable down to this limit, not the
theoretical maximum (which would also include the condensation enthalpy below the dew
point).

Mass flow caveat
----------------------
``DISTRICT_MASS_FLOW_KG_PER_S`` (8 kg/s) is the mass flow of the heat network model, NOT
the simulated lab mass flow (``dry_air_mass_flow_kg_per_s`` in
``data.trajectory.diagnostics``, two orders of magnitude smaller). The kW numbers from this
module are a scaling calculation -- they answer "how large would the error be if a plant
ran with this mass flow, at exactly this temperature/humidity prediction error", not a
plant statement about the actual lab fluidized bed.
``REFERENCE_LINE_RECOVERABLE_HEAT_KW`` (446.6 kW) is the recoverable heat of the
fluidized-bed line at the site, as a reference quantity for putting the absolute error
into context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..simulation.backend import FloatArray, NumpyBackend
from ..simulation.thermo import dew_point_celsius
from .evaluator import per_run_nrmse, per_run_rmse_median, pooled_nrmse
from .exhaust_reconstruction import EXHAUST_TARGET_NAMES, reconstruct_exhaust_arrays

if TYPE_CHECKING:
    from ..data.windows import SequenceDataset
    from ..models.base import SequenceModel
    from ..simulation.backend import Array, ArrayBackend
    from ..simulation.config import GasConfig
    from .experiments import GeneratedData

__all__ = [
    "DEW_POINT_MARGIN_K",
    "DISTRICT_MASS_FLOW_KG_PER_S",
    "RECOVERABLE_POWER_NAME",
    "REFERENCE_LINE_RECOVERABLE_HEAT_KW",
    "RecoverablePowerResult",
    "compute_recoverable_power_kw",
    "evaluate_recoverable_power",
]

DISTRICT_MASS_FLOW_KG_PER_S = 8.0
"""Massenstrom des Wärmeverbundmodells -- siehe Moduldocstring, Massenstrom-Vorbehalt."""

DEW_POINT_MARGIN_K = 3.0
"""Sicherheitsmarge zum Taupunkt gegen Kondensation im (nicht modellierten) Übertrager."""

REFERENCE_LINE_RECOVERABLE_HEAT_KW = 446.6
"""Rückgewinnbare Wärme der Wirbelschichtlinie am Standort -- Bezugsgröße für den
absoluten Q_rec-Fehler, keine Grenze/Kappung."""

RECOVERABLE_POWER_NAME = ("recoverable_power_kw",)
"""Einelementiges Namens-Tupel -- ``pooled_nrmse``/``per_run_nrmse``/``per_run_rmse_median``
erwarten eine Namensliste je Spalte, hier genau eine abgeleitete Größe."""


def compute_recoverable_power_kw(
    backend: ArrayBackend,
    gas: GasConfig,
    outlet_temperature_celsius: Array,
    exhaust_humidity_kg_per_kg: Array,
    *,
    mass_flow_kg_per_s: float = DISTRICT_MASS_FLOW_KG_PER_S,
    dew_point_margin_k: float = DEW_POINT_MARGIN_K,
) -> Array:
    r"""Element-wise :math:`\dot Q_{rec}` from exhaust air temperature and humidity, in kW.

    Pure, backend-agnostic function (like ``simulation.thermo``) -- no model call, no
    windowing logic. Accepts ``outlet_temperature_celsius``/``exhaust_humidity_kg_per_kg``
    of any shape (element-wise operations), typically
    ``(n_windows, horizon)`` from :func:`~.exhaust_reconstruction.reconstruct_exhaust_arrays`.
    """
    dew_point = dew_point_celsius(backend, gas, exhaust_humidity_kg_per_kg)
    specific_heat_humid_air = (
        gas.cp_dry_air_j_per_kg_k + exhaust_humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k
    )
    delta_t = outlet_temperature_celsius - (dew_point + dew_point_margin_k)
    return mass_flow_kg_per_s * delta_t * specific_heat_humid_air / 1000.0


@dataclass(frozen=True, slots=True)
class RecoverablePowerResult:
    """NRMSE/absolute error of ``Q_rec``, prediction against truth -- see module docstring."""

    horizons: tuple[int, ...]
    nrmse_by_horizon: dict[int, float]
    nrmse_run_spread: dict[int, tuple[float, float, float]]
    rmse_run_median_kw: dict[int, float]
    n_windows: int
    n_runs: int
    mass_flow_kg_per_s: float
    dew_point_margin_k: float
    reference_line_kw: float = REFERENCE_LINE_RECOVERABLE_HEAT_KW

    def format(self) -> str:
        """Table for run logs, in the same style as ``ExhaustReconstructionResult.format``."""
        columns = "".join(f"{f'h={h}':>10}" for h in self.horizons)
        primary = "".join(f"{self.nrmse_run_spread[h][0]:>10.4f}" for h in self.horizons)
        physical = "".join(f"{self.rmse_run_median_kw[h]:>10.2f}" for h in self.horizons)
        relative = "".join(
            f"{100.0 * self.rmse_run_median_kw[h] / self.reference_line_kw:>9.2f}%"
            for h in self.horizons
        )
        return "\n".join(
            [
                f"Q_rec ({self.n_runs} Läufe, {self.n_windows} Fenster, "
                f"m_L={self.mass_flow_kg_per_s} kg/s, Marge={self.dew_point_margin_k} K) "
                f"-- Skalierungsrechnung, siehe Moduldocstring",
                f"{'':<28}{columns}",
                f"{'NRMSE':<28}{primary}",
                f"{'RMSE physikalisch [kW]':<28}{physical}",
                f"{'  ggü. Linie (446.6 kW)':<28}{relative}",
            ]
        )

    def as_flat_metrics(self) -> dict[str, float]:
        """Flat key-value pairs for tracking.

        ``q_rec_`` prefix -- can be merged without collisions alongside
        ``ExhaustReconstructionResult.as_flat_metrics()`` in the same ``RunRecord``.
        """
        flat: dict[str, float] = {
            "q_rec_n_windows": float(self.n_windows),
            "q_rec_n_test_runs": float(self.n_runs),
            "q_rec_mass_flow_kg_per_s": self.mass_flow_kg_per_s,
            "q_rec_dew_point_margin_k": self.dew_point_margin_k,
            "q_rec_reference_line_kw": self.reference_line_kw,
        }
        for horizon, value in self.nrmse_by_horizon.items():
            flat[f"q_rec_nrmse/h{horizon}"] = value
        for horizon, value in self.rmse_run_median_kw.items():
            flat[f"q_rec_rmse_run_median_kw/h{horizon}"] = value
        for horizon, (median, minimum, maximum) in self.nrmse_run_spread.items():
            flat[f"q_rec_nrmse_run_median/h{horizon}"] = median
            flat[f"q_rec_nrmse_run_min/h{horizon}"] = minimum
            flat[f"q_rec_nrmse_run_max/h{horizon}"] = maximum
        return flat


def evaluate_recoverable_power(
    model: SequenceModel,
    data: GeneratedData,
    test: SequenceDataset,
    *,
    horizons: tuple[int, ...] | None = None,
    batch_size: int = 4096,
    mass_flow_kg_per_s: float = DISTRICT_MASS_FLOW_KG_PER_S,
    dew_point_margin_k: float = DEW_POINT_MARGIN_K,
) -> RecoverablePowerResult:
    """``Q_rec`` from PREDICTED channels against ``Q_rec`` from truth channels.

    Builds on :func:`~.exhaust_reconstruction.reconstruct_exhaust_arrays` (the same
    prerequisites on ``test.target_names``, the same leak/mismatch protection) and applies
    :func:`compute_recoverable_power_kw` element-wise to the
    ``outlet_temperature_celsius``/``Y_gas_out_kg_per_kg`` columns already reconstructed
    there -- no second reconstruction pass.
    """
    reconstructed_array, reference_array, run_array, resolved_horizons = reconstruct_exhaust_arrays(
        model, data, test, horizons=horizons, batch_size=batch_size
    )

    outlet_index = EXHAUST_TARGET_NAMES.index("outlet_temperature_celsius")
    humidity_index = EXHAUST_TARGET_NAMES.index("Y_gas_out_kg_per_kg")

    backend = NumpyBackend()
    gas = data.process_config.gas

    def _q_rec(source: FloatArray) -> FloatArray:
        value = compute_recoverable_power_kw(
            backend,
            gas,
            source[..., outlet_index],
            source[..., humidity_index],
            mass_flow_kg_per_s=mass_flow_kg_per_s,
            dew_point_margin_k=dew_point_margin_k,
        )
        return np.asarray(value)[..., np.newaxis]

    q_rec_reconstructed = _q_rec(reconstructed_array)
    q_rec_reference = _q_rec(reference_array)

    by_horizon = pooled_nrmse(
        q_rec_reconstructed, q_rec_reference, resolved_horizons, RECOVERABLE_POWER_NAME
    )
    spread = per_run_nrmse(
        q_rec_reconstructed, q_rec_reference, run_array, resolved_horizons, RECOVERABLE_POWER_NAME
    )
    rmse_physical = per_run_rmse_median(
        q_rec_reconstructed, q_rec_reference, run_array, resolved_horizons, RECOVERABLE_POWER_NAME
    )
    name = RECOVERABLE_POWER_NAME[0]

    return RecoverablePowerResult(
        horizons=resolved_horizons,
        nrmse_by_horizon=by_horizon[name],
        nrmse_run_spread=spread[name],
        rmse_run_median_kw=rmse_physical[name],
        n_windows=int(reference_array.shape[0]),
        n_runs=int(np.unique(run_array).size),
        mass_flow_kg_per_s=mass_flow_kg_per_s,
        dew_point_margin_k=dew_point_margin_k,
    )

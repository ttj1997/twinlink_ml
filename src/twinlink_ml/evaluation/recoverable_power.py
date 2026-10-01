r"""X3: rückgewinnbare Leistung ``Q_rec`` aus VORHERGESAGTEN Abluft-Größen (Befund 50).

.. math:: \dot Q_{rec} = \dot m_L \left(T_{out} - (T_{dp}(Y_{out}) + 3\,\mathrm K)\right)
    \left(c_{p,L} + Y_{out}\, c_{p,D}\right)

``T_out``/``Y_out`` sind ``outlet_temperature_celsius``/``Y_gas_out_kg_per_kg`` aus
:func:`~twinlink_ml.evaluation.exhaust_reconstruction.reconstruct_exhaust_arrays` -- **derselbe**
algebraische Pfad wie X1/X2 (``simulation.exhaust.reconstruct_exhaust_state``), hier nur mit
einer weiteren, nichtlinearen Größe obendrauf. Kein neuer Rückrechnungscode: dieses Modul
ruft ausschließlich ``reconstruct_exhaust_arrays`` und wendet die Q_rec-Formel elementweise
auf dessen ``reconstructed``/``reference``-Arrays an -- Vorhersage-Q_rec gegen Wahrheits-Q_rec,
exakt dieselbe Vorhersage/Wahrheit-Gegenüberstellung wie bei ``T_gas_out_celsius`` selbst.

3-K-Marge zum Taupunkt: verhindert Kondensation im (hier nicht modellierten) Wärmeübertrager
-- ``Q_rec`` ist die bis zu dieser Grenze sensibel rückgewinnbare Leistung, nicht die
theoretisch maximale (die würde auch die Kondensationsenthalpie unterhalb des Taupunkts
einschließen).

Massenstrom-Vorbehalt
----------------------
``DISTRICT_MASS_FLOW_KG_PER_S`` (8 kg/s) ist der Massenstrom des Wärmeverbundmodells, NICHT
der simulierte Labormassenstrom (``dry_air_mass_flow_kg_per_s`` in
``data.trajectory.diagnostics``, zwei Größenordnungen kleiner). Die kW-Zahlen aus diesem
Modul sind eine Skalierungsrechnung -- sie beantworten "wie groß wäre der Fehler, wenn eine
Anlage mit diesem Massenstrom liefe, bei genau diesem Temperatur-/Feuchte-Vorhersagefehler",
keine Anlagenaussage über die tatsächliche Laborwirbelschicht.
``REFERENCE_LINE_RECOVERABLE_HEAT_KW`` (446.6 kW) ist die rückgewinnbare Wärme der
Wirbelschichtlinie am Standort, als Bezugsgröße für die Einordnung des absoluten Fehlers.
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
    r"""Elementweise :math:`\dot Q_{rec}` aus Abluft-Temperatur und -Feuchte, in kW.

    Reine, backend-agnostische Funktion (wie ``simulation.thermo``) -- kein Modellaufruf,
    keine Windowing-Logik. Nimmt ``outlet_temperature_celsius``/``exhaust_humidity_kg_per_kg``
    beliebiger Form entgegen (elementweise Operationen), typischerweise
    ``(n_windows, horizon)`` aus :func:`~.exhaust_reconstruction.reconstruct_exhaust_arrays`.
    """
    dew_point = dew_point_celsius(backend, gas, exhaust_humidity_kg_per_kg)
    specific_heat_humid_air = (
        gas.cp_dry_air_j_per_kg_k + exhaust_humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k
    )
    delta_t = outlet_temperature_celsius - (dew_point + dew_point_margin_k)
    return mass_flow_kg_per_s * delta_t * specific_heat_humid_air / 1000.0


@dataclass(frozen=True, slots=True)
class RecoverablePowerResult:
    """NRMSE/absoluter Fehler von ``Q_rec``, Vorhersage gegen Wahrheit -- siehe Moduldocstring."""

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
        """Tabelle für Run-Logs, im selben Stil wie ``ExhaustReconstructionResult.format``."""
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
        """Flache Schlüssel-Wert-Paare für das Tracking.

        ``q_rec_``-Präfix -- kollisionsfrei neben
        ``ExhaustReconstructionResult.as_flat_metrics()`` im selben ``RunRecord`` mergebar.
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
    """``Q_rec`` aus VORHERGESAGTEN Kanälen gegen ``Q_rec`` aus Wahrheits-Kanälen.

    Baut auf :func:`~.exhaust_reconstruction.reconstruct_exhaust_arrays` auf (dieselben
    Voraussetzungen an ``test.target_names``, derselbe Leck-/Fehlpaarungs-Schutz) und wendet
    :func:`compute_recoverable_power_kw` elementweise auf die dort bereits rekonstruierten
    ``outlet_temperature_celsius``/``Y_gas_out_kg_per_kg``-Spalten an -- kein zweiter
    Rückrechnungs-Durchlauf.
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

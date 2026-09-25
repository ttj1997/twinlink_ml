r"""End-zu-End-Güte der Abluft-Rückrechnung aus vorhergesagter T_bed/Feuchte (Befund 41g).

Befund 41 hat gezeigt, dass ``simulation.exhaust.reconstruct_exhaust_state()`` bitgleich
gegen ``solve_steady_state()`` übereinstimmt, wenn man sie mit den **wahren, simulierten**
Werten von ``bed_temperature_celsius``/``product_moisture_kg_per_kg`` füttert. Das ist eine
Verdrahtungsprüfung, keine Aussage zur tatsächlichen Nutzlast: die Funktion soll mit
**vorhergesagten** Werten eines Sequenzmodells laufen. Dieses Modul schließt genau diese
Lücke, ohne ``exhaust.py``/``evaluator.py`` selbst zu ändern -- es ruft beide unverändert
auf und arbeitet auf der rohen, ungefensterten ``Trajectory`` (``GeneratedData.trajectory``),
weil die für die Rückrechnung nötigen Diagnosegrößen (``dry_air_mass_flow_kg_per_s``,
``inlet_humidity_kg_per_kg``) beim Fenstern in ``data.windows`` verworfen werden.

Massenstrom-Leck (aufgedeckt vor der ersten Kostenprobe, nicht nachträglich gefunden)
--------------------------------------------------------------------------------------
``dry_air_mass_flow_kg_per_s`` ist entgegen dem ersten Anschein **nicht** rein aus
Stellgröße (Gasgeschwindigkeit) und Störgröße ableitbar: ``greybox._rhs`` wertet die
Luftdichte bei ``bed_temperature`` aus, nicht bei Einlass-/Umgebungstemperatur
(``gas_density_kg_per_m3(gas, bed_temperature, ...)``, ideal durchmischtes Bett). Bei
einem Mehrschritt-Rollout (h > 1) ist die künftige, wahre Betttemperatur im Deployment
nicht bekannt -- genau deshalb wird sie vorhergesagt. Den wahren, aus der wahren
zukünftigen T_bed abgeleiteten Massenstrom in die Rekonstruktions-Seite zu speisen, wäre
ein echtes Leck und würde die End-zu-End-Metrik optimistischer machen, als ein reales
System leisten könnte. ``inlet_absolute_humidity_kg_per_kg`` (Y_in) ist davon nicht
betroffen -- sie hängt nur von Umgebungstemperatur/-feuchte ab (Störgrößen), nie von T_bed.

Die Korrektur (``rescale_dry_air_mass_flow_kg_per_s``) skaliert den wahren Massenstrom mit
dem Dichteverhältnis bei vorhergesagter vs. wahrer Betttemperatur -- dieselbe Formel wie in
``greybox._rhs``, hier nur mit vorhergesagter statt wahrer Temperatur ausgewertet, und ohne
Umweg über Gasgeschwindigkeit/Querschnittsfläche (die kürzen sich im Verhältnis heraus).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.splits import SplitName
from ..data.windows import SequenceDataset
from ..models.base import SequenceModel
from ..simulation.backend import Array, ArrayBackend, FloatArray, NumpyBackend
from ..simulation.config import GasConfig
from ..simulation.exhaust import reconstruct_exhaust_state
from ..simulation.sorption import isotherm_from_material
from ..simulation.thermo import gas_density_kg_per_m3
from .evaluator import per_run_nrmse, per_run_rmse_median, pooled_nrmse
from .experiments import GeneratedData

__all__ = [
    "EXHAUST_TARGET_NAMES",
    "ExhaustReconstructionResult",
    "evaluate_exhaust_reconstruction",
    "reconstruct_exhaust_arrays",
    "rescale_dry_air_mass_flow_kg_per_s",
]

EXHAUST_TARGET_NAMES: tuple[str, str, str] = (
    "T_gas_out_celsius",
    "Y_gas_out_kg_per_kg",
    "outlet_temperature_celsius",
)
"""``outlet_temperature_celsius`` seit ``ExhaustConfig``/X1 ergaenzt: die um
``exhaust.outlet_duct_heat_loss_k`` reduzierte Bett-Austrittstemperatur -- die eigentliche
Verbundschnittstellengroesse, im Unterschied zu ``T_gas_out_celsius`` (Gastemperatur direkt
am Bett-Austritt, vor jedem Leitungsverlust). Haengt NUR von der Betttemperatur-Vorhersage
und ``exhaust.outlet_duct_heat_loss_k`` ab (siehe ``simulation.exhaust.
reconstruct_exhaust_state``) -- bei ``outlet_duct_heat_loss_k=0.0`` (aktueller Platzhalter-
Default) numerisch bitgleich zu ``T_gas_out_celsius``."""


def rescale_dry_air_mass_flow_kg_per_s(
    backend: ArrayBackend,
    gas: GasConfig,
    *,
    true_dry_air_mass_flow_kg_per_s: Array,
    true_bed_temperature_celsius: Array,
    predicted_bed_temperature_celsius: Array,
    inlet_absolute_humidity_kg_per_kg: Array,
) -> Array:
    r"""Trockenluftmassenstrom auf vorhergesagte statt wahre Betttemperatur umrechnen.

    ``greybox._rhs`` berechnet ``dry_air_mass_flow_kg_per_s`` als
    :math:`\rho(T_{bed}, Y_{in}) \cdot u_{gas} \cdot A / (1 + Y_{in})`. Gasgeschwindigkeit
    und Querschnittsfläche sind hier unbekannt (nicht Teil der ``diagnostics``), kürzen
    sich aber im Verhältnis zweier Massenströme bei gleicher Anregung, aber
    unterschiedlicher Betttemperatur exakt heraus:

    .. math:: \dot m_{L,tr}(T_{pred}) = \dot m_{L,tr}(T_{true})
              \cdot \frac{\rho(T_{pred}, Y_{in})}{\rho(T_{true}, Y_{in})}

    Bei ``T_pred == T_true`` ist das Verhältnis exakt 1.0 (dasselbe Gleitkommaergebnis
    geteilt durch sich selbst), die Rückgabe also bitgleich zum wahren Massenstrom --
    keine neue Physik, nur eine Anwendung von ``thermo.gas_density_kg_per_m3`` (unverändert
    importiert) auf eine andere Temperatur als die, mit der ``greybox._rhs`` es tut.
    """
    density_at_true = gas_density_kg_per_m3(
        backend, gas, true_bed_temperature_celsius, inlet_absolute_humidity_kg_per_kg
    )
    density_at_predicted = gas_density_kg_per_m3(
        backend, gas, predicted_bed_temperature_celsius, inlet_absolute_humidity_kg_per_kg
    )
    return true_dry_air_mass_flow_kg_per_s * density_at_predicted / density_at_true


@dataclass(frozen=True, slots=True)
class ExhaustReconstructionResult:
    """NRMSE zwischen Abluft-Rückrechnung aus Vorhersage und aus Wahrheit.

    Beide Rückrechnungen laufen durch dieselbe, unveränderte
    ``exhaust.reconstruct_exhaust_state()`` -- Referenz mit wahrem
    ``bed_temperature_celsius``/``product_moisture_kg_per_kg`` (und wahrem
    ``dry_air_mass_flow_kg_per_s``), Rekonstruktion mit den Werten des Sequenzmodells (und
    dem darauf umgerechneten Massenstrom, siehe ``rescale_dry_air_mass_flow_kg_per_s``).
    ``inlet_absolute_humidity_kg_per_kg`` ist in beiden Aufrufen wahr (echte Störgröße,
    kein Leck).
    """

    horizons: tuple[int, ...]
    nrmse_by_horizon: dict[str, dict[int, float]]
    nrmse_run_spread: dict[str, dict[int, tuple[float, float, float]]]
    n_windows: int
    n_runs: int
    rmse_run_median_physical: dict[str, dict[int, float]] = field(default_factory=dict)
    """Unnormierter RMSE je Lauf, Median über Läufe, physikalische Einheiten (``°C``/``K`` für
    ``T_gas_out_celsius``/``outlet_temperature_celsius``, ``kg/kg`` für
    ``Y_gas_out_kg_per_kg``) -- Ergänzung zur NRMSE, Pendant zu
    ``EvaluationResult.rmse_run_median_physical`` (siehe dortiger Moduldocstring "Absoluter
    RMSE zusätzlich zur NRMSE"). Für ``outlet_temperature_celsius`` bei
    ``exhaust.outlet_duct_heat_loss_k=0.0`` bitgleich zum RMSE von ``T_gas_out_celsius`` --
    beide sind bei diesem Verlustwert dieselbe Zahlenreihe, nur unterschiedlich benannt."""

    def format(self) -> str:
        """Tabelle für Run-Logs, im selben Stil wie ``EvaluationResult.format``."""
        columns = "".join(f"{f'h={h}':>10}" for h in self.horizons)
        lines = [
            f"Abluft-Rückrechnung, Vorhersage vs. Wahrheit ({self.n_runs} Läufe, "
            f"{self.n_windows} Fenster)",
            f"{'Zielgröße':<28}{columns}",
        ]
        for target in EXHAUST_TARGET_NAMES:
            primary = "".join(
                f"{self.nrmse_run_spread[target][h][0]:>10.4f}" for h in self.horizons
            )
            lines.append(f"{target:<28}{primary}")
            pooled = "".join(f"{self.nrmse_by_horizon[target][h]:>10.4f}" for h in self.horizons)
            lines.append(f"{'  gepoolt (nicht primär)':<28}{pooled}")
            if target in self.rmse_run_median_physical:
                physical = "".join(
                    f"{self.rmse_run_median_physical[target][h]:>10.4f}" for h in self.horizons
                )
                lines.append(f"{'  RMSE physikalisch':<28}{physical}")
        return "\n".join(lines)

    def as_flat_metrics(self) -> dict[str, float]:
        """Flache Schlüssel-Wert-Paare für das Tracking.

        ``exhaust_``-Präfix -- kollisionsfrei neben
        ``EvaluationResult.as_flat_metrics()`` im selben ``RunRecord`` mergebar.
        """
        flat: dict[str, float] = {
            "exhaust_n_windows": float(self.n_windows),
            "exhaust_n_test_runs": float(self.n_runs),
        }
        for target, by_horizon in self.nrmse_by_horizon.items():
            for horizon, value in by_horizon.items():
                flat[f"exhaust_nrmse/{target}/h{horizon}"] = value
        for target, rmse_by_horizon in self.rmse_run_median_physical.items():
            for horizon, value in rmse_by_horizon.items():
                flat[f"exhaust_rmse_run_median_physical/{target}/h{horizon}"] = value
        for target, spread_by_horizon in self.nrmse_run_spread.items():
            for horizon, (median, minimum, maximum) in spread_by_horizon.items():
                flat[f"exhaust_nrmse_run_median/{target}/h{horizon}"] = median
                flat[f"exhaust_nrmse_run_min/{target}/h{horizon}"] = minimum
                flat[f"exhaust_nrmse_run_max/{target}/h{horizon}"] = maximum
        return flat


def reconstruct_exhaust_arrays(
    model: SequenceModel,
    data: GeneratedData,
    test: SequenceDataset,
    *,
    horizons: tuple[int, ...] | None = None,
    batch_size: int = 4096,
) -> tuple[FloatArray, FloatArray, FloatArray, tuple[int, ...]]:
    """Rohe ``(reconstructed, reference, runs, horizons)``-Arrays der Abluft-Rückrechnung.

    Eigene Funktion (nicht nur ein Codeblock in :func:`evaluate_exhaust_reconstruction`),
    damit eine weitere abgeleitete Größe -- etwa die rückgewinnbare Leistung ``Q_rec`` aus
    Vorhersage-``T_out``/``Y_out`` (X3, ``docs/04_stand_ap1.md``) -- dieselben
    (Vorhersage-, Wahrheits-)Arrays weiterverwenden kann, ohne den Batch-/Leck-Schutz-Code
    oben ein zweites Mal zu schreiben. ``reconstructed``/``reference`` haben Form
    ``(n_windows, horizon, len(EXHAUST_TARGET_NAMES))``, Spaltenreihenfolge wie
    :data:`EXHAUST_TARGET_NAMES`.

    Setzt voraus, dass ``bed_temperature_celsius`` und ``product_moisture_kg_per_kg`` in
    ``test.target_names`` stehen (explizite Erweiterung, siehe Befund 41f -- kein globaler
    Default). Greift für die beiden benötigten, in ``WindowBatch`` nicht vorhandenen
    Diagnosegrößen (``dry_air_mass_flow_kg_per_s``, ``inlet_humidity_kg_per_kg``) auf die
    ungefensterte ``data.trajectory.diagnostics`` zurück, indiziert über dieselbe
    Split-Reihenfolge, die ``TrajectoryWindowDataset.from_trajectory`` beim Aufbau
    verwendet (``data.split.as_mapping["test"]`` gegen ``data.trajectory.run_id``).
    """
    for name in ("bed_temperature_celsius", "product_moisture_kg_per_kg"):
        if name not in test.target_names:
            raise ValueError(
                f"reconstruct_exhaust_arrays braucht '{name}' in test.target_names -- "
                "explizit als target_names-Override angeben (Befund 41f), kein globaler Default"
            )

    backend = NumpyBackend()
    gas = data.process_config.gas
    material = data.process_config.material
    isotherm = isotherm_from_material(material)

    bed_index = test.target_names.index("bed_temperature_celsius")
    moisture_index = test.target_names.index("product_moisture_kg_per_kg")
    history_length = test.config.history_length
    full_horizon = test.config.horizon
    if horizons is None:
        horizons = tuple(
            sorted({1, max(1, full_horizon // 4), max(1, full_horizon // 2), full_horizon})
        )
    invalid = [h for h in horizons if not 1 <= h <= full_horizon]
    if invalid:
        raise ValueError(f"Horizonte {invalid} liegen außerhalb 1..{full_horizon}")

    row_by_run_id = {run_id: row for row, run_id in enumerate(data.trajectory.run_id)}
    test_run_ids = data.split.as_mapping[SplitName.TEST]
    rows_for_split_position = np.asarray(
        [row_by_run_id[run_id] for run_id in test_run_ids], dtype=np.intp
    )

    dry_flow_diagnostic = data.trajectory.diagnostics["dry_air_mass_flow_kg_per_s"]
    inlet_humidity_diagnostic = data.trajectory.diagnostics["inlet_humidity_kg_per_kg"]
    output_names = data.trajectory.metadata["output_names"]
    bed_output_index = output_names.index("bed_temperature_celsius")
    moisture_output_index = output_names.index("product_moisture_kg_per_kg")

    reconstructed: list[FloatArray] = []
    reference: list[FloatArray] = []
    runs: list[FloatArray] = []

    for batch in test.iter_batches(batch_size):
        prediction = test.target_scaler.inverse(model.predict(batch))
        if batch.future_targets_truth is None:
            raise ValueError(
                "reconstruct_exhaust_arrays braucht future_targets_truth -- "
                "Datensatz ohne rauschfreie Referenz erzeugt"
            )
        truth = test.target_scaler.inverse(batch.future_targets_truth)

        predicted_bed = prediction[..., bed_index]
        predicted_moisture = prediction[..., moisture_index]
        true_bed = truth[..., bed_index]
        true_moisture = truth[..., moisture_index]

        split_positions = batch.run_index.astype(np.intp)
        starts = batch.window_start.astype(np.intp)
        time_index = starts[:, None] + history_length + np.arange(full_horizon)[None, :]
        trajectory_rows = rows_for_split_position[split_positions][:, None]

        # Selbstpruefung: `test` MUSS aus `data.trajectory`/`data.split` gebaut worden sein --
        # sonst passen `run_index`/`window_start` nur zufaellig in der Form, zeigen aber auf
        # die falsche Trajektorie (z.B. Trainingspool statt Konstantlauf-Holdout-Pool
        # uebergeben). Ohne diese Pruefung waere das Ergebnis nicht falsch geformt, nur
        # STILL falsch -- reines Formmatching haette dieselbe Fensteranzahl/denselben
        # Horizont fuer jeden beliebigen zweiten Pool derselben DataSpec. Vergleich gegen
        # ``data.trajectory.outputs`` (dieselbe Groesse wie ``future_targets_truth``, nur
        # unabhaengig ueber Trajectory-Zeile/Zeit statt ueber Fensterindex bezogen) deckt das
        # zuverlaessig auf.
        expected_true_bed = data.trajectory.outputs[trajectory_rows, time_index, bed_output_index]
        expected_true_moisture = data.trajectory.outputs[
            trajectory_rows, time_index, moisture_output_index
        ]
        if not np.allclose(expected_true_bed, true_bed, atol=1e-6, rtol=1e-6) or not np.allclose(
            expected_true_moisture, true_moisture, atol=1e-6, rtol=1e-6
        ):
            raise ValueError(
                "test stimmt nicht mit data.trajectory/data.split ueberein -- "
                "reconstruct_exhaust_arrays braucht dasselbe GeneratedData-Objekt, "
                "aus dem test (via TrajectoryWindowDataset.from_trajectory) gebaut wurde "
                "(z.B. Trainingspool statt Konstantlauf-Holdout-Pool uebergeben?)"
            )

        true_dry_flow = dry_flow_diagnostic[trajectory_rows, time_index]
        true_inlet_humidity = inlet_humidity_diagnostic[trajectory_rows, time_index]

        t_gas_reference, y_gas_reference, outlet_reference = reconstruct_exhaust_state(
            backend,
            gas,
            material,
            isotherm,
            data.process_config.exhaust,
            bed_temperature_celsius=true_bed,
            particle_moisture_kg_per_kg=true_moisture,
            dry_air_mass_flow_kg_per_s=true_dry_flow,
            inlet_absolute_humidity_kg_per_kg=true_inlet_humidity,
        )

        predicted_dry_flow = rescale_dry_air_mass_flow_kg_per_s(
            backend,
            gas,
            true_dry_air_mass_flow_kg_per_s=true_dry_flow,
            true_bed_temperature_celsius=true_bed,
            predicted_bed_temperature_celsius=predicted_bed,
            inlet_absolute_humidity_kg_per_kg=true_inlet_humidity,
        )
        t_gas_reconstructed, y_gas_reconstructed, outlet_reconstructed = reconstruct_exhaust_state(
            backend,
            gas,
            material,
            isotherm,
            data.process_config.exhaust,
            bed_temperature_celsius=predicted_bed,
            particle_moisture_kg_per_kg=predicted_moisture,
            dry_air_mass_flow_kg_per_s=predicted_dry_flow,
            inlet_absolute_humidity_kg_per_kg=true_inlet_humidity,
        )

        reference.append(
            np.stack([t_gas_reference, y_gas_reference, outlet_reference], axis=-1)
        )
        reconstructed.append(
            np.stack([t_gas_reconstructed, y_gas_reconstructed, outlet_reconstructed], axis=-1)
        )
        runs.append(batch.run_index)

    reference_array = np.concatenate(reference, axis=0)
    reconstructed_array = np.concatenate(reconstructed, axis=0)
    run_array = np.concatenate(runs, axis=0)

    return reconstructed_array, reference_array, run_array, horizons


def evaluate_exhaust_reconstruction(
    model: SequenceModel,
    data: GeneratedData,
    test: SequenceDataset,
    *,
    horizons: tuple[int, ...] | None = None,
    batch_size: int = 4096,
) -> ExhaustReconstructionResult:
    """T_gas_out/Y_gas_out/outlet_temperature_celsius: Vorhersage- gegen Wahrheits-Rückrechnung.

    Dünner Aggregations-Wrapper um :func:`reconstruct_exhaust_arrays` -- siehe dort für
    Voraussetzungen (``test.target_names``) und den Leck-/Fehlpaarungs-Schutz.
    """
    reconstructed_array, reference_array, run_array, resolved_horizons = (
        reconstruct_exhaust_arrays(model, data, test, horizons=horizons, batch_size=batch_size)
    )

    by_horizon = pooled_nrmse(
        reconstructed_array, reference_array, resolved_horizons, EXHAUST_TARGET_NAMES
    )
    spread = per_run_nrmse(
        reconstructed_array, reference_array, run_array, resolved_horizons, EXHAUST_TARGET_NAMES
    )
    rmse_physical = per_run_rmse_median(
        reconstructed_array, reference_array, run_array, resolved_horizons, EXHAUST_TARGET_NAMES
    )

    return ExhaustReconstructionResult(
        horizons=resolved_horizons,
        nrmse_by_horizon=by_horizon,
        nrmse_run_spread=spread,
        n_windows=int(reference_array.shape[0]),
        n_runs=int(np.unique(run_array).size),
        rmse_run_median_physical=rmse_physical,
    )

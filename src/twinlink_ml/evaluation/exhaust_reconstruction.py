r"""End-to-end quality of the exhaust air reconstruction from predicted T_bed/moisture (Befund 41g).

Befund 41 showed that ``simulation.exhaust.reconstruct_exhaust_state()`` agrees bit for bit
with ``solve_steady_state()`` when it is fed the **true, simulated** values of
``bed_temperature_celsius``/``product_moisture_kg_per_kg``. That is a wiring check, not a
statement about the actual payload: the function is meant to run with **predicted** values
of a sequence model. This module closes exactly that gap without changing
``exhaust.py``/``evaluator.py`` themselves -- it calls both unchanged and works on the raw,
unwindowed ``Trajectory`` (``GeneratedData.trajectory``), because the diagnostic quantities
needed for the reconstruction (``dry_air_mass_flow_kg_per_s``,
``inlet_humidity_kg_per_kg``) are discarded during windowing in ``data.windows``.

Mass flow leak (uncovered before the first cost probe, not found afterwards)
--------------------------------------------------------------------------------------
Contrary to first appearance, ``dry_air_mass_flow_kg_per_s`` can **not** be derived purely
from the action (gas velocity) and the disturbance: ``greybox._rhs`` evaluates the air
density at ``bed_temperature``, not at the inlet/ambient temperature
(``gas_density_kg_per_m3(gas, bed_temperature, ...)``, ideally mixed bed). In a multi-step
rollout (h > 1) the future true bed temperature is not known in deployment -- that is
exactly why it is predicted. Feeding the true mass flow derived from the true future T_bed
into the reconstruction side would be a real leak and would make the end-to-end metric
more optimistic than a real system could achieve. ``inlet_absolute_humidity_kg_per_kg``
(Y_in) is not affected by this -- it depends only on ambient temperature/humidity
(disturbances), never on T_bed.

The correction (``rescale_dry_air_mass_flow_kg_per_s``) scales the true mass flow by the
density ratio at predicted vs. true bed temperature -- the same formula as in
``greybox._rhs``, here only evaluated with the predicted instead of the true temperature,
and without the detour via gas velocity/cross-sectional area (they cancel in the ratio).
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
    r"""Convert the dry air mass flow to the predicted instead of the true bed temperature.

    ``greybox._rhs`` computes ``dry_air_mass_flow_kg_per_s`` as
    :math:`\rho(T_{bed}, Y_{in}) \cdot u_{gas} \cdot A / (1 + Y_{in})`. Gas velocity and
    cross-sectional area are unknown here (not part of the ``diagnostics``), but cancel
    exactly in the ratio of two mass flows with the same excitation but different bed
    temperature:

    .. math:: \dot m_{L,tr}(T_{pred}) = \dot m_{L,tr}(T_{true})
              \cdot \frac{\rho(T_{pred}, Y_{in})}{\rho(T_{true}, Y_{in})}

    With ``T_pred == T_true`` the ratio is exactly 1.0 (the same floating-point result
    divided by itself), so the return value is bit-identical to the true mass flow --
    no new physics, only an application of ``thermo.gas_density_kg_per_m3`` (imported
    unchanged) to a different temperature than the one ``greybox._rhs`` uses.
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
    """NRMSE between the exhaust air reconstruction from the prediction and from the truth.

    Both reconstructions run through the same, unchanged
    ``exhaust.reconstruct_exhaust_state()`` -- reference with the true
    ``bed_temperature_celsius``/``product_moisture_kg_per_kg`` (and the true
    ``dry_air_mass_flow_kg_per_s``), reconstruction with the values of the sequence model
    (and the mass flow converted to them, see ``rescale_dry_air_mass_flow_kg_per_s``).
    ``inlet_absolute_humidity_kg_per_kg`` is true in both calls (a genuine disturbance,
    no leak).
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
        """Table for run logs, in the same style as ``EvaluationResult.format``."""
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
        """Flat key-value pairs for tracking.

        ``exhaust_`` prefix -- can be merged without collisions alongside
        ``EvaluationResult.as_flat_metrics()`` in the same ``RunRecord``.
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
    """Raw ``(reconstructed, reference, runs, horizons)`` arrays of the exhaust air reconstruction.

    A separate function (not just a code block in :func:`evaluate_exhaust_reconstruction`),
    so that a further derived quantity -- such as the recoverable power ``Q_rec`` from the
    predicted ``T_out``/``Y_out`` (X3, ``docs/04_stand_ap1.md``) -- can reuse the same
    (prediction, truth) arrays without writing the batch/leak-protection code above a
    second time. ``reconstructed``/``reference`` have shape
    ``(n_windows, horizon, len(EXHAUST_TARGET_NAMES))``, column order as in
    :data:`EXHAUST_TARGET_NAMES`.

    Requires ``bed_temperature_celsius`` and ``product_moisture_kg_per_kg`` to be in
    ``test.target_names`` (explicit extension, see Befund 41f -- no global default). For
    the two required diagnostic quantities not present in ``WindowBatch``
    (``dry_air_mass_flow_kg_per_s``, ``inlet_humidity_kg_per_kg``) it falls back on the
    unwindowed ``data.trajectory.diagnostics``, indexed via the same split order that
    ``TrajectoryWindowDataset.from_trajectory`` uses during construction
    (``data.split.as_mapping["test"]`` against ``data.trajectory.run_id``).
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

        # Self-check: `test` MUST have been built from `data.trajectory`/`data.split` --
        # otherwise `run_index`/`window_start` only happen to fit in shape but point to the
        # wrong trajectory (e.g. training pool passed instead of constant-run holdout
        # pool). Without this check the result would not be wrongly shaped, only
        # SILENTLY wrong -- pure shape matching would give the same number of windows/the
        # same horizon for any second pool of the same DataSpec. Comparing against
        # ``data.trajectory.outputs`` (the same quantity as ``future_targets_truth``, only
        # referenced independently via trajectory row/time instead of via window index)
        # reliably exposes this.
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

        reference.append(np.stack([t_gas_reference, y_gas_reference, outlet_reference], axis=-1))
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
    """T_gas_out/Y_gas_out/outlet_temperature_celsius: prediction vs. truth reconstruction.

    Thin aggregation wrapper around :func:`reconstruct_exhaust_arrays` -- see there for the
    prerequisites (``test.target_names``) and the leak/mismatch protection.
    """
    reconstructed_array, reference_array, run_array, resolved_horizons = reconstruct_exhaust_arrays(
        model, data, test, horizons=horizons, batch_size=batch_size
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

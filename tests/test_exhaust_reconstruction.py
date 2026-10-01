"""Regressionstests für die Abluft-Rückrechnung (``evaluation/exhaust_reconstruction.py``).

Zwei unabhängige Fehlklassen, zwei Testgruppen:

1. **Massenstrom-Leck** (Befund 42): ``dry_air_mass_flow_kg_per_s`` hängt
   konstruktionsbedingt von der Betttemperatur ab (``greybox._rhs`` wertet die Luftdichte
   bei ``bed_temperature`` aus, nicht bei Einlasstemperatur). Würde man für die
   "Rekonstruktion aus Modellvorhersage" trotzdem den wahren, aus der wahren Betttemperatur
   abgeleiteten Massenstrom verwenden, wäre die End-zu-End-Metrik optimistischer, als ein
   reales System mit vorhergesagtem statt wahrem T_bed leisten könnte -- ein Leck aus der
   Zukunft, im selben Sinn, den die bestehenden ``rollout_does_not_see_future_targets``-Tests
   für die Sequenzmodelle selbst prüfen.

2. **Stille Fehlpaarung** (Befund 42(b)): ``evaluate_exhaust_reconstruction()`` braucht ein
   ``(GeneratedData, SequenceDataset)``-Paar, das tatsächlich zusammengehört (``test`` muss
   aus ``data.trajectory``/``data.split`` gebaut worden sein). Eine erste Fassung prüfte das
   nicht -- bei zufällig passender Fensterzahl (z. B. Trainingspool und Konstantlauf-Holdout
   mit gleichem ``n_test``) wäre eine Verwechslung klaglos durchgelaufen und hätte still
   falsche Zahlen geliefert, kein Absturz. Gefunden durch eine gezielte Gegenprobe (falsches
   Paar absichtlich übergeben), nicht durch einen Fehlerbericht.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from twinlink_ml.data.splits import SplitName
from twinlink_ml.data.windows import TrajectoryWindowDataset
from twinlink_ml.evaluation import DataSpec, ValidationPolicy, build_datasets, generate_data
from twinlink_ml.evaluation.exhaust_reconstruction import (
    evaluate_exhaust_reconstruction,
    rescale_dry_air_mass_flow_kg_per_s,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, ProcessConfig, load_process_config
from twinlink_ml.simulation.backend import ArrayBackend
from twinlink_ml.simulation.steady_state import solve_steady_state

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_rescaled_dry_air_mass_flow_matches_true_value_when_prediction_is_exact(
    config: ProcessConfig, simulator, backend: ArrayBackend
) -> None:
    """T_pred == T_true: keine neue Physik darf hier bereits eine Abweichung erzeugen."""
    solution = solve_steady_state(simulator)
    assert solution.feasible, solution.infeasibility_reasons

    true_bed_temperature = backend.asarray(np.array([solution.bed_temperature_celsius]))
    true_dry_flow = backend.asarray(np.array([solution.dry_air_mass_flow_kg_per_s]))
    inlet_humidity = backend.asarray(np.array([solution.diagnostics["inlet_humidity_kg_per_kg"]]))

    rescaled = rescale_dry_air_mass_flow_kg_per_s(
        backend,
        config.gas,
        true_dry_air_mass_flow_kg_per_s=true_dry_flow,
        true_bed_temperature_celsius=true_bed_temperature,
        predicted_bed_temperature_celsius=true_bed_temperature,
        inlet_absolute_humidity_kg_per_kg=inlet_humidity,
    )

    assert np.array_equal(backend.to_numpy(rescaled), backend.to_numpy(true_dry_flow)), (
        "Degenerierter Fall (T_pred == T_true) muss bitgleich zum wahren Massenstrom sein"
    )


def test_rescaled_dry_air_mass_flow_decreases_for_overpredicted_bed_temperature(
    config: ProcessConfig, simulator, backend: ArrayBackend
) -> None:
    """Plausibler Vorhersagefehler (+3 °C) muss den Massenstrom nachweisbar senken.

    Ideales Gasgesetz: höhere Temperatur -> geringere Dichte -> bei gleicher
    Gasgeschwindigkeit (in der Rekonstruktion unbekannt, kürzt sich aber im Verhältnis
    heraus, siehe Docstring von ``rescale_dry_air_mass_flow_kg_per_s``) geringerer
    Massenstrom. Ohne diesen Test wäre nicht belegt, dass die Korrektur überhaupt etwas
    bewirkt, statt nur rechnerisch plausibel zu klingen.
    """
    solution = solve_steady_state(simulator)
    assert solution.feasible, solution.infeasibility_reasons

    true_bed_temperature = backend.asarray(np.array([solution.bed_temperature_celsius]))
    predicted_bed_temperature = backend.asarray(np.array([solution.bed_temperature_celsius + 3.0]))
    true_dry_flow = backend.asarray(np.array([solution.dry_air_mass_flow_kg_per_s]))
    inlet_humidity = backend.asarray(np.array([solution.diagnostics["inlet_humidity_kg_per_kg"]]))

    rescaled = rescale_dry_air_mass_flow_kg_per_s(
        backend,
        config.gas,
        true_dry_air_mass_flow_kg_per_s=true_dry_flow,
        true_bed_temperature_celsius=true_bed_temperature,
        predicted_bed_temperature_celsius=predicted_bed_temperature,
        inlet_absolute_humidity_kg_per_kg=inlet_humidity,
    )

    rescaled_value = float(backend.to_numpy(rescaled)[0])
    true_value = float(backend.to_numpy(true_dry_flow)[0])
    assert rescaled_value < true_value, (
        "+3 °C Vorhersagefehler muss den rekonstruierten Massenstrom senken "
        "(ideales Gasgesetz), nicht nur irgendeine Abweichung erzeugen"
    )
    # Deutlich messbar, nicht nur eine Gleitkomma-Nuance -- bei ~3 K Unterschied um eine
    # Betttemperatur im Bereich einiger 10 °C sollte die relative Änderung bei ~1 % liegen.
    relative_change = (true_value - rescaled_value) / true_value
    assert relative_change > 1e-4, (
        f"Effekt zu klein, um kein Rundungsartefakt zu sein: {relative_change}"
    )


# --------------------------------------------------------- Stille Fehlpaarung (Befund 42b)
#
# Winzige Groessen (n_train_max=8, run_duration_s=4000) statt Produktionsgroesse -- geprueft
# wird die Verdrahtung (wirft der Fehler bei einer Verwechslung?), nicht die Guete einer
# Vorhersage. Muster identisch zu run_arch_constant_holdout_worker.py (Befund 36): ein
# Trainingspool ("excited") und ein separater Konstantlauf-Holdout-Pool, beide via
# generate_data() erzeugt, test_constant via TrajectoryWindowDataset.from_trajectory(...)
# mit den Trainings-Skalierern gebaut -- absichtlich mit gleichem n_test, damit eine
# Verwechslung nur an den WERTEN sichtbar wuerde, nicht an der Form.

CONFIG_PATH = "conf/process/greybox_gab_placeholder.yaml"
HOLDOUT_SEED = 999
FOUR_TARGETS = (
    "energy_target",
    "product_moisture_kg_per_kg",
    "d50_um",
    "bed_temperature_celsius",
)


def _tiny_excited_spec() -> DataSpec:
    return DataSpec(
        sample_time_s=25.0,
        run_duration_s=4000.0,
        n_train_max=8,
        n_val=4,
        n_test=4,
        history_length=16,
        horizon=8,
        stride=8,
        target_names=FOUR_TARGETS,
    )


def _tiny_holdout_spec(excited: DataSpec) -> DataSpec:
    return dataclasses.replace(
        excited, n_train_max=2, n_val=2, n_test=excited.n_test, constant_run_fraction=1.0
    )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_evaluate_exhaust_reconstruction_rejects_mismatched_data_and_test() -> None:
    """Falsches (GeneratedData, SequenceDataset)-Paar muss ValueError werfen, nicht durchlaufen.

    Vor Befund 42(b) prüfte die Funktion nicht, ob ``test`` tatsächlich aus
    ``data.trajectory``/``data.split`` gebaut wurde -- bei gleicher Fensterzahl (hier: beide
    Pools mit ``n_test=4``) lief eine Verwechslung klaglos durch und hätte still falsche
    Zahlen geliefert (Diagnostics vom falschen Pool, nur formal passend indiziert). Diese
    Gegenprobe reproduziert genau die Verwechslung, die den Fehler ursprünglich aufdeckte:
    ``excited`` (Trainingspool) zusammen mit ``test_constant`` (Konstantlauf-Testmenge).
    """
    from twinlink_ml.models.composite import CompositeGruSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)

    excited_spec = _tiny_excited_spec()
    holdout_spec = _tiny_holdout_spec(excited_spec)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        excited = generate_data(config, excited_spec, master_seed=1)
        holdout = generate_data(config, holdout_spec, master_seed=HOLDOUT_SEED)

    train, validation, test_excited = build_datasets(
        excited, excited_spec.n_train_max, policy=ValidationPolicy.FIXED_POOL
    )
    test_constant = TrajectoryWindowDataset.from_trajectory(
        holdout.trajectory,
        holdout.split,
        SplitName.TEST,
        excited_spec.window,
        scalers=train.scalers,
    )
    assert test_excited.n_windows == test_constant.n_windows, (
        "Testaufbau ungueltig -- die Gegenprobe braucht gleiche Fensterzahl in beiden "
        "Pools, sonst waere ein Fehler auch ohne die Konsistenzpruefung an der Form sichtbar"
    )

    model = CompositeGruSequenceModel(TorchSequenceConfig(hidden_size=8, max_epochs=1, patience=0))
    model.fit(train, seed=excited.seeds.model_seed, validation=validation)

    # Korrekte Paarungen muessen weiterhin funktionieren -- die Pruefung darf nicht pauschal
    # ueber-scharf sein und auch richtige Aufrufe blockieren.
    evaluate_exhaust_reconstruction(model, excited, test_excited, horizons=(1, 8))
    evaluate_exhaust_reconstruction(model, holdout, test_constant, horizons=(1, 8))

    # Die eigentliche Gegenprobe: vertauschtes Paar muss scheitern.
    with pytest.raises(ValueError, match="stimmt nicht mit data.trajectory"):
        evaluate_exhaust_reconstruction(model, excited, test_constant, horizons=(1, 8))

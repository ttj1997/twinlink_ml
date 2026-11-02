"""Regression tests for the exhaust air back-calculation (``evaluation/exhaust_reconstruction.py``).

Two independent error classes, two test groups:

1. **Mass-flow leak** (Befund 42): ``dry_air_mass_flow_kg_per_s`` depends
   by construction on the bed temperature (``greybox._rhs`` evaluates the air density
   at ``bed_temperature``, not at the inlet temperature). If one nevertheless used, for the
   "reconstruction from model prediction", the true mass flow derived from the true bed
   temperature, the end-to-end metric would be more optimistic than a
   real system with predicted instead of true T_bed could achieve -- a leak from the
   future, in the same sense that the existing ``rollout_does_not_see_future_targets`` tests
   check for the sequence models themselves.

2. **Silent mismatch** (Befund 42(b)): ``evaluate_exhaust_reconstruction()`` needs a
   ``(GeneratedData, SequenceDataset)`` pair that actually belongs together (``test`` must
   have been built from ``data.trajectory``/``data.split``). A first version did not check
   this -- with a coincidentally matching window count (e.g. training pool and constant-run
   holdout with the same ``n_test``) a mix-up would have run through without complaint and
   silently delivered wrong numbers, no crash. Found through a targeted cross-check (wrong
   pair passed deliberately), not through a bug report.
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
    """T_pred == T_true: no new physics may already produce a deviation here."""
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
    """A plausible prediction error (+3 °C) must demonstrably lower the mass flow.

    Ideal gas law: higher temperature -> lower density -> at the same
    gas velocity (unknown in the reconstruction, but cancels out in the ratio,
    see the docstring of ``rescale_dry_air_mass_flow_kg_per_s``) lower
    mass flow. Without this test there would be no evidence that the correction has any
    effect at all, instead of merely sounding arithmetically plausible.
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
    # Clearly measurable, not just a floating-point nuance -- with a ~3 K difference around a
    # bed temperature in the range of some 10s of °C the relative change should be ~1 %.
    relative_change = (true_value - rescaled_value) / true_value
    assert relative_change > 1e-4, (
        f"Effekt zu klein, um kein Rundungsartefakt zu sein: {relative_change}"
    )


# ---------------------------------------------------------- Silent mismatch (Befund 42b)
#
# Tiny sizes (n_train_max=8, run_duration_s=4000) instead of production size -- what is
# checked is the wiring (does the error raise on a mix-up?), not the quality of a
# prediction. Pattern identical to run_arch_constant_holdout_worker.py (Befund 36): a
# training pool ("excited") and a separate constant-run holdout pool, both generated via
# generate_data(), test_constant built via TrajectoryWindowDataset.from_trajectory(...)
# with the training scalers -- deliberately with the same n_test, so that a
# mix-up would only become visible in the VALUES, not in the shape.

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
    """A wrong (GeneratedData, SequenceDataset) pair must raise ValueError, not run through.

    Before Befund 42(b) the function did not check whether ``test`` was actually built from
    ``data.trajectory``/``data.split`` -- with the same window count (here: both
    pools with ``n_test=4``) a mix-up ran through without complaint and would silently have
    delivered wrong numbers (diagnostics from the wrong pool, only formally indexed to match).
    This cross-check reproduces exactly the mix-up that originally uncovered the error:
    ``excited`` (training pool) together with ``test_constant`` (constant-run test set).
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

    # Correct pairings must continue to work -- the check must not be
    # over-sharp across the board and also block correct calls.
    evaluate_exhaust_reconstruction(model, excited, test_excited, horizons=(1, 8))
    evaluate_exhaust_reconstruction(model, holdout, test_constant, horizons=(1, 8))

    # The actual cross-check: a swapped pair must fail.
    with pytest.raises(ValueError, match="stimmt nicht mit data.trajectory"):
        evaluate_exhaust_reconstruction(model, excited, test_constant, horizons=(1, 8))
